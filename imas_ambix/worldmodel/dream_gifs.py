"""Full-resolution reconstruction + dream GIFs for the plan-conditioned world model.

The 4x4 token-grid view of the camera prediction is useless as a demo.  This
builds the two qualitative GIFs a reader actually wants — both decoded back to
real 256x256 camera images through the frozen Open-MAGVIT2 VQModel:

1. **Reconstruction GIF** — side-by-side ``GT | model teacher-forced prediction``.
   At every grid step the model sees the TRUE tokens up to that step and emits
   its next-token rbb prediction; we decode that prediction (and the GT) to
   full-res images and animate them over the target window.  This shows whether
   the model's one-step rbb forecast is plasma-like.
2. **Dream GIF** — side-by-side ``GT | model autoregressive rollout``.  The
   model is given only the shot's real plan + the short context window, then
   rolls the rbb stream forward consuming its OWN predictions (the "dream").
   This shows whether the dream stays coherent/evolving or collapses/drifts.

ID space (the validated mapping — mirrors
:mod:`imas_ambix.camdyn.reconstruction_demo`)
---------------------------------------------------------------------------
The rbb modality runs at FULL resolution (``camera_grid_stride=1`` ⇒ 256
channels = the whole 16x16 frame grid, row-major ``for r for c``).  The world
model consumes the on-disk frame-store ids DIRECTLY (``_read_camera`` does NOT
rebase the camera, unlike the signal_hf groups), so the model's predicted rbb
token ids live in the SAME store-id space as the on-disk GT grids.  Both are
decoded by the established
:func:`imas_ambix.camdyn.reconstruction_demo.run_decode_subprocess`, which
subtracts ``REGISTRY_OFFSET`` (=4) and runs the VQModel under the MAGVIT2 venv —
so GT and prediction take an identical decode path and a decoded prediction is
never mislabelled.

Two-venv handoff (mirrors demo_artifacts.py)
--------------------------------------------
PHASE A (this venv, GPU): load the checkpoint once, assemble the shot, run the
teacher-forced forward + the autoregressive rollout, dump a ``(N,F,16,16)``
store-id token bundle in the reconstruction_demo format.
PHASE B (MAGVIT2 venv, GPU): decode the whole stack to ``(N,F,256,256,3)`` uint8.
PHASE C (this venv): assemble the two GIFs.

Run (single GPU; keep the neighbour's LLM server + the live retrain up)::

    .venv/bin/python -m imas_ambix.worldmodel.dream_gifs \\
        --checkpoint /work/projects/imas_gpu/worldmodel/ckpt/1219524/ckpt-00012500.pt \\
        --shot 23735 \\
        --out-dir docs/figures/world-model-demonstration

Re-runnable on the FINAL checkpoint (latest.pt, step 20000) by just changing
``--checkpoint`` (and the title strip reflects the loaded step).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import tempfile
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

#: An interim, genuinely-trained full-res rbb checkpoint (step 12,500).  The
#: FINAL run lands ``latest.pt`` at step 20,000 — pass ``--checkpoint`` to
#: re-render on it.
DEFAULT_CHECKPOINT = Path(
    "/work/projects/imas_gpu/worldmodel/ckpt/1219524/ckpt-00012500.pt"
)

#: Held-out shot (discovery index 4000, beyond the retrain's 3000-shot set).
DEFAULT_SHOT = 23735

DEFAULT_OUT_DIR = Path("docs/figures/world-model-demonstration")

#: The camera the full-res rbb head predicts and we decode.
REFERENCE_CAMERA = "rbb"
GRID_H, GRID_W = 16, 16

#: The training window contract (model token context = plan_steps 64 + obs_steps
#: 64; the common grid is n_steps=64 with the first context_steps=16 given).
WINDOW_N_STEPS = 64
WINDOW_CONTEXT_STEPS = 16

#: GIF playback.
FPS = 8


# ---------------------------------------------------------------------------
# Clean-cancellation stop flag (repo §2b GPU-safety pattern)
# ---------------------------------------------------------------------------


class _StopFlag:
    """SIGTERM/SIGINT-set stop flag the loops poll (clean cancel in < 5 s)."""

    def __init__(self) -> None:
        self.stop = False

    def install(self) -> None:
        def _handler(signum, _frame):  # noqa: ANN001
            logger.warning("received signal %s — setting STOP flag", signum)
            self.stop = True

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, _handler)
            except (ValueError, OSError):
                logger.debug("could not install handler for %s", sig)


# ---------------------------------------------------------------------------
# Teacher-forced full-window rbb prediction
# ---------------------------------------------------------------------------


def _teacher_forced_rbb(
    model,
    sample,
    obs_names,
    plan_names,
    *,
    chunk_channels: int = 32,
):
    """Teacher-forced next-token rbb prediction over the full window.

    Unlike the autoregressive rollout, this feeds the model the TRUE tokens at
    every step (one ``encode`` pass over the whole window) and reads the rbb
    head's per-channel argmax at each step.  ``logits`` at step ``t`` predict the
    token at step ``t+1``, so the prediction for grid step ``s`` (``s >= 1``)
    comes from the hidden state at step ``s-1``.  Returns ``(n_steps, 256)``
    store-id token ids — step 0 copied from truth (no predecessor).

    Channel-chunked (mirrors :func:`eval._chunked_argmax_step`) so the
    full-resolution 256-channel x 2^18-vocab head never materialises the
    all-channel logit tensor.
    """
    import torch

    from imas_ambix.worldmodel.eval import _chunked_argmax_step
    from imas_ambix.worldmodel.train import pad_collate_batch

    channels = {m.name: int(m.n_channels) for m in model.config.modalities}
    batch = pad_collate_batch([sample], obs_names, plan_names, channels)
    n_steps = sample.n_steps

    with torch.no_grad():
        obs_hidden = model.encode(batch)  # (1, obs_len, d) — true tokens fed in
        fixed_ch = int(model.channel_query[REFERENCE_CAMERA].shape[0])
        pred = np.zeros((n_steps, fixed_ch), dtype=np.int64)
        # step 0 has no predecessor hidden state — copy the truth there.
        true_rbb = np.asarray(batch["tokens"][REFERENCE_CAMERA][0], dtype=np.int64)
        pred[0] = true_rbb[0, :fixed_ch] if true_rbb.shape[1] >= fixed_ch else 0
        obs_len = int(obs_hidden.shape[1])
        for s in range(1, min(n_steps, obs_len)):
            step_pred = _chunked_argmax_step(
                model,
                obs_hidden,
                REFERENCE_CAMERA,
                s - 1,
                chunk_channels=chunk_channels,
            )  # (1, fixed_ch)
            pred[s] = step_pred[0].cpu().numpy().astype(np.int64)
    return pred  # (n_steps, 256) store-id


# ---------------------------------------------------------------------------
# Token-grid extraction helpers
# ---------------------------------------------------------------------------


def _flat_to_grid(flat: np.ndarray) -> np.ndarray:
    """``(T, 256) -> (T, 16, 16)`` row-major (matches ``_read_camera`` flatten)."""
    flat = np.asarray(flat, dtype=np.int64)
    t = flat.shape[0]
    c = flat.shape[1]
    if c != GRID_H * GRID_W:
        raise ValueError(
            f"expected {GRID_H * GRID_W} rbb channels (full-res stride=1), got {c}"
        )
    return flat.reshape(t, GRID_H, GRID_W)


def _gt_rbb_grids(sample) -> np.ndarray:
    """GT rbb store-id grids ``(n_steps, 16, 16)`` from the assembled sample.

    The sample holds the on-disk frame-store ids (NOT rebased for the camera),
    so these are the same store-id space the model predicts in — decode subtracts
    ``REGISTRY_OFFSET`` once for both.
    """
    if REFERENCE_CAMERA not in sample.tokens:
        raise ValueError(
            f"assembled sample carries no {REFERENCE_CAMERA!r} tokens — pick an "
            "rbb-bearing held-out shot"
        )
    return _flat_to_grid(np.asarray(sample.tokens[REFERENCE_CAMERA], dtype=np.int64))


# ---------------------------------------------------------------------------
# Phase A — predict (teacher-forced + dream) and dump the token bundle
# ---------------------------------------------------------------------------


def run_phase_a(
    *,
    checkpoint: Path,
    shot_id: int,
    token_bundle: Path,
    device: str = "cuda",
) -> dict:
    """Load the model once, assemble the shot, build GT + TF + dream rbb grids.

    Writes a ``(N,F,16,16)`` store-id token bundle in the reconstruction_demo
    format with a JSON index tagging each slice as ``gt`` / ``teacher_forced`` /
    ``dream`` so phase B decodes everything in one batched pass.  Returns a
    summary dict (step, n_steps, context_steps, mismatch rates).
    """
    import torch

    from imas_ambix.worldmodel.dataset import (
        WorldModelWindowConfig,
        build_shot_sample,
        default_modalities,
    )
    from imas_ambix.worldmodel.eval import _model_obs_plan_names, rollout
    from imas_ambix.worldmodel.train import load_model_from_checkpoint

    stop = _StopFlag()
    stop.install()

    # determinism (repo §2b) — matches the training / encode flags.
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if device == "cuda" and torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    else:
        device = "cpu"

    modalities = default_modalities()
    window = WorldModelWindowConfig(
        n_steps=WINDOW_N_STEPS, context_steps=WINDOW_CONTEXT_STEPS
    )

    logger.info("assembling held-out shot %s", shot_id)
    sample = build_shot_sample(shot_id, modalities, window)
    if REFERENCE_CAMERA not in sample.tokens:
        raise ValueError(
            f"shot {shot_id} assembled WITHOUT {REFERENCE_CAMERA} tokens — cannot "
            "build the camera demo; pick another rbb-bearing held-out shot"
        )

    logger.info("loading checkpoint on %s: %s", device, checkpoint)
    model, payload = load_model_from_checkpoint(checkpoint, map_location=device)
    model.eval()
    obs_names, plan_names = _model_obs_plan_names(model)
    step = int(payload.get("step", -1))
    logger.info(
        "checkpoint step=%d params=%d ctx_len=%d",
        step,
        int(model.num_parameters()),
        int(model.context_length()),
    )

    ctx = int(sample.context_steps)
    n_steps = sample.n_steps

    # Move the model to CPU for the predict phase — ``eval.rollout`` and
    # :func:`_teacher_forced_rbb` build their batch tensors on CPU
    # (``pad_collate_batch``) and the model reads its device from the
    # parameters, so a CUDA model would device-mismatch the CPU inputs.  This is
    # the ESTABLISHED CPU-eval path (``demo_artifacts.run_phase_a`` /
    # ``train._run_periodic_eval`` do exactly this); it also frees the GPU during
    # the CPU-bound rollout, leaving the neighbour's LLM server + the retrain
    # their cards.  The decode (phase B) runs on the GPU under the MAGVIT2 venv.
    if device == "cuda" and torch.cuda.is_available():
        model.to("cpu")
        torch.cuda.empty_cache()
        logger.info("moved model to CPU for the predict phase (frees the GPU)")

    # GT grids (store-id space, full window).
    gt_grids = _gt_rbb_grids(sample)  # (n_steps, 16, 16)

    # Teacher-forced one-step rbb prediction over the full window.
    t0 = time.time()
    tf_flat = _teacher_forced_rbb(model, sample, obs_names, plan_names)
    tf_grids = _flat_to_grid(tf_flat)  # (n_steps, 16, 16)
    logger.info("teacher-forced rbb prediction done in %.1fs", time.time() - t0)

    # Autoregressive dream: context kept, target generated from the model's own
    # predictions (the established eval.rollout, chunked-CE-safe at full res).
    t0 = time.time()
    with torch.no_grad():
        predicted = rollout(model, sample, obs_names, plan_names)
    if REFERENCE_CAMERA not in predicted:
        raise ValueError(
            f"rollout produced no {REFERENCE_CAMERA} prediction — the model lacks "
            "the rbb head or the sample dropped it"
        )
    dream_flat = np.asarray(predicted[REFERENCE_CAMERA], dtype=np.int64)  # (T, c)
    # rollout returns the overlap channel width; rbb is full-res so c == 256, but
    # guard anyway by padding to 256 with the GT id (only affects decode, never
    # scored) so the reshape to 16x16 is always valid.
    if dream_flat.shape[1] < GRID_H * GRID_W:
        pad_c = GRID_H * GRID_W - dream_flat.shape[1]
        gt_flat = np.asarray(sample.tokens[REFERENCE_CAMERA], dtype=np.int64)
        dream_flat = np.concatenate(
            [dream_flat, gt_flat[:, dream_flat.shape[1] : dream_flat.shape[1] + pad_c]],
            axis=1,
        )
    dream_grids = _flat_to_grid(dream_flat[:, : GRID_H * GRID_W])
    logger.info("autoregressive dream rollout done in %.1fs", time.time() - t0)

    # token-mismatch over the target window (a quick honesty readout).
    valid = np.asarray(sample.valid[REFERENCE_CAMERA], dtype=bool).reshape(
        n_steps, GRID_H, GRID_W
    )
    tgt_valid = valid[ctx:]
    n_tgt = int(tgt_valid.sum())
    tf_mismatch = (
        float(((tf_grids[ctx:] != gt_grids[ctx:]) & tgt_valid).sum()) / n_tgt
        if n_tgt
        else float("nan")
    )
    dream_mismatch = (
        float(((dream_grids[ctx:] != gt_grids[ctx:]) & tgt_valid).sum()) / n_tgt
        if n_tgt
        else float("nan")
    )
    # how much the dream evolves vs collapses: fraction of target cells that
    # CHANGE from the last context frame (0 == frozen/collapsed, high == evolving).
    last_ctx = dream_grids[ctx - 1]
    dream_change = float((dream_grids[ctx:] != last_ctx[None]).mean())

    # release the model (repo §2b)
    try:
        del model
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001
        logger.warning("model release note: %r", exc)

    # ── write the token bundle (reconstruction_demo format) ─────────────────
    # one (N,F,16,16) store-id stack; the JSON index tags each slice's role.
    grids_stack = np.stack([gt_grids, tf_grids, dream_grids]).astype(np.int64)
    index = [
        {"role": "gt", "slot": 0},
        {"role": "teacher_forced", "slot": 1},
        {"role": "dream", "slot": 2},
    ]
    meta = {
        "format": "reconstruction_demo",
        "camera": REFERENCE_CAMERA,
        "id_space": "store-id (model native == on-disk frame store; decode -4)",
        "grid_hw": [GRID_H, GRID_W],
        "shot_id": int(shot_id),
        "checkpoint": str(checkpoint),
        "checkpoint_step": step,
        "n_steps": int(n_steps),
        "context_steps": int(ctx),
        "grid_time": np.asarray(sample.grid_time, dtype=np.float64).tolist(),
    }
    token_bundle.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        token_bundle,
        grids=grids_stack,
        index=json.dumps(index),
        meta=json.dumps(meta),
    )
    logger.info("wrote token bundle -> %s", token_bundle)

    return {
        "checkpoint": str(checkpoint),
        "checkpoint_step": step,
        "shot_id": int(shot_id),
        "n_steps": int(n_steps),
        "context_steps": int(ctx),
        "n_target_valid": n_tgt,
        "teacher_forced_token_mismatch": tf_mismatch,
        "dream_token_mismatch": dream_mismatch,
        "dream_change_fraction": dream_change,
        "grid_time": meta["grid_time"],
    }


# ---------------------------------------------------------------------------
# Phase C — assemble the GIFs from the decoded images
# ---------------------------------------------------------------------------

#: Native rbb frame aspect (rows, cols) — crop the square decode back to the
#: camera aspect for display (mirrors reconstruction_demo._to_aspect).
ORIGINAL_HW = (112, 156)


def _to_aspect(img_square: np.ndarray) -> np.ndarray:
    """Resize a 256x256 decoded image to the native rbb aspect (grayscale)."""
    from PIL import Image

    if img_square.ndim == 3:
        img_square = img_square[..., 0]
    im = Image.fromarray(img_square.astype(np.uint8)).resize(
        (ORIGINAL_HW[1], ORIGINAL_HW[0]), Image.BILINEAR
    )
    return np.asarray(im)


def _panel_frame(
    gt_img: np.ndarray,
    pred_img: np.ndarray,
    *,
    left_title: str,
    right_title: str,
    banner: str,
    in_target: bool,
) -> np.ndarray:
    """Render one side-by-side (GT | prediction) panel to an RGB uint8 array.

    A one-line title strip sits above each panel; a shared banner runs along the
    top.  A coloured frontier tint flags target-window frames (model is now
    forecasting / dreaming) vs the given context frames.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0), dpi=100)
    fig.subplots_adjust(top=0.80, bottom=0.02, left=0.02, right=0.98, wspace=0.04)
    for ax, img, title in (
        (axes[0], gt_img, left_title),
        (axes[1], pred_img, right_title),
    ):
        ax.imshow(img, cmap="inferno", vmin=0, vmax=255, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=9, color=("#d62728" if in_target else "#222222"))
    fig.suptitle(banner, fontsize=8, y=0.985)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return buf


def _save_gif(frames: list[np.ndarray], out_path: Path, *, fps: int = FPS) -> tuple:
    """Write an animated, looping GIF (via PIL).  Returns ``(height, width)`` px.

    PIL is the only image lib guaranteed in the venv (imageio is absent and the
    betelgeuse node has no outbound network to install it), so the RGB frames are
    written with ``Image.save(save_all=...)`` — a looping GIF at ``fps``.
    """
    from PIL import Image

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil_frames = [Image.fromarray(np.asarray(f, dtype=np.uint8)) for f in frames]
    duration_ms = int(round(1000.0 / fps))
    pil_frames[0].save(
        str(out_path),
        format="GIF",
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    h, w = frames[0].shape[:2]
    return int(h), int(w)


def assemble_gifs(
    *,
    image_bundle: Path,
    summary: dict,
    out_dir: Path,
) -> dict:
    """Build the reconstruction + dream GIFs from the decoded image bundle.

    The decoded bundle holds ``(N,F,256,256,3)`` images with the same role index
    the token bundle carried (gt / teacher_forced / dream).  We pair GT with each
    prediction column, animate over the FULL window (context + target), and tag
    each frame's title strip with the grid time + the interim banner.
    """
    data = np.load(str(image_bundle), allow_pickle=True)
    images = np.asarray(data["images"], dtype=np.uint8)  # (N,F,256,256,3)
    index = json.loads(str(data["index"]))
    slot = {e["role"]: e["slot"] for e in index}

    gt = images[slot["gt"]]  # (F,256,256,3)
    tf = images[slot["teacher_forced"]]
    dream = images[slot["dream"]]

    ctx = int(summary["context_steps"])
    n_steps = int(summary["n_steps"])
    step = int(summary["checkpoint_step"])
    shot_id = int(summary["shot_id"])
    grid_time = np.asarray(summary["grid_time"], dtype=np.float64)
    n_frames = min(gt.shape[0], n_steps)

    interim = f"interim — step {step:,}, not final"
    out_paths: dict[str, str] = {}
    dims: dict[str, list[int]] = {}

    for role, pred_stack, fname, right_title_base, banner_kind in (
        (
            "teacher_forced",
            tf,
            "fullres-reconstruction-interim.gif",
            "model (teacher-forced)",
            "reconstruction",
        ),
        (
            "dream",
            dream,
            "fullres-dream-interim.gif",
            "model (dream / autoregressive)",
            "dream",
        ),
    ):
        frames: list[np.ndarray] = []
        for fi in range(n_frames):
            in_target = fi >= ctx
            t_ms = (
                float(grid_time[fi] - grid_time[ctx]) * 1e3 if grid_time.size else 0.0
            )
            phase = "FORECAST" if in_target else "context"
            banner = (
                f"WM {banner_kind}  |  shot {shot_id}  |  rbb full-res decode  |  "
                f"{interim}"
            )
            left_title = f"ground truth   t={t_ms:+.0f} ms"
            right_title = f"{right_title_base}   [{phase}]"
            gt_img = _to_aspect(gt[fi])
            pred_img = _to_aspect(pred_stack[fi])
            frames.append(
                _panel_frame(
                    gt_img,
                    pred_img,
                    left_title=left_title,
                    right_title=right_title,
                    banner=banner,
                    in_target=in_target,
                )
            )
        out_path = out_dir / fname
        h, w = _save_gif(frames, out_path)
        out_paths[role] = str(out_path)
        dims[role] = [h, w]
        logger.info(
            "wrote %s GIF -> %s (%dx%d, %d frames)", role, out_path, w, h, n_frames
        )

    return {"gif_paths": out_paths, "gif_dims": dims, "n_frames": n_frames}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def build(
    *,
    checkpoint: Path,
    shot_id: int,
    out_dir: Path,
    device: str = "cuda",
    work_dir: Path | None = None,
) -> dict:
    """Phase A (predict) -> phase B (MAGVIT2 decode) -> phase C (assemble GIFs)."""
    from imas_ambix.camdyn.reconstruction_demo import run_decode_subprocess

    out_dir = Path(out_dir)
    work_dir = work_dir or Path(
        tempfile.mkdtemp(prefix="wm-dream-", dir=os.environ.get("TMPDIR", "/tmp"))
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    token_bundle = work_dir / "tokens.npz"
    image_bundle = work_dir / "images.npz"

    summary = run_phase_a(
        checkpoint=checkpoint,
        shot_id=shot_id,
        token_bundle=token_bundle,
        device=device,
    )

    # PHASE B — decode the GT + prediction grids to 256x256 via the MAGVIT2 venv.
    logger.info("decoding token grids via the MAGVIT2 venv")
    run_decode_subprocess(token_bundle, image_bundle, "cuda")
    if not image_bundle.exists():
        raise RuntimeError(
            f"decode produced no image bundle at {image_bundle} — cannot build GIFs"
        )

    # PHASE C — assemble the GIFs.
    gif_summary = assemble_gifs(
        image_bundle=image_bundle, summary=summary, out_dir=out_dir
    )
    summary.update(gif_summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    p.add_argument("--shot", type=int, default=DEFAULT_SHOT)
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--device", default="cuda")
    p.add_argument("--work-dir", default=None)
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    summary = build(
        checkpoint=Path(args.checkpoint),
        shot_id=args.shot,
        out_dir=Path(args.out_dir),
        device=args.device,
        work_dir=Path(args.work_dir) if args.work_dir else None,
    )

    print("\n=== full-res reconstruction + dream GIF summary ===")
    print(f"checkpoint: {summary['checkpoint']} (step {summary['checkpoint_step']})")
    print(
        f"shot: {summary['shot_id']}  n_steps={summary['n_steps']} "
        f"context={summary['context_steps']}"
    )
    print(
        f"teacher-forced rbb token mismatch (target window): "
        f"{summary['teacher_forced_token_mismatch']:.4f}"
    )
    print(
        f"dream rbb token mismatch (target window): "
        f"{summary['dream_token_mismatch']:.4f}"
    )
    print(
        f"dream change-from-last-context fraction: "
        f"{summary['dream_change_fraction']:.4f} "
        f"(0=frozen/collapsed, high=evolving)"
    )
    print(f"GIFs: {summary['gif_paths']}")
    print(f"pixel dims (HxW): {summary['gif_dims']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
