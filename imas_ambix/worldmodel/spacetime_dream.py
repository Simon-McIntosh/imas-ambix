"""Decode the spatiotemporal camera transformer to GT-vs-prediction images.

Two-venv handoff (mirrors :mod:`imas_ambix.worldmodel.dream_gifs`)
------------------------------------------------------------------
PHASE A (this venv, GPU): load a spacetime checkpoint, assemble a held-out rbb
window, run the teacher-forced next-frame prediction AND the autoregressive
dream, dump a ``(N, F, 16, 16)`` STORE-id token bundle (gt / teacher_forced /
dream) in the reconstruction_demo format.
PHASE B (Open-MAGVIT2 venv, GPU):
:func:`imas_ambix.camdyn.reconstruction_demo.run_decode_subprocess` decodes the
whole stack to ``(N, F, 256, 256, 3)`` uint8 — it subtracts ``REGISTRY_OFFSET``
itself, so the token bundle MUST be STORE-ids (see
:func:`imas_ambix.worldmodel.spacetime_dataset.local_to_store`).
PHASE C (this venv): assemble side-by-side GIFs + a contact-sheet PNG.

ID space
--------
The model predicts LOCAL ids (store-id − 4); GT frames are also held LOCAL in
the sample.  Both are mapped back to STORE-id with the single ``local_to_store``
inverse before the bundle is written, so GT and prediction take an IDENTICAL
decode path and a decoded prediction is never mislabelled.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
import time
from pathlib import Path

import numpy as np

from imas_ambix.worldmodel.spacetime_dataset import (
    GRID_H,
    GRID_W,
    REFERENCE_CAMERA,
    SpacetimeWindowConfig,
    assemble_window,
    local_to_store,
)

logger = logging.getLogger(__name__)

ORIGINAL_HW = (112, 156)  # native rbb aspect for display
FPS = 8


# ---------------------------------------------------------------------------
# Phase A — predict + dump the token bundle
# ---------------------------------------------------------------------------


def run_phase_a(
    *,
    checkpoint: Path,
    shot_id: int,
    token_bundle: Path,
    window: SpacetimeWindowConfig,
    camera: str = REFERENCE_CAMERA,
    device: str = "cuda",
    token_root: Path | None = None,
) -> dict:
    """Load the model once, build GT + teacher-forced + dream STORE-id grids."""
    import torch

    from imas_ambix.worldmodel.spacetime_train import (
        autoregressive_dream,
        load_model_from_checkpoint,
        teacher_forced_frames,
    )

    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if device == "cuda" and torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    else:
        device = "cpu"

    logger.info("assembling held-out shot %s camera %s", shot_id, camera)
    sample = assemble_window(int(shot_id), window, camera=camera, token_root=token_root)

    logger.info("loading checkpoint on %s: %s", device, checkpoint)
    model, payload = load_model_from_checkpoint(checkpoint, map_location=device)
    model.eval()
    step = int(payload.get("step", -1))
    logger.info("checkpoint step=%d params=%d", step, int(model.num_parameters()))

    dev = torch.device(device)
    ctx = int(sample.context_frames)
    n_frames = int(sample.frames.shape[0])

    # GT local grids -> store-id grids.
    gt_local = np.asarray(sample.frames, dtype=np.int64).reshape(
        n_frames, GRID_H, GRID_W
    )

    t0 = time.time()
    tf_local = teacher_forced_frames(model, sample, device=dev)
    tf_local = tf_local.reshape(n_frames, GRID_H, GRID_W)
    logger.info("teacher-forced prediction in %.1fs", time.time() - t0)

    t0 = time.time()
    dream_local = autoregressive_dream(model, sample, device=dev)
    dream_local = dream_local.reshape(n_frames, GRID_H, GRID_W)
    logger.info("autoregressive dream in %.1fs", time.time() - t0)

    # honesty readouts over the forecast window.
    tf_mismatch = float((tf_local[ctx:] != gt_local[ctx:]).mean())
    dream_mismatch = float((dream_local[ctx:] != gt_local[ctx:]).mean())
    last_ctx = dream_local[ctx - 1]
    dream_change = float((dream_local[ctx:] != last_ctx[None]).mean())

    try:
        del model
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001
        logger.warning("model release note: %r", exc)

    grids_stack = np.stack(
        [
            local_to_store(gt_local),
            local_to_store(tf_local),
            local_to_store(dream_local),
        ]
    ).astype(np.int64)
    index = [
        {"role": "gt", "slot": 0},
        {"role": "teacher_forced", "slot": 1},
        {"role": "dream", "slot": 2},
    ]
    meta = {
        "format": "reconstruction_demo",
        "camera": camera,
        "id_space": "store-id (local + REGISTRY_OFFSET; decode subtracts it)",
        "grid_hw": [GRID_H, GRID_W],
        "shot_id": int(shot_id),
        "checkpoint": str(checkpoint),
        "checkpoint_step": step,
        "n_frames": int(n_frames),
        "context_frames": int(ctx),
        "frame_time": np.asarray(sample.frame_time, dtype=np.float64).tolist(),
    }
    token_bundle.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        token_bundle, grids=grids_stack, index=json.dumps(index), meta=json.dumps(meta)
    )
    logger.info("wrote token bundle -> %s", token_bundle)

    return {
        "checkpoint": str(checkpoint),
        "checkpoint_step": step,
        "shot_id": int(shot_id),
        "camera": camera,
        "n_frames": int(n_frames),
        "context_frames": int(ctx),
        "teacher_forced_token_mismatch": tf_mismatch,
        "dream_token_mismatch": dream_mismatch,
        "dream_change_fraction": dream_change,
        "frame_time": meta["frame_time"],
    }


# ---------------------------------------------------------------------------
# Pixel-space honesty metric — model vs persistence on the forecast window
# ---------------------------------------------------------------------------


def forecast_pixel_errors(
    gt: np.ndarray, pred: np.ndarray, ctx: int
) -> dict[str, float | bool]:
    """Mean decoded-pixel error of a prediction vs a persistence baseline.

    A held-out token-mismatch near the huge-vocab saturation point can coexist
    with coherent-LOOKING but non-forecasting video; the honest signal is the
    decoded image error against the trivial *persistence* baseline — freeze the
    last context frame ``gt[ctx-1]`` across the forecast window.  Only forecast
    frames (``fi >= ctx``) are scored; context frames are excluded.

    Both inputs are ``(F, H, W[, C])`` decoded image stacks on the same scale
    (typically uint8 256x256x3).  Returns the mean absolute pixel error for the
    model and for persistence, their ratio (model / persistence), and whether
    the model beats persistence (lower error is better).
    """
    g = np.asarray(gt, dtype=np.float64)
    p = np.asarray(pred, dtype=np.float64)
    if g.shape[0] != p.shape[0]:
        raise ValueError(f"frame-count mismatch: gt {g.shape[0]} vs pred {p.shape[0]}")
    if not 1 <= ctx < g.shape[0]:
        raise ValueError(f"ctx {ctx} out of range for {g.shape[0]} frames")

    fc = slice(ctx, g.shape[0])
    persistence = g[ctx - 1]  # last observed frame, frozen across the forecast
    model_err = float(np.abs(p[fc] - g[fc]).mean())
    pers_err = float(np.abs(persistence[None] - g[fc]).mean())
    ratio = float("inf") if pers_err == 0.0 else model_err / pers_err
    return {
        "model_pixel_error": model_err,
        "persistence_pixel_error": pers_err,
        "ratio": ratio,
        "model_beats_persistence": bool(model_err < pers_err),
    }


# ---------------------------------------------------------------------------
# Phase C — assemble the figures
# ---------------------------------------------------------------------------


def _to_aspect(img_square: np.ndarray) -> np.ndarray:
    from PIL import Image

    if img_square.ndim == 3:
        img_square = img_square[..., 0]
    im = Image.fromarray(img_square.astype(np.uint8)).resize(
        (ORIGINAL_HW[1], ORIGINAL_HW[0]), Image.BILINEAR
    )
    return np.asarray(im)


def _panel_frame(gt_img, pred_img, *, left_title, right_title, banner, in_target):
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


def _save_gif(frames, out_path: Path, *, fps: int = FPS):
    from PIL import Image

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil = [Image.fromarray(np.asarray(f, dtype=np.uint8)) for f in frames]
    pil[0].save(
        str(out_path),
        format="GIF",
        save_all=True,
        append_images=pil[1:],
        duration=int(round(1000.0 / fps)),
        loop=0,
        optimize=False,
    )
    return frames[0].shape[:2]


def _contact_sheet(
    gt, pred, *, ctx: int, role: str, step: int, shot_id: int, out_path: Path
):
    """A single PNG: rows = GT / prediction, columns = a spread of frames.

    Gives an at-a-glance still even where a GIF cannot render (the plan doc).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = gt.shape[0]
    n_col = min(8, n)
    cols = np.linspace(0, n - 1, n_col).round().astype(int)
    fig, axes = plt.subplots(2, n_col, figsize=(1.5 * n_col, 3.2), dpi=120)
    if n_col == 1:
        axes = axes.reshape(2, 1)
    for j, fi in enumerate(cols):
        for r, (img_stack, lbl) in enumerate(((gt, "GT"), (pred, role))):
            ax = axes[r, j]
            ax.imshow(
                _to_aspect(img_stack[fi]),
                cmap="inferno",
                vmin=0,
                vmax=255,
                interpolation="nearest",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            tgt = fi >= ctx
            if r == 0:
                ax.set_title(
                    f"f{fi}{' (fc)' if tgt else ''}",
                    fontsize=7,
                    color="#d62728" if tgt else "#222",
                )
            if j == 0:
                ax.set_ylabel(lbl, fontsize=9)
    fig.suptitle(
        f"spacetime {role} | shot {shot_id} | rbb full-res decode | step {step:,}",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path))
    plt.close(fig)


def assemble_figures(*, image_bundle: Path, summary: dict, out_dir: Path) -> dict:
    data = np.load(str(image_bundle), allow_pickle=True)
    images = np.asarray(data["images"], dtype=np.uint8)  # (N,F,256,256,3)
    index = json.loads(str(data["index"]))
    slot = {e["role"]: e["slot"] for e in index}
    gt, tf, dream = (
        images[slot["gt"]],
        images[slot["teacher_forced"]],
        images[slot["dream"]],
    )

    ctx = int(summary["context_frames"])
    n_frames = int(summary["n_frames"])
    step = int(summary["checkpoint_step"])
    shot_id = int(summary["shot_id"])
    ftime = np.asarray(summary["frame_time"], dtype=np.float64)
    n = min(gt.shape[0], n_frames)

    # Honest verdict: decoded-pixel error vs persistence on the forecast window.
    # The token-mismatch can saturate while the video still looks coherent; this
    # is the metric that distinguishes a forecaster from a coherent generator.
    summary["dream_pixel"] = forecast_pixel_errors(gt[:n], dream[:n], ctx)
    summary["teacher_forced_pixel"] = forecast_pixel_errors(gt[:n], tf[:n], ctx)

    out_paths: dict[str, str] = {}
    for role, pred_stack, fname, right in (
        (
            "teacher_forced",
            tf,
            "spacetime-reconstruction.gif",
            "model (teacher-forced)",
        ),
        ("dream", dream, "spacetime-dream.gif", "model (dream / autoregressive)"),
    ):
        frames = []
        for fi in range(n):
            in_target = fi >= ctx
            t_ms = float(ftime[fi] - ftime[ctx]) * 1e3 if ftime.size > ctx else 0.0
            banner = f"spacetime {role} | shot {shot_id} | rbb full-res | step {step:,}"
            frames.append(
                _panel_frame(
                    _to_aspect(gt[fi]),
                    _to_aspect(pred_stack[fi]),
                    left_title=f"ground truth   t={t_ms:+.0f} ms",
                    right_title=f"{right}   [{'FORECAST' if in_target else 'context'}]",
                    banner=banner,
                    in_target=in_target,
                )
            )
        gif = out_dir / fname
        _save_gif(frames, gif)
        out_paths[role] = str(gif)
        png = out_dir / fname.replace(".gif", ".png")
        _contact_sheet(
            gt[:n],
            pred_stack[:n],
            ctx=ctx,
            role=role,
            step=step,
            shot_id=shot_id,
            out_path=png,
        )
        out_paths[role + "_png"] = str(png)
        logger.info("wrote %s -> %s (+ %s)", role, gif, png)
    return {"figure_paths": out_paths, "n_frames": n}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def build(
    *,
    checkpoint: Path,
    shot_id: int,
    out_dir: Path,
    window: SpacetimeWindowConfig,
    camera: str = REFERENCE_CAMERA,
    device: str = "cuda",
    token_root: Path | None = None,
    work_dir: Path | None = None,
) -> dict:
    from imas_ambix.camdyn.reconstruction_demo import run_decode_subprocess

    out_dir = Path(out_dir)
    work_dir = work_dir or Path(
        tempfile.mkdtemp(prefix="st-dream-", dir=os.environ.get("TMPDIR", "/tmp"))
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    token_bundle = work_dir / "tokens.npz"
    image_bundle = work_dir / "images.npz"

    summary = run_phase_a(
        checkpoint=checkpoint,
        shot_id=shot_id,
        token_bundle=token_bundle,
        window=window,
        camera=camera,
        device=device,
        token_root=token_root,
    )
    logger.info("decoding token grids via the MAGVIT2 venv")
    run_decode_subprocess(token_bundle, image_bundle, "cuda")
    if not image_bundle.exists():
        raise RuntimeError(f"decode produced no image bundle at {image_bundle}")
    summary.update(
        assemble_figures(image_bundle=image_bundle, summary=summary, out_dir=out_dir)
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--shot", type=int, required=True)
    p.add_argument(
        "--out-dir", default="/work/projects/imas_gpu/worldmodel/spacetime_smoke"
    )
    p.add_argument("--camera", default=REFERENCE_CAMERA)
    p.add_argument("--n-frames", type=int, default=24)
    p.add_argument("--n-plan", type=int, default=8)
    p.add_argument("--context-frames", type=int, default=8)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--token-root", default=None)
    p.add_argument("--work-dir", default=None)
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    window = SpacetimeWindowConfig(
        n_frames=args.n_frames,
        n_plan=args.n_plan,
        context_frames=args.context_frames,
        frame_stride=args.frame_stride,
    )
    summary = build(
        checkpoint=Path(args.checkpoint),
        shot_id=args.shot,
        out_dir=Path(args.out_dir),
        window=window,
        camera=args.camera,
        device=args.device,
        token_root=Path(args.token_root) if args.token_root else None,
        work_dir=Path(args.work_dir) if args.work_dir else None,
    )
    print("\n=== spacetime dream summary ===")
    print(f"checkpoint: {summary['checkpoint']} (step {summary['checkpoint_step']})")
    print(
        f"shot {summary['shot_id']} n_frames={summary['n_frames']} "
        f"ctx={summary['context_frames']}"
    )
    print(
        "teacher-forced token mismatch (forecast): "
        f"{summary['teacher_forced_token_mismatch']:.4f}"
    )
    print(f"dream token mismatch (forecast): {summary['dream_token_mismatch']:.4f}")
    print(f"dream change-from-last-context: {summary['dream_change_fraction']:.4f}")
    dp = summary.get("dream_pixel")
    if dp is not None:
        verdict = (
            "BEATS persistence"
            if dp["model_beats_persistence"]
            else "LOSES to persistence"
        )
        print(
            "dream pixel-error vs persistence (forecast): "
            f"model={dp['model_pixel_error']:.3f}  "
            f"persistence={dp['persistence_pixel_error']:.3f}  "
            f"ratio={dp['ratio']:.2f}x  -> {verdict}"
        )
    print(f"figures: {summary['figure_paths']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
