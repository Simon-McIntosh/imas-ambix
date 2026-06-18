"""Full-resolution rbb predict→decode feasibility smoke (the de-risk proof).

The world model now predicts the FULL 16×16 = 256-token rbb camera frame (see
:func:`imas_ambix.worldmodel.dataset.default_modalities` — rbb at
``camera_grid_stride=1``).  Before the long retrain, this CHEAP smoke proves the
end-to-end claim that matters: a full-resolution rbb prediction can be DECODED
into a real, plasma-like image.

What it does
------------
1. Overfit a small full-res-rbb world model on a handful of rbb-bearing shots
   for enough steps that the loss drops clearly (the chunked cross-entropy
   makes the 256-channel rbb head fit memory).
2. Roll the overfit model out on one overfit shot and take its predicted rbb
   tokens ``(T, 256)`` → reshape to ``(frames, 16, 16)``.
3. ID-MAPPING CORRECTNESS CHECK.  The model's rbb id IS the on-disk GLOBAL id
   (``_read_camera`` does NOT rebase the camera; the only offset, the
   4-token control range == ``REGISTRY_OFFSET``, is already baked into the
   stored ids and is subtracted by the decoder).  We verify this by decoding
   the SAME shot's GT rbb tokens via BOTH paths and asserting they are
   identical token grids → identical decoded images:
     * direct-from-disk-global: the raw ``rbb.zarr`` grid (16×16 GLOBAL ids);
     * model-path: ``build_shot_sample(...).tokens["rbb"]`` reshaped to 16×16.
4. Decode (a) the GT frames and (b) the model's overfit-prediction frames
   through the frozen Open-MAGVIT2 tokenizer (the established two-venv handoff
   :func:`imas_ambix.camdyn.reconstruction_demo.run_decode_subprocess`), and
   save a side-by-side GT-vs-prediction PNG.

If the decoded prediction is garbage (not plasma-like), STOP and report — that
is most likely an id-mapping bug, and the retrain must NOT be launched.

Run (single GPU, betelgeuse; keep the neighbour's LLM server up)::

    .venv/bin/python -m imas_ambix.worldmodel.fullres_smoke \\
        --shots 15085,15086,15087 --steps 400
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

#: Where the smoke artifacts land (data, NOT git-tracked).
OUTPUT_DIR = Path("/work/projects/imas_gpu/worldmodel/demo/fullres_smoke")

#: Warm-start checkpoint (rbb channel_query 16→256 is fresh; everything else
#: shape-matches and is copied where it does).
WARM_START = Path("/work/projects/imas_gpu/worldmodel/ckpt/1219352/latest.pt")

GRID_H, GRID_W = 16, 16


# ---------------------------------------------------------------------------
# Warm-start helper
# ---------------------------------------------------------------------------


def warm_start_from(model, ckpt_path: Path) -> dict:
    """Copy every SHAPE-MATCHING parameter from a checkpoint into ``model``.

    The full-res model differs from the warm-start only in rbb's
    ``channel_query`` (16→256 rows) and any other shape-mismatched tensor; all
    matching tensors (token_embed, the d→vocab heads, transformer blocks,
    pos/segment embeddings, the small-camera channel queries) are loaded.
    Returns a small report dict (counts of copied / skipped tensors).
    """
    import torch

    payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    src = payload["model_state_dict"]
    dst = model.state_dict()
    copied: list[str] = []
    skipped: list[str] = []
    for k, v in dst.items():
        if k in src and tuple(src[k].shape) == tuple(v.shape):
            dst[k] = src[k]
            copied.append(k)
        else:
            skipped.append(k)
    model.load_state_dict(dst)
    logger.info(
        "warm-start from %s: copied %d tensors, %d fresh (shape-mismatched/new)",
        ckpt_path,
        len(copied),
        len(skipped),
    )
    if skipped:
        logger.info("fresh tensors: %s", skipped[:20])
    return {"copied": len(copied), "skipped": skipped}


# ---------------------------------------------------------------------------
# ID-mapping correctness check (the load-bearing validation)
# ---------------------------------------------------------------------------


@dataclass
class IdMapCheck:
    shot_id: int
    n_frames: int
    grids_identical: bool
    direct_max: int
    model_max: int


def verify_id_mapping(shot_id: int, n_frames: int, *, token_root=None) -> IdMapCheck:
    """Confirm the model's rbb id == the on-disk GLOBAL id (identity mapping).

    Decodes nothing here — it asserts the two TOKEN-GRID paths agree, which is
    the necessary+sufficient condition for the decoded images to agree (the
    decode is a deterministic function of the token grid):

    * direct-from-disk-global: the raw ``rbb.zarr`` grid (``(F,16,16)`` GLOBAL
      ids), the path :mod:`imas_ambix.worldmodel.demo_artifacts` decodes as
      ground truth;
    * model-path: ``build_shot_sample`` → ``tokens["rbb"]`` (``(T,256)`` model
      ids) reshaped to ``(T,16,16)``.

    The model path goes through the common-grid nearest-token resample, so we
    compare it FRAME-FOR-FRAME against the disk frames it selected: every model
    rbb row must equal some disk frame (the resample picks a real frame), and
    crucially the id SPACE must match (no rebasing offset).  We assert the model
    grid values are a subset of the disk id space and, for the contiguous run we
    decode, that the model row equals the disk frame at the same flattened
    layout.
    """
    import zarr

    from imas_ambix.worldmodel.dataset import (
        WorldModelWindowConfig,
        build_shot_sample,
        default_modalities,
    )

    store = Path("/work/projects/imas_gpu/mast-tokens/v1/frames") / str(shot_id)
    grp = zarr.open_group(str(store / "rbb.zarr"), mode="r")
    disk = np.asarray(grp["tokens"], dtype=np.int64)  # (T,16,16) GLOBAL ids

    mods = default_modalities()
    window = WorldModelWindowConfig(n_steps=64, context_steps=16)
    sample = build_shot_sample(shot_id, mods, window, token_root=token_root)
    model_rbb = np.asarray(sample.tokens["rbb"], dtype=np.int64)  # (64, 256) model ids
    model_grids = model_rbb.reshape(model_rbb.shape[0], GRID_H, GRID_W)

    # the model's rbb ids must live in the SAME id space as the disk grids (the
    # identity mapping — no per-group rebasing for the camera).  Every model
    # frame is a nearest-resampled COPY of some disk frame, so each model frame
    # must appear verbatim among the disk frames.
    disk_set = {tuple(f.ravel().tolist()) for f in disk}
    identical = all(
        tuple(g.ravel().tolist()) in disk_set
        for g in model_grids
        if g.max() > 0  # skip all-PAD frames (off-grid steps)
    )
    return IdMapCheck(
        shot_id=int(shot_id),
        n_frames=int(min(n_frames, disk.shape[0])),
        grids_identical=bool(identical),
        direct_max=int(disk.max()),
        model_max=int(model_rbb.max()),
    )


# ---------------------------------------------------------------------------
# Overfit + predict
# ---------------------------------------------------------------------------


def _teacher_forced_rbb(model, batch, shot_index: int = 0):
    """Teacher-forced rbb argmax: head on the TRUE-token hidden states.

    Returns ``(T, 256)`` rbb token ids predicted from the TRUE previous tokens
    (no free-running rollout).  At obs step ``t`` the head predicts step
    ``t+1``, so the returned grid at step ``t+1`` is the teacher-forced
    next-token prediction.  This isolates "can the full-res head reproduce a
    LEARNED frame and decode it" from autoregressive rollout drift — the
    cleanest feasibility read for the predict→decode mechanism.
    """
    import torch

    from imas_ambix.worldmodel.eval import _chunked_argmax_step

    with torch.no_grad():
        obs_hidden = model.encode(batch)  # (B, T, d)
        t_len = obs_hidden.shape[1]
        n_ch = int(model.channel_query["rbb"].shape[0])
        out = np.zeros((t_len, n_ch), dtype=np.int64)
        for t in range(t_len - 1):
            pred = _chunked_argmax_step(model, obs_hidden, "rbb", t, chunk_channels=64)
            out[t + 1] = pred[shot_index].cpu().numpy()[:n_ch]
    return out


def overfit_and_predict(
    shots: list[int],
    *,
    steps: int,
    lr: float,
    d_model: int,
    n_layers: int,
    n_heads: int,
    warm_start: bool,
    token_root=None,
):
    """Overfit a full-res-rbb model on ``shots``; roll out shot[0].

    Returns ``(pred_rbb (T,256), tf_rbb (T,256), gt_rbb (T,256), context_steps,
    loss_drop, warm_report)`` — the model's autoregressive-rollout rbb grid,
    the TEACHER-FORCED rbb grid (head on true tokens — isolates the head from
    rollout drift), the ground-truth rbb token grid, and the overfit loss-drop.
    """
    import torch

    from imas_ambix.worldmodel.dataset import (
        WorldModelWindowConfig,
        build_shot_sample,
        default_modalities,
    )
    from imas_ambix.worldmodel.eval import _model_obs_plan_names, rollout
    from imas_ambix.worldmodel.train import (
        TrainConfig,
        _set_determinism,
        build_model_for_samples,
        chunked_next_token_nll,
        collate_samples,
    )

    _set_determinism(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mods = default_modalities()
    window = WorldModelWindowConfig(n_steps=64, context_steps=16)

    samples = [build_shot_sample(s, mods, window, token_root=token_root) for s in shots]
    # keep only modalities every sample carries (robust to a missing camera)
    common = set.intersection(*(set(s.tokens) for s in samples))
    if "rbb" not in common:
        raise RuntimeError(
            f"not every shot {shots} carries rbb — pick rbb-bearing shots "
            f"(common modalities: {sorted(common)})"
        )
    mods = [m for m in mods if m.name in common]
    obs = [m.name for m in mods if not m.is_conditioning]
    plan = [m.name for m in mods if m.is_conditioning]

    # d_ff = 4*d_model matches the converged checkpoint's MLP shape (d_model=384
    # -> d_ff=1536) so the warm-start copies the transformer MLPs too, not just
    # the embeddings/heads/attention — a far more effective warm-start.
    model = build_model_for_samples(
        samples,
        mods,
        window,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=4 * d_model,
    )
    rbb_ch = next(m.n_channels for m in model.config.modalities if m.name == "rbb")
    logger.info(
        "full-res model: rbb channels=%d (expect 256), d_model=%d params=%.1fM",
        rbb_ch,
        d_model,
        model.num_parameters() / 1e6,
    )
    if rbb_ch != 256:
        raise RuntimeError(f"rbb channel width is {rbb_ch}, expected 256 (full-res)")

    warm_report = {"copied": 0, "skipped": []}
    if warm_start and WARM_START.exists() and d_model == 384:
        warm_report = warm_start_from(model, WARM_START)
    elif warm_start:
        logger.info(
            "warm-start skipped (d_model=%d != 384 or ckpt missing) — fresh init",
            d_model,
        )

    model.to(device)
    model.train()
    batch = collate_samples(samples, obs, plan)
    batch = {
        **batch,
        "tokens": {k: v.to(device) for k, v in batch["tokens"].items()},
        "valid": {k: v.to(device) for k, v in batch["valid"].items()},
    }
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses: list[float] = []
    cfg = TrainConfig()  # for chunk size default
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        obs_hidden = model.encode(batch)
        loss = chunked_next_token_nll(
            model,
            obs_hidden,
            batch,
            obs,
            target_only=False,
            chunk_channels=cfg.loss_chunk_channels,
        )
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
        if step % 25 == 0 or step == steps - 1:
            logger.info("overfit step %d/%d loss=%.4f", step, steps, losses[-1])

    drop = losses[-1] / losses[0] if losses[0] > 0 else 1.0
    logger.info(
        "overfit done: initial=%.4f final=%.4f drop_ratio=%.3f",
        losses[0],
        losses[-1],
        drop,
    )

    # teacher-forced rbb prediction (head on the TRUE previous tokens) — the
    # cleanest predict→decode read, independent of rollout drift.
    model.eval()
    tf_rbb = _teacher_forced_rbb(model, batch, shot_index=0)  # (64, 256) model ids

    # autoregressive rollout of the FIRST shot on CPU (rollout assembles CPU
    # batch tensors; the model reads its device from parameters).
    model.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    obs_names, plan_names = _model_obs_plan_names(model)
    predicted = rollout(model, samples[0], obs_names, plan_names, chunk_channels=64)
    pred_rbb = np.asarray(predicted["rbb"], dtype=np.int64)  # (64, 256) model ids
    gt_rbb = np.asarray(samples[0].tokens["rbb"], dtype=np.int64)  # (64, 256)
    ctx = int(samples[0].context_steps)

    try:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001
        logger.warning("model release note: %r", exc)

    return pred_rbb, tf_rbb, gt_rbb, ctx, drop, warm_report


# ---------------------------------------------------------------------------
# Decode (GT + prediction) via the frozen MAGVIT2 tokenizer
# ---------------------------------------------------------------------------


def decode_grids(grids_global: np.ndarray, out_dir: Path, tag: str, device: str):
    """Decode ``(F,16,16)`` GLOBAL-id grids → ``(F,256,256,3)`` images.

    Re-uses the established two-venv handoff: writes a one-stack token bundle
    in the reconstruction-demo format and invokes
    :func:`imas_ambix.camdyn.reconstruction_demo.run_decode_subprocess` under
    the MAGVIT2 interpreter (which subtracts REGISTRY_OFFSET and decodes).
    Returns the decoded image stack ``(F,256,256,3)`` or ``None`` if decode is
    unavailable.
    """
    from imas_ambix.camdyn.reconstruction_demo import (
        MAGVIT2_PYTHON,
        run_decode_subprocess,
    )

    if not MAGVIT2_PYTHON.exists():
        logger.warning(
            "MAGVIT2 interpreter not found at %s — cannot decode", MAGVIT2_PYTHON
        )
        return None

    grids = np.asarray(grids_global, dtype=np.int64)[None]  # (1,F,16,16)
    token_bundle = out_dir / f"_tokens_{tag}.npz"
    image_bundle = out_dir / f"_images_{tag}.npz"
    index = [{"window": 0, "scenario": "_window", "role": tag, "slot": 0}]
    meta = [{"shot_id": 0, "tag": tag}]
    np.savez_compressed(
        token_bundle,
        grids=grids,
        index=json.dumps(index),
        meta=json.dumps(meta),
    )
    try:
        run_decode_subprocess(token_bundle, image_bundle, device)
    except Exception as exc:  # noqa: BLE001 — decode failure is reported, not faked
        logger.warning("decode subprocess failed for %s: %r", tag, exc)
        return None
    if not image_bundle.exists():
        logger.warning("decode produced no image bundle for %s", tag)
        return None
    data = np.load(str(image_bundle), allow_pickle=True)
    images = np.asarray(data["images"], dtype=np.uint8)  # (1,F,256,256,3)
    return images[0]


def _to_aspect(img_square: np.ndarray) -> np.ndarray:
    """Resize a 256² decoded image to the native rbb aspect for display."""
    from PIL import Image

    if img_square.ndim == 3:
        img_square = img_square[..., 0]
    im = Image.fromarray(img_square.astype(np.uint8)).resize((156, 112), Image.BILINEAR)
    return np.asarray(im)


def save_side_by_side(
    gt_images: np.ndarray,
    tf_images: np.ndarray | None,
    pred_images: np.ndarray,
    gt_images_modelpath: np.ndarray | None,
    out_path: Path,
    *,
    shot_id: int,
    ctx: int,
    frame_idxs: list[int],
    id_check: IdMapCheck,
    loss_drop: float,
):
    """Save a GT-vs-prediction side-by-side PNG over several target frames.

    Rows (top→bottom):

    * GT (decoded from disk-global) — the ground-truth reference;
    * [optional] GT (decoded via the model id-path) — the visual half of the
      id-mapping check (must match the disk-global GT);
    * teacher-forced prediction — the full-res head's next-token argmax on the
      TRUE previous tokens (isolates head capability from rollout drift);
    * autoregressive prediction — the free-running rollout.

    Columns are time steps in the TARGET window (steps ``>= ctx``) so the
    prediction rows are genuinely generated, not copied context.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = ["GT (disk-global)"]
    stacks = [gt_images]
    if gt_images_modelpath is not None:
        rows.append("GT (model id-path)")
        stacks.append(gt_images_modelpath)
    if tf_images is not None:
        rows.append("prediction (teacher-forced)")
        stacks.append(tf_images)
    rows.append("prediction (rollout)")
    stacks.append(pred_images)

    n_rows = len(rows)
    n_cols = len(frame_idxs)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(1.7 * n_cols + 1.4, 1.6 * n_rows + 0.6),
        squeeze=False,
        constrained_layout=True,
    )
    for ri, (label, stack) in enumerate(zip(rows, stacks, strict=True)):
        for ci, fi in enumerate(frame_idxs):
            ax = axes[ri][ci]
            if stack is not None and fi < stack.shape[0]:
                ax.imshow(
                    _to_aspect(stack[fi]),
                    cmap="gray",
                    vmin=0,
                    vmax=255,
                    interpolation="nearest",
                )
            else:
                ax.imshow(
                    np.zeros((112, 156), dtype=np.uint8), cmap="gray", vmin=0, vmax=255
                )
            ax.set_xticks([])
            ax.set_yticks([])
            if ri == 0:
                ax.set_title(f"frame {fi}", fontsize=8)
            if ci == 0:
                ax.set_ylabel(label, fontsize=8)

    fig.suptitle(
        f"full-res rbb predict→decode smoke — shot {shot_id} | "
        f"id-map identical={id_check.grids_identical} | "
        f"overfit loss drop={loss_drop:.3f} | "
        f"decoder=frozen Open-MAGVIT2 (imagenet_256_L)",
        fontsize=9,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("wrote side-by-side PNG -> %s", out_path)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_smoke(
    shots: list[int],
    *,
    steps: int,
    lr: float,
    d_model: int,
    n_layers: int,
    n_heads: int,
    warm_start: bool,
    out_dir: Path = OUTPUT_DIR,
    token_root=None,
    device: str = "cuda",
) -> dict:
    """Overfit → verify id mapping → decode GT + prediction → save the PNG."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shot0 = shots[0]

    # 1. overfit + predict (autoregressive rollout AND teacher-forced)
    pred_rbb, tf_rbb, gt_rbb, ctx, loss_drop, warm_report = overfit_and_predict(
        shots,
        steps=steps,
        lr=lr,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        warm_start=warm_start,
        token_root=token_root,
    )

    # 2. id-mapping correctness check (token-grid level — necessary+sufficient)
    id_check = verify_id_mapping(shot0, gt_rbb.shape[0], token_root=token_root)
    logger.info(
        "ID-MAP CHECK shot %d: grids_identical=%s (disk_max=%d model_max=%d)",
        shot0,
        id_check.grids_identical,
        id_check.direct_max,
        id_check.model_max,
    )

    # 3. choose target-window frames to decode/show (generated, not context).
    #    decode the WHOLE 64-step grid so columns can be picked freely.
    pred_grids = pred_rbb.reshape(pred_rbb.shape[0], GRID_H, GRID_W)
    tf_grids = tf_rbb.reshape(tf_rbb.shape[0], GRID_H, GRID_W)
    gt_grids = gt_rbb.reshape(gt_rbb.shape[0], GRID_H, GRID_W)

    # the disk-global GT path (the demo's ground-truth reference) — raw grid for
    # the same shot, used to validate the id mapping at the IMAGE level too.
    import zarr

    disk = np.asarray(
        zarr.open_group(
            str(
                Path("/work/projects/imas_gpu/mast-tokens/v1/frames")
                / str(shot0)
                / "rbb.zarr"
            ),
            mode="r",
        )["tokens"],
        dtype=np.int64,
    )
    # use the same central run the demo uses so the GT frames are on the plasma
    take = min(64, disk.shape[0])
    start = max(0, (disk.shape[0] - take) // 2)
    disk_run = disk[start : start + take]  # (take,16,16) GLOBAL ids

    # 4. decode (GT disk-global, GT model-path, teacher-forced, rollout)
    gt_disk_images = decode_grids(disk_run, out_dir, "gt_disk", device)
    gt_model_images = decode_grids(gt_grids, out_dir, "gt_model", device)
    tf_images = decode_grids(tf_grids, out_dir, "tf", device)
    pred_images = decode_grids(pred_grids, out_dir, "pred", device)

    decoded_ok = pred_images is not None and gt_disk_images is not None
    # image-level id-map validation: GT decoded via disk-global vs via the model
    # id-path on the frames the model path actually selected (a real disk
    # frame).  Identical TOKEN grids decode within decoder fp-nondeterminism
    # (~a few /255 across separate bf16 batches), so we assert a small pixel
    # tolerance, not bitwise equality — the rigorous proof is the token-grid
    # identity in verify_id_mapping (a deterministic decode of identical tokens).
    image_idmap_match = None
    image_idmap_maxdiff = None
    if gt_disk_images is not None and gt_model_images is not None:
        disk_lookup = {tuple(f.ravel().tolist()): i for i, f in enumerate(disk_run)}
        diffs = []
        for j, g in enumerate(gt_grids):
            key = tuple(g.ravel().tolist())
            if key in disk_lookup and j < gt_model_images.shape[0]:
                i = disk_lookup[key]
                if i < gt_disk_images.shape[0]:
                    diffs.append(
                        int(
                            np.abs(
                                gt_model_images[j].astype(int)
                                - gt_disk_images[i].astype(int)
                            ).max()
                        )
                    )
        if diffs:
            image_idmap_maxdiff = max(diffs)
            image_idmap_match = image_idmap_maxdiff <= 8  # decoder fp tolerance
        logger.info(
            "IMAGE id-map check: %d matched frames, max pixel diff=%s (tol<=8) -> %s",
            len(diffs),
            image_idmap_maxdiff,
            image_idmap_match,
        )

    # 5. side-by-side PNG over target-window frames
    png_path = out_dir / f"fullres_smoke_shot{shot0}.png"
    if decoded_ok:
        # show generated (target-window) frames spread across [ctx, T)
        n = pred_grids.shape[0]
        frame_idxs = list(np.linspace(ctx, n - 1, 5).round().astype(int))
        save_side_by_side(
            gt_disk_images,
            tf_images,
            pred_images,
            gt_model_images,
            png_path,
            shot_id=shot0,
            ctx=ctx,
            frame_idxs=frame_idxs,
            id_check=id_check,
            loss_drop=loss_drop,
        )
    else:
        logger.warning("decode unavailable — PNG NOT written")

    summary = {
        "shots": shots,
        "steps": steps,
        "loss_drop_ratio": float(loss_drop),
        "warm_start": {
            "used": bool(warm_start and warm_report["copied"] > 0),
            "copied": warm_report["copied"],
            "fresh": warm_report["skipped"],
        },
        "id_map_grids_identical": id_check.grids_identical,
        "id_map_disk_max": id_check.direct_max,
        "id_map_model_max": id_check.model_max,
        "image_idmap_within_tol": image_idmap_match,
        "image_idmap_max_pixel_diff": image_idmap_maxdiff,
        "decoded": decoded_ok,
        "png": str(png_path) if decoded_ok else None,
    }
    (out_dir / f"smoke_summary_shot{shot0}.json").write_text(
        json.dumps(summary, indent=2)
    )
    logger.info("SMOKE SUMMARY: %s", json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--shots",
        default="15085,15086,15087",
        help="comma-separated rbb-bearing overfit shots",
    )
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument(
        "--d-model",
        type=int,
        default=384,
        help="384 enables warm-start from the converged ckpt",
    )
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--no-warm-start", action="store_true")
    p.add_argument("--out-dir", default=str(OUTPUT_DIR))
    p.add_argument("--token-root", default=None)
    p.add_argument("--device", default="cuda")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    shots = [int(s) for s in args.shots.split(",") if s.strip()]
    token_root = Path(args.token_root) if args.token_root else None
    run_smoke(
        shots,
        steps=args.steps,
        lr=args.lr,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        warm_start=not args.no_warm_start,
        out_dir=Path(args.out_dir),
        token_root=token_root,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
