"""Decoded camera-image demonstration grids for the camera-dynamics arm.

This renders the qualitative companion to the paired :mod:`arm_compare`
verdict: image grids that show, for a held-out window, what the trained
**dynamics** arm reconstructs versus the trivial **zero-order-hold** (ZOH)
predictor versus the ground truth — decoded back to camera images through
the frozen Open-MAGVIT2 tokenizer.

Each figure is a grid whose rows are
(top→bottom)::

    ground truth (raw camera frames)
    [optional] ground truth (decoded from the true tokens)
    zero-order hold     (carry-forward of the last OBSERVED token, decoded)
    dynamics prediction (bit-head MAP token ids, decoded)

and whose COLUMNS increase in time, labelled with the time offset (ms)
of each shown frame relative to the window/frontier start.

Three scenarios::

    frontier      temporal-frontier: the model sees the FULL frames up to a
                  mid-window frontier (frame 8 of 16) and must forecast the
                  rest.  ZOH = persistence of the last pre-frontier frame.
    clipped       a frozen named-geometry clip (``fixed_section2``): the
                  model sees only a small sub-window stream and must emit the
                  full frame.  The visible clip box is outlined on every
                  image.  ZOH = carry-forward of ever-observed cells (mostly
                  empty outside the clip — that contrast is the point).
    signals_only  full mask: no camera input at all — reconstruct from the
                  actuator conditioning alone.  ZOH row is fully empty.

The dynamics-arm prediction is the bit-head MAP token id (per cell,
``id = Σ_b (logit_b > 0) << b`` — the model's exact most-likely token,
see :func:`imas_ambix.camdyn.model.score_window_bits`).

Decode architecture
--------------------
Token → image decoding goes through the SAME frozen VQModel the corpus
encoder and the bench use (``imagenet_256_L``, NEVER fine-tuned).  That
model lives in a separate venv (the Open-MAGVIT2 source + weights), so the
demo runs in two phases:

  1. **Predict phase** (this package's venv, on GPU): load the dynamics
     checkpoint, materialise the held-out windows, run the model forward,
     compute the predicted / ZOH / true token grids, and dump them to an
     ``.npz`` token bundle (global ids, the model's native space).
  2. **Decode phase** (the Open-MAGVIT2 venv): re-invoke this module's
     ``--decode-phase`` entry point with the magvit2 interpreter; it loads
     the VQModel once, decodes every unique token grid to a 256² image,
     and writes a decoded-image bundle.

The predict phase then loads the decoded bundle + the raw level-1 frames
and lays out the figures.  ZOH/never-observed cells (id ``< 0``) are
rendered as black tiles in the token grid before decoding, and the caption
flags that the decoder is the frozen baseline tokenizer (so the reader
separates tokenizer reconstruction loss from prediction loss).

Run (predict + decode + figures, on a GPU node)::

    .venv/bin/python -m imas_ambix.camdyn.reconstruction_demo \\
        --out docs/figures/camera-dynamics-wm/
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Locations of the frozen tokenizer (decode phase) — mirror the bench config.
# ---------------------------------------------------------------------------

MAGVIT2_PYTHON = Path(
    "/work/projects/imas_gpu/mast-tokens/v1/open-magvit2/.venv/bin/python"
)
MAGVIT2_ROOT = Path("/work/projects/imas_gpu/mast-tokens/v1/open-magvit2")
DYNAMICS_CKPT = Path(
    "/work/projects/imas_gpu/mast-checkpoints/camdyn/cap_v1_dynamics/final.pt"
)
#: The TRAINED per-frame baseline arm (temporal attention OFF).  This is the
#: STRONG visual baseline — a real 202M model that emits coherent full frames
#: — not the trivial zero-order-hold floor.
BASELINE_CKPT = Path(
    "/work/projects/imas_gpu/mast-checkpoints/camdyn/cap_v1_baseline/final.pt"
)

#: Registry shift between the stored (global) token ids and the local LFQ
#: codebook ids the VQModel decoder expects (``len(CONTROL_TOKENS) == 4``;
#: see :data:`imas_ambix.data.stream_encode.REGISTRY_OFFSET`).
REGISTRY_OFFSET = 4
GRID_H, GRID_W = 16, 16

#: Accent colour for the visible-clip outline / frontier marker.
ACCENT = "#d62728"

# ---------------------------------------------------------------------------
# Window selection
# ---------------------------------------------------------------------------

#: Reference shot (the report's headline shot) + two more high-activity
#: held-out shots, all present in ``camdyn_split_v0.json`` held_out.  All
#: three have an established flat-top with a bright, structured plasma
#: image (mean raw intensity > 180 in the selected window).
DEMO_SHOTS = (24065, 24446, 23937)


@dataclass
class DemoWindow:
    """One selected held-out window + the per-scenario token grids.

    All token arrays are GLOBAL ids (the model's native space, with the
    registry offset); the decode phase subtracts the offset.  A value
    ``< 0`` marks a cell with no prediction (ZOH never-observed) and is
    rendered as a black tile.
    """

    shot_id: int
    start: int
    frame_time: np.ndarray  # (F,) s
    dt: np.ndarray  # (F,) s
    valid: np.ndarray  # (F,) bool
    true_tokens: np.ndarray  # (F,H,W) int  global ids
    motion_fraction: float


def _motion_fraction(tokens: np.ndarray, frame_time: np.ndarray) -> float:
    """Mean fraction of moving (id-changing) cells over a window."""
    from imas_ambix.camdyn.metrics import motion_weighted_subset

    moving = motion_weighted_subset(tokens, frame_time)
    return float(moving.mean())


def _window_brightness(shot_id: int, starts, n_frames: int) -> np.ndarray | None:
    """Mean raw-frame intensity per candidate window (the activity proxy).

    Token-id "motion" saturates at ~1.0 on the fast rbb cadence (every cell
    changes within the ±window), so it does NOT discriminate quiescent
    early/dark windows from the structured flat-top.  Raw-frame brightness
    does: a window over an established plasma has a bright, structured image
    (mean intensity ≫ the near-dark ramp-up / aborted-shot frames).  Returns
    ``(len(starts),)`` mean intensities, or None if the raw frames are
    unavailable (the caller then falls back to enumeration order).
    """
    from imas_ambix.camdyn.dataset import level1_shot_path

    path = level1_shot_path(shot_id)
    if path is None or not Path(path).exists():
        return None
    try:
        import xarray as xr

        ds = xr.open_zarr(str(path / "rbb"), consolidated=False)
        data_vars = list(ds.data_vars)
        if not data_vars:
            return None
        raw = np.asarray(ds[data_vars[0]].values)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[demo] raw frames for shot %d unavailable: %s", shot_id, exc)
        return None
    out = np.zeros(len(starts), dtype=np.float64)
    for k, s in enumerate(starts):
        end = min(int(s) + n_frames, raw.shape[0])
        out[k] = float(raw[int(s) : end].mean()) if end > s else 0.0
    return out


def select_windows(
    shot_ids,
    *,
    n_frames: int,
    stride: int,
    windows_per_shot: int = 1,
    seed: int = 0,
) -> list[DemoWindow]:
    """Pick the brightest (most plasma-active) held-out window(s) per shot.

    Enumerates every ``stride``-spaced window of ``n_frames`` frames in each
    shot and ranks them by mean raw-frame intensity — the honest activity
    proxy (token "motion" saturates and cannot tell a dark ramp-up window
    from a structured flat-top one).  The brightest windows sit on the
    established plasma where the camera sees real structure (strike points,
    filaments) rather than near-dark sensor noise; the motion fraction is
    still recorded for the caption.
    """
    from imas_ambix.camdyn.dataset import (
        FrameTokenDataset,
        FrameWindowConfig,
        discover_token_shots,
    )

    cfg = FrameWindowConfig(n_frames=n_frames, stride=stride, seed=seed)
    out: list[DemoWindow] = []
    for sid in shot_ids:
        specs = discover_token_shots(shot_ids=[sid], read_n_frames=True)
        if not specs:
            logger.warning("[demo] shot %d has no tokens on disk — skipping", sid)
            continue
        ds = FrameTokenDataset(specs, cfg)
        if len(ds) == 0:
            logger.warning("[demo] shot %d: no full windows — skipping", sid)
            continue
        # rank candidate windows by raw-frame brightness without
        # materialising every window's tokens (only the chosen few are read).
        starts = [ds._windows[i][1] for i in range(len(ds))]
        bright = _window_brightness(int(sid), starts, n_frames)
        order = (
            list(np.argsort(-bright)) if bright is not None else list(range(len(ds)))
        )
        for idx in order[:windows_per_shot]:
            win = ds[int(idx)]
            mf = _motion_fraction(win.tokens, win.frame_time)
            out.append(
                DemoWindow(
                    shot_id=int(win.shot_id),
                    start=int(win.start),
                    frame_time=np.asarray(win.frame_time, dtype=np.float64),
                    dt=np.asarray(win.dt, dtype=np.float64),
                    valid=np.asarray(win.valid_frames, dtype=bool),
                    true_tokens=np.asarray(win.tokens, dtype=np.int64),
                    motion_fraction=mf,
                )
            )
    return out


# ---------------------------------------------------------------------------
# Scenario masks (visibility: True = the model sees the cell)
# ---------------------------------------------------------------------------


def scenario_mask(scenario: str, n_frames: int, frontier: int) -> np.ndarray:
    """Visibility mask ``(F,H,W)`` for one scenario (True = visible)."""
    if scenario == "frontier":
        m = np.zeros((n_frames, GRID_H, GRID_W), dtype=bool)
        m[:frontier] = True  # full frames up to the frontier, rest masked
        return m
    if scenario == "clipped":
        from imas_ambix.camdyn.masking import named_geometry_mask

        return named_geometry_mask("fixed_section2", n_frames).astype(bool)
    if scenario == "signals_only":
        return np.zeros((n_frames, GRID_H, GRID_W), dtype=bool)
    raise ValueError(f"unknown scenario {scenario!r}")


def clip_box(scenario: str) -> tuple[int, int, int, int] | None:
    """Return the ``(r0, r1, c0, c1)`` visible box for an outline, or None.

    Only the fixed clip scenario has a static box to outline; the frontier
    scenario marks the whole frame (handled in the figure), signals-only has
    no visible region.
    """
    if scenario != "clipped":
        return None
    from imas_ambix.camdyn.masking import named_geometry_mask

    m0 = named_geometry_mask("fixed_section2", 1)[0]
    rows = np.where(m0.any(axis=1))[0]
    cols = np.where(m0.any(axis=0))[0]
    return int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1


# ---------------------------------------------------------------------------
# Predict phase — model forward → predicted / ZOH token grids
# ---------------------------------------------------------------------------


def _zscore(values, stats):
    mu, sd = stats
    return (np.asarray(values, dtype=np.float32) - mu) / sd


def predict_window(
    model,
    torch,
    device,
    win: DemoWindow,
    cond_stats,
    scenario: str,
    frontier: int,
):
    """Run the model + ZOH for one window/scenario.

    Returns ``(visible, pred_tokens, zoh_tokens)`` — all ``(F,H,W)``.
    ``pred_tokens`` / ``zoh_tokens`` are GLOBAL ids; a ``-1`` cell in
    ``zoh_tokens`` marks a never-observed location (rendered black).
    """
    from imas_ambix.camdyn.arm_compare import _carry_forward_pred
    from imas_ambix.camdyn.conditioning import CONDITIONING_CHANNELS, load_conditioning
    from imas_ambix.camdyn.dataset import discover_token_shots

    n_frames = win.true_tokens.shape[0]
    visible = scenario_mask(scenario, n_frames, frontier)

    # Conditioning held to this window's frame times (z-scored with the ckpt
    # stats, exactly as training/eval does).
    specs = discover_token_shots(shot_ids=[win.shot_id], read_n_frames=False)
    level1_path = specs[0].level1_path if specs else None
    cond = load_conditioning(
        level1_path, win.frame_time, win.shot_id, channels=CONDITIONING_CHANNELS
    )
    cv = _zscore(cond.values, cond_stats)[None]  # (1,F,C)
    cm = cond.missing[None].astype(np.float32)  # (1,F,C)
    dt = win.dt[None].astype(np.float32)  # (1,F)

    tokens_t = torch.from_numpy(win.true_tokens[None]).to(device)
    vis_t = torch.from_numpy(visible[None]).to(device)
    cv_t = torch.from_numpy(cv).to(device)
    cm_t = torch.from_numpy(cm).to(device)
    dt_t = torch.from_numpy(dt).to(device)

    with torch.no_grad():
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=(device.type == "cuda"),
        ):
            logits = model.module(tokens_t, vis_t, cv_t, cm_t, dt_t)
        bl = logits.float().cpu().numpy()[0]  # (F,H,W,bits)

    # Bit-head MAP token id (the model's exact most-likely id per cell).
    shifts = np.arange(bl.shape[-1], dtype=np.int64)
    pred_tokens = ((bl > 0.0).astype(np.int64) << shifts).sum(axis=-1)  # (F,H,W)

    # ZOH: causal carry-forward of the last observed token (−1 = never seen).
    zoh_tokens = _carry_forward_pred(win.true_tokens, visible)

    return visible, pred_tokens, zoh_tokens


def _bit_map_tokens(bit_logits: np.ndarray) -> np.ndarray:
    """Bit-head MAP token id per cell — ``id = Σ_b (logit_b > 0) << b``."""
    bl = np.asarray(bit_logits)
    shifts = np.arange(bl.shape[-1], dtype=np.int64)
    return ((bl > 0.0).astype(np.int64) << shifts).sum(axis=-1)


def predict_window_arm(
    model,
    torch,
    device,
    win: DemoWindow,
    cond_stats,
    scenario: str,
    frontier: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward ONE arm over one window/scenario → ``(visible, pred_tokens)``.

    ``pred_tokens`` ``(F,H,W)`` are the arm's bit-head MAP global token ids
    (the model's exact most-likely token per cell).  Both the trained
    dynamics and per-frame baseline arms have the identical forward
    signature, so this scores either with the arm's own conditioning stats.
    """
    from imas_ambix.camdyn.conditioning import CONDITIONING_CHANNELS, load_conditioning
    from imas_ambix.camdyn.dataset import discover_token_shots

    n_frames = win.true_tokens.shape[0]
    visible = scenario_mask(scenario, n_frames, frontier)

    specs = discover_token_shots(shot_ids=[win.shot_id], read_n_frames=False)
    level1_path = specs[0].level1_path if specs else None
    cond = load_conditioning(
        level1_path, win.frame_time, win.shot_id, channels=CONDITIONING_CHANNELS
    )
    cv = _zscore(cond.values, cond_stats)[None]
    cm = cond.missing[None].astype(np.float32)
    dt = win.dt[None].astype(np.float32)

    tokens_t = torch.from_numpy(win.true_tokens[None]).to(device)
    vis_t = torch.from_numpy(visible[None]).to(device)
    cv_t = torch.from_numpy(cv).to(device)
    cm_t = torch.from_numpy(cm).to(device)
    dt_t = torch.from_numpy(dt).to(device)

    with torch.no_grad():
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=(device.type == "cuda"),
        ):
            logits = model.module(tokens_t, vis_t, cv_t, cm_t, dt_t)
        bl = logits.float().cpu().numpy()[0]  # (F,H,W,bits)

    return visible, _bit_map_tokens(bl)


def run_predict_phase(
    shots, *, n_frames, frontier, stride, windows_per_shot, scenarios
):
    """Load the dynamics arm, materialise windows, build all token bundles."""
    import torch

    from imas_ambix.camdyn.arm_compare import _load_arm

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("[demo] loading dynamics arm on %s", dev)
    model, _full_cfg, cond_stats = _load_arm(DYNAMICS_CKPT, torch, dev)

    windows = select_windows(
        shots,
        n_frames=n_frames,
        stride=stride,
        windows_per_shot=windows_per_shot,
    )
    if not windows:
        raise RuntimeError("no demo windows could be selected from the held-out shots")
    for w in windows:
        logger.info(
            "[demo] shot %d start %d motion=%.3f t=%.1f-%.1f ms",
            w.shot_id,
            w.start,
            w.motion_fraction,
            w.frame_time[0] * 1e3,
            w.frame_time[-1] * 1e3,
        )

    bundle: list[dict] = []
    for w in windows:
        entry: dict = {
            "shot_id": w.shot_id,
            "start": w.start,
            "frame_time": w.frame_time,
            "dt": w.dt,
            "valid": w.valid,
            "motion_fraction": w.motion_fraction,
            "true_tokens": w.true_tokens,
            "scenarios": {},
        }
        for scenario in scenarios:
            visible, pred, zoh = predict_window(
                model, torch, dev, w, cond_stats, scenario, frontier
            )
            entry["scenarios"][scenario] = {
                "visible": visible,
                "pred_tokens": pred,
                "zoh_tokens": zoh,
            }
        bundle.append(entry)

    # release the model (repo §2b: try/finally + empty_cache)
    try:
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001
        logger.warning("[demo] model release note: %s", exc)

    return bundle


# ---------------------------------------------------------------------------
# Token-bundle (de)serialisation — the predict↔decode handoff on disk
# ---------------------------------------------------------------------------


def save_token_bundle(bundle, path: Path) -> None:
    """Flatten the predict-phase bundle to a single ``.npz`` for decoding.

    Every token-grid WINDOW that needs an image is stacked into one
    ``(N,F,16,16)`` array of GLOBAL ids so the decode phase can run a single
    batched forward over ``N·F`` frames; a JSON index records which slice
    belongs to which (window, scenario, row).
    """
    grids: list[np.ndarray] = []
    index: list[dict] = []
    meta: list[dict] = []

    def _add(grid, win_i, scenario, role):
        index.append(
            {"window": win_i, "scenario": scenario, "role": role, "slot": len(grids)}
        )
        grids.append(np.asarray(grid, dtype=np.int64))

    for wi, entry in enumerate(bundle):
        meta.append(
            {
                "shot_id": int(entry["shot_id"]),
                "start": int(entry["start"]),
                "frame_time": np.asarray(entry["frame_time"]).tolist(),
                "dt": np.asarray(entry["dt"]).tolist(),
                "valid": np.asarray(entry["valid"]).astype(bool).tolist(),
                "motion_fraction": float(entry["motion_fraction"]),
                "scenarios": list(entry["scenarios"].keys()),
            }
        )
        _add(entry["true_tokens"], wi, "_window", "true")
        for scenario, sc in entry["scenarios"].items():
            _add(sc["visible"].astype(np.int64), wi, scenario, "visible")
            _add(sc["pred_tokens"], wi, scenario, "pred")
            _add(sc["zoh_tokens"], wi, scenario, "zoh")

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        grids=np.stack(grids).astype(np.int64),
        index=json.dumps(index),
        meta=json.dumps(meta),
    )


# ---------------------------------------------------------------------------
# Decode phase — runs under the Open-MAGVIT2 venv (VQModel)
# ---------------------------------------------------------------------------


def decode_phase(token_bundle: Path, image_bundle: Path, device: str) -> None:
    """Decode every token grid in *token_bundle* → images in *image_bundle*.

    Runs inside the Open-MAGVIT2 venv.  Loads the VQModel ONCE, subtracts
    the registry offset, renders never-observed cells (id ``< 0``) as the
    codebook id 0 with a companion blackout mask so the figure can paint
    them black, and decodes the full stack in one batched pass.
    """
    sys.path.insert(0, str(MAGVIT2_ROOT))
    from imas_ambix.bench.stream_worker import decode_batch, load_model

    data = np.load(str(token_bundle), allow_pickle=True)
    grids = np.asarray(data["grids"], dtype=np.int64)  # (N,F,16,16) global ids
    n, f, h, w = grids.shape
    flat = grids.reshape(n * f, h, w)

    # never-observed (id < 0) → decode as id 0, paint black afterwards.
    blackout = flat < 0
    local = flat - REGISTRY_OFFSET
    local = np.where(blackout, 0, local)
    local = np.clip(local, 0, (1 << 18) - 1).astype(np.int64)

    model = load_model(MAGVIT2_ROOT, device)
    try:
        images = decode_batch(model, local, device, 8, (256, 256))  # (N*F,256,256,3)
    finally:
        try:
            del model
            if device.startswith("cuda"):
                import torch

                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    # zero out the blackout cells in image space (each token cell is a
    # 256/16 = 16-px block).
    block = 256 // GRID_H
    images = images.copy()
    for i in range(n * f):
        bo = blackout[i]
        if not bo.any():
            continue
        for r in range(h):
            for c in range(w):
                if bo[r, c]:
                    images[
                        i, r * block : (r + 1) * block, c * block : (c + 1) * block
                    ] = 0

    images = images.reshape(n, f, 256, 256, images.shape[-1])  # (N,F,256,256,3)

    image_bundle.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        image_bundle,
        images=images.astype(np.uint8),
        index=data["index"],
        meta=data["meta"],
    )


def run_decode_subprocess(token_bundle: Path, image_bundle: Path, device: str) -> None:
    """Re-invoke this module under the magvit2 venv to decode the bundle."""
    if not MAGVIT2_PYTHON.exists():
        raise RuntimeError(
            f"Open-MAGVIT2 decode interpreter not found at {MAGVIT2_PYTHON}. "
            "The frozen tokenizer decoder cannot be located on disk — cannot "
            "decode tokens to images (no download possible on betelgeuse)."
        )
    import os

    repo_root = Path(__file__).resolve().parent.parent.parent
    # The magvit2 venv may not have imas_ambix installed, so prepend the repo
    # root to sys.path and drive decode_phase via env vars (keeps the inline
    # `-c` payload free of embedded long paths).
    payload = (
        "import os, sys; sys.path.insert(0, os.environ['AMBIX_REPO_ROOT']); "
        "from imas_ambix.camdyn.reconstruction_demo import decode_phase; "
        "from pathlib import Path; "
        "decode_phase(Path(os.environ['AMBIX_TOKEN_BUNDLE']), "
        "Path(os.environ['AMBIX_IMAGE_BUNDLE']), os.environ['AMBIX_DECODE_DEVICE'])"
    )
    env = dict(os.environ)
    env["AMBIX_REPO_ROOT"] = str(repo_root)
    env["AMBIX_TOKEN_BUNDLE"] = str(token_bundle)
    env["AMBIX_IMAGE_BUNDLE"] = str(image_bundle)
    env["AMBIX_DECODE_DEVICE"] = device
    cmd = [str(MAGVIT2_PYTHON), "-c", payload]
    logger.info("[demo] decoding token bundle via magvit2 venv")
    subprocess.run(cmd, check=True, env=env)


# ---------------------------------------------------------------------------
# Raw ground-truth frames (level-1) for the same windows
# ---------------------------------------------------------------------------


def load_raw_frames(shot_id: int, start: int, n_frames: int) -> np.ndarray | None:
    """Raw rbb frames ``(F,112,156)`` uint8 for ``[start, start+n_frames)``.

    Returns None if the level-1 store / frames are unavailable.
    """
    from imas_ambix.camdyn.dataset import level1_shot_path

    path = level1_shot_path(shot_id)
    if path is None or not Path(path).exists():
        return None
    try:
        import xarray as xr

        ds = xr.open_zarr(str(path / "rbb"), consolidated=False)
        data_vars = list(ds.data_vars)
        if not data_vars:
            return None
        frames = np.asarray(ds[data_vars[0]].values)
        end = min(start + n_frames, frames.shape[0])
        return frames[start:end]
    except Exception as exc:  # noqa: BLE001
        logger.warning("[demo] raw frames for shot %d unavailable: %s", shot_id, exc)
        return None


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------

#: Native rbb frame aspect (rows, cols) — used to crop the square decoded
#: image back to the camera aspect, mirroring the encode preprocessing
#: (which resized the native frame to 256² before tokenising).
ORIGINAL_HW = (112, 156)


def _to_aspect(img_square: np.ndarray) -> np.ndarray:
    """Resize a 256² decoded image to the native rbb aspect for display.

    The encoder resized the native ``(112,156)`` frame to ``256²`` before
    tokenising (a pure bilinear resize, no crop — see
    ``frames_to_input_device``), so the faithful inverse for display is the
    same bilinear resize back to ``(112,156)``.
    """
    from PIL import Image

    if img_square.ndim == 3:
        img_square = img_square[..., 0]  # grayscale camera → single channel
    im = Image.fromarray(img_square.astype(np.uint8)).resize(
        (ORIGINAL_HW[1], ORIGINAL_HW[0]), Image.BILINEAR
    )
    return np.asarray(im)


def _column_frames(n_frames: int, frontier: int, scenario: str) -> list[int]:
    """Frame indices to show as columns for a scenario."""
    if scenario == "frontier":
        # post-frontier offsets +1,+2,+4,+6,+(F-1-frontier) capped to the window
        offsets = [1, 2, 4, 6, n_frames - 1 - frontier]
        cols = []
        for o in offsets:
            fi = frontier + o
            if 0 <= fi < n_frames and fi not in cols:
                cols.append(fi)
        return cols
    # clipped / signals-only: spread evenly across the window
    return list(np.linspace(0, n_frames - 1, 5).round().astype(int))


def display_limits(
    gt_frame: np.ndarray, *, lo_pct: float = 1.0, hi_pct: float = 99.0
) -> tuple[float, float]:
    """Robust per-frame display limits ``(vmin, vmax)`` from a GROUND-TRUTH frame.

    The token/decode pipeline normalises per-SHOT (one global min/max over the
    whole shot), so the dim ramp-up frames vanish and intra-pulse brightness
    evolution swamps the structure.  For display we instead take the
    ``lo_pct``/``hi_pct`` percentiles of the *ground-truth* frame at each time
    column and apply the SAME vmin/vmax to all three rows (GT / baseline /
    dynamics) in that column: structure becomes legible while genuine
    over/under-shoot in the reconstructions stays honestly visible (a
    reconstruction brighter than the GT 99th pct clips to white, as it should).

    Degenerate frames (flat or empty) fall back to a unit span so imshow does
    not divide by zero.
    """
    f = np.asarray(gt_frame, dtype=np.float64)
    finite = f[np.isfinite(f)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.percentile(finite, lo_pct))
    vmax = float(np.percentile(finite, hi_pct))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(finite.min())
        vmax = float(finite.max())
        if vmax <= vmin:
            vmax = vmin + 1.0
    return vmin, vmax


def _imshow_cam(ax, img, *, vmin=0, vmax=255):
    ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])


def _outline(ax, scenario: str, frontier_here: bool):
    """Outline the visible region (clip box) on a camera-aspect axis."""
    import matplotlib.patches as mpatches

    box = clip_box(scenario)
    if box is None:
        return
    r0, r1, c0, c1 = box
    # map token-grid box → camera-pixel box
    rr0 = r0 / GRID_H * ORIGINAL_HW[0]
    rr1 = r1 / GRID_H * ORIGINAL_HW[0]
    cc0 = c0 / GRID_W * ORIGINAL_HW[1]
    cc1 = c1 / GRID_W * ORIGINAL_HW[1]
    ax.add_patch(
        mpatches.Rectangle(
            (cc0 - 0.5, rr0 - 0.5),
            cc1 - cc0,
            rr1 - rr0,
            fill=False,
            edgecolor=ACCENT,
            linewidth=1.4,
        )
    )


SCENARIO_TITLE = {
    "frontier": "temporal-frontier forecast",
    "clipped": "clipped-view infill (fixed_section2)",
    "signals_only": "signals-only reconstruction (full mask)",
}


def assemble_figure(
    scenario: str,
    bundle_meta,
    images,
    index,
    raw_by_window,
    *,
    frontier: int,
    out_path: Path,
    show_decoded_gt: bool = True,
):
    """Lay out one scenario's multi-shot grid figure and save it."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # slot lookup: (window, scenario, role) -> image index
    slot = {(e["window"], e["scenario"], e["role"]): e["slot"] for e in index}

    windows = [wi for wi, m in enumerate(bundle_meta) if scenario in m["scenarios"]]
    if not windows:
        raise RuntimeError(f"no windows carry scenario {scenario!r}")

    # rows per window block
    rows = ["truth (raw)"]
    if show_decoded_gt:
        rows.append("truth (decoded)")
    rows += ["zero-order hold", "dynamics prediction"]
    n_rows_per = len(rows)

    # columns
    n_frames = len(bundle_meta[windows[0]]["frame_time"])
    cols = _column_frames(n_frames, frontier, scenario)
    n_cols = len(cols)

    n_blocks = len(windows)
    fig_h = 1.5 * n_rows_per * n_blocks + 0.6
    fig_w = 1.7 * n_cols + 1.4
    fig, axes = plt.subplots(
        n_rows_per * n_blocks,
        n_cols,
        figsize=(fig_w, fig_h),
        squeeze=False,
        constrained_layout=True,
    )

    f0 = frontier if scenario == "frontier" else 0

    for bi, wi in enumerate(windows):
        m = bundle_meta[wi]
        ft = np.asarray(m["frame_time"], dtype=float)
        raw = raw_by_window.get(wi)
        true_decoded = images[slot[(wi, "_window", "true")]]
        pred = images[slot[(wi, scenario, "pred")]]
        zoh = images[slot[(wi, scenario, "zoh")]]

        for ci, fi in enumerate(cols):
            dt_ms = (ft[fi] - ft[f0]) * 1e3
            row0 = bi * n_rows_per

            # row: raw GT
            ax = axes[row0][ci]
            if raw is not None and fi < raw.shape[0]:
                _imshow_cam(ax, raw[fi])
            else:
                _imshow_cam(ax, np.zeros(ORIGINAL_HW, dtype=np.uint8))
            _outline(ax, scenario, scenario == "frontier")
            ax.set_title(f"{dt_ms:+.1f} ms", fontsize=8)
            if ci == 0:
                ax.set_ylabel(f"shot {m['shot_id']}\n{rows[0]}", fontsize=8)

            ri = 1
            if show_decoded_gt:
                ax = axes[row0 + ri][ci]
                _imshow_cam(ax, _to_aspect(true_decoded[fi]))
                _outline(ax, scenario, scenario == "frontier")
                if ci == 0:
                    ax.set_ylabel(rows[ri], fontsize=8)
                ri += 1

            # ZOH row
            ax = axes[row0 + ri][ci]
            _imshow_cam(ax, _to_aspect(zoh[fi]))
            _outline(ax, scenario, scenario == "frontier")
            if ci == 0:
                ax.set_ylabel(rows[ri], fontsize=8)
            ri += 1

            # dynamics prediction row
            ax = axes[row0 + ri][ci]
            _imshow_cam(ax, _to_aspect(pred[fi]))
            _outline(ax, scenario, scenario == "frontier")
            if ci == 0:
                ax.set_ylabel(rows[ri], fontsize=8)

    # caption
    first = bundle_meta[windows[0]]
    ft0 = np.asarray(first["frame_time"], dtype=float)
    frontier_ms = (ft0[frontier] - ft0[0]) * 1e3 if scenario == "frontier" else None
    bits = [
        SCENARIO_TITLE.get(scenario, scenario),
        f"shots {', '.join(str(bundle_meta[wi]['shot_id']) for wi in windows)}",
        f"window {ft0[0] * 1e3:.0f}-{ft0[-1] * 1e3:.0f} ms",
    ]
    if scenario == "frontier":
        bits.append(
            f"frontier @ frame {frontier} (t0={frontier_ms:+.1f} ms, offsets shown)"
        )
    bits.append(
        "decoder = frozen baseline Open-MAGVIT2 (imagenet_256_L, never fine-tuned)"
    )
    fig.suptitle(
        "camera-dynamics-wm — " + " | ".join(bits),
        fontsize=10,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("[demo] wrote %s", out_path)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

SCENARIO_FILE = {
    "frontier": "fig-cdw-demo-frontier.png",
    "clipped": "fig-cdw-demo-clipped.png",
    "signals_only": "fig-cdw-demo-signals-only.png",
}


def build_demo(
    out_dir: Path,
    *,
    shots=DEMO_SHOTS,
    n_frames: int = 16,
    frontier: int = 8,
    stride: int = 8,
    windows_per_shot: int = 1,
    scenarios=("frontier", "clipped", "signals_only"),
    work_dir: Path | None = None,
) -> list[Path]:
    """Predict → decode → assemble all scenario figures.  Returns figure paths."""
    import os
    import tempfile

    out_dir = Path(out_dir)
    # Scratch (predict↔decode token/image bundles) lives in TMPDIR, NOT under
    # the committed figures dir — only the PNGs land in out_dir.
    work_dir = work_dir or Path(
        tempfile.mkdtemp(prefix="camdyn-demo-", dir=os.environ.get("TMPDIR", "/tmp"))
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    token_bundle = work_dir / "tokens.npz"
    image_bundle = work_dir / "images.npz"

    bundle = run_predict_phase(
        shots,
        n_frames=n_frames,
        frontier=frontier,
        stride=stride,
        windows_per_shot=windows_per_shot,
        scenarios=scenarios,
    )
    save_token_bundle(bundle, token_bundle)
    run_decode_subprocess(
        token_bundle,
        image_bundle,
        "cuda",
    )

    data = np.load(str(image_bundle), allow_pickle=True)
    images = np.asarray(data["images"], dtype=np.uint8)
    index = json.loads(str(data["index"]))
    meta = json.loads(str(data["meta"]))

    # raw frames per window (level-1)
    raw_by_window: dict[int, np.ndarray] = {}
    for wi, m in enumerate(meta):
        raw = load_raw_frames(int(m["shot_id"]), int(m["start"]), n_frames)
        if raw is not None:
            raw_by_window[wi] = raw

    written: list[Path] = []
    for scenario in scenarios:
        out_path = out_dir / SCENARIO_FILE[scenario]
        assemble_figure(
            scenario,
            meta,
            images,
            index,
            raw_by_window,
            frontier=frontier,
            out_path=out_path,
            show_decoded_gt=True,
        )
        written.append(out_path)
    return written


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        default="docs/figures/camera-dynamics-wm/",
        help="output directory for the figures",
    )
    p.add_argument(
        "--frontier", type=int, default=8, help="frontier frame (of n_frames)"
    )
    p.add_argument("--n-frames", type=int, default=16)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--windows-per-shot", type=int, default=1)
    p.add_argument(
        "--shots",
        default=",".join(str(s) for s in DEMO_SHOTS),
        help="comma-separated held-out shot ids",
    )
    p.add_argument(
        "--scenarios",
        default="frontier,clipped,signals_only",
        help="comma-separated scenarios",
    )
    # decode-phase entry point (invoked under the magvit2 venv)
    p.add_argument("--decode-phase", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--token-bundle", default=None, help=argparse.SUPPRESS)
    p.add_argument("--image-bundle", default=None, help=argparse.SUPPRESS)
    p.add_argument("--device", default="cuda", help=argparse.SUPPRESS)
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.decode_phase:
        decode_phase(Path(args.token_bundle), Path(args.image_bundle), args.device)
        return 0

    shots = [int(s) for s in args.shots.split(",") if s.strip()]
    scenarios = tuple(s.strip() for s in args.scenarios.split(",") if s.strip())
    written = build_demo(
        Path(args.out),
        shots=shots,
        n_frames=args.n_frames,
        frontier=args.frontier,
        stride=args.stride,
        windows_per_shot=args.windows_per_shot,
        scenarios=scenarios,
    )
    for w in written:
        logger.info("[demo] figure: %s", w)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
