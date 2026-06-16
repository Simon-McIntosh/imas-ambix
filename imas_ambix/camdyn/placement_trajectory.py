"""Is the persistent-filament blur REDUCIBLE (worth a retrain) or at a FLOOR?

The decisive eval-only gate for the camera-dynamics world model.  No
training.  Two ranked questions:

RANK 0 — decontaminate the placement metric
--------------------------------------------
An adversarial review proved the whole-edge-band located-SSIM in
:mod:`structure_fidelity` is ~90 % background-copyable: on shot 24446 the
MAP decode and the frozen-last-frame PERSISTENCE decode score the SAME
edge-band located-SSIM (0.877 each).  A band-wide SSIM is dominated by the
static divertor/background pixels that BOTH the model and persistence copy
correctly, so it cannot see whether the model placed the bright FILAMENTS
in the right cells.

The fix: restrict the located-SSIM to the FILAMENT CELLS that
:func:`structure_fidelity.calibrate_hedging_threshold` already isolates
(token-grid cells in the lower-edge band whose GT high-frequency striation
energy is in the top tercile), and report the RESIDUAL over persistence —
``(MAP − persistence)`` and ``(oracle − persistence)`` on those cells.
Persistence is the reference, so its residual is 0 by construction; a
spectrum-matched COLOURED-NOISE control must score ≤ 0.  If the residual
does not separate MAP from persistence/noise with real dynamic range, the
metric still cannot see placement and we say so.

RANK 1 — checkpoint-trajectory reducibility gate
-------------------------------------------------
For every saved step checkpoint (step1000 … step19000, final) of BOTH arms
we compute the decontaminated filament-residual placement metric (MAP and
persistence are DETERMINISTIC — no seed loop needed) and the filament-cell
predictive entropy, and plot them vs training step.

Decision rule (stated against the curves):

* If at step19000→final the dynamics filament-residual is STILL CLIMBING
  and the filament-cell entropy still FALLING → placement is REDUCIBLE
  (the arms are undertrained, ~0.65 epoch) → a convergence retrain
  (cap_v1 to ~60k steps) is justified.
* If both have PLATEAUED by mid-training → placement is at a FLOOR → do NOT
  spend a placement retrain; ship MAP-for-estimate + coherent-sample-for-viz
  and put any retrain budget on a temporal-evolution objective (which needs
  its own validated motion readout first).

Run (eval-only, deterministic, GPU node)::

    .venv/bin/python -m imas_ambix.camdyn.placement_trajectory

Smoke (1 checkpoint, 1 window) before the full loop::

    .venv/bin/python -m imas_ambix.camdyn.placement_trajectory --smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from pathlib import Path

import numpy as np

from imas_ambix.camdyn import recon_movie as mv
from imas_ambix.camdyn import reconstruction_demo as rd
from imas_ambix.camdyn import structure_fidelity as sf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CKPT_ROOT = Path("/work/projects/imas_gpu/mast-checkpoints/camdyn")
ARM_DIRS = {
    "dynamics": CKPT_ROOT / "cap_v1_dynamics",
    "baseline": CKPT_ROOT / "cap_v1_baseline",
}

#: Step checkpoints in the trajectory (final.pt is the headline endpoint).
STEP_POINTS = tuple(range(1000, 20000, 1000))  # 1000 … 19000

DEFAULT_FIGURE = Path(
    "docs/figures/camera-dynamics-wm/fig-cdw-placement-trajectory.png"
)
DEFAULT_JSON = Path("imas_ambix/camdyn/artifacts/placement_trajectory.json")

#: Cond-ablation final-checkpoint re-read (RANK 0 second deliverable).
COND_ABLATION_FINAL = {
    "full": CKPT_ROOT / "cap_v1_dynamics" / "final.pt",
    "ip_ne": CKPT_ROOT / "cond_ip_ne" / "final.pt",
    "none": CKPT_ROOT / "cond_none" / "final.pt",
}
DEFAULT_COND_JSON = Path("imas_ambix/camdyn/artifacts/cond_ablation_final.json")

GRID_H, GRID_W = mv.GRID_H, mv.GRID_W
IMG = 256
CELL = IMG // GRID_H  # 16 px per token cell


# ---------------------------------------------------------------------------
# Filament-cell identification (reuse the calibrate_hedging_threshold logic)
# ---------------------------------------------------------------------------


def filament_cell_mask(raw_frames: np.ndarray, frontier: int) -> dict:
    """Token-grid FILAMENT / STATIC-BACKGROUND cell masks from RAW GT frames.

    This is the SAME split :func:`structure_fidelity.calibrate_hedging_threshold`
    uses to decontaminate the entropy read, lifted out so the placement metric
    can restrict its SSIM to exactly those cells:

    * ``filament`` — lower-edge-band cells whose GT high-frequency striation
      energy is in the top tercile of edge cells (where the bright SOL/divertor
      filaments live);
    * ``static_bg`` — non-edge cells in the bottom tercile of post-frontier
      temporal variance (the confidently-copyable background).

    Returns ``{filament, static_bg, n_filament, n_static_bg}`` with the two
    masks as ``(GRID_H, GRID_W)`` booleans, or ``calibrated=False`` if the GT
    frames are missing / the bands are empty.
    """
    from PIL import Image

    gh, gw = GRID_H, GRID_W
    if raw_frames is None or raw_frames.shape[0] <= frontier:
        return {"calibrated": False, "reason": "no raw frames"}
    post = list(range(frontier, raw_frames.shape[0]))
    gt = np.stack(
        [
            np.asarray(
                Image.fromarray(
                    np.clip(raw_frames[fi].astype(np.float64), 0, None).astype(np.uint8)
                    if raw_frames[fi].dtype != np.uint8
                    else raw_frames[fi]
                ).resize((gw, gh), Image.BILINEAR),
                dtype=np.float64,
            )
            for fi in post
        ]
    )  # (P, gh, gw)

    temporal_var = gt.var(axis=0)
    edge_band = np.zeros((gh, gw), dtype=bool)
    edge_band[gh - 4 :, :] = True
    hf = np.zeros((gh, gw), dtype=np.float64)
    for fi in range(gt.shape[0]):
        fr = gt[fi]
        pad = np.pad(fr, 1, mode="edge")
        local_mean = (
            pad[:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:] + fr
        ) / 5.0
        hf += np.abs(fr - local_mean)
    hf /= gt.shape[0]

    edge_hf = hf[edge_band]
    if edge_hf.size == 0:
        return {"calibrated": False, "reason": "empty edge band"}
    hf_hi = np.percentile(edge_hf, 66.0)
    var_lo = np.percentile(temporal_var, 33.0)
    filament = edge_band & (hf >= hf_hi)
    static_bg = (~edge_band) & (temporal_var <= var_lo)
    if filament.sum() == 0:
        return {"calibrated": False, "reason": "empty filament cell set"}
    return {
        "calibrated": True,
        "filament": filament,
        "static_bg": static_bg,
        "n_filament": int(filament.sum()),
        "n_static_bg": int(static_bg.sum()),
    }


def _cell_mask_to_pixels(cell_mask: np.ndarray) -> np.ndarray:
    """Upsample a ``(GRID_H, GRID_W)`` cell mask to a ``(256,256)`` pixel mask.

    Each token cell maps to a ``CELL × CELL`` pixel block (16²) — the filament
    cells become the pixel region the masked SSIM is averaged over.
    """
    return np.kron(np.asarray(cell_mask, dtype=bool), np.ones((CELL, CELL), bool))


# ---------------------------------------------------------------------------
# Masked located-SSIM restricted to a pixel region
# ---------------------------------------------------------------------------


def _ssim_pixel_map(a: np.ndarray, b: np.ndarray, *, win: int = 7) -> np.ndarray:
    """Per-pixel SSIM map between two float images (no skimage dependency).

    Identical formulation to :func:`structure_fidelity.ssim_map` (per-pair
    robust 1–99 % normalisation, box-mean windows, standard stabilisers) but
    returns the per-pixel SSIM array so a region mask can select where the
    filaments are instead of averaging the whole frame.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    def _norm(x):
        lo, hi = np.percentile(x, 1.0), np.percentile(x, 99.0)
        if hi <= lo:
            hi = lo + 1.0
        return np.clip((x - lo) / (hi - lo), 0.0, 1.0)

    a = _norm(a)
    b = _norm(b)
    # SSIM stabilisers for a unit dynamic range (images normalised to [0,1])
    c1 = 0.01**2
    c2 = 0.03**2

    def _box(x):
        return sf._uniform_box(x, win)

    mu_a = _box(a)
    mu_b = _box(b)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    var_a = _box(a * a) - mu_a2
    var_b = _box(b * b) - mu_b2
    cov_ab = _box(a * b) - mu_ab
    ssim = ((2 * mu_ab + c1) * (2 * cov_ab + c2)) / (
        (mu_a2 + mu_b2 + c1) * (var_a + var_b + c2)
    )
    return np.clip(ssim, -1.0, 1.0)


def filament_located_ssim(
    decoded_imgs: np.ndarray,
    gt_imgs: np.ndarray,
    post: list[int],
    pixel_mask: np.ndarray,
) -> float:
    """Mean located-SSIM(pred, RAW GT) over post frames, RESTRICTED to filament px.

    The decontaminated placement number: the per-pixel SSIM map vs raw GT is
    averaged over the FILAMENT pixel region only (not the whole edge band), so
    a decode that copies the static background but misplaces the bright
    filaments scores low.  NaN if the region is empty.
    """
    if pixel_mask is None or not pixel_mask.any():
        return float("nan")
    vals = []
    for fi in post:
        if fi >= gt_imgs.shape[0]:
            continue
        pred = sf._to_gray256(decoded_imgs[fi])
        gt = sf._to_gray256(gt_imgs[fi])
        smap = _ssim_pixel_map(pred, gt)
        vals.append(float(smap[pixel_mask].mean()))
    return float(np.mean(vals)) if vals else float("nan")


# ---------------------------------------------------------------------------
# Per-checkpoint, per-window deterministic placement read
# ---------------------------------------------------------------------------


def _decode_three(grids: dict, work_dir: Path, device: str) -> dict:
    """Decode MAP / persistence / oracle token grids in ONE OMAG2 pass."""
    return sf._decode_grids(grids, work_dir, device)


def evaluate_placement(
    model,
    torch,
    device,
    win: rd.DemoWindow,
    cond_stats,
    work_dir: Path,
    *,
    frontier: int = sf.FRONTIER,
    seed0: int = sf.SEED0,
) -> dict:
    """Decontaminated filament-residual placement read for ONE (model, window).

    Deterministic: MAP and persistence need no seeds.  The oracle joint sample
    is drawn once (seed0) as the probe-only upper bound.  Returns the
    filament-restricted located-SSIM for each role, the residuals over
    persistence, the coloured-noise control residual (dynamic-range check), and
    the filament-cell predictive entropy.
    """
    visible, bit_logits = sf.forward_bit_logits(
        model, torch, device, win, cond_stats, sf.SCENARIO, frontier
    )
    post = list(range(frontier, win.true_tokens.shape[0]))
    true_tokens = win.true_tokens

    gt256 = sf._raw_gt_256(win)
    raw_native = rd.load_raw_frames(win.shot_id, win.start, win.true_tokens.shape[0])
    if gt256 is None or raw_native is None:
        return {"error": "no raw GT frames", "shot_id": int(win.shot_id)}

    cells = filament_cell_mask(raw_native, frontier)
    if not cells.get("calibrated"):
        return {
            "error": cells.get("reason", "uncalibrated"),
            "shot_id": int(win.shot_id),
        }
    fil_px = _cell_mask_to_pixels(cells["filament"])

    # deterministic token grids + the probe-only oracle
    map_tok = sf.decode_map(bit_logits)
    persist_tok = mv.persistence_tokens(true_tokens, frontier)
    oracle_tok = sf.decode_oracle_joint(
        bit_logits, true_tokens, temperature=1.0, rng=np.random.default_rng(seed0)
    )
    grids = {"map": map_tok, "persistence": persist_tok, "oracle_joint": oracle_tok}
    # the OMAG2 decode subprocess takes a device STRING, not a torch.device
    dev_str = "cuda" if getattr(device, "type", str(device)) == "cuda" else "cpu"
    decoded = _decode_three(grids, work_dir, dev_str)

    # coloured-noise control (image space)
    cn = sf.coloured_noise_like(gt256, post, np.random.default_rng(seed0))

    loc = {
        "map": filament_located_ssim(decoded["map"], gt256, post, fil_px),
        "persistence": filament_located_ssim(
            decoded["persistence"], gt256, post, fil_px
        ),
        "oracle_joint": filament_located_ssim(
            decoded["oracle_joint"], gt256, post, fil_px
        ),
        "coloured_noise": filament_located_ssim(cn, gt256, post, fil_px),
    }

    # filament-cell predictive entropy (post-frontier mean over filament cells)
    ent = sf.per_bit_entropy(bit_logits)  # (F,gh,gw)
    ent_post = ent[np.asarray(post, dtype=int)].mean(axis=0)  # (gh,gw)
    fil_ent = float(ent_post[cells["filament"]].mean())
    bg_ent = (
        float(ent_post[cells["static_bg"]].mean())
        if cells["n_static_bg"]
        else float("nan")
    )

    persist_loc = loc["persistence"]
    return {
        "shot_id": int(win.shot_id),
        "window_ms": [
            float(win.frame_time[0] * 1e3),
            float(win.frame_time[-1] * 1e3),
        ],
        "n_filament_cells": cells["n_filament"],
        "n_static_bg_cells": cells["n_static_bg"],
        "filament_located_ssim": loc,
        "residual_vs_persistence": {
            "map": float(loc["map"] - persist_loc),
            "oracle_joint": float(loc["oracle_joint"] - persist_loc),
            "persistence": 0.0,  # reference, by construction
            "coloured_noise": float(loc["coloured_noise"] - persist_loc),
        },
        "filament_cell_entropy": fil_ent,
        "static_bg_cell_entropy": bg_ent,
    }


# ---------------------------------------------------------------------------
# Window selection (shared with the structure-fidelity probe)
# ---------------------------------------------------------------------------


def _select_windows(shots) -> list[rd.DemoWindow]:
    sws = sf.select_structure_windows(shots=tuple(shots))
    return [sw.window for sw in sws]


# ---------------------------------------------------------------------------
# Trajectory driver — every checkpoint, both arms
# ---------------------------------------------------------------------------


def run_trajectory(
    *,
    device: str = "cuda",
    shots=sf.FLATTOP_SHOTS,
    step_points=STEP_POINTS,
    smoke: bool = False,
) -> dict:
    """Compute the decontaminated placement read at every checkpoint, both arms."""
    import contextlib

    import torch

    from imas_ambix.camdyn.arm_compare import _load_arm

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    logger.info("[placement] device = %s", dev)

    if smoke:
        shots = (shots[0],)
        step_points = (step_points[0],)
        arms = {"dynamics": ARM_DIRS["dynamics"]}
    else:
        arms = ARM_DIRS

    windows = _select_windows(shots)
    if not windows:
        raise RuntimeError("no flat-top edge windows could be selected")
    logger.info(
        "[placement] %d window(s) on shots %s; %d step ckpts + final; arms=%s",
        len(windows),
        [w.shot_id for w in windows],
        len(step_points),
        list(arms),
    )

    work_dir = Path(
        tempfile.mkdtemp(prefix="placement-", dir=os.environ.get("TMPDIR", "/tmp"))
    )

    # checkpoint label → file, ordered (step1000 … step19000, final)
    ckpt_labels: list[tuple[str, int | None]] = [(f"step{s}", s) for s in step_points]
    if not smoke:
        ckpt_labels.append(("final", None))

    trajectory: dict = {arm: [] for arm in arms}
    for arm, arm_dir in arms.items():
        for label, step in ckpt_labels:
            ckpt = arm_dir / f"{label}.pt"
            if not ckpt.exists():
                logger.warning("[placement] missing %s — skipping", ckpt)
                continue
            logger.info("[placement] loading %s / %s", arm, label)
            model, _cfg, cond_stats = _load_arm(ckpt, torch, dev)
            try:
                per_window = []
                for win in windows:
                    res = evaluate_placement(
                        model, torch, dev, win, cond_stats, work_dir
                    )
                    per_window.append(res)
                ok = [r for r in per_window if "error" not in r]
                agg = _aggregate(ok)
                trajectory[arm].append(
                    {
                        "label": label,
                        "step": step if step is not None else _final_step(arm_dir),
                        "per_window": per_window,
                        "aggregate": agg,
                    }
                )
                logger.info(
                    "[placement] %s/%s  MAP-persist=%+.4f  oracle-persist=%+.4f  "
                    "fil_ent=%.3f  cn-persist=%+.4f",
                    arm,
                    label,
                    agg.get("residual_map_vs_persistence", float("nan")),
                    agg.get("residual_oracle_vs_persistence", float("nan")),
                    agg.get("filament_cell_entropy", float("nan")),
                    agg.get("residual_coloured_noise_vs_persistence", float("nan")),
                )
            finally:
                with contextlib.suppress(Exception):
                    del model
                if dev.type == "cuda":
                    torch.cuda.empty_cache()

    return {
        "task": (
            "decontaminated filament-placement metric + checkpoint-trajectory "
            "reducibility gate"
        ),
        "metric": (
            "filament_located_ssim = per-pixel SSIM(pred, RAW GT 256) averaged "
            "over the FILAMENT pixel region only (top-HF lower-edge token cells, "
            "16px blocks); residual = role_ssim - persistence_ssim on those px. "
            "Persistence residual = 0 (reference); coloured-noise residual must "
            "be <= 0 (dynamic-range check)."
        ),
        "scenario": sf.SCENARIO,
        "frontier_frame": sf.FRONTIER,
        "shots": [int(w.shot_id) for w in windows],
        "step_points": list(step_points),
        "deterministic": (
            "MAP + persistence are deterministic; oracle joint drawn once (seed "
            f"{sf.SEED0}); no seed loop"
        ),
        "trajectory": trajectory,
    }


def _final_step(arm_dir: Path) -> int:
    """Best-effort training step of final.pt (read from checkpoint metadata)."""
    try:
        import torch

        ck = torch.load(
            str(arm_dir / "final.pt"), map_location="cpu", weights_only=False
        )
        return int(ck.get("step", ck.get("global_step", 20000)))
    except Exception:  # noqa: BLE001
        return 20000


def _aggregate(window_results: list[dict]) -> dict:
    """Mean over windows of the placement residuals + filament entropy."""
    if not window_results:
        return {}

    def _m(key_path):
        vals = []
        for r in window_results:
            cur = r
            for k in key_path:
                cur = cur.get(k, {}) if isinstance(cur, dict) else {}
            if isinstance(cur, (int, float)) and np.isfinite(cur):
                vals.append(float(cur))
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "n_windows": len(window_results),
        "map_filament_ssim": _m(["filament_located_ssim", "map"]),
        "persistence_filament_ssim": _m(["filament_located_ssim", "persistence"]),
        "oracle_filament_ssim": _m(["filament_located_ssim", "oracle_joint"]),
        "coloured_noise_filament_ssim": _m(["filament_located_ssim", "coloured_noise"]),
        "residual_map_vs_persistence": _m(["residual_vs_persistence", "map"]),
        "residual_oracle_vs_persistence": _m(
            ["residual_vs_persistence", "oracle_joint"]
        ),
        "residual_coloured_noise_vs_persistence": _m(
            ["residual_vs_persistence", "coloured_noise"]
        ),
        "filament_cell_entropy": _m(["filament_cell_entropy"]),
        "static_bg_cell_entropy": _m(["static_bg_cell_entropy"]),
    }


# ---------------------------------------------------------------------------
# Reducibility verdict
# ---------------------------------------------------------------------------


def _series(traj_arm: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    """(steps, values) for one aggregate key across an arm's checkpoints."""
    steps, vals = [], []
    for pt in traj_arm:
        agg = pt.get("aggregate", {})
        v = agg.get(key, np.nan)
        if np.isfinite(v):
            steps.append(pt["step"])
            vals.append(v)
    order = np.argsort(steps)
    return np.asarray(steps)[order], np.asarray(vals)[order]


def reducibility_verdict(trajectory: dict) -> dict:
    """Is the dynamics filament-residual still climbing / entropy falling at the end?

    Compares the last-quarter slope of the dynamics arm's residual-MAP-vs-
    persistence and filament-cell-entropy series.  REDUCIBLE if the residual is
    still climbing AND entropy still falling into the final checkpoint; FLOOR if
    both have flattened by mid-training.
    """
    dyn = trajectory.get("dynamics", [])
    s_res, v_res = _series(dyn, "residual_map_vs_persistence")
    s_ent, v_ent = _series(dyn, "filament_cell_entropy")

    def _tail_slope(s, v):
        if s.size < 3:
            return float("nan"), float("nan")
        # slope over the last third of the trajectory, normalised per 1000 steps
        k = max(2, s.size // 3)
        slope = float(np.polyfit(s[-k:].astype(float), v[-k:], 1)[0]) * 1000.0
        # full-trajectory slope (for the "flattened" reference)
        slope_all = float(np.polyfit(s.astype(float), v, 1)[0]) * 1000.0
        return slope, slope_all

    res_tail, res_all = _tail_slope(s_res, v_res)
    ent_tail, ent_all = _tail_slope(s_ent, v_ent)

    # Absolute slope floors below which a tail trend is physically negligible
    # (noise-level drift on a plateaued curve).  These are set well below the
    # early-training slopes: the placement residual rises ~+0.05/1k and the
    # filament entropy falls ~-0.33/1k over the first ~3000 steps; a tail slope
    # an order of magnitude below that is a plateau, not a continuing trend.
    # Without an absolute floor a near-zero, noise-dominated tail slope can pass
    # a purely RELATIVE "fraction-of-full-slope" test when the full slope is
    # itself tiny, falsely flagging a flat curve as still-moving.
    res_climb_floor = 0.005  # +SSIM residual per 1000 steps
    ent_fall_floor = 0.10  # -nats per 1000 steps

    # "still climbing" — tail residual slope materially positive: above the
    # absolute floor AND a meaningful fraction of the full-trajectory slope.
    res_climbing = bool(
        np.isfinite(res_tail)
        and res_tail >= res_climb_floor
        and res_tail >= 0.25 * abs(res_all)
    )
    ent_falling = bool(
        np.isfinite(ent_tail)
        and ent_tail <= -ent_fall_floor
        and abs(ent_tail) >= 0.25 * abs(ent_all)
    )

    if res_climbing and ent_falling:
        verdict = "REDUCIBLE"
        detail = (
            "dynamics filament-placement residual is STILL CLIMBING and "
            "filament-cell entropy STILL FALLING into final.pt — the arms are "
            "undertrained; a convergence retrain (cap_v1 to ~60k steps) is "
            "justified for placement"
        )
    elif not res_climbing and not ent_falling:
        verdict = "FLOOR"
        detail = (
            "dynamics filament-placement residual and filament-cell entropy have "
            "PLATEAUED by mid-training — placement is at a floor; ship MAP-for-"
            "estimate + coherent-sample-for-viz and spend retrain budget on a "
            "temporal-evolution objective (needs a validated motion readout first)"
        )
    else:
        verdict = "MIXED"
        detail = (
            "residual and entropy disagree on direction at the tail "
            f"(residual {'climbing' if res_climbing else 'flat'}, entropy "
            f"{'falling' if ent_falling else 'flat'}) — inspect the curves before "
            "committing retrain budget"
        )

    return {
        "verdict": verdict,
        "detail": detail,
        "dynamics_residual_map_vs_persistence_tail_slope_per1k": res_tail,
        "dynamics_residual_map_vs_persistence_full_slope_per1k": res_all,
        "dynamics_filament_entropy_tail_slope_per1k": ent_tail,
        "dynamics_filament_entropy_full_slope_per1k": ent_all,
        "residual_still_climbing": res_climbing,
        "entropy_still_falling": ent_falling,
        "residual_climb_floor_per1k": res_climb_floor,
        "entropy_fall_floor_per1k": ent_fall_floor,
        "final_residual_map_vs_persistence": float(v_res[-1])
        if v_res.size
        else float("nan"),
        "final_filament_cell_entropy": float(v_ent[-1]) if v_ent.size else float("nan"),
    }


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def render_trajectory_figure(trajectory: dict, verdict: dict, out_path: Path) -> None:
    """Plot (a) filament-residual placement metric and (b) filament-cell entropy
    vs training step, both arms, with the reducibility verdict in the caption."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    colours = {"dynamics": "#2ca02c", "baseline": "#1f77b4"}

    for arm, traj in trajectory.items():
        col = colours.get(arm, "#888")
        s_map, v_map = _series(traj, "residual_map_vs_persistence")
        s_ora, v_ora = _series(traj, "residual_oracle_vs_persistence")
        s_cn, v_cn = _series(traj, "residual_coloured_noise_vs_persistence")
        s_ent, v_ent = _series(traj, "filament_cell_entropy")
        if s_map.size:
            ax1.plot(
                s_map,
                v_map,
                "-o",
                color=col,
                ms=4,
                label=f"{arm} MAP−persist",
            )
        if s_ora.size:
            ax1.plot(
                s_ora,
                v_ora,
                "--^",
                color=col,
                ms=4,
                alpha=0.7,
                label=f"{arm} oracle−persist",
            )
        if s_cn.size:
            ax1.plot(
                s_cn,
                v_cn,
                ":x",
                color=col,
                ms=4,
                alpha=0.5,
                label=f"{arm} cnoise−persist",
            )
        if s_ent.size:
            ax2.plot(s_ent, v_ent, "-o", color=col, ms=4, label=arm)

    ax1.axhline(0.0, color="k", lw=0.8, alpha=0.5)
    ax1.set_xlabel("training step")
    ax1.set_ylabel("filament-cell located-SSIM residual vs persistence")
    ax1.set_title(
        "decontaminated filament PLACEMENT residual\n"
        "(>0 = beats copy-last-frame on filament cells)"
    )
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=7, ncol=2)

    ax2.set_xlabel("training step")
    ax2.set_ylabel("filament-cell predictive entropy (nats)")
    ax2.set_title("filament-region predictive entropy")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)

    fig.suptitle(
        "camera-dynamics-wm placement reducibility gate — VERDICT: "
        f"{verdict['verdict']}\n{verdict['detail']}",
        fontsize=10,
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("[placement] wrote %s", out_path)


# ---------------------------------------------------------------------------
# Cond-ablation re-read at the headline (final.pt) checkpoint
# ---------------------------------------------------------------------------


def rerun_cond_ablation_final(*, device: str = "cuda") -> dict:
    """Re-read both cond-ablation selectors at the FULL arm's final.pt headline.

    The committed ``cond_ablation.json`` scored the full arm at step8000 (the
    matched-budget point).  This rescores at the full arm's final.pt (20000
    steps) — the headline checkpoint — against the SAME ip_ne / none arms, for
    BOTH the held_out and fixed_section2 selectors, so we can settle whether
    full beats none on NLL / top-1 at the headline.
    """
    from imas_ambix.camdyn.cond_ablation import compare_conditioning

    return compare_conditioning(
        {k: str(v) for k, v in COND_ABLATION_FINAL.items()}, device=device
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run(
    *,
    device: str = "cuda",
    figure: Path = DEFAULT_FIGURE,
    json_path: Path = DEFAULT_JSON,
    cond_json: Path = DEFAULT_COND_JSON,
    smoke: bool = False,
    skip_cond: bool = False,
) -> dict:
    summary = run_trajectory(device=device, smoke=smoke)
    verdict = reducibility_verdict(summary["trajectory"])
    summary["reducibility_verdict"] = verdict

    if not smoke:
        render_trajectory_figure(summary["trajectory"], verdict, figure)
        summary["figure"] = str(figure)

    # strip nothing heavy — placement results are scalar
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("[placement] written %s", json_path)

    if not skip_cond and not smoke:
        cond = rerun_cond_ablation_final(device=device)
        cond["headline_note"] = (
            "RE-READ at full=cap_v1_dynamics/final.pt (20000 steps); ip_ne/none "
            "are their own final.pt (6000 steps, never extended). Supersedes the "
            "step8000 matched-budget cond_ablation.json for the headline verdict. "
            "favours_dynamics convention: NLL diff = none_nll - arm_nll (positive "
            "= arm beats none); top1 diff = arm_acc - none_acc."
        )
        cond_json = Path(cond_json)
        cond_json.write_text(json.dumps(cond, indent=2), encoding="utf-8")
        logger.info("[placement] cond-ablation final written %s", cond_json)
        summary["cond_ablation_final"] = cond

    _print_report(summary)
    return summary


def _print_report(summary: dict) -> None:
    v = summary.get("reducibility_verdict", {})
    print("\n" + "=" * 80)
    print("PLACEMENT REDUCIBILITY GATE — decontaminated filament-residual trajectory")
    print("=" * 80)
    print(f"shots       : {summary['shots']}")
    for arm, traj in summary["trajectory"].items():
        print(f"\n[{arm}]")
        print(
            f"  {'ckpt':>10}{'MAP-persist':>14}{'oracle-persist':>16}"
            f"{'fil_entropy':>13}{'cn-persist':>12}"
        )
        for pt in traj:
            a = pt["aggregate"]
            cn = a.get("residual_coloured_noise_vs_persistence", float("nan"))
            print(
                f"  {pt['label']:>10}"
                f"{a.get('residual_map_vs_persistence', float('nan')):>14.4f}"
                f"{a.get('residual_oracle_vs_persistence', float('nan')):>16.4f}"
                f"{a.get('filament_cell_entropy', float('nan')):>13.3f}"
                f"{cn:>12.4f}"
            )
    print("-" * 80)
    print(f"VERDICT: {v.get('verdict')}")
    print(f"  {v.get('detail')}")
    print(
        f"  residual tail slope (per 1k): "
        f"{v.get('dynamics_residual_map_vs_persistence_tail_slope_per1k'):+.5f}  "
        f"(full {v.get('dynamics_residual_map_vs_persistence_full_slope_per1k'):+.5f})"
    )
    print(
        f"  entropy  tail slope (per 1k): "
        f"{v.get('dynamics_filament_entropy_tail_slope_per1k'):+.5f}  "
        f"(full {v.get('dynamics_filament_entropy_full_slope_per1k'):+.5f})"
    )
    if "cond_ablation_final" in summary:
        ho = summary["cond_ablation_final"]["held_out"]
        fs = summary["cond_ablation_final"]["named_geometry"]["fixed_section2"]
        print("-" * 80)
        print("COND-ABLATION at FULL final.pt (headline):")
        for sel, blk in (("held_out", ho), ("fixed_section2", fs)):
            fn = blk.get("full_vs_none_nll", {})
            ft = blk.get("full_vs_none_top1", {})
            print(
                f"  [{sel}] full_vs_none_NLL  mean={fn.get('mean'):+.4f} "
                f"CI=[{fn.get('lo'):+.4f},{fn.get('hi'):+.4f}] "
                f"full_beats_none={fn.get('favours_dynamics')}"
            )
            print(
                f"  [{sel}] full_vs_none_TOP1 mean={ft.get('mean'):+.6f} "
                f"CI=[{ft.get('lo'):+.6f},{ft.get('hi'):+.6f}] "
                f"full_beats_none={ft.get('favours_dynamics')}"
            )
    print("=" * 80 + "\n")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda")
    p.add_argument("--figure", default=str(DEFAULT_FIGURE))
    p.add_argument("--json", default=str(DEFAULT_JSON))
    p.add_argument("--cond-json", default=str(DEFAULT_COND_JSON))
    p.add_argument("--smoke", action="store_true", help="1 ckpt, 1 window, no cond")
    p.add_argument("--skip-cond", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    run(
        device=args.device,
        figure=Path(args.figure),
        json_path=Path(args.json),
        cond_json=Path(args.cond_json),
        smoke=args.smoke,
        skip_cond=args.skip_cond,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
