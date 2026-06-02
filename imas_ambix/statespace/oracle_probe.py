"""S9 oracle probe — IS the MSE pitch recoverable from NON-MSE diagnostics?

THE FEASIBILITY GATE.  This is a *supervised diagnostic*, not a model we keep.
It answers one question that decides whether S9-D4 (the neural filter) earns the
4xH200: does any non-MSE diagnostic channel carry interior-current / pitch
information at all?

Methodology (read carefully — a slip here flips the verdict):

* **CALIBRATION shots only.**  We use ONLY ``partition == "calibration"`` shots
  from the D1 manifest.  The 112 HELD-OUT shots are never touched here.
* **Shot-level train/test split.**  A 70/30 split *by shot* (seeded), never by
  slice — within-shot slices are heavily autocorrelated, so a slice-level split
  leaks and inflates every arm equally.  Normalisation and grid-node positions
  are fit on TRAIN shots only.
* **MSE pitch is a SUPERVISED LABEL here.**  That is legitimate: this probe
  measures information content, it is not the MSE-free filter.  The label is the
  D1-gated pitch truth (rail + error gate via
  :func:`mse_split.pitch_point_gate`) at the MSE sightlines, interpolated onto a
  FIXED radial grid so the target is fixed-dim and cross-shot comparable.
* **Extract once, ablate by column-slicing.**  ONE feature matrix per gated
  slice with named modality column blocks; every arm is a column subset of the
  *same* matrix on the *same* slice population — guaranteeing comparability.

The decisive ablation (all arms scored on the identical test points):

  A. magnetics-only (ama/amb/amc/ane/Ip)              [re-tests Stage-2]
  B. magnetics + Thomson
  C. magnetics + SXR + bolo
  D. magnetics + ALL (SXR + bolo + Thomson + camera)
  E. attribution: magnetics + each single non-mag modality alone

Decision rule (stated explicitly in the artifact):

  * arm D (and/or arms in E) CLEARLY below persistence AND below magnetics-only
    -> INFO EXISTS -> D4 earns the H200; report WHICH modalities carry it.
  * all arms ~ persistence -> MSE-free recovery INFEASIBLE from this corpus ->
    clean negative (Stage-2 extended to the full multimodal set), no H200.

Run: ``uv run python -m imas_ambix.statespace.oracle_probe``  (CPU, minutes).
"""

# Uppercase physics-symbol names (R, R0, Ip, …) follow the tokamak convention.
# mse_split aliased to M to match the sibling mse_eval.py convention.
# ruff: noqa: N803, N806, N812
from __future__ import annotations

import os

# Pin BLAS/OpenMP thread pools BEFORE importing numpy/sklearn.  On a shared
# login node the HistGradientBoostingRegressor's OpenMP pool otherwise spreads
# over all 64 cores and thrashes against concurrent jobs — a single fit went
# from 0.2 s (pinned) to > 200 s (unpinned) during development.  4 threads is
# ample for this MLP/GBM-scale probe and keeps it a good login-node citizen.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from imas_ambix.data.paths import LEVEL1_DIR  # noqa: E402
from imas_ambix.statespace import mse_split as M  # noqa: E402
from imas_ambix.statespace.align import (  # noqa: E402
    align_camera_to_grid,
    align_chord2d_to_grid,
)
from imas_ambix.statespace.baseline import (  # noqa: E402
    _AMA_CHANNELS,
    _AMB_CHANNELS,
    _AMC_CHANNELS,
    _ANE_CHANNELS,
)

logger = logging.getLogger(__name__)

MANIFEST_PATH = Path("/work/projects/imas_gpu/mast/manifests/mse_heldout_split_v0.json")
ARTIFACT_PATH = Path(__file__).resolve().parent / "artifacts" / "oracle_probe_v0.json"

# ---------------------------------------------------------------------------
# Target definition: fixed radial grid of pitch nodes
# ---------------------------------------------------------------------------
#
# The pooled MSE sightlines on MAST span ~0.7-1.4 m major radius.  We predict
# pitch at a small fixed set of radial nodes placed in the inner-quantile range
# of pooled sightline radii (computed on TRAIN shots).  Per-shot, nodes outside
# that shot's actual sightline coverage are masked (never scored — no
# extrapolation).  N_NODES kept small (feasibility probe, MLP/GBM scale).
N_NODES = 7
NODE_QLO, NODE_QHI = 0.10, 0.90  # inner-quantile band of pooled sightline radii

# Camera cheap features: pooled grid + a few robust stats. NO CNN.
CAM_POOL = 4  # 4x4 mean-pooled frame -> 16 features
N_CAM_STATS = 4  # mean, std, p10, p90 over the (cropped) frame

# Modality column-block names (order defines the concatenated feature matrix).
MODALITIES = ["mag", "thomson", "sxr", "bolo", "camera"]


# ---------------------------------------------------------------------------
# Feature extraction — one matrix per gated slice, named modality blocks
# ---------------------------------------------------------------------------


@dataclass
class ShotFeatures:
    """Per-shot aligned features + gated pitch target on the fixed radial grid."""

    shot_id: int
    blocks: dict[str, np.ndarray]  # name -> (S, F_block) feature block
    present: dict[str, bool]  # name -> modality present for this shot
    y: np.ndarray  # (S, N_NODES) gated pitch target (NaN where node masked)
    sightline_r: np.ndarray  # (C,) sightline major radii (for node masking)
    rax_proxy: np.ndarray  # (S,) per-slice axis-crossing R (diagnostic only)


def _read_signal_1d(grp, channels, slice_t):
    """Interp named 1-D channels of a zarr group onto slice times -> (S, C)."""
    keys = set(grp.array_keys())
    if "time" not in keys:
        return None
    t = np.asarray(grp["time"], dtype=np.float64)
    if t.size < 2:
        return None
    cols = []
    for ch in channels:
        if ch not in keys:
            cols.append(np.full(slice_t.size, np.nan))
            continue
        try:
            a = np.asarray(grp[ch], dtype=np.float64)
        except Exception:
            cols.append(np.full(slice_t.size, np.nan))
            continue
        if a.ndim != 1 or a.shape[0] != t.shape[0]:
            cols.append(np.full(slice_t.size, np.nan))
            continue
        # interp inside native range; NaN outside (no fabricated extrapolation)
        out = np.interp(slice_t, t, a, left=np.nan, right=np.nan)
        cols.append(out)
    return np.stack(cols, axis=1)


def _read_chord2d(store, group, n_ch, slice_t):
    """Align a chord-2D group (xsx/abm) onto slice times -> (S, n_ch) or None."""
    if group not in store:
        return None
    grp = store[group]
    keys = set(grp.array_keys())
    if "time" not in keys:
        return None
    t = np.asarray(grp["time"], dtype=np.float64)
    if t.size < 2:
        return None
    try:
        if group == "xsx":
            cams = []
            for cam_key in ("hcam_l", "hcam_u", "tcam"):
                if cam_key not in keys:
                    cams.append(np.full((t.size, 18), np.nan, dtype=np.float32))
                    continue
                cam = np.asarray(grp[cam_key], dtype=np.float32)
                if cam.shape[1] == t.size:
                    cam = cam.T
                elif cam.shape[0] != t.size:
                    cams.append(np.full((t.size, 18), np.nan, dtype=np.float32))
                    continue
                cams.append(cam)
            arr_native = np.concatenate(cams, axis=1)
        elif group == "abm":
            ibol = np.asarray(grp["i-bol"], dtype=np.float32)
            if ibol.shape[0] != t.size and ibol.shape[1] == t.size:
                ibol = ibol.T
            arr_native = ibol
        else:
            return None
    except Exception:
        return None
    aligned = align_chord2d_to_grid(arr_native, t, slice_t)  # NaN outside range
    if aligned.shape[1] != n_ch:
        out = np.full((slice_t.size, n_ch), np.nan, dtype=np.float32)
        nc = min(aligned.shape[1], n_ch)
        out[:, :nc] = aligned[:, :nc]
        aligned = out
    return aligned


def _read_camera_features(store, slice_t):
    """Cheap camera features aligned to slice times -> (S, CAM_POOL^2 + stats).

    Pooled 4x4 frame + (mean, std, p10, p90).  Nearest-frame alignment with
    clamp; per-slice in-range flag is appended.  NO CNN.
    """
    if "rbb" not in store:
        return None
    grp = store["rbb"]
    keys = set(grp.array_keys())
    if "data" not in keys or "time" not in keys:
        return None
    try:
        frames = np.asarray(grp["data"], dtype=np.float32)  # (T, H, W) uint8
        t_cam = np.asarray(grp["time"], dtype=np.float64)
    except Exception:
        return None
    if frames.ndim != 3 or t_cam.size < 2:
        return None
    T, H, W = frames.shape
    bh, bw = H // CAM_POOL, W // CAM_POOL
    if bh == 0 or bw == 0:
        return None
    Hc, Wc = bh * CAM_POOL, bw * CAM_POOL
    cropped = frames[:, :Hc, :Wc]
    pooled = cropped.reshape(T, CAM_POOL, bh, CAM_POOL, bw).mean(axis=(2, 4))
    pooled = pooled.reshape(T, CAM_POOL * CAM_POOL)  # (T, 16)
    flat = cropped.reshape(T, -1)
    stats = np.stack(
        [
            flat.mean(axis=1),
            flat.std(axis=1),
            np.percentile(flat, 10, axis=1),
            np.percentile(flat, 90, axis=1),
        ],
        axis=1,
    )  # (T, 4)
    feat_native = np.concatenate([pooled, stats], axis=1)  # (T, 20)
    # nearest-frame alignment (clamps outside range)
    feat_3d = feat_native[:, :, None]  # reuse the camera aligner (T,F,1)
    aligned = align_camera_to_grid(feat_3d, t_cam, slice_t)[:, :, 0]  # (S, 20)
    # per-slice in-range flag: 1 if slice time within native camera window
    in_range = ((slice_t >= t_cam.min()) & (slice_t <= t_cam.max())).astype(np.float32)[
        :, None
    ]
    return np.concatenate([aligned, in_range], axis=1)  # (S, 21)


def _target_on_grid(pitch_gated, sightline_r, node_r):
    """Interpolate gated pitch(R) onto fixed node radii per slice -> (S, N).

    Nodes outside a slice's finite-sightline coverage are NaN (never scored).
    Also returns a per-slice axis-crossing R proxy (diagnostic only).
    """
    S, C = pitch_gated.shape
    N = node_r.size
    y = np.full((S, N), np.nan)
    rax = np.full(S, np.nan)
    for s in range(S):
        p = pitch_gated[s]
        fin = np.isfinite(p)
        if fin.sum() < 4:
            continue
        rr = sightline_r[fin]
        pp = p[fin]
        srt = np.argsort(rr)
        rr, pp = rr[srt], pp[srt]
        # interp pitch onto nodes; mask nodes outside this shot's coverage
        in_cov = (node_r >= rr.min()) & (node_r <= rr.max())
        if in_cov.any():
            y[s, in_cov] = np.interp(node_r[in_cov], rr, pp)
        # axis-crossing proxy (nearest zero crossing) — diagnostic
        sgn = np.sign(pp)
        zc = np.where(np.diff(sgn) != 0)[0]
        if zc.size:
            i = zc[0]
            denom = pp[i + 1] - pp[i]
            if denom != 0:
                rax[s] = rr[i] - pp[i] * (rr[i + 1] - rr[i]) / denom
    return y, rax


def extract_shot(shot_id, entry, node_r):
    """Build a :class:`ShotFeatures` for one CAL shot (None if unusable)."""
    import zarr  # noqa: PLC0415

    path = LEVEL1_DIR / f"{shot_id}.zarr"
    if not path.exists():
        return None
    tr = M.read_ams_shot(path)
    if tr is None:
        return None

    # gated pitch-valid slices (D1 gate)
    t = tr.time
    point_gate = M.pitch_point_gate(tr.pitch, tr.pitch_error)  # (K, C)
    pv = M.pitch_valid_mask(tr)  # (K,) >= PITCH_VALID_MIN_CH gated channels
    sl = np.where(pv)[0]
    if sl.size == 0:
        return None
    slice_t = np.asarray(t[sl], dtype=np.float64)
    pitch = np.asarray(tr.pitch, dtype=np.float64)[sl]  # (S, C)
    gate = point_gate[sl]  # (S, C)
    pitch_gated = np.where(gate, pitch, np.nan)
    sightline_r = np.asarray(tr.active_channel_rpos, dtype=np.float64)  # (C,)

    y, rax = _target_on_grid(pitch_gated, sightline_r, node_r)
    # require at least one finite target node per slice
    keep = np.isfinite(y).any(axis=1)
    if not keep.any():
        return None
    sl_t = slice_t[keep]
    y = y[keep]
    rax = rax[keep]

    try:
        store = zarr.open_group(str(path), mode="r")
    except Exception:
        return None

    blocks: dict[str, np.ndarray] = {}
    present: dict[str, bool] = {}

    # --- magnetics block: ama + amb + amc + ane (Ip = amc/plasma_current) -----
    mag_parts = []
    mag_ok = True
    for g, chans in (
        ("ama", _AMA_CHANNELS),
        ("amb", _AMB_CHANNELS),
        ("amc", _AMC_CHANNELS),
        ("ane", _ANE_CHANNELS),
    ):
        if g not in store:
            mag_parts.append(np.full((sl_t.size, len(chans)), np.nan))
            if g in ("ama", "amb", "amc"):
                mag_ok = False
            continue
        m = _read_signal_1d(store[g], chans, sl_t)
        if m is None:
            m = np.full((sl_t.size, len(chans)), np.nan)
            if g in ("ama", "amb", "amc"):
                mag_ok = False
        mag_parts.append(m)
    blocks["mag"] = np.concatenate(mag_parts, axis=1)  # (S, 122)
    present["mag"] = mag_ok
    if not mag_ok:
        return None  # magnetics is the spine of every arm

    # --- thomson block (14 feats via integrated_inputs) ----------------------
    from imas_ambix.statespace.integrated_inputs import (  # noqa: PLC0415
        N_THOMSON_FEATURES,
        load_thomson_stream,
    )

    try:
        ts = load_thomson_stream(store, sl_t)
        blocks["thomson"] = np.asarray(ts.features, dtype=np.float64)
        present["thomson"] = ts.system != "none" and ts.n_measurements > 0
    except Exception:
        blocks["thomson"] = np.full((sl_t.size, N_THOMSON_FEATURES), np.nan)
        present["thomson"] = False

    # --- sxr block (54 chords) -----------------------------------------------
    sxr = _read_chord2d(store, "xsx", 54, sl_t)
    if sxr is None:
        sxr = np.full((sl_t.size, 54), np.nan)
        present["sxr"] = False
    else:
        present["sxr"] = True
    blocks["sxr"] = sxr

    # --- bolo block (32 chords) ----------------------------------------------
    bolo = _read_chord2d(store, "abm", 32, sl_t)
    if bolo is None:
        bolo = np.full((sl_t.size, 32), np.nan)
        present["bolo"] = False
    else:
        present["bolo"] = True
    blocks["bolo"] = bolo

    # --- camera block (cheap pooled + stats + in-range flag) -----------------
    cam = _read_camera_features(store, sl_t)
    if cam is None:
        cam = np.full((sl_t.size, CAM_POOL * CAM_POOL + N_CAM_STATS + 1), np.nan)
        present["camera"] = False
    else:
        present["camera"] = True
    blocks["camera"] = cam

    return ShotFeatures(
        shot_id=shot_id,
        blocks=blocks,
        present=present,
        y=y,
        sightline_r=sightline_r,
        rax_proxy=rax,
    )


# ---------------------------------------------------------------------------
# Node radii (fit on TRAIN shots only)
# ---------------------------------------------------------------------------


def fit_node_radii(manifest, train_ids):
    """Place N_NODES in the inner-quantile band of pooled TRAIN sightline radii."""
    radii = []
    for sid in train_ids:
        e = manifest["shots"].get(str(sid))
        if e is None:
            continue
        radii.extend(e["active_channel_rpos"])
    radii = np.asarray(radii, dtype=np.float64)
    radii = radii[np.isfinite(radii)]
    lo = float(np.quantile(radii, NODE_QLO))
    hi = float(np.quantile(radii, NODE_QHI))
    return np.linspace(lo, hi, N_NODES)


# ---------------------------------------------------------------------------
# Arms (column-subset specs over the shared feature matrix)
# ---------------------------------------------------------------------------

ARMS: dict[str, list[str]] = {
    "A_mag": ["mag"],
    "B_mag_thomson": ["mag", "thomson"],
    "C_mag_sxr_bolo": ["mag", "sxr", "bolo"],
    "D_all": ["mag", "thomson", "sxr", "bolo", "camera"],
    # attribution arms
    "E_mag_sxr": ["mag", "sxr"],
    "E_mag_bolo": ["mag", "bolo"],
    "E_mag_thomson": ["mag", "thomson"],
    "E_mag_camera": ["mag", "camera"],
}


def assemble_matrix(shots, mods):
    """Stack the requested modality blocks across shots -> (X, y, shot_idx).

    A per-modality present-flag column is appended for each non-mag modality so
    the regressor can distinguish "absent" from "zero".  NaN is left in place
    (HistGradientBoostingRegressor ingests NaN natively).
    """
    X_parts_per_shot = []
    y_all = []
    shot_idx = []
    for k, sf in enumerate(shots):
        parts = []
        for mod in mods:
            parts.append(sf.blocks[mod])
            if mod != "mag":
                flag = np.full(
                    (sf.blocks[mod].shape[0], 1),
                    1.0 if sf.present[mod] else 0.0,
                )
                parts.append(flag)
        X_parts_per_shot.append(np.concatenate(parts, axis=1))
        y_all.append(sf.y)
        shot_idx.append(np.full(sf.y.shape[0], k, dtype=int))
    X = np.concatenate(X_parts_per_shot, axis=0)
    y = np.concatenate(y_all, axis=0)
    si = np.concatenate(shot_idx, axis=0)
    return X, y, si


# ---------------------------------------------------------------------------
# Scoring helpers (LOCAL — D1's score()/Persistence are held-out-only)
# ---------------------------------------------------------------------------


def pooled_rmse(y_true, y_pred):
    """Plain pooled RMSE over all finite (slice x node) target points."""
    d = (y_pred - y_true).reshape(-1)
    d = d[np.isfinite(d)]
    return float(np.sqrt(np.mean(d**2))) if d.size else float("nan")


def per_shot_rmse(y_true, y_pred, shot_idx):
    """Per-shot RMSE then mean over shots (mirrors D1's aggregation shape)."""
    vals = []
    for k in np.unique(shot_idx):
        m = shot_idx == k
        d = (y_pred[m] - y_true[m]).reshape(-1)
        d = d[np.isfinite(d)]
        if d.size:
            vals.append(np.sqrt(np.mean(d**2)))
    return float(np.mean(vals)) if vals else float("nan")


def persistence_pred(test_shots, node_r):
    """Local persistence: freeze each shot's target at its first valid slice.

    Per node, fill from the first slice with a finite value at that node; if a
    node is never finite in the shot, fall back to the shot's first-slice median.
    Returns y_pred aligned to the test target stack.
    """
    preds = []
    for sf in test_shots:
        y = sf.y  # (S, N)
        S, N = y.shape
        frozen = np.full(N, np.nan)
        for j in range(N):
            col = y[:, j]
            fin = np.where(np.isfinite(col))[0]
            if fin.size:
                frozen[j] = col[fin[0]]
        med = np.nanmedian(frozen) if np.isfinite(frozen).any() else 0.0
        frozen = np.where(np.isfinite(frozen), frozen, med)
        preds.append(np.broadcast_to(frozen, (S, N)).copy())
    return np.concatenate(preds, axis=0)


def climatology_pred(train_shots, test_shots):
    """Climatology: predict the per-node TRAIN mean, ignoring ALL features.

    THE DISCRIMINATOR.  When every feature arm collapses to the same RMSE, it
    is either a true negative (features carry nothing → GBM falls back to the
    node mean) OR silent feature-deadness (alignment/NaN bug → same fallback).
    Both collapse to this climatology RMSE.  Reading the arms against it tells
    the two worlds apart:

      * climatology >> arm A_mag  -> features are ALIVE, magnetics tracks pitch,
        and a flat D-vs-A is a genuine negative for the extra modalities.
      * climatology ~= every arm  -> features add nothing beyond the node mean
        (true negative or dead features); the magnetics "win over persistence"
        is only "mean beats freeze", not pitch-tracking — report it as such.

    The mean is fit on TRAIN nodes only and broadcast to every TEST slice.
    """
    N = train_shots[0].y.shape[1]
    ytr = np.concatenate([sf.y for sf in train_shots], axis=0)  # (Ntr, N)
    node_mean = np.full(N, np.nan)
    for j in range(N):
        col = ytr[:, j]
        col = col[np.isfinite(col)]
        if col.size:
            node_mean[j] = float(np.mean(col))
    glob = np.nanmean(node_mean) if np.isfinite(node_mean).any() else 0.0
    node_mean = np.where(np.isfinite(node_mean), node_mean, glob)
    preds = []
    for sf in test_shots:
        S = sf.y.shape[0]
        preds.append(np.broadcast_to(node_mean, (S, N)).copy())
    return np.concatenate(preds, axis=0)


# ---------------------------------------------------------------------------
# Bootstrap CI over test shots
# ---------------------------------------------------------------------------


def bootstrap_ci(y_true, y_pred, shot_idx, n_boot=500, seed=0):
    """Bootstrap the per-shot-mean RMSE over test shots -> (lo, hi) 95% CI."""
    rng = np.random.default_rng(seed)
    shots = np.unique(shot_idx)
    # precompute per-shot squared-error sums + counts
    per = {}
    for k in shots:
        m = shot_idx == k
        d = (y_pred[m] - y_true[m]).reshape(-1)
        d = d[np.isfinite(d)]
        per[k] = (float(np.sum(d**2)), int(d.size))
    boots = []
    for _ in range(n_boot):
        samp = rng.choice(shots, size=shots.size, replace=True)
        rmses = []
        for k in samp:
            ss, n = per[k]
            if n:
                rmses.append(np.sqrt(ss / n))
        if rmses:
            boots.append(np.mean(rmses))
    if not boots:
        return float("nan"), float("nan")
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _per_shot_sumsq(y_true, y_pred, shot_idx):
    """Per-shot (sum sq err, count) over finite points -> dict[shot -> (ss, n)]."""
    out = {}
    for k in np.unique(shot_idx):
        m = shot_idx == k
        d = (y_pred[m] - y_true[m]).reshape(-1)
        d = d[np.isfinite(d)]
        out[int(k)] = (float(np.sum(d**2)), int(d.size))
    return out


def paired_bootstrap_diff(y_true, pred_a, pred_b, shot_idx, n_boot=2000, seed=0):
    """Paired bootstrap of (RMSE_a - RMSE_b) over SHARED test shots.

    Both predictors are scored on the SAME test points, so their per-shot errors
    are correlated; a paired resample (one shared shot resample per iteration)
    is the correct, non-conservative test.  Returns
    ``(mean_diff, lo, hi, frac_pos)`` where the CI is the 2.5/97.5 percentile of
    the difference and ``frac_pos`` is the fraction of resamples with diff > 0.
    A negative difference whose CI excludes 0 means ``a`` is significantly better
    (lower RMSE) than ``b``.
    """
    rng = np.random.default_rng(seed)
    ssa = _per_shot_sumsq(y_true, pred_a, shot_idx)
    ssb = _per_shot_sumsq(y_true, pred_b, shot_idx)
    shots = np.array(sorted(ssa))
    diffs = []
    for _ in range(n_boot):
        samp = rng.choice(shots, size=shots.size, replace=True)
        ra, rb = [], []
        for k in samp:
            sa, na = ssa[int(k)]
            sb, nb = ssb[int(k)]
            if na:
                ra.append(np.sqrt(sa / na))
            if nb:
                rb.append(np.sqrt(sb / nb))
        if ra and rb:
            diffs.append(np.mean(ra) - np.mean(rb))
    if not diffs:
        return float("nan"), float("nan"), float("nan"), float("nan")
    diffs = np.asarray(diffs)
    return (
        float(np.mean(diffs)),
        float(np.percentile(diffs, 2.5)),
        float(np.percentile(diffs, 97.5)),
        float(np.mean(diffs > 0)),
    )


def per_node_rmse(y_true, y_pred, node_r):
    """Per-radial-node pooled RMSE -> list of {node, r_m, rmse, n}."""
    out = []
    for j in range(y_true.shape[1]):
        d = (y_pred[:, j] - y_true[:, j]).reshape(-1)
        d = d[np.isfinite(d)]
        out.append(
            {
                "node": j,
                "r_m": round(float(node_r[j]), 3),
                "rmse": float(np.sqrt(np.mean(d**2))) if d.size else float("nan"),
                "n": int(d.size),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Main probe
# ---------------------------------------------------------------------------


@dataclass
class ProbeConfig:
    n_shots: int = 150
    test_frac: float = 0.30
    seed: int = 0
    gbm_seed: int = 0
    max_iter: int = 200  # early stopping typically halts well before this
    learning_rate: float = 0.06
    # "clearly below" margin: a fractional RMSE reduction this large is the
    # threshold for declaring info-exists (vs noise at ~30 test shots).
    clear_margin_frac: float = 0.05


def run(cfg: ProbeConfig) -> dict:
    from sklearn.ensemble import (  # noqa: PLC0415
        HistGradientBoostingRegressor,
    )

    t0 = time.time()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    cal_ids = sorted(
        int(s) for s, e in manifest["shots"].items() if e["partition"] == "calibration"
    )
    rng = np.random.default_rng(cfg.seed)
    chosen = sorted(
        rng.choice(cal_ids, size=min(cfg.n_shots, len(cal_ids)), replace=False).tolist()
    )

    # shot-level split FIRST (so node radii are fit on TRAIN shots only)
    n_test = max(1, int(round(len(chosen) * cfg.test_frac)))
    perm = rng.permutation(len(chosen))
    test_set = {chosen[i] for i in perm[:n_test]}
    train_ids = [s for s in chosen if s not in test_set]

    node_r = fit_node_radii(manifest, train_ids)
    logger.info("node radii (m): %s", np.round(node_r, 3).tolist())

    # extract features (cached on disk keyed by shot-set + seed so stat
    # refinements do not re-pay the ~8.5 min extraction)
    import hashlib  # noqa: PLC0415
    import pickle  # noqa: PLC0415

    key = hashlib.md5(
        f"{cfg.n_shots}_{cfg.seed}_{N_NODES}_{CAM_POOL}".encode()
    ).hexdigest()[:12]
    cache = Path("/tmp") / f"oracle_probe_feats_{key}.pkl"
    if cache.exists():
        logger.info("loading cached features from %s", cache)
        with cache.open("rb") as fh:
            blob = pickle.load(fh)
        train_shots = blob["train_shots"]
        test_shots = blob["test_shots"]
        presence = blob["presence"]
        n_attempt = blob["n_attempt"]
        node_r = blob["node_r"]
    else:
        train_shots, test_shots = [], []
        presence = {m: 0 for m in MODALITIES}
        n_attempt = 0
        for sid in chosen:
            n_attempt += 1
            sf = extract_shot(sid, manifest["shots"][str(sid)], node_r)
            if sf is None:
                continue
            for m in MODALITIES:
                presence[m] += int(sf.present[m])
            if sid in test_set:
                test_shots.append(sf)
            else:
                train_shots.append(sf)
        with cache.open("wb") as fh:
            pickle.dump(
                {
                    "train_shots": train_shots,
                    "test_shots": test_shots,
                    "presence": presence,
                    "n_attempt": n_attempt,
                    "node_r": node_r,
                },
                fh,
            )
    n_used = len(train_shots) + len(test_shots)
    presence_frac = {m: presence[m] / max(n_used, 1) for m in MODALITIES}
    logger.info(
        "extracted %d/%d shots (%d train, %d test) in %.1fs; presence=%s",
        n_used,
        n_attempt,
        len(train_shots),
        len(test_shots),
        time.time() - t0,
        {m: round(v, 2) for m, v in presence_frac.items()},
    )
    if not train_shots or not test_shots:
        raise RuntimeError("empty train or test set after extraction")

    n_train_slices = sum(sf.y.shape[0] for sf in train_shots)
    n_test_slices = sum(sf.y.shape[0] for sf in test_shots)

    # persistence reference on the SAME test points
    y_test_ref = np.concatenate([sf.y for sf in test_shots], axis=0)
    test_shot_idx = np.concatenate(
        [np.full(sf.y.shape[0], k) for k, sf in enumerate(test_shots)]
    )
    y_pers = persistence_pred(test_shots, node_r)
    pers_pooled = pooled_rmse(y_test_ref, y_pers)
    pers_pershot = per_shot_rmse(y_test_ref, y_pers, test_shot_idx)
    pers_lo, pers_hi = bootstrap_ci(y_test_ref, y_pers, test_shot_idx, seed=11)

    # climatology reference (predict the TRAIN per-node mean) — THE
    # discriminator between a true negative and silent feature-deadness.
    y_clim = climatology_pred(train_shots, test_shots)
    clim_pooled = pooled_rmse(y_test_ref, y_clim)
    clim_pershot = per_shot_rmse(y_test_ref, y_clim, test_shot_idx)
    clim_lo, clim_hi = bootstrap_ci(y_test_ref, y_clim, test_shot_idx, seed=13)
    logger.info(
        "persistence per_shot=%.4f CI=[%.4f,%.4f] | climatology per_shot=%.4f "
        "CI=[%.4f,%.4f]",
        pers_pershot,
        pers_lo,
        pers_hi,
        clim_pershot,
        clim_lo,
        clim_hi,
    )

    # per-arm: train one HGBR per node on TRAIN, score on TEST.  Keep each arm's
    # joint per-shot prediction stack so the paired bootstrap + per-node
    # breakdown can run on the SAME test points.
    arm_results: dict[str, dict] = {}
    arm_pred: dict[str, np.ndarray] = {}  # arm -> (n_test_pts, N_NODES) yhat
    for arm_name, mods in ARMS.items():
        Xtr, ytr, _ = assemble_matrix(train_shots, mods)
        Xte, yte, te_idx = assemble_matrix(test_shots, mods)
        yhat = np.full_like(yte, np.nan)
        for j in range(N_NODES):
            fin = np.isfinite(ytr[:, j])
            if fin.sum() < 50:
                continue
            reg = HistGradientBoostingRegressor(
                max_iter=cfg.max_iter,
                learning_rate=cfg.learning_rate,
                random_state=cfg.gbm_seed,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=20,
            )
            reg.fit(Xtr[fin], ytr[fin, j])
            te_fin = np.isfinite(yte[:, j])
            if te_fin.any():
                yhat[te_fin, j] = reg.predict(Xte[te_fin])
        arm_pred[arm_name] = yhat
        pooled = pooled_rmse(yte, yhat)
        pershot = per_shot_rmse(yte, yhat, te_idx)
        lo, hi = bootstrap_ci(yte, yhat, te_idx, seed=cfg.gbm_seed + 7)
        arm_results[arm_name] = {
            "modalities": mods,
            "rmse_pooled": pooled,
            "rmse_per_shot": pershot,
            "rmse_per_shot_ci95": [lo, hi],
            "rmse_per_node": per_node_rmse(yte, yhat, node_r),
            "n_features": Xtr.shape[1],
        }
        logger.info(
            "arm %-16s pooled=%.4f per_shot=%.4f CI=[%.4f,%.4f] feats=%d",
            arm_name,
            pooled,
            pershot,
            lo,
            hi,
            Xtr.shape[1],
        )

    # --- paired bootstrap of the DIFFERENCES on the shared test points --------
    # Marginal CIs overlap because clim/A/D are scored on the SAME 44 shots;
    # the correct test is a paired resample of the per-shot RMSE difference.
    bA = cfg.gbm_seed + 21
    paired = {
        "A_minus_clim": paired_bootstrap_diff(
            y_test_ref, arm_pred["A_mag"], y_clim, test_shot_idx, seed=bA
        ),
        "D_minus_A": paired_bootstrap_diff(
            y_test_ref, arm_pred["D_all"], arm_pred["A_mag"], test_shot_idx, seed=bA + 1
        ),
    }
    for arm_name in ("E_mag_sxr", "E_mag_bolo", "E_mag_thomson", "E_mag_camera"):
        paired[f"{arm_name}_minus_A"] = paired_bootstrap_diff(
            y_test_ref,
            arm_pred[arm_name],
            arm_pred["A_mag"],
            test_shot_idx,
            seed=bA + 2,
        )
    paired_out = {
        k: {"mean_diff": v[0], "ci95": [v[1], v[2]], "frac_pos": v[3]}
        for k, v in paired.items()
    }
    # per-node climatology RMSE for the interior-vs-edge story
    clim_per_node = per_node_rmse(y_test_ref, y_clim, node_r)
    for k, v in paired_out.items():
        logger.info(
            "paired %-22s mean_diff=%+.4f CI=[%+.4f,%+.4f]",
            k,
            v["mean_diff"],
            v["ci95"][0],
            v["ci95"][1],
        )

    # --- verdict (paired-difference CIs on the shared test points) ----------
    # clim/A/D are scored on the SAME test shots, so their per-shot errors are
    # correlated — the correct test is the paired bootstrap of the RMSE
    # difference (above), significant iff the difference-CI excludes 0.  The
    # marginal CIs above only describe absolute spread.
    margin = 1 - cfg.clear_margin_frac
    rmse_A = arm_results["A_mag"]["rmse_per_shot"]
    rmse_D = arm_results["D_all"]["rmse_per_shot"]

    def _sig_better(diff_key):
        # (a - b) significantly NEGATIVE  ->  a has lower RMSE than b
        v = paired_out[diff_key]
        return bool(np.isfinite(v["ci95"][1]) and v["ci95"][1] < 0.0)

    # are the FEATURES alive? (magnetics significantly below climatology)
    mag_beats_clim = _sig_better("A_minus_clim")

    # D-vs-A: require BOTH statistical significance (paired CI < 0) AND a
    # material 5% improvement — a 0.6% gain is not worth the H200 even if real.
    D_sig_below_A = _sig_better("D_minus_A")
    D_material = bool(np.isfinite(rmse_D) and rmse_D < rmse_A * margin)
    D_below_A = D_sig_below_A and D_material

    # attribution: which single-modality E arms BOTH significantly + materially
    # beat magnetics-only?
    carriers = []
    for arm_name in ("E_mag_sxr", "E_mag_bolo", "E_mag_thomson", "E_mag_camera"):
        r = arm_results[arm_name]["rmse_per_shot"]
        e_sig = _sig_better(f"{arm_name}_minus_A")
        e_mat = bool(np.isfinite(r) and r < rmse_A * margin)
        if e_sig and e_mat:
            carriers.append(arm_name.replace("E_mag_", ""))

    info_exists = bool(D_below_A) or len(carriers) > 0
    if info_exists:
        verdict = "INFO_EXISTS"
        carrier_note = (
            "non-magnetics modalities carry interior pitch info beyond "
            f"magnetics-only (paired-CI significant + >5% material): "
            f"{carriers or ['(D beats A jointly)']} -> D4 earns the H200."
        )
    elif not mag_beats_clim:
        # Every arm — magnetics included — ties climatology.  Either a true
        # negative or dead features; the magnetics 'win over persistence' is
        # just 'mean beats freeze'.  Still no H200, but flag the ambiguity.
        verdict = "INFEASIBLE_NEGATIVE"
        carrier_note = (
            "ALL arms (incl. magnetics) tie climatology on the paired test — no "
            "feature beats the per-node TRAIN mean.  MSE-free recovery "
            "infeasible from this corpus; magnetics does NOT track interior "
            "pitch beyond the mean (the persistence win is 'mean beats freeze', "
            "not pitch-tracking).  Target may be near-constant per node — "
            "interpret with care.  No H200."
        )
    else:
        # Features ALIVE (magnetics significantly beats climatology, paired),
        # but the extra modalities add nothing beyond magnetics — a clean,
        # defensible negative.
        verdict = "INFEASIBLE_NEGATIVE"
        carrier_note = (
            "magnetics SIGNIFICANTLY beats climatology (paired CI < 0 — features "
            "are alive and track pitch), but NO non-magnetics modality adds "
            "interior pitch info beyond magnetics-only.  Magnetics already beats "
            "persistence.  The negative is specifically 'SXR/bolo/Thomson/camera "
            "add nothing beyond magnetics' — NOT 'nothing beats persistence'.  "
            "See per-node RMSE for the interior-vs-edge story.  No H200."
        )

    result = {
        "version": "oracle_probe_v0",
        "question": (
            "Is MSE pitch recoverable from NON-MSE diagnostics? "
            "(supervised feasibility probe on CALIBRATION shots)"
        ),
        "verdict": verdict,
        "verdict_detail": {
            "info_exists": info_exists,
            "D_significantly_and_materially_below_A": bool(D_below_A),
            "D_paired_significant_below_A": bool(D_sig_below_A),
            "D_material_5pct_below_A": bool(D_material),
            "magnetics_beats_climatology_paired": bool(mag_beats_clim),
            "carriers": carriers,
            "carrier_note": carrier_note,
            "clear_margin_frac": cfg.clear_margin_frac,
            "rule": (
                "INFO_EXISTS requires an arm to BOTH (i) be significantly below "
                "magnetics-only on the PAIRED bootstrap of the per-shot RMSE "
                "difference (difference-CI upper bound < 0) AND (ii) clear a 5% "
                "material margin.  Otherwise INFEASIBLE_NEGATIVE.  The paired "
                "test (shared test shots, one resample) is correct where "
                "marginal CIs over-conservatively overlap.  The climatology "
                "paired test (A vs node-mean) tells a true negative (mag beats "
                "clim) from feature-deadness (mag ties clim)."
            ),
        },
        "magnetics_only_rmse_per_shot": rmse_A,
        "paired_differences": paired_out,
        "persistence": {
            "rmse_pooled": pers_pooled,
            "rmse_per_shot": pers_pershot,
            "rmse_per_shot_ci95": [pers_lo, pers_hi],
            "note": "freeze pitch at first gated slice, hold (D1-style sanity)",
        },
        "climatology": {
            "rmse_pooled": clim_pooled,
            "rmse_per_shot": clim_pershot,
            "rmse_per_shot_ci95": [clim_lo, clim_hi],
            "rmse_per_node": clim_per_node,
            "note": (
                "predict per-node TRAIN mean, ignore features; discriminator "
                "between true negative and silent feature-deadness"
            ),
        },
        "arms": arm_results,
        "target": {
            "definition": (
                f"gated MSE pitch (rad) interpolated onto {N_NODES} fixed radial "
                f"nodes in the [{NODE_QLO},{NODE_QHI}] quantile band of pooled "
                "TRAIN sightline radii; per-shot nodes outside coverage masked "
                "(never scored)"
            ),
            "node_radii_m": node_r.tolist(),
            "gate": "mse_split.pitch_point_gate (rail<1.5 rad, err<=0.3 rad)",
            "rmse_definition": "per-shot RMSE then mean over test shots (primary)",
        },
        "data": {
            "partition": "calibration",
            "n_shots_requested": cfg.n_shots,
            "n_shots_used": n_used,
            "n_train_shots": len(train_shots),
            "n_test_shots": len(test_shots),
            "n_train_slices": int(n_train_slices),
            "n_test_slices": int(n_test_slices),
            "train_shot_ids": sorted(sf.shot_id for sf in train_shots),
            "test_shot_ids": sorted(sf.shot_id for sf in test_shots),
            "modality_presence_frac": {
                m: round(v, 3) for m, v in presence_frac.items()
            },
            "split": "70/30 by SHOT (seeded); never by slice",
            "seed": cfg.seed,
            "gbm_seed": cfg.gbm_seed,
        },
        "model": {
            "regressor": "sklearn HistGradientBoostingRegressor (per-node)",
            "max_iter": cfg.max_iter,
            "learning_rate": cfg.learning_rate,
            "missing": "NaN ingested natively; per-modality present-flag column",
            "camera_features": (
                f"{CAM_POOL}x{CAM_POOL} pooled + 4 stats + in-range flag (no CNN)"
            ),
        },
        "runtime_s": round(time.time() - t0, 1),
    }
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-shots", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=str(ARTIFACT_PATH))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = ProbeConfig(n_shots=args.n_shots, seed=args.seed)
    result = run(cfg)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n[oracle_probe] VERDICT: {result['verdict']}")
    print(f"[oracle_probe] wrote {out}")
    print(json.dumps(result["verdict_detail"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
