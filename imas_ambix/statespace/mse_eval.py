"""D1 — held-out-MSE eval harness + EVAL-INTERFACE CONTRACT v0.

THE GATE.  Nothing downstream in S9 (D2 EnKF baseline, D4 neural filter) is
scorable until this lands.  Both D2 and D4 import this module and implement to
the :class:`ShotPrediction` contract, call the SHARED forward/inverse
observation models here, and are scored by :func:`score`.

What lives here
---------------
* :class:`ShotPrediction` — the canonical per-shot prediction dataclass.
* :func:`pitch_from_current_profile` — the CANONICAL SHARED forward
  observation model (current/iota profile → MSE pitch per sightline).  D2 and
  D4 both call this, so a head-to-head isolates *state inference*, not
  observation physics.  Pure-array / torch-portable, vectorised over
  sightlines and time.
* :func:`invert_pitch_to_q0rax` — the SHARED pitch → (q0, rax) map.  Applied
  identically to the predicted pitch AND the truth pitch, so model-q0 vs
  truth-q0 differ only by *model error*, not by inversion method.
* :class:`PersistencePredictor` — sanity baseline (freeze pitch at the first
  beam-on slice); a concrete target for D2/D4 to beat.
* :func:`score` — computes the pre-registered metrics
  (RMSE / CRPS / NLL / coverage[0.88,0.92]) per-shot then aggregates as the
  mean over shots.

Pre-registered metric structure
--------------------------------
PRIMARY  = pitch (rad), CLEAN — no inversion, no method mismatch.  This is the
           unambiguous gate axis.
SECONDARY = physically-gated, error-weighted q0_kappa1.85_4pt + rax_4pt.
           q0/rax carry a METHOD-MISMATCH CAVEAT: the truth's raw q0_4pt/rax_4pt
           come from a proprietary 2pt/4pt fit (fixed κ=1.85) we deliberately
           do NOT replicate.  The harness scores the method-matched secondary
           (truth-q0 := invert_pitch_to_q0rax(truth_pitch)) so both sides share
           the inversion, and additionally reports inv(truth_pitch) vs raw
           q0_4pt agreement as a provisional cross-check.  The gate verdict
           rests on PRIMARY pitch.

Slice population
----------------
PRIMARY pitch is scored on the manifest ``pitch_valid_mask`` slices.
SECONDARY q0/rax is scored on the co-finite gated set
``pitch_valid_mask & {q0,rax}_gated_mask`` (the populations only partially
overlap in the raw corpus; see mse_split notes).
"""

# This module uses uppercase physics-symbol names throughout (R, R0, B_tor,
# B_pol, K, C, G, …) — the standard tokamak / matrix-dimension convention.
# ruff: noqa: N803, N806, N812
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from imas_ambix.statespace import calibration as cal

logger = logging.getLogger(__name__)

MU0 = 4.0e-7 * np.pi

# Default machine constants for the shared forward model (MAST-ish).  D2/D4 may
# pass per-shot values; these are only fallbacks for tests.
DEFAULT_R0 = 0.85  # geometric major radius (m)
DEFAULT_BT0 = 0.5  # vacuum toroidal field at R0 (T)

# Coverage gate: pre-registered acceptance band on the 90% interval.
COVERAGE_GATE_LO = 0.88
COVERAGE_GATE_HI = 0.92


# ---------------------------------------------------------------------------
# EVAL-INTERFACE CONTRACT v0
# ---------------------------------------------------------------------------


@dataclass
class ShotPrediction:
    """A predictor's output for ONE held-out shot.

    All arrays are plain numpy.  ``t`` MUST equal the shot's manifest
    ``beam_on_slice_times``; channel order MUST equal the manifest
    ``active_channel_ids``.

    Attributes
    ----------
    t:
        (K,) slice times (s).
    pitch_mean, pitch_std:
        (K, C) predicted pitch (rad) and its 1σ at the C active MSE channels.
    pitch_samples:
        Optional (K, C, M) ensemble samples for non-Gaussian CRPS.  If present,
        CRPS uses the energy form; otherwise the Gaussian closed form is used.
    q0_mean, q0_std, rax_mean, rax_std:
        (K,) on-axis derived quantities.  If left empty, :func:`score` derives
        them from the predicted pitch via :func:`invert_pitch_to_q0rax`
        (method-matched).  Predictors MAY supply them directly if they prefer
        to propagate uncertainty themselves.
    """

    t: np.ndarray
    pitch_mean: np.ndarray
    pitch_std: np.ndarray
    pitch_samples: np.ndarray | None = None
    q0_mean: np.ndarray | None = None
    q0_std: np.ndarray | None = None
    rax_mean: np.ndarray | None = None
    rax_std: np.ndarray | None = None

    def validate(self, n_channels: int) -> None:
        t = np.asarray(self.t)
        K = t.shape[0]
        for name in ("pitch_mean", "pitch_std"):
            a = np.asarray(getattr(self, name))
            if a.shape != (K, n_channels):
                raise ValueError(f"{name} shape {a.shape} != (K={K}, C={n_channels})")
        if self.pitch_samples is not None:
            s = np.asarray(self.pitch_samples)
            if s.ndim != 3 or s.shape[:2] != (K, n_channels):
                raise ValueError(
                    f"pitch_samples shape {s.shape} != (K={K}, C={n_channels}, M)"
                )


# ---------------------------------------------------------------------------
# CANONICAL SHARED forward observation model
# ---------------------------------------------------------------------------


def pitch_from_current_profile(
    profile: np.ndarray,
    rho_grid: np.ndarray,
    sightline_rpos: np.ndarray,
    R0: float,
    B_tor0_or_Ip: float,
    *,
    kind: str = "j",
) -> np.ndarray:
    """SHARED forward model: current/ι profile → MSE pitch per sightline.

    Cylindrical Ampère closure (the canonical observation physics that BOTH the
    EnKF (D2) and the neural filter (D4) import, so a head-to-head isolates
    state inference, not observation physics):

        B_pol(r) = μ0 · I_enc(r) / (2π r),   I_enc(r) = ∫_0^r j(r') 2π r' dr'
        B_tor(R) = B_tor0 · R0 / R
        pitch(R) = arctan( B_pol(R) / B_tor(R) )

    Vectorised over sightlines AND time — no per-element python loops — so D4
    can port it to a differentiable torch version (every op below has a torch
    analogue: cumulative sum, broadcasting, arctan2, interp via searchsorted).

    Parameters
    ----------
    profile:
        Current-density-like profile, shape (..., G) over ``rho_grid``.
        Leading dims are batch/time and are preserved.  Interpretation set by
        ``kind``:
          * ``kind='j'``   — toroidal current density j(ρ) [A/m²].
          * ``kind='iota'``— rotational transform ι(ρ); converted to an
            effective B_pol via ι = R0 B_pol / (r B_tor) ⇒
            B_pol = ι · r · B_tor / R0.
    rho_grid:
        (G,) minor-radius grid (m), strictly increasing, ρ[0] ≈ 0 (axis).
    sightline_rpos:
        (C,) MSE sightline MAJOR radii (m).  Minor radius r = |R − R0|.
    R0:
        Geometric major radius (m).
    B_tor0_or_Ip:
        For ``kind='j'`` and ``kind='iota'`` this is the vacuum toroidal field
        B_tor0 at R0 (T).  (Iₚ is implied by the integral of ``profile``; the
        argument name keeps the contract signature flexible for callers that
        prefer to pass a current scale.)
    kind:
        ``'j'`` or ``'iota'``.

    Returns
    -------
    pitch : np.ndarray
        Shape (..., C) — MSE pitch (rad) at each sightline, batch/time preserved.
    """
    profile = np.asarray(profile, dtype=np.float64)
    rho = np.asarray(rho_grid, dtype=np.float64).reshape(-1)
    R = np.asarray(sightline_rpos, dtype=np.float64).reshape(-1)
    G = rho.shape[0]
    if profile.shape[-1] != G:
        raise ValueError(f"profile last dim {profile.shape[-1]} != grid {G}")

    # Minor radius magnitude of each sightline; the poloidal field circulates
    # the axis, so its projection along the MSE sightline flips sign between the
    # inboard (R < R0) and outboard (R > R0) side. The signed factor produces
    # the physical zero crossing of pitch AT the magnetic axis (R ≈ R0).
    r_sight = np.abs(R - R0)  # (C,) minor-radius magnitude
    side = np.sign(R - R0)  # (C,) +1 outboard, −1 inboard
    side = np.where(side == 0, 1.0, side)
    B_tor = B_tor0_or_Ip * R0 / np.where(R != 0, R, R0)  # (C,)

    if kind == "iota":
        # Interpolate ι onto each sightline minor radius (vectorised).
        iota_s = _interp_last_axis(profile, rho, r_sight)  # (..., C)
        B_pol = side * iota_s * r_sight * B_tor / R0  # broadcast (...,C)*(C,)
        return np.arctan2(B_pol, B_tor)

    if kind != "j":
        raise ValueError(f"kind must be 'j' or 'iota', got {kind!r}")

    # Enclosed current via the trapezoid of j(r')·2π r' dr' (cumulative).
    integrand = profile * (2.0 * np.pi * rho)  # (..., G)
    drho = np.diff(rho)  # (G-1,)
    # trapezoid cumulative sum along the grid axis
    seg = 0.5 * (integrand[..., 1:] + integrand[..., :-1]) * drho  # (..., G-1)
    I_enc_grid = np.concatenate(
        [np.zeros(profile.shape[:-1] + (1,)), np.cumsum(seg, axis=-1)], axis=-1
    )  # (..., G)
    # B_pol on the grid, then interpolate to each sightline minor radius.
    with np.errstate(divide="ignore", invalid="ignore"):
        B_pol_grid = MU0 * I_enc_grid / (2.0 * np.pi * rho)  # (..., G)
    B_pol_grid[..., 0] = 0.0  # on-axis B_pol → 0 (limit of I_enc/r)
    B_pol_s = side * _interp_last_axis(B_pol_grid, rho, r_sight)  # (..., C)
    return np.arctan2(B_pol_s, B_tor)


def _interp_last_axis(
    values: np.ndarray, grid: np.ndarray, query: np.ndarray
) -> np.ndarray:
    """Linear interp of ``values`` (..., G) over ``grid`` (G,) at ``query`` (C,).

    Vectorised over the leading (batch/time) dims via searchsorted + gather —
    the torch-portable pattern (no python loop over batch).
    """
    grid = np.asarray(grid, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64)
    G = grid.shape[0]
    # locate bracketing indices
    idx = np.searchsorted(grid, query, side="right") - 1
    idx = np.clip(idx, 0, G - 2)  # (C,)
    g0 = grid[idx]
    g1 = grid[idx + 1]
    w = np.where(g1 > g0, (query - g0) / (g1 - g0), 0.0)  # (C,)
    v0 = values[..., idx]  # (..., C)
    v1 = values[..., idx + 1]  # (..., C)
    out = v0 + (v1 - v0) * w
    # clamp queries beyond the grid to the endpoints
    out = np.where(query <= grid[0], values[..., :1], out)
    out = np.where(query >= grid[-1], values[..., -1:], out)
    return out


# ---------------------------------------------------------------------------
# SHARED pitch → (q0, rax) inverse map  (SECONDARY, method-matched)
# ---------------------------------------------------------------------------


def invert_pitch_to_q0rax(
    pitch_by_channel: np.ndarray,
    geometry: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """SHARED inverse: pitch profile → (q0, rax).  SECONDARY metric.

    Applied IDENTICALLY to predicted pitch and truth pitch, so the resulting
    q0/rax differ only by model error, not by inversion method.  We deliberately
    do NOT replicate the proprietary MSE 2pt/4pt κ=1.85 pipeline — q0/rax are
    SECONDARY and carry a method-mismatch caveat (see module docstring).

    Definitions used here (a reasonable, physical shared inversion):

    * ``rax`` = the major radius where the pitch profile crosses zero
      (B_pol → 0 ⇒ magnetic axis), chosen as the crossing nearest the centroid
      of the sightline radii (robust to noisy near-axis channels).
    * ``q0``  = on-axis safety factor estimated from the local pitch *gradient*
      at the axis crossing via the cylindrical limit
          q0 ≈ (B_tor0 / R0) / (dB_pol/dr)|_axis,
      with B_pol = B_tor · tan(pitch) and dB_pol/dr from a local linear fit of
      tan(pitch) vs r around the crossing.  q0 is clamped to the physical gate
      [Q0_MIN, Q0_MAX] used by the split's gating mask.

    Parameters
    ----------
    pitch_by_channel:
        (K, C) pitch (rad) per slice per sightline (radial channel order).
    geometry:
        dict with keys ``rpos`` ((C,) sightline major radii), ``R0``, ``Bt0``.

    Returns
    -------
    (q0, rax) : tuple of (K,) arrays.  NaN where no physical crossing is found.
    """
    from imas_ambix.statespace.mse_split import Q0_MAX, Q0_MIN  # noqa: PLC0415

    pitch = np.asarray(pitch_by_channel, dtype=np.float64)
    if pitch.ndim == 1:
        pitch = pitch[np.newaxis, :]
    R = np.asarray(geometry["rpos"], dtype=np.float64).reshape(-1)  # (C,)
    R0 = float(geometry.get("R0", DEFAULT_R0))
    # Bt0 cancels in the q0 = (1/R0)/slope estimate (B_pol/B_tor ratio), so it is
    # not used here; it is part of the geometry dict for the forward model.
    K, C = pitch.shape

    r_centroid = float(np.nanmean(R))
    tan_p = np.tan(pitch)  # B_pol ∝ tan(pitch) (B_tor slowly varying)

    q0 = np.full(K, np.nan)
    rax = np.full(K, np.nan)
    for k in range(K):
        fin = np.isfinite(pitch[k]) & (np.abs(pitch[k]) <= np.pi / 2.0)
        if fin.sum() < 4:
            continue
        rr = R[fin]
        pp = pitch[k, fin]
        tp = tan_p[k, fin]
        # sort by radius (channels already radial, but be safe)
        srt = np.argsort(rr)
        rr, pp, tp = rr[srt], pp[srt], tp[srt]
        # zero crossings of pitch
        sgn = np.sign(pp)
        zc = np.where(np.diff(sgn) != 0)[0]
        if zc.size == 0:
            continue
        # choose the crossing whose interpolated R is nearest the centroid
        cross_R = []
        for i in zc:
            denom = pp[i + 1] - pp[i]
            if denom == 0:
                continue
            r0 = rr[i] - pp[i] * (rr[i + 1] - rr[i]) / denom
            cross_R.append((abs(r0 - r_centroid), r0, i))
        if not cross_R:
            continue
        cross_R.sort()
        _, r0, i = cross_R[0]
        rax[k] = r0
        # local gradient of tan(pitch) vs minor radius around the crossing
        r_minor = np.abs(rr - R0)
        # use a 4-point window centred on the crossing for the slope
        lo = max(0, i - 1)
        hi = min(len(rr), i + 3)
        if hi - lo >= 2:
            slope = np.polyfit(r_minor[lo:hi], tp[lo:hi], 1)[0]
            if np.isfinite(slope) and slope != 0:
                # B_pol = Bt0 * tan(pitch); dB_pol/dr = Bt0 * slope
                # q0 ≈ (Bt0/R0) / (dB_pol/dr) = (1/R0)/slope
                q = abs((1.0 / R0) / slope)
                q0[k] = float(np.clip(q, Q0_MIN, Q0_MAX))
    return q0, rax


# ---------------------------------------------------------------------------
# Reference baseline: persistence (freeze pitch at first beam-on slice)
# ---------------------------------------------------------------------------


class PersistencePredictor:
    """Sanity baseline: freeze pitch at the first pitch-valid beam-on slice.

    pitch_std is set to the per-channel ``pitcha_error`` at the freeze slice
    (so coverage is non-degenerate).  q0/rax are left for :func:`score` to
    derive via the shared inversion (method-matched).
    """

    name = "persistence"

    def predict(self, manifest: dict, truth: MseTruth) -> dict[int, ShotPrediction]:
        preds: dict[int, ShotPrediction] = {}
        for sid_str, entry in manifest["shots"].items():
            sid = int(sid_str)
            if entry["partition"] != "held_out":
                continue
            tr = truth.get(sid)
            if tr is None:
                continue
            t = np.asarray(entry["beam_on_slice_times"])
            pv = np.asarray(entry["pitch_valid_mask"], dtype=bool)
            C = len(entry["active_channel_ids"])
            valid_idx = np.where(pv)[0]
            if valid_idx.size == 0:
                continue
            f = valid_idx[0]
            frozen_pitch = tr.pitch[f]  # (C,)
            frozen_err = tr.pitch_error[f]  # (C,)
            # default std where error missing/non-finite
            std = np.where(np.isfinite(frozen_err) & (frozen_err > 0), frozen_err, 0.1)
            pitch_mean = np.broadcast_to(frozen_pitch, (t.shape[0], C)).copy()
            pitch_std = np.broadcast_to(std, (t.shape[0], C)).copy()
            # fill non-finite frozen channels with the median (stay finite)
            med = (
                np.nanmedian(frozen_pitch[np.isfinite(frozen_pitch)])
                if np.isfinite(frozen_pitch).any()
                else 0.0
            )
            pitch_mean = np.where(np.isfinite(pitch_mean), pitch_mean, med)
            preds[sid] = ShotPrediction(t=t, pitch_mean=pitch_mean, pitch_std=pitch_std)
        return preds


# ---------------------------------------------------------------------------
# Truth loader (reads pitch/q0/rax on demand from the level-1 corpus)
# ---------------------------------------------------------------------------


@dataclass
class MseTruth:
    """On-demand truth provider for scored shots (single-sourced from level-1)."""

    level1_dir: Path
    _cache: dict = field(default_factory=dict)

    def get(self, shot_id: int):
        from imas_ambix.statespace.mse_split import read_ams_shot  # noqa: PLC0415

        if shot_id not in self._cache:
            self._cache[shot_id] = read_ams_shot(self.level1_dir / f"{shot_id}.zarr")
        return self._cache[shot_id]


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _rmse(y_true: np.ndarray, y_mean: np.ndarray) -> float:
    d = y_mean - y_true
    d = d[np.isfinite(d)]
    return float(np.sqrt(np.mean(d**2))) if d.size else float("nan")


def _coverage90(y_true, y_mean, y_std) -> float:
    return cal.interval_coverage(y_true, y_mean, y_std, alpha=0.10)


def _crps(y_true, y_mean, y_std, samples=None) -> float:
    if samples is not None:
        return cal.crps_ensemble(y_true, samples)
    return cal.crps_gaussian(y_true, y_mean, y_std)


def _transient_mask(
    pitch_truth: np.ndarray, t: np.ndarray, thresh_q: float = 0.75
) -> np.ndarray:
    """(K,) bool — high |d(pitch)/dt| slices (transient window).

    Uses the per-slice mean |Δpitch/Δt| over channels; threshold = upper
    quartile.  Simple v0 definition.
    """
    if t.shape[0] < 3:
        return np.zeros(t.shape[0], dtype=bool)
    dt = np.gradient(t)
    finite = np.isfinite(pitch_truth)
    has_any = finite.any(axis=1)
    # sum/count avoids nanmean's "empty slice" warning on all-NaN rows
    safe = np.where(finite, pitch_truth, 0.0)
    counts = finite.sum(axis=1)
    per_slice = np.where(
        has_any, safe.sum(axis=1) / np.where(counts > 0, counts, 1), 0.0
    )
    dp = np.gradient(per_slice)  # (K,)
    rate = np.abs(dp / np.where(dt != 0, dt, np.nan))
    finite = rate[np.isfinite(rate)]
    if finite.size == 0:
        return np.zeros(t.shape[0], dtype=bool)
    thr = np.quantile(finite, thresh_q)
    return np.nan_to_num(rate, nan=0.0) >= thr


def _flatten_finite(*arrays):
    """Stack arrays and keep positions where ALL are finite."""
    arrs = [np.asarray(a).reshape(-1) for a in arrays]
    mask = np.ones(arrs[0].shape, dtype=bool)
    for a in arrs:
        mask &= np.isfinite(a)
    return [a[mask] for a in arrs], mask


def _wmean(values: np.ndarray, weights: np.ndarray | None) -> float:
    """Weighted mean (plain mean when ``weights`` is None)."""
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return float("nan")
    if weights is None:
        return float(np.mean(v))
    w = np.asarray(weights, dtype=np.float64)
    sw = float(np.sum(w))
    if sw <= 0:
        return float(np.mean(v))
    return float(np.sum(w * v) / sw)


def _metric_block(y_true, y_mean, y_std, samples=None, weights=None) -> dict:
    """Per-point metrics, optionally error-WEIGHTED.

    When ``weights`` is supplied (per-point, same shape as the inputs before
    flattening), RMSE / CRPS / NLL are computed as weighted means so noisy
    (high-pitcha_error) points count less.  ``weights`` are typically
    inverse-variance (1 / pitcha_error²), normalised internally.

    COVERAGE (``cov90``) is intentionally left UNWEIGHTED — it is the
    pre-registered gate metric (acceptance band [0.88, 0.92]) and is registered
    against the plain empirical fraction-in-interval, not a low-error-biased
    weighted fraction.  Weighting it would change which calibration property the
    gate tests.
    """
    yt_full = np.asarray(y_true).reshape(-1)
    ym_full = np.asarray(y_mean).reshape(-1)
    ys_full = np.asarray(y_std).reshape(-1)
    finite = np.isfinite(yt_full) & np.isfinite(ym_full) & np.isfinite(ys_full)
    if weights is not None:
        wf = np.asarray(weights, dtype=np.float64).reshape(-1)
        finite &= np.isfinite(wf)
    yt, ym, ys = yt_full[finite], ym_full[finite], ys_full[finite]
    w = (
        np.asarray(weights, dtype=np.float64).reshape(-1)[finite]
        if weights is not None
        else None
    )
    if yt.size == 0:
        return {
            "rmse": float("nan"),
            "crps": float("nan"),
            "nll": float("nan"),
            "cov90": float("nan"),
            "n": 0,
        }
    # per-point quantities, then (weighted) mean
    sq_err = (ym - yt) ** 2
    crps_pp = _crps_per_point(yt, ym, ys, samples=samples, finite=finite)
    nll_pp = _nll_per_point(yt, ym, ys)
    inside = _inside90_per_point(yt, ym, ys)
    return {
        "rmse": float(np.sqrt(_wmean(sq_err, w))),
        "crps": _wmean(crps_pp, w),
        "nll": _wmean(nll_pp, w),
        # cov90 UNWEIGHTED — pre-registered gate metric (see docstring)
        "cov90": float(np.mean(inside.astype(np.float64))),
        "n": int(yt.size),
    }


def _pitch_block(
    pt_gated: np.ndarray,
    pm: np.ndarray,
    ps: np.ndarray,
    weights: np.ndarray,
    slice_mask: np.ndarray,
) -> dict:
    """Error-weighted pitch metric block restricted to ``slice_mask`` slices."""
    sm = slice_mask[:, None]
    return _metric_block(np.where(sm, pt_gated, np.nan), pm, ps, weights=weights)


def _crps_per_point(yt, ym, ys, samples=None, finite=None) -> np.ndarray:
    """Per-point Gaussian CRPS (closed form), shape (n_finite,)."""
    from scipy.stats import norm  # noqa: PLC0415

    sigma = np.abs(ys)
    sigma = np.where(sigma > 0, sigma, 1e-12)
    z = (yt - ym) / sigma
    return sigma * (
        z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - 1.0 / np.sqrt(np.pi)
    )


def _nll_per_point(yt, ym, ys) -> np.ndarray:
    sigma = np.abs(ys)
    sigma = np.where(sigma > 0, sigma, 1e-12)
    return 0.5 * (np.log(2.0 * np.pi * sigma**2) + ((yt - ym) / sigma) ** 2)


def _inside90_per_point(yt, ym, ys) -> np.ndarray:
    from scipy.stats import norm  # noqa: PLC0415

    zc = float(norm.ppf(1.0 - 0.10 / 2.0))
    return (yt >= ym - zc * ys) & (yt <= ym + zc * ys)


# ---------------------------------------------------------------------------
# score()
# ---------------------------------------------------------------------------


def score(
    predictions: dict[int, ShotPrediction],
    manifest: dict,
    truth: MseTruth | None = None,
) -> dict:
    """Score predictions on the HELD-OUT shots of ``manifest``.

    CANONICAL CONTRACT call is ``score(predictions, manifest)`` (2-arg): the
    truth is loaded on demand from the level-1 corpus.  Tests may pass an
    explicit ``truth`` provider (e.g. a synthetic stand-in).

    Returns the nested metric dict described in the contract:
        {primary:{pitch:{rmse,crps,nll,cov90, by_window:{quiescent,transient}}},
         secondary:{q0:{...}, rax:{...}},
         meta:{n_shots, coverage_gate, secondary_cross_check}}
    computed per-shot then aggregated as the MEAN over shots (N reported).
    """
    from imas_ambix.statespace import mse_split as M  # noqa: PLC0415

    if truth is None:
        from imas_ambix.data.paths import LEVEL1_DIR  # noqa: PLC0415

        truth = MseTruth(level1_dir=LEVEL1_DIR)
    shots_meta = manifest["shots"]
    per_shot_primary, per_shot_primary_q, per_shot_primary_tr = [], [], []
    per_shot_q0, per_shot_rax = [], []
    cross_check = []  # inv(truth_pitch) vs raw q0_4pt agreement

    scored_ids = []
    for sid, pred in predictions.items():
        entry = shots_meta.get(str(sid))
        if entry is None or entry["partition"] != "held_out":
            continue
        tr = truth.get(sid)
        if tr is None:
            continue
        C = len(entry["active_channel_ids"])
        try:
            pred.validate(C)
        except ValueError as e:
            logger.warning("Shot %s prediction rejected: %s", sid, e)
            continue

        t = np.asarray(entry["beam_on_slice_times"])
        pv = np.asarray(entry["pitch_valid_mask"], dtype=bool)
        q0g = np.asarray(entry["q0_gated_mask"], dtype=bool)
        raxg = np.asarray(entry["rax_gated_mask"], dtype=bool)

        # --- PRIMARY: pitch on the PHYSICALLY-GATED point set ----------------
        # Apply the SHARED per-point pitch gate (rail + error) so railed /
        # high-uncertainty truth points are dropped, and error-WEIGHT the
        # metrics by inverse pitcha_error² so noisy points count less.
        pt_truth = np.array(tr.pitch, dtype=np.float64)  # (K, C)
        pm = pred.pitch_mean
        ps = pred.pitch_std
        point_gate = M.pitch_point_gate(tr.pitch, tr.pitch_error)  # (K, C) bool
        # mask gated-out truth points to NaN → dropped by _metric_block
        pt_gated = np.where(point_gate, pt_truth, np.nan)
        # inverse-variance weights from per-point pitcha_error (clip tiny errs)
        with np.errstate(invalid="ignore", divide="ignore"):
            pe = np.abs(np.asarray(tr.pitch_error, dtype=np.float64))
            pe = np.where(np.isfinite(pe) & (pe > 1e-3), pe, np.nan)
            weights = 1.0 / pe**2
        # points with no usable error get the median weight (count, not drop)
        med_w = np.nanmedian(weights) if np.isfinite(weights).any() else 1.0
        weights = np.where(np.isfinite(weights), weights, med_w)

        block_all = _pitch_block(pt_gated, pm, ps, weights, pv)
        # by-window (transient computed on the gated truth)
        trans = _transient_mask(pt_gated, t) & pv
        quies = pv & ~trans
        block_q = (
            _pitch_block(pt_gated, pm, ps, weights, quies) if quies.any() else None
        )
        block_tr = (
            _pitch_block(pt_gated, pm, ps, weights, trans) if trans.any() else None
        )
        per_shot_primary.append(block_all)
        per_shot_primary_q.append(block_q)
        per_shot_primary_tr.append(block_tr)

        # --- SECONDARY: method-matched q0/rax via shared inversion -----------
        geom = {
            "rpos": np.asarray(entry["active_channel_rpos"]),
            "R0": DEFAULT_R0,
            "Bt0": DEFAULT_BT0,
        }
        # truth q0/rax = inversion of truth pitch (method-matched)
        q0_truth, rax_truth = invert_pitch_to_q0rax(pt_truth, geom)
        # predicted q0/rax: use predictor-supplied if present, else invert pred pitch
        if pred.q0_mean is not None and pred.rax_mean is not None:
            q0_pred = np.asarray(pred.q0_mean)
            rax_pred = np.asarray(pred.rax_mean)
            q0_std = (
                np.asarray(pred.q0_std)
                if pred.q0_std is not None
                else np.full_like(q0_pred, 0.2)
            )
            rax_std = (
                np.asarray(pred.rax_std)
                if pred.rax_std is not None
                else np.full_like(rax_pred, 0.02)
            )
        else:
            q0_pred, rax_pred = invert_pitch_to_q0rax(pm, geom)
            # propagate pitch std → q0/rax by a small MC ensemble
            q0_pred, q0_std, rax_pred, rax_std = _mc_invert(pm, ps, geom)

        # gated secondary slice population: co-finite pitch-valid AND gated
        sec_q0_mask = pv & q0g & np.isfinite(q0_truth) & np.isfinite(q0_pred)
        sec_rax_mask = pv & raxg & np.isfinite(rax_truth) & np.isfinite(rax_pred)
        per_shot_q0.append(
            _metric_block(
                q0_truth[sec_q0_mask], q0_pred[sec_q0_mask], q0_std[sec_q0_mask]
            )
            if sec_q0_mask.any()
            else None
        )
        per_shot_rax.append(
            _metric_block(
                rax_truth[sec_rax_mask], rax_pred[sec_rax_mask], rax_std[sec_rax_mask]
            )
            if sec_rax_mask.any()
            else None
        )

        # cross-check: inv(truth_pitch) vs RAW q0_4pt on the raw gate
        raw_mask = q0g & np.isfinite(q0_truth) & np.isfinite(tr.q0)
        if raw_mask.any():
            d = q0_truth[raw_mask] - tr.q0[raw_mask]
            cross_check.append(float(np.median(np.abs(d))))
        scored_ids.append(sid)

    def _agg(blocks: list) -> dict:
        clean = [b for b in blocks if b is not None and b.get("n", 0) > 0]
        if not clean:
            return {
                "rmse": float("nan"),
                "crps": float("nan"),
                "nll": float("nan"),
                "cov90": float("nan"),
                "n_shots": 0,
            }
        return {
            "rmse": float(np.mean([b["rmse"] for b in clean])),
            "crps": float(np.mean([b["crps"] for b in clean])),
            "nll": float(np.mean([b["nll"] for b in clean])),
            "cov90": float(np.mean([b["cov90"] for b in clean])),
            "n_shots": len(clean),
        }

    primary_all = _agg(per_shot_primary)
    result = {
        "primary": {
            "pitch": {
                **primary_all,
                "by_window": {
                    "quiescent": _agg(per_shot_primary_q),
                    "transient": _agg(per_shot_primary_tr),
                },
            }
        },
        "secondary": {
            "q0": _agg(per_shot_q0),
            "rax": _agg(per_shot_rax),
            "caveat": (
                "method-matched (truth-q0 := invert_pitch_to_q0rax(truth_pitch)); "
                "raw q0_4pt NOT replicated — provisional secondary."
            ),
        },
        "meta": {
            "n_shots": len(scored_ids),
            "scored_shot_ids": sorted(scored_ids),
            "coverage_gate": [COVERAGE_GATE_LO, COVERAGE_GATE_HI],
            "primary_cov90_in_gate": bool(
                COVERAGE_GATE_LO <= primary_all["cov90"] <= COVERAGE_GATE_HI
            )
            if np.isfinite(primary_all["cov90"])
            else False,
            "secondary_cross_check_median_abs_q0_resid": (
                float(np.mean(cross_check)) if cross_check else float("nan")
            ),
        },
    }
    return result


def _mc_invert(pitch_mean, pitch_std, geom, n_mc: int = 16, seed: int = 0):
    """Propagate pitch (mean,std) → (q0,rax) mean/std by a small MC ensemble."""
    rng = np.random.default_rng(seed)
    K, C = pitch_mean.shape
    q0_s = np.full((n_mc, K), np.nan)
    rax_s = np.full((n_mc, K), np.nan)
    for m in range(n_mc):
        sample = pitch_mean + pitch_std * rng.standard_normal((K, C))
        q0_s[m], rax_s[m] = invert_pitch_to_q0rax(sample, geom)
    q0_mean = np.nanmean(q0_s, axis=0)
    q0_std = np.nanstd(q0_s, axis=0)
    rax_mean = np.nanmean(rax_s, axis=0)
    rax_std = np.nanstd(rax_s, axis=0)
    # floor std so coverage is non-degenerate
    q0_std = np.where(np.isfinite(q0_std) & (q0_std > 1e-3), q0_std, 0.1)
    rax_std = np.where(np.isfinite(rax_std) & (rax_std > 1e-4), rax_std, 0.01)
    return q0_mean, q0_std, rax_mean, rax_std


def load_manifest(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Smoke test (runnable via `uv run python -m imas_ambix.statespace.mse_eval`)
# ---------------------------------------------------------------------------


def _smoke() -> int:
    """Build a small split, run PersistencePredictor, assert finite metrics.

    Returns a process exit code (0 = pass).
    """
    import os  # noqa: PLC0415

    from imas_ambix.data.paths import LEVEL1_DIR, MANIFEST_DIR  # noqa: PLC0415
    from imas_ambix.statespace import mse_split as M  # noqa: PLC0415

    inv = MANIFEST_DIR / "statespace_family_inventory.json"
    sg = M.load_inventory_shot_groups(inv)

    # small subset: first ~25 present ams shots
    ams = M.ams_shots(sg)
    present = [s for s in ams if os.path.exists(f"{LEVEL1_DIR}/{s}.zarr")][:25]
    beam_on = M.find_beam_on_shots(present, LEVEL1_DIR, max_workers=8)
    print(f"[smoke] {len(beam_on)} beam-on of {len(present)} present ams shots")

    # Force a small held-out set for the smoke (treat all beam-on as held_out).
    split = M.MseSplit(
        train=sorted(set(M.tier1_union_shots(sg)) - set(beam_on)),
        calibration=[],
        held_out=beam_on,
        train_input_groups=sorted(
            set(M.TIER1_MAGNETICS)
            | set(M.TIER1_CAMERA)
            | set(M.TIER1_BOLOMETER)
            | set(M.TIER1_SXR)
        ),
        n_ams_total=len(ams),
        n_ams_beam_on=len(beam_on),
        n_ams_beam_off=len(ams) - len(beam_on),
    )

    # Gate assertion 1: TRAIN ∩ eval == ∅ and no MSE in train inputs.
    split.assert_no_mse_in_train()
    eval_set = set(split.calibration) | set(split.held_out)
    assert not (set(split.train) & eval_set), "TRAIN overlaps eval!"
    assert "ams" not in split.train_input_groups, "MSE group in TRAIN inputs!"
    print(
        f"[smoke] PASS: TRAIN ({len(split.train)}) ∩ eval ({len(eval_set)}) = ∅; "
        "no MSE group in TRAIN inputs"
    )

    entries = M.build_manifests_parallel(split, LEVEL1_DIR, max_workers=8)
    manifest = {
        "version": "smoke",
        "summary": split.summary_dict(),
        "shots": {str(k): v for k, v in entries.items()},
    }

    truth = MseTruth(level1_dir=LEVEL1_DIR)
    pred = PersistencePredictor().predict(manifest, truth)
    print(f"[smoke] PersistencePredictor produced {len(pred)} shot predictions")

    # Shape contract check
    for sid, p in pred.items():
        C = len(manifest["shots"][str(sid)]["active_channel_ids"])
        p.validate(C)

    result = score(pred, manifest, truth)
    print("[smoke] metrics:")
    print(json.dumps(result, indent=2))

    # Assertions: primary pitch metrics finite + sensible
    pp = result["primary"]["pitch"]
    assert pp["n_shots"] > 0, "no shots scored"
    assert np.isfinite(pp["rmse"]), "pitch RMSE not finite"
    assert np.isfinite(pp["crps"]), "pitch CRPS not finite"
    assert np.isfinite(pp["nll"]), "pitch NLL not finite"
    assert 0.0 <= pp["cov90"] <= 1.0, "pitch coverage out of [0,1]"
    print("[smoke] PASS: primary pitch metrics finite and in-range")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(_smoke())
