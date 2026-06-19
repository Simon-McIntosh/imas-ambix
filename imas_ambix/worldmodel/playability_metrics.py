"""Playability metrics for camera world-model rollouts vs a persistence baseline.

Why a new metric module (and why pixel-MSE-vs-persistence is the WRONG gate)
---------------------------------------------------------------------------
The camera world model collapses to persistence under a greedy *argmax* rollout.
The previously-committed scorer (``spacetime_dream.forecast_pixel_errors``) is a
per-pixel absolute error of a SINGLE prediction against the persistence baseline
(freeze the last context frame).  That metric structurally favours persistence on
this data: the plasma frames change little frame-to-frame, so the frozen frame is
a low-error point estimate, and ANY plausible motion a generative model adds is
penalised wherever it does not land exactly on the truth.  This is the standard
*perception–distortion* trade-off — a distortion (MSE/MAE) optimum is the blurry
conditional mean / the frozen frame, NOT a sharp realistic sample.  So pixel-MAE
cannot, by construction, credit a model that has learnt the right *distribution*
of plasma evolution; it can only ever say "persistence wins".  We keep it, but
only as a luminance-FAIR SANITY readout, never as the gate.

What this module scores instead
-------------------------------
A model that escapes persistence collapse should (a) match the DISTRIBUTION of
plausible futures, and (b) reproduce the right amount of frame-to-frame MOTION.
Three families, each able to credit a generative model:

1. ``ensemble_crps`` — a per-pixel CRPS (Continuous Ranked Probability Score)
   computed over an ENSEMBLE of N sampled rollouts.  CRPS is a strictly proper
   scoring rule for a probabilistic forecast: it rewards an ensemble that brackets
   the truth with appropriate spread and punishes both bias and over/under-
   dispersion.  Crucially, persistence is a DEGENERATE (zero-spread) ensemble, for
   which the CRPS reduces exactly to the MAE — so a well-calibrated sampled
   ensemble can BEAT persistence's CRPS even where its per-member MAE does not.
   This is the distributional verdict.  (A learned-feature FVD would be the
   textbook choice, but no I3D/video-feature extractor is staged offline on the
   betelgeuse GPU node — no outbound network — so a per-pixel proper scoring rule
   over the sampled ensemble is the documented offline proxy.)

2. ``motion_fraction`` / ``motion_report`` — the frame-to-frame change fraction
   (fraction of pixels whose value moves by more than a tolerance between
   consecutive frames) of a rollout, compared with the GT's own change fraction.
   A collapsed (persistence) rollout has change fraction ~0; a healthy rollout
   tracks the GT's change fraction.  This is the non-collapse measure and reads
   directly in pixel space (unlike the token-space change fraction, which stays
   high under collapse because redundant codebook ids decode to near-identical
   dark frames).

3. ``ssim_report`` — a Gaussian-windowed SSIM of the rollout vs the persistence
   baseline, plus a luminance-normalised MAE.  SSIM is structural (less dominated
   by the dark-background luminance than raw MAE), but it is STILL a distortion
   metric and on near-static scenes persistence remains hard to beat — so this is
   reported as a SANITY check WITH that caveat, never as the gate.

A control-divergence measure (does changing the actuator plan change the dream?)
is the right M3 object and is intentionally LEFT as a stub here
(:func:`control_divergence_stub`) — it needs the actuator-conditioned model from a
later milestone, not this checkpoint.

All functions operate on decoded uint8/float image stacks ``(F, H, W[, C])`` on a
common scale and score ONLY the forecast frames (``f >= ctx``); the context frames
are excluded exactly as in the committed pixel scorer.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _to_gray_f64(stack: np.ndarray) -> np.ndarray:
    """``(F, H, W[, C]) -> (F, H, W)`` float64 luminance (mean over channels)."""
    arr = np.asarray(stack, dtype=np.float64)
    if arr.ndim == 4:
        arr = arr.mean(axis=-1)
    if arr.ndim != 3:
        raise ValueError(f"expected (F,H,W[,C]) image stack, got shape {arr.shape}")
    return arr


def _check_forecast(n_frames: int, ctx: int) -> slice:
    if not 1 <= ctx < n_frames:
        raise ValueError(f"ctx {ctx} out of range for {n_frames} frames")
    return slice(ctx, n_frames)


def persistence_stack(gt: np.ndarray, ctx: int) -> np.ndarray:
    """The persistence baseline as a full ``(F, ...)`` stack.

    Frames ``< ctx`` are the truth (context); every forecast frame ``>= ctx`` is
    the last context frame ``gt[ctx-1]`` frozen.  Returned on the GT's own dtype
    so downstream metrics treat it like any other prediction.
    """
    g = np.asarray(gt)
    _check_forecast(g.shape[0], ctx)
    out = g.copy()
    out[ctx:] = g[ctx - 1][None]
    return out


# ---------------------------------------------------------------------------
# 1. Distributional score — per-pixel ensemble CRPS
# ---------------------------------------------------------------------------


def _crps_ensemble_pixelwise(ensemble: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Per-element CRPS of an ``(N, ...)`` ensemble against ``(...)`` truth.

    Uses the standard energy-form estimator

        CRPS ≈ mean_i |X_i - y|  -  0.5 * mean_{i,j} |X_i - X_j|

    evaluated independently per pixel.  The first term is the ensemble's mean
    absolute error to the truth; the second rewards ensemble SPREAD (a zero-spread
    / degenerate ensemble loses the −0.5·spread credit and so its CRPS equals its
    MAE).  Returns the per-element CRPS array (same trailing shape as ``truth``).
    """
    ens = np.asarray(ensemble, dtype=np.float64)
    y = np.asarray(truth, dtype=np.float64)
    n = ens.shape[0]
    if ens.shape[1:] != y.shape:
        raise ValueError(f"ensemble {ens.shape[1:]} vs truth {y.shape} shape mismatch")
    # term 1: E|X - y|
    skill = np.abs(ens - y[None]).mean(axis=0)
    # term 2: 0.5 E|X - X'| — the unbiased pairwise-difference (sum over i<j).
    # Computed as 1/N^2 sum_{i,j}|X_i - X_j| (biased-by-1/N but standard for the
    # finite-ensemble CRPS estimator; the bias is identical across compared
    # ensembles so it does not affect the persistence comparison).
    if n < 2:
        spread = np.zeros_like(skill)
    else:
        acc = np.zeros_like(skill)
        for i in range(n):
            acc += np.abs(ens[i][None] - ens).sum(axis=0)
        spread = acc / (n * n)
    return skill - 0.5 * spread


def ensemble_crps(
    gt: np.ndarray,
    rollout_ensemble: np.ndarray,
    ctx: int,
) -> dict[str, float | int | bool]:
    """Mean per-pixel CRPS of a sampled rollout ENSEMBLE vs persistence.

    Parameters
    ----------
    gt:
        ``(F, H, W[, C])`` ground-truth decoded frames.
    rollout_ensemble:
        ``(N, F, H, W[, C])`` decoded frames for ``N`` sampled rollouts (same
        window, same context).  ``N == 1`` is allowed (then the CRPS is the
        single-member MAE — no spread credit — which is the honest degenerate
        case for argmax).
    ctx:
        Number of leading context frames (scored only on ``f >= ctx``).

    Returns
    -------
    dict with the model and persistence mean CRPS over the forecast window, their
    ratio (model / persistence; ``< 1`` ⇒ model better), the ensemble size, and
    ``model_beats_persistence`` (lower CRPS is better).
    """
    g = _to_gray_f64(gt)
    ens = np.stack([_to_gray_f64(r) for r in np.asarray(rollout_ensemble)], axis=0)
    fc = _check_forecast(g.shape[0], ctx)
    n = int(ens.shape[0])

    model_crps = float(_crps_ensemble_pixelwise(ens[:, fc], g[fc]).mean())
    # persistence is a single, degenerate (zero-spread) ensemble member.
    pers = persistence_stack(g, ctx)
    pers_crps = float(_crps_ensemble_pixelwise(pers[None, fc], g[fc]).mean())
    ratio = float("inf") if pers_crps == 0.0 else model_crps / pers_crps
    return {
        "model_crps": model_crps,
        "persistence_crps": pers_crps,
        "ratio": ratio,
        "ensemble_size": n,
        "model_beats_persistence": bool(model_crps < pers_crps),
    }


# ---------------------------------------------------------------------------
# 2. Motion / non-collapse — frame-to-frame change fraction
# ---------------------------------------------------------------------------


def change_fraction(stack: np.ndarray, ctx: int, *, tol: float = 2.0) -> float:
    """Mean fraction of pixels that move > ``tol`` between consecutive frames.

    Computed over the forecast window (the transitions ``ctx-1 -> ctx``, …,
    ``F-2 -> F-1``).  A collapsed (frozen-frame) rollout has change fraction ~0;
    a moving scene has a positive fraction.  ``tol`` is in the stack's value units
    (default 2 on a 0..255 uint8 scale — robust to decode quantisation jitter).
    """
    g = _to_gray_f64(stack)
    n = g.shape[0]
    _check_forecast(n, ctx)
    diffs = np.abs(g[ctx:] - g[ctx - 1 : n - 1])  # (n-ctx, H, W) consecutive deltas
    return float((diffs > float(tol)).mean())


def motion_report(
    gt: np.ndarray,
    pred: np.ndarray,
    ctx: int,
    *,
    tol: float = 2.0,
) -> dict[str, float]:
    """Change-fraction of a (single) rollout vs the GT's own change-fraction.

    ``collapse_ratio`` = pred change fraction / GT change fraction: ~0 means the
    rollout collapsed to a frozen frame, ~1 means it reproduces the GT's amount of
    motion (the non-collapse target).  Also reports the persistence baseline's
    change fraction (≈0 by construction — the frozen-frame reference for
    "collapsed").
    """
    g = _to_gray_f64(gt)
    gt_cf = change_fraction(g, ctx, tol=tol)
    pred_cf = change_fraction(pred, ctx, tol=tol)
    pers_cf = change_fraction(persistence_stack(g, ctx), ctx, tol=tol)
    ratio = float("inf") if gt_cf == 0.0 else pred_cf / gt_cf
    return {
        "gt_change_fraction": gt_cf,
        "pred_change_fraction": pred_cf,
        "persistence_change_fraction": pers_cf,
        "collapse_ratio": ratio,
    }


# ---------------------------------------------------------------------------
# 3. Luminance-fair pixel SANITY — windowed SSIM (NOT the gate)
# ---------------------------------------------------------------------------


def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    ax = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    g = np.exp(-(ax**2) / (2.0 * sigma**2))
    g = g / g.sum()
    return np.outer(g, g)


def _conv2d_valid(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """2-D 'valid' correlation of a single ``(H, W)`` image with ``kernel``.

    A small dependency-free convolution (the inference venv has no scikit-image);
    uses :func:`scipy.signal.fftconvolve` when available, else a direct stride
    sum.  Kernel is symmetric so correlation == convolution.
    """
    try:
        from scipy.signal import fftconvolve  # noqa: PLC0415

        return fftconvolve(img, kernel, mode="valid")
    except Exception:  # noqa: BLE001 — fall back to a direct windowed sum
        kh, kw = kernel.shape
        h, w = img.shape
        out = np.zeros((h - kh + 1, w - kw + 1), dtype=np.float64)
        for i in range(kh):
            for j in range(kw):
                out += kernel[i, j] * img[i : i + out.shape[0], j : j + out.shape[1]]
        return out


def _ssim_pair(a: np.ndarray, b: np.ndarray, *, data_range: float = 255.0) -> float:
    """Mean Gaussian-windowed SSIM between two ``(H, W)`` float images."""
    kernel = _gaussian_kernel()
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_a = _conv2d_valid(a, kernel)
    mu_b = _conv2d_valid(b, kernel)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sa = _conv2d_valid(a * a, kernel) - mu_a2
    sb = _conv2d_valid(b * b, kernel) - mu_b2
    sab = _conv2d_valid(a * b, kernel) - mu_ab
    num = (2 * mu_ab + c1) * (2 * sab + c2)
    den = (mu_a2 + mu_b2 + c1) * (sa + sb + c2)
    return float((num / den).mean())


def ssim_report(
    gt: np.ndarray,
    pred: np.ndarray,
    ctx: int,
    *,
    data_range: float = 255.0,
) -> dict[str, float | bool | str]:
    """Forecast-window SSIM + luminance-normalised MAE of pred vs persistence.

    SANITY ONLY — SSIM is structural so it is fairer to a generative model than
    raw MAE, but it remains a DISTORTION metric and on a near-static plasma scene
    persistence is still hard to beat.  Reported alongside the distributional
    verdict, never as the gate; the ``caveat`` field states this in-band.

    ``mean_ssim`` is the per-forecast-frame mean SSIM of the prediction to the
    GT; ``persistence_ssim`` the same for the frozen-frame baseline.
    ``lum_norm_mae_ratio`` is the prediction's luminance-normalised MAE divided by
    persistence's (each frame mean-subtracted before the MAE, so the dark-
    background DC level cannot dominate); ``< 1`` ⇒ the prediction is closer.
    """
    g = _to_gray_f64(gt)
    p = _to_gray_f64(pred)
    pers = persistence_stack(g, ctx)
    fc = _check_forecast(g.shape[0], ctx)

    pred_ssim = float(
        np.mean(
            [
                _ssim_pair(p[f], g[f], data_range=data_range)
                for f in range(*fc.indices(g.shape[0]))
            ]
        )
    )
    pers_ssim = float(
        np.mean(
            [
                _ssim_pair(pers[f], g[f], data_range=data_range)
                for f in range(*fc.indices(g.shape[0]))
            ]
        )
    )

    def _lum_norm_mae(stack: np.ndarray) -> float:
        s = stack[fc]
        ref = g[fc]
        s = s - s.mean(axis=(1, 2), keepdims=True)
        ref0 = ref - ref.mean(axis=(1, 2), keepdims=True)
        return float(np.abs(s - ref0).mean())

    pred_lm = _lum_norm_mae(p)
    pers_lm = _lum_norm_mae(pers)
    lm_ratio = float("inf") if pers_lm == 0.0 else pred_lm / pers_lm
    return {
        "mean_ssim": pred_ssim,
        "persistence_ssim": pers_ssim,
        "ssim_beats_persistence": bool(pred_ssim > pers_ssim),
        "lum_norm_mae_ratio": lm_ratio,
        "caveat": (
            "SANITY ONLY: SSIM/luminance-normalised MAE are distortion metrics; on "
            "a near-static scene persistence is hard to beat by construction — use "
            "ensemble_crps + motion_report as the verdict, not these."
        ),
    }


# ---------------------------------------------------------------------------
# Control-divergence — M3 stub
# ---------------------------------------------------------------------------


def control_divergence_stub() -> dict[str, str]:
    """Placeholder for the M3 controllability measure (not this milestone).

    The right test of *playability* is: does perturbing the ACTUATOR plan change
    the dream (and in the physically expected direction)?  That needs the
    actuator-conditioned model from a later milestone — this checkpoint conditions
    on measured diagnostics, which are FIXED context, so a control sweep is not
    yet meaningful.  Documented here so the harness surface is complete; M3 fills
    it in.
    """
    return {
        "status": "stub",
        "reason": (
            "control-divergence (vary actuator plan -> measure dream change) needs "
            "the actuator-conditioned model; deferred to a later milestone"
        ),
    }


__all__ = [
    "change_fraction",
    "control_divergence_stub",
    "ensemble_crps",
    "motion_report",
    "persistence_stack",
    "ssim_report",
]
