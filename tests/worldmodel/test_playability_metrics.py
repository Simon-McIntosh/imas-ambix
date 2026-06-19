"""Unit tests for the playability metric harness.

These pin the load-bearing PROPERTIES of each metric so the held-out verdict is
reproducible and the metrics behave as designed:

* ensemble CRPS reduces to MAE for a degenerate (zero-spread) ensemble, and a
  spread ensemble that brackets the truth can BEAT persistence's CRPS even where
  its per-member MAE would not (the perception–distortion escape);
* the motion / change-fraction is ~0 for a collapsed (frozen) rollout and
  positive + close to GT for a moving one;
* SSIM is 1.0 for an exact match and the luminance-normalised MAE ignores a pure
  DC offset (the luminance-fair property);
* only forecast frames are scored.
"""

from __future__ import annotations

import numpy as np

from imas_ambix.worldmodel.playability_metrics import (
    change_fraction,
    ensemble_crps,
    motion_report,
    persistence_stack,
    ssim_report,
)


def _const_frame(h, w, val):
    return np.full((h, w, 3), val, dtype=np.float64)


def _stack(vals, h=8, w=8):
    return np.stack([_const_frame(h, w, v) for v in vals], axis=0)


# ---------------------------------------------------------------------------
# persistence_stack
# ---------------------------------------------------------------------------


def test_persistence_stack_freezes_last_context():
    gt = _stack([10, 20, 30, 40])
    pers = persistence_stack(gt, ctx=2)
    # context frames untouched
    assert np.all(pers[0] == 10) and np.all(pers[1] == 20)
    # forecast frames frozen at gt[ctx-1] == frame 1 (value 20)
    assert np.all(pers[2] == 20) and np.all(pers[3] == 20)


# ---------------------------------------------------------------------------
# ensemble CRPS
# ---------------------------------------------------------------------------


def test_crps_degenerate_ensemble_equals_mae():
    """A single-member (zero-spread) ensemble's CRPS == its MAE to truth."""
    gt = _stack([10, 20, 30, 40])
    # one member that is wrong by a constant 5 on the forecast frames
    member = _stack([10, 20, 35, 45])
    out = ensemble_crps(gt, member[None], ctx=2)
    # forecast frames are 30,40; member is 35,45 -> |diff| = 5 each -> MAE 5
    assert out["ensemble_size"] == 1
    assert abs(out["model_crps"] - 5.0) < 1e-9
    # persistence freezes frame1 (20): off by |30-20|,|40-20| -> mean 15
    assert abs(out["persistence_crps"] - 15.0) < 1e-9
    assert out["model_beats_persistence"] is True


def test_crps_spread_ensemble_beats_persistence_where_mae_would_not():
    """An ensemble bracketing the truth beats persistence on CRPS though its
    per-member MAE is worse — the perception–distortion escape the harness exists
    to credit.

    GT forecast frame = 30 (single forecast frame, ctx=2 of 3).  Three members at
    10, 30, 50 bracket it.  Per-MEMBER MAE = (|10-30|+0+|50-30|)/3 = 13.33, WORSE
    than persistence's |30-20| = 10.  But the energy-form CRPS
    ``E|X-y| - 0.5 E|X-X'|`` = 13.33 - 0.5·(160/9) = 4.444 BEATS persistence's
    CRPS of 10, because the ensemble spread is rewarded.  So a distortion (MAE)
    metric says "persistence wins" while the proper score says "the ensemble
    wins" — exactly the discrimination this harness adds.
    """
    gt = _stack([10, 20, 30])  # ctx=2 -> forecast frame index 2, value 30
    members = [_stack([10, 20, v]) for v in (10, 30, 50)]
    ens = np.stack(members, axis=0)
    out = ensemble_crps(gt, ens, ctx=2)
    assert out["ensemble_size"] == 3
    # per-member MAE (13.33) is WORSE than persistence (10) ...
    per_member_mae = float(np.mean([abs(v - 30) for v in (10, 30, 50)]))
    assert per_member_mae > out["persistence_crps"]
    # ... yet the ensemble CRPS beats persistence.
    assert abs(out["model_crps"] - 4.4444444) < 1e-4
    assert abs(out["persistence_crps"] - 10.0) < 1e-9
    assert out["model_beats_persistence"] is True


def test_crps_only_forecast_frames_scored():
    """Errors on context frames must not count toward the CRPS."""
    gt = _stack([10, 20, 30])
    member = _stack([99, 99, 30])  # garbage in context, exact on forecast
    out = ensemble_crps(gt, member[None], ctx=2)
    assert abs(out["model_crps"] - 0.0) < 1e-9


# ---------------------------------------------------------------------------
# motion / change fraction
# ---------------------------------------------------------------------------


def test_change_fraction_zero_for_frozen_rollout():
    """A collapsed (frozen-frame) rollout has ~0 change fraction."""
    gt = _stack([10, 20, 30, 40])
    frozen = persistence_stack(gt, ctx=2)
    assert change_fraction(frozen, ctx=2, tol=1.0) == 0.0


def test_change_fraction_positive_for_moving_rollout():
    gt = _stack([10, 20, 30, 40])  # every forecast transition moves by 10
    cf = change_fraction(gt, ctx=2, tol=1.0)
    assert cf == 1.0  # every pixel moves > tol on every forecast transition


def test_motion_report_collapse_ratio():
    """A frozen rollout has collapse_ratio ~0; a GT-matching one ~1."""
    gt = _stack([10, 20, 30, 40])
    frozen = persistence_stack(gt, ctx=2)
    rep_frozen = motion_report(gt, frozen, ctx=2, tol=1.0)
    assert rep_frozen["collapse_ratio"] == 0.0
    assert rep_frozen["gt_change_fraction"] > 0.0
    rep_match = motion_report(gt, gt, ctx=2, tol=1.0)
    assert abs(rep_match["collapse_ratio"] - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# SSIM sanity
# ---------------------------------------------------------------------------


def test_ssim_exact_match_is_one():
    rng = np.random.default_rng(0)
    gt = rng.integers(0, 256, size=(4, 16, 16, 3)).astype(np.float64)
    out = ssim_report(gt, gt.copy(), ctx=2)
    assert out["mean_ssim"] > 0.999


def test_lum_norm_mae_ignores_dc_offset():
    """A pure constant brightness offset must not register as error.

    The prediction equals GT plus a constant 30 everywhere; raw MAE would be 30,
    but the luminance-normalised MAE subtracts each frame's mean first, so the
    prediction is luminance-fair-identical -> ratio 0 (persistence also 0 only if
    static; here GT moves so persistence has positive lum-norm MAE).
    """
    gt = _stack([10, 20, 30, 40], h=16, w=16)
    # add structure so SSIM/variance terms are well defined, then a DC offset
    rng = np.random.default_rng(1)
    texture = rng.normal(0, 5, size=gt.shape)
    gt = gt + texture
    pred = gt + 30.0  # pure DC offset
    out = ssim_report(gt, pred, ctx=2)
    # luminance-normalised: pred matches gt after mean removal -> ~0 numerator
    assert out["lum_norm_mae_ratio"] < 1e-6
