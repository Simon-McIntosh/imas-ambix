"""Tests for the GS-readout-vs-referee scoring core (the gate-2 skill).

The gate scores the topology READ from the model's solved ψ against the
firewalled EFIT referee, per-quantity, exactly as the absolute-magnetics oracle
does: ``skill = 1 − RMSE_model / RMSE_baseline`` (baseline = train-mean
predictor).  The X-point is an order-invariant null set, so its error is a
PERMUTATION-INVARIANT match of the ≤2 predicted slots to the ≤2 reference slots.
"""

from __future__ import annotations

import numpy as np

from imas_ambix.latent.evaluate import (
    matched_xpoint_error,
    per_quantity_skill,
)


def test_matched_xpoint_error_is_permutation_invariant():
    pred = np.array([[1.0, 0.5], [1.2, -0.5]])
    ref_a = np.array([[1.0, 0.5], [1.2, -0.5]])
    ref_b = np.array([[1.2, -0.5], [1.0, 0.5]])  # swapped slots
    e_a = matched_xpoint_error(pred, ref_a)
    e_b = matched_xpoint_error(pred, ref_b)
    np.testing.assert_allclose(e_a, e_b)  # swap must not change the error
    np.testing.assert_allclose(e_a, 0.0, atol=1e-12)


def test_matched_xpoint_error_handles_absent_slots():
    pred = np.array([[1.0, 0.5], [np.nan, np.nan]])  # one predicted null
    ref = np.array([[1.05, 0.55], [np.nan, np.nan]])  # one reference null
    err = matched_xpoint_error(pred, ref)
    assert np.isfinite(err)
    np.testing.assert_allclose(err, np.hypot(0.05, 0.05), atol=1e-9)


def test_per_quantity_skill_matches_oracle_formula():
    """skill_i = 1 − RMSE_model_i / RMSE_baseline_i, per component."""
    # two samples, axis_R only for simplicity
    model = np.array([[1.01], [0.99]])
    ref = np.array([[1.00], [1.00]])
    baseline = np.array([[1.10], [0.90]])  # a worse (train-mean-like) predictor
    names = ["axis_R"]
    skill = per_quantity_skill(model, ref, baseline, names)
    rmse_m = np.sqrt(np.mean((model - ref) ** 2))
    rmse_b = np.sqrt(np.mean((baseline - ref) ** 2))
    np.testing.assert_allclose(skill["axis_R"], 1.0 - rmse_m / rmse_b, rtol=1e-9)
    assert skill["axis_R"] > 0  # model beats baseline


def test_per_quantity_skill_respects_mask():
    """A component with no finite reference yields NaN skill, not a crash."""
    model = np.array([[1.0, np.nan], [1.0, np.nan]])
    ref = np.array([[1.0, np.nan], [1.0, np.nan]])
    baseline = np.array([[1.5, np.nan], [0.5, np.nan]])
    names = ["axis_R", "axis_Z"]
    skill = per_quantity_skill(model, ref, baseline, names)
    assert np.isnan(skill["axis_Z"])  # no finite reference → undefined skill
    assert np.isfinite(skill["axis_R"])
