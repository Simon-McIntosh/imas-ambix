from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from nova.transport.current_diffusion import EtaProfile

from imas_ambix.closure import (
    corrected_resistivity,
    fit_resistivity_correction,
    paired_bootstrap_comparison,
)

_ARTIFACT = (
    Path(__file__).parents[2]
    / "imas_ambix"
    / "closure"
    / "artifacts"
    / "current_diffusion_calibration.json"
)


def test_correction_is_multiplicative_and_preserves_profile_family() -> None:
    baseline = EtaProfile(eta0=5.0e-8, contrast=2.0, shape=2.0)
    corrected = corrected_resistivity(baseline, 0.9)

    assert corrected.eta0 == pytest.approx(0.9 * baseline.eta0)
    assert corrected.contrast == baseline.contrast
    assert corrected.shape == baseline.shape
    with pytest.raises(ValueError):
        corrected_resistivity(baseline, 2.0)


def test_tight_prior_pulls_a_scalar_fit_toward_untuned_closure() -> None:
    fit = fit_resistivity_correction(
        lambda multiplier: (multiplier - 0.8) ** 2,
        prior_log_sigma=0.15,
        prior_weight=0.01,
    )

    assert fit.success
    assert 0.8 < fit.multiplier < 1.0
    assert fit.prior_penalty > 0.0


def test_paired_interval_controls_the_beat_verdict() -> None:
    beat = paired_bootstrap_comparison(
        np.array([0.8, 0.7, 0.9, 0.6, 0.75]),
        np.ones(5),
        draws=4000,
        seed=11,
    )
    miss = paired_bootstrap_comparison(
        np.array([0.8, 1.1, 0.9, 1.2, 1.0]),
        np.ones(5),
        draws=4000,
        seed=11,
    )

    assert beat["verdict"] == "BEAT"
    assert beat["paired_bootstrap_confidence_interval_95"][1] < 0.0
    assert miss["verdict"] == "MISS"
    lower, upper = miss["paired_bootstrap_confidence_interval_95"]
    assert lower <= 0.0 <= upper


def test_banked_artifact_records_the_complete_heldout_measurement() -> None:
    payload = json.loads(_ARTIFACT.read_text())
    corpus = payload["corpus"]
    method = payload["method"]
    comparison = payload["comparison"]
    li = payload["li_diagnostic"]

    assert payload["status"] == "complete"
    assert corpus["banked_shots"] == 20
    assert corpus["fit_shot_count"] == 15
    assert corpus["heldout_shot_count"] == 5
    assert corpus["split_disjoint"]
    assert not set(corpus["fit_shots"]) & set(corpus["heldout_shots"])
    assert method["same_rank_and_basis_in_both_arms"]
    assert method["basis_rank"] == 3
    assert method["free_form_profile_admitted"] is False
    assert payload["fit"]["shape_unchanged"]
    assert len(comparison["heldout_shots"]) == 5
    assert np.isfinite(comparison["tuned_error_mean"])
    assert np.isfinite(comparison["untuned_error_mean"])
    assert np.isfinite(comparison["paired_difference_tuned_minus_untuned"])
    interval = comparison["paired_bootstrap_confidence_interval_95"]
    assert len(interval) == 2 and np.all(np.isfinite(interval))
    expected = "BEAT" if interval[1] < 0.0 else "MISS"
    assert comparison["verdict"] == expected
    assert li["eligible_frames"] > 0
    assert np.isfinite(li["tuned_skill"])
    assert np.isfinite(li["untuned_skill"])
