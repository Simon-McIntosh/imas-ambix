"""Tests for Dalpha-selected ELM-frame response morphology scoring."""

from __future__ import annotations

import numpy as np
import pytest

from imas_ambix.camdyn.elm_morphology import (
    aligned_frame_indices,
    summarise_paired_scores,
)
from imas_ambix.camdyn.metrics import (
    ELM_MORPHOLOGY_HORIZON_MS,
    elm_edge_divertor_mask,
    elm_frame_morphology_fidelity,
)


def test_edge_divertor_mask_is_fixed_spatial_support():
    mask = elm_edge_divertor_mask()

    assert mask.shape == (16, 16)
    assert mask.dtype == np.bool_
    assert mask[-5:, :].all()
    assert mask[:, :3].all()
    assert mask[:, -3:].all()
    assert not mask[:11, 3:13].any()


def test_morphology_fidelity_rewards_the_correct_response_pattern():
    reference = np.full((32, 32), 40.0)
    target = reference.copy()
    target[-10:, 4:16] += 100.0
    perfect = target.copy()
    static = reference.copy()

    perfect_score = elm_frame_morphology_fidelity(perfect, target, reference)
    static_score = elm_frame_morphology_fidelity(static, target, reference)

    assert perfect_score["morphology_fidelity"] == pytest.approx(1.0)
    assert perfect_score["response_correlation"] == pytest.approx(1.0)
    assert perfect_score["brightness_fidelity"] == pytest.approx(1.0)
    assert static_score["morphology_fidelity"] == 0.0


def test_morphology_fidelity_ignores_changes_outside_registered_region():
    reference = np.zeros((32, 32), dtype=np.float64)
    target = reference.copy()
    predicted = reference.copy()
    target[-8:, :8] = 100.0
    predicted[:] = target
    predicted[10, 16] = 255.0

    score = elm_frame_morphology_fidelity(predicted, target, reference)
    assert score["morphology_fidelity"] == pytest.approx(1.0)


def test_morphology_fidelity_rejects_spatially_shuffled_response():
    reference = np.zeros((32, 32), dtype=np.float64)
    target = reference.copy()
    predicted = reference.copy()
    target[-10:, :8] = 120.0
    predicted[-10:, -8:] = 120.0

    score = elm_frame_morphology_fidelity(predicted, target, reference)

    assert score["response_correlation"] < 0.0
    assert score["morphology_fidelity"] == 0.0


def test_aligned_indices_realise_physical_horizon_without_duplicates():
    frame_time = np.arange(2000, dtype=np.float64) * 0.0005
    burst_time = 0.4

    idx = aligned_frame_indices(frame_time, burst_time)

    assert idx is not None
    assert np.all(np.diff(idx) > 0)
    actual_ms = (frame_time[idx[12]] - frame_time[idx[8]]) * 1e3
    assert actual_ms == pytest.approx(ELM_MORPHOLOGY_HORIZON_MS)
    assert frame_time[idx[12]] == pytest.approx(burst_time)


def test_aligned_indices_reject_camera_too_slow_for_distinct_samples():
    frame_time = np.arange(40, dtype=np.float64) * 0.01

    assert aligned_frame_indices(frame_time, 0.2) is None


def test_paired_summary_bootstraps_over_windows_and_reproduces_gap():
    baseline = [
        {
            "morphology_fidelity": 0.60 + i * 0.001,
            "edge_divertor_nll": 1.30 + i * 0.001,
            "edge_divertor_top1": 0.1,
        }
        for i in range(12)
    ]
    dynamics = [
        {
            "morphology_fidelity": 0.72 + i * 0.001,
            "edge_divertor_nll": 1.05 + i * 0.001,
            "edge_divertor_top1": 0.2,
        }
        for i in range(12)
    ]

    summary = summarise_paired_scores(baseline, dynamics)

    morph = summary["dynamics_minus_baseline_morphology"]
    nll = summary["baseline_minus_dynamics_nll"]
    assert morph["n_pairs"] == 12
    assert morph["mean"] == pytest.approx(0.12)
    assert morph["favours_dynamics"] is True
    assert nll["mean"] == pytest.approx(0.25)
    assert nll["favours_dynamics"] is True
    assert summary["reproduces_existing_arm_gap"] is True
    assert summary["edge_divertor_nll_favours_dynamics"] is True


def test_morphology_verdict_preserves_separate_nll_qualification():
    baseline = [
        {
            "morphology_fidelity": 0.1,
            "edge_divertor_nll": 1.0,
            "edge_divertor_top1": 0.1,
        }
        for _ in range(8)
    ]
    dynamics = [
        {
            "morphology_fidelity": 0.2,
            "edge_divertor_nll": 1.2,
            "edge_divertor_top1": 0.1,
        }
        for _ in range(8)
    ]

    summary = summarise_paired_scores(baseline, dynamics)

    assert summary["reproduces_existing_arm_gap"] is True
    assert summary["edge_divertor_nll_favours_dynamics"] is False


def test_paired_summary_refuses_unpaired_windows():
    row = {
        "morphology_fidelity": 0.5,
        "edge_divertor_nll": 1.0,
        "edge_divertor_top1": 0.0,
    }
    with pytest.raises(ValueError, match="paired"):
        summarise_paired_scores([row], [])
