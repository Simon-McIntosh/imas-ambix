"""Frozen camera-latent MSE pitch probe tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from imas_ambix.camdyn.pitch_probe import (
    RADIAL_NODES,
    ShotLatents,
    build_shot_prediction,
    fit_radial_probe,
    nearest_unique_indices,
    pitch_at_nodes,
)


def test_nearest_alignment_is_bounded_and_deduplicated():
    native = np.array([0.000, 0.001, 0.002, 0.003])
    query = np.array([0.0002, 0.0003, 0.0014, 0.0100])
    query_index, native_index, tolerance = nearest_unique_indices(native, query)
    assert tolerance == pytest.approx(0.00075)
    assert query_index.tolist() == [0, 2]
    assert native_index.tolist() == [0, 1]


def test_pitch_nodes_respect_physical_gate_and_sightline_support():
    radii = np.array([0.70, 0.80, 0.90, 1.00])
    nodes = np.array([0.65, 0.75, 0.85, 0.95, 1.05])
    pitch = np.array([[0.10, 0.20, 0.30, 0.40]])
    error = np.array([[0.01, 0.01, 0.01, 0.01]])
    result = pitch_at_nodes(pitch, error, radii, nodes)
    assert np.isnan(result[0, 0])
    assert result[0, 1:4].tolist() == pytest.approx([0.15, 0.25, 0.35])
    assert np.isnan(result[0, 4])

    error[0, 1:] = 1.0
    assert np.isnan(pitch_at_nodes(pitch, error, radii, nodes)).all()


def test_node_ridge_recovers_an_informative_frozen_representation():
    rng = np.random.default_rng(8)
    features = rng.normal(size=(200, 12))
    targets = np.stack(
        [
            features[:, 0]
            + 0.1 * node
            + rng.normal(scale=0.02, size=200)
            for node in range(RADIAL_NODES)
        ],
        axis=1,
    )
    targets[:20, 0] = np.nan
    probe = fit_radial_probe(features, targets)
    prediction = probe.predict(features)
    assert np.sqrt(np.nanmean((prediction - targets) ** 2)) < 0.05
    assert np.isfinite(probe.sigma).all()


def test_prediction_maps_only_selected_slices_to_native_sightlines():
    rng = np.random.default_rng(3)
    features = rng.normal(size=(32, 6))
    targets = rng.normal(scale=0.2, size=(32, RADIAL_NODES))
    probe = fit_radial_probe(features, targets)
    selected = ShotLatents(
        shot_id=7,
        baseline=features[:2],
        dynamics=features[:2],
        slice_indices=np.array([1, 3]),
    )
    entry = {
        "beam_on_slice_times": [0.0, 0.001, 0.002, 0.003, 0.004],
        "active_channel_ids": [1, 2, 3],
        "active_channel_rpos": [0.78, 0.90, 1.02],
    }
    nodes = np.linspace(0.75, 1.05, RADIAL_NODES)
    prediction = build_shot_prediction(selected, features[:2], probe, nodes, entry)
    assert prediction.pitch_mean.shape == (5, 3)
    assert np.isfinite(prediction.pitch_mean[[1, 3]]).all()
    assert np.isnan(prediction.pitch_mean[[0, 2, 4]]).all()
    assert np.isfinite(prediction.pitch_std[[1, 3]]).all()


def test_committed_artifact_uses_the_locked_cohort_and_both_arms():
    path = Path("imas_ambix/camdyn/artifacts/pitch_probe.json")
    if not path.exists():
        pytest.skip("measurement artifact is produced by the GPU probe")
    artifact = json.loads(path.read_text(encoding="utf-8"))
    cohort = artifact["heldout_cohort"]
    assert cohort["locked_shots"] == 112
    assert cohort["tokenless_exclusions"] == [22426, 24404, 24666, 25016]
    assert cohort["expected_with_tokens"] == 108
    assert cohort["usable_shots"] > 1
    assert set(artifact["arms"]) == {"baseline", "dynamics"}
    assert (
        artifact["protocol"]["eval_harness"]
        == "imas_ambix.statespace.mse_eval.score"
    )
    assert (
        artifact["paired_dynamics_minus_baseline"]["rmse"]["n_shots"]
        == cohort["usable_shots"]
    )
