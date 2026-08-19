"""Frozen camera-representation diagnostic readout tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from imas_ambix.camdyn.physics_probes import (
    DIAGNOSTIC_TARGETS,
    align_nearest_native,
    deterministic_window_starts,
    paired_bootstrap_difference,
    score_frozen_target,
)


def test_native_alignment_rejects_distant_frames():
    signal_time = np.array([0.0, 0.01, 0.02])
    signal_value = np.array([1.0, 2.0, 3.0])
    frame_time = np.array([0.001, 0.011, 0.050])
    aligned, tolerance = align_nearest_native(signal_time, signal_value, frame_time)
    assert tolerance == pytest.approx(0.0075)
    assert aligned[:2].tolist() == [1.0, 2.0]
    assert np.isnan(aligned[2])


def test_windows_use_common_time_support_without_target_values():
    camera_time = np.linspace(0.0, 1.0, 101)
    target_times = [
        np.linspace(0.1, 0.9, 20),
        np.linspace(0.2, 0.8, 30),
        np.linspace(0.15, 0.85, 10),
        np.linspace(0.05, 0.95, 40),
    ]
    starts = deterministic_window_starts(camera_time, target_times, n_frames=16)
    assert starts == [27, 42, 58]


def test_paired_interval_is_oriented_to_lower_error():
    interval = paired_bootstrap_difference(
        np.linspace(-0.3, -0.1, 40), seed=3, n_boot=2000
    )
    assert interval["hi"] < 0
    assert interval["dynamics_better"]
    assert interval["n_shots"] == 40


def test_matched_probe_scores_a_more_informative_representation():
    rng = np.random.default_rng(4)
    train_n, held_n, dim = 320, 160, 12
    train_target = rng.normal(size=train_n)
    held_target = rng.normal(size=held_n)
    train_dynamics = rng.normal(scale=0.15, size=(train_n, dim))
    held_dynamics = rng.normal(scale=0.15, size=(held_n, dim))
    train_dynamics[:, 0] += train_target
    held_dynamics[:, 0] += held_target
    train_baseline = rng.normal(size=(train_n, dim))
    held_baseline = rng.normal(size=(held_n, dim))
    shots = np.repeat(np.arange(20), held_n // 20)

    result = score_frozen_target(
        train_baseline,
        train_dynamics,
        train_target,
        held_baseline,
        held_dynamics,
        held_target,
        shots,
        seed=5,
    )

    assert result["arms"]["dynamics"]["rmse"] < result["arms"]["baseline"]["rmse"]
    assert result["arms"]["dynamics"]["crps"] < result["arms"]["baseline"]["crps"]
    assert result["paired_dynamics_minus_baseline"]["rmse"]["dynamics_better"]
    assert result["paired_dynamics_minus_baseline"]["crps"]["dynamics_better"]


def test_committed_artifact_has_both_arms_and_four_diagnostics():
    path = Path("imas_ambix/camdyn/artifacts/physics_probes.json")
    if not path.exists():
        pytest.skip("measurement artifact is produced by the GPU readout")
    artifact = json.loads(path.read_text(encoding="utf-8"))
    assert set(artifact["targets"]) == {target.key for target in DIAGNOSTIC_TARGETS}
    for result in artifact["targets"].values():
        assert set(result["arms"]) == {"baseline", "dynamics"}
        assert set(result["paired_dynamics_minus_baseline"]) == {"rmse", "crps"}
        assert result["n_heldout_shots"] > 1
