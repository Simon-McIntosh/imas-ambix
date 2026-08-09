"""Focused checks for Nova propagation, conditioning, and artifacts."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from imas_ambix.statespace.nova_ensemble_estimator import (
    EstimatorConfig,
    EstimatorFailure,
    NovaEnsembleEstimator,
    merge_shards,
    write_result,
)


@pytest.fixture(scope="module")
def conditioned_products():
    config = EstimatorConfig(members=8, steps=270, fixed_lag_steps=60, seed=41)
    estimator = NovaEnsembleEstimator(config)
    ordinary = estimator.run()
    perturbation = np.zeros_like(ordinary.observations)
    perturbation[180:, 1] = 0.08
    changed_future = estimator.run(observation_perturbation=perturbation)
    return config, ordinary, changed_future


def test_physical_and_provenance_contract(conditioned_products):
    config, result, _ = conditioned_products
    assert result.provenance.nova_revision == (
        "fdbfd15b179ffbd562a2ac2b6e4961cc7442ab1e"
    )
    assert result.provenance.backend == "cpu"
    assert result.provenance.x64_enabled
    assert result.provenance.dtype == "float64"
    assert result.provenance.topology == "single-null"
    assert result.provenance.clock_hz == 1000
    assert result.causal_forecast.dtype == np.float64
    assert result.metrics["physics"]["finite_members"] == config.members
    assert result.metrics["physics"]["max_ledger_identity_error"] <= 1.0e-12
    assert result.metrics["physics"]["max_boundary_current_relative_error"] < 0.04
    assert result.metrics["physics"]["common_random_numbers"] is True
    assert result.metrics["physics"]["correction_frequency_hz"] == 100
    assert result.camera_proxy == {
        "product": "flux_surface_emissivity_proxy",
        "validated_checkpoint": False,
        "label": "proxy",
    }


def test_conditioning_reduces_innovation_and_proper_score(conditioned_products):
    _, result, _ = conditioned_products
    innovation = result.metrics["innovation_rmse"]
    score = result.metrics["proper_score"]
    assert innovation["analysis"] < innovation["forecast"]
    assert score["analysis"] < score["forecast"]
    assert set(result.metrics["horizons"]) == {
        "10_ms",
        "50_ms",
        "100_ms",
        "250_ms",
    }
    assert score["persistence"] > 0.0
    assert score["conventional_enkf"] > 0.0


def test_causal_products_ignore_future_observations_while_smoother_responds(
    conditioned_products,
):
    _, ordinary, changed_future = conditioned_products
    np.testing.assert_array_equal(
        ordinary.causal_forecast[:180], changed_future.causal_forecast[:180]
    )
    np.testing.assert_array_equal(
        ordinary.causal_analysis[:180], changed_future.causal_analysis[:180]
    )
    assert not np.array_equal(
        ordinary.full_sequence_smoothing[:180],
        changed_future.full_sequence_smoothing[:180],
    )


def test_out_of_distribution_uncertainty_widens_and_actuator_edit_moves_ensemble(
    conditioned_products,
):
    _, result, _ = conditioned_products
    uncertainty = result.metrics["uncertainty"]
    response = result.metrics["actuator_response"]
    assert (
        uncertainty["out_of_distribution_spread"]
        > (uncertainty["in_distribution_spread"])
    )
    assert uncertainty["widening_ratio"] > 1.0
    assert response["edited_displacement"] > response["same_plan_spread"]
    assert response["displacement_to_spread"] > 1.0


def test_runtime_merge_and_schema_failures_are_rejected(
    conditioned_products,
    tmp_path,
):
    config, result, _ = conditioned_products
    nonuniform = np.arange(config.steps, dtype=np.float64) * 0.001
    nonuniform[20] += 2.0e-5
    with pytest.raises(EstimatorFailure, match="uniformly sampled"):
        NovaEnsembleEstimator(config).run(clock=nonuniform)
    with pytest.raises(EstimatorFailure, match="requested backend"):
        NovaEnsembleEstimator(replace(config, backend="gpu")).run()

    write_result(
        result,
        tmp_path,
        name="shard-000",
        shard_index=0,
        shard_count=2,
        seed=config.seed,
    )
    with pytest.raises(EstimatorFailure, match="missing shard"):
        merge_shards(tmp_path, tmp_path / "incomplete")
    write_result(
        result,
        tmp_path,
        name="shard-001",
        shard_index=1,
        shard_count=2,
        seed=config.seed,
    )
    metadata_path = tmp_path / "shard-001.json"
    baseline = json.loads(metadata_path.read_text())
    corruptions = (
        (lambda row: row.update(seed=config.seed + 1), "seed"),
        (
            lambda row: row["provenance"].update(nova_revision="mismatch"),
            "revision",
        ),
        (lambda row: row["provenance"].update(backend="gpu"), "backend"),
        (lambda row: row["provenance"].update(dtype="float32"), "dtype"),
        (lambda row: row["shape"]["truth"].__setitem__(0, config.steps - 1), "shape"),
        (lambda row: row.update(failure=True), "failed shard"),
    )
    for corrupt, message in corruptions:
        metadata = deepcopy(baseline)
        corrupt(metadata)
        metadata_path.write_text(json.dumps(metadata))
        with pytest.raises(EstimatorFailure, match=message):
            merge_shards(tmp_path, tmp_path / "mismatch")
    metadata_path.write_text(json.dumps(baseline))

    duplicate_path = tmp_path / "shard-duplicate.json"
    duplicate = deepcopy(baseline)
    duplicate["shard_index"] = 0
    duplicate_path.write_text(json.dumps(duplicate))
    with pytest.raises(EstimatorFailure, match="duplicate shard"):
        merge_shards(tmp_path, tmp_path / "duplicate")
    duplicate_path.unlink()

    metadata_output, array_output = merge_shards(tmp_path, tmp_path / "complete")
    assert metadata_output.is_file()
    with np.load(array_output) as merged:
        assert merged["causal_analysis"].shape[1] == 2 * config.members
