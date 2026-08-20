"""Focused checks for Nova propagation, conditioning, and artifacts."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from imas_ambix.statespace.nova_ensemble_estimator import (
    NOVA_REVISION,
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
    assert result.provenance.nova_revision == NOVA_REVISION
    assert result.provenance.backend == "cpu"
    assert result.provenance.x64_enabled
    assert result.provenance.dtype == "float64"
    assert result.provenance.topology == "circular-nested-flux-surfaces"
    assert result.provenance.clock_hz == 1000
    assert result.truth.shape == (config.steps, 4)
    assert result.observations.shape == (config.steps, 4)
    assert result.causal_forecast.dtype == np.float64
    assert result.metrics["physics"]["finite_members"] == config.members
    assert result.metrics["physics"]["max_ledger_identity_error"] <= 1.0e-12
    assert result.metrics["physics"]["max_boundary_current_relative_error"] < 0.04
    assert result.metrics["physics"]["common_random_numbers"] is True
    assert result.metrics["physics"]["correction_frequency_hz"] == 100
    expected_flux_shape = (config.steps, config.members, 21)
    assert result.equilibrium_flux.shape == expected_flux_shape
    assert result.edited_equilibrium_flux.shape == expected_flux_shape
    assert result.equilibrium_flux.dtype == np.float64
    assert result.edited_equilibrium_flux.dtype == np.float64
    assert np.isfinite(result.equilibrium_flux).all()
    assert np.isfinite(result.edited_equilibrium_flux).all()
    np.testing.assert_array_equal(
        result.equilibrium_flux[0], result.edited_equilibrium_flux[0]
    )
    assert (
        np.mean(
            result.edited_equilibrium_flux[-1, :, -1]
            - result.equilibrium_flux[-1, :, -1]
        )
        > 0.0
    )
    assert result.camera_proxy == {
        "product": "flux_surface_emissivity_proxy",
        "validated_checkpoint": False,
        "label": "proxy",
    }


def test_conditioning_reduces_innovation_and_proper_score(conditioned_products):
    config, result, _ = conditioned_products
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
    assert np.isfinite(score["conventional_enkf"])
    comparator = result.metrics["comparators"]["conventional_enkf"]
    assert comparator["identity"] == "random_walk_ensemble_kalman_filter"
    assert comparator["transition"] == "persistence_plus_gaussian_process_noise"
    assert comparator["observation_operator"] == "identity"
    assert comparator["cohort"] == "same synthetic cohort"
    assert comparator["ensemble_shape"] == [config.steps, config.members, 4]


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
    np.testing.assert_array_equal(
        ordinary.equilibrium_flux,
        changed_future.equilibrium_flux,
    )
    np.testing.assert_array_equal(
        ordinary.edited_equilibrium_flux,
        changed_future.edited_equilibrium_flux,
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

    shard_members = config.members // 2
    first_shard = NovaEnsembleEstimator(
        replace(config, members=shard_members, member_offset=0)
    ).run()
    second_shard = NovaEnsembleEstimator(
        replace(config, members=shard_members, member_offset=shard_members)
    ).run()
    np.testing.assert_array_equal(first_shard.truth, second_shard.truth)
    np.testing.assert_array_equal(first_shard.observations, second_shard.observations)
    write_result(
        first_shard,
        tmp_path,
        name="shard-000",
        shard_index=0,
        shard_count=2,
        seed=config.seed,
    )
    with pytest.raises(EstimatorFailure, match="missing shard"):
        merge_shards(tmp_path, tmp_path / "incomplete")
    write_result(
        second_shard,
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
        (
            lambda row: row["estimator_config"].update(observation_noise=0.5),
            "configuration",
        ),
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
    merged_metadata = json.loads(metadata_output.read_text())
    merged_metrics = merged_metadata["metrics"]
    assert merged_metadata["shape"]["causal_analysis"] == [
        config.steps,
        config.members,
        4,
    ]
    assert merged_metadata["shape"]["equilibrium_flux"] == [
        config.steps,
        config.members,
        21,
    ]
    assert merged_metadata["shape"]["edited_equilibrium_flux"] == [
        config.steps,
        config.members,
        21,
    ]
    equilibrium_metadata = merged_metadata["equilibrium_products"]
    assert equilibrium_metadata["equilibrium_flux"]["axes"] == [
        "clock",
        "member",
        "radial_face",
    ]
    assert (
        "not a two-dimensional Grad-Shafranov map"
        in equilibrium_metadata["equilibrium_flux"]["description"]
    )
    assert merged_metadata["estimator_config"]["observation_noise"] == (
        config.observation_noise
    )
    assert merged_metrics["physics"]["finite_members"] == config.members
    assert merged_metrics["physics"]["aggregation_rule"] == (
        "maximum physical error across shards"
    )
    for key in (
        "innovation_rmse",
        "proper_score",
        "horizons",
        "uncertainty",
        "actuator_response",
    ):
        actual = merged_metrics[key]
        expected = result.metrics[key]
        if key == "horizons":
            for horizon in actual:
                for measure in actual[horizon]:
                    np.testing.assert_allclose(
                        actual[horizon][measure],
                        expected[horizon][measure],
                        rtol=0.0,
                        atol=1.0e-15,
                    )
        else:
            for measure in actual:
                np.testing.assert_allclose(
                    actual[measure],
                    expected[measure],
                    rtol=0.0,
                    atol=1.0e-15,
                )
    comparator = merged_metrics["comparators"]["conventional_enkf"]
    assert comparator["identity"] == "random_walk_ensemble_kalman_filter"
    assert comparator["ensemble_shape"] == [config.steps, config.members, 4]
    runtime = merged_metrics["runtime"]
    assert len(runtime["per_shard"]) == 2
    assert runtime["parallel_wall_time_s"] is None
    assert runtime["aggregate_member_steps_per_s"] is None
    assert "were not measured" in runtime["aggregation_rule"]
    assert runtime["serial_elapsed_sum_s"] == sum(
        row["elapsed_s"] for row in runtime["per_shard"]
    )
    assert runtime["member_steps_total"] == config.members * config.steps
    with np.load(array_output) as merged:
        for key in ("clock", "truth", "observations"):
            np.testing.assert_array_equal(merged[key], getattr(result, key))
        for key in (
            "causal_forecast",
            "causal_analysis",
            "fixed_lag_smoothing",
            "full_sequence_smoothing",
            "edited_actuator",
            "nominal_actuator",
            "equilibrium_flux",
            "edited_equilibrium_flux",
        ):
            np.testing.assert_array_equal(merged[key], getattr(result, key))
