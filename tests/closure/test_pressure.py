from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from imas_ambix.closure import (
    CollisionalityReceipt,
    PressureClosureSample,
    dimensionless_collisionality_proxy,
    fit_pressure_closure,
)

_ARTIFACT = (
    Path(__file__).parents[2]
    / "imas_ambix"
    / "closure"
    / "artifacts"
    / "pressure_closure_calibration.json"
)


def _sample(
    shot: str, collisionality: float, multiplier: float
) -> PressureClosureSample:
    receipt = CollisionalityReceipt(
        value=collisionality,
        density_m3=3.0e19,
        temperature_ev=800.0,
        major_radius_m=1.7,
        effective_minor_radius_m=0.55,
        inverse_aspect_ratio=0.55 / 1.7,
        coulomb_logarithm=17.0,
    )
    return PressureClosureSample(
        shot=shot,
        time_ms=1000.0,
        total_pressure_pa=multiplier * 1000.0,
        electron_pressure_pa=1000.0,
        raw_multiplier=multiplier,
        collisionality=receipt,
        actuators={},
    )


def test_collisionality_proxy_tracks_density_temperature_and_geometry() -> None:
    reference = dimensionless_collisionality_proxy(3.0e19, 800.0, 1.7, 0.55)
    denser = dimensionless_collisionality_proxy(6.0e19, 800.0, 1.7, 0.55)
    hotter = dimensionless_collisionality_proxy(3.0e19, 1600.0, 1.7, 0.55)
    larger = dimensionless_collisionality_proxy(3.0e19, 800.0, 2.0, 0.55)

    assert reference.value > 0.0
    assert denser.value > reference.value
    assert hotter.value < reference.value
    assert larger.value > reference.value


def test_fitted_multiplier_is_physically_floored() -> None:
    collisionalities = np.geomspace(0.02, 2.0, 30)
    samples = []
    for index, collisionality in enumerate(collisionalities):
        multiplier = np.exp(0.3 + 0.2 * np.log(collisionality / 0.2))
        samples.append(_sample(f"shot-{index % 5}", collisionality, multiplier))
    calibration = fit_pressure_closure(samples, bootstrap_draws=200)

    predictions = calibration.multiplier(np.geomspace(1.0e-5, 1.0e3, 2000))
    assert np.all(predictions >= 1.0)
    assert calibration.collisionality_slope == pytest.approx(0.2, abs=0.03)


def test_banked_artifact_records_complete_physical_calibration() -> None:
    payload = json.loads(_ARTIFACT.read_text())
    corpus = payload["corpus"]
    fit = payload["fit"]

    assert corpus["banked_shots"] == 20
    assert corpus["banked_frames"] == 4141
    assert fit["shot_count"] == 20
    assert fit["sample_count"] == corpus["eligible_frames"]
    assert np.isfinite(fit["collisionality_slope"])
    assert len(fit["collisionality_slope_confidence_interval"]) == 2
    assert all(np.isfinite(fit["collisionality_slope_confidence_interval"]))
    assert np.isfinite(fit["residual_scatter_multiplier"])
    assert fit["fitted_multiplier_summary"]["minimum"] >= 1.0
    assert payload["per_shot"]
    assert all(
        receipt["fitted_multiplier"]["minimum"] >= 1.0
        for receipt in payload["per_shot"]
    )
    heating = payload["auxiliary_heating_dependence"]
    assert set(heating["channels"]) == set(corpus["auxiliary_heating_channels"])
    assert len(corpus["actuator_channels_present_in_every_shot"]) >= 1
