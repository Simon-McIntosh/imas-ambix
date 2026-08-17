from __future__ import annotations

import numpy as np
import pytest

from imas_ambix.thomson import (
    ChannelValidityPolicy,
    ElmPhase,
    IsofluxPairer,
    IsothermAsymmetryOperator,
    PedestalCalibration,
    PedestalFootDetector,
    TopologyClass,
    ValidityReason,
)


def _synthetic_calibration() -> PedestalCalibration:
    rng = np.random.default_rng(417)
    psi_n = np.linspace(0.45, 1.25, 600)
    feature = -3.2 * (psi_n - np.quantile(psi_n, 0.1))
    feature += rng.normal(0.0, 0.035, psi_n.size)
    topology = np.full(psi_n.shape, TopologyClass.CORE_VERTICAL.value)
    return PedestalCalibration.fit(feature, psi_n, topology)


def test_pedestal_detector_recovers_synthetic_separatrix() -> None:
    calibration = _synthetic_calibration()
    coordinate = np.linspace(-0.22, 0.22, 80)
    psi_n = np.linspace(0.45, 1.25, coordinate.size)
    known_separatrix = float(np.interp(1.0, psi_n, coordinate))
    temperature = 1800.0 * np.exp(-3.2 * psi_n)

    estimate = PedestalFootDetector(calibration).locate(
        coordinate,
        temperature,
        topology=TopologyClass.CORE_VERTICAL,
    )

    assert estimate.bracketed
    assert estimate.psi_n == 1.0
    assert estimate.coordinate_m == pytest.approx(known_separatrix, abs=0.012)
    assert estimate.coordinate_sigma_m > 0.0


def test_validity_policy_widens_sigma_without_dropping_channels() -> None:
    policy = ChannelValidityPolicy()
    quiet_core = policy.assess(
        800.0,
        3.0e19,
        topology=TopologyClass.CORE_VERTICAL,
    )
    elm_edge = policy.assess(
        800.0,
        3.0e19,
        topology=TopologyClass.TANGENTIAL_EDGE,
        elm_phase=ElmPhase.CRASH,
    )
    invalid_divertor = policy.assess(
        np.nan,
        3.0e19,
        topology=TopologyClass.DIVERTOR,
    )

    assert quiet_core.reason is ValidityReason.VALID
    assert elm_edge.reason is ValidityReason.ELM_AFFECTED
    assert invalid_divertor.reason is ValidityReason.NONFINITE
    assert quiet_core.retained and elm_edge.retained and invalid_divertor.retained
    assert quiet_core.sigma_multiplier == 1.0
    assert elm_edge.sigma_multiplier == 5.0
    assert invalid_divertor.sigma_multiplier == 36.0


def test_isoflux_pairs_recover_known_psi_n_on_non_collinear_chords() -> None:
    first_psi_n = np.array([0.62, 0.78, 0.94, 1.08])
    second_psi_n = np.array([0.63, 0.79, 0.95, 1.09])
    first_temperature = 2200.0 * np.exp(-2.7 * first_psi_n)
    second_temperature = 2200.0 * np.exp(-2.7 * second_psi_n)

    pairs = IsofluxPairer().pair(
        first_temperature,
        second_temperature,
        first_psi_n,
        second_psi_n,
        first_direction_rz=(0.0, 1.0),
        second_direction_rz=(1.0, -0.08),
    )

    assert len(pairs) == 4
    assert [pair.first_index for pair in pairs] == [0, 1, 2, 3]
    assert [pair.psi_n for pair in pairs] == pytest.approx(
        [0.625, 0.785, 0.945, 1.085], abs=1.0e-12
    )
    assert all(pair.geometry_determinant > 0.99 for pair in pairs)


def test_isoflux_pairer_rejects_collinear_chords() -> None:
    with pytest.raises(ValueError, match="non-collinear"):
        IsofluxPairer().pair(
            np.array([100.0]),
            np.array([100.0]),
            np.array([0.9]),
            np.array([0.9]),
            first_direction_rz=(1.0, 0.0),
            second_direction_rz=(2.0, 0.0),
        )


def test_training_free_isotherm_asymmetry_round_trip() -> None:
    operator = IsothermAsymmetryOperator()
    expected = 1.17
    radii = operator.synthesize_radii(
        expected,
        reference_major_radius_m=1.68,
        minor_radius_m=0.58,
        isotherm_half_width_m=0.31,
    )

    measured = operator.measure(
        *radii,
        reference_major_radius_m=1.68,
        minor_radius_m=0.58,
    )

    assert measured == pytest.approx(expected, rel=1.0e-12)
    assert not hasattr(operator, "fit")
