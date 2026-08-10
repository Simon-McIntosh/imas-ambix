"""Canonical DDv4/COCOS-17 invariants at the Ambix data boundary."""

from __future__ import annotations

import numpy as np
import pytest

from imas_ambix.cocos import (
    CANONICAL_COCOS,
    MAST_SOURCE_COCOS,
    ConventionContractError,
    canonical_factor,
    mast_angle_to_canonical,
    project_poloidal_field,
    require_canonical_contract,
)


def test_only_an_exact_ddv4_pin_and_cocos_seventeen_are_canonical():
    require_canonical_contract("4.1.1", CANONICAL_COCOS)

    with pytest.raises(ConventionContractError, match="DDv4 only"):
        require_canonical_contract("3.40.0", CANONICAL_COCOS)
    with pytest.raises(ConventionContractError, match="major.minor.patch"):
        require_canonical_contract("4", CANONICAL_COCOS)
    with pytest.raises(ConventionContractError, match="canonical COCOS 17"):
        require_canonical_contract("4.1.1", 11)


def test_mast_scalar_factors_land_on_cocos_seventeen():
    psi_factor = canonical_factor("psi_like", source_cocos=MAST_SOURCE_COCOS)
    derivative_factor = canonical_factor("dodpsi_like", source_cocos=MAST_SOURCE_COCOS)
    assert psi_factor == pytest.approx(2.0 * np.pi)
    assert derivative_factor == pytest.approx(1.0 / (2.0 * np.pi))
    assert canonical_factor("q_like", source_cocos=MAST_SOURCE_COCOS) == -1.0
    assert canonical_factor("ip_like", source_cocos=MAST_SOURCE_COCOS) == 1.0


def test_mast_probe_axis_is_converted_without_changing_its_directed_field():
    source_angles = np.array([0.0, 90.0, 45.0])
    canonical_angles = mast_angle_to_canonical(source_angles)
    assert canonical_angles.tolist() == [0.0, -90.0, -45.0]

    br = np.array([3.0, 3.0, 3.0])
    bz = np.array([2.0, 2.0, 2.0])
    source_projection = br * np.cos(np.deg2rad(source_angles)) + bz * np.sin(
        np.deg2rad(source_angles)
    )
    canonical_projection = project_poloidal_field(br, bz, canonical_angles)
    assert canonical_projection == pytest.approx(source_projection)


@pytest.mark.parametrize(
    ("ip", "b0", "psi_axis", "psi_boundary", "q"),
    [
        (8.113e5, -0.406, 0.0815, -0.0285, 4.513),
        (-8.113e5, 0.406, -0.0815, 0.0285, 4.513),
    ],
)
def test_both_mast_field_polarities_satisfy_the_canonical_sign_identities(
    ip: float,
    b0: float,
    psi_axis: float,
    psi_boundary: float,
    q: float,
):
    psi_scale = canonical_factor("psi_like", source_cocos=MAST_SOURCE_COCOS)
    q_scale = canonical_factor("q_like", source_cocos=MAST_SOURCE_COCOS)
    canonical_axis = psi_axis * psi_scale
    canonical_boundary = psi_boundary * psi_scale
    canonical_q = q * q_scale

    sigma_bp = int(np.sign(canonical_boundary - canonical_axis) * np.sign(ip))
    sigma_rho_theta_phi = int(np.sign(canonical_q) * np.sign(ip) * np.sign(b0))
    assert sigma_bp == -1
    assert sigma_rho_theta_phi == 1
