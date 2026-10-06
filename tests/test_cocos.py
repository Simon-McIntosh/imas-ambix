"""Canonical DDv4/COCOS-17 invariants at the Ambix data boundary."""

from __future__ import annotations

from math import tau

import numpy as np
import pytest

from imas_ambix.cocos import (
    CANONICAL_COCOS,
    MAST_SOURCE_COCOS,
    ConventionContractError,
    canonical_factor,
    identify_source_cocos,
    project_poloidal_field,
    require_canonical_contract,
    source_cocos_digits,
)
from imas_ambix.data.cocos_convention import (
    MAST_SOURCE_COCOS as DECLARED_MAST_SOURCE_COCOS,
)

SOURCE_CONVENTION_DECLARATION = (
    "COCOS 3 is an external owner assumption pending a facility statement of "
    "positive-phi direction, not a measurement"
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
    assert MAST_SOURCE_COCOS == DECLARED_MAST_SOURCE_COCOS == 3, (
        SOURCE_CONVENTION_DECLARATION
    )
    psi_factor = canonical_factor("psi_like", source_cocos=MAST_SOURCE_COCOS)
    derivative_factor = canonical_factor("dodpsi_like", source_cocos=MAST_SOURCE_COCOS)
    assert psi_factor == pytest.approx(tau), SOURCE_CONVENTION_DECLARATION
    assert derivative_factor == pytest.approx(1.0 / tau), SOURCE_CONVENTION_DECLARATION
    assert canonical_factor("q_like", source_cocos=MAST_SOURCE_COCOS) == -1.0, (
        SOURCE_CONVENTION_DECLARATION
    )
    assert canonical_factor("ip_like", source_cocos=MAST_SOURCE_COCOS) == 1.0, (
        SOURCE_CONVENTION_DECLARATION
    )


def test_measured_digits_identify_diii_d_source_cocos() -> None:
    source = identify_source_cocos(
        sigma_bp=1,
        e_bp=0,
        sigma_r_phi_z=1,
        sigma_rho_theta_phi=-1,
    )
    assert source == 5
    assert source_cocos_digits(source) == (1, 0, 1, -1)
    assert canonical_factor("psi_like", source_cocos=source) == -tau
    assert canonical_factor("ip_like", source_cocos=source) == 1.0
    assert canonical_factor("b0_like", source_cocos=source) == 1.0
    assert canonical_factor("q_like", source_cocos=source) == -1.0
    assert canonical_factor("dodpsi_like", source_cocos=source) == -1.0 / tau


def test_ddv4_poloidal_angle_projects_along_its_directed_axis():
    br = np.array([3.0, 3.0, 3.0])
    bz = np.array([2.0, 2.0, 2.0])
    angles = np.array([0.0, -90.0, 90.0])

    projection = project_poloidal_field(br, bz, angles)

    assert projection == pytest.approx([3.0, 2.0, -2.0])


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
    ip_scale = canonical_factor("ip_like", source_cocos=MAST_SOURCE_COCOS)
    q_scale = canonical_factor("q_like", source_cocos=MAST_SOURCE_COCOS)
    canonical_axis = psi_axis * psi_scale
    canonical_boundary = psi_boundary * psi_scale
    canonical_ip = ip * ip_scale
    canonical_q = q * q_scale

    sigma_bp = int(np.sign(canonical_boundary - canonical_axis) * np.sign(canonical_ip))
    sigma_rho_theta_phi = int(
        np.sign(canonical_q) * np.sign(canonical_ip) * np.sign(b0)
    )
    assert sigma_bp == -1
    assert sigma_rho_theta_phi == 1
