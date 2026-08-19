from __future__ import annotations

import os
from math import tau
from pathlib import Path

import numpy as np
import pytest

from imas_ambix.challenge.convention import (
    DIIID_CONVENTION,
    DIIID_SOURCE_COCOS,
    MINIMUM_AUDIT_SHOTS,
    measure_diiid_convention,
)
from imas_ambix.cocos import CANONICAL_COCOS, identify_source_cocos


def _train_paths() -> list[Path]:
    root = Path(
        os.environ.get(
            "SOPHELIO_DIIID_TRAIN",
            "/work/projects/imas_gpu/sophelio/raw/data/diii_d_train",
        )
    )
    return sorted(root.glob("*.parquet"))


def test_measured_digits_and_transform_pin_cocos_five_to_seventeen() -> None:
    authority = DIIID_CONVENTION
    assert DIIID_SOURCE_COCOS == 5
    assert authority.source_cocos == identify_source_cocos(
        sigma_bp=1,
        e_bp=0,
        sigma_r_phi_z=1,
        sigma_rho_theta_phi=-1,
    )
    assert authority.source_digits == (1, 0, 1, -1)
    assert authority.target_cocos == CANONICAL_COCOS == 17
    assert authority.psi_to_canonical == -tau
    assert authority.total_flux_to_canonical == -1.0
    assert authority.ip_to_canonical == 1.0
    assert authority.toroidal_field_to_canonical == 1.0
    assert authority.q_to_canonical == -1.0
    assert authority.derivative_to_canonical == -1.0 / tau

    source_flux = np.array([-0.4, -0.1, 0.2])
    canonical_flux = authority.canonical_flux(source_flux)
    np.testing.assert_allclose(authority.source_flux(canonical_flux), source_flux)


def test_twenty_shot_receipt_identifies_every_measured_factor() -> None:
    paths = _train_paths()
    if len(paths) < MINIMUM_AUDIT_SHOTS:
        pytest.skip(
            f"real corpus has {len(paths)} of {MINIMUM_AUDIT_SHOTS} required shots"
        )

    receipt = measure_diiid_convention(paths, shots=MINIMUM_AUDIT_SHOTS)

    assert receipt.shots == MINIMUM_AUDIT_SHOTS
    assert receipt.per_radian_wins == receipt.shots
    assert receipt.psi_ip_positive == receipt.shots
    assert receipt.q_ip_bcoil_negative == receipt.shots
    assert receipt.delta_star_ip_positive == receipt.shots
    assert receipt.per_radian_ratio_median == pytest.approx(1.08030, abs=5.0e-4)
    assert receipt.total_flux_ratio_median == pytest.approx(0.171935, abs=5.0e-4)
    assert all(frame.bcoil < 0.0 for frame in receipt.frames)
