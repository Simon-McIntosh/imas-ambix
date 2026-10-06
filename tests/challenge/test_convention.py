from __future__ import annotations

import os
from math import tau
from pathlib import Path

import numpy as np
import pytest

from imas_alambic.cocos import CANONICAL_COCOS, identify_source_cocos
from imas_ambix.challenge.convention import (
    DIIID_CONVENTION,
    DIIID_SOURCE_COCOS,
    MINIMUM_AUDIT_SHOTS,
    ConventionEligibility,
    measure_convention,
    measure_diiid_convention,
)
from imas_ambix.challenge.loader import EfitLabels


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
    assert receipt.psi_ip_negative == receipt.shots
    assert receipt.q_ip_bcoil_positive == receipt.shots
    assert receipt.delta_star_ip_positive == receipt.shots
    assert receipt.per_radian_ratio_median == pytest.approx(1.08030, abs=5.0e-4)
    assert receipt.total_flux_ratio_median == pytest.approx(0.171935, abs=5.0e-4)
    assert all(frame.bcoil < 0.0 for frame in receipt.frames)


def _synthetic_labels(*, flux_sign: float = 1.0, frames: int = 3) -> EfitLabels:
    """Build canonical EfitLabels whose flux sense follows ``flux_sign``."""

    radius = np.linspace(1.0, 2.0, 65)
    height = np.linspace(-0.5, 0.5, 65)
    grid_z, grid_r = np.meshgrid(height, radius, indexing="ij")
    flux = flux_sign * ((grid_r - 1.5) ** 2 + grid_z**2)
    angle = np.linspace(0.0, tau, 32, endpoint=False)
    return EfitLabels(
        time_ms=np.arange(frames, dtype=float),
        psirz=np.tile(flux, (frames, 1, 1)),
        grid_r_m=radius,
        grid_z_m=height,
        lcfs_r_m=np.tile(1.5 + 0.4 * np.cos(angle), (frames, 1)),
        lcfs_z_m=np.tile(0.4 * np.sin(angle), (frames, 1)),
        scalars={
            "efit_r_axis": np.full(frames, 1.5),
            "efit_z_axis": np.zeros(frames),
            "efit_lcfs_n": np.full(frames, 32),
            "efit_q95": np.full(frames, 3.0),
            "magnetics_plasma_current": np.full(frames, 1200.0),
            "magnetics_bcoil": np.full(frames, -50.0),
        },
        cocos=CANONICAL_COCOS,
    )


def _synthetic_eligibility() -> ConventionEligibility:
    return ConventionEligibility(
        minimum_plasma_current_ka=500.0,
        standard_field_bcoil_maximum=0.0,
        minimum_audit_shots=3,
    )


def test_measure_convention_reads_synthetic_efit_labels() -> None:
    receipt = measure_convention(
        (_synthetic_labels() for _ in range(3)), _synthetic_eligibility(), shots=3
    )

    assert receipt.shots == 3
    assert receipt.psi_ip_positive == receipt.shots
    assert receipt.q_ip_bcoil_negative == receipt.shots


def test_measure_convention_tracks_the_synthetic_flux_sense() -> None:
    receipt = measure_convention(
        (_synthetic_labels(flux_sign=-1.0) for _ in range(3)),
        _synthetic_eligibility(),
        shots=3,
    )

    assert receipt.psi_ip_negative == receipt.shots
    assert receipt.psi_ip_positive == 0
