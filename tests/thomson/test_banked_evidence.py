from __future__ import annotations

from math import tau
from pathlib import Path

import numpy as np
import pytest

from imas_ambix.challenge.convention import DIIID_CONVENTION
from imas_ambix.challenge.loader import load_shot
from imas_ambix.thomson import (
    IsothermAsymmetryOperator,
    PedestalCalibration,
    TopologyClass,
    banked_equilibrium_moments,
    collect_pedestal_samples,
)
from imas_ambix.thomson.bank import _bilinear_sample, _sample_psi_n

_ROOT = Path("/work/projects/imas_gpu/sophelio/prototype-subset/proto-extract")
_SOURCE = _ROOT / "source"
_BANK = _ROOT / "bank"


def _paired_paths() -> list[tuple[Path, Path]]:
    pairs = []
    for bank_path in sorted(_BANK.glob("*_flux_trajectory.npz")):
        stem = bank_path.name.removesuffix("_flux_trajectory.npz")
        parquet_path = _SOURCE / f"{stem}.parquet"
        if parquet_path.exists():
            pairs.append((parquet_path, bank_path))
    return pairs


@pytest.mark.skipif(not _BANK.exists(), reason="banked extraction corpus unavailable")
def test_canonical_flux_migration_preserves_sampled_psi_n() -> None:
    parquet_path, bank_path = _paired_paths()[0]
    shot = load_shot(parquet_path)
    with np.load(bank_path, allow_pickle=False) as loaded:
        bank = {name: loaded[name] for name in loaded.files}

    radius = np.linspace(shot.labels.grid_r_m[8], shot.labels.grid_r_m[-9], 9)
    height = np.linspace(shot.labels.grid_z_m[8], shot.labels.grid_z_m[-9], 9)
    sample_r, sample_z = np.meshgrid(radius, height)
    sample_r = sample_r.ravel()
    sample_z = sample_z.ravel()

    errors = []
    frame_count = min(len(shot.labels.time_ms), len(bank["axis_flux_wb"]))
    for frame in (0, frame_count // 2, frame_count - 1):
        source_map = DIIID_CONVENTION.source_flux(shot.labels.psirz[frame])
        source_samples = _bilinear_sample(
            shot.labels.grid_r_m,
            shot.labels.grid_z_m,
            source_map,
            sample_r,
            sample_z,
        )
        source_axis = float(bank["axis_flux_wb"][frame]) / tau
        source_boundary = float(bank["boundary_flux_wb"][frame]) / tau
        expected = (source_samples - source_axis) / (source_boundary - source_axis)
        canonical = _sample_psi_n(
            shot,
            bank,
            frame,
            sample_r,
            sample_z,
        )
        errors.append(float(np.nanmax(np.abs(canonical - expected))))

    assert max(errors) <= 2.0e-15


@pytest.mark.skipif(not _BANK.exists(), reason="banked extraction corpus unavailable")
def test_detector_uncertainty_heldout_calibration_band() -> None:
    pairs = _paired_paths()
    assert len(pairs) >= 8
    split = max(6, int(0.75 * len(pairs)))
    train_parts = [collect_pedestal_samples(*pair) for pair in pairs[:split]]
    heldout_parts = [collect_pedestal_samples(*pair) for pair in pairs[split:]]
    train_feature = np.concatenate([part[0] for part in train_parts])
    train_psi_n = np.concatenate([part[1] for part in train_parts])
    train_topology = np.concatenate([part[2] for part in train_parts])
    heldout_feature = np.concatenate([part[0] for part in heldout_parts])
    heldout_psi_n = np.concatenate([part[1] for part in heldout_parts])
    heldout_topology = np.concatenate([part[2] for part in heldout_parts])

    calibration = PedestalCalibration.fit(
        train_feature,
        train_psi_n,
        train_topology,
    )
    coverage, sample_count = calibration.coverage(
        heldout_feature,
        heldout_psi_n,
        heldout_topology,
    )

    assert sample_count >= 500
    assert 0.50 <= coverage <= 0.82
    payload = calibration.to_dict()
    assert payload[TopologyClass.CORE_VERTICAL.value]["sample_count"] >= 500
    assert payload[TopologyClass.TANGENTIAL_EDGE.value]["sample_count"] >= 100


@pytest.mark.skipif(not _BANK.exists(), reason="banked extraction corpus unavailable")
def test_shafranov_operator_matches_banked_beta_p_plus_li_half() -> None:
    pairs = _paired_paths()
    moments = []
    for parquet_path, bank_path in pairs[:4]:
        moments.extend(banked_equilibrium_moments(parquet_path, bank_path, stride=40))
    assert len(moments) >= 12
    operator = IsothermAsymmetryOperator()
    recovered = []
    expected = []
    for moment in moments:
        radii = operator.synthesize_radii(
            moment.beta_p_plus_li_half,
            reference_major_radius_m=1.68,
            minor_radius_m=0.58,
            isotherm_half_width_m=0.28,
        )
        recovered.append(
            operator.measure(
                *radii,
                reference_major_radius_m=1.68,
                minor_radius_m=0.58,
            )
        )
        expected.append(moment.beta_p_plus_li_half)
    assert np.asarray(recovered) == pytest.approx(expected, rel=1.0e-12)
    assert np.all(np.asarray([moment.beta_p for moment in moments]) >= 0.0)
