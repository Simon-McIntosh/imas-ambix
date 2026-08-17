from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from imas_ambix.thomson import (
    IsothermAsymmetryOperator,
    PedestalCalibration,
    TopologyClass,
    banked_equilibrium_moments,
    collect_pedestal_samples,
)

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
