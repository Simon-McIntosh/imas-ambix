"""Bank a held-out DIII-D pedestal calibration and equilibrium receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .bank import banked_equilibrium_moments, collect_pedestal_samples
from .calibration import PedestalCalibration


def build_calibration_artifact(root: Path) -> dict[str, object]:
    source = root / "source"
    bank = root / "bank"
    pairs = []
    for bank_path in sorted(bank.glob("*_flux_trajectory.npz")):
        stem = bank_path.name.removesuffix("_flux_trajectory.npz")
        parquet_path = source / f"{stem}.parquet"
        if parquet_path.exists():
            pairs.append((parquet_path, bank_path))
    if len(pairs) < 8:
        raise RuntimeError(
            f"at least eight paired shots are required, found {len(pairs)}"
        )
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
    coverage, coverage_samples = calibration.coverage(
        heldout_feature,
        heldout_psi_n,
        heldout_topology,
    )
    moments = []
    for parquet_path, bank_path in pairs:
        moments.extend(banked_equilibrium_moments(parquet_path, bank_path, stride=40))
    moment_values = np.asarray(
        [moment.beta_p_plus_li_half for moment in moments], dtype=float
    )
    return {
        "corpus": {
            "paired_shots": len(pairs),
            "training_shots": split,
            "heldout_shots": len(pairs) - split,
            "training_samples": int(train_feature.size),
            "heldout_samples": int(heldout_feature.size),
        },
        "uncertainty_receipt": {
            "nominal_interval": "one_sigma",
            "preregistered_coverage_band": [0.50, 0.82],
            "heldout_coverage": coverage,
            "heldout_samples": coverage_samples,
            "passes": bool(0.50 <= coverage <= 0.82),
        },
        "calibration_curves": calibration.to_dict(),
        "banked_equilibrium_receipt": {
            "sample_count": len(moments),
            "beta_p_plus_li_half_min": float(np.min(moment_values)),
            "beta_p_plus_li_half_median": float(np.median(moment_values)),
            "beta_p_plus_li_half_max": float(np.max(moment_values)),
            "operator": "analytic_isotherm_centre_shift",
            "training_samples": 0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    artifact = build_calibration_artifact(arguments.root)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps(artifact["uncertainty_receipt"], indent=2))
    print(json.dumps(artifact["banked_equilibrium_receipt"], indent=2))


if __name__ == "__main__":
    main()
