from __future__ import annotations

import os
from math import tau
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from imas_ambix.challenge.convention import DIIID_CONVENTION
from imas_ambix.challenge.facts import build_report
from imas_ambix.challenge.loader import load_shot, validate_shot_schema
from imas_ambix.cocos import CANONICAL_COCOS


def _real_slice() -> list[Path]:
    root = Path(
        os.environ.get(
            "SOPHELIO_DIIID_TRAIN",
            "/work/projects/imas_gpu/sophelio/raw/data/diii_d_train",
        )
    )
    return sorted(root.glob("*.parquet"))[:100]


def test_real_diii_d_schema_on_one_hundred_shots() -> None:
    paths = _real_slice()
    if len(paths) < 100:
        pytest.skip(f"real corpus slice has {len(paths)} of 100 required shots")
    validated = 0
    for path in paths:
        shot = validate_shot_schema(path)
        assert shot.source == "DIII-D"
        assert shot.labels.psirz.shape[1:] == (65, 65)
        assert shot.labels.cocos == CANONICAL_COCOS
        validated += 1
    assert validated == 100


def test_facts_report_covers_every_circulated_claim() -> None:
    paths = _real_slice()
    if len(paths) < 100:
        pytest.skip(f"real corpus slice has {len(paths)} of 100 required shots")
    report = build_report(paths)
    claims = report["claims"]
    assert len(claims) == 5
    assert {claim["verdict"] for claim in claims} <= {
        "confirmed",
        "corrected",
        "unreachable-from-slice",
    }
    assert report["measurements"]["shots"] == 100


def test_loader_serves_diii_d_labels_in_the_canonical_convention() -> None:
    paths = _real_slice()
    if not paths:
        pytest.skip("real DIII-D corpus is unavailable")
    path = paths[0]
    raw = pq.read_table(
        path,
        columns=[
            "efit_psirz",
            "efit_q95",
            "magnetics_plasma_current",
            "magnetics_bcoil",
        ],
    )
    shot = load_shot(path)

    raw_flux = np.asarray(raw["efit_psirz"][0].as_py(), dtype=float)
    raw_q95 = np.asarray(raw["efit_q95"][0].as_py(), dtype=float)
    raw_ip = np.asarray(raw["magnetics_plasma_current"][0].as_py(), dtype=float)
    raw_bcoil = np.asarray(raw["magnetics_bcoil"][0].as_py(), dtype=float)

    np.testing.assert_allclose(shot.labels.psirz, -tau * raw_flux)
    np.testing.assert_allclose(shot.labels.scalars["efit_q95"], -raw_q95)
    np.testing.assert_allclose(
        shot.actuators["magnetics_plasma_current"].values,
        DIIID_CONVENTION.canonical_plasma_current(raw_ip),
    )
    np.testing.assert_allclose(
        shot.actuators["magnetics_bcoil"].values,
        DIIID_CONVENTION.canonical_toroidal_field(raw_bcoil),
    )
