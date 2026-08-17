from __future__ import annotations

import os
from pathlib import Path

import pytest

from imas_ambix.challenge.facts import build_report
from imas_ambix.challenge.loader import validate_shot_schema


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
