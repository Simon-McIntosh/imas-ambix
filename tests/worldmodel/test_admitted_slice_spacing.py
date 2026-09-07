from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from imas_ambix.worldmodel.admitted_slice_spacing import (
    score_session,
    summarize_admitted_times,
)
from imas_ambix.worldmodel.flux_label_dataset import (
    EXPECTED_CARRIER_IDENTITY,
    EXPECTED_POLICY_DIGEST,
)


def _write_session(
    root: Path,
    shot_id: int,
    session_times: list[float],
    rows: list[dict[str, object]],
) -> None:
    xr.Dataset(coords={"time": np.asarray(session_times, dtype=np.float64)}).to_netcdf(
        root / f"{shot_id}.nc", group="steering", engine="h5netcdf"
    )
    manifest = {
        "status": "complete",
        "shot": shot_id,
        "policy_digest": EXPECTED_POLICY_DIGEST,
        "carrier_identity": EXPECTED_CARRIER_IDENTITY,
        "slices": rows,
    }
    (root / f"{shot_id}.manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def test_synthetic_session_recovers_planted_non_uniform_gaps(tmp_path: Path) -> None:
    shot_id = 22086
    rows = [
        {"row": 0, "written": True, "converged": True},
        {"row": 1, "written": True, "converged": True},
        {"row": 2, "written": True, "converged": False},
        {"row": 3, "written": False, "converged": False},
        {"row": 4, "written": True, "converged": True},
        {"row": 5, "written": True, "converged": True},
    ]
    _write_session(tmp_path, shot_id, [0.0, 0.005, 0.010, 0.015, 0.030], rows)

    result = score_session(shot_id, session_root=tmp_path)

    assert result["admitted_slice_count"] == 4
    assert result["written_slice_count"] == 5
    assert result["written_but_unconverged_count"] == 1
    assert result["gaps_s"] == pytest.approx([0.005, 0.010, 0.015])
    assert result["non_nominal_gap_count"] == 2
    assert result["non_nominal_gap_fraction"] == pytest.approx(2.0 / 3.0)
    assert result["longest_uniform_gap_run"] == 1
    assert result["gap_distribution"]["histogram"]["bins"] == [
        {"gap_s": 0.005, "count": 1},
        {"gap_s": 0.01, "count": 1},
        {"gap_s": 0.015, "count": 1},
    ]


def test_uniform_session_is_reported_as_uniform(tmp_path: Path) -> None:
    shot_id = 21978
    rows = [{"row": index, "written": True, "converged": True} for index in range(4)]
    _write_session(tmp_path, shot_id, [0.0, 0.005, 0.010, 0.015], rows)

    result = score_session(shot_id, session_root=tmp_path)

    assert result["all_gaps_nominal"] is True
    assert result["non_nominal_gap_count"] == 0
    assert result["non_nominal_gap_fraction"] == 0.0
    assert result["longest_uniform_gap_run"] == 3
    assert result["gap_distribution"]["minimum_s"] == pytest.approx(0.005)
    assert result["gap_distribution"]["median_s"] == pytest.approx(0.005)
    assert result["gap_distribution"]["maximum_s"] == pytest.approx(0.005)


def test_spacing_tolerance_is_one_part_in_a_thousand() -> None:
    result = summarize_admitted_times(
        np.asarray([0.0, 0.005005, 0.0100101], dtype=np.float64)
    )

    assert result["absolute_tolerance_s"] == pytest.approx(0.000005)
    assert result["non_nominal_gap_count"] == 1
