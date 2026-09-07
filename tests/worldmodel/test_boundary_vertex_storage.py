from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from imas_ambix.worldmodel.boundary_vertex_storage import (
    compare_boundary_storage,
    read_boundary_storage,
)


def _write_session(path: Path, boundary: np.ndarray) -> None:
    values = np.asarray(boundary, dtype=np.float64)
    r_values = values[:, 0][None, :, None]
    z_values = values[:, 1][None, :, None]
    session = xr.Dataset(
        data_vars={
            "flux_surface_r": (
                ("n_surface", "n_theta", "time"),
                r_values,
            ),
            "flux_surface_z": (
                ("n_surface", "n_theta", "time"),
                z_values,
            ),
            "flux_surface_psi_norm": (("n_surface",), np.asarray([1.0])),
        },
        coords={"time": np.asarray([0.02])},
    )
    session.to_netcdf(path, group="steering", engine="h5netcdf")


def test_trailing_nonfinite_row_remains_in_stored_length(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.nc"
    repaired_path = tmp_path / "repaired.nc"
    _write_session(baseline_path, np.asarray([[1.0, 0.0], [1.1, 0.1]]))
    _write_session(
        repaired_path,
        np.asarray([[1.0, 0.0], [1.1, 0.1], [np.nan, np.nan]]),
    )

    result = compare_boundary_storage(baseline_path, repaired_path)
    repaired = result["repaired"]["boundary_array"]

    assert repaired["full_shape"] == [1, 3, 2]
    assert repaired["stored_vertex_count"] == 3
    assert repaired["slices"][0]["last_row"] == ["NaN", "NaN"]
    assert repaired["slices"][0]["trailing_nonfinite_row_count"] == 1
    assert repaired["slices"][0]["final_row_equals_first_row"] is False
    assert result["comparison"]["category"] == "stored_length_differs"


def test_absent_named_group_raises(tmp_path: Path) -> None:
    path = tmp_path / "root-only.nc"
    xr.Dataset({"value": (("row",), np.asarray([1.0]))}).to_netcdf(
        path, engine="h5netcdf"
    )

    with pytest.raises(ValueError, match="no readable NetCDF group /steering"):
        read_boundary_storage(path)
