"""Tests for directly timed storage-arm benchmark measurements."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest
import zarr

from imas_ambix.bench.store_arms import (
    FIXED_CHANNELS,
    FIXED_MAST_SHOT,
    IDSZarrWriter,
    payload_to_ids,
    read_dd_zarr,
    read_fixed_native_payload,
    read_imas_netcdf,
    write_imas_netcdf,
)
from imas_ambix.bench.store_read_rates import (
    ARMS,
    PATTERNS,
    BenchmarkConfig,
    measure_access_patterns,
    measure_transport,
)

LEVEL2_ROOT = Path("/work/projects/imas_gpu/mast/level2/shots")


def test_access_matrix_has_twelve_directly_timed_cells():
    config = BenchmarkConfig(
        shots=(10, 11, 12, 13),
        batch_size=4,
        slice_size=4,
        static_repetitions=2,
        slice_repetitions=2,
        batch_repetitions=2,
        random_windows=4,
    )

    def reader(shot: int, start: int | None, stop: int | None):
        values = np.arange(12, dtype=np.float64) + shot
        return tuple(values[slice(start, stop)] for _ in FIXED_CHANNELS)

    readers = {arm: reader for arm in ARMS}
    cells = measure_access_patterns(
        config, readers, {shot: 12 for shot in config.shots}
    )

    assert len(cells) == 12
    assert {(cell.arm, cell.access_pattern) for cell in cells} == {
        (arm, pattern) for arm in ARMS for pattern in PATTERNS
    }
    assert all(cell.wall_seconds > 0 for cell in cells)
    assert all(cell.sample_count > 0 for cell in cells)
    assert all(cell.throughput_samples_per_second > 0 for cell in cells)


def test_transport_times_identical_live_and_mapped_objects(tmp_path):
    local_array = tmp_path / "local.zarr"
    remote_array = tmp_path / "remote.zarr"
    values = np.arange(128, dtype=np.float64)

    zarr.save_array(local_array, values)
    zarr.save_array(remote_array, values)
    cells, evidence = measure_transport(
        local_array,
        str(remote_array),
        repetitions=3,
    )

    assert [cell.route for cell in cells] == [
        "live_https_transfer",
        "mapped_on_disk",
    ]
    assert all(cell.sample_count == 384 for cell in cells)
    assert all(cell.wall_seconds > 0 for cell in cells)
    assert evidence["decoded_arrays_identical"] is True
    assert evidence["remote_shape"] == [128]
    assert evidence["mapped_shape"] == [128]


@pytest.mark.skipif(
    not (LEVEL2_ROOT / f"{FIXED_MAST_SHOT}.zarr").exists(),
    reason="FAIR-MAST level-2 mirror not mounted",
)
def test_project_environment_writes_and_reads_both_converted_arms(tmp_path):
    payload = read_fixed_native_payload(LEVEL2_ROOT)
    ids = payload_to_ids(payload)
    paths = tuple(payload.arrays)

    netcdf_path = tmp_path / "summary.nc"
    write_imas_netcdf(ids, netcdf_path)
    netcdf_arrays = read_imas_netcdf(netcdf_path, paths)

    zarr_path = tmp_path / "summary.zarr"
    IDSZarrWriter(ids, paths).write(zarr_path)
    zarr_arrays = read_dd_zarr(zarr_path, paths)

    for path, native in payload.arrays.items():
        assert np.array_equal(native.values, netcdf_arrays[path])
        assert np.array_equal(native.values, zarr_arrays[path])


def test_tensorizer_subclass_reaches_no_private_imas_names():
    tree = ast.parse(inspect.getsource(IDSZarrWriter))
    private_attributes = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr.startswith("_")
        and not node.attr.startswith("__")
    }
    assert private_attributes == set()
    assert IDSZarrWriter.__mro__[1].__name__ == "IDSTensorizer"
    assert {
        "get_dimensions",
        "get_attributes",
        "get_shape_dimensions",
        "get_shape_attributes",
    }.issubset(IDSZarrWriter.__dict__)
