"""Parity and public-seam checks for the three benchmark storage arms."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest
import zarr
from imas.backends.netcdf.ids_tensorizer import IDSTensorizer

from imas_ambix.bench.store_arms import (
    FIXED_CHANNELS,
    FIXED_MAST_SHOT,
    FIXED_SAMPLE_START,
    FIXED_SAMPLE_STOP,
    IDSZarrWriter,
    payload_to_ids,
    read_dd_zarr,
    read_fixed_native_payload,
    read_imas_netcdf,
    write_imas_netcdf,
)

LEVEL2_ROOT = Path("/work/projects/imas_gpu/mast/level2/shots")


@pytest.fixture(scope="module")
def fixed_payload():
    """Read the real fixed shot when the MAST mirror is mounted."""
    shot_path = LEVEL2_ROOT / f"{FIXED_MAST_SHOT}.zarr"
    if not shot_path.exists():
        pytest.skip("FAIR-MAST level-2 mirror not mounted")
    return read_fixed_native_payload(LEVEL2_ROOT)


def test_fixed_payload_is_one_unchanged_finite_shot_slice(fixed_payload):
    assert fixed_payload.shot == 11766
    assert fixed_payload.sample_start == 0
    assert fixed_payload.sample_stop == 64
    assert fixed_payload.sample_count == 64
    assert tuple(fixed_payload.arrays) == tuple(
        binding.dd_path for binding in FIXED_CHANNELS
    )
    for payload_array in fixed_payload.arrays.values():
        assert payload_array.values.shape == (FIXED_SAMPLE_STOP - FIXED_SAMPLE_START,)
        assert payload_array.values.dtype == np.float64
        assert np.isfinite(payload_array.values).all()


def test_all_three_arms_are_array_equal_and_record_public_writers(
    fixed_payload, tmp_path
):
    ids = payload_to_ids(fixed_payload)
    paths = tuple(fixed_payload.arrays)

    netcdf_path = tmp_path / "fixed-summary.nc"
    netcdf_receipt = write_imas_netcdf(ids, netcdf_path)
    netcdf_arrays = read_imas_netcdf(netcdf_path, paths)

    zarr_path = tmp_path / "fixed-summary.zarr"
    zarr_writer = IDSZarrWriter(ids, paths)
    zarr_receipt = zarr_writer.write(zarr_path)
    zarr_arrays = read_dd_zarr(zarr_path, paths)

    for path, native in fixed_payload.arrays.items():
        assert np.array_equal(native.values, netcdf_arrays[path]), path
        assert np.array_equal(native.values, zarr_arrays[path]), path

    assert netcdf_receipt.entrypoint == "imas.DBEntry.put"
    assert netcdf_receipt.called_methods == ("put",)
    assert netcdf_receipt.private_names == ()
    assert zarr_receipt.base_class == (
        "imas.backends.netcdf.ids_tensorizer.IDSTensorizer"
    )
    assert zarr_receipt.called_methods == (
        "include_coordinate_paths",
        "collect_filled_data",
        "determine_data_shapes",
        "get_dimensions",
        "tensorize",
        "get_attributes",
    )
    assert zarr_receipt.conditional_methods == (
        "get_shape_dimensions",
        "get_shape_attributes",
    )
    assert zarr_receipt.private_names == ()
    assert IDSZarrWriter.__mro__[1] is IDSTensorizer


def test_every_zarr_array_carries_dd_path_and_units(fixed_payload, tmp_path):
    ids = payload_to_ids(fixed_payload)
    zarr_path = tmp_path / "annotated-summary.zarr"
    IDSZarrWriter(ids, tuple(fixed_payload.arrays)).write(zarr_path)

    group = zarr.open_group(zarr_path, mode="r")
    assert set(group.array_keys()) == {
        path.replace("/", ".") for path in fixed_payload.arrays
    }
    for path, source in fixed_payload.arrays.items():
        stored = group[path.replace("/", ".")]
        assert stored.attrs["dd_path"] == source.dd_path
        assert source.dd_path == f"summary/{path}"
        assert stored.attrs["units"] == source.units


def test_zarr_subclass_uses_no_single_underscore_attributes():
    """Single-underscore attributes would make the sibling seam private."""
    tree = ast.parse(inspect.getsource(IDSZarrWriter))
    reached_private_names = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr.startswith("_")
        and not node.attr.startswith("__")
    }
    assert reached_private_names == set()
