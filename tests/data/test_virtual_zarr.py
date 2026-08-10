"""Lazy, read-only canonical views over unchanged physical arrays."""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from imas_ambix.data.signal_map import MAP_SCHEMA_VERSION, SignalMap, SignalRule
from imas_ambix.data.virtual_zarr import VirtualZarrView


def _map() -> SignalMap:
    signal = SignalRule(
        semantic_id="poloidal_flux",
        source_group="efm",
        source_array="psi",
        source_unit="Wb/rad",
        target_path="equilibrium/time_slice/profiles_2d/psi",
        target_unit="Wb",
        target_index=None,
        transformation="psi_like",
        source_cocos=3,
        unit_factor=1.0,
        channel_factor=1.0,
        standard_name=None,
        evidence="measured COCOS receipt",
    )
    return SignalMap.create(
        schema_version=MAP_SCHEMA_VERSION,
        set_version="0.1.0",
        machine="mast",
        system="equilibrium",
        source_dataset="fair-mast-level1",
        target_dd_version="4.1.1",
        target_cocos=17,
        discovery_producer="imas-codex",
        discovery_receipt="sha256:discovery",
        signals=(signal,),
    )


class _CountingArray:
    shape = (100,)
    chunks = (10,)
    dtype = np.dtype("float32")

    def __init__(self):
        self.selections = []
        self.values = np.arange(100, dtype=np.float32)

    def __getitem__(self, selection):
        self.selections.append(selection)
        return self.values[selection]


def test_metadata_and_binding_do_not_read_a_chunk():
    physical = _CountingArray()
    view = VirtualZarrView({"efm": {"psi": physical}}, _map().compile(30420))

    assert view.keys() == ("poloidal_flux",)
    array = view["poloidal_flux"]
    assert array.shape == (100,)
    assert array.chunks == (10,)
    assert array.attrs["cocos"] == 17
    assert physical.selections == []


def test_an_explicit_slice_reads_only_that_slice_and_applies_cocos():
    physical = _CountingArray()
    view = VirtualZarrView({"efm": {"psi": physical}}, _map().compile(30420))

    converted = view["poloidal_flux"][10:13]
    assert physical.selections == [slice(10, 13)]
    assert converted == pytest.approx(np.arange(10, 13) * 2.0 * np.pi)


def test_implicit_materialisation_and_writes_fail_loudly():
    physical = _CountingArray()
    view = VirtualZarrView({"efm": {"psi": physical}}, _map().compile(30420))
    array = view["poloidal_flux"]

    with pytest.raises(TypeError, match="explicit slice"):
        np.asarray(array)
    with pytest.raises(TypeError, match="read-only"):
        array[:] = 0.0
    with pytest.raises(TypeError, match="read-only"):
        view["new"] = array
    assert physical.selections == []


def test_physical_zarr_bytes_remain_unchanged(tmp_path):
    path = tmp_path / "shot.zarr"
    root = zarr.open_group(path, mode="w")
    source = root.create_group("efm")
    raw = np.arange(12, dtype=np.float32)
    source.create_array("psi", data=raw, chunks=(4,))

    view = VirtualZarrView.open(str(path), _map(), shot=30420)
    assert view["poloidal_flux"][4:8] == pytest.approx(raw[4:8] * 2.0 * np.pi)

    reopened = zarr.open_group(path, mode="r")
    assert np.asarray(reopened["efm"]["psi"][:]) == pytest.approx(raw)
