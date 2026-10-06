"""JT-60SA packaged machine-map catalogue over the converted per-phase store.

The catalogue at ``imas_ambix/data/machine_maps/jt-60sa.json`` declares the
converted SELENE-deck description of ``JT60SA_DESCRIPTION_DIR`` as two
range-scoped maps (OP1, OP2).  These tests check the declaration, that both
phases emit and adapt through one machine-name-free code path against the real
store, and that the phase boundary changes only the vessel.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from imas_ambix.data.description_identity import machine_description_bytes
from imas_ambix.data.description_reader import read_geometry_table
from imas_ambix.data.geometry_adapter import geometry_table_from_description
from imas_ambix.data.machine_map import (
    load_machine_map,
    load_packaged_machine_map,
    map_for_shot,
)
from imas_ambix.data.paths import JT60SA_DESCRIPTION_DIR
from imas_ambix.data.transform_engine import (
    BindingTransformError,
    transform_machine_description,
)

STORE_AVAILABLE = JT60SA_DESCRIPTION_DIR.is_dir()
requires_store = pytest.mark.skipif(
    not STORE_AVAILABLE,
    reason="the JT-60SA converted description store is not mounted",
)

PHASE_SHOTS = {"OP1": 101173, "OP2": 101174}
COIL_ELEMENT_COUNTS = {
    "CS1": 40, "CS2": 40, "CS3": 40, "CS4": 40,
    "EF1": 16, "EF2": 16, "EF3": 16, "EF4": 16, "EF5": 16, "EF6": 16,
    "FPPC_UP": 6, "FPPC_DOWN": 6,
}
COIL_GEOMETRY_PREFIX = "pf_active/coil/element/geometry/"
COIL_NAME_PATH = "pf_active/coil/element/name"


def _bindings(catalog, binding_set):
    return {
        binding.name: binding
        for binding in catalog.binding_sets[binding_set]
    }


def _emit(shot):
    catalog = load_packaged_machine_map("jt-60sa")
    description = transform_machine_description(
        catalog, shot, "netcdf", JT60SA_DESCRIPTION_DIR
    )
    assert description.status == "emitted", description.detail
    assert not description.missing_bindings
    return catalog, description


def _ids_payload(description):
    """Stable per-IDS content keyed by DD path and value bytes.

    Phase maps select different binding sets, so a whole-description digest
    differs at the boundary for a reason that is not geometry.  Comparing the
    emitted DD content per IDS isolates the physical change.
    """
    payload: dict[str, list[tuple[str, str, str, tuple[int, ...], str]]] = {}
    for array in description.arrays:
        ids_name = array.dd_path.split("/", 1)[0]
        values = np.asarray(array.values)
        payload.setdefault(ids_name, []).append(
            (
                array.dd_path,
                array.source_group,
                values.dtype.str,
                tuple(values.shape),
                np.ascontiguousarray(values).tobytes().hex(),
            )
        )
    return {ids_name: sorted(rows) for ids_name, rows in payload.items()}


def test_catalogue_declares_the_two_phase_store_and_maps():
    catalog = load_packaged_machine_map("jt-60sa")

    assert catalog.description_store_format == "netcdf"
    assert catalog.description_store_root == "JT60SA_DESCRIPTION_DIR"
    assert catalog.description_store_layout == "static-over-map"
    assert catalog.probe_angle_source == "description"

    assert [map_.name for map_ in catalog.maps] == ["OP1", "OP2"]
    op1, op2 = catalog.maps
    assert (op1.first_shot, op1.last_shot) == (100001, 101173)
    assert (op2.first_shot, op2.last_shot) == (101174, 999999)

    assert map_for_shot(catalog, 101173) is op1
    assert map_for_shot(catalog, 101174) is op2
    for uncovered in (60033, 1000000):
        with pytest.raises(LookupError):
            map_for_shot(catalog, uncovered)


def test_pf_active_coils_declared_one_family_and_assembly_per_coil():
    catalog = load_packaged_machine_map("jt-60sa")

    for phase, binding_set in (("OP1", "jt60sa-description-op1"),
                               ("OP2", "jt60sa-description-op2")):
        bindings = _bindings(catalog, binding_set)
        for coil, count in COIL_ELEMENT_COUNTS.items():
            stem = f"jt60sa-{phase.lower()}-pf-active-{coil.lower().replace('_', '-')}"
            geometry = [
                bindings[f"{stem}-{role}"]
                for role in ("r", "z", "width", "height")
            ]
            name_binding = bindings[f"{stem}-coordinate-element"]
            # Every coil-element binding selects this coil's struct-array entry
            # by the coil name the store carries, and counts only the leaf and
            # the struct levels below the selected coil.
            for binding in (*geometry, name_binding):
                assert binding.struct_array_entry == coil
                assert binding.source_rank == 1
            assert name_binding.dd_path == COIL_NAME_PATH
            for binding in geometry:
                assert binding.dd_path.startswith(COIL_GEOMETRY_PREFIX)

            member_names = {binding.name for binding in geometry}
            matches = [
                assembly
                for assembly in catalog.structure_assemblies
                if member_names.issubset(assembly.member_bindings)
            ]
            assert len(matches) == 1, f"{coil} must resolve to one assembly"
            assembly = matches[0]
            assert assembly.name_binding == name_binding.name
            assert len(assembly.element_identifiers) == count


def test_ragged_struct_arrays_declared_per_entry():
    catalog = load_packaged_machine_map("jt-60sa")

    op1 = _bindings(catalog, "jt60sa-description-op1")
    op2 = _bindings(catalog, "jt60sa-description-op2")
    op1_vessel = {
        binding.struct_array_entry
        for binding in op1.values()
        if binding.dd_path.startswith("pf_passive/loop/element/")
    }
    op2_vessel = {
        binding.struct_array_entry
        for binding in op2.values()
        if binding.dd_path.startswith("pf_passive/loop/element/")
    }
    # The OP1 vessel is two ragged loops; the OP2 vessel is one.
    assert op1_vessel == {"VV1", "VV2"}
    assert op2_vessel == {"VV1"}

    # A single loop assembly per vessel loop, as for a coil.
    vessels = [
        assembly
        for assembly in catalog.structure_assemblies
        if assembly.structure_path == "pf_passive/loop/element/geometry/rectangle"
    ]
    assert len(vessels) == 3


@requires_store
@pytest.mark.parametrize("phase,shot", sorted(PHASE_SHOTS.items()))
def test_each_phase_emits_and_adapts_through_one_code_path(phase, shot):
    catalog, description = _emit(shot)
    table = read_geometry_table(shot, machine="jt-60sa")
    direct = geometry_table_from_description(description, catalog)

    assert table.signature == direct.signature
    assert len(table.active_circuits) == 12
    assert len(table.b_probes) == 17
    assert len(table.flux_loops) == 27
    assert len(table.amc_current_channels) == 12
    expected_vessel = 120 if phase == "OP1" else 98
    assert len(table.passive_structures) == expected_vessel


@requires_store
def test_both_phases_share_the_pf_and_magnetics_geometry():
    _, op1 = _emit(PHASE_SHOTS["OP1"])
    _, op2 = _emit(PHASE_SHOTS["OP2"])
    t1 = geometry_table_from_description(op1, load_packaged_machine_map("jt-60sa"))
    t2 = geometry_table_from_description(op2, load_packaged_machine_map("jt-60sa"))

    # One code path: the phase-independent PF coils, probes and flux loops are
    # identical between the tables; only the vessel and wall differ.
    assert t1.pf_filaments == t2.pf_filaments
    assert t1.active_circuits == t2.active_circuits == list(range(1, 13))
    # ``angle_deg`` is NaN (absent from the description), so compare the finite
    # probe coordinates rather than probe identity.
    assert [(p.r, p.z, p.length) for p in t1.b_probes] == [
        (p.r, p.z, p.length) for p in t2.b_probes
    ]
    assert t1.flux_loops == t2.flux_loops
    assert t1.passive_structures != t2.passive_structures
    assert t1.limiter_r != t2.limiter_r


@requires_store
def test_each_phase_carries_the_twelve_pf_coils_with_their_element_counts():
    for shot in PHASE_SHOTS.values():
        _, description = _emit(shot)
        coil_arrays = [
            array
            for array in description.arrays
            if array.dd_path == "pf_active/coil/element/geometry/rectangle/r"
        ]
        counts = sorted(int(np.asarray(array.values).size) for array in coil_arrays)
        assert counts == sorted(COIL_ELEMENT_COUNTS.values())
        assert sum(counts) == 268


@requires_store
def test_phase_identity_is_deterministic_and_partitions_the_range():
    for shot in PHASE_SHOTS.values():
        _, first = _emit(shot)
        _, second = _emit(shot)
        assert machine_description_bytes(first) == machine_description_bytes(second)

    _, op1_a = _emit(100001)
    _, op1_b = _emit(101173)
    assert machine_description_bytes(op1_a) == machine_description_bytes(op1_b)

    _, op2_a = _emit(101174)
    _, op2_b = _emit(150000)
    assert machine_description_bytes(op2_a) == machine_description_bytes(op2_b)


@requires_store
def test_phase_boundary_changes_only_the_vessel_and_wall():
    _, before = _emit(101173)
    _, after = _emit(101174)
    before_payload = _ids_payload(before)
    after_payload = _ids_payload(after)

    assert before_payload != after_payload
    changed_ids = sorted(
        ids_name
        for ids_name in set(before_payload) | set(after_payload)
        if before_payload.get(ids_name) != after_payload.get(ids_name)
    )
    unchanged_ids = sorted(
        (set(before_payload) & set(after_payload)) - set(changed_ids)
    )

    # The vessel and the wall move at the boundary; pf_active and magnetics do
    # not, so the boundary is a description change and not a re-addressing.
    assert "pf_passive" in changed_ids
    assert "pf_active" in unchanged_ids
    assert "magnetics" in unchanged_ids

    print(
        "JT60SA_PHASE_BOUNDARY "
        f"boundary=101174 changed_ids={','.join(changed_ids)} "
        f"unchanged_ids={','.join(unchanged_ids)}"
    )


def test_dropping_the_coil_selector_refuses_the_ragged_read(tmp_path: Path):
    """Negative control: without the coil selector the ragged read must fail."""
    source = (
        Path(__file__).resolve().parents[2]
        / "imas_ambix"
        / "data"
        / "machine_maps"
        / "jt-60sa.json"
    )
    payload = json.loads(source.read_text())
    stripped = 0
    for binding_set in payload["binding_sets"]:
        for binding in binding_set["bindings"]:
            if (
                "pf-active" in binding["name"]
                and binding["dd_path"].startswith("pf_active/coil/element/")
                and "struct_array_entry" in binding
            ):
                del binding["struct_array_entry"]
                stripped += 1
    assert stripped == 120
    mutated = tmp_path / "jt-60sa.json"
    mutated.write_text(json.dumps(payload))
    catalog = load_machine_map(mutated)

    with pytest.raises(BindingTransformError, match="ragged"):
        transform_machine_description(catalog, 101173, "netcdf", JT60SA_DESCRIPTION_DIR)
