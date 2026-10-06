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

from imas_alambic.machine_map import (
    load_machine_map,
    load_packaged_machine_map,
    map_for_shot,
)
from imas_alambic.transform_engine import (
    BindingTransformError,
    transform_machine_description,
)
from imas_ambix.data.description_identity import machine_description_bytes
from imas_ambix.data.description_reader import read_geometry_table
from imas_ambix.data.geometry_adapter import geometry_table_from_description
from imas_ambix.data.paths import JT60SA_DESCRIPTION_DIR

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
TURNS_PATH = "pf_active/coil/element/turns_with_sign"
# The ampere-turns per ampere each drive circuit carries, summed over its
# connections.  These are the SELENE deck's coil turn totals: 40 filaments of
# 13.725 turns per CS coil (549), the six EF windings at 142, 154, 247, 353,
# 152 and 180, and 23 for each FPPC coil.
DECLARED_AMPERE_TURNS = {
    "cs1": 549.0, "cs2": 549.0, "cs3": 549.0, "cs4": 549.0,
    "ef1": 142.0, "ef2": 154.0, "ef3": 247.0,
    "ef4": 353.0, "ef5": 152.0, "ef6": 180.0,
    "fppc-up": 23.0, "fppc-down": 23.0,
}


def _circuit_token(identifier: str) -> str:
    return identifier.split("circuit-", 1)[1]


def _declared_ampere_turns(topology) -> dict[str, float]:
    totals: dict[str, float] = {}
    for connection in topology.connections:
        token = _circuit_token(connection.circuit_identifier)
        totals[token] = totals.get(token, 0.0) + (
            connection.turns * connection.current_weight * connection.direction
        )
    return totals


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


def _binding_payload(description):
    """Per-binding content keyed by a phase-neutral binding name.

    Phase maps select different binding sets whose names carry the phase
    token (``jt60sa-op1-…`` / ``jt60sa-op2-…``).  Stripping that token lets a
    binding be matched across the boundary, so the vessel, cryostat, PF coil
    and magnetics content can each be compared on its own.
    """
    payload: dict[str, tuple[str, str, tuple[int, ...], str]] = {}
    for array in description.arrays:
        key = (
            array.binding_name.replace("jt60sa-op1-", "jt60sa-")
            .replace("jt60sa-op2-", "jt60sa-")
        )
        values = np.asarray(array.values)
        payload[key] = (
            array.dd_path,
            values.dtype.str,
            tuple(values.shape),
            np.ascontiguousarray(values).tobytes().hex(),
        )
    return payload


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


def test_drive_topologies_declare_the_deck_ampere_turns_per_ampere():
    """Both phase drive topologies state each circuit's deck turn total.

    Every connection carries its element's ``turns_with_sign`` from
    ``pf_active``, so the ampere-turns per ampere summed over a circuit's
    connections is the SELENE deck's coil turn total — 549 for each CS coil, the
    six EF windings and 23 for each FPPC coil — rather than the circuit's
    filament count.
    """
    catalog = load_packaged_machine_map("jt-60sa")

    assert [topology.name for topology in catalog.drive_topologies] == [
        "jt60sa-pf-drive-op1",
        "jt60sa-pf-drive-op2",
    ]
    for topology in catalog.drive_topologies:
        assert topology.turns_path == TURNS_PATH
        totals = _declared_ampere_turns(topology)
        assert totals == pytest.approx(DECLARED_AMPERE_TURNS)


@requires_store
def test_drive_connection_turns_equal_the_stored_turns_with_sign():
    """Each declared turn value is the store's own ``turns_with_sign``.

    The provenance the topology states (``turns_path``) is checked, not
    trusted: every connection's ``turns`` matches the ``turns_with_sign`` of the
    pf_active element its geometry identifier names, in both phase stores.
    """
    import imas

    catalog = load_packaged_machine_map("jt-60sa")
    for phase, topology in zip(("OP1", "OP2"), catalog.drive_topologies, strict=True):
        with imas.DBEntry(
            JT60SA_DESCRIPTION_DIR / phase / "pf_active.nc",
            "r",
            dd_version=catalog.dd_version,
        ) as entry:
            pf_active = entry.get("pf_active", autoconvert=False)
        stored = {
            str(element.name): float(element.turns_with_sign)
            for coil in pf_active.coil
            for element in coil.element
        }
        for connection in topology.connections:
            element_name = connection.geometry_element_identifier.rsplit("/", 1)[-1]
            assert connection.turns == pytest.approx(
                stored[element_name], rel=1e-12
            )


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


def test_no_element_identifier_is_a_trailing_slash_placeholder():
    """Every structure-assembly element identifier names a real element.

    A placeholder identifier ends in ``/`` so its last path segment is empty,
    matching the empty element name the converter used to write.  Once the
    converter names each element, the identifier must carry that name, so no
    identifier in the catalogue may end in a slash.
    """
    source = (
        Path(__file__).resolve().parents[2]
        / "imas_ambix"
        / "data"
        / "machine_maps"
        / "jt-60sa.json"
    )
    payload = json.loads(source.read_text())
    offenders = [
        identifier
        for assembly in payload["structure_assemblies"]
        for identifier in assembly["element_identifiers"]
        if identifier.endswith("/")
    ]
    assert offenders == [], f"placeholder element identifiers remain: {offenders[:3]}"
    # The identifiers also survive catalogue validation and name their element.
    catalog = load_packaged_machine_map("jt-60sa")
    for assembly in catalog.structure_assemblies:
        for identifier in assembly.element_identifiers:
            assert identifier and not identifier.endswith("/")


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
    # The vessel and the cryostat are separate ragged loops in both phases: the
    # vessel changes at the boundary, the cryostat is the same 57 filaments.
    assert op1_vessel == {"VV", "CRYOSTAT"}
    assert op2_vessel == {"VV", "CRYOSTAT"}

    # A single loop assembly per vessel and cryostat loop, in each phase.
    vessels = [
        assembly
        for assembly in catalog.structure_assemblies
        if assembly.structure_path == "pf_passive/loop/element/geometry/rectangle"
    ]
    assert len(vessels) == 4


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
    # Both phases carry the vessel loop and the shared 57-filament cryostat
    # loop; only the vessel filament count differs (63 OP1, 98 OP2).
    expected_vessel = 120 if phase == "OP1" else 155
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
def test_description_route_supplies_the_stored_poloidal_angle_for_every_probe():
    """The catalogue binds the DD probe angle and the reader reports it in degrees.

    Both phase stores hold every tangential probe's sensing-axis angle
    ``theta = (90 deg - omega) mod 360 deg`` in radians.  The description
    binding carries ``identity``, so ``read_geometry_table`` supplies each
    probe's ``angle_deg`` from the stored ``poloidal_angle``, converted to
    degrees.
    """
    import imas

    shot = 100595
    catalogue = load_packaged_machine_map("jt-60sa")
    description = transform_machine_description(
        catalogue, shot, "netcdf", JT60SA_DESCRIPTION_DIR
    )
    assert description.status == "emitted", description.detail
    angle_arrays = [
        array
        for array in description.arrays
        if array.dd_path == "magnetics/b_field_pol_probe/poloidal_angle"
    ]
    assert len(angle_arrays) == 1
    assert angle_arrays[0].target_unit == "rad"

    stored_deg: dict[str, float] = {}
    with imas.DBEntry(
        JT60SA_DESCRIPTION_DIR / "OP1" / "magnetics.nc",
        "r",
        dd_version=catalogue.dd_version,
    ) as entry:
        magnetics = entry.get("magnetics", autoconvert=False)
        for probe in magnetics.b_field_pol_probe:
            stored_deg[str(probe.name)] = float(np.rad2deg(probe.poloidal_angle))
    assert len(stored_deg) == 17

    table = read_geometry_table(shot, machine="jt-60sa")
    probes = [item for item in table.sensor_map if item.kind == "b_probe"]
    assert [item.amb_channel for item in probes] == list(stored_deg)
    for probe in probes:
        assert probe.angle_deg == pytest.approx(
            stored_deg[probe.amb_channel], abs=1e-9
        )
        assert probe.flag == ""


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
    before_payload = _binding_payload(before)
    after_payload = _binding_payload(after)

    assert before_payload != after_payload
    shared = set(before_payload) & set(after_payload)
    changed = {key for key in shared if before_payload[key] != after_payload[key]}

    # The vacuum vessel moves at the boundary: each phase's deck carries its own
    # vessel run, OP1's 63 filaments giving way to OP2's 98.
    vessel_keys = [key for key in shared if key.startswith("jt60sa-pf-passive-vv")]
    assert vessel_keys
    assert all(key in changed for key in vessel_keys)

    # The cryostat loop is the same 57 filaments in both phases, so those
    # bindings are byte-equal across the boundary.
    cryostat_keys = [
        key for key in shared if key.startswith("jt60sa-pf-passive-cryostat")
    ]
    assert cryostat_keys
    assert all(
        before_payload[key] == after_payload[key] for key in cryostat_keys
    )

    # The wall moves: each phase's limiter and vessel unit come from its own
    # coil_vv deck, and the first wall changed for OP2, so the shared limiter
    # bindings differ across the boundary.
    limiter_keys = [key for key in shared if key.startswith("jt60sa-wall-limiter-")]
    assert limiter_keys
    assert all(key in changed for key in limiter_keys)
    # Both phases now carry the vessel unit (annular inner and outer outlines).
    # The vessel's contour table is the same in both decks, so those bindings
    # are byte-equal across the boundary.
    wall_vessel_keys = [
        key for key in shared if key.startswith("jt60sa-wall-vessel-")
    ]
    assert wall_vessel_keys
    assert all(before_payload[key] == after_payload[key] for key in wall_vessel_keys)

    # pf_active and magnetics are phase-independent, so the boundary is a
    # description change and not a re-addressing.
    for family in ("jt60sa-pf-active-", "jt60sa-magnetics-"):
        family_keys = [key for key in shared if key.startswith(family)]
        assert family_keys
        assert all(key not in changed for key in family_keys)

    print(
        "JT60SA_PHASE_BOUNDARY "
        f"boundary=101174 changed={len(changed)} shared={len(shared)}"
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
