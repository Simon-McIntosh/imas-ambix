"""Machine maps are closed-world data declarations, not reader code."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import imas
import numpy as np
import pytest
from imas.ids_struct_array import IDSStructArray

from imas_ambix.data.geometry_transitions import (
    build_geometry_transitions,
    load_geometry_table_payload,
)
from imas_ambix.data.machine_map import (
    LINKML_SCHEMA_PATH,
    MachineMapError,
    assert_transition_alignment,
    load_linkml_schema,
    load_machine_map,
    load_packaged_machine_map,
    map_for_shot,
)
from imas_ambix.data.manifest import load_index

LEVEL2_ROOT = Path("/work/projects/imas_gpu/mast/level2/shots")
MACHINE_DESCRIPTION_GROUPS = ("magnetics", "pf_active", "pf_passive", "wall")
EXPECTED_BOUND_CHANNEL_COUNTS = {
    "magnetics": 70,
    "pf_active": 71,
    "pf_passive": 110,
    "wall": 3,
}
PASSIVE_LOOP_FAMILIES = (
    "botcol",
    "coil_cases",
    "endcrown_l",
    "endcrown_u",
    "incon",
    "lhorw",
    "mid",
    "p2larm",
    "p2ldivpl",
    "p2uarm",
    "p2udivpl",
    "ring",
    "rodgr",
    "topcol",
    "uhorw",
    "vertw",
)
LEGACY_DESCRIPTION_FIELDS = (
    "magpr_r",
    "magpr_z",
    "magpr_ang",
    "magpr_len.cc",
    "magpr_len.ccbv",
    "magpr_len.obr",
    "magpr_len.obv",
    "magpr_len.omv",
    "silop_r",
    "silop_z",
    "pf_active.element.r",
    "pf_active.element.z",
    "pf_active.element.width",
    "pf_active.element.height",
    "pf_passive.loop.name",
    "pf_passive.element.name",
    "pf_passive.element.r",
    "pf_passive.element.z",
    "pf_passive.element.width",
    "pf_passive.element.height",
    "pf_active.element.turns_with_sign",
    "pf_passive.element.turns_with_sign",
    "fcoil_circ",
    "fcoil_xmult",
    "amc_current_channels",
    "limiterr",
    "limiterz",
)


def _dd_leaf(dd_version: str, path: str):
    ids_name, leaf = path.split("/", maxsplit=1)
    return imas.IDSFactory(dd_version).new(ids_name).metadata[leaf]


def _dd_struct_array_count(dd_version: str, path: str) -> int:
    ids_name, relative_path = path.split("/", maxsplit=1)
    node = imas.IDSFactory(dd_version).new(ids_name)
    count = 0
    for component in relative_path.split("/")[:-1]:
        node = getattr(node, component)
        if isinstance(node, IDSStructArray):
            count += 1
            node.resize(1)
            node = node[0]
    return count


def _source_data_kind(data_type: object) -> str:
    declaration = json.dumps(data_type, sort_keys=True).lower()
    if "string" in declaration or "utf" in declaration:
        return "STR"
    if "float" in declaration:
        return "FLT"
    if "int" in declaration:
        return "INT"
    raise AssertionError(f"unsupported source data type {data_type!r}")


@lru_cache(maxsize=1)
def _machine_description_inventory() -> tuple[
    list[int], dict[str, dict[str, dict[str, set[object]]]]
]:
    shots: list[int] = []
    inventory: dict[str, dict[str, dict[str, set[object]]]] = {
        group: {} for group in MACHINE_DESCRIPTION_GROUPS
    }
    for entry in os.scandir(LEVEL2_ROOT):
        if not entry.is_dir() or not entry.name.endswith(".zarr"):
            continue
        shot = int(entry.name.removesuffix(".zarr"))
        metadata_path = Path(entry.path) / "zarr.json"
        metadata = json.loads(metadata_path.read_text())["consolidated_metadata"][
            "metadata"
        ]
        shots.append(shot)
        for group in MACHINE_DESCRIPTION_GROUPS:
            prefix = f"{group}/"
            for path, row in metadata.items():
                if not path.startswith(prefix) or row.get("node_type") != "array":
                    continue
                name = path.removeprefix(prefix)
                observation = inventory[group].setdefault(
                    name,
                    {"units": set(), "ranks": set(), "data_kinds": set()},
                )
                unit = str(row.get("attributes", {}).get("units", ""))
                observation["units"].add(unit)
                observation["ranks"].add(len(row["shape"]))
                observation["data_kinds"].add(_source_data_kind(row["data_type"]))
    return sorted(shots), inventory


def test_linkml_schema_is_declarative_and_has_no_executable_language():
    schema = load_linkml_schema()
    assert LINKML_SCHEMA_PATH.name == "schema.yaml"
    assert schema["classes"]["MachineMapCatalog"]["tree_root"] is True
    assert "SourceQualification" in schema["classes"]
    assert "CircuitConnection" in schema["classes"]
    assert "DriveTopology" in schema["classes"]
    assert "StructureAssembly" in schema["classes"]
    assert set(schema["enums"]["BindingRole"]["permissible_values"]) == {
        "value",
        "identifier",
        "dimension-coordinate",
    }
    assert set(schema["enums"]["SourceStatus"]["permissible_values"]) == {
        "corpus-observed",
        "legacy-only",
    }
    forbidden = {"condition", "expression", "code_hook", "transform_code"}
    assert forbidden.isdisjoint(schema["slots"])


def test_mast_catalog_accounts_for_every_machine_description_array_in_the_corpus():
    if not LEVEL2_ROOT.is_dir():
        pytest.skip("FAIR-MAST level-2 mirror is not mounted")
    catalog = load_packaged_machine_map("mast")
    shots, inventory = _machine_description_inventory()
    bindings = catalog.binding_sets["mast-machine-description"]

    assert len(shots) == 11_573
    assert (shots[0], shots[-1]) == (11_766, 30_471)
    assert catalog.bound_channel_counts == EXPECTED_BOUND_CHANNEL_COUNTS
    assert catalog.bound_channel_count == 254
    assert catalog.qualified_channel_counts == {
        "magnetics": 4,
        "pf_active": 1,
        "pf_passive": 1,
    }
    assert catalog.qualified_channel_count == 6
    assert len(catalog.maps) * catalog.bound_channel_count == 508
    assert catalog.validation_gaps == ()
    assert (
        sum(
            item.source_status == "corpus-observed"
            for item in catalog.source_qualifications
        )
        == 1
    )
    assert (
        sum(
            item.source_status == "legacy-only"
            for item in catalog.source_qualifications
        )
        == 5
    )

    bindings_by_group: dict[str, dict[str, object]] = {
        group: {
            binding.source_array: binding
            for binding in bindings
            if binding.source_group == group
        }
        for group in MACHINE_DESCRIPTION_GROUPS
    }
    qualifications_by_group: dict[str, dict[str, object]] = {
        group: {
            item.source_array: item
            for item in catalog.source_qualifications
            if item.source_group == group and item.source_status == "corpus-observed"
        }
        for group in MACHINE_DESCRIPTION_GROUPS
    }
    for group in MACHINE_DESCRIPTION_GROUPS:
        bound_sources = set(bindings_by_group[group])
        qualified_sources = set(qualifications_by_group[group])
        assert bound_sources.isdisjoint(qualified_sources)
        assert bound_sources | qualified_sources == set(inventory[group])

    for binding in bindings:
        leaf = _dd_leaf(catalog.dd_version, binding.dd_path)
        assert (leaf.units or "1") == binding.target_unit
        observed_units = inventory[binding.source_group][binding.source_array]["units"]
        nonempty_units = {unit for unit in observed_units if unit}
        if binding.source_role == "value":
            assert binding.sign_convention == "identity"
            expected_source_unit = (
                "degree" if binding.target_unit == "rad" else binding.target_unit
            )
            assert binding.source_unit == expected_source_unit
            if binding.source_unit in {"degree", "m"}:
                assert nonempty_units == {"SI, degrees, m"}
            elif binding.source_unit == "s":
                assert {unit.lower() for unit in nonempty_units} == {"s"}
            else:
                assert nonempty_units == {binding.source_unit}
        else:
            assert binding.source_unit == binding.target_unit == "1"
            assert binding.sign_convention == "not-applicable"
            assert not nonempty_units


def test_every_binding_targets_a_writable_shape_compatible_dd_leaf():
    if not LEVEL2_ROOT.is_dir():
        pytest.skip("FAIR-MAST level-2 mirror is not mounted")
    _, inventory = _machine_description_inventory()

    for machine in ("mast", "diii-d"):
        catalog = load_packaged_machine_map(machine)
        for bindings in catalog.binding_sets.values():
            for binding in bindings:
                leaf = _dd_leaf(catalog.dd_version, binding.dd_path)
                assert leaf.data_type.name not in {"STRUCTURE", "STRUCT_ARRAY"}
                if machine != "mast":
                    assert leaf.data_type.name == "FLT"
                    continue

                observation = inventory[binding.source_group][binding.source_array]
                assert observation["data_kinds"] == {leaf.data_type.name}
                assert observation["ranks"] == {binding.source_rank}
                for source_rank in observation["ranks"]:
                    expanded_dimensions = source_rank - leaf.ndim
                    assert expanded_dimensions >= 0, binding.name
                    if expanded_dimensions:
                        assert expanded_dimensions <= _dd_struct_array_count(
                            catalog.dd_version, binding.dd_path
                        ), binding.name


def test_saddle_trajectories_are_twelve_named_loops_with_twenty_eight_points():
    if not LEVEL2_ROOT.is_dir():
        pytest.skip("FAIR-MAST level-2 mirror is not mounted")
    catalog = load_packaged_machine_map("mast")
    bindings = {
        item.source_array: item
        for item in catalog.binding_sets["mast-machine-description"]
    }
    qualified_sources = {item.source_array for item in catalog.source_qualifications}
    metadata = json.loads((LEVEL2_ROOT / "12417.zarr" / "zarr.json").read_text())[
        "consolidated_metadata"
    ]["metadata"]

    expected_sources = {
        f"b_field_tor_probe_saddle_{band}_{axis}"
        for band in "lmu"
        for axis in ("r", "z", "phi")
    }
    assert expected_sources.isdisjoint(qualified_sources)
    for source in expected_sources:
        binding = bindings[source]
        axis = source.rsplit("_", maxsplit=1)[1]
        assert binding.source_rank == 2
        assert tuple(metadata[f"magnetics/{source}"]["shape"]) == (12, 28)
        assert binding.dd_path == f"magnetics/flux_loop/position/{axis}"
        assert "axis 0 matches the 12" in binding.evidence
        assert "axis 1 matches coordinate 0..27" in binding.evidence

    saddle_assemblies = {
        item.name: item
        for item in catalog.structure_assemblies
        if item.name.startswith("mast-magnetics-saddle-")
    }
    assert len(saddle_assemblies) == 3
    for band in "lmu":
        assembly = saddle_assemblies[f"mast-magnetics-saddle-{band}-flux-loops"]
        assert assembly.structure_path == "magnetics/flux_loop/position"
        assert assembly.type_path == "magnetics/flux_loop/type/index"
        assert assembly.type_index == 2
        name_binding = bindings[f"b_field_tor_probe_saddle_{band}_geometry_channel"]
        assert assembly.name_binding == name_binding.name
        assert name_binding.dd_path == "magnetics/flux_loop/name"
        assert len(assembly.member_bindings) == 3

    coordinate = next(
        item
        for item in catalog.source_qualifications
        if item.source_array == "coordinate"
    )
    assert coordinate.source_shape == (28,)
    assert "no writable coordinate-index leaf" in coordinate.reason


def test_passive_loop_elements_and_shape_angles_are_executable_oblique_geometry():
    if not LEVEL2_ROOT.is_dir():
        pytest.skip("FAIR-MAST level-2 mirror is not mounted")
    catalog = load_packaged_machine_map("mast")
    bindings = {
        item.source_array: item
        for item in catalog.binding_sets["mast-machine-description"]
        if item.source_group == "pf_passive"
    }
    expected_rectangle_sources = {
        f"{family}_{suffix}"
        for family in ("coil_cases",)
        for suffix in ("geometry_channel", "r", "z", "width", "height")
    }
    oblique_families = tuple(
        family for family in PASSIVE_LOOP_FAMILIES if family != "coil_cases"
    )
    expected_oblique_sources = {
        f"{family}_{suffix}"
        for family in oblique_families
        for suffix in (
            "geometry_channel",
            "r",
            "z",
            "width",
            "height",
            "shapeAngle1",
            "shapeAngle2",
        )
    }
    assert set(bindings) == expected_rectangle_sources | expected_oblique_sources
    assert len(bindings) == 110

    for family in oblique_families:
        name_binding = bindings[f"{family}_geometry_channel"]
        assert name_binding.dd_path == "pf_passive/loop/element/name"
        expected_targets = {
            "r": "r",
            "z": "z",
            "width": "length_alpha",
            "height": "length_beta",
            "shapeAngle1": "alpha",
            "shapeAngle2": "beta",
        }
        for source_suffix, target_suffix in expected_targets.items():
            binding = bindings[f"{family}_{source_suffix}"]
            assert binding.dd_path == (
                "pf_passive/loop/element/geometry/oblique/" + target_suffix
            )
        assert "first side inclination" in bindings[f"{family}_shapeAngle1"].evidence
        assert "second side inclination" in bindings[f"{family}_shapeAngle2"].evidence
        assert all(
            "must not be converted as lengths" in bindings[f"{family}_{angle}"].evidence
            for angle in ("shapeAngle1", "shapeAngle2")
        )

    metadata = json.loads((LEVEL2_ROOT / "12417.zarr" / "zarr.json").read_text())[
        "consolidated_metadata"
    ]["metadata"]
    for family in oblique_families:
        for angle in ("shapeAngle1", "shapeAngle2"):
            binding = bindings[f"{family}_{angle}"]
            assert binding.source_rank == len(
                metadata[f"pf_passive/{family}_{angle}"]["shape"]
            )
            assert binding.source_unit == "degree"

    oblique_assemblies = {
        item.name: item
        for item in catalog.structure_assemblies
        if item.name.startswith("mast-pf-passive-")
    }
    assert len(oblique_assemblies) == 15
    assert {item.type_index for item in oblique_assemblies.values()} == {3}
    assert {item.type_path for item in oblique_assemblies.values()} == {
        "pf_passive/loop/element/geometry/geometry_type"
    }


def test_every_legacy_description_field_has_a_binding_or_qualification():
    catalog = load_packaged_machine_map("mast")
    bindings = catalog.binding_sets["mast-machine-description"]
    binding_names = {
        (item.source_group, item.source_array): item.name for item in bindings
    }
    qualification_names = {
        (item.source_group, item.source_array): item.name
        for item in catalog.source_qualifications
    }
    topology_names = tuple(item.name for item in catalog.drive_topologies)

    def binding_matches(group: str, suffix: str) -> tuple[str, ...]:
        return tuple(
            item.name
            for item in bindings
            if item.source_group == group and item.source_array.endswith(suffix)
        )

    accounting = {
        "magpr_r": tuple(
            item.name
            for item in bindings
            if item.source_group == "magnetics"
            and item.source_array.startswith("b_field_pol_probe_")
            and item.source_array.endswith("_r")
        ),
        "magpr_z": tuple(
            item.name
            for item in bindings
            if item.source_group == "magnetics"
            and item.source_array.startswith("b_field_pol_probe_")
            and item.source_array.endswith("_z")
        ),
        "magpr_ang": (qualification_names[("magnetics", "magpr_ang")],),
        "magpr_len.cc": (
            qualification_names[("magnetics", "b_field_pol_probe_cc_length")],
        ),
        "magpr_len.ccbv": (
            binding_names[("magnetics", "b_field_pol_probe_ccbv_length")],
        ),
        "magpr_len.obr": (
            binding_names[("magnetics", "b_field_pol_probe_obr_length")],
        ),
        "magpr_len.obv": (
            binding_names[("magnetics", "b_field_pol_probe_obv_length")],
        ),
        "magpr_len.omv": (
            qualification_names[("magnetics", "b_field_pol_probe_omv_length")],
        ),
        "silop_r": (binding_names[("magnetics", "flux_loop_r")],),
        "silop_z": (binding_names[("magnetics", "flux_loop_z")],),
        "pf_active.element.r": binding_matches("pf_active", "_r"),
        "pf_active.element.z": binding_matches("pf_active", "_z"),
        "pf_active.element.width": binding_matches("pf_active", "_width"),
        "pf_active.element.height": binding_matches("pf_active", "_height"),
        "pf_passive.loop.name": (qualification_names[("pf_passive", "loop_names")],),
        "pf_passive.element.name": binding_matches("pf_passive", "_geometry_channel"),
        "pf_passive.element.r": binding_matches("pf_passive", "_r"),
        "pf_passive.element.z": binding_matches("pf_passive", "_z"),
        "pf_passive.element.width": binding_matches("pf_passive", "_width"),
        "pf_passive.element.height": binding_matches("pf_passive", "_height"),
        "pf_active.element.turns_with_sign": (*topology_names,),
        "pf_passive.element.turns_with_sign": (*topology_names,),
        "fcoil_circ": topology_names,
        "fcoil_xmult": topology_names,
        "amc_current_channels": (
            qualification_names[("pf_active", "amc_current_channels")],
        ),
        "limiterr": (binding_names[("wall", "limiter_r")],),
        "limiterz": (binding_names[("wall", "limiter_z")],),
    }

    assert tuple(accounting) == LEGACY_DESCRIPTION_FIELDS
    unaccounted = []
    for field, declarations in accounting.items():
        print(f"LEGACY_FIELD_ACCOUNTING field={field} declarations={declarations}")
        if not declarations:
            unaccounted.append(field)
    assert unaccounted == []

    assert len(catalog.drive_topologies) == 3
    for topology in catalog.drive_topologies:
        assert topology.circuit_identity_source == "fcoil_circ"
        assert topology.current_scale_source == "fcoil_xmult"
        assert topology.current_channel_source == "amc_current_channels"
        assert (
            topology.current_channel_qualification
            == accounting["amc_current_channels"][0]
        )
        assert topology.passive_loop_names == PASSIVE_LOOP_FAMILIES


def test_turn_magnitudes_are_positive_and_direction_is_connectivity_only():
    for machine in ("mast", "diii-d"):
        catalog = load_packaged_machine_map(machine)
        for bindings in catalog.binding_sets.values():
            for binding in bindings:
                if (
                    "turn" in binding.source_array.lower()
                    or "turn" in binding.dd_path.lower()
                ):
                    assert binding.sign_convention != "negate"
        for qualification in catalog.source_qualifications:
            declaration = " ".join(
                (qualification.name, qualification.source_array, qualification.reason)
            ).lower()
            assert "turns_with_sign" not in declaration
        for topology in catalog.drive_topologies:
            assert topology.circuit_identifier_path == "pf_active/circuit/name"
            assert topology.supply_identifier_path == "pf_active/supply/name"
            assert topology.element_identifier_path == "pf_active/coil/element/name"
            assert topology.turns_path == "pf_active/coil/element/turns_with_sign"
            assert topology.connections_path == "pf_active/circuit/connections"
            assert all(connection.turns > 0 for connection in topology.connections)
            assert {connection.direction for connection in topology.connections} <= {
                -1,
                1,
            }
            circuit_identifier = _dd_leaf(
                catalog.dd_version, topology.circuit_identifier_path
            )
            supply_identifier = _dd_leaf(
                catalog.dd_version, topology.supply_identifier_path
            )
            element_identifier = _dd_leaf(
                catalog.dd_version, topology.element_identifier_path
            )
            turns = _dd_leaf(catalog.dd_version, topology.turns_path)
            connections = _dd_leaf(catalog.dd_version, topology.connections_path)
            assert circuit_identifier.data_type.name == "STR"
            assert supply_identifier.data_type.name == "STR"
            assert element_identifier.data_type.name == "STR"
            assert "identifier" in circuit_identifier.documentation
            assert "identifier" in supply_identifier.documentation
            assert "identifier" in element_identifier.documentation
            assert turns.data_type.name == "FLT"
            assert turns.units == "1"
            assert "Should be positive" in turns.documentation
            assert (connections.data_type.name, connections.ndim) == ("INT", 2)
            assert "positive side" in connections.documentation
            assert "negative side" in connections.documentation


def test_sparse_connectivity_reconstructs_every_legacy_signed_drive_element():
    catalog = load_packaged_machine_map("mast")
    payload = load_geometry_table_payload()
    topologies = {
        topology.source_location.rsplit("/", maxsplit=1)[-1]: topology
        for topology in catalog.drive_topologies
    }
    total_elements = 0
    direction_counts = {-1: 0, 1: 0}

    for signature, campaign in payload["campaigns"].items():
        topology = topologies[signature]
        filaments = campaign["pf_filaments"]
        expected_circuits = tuple(
            f"fcoil-circuit-{int(item['circuit']):03d}" for item in filaments
        )
        expected_elements = tuple(
            f"fcoil-element-{index:04d}" for index in range(len(filaments))
        )
        expected_supplies = tuple(
            f"fcoil-supply-{int(item['circuit']):03d}" for item in filaments
        )
        legacy_signed_drive = np.asarray(
            [float(item["turns"]) * float(item["xmult"]) for item in filaments]
        )
        circuit_identifiers = tuple(
            connection.circuit_identifier for connection in topology.connections
        )
        element_identifiers = tuple(
            connection.element_identifier for connection in topology.connections
        )
        supply_identifiers = tuple(
            connection.supply_identifier for connection in topology.connections
        )
        positive_turns = np.asarray(
            [connection.turns for connection in topology.connections]
        )

        unique_circuits = tuple(dict.fromkeys(circuit_identifiers))
        circuit_rows = {
            identifier: index for index, identifier in enumerate(unique_circuits)
        }
        connectivity = np.zeros(
            (len(unique_circuits), len(topology.connections)), dtype=np.int8
        )
        for column, connection in enumerate(topology.connections):
            connectivity[circuit_rows[connection.circuit_identifier], column] = (
                connection.direction
            )
            direction_counts[connection.direction] += 1

        selected_direction = connectivity[
            [circuit_rows[identifier] for identifier in circuit_identifiers],
            np.arange(len(topology.connections)),
        ]
        reconstructed = positive_turns * selected_direction
        assert circuit_identifiers == expected_circuits
        assert supply_identifiers == expected_supplies
        assert element_identifiers == expected_elements
        assert np.count_nonzero(connectivity, axis=0).tolist() == [1] * len(filaments)
        assert np.array_equal(reconstructed, legacy_signed_drive)
        total_elements += len(filaments)
        negative_count = np.count_nonzero(reconstructed < 0)
        print(
            "CIRCUIT_DRIVE_RECONSTRUCTION "
            f"signature={signature} elements={len(filaments)} "
            f"circuits={len(unique_circuits)} negative={negative_count}"
        )

    assert total_elements == 2_946
    assert direction_counts == {-1: 0, 1: 2_946}
    for machine_map in catalog.maps:
        digest = machine_map.transition.rsplit("-", maxsplit=1)[-1]
        assert machine_map.drive_topology == f"mast-legacy-fcoil-drive-{digest}"


def test_every_mast_map_range_is_one_geometry_transition():
    if not LEVEL2_ROOT.is_dir():
        pytest.skip("FAIR-MAST level-2 mirror is not mounted")
    catalog = load_packaged_machine_map("mast")
    shots, _ = _machine_description_inventory()
    transitions = build_geometry_transitions(
        shots,
        load_index(),
        load_geometry_table_payload(),
    )

    declared_ranges = [
        (item.first_shot, item.last_shot, item.transition) for item in catalog.maps
    ]
    emitted_ranges = [
        (item.first_shot, item.last_shot, item.name) for item in transitions
    ]
    mismatches = set(declared_ranges).symmetric_difference(emitted_ranges)

    assert declared_ranges == [
        (11_766, 12_416, "mast-geometry-11766-9425ae4a8bf3bc15"),
        (12_417, 30_471, "mast-geometry-12417-edd753d282903679"),
    ]
    assert len(transitions) == len(catalog.maps) == 2
    assert len(mismatches) == 0
    assert_transition_alignment(catalog, transitions)
    bindings = catalog.binding_sets["mast-machine-description"]
    assert all(catalog.bindings_for(item) == bindings for item in catalog.maps)
    assert len({item.name for item in bindings}) == catalog.bound_channel_count
    assert len({item.name for item in catalog.source_qualifications}) == (
        catalog.qualified_channel_count
    )
    topology_names = {item.name for item in catalog.drive_topologies}
    for transition in transitions:
        machine_map = map_for_shot(catalog, transition.first_shot)
        assert machine_map.transition == transition.name
        assert machine_map.last_shot == transition.last_shot
        assert machine_map.drive_topology in topology_names
    assert len({item.drive_topology for item in catalog.maps}) == 2
    print(
        "CATALOG_TRANSITION_ALIGNMENT "
        f"maps={len(catalog.maps)} transitions={len(transitions)} "
        f"mismatches={len(mismatches)} executable={dict(catalog.bound_channel_counts)} "
        f"qualified={dict(catalog.qualified_channel_counts)}"
    )


def test_diii_d_public_map_retains_only_direct_plasma_current():
    catalog = load_packaged_machine_map("diii-d")
    machine_map = catalog.maps[0]
    bindings = catalog.bindings_for(machine_map)
    gaps = {gap.binding: gap.reason for gap in catalog.validation_gaps}

    assert catalog.source == "https://github.com/MIT-PSFC/disruption-py"
    assert catalog.source_revision == "dec5c58a3e3970bc6817f33efb615fea11057fce"
    assert machine_map.validation_state == "source-only"
    assert catalog.bound_channel_count == len(bindings) == len(gaps) == 1
    assert catalog.source_qualifications == ()
    assert gaps.keys() == {binding.name for binding in bindings}
    assert all("No DIII-D pulse corpus" in reason for reason in gaps.values())
    assert all(binding.sign_convention == "unknown-unvalidated" for binding in bindings)
    assert [
        (binding.source_group, binding.source_array, binding.dd_path)
        for binding in bindings
    ] == [("d3d", "ip", "magnetics/ip/data")]
    for binding in bindings:
        leaf = _dd_leaf(catalog.dd_version, binding.dd_path)
        assert leaf.data_type.name == "FLT"
        assert leaf.units == binding.target_unit
        assert catalog.source_revision in binding.evidence


def test_machine_maps_exclude_reconstruction_derived_bindings():
    disallowed = {"equilibrium", "q95", "q_95", "li_3", "betan", "beta_tor_norm"}
    for machine in ("mast", "diii-d"):
        catalog = load_packaged_machine_map(machine)
        for bindings in catalog.binding_sets.values():
            for binding in bindings:
                declaration = "/".join(
                    (binding.source_group, binding.source_array, binding.dd_path)
                ).lower()
                assert not any(term in declaration for term in disallowed)


def test_loader_rejects_an_undeclared_conditional_slot(tmp_path):
    source = json.loads((LINKML_SCHEMA_PATH.parent / "mast.json").read_text())
    source["maps"][0]["condition"] = "shot > 12000"
    invalid = tmp_path / "conditional.json"
    invalid.write_text(json.dumps(source))

    with pytest.raises(MachineMapError, match="extra=.+condition"):
        load_machine_map(invalid)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (("turns", -1.0, "number > 0"), ("direction", 0, "must be -1 or 1")),
)
def test_loader_rejects_nonpositive_turns_and_unsigned_connections(
    tmp_path, field, value, message
):
    source = json.loads((LINKML_SCHEMA_PATH.parent / "mast.json").read_text())
    source["drive_topologies"][0]["connections"][0][field] = value
    invalid = tmp_path / f"invalid-{field}.json"
    invalid.write_text(json.dumps(source))

    with pytest.raises(MachineMapError, match=message):
        load_machine_map(invalid)
