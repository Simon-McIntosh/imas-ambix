"""Machine maps are closed-world data declarations, not reader code."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import imas
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
MACHINE_DESCRIPTION_GROUPS = ("magnetics", "pf_active", "wall")
EXPECTED_BOUND_CHANNEL_COUNTS = {"magnetics": 61, "pf_active": 71, "wall": 3}
EXPECTED_QUALIFIED_SOURCES = {
    "b_field_tor_probe_saddle_l_phi": (12, 28),
    "b_field_tor_probe_saddle_l_r": (12, 28),
    "b_field_tor_probe_saddle_l_z": (12, 28),
    "b_field_tor_probe_saddle_m_phi": (12, 28),
    "b_field_tor_probe_saddle_m_r": (12, 28),
    "b_field_tor_probe_saddle_m_z": (12, 28),
    "b_field_tor_probe_saddle_u_phi": (12, 28),
    "b_field_tor_probe_saddle_u_r": (12, 28),
    "b_field_tor_probe_saddle_u_z": (12, 28),
    "coordinate": (28,),
}


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
    assert set(schema["enums"]["BindingRole"]["permissible_values"]) == {
        "value",
        "identifier",
        "dimension-coordinate",
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
    assert catalog.bound_channel_count == 135
    assert catalog.qualified_channel_counts == {"magnetics": 10}
    assert catalog.qualified_channel_count == 10
    assert len(catalog.maps) * catalog.bound_channel_count == 1_485
    assert catalog.validation_gaps == ()

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
            if item.source_group == group
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
                for source_rank in observation["ranks"]:
                    expanded_dimensions = source_rank - leaf.ndim
                    assert expanded_dimensions in {0, 1}, binding.name
                    if expanded_dimensions:
                        assert _dd_struct_array_count(
                            catalog.dd_version, binding.dd_path
                        ), binding.name


def test_unsupported_saddle_outlines_are_explicit_source_qualifications():
    if not LEVEL2_ROOT.is_dir():
        pytest.skip("FAIR-MAST level-2 mirror is not mounted")
    catalog = load_packaged_machine_map("mast")
    qualifications = {item.source_array: item for item in catalog.source_qualifications}
    representative_metadata = json.loads(
        (LEVEL2_ROOT / "12417.zarr" / "zarr.json").read_text()
    )["consolidated_metadata"]["metadata"]

    assert {
        source: item.source_shape for source, item in qualifications.items()
    } == EXPECTED_QUALIFIED_SOURCES
    for source, item in qualifications.items():
        row = representative_metadata[f"magnetics/{source}"]
        assert tuple(row["shape"]) == item.source_shape
        assert "DD 4.1.1" in item.reason
        assert "writable" in item.reason
        assert "DD metadata" in item.evidence
    assert "STRUCTURE" in qualifications["coordinate"].evidence
    assert all(
        "FLT_0D" in item.evidence
        for source, item in qualifications.items()
        if source != "coordinate"
    )


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

    assert len(transitions) == len(catalog.maps) == 11
    assert_transition_alignment(catalog, transitions)
    for transition in transitions:
        machine_map = map_for_shot(catalog, transition.first_shot)
        assert machine_map.transition == transition.name
        assert machine_map.last_shot == transition.last_shot


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
