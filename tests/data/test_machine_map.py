"""Machine maps are closed-world data declarations, not reader code."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import imas
import pytest

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
EXPECTED_BOUND_CHANNEL_COUNTS = {"magnetics": 71, "pf_active": 71, "wall": 3}


def _dd_leaf(dd_version: str, path: str):
    ids_name, leaf = path.split("/", maxsplit=1)
    return imas.IDSFactory(dd_version).new(ids_name).metadata[leaf]


@lru_cache(maxsize=1)
def _machine_description_inventory() -> tuple[
    list[int], dict[str, dict[str, set[str]]]
]:
    shots: list[int] = []
    inventory: dict[str, dict[str, set[str]]] = {
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
                unit = str(row.get("attributes", {}).get("units", ""))
                inventory[group].setdefault(name, set()).add(unit)
    return sorted(shots), inventory


def test_linkml_schema_is_declarative_and_has_no_executable_language():
    schema = load_linkml_schema()
    assert LINKML_SCHEMA_PATH.name == "schema.yaml"
    assert schema["classes"]["MachineMapCatalog"]["tree_root"] is True
    assert set(schema["enums"]["BindingRole"]["permissible_values"]) == {
        "value",
        "identifier",
        "dimension-coordinate",
    }
    forbidden = {"condition", "expression", "code_hook", "transform_code"}
    assert forbidden.isdisjoint(schema["slots"])


def test_mast_maps_bind_every_machine_description_array_in_the_corpus():
    if not LEVEL2_ROOT.is_dir():
        pytest.skip("FAIR-MAST level-2 mirror is not mounted")
    catalog = load_packaged_machine_map("mast")
    shots, inventory = _machine_description_inventory()
    bindings = catalog.binding_sets["mast-machine-description"]

    assert len(shots) == 11_573
    assert (shots[0], shots[-1]) == (11_766, 30_471)
    assert catalog.bound_channel_counts == EXPECTED_BOUND_CHANNEL_COUNTS
    assert catalog.bound_channel_count == 145
    assert len(catalog.maps) * catalog.bound_channel_count == 1_595
    assert catalog.validation_gaps == ()

    bindings_by_group: dict[str, dict[str, object]] = {
        group: {
            binding.source_array: binding
            for binding in bindings
            if binding.source_group == group
        }
        for group in MACHINE_DESCRIPTION_GROUPS
    }
    for group in MACHINE_DESCRIPTION_GROUPS:
        assert set(bindings_by_group[group]) == set(inventory[group])

    for binding in bindings:
        leaf = _dd_leaf(catalog.dd_version, binding.dd_path)
        assert (leaf.units or "1") == binding.target_unit
        observed_units = inventory[binding.source_group][binding.source_array]
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
