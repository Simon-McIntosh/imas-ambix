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


def _dd_leaf(dd_version: str, path: str):
    ids_name, leaf = path.split("/", maxsplit=1)
    return imas.IDSFactory(dd_version).new(ids_name).metadata[leaf]


@lru_cache(maxsize=1)
def _interferometer_inventory() -> tuple[list[int], set[str], dict[str, set[str]]]:
    shots: list[int] = []
    arrays: set[str] = set()
    units: dict[str, set[str]] = {}
    for entry in os.scandir(LEVEL2_ROOT):
        if not entry.is_dir() or not entry.name.endswith(".zarr"):
            continue
        shot = int(entry.name.removesuffix(".zarr"))
        metadata_path = Path(entry.path) / "zarr.json"
        metadata = json.loads(metadata_path.read_text())["consolidated_metadata"][
            "metadata"
        ]
        prefix = "interferometer/"
        names = {
            path.removeprefix(prefix)
            for path, row in metadata.items()
            if path.startswith(prefix) and row.get("node_type") == "array"
        }
        shots.append(shot)
        arrays.update(names)
        for name in names:
            unit = str(metadata[f"{prefix}{name}"]["attributes"].get("units", ""))
            units.setdefault(name, set()).add(unit)
    return sorted(shots), arrays, units


def test_linkml_schema_is_declarative_and_has_no_executable_language():
    schema = load_linkml_schema()
    assert LINKML_SCHEMA_PATH.name == "schema.yaml"
    assert schema["classes"]["MachineMapCatalog"]["tree_root"] is True
    forbidden = {"condition", "expression", "code_hook", "transform_code"}
    assert forbidden.isdisjoint(schema["slots"])


def test_mast_maps_validate_and_cover_every_mapped_corpus_channel():
    if not LEVEL2_ROOT.is_dir():
        pytest.skip("FAIR-MAST level-2 mirror is not mounted")
    catalog = load_packaged_machine_map("mast")
    shots, arrays, units = _interferometer_inventory()
    bindings = catalog.binding_sets["mast-interferometer"]

    assert len(shots) == 11_573
    assert (shots[0], shots[-1]) == (11_766, 30_471)
    assert (
        arrays
        == {binding.source_array for binding in bindings}
        == {
            "time",
            "n_e_line",
        }
    )
    assert units == {"time": {"s"}, "n_e_line": {"1 / m ** 2"}}
    assert catalog.bound_channel_count == 2
    assert len(catalog.maps) * catalog.bound_channel_count == 22
    assert catalog.validation_gaps == ()

    for binding in bindings:
        leaf = _dd_leaf(catalog.dd_version, binding.dd_path)
        assert leaf.data_type.name == "FLT"
        assert leaf.units == binding.target_unit
        assert binding.sign_convention == "identity"


def test_every_mast_map_range_is_one_geometry_transition():
    if not LEVEL2_ROOT.is_dir():
        pytest.skip("FAIR-MAST level-2 mirror is not mounted")
    catalog = load_packaged_machine_map("mast")
    shots, _, _ = _interferometer_inventory()
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


def test_diii_d_public_map_is_valid_and_every_binding_is_qualified():
    catalog = load_packaged_machine_map("diii-d")
    machine_map = catalog.maps[0]
    bindings = catalog.bindings_for(machine_map)
    gaps = {gap.binding: gap.reason for gap in catalog.validation_gaps}

    assert catalog.source == "https://github.com/MIT-PSFC/disruption-py"
    assert catalog.source_revision == "dec5c58a3e3970bc6817f33efb615fea11057fce"
    assert machine_map.validation_state == "source-only"
    assert catalog.bound_channel_count == len(bindings) == len(gaps) == 4
    assert gaps.keys() == {binding.name for binding in bindings}
    assert all("No DIII-D pulse corpus" in reason for reason in gaps.values())
    assert all(binding.sign_convention == "unknown-unvalidated" for binding in bindings)
    for binding in bindings:
        leaf = _dd_leaf(catalog.dd_version, binding.dd_path)
        assert leaf.data_type.name == "FLT"
        assert leaf.units == binding.target_unit
        assert "dec5c58a" in binding.evidence


def test_loader_rejects_an_undeclared_conditional_slot(tmp_path):
    source = json.loads((LINKML_SCHEMA_PATH.parent / "mast.json").read_text())
    source["maps"][0]["condition"] = "shot > 12000"
    invalid = tmp_path / "conditional.json"
    invalid.write_text(json.dumps(source))

    with pytest.raises(MachineMapError, match="extra=.+condition"):
        load_machine_map(invalid)
