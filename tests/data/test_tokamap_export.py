"""Tests for the machine-map to tokamap directory export."""

from __future__ import annotations

import dataclasses
import json
import math
import shutil
import subprocess
from pathlib import Path

import imas
import pytest

from imas_ambix.cocos import canonical_factor
from imas_ambix.data.machine_map import load_packaged_machine_map
from imas_ambix.data.signal_map import load_packaged_signal_map
from imas_ambix.data.tokamap_export import (
    PARTITION_ATTRIBUTE,
    PARTITION_SELECTOR,
    export_tokamap_directory,
)

MACHINE = "mast"
SIGNAL_SYSTEM = "magnetics"

# Independent source-to-target unit factors: the exporter must agree with
# these without consulting its own table.
_EXPECTED_UNITS = {
    ("degree", "rad"): math.pi / 180.0,
    ("rad", "degree"): 180.0 / math.pi,
}
_EXPECTED_SIGNS = {
    "identity": 1.0,
    "not-applicable": 1.0,
    "negate": -1.0,
    "unknown-unvalidated": 1.0,
}


def _expected_unit(source_unit: str, target_unit: str) -> float:
    if source_unit == target_unit:
        return 1.0
    return _EXPECTED_UNITS[(source_unit.strip().lower(), target_unit.strip().lower())]


def _transformation(dd_path: str) -> str | None:
    ids_name, relative_path = dd_path.split("/", maxsplit=1)
    metadata = imas.IDSFactory("4.1.1").new(ids_name).metadata
    components = relative_path.split("/")
    for size in range(len(components), 0, -1):
        node = metadata["/".join(components[:size])]
        transformation = getattr(node, "cocos_label_transformation", None)
        if transformation:
            return str(transformation)
    return None


@pytest.fixture(scope="module")
def catalog():
    return load_packaged_machine_map(MACHINE)


@pytest.fixture(scope="module")
def signal_map():
    return load_packaged_signal_map(MACHINE, SIGNAL_SYSTEM)


def _bindings_by_name(cat) -> dict:
    index = {}
    for machine_map in cat.maps:
        for binding in cat.bindings_for(machine_map):
            index.setdefault(binding.name, binding)
    return index


def _validator_command(directory: Path) -> list[str]:
    executable = shutil.which("tokamap-validator")
    if executable is not None:
        return [executable, str(directory)]
    return [
        "python3",
        "-c",
        "from tokamap.validator.main import run; run()",
        str(directory),
    ]


def test_export_passes_validator_for_every_bound_ids_group(
    catalog, signal_map, tmp_path
):
    directory = tmp_path / "export"
    result = export_tokamap_directory(
        catalog, [signal_map], directory=directory
    )

    bound_groups = sorted(
        {
            binding.dd_path.split("/", maxsplit=1)[0]
            for machine_map in catalog.maps
            for binding in catalog.bindings_for(machine_map)
        }
    )
    assert result.groups == tuple(bound_groups)
    for group in bound_groups:
        assert (directory / group).is_dir(), group

    completed = subprocess.run(
        _validator_command(directory),
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Validation completed successfully." in completed.stdout


def _assert_scale_round_trip(result, cat) -> int:
    index = _bindings_by_name(cat)
    catalogue_entries = 0
    for entry in result.entries:
        if entry.kind != "catalogue":
            continue
        binding = index[entry.source_name]
        catalogue_entries += 1
        unit = _expected_unit(binding.source_unit, binding.target_unit)
        sign = _EXPECTED_SIGNS[binding.sign_convention]
        transformation = _transformation(binding.dd_path)
        cocos = (
            canonical_factor(
                transformation,
                source_cocos=cat.cocos_for_binding(binding),
            )
            if transformation is not None
            else 1.0
        )
        assert entry.scale == pytest.approx(unit * sign * cocos, abs=1e-12)
    return catalogue_entries


def test_exported_scale_round_trips_unit_sign_and_cocos(catalog, signal_map, tmp_path):
    directory = tmp_path / "export"
    result = export_tokamap_directory(catalog, [signal_map], directory=directory)
    assert _assert_scale_round_trip(result, catalog) > 0

    # A declared source COCOS that makes the collapse numerically visible:
    # COCOS 6 to the canonical COCOS 17 negates an ip_like target, so dropping
    # the COCOS factor from the exported scale cannot pass unnoticed.
    even_catalog = dataclasses.replace(catalog, source_cocos=6)
    even_directory = tmp_path / "even"
    even_result = export_tokamap_directory(
        even_catalog, [signal_map], directory=even_directory
    )
    assert _assert_scale_round_trip(even_result, even_catalog) > 0

    by_path = {
        entry.dd_path: entry
        for entry in even_result.entries
        if entry.kind == "catalogue"
    }
    current = by_path["pf_active/coil/current/data"]
    assert current.cocos_factor == pytest.approx(-1.0, abs=1e-12)
    assert current.scale == pytest.approx(-1.0, abs=1e-12)

    # The degree to radian bindings carry the unit factor on disk as well.
    angle = by_path["pf_passive/loop/element/geometry/oblique/alpha"]
    assert angle.scale == pytest.approx(math.pi / 180.0, abs=1e-12)


def test_shot_partitions_follow_catalogue_ranges(catalog, signal_map, tmp_path):
    directory = tmp_path / "export"
    result = export_tokamap_directory(catalog, [signal_map], directory=directory)

    expected = sorted(machine_map.first_shot for machine_map in catalog.maps)
    assert result.partitions == tuple(expected)

    config = json.loads((directory / "mappings.cfg.json").read_text())
    assert config["partitions"] == [
        {"attribute": PARTITION_ATTRIBUTE, "selector": PARTITION_SELECTOR}
    ]
    for group in result.groups:
        assert sorted(int(entry.name) for entry in (directory / group).iterdir()) == (
            expected
        )
        for first_shot in expected:
            leaf = directory / group / str(first_shot)
            assert (leaf / "globals.json").is_file()
            assert (leaf / "mappings.json").is_file()


def test_unknown_unvalidated_sign_is_recorded_not_silently_signed(catalog, tmp_path):
    binding_set_name, bindings = next(iter(catalog.binding_sets.items()))
    target = bindings[0]
    rewritten = tuple(
        dataclasses.replace(binding, sign_convention="unknown-unvalidated")
        if binding.name == target.name
        else binding
        for binding in bindings
    )
    assert rewritten[0].sign_convention == "unknown-unvalidated"
    unvalidated = dataclasses.replace(
        catalog, binding_sets={binding_set_name: rewritten}
    )

    directory = tmp_path / "unvalidated"
    result = export_tokamap_directory(unvalidated, directory=directory)

    entry = next(
        item for item in result.entries if item.source_name == target.name
    )
    assert entry.sign_factor == 1.0
    assert "unknown-unvalidated" in entry.comment
    assert "no sign applied" in entry.comment

    written = json.loads(
        (directory / entry.group / str(entry.partition) / "mappings.json").read_text()
    )
    assert "unknown-unvalidated" in written[entry.key]["comment"]
    assert written[entry.key]["scale"] == pytest.approx(
        entry.unit_factor * entry.cocos_factor, abs=1e-12
    )


def test_each_group_file_holds_only_that_ids(catalog, signal_map, tmp_path):
    directory = tmp_path / "export"
    result = export_tokamap_directory(catalog, [signal_map], directory=directory)

    # A key is relative to its group, so membership is checked through the
    # record's Data Dictionary path, not through the key.
    paths_by_group: dict[str, dict[str, str]] = {group: {} for group in result.groups}
    for entry in result.entries:
        assert entry.dd_path.startswith(f"{entry.group}/"), (
            entry.group,
            entry.dd_path,
        )
        paths_by_group[entry.group].setdefault(entry.key, entry.dd_path)

    on_disk_total = 0
    for group in result.groups:
        for first_shot in result.partitions:
            leaf = directory / group / str(first_shot) / "mappings.json"
            mappings = json.loads(leaf.read_text())
            on_disk_total += len(mappings)
            for key in mappings:
                assert key in paths_by_group[group], (group, key)
                source_path = paths_by_group[group][key]
                assert source_path.startswith(f"{group}/"), (group, key, source_path)

    for left in result.groups:
        for right in result.groups:
            if left < right:
                shared = set(paths_by_group[left].values()) & set(
                    paths_by_group[right].values()
                )
                assert not shared, (left, right, shared)

    assert sum(result.group_entry_counts.values()) == len(result.entries)
    assert on_disk_total == len(result.entries)
    assert result.mappings_file_count == len(result.groups) * len(result.partitions)
    for group in result.groups:
        assert result.group_entry_counts[group] == len(paths_by_group[group]) * len(
            result.partitions
        )
    assert len(result.entries) > 0


def test_group_keys_are_relative_and_verbatim(catalog, signal_map, tmp_path):
    directory = tmp_path / "export"
    export_tokamap_directory(catalog, [signal_map], directory=directory)

    first_shot = sorted(machine_map.first_shot for machine_map in catalog.maps)[0]
    mappings = json.loads(
        (directory / "magnetics" / str(first_shot) / "mappings.json").read_text()
    )
    # A known MAST key appears exactly as the reference mappings write it: the
    # structure array marked, and the IDS group not repeated in the key.
    assert "flux_loop[#]/name" in mappings
    assert not any(key.startswith("magnetics/") for key in mappings)
