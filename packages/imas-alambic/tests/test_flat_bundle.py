"""A bundle declares its machine once and selects its addressing by that key.

A flat single-machine bundle names its one machine with the singular ``machine``
key in ``bundle.json`` and holds its files directly under the bundle; a
multi-machine bundle names a ``machines`` list and keeps a machine-name level
inside the tree.  These tests pin both addressings, the refusal of a bundle that
names its machines neither way or both ways, and resolution of a flat bundle's
catalogue, signal maps and description store by machine name.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from imas_alambic.machine_map import (
    MachineMapError,
    bundle_for_machine,
    load_bundle_descriptor,
    load_packaged_machine_map,
    resolve_store_root,
)
from imas_alambic.signal_map import (
    MAP_SCHEMA_VERSION,
    SignalMap,
    SignalRule,
    load_packaged_signal_map,
)

_SYSTEMS = ("pf_active", "magnetics")


def _signal() -> SignalRule:
    return SignalRule(
        semantic_id="plasma_current",
        source_group="amc",
        source_array="plasma_current",
        source_unit="kA",
        target_path="magnetics/ip/data",
        target_unit="A",
        target_index=0,
        transformation="ip_like",
        source_cocos=3,
        unit_factor=1000.0,
        channel_factor=1.0,
        standard_name=None,
        evidence="receipt sha256:synthetic",
        validation_state="source-only",
    )


def _signal_payload(machine: str, system: str) -> dict:
    return SignalMap.create(
        schema_version=MAP_SCHEMA_VERSION,
        set_version="0.1.0",
        machine=machine,
        system=system,
        source_dataset="synthetic",
        target_dd_version="4.1.1",
        target_cocos=17,
        discovery_producer="imas-codex",
        discovery_receipt="sha256:discovery",
        signals=(_signal(),),
        calibrations=(),
        blocked=(),
    ).as_dict()


def _catalog_payload(machine: str, description_role: str) -> dict:
    binding = {
        "name": "ip",
        "source_group": "amc",
        "source_array": "plasma_current",
        "source_rank": 0,
        "source_role": "value",
        "source_location": "file:///synthetic/amc",
        "dd_path": "magnetics/ip/data",
        "source_unit": "kA",
        "target_unit": "A",
        "sign_convention": "identity",
        "evidence": "receipt sha256:synthetic",
    }
    return {
        "schema_version": "1.0.0",
        "dd_version": "4.1.1",
        "source": "synthetic",
        "source_revision": "0",
        "source_cocos": 0,
        "description_store_format": "netcdf",
        "description_store_root": description_role,
        "description_store_layout": "per-shot",
        "probe_angle_source": "description",
        "binding_sets": [{"name": "bs", "bindings": [binding]}],
        "maps": [
            {
                "name": "map",
                "machine": machine,
                "first_shot": 0,
                "last_shot": 100,
                "transition": None,
                "binding_set": "bs",
                "drive_topology": None,
                "description_supplement": None,
                "validation_state": "corpus-validated",
            }
        ],
        "validation_gaps": [],
        "source_qualifications": [],
        "sensor_identity_rules": [],
        "identity_qualifications": [],
        "flux_loop_position_declarations": [],
        "circuit_current_joins": [],
        "drive_topologies": [],
        "structure_assemblies": [],
        "acquisition_declarations": [],
        "description_supplements": [],
    }


def _write_bundle(root: Path, payload: dict) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "bundle.json").write_text(json.dumps(payload))
    return root


def _write_flat_bundle(root: Path, machine: Path | str) -> Path:
    machine = str(machine)
    _write_bundle(
        root,
        {
            "name": "facility",
            "version": "1.0.0",
            "machine": machine,
            "store_roots": {"description": "machine_description"},
        },
    )
    (root / "machine_map.json").write_text(
        json.dumps(_catalog_payload(machine, "description"))
    )
    (root / "maps").mkdir(exist_ok=True)
    for system in _SYSTEMS:
        (root / "maps" / f"{system}.json").write_text(
            json.dumps(_signal_payload(machine, system))
        )
    (root / "machine_description").mkdir(exist_ok=True)
    return root


def _no_entry_points(monkeypatch) -> None:
    monkeypatch.setattr("imas_alambic.machine_map.entry_points", lambda group: [])


def test_a_flat_bundle_is_addressed_from_its_single_machine_key(tmp_path):
    root = _write_flat_bundle(tmp_path / "facility", "jt-60sa")

    bundle = load_bundle_descriptor(root)

    assert bundle.single_machine is True
    assert bundle.machines == ("jt-60sa",)
    assert bundle.machine_map_path("jt-60sa") == root / "machine_map.json"
    assert bundle.signal_map_path("jt-60sa", "pf_active") == (
        root / "maps" / "pf_active.json"
    )


def test_a_multi_machine_bundle_is_addressed_from_its_machines_list(tmp_path):
    root = _write_bundle(
        tmp_path / "multi",
        {
            "name": "ambix",
            "version": "1.0.0",
            "machines": ["mast", "diii-d", "jt-60sa"],
        },
    )

    bundle = load_bundle_descriptor(root)

    assert bundle.single_machine is False
    assert bundle.machines == ("mast", "diii-d", "jt-60sa")
    assert bundle.machine_map_path("mast") == root / "machine_maps" / "mast.json"
    assert bundle.signal_map_path("mast", "pf_active") == (
        root / "maps" / "mast" / "pf_active.json"
    )


def test_a_bundle_declaring_both_machine_keys_is_refused_naming_the_bundle(tmp_path):
    root = _write_bundle(
        tmp_path / "both",
        {"name": "conflict", "machine": "a", "machines": ["a"]},
    )

    with pytest.raises(MachineMapError) as raised:
        load_bundle_descriptor(root)

    message = str(raised.value)
    assert "conflict" in message
    assert "both" in message


def test_a_bundle_declaring_neither_machine_key_is_refused_naming_the_bundle(tmp_path):
    root = _write_bundle(tmp_path / "neither", {"name": "empty"})

    with pytest.raises(MachineMapError) as raised:
        load_bundle_descriptor(root)

    message = str(raised.value)
    assert "empty" in message
    assert "neither" in message


def test_a_flat_bundle_on_the_map_path_resolves_catalogue_signals_and_store(
    tmp_path, monkeypatch
):
    root = _write_flat_bundle(tmp_path / "facility", "synth-machine")
    monkeypatch.setenv("IMAS_ALAMBIC_MAP_PATH", str(root))
    _no_entry_points(monkeypatch)

    bundle = bundle_for_machine("synth-machine")
    assert bundle.single_machine is True

    catalog = load_packaged_machine_map("synth-machine")
    assert catalog.description_store_root == "description"
    assert catalog.description_store_root_path() == root / "machine_description"
    assert resolve_store_root("description") == root / "machine_description"

    for system in _SYSTEMS:
        signal_map = load_packaged_signal_map("synth-machine", system)
        assert (signal_map.machine, signal_map.system) == ("synth-machine", system)


def test_the_duplicate_machine_refusal_holds_across_flat_and_multi_machine(
    tmp_path, monkeypatch
):
    flat = _write_flat_bundle(tmp_path / "flat", "shared")
    multi = _write_bundle(tmp_path / "multi", {"name": "ambix", "machines": ["shared"]})
    monkeypatch.setenv("IMAS_ALAMBIC_MAP_PATH", f"{flat}:{multi}")
    _no_entry_points(monkeypatch)

    with pytest.raises(MachineMapError) as raised:
        bundle_for_machine("shared")

    message = str(raised.value)
    assert "facility" in message and "ambix" in message
