"""Bundle discovery on synthetic bundles named in ``IMAS_ALAMBIC_MAP_PATH``."""

from __future__ import annotations

import json
from dataclasses import replace
from types import MappingProxyType
from typing import TYPE_CHECKING

import pytest

from imas_alambic.machine_map import (
    MachineMapError,
    bundle_for_machine,
    discover_bundles,
    load_bundle_descriptor,
    resolve_store_root,
)

if TYPE_CHECKING:
    from pathlib import Path


def _bundle(root: Path, *, name: str, machines: list[str], store_roots=None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "bundle.json").write_text(
        json.dumps(
            {
                "name": name,
                "version": "1.0.0",
                "machines": machines,
                "store_roots": store_roots or {},
            }
        )
    )
    return root


def test_a_synthetic_bundle_on_the_map_path_resolves_by_name(tmp_path, monkeypatch):
    _bundle(tmp_path / "synth", name="synth", machines=["synth-machine"])
    monkeypatch.setenv("IMAS_ALAMBIC_MAP_PATH", str(tmp_path / "synth"))

    machine = bundle_for_machine("synth-machine")

    assert machine.name == "synth"
    assert machine.machine_map_path("synth-machine") == (
        tmp_path / "synth" / "machine_maps" / "synth-machine.json"
    )


def test_the_same_directory_reached_twice_is_one_bundle(tmp_path, monkeypatch):
    bundle = _bundle(tmp_path / "synth", name="synth", machines=["synth-machine"])
    monkeypatch.setenv("IMAS_ALAMBIC_MAP_PATH", f"{bundle}{os_sep()}{bundle}")
    monkeypatch.setattr("imas_alambic.machine_map.entry_points", lambda group: [])

    assert len(discover_bundles()) == 1


def test_a_machine_in_two_bundles_is_refused_naming_both(tmp_path, monkeypatch):
    first = _bundle(tmp_path / "a", name="alpha", machines=["shared"])
    second = _bundle(tmp_path / "b", name="beta", machines=["shared"])
    monkeypatch.setenv("IMAS_ALAMBIC_MAP_PATH", f"{first}{os_sep()}{second}")

    with pytest.raises(MachineMapError) as raised:
        bundle_for_machine("shared")

    message = str(raised.value)
    assert "alpha" in message and "beta" in message


def test_an_unknown_machine_is_refused(tmp_path, monkeypatch):
    _bundle(tmp_path / "synth", name="synth", machines=["synth-machine"])
    monkeypatch.setenv("IMAS_ALAMBIC_MAP_PATH", str(tmp_path / "synth"))

    with pytest.raises(MachineMapError, match="no bundle carries"):
        bundle_for_machine("absent")


def test_relative_store_roots_resolve_under_the_bundle(tmp_path, monkeypatch):
    _bundle(
        tmp_path / "synth",
        name="synth",
        machines=[],
        store_roots={"REL": "machines", "ABS": str(tmp_path / "abs")},
    )
    monkeypatch.setenv("IMAS_ALAMBIC_MAP_PATH", str(tmp_path / "synth"))

    assert resolve_store_root("REL") == tmp_path / "synth" / "machines"
    assert resolve_store_root("ABS") == tmp_path / "abs"


def test_an_entry_point_may_supply_a_ready_bundle(tmp_path, monkeypatch):
    root = _bundle(tmp_path / "synth", name="synth", machines=["synth-machine"])
    declared = replace(
        load_bundle_descriptor(root),
        store_roots=MappingProxyType({"EXT": tmp_path / "ext"}),
    )
    monkeypatch.delenv("IMAS_ALAMBIC_MAP_PATH", raising=False)
    monkeypatch.delenv("IMAS_ALAMBIC_HOME", raising=False)
    monkeypatch.setattr(
        "imas_alambic.machine_map.entry_points",
        lambda group: [_FakeEntryPoint(declared)],
    )

    (bundle,) = discover_bundles()

    assert bundle is declared
    assert resolve_store_root("EXT") == tmp_path / "ext"


def test_a_declared_bundle_wins_over_the_same_directory_on_the_map_path(
    tmp_path, monkeypatch
):
    root = _bundle(
        tmp_path / "synth",
        name="synth",
        machines=["synth-machine"],
        store_roots={"JSON": "json-root"},
    )
    declared = replace(
        load_bundle_descriptor(root),
        store_roots=MappingProxyType({"EXT": tmp_path / "ext"}),
    )
    monkeypatch.setenv("IMAS_ALAMBIC_MAP_PATH", str(root))
    monkeypatch.setattr(
        "imas_alambic.machine_map.entry_points",
        lambda group: [_FakeEntryPoint(declared)],
    )

    assert len(discover_bundles()) == 1
    assert resolve_store_root("EXT") == tmp_path / "ext"


class _FakeEntryPoint:
    def __init__(self, target):
        self._target = target

    def load(self):
        return self._target


def os_sep() -> str:
    import os

    return os.pathsep
