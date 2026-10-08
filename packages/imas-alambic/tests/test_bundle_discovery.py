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
    bundles_carrying,
    discover_bundles,
    load_bundle_descriptor,
    load_packaged_machine_map,
    resolve_store_root,
)
from imas_alambic.settings import ENV_MAP_PATH

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


def test_bundles_carrying_returns_every_reachable_bundle_that_declares_it(
    tmp_path, monkeypatch
):
    first = _bundle(tmp_path / "a", name="alpha", machines=["shared"])
    second = _bundle(tmp_path / "b", name="beta", machines=["shared"])
    other = _bundle(tmp_path / "c", name="gamma", machines=["elsewhere"])
    monkeypatch.setenv(
        ENV_MAP_PATH, f"{first}{os_sep()}{second}{os_sep()}{other}"
    )
    monkeypatch.setattr("imas_alambic.machine_map.entry_points", lambda group: [])

    carrying = bundles_carrying("shared")
    assert {bundle.name for bundle in carrying} == {"alpha", "beta"}
    (only,) = bundles_carrying("elsewhere")
    assert only.name == "gamma"
    assert bundles_carrying("absent") == ()


def test_an_unknown_machine_is_refused(tmp_path, monkeypatch):
    _bundle(tmp_path / "synth", name="synth", machines=["synth-machine"])
    monkeypatch.setenv("IMAS_ALAMBIC_MAP_PATH", str(tmp_path / "synth"))

    with pytest.raises(MachineMapError, match="no bundle carries"):
        bundle_for_machine("absent")


def test_the_unknown_machine_refusal_names_the_map_path_variable(tmp_path, monkeypatch):
    _bundle(tmp_path / "synth", name="synth", machines=["synth-machine"])
    monkeypatch.setenv(ENV_MAP_PATH, str(tmp_path / "synth"))

    with pytest.raises(MachineMapError) as raised:
        bundle_for_machine("absent")

    message = str(raised.value)
    assert "absent" in message
    assert ENV_MAP_PATH in message


def test_a_map_path_that_does_not_carry_jt60sa_is_refused_naming_both(
    tmp_path, monkeypatch
):
    # A named directory must hold a bundle.json, so the path names a valid
    # bundle that simply does not carry the machine; that is what reaches
    # bundle_for_machine's refusal.
    _bundle(tmp_path / "synth", name="synth", machines=["synth-machine"])
    monkeypatch.setenv(ENV_MAP_PATH, str(tmp_path / "synth"))
    monkeypatch.delenv("IMAS_ALAMBIC_HOME", raising=False)
    monkeypatch.setattr("imas_alambic.machine_map.entry_points", lambda group: [])

    with pytest.raises(MachineMapError) as raised:
        load_packaged_machine_map("jt-60sa")

    message = str(raised.value)
    assert "jt-60sa" in message
    assert ENV_MAP_PATH in message


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


def test_an_entry_point_may_supply_several_bundles(tmp_path, monkeypatch):
    """A target that returns an iterable publishes a ready bundle and a directory."""

    ready_root = _bundle(tmp_path / "ready", name="ready", machines=["ready-machine"])
    ready = load_bundle_descriptor(ready_root)
    directory = _bundle(
        tmp_path / "directory", name="directory", machines=["dir-machine"]
    )
    monkeypatch.delenv(ENV_MAP_PATH, raising=False)
    monkeypatch.delenv("IMAS_ALAMBIC_HOME", raising=False)
    monkeypatch.setattr(
        "imas_alambic.machine_map.entry_points",
        lambda group: [_FakeEntryPoint([ready, directory])],
    )

    bundles = discover_bundles()

    assert [bundle.name for bundle in bundles] == ["ready", "directory"]
    assert bundles[0] is ready
    assert bundles[1].root == directory


class _FakeEntryPoint:
    def __init__(self, target):
        self._target = target

    def load(self):
        return self._target


def os_sep() -> str:
    import os

    return os.pathsep
