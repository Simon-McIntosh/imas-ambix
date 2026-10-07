"""Settings resolve at flag, then variable, then home, each with its source.

``resolve_settings`` is the one reader of the ``IMAS_ALAMBIC_*`` variables, so
these tests inject the environment instead of mutating it and pin the
precedence, the home-derived defaults, the cache fallback, the machine
inference and the refusal that names the variable to set.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from imas_alambic.machine_map import MachineMapError
from imas_alambic.settings import (
    ENV_CACHE,
    ENV_HOME,
    ENV_IDS_ROOT,
    ENV_MAP_PATH,
    SettingsFlags,
    require_setting,
    resolve_settings,
)

if TYPE_CHECKING:
    from pathlib import Path


def _no_entry_points(monkeypatch) -> None:
    monkeypatch.setattr("imas_alambic.machine_map.entry_points", lambda group: [])


def _bundle(root: Path, *, machine: str, store_roots: dict | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "bundle.json").write_text(
        json.dumps(
            {
                "name": "facility",
                "version": "1.0.0",
                "machine": machine,
                "store_roots": store_roots or {},
            }
        )
    )
    return root


def test_home_derives_maps_and_ids_and_infers_the_machine(tmp_path, monkeypatch):
    _no_entry_points(monkeypatch)
    home = tmp_path / "IMASDB"
    _bundle(home / "maps" / "current", machine="jt-60sa")

    settings = resolve_settings(SettingsFlags(), {ENV_HOME: str(home)})

    assert settings.home.value == home
    assert settings.home.source == ENV_HOME
    assert settings.maps.value == home / "maps" / "current"
    assert settings.maps.source == f"{ENV_HOME}/maps/current"
    assert settings.ids_root.value == home / "ids"
    assert settings.ids_root.source == f"{ENV_HOME}/ids"
    assert settings.machine.value == "jt-60sa"
    assert "inferred" in settings.machine.source


def test_a_flag_beats_the_variable_and_the_variable_beats_the_home(
    tmp_path, monkeypatch
):
    _no_entry_points(monkeypatch)
    # Both map paths must be real bundles, because the cache role and the
    # machine inference search them.
    _bundle(tmp_path / "flag-map", machine="user")
    _bundle(tmp_path / "env-map", machine="user")
    environ = {
        ENV_HOME: str(tmp_path / "home"),
        ENV_MAP_PATH: str(tmp_path / "env-map"),
        ENV_IDS_ROOT: str(tmp_path / "env-ids"),
        ENV_CACHE: str(tmp_path / "env-cache"),
    }

    flagged = resolve_settings(
        SettingsFlags(
            maps=str(tmp_path / "flag-map"),
            ids_root=str(tmp_path / "flag-ids"),
            cache=str(tmp_path / "flag-cache"),
            machine="flagged",
        ),
        environ,
    )
    assert flagged.maps.value == str(tmp_path / "flag-map")
    assert flagged.maps.source == "--maps"
    assert flagged.ids_root.source == "--out"
    assert flagged.cache.source == "--cache"
    assert flagged.machine.value == "flagged"
    assert flagged.machine.source == "--machine"

    varied = resolve_settings(SettingsFlags(), environ)
    assert varied.maps.value == str(tmp_path / "env-map")
    assert varied.maps.source == ENV_MAP_PATH
    assert varied.ids_root.source == ENV_IDS_ROOT
    assert varied.cache.value == str(tmp_path / "env-cache")
    assert varied.cache.source == ENV_CACHE


def test_cache_falls_back_from_the_bundle_role_to_the_user_default(
    tmp_path, monkeypatch
):
    _no_entry_points(monkeypatch)
    root = _bundle(
        tmp_path / "facility",
        machine="jt-60sa",
        store_roots={"eddb_cache": str(tmp_path / "bundle-cache")},
    )
    settings = resolve_settings(SettingsFlags(), {ENV_MAP_PATH: str(root)})
    assert settings.cache.value == tmp_path / "bundle-cache"
    assert "eddb_cache" in settings.cache.source

    bare = _bundle(tmp_path / "bare", machine="jt-60sa")
    fallback = resolve_settings(SettingsFlags(), {ENV_MAP_PATH: str(bare)})
    assert str(fallback.cache.value).endswith(".cache/imas-alambic")
    assert "default" in fallback.cache.source


def test_several_reachable_machines_leaves_the_machine_unset_naming_the_flag(
    tmp_path, monkeypatch
):
    _no_entry_points(monkeypatch)
    first = _bundle(tmp_path / "one", machine="alpha")
    second = _bundle(tmp_path / "two", machine="beta")
    settings = resolve_settings(SettingsFlags(), {ENV_MAP_PATH: f"{first}:{second}"})
    assert settings.machine.value is None
    assert "--machine" in settings.machine.source
    assert "alpha" in settings.machine.source and "beta" in settings.machine.source


def test_require_setting_names_the_variable_that_sets_it(monkeypatch):
    _no_entry_points(monkeypatch)
    settings = resolve_settings(SettingsFlags(), {})
    with pytest.raises(MachineMapError) as raised:
        require_setting("ids_root", settings.ids_root, f"{ENV_IDS_ROOT} or {ENV_HOME}")
    message = str(raised.value)
    assert "ids_root" in message
    assert ENV_IDS_ROOT in message and ENV_HOME in message
