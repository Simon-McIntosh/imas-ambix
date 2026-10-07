"""Settings resolve at flag, then variable, then home, each with its source.

``resolve_settings`` is the one reader of the ``IMAS_ALAMBIC_*`` variables, so
these tests inject the environment instead of mutating it and pin the
precedence, the home-derived defaults, the cache fallback, the machine
inference and the refusal that names the variable to set.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from imas_alambic.eddb_remote import (
    DEFAULT_SSH_COMMAND,
    extractor_for_host,
    ssh_command_for_host,
)
from imas_alambic.machine_map import MachineMapError
from imas_alambic.settings import (
    ENV_CACHE,
    ENV_EDDB_HOST,
    ENV_HOME,
    ENV_IDS_ROOT,
    ENV_MAP_PATH,
    SettingsFlags,
    require_setting,
    resolve_settings,
)


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


def test_ids_root_flag_beats_every_earlier_source(tmp_path, monkeypatch):
    _no_entry_points(monkeypatch)
    home = tmp_path / "IMASDB"
    _bundle(
        home / "maps" / "current",
        machine="jt-60sa",
        store_roots={"ids_root": str(tmp_path / "bundle-ids")},
    )
    environ = {ENV_HOME: str(home), ENV_IDS_ROOT: str(tmp_path / "env-ids")}

    flagged = resolve_settings(
        SettingsFlags(ids_root=str(tmp_path / "flag-ids")), environ
    )

    assert flagged.ids_root.value == str(tmp_path / "flag-ids")
    assert flagged.ids_root.source == "--out"


def test_ids_root_variable_beats_home_and_the_bundle_role(tmp_path, monkeypatch):
    _no_entry_points(monkeypatch)
    home = tmp_path / "IMASDB"
    _bundle(
        home / "maps" / "current",
        machine="jt-60sa",
        store_roots={"ids_root": str(tmp_path / "bundle-ids")},
    )

    settings = resolve_settings(
        SettingsFlags(), {ENV_HOME: str(home), ENV_IDS_ROOT: str(tmp_path / "env-ids")}
    )

    assert settings.ids_root.value == str(tmp_path / "env-ids")
    assert settings.ids_root.source == ENV_IDS_ROOT


def test_ids_root_home_beats_the_bundle_role(tmp_path, monkeypatch):
    _no_entry_points(monkeypatch)
    home = tmp_path / "IMASDB"
    _bundle(
        home / "maps" / "current",
        machine="jt-60sa",
        store_roots={"ids_root": str(tmp_path / "bundle-ids")},
    )

    settings = resolve_settings(SettingsFlags(), {ENV_HOME: str(home)})

    assert settings.ids_root.value == home / "ids"
    assert settings.ids_root.source == f"{ENV_HOME}/ids"


def test_ids_root_comes_from_the_bundle_role_when_nothing_else_names_it(
    tmp_path, monkeypatch
):
    _no_entry_points(monkeypatch)
    root = _bundle(
        tmp_path / "facility",
        machine="jt-60sa",
        store_roots={"ids_root": str(tmp_path / "bundle-ids")},
    )

    settings = resolve_settings(SettingsFlags(), {ENV_MAP_PATH: str(root)})

    assert settings.ids_root.value == tmp_path / "bundle-ids"
    assert settings.ids_root.source == "bundle ids_root role"


def test_ids_root_is_unset_when_no_bundle_declares_the_role(tmp_path, monkeypatch):
    _no_entry_points(monkeypatch)
    root = _bundle(tmp_path / "bare", machine="jt-60sa")

    settings = resolve_settings(SettingsFlags(), {ENV_MAP_PATH: str(root)})

    assert settings.ids_root.value is None
    assert not settings.ids_root.is_set


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


def test_every_setting_is_reported_when_no_bundle_is_reachable(tmp_path, monkeypatch):
    _no_entry_points(monkeypatch)
    home = tmp_path / "IMASDB"
    home.mkdir()
    # The home-derived map search path names maps/current, which does not exist
    # yet, so the search-path entry carries no descriptor and bundle discovery
    # refuses.  Config must still report every setting.
    settings = resolve_settings(SettingsFlags(), {ENV_HOME: str(home)})

    assert settings.home.value == home
    assert settings.maps.value == home / "maps" / "current"
    assert settings.ids_root.value == home / "ids"
    assert settings.ids_root.source == f"{ENV_HOME}/ids"

    # The machine and the cache search the bundle, so each carries the refusal
    # as its source with no value rather than raising out of resolve_settings.
    assert settings.machine.value is None
    assert "cannot read bundle descriptor" in settings.machine.source
    assert settings.cache.value is None
    assert "cannot read bundle descriptor" in settings.cache.source

    # A command that needs the machine still refuses while it is unresolved.
    with pytest.raises(MachineMapError):
        require_setting("machine", settings.machine, "--machine")


def test_require_setting_names_the_variable_that_sets_it(monkeypatch):
    _no_entry_points(monkeypatch)
    settings = resolve_settings(SettingsFlags(), {})
    with pytest.raises(MachineMapError) as raised:
        require_setting("ids_root", settings.ids_root, f"{ENV_IDS_ROOT} or {ENV_HOME}")
    message = str(raised.value)
    assert "ids_root" in message
    assert ENV_IDS_ROOT in message and ENV_HOME in message


def test_eddb_host_flag_beats_the_variable(monkeypatch):
    _no_entry_points(monkeypatch)

    flagged = resolve_settings(
        SettingsFlags(eddb_host="flag-host"), {ENV_EDDB_HOST: "env-host"}
    )
    assert flagged.eddb_host.value == "flag-host"
    assert flagged.eddb_host.source == "--eddb-host"

    varied = resolve_settings(SettingsFlags(), {ENV_EDDB_HOST: "env-host"})
    assert varied.eddb_host.value == "env-host"
    assert varied.eddb_host.source == ENV_EDDB_HOST


def test_eddb_host_unset_keeps_the_jt60sa_ssh_route(monkeypatch):
    _no_entry_points(monkeypatch)

    settings = resolve_settings(SettingsFlags(), {})

    assert settings.eddb_host.value is None
    assert "jt-60sa" in settings.eddb_host.source
    assert ssh_command_for_host("jt-60sa") == DEFAULT_SSH_COMMAND
    route = extractor_for_host(settings.eddb_host.value).ssh_command
    assert route == DEFAULT_SSH_COMMAND


def test_eddb_host_local_gives_the_local_transport_with_the_engine_interpreter(
    monkeypatch,
):
    _no_entry_points(monkeypatch)

    settings = resolve_settings(SettingsFlags(eddb_host="local"), {})
    assert settings.eddb_host.value == "local"

    extractor = extractor_for_host("local")
    assert extractor.ssh_command == ()
    assert extractor.remote_python == sys.executable


def test_eddb_host_composes_the_ssh_route_keeping_config_and_module_shell(monkeypatch):
    _no_entry_points(monkeypatch)

    settings = resolve_settings(SettingsFlags(eddb_host="a-host"), {})
    assert settings.eddb_host.value == "a-host"

    extractor = extractor_for_host("a-host")
    assert extractor.ssh_command == ("ssh", "-F", "~/.ssh/config", "a-host")

    argv = extractor._argv()
    assert argv[:4] == ["ssh", "-F", os.path.expanduser("~/.ssh/config"), "a-host"]
    joined = " ".join(argv)
    assert "module unload" in joined and "module load" in joined


def test_config_reports_eddb_host_with_its_source(monkeypatch):
    _no_entry_points(monkeypatch)
    monkeypatch.setenv(ENV_EDDB_HOST, "local")

    from click.testing import CliRunner

    from imas_alambic.cli import main

    result = CliRunner().invoke(main, ["config"])

    assert result.exit_code == 0, result.output
    assert f"eddb_host: local  [{ENV_EDDB_HOST}]" in result.output


def test_the_facility_launcher_selects_the_local_eddb_transport():
    script = Path(__file__).resolve().parents[1] / "install" / "imasdb.sh"
    launcher = script.read_text().split("<<'LAUNCHER'", 1)[1].split("\nLAUNCHER", 1)[0]

    assert "IMAS_ALAMBIC_EDDB_HOST=local" in launcher
    assert "export IMAS_ALAMBIC_EDDB_HOST" in launcher
    assert "IMAS_ALAMBIC_HOME=$home" in launcher
