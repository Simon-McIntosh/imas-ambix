"""Resolve every imas-alambic setting from its flag, variable and home default.

One owner for where the engine reads and writes: the map search path, the IDS
root, the writer's cache and the machine.  ``resolve_settings`` is the only
reader of the ``IMAS_ALAMBIC_*`` variables.  A setting resolves at the highest
precedence that names it:

1. a command-line flag,
2. its ``IMAS_ALAMBIC_*`` variable,
3. a default derived from ``IMAS_ALAMBIC_HOME`` -- ``maps/current`` for the map
   search path and ``ids`` for the IDS root.

The cache has one more step than the others.  It falls back, so a bundle that
declares an ``eddb_cache`` store root keeps the development cache where it
already is, to ``~/.cache/imas-alambic``.

Every setting carries the source it came from, so ``imas-alambic config`` can
print each value beside where it was read.  The machine is never a variable: it
is inferred when exactly one machine is reachable through the bundles, and a
command that needs one and finds several is refused unless ``--machine`` names
it.  No function here mutates the environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from imas_alambic.machine_map import (
    MachineMapError,
    discover_bundles,
    resolve_store_root,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

#: The environment variables this module is the sole reader of.
ENV_HOME = "IMAS_ALAMBIC_HOME"
ENV_MAP_PATH = "IMAS_ALAMBIC_MAP_PATH"
ENV_IDS_ROOT = "IMAS_ALAMBIC_IDS_ROOT"
ENV_CACHE = "IMAS_ALAMBIC_CACHE"

#: The map search path variable, under the name bundle discovery knows it by.
_BUNDLE_ENV_VAR = ENV_MAP_PATH

#: The store role a bundle declares for the writer's on-demand EDDB cache, and
#: the per-user fallback when no reachable bundle declares one.
_CACHE_ROLE = "eddb_cache"
_CACHE_FALLBACK = Path("~/.cache/imas-alambic")


@dataclass(frozen=True)
class Setting:
    """One resolved setting: its value, or ``None``, and where it came from."""

    value: str | Path | None
    source: str

    @property
    def is_set(self) -> bool:
        """Whether the setting resolved to a value."""

        return self.value is not None


@dataclass(frozen=True)
class SettingsFlags:
    """The command-line flags that can set a setting, each unset by default."""

    maps: str | None = None
    ids_root: str | None = None
    cache: str | None = None
    machine: str | None = None


@dataclass(frozen=True)
class ResolvedSettings:
    """Every setting, each with the source it resolved from."""

    home: Setting
    maps: Setting
    ids_root: Setting
    cache: Setting
    machine: Setting

    def as_pairs(self) -> tuple[tuple[str, Setting], ...]:
        """Return ``(name, setting)`` pairs in the order ``config`` prints."""

        return (
            ("home", self.home),
            ("maps", self.maps),
            ("ids_root", self.ids_root),
            ("cache", self.cache),
            ("machine", self.machine),
        )


def require_setting(name: str, setting: Setting, variable: str) -> str:
    """Return a setting's value as text, refusing a missing one by name.

    A missing setting names the variable or flag that would set it, so the
    refusal says which thing to provide rather than only that something is
    absent.
    """

    if setting.value is None:
        raise MachineMapError(f"setting {name!r} is not set; set {variable}")
    return str(setting.value)


def _home(environ: Mapping[str, str]) -> Setting:
    raw = environ.get(ENV_HOME)
    if raw:
        return Setting(Path(raw), ENV_HOME)
    return Setting(None, f"{ENV_HOME} (unset)")


def _maps(flags: SettingsFlags, environ: Mapping[str, str], home: Setting) -> Setting:
    if flags.maps is not None:
        return Setting(flags.maps, "--maps")
    raw = environ.get(ENV_MAP_PATH)
    if raw:
        return Setting(raw, ENV_MAP_PATH)
    if home.value is not None:
        return Setting(
            Path(home.value) / "maps" / "current", f"{ENV_HOME}/maps/current"
        )
    return Setting(None, f"{ENV_HOME} (unset)")


def _ids_root(
    flags: SettingsFlags, environ: Mapping[str, str], home: Setting
) -> Setting:
    if flags.ids_root is not None:
        return Setting(flags.ids_root, "--out")
    raw = environ.get(ENV_IDS_ROOT)
    if raw:
        return Setting(raw, ENV_IDS_ROOT)
    if home.value is not None:
        return Setting(Path(home.value) / "ids", f"{ENV_HOME}/ids")
    return Setting(None, f"{ENV_HOME} (unset)")


def _bundles(search_path: object):
    # An explicit empty tuple is a search path that names no directory, which
    # stops the machine inference from asking resolve_settings for it again.
    return discover_bundles(() if search_path is None else search_path)


def _cache_root_from_bundle(maps: Setting) -> Path | None:
    return resolve_store_root(
        _CACHE_ROLE,
        kind="cache",
        optional=True,
        search_path=() if maps.value is None else maps.value,
    )


def _cache(flags: SettingsFlags, environ: Mapping[str, str], maps: Setting) -> Setting:
    if flags.cache is not None:
        return Setting(flags.cache, "--cache")
    raw = environ.get(ENV_CACHE)
    if raw:
        return Setting(raw, ENV_CACHE)
    declared = _cache_root_from_bundle(maps)
    if declared is not None:
        return Setting(declared, f"bundle {_CACHE_ROLE} role")
    return Setting(_CACHE_FALLBACK.expanduser(), f"default {_CACHE_FALLBACK}")


def _machine(flags: SettingsFlags, maps: Setting) -> Setting:
    if flags.machine is not None:
        return Setting(flags.machine, "--machine")
    reachable = sorted(
        {machine for bundle in _bundles(maps.value) for machine in bundle.machines}
    )
    if len(reachable) == 1:
        return Setting(reachable[0], "inferred from the single reachable bundle")
    if not reachable:
        return Setting(None, "no machine is reachable through the bundles")
    return Setting(
        None,
        f"{len(reachable)} machines are reachable ({', '.join(reachable)}); "
        "pass --machine",
    )


def resolve_settings(
    flags: SettingsFlags | None = None,
    environ: Mapping[str, str] | None = None,
) -> ResolvedSettings:
    """Resolve every setting from its flag, its variable and its home default.

    ``environ`` defaults to the process environment and is injectable so a test
    can pin a resolution without touching the environment.  Nothing here raises
    on a missing setting: ``config`` must be able to report what is unset, so a
    command that needs a value refuses it at the point of use.
    """

    flags = SettingsFlags() if flags is None else flags
    environ = os.environ if environ is None else environ
    home = _home(environ)
    maps = _maps(flags, environ, home)
    return ResolvedSettings(
        home=home,
        ids_root=_ids_root(flags, environ, home),
        maps=maps,
        cache=_cache(flags, environ, maps),
        machine=_machine(flags, maps),
    )


__all__ = [
    "ENV_CACHE",
    "ENV_HOME",
    "ENV_IDS_ROOT",
    "ENV_MAP_PATH",
    "ResolvedSettings",
    "Setting",
    "SettingsFlags",
    "require_setting",
    "resolve_settings",
]
