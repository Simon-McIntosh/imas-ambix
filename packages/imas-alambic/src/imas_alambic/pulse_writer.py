"""Join the static machine description and the dynamic signal maps into IDSs.

The static half of a JT-60SA pulse is the machine description the catalogue
selects for the shot's phase, written whole by the converter as one DD netCDF
file per IDS.  The dynamic half is the compiled signal map, which turns raw
EDDB channels into canonical values in the unit, sign and convention the
target path leaves need.  Nothing joined them before: the two halves were read
separately and a reconstruction wanted both.

This module is that join.  :func:`write_pulse` takes the machine, the shot, a
run and an output root, reads the phase description whole, writes into it the
time-dependent leaves the signal maps serve, and writes one DD netCDF file
holding every description IDS at :func:`pulse_path`.  It returns a
:class:`WriteReceipt` naming each IDS written, each time-dependent leaf filled,
and each declared signal left out with its reason.  One public rule owns the
layout: :func:`pulse_path` returns ``{out_root}/{pulse}_{run}.nc``, and every
reader builds the name through it.

The static content is the converter's own output, never re-derived.  Each
signal's values come from the compiled map through
:meth:`VirtualZarrView.for_target`, so a unit, sign or COCOS factor is applied
where it is applied today; the time base comes from :func:`read_channel`,
because the view returns values with none.  A signal the maps block is not
written.

``ids_properties.homogeneous_time`` is 1 only when every signal written into
that IDS shares one time base; otherwise it is 0 and each signal carries its
own time vector.  An IDS the map serves nothing into is written unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import imas
import numpy as np
from imas.ids_struct_array import IDSStructArray

from imas_alambic.eddb import eddb_token, normalised_shot, read_channel
from imas_alambic.machine_map import (
    MachineMap,
    MachineMapCatalog,
    bundle_for_machine,
    load_packaged_machine_map,
    map_for_shot,
)
from imas_alambic.settings import SettingsFlags, resolve_settings
from imas_alambic.signal_map import SignalMap, load_signal_map
from imas_alambic.virtual_zarr import VirtualZarrView

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from imas_alambic.signal_map import SignalRule

#: The description IDSs a pulse is written as, in write order.  The three the
#: maps serve receive dynamic signals; the other two are written whole from the
#: phase description with no dynamic leaf.
IDS_NAMES = ("pf_active", "pf_passive", "wall", "magnetics", "tf")

#: Which signal map system feeds which description IDS.
_SYSTEM_FOR_IDS = {"pf_active": "pf_active", "magnetics": "magnetics", "tf": "tf"}

#: The layout and format the catalogue must declare for the writer to read one
#: netCDF file per IDS from a directory per phase.
_STORE_LAYOUT = "static-over-map"
_STORE_FORMAT = "netcdf"


class PulseWriteError(RuntimeError):
    """Raised when a pulse cannot be written from the description and cache."""


@dataclass(frozen=True, order=True)
class WrittenLeaf:
    """One time-dependent leaf filled from a signal map."""

    ids: str
    semantic_id: str
    target_path: str
    time_path: str
    samples: int


@dataclass(frozen=True, order=True)
class ExcludedSignal:
    """One declared signal left out of the write, with its reason."""

    ids: str
    signal: str
    reason: str


@dataclass(frozen=True)
class WriteReceipt:
    """What one :func:`write_pulse` call wrote and what it left out."""

    machine: str
    shot: str
    phase: str
    run: int
    path: str
    description_root: str
    out_dir: str
    ids_written: tuple[str, ...]
    leaves: tuple[WrittenLeaf, ...]
    excluded: tuple[ExcludedSignal, ...]
    files: Mapping[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the receipt."""

        return {
            "machine": self.machine,
            "shot": self.shot,
            "phase": self.phase,
            "run": self.run,
            "path": self.path,
            "description_root": self.description_root,
            "out_dir": self.out_dir,
            "ids_written": list(self.ids_written),
            "leaves": [
                {
                    "ids": leaf.ids,
                    "semantic_id": leaf.semantic_id,
                    "target_path": leaf.target_path,
                    "time_path": leaf.time_path,
                    "samples": leaf.samples,
                }
                for leaf in self.leaves
            ],
            "excluded": [
                {"ids": item.ids, "signal": item.signal, "reason": item.reason}
                for item in self.excluded
            ],
            "files": dict(self.files),
        }


def _phase_directory(catalog: MachineMapCatalog, phase_map: MachineMap) -> Path:
    """Return the phase directory the description store declares, or refuse."""

    if catalog.description_store_format != _STORE_FORMAT:
        raise PulseWriteError(
            f"machine description store format {catalog.description_store_format!r} "
            f"is not {_STORE_FORMAT!r}; the writer reads one netCDF file per IDS"
        )
    if catalog.description_store_layout != _STORE_LAYOUT:
        raise PulseWriteError(
            f"machine description store layout {catalog.description_store_layout!r} "
            f"is not {_STORE_LAYOUT!r}; the writer addresses one directory per phase"
        )
    return catalog.description_store_root_path() / phase_map.name


def _read_description(path: Path, ids_name: str, dd_version: str):
    """Read one description IDS whole, refusing a missing or unreadable file."""

    if not path.is_file():
        raise PulseWriteError(f"description file is absent: {path}")
    try:
        with imas.DBEntry(path, "r", dd_version=dd_version) as entry:
            return entry.get(ids_name, autoconvert=False)
    except OSError as error:
        raise PulseWriteError(f"cannot read description {path}: {error}") from error


def _struct_array_index(rule: SignalRule) -> int:
    """The element a served signal selects within its struct array.

    A map rule with an explicit ``target_index`` names the element directly.
    ``magnetics/ip`` is a struct array the map serves without an index, so its
    one signal lands in the first element.
    """

    if rule.target_index is not None:
        return rule.target_index
    return 0


def _navigate(parent: object, components: Sequence[str], index: int) -> object:
    """Walk the struct-array levels of a target path, sizing the array to fit."""

    node = parent
    for component in components:
        child = getattr(node, component)
        if isinstance(child, IDSStructArray):
            if len(child) <= index:
                child.resize(index + 1)
            node = child[index]
        else:
            node = child
    return node


def _served_entries(
    signal_map: SignalMap,
    view: VirtualZarrView,
    cache_root: Path,
    shot_token: str,
) -> list[tuple[SignalRule, np.ndarray, np.ndarray]]:
    """Resolve every served signal to its canonical values and cache time base."""

    entries: list[tuple[SignalRule, np.ndarray, np.ndarray]] = []
    for rule in signal_map.signals:
        array = view.for_target(rule.target_path, rule.target_index)
        values = np.asarray(array[...], dtype=float)
        if values.ndim > 1:
            values = values[0]
        record = read_channel(
            cache_root, shot_token, rule.source_group, rule.source_array
        )
        time = np.asarray(record.time, dtype=float)
        if values.shape[-1] != time.shape[-1]:
            raise PulseWriteError(
                f"signal {rule.semantic_id!r} carries {values.shape[-1]} samples "
                f"for a {time.shape[-1]}-sample time base from "
                f"{rule.source_group}/{rule.source_array}"
            )
        entries.append((rule, values, time))
    return entries


def _shared_time_base(
    entries: Sequence[tuple[SignalRule, np.ndarray, np.ndarray]],
) -> np.ndarray | None:
    """Return the shared time base when every entry has one, else ``None``."""

    first = entries[0][2]
    for _rule, _values, time in entries[1:]:
        if time.shape != first.shape or not np.array_equal(time, first):
            return None
    return first


def _write_one_ids(
    ids_name: str,
    description: object,
    entries: Sequence[tuple[SignalRule, np.ndarray, np.ndarray]],
) -> list[WrittenLeaf]:
    """Write the served signals into one description IDS and report its leaves.

    ``homogeneous_time`` is 1 when every served signal shares one time base, so
    the IDS's own ``time`` node carries it and no signal carries its own;
    otherwise it is 0 and each signal's ``time`` sibling carries its base.
    """

    shared = _shared_time_base(entries)
    homogeneous = shared is not None
    description.ids_properties.homogeneous_time = 1 if homogeneous else 0
    if homogeneous:
        description.time = np.ascontiguousarray(shared, dtype=float)

    leaves: list[WrittenLeaf] = []
    for rule, values, time in entries:
        components = rule.target_path.split("/")
        holder = _navigate(description, components[1:-1], _struct_array_index(rule))
        setattr(holder, components[-1], np.ascontiguousarray(values, dtype=float))
        if homogeneous:
            time_path = f"{ids_name}/time"
        else:
            holder.time = np.ascontiguousarray(time, dtype=float)
            time_path = f"{ids_name}/{'/'.join(components[1:-1])}/time"
        leaves.append(
            WrittenLeaf(
                ids=ids_name,
                semantic_id=rule.semantic_id,
                target_path=rule.target_path,
                time_path=time_path,
                samples=int(values.shape[-1]),
            )
        )
    return leaves


def _excluded_signals(ids_name: str, signal_map: SignalMap) -> list[ExcludedSignal]:
    """Report every declared signal the map blocks, with the map's reason."""

    return [
        ExcludedSignal(
            ids=ids_name,
            signal=f"{blocked.source_group}/{blocked.source_array}",
            reason=blocked.reason,
        )
        for blocked in signal_map.blocked
    ]


def pulse_path(out_root: Path | str, pulse: object, run: int = 0) -> Path:
    """Return the run file's path: ``<out_root>/<pulse>_<run>.nc``.

    The pulse is the bare integer the cache names it by and the run is
    unpadded, so ``pulse 101154`` at run 0 is ``101154_0.nc``.  This is the one
    rule that owns the run file's name; every reader builds the name through it.
    """

    return Path(out_root) / f"{normalised_shot(pulse)}_{int(run)}.nc"


def write_pulse(
    machine: str,
    shot: object,
    out_root: Path | str,
    *,
    run: int = 0,
    overwrite: bool = False,
    maps: str | None = None,
    cache: str | None = None,
) -> WriteReceipt:
    """Write one shot's description IDSs with the signals the maps serve.

    Reads the phase description whole from the catalogue's store and the
    dynamic values from the shot's EDDB cache, writes one DD netCDF file
    holding every description IDS at :func:`pulse_path`, and returns a
    :class:`WriteReceipt`.  The cache root is the resolved ``eddb_cache``
    setting.  Refuses a shot with no cached pulse rather than writing empty
    signals, and refuses an existing run file unless ``overwrite`` is set.
    """

    settings = resolve_settings(SettingsFlags(maps=maps, cache=cache))
    catalog = load_packaged_machine_map(machine, search_path=settings.maps.value)
    shot_int = int(normalised_shot(shot))
    shot_token = eddb_token(shot)
    phase_map = map_for_shot(catalog, shot_int)
    description_root = _phase_directory(catalog, phase_map)

    cache_root = Path(str(settings.cache.value))
    cache_dir = cache_root / f"{shot_int}.zarr"
    if not cache_dir.is_dir():
        raise PulseWriteError(
            f"shot {shot_token} has no EDDB cache; {cache_dir} is absent"
        )

    out_dir = Path(out_root)
    path = pulse_path(out_dir, shot_int, run)
    if path.exists() and not overwrite:
        raise PulseWriteError(
            f"run file {path} already exists; pass --overwrite to replace it"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = bundle_for_machine(machine, search_path=settings.maps.value)
    maps_by_ids = {
        ids_name: load_signal_map(bundle.signal_map_path(machine, system))
        for ids_name, system in _SYSTEM_FOR_IDS.items()
    }
    views = {
        ids_name: VirtualZarrView.open(str(cache_dir), signal_map, shot=shot_int)
        for ids_name, signal_map in maps_by_ids.items()
    }

    leaves: list[WrittenLeaf] = []
    excluded: list[ExcludedSignal] = []

    with imas.DBEntry(path, "w", dd_version=catalog.dd_version) as entry:
        for ids_name in IDS_NAMES:
            description = _read_description(
                description_root / f"{ids_name}.nc", ids_name, catalog.dd_version
            )
            signal_map = maps_by_ids.get(ids_name)
            if signal_map is not None:
                entries = _served_entries(
                    signal_map, views[ids_name], cache_root, shot_token
                )
                leaves.extend(_write_one_ids(ids_name, description, entries))
                excluded.extend(_excluded_signals(ids_name, signal_map))
            description.validate()
            entry.put(description)

    return WriteReceipt(
        machine=machine,
        shot=shot_token,
        phase=phase_map.name,
        run=int(run),
        path=str(path),
        description_root=str(description_root),
        out_dir=str(out_dir),
        ids_written=tuple(IDS_NAMES),
        leaves=tuple(sorted(leaves)),
        excluded=tuple(sorted(excluded)),
        files=MappingProxyType({ids_name: str(path) for ids_name in IDS_NAMES}),
    )


__all__ = [
    "IDS_NAMES",
    "ExcludedSignal",
    "WriteReceipt",
    "WrittenLeaf",
    "PulseWriteError",
    "pulse_path",
    "write_pulse",
]
