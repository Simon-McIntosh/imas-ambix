"""Export an ambix machine-map catalogue as a tokamap mapping directory.

TokaMap has no slot for a declared source unit, a named sign convention, a
source COCOS or a validation state; its only conversion slot is a bare
``scale`` on a ``DATA_SOURCE`` entry.  The forward collapse is therefore
lossy in provenance but exact in value: an ambix binding carries a unit
factor, a sign-convention factor and a COCOS factor, and their product is
the tokamap ``scale``.

The exporter writes the directory layout ``tokamap-validator`` accepts: a
top-level ``mappings.cfg.json`` and ``globals.json``, one directory per Data
Dictionary IDS group, and, beneath each group, one directory per partition
selected by the catalogue's shot ranges.  Each leaf carries its own
``globals.json`` and ``mappings.json``, and each leaf holds only the mappings
whose Data Dictionary path belongs to its own group.  A mapping's tokamap key
keeps the IDS prefix, so a group's file can be read for foreign bindings
without consulting anything else.

A DD path that several source arrays feed -- two probe families both
declaring ``b_field_pol_probe/name`` is an example -- has no tokamap
representation, because a JSON object holds one entry per key.  Those
alternatives collapse to a single entry, first declaration wins: tokamap has
no conditional, and the ambix catalogue already records the exclusion through
its qualifications.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import imas

from imas_ambix.cocos import canonical_factor

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from imas_ambix.data.machine_map import (
        ChannelBinding,
        MachineMap,
        MachineMapCatalog,
    )
    from imas_ambix.data.signal_map import SignalMap, SignalRule

TOKAMAP_FORMAT_VERSION = "1.0.0"
"""TokaMap directory format version written into ``mappings.cfg.json``."""

PARTITION_ATTRIBUTE = "shot"
"""Tokamap partition attribute the catalogue's shot ranges select."""

PARTITION_SELECTOR = "MAX_BELOW"
"""Selector mapping the catalogue's contiguous ranges onto directory names."""


class TokamapExportError(ValueError):
    """Raised when a catalogue binding cannot be expressed in tokamap."""


@dataclass(frozen=True)
class TokamapEntry:
    """One ``DATA_SOURCE`` mapping written under a group and partition."""

    group: str
    partition: int
    key: str
    kind: str
    source_name: str
    dd_path: str
    unit_factor: float
    sign_factor: float
    cocos_factor: float
    scale: float
    comment: str


@dataclass(frozen=True)
class TokamapExport:
    """The result of exporting one directory, counting what landed on disk.

    ``entries`` is the deduplicated list of mappings the exporter produced,
    while ``group_entry_counts`` and ``mappings_file_count`` describe the
    files themselves: each group's ``mappings.json`` is counted as written, so
    a group that received another group's bindings cannot hide behind a
    ``len(entries)`` that de-duplicates them away.
    """

    directory: Path
    groups: tuple[str, ...]
    partitions: tuple[int, ...]
    entries: tuple[TokamapEntry, ...] = field(default_factory=tuple)
    group_entry_counts: Mapping[str, int] = field(default_factory=dict)
    mappings_file_count: int = 0


_UNIT_CONVERSIONS: Mapping[tuple[str, str], float] = {
    ("degree", "rad"): math.pi / 180.0,
    ("deg", "rad"): math.pi / 180.0,
    ("rad", "degree"): 180.0 / math.pi,
    ("rad", "deg"): 180.0 / math.pi,
}

_SIGN_FACTORS: Mapping[str, float] = {
    "identity": 1.0,
    "not-applicable": 1.0,
    "negate": -1.0,
    "unknown-unvalidated": 1.0,
}


def _unit_factor(source_unit: str, target_unit: str) -> float:
    """Return the multiplicative factor from a source to a target unit."""

    if source_unit == target_unit:
        return 1.0
    key = (source_unit.strip().lower(), target_unit.strip().lower())
    try:
        return _UNIT_CONVERSIONS[key]
    except KeyError as error:
        raise TokamapExportError(
            f"no conversion from unit {source_unit!r} to {target_unit!r}"
        ) from error


def _sign_factor(sign_convention: str) -> float:
    """Return the multiplicative factor of a named sign convention."""

    try:
        return _SIGN_FACTORS[sign_convention]
    except KeyError as error:
        raise TokamapExportError(
            f"unknown sign convention {sign_convention!r}"
        ) from error


@cache
def _ids_metadata(dd_version: str, ids_name: str) -> Any:
    return imas.IDSFactory(dd_version).new(ids_name).metadata


@cache
def _cocos_transformation(dd_version: str, dd_path: str) -> str | None:
    """Resolve the nearest COCOS class declared on a DD target or its parents."""

    ids_name, relative_path = dd_path.split("/", maxsplit=1)
    metadata = _ids_metadata(dd_version, ids_name)
    components = relative_path.split("/")
    for size in range(len(components), 0, -1):
        node = metadata["/".join(components[:size])]
        transformation = getattr(node, "cocos_label_transformation", None)
        if transformation:
            return str(transformation)
    return None


def _data_type_name(node: Any) -> str:
    data_type = getattr(node, "data_type", None)
    return str(getattr(data_type, "name", data_type))


@cache
def _tokamap_key(dd_version: str, dd_path: str) -> str:
    """Render a DD path as a tokamap key that keeps its IDS group prefix.

    The key is the full Data Dictionary path, with every structure array
    marked ``[#]`` because tokamap expands that dimension.  Carrying the IDS
    prefix means a binding's group is readable from its key alone, so each
    group's ``mappings.json`` can be checked for bindings that belong to
    another IDS.
    """

    ids_name, relative_path = dd_path.split("/", maxsplit=1)
    metadata = _ids_metadata(dd_version, ids_name)
    components = relative_path.split("/")
    rendered: list[str] = [ids_name]
    for index, component in enumerate(components):
        node = metadata["/".join(components[: index + 1])]
        if _data_type_name(node) == "STRUCT_ARRAY":
            rendered.append(f"{component}[#]")
        else:
            rendered.append(component)
    return "/".join(rendered)


def _comment(
    *,
    sign_convention: str,
    source_unit: str,
    target_unit: str,
    transformation: str | None,
    cocos_factor: float,
    evidence: str,
) -> str:
    sign_note = (
        "no sign applied, sign convention unvalidated"
        if sign_convention == "unknown-unvalidated"
        else "sign applied"
    )
    return (
        f"sign_convention={sign_convention} ({sign_note}); "
        f"unit {source_unit}->{target_unit}; "
        f"cocos {transformation or 'none'} x{cocos_factor!r}; "
        f"{evidence}"
    )


def _catalogue_entry(
    binding: ChannelBinding,
    catalog: MachineMapCatalog,
) -> tuple[str, dict[str, Any], TokamapEntry]:
    unit_factor = _unit_factor(binding.source_unit, binding.target_unit)
    sign_factor = _sign_factor(binding.sign_convention)
    transformation = _cocos_transformation(catalog.dd_version, binding.dd_path)
    if transformation is None:
        cocos_factor = 1.0
    else:
        source_cocos = catalog.cocos_for_binding(binding)
        if source_cocos is None:
            raise TokamapExportError(
                f"binding {binding.name!r} targets COCOS-dependent path "
                f"{binding.dd_path!r} with no declared source COCOS"
            )
        cocos_factor = canonical_factor(
            transformation, source_cocos=int(source_cocos)
        )
    scale = unit_factor * sign_factor * cocos_factor
    comment = _comment(
        sign_convention=binding.sign_convention,
        source_unit=binding.source_unit,
        target_unit=binding.target_unit,
        transformation=transformation,
        cocos_factor=cocos_factor,
        evidence=binding.evidence,
    )
    key = _tokamap_key(catalog.dd_version, binding.dd_path)
    mapping = {
        "map_type": "DATA_SOURCE",
        "args": {
            "source_group": binding.source_group,
            "source_array": binding.source_array,
            "source_role": binding.source_role,
        },
        "data_source": binding.source_location,
        "scale": scale,
        "comment": comment,
    }
    record = TokamapEntry(
        group=binding.dd_path.split("/", maxsplit=1)[0],
        partition=-1,
        key=key,
        kind="catalogue",
        source_name=binding.name,
        dd_path=binding.dd_path,
        unit_factor=unit_factor,
        sign_factor=sign_factor,
        cocos_factor=cocos_factor,
        scale=scale,
        comment=comment,
    )
    return key, mapping, record


def _signal_entry(
    rule: SignalRule,
    dd_version: str,
) -> tuple[str, dict[str, Any], TokamapEntry]:
    unit_factor = float(rule.unit_factor)
    sign_factor = float(rule.channel_factor)
    cocos_factor = float(rule.convention_factor)
    scale = unit_factor * sign_factor * cocos_factor
    key = _tokamap_key(dd_version, rule.target_path)
    comment = (
        f"semantic_id={rule.semantic_id}; "
        f"unit {rule.source_unit}->{rule.target_unit}; "
        f"transformation {rule.transformation} x{cocos_factor!r}; "
        f"{rule.evidence}"
    )
    mapping = {
        "map_type": "DATA_SOURCE",
        "args": {
            "source_group": rule.source_group,
            "source_array": rule.source_array,
        },
        "data_source": rule.source_array,
        "scale": scale,
        "comment": comment,
    }
    record = TokamapEntry(
        group=rule.target_path.split("/", maxsplit=1)[0],
        partition=-1,
        key=key,
        kind="signal",
        source_name=rule.semantic_id,
        dd_path=rule.target_path,
        unit_factor=unit_factor,
        sign_factor=sign_factor,
        cocos_factor=cocos_factor,
        comment=comment,
        scale=scale,
    )
    return key, mapping, record


def _leaf_mappings(
    catalog: MachineMapCatalog,
    machine_map: MachineMap,
    signal_maps: tuple[SignalMap, ...],
    group: str,
) -> tuple[dict[str, Any], list[tuple[str, dict[str, Any], TokamapEntry]]]:
    """Return the mappings for one IDS group's leaf directory.

    Only bindings and signals whose Data Dictionary path belongs to ``group``
    are emitted: a group's ``mappings.json`` must hold that IDS and no other,
    because tokamap applies the whole file to the group it is filed under.
    """

    mappings: dict[str, Any] = {}
    records: list[tuple[str, dict[str, Any], TokamapEntry]] = []
    for binding in catalog.bindings_for(machine_map):
        if binding.dd_path.split("/", maxsplit=1)[0] != group:
            continue
        key, mapping, record = _catalogue_entry(binding, catalog)
        if key not in mappings:
            mappings[key] = mapping
            records.append((key, mapping, record))
    for signal_map in signal_maps:
        for rule in signal_map.signals:
            if rule.target_path.split("/", maxsplit=1)[0] != group:
                continue
            key, mapping, record = _signal_entry(rule, catalog.dd_version)
            if key not in mappings:
                mappings[key] = mapping
                records.append((key, mapping, record))
    return mappings, records


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def export_tokamap_directory(
    catalog: MachineMapCatalog,
    signal_maps: Iterable[SignalMap] = (),
    *,
    directory: Path | str,
    experiment: str | None = None,
    author: str = "ambix",
    version: str = TOKAMAP_FORMAT_VERSION,
) -> TokamapExport:
    """Write a tokamap directory from a catalogue and its signal maps.

    Every catalogue binding is emitted as a ``DATA_SOURCE`` mapping whose
    ``scale`` is the product of its unit factor, its sign-convention factor
    and its COCOS factor.  Shot partitions follow the catalogue's map ranges:
    each map contributes one partition directory named by its ``first_shot``,
    selected at read time by ``MAX_BELOW``.
    """

    signal_maps = tuple(signal_maps)
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)

    if experiment is None:
        experiment = catalog.maps[0].machine

    groups = sorted(
        {
            binding.dd_path.split("/", maxsplit=1)[0]
            for machine_map in catalog.maps
            for binding in catalog.bindings_for(machine_map)
        }
        | {
            rule.target_path.split("/", maxsplit=1)[0]
            for signal_map in signal_maps
            for rule in signal_map.signals
        }
    )
    partitions = sorted({machine_map.first_shot for machine_map in catalog.maps})

    _write_json(
        root / "mappings.cfg.json",
        {
            "metadata": {
                "experiment": experiment,
                "author": author,
                "version": version,
            },
            "partitions": [
                {"attribute": PARTITION_ATTRIBUTE, "selector": PARTITION_SELECTOR}
            ],
            "groups": list(groups),
        },
    )
    _write_json(root / "globals.json", {"source": catalog.source})

    records: list[TokamapEntry] = []
    group_entry_counts: dict[str, int] = {}
    mappings_file_count = 0
    for group in groups:
        group_total = 0
        for machine_map in catalog.maps:
            leaf = root / group / str(machine_map.first_shot)
            leaf.mkdir(parents=True, exist_ok=True)
            _write_json(
                leaf / "globals.json",
                {
                    "machine": machine_map.machine,
                    "map": machine_map.name,
                    "shot_first": machine_map.first_shot,
                    "shot_last": machine_map.last_shot,
                },
            )
            mappings, leaf_records = _leaf_mappings(
                catalog, machine_map, signal_maps, group
            )
            _write_json(leaf / "mappings.json", mappings)
            mappings_file_count += 1
            group_total += len(mappings)
            for _key, _mapping, record in leaf_records:
                records.append(
                    TokamapEntry(
                        group=group,
                        partition=machine_map.first_shot,
                        key=record.key,
                        kind=record.kind,
                        source_name=record.source_name,
                        dd_path=record.dd_path,
                        unit_factor=record.unit_factor,
                        sign_factor=record.sign_factor,
                        cocos_factor=record.cocos_factor,
                        scale=record.scale,
                        comment=record.comment,
                    )
                )
        group_entry_counts[group] = group_total

    return TokamapExport(
        directory=root,
        groups=tuple(groups),
        partitions=tuple(partitions),
        entries=tuple(records),
        group_entry_counts=group_entry_counts,
        mappings_file_count=mappings_file_count,
    )


__all__ = [
    "PARTITION_ATTRIBUTE",
    "PARTITION_SELECTOR",
    "TOKAMAP_FORMAT_VERSION",
    "TokamapEntry",
    "TokamapExport",
    "TokamapExportError",
    "export_tokamap_directory",
]
