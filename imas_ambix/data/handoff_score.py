"""Score an imported mapping hand-off against a machine's signal maps."""

from __future__ import annotations

import math
from collections import Counter
from functools import cache
from typing import TYPE_CHECKING, Any

import imas

from imas_ambix.data.generated_mapping_import import import_generated_mappings

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from imas_alambic.machine_map import MachineMapCatalog
    from imas_alambic.signal_map import SignalMap, SignalRule


@cache
def _target_cocos_label(dd_version: str, path: str) -> str | None:
    ids_name, relative_path = path.split("/", maxsplit=1)
    metadata = imas.IDSFactory(dd_version).new(ids_name).metadata
    parts = relative_path.split("/")
    for length in range(len(parts), 0, -1):
        label = getattr(
            metadata["/".join(parts[:length])], "cocos_label_transformation", None
        )
        if label:
            return str(label)
    return None


def _scale(rule: SignalRule) -> float:
    return float(rule.unit_factor * rule.channel_factor * rule.convention_factor)


def _agrees(draft: SignalRule, reference: SignalRule, *, sign_unscored: bool) -> bool:
    draft_scale = _scale(draft)
    reference_scale = _scale(reference)
    return (
        draft.target_path == reference.target_path
        and draft.target_index == reference.target_index
        and draft.source_group == reference.source_group
        and draft.source_array == reference.source_array
        and (
            sign_unscored
            or math.copysign(1, draft_scale) == math.copysign(1, reference_scale)
        )
        and math.isclose(
            abs(draft_scale) if sign_unscored else draft_scale,
            abs(reference_scale) if sign_unscored else reference_scale,
            rel_tol=1e-9,
            abs_tol=1e-12,
        )
    )


def _handoff_label(
    rows: Sequence[Mapping[str, Any]], rule: SignalRule, dd_version: str
) -> str | None:
    for row in rows:
        if (
            row.get("signal_id") == rule.semantic_id
            and row.get("target_path") == rule.target_path
            and row.get("source_group") == rule.source_group
            and row.get("source_array") == rule.source_array
        ):
            label = row.get("cocos_label")
            return (
                _target_cocos_label(dd_version, rule.target_path)
                if label is None
                else str(label)
            )
    return _target_cocos_label(dd_version, rule.target_path)


def _cocos_dependent(label: str | None) -> bool:
    if label is None or label in {"none", "one_like"}:
        return False
    if label in {
        "psi_like",
        "ip_like",
        "b0_like",
        "q_like",
        "pol_angle_like",
        "tor_angle_like",
        "dodpsi_like",
    } or label.startswith("grid_type"):
        return True
    raise ValueError(f"unknown COCOS label {label!r}")


def _unplaced(reason: str) -> bool:
    return reason.startswith(
        (
            "member identifier ",
            "no machine-description member array",
            "target path, member identifier",
        )
    )


def _measure(
    exported: int,
    unresolved: Sequence[Any],
    values: Sequence[SignalRule | tuple[SignalRule, bool]],
    reference: Sequence[SignalRule] | None,
    signals: Sequence[Mapping[str, Any]],
    catalogue: MachineMapCatalog,
    dd_version: str,
) -> dict[str, Any]:
    unplaced = sum(_unplaced(row.reason) for row in unresolved)
    reasons = Counter(row.reason for row in unresolved if not _unplaced(row.reason))
    matches: list[bool] = []
    conflicts: list[dict[str, Any]] = []
    if reference is not None:
        for rule, withheld in values:
            label = _handoff_label(signals, rule, dd_version)
            sign_unscored = withheld or (
                catalogue.source_cocos in (None, 0) and _cocos_dependent(label)
            )
            if any(
                _agrees(rule, prior, sign_unscored=sign_unscored)
                for prior in reference
            ):
                matches.append(sign_unscored)
                continue
            on_target = [
                prior
                for prior in reference
                if prior.target_path == rule.target_path
                and prior.target_index == rule.target_index
            ]
            if on_target:
                conflicts.append(
                    {
                        "target_path": rule.target_path,
                        "target_index": rule.target_index,
                        "imported_source_array": rule.source_array,
                        "hand_built_source_arrays": sorted(
                            prior.source_array for prior in on_target
                        ),
                    }
                )
    agreeing: int | str = "no reference" if reference is None else len(matches)
    return {
        "exported": exported,
        "imported": exported - len(unresolved),
        "agreeing": agreeing,
        "conflicting": len(conflicts),
        "conflicts": conflicts,
        "unplaced": unplaced,
        "refused": sum(reasons.values()),
        "refused_reasons": dict(sorted(reasons.items())),
        "sign_unscored": sum(matches),
    }


def _structure(target_path: str) -> str | None:
    """The target path's first segment after the IDS name, or None if absent."""
    parts = target_path.split("/")
    return "/".join(parts[:2]) if len(parts) >= 2 else None


def score_handoff(
    document: Mapping[str, Any],
    catalogue: MachineMapCatalog,
    description_members: Mapping[str, Sequence[str]],
    hand_built_maps: Sequence[SignalMap],
) -> dict[str, Any]:
    """Count every row and compare imported values with hand-built rules.

    Rows are grouped by IDS and by structure, the target path's first segment
    after the IDS name, so systems that share one IDS stay separate rows.
    """
    imported = import_generated_mappings(document, catalogue, description_members)
    references = {mapping.system: mapping.signals for mapping in hand_built_maps}
    drafts = {mapping.system: mapping.signals for mapping in imported.maps}
    pending: dict[str, list[SignalRule]] = {}
    for item in imported.pending:
        pending.setdefault(item.ids_name, []).append(item.rule)
    by_ids: dict[str, dict[str, Any]] = {}
    by_structure: dict[str, dict[str, Any]] = {}
    for item in document["ids"]:
        ids_name = item["ids_name"]
        rows = list(item["signals"]) + list(item["unexpanded"])
        unresolved = [row for row in imported.unresolved if row.ids_name == ids_name]
        values = [(rule, False) for rule in drafts.get(ids_name, ())]
        values += [(rule, True) for rule in pending.get(ids_name, ())]
        reference = references.get(ids_name)
        by_ids[ids_name] = _measure(
            len(rows),
            unresolved,
            values,
            reference,
            item["signals"],
            catalogue,
            document["dd_version"],
        )
        structure_rows: dict[str, list[Mapping[str, Any]]] = {}
        structure_unresolved: dict[str, list[Any]] = {}
        structure_values: dict[str, list[tuple[SignalRule, bool]]] = {}
        for row in rows:
            structure = _structure(row["target_path"])
            if structure is not None:
                structure_rows.setdefault(structure, []).append(row)
        for row in unresolved:
            structure = _structure(row.target_path)
            if structure is not None:
                structure_unresolved.setdefault(structure, []).append(row)
        for value in values:
            structure = _structure(value[0].target_path)
            if structure is not None:
                structure_values.setdefault(structure, []).append(value)
        names = set(structure_rows) | set(structure_unresolved) | set(
            structure_values
        )
        for structure in names:
            if reference is None:
                structure_reference: Sequence[SignalRule] | None = None
            else:
                structure_reference = [
                    prior
                    for prior in reference
                    if _structure(prior.target_path) == structure
                ] or None
            by_structure[structure] = _measure(
                len(structure_rows.get(structure, ())),
                structure_unresolved.get(structure, ()),
                structure_values.get(structure, ()),
                structure_reference,
                item["signals"],
                catalogue,
                document["dd_version"],
            )
    total_reasons = Counter()
    for row in by_ids.values():
        total_reasons.update(row["refused_reasons"])
    total = {
        "exported": sum(row["exported"] for row in by_ids.values()),
        "imported": sum(row["imported"] for row in by_ids.values()),
        "agreeing": sum(
            row["agreeing"]
            for row in by_ids.values()
            if isinstance(row["agreeing"], int)
        ),
        "conflicting": sum(row["conflicting"] for row in by_ids.values()),
        "unplaced": sum(row["unplaced"] for row in by_ids.values()),
        "refused": sum(row["refused"] for row in by_ids.values()),
        "refused_reasons": dict(sorted(total_reasons.items())),
        "sign_unscored": sum(row["sign_unscored"] for row in by_ids.values()),
    }
    return {
        "machine": document["facility"],
        "by_ids": by_ids,
        "by_structure": by_structure,
        "total": total,
    }
