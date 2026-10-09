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


def _agrees(draft: SignalRule, reference: SignalRule) -> bool:
    draft_scale = _scale(draft)
    reference_scale = _scale(reference)
    return (
        draft.target_path == reference.target_path
        and draft.target_index == reference.target_index
        and draft.source_group == reference.source_group
        and draft.source_array == reference.source_array
        and math.copysign(1, draft_scale) == math.copysign(1, reference_scale)
        and math.isclose(draft_scale, reference_scale, rel_tol=1e-9, abs_tol=1e-12)
    )


def _unplaced(reason: str) -> bool:
    return reason.startswith(
        (
            "member identifier ",
            "no machine-description member array",
            "target path, member identifier",
        )
    )


def score_handoff(
    document: Mapping[str, Any],
    catalogue: MachineMapCatalog,
    description_members: Mapping[str, Sequence[str]],
    hand_built_maps: Sequence[SignalMap],
) -> dict[str, Any]:
    """Count every hand-off row and compare imported values with hand-built rules."""
    imported = import_generated_mappings(document, catalogue, description_members)
    references = {mapping.system: mapping.signals for mapping in hand_built_maps}
    drafts = {mapping.system: mapping.signals for mapping in imported.maps}
    by_ids: dict[str, dict[str, Any]] = {}
    for item in document["ids"]:
        ids_name = item["ids_name"]
        exported = len(item["signals"]) + len(item["unexpanded"])
        unresolved = [row for row in imported.unresolved if row.ids_name == ids_name]
        unplaced = sum(_unplaced(row.reason) for row in unresolved)
        reasons = Counter(row.reason for row in unresolved if not _unplaced(row.reason))
        values = drafts.get(ids_name, ())
        unscored = sum(
            bool(_target_cocos_label(document["dd_version"], rule.target_path))
            and catalogue.source_cocos in (None, 0)
            for rule in values
        )
        reference = references.get(ids_name)
        agreeing: int | str = (
            "no reference"
            if reference is None
            else sum(
                not (
                    _target_cocos_label(document["dd_version"], rule.target_path)
                    and catalogue.source_cocos in (None, 0)
                )
                and any(_agrees(rule, prior) for prior in reference)
                for rule in values
            )
        )
        by_ids[ids_name] = {
            "exported": exported,
            "imported": exported - len(unresolved),
            "agreeing": agreeing,
            "unplaced": unplaced,
            "refused": sum(reasons.values()),
            "refused_reasons": dict(sorted(reasons.items())),
            "unscored_sign": unscored,
            "sign": "unscored" if unscored else "scored",
        }
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
        "unplaced": sum(row["unplaced"] for row in by_ids.values()),
        "refused": sum(row["refused"] for row in by_ids.values()),
        "refused_reasons": dict(sorted(total_reasons.items())),
        "unscored_sign": sum(row["unscored_sign"] for row in by_ids.values()),
        "sign": "unscored"
        if any(row["unscored_sign"] for row in by_ids.values())
        else "scored",
    }
    return {"machine": document["facility"], "by_ids": by_ids, "total": total}
