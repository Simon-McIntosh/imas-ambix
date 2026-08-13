"""Shot-range boundaries for the FAIR-MAST machine geometry.

The geometry table groups sampled shots by static setup signature.  Sorting
those samples exposes every observed change, including a return to a geometry
seen earlier.  This module projects those observations onto a requested corpus
and produces one explicit, non-overlapping range for every observed setup run.
"""

from __future__ import annotations

import json
from bisect import bisect_left, bisect_right
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from imas_ambix.data.paths import MANIFEST_DIR

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    import pandas as pd


_COLLECTION_FIELDS: dict[str, tuple[str, ...]] = {
    "b_probes": ("r", "z", "angle_deg", "length"),
    "flux_loops": ("r", "z"),
    "pf_filaments": (
        "r",
        "z",
        "turns",
        "width",
        "height",
        "circuit",
        "xmult",
    ),
}
_SEQUENCE_FIELDS = ("limiter_r", "limiter_z")
_SCALAR_FIELDS = ("r0", "minor_radius")


@dataclass(frozen=True)
class GeometryTransition:
    """One inclusive shot range carrying a single machine setup."""

    name: str
    first_shot: int
    last_shot: int
    geometry_signature: str
    campaign: str
    changed_fields: tuple[str, ...]

    def contains(self, shot_id: int) -> bool:
        """Return whether ``shot_id`` lies in this inclusive range."""
        return self.first_shot <= int(shot_id) <= self.last_shot

    def to_dict(self) -> dict[str, object]:
        """Return the JSON-compatible manifest representation."""
        row = asdict(self)
        row["changed_fields"] = list(self.changed_fields)
        return row


def load_geometry_table_payload(
    path: Path | str = MANIFEST_DIR / "gs_geometry_tables.json",
) -> dict[str, Any]:
    """Load the immutable geometry-table artifact without rewriting it."""
    return json.loads(Path(path).read_text())


def _campaign_lookup(index: pd.DataFrame) -> tuple[dict[int, str], dict[str, int]]:
    required = {"shot_id", "campaign"}
    missing = required.difference(index.columns)
    if missing:
        raise ValueError(f"campaign index is missing columns: {sorted(missing)}")

    by_shot: dict[int, str] = {}
    first_shot: dict[str, int] = {}
    for shot, campaign in index.loc[:, ["shot_id", "campaign"]].itertuples(
        index=False, name=None
    ):
        shot_id = int(shot)
        name = str(campaign)
        prior = by_shot.setdefault(shot_id, name)
        if prior != name:
            raise ValueError(
                f"shot {shot_id} has conflicting campaigns {prior!r} and {name!r}"
            )
        first_shot[name] = min(first_shot.get(name, shot_id), shot_id)
    return by_shot, first_shot


def _geometry_rows(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw = payload.get("campaigns", payload)
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("geometry table payload has no campaigns")

    rows: dict[str, Mapping[str, Any]] = {}
    for key, value in raw.items():
        if not isinstance(value, Mapping):
            raise ValueError(f"geometry table {key!r} is not a mapping")
        signature = str(value.get("signature_key", key))
        if signature in rows:
            raise ValueError(f"duplicate geometry signature {signature!r}")
        rows[signature] = value
    return rows


def _sample_timeline(
    rows: Mapping[str, Mapping[str, Any]],
) -> list[tuple[int, str]]:
    samples: dict[int, str] = {}
    for signature, row in rows.items():
        for shot in row.get("shots", ()):
            shot_id = int(shot)
            prior = samples.setdefault(shot_id, signature)
            if prior != signature:
                raise ValueError(
                    f"geometry sample {shot_id} has signatures {prior!r} and "
                    f"{signature!r}"
                )

    timeline: list[tuple[int, str]] = []
    for shot_id, signature in sorted(samples.items()):
        if not timeline or timeline[-1][1] != signature:
            timeline.append((shot_id, signature))
    if not timeline:
        raise ValueError("geometry table payload has no sampled shots")
    return timeline


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=True)


def geometry_fields_changed(
    before: Mapping[str, Any] | None,
    after: Mapping[str, Any],
) -> tuple[str, ...]:
    """Name the geometry leaves that differ between two setup tables."""
    changed: list[str] = []
    for collection, fields in _COLLECTION_FIELDS.items():
        after_rows = list(after.get(collection, ()))
        before_rows = [] if before is None else list(before.get(collection, ()))
        if before is None or len(before_rows) != len(after_rows):
            changed.append(f"{collection}.count")
        for field in fields:
            after_values = [row.get(field) for row in after_rows]
            before_values = [row.get(field) for row in before_rows]
            if before is None or _canonical(before_values) != _canonical(after_values):
                changed.append(f"{collection}.{field}")

    for field in (*_SEQUENCE_FIELDS, *_SCALAR_FIELDS):
        if before is None or _canonical(before.get(field)) != _canonical(
            after.get(field)
        ):
            changed.append(field)
    return tuple(changed)


def build_geometry_transitions(
    corpus_shot_ids: Iterable[int],
    campaign_index: pd.DataFrame,
    geometry_payload: Mapping[str, Any],
) -> tuple[GeometryTransition, ...]:
    """Project observed setup changes onto an inclusive corpus shot range.

    Geometry samples need not be members of the target corpus.  A change at a
    missing shot takes effect at the first corpus shot after it.  The latest
    sample at or before the corpus start supplies the initial setup.
    """
    corpus = sorted({int(shot) for shot in corpus_shot_ids})
    if not corpus:
        return ()

    rows = _geometry_rows(geometry_payload)
    timeline = _sample_timeline(rows)
    by_shot, _ = _campaign_lookup(campaign_index)
    corpus_first, corpus_last = corpus[0], corpus[-1]

    active_positions = [shot for shot, _ in timeline]
    active_index = bisect_right(active_positions, corpus_first) - 1
    if active_index < 0:
        raise ValueError(
            f"no geometry sample exists at or before corpus start {corpus_first}"
        )

    projected: list[tuple[int, str]] = [(corpus_first, timeline[active_index][1])]
    for observed_shot, signature in timeline[active_index + 1 :]:
        position = bisect_left(corpus, observed_shot)
        if position == len(corpus):
            break
        first_shot = corpus[position]
        if first_shot > corpus_last:
            break
        if projected[-1][0] == first_shot:
            projected[-1] = (first_shot, signature)
        elif projected[-1][1] != signature:
            projected.append((first_shot, signature))

    transitions: list[GeometryTransition] = []
    previous_row: Mapping[str, Any] | None = None
    for position, (first_shot, signature) in enumerate(projected):
        if first_shot not in by_shot:
            raise ValueError(f"campaign index has no row for shot {first_shot}")
        last_shot = (
            projected[position + 1][0] - 1
            if position + 1 < len(projected)
            else corpus_last
        )
        row = rows[signature]
        digest = signature.rsplit("-", maxsplit=1)[-1]
        transitions.append(
            GeometryTransition(
                name=f"mast-geometry-{first_shot}-{digest}",
                first_shot=first_shot,
                last_shot=last_shot,
                geometry_signature=signature,
                campaign=by_shot[first_shot],
                changed_fields=geometry_fields_changed(previous_row, row),
            )
        )
        previous_row = row
    return tuple(transitions)


def transition_for_shot(
    transitions: Sequence[GeometryTransition], shot_id: int
) -> GeometryTransition:
    """Return the single declared geometry range covering ``shot_id``."""
    if not transitions:
        raise LookupError("no geometry transitions are declared")
    starts = [transition.first_shot for transition in transitions]
    position = bisect_right(starts, int(shot_id)) - 1
    if position < 0 or not transitions[position].contains(shot_id):
        raise LookupError(f"shot {shot_id} is outside the declared geometry ranges")
    return transitions[position]


def count_transitions_inside_campaigns(
    transitions: Sequence[GeometryTransition], campaign_index: pd.DataFrame
) -> int:
    """Count boundaries after the first indexed shot of their campaign."""
    _, first_shot = _campaign_lookup(campaign_index)
    return sum(
        transition.first_shot > first_shot[transition.campaign]
        for transition in transitions
    )
