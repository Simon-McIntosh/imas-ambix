"""Shot-range boundaries for the FAIR-MAST machine geometry.

The geometry table groups sampled shots by static setup signature.  A setup
signature is intentionally sensitive to the numerical representation, so a
different element count or a re-meshed conductor changes that signature even
when the represented machine does not move.  Sorting the samples therefore
produces *candidate* boundaries, not necessarily physical geometry changes.

Every candidate is classified here before it can open a range.  A change in
element count alone is discretisation.  Changes in element positions, sizes,
weights, or circuit numbering are also discretisation when the named conductor
outlines agree within :data:`CONDUCTOR_OUTLINE_TOLERANCE_M`.  Small coordinate
rounding in otherwise stable geometry is tolerated separately.  Only a
physical-geometry change opens a new, explicit, non-overlapping shot range.
"""

from __future__ import annotations

import json
import math
from bisect import bisect_left, bisect_right
from collections import defaultdict
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

CONDUCTOR_OUTLINE_TOLERANCE_M = 1e-5
"""Maximum conductor-outline residual treated as the same geometry (10 um)."""

GEOMETRY_POSITION_TOLERANCE_M = 5e-4
"""Maximum non-conductor coordinate rounding treated as unchanged (0.5 mm)."""

GEOMETRY_CHANGE = "geometry_change"
DISCRETISATION_ONLY = "discretisation_only"


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


@dataclass(frozen=True)
class GeometryBoundaryAssessment:
    """Classification of one sampled setup boundary before range creation."""

    name: str
    first_shot: int
    before_signature: str
    after_signature: str
    before_element_count: int
    after_element_count: int
    changed_fields: tuple[str, ...]
    classification: str
    reason: str
    conductor_outline_residual_m: float
    geometry_position_residual_m: float

    @property
    def opens_range(self) -> bool:
        """Return whether this boundary represents a physical geometry change."""
        return self.classification == GEOMETRY_CHANGE

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible evidence representation."""
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


def _projected_timeline(
    corpus: list[int], timeline: list[tuple[int, str]]
) -> list[tuple[int, str]]:
    """Project sampled setup changes onto the first affected corpus shots."""
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
    return projected


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


def _element_groups(row: Mapping[str, Any]) -> dict[int, list[Mapping[str, Any]]]:
    groups: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for element in row.get("pf_filaments", ()):
        groups[int(element.get("circuit", 0))].append(element)
    return dict(groups)


def _circuit_outline(elements: list[Mapping[str, Any]]) -> tuple[float, ...]:
    """Return the axis-aligned outline represented by one element group."""
    radial = [
        (
            float(item.get("r", 0.0)) - abs(float(item.get("width", 0.0))) / 2,
            float(item.get("r", 0.0)) + abs(float(item.get("width", 0.0))) / 2,
        )
        for item in elements
    ]
    vertical = [
        (
            float(item.get("z", 0.0)) - abs(float(item.get("height", 0.0))) / 2,
            float(item.get("z", 0.0)) + abs(float(item.get("height", 0.0))) / 2,
        )
        for item in elements
    ]
    return (
        min(edge[0] for edge in radial),
        min(edge[0] for edge in vertical),
        max(edge[1] for edge in radial),
        max(edge[1] for edge in vertical),
    )


def _is_subdivided_conductor(elements: list[Mapping[str, Any]]) -> bool:
    """Identify a conductor group rather than a singleton mesh element."""
    return len(elements) > 1 or any(
        abs(float(item.get("xmult", 1.0))) < 1.0 - 1e-9 for item in elements
    )


def _passive_reference_outlines(
    row: Mapping[str, Any],
) -> dict[str, tuple[float, ...]]:
    """Return stable named passive-conductor reference coordinates."""
    outlines: dict[str, tuple[float, ...]] = {}
    for position, item in enumerate(row.get("passive_structures", ())):
        name = str(item.get("name", position))
        status = "obsolete" if bool(item.get("obsolete", False)) else "current"
        r = float(item.get("r", 0.0))
        z = float(item.get("z", 0.0))
        outlines[f"passive:{name}:{status}"] = (r, z, r, z)
    return outlines


def _mapping_residual(
    before: Mapping[str, tuple[float, ...]],
    after: Mapping[str, tuple[float, ...]],
) -> float:
    if before.keys() != after.keys():
        return math.inf
    return max(
        (
            abs(left - right)
            for key in before
            for left, right in zip(before[key], after[key], strict=True)
        ),
        default=0.0,
    )


def conductor_outline_residual_m(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> float:
    """Return the maximum named-conductor outline displacement in metres.

    Multi-element or fractionally weighted circuits are compared by their
    represented rectangular envelope, which is unchanged when a pack is split
    into more cells.  Singleton passive mesh elements are deliberately not
    matched by their transient circuit numbers; the payload's named passive
    structure references carry their stable physical outline coordinates.
    Payloads without those references fall back to comparing every circuit.
    """
    before_groups = _element_groups(before)
    after_groups = _element_groups(after)
    circuit_ids = set(before_groups) | set(after_groups)
    selected = {
        circuit
        for circuit in circuit_ids
        if _is_subdivided_conductor(before_groups.get(circuit, []))
        or _is_subdivided_conductor(after_groups.get(circuit, []))
    }
    passive_before = _passive_reference_outlines(before)
    passive_after = _passive_reference_outlines(after)
    if not selected and not passive_before and not passive_after:
        selected = circuit_ids

    before_outlines = dict(passive_before)
    after_outlines = dict(passive_after)
    for circuit in selected:
        if circuit in before_groups:
            before_outlines[f"circuit:{circuit}"] = _circuit_outline(
                before_groups[circuit]
            )
        if circuit in after_groups:
            after_outlines[f"circuit:{circuit}"] = _circuit_outline(
                after_groups[circuit]
            )
    return _mapping_residual(before_outlines, after_outlines)


def _indexed_geometry(
    row: Mapping[str, Any], collection: str, fields: tuple[str, ...]
) -> dict[int, tuple[float, ...]]:
    indexed: dict[int, tuple[float, ...]] = {}
    for position, item in enumerate(row.get(collection, ())):
        key = int(item.get("index", position))
        if key in indexed:
            return {}
        indexed[key] = tuple(float(item.get(field, 0.0)) for field in fields)
    return indexed


def _position_residual_m(before: Mapping[str, Any], after: Mapping[str, Any]) -> float:
    """Return the largest non-conductor physical-coordinate displacement."""
    residuals = [
        _mapping_residual(
            _indexed_geometry(before, "b_probes", ("r", "z", "length")),
            _indexed_geometry(after, "b_probes", ("r", "z", "length")),
        ),
        _mapping_residual(
            _indexed_geometry(before, "flux_loops", ("r", "z")),
            _indexed_geometry(after, "flux_loops", ("r", "z")),
        ),
    ]
    before_limiter_r = list(before.get("limiter_r", ()))
    before_limiter_z = list(before.get("limiter_z", ()))
    after_limiter_r = list(after.get("limiter_r", ()))
    after_limiter_z = list(after.get("limiter_z", ()))
    if len(before_limiter_r) != len(before_limiter_z):
        residuals.append(math.inf)
    if len(after_limiter_r) != len(after_limiter_z):
        residuals.append(math.inf)
    before_limiter = list(zip(before_limiter_r, before_limiter_z, strict=False))
    after_limiter = list(zip(after_limiter_r, after_limiter_z, strict=False))
    if len(before_limiter) != len(after_limiter):
        residuals.append(math.inf)
    else:
        residuals.append(
            max(
                (
                    abs(float(left) - float(right))
                    for before_point, after_point in zip(
                        before_limiter, after_limiter, strict=True
                    )
                    for left, right in zip(before_point, after_point, strict=True)
                ),
                default=0.0,
            )
        )
    residuals.extend(
        abs(float(before.get(field, 0.0)) - float(after.get(field, 0.0)))
        for field in _SCALAR_FIELDS
    )
    return max(residuals, default=0.0)


def _nonpositional_geometry_changed(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> bool:
    before_angles = [item.get("angle_deg") for item in before.get("b_probes", ())]
    after_angles = [item.get("angle_deg") for item in after.get("b_probes", ())]
    return _canonical(before_angles) != _canonical(after_angles)


def _assess_projected_boundaries(
    projected: list[tuple[int, str]],
    rows: Mapping[str, Mapping[str, Any]],
    *,
    outline_tolerance_m: float,
    position_tolerance_m: float,
) -> tuple[GeometryBoundaryAssessment, ...]:
    assessments: list[GeometryBoundaryAssessment] = []
    for (before_shot, before_signature), (first_shot, after_signature) in zip(
        projected, projected[1:], strict=False
    ):
        del before_shot
        before = rows[before_signature]
        after = rows[after_signature]
        changed_fields = geometry_fields_changed(before, after)
        outline_residual = conductor_outline_residual_m(before, after)
        position_residual = _position_residual_m(before, after)
        element_fields = {
            field for field in changed_fields if field.startswith("pf_filaments.")
        }

        if element_fields == {"pf_filaments.count"} and len(changed_fields) == 1:
            classification = DISCRETISATION_ONLY
            reason = "element count alone changed"
        elif _nonpositional_geometry_changed(before, after):
            classification = GEOMETRY_CHANGE
            reason = "probe orientation changed"
        elif outline_residual > outline_tolerance_m:
            classification = GEOMETRY_CHANGE
            reason = (
                f"conductor outline residual {outline_residual:.9g} m exceeds "
                f"{outline_tolerance_m:.9g} m"
            )
        elif position_residual > position_tolerance_m:
            classification = GEOMETRY_CHANGE
            reason = (
                f"geometry position residual {position_residual:.9g} m exceeds "
                f"{position_tolerance_m:.9g} m"
            )
        else:
            classification = DISCRETISATION_ONLY
            reason = (
                f"element mesh changed with conductor outline residual "
                f"{outline_residual:.9g} m <= {outline_tolerance_m:.9g} m and "
                f"geometry position residual {position_residual:.9g} m <= "
                f"{position_tolerance_m:.9g} m"
            )

        digest = after_signature.rsplit("-", maxsplit=1)[-1]
        assessments.append(
            GeometryBoundaryAssessment(
                name=f"mast-geometry-{first_shot}-{digest}",
                first_shot=first_shot,
                before_signature=before_signature,
                after_signature=after_signature,
                before_element_count=len(before.get("pf_filaments", ())),
                after_element_count=len(after.get("pf_filaments", ())),
                changed_fields=changed_fields,
                classification=classification,
                reason=reason,
                conductor_outline_residual_m=outline_residual,
                geometry_position_residual_m=position_residual,
            )
        )
    return tuple(assessments)


def classify_geometry_boundaries(
    corpus_shot_ids: Iterable[int],
    geometry_payload: Mapping[str, Any],
    *,
    outline_tolerance_m: float = CONDUCTOR_OUTLINE_TOLERANCE_M,
    position_tolerance_m: float = GEOMETRY_POSITION_TOLERANCE_M,
) -> tuple[GeometryBoundaryAssessment, ...]:
    """Classify every projected setup boundary as physical or discretisation."""
    corpus = sorted({int(shot) for shot in corpus_shot_ids})
    if not corpus:
        return ()
    rows = _geometry_rows(geometry_payload)
    projected = _projected_timeline(corpus, _sample_timeline(rows))
    return _assess_projected_boundaries(
        projected,
        rows,
        outline_tolerance_m=outline_tolerance_m,
        position_tolerance_m=position_tolerance_m,
    )


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
    corpus_last = corpus[-1]
    projected = _projected_timeline(corpus, timeline)
    assessments = _assess_projected_boundaries(
        projected,
        rows,
        outline_tolerance_m=CONDUCTOR_OUTLINE_TOLERANCE_M,
        position_tolerance_m=GEOMETRY_POSITION_TOLERANCE_M,
    )
    projected = [projected[0]] + [
        item
        for item, assessment in zip(projected[1:], assessments, strict=True)
        if assessment.opens_range
    ]

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
