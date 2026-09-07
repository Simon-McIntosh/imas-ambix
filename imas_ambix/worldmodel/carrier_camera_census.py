"""Rank Nova labeller sessions by presentation-camera richness.

The census is deliberately metadata-only: manifest rows establish usable
converged labels, level-1 Zarr metadata establishes the available plasma-camera
streams, and the presence of a Thomson group establishes that the renderer can
draw its scattering-volume locus.  Corrupt or incomplete sessions remain in
the output with explicit exclusion reasons instead of disappearing from the
population.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from imas_ambix.camdyn.dataset import level1_shot_path
from imas_ambix.data.paths import LEVEL1_DIR
from imas_ambix.worldmodel.flux_decoder_video import _manifest_selection
from imas_ambix.worldmodel.flux_label_dataset import DEFAULT_SESSION_ROOT

if TYPE_CHECKING:
    from collections.abc import Callable

PLASMA_CAMERA_GROUPS = ("rba", "rbb", "rbc")
THOMSON_GROUPS = ("atm", "ayc", "aye")
MIN_CONVERGED_SLICES = 40
DEFAULT_TOP_COUNT = 5
DEMO_CAMERA_GROUP = "rbb"
DEMO_FRAME_HEIGHT = 128
DEMO_FRAME_WIDTH = 172
DEMO_CADENCE_US = 20.0
DEMO_CADENCE_TOLERANCE_US = 5.0
SQUARE_CAMERA_GROUP = "rbb"
SQUARE_FRAME_HEIGHT = 512
SQUARE_FRAME_WIDTH = 512
SQUARE_MIN_CADENCE_US = 550.0
SQUARE_MAX_CADENCE_US = 800.0
CADENCE_BAND_WIDTH_US = 10.0

# Each scale is the point where that factor contributes one half.  Frame count
# is intentionally saturating: beyond a thousand frames, more cadence has less
# presentation value than gaining spatial area or shot coverage.
SCORE_SCALES = {
    "converged_slice_count": 40.0,
    "frame_area": 100_000.0,
    "temporal_span_s": 0.5,
    "frame_count": 1_000.0,
}


@dataclass(frozen=True, slots=True)
class CameraMetrics:
    """Metadata for one usable plasma-camera stream."""

    camera_group: str
    frame_height: int
    frame_width: int
    frame_area: int
    frame_count: int
    temporal_span_s: float
    cadence_us: float | None


@dataclass(slots=True)
class SessionRanking:
    """One corpus session's complete ranking record."""

    shot: int
    manifest_status: str
    manifest_slice_count: int
    session_frame_count: int
    converged_slice_count: int
    thomson_groups: list[str] = field(default_factory=list)
    available_cameras: list[CameraMetrics] = field(default_factory=list)
    camera_group: str | None = None
    frame_height: int | None = None
    frame_width: int | None = None
    frame_area: int | None = None
    frame_count: int | None = None
    temporal_span_s: float | None = None
    cadence_us: float | None = None
    score: float = 0.0
    score_components: dict[str, float] = field(default_factory=dict)
    eligible: bool = False
    exclusion_reasons: list[str] = field(default_factory=list)
    camera_errors: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    rank: int = 0


def _saturating_component(value: float, scale: float) -> float:
    value = max(0.0, float(value))
    return value / (value + scale)


def camera_richness_score(
    *,
    converged_slice_count: int,
    frame_area: int,
    temporal_span_s: float,
    frame_count: int,
) -> tuple[float, dict[str, float]]:
    """Return an equal-factor geometric score and its four components.

    Saturation prevents a narrow high-cadence strip from winning on frame count
    alone.  The geometric combination keeps every factor load-bearing: a weak
    camera dimension cannot be fully compensated by one exceptionally large
    number elsewhere.
    """
    raw = {
        "converged_slice_count": float(converged_slice_count),
        "frame_area": float(frame_area),
        "temporal_span_s": float(temporal_span_s),
        "frame_count": float(frame_count),
    }
    components = {
        name: _saturating_component(value, SCORE_SCALES[name])
        for name, value in raw.items()
    }
    score = math.prod(components.values()) ** (1.0 / len(components))
    return float(score), components


def _session_frame_count(path: Path) -> int:
    import h5netcdf  # noqa: PLC0415

    with h5netcdf.File(path, "r") as root:
        if "steering" not in root.groups:
            raise KeyError(f"{path} has no steering group")
        group = root.groups["steering"]
        if "time" not in group.dimensions:
            raise KeyError(f"{path} steering group has no time dimension")
        return int(group.dimensions["time"].size)


def _camera_metrics(group: Any, camera_group: str) -> CameraMetrics:
    arrays = set(group.array_keys())
    if not {"data", "time"}.issubset(arrays):
        raise KeyError(f"{camera_group} does not carry both data and time")
    data = group["data"]
    if len(data.shape) not in (3, 4):
        raise ValueError(
            f"{camera_group}/data must have shape (time, height, width[, channel])"
        )
    frame_count = int(data.shape[0])
    height = int(data.shape[1])
    width = int(data.shape[2])
    if min(frame_count, height, width) <= 0:
        raise ValueError(f"{camera_group}/data has an empty dimension")
    times = np.asarray(group["time"], dtype=np.float64)
    if times.ndim != 1 or int(times.size) != frame_count:
        raise ValueError(f"{camera_group} data and time lengths differ")
    if not np.isfinite(times).all():
        raise ValueError(f"{camera_group}/time contains a non-finite value")
    span = float(times.max() - times.min()) if times.size > 1 else 0.0
    cadence_us = (
        float(np.median(np.abs(np.diff(times)))) * 1_000_000.0
        if times.size > 1
        else None
    )
    return CameraMetrics(
        camera_group=camera_group,
        frame_height=height,
        frame_width=width,
        frame_area=height * width,
        frame_count=frame_count,
        temporal_span_s=span,
        cadence_us=cadence_us,
    )


def _best_camera(
    store: Any,
    converged_slice_count: int,
) -> tuple[
    list[CameraMetrics],
    CameraMetrics | None,
    float,
    dict[str, float],
    dict[str, str],
]:
    groups = set(store.group_keys())
    candidates: list[tuple[float, CameraMetrics, dict[str, float]]] = []
    available: list[CameraMetrics] = []
    errors: dict[str, str] = {}
    for camera_group in PLASMA_CAMERA_GROUPS:
        if camera_group not in groups:
            continue
        try:
            metrics = _camera_metrics(store[camera_group], camera_group)
            score, components = camera_richness_score(
                converged_slice_count=converged_slice_count,
                frame_area=metrics.frame_area,
                temporal_span_s=metrics.temporal_span_s,
                frame_count=metrics.frame_count,
            )
        except (KeyError, TypeError, ValueError) as exc:
            errors[camera_group] = f"{type(exc).__name__}: {exc}"
            continue
        available.append(metrics)
        candidates.append((score, metrics, components))
    if not candidates:
        return available, None, 0.0, {}, errors
    score, metrics, components = min(
        candidates,
        key=lambda item: (-item[0], item[1].camera_group),
    )
    return available, metrics, score, components, errors


def _is_demo_camera(metrics: CameraMetrics) -> bool:
    return (
        metrics.camera_group == DEMO_CAMERA_GROUP
        and metrics.frame_height == DEMO_FRAME_HEIGHT
        and metrics.frame_width == DEMO_FRAME_WIDTH
        and metrics.cadence_us is not None
        and abs(metrics.cadence_us - DEMO_CADENCE_US) <= DEMO_CADENCE_TOLERANCE_US
    )


def _is_square_camera(metrics: CameraMetrics) -> bool:
    return (
        metrics.camera_group == SQUARE_CAMERA_GROUP
        and metrics.frame_height == SQUARE_FRAME_HEIGHT
        and metrics.frame_width == SQUARE_FRAME_WIDTH
        and metrics.cadence_us is not None
        and SQUARE_MIN_CADENCE_US <= metrics.cadence_us <= SQUARE_MAX_CADENCE_US
    )


def _cadence_band(cadence_us: float | None) -> tuple[float | None, float | None]:
    if cadence_us is None:
        return None, None
    centre = (
        math.floor(cadence_us / CADENCE_BAND_WIDTH_US + 0.5) * CADENCE_BAND_WIDTH_US
    )
    half_width = CADENCE_BAND_WIDTH_US / 2.0
    return centre - half_width, centre + half_width


def _rbb_geometry_distribution(
    records: list[SessionRanking],
    *,
    session_total: int,
    measured_at_utc: str,
) -> list[dict[str, Any]]:
    geometries: dict[tuple[int, int, float | None, float | None], list[int]] = {}
    for record in records:
        for camera in record.available_cameras:
            if camera.camera_group != "rbb":
                continue
            lower, upper = _cadence_band(camera.cadence_us)
            key = (camera.frame_height, camera.frame_width, lower, upper)
            geometries.setdefault(key, []).append(record.shot)

    rows = []
    for (height, width, lower, upper), shots in geometries.items():
        statuses = {
            record.shot: record.manifest_status
            for record in records
            if record.shot in shots
        }
        complete_shots = sorted(
            shot for shot in shots if statuses.get(shot) == "complete"
        )
        rows.append(
            {
                "frame_height": height,
                "frame_width": width,
                "cadence_band_lower_us": lower,
                "cadence_band_upper_us": upper,
                "cadence_band_upper_inclusive": False if upper is not None else None,
                "shot_count": len(shots),
                "shots": sorted(shots),
                "complete_shot_count": len(complete_shots),
                "complete_shots": complete_shots,
                "session_total": session_total,
                "measured_at_utc": measured_at_utc,
            }
        )
    rows.sort(
        key=lambda row: (
            -row["shot_count"],
            row["frame_height"],
            row["frame_width"],
            row["cadence_band_lower_us"] is None,
            row["cadence_band_lower_us"] or 0.0,
        )
    )
    return rows


def _family_summary(
    records: list[SessionRanking],
    *,
    family_name: str,
    definition: dict[str, Any],
    predicate: Callable[[CameraMetrics], bool],
    session_total: int,
    measured_at_utc: str,
) -> dict[str, Any]:
    shots = sorted(
        record.shot
        for record in records
        if any(predicate(camera) for camera in record.available_cameras)
    )
    count = len(shots)
    complete_shots = sorted(
        record.shot
        for record in records
        if record.shot in shots and record.manifest_status == "complete"
    )
    shot_word = "shot" if count == 1 else "shots"
    return {
        "family": family_name,
        "definition": definition,
        "shot_count": count,
        "shots": shots,
        "complete_shot_count": len(complete_shots),
        "complete_shots": complete_shots,
        "present_in_measured_sessions": bool(shots),
        "finding": (
            f"{count} {shot_word} found among {session_total} sessions at "
            f"{measured_at_utc}; {len(complete_shots)} had complete manifests"
        ),
        "session_total": session_total,
        "measured_at_utc": measured_at_utc,
    }


def _read_session_ranking(manifest_path: Path, level1_root: Path) -> SessionRanking:
    shot_from_name = int(manifest_path.name.removesuffix(".manifest.json"))
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{manifest_path} does not contain a JSON object")
    shot = int(payload.get("shot", shot_from_name))
    if shot != shot_from_name:
        raise ValueError(
            f"{manifest_path} names shot {shot}, expected {shot_from_name}"
        )
    slices = payload.get("slices")
    if not isinstance(slices, list):
        raise ValueError(f"{manifest_path} has no slice-row list")

    session_path = manifest_path.with_suffix("").with_suffix(".nc")
    if session_path.is_file():
        session_count = _session_frame_count(session_path)
        _, selected_shot, selected, manifest_slice_count = _manifest_selection(
            session_path, session_count
        )
        if selected_shot != shot:
            raise ValueError(
                f"{session_path} selection names shot {selected_shot}, expected {shot}"
            )
        converged_slice_count = len(selected)
    else:
        session_count = 0
        manifest_slice_count = len(slices)
        converged_slice_count = sum(
            bool(row.get("written", False)) and bool(row.get("converged", False))
            for row in slices
            if isinstance(row, dict)
        )

    record = SessionRanking(
        shot=shot,
        manifest_status=str(payload.get("status", "unknown")),
        manifest_slice_count=manifest_slice_count,
        session_frame_count=session_count,
        converged_slice_count=converged_slice_count,
    )
    if record.manifest_status != "complete":
        record.exclusion_reasons.append("manifest_not_complete")
    if not session_path.is_file():
        record.exclusion_reasons.append("missing_session_file")

    level1_path = level1_shot_path(shot, level1_dir=level1_root)
    if not level1_path.is_dir():
        record.exclusion_reasons.append("missing_level1_shot")
    else:
        import zarr  # noqa: PLC0415

        store = zarr.open_group(str(level1_path), mode="r")
        groups = set(store.group_keys())
        record.thomson_groups = [name for name in THOMSON_GROUPS if name in groups]
        available, metrics, score, components, errors = _best_camera(
            store, converged_slice_count
        )
        record.available_cameras = available
        record.camera_errors = errors
        if metrics is not None:
            record.camera_group = metrics.camera_group
            record.frame_height = metrics.frame_height
            record.frame_width = metrics.frame_width
            record.frame_area = metrics.frame_area
            record.frame_count = metrics.frame_count
            record.temporal_span_s = metrics.temporal_span_s
            record.cadence_us = metrics.cadence_us
            record.score = score
            record.score_components = components

    if not record.thomson_groups:
        record.exclusion_reasons.append("missing_thomson_group")
    if record.converged_slice_count < MIN_CONVERGED_SLICES:
        record.exclusion_reasons.append("fewer_than_40_converged_slices")
    if record.camera_group is None:
        record.exclusion_reasons.append("missing_usable_plasma_camera")
    record.eligible = not record.exclusion_reasons
    return record


def census_sessions(
    *,
    session_root: Path = DEFAULT_SESSION_ROOT,
    level1_root: Path = LEVEL1_DIR,
) -> list[SessionRanking]:
    """Read and rank every manifest under *session_root*."""
    manifest_paths = sorted(
        Path(session_root).glob("*.manifest.json"),
        key=lambda path: int(path.name.removesuffix(".manifest.json")),
    )
    if not manifest_paths:
        raise FileNotFoundError(f"no labeller manifests found under {session_root}")

    records: list[SessionRanking] = []
    for manifest_path in manifest_paths:
        shot = int(manifest_path.name.removesuffix(".manifest.json"))
        try:
            record = _read_session_ranking(manifest_path, Path(level1_root))
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            record = SessionRanking(
                shot=shot,
                manifest_status="unreadable",
                manifest_slice_count=0,
                session_frame_count=0,
                converged_slice_count=0,
                exclusion_reasons=["session_read_error"],
                error=f"{type(exc).__name__}: {exc}",
            )
        records.append(record)

    records.sort(key=lambda row: (not row.eligible, -row.score, row.shot))
    for rank, record in enumerate(records, start=1):
        record.rank = rank
    return records


def build_ranking_payload(
    *,
    session_root: Path = DEFAULT_SESSION_ROOT,
    level1_root: Path = LEVEL1_DIR,
    top_count: int = DEFAULT_TOP_COUNT,
) -> dict[str, Any]:
    """Build the committed census payload including the winner and top rows."""
    if top_count <= 0:
        raise ValueError("top_count must be positive")
    records = census_sessions(session_root=session_root, level1_root=level1_root)
    eligible = [record for record in records if record.eligible]
    if len(eligible) < top_count:
        raise RuntimeError(
            f"only {len(eligible)} eligible sessions; need {top_count} for the ranking"
        )
    ranking = [asdict(record) for record in records]
    encoded_ranking = json.dumps(ranking, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    top = ranking[:top_count]
    measured_at_utc = datetime.now(UTC).isoformat()
    session_total = len(records)
    available_camera_group_total = sum(
        len(record.available_cameras) for record in records
    )
    camera_group_counts = {
        camera_group: sum(
            any(
                camera.camera_group == camera_group
                for camera in record.available_cameras
            )
            for record in records
        )
        for camera_group in PLASMA_CAMERA_GROUPS
    }
    manifest_status_counts = {
        status: sum(record.manifest_status == status for record in records)
        for status in sorted({record.manifest_status for record in records})
    }
    demo_family = _family_summary(
        records,
        family_name="demo-rbb",
        definition={
            "camera_group": DEMO_CAMERA_GROUP,
            "frame_height": DEMO_FRAME_HEIGHT,
            "frame_width": DEMO_FRAME_WIDTH,
            "cadence_target_us": DEMO_CADENCE_US,
            "cadence_tolerance_us": DEMO_CADENCE_TOLERANCE_US,
            "cadence_minimum_us": DEMO_CADENCE_US - DEMO_CADENCE_TOLERANCE_US,
            "cadence_maximum_us": DEMO_CADENCE_US + DEMO_CADENCE_TOLERANCE_US,
            "cadence_bounds_inclusive": True,
            "dimension_tolerance_pixels": 0,
        },
        predicate=_is_demo_camera,
        session_total=session_total,
        measured_at_utc=measured_at_utc,
    )
    square_family = _family_summary(
        records,
        family_name="square-rbb",
        definition={
            "camera_group": SQUARE_CAMERA_GROUP,
            "frame_height": SQUARE_FRAME_HEIGHT,
            "frame_width": SQUARE_FRAME_WIDTH,
            "cadence_minimum_us": SQUARE_MIN_CADENCE_US,
            "cadence_maximum_us": SQUARE_MAX_CADENCE_US,
            "cadence_bounds_inclusive": True,
            "dimension_tolerance_pixels": 0,
        },
        predicate=_is_square_camera,
        session_total=session_total,
        measured_at_utc=measured_at_utc,
    )
    return {
        "schema": "carrier-camera-ranking",
        "generated_at_utc": measured_at_utc,
        "measurement": {
            "session_total": session_total,
            "available_camera_group_total": available_camera_group_total,
            "available_camera_group_counts": camera_group_counts,
            "manifest_status_counts": manifest_status_counts,
            "measured_at_utc": measured_at_utc,
            "scope": (
                "All figures in this report use the manifest paths discovered "
                "for this pass; the corpus remains under active production."
            ),
        },
        "session_root": str(Path(session_root)),
        "level1_root": str(Path(level1_root)),
        "camera_groups": list(PLASMA_CAMERA_GROUPS),
        "camera_inventory_fields": [
            "camera_group",
            "frame_height",
            "frame_width",
            "frame_area",
            "frame_count",
            "temporal_span_s",
            "cadence_us",
        ],
        "thomson_groups": list(THOMSON_GROUPS),
        "minimum_converged_slices": MIN_CONVERGED_SLICES,
        "score_definition": {
            "formula": "geometric_mean(value / (value + scale))",
            "scales": SCORE_SCALES,
            "tie_break": "shot ascending",
        },
        "session_count": session_total,
        "eligible_count": len(eligible),
        "excluded_count": len(records) - len(eligible),
        "ranking_sha256": hashlib.sha256(encoded_ranking).hexdigest(),
        "winner": top[0],
        "top_five": top,
        "rbb_geometry_distribution": {
            "cadence_band_width_us": CADENCE_BAND_WIDTH_US,
            "cadence_band_rule": (
                "nearest 10 microseconds; lower bound inclusive, upper bound exclusive"
            ),
            "rows": _rbb_geometry_distribution(
                records,
                session_total=session_total,
                measured_at_utc=measured_at_utc,
            ),
            "session_total": session_total,
            "measured_at_utc": measured_at_utc,
        },
        "demo_camera_family": demo_family,
        "square_camera_family": square_family,
        "trainability_scope": (
            "Camera metadata does not encode camera-topology cohort membership. "
            "This report measures geometry and counts only; cohort exclusion is "
            "applied elsewhere and trainability is not inferred here."
        ),
        "ranking": ranking,
    }


def write_ranking(payload: dict[str, Any], output_path: Path) -> None:
    """Write *payload* atomically as indented JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-root", type=Path, default=DEFAULT_SESSION_ROOT)
    parser.add_argument("--level1-root", type=Path, default=LEVEL1_DIR)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-count", type=int, default=DEFAULT_TOP_COUNT)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the census command."""
    args = _parser().parse_args(argv)
    payload = build_ranking_payload(
        session_root=args.session_root,
        level1_root=args.level1_root,
        top_count=args.top_count,
    )
    write_ranking(payload, args.output)
    winner = payload["winner"]
    print(
        json.dumps(
            {
                "session_count": payload["session_count"],
                "eligible_count": payload["eligible_count"],
                "winner": winner["shot"],
                "winner_camera": winner["camera_group"],
                "winner_score": winner["score"],
                "demo_camera_family_shot_count": payload["demo_camera_family"][
                    "shot_count"
                ],
                "square_camera_family_shot_count": payload["square_camera_family"][
                    "shot_count"
                ],
                "measured_at_utc": payload["measurement"]["measured_at_utc"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_TOP_COUNT",
    "CADENCE_BAND_WIDTH_US",
    "DEMO_CADENCE_TOLERANCE_US",
    "DEMO_CADENCE_US",
    "DEMO_CAMERA_GROUP",
    "DEMO_FRAME_HEIGHT",
    "DEMO_FRAME_WIDTH",
    "MIN_CONVERGED_SLICES",
    "PLASMA_CAMERA_GROUPS",
    "SCORE_SCALES",
    "SQUARE_CAMERA_GROUP",
    "SQUARE_FRAME_HEIGHT",
    "SQUARE_FRAME_WIDTH",
    "SQUARE_MAX_CADENCE_US",
    "SQUARE_MIN_CADENCE_US",
    "THOMSON_GROUPS",
    "CameraMetrics",
    "SessionRanking",
    "build_ranking_payload",
    "camera_richness_score",
    "census_sessions",
    "main",
    "write_ranking",
]
