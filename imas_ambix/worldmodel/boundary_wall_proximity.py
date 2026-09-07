"""Measure whether admitted Nova flux boundaries remain attached to the wall."""

from __future__ import annotations

import argparse
import json
import subprocess
import textwrap
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from imas_ambix.data.paths import LEVEL1_DIR
from imas_ambix.latent.wall_mask import _inside_polygon
from imas_ambix.worldmodel import flux_decoder_video
from imas_ambix.worldmodel.camera_topology_targets import _segment_intersections
from imas_ambix.worldmodel.flux_conditioning import (
    MAST_LIMITER_R,
    MAST_LIMITER_Z,
)
from imas_ambix.worldmodel.flux_label_dataset import (
    DEFAULT_SESSION_ROOT,
    EXPECTED_CARRIER_IDENTITY,
    EXPECTED_POLICY_DIGEST,
    _conditioned_row_is_free,
    _conditioning_improved_centroid,
    _load_companion,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

FROZEN_CARRIER_SHOTS = (21978, 21983, 21985, 21986, 21989, 22086)
PROVIDED_MISSING_NOMINAL_COUNTS = {
    21978: 22,
    21983: 24,
    21985: 22,
    21986: 16,
    21989: 25,
    22086: 10,
}
FLOATING_DISTANCE_MM = 50.0
DEFAULT_OUTPUT_DIR = Path("docs/figures/physics-carried-playable-plasma/label-quality")
DEFAULT_REPAIRED_SESSION_ROOT = Path(
    "/work/projects/imas_gpu/sophelio/labeller_sessions/"
    "boundary-repair-validation-20260907T1124Z"
)


@dataclass(frozen=True, slots=True)
class BoundaryMeasurement:
    """Wall proximity and area for one slice admitted by label semantics."""

    shot_id: int
    manifest_row: int
    session_index: int
    time_s: float
    diverted: bool
    recorded_conditioned: bool
    conditioned: bool
    reclassified_as_free: bool
    conditioned_branch_guard_ok: bool
    surface_category: str
    selected_surface_index: int | None
    selected_surface_psi_norm: float | None
    selected_surface_raw_vertex_count: int | None
    selected_surface_canonical_vertex_count: int | None
    boundary_to_limiter_distance_mm: float | None
    boundary_area_m2: float | None
    boundary_points_inside_limiter: bool | None
    floating_boundary: bool | None
    measurement_error: str | None


SurfaceCategory = Literal[
    "nominal_outer_surface",
    "fallback_outermost_finite_surface",
    "no_qualifying_surface",
]


@dataclass(frozen=True, slots=True)
class SurfaceSelection:
    """The outermost usable flux-surface row and how it was selected."""

    category: SurfaceCategory
    index: int | None
    psi_norm: float | None
    r: np.ndarray | None
    z: np.ndarray | None


def _source_revision() -> str:
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _finite_polygon(r: np.ndarray, z: np.ndarray, *, name: str) -> np.ndarray:
    r_values = np.asarray(r, dtype=np.float64).reshape(-1)
    z_values = np.asarray(z, dtype=np.float64).reshape(-1)
    if r_values.shape != z_values.shape:
        raise ValueError(f"{name} R and Z coordinates must align")
    points = np.column_stack((r_values, z_values))
    points = points[np.isfinite(points).all(axis=1)]
    if points.shape[0] > 1 and np.allclose(points[0], points[-1]):
        points = points[:-1]
    if points.shape[0] < 3:
        raise ValueError(f"{name} needs at least three finite vertices")
    return points


def _surface_vertex_counts(r: np.ndarray, z: np.ndarray) -> tuple[int, int]:
    r_values = np.asarray(r, dtype=np.float64).reshape(-1)
    z_values = np.asarray(z, dtype=np.float64).reshape(-1)
    if r_values.shape != z_values.shape:
        raise ValueError("surface R and Z coordinates must align")
    finite = np.column_stack((r_values, z_values))
    finite = finite[np.isfinite(finite).all(axis=1)]
    raw_count = int(finite.shape[0])
    canonical_count = raw_count
    if raw_count > 1 and np.allclose(finite[0], finite[-1]):
        canonical_count -= 1
    return raw_count, canonical_count


def _vertices_to_segments_distance(vertices: np.ndarray, polygon: np.ndarray) -> float:
    starts = polygon
    vectors = np.roll(polygon, -1, axis=0) - starts
    squared_lengths = np.einsum("ij,ij->i", vectors, vectors)
    valid = squared_lengths > 0.0
    if not np.any(valid):
        raise ValueError("polygon has no non-zero-length edge")
    starts = starts[valid]
    vectors = vectors[valid]
    squared_lengths = squared_lengths[valid]
    offsets = vertices[:, None, :] - starts[None, :, :]
    positions = np.einsum("vsi,si->vs", offsets, vectors) / squared_lengths
    positions = np.clip(positions, 0.0, 1.0)
    nearest = starts[None, :, :] + positions[:, :, None] * vectors[None, :, :]
    distances = np.linalg.norm(vertices[:, None, :] - nearest, axis=2)
    return float(np.min(distances))


def boundary_polygon_metrics(
    boundary_r: np.ndarray,
    boundary_z: np.ndarray,
    *,
    limiter_r: np.ndarray = MAST_LIMITER_R,
    limiter_z: np.ndarray = MAST_LIMITER_Z,
) -> dict[str, float | bool]:
    """Return minimum polyline distance, enclosed area, and containment."""
    boundary = _finite_polygon(boundary_r, boundary_z, name="boundary")
    limiter = _finite_polygon(limiter_r, limiter_z, name="limiter")
    closed_boundary = np.vstack((boundary, boundary[0]))
    intersections = _segment_intersections([closed_boundary], limiter)
    distance_m = (
        0.0
        if intersections.size
        else min(
            _vertices_to_segments_distance(boundary, limiter),
            _vertices_to_segments_distance(limiter, boundary),
        )
    )
    area_m2 = 0.5 * abs(
        float(
            np.sum(
                boundary[:, 0] * np.roll(boundary[:, 1], -1)
                - boundary[:, 1] * np.roll(boundary[:, 0], -1)
            )
        )
    )
    inside = _inside_polygon(
        boundary[:, 0], boundary[:, 1], limiter[:, 0], limiter[:, 1]
    )
    return {
        "boundary_to_limiter_distance_m": distance_m,
        "boundary_area_m2": area_m2,
        "boundary_points_inside_limiter": bool(np.all(inside)),
    }


def compare_boundary_polygons(
    baseline_r: np.ndarray,
    baseline_z: np.ndarray,
    repaired_r: np.ndarray,
    repaired_z: np.ndarray,
    *,
    limiter_r: np.ndarray = MAST_LIMITER_R,
    limiter_z: np.ndarray = MAST_LIMITER_Z,
) -> dict[str, Any]:
    """Compare two boundaries after canonicalizing a closing vertex."""
    baseline_metric = boundary_polygon_metrics(
        baseline_r, baseline_z, limiter_r=limiter_r, limiter_z=limiter_z
    )
    repaired_metric = boundary_polygon_metrics(
        repaired_r, repaired_z, limiter_r=limiter_r, limiter_z=limiter_z
    )
    baseline_raw, baseline_canonical = _surface_vertex_counts(baseline_r, baseline_z)
    repaired_raw, repaired_canonical = _surface_vertex_counts(repaired_r, repaired_z)
    distance_difference_mm = 1000.0 * (
        float(repaired_metric["boundary_to_limiter_distance_m"])
        - float(baseline_metric["boundary_to_limiter_distance_m"])
    )
    area_difference_m2 = float(repaired_metric["boundary_area_m2"]) - float(
        baseline_metric["boundary_area_m2"]
    )
    return {
        "baseline_raw_vertex_count": baseline_raw,
        "baseline_canonical_vertex_count": baseline_canonical,
        "repaired_raw_vertex_count": repaired_raw,
        "repaired_canonical_vertex_count": repaired_canonical,
        "baseline_boundary_to_limiter_distance_mm": 1000.0
        * float(baseline_metric["boundary_to_limiter_distance_m"]),
        "repaired_boundary_to_limiter_distance_mm": 1000.0
        * float(repaired_metric["boundary_to_limiter_distance_m"]),
        "boundary_to_limiter_distance_difference_mm": distance_difference_mm,
        "baseline_boundary_area_m2": float(baseline_metric["boundary_area_m2"]),
        "repaired_boundary_area_m2": float(repaired_metric["boundary_area_m2"]),
        "boundary_area_difference_m2": area_difference_m2,
        "geometrically_identical": bool(
            distance_difference_mm == 0.0 and area_difference_m2 == 0.0
        ),
    }


def _load_manifest(path: Path, shot_id: int) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(f"{path} is not an atomically complete session")
    if int(manifest.get("shot", -1)) != shot_id:
        raise ValueError(f"{path} shot identity does not match {shot_id}")
    if str(manifest.get("policy_digest", "")) != EXPECTED_POLICY_DIGEST:
        raise ValueError(f"{path} does not carry the pinned policy digest")
    if str(manifest.get("carrier_identity", "")) != EXPECTED_CARRIER_IDENTITY:
        raise ValueError(f"{path} does not carry the pinned carrier identity")
    if not isinstance(manifest.get("slices"), list):
        raise ValueError(f"{path} has no slice-row list")
    return manifest


def _slice_array(session: Any, name: str, index: int) -> np.ndarray:
    value = session[name]
    if "time" in value.dims:
        value = value.isel(time=index)
    return np.asarray(value, dtype=np.float64)


def select_outermost_finite_surface(
    levels: np.ndarray, surface_r: np.ndarray, surface_z: np.ndarray
) -> SurfaceSelection:
    """Prefer a finite psi_norm=1 surface, then the highest usable level."""
    levels = np.asarray(levels, dtype=np.float64).reshape(-1)
    surface_r = np.asarray(surface_r, dtype=np.float64)
    surface_z = np.asarray(surface_z, dtype=np.float64)
    if surface_r.shape != surface_z.shape or surface_r.shape[0] != levels.size:
        raise ValueError("flux surfaces and psi_norm levels do not align")
    finite_counts = np.count_nonzero(
        np.isfinite(surface_r) & np.isfinite(surface_z), axis=1
    )
    finite_levels = np.flatnonzero(np.isfinite(levels))
    nominal_index: int | None = None
    if finite_levels.size:
        closest = int(finite_levels[np.argmin(np.abs(levels[finite_levels] - 1.0))])
        if np.isclose(levels[closest], 1.0, rtol=0.0, atol=1.0e-6):
            nominal_index = closest
            if finite_counts[closest] >= 3:
                return SurfaceSelection(
                    category="nominal_outer_surface",
                    index=closest,
                    psi_norm=float(levels[closest]),
                    r=surface_r[closest],
                    z=surface_z[closest],
                )
    qualifying = np.flatnonzero(np.isfinite(levels) & (finite_counts >= 3))
    if nominal_index is not None:
        qualifying = qualifying[qualifying != nominal_index]
    if not qualifying.size:
        return SurfaceSelection(
            category="no_qualifying_surface",
            index=None,
            psi_norm=None,
            r=None,
            z=None,
        )
    selected = int(qualifying[np.argmax(levels[qualifying])])
    return SurfaceSelection(
        category="fallback_outermost_finite_surface",
        index=selected,
        psi_norm=float(levels[selected]),
        r=surface_r[selected],
        z=surface_z[selected],
    )


def _select_surface(session: Any, index: int) -> SurfaceSelection:
    levels = _slice_array(session, "flux_surface_psi_norm", index)
    surface_r = _slice_array(session, "flux_surface_r", index)
    surface_z = _slice_array(session, "flux_surface_z", index)
    return select_outermost_finite_surface(levels, surface_r, surface_z)


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if not finite.size:
        return {
            "count": 0,
            "minimum": None,
            "median": None,
            "p90": None,
            "maximum": None,
        }
    return {
        "count": int(finite.size),
        "minimum": float(np.min(finite)),
        "median": float(np.median(finite)),
        "p90": float(np.quantile(finite, 0.9)),
        "maximum": float(np.max(finite)),
    }


def summarize_measurements(
    measurements: Sequence[BoundaryMeasurement],
) -> dict[str, Any]:
    """Summarize admitted measurements overall and by diverted state."""
    result: dict[str, Any] = {}
    groups = {
        "all": list(measurements),
        "diverted_false": [item for item in measurements if not item.diverted],
        "diverted_true": [item for item in measurements if item.diverted],
    }
    for name, group in groups.items():
        measured = [
            item for item in group if item.boundary_to_limiter_distance_mm is not None
        ]
        floating_count = sum(item.floating_boundary is True for item in group)
        count = len(group)
        category_counts = {
            category: sum(item.surface_category == category for item in group)
            for category in (
                "nominal_outer_surface",
                "fallback_outermost_finite_surface",
                "no_qualifying_surface",
            )
        }
        category_fractions = {
            category: value / count if count else 0.0
            for category, value in category_counts.items()
        }
        missing_outer_count = (
            category_counts["fallback_outermost_finite_surface"]
            + category_counts["no_qualifying_surface"]
        )
        result[name] = {
            "admitted_slice_count": count,
            "measured_slice_count": len(measured),
            "measurement_unavailable_count": count - len(measured),
            "surface_category_counts": category_counts,
            "surface_category_fractions": category_fractions,
            "finding_category_counts": {
                "floating_boundary": floating_count,
                "nominal_outer_surface_missing": missing_outer_count,
                "no_qualifying_surface": category_counts["no_qualifying_surface"],
            },
            "nominal_outer_surface_missing_count": missing_outer_count,
            "nominal_outer_surface_missing_fraction": (
                missing_outer_count / count if count else 0.0
            ),
            "boundary_to_limiter_distance_mm": _distribution(
                [
                    item.boundary_to_limiter_distance_mm
                    for item in measured
                    if item.boundary_to_limiter_distance_mm is not None
                ]
            ),
            "boundary_area_m2": _distribution(
                [
                    item.boundary_area_m2
                    for item in measured
                    if item.boundary_area_m2 is not None
                ]
            ),
            "floating_boundary_count": floating_count,
            "floating_boundary_fraction": floating_count / count if count else 0.0,
            "floating_boundary_fraction_of_measured": (
                floating_count / len(measured) if measured else 0.0
            ),
            "boundary_not_fully_inside_count": sum(
                item.boundary_points_inside_limiter is False for item in group
            ),
        }
    return result


def score_session(
    shot_id: int, *, session_root: Path = DEFAULT_SESSION_ROOT
) -> dict[str, Any]:
    """Measure every slice admitted by the label-semantic predicate."""
    import xarray as xr  # noqa: PLC0415

    root = Path(session_root)
    manifest_path = root / f"{shot_id}.manifest.json"
    companion_path = root / f"{shot_id}.npz"
    session_path = root / f"{shot_id}.nc"
    manifest = _load_manifest(manifest_path, shot_id)
    slices = manifest["slices"]
    (
        companion_rows,
        companion_times,
        conditioned,
        guard_ok,
        free_centroid_error_m,
        conditioned_centroid_error_m,
    ) = _load_companion(companion_path, slices)
    row_to_session = {
        int(row): index for index, row in enumerate(companion_rows.tolist())
    }

    measurements: list[BoundaryMeasurement] = []
    dropped = {
        "unwritten": sum(not bool(row.get("written", False)) for row in slices),
        "semantically_unconverged": 0,
        "conditioned_guard_failed": 0,
        "conditioning_comparison_unavailable_rows": 0,
        "conditioning_ineffective_rows": 0,
    }
    reclassified_count = 0
    with xr.open_dataset(session_path, group="steering", engine="h5netcdf") as session:
        session_times = np.asarray(session["time"], dtype=np.float64).reshape(-1)
        surface_storage_vertex_count = int(session["flux_surface_r"].shape[-1])
        if session_times.shape != companion_times.shape or not np.allclose(
            session_times, companion_times, rtol=0.0, atol=1.0e-9
        ):
            raise ValueError(f"{session_path} times do not align with its companion")
        for row in slices:
            if not bool(row.get("written", False)):
                continue
            session_index = row_to_session[int(row["row"])]
            recorded_conditioned = bool(conditioned[session_index])
            reclassified_as_free = _conditioned_row_is_free(
                row, conditioned=recorded_conditioned
            )
            reclassified_count += int(reclassified_as_free)
            semantically_converged = bool(
                row.get("free_converged", False)
                if reclassified_as_free
                else row.get("converged", False)
            )
            if not semantically_converged:
                dropped["semantically_unconverged"] += 1
                continue
            genuinely_conditioned = recorded_conditioned and not reclassified_as_free
            branch_ok = bool(guard_ok[session_index])
            if genuinely_conditioned and not branch_ok:
                dropped["conditioned_guard_failed"] += 1
                continue
            if genuinely_conditioned:
                free_error = float(free_centroid_error_m[session_index])
                conditioned_error = float(conditioned_centroid_error_m[session_index])
                if not np.isfinite((free_error, conditioned_error)).all():
                    dropped["conditioning_comparison_unavailable_rows"] += 1
                    continue
                if not _conditioning_improved_centroid(free_error, conditioned_error):
                    dropped["conditioning_ineffective_rows"] += 1
                    continue

            diverted = bool(_slice_array(session, "diverted", session_index).item())
            measurement_error: str | None = None
            distance_mm: float | None = None
            area_m2: float | None = None
            boundary_inside: bool | None = None
            selection = _select_surface(session, session_index)
            raw_vertex_count: int | None = None
            canonical_vertex_count: int | None = None
            if selection.r is not None and selection.z is not None:
                raw_vertex_count, canonical_vertex_count = _surface_vertex_counts(
                    selection.r, selection.z
                )
                metric = boundary_polygon_metrics(selection.r, selection.z)
                distance_mm = 1000.0 * float(metric["boundary_to_limiter_distance_m"])
                area_m2 = float(metric["boundary_area_m2"])
                boundary_inside = bool(metric["boundary_points_inside_limiter"])
            else:
                measurement_error = "no flux-surface level has three finite vertices"
            measurements.append(
                BoundaryMeasurement(
                    shot_id=int(shot_id),
                    manifest_row=int(row["row"]),
                    session_index=session_index,
                    time_s=float(companion_times[session_index]),
                    diverted=diverted,
                    recorded_conditioned=recorded_conditioned,
                    conditioned=genuinely_conditioned,
                    reclassified_as_free=reclassified_as_free,
                    conditioned_branch_guard_ok=branch_ok,
                    surface_category=selection.category,
                    selected_surface_index=selection.index,
                    selected_surface_psi_norm=selection.psi_norm,
                    selected_surface_raw_vertex_count=raw_vertex_count,
                    selected_surface_canonical_vertex_count=canonical_vertex_count,
                    boundary_to_limiter_distance_mm=distance_mm,
                    boundary_area_m2=area_m2,
                    boundary_points_inside_limiter=boundary_inside,
                    floating_boundary=(
                        None
                        if distance_mm is None
                        else distance_mm > FLOATING_DISTANCE_MM
                    ),
                    measurement_error=measurement_error,
                )
            )

    return {
        "shot_id": int(shot_id),
        "manifest_path": str(manifest_path.resolve()),
        "session_path": str(session_path.resolve()),
        "manifest_slice_count": len(slices),
        "written_slice_count": int(companion_rows.size),
        "surface_storage_vertex_count": surface_storage_vertex_count,
        "reclassified_as_free_count": reclassified_count,
        "dropped_slices": dropped,
        "summary": summarize_measurements(measurements),
        "slices": [asdict(item) for item in measurements],
    }


def _metric_view(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "surface_category": item["surface_category"],
        "selected_surface_index": item["selected_surface_index"],
        "selected_surface_psi_norm": item["selected_surface_psi_norm"],
        "raw_vertex_count": item["selected_surface_raw_vertex_count"],
        "canonical_vertex_count": item["selected_surface_canonical_vertex_count"],
        "boundary_to_limiter_distance_mm": item["boundary_to_limiter_distance_mm"],
        "boundary_area_m2": item["boundary_area_m2"],
        "measurement_error": item["measurement_error"],
    }


def _difference_summary(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    distance = np.asarray(
        [
            item["boundary_to_limiter_distance_difference_mm"]
            for item in items
            if item["boundary_to_limiter_distance_difference_mm"] is not None
        ],
        dtype=np.float64,
    )
    area = np.asarray(
        [
            item["boundary_area_difference_m2"]
            for item in items
            if item["boundary_area_difference_m2"] is not None
        ],
        dtype=np.float64,
    )
    return {
        "paired_slice_count": len(items),
        "measurable_pair_count": int(distance.size),
        "exact_zero_distance_difference_count": int(np.count_nonzero(distance == 0.0)),
        "exact_zero_area_difference_count": int(np.count_nonzero(area == 0.0)),
        "maximum_absolute_distance_difference_mm": (
            float(np.max(np.abs(distance))) if distance.size else None
        ),
        "maximum_absolute_area_difference_m2": (
            float(np.max(np.abs(area))) if area.size else None
        ),
    }


def _vertex_count_summary(
    items: Sequence[Mapping[str, Any]], side: str
) -> dict[str, Any]:
    raw: dict[str, int] = {}
    canonical: dict[str, int] = {}
    closing_vertex_count = 0
    for item in items:
        side_item = item[side]
        raw_value = side_item["raw_vertex_count"]
        canonical_value = side_item["canonical_vertex_count"]
        raw_key = "unavailable" if raw_value is None else str(int(raw_value))
        canonical_key = (
            "unavailable" if canonical_value is None else str(int(canonical_value))
        )
        raw[raw_key] = raw.get(raw_key, 0) + 1
        canonical[canonical_key] = canonical.get(canonical_key, 0) + 1
        closing_vertex_count += int(
            raw_value is not None
            and canonical_value is not None
            and int(raw_value) == int(canonical_value) + 1
        )
    return {
        "raw_finite_vertex_count_frequency": raw,
        "canonical_vertex_count_frequency": canonical,
        "duplicated_closing_vertex_slice_count": closing_vertex_count,
    }


def _renderer_seed_context(
    shot_id: int, *, session_root: Path, admitted_slices: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    session_path = Path(session_root) / f"{shot_id}.nc"
    session = flux_decoder_video._read_session(session_path)
    session_times = np.asarray(session["time"], dtype=np.float64).reshape(-1)
    mode, manifest_shot, selected, manifest_count = (
        flux_decoder_video._manifest_selection(session_path, session_times.size)
    )
    if mode != "labeller" or manifest_shot != shot_id:
        raise ValueError(f"{session_path} is not the requested labeller session")
    camera_times = flux_decoder_video._camera_times(shot_id, level1_root=LEVEL1_DIR)
    _, camera_deltas = flux_decoder_video._nearest_indices(
        camera_times, session_times[selected]
    )
    selected = [
        index
        for index, delta in zip(selected, camera_deltas, strict=True)
        if abs(float(delta)) <= flux_decoder_video.MAX_CAMERA_DELTA_SECONDS
    ]
    seed = flux_decoder_video._resolve_seed_window(
        selected,
        session_times,
        session_times,
        camera_times,
        requested_start_slice=0,
    )
    first_admitted = min(admitted_slices, key=lambda item: int(item["session_index"]))
    cold_index = 0
    seed_indices = [int(value) for value in seed.session_slice_indices]
    inside = cold_index in seed_indices
    adjacent = bool(
        not inside
        and seed_indices
        and cold_index in {min(seed_indices) - 1, max(seed_indices) + 1}
    )
    if inside:
        relation = "inside_seed_window"
    elif adjacent:
        relation = "adjacent_before_seed_window"
    else:
        relation = "outside_seed_window"
    admitted_indices = {int(item["session_index"]) for item in admitted_slices}
    return {
        "manifest_slice_count": manifest_count,
        "cold_slice": {
            "session_index": cold_index,
            "time_s": float(session_times[cold_index]),
            "admitted": cold_index in admitted_indices,
            "relation_to_seed_window": relation,
        },
        "first_admitted_slice": {
            "session_index": int(first_admitted["session_index"]),
            "manifest_row": int(first_admitted["manifest_row"]),
            "time_s": float(first_admitted["time_s"]),
        },
        "first_renderer_slice": {
            "session_index": int(seed.selected[0]),
            "time_s": float(session_times[seed.selected[0]]),
        },
        "seed_window": {
            "history_spacing_s": flux_decoder_video.HISTORY_SPACING_SECONDS,
            "session_slice_indices": seed_indices,
            "query_times_s": seed.query_times.tolist(),
            "camera_frame_indices": seed.camera_frame_indices.tolist(),
            "camera_frame_times_s": (
                seed.query_times + seed.camera_time_deltas
            ).tolist(),
            "camera_time_deltas_s": seed.camera_time_deltas.tolist(),
        },
    }


def _compare_shot(
    shot_id: int, *, baseline_root: Path, repaired_root: Path
) -> dict[str, Any]:
    baseline = score_session(shot_id, session_root=baseline_root)
    repaired = score_session(shot_id, session_root=repaired_root)
    baseline_by_row = {int(item["manifest_row"]): item for item in baseline["slices"]}
    repaired_by_row = {int(item["manifest_row"]): item for item in repaired["slices"]}
    baseline_rows = set(baseline_by_row)
    repaired_rows = set(repaired_by_row)
    paired: list[dict[str, Any]] = []
    for row in sorted(baseline_rows & repaired_rows):
        before = baseline_by_row[row]
        after = repaired_by_row[row]
        if not np.isclose(
            float(before["time_s"]), float(after["time_s"]), rtol=0.0, atol=1.0e-9
        ):
            raise ValueError(f"shot {shot_id} row {row} has mismatched slice times")
        if bool(before["diverted"]) != bool(after["diverted"]):
            raise ValueError(f"shot {shot_id} row {row} changed diverted class")
        before_distance = before["boundary_to_limiter_distance_mm"]
        after_distance = after["boundary_to_limiter_distance_mm"]
        before_area = before["boundary_area_m2"]
        after_area = after["boundary_area_m2"]
        measurable = all(
            value is not None
            for value in (before_distance, after_distance, before_area, after_area)
        )
        paired.append(
            {
                "manifest_row": row,
                "session_index": int(before["session_index"]),
                "time_s": float(before["time_s"]),
                "diverted": bool(before["diverted"]),
                "baseline": _metric_view(before),
                "repaired": _metric_view(after),
                "boundary_to_limiter_distance_difference_mm": (
                    float(after_distance) - float(before_distance)
                    if measurable
                    else None
                ),
                "boundary_area_difference_m2": (
                    float(after_area) - float(before_area) if measurable else None
                ),
            }
        )
    summaries = {
        "all": _difference_summary(paired),
        "diverted_false": _difference_summary(
            [item for item in paired if not item["diverted"]]
        ),
        "diverted_true": _difference_summary(
            [item for item in paired if item["diverted"]]
        ),
    }
    return {
        "shot_id": shot_id,
        "admitted_slice_counts": {
            "baseline": len(baseline["slices"]),
            "repaired": len(repaired["slices"]),
            "paired": len(paired),
            "baseline_only": len(baseline_rows - repaired_rows),
            "repaired_only": len(repaired_rows - baseline_rows),
        },
        "surface_storage_vertex_counts": {
            "baseline": int(baseline["surface_storage_vertex_count"]),
            "repaired": int(repaired["surface_storage_vertex_count"]),
        },
        "selected_surface_vertex_counts": {
            "baseline": _vertex_count_summary(paired, "baseline"),
            "repaired": _vertex_count_summary(paired, "repaired"),
        },
        "baseline_only_slices": [
            {
                "manifest_row": row,
                "session_index": int(baseline_by_row[row]["session_index"]),
                "time_s": float(baseline_by_row[row]["time_s"]),
            }
            for row in sorted(baseline_rows - repaired_rows)
        ],
        "repaired_only_slices": [
            {
                "manifest_row": row,
                "session_index": int(repaired_by_row[row]["session_index"]),
                "time_s": float(repaired_by_row[row]["time_s"]),
            }
            for row in sorted(repaired_rows - baseline_rows)
        ],
        "difference_summary": summaries,
        "seed_location": _renderer_seed_context(
            shot_id, session_root=repaired_root, admitted_slices=repaired["slices"]
        ),
        "paired_slices": paired,
    }


def build_repair_null_report(
    *,
    baseline_root: Path = DEFAULT_SESSION_ROOT,
    repaired_root: Path = DEFAULT_REPAIRED_SESSION_ROOT,
    shot_ids: Sequence[int] = FROZEN_CARRIER_SHOTS,
) -> dict[str, Any]:
    """Compare original and repaired session geometry on common carriers."""
    shots = [
        _compare_shot(
            int(shot), baseline_root=baseline_root, repaired_root=repaired_root
        )
        for shot in shot_ids
    ]
    pairs = [item for shot in shots for item in shot["paired_slices"]]
    summaries = {
        "all": _difference_summary(pairs),
        "diverted_false": _difference_summary(
            [item for item in pairs if not item["diverted"]]
        ),
        "diverted_true": _difference_summary(
            [item for item in pairs if item["diverted"]]
        ),
    }
    cold_inside = sum(
        shot["seed_location"]["cold_slice"]["relation_to_seed_window"]
        == "inside_seed_window"
        for shot in shots
    )
    cold_adjacent = sum(
        shot["seed_location"]["cold_slice"]["relation_to_seed_window"]
        == "adjacent_before_seed_window"
        for shot in shots
    )
    distance_tolerance_mm = 1.0e-9
    area_tolerance_m2 = 1.0e-12
    distance_maximum = summaries["all"]["maximum_absolute_distance_difference_mm"]
    area_maximum = summaries["all"]["maximum_absolute_area_difference_m2"]
    reproduces_null = bool(
        distance_maximum is not None
        and area_maximum is not None
        and float(distance_maximum) <= distance_tolerance_mm
        and float(area_maximum) <= area_tolerance_m2
    )
    verdict = (
        f"The independent wall-distance and enclosed-area instrument "
        f"{'reproduces' if reproduces_null else 'does not reproduce'} the reported "
        f"repair null across {len(pairs)} paired admitted slices; {cold_inside} of "
        f"{len(shots)} unrepairable cold slices fall inside the renderer seed window "
        f"and {cold_adjacent} sit immediately adjacent before it."
    )
    return {
        "schema": "repair-null-check",
        "generated_at": datetime.now(UTC).isoformat(),
        "source_revision": _source_revision(),
        "baseline_session_root": str(Path(baseline_root).resolve()),
        "repaired_session_root": str(Path(repaired_root).resolve()),
        "shots_requested": [int(shot) for shot in shot_ids],
        "comparison": (
            "Slices are paired by manifest row and equal session time. Boundary "
            "distance and area are computed after removing a duplicated closing "
            "vertex, while raw and canonical vertex counts remain explicit."
        ),
        "null_tolerances": {
            "boundary_to_limiter_distance_mm": distance_tolerance_mm,
            "boundary_area_m2": area_tolerance_m2,
        },
        "reported_vertex_serialization": {
            "baseline_vertex_count": 64,
            "repaired_vertex_count": 65,
            "interpretation": (
                "A repaired polyline may append its first point as a closing vertex; "
                "canonical counts and geometry remove that duplicate while raw "
                "observations remain reported."
            ),
        },
        "observed_selected_surface_vertex_counts": {
            "baseline": _vertex_count_summary(pairs, "baseline"),
            "repaired": _vertex_count_summary(pairs, "repaired"),
        },
        "difference_summary": summaries,
        "cold_slice_seed_summary": {
            "shot_count": len(shots),
            "inside_seed_window_count": cold_inside,
            "adjacent_before_seed_window_count": cold_adjacent,
            "outside_seed_window_count": len(shots) - cold_inside - cold_adjacent,
        },
        "reproduces_reported_null": reproduces_null,
        "shots": shots,
        "verdict": verdict,
    }


def build_report(
    *,
    session_root: Path = DEFAULT_SESSION_ROOT,
    shot_ids: Sequence[int] = FROZEN_CARRIER_SHOTS,
) -> dict[str, Any]:
    """Measure all requested carrier sessions and assemble the receipt."""
    shots = [score_session(int(shot), session_root=session_root) for shot in shot_ids]
    all_measurements = [
        BoundaryMeasurement(**item) for shot in shots for item in shot["slices"]
    ]
    aggregate = summarize_measurements(all_measurements)
    overall = aggregate["all"]
    non_diverted = aggregate["diverted_false"]
    diverted = aggregate["diverted_true"]
    floating_count = int(overall["floating_boundary_count"])
    total = int(overall["admitted_slice_count"])
    fraction = float(overall["floating_boundary_fraction"])
    unavailable = int(overall["measurement_unavailable_count"])
    missing_outer_count = int(overall["nominal_outer_surface_missing_count"])
    fallback_count = int(
        overall["surface_category_counts"]["fallback_outermost_finite_surface"]
    )
    confined = (
        floating_count > 0
        and int(non_diverted["floating_boundary_count"]) == floating_count
        and int(diverted["floating_boundary_count"]) == 0
    )
    confinement = (
        "the effect is confined to the non-diverted group as predicted"
        if confined
        else "the effect is not confined to the non-diverted group"
    )
    missing_confined = (
        missing_outer_count > 0
        and int(non_diverted["nominal_outer_surface_missing_count"])
        == missing_outer_count
        and int(diverted["nominal_outer_surface_missing_count"]) == 0
    )
    missing_confinement = (
        "the missing nominal surfaces are confined to the non-diverted group"
        if missing_confined
        else "the missing nominal surfaces are not confined to the non-diverted group"
    )
    verdict = (
        f"{floating_count}/{total} trainable slices ({fraction:.1%}) carry a "
        f"measured floating boundary beyond {FLOATING_DISTANCE_MM:.0f} mm; "
        f"{missing_outer_count}/{total} lack a finite nominal outer surface "
        f"({fallback_count} use a visible fallback and {unavailable} have no "
        f"qualifying level); {confinement}, and {missing_confinement}."
    )
    observed_missing_counts = {
        int(shot["shot_id"]): int(
            shot["summary"]["all"]["nominal_outer_surface_missing_count"]
        )
        for shot in shots
    }
    supplied_total = sum(PROVIDED_MISSING_NOMINAL_COUNTS.values())
    observed_total = sum(observed_missing_counts.values())
    return {
        "schema": "boundary-wall-proximity",
        "generated_at": datetime.now(UTC).isoformat(),
        "source_revision": _source_revision(),
        "session_root": str(Path(session_root).resolve()),
        "shots_requested": [int(shot) for shot in shot_ids],
        "pins": {
            "policy_digest": EXPECTED_POLICY_DIGEST,
            "carrier_identity": EXPECTED_CARRIER_IDENTITY,
        },
        "limiter": {
            "source": "imas_ambix.worldmodel.flux_conditioning.MAST_LIMITER_R/Z",
            "vertex_count": len(MAST_LIMITER_R),
            "containment_helper": "imas_ambix.latent.wall_mask._inside_polygon",
        },
        "admission_rule": (
            "written, semantically converged rows; conditioned exception with zero "
            "trips is reclassified as free; genuinely conditioned rows require a "
            "true branch guard and a finite, materially improved centroid error"
        ),
        "measurement": (
            "minimum Euclidean distance between the psi_norm=1 boundary polyline "
            "and the 36-vertex limiter polygon, plus shoelace enclosed area; when "
            "psi_norm=1 has fewer than three finite vertices, use the highest "
            "psi_norm level with at least three finite vertices and record it"
        ),
        "floating_boundary_threshold_mm": FLOATING_DISTANCE_MM,
        "missing_nominal_surface_census_reconciliation": {
            "provided_total": supplied_total,
            "current_total": observed_total,
            "total_delta": observed_total - supplied_total,
            "provided_per_shot": {
                str(shot): count
                for shot, count in PROVIDED_MISSING_NOMINAL_COUNTS.items()
            },
            "current_per_shot": {
                str(shot): count for shot, count in observed_missing_counts.items()
            },
            "per_shot_delta": {
                str(shot): observed_missing_counts.get(shot, 0) - count
                for shot, count in PROVIDED_MISSING_NOMINAL_COUNTS.items()
            },
            "statement": (
                f"The supplied diagnostic counted {supplied_total} missing nominal "
                f"surfaces; the pinned files currently contain {observed_total}."
            ),
        },
        "shots": shots,
        "aggregate": aggregate,
        "floating_confined_to_non_diverted": confined,
        "missing_outer_surface_confined_to_non_diverted": missing_confined,
        "verdict": verdict,
    }


def write_figure(report: Mapping[str, Any], output: Path) -> None:
    """Draw distance distributions and distance-versus-area measurements."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: PLC0415
    from matplotlib.figure import Figure  # noqa: PLC0415

    slices = [item for shot in report["shots"] for item in shot["slices"]]
    figure = Figure(figsize=(11.5, 5.5), dpi=160, constrained_layout=True)
    FigureCanvasAgg(figure)
    distribution_axis, scatter_axis = figure.subplots(1, 2)
    colors = {False: "#D55E00", True: "#0072B2"}
    labels = {False: "non-diverted", True: "diverted"}
    for position, diverted in enumerate((False, True)):
        group = [item for item in slices if bool(item["diverted"]) is diverted]
        measured = [
            item
            for item in group
            if item["boundary_to_limiter_distance_mm"] is not None
            and item["boundary_area_m2"] is not None
        ]
        distances = np.asarray(
            [item["boundary_to_limiter_distance_mm"] for item in measured],
            dtype=np.float64,
        )
        areas = np.asarray([item["boundary_area_m2"] for item in measured])
        jitter = np.linspace(-0.16, 0.16, distances.size) if distances.size else []
        distribution_axis.scatter(
            np.asarray(jitter) + position,
            distances,
            s=18,
            alpha=0.58,
            color=colors[diverted],
            edgecolors="none",
        )
        scatter_axis.scatter(
            distances,
            areas,
            s=20,
            alpha=0.62,
            color=colors[diverted],
            label=labels[diverted],
            edgecolors="none",
        )
    distribution_axis.axhline(
        FLOATING_DISTANCE_MM, color="#222222", linestyle="--", linewidth=1.1
    )
    distribution_axis.set_xticks((0, 1), ("non-diverted", "diverted"))
    distribution_axis.set_ylabel("Boundary-to-limiter distance (mm)")
    distribution_axis.set_title("Wall proximity by recorded topology")
    distribution_axis.grid(axis="y", alpha=0.22)
    scatter_axis.axvline(
        FLOATING_DISTANCE_MM, color="#222222", linestyle="--", linewidth=1.1
    )
    scatter_axis.set_xlabel("Boundary-to-limiter distance (mm)")
    scatter_axis.set_ylabel("Enclosed boundary area (m²)")
    scatter_axis.set_title("Floating distance versus boundary area")
    scatter_axis.grid(alpha=0.22)
    scatter_axis.legend()
    figure.suptitle(textwrap.fill(str(report["verdict"]), width=105), fontsize=11)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)


def write_report(report: Mapping[str, Any], output_dir: Path) -> tuple[Path, Path]:
    """Write the JSON receipt and matching evidence figure."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "boundary-wall-proximity.json"
    png_path = directory / "boundary-wall-proximity.png"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_figure(report, png_path)
    return json_path, png_path


def write_repair_null_figure(report: Mapping[str, Any], output: Path) -> None:
    """Plot absolute paired changes by topology with exact zeros visible."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: PLC0415
    from matplotlib.figure import Figure  # noqa: PLC0415

    pairs = [item for shot in report["shots"] for item in shot["paired_slices"]]
    figure = Figure(figsize=(11.5, 5.5), dpi=160, constrained_layout=True)
    FigureCanvasAgg(figure)
    distance_axis, area_axis = figure.subplots(1, 2)
    colors = {False: "#D55E00", True: "#0072B2"}
    labels = {False: "non-diverted", True: "diverted"}
    panels = (
        (
            distance_axis,
            "boundary_to_limiter_distance_difference_mm",
            "Absolute distance difference (mm)",
        ),
        (area_axis, "boundary_area_difference_m2", "Absolute area difference (m²)"),
    )
    for axis, key, ylabel in panels:
        nonzero = np.asarray(
            [abs(float(item[key])) for item in pairs if item[key] not in (None, 0.0)],
            dtype=np.float64,
        )
        floor = float(np.min(nonzero) / 10.0) if nonzero.size else 1.0e-16
        for position, diverted in enumerate((False, True)):
            group = [
                item
                for item in pairs
                if bool(item["diverted"]) is diverted and item[key] is not None
            ]
            values = np.asarray(
                [
                    floor if float(item[key]) == 0.0 else abs(float(item[key]))
                    for item in group
                ]
            )
            jitter = np.linspace(-0.17, 0.17, values.size) if values.size else []
            axis.scatter(
                np.asarray(jitter) + position,
                values,
                s=18,
                alpha=0.62,
                color=colors[diverted],
                edgecolors="none",
                label=labels[diverted],
            )
        axis.axhline(
            floor,
            color="#333333",
            linestyle=":",
            linewidth=1.0,
            label="exact zero display floor",
        )
        axis.set_yscale("log")
        axis.set_xticks((0, 1), ("non-diverted", "diverted"))
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", which="both", alpha=0.22)
    distance_axis.set_title("Wall-distance change")
    area_axis.set_title("Enclosed-area change")
    handles, legend_labels = area_axis.get_legend_handles_labels()
    unique = dict(zip(legend_labels, handles, strict=True))
    area_axis.legend(unique.values(), unique.keys(), loc="best")
    figure.suptitle(textwrap.fill(str(report["verdict"]), width=110), fontsize=11)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)


def write_repair_null_report(
    report: Mapping[str, Any], output_dir: Path
) -> tuple[Path, Path]:
    """Write the paired repair receipt and its evidence figure."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "repair-null-check.json"
    png_path = directory / "repair-null-check.png"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_repair_null_figure(report, png_path)
    return json_path, png_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-root", type=Path, default=DEFAULT_SESSION_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--shots", nargs="+", type=int, default=list(FROZEN_CARRIER_SHOTS)
    )
    parser.add_argument(
        "--comparison-root",
        type=Path,
        help="compare the session root with this repaired-session root",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the carrier boundary-to-wall audit."""
    args = _parser().parse_args(argv)
    if args.comparison_root is None:
        report = build_report(session_root=args.session_root, shot_ids=args.shots)
        paths = write_report(report, args.output_dir)
    else:
        report = build_repair_null_report(
            baseline_root=args.session_root,
            repaired_root=args.comparison_root,
            shot_ids=args.shots,
        )
        paths = write_repair_null_report(report, args.output_dir)
    print(json.dumps({"verdict": report["verdict"], "outputs": paths}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
