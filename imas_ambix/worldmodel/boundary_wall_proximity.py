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

from imas_ambix.latent.wall_mask import _inside_polygon
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
FLOATING_DISTANCE_MM = 50.0
DEFAULT_OUTPUT_DIR = Path("docs/figures/physics-carried-playable-plasma/label-quality")


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
    distance_m = min(
        _vertices_to_segments_distance(boundary, limiter),
        _vertices_to_segments_distance(limiter, boundary),
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
        missing_outer_count = (
            category_counts["fallback_outermost_finite_surface"]
            + category_counts["no_qualifying_surface"]
        )
        result[name] = {
            "admitted_slice_count": count,
            "measured_slice_count": len(measured),
            "measurement_unavailable_count": count - len(measured),
            "surface_category_counts": category_counts,
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
            if selection.r is not None and selection.z is not None:
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
        "reclassified_as_free_count": reclassified_count,
        "dropped_slices": dropped,
        "summary": summarize_measurements(measurements),
        "slices": [asdict(item) for item in measurements],
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-root", type=Path, default=DEFAULT_SESSION_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--shots", nargs="+", type=int, default=list(FROZEN_CARRIER_SHOTS)
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the carrier boundary-to-wall audit."""
    args = _parser().parse_args(argv)
    report = build_report(session_root=args.session_root, shot_ids=args.shots)
    paths = write_report(report, args.output_dir)
    print(json.dumps({"verdict": report["verdict"], "outputs": paths}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
