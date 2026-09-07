"""Validate the renderer limiter polygon against the MAST wall description.

Distances are signed by containment in the reference polygon: negative means
the sampled point lies inside the reference, positive means it lies outside,
and zero means it lies on the reference polyline.  Measuring in both
directions exposes both protrusion and corner-cutting by a coarse polygon.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import textwrap
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

from imas_ambix.latent.wall_mask import _inside_polygon
from imas_ambix.worldmodel.camera_topology_targets import (
    MAST_WALL_SOURCE_SHOT,
    _load_wall,
)
from imas_ambix.worldmodel.equilibrium_labels import (
    DEFAULT_LEVEL2_ROOT,
    equilibrium_store_path,
)
from imas_ambix.worldmodel.flux_conditioning import (
    MAST_LIMITER_R,
    MAST_LIMITER_Z,
)

DEFAULT_REQUESTED_SHOT = 22086
DEFAULT_DENSE_SAMPLE_COUNT = 4096
DEFECT_THRESHOLD_MM = 50.0
ON_POLYLINE_TOLERANCE_M = 1.0e-12
DEFAULT_OUTPUT_JSON = Path(
    "docs/figures/physics-carried-playable-plasma/label-quality/"
    "limiter-polygon-validation.json"
)
DEFAULT_OUTPUT_PNG = DEFAULT_OUTPUT_JSON.with_suffix(".png")


@dataclass(frozen=True, slots=True)
class WallGeometry:
    """Machine-description wall coordinates and selection provenance."""

    r: np.ndarray
    z: np.ndarray
    requested_shot_id: int
    source_shot_id: int
    digest: str
    requested_store: Path
    source_store: Path
    source_branch: str
    description_2d_entry_count: int
    limiter_unit_count: int


def _source_revision() -> str:
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _polygon(r: Sequence[float], z: Sequence[float], *, name: str) -> np.ndarray:
    radius = np.asarray(r, dtype=np.float64).reshape(-1)
    height = np.asarray(z, dtype=np.float64).reshape(-1)
    if radius.shape != height.shape:
        raise ValueError(f"{name} R and Z coordinates must align")
    points = np.column_stack((radius, height))
    if not np.isfinite(points).all():
        raise ValueError(f"{name} coordinates must be finite")
    if points.shape[0] > 1 and np.allclose(
        points[0], points[-1], rtol=0.0, atol=ON_POLYLINE_TOLERANCE_M
    ):
        points = points[:-1]
    if points.shape[0] > 1:
        keep = np.concatenate(
            ([True], np.linalg.norm(np.diff(points, axis=0), axis=1) > 0.0)
        )
        points = points[keep]
    if points.shape[0] < 3 or np.unique(points, axis=0).shape[0] < 3:
        raise ValueError(f"{name} needs at least three distinct vertices")
    return points


def _nearest_polyline_points(
    query: np.ndarray, polygon: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts = polygon
    vectors = np.roll(polygon, -1, axis=0) - starts
    squared_lengths = np.einsum("ij,ij->i", vectors, vectors)
    valid = squared_lengths > 0.0
    if not np.any(valid):
        raise ValueError("reference polygon has no non-zero-length edge")
    starts = starts[valid]
    vectors = vectors[valid]
    squared_lengths = squared_lengths[valid]
    segment_ids = np.flatnonzero(valid)

    offsets = query[:, None, :] - starts[None, :, :]
    positions = np.einsum("qsi,si->qs", offsets, vectors) / squared_lengths
    positions = np.clip(positions, 0.0, 1.0)
    candidates = starts[None, :, :] + positions[:, :, None] * vectors[None, :, :]
    squared_distances = np.sum((query[:, None, :] - candidates) ** 2, axis=2)
    nearest_index = np.argmin(squared_distances, axis=1)
    rows = np.arange(query.shape[0])
    nearest = candidates[rows, nearest_index]
    distance = np.sqrt(squared_distances[rows, nearest_index])
    return distance, nearest, segment_ids[nearest_index]


def signed_point_to_polygon_distances(
    query_r: Sequence[float],
    query_z: Sequence[float],
    reference_r: Sequence[float],
    reference_z: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return signed distances, nearest points, and nearest segment indices."""
    query = _polygon(query_r, query_z, name="query polygon")
    reference = _polygon(reference_r, reference_z, name="reference polygon")
    distance, nearest, segment = _nearest_polyline_points(query, reference)
    inside = _inside_polygon(query[:, 0], query[:, 1], reference[:, 0], reference[:, 1])
    signed = np.where(inside, -distance, distance)
    signed[distance <= ON_POLYLINE_TOLERANCE_M] = 0.0
    return signed, nearest, segment


def resample_closed_polygon(
    r: Sequence[float], z: Sequence[float], sample_count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a closed polygon uniformly by circuit arclength."""
    if sample_count < 3:
        raise ValueError("dense sample count must be at least three")
    polygon = _polygon(r, z, name="polygon")
    vectors = np.roll(polygon, -1, axis=0) - polygon
    lengths = np.linalg.norm(vectors, axis=1)
    perimeter = float(np.sum(lengths))
    if not np.isfinite(perimeter) or perimeter <= 0.0:
        raise ValueError("polygon perimeter must be finite and positive")
    circuit_distance = np.linspace(0.0, perimeter, sample_count, endpoint=False)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    segment = np.searchsorted(cumulative[1:], circuit_distance, side="right")
    local = (circuit_distance - cumulative[segment]) / lengths[segment]
    points = polygon[segment] + local[:, None] * vectors[segment]
    return points, circuit_distance / perimeter


def _vertex_circuit_fraction(polygon: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(np.roll(polygon, -1, axis=0) - polygon, axis=1)
    return np.concatenate(([0.0], np.cumsum(lengths[:-1]))) / np.sum(lengths)


def _poloidal_angle_degrees(points: np.ndarray, centre: np.ndarray) -> np.ndarray:
    angle = np.degrees(np.arctan2(points[:, 1] - centre[1], points[:, 0] - centre[0]))
    return np.mod(angle, 360.0)


def _distribution(signed_distance_m: np.ndarray) -> dict[str, float | int]:
    signed_mm = np.asarray(signed_distance_m, dtype=np.float64) * 1000.0
    absolute_mm = np.abs(signed_mm)
    return {
        "count": int(signed_mm.size),
        "mean_signed_mm": float(np.mean(signed_mm)),
        "median_signed_mm": float(np.median(signed_mm)),
        "p90_signed_mm": float(np.quantile(signed_mm, 0.9)),
        "p90_absolute_mm": float(np.quantile(absolute_mm, 0.9)),
        "maximum_absolute_mm": float(np.max(absolute_mm)),
        "minimum_signed_mm": float(np.min(signed_mm)),
        "maximum_signed_mm": float(np.max(signed_mm)),
        "inside_count": int(np.count_nonzero(signed_mm < 0.0)),
        "outside_count": int(np.count_nonzero(signed_mm > 0.0)),
        "on_polyline_count": int(np.count_nonzero(signed_mm == 0.0)),
    }


def _direction_report(
    query: np.ndarray,
    reference: np.ndarray,
    circuit_fraction: np.ndarray,
    *,
    centre: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    signed, nearest, segment = signed_point_to_polygon_distances(
        query[:, 0], query[:, 1], reference[:, 0], reference[:, 1]
    )
    angles = _poloidal_angle_degrees(query, centre)
    records = [
        {
            "sample_index": int(index),
            "r_m": float(point[0]),
            "z_m": float(point[1]),
            "circuit_fraction": float(circuit_fraction[index]),
            "poloidal_angle_deg": float(angles[index]),
            "signed_distance_mm": float(signed[index] * 1000.0),
            "nearest_r_m": float(nearest[index, 0]),
            "nearest_z_m": float(nearest[index, 1]),
            "nearest_segment_index": int(segment[index]),
        }
        for index, point in enumerate(query)
    ]
    return {
        "summary": _distribution(signed),
        "samples": records,
    }, nearest


def compare_limiter_polygons(
    constant_r: Sequence[float],
    constant_z: Sequence[float],
    machine_r: Sequence[float],
    machine_z: Sequence[float],
    *,
    dense_sample_count: int = DEFAULT_DENSE_SAMPLE_COUNT,
) -> dict[str, Any]:
    """Compare constant and machine polygons with bidirectional distances."""
    constant_input_count = len(np.asarray(constant_r).reshape(-1))
    machine_input_count = len(np.asarray(machine_r).reshape(-1))
    constant = _polygon(constant_r, constant_z, name="renderer limiter")
    machine = _polygon(machine_r, machine_z, name="machine-description wall")
    dense_machine, dense_fraction = resample_closed_polygon(
        machine[:, 0], machine[:, 1], dense_sample_count
    )
    constant_fraction = _vertex_circuit_fraction(constant)
    centre = np.mean(machine, axis=0)

    constant_to_machine, constant_nearest = _direction_report(
        constant,
        machine,
        constant_fraction,
        centre=centre,
    )
    machine_to_constant, machine_nearest = _direction_report(
        dense_machine,
        constant,
        dense_fraction,
        centre=centre,
    )
    directions = {
        "constant_vertices_to_machine_wall": constant_to_machine,
        "dense_machine_wall_to_constant_polygon": machine_to_constant,
    }
    candidates: list[dict[str, Any]] = []
    for direction, report in directions.items():
        sample = max(
            report["samples"], key=lambda item: abs(item["signed_distance_mm"])
        )
        candidates.append({"direction": direction, **sample})
    largest = max(candidates, key=lambda item: abs(item["signed_distance_mm"]))
    maximum_absolute_mm = max(
        float(report["summary"]["maximum_absolute_mm"])
        for report in directions.values()
    )
    threshold_supported = maximum_absolute_mm <= DEFECT_THRESHOLD_MM
    if threshold_supported:
        verdict = (
            "ACCURATE ENOUGH: the bidirectional polygon mismatch peaks at "
            f"{maximum_absolute_mm:.3f} mm, below the 50 mm defect threshold."
        )
    else:
        verdict = (
            "RE-MEASURE AGAINST MACHINE DESCRIPTION: the bidirectional polygon "
            f"mismatch reaches {maximum_absolute_mm:.3f} mm, comparable to or "
            "larger than the 50 mm defect threshold."
        )
    return {
        "sign_convention": (
            "negative means the sample lies inside the reference polygon; "
            "positive means outside; zero means on its polyline"
        ),
        "defect_threshold_mm": DEFECT_THRESHOLD_MM,
        "threshold_supported": threshold_supported,
        "constant_polygon": {
            "input_vertex_count": constant_input_count,
            "effective_vertex_count": int(constant.shape[0]),
            "endpoint_repeated": constant_input_count != constant.shape[0],
            "r_bounds_m": [
                float(np.min(constant[:, 0])),
                float(np.max(constant[:, 0])),
            ],
            "z_bounds_m": [
                float(np.min(constant[:, 1])),
                float(np.max(constant[:, 1])),
            ],
        },
        "machine_polygon": {
            "input_vertex_count": machine_input_count,
            "effective_vertex_count": int(machine.shape[0]),
            "endpoint_repeated": machine_input_count != machine.shape[0],
            "dense_sample_count": int(dense_sample_count),
            "r_bounds_m": [float(np.min(machine[:, 0])), float(np.max(machine[:, 0]))],
            "z_bounds_m": [float(np.min(machine[:, 1])), float(np.max(machine[:, 1]))],
        },
        **directions,
        "largest_single_deviation": largest,
        "verdict": verdict,
        "plot_data": {
            "constant_points_m": constant.tolist(),
            "machine_points_m": machine.tolist(),
            "dense_machine_points_m": dense_machine.tolist(),
            "constant_nearest_machine_points_m": constant_nearest.tolist(),
            "dense_machine_nearest_constant_points_m": machine_nearest.tolist(),
        },
    }


def load_machine_description_wall(
    requested_shot_id: int = DEFAULT_REQUESTED_SHOT,
    *,
    level2_root: Path = DEFAULT_LEVEL2_ROOT,
) -> WallGeometry:
    """Load the wall through the camera-target reader's era-fill path."""
    import zarr  # noqa: PLC0415

    root = Path(level2_root)
    requested_store = equilibrium_store_path(int(requested_shot_id), root)
    store = zarr.open_group(str(requested_store), mode="r")
    wall_r, wall_z, source_shot, digest = _load_wall(
        store, root, int(requested_shot_id)
    )
    expected_digest = hashlib.sha256(
        np.asarray(wall_r).tobytes() + np.asarray(wall_z).tobytes()
    ).hexdigest()
    if expected_digest != digest:
        raise ValueError("wall digest does not match loaded machine coordinates")
    source_store = equilibrium_store_path(source_shot, root)
    source_group = (
        store
        if source_shot == int(requested_shot_id)
        else zarr.open_group(str(source_store), mode="r")
    )
    wall_group = source_group["wall"]
    limiter_unit_count = int(
        np.asarray(wall_group["limiter_geometry_channel"]).reshape(-1).size
    )
    source_branch = (
        "shot_wall" if source_shot == int(requested_shot_id) else "era_constant_fill"
    )
    return WallGeometry(
        r=np.asarray(wall_r, dtype=np.float64),
        z=np.asarray(wall_z, dtype=np.float64),
        requested_shot_id=int(requested_shot_id),
        source_shot_id=int(source_shot),
        digest=str(digest),
        requested_store=requested_store,
        source_store=source_store,
        source_branch=source_branch,
        description_2d_entry_count=1,
        limiter_unit_count=limiter_unit_count,
    )


def build_report(
    *,
    requested_shot_id: int = DEFAULT_REQUESTED_SHOT,
    level2_root: Path = DEFAULT_LEVEL2_ROOT,
    dense_sample_count: int = DEFAULT_DENSE_SAMPLE_COUNT,
) -> dict[str, Any]:
    """Load the MAST wall and assemble the polygon-validation receipt."""
    wall = load_machine_description_wall(requested_shot_id, level2_root=level2_root)
    comparison = compare_limiter_polygons(
        MAST_LIMITER_R,
        MAST_LIMITER_Z,
        wall.r,
        wall.z,
        dense_sample_count=dense_sample_count,
    )
    return {
        "schema": "limiter-polygon-validation",
        "generated_at": datetime.now(UTC).isoformat(),
        "source_revision": _source_revision(),
        "constant_source": ("imas_ambix.worldmodel.flux_conditioning.MAST_LIMITER_R/Z"),
        "machine_description_reader": (
            "imas_ambix.worldmodel.camera_topology_targets._load_wall"
        ),
        "wall_provenance": {
            "level2_root": str(Path(level2_root)),
            "requested_shot_id": wall.requested_shot_id,
            "wall_source_shot_id": wall.source_shot_id,
            "wall_source_branch": wall.source_branch,
            "wall_digest": wall.digest,
            "requested_store": str(wall.requested_store),
            "source_store": str(wall.source_store),
            "era_constant_source_shot_id": MAST_WALL_SOURCE_SHOT,
            "wall_description_2d_entry_count": wall.description_2d_entry_count,
            "limiter_unit_count": wall.limiter_unit_count,
            "collection_count_method": (
                "the flat wall group represents one selected description_2d; "
                "limiter units are counted from limiter_geometry_channel"
            ),
        },
        **comparison,
    }


def write_figure(report: Mapping[str, Any], output: Path) -> None:
    """Overlay both polygons and plot signed mismatch around the circuit."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: PLC0415
    from matplotlib.figure import Figure  # noqa: PLC0415

    plot_data = report["plot_data"]
    constant = np.asarray(plot_data["constant_points_m"], dtype=np.float64)
    machine = np.asarray(plot_data["machine_points_m"], dtype=np.float64)
    dense = np.asarray(plot_data["dense_machine_points_m"], dtype=np.float64)
    dense_nearest = np.asarray(
        plot_data["dense_machine_nearest_constant_points_m"], dtype=np.float64
    )
    directions = (
        report["constant_vertices_to_machine_wall"],
        report["dense_machine_wall_to_constant_polygon"],
    )

    figure = Figure(figsize=(12.0, 7.2), dpi=160, constrained_layout=True)
    FigureCanvasAgg(figure)
    overlay_axis, deviation_axis = figure.subplots(1, 2)
    machine_closed = np.vstack((machine, machine[0]))
    constant_closed = np.vstack((constant, constant[0]))
    overlay_axis.fill(
        machine_closed[:, 0],
        machine_closed[:, 1],
        color="#D55E00",
        alpha=0.16,
        label="machine-description area",
    )
    overlay_axis.fill(
        constant_closed[:, 0],
        constant_closed[:, 1],
        color="#0072B2",
        alpha=0.16,
        label="renderer-constant area",
    )
    overlay_axis.plot(
        machine_closed[:, 0],
        machine_closed[:, 1],
        color="#D55E00",
        linewidth=1.8,
        label="machine-description wall",
    )
    overlay_axis.plot(
        constant_closed[:, 0],
        constant_closed[:, 1],
        color="#0072B2",
        linewidth=1.1,
        linestyle="--",
        label="36-vertex constant",
    )
    stride = max(1, dense.shape[0] // 128)
    for point, nearest in zip(dense[::stride], dense_nearest[::stride], strict=True):
        overlay_axis.plot(
            [point[0], nearest[0]],
            [point[1], nearest[1]],
            color="#7A5195",
            alpha=0.20,
            linewidth=0.5,
        )
    largest = report["largest_single_deviation"]
    overlay_axis.scatter(
        [largest["r_m"]],
        [largest["z_m"]],
        s=44,
        color="black",
        zorder=5,
        label=f"largest deviation: {abs(largest['signed_distance_mm']):.3f} mm",
    )
    overlay_axis.set_aspect("equal")
    overlay_axis.set_xlabel("R (m)")
    overlay_axis.set_ylabel("Z (m)")
    overlay_axis.set_title("Wall overlay; single-colour shading is mismatch")
    overlay_axis.grid(alpha=0.18)
    overlay_axis.legend(loc="best", fontsize=8)

    labels = ("constant → machine", "machine → constant")
    colors = ("#0072B2", "#D55E00")
    plotted_deviations: list[np.ndarray] = []
    for direction, label, color in zip(directions, labels, colors, strict=True):
        samples = direction["samples"]
        fraction = np.asarray(
            [sample["circuit_fraction"] for sample in samples], dtype=np.float64
        )
        signed_mm = np.asarray(
            [sample["signed_distance_mm"] for sample in samples], dtype=np.float64
        )
        plotted_deviations.append(signed_mm)
        deviation_axis.plot(
            fraction, signed_mm, color=color, linewidth=1.2, label=label
        )
    deviation_axis.axhline(0.0, color="black", linewidth=0.8)
    maximum_plotted_mm = max(
        float(np.max(np.abs(values))) for values in plotted_deviations
    )
    display_half_range_mm = max(1.0e-3, 1.15 * maximum_plotted_mm)
    deviation_axis.set_ylim(-display_half_range_mm, display_half_range_mm)
    deviation_axis.text(
        0.02,
        0.03,
        f"Zoomed mismatch scale; threshold is ±{DEFECT_THRESHOLD_MM:g} mm",
        transform=deviation_axis.transAxes,
        fontsize=8,
        color="#555555",
    )
    deviation_axis.set_xlabel("Poloidal circuit fraction")
    deviation_axis.set_ylabel("Signed nearest-wall deviation (mm)")
    deviation_axis.set_title("Negative is inside reference; positive is outside")
    deviation_axis.grid(alpha=0.18)
    deviation_axis.legend(loc="best")
    figure.suptitle(textwrap.fill(str(report["verdict"]), width=105), fontsize=11)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)


def write_report(
    report: Mapping[str, Any], output_json: Path, output_png: Path
) -> None:
    """Write the machine-wall receipt and its evidence figure."""
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_figure(report, output_png)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requested-shot", type=int, default=DEFAULT_REQUESTED_SHOT)
    parser.add_argument("--level2-root", type=Path, default=DEFAULT_LEVEL2_ROOT)
    parser.add_argument(
        "--dense-sample-count", type=int, default=DEFAULT_DENSE_SAMPLE_COUNT
    )
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-png", type=Path, default=DEFAULT_OUTPUT_PNG)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the limiter-polygon validation."""
    args = _parser().parse_args(argv)
    report = build_report(
        requested_shot_id=args.requested_shot,
        level2_root=args.level2_root,
        dense_sample_count=args.dense_sample_count,
    )
    write_report(report, args.output_json, args.output_png)
    print(report["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
