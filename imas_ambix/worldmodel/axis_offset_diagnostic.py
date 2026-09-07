"""Diagnose signed magnetic-axis offsets against evaluator-only EFIT geometry.

The EFIT reconstruction enters exclusively through
:mod:`imas_ambix.worldmodel.equilibrium_labels` and is never exposed as a
training input.  The diagnostic keeps every converged slice in its receipt,
then reports the free flat-top subset used to interpret the physics-fidelity
axis miss.  Nova's current centroid is retained as a third geometric referent.
"""

from __future__ import annotations

import argparse
import json
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from imas_ambix.data.paths import LEVEL1_DIR
from imas_ambix.worldmodel import equilibrium_labels
from imas_ambix.worldmodel.physics_fidelity_gate import (
    DEFAULT_SESSION_ROOT,
    FROZEN_CARRIER_SHOTS,
    _load_companion,
    _load_efit_current,
    _load_manifest,
    _slice_array,
    flat_top_mask_from_current,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

RADIAL_SIGN_COHERENCE = 0.90
RADIAL_DOMINANCE_RATIO = 0.75


@dataclass(frozen=True, slots=True)
class AxisOffsetSlice:
    """Signed axis and centroid comparison for one converged Nova slice."""

    manifest_row: int
    session_index: int
    time_s: float
    conditioned_recorded: bool
    conditioned_applied: bool
    conditioned_flag_corrected: bool
    conditioned_branch_guard_ok: bool
    flat_top: bool
    evidence_eligible: bool
    nova_axis_r_m: float | None
    nova_axis_z_m: float | None
    efit_axis_r_m: float | None
    efit_axis_z_m: float | None
    current_centroid_r_m: float | None
    current_centroid_z_m: float | None
    d_r_cm: float | None
    d_z_cm: float | None
    axis_offset_cm: float | None
    nova_axis_to_current_centroid_cm: float | None
    efit_axis_to_current_centroid_cm: float | None
    exclusion_reason: str | None


def signed_axis_offset_cm(
    nova_axis_r_m: float,
    nova_axis_z_m: float,
    efit_axis_r_m: float,
    efit_axis_z_m: float,
) -> tuple[float, float]:
    """Return signed ``(Nova - EFIT)`` magnetic-axis components in centimetres."""
    values = np.asarray(
        (nova_axis_r_m, nova_axis_z_m, efit_axis_r_m, efit_axis_z_m),
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise ValueError("magnetic-axis coordinates must be finite")
    return (
        100.0 * float(nova_axis_r_m - efit_axis_r_m),
        100.0 * float(nova_axis_z_m - efit_axis_z_m),
    )


def point_distance_cm(
    first_r_m: float,
    first_z_m: float,
    second_r_m: float,
    second_z_m: float,
) -> float:
    """Return Euclidean poloidal distance between two finite points in centimetres."""
    values = np.asarray(
        (first_r_m, first_z_m, second_r_m, second_z_m), dtype=np.float64
    )
    if not np.isfinite(values).all():
        raise ValueError("point coordinates must be finite")
    return 100.0 * float(np.hypot(first_r_m - second_r_m, first_z_m - second_z_m))


def conditioned_solve_applied(
    manifest_row: Mapping[str, Any], conditioned_recorded: bool
) -> tuple[bool, bool]:
    """Resolve whether the EFIT-centroid-conditioned solve actually ran.

    Early session writers set the companion flag before deriving the centroid
    target.  A recorded conditioned row with an exception and zero conditioned
    trips therefore never reached the constrained solve and remains independent
    of the EFIT scalar.  The second return value records that correction.
    """
    if not conditioned_recorded:
        return False, False
    exception = manifest_row.get("exception")
    trips = manifest_row.get("conditioned_trips")
    derivation_failed = bool(exception) and trips is not None and int(trips) == 0
    return not derivation_failed, derivation_failed


def _optional_scalar(value: Any) -> float | None:
    scalar = float(np.asarray(value, dtype=np.float64).item())
    return scalar if np.isfinite(scalar) else None


def _signed_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if not finite.size:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "positive_fraction": None,
        }
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "std": float(np.std(finite)),
        "positive_fraction": float(np.mean(finite > 0.0)),
    }


def _slice_receipt(item: AxisOffsetSlice) -> dict[str, Any]:
    """Serialise a slice while keeping the scientific dR/dZ notation in data."""
    receipt = asdict(item)
    receipt["dR_cm"] = receipt.pop("d_r_cm")
    receipt["dZ_cm"] = receipt.pop("d_z_cm")
    return receipt


def _distance_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if not finite.size:
        return {"count": 0, "mean": None, "median": None, "std": None}
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "std": float(np.std(finite)),
    }


def aggregate_slices(slices: Sequence[AxisOffsetSlice]) -> dict[str, Any]:
    """Aggregate signed offsets and centroid distances on eligible slices."""
    evidence = [item for item in slices if item.evidence_eligible]
    finite = [item for item in evidence if item.d_r_cm is not None]
    flat_top_times = [item.time_s for item in slices if item.flat_top]
    corrected_count = sum(item.conditioned_flag_corrected for item in slices)
    return {
        "candidate_slice_count": len(slices),
        "manifest_converged_slice_count": len(slices) - corrected_count,
        "recovered_free_solution_count": corrected_count,
        "flat_top_slice_count": sum(item.flat_top for item in slices),
        "flat_top_time_start_s": min(flat_top_times) if flat_top_times else None,
        "flat_top_time_end_s": max(flat_top_times) if flat_top_times else None,
        "conditioned_recorded_count": sum(item.conditioned_recorded for item in slices),
        "conditioned_applied_count": sum(item.conditioned_applied for item in slices),
        "conditioned_flag_corrected_count": corrected_count,
        "excluded_conditioned_flat_top_count": sum(
            item.flat_top and item.conditioned_applied for item in slices
        ),
        "evidence_slice_count": len(finite),
        "dR_cm": _signed_summary(
            [float(item.d_r_cm) for item in finite if item.d_r_cm is not None]
        ),
        "dZ_cm": _signed_summary(
            [float(item.d_z_cm) for item in finite if item.d_z_cm is not None]
        ),
        "axis_offset_cm": _distance_summary(
            [
                float(item.axis_offset_cm)
                for item in finite
                if item.axis_offset_cm is not None
            ]
        ),
        "nova_axis_to_current_centroid_cm": _distance_summary(
            [
                float(item.nova_axis_to_current_centroid_cm)
                for item in finite
                if item.nova_axis_to_current_centroid_cm is not None
            ]
        ),
        "efit_axis_to_current_centroid_cm": _distance_summary(
            [
                float(item.efit_axis_to_current_centroid_cm)
                for item in finite
                if item.efit_axis_to_current_centroid_cm is not None
            ]
        ),
    }


def classify_offset(summary: Mapping[str, Any]) -> tuple[str, str]:
    """Classify the aggregate sign pattern without changing the failed gate."""
    radial = summary["dR_cm"]
    count = int(radial["count"])
    if count == 0:
        return "mixture", "Mixture: no finite free flat-top axis pairs were available."
    positive = float(radial["positive_fraction"])
    mean_r = float(radial["mean"])
    median_offset = float(summary["axis_offset_cm"]["median"])
    nova_centroid = summary["nova_axis_to_current_centroid_cm"]["median"]
    efit_centroid = summary["efit_axis_to_current_centroid_cm"]["median"]
    centroid_text = (
        "current-centroid medians are unavailable"
        if nova_centroid is None or efit_centroid is None
        else (
            f"the current centroid is {float(nova_centroid):.2f} cm from Nova's "
            f"axis and {float(efit_centroid):.2f} cm from EFIT's"
        )
    )
    coherent = (
        positive >= RADIAL_SIGN_COHERENCE or positive <= 1.0 - RADIAL_SIGN_COHERENCE
    )
    radial_dominant = abs(mean_r) >= RADIAL_DOMINANCE_RATIO * median_offset
    if coherent and radial_dominant:
        direction = "outboard" if mean_r > 0.0 else "inboard"
        return (
            "outboard_displacement" if mean_r > 0.0 else "convention_or_resolution",
            f"Displacement: Nova's axis is coherently {direction} of EFIT "
            f"(dR positive on {positive:.1%} of evidence slices), with the signed "
            f"radial component dominating the miss; {centroid_text}.",
        )
    centred_signs = 0.40 <= positive <= 0.60
    if centred_signs and abs(mean_r) <= 0.5 * float(radial["std"]):
        return (
            "convention_or_resolution",
            "Convention or resolution mismatch: dR changes sign and its mean is "
            f"small relative to slice-to-slice radial scatter; {centroid_text}.",
        )
    return (
        "mixture",
        f"Mixture: Nova is inboard on {1.0 - positive:.1%} of evidence slices, "
        "but both components change sign by slice or shot and no coherent radial "
        f"displacement explains the miss; {centroid_text}, so an axis-centroid "
        "swap is not supported.",
    )


def score_shot(
    shot_id: int,
    *,
    session_root: Path = DEFAULT_SESSION_ROOT,
    level2_root: Path = equilibrium_labels.DEFAULT_LEVEL2_ROOT,
    level1_root: Path = LEVEL1_DIR,
) -> dict[str, Any]:
    """Compare Nova, EFIT and current-centroid positions for one carrier shot."""
    import xarray as xr  # noqa: PLC0415

    root = Path(session_root)
    session_path = root / f"{shot_id}.nc"
    manifest_path = root / f"{shot_id}.manifest.json"
    companion_path = root / f"{shot_id}.npz"
    manifest = _load_manifest(manifest_path, shot_id)
    rows = manifest["slices"]
    companion_rows, companion_times, conditioned, guard_ok = _load_companion(
        companion_path, rows
    )
    row_to_session = {int(row): index for index, row in enumerate(companion_rows)}
    written_rows = [row for row in rows if bool(row.get("written", False))]
    converged: list[Mapping[str, Any]] = []
    for row in written_rows:
        session_index = row_to_session[int(row["row"])]
        _, corrected = conditioned_solve_applied(row, bool(conditioned[session_index]))
        free_solution_recovered = corrected and bool(row.get("free_converged", False))
        if bool(row.get("converged", False)) or free_solution_recovered:
            converged.append(row)
    session_indices = np.asarray(
        [row_to_session[int(row["row"])] for row in converged], dtype=np.int64
    )
    slice_times = companion_times[session_indices]
    current_times, plasma_current = _load_efit_current(shot_id, Path(level1_root))
    flat_top, current_receipt = flat_top_mask_from_current(
        current_times, plasma_current, slice_times
    )
    geometry = equilibrium_labels.load_equilibrium_geometry(
        shot_id, slice_times, level2_root=Path(level2_root)
    )

    results: list[AxisOffsetSlice] = []
    with xr.open_dataset(session_path, group="steering", engine="h5netcdf") as session:
        session_times = np.asarray(session["time"], dtype=np.float64).reshape(-1)
        if session_times.shape != companion_times.shape or not np.allclose(
            session_times, companion_times, rtol=0.0, atol=1.0e-9
        ):
            raise ValueError(f"{session_path} times do not align with its companion")
        for position, (row, index, is_flat_top) in enumerate(
            zip(converged, session_indices, flat_top, strict=True)
        ):
            session_index = int(index)
            recorded = bool(conditioned[session_index])
            applied, corrected = conditioned_solve_applied(row, recorded)
            exclusion: str | None = None
            nova_r = _optional_scalar(
                _slice_array(session, "magnetic_axis_r", session_index)
            )
            nova_z = _optional_scalar(
                _slice_array(session, "magnetic_axis_z", session_index)
            )
            efit_r = (
                float(geometry.target[position, 0])
                if bool(geometry.finite_mask[position, 0])
                else None
            )
            efit_z = (
                float(geometry.target[position, 1])
                if bool(geometry.finite_mask[position, 1])
                else None
            )
            centroid_r = _optional_scalar(
                _slice_array(session, "current_centroid_r", session_index)
            )
            centroid_z = _optional_scalar(
                _slice_array(session, "current_centroid_z", session_index)
            )
            d_r: float | None = None
            d_z: float | None = None
            axis_distance: float | None = None
            nova_centroid: float | None = None
            efit_centroid: float | None = None
            try:
                d_r, d_z = signed_axis_offset_cm(nova_r, nova_z, efit_r, efit_z)
                axis_distance = float(np.hypot(d_r, d_z))
            except TypeError, ValueError:
                exclusion = "non_finite_axis"
            with suppress(TypeError, ValueError):
                nova_centroid = point_distance_cm(
                    nova_r, nova_z, centroid_r, centroid_z
                )
            with suppress(TypeError, ValueError):
                efit_centroid = point_distance_cm(
                    efit_r, efit_z, centroid_r, centroid_z
                )
            eligible = bool(is_flat_top and not applied and d_r is not None)
            if not is_flat_top:
                exclusion = "outside_flat_top"
            elif applied:
                exclusion = "conditioned_from_efit_centroid"
            results.append(
                AxisOffsetSlice(
                    manifest_row=int(row["row"]),
                    session_index=session_index,
                    time_s=float(slice_times[position]),
                    conditioned_recorded=recorded,
                    conditioned_applied=applied,
                    conditioned_flag_corrected=corrected,
                    conditioned_branch_guard_ok=bool(guard_ok[session_index]),
                    flat_top=bool(is_flat_top),
                    evidence_eligible=eligible,
                    nova_axis_r_m=nova_r,
                    nova_axis_z_m=nova_z,
                    efit_axis_r_m=efit_r,
                    efit_axis_z_m=efit_z,
                    current_centroid_r_m=centroid_r,
                    current_centroid_z_m=centroid_z,
                    d_r_cm=d_r,
                    d_z_cm=d_z,
                    axis_offset_cm=axis_distance,
                    nova_axis_to_current_centroid_cm=nova_centroid,
                    efit_axis_to_current_centroid_cm=efit_centroid,
                    exclusion_reason=exclusion,
                )
            )

    return {
        "shot_id": int(shot_id),
        "session_path": str(session_path.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "flat_top_current": current_receipt,
        "summary": aggregate_slices(results),
        "slices": [_slice_receipt(item) for item in results],
    }


def score_carriers(
    shots: Sequence[int] = FROZEN_CARRIER_SHOTS,
    *,
    session_root: Path = DEFAULT_SESSION_ROOT,
    level2_root: Path = equilibrium_labels.DEFAULT_LEVEL2_ROOT,
    level1_root: Path = LEVEL1_DIR,
) -> dict[str, Any]:
    """Score the carrier set and return a JSON-ready signed-offset diagnosis."""
    shot_results = [
        score_shot(
            int(shot),
            session_root=Path(session_root),
            level2_root=Path(level2_root),
            level1_root=Path(level1_root),
        )
        for shot in shots
    ]
    all_slices = [
        AxisOffsetSlice(
            **{
                **{
                    key: value
                    for key, value in item.items()
                    if key not in {"dR_cm", "dZ_cm"}
                },
                "d_r_cm": item["dR_cm"],
                "d_z_cm": item["dZ_cm"],
            }
        )
        for shot in shot_results
        for item in shot["slices"]
    ]
    aggregate = aggregate_slices(all_slices)
    classification, verdict = classify_offset(aggregate)
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "diagnostic": "signed_axis_offset",
        "classification": classification,
        "verdict": verdict,
        "offset_definition": {
            "dR_cm": "100 * (Nova magnetic_axis_r - EFIT magnetic_axis_r)",
            "dZ_cm": "100 * (Nova magnetic_axis_z - EFIT magnetic_axis_z)",
            "axis_offset_cm": "hypot(dR_cm, dZ_cm)",
        },
        "flat_top_definition": {
            "signal": "level1 efm/plasma_current_c",
            "slice_rule": (
                "interpolated absolute current >= 80% of shot maximum absolute current"
            ),
        },
        "evidence_rule": (
            "converged free Nova slices in flat top with finite Nova and EFIT axes; "
            "a legacy conditioned flag is corrected only when target derivation failed "
            "before a zero-trip conditioned solve"
        ),
        "sources": {
            "session_root": str(Path(session_root).resolve()),
            "level2_root": str(Path(level2_root).resolve()),
            "level1_root": str(Path(level1_root).resolve()),
            "efit_loader": (
                "imas_ambix.worldmodel.equilibrium_labels.load_equilibrium_geometry"
            ),
            "nova_axis": "steering/magnetic_axis_r,z",
            "nova_current_centroid": "steering/current_centroid_r,z",
        },
        "shots": shot_results,
        "aggregate": aggregate,
    }


def write_figure(diagnostic: Mapping[str, Any], path: Path) -> None:
    """Write signed components, vector cloud and centroid-reference distances."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: PLC0415
    from matplotlib.figure import Figure  # noqa: PLC0415

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(12.5, 8.0), dpi=150, constrained_layout=True)
    FigureCanvasAgg(figure)
    component_axes, vector_axes, centroid_axes = figure.subplots(1, 3)
    palette = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00")
    shot_labels: list[str] = []
    nova_centroid_medians: list[float] = []
    efit_centroid_medians: list[float] = []
    for color, shot in zip(palette, diagnostic["shots"], strict=False):
        slices = shot["slices"]
        eligible = [item for item in slices if item["evidence_eligible"]]
        times = np.asarray([item["time_s"] for item in eligible])
        d_r = np.asarray([item["dR_cm"] for item in eligible])
        d_z = np.asarray([item["dZ_cm"] for item in eligible])
        label = str(shot["shot_id"])
        component_axes.plot(times, d_r, "o", color=color, markersize=3, label=label)
        component_axes.plot(times, d_z, "x", color=color, markersize=4)
        vector_axes.scatter(d_r, d_z, s=15, color=color, label=label)
        summary = shot["summary"]
        shot_labels.append(label)
        nova_centroid_medians.append(
            summary["nova_axis_to_current_centroid_cm"]["median"]
        )
        efit_centroid_medians.append(
            summary["efit_axis_to_current_centroid_cm"]["median"]
        )
    component_axes.axhline(0.0, color="#333333", linewidth=0.8)
    component_axes.set_title("Signed axis components")
    component_axes.set_xlabel("Time (s)")
    component_axes.set_ylabel("Nova - EFIT (cm)")
    component_axes.grid(alpha=0.2)
    component_axes.legend(title="Shot; circles dR, crosses dZ", fontsize=7)
    vector_axes.axhline(0.0, color="#333333", linewidth=0.8)
    vector_axes.axvline(0.0, color="#333333", linewidth=0.8)
    vector_axes.set_aspect("equal", adjustable="datalim")
    vector_axes.set_title("Offset vectors")
    vector_axes.set_xlabel("dR (cm)")
    vector_axes.set_ylabel("dZ (cm)")
    vector_axes.grid(alpha=0.2)
    positions = np.arange(len(shot_labels), dtype=np.float64)
    width = 0.36
    centroid_axes.bar(
        positions - width / 2,
        nova_centroid_medians,
        width,
        color="#0072B2",
        label="Nova axis",
    )
    centroid_axes.bar(
        positions + width / 2,
        efit_centroid_medians,
        width,
        color="#E69F00",
        label="EFIT axis",
    )
    centroid_axes.set_xticks(positions, shot_labels, rotation=45)
    centroid_axes.set_title("Median distance to current centroid")
    centroid_axes.set_xlabel("Shot")
    centroid_axes.set_ylabel("Distance (cm)")
    centroid_axes.grid(axis="y", alpha=0.2)
    centroid_axes.legend(fontsize=8)
    aggregate = diagnostic["aggregate"]
    figure.suptitle(
        f"Signed axis diagnostic: {str(diagnostic['classification']).upper()}\n"
        f"dR > 0 on {aggregate['dR_cm']['positive_fraction']:.1%}; "
        f"mean dR {aggregate['dR_cm']['mean']:.2f} cm, "
        f"mean dZ {aggregate['dZ_cm']['mean']:.2f} cm; "
        "centroid medians Nova/EFIT "
        f"{aggregate['nova_axis_to_current_centroid_cm']['median']:.2f}/"
        f"{aggregate['efit_axis_to_current_centroid_cm']['median']:.2f} cm"
    )
    figure.savefig(output)


def write_diagnostic(
    diagnostic: Mapping[str, Any], output_dir: Path
) -> tuple[Path, Path]:
    """Write the detailed JSON diagnostic and companion figure."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "axis-offset.json"
    figure_path = directory / "axis-offset.png"
    json_path.write_text(
        json.dumps(diagnostic, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_figure(diagnostic, figure_path)
    return json_path, figure_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-root", type=Path, default=DEFAULT_SESSION_ROOT)
    parser.add_argument(
        "--level2-root", type=Path, default=equilibrium_labels.DEFAULT_LEVEL2_ROOT
    )
    parser.add_argument("--level1-root", type=Path, default=LEVEL1_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--shots", nargs="+", type=int, default=list(FROZEN_CARRIER_SHOTS)
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    diagnostic = score_carriers(
        args.shots,
        session_root=args.session_root,
        level2_root=args.level2_root,
        level1_root=args.level1_root,
    )
    paths = write_diagnostic(diagnostic, args.output_dir)
    print(
        json.dumps(
            {
                "classification": diagnostic["classification"],
                "verdict": diagnostic["verdict"],
                "aggregate": diagnostic["aggregate"],
                "shots": [
                    {"shot_id": shot["shot_id"], **shot["summary"]}
                    for shot in diagnostic["shots"]
                ],
                "outputs": [str(path) for path in paths],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
