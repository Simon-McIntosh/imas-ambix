"""Score Nova carrier equilibria against the evaluator-only EFIT geometry.

The EFIT reconstruction enters only through :mod:`equilibrium_labels`.  It is
never returned as a model input.  Each semantically converged Nova slice is
retained in the receipt, while genuinely conditioned slices are labelled and
excluded from the evidence denominator because their solve inherited an EFIT
centroid scalar.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from imas_ambix.camdyn.dataset import level1_shot_path
from imas_ambix.data.paths import LEVEL1_DIR
from imas_ambix.worldmodel import equilibrium_labels
from imas_ambix.worldmodel.flux_label_dataset import (
    DEFAULT_SESSION_ROOT,
    EXPECTED_CARRIER_IDENTITY,
    EXPECTED_POLICY_DIGEST,
    _conditioned_row_is_free,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

FROZEN_CARRIER_SHOTS = (21978, 21983, 21985, 21986, 21989, 22086)
BOUNDARY_LIMIT_CM = 2.0
AXIS_LIMIT_CM = 2.0
MIN_PASS_FRACTION = 0.90
FLAT_TOP_CURRENT_FRACTION = 0.80
SUPERSEDED_PUBLISHED_COUNTS = {
    "joint": {"pass_count": 3, "denominator": 51},
    "boundary": {"pass_count": 24, "denominator": 51},
    "axis": {"pass_count": 3, "denominator": 51},
}
SUPERSEDED_GATE_PASSED = False


@dataclass(frozen=True, slots=True)
class SliceFidelity:
    """One semantically converged Nova slice and its EFIT comparison."""

    manifest_row: int
    session_index: int
    time_s: float
    recorded_conditioned: bool
    conditioned: bool
    reclassified_as_free: bool
    recorded_converged: bool
    semantically_converged: bool
    nova_axis_finite: bool
    conditioned_branch_guard_ok: bool
    flat_top: bool
    evidence_eligible: bool
    boundary_evidence_eligible: bool
    axis_evidence_eligible: bool
    joint_evidence_eligible: bool
    boundary_rms_cm: float | None
    axis_offset_cm: float | None
    boundary_within_limit: bool | None
    axis_within_limit: bool | None
    joint_within_limits: bool | None
    nova_solve_wall_seconds: float | None
    exclusion_reason: str | None
    boundary_exclusion_reason: str | None
    axis_exclusion_reason: str | None
    joint_exclusion_reason: str | None


def _slice_semantics(
    row: Mapping[str, Any], *, recorded_conditioned: bool
) -> tuple[bool, bool, bool]:
    """Return convergence, conditioning, and reclassification under one rule."""
    reclassified_as_free = _conditioned_row_is_free(
        row, conditioned=recorded_conditioned
    )
    semantically_converged = bool(
        row.get("free_converged", False)
        if reclassified_as_free
        else row.get("converged", False)
    )
    genuinely_conditioned = recorded_conditioned and not reclassified_as_free
    return semantically_converged, genuinely_conditioned, reclassified_as_free


def radius_rms_distance_m(
    nova_radii: np.ndarray,
    efit_radii: np.ndarray,
    efit_finite_mask: np.ndarray,
) -> float:
    """Return RMS radial-boundary error on finite referee radius components."""
    nova = np.asarray(nova_radii, dtype=np.float64).reshape(-1)
    efit = np.asarray(efit_radii, dtype=np.float64).reshape(-1)
    referee_mask = np.asarray(efit_finite_mask, dtype=bool).reshape(-1)
    if nova.shape != efit.shape or nova.shape != referee_mask.shape:
        raise ValueError("Nova radii, EFIT radii, and referee mask must align")
    valid = referee_mask & np.isfinite(nova) & np.isfinite(efit)
    if not np.any(valid):
        raise ValueError("no finite EFIT LCFS radius is available")
    return float(np.sqrt(np.mean((nova[valid] - efit[valid]) ** 2)))


def axis_offset_m(
    nova_axis_r: float,
    nova_axis_z: float,
    efit_axis_r: float,
    efit_axis_z: float,
) -> float:
    """Return Euclidean magnetic-axis displacement in the poloidal plane."""
    values = np.asarray(
        (nova_axis_r, nova_axis_z, efit_axis_r, efit_axis_z), dtype=np.float64
    )
    if not np.isfinite(values).all():
        raise ValueError("magnetic-axis coordinates must be finite")
    return float(np.hypot(nova_axis_r - efit_axis_r, nova_axis_z - efit_axis_z))


def flat_top_mask_from_current(
    current_times: np.ndarray,
    plasma_current_a: np.ndarray,
    slice_times: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Select slices where interpolated ``|I_p,efm|`` is at least 80% of max."""
    times = np.asarray(current_times, dtype=np.float64).reshape(-1)
    current = np.abs(np.asarray(plasma_current_a, dtype=np.float64).reshape(-1))
    query = np.asarray(slice_times, dtype=np.float64).reshape(-1)
    if times.shape != current.shape or times.size < 4:
        raise ValueError("plasma-current values need a matching time axis")
    finite = np.isfinite(times) & np.isfinite(current)
    times = times[finite]
    current = current[finite]
    order = np.argsort(times, kind="stable")
    times = times[order]
    current = current[order]
    unique = np.concatenate(([True], np.diff(times) > 0.0))
    times = times[unique]
    current = current[unique]
    if times.size < 4:
        raise ValueError(
            "plasma-current time axis has fewer than four distinct samples"
        )
    peak = float(np.max(current))
    if peak <= 0.0:
        raise ValueError("EFIT plasma-current trace has no non-zero sample")
    flat_top_threshold = FLAT_TOP_CURRENT_FRACTION * peak
    interpolated = np.interp(query, times, current, left=np.nan, right=np.nan)
    mask = np.isfinite(interpolated) & (interpolated >= flat_top_threshold)
    return mask, {
        "peak_current_a": peak,
        "flat_top_threshold_a": flat_top_threshold,
    }


def _load_efit_current(
    shot_id: int, level1_root: Path
) -> tuple[np.ndarray, np.ndarray]:
    import zarr  # noqa: PLC0415

    path = level1_shot_path(shot_id, level1_dir=level1_root)
    store = zarr.open_group(str(path), mode="r")
    if "efm" not in set(store.group_keys()):
        raise KeyError(f"shot {shot_id}: no efm group at {path}")
    group = store["efm"]
    if not {"time", "plasma_current_c"}.issubset(set(group.array_keys())):
        raise KeyError(f"shot {shot_id}: efm lacks time or plasma_current_c")
    return (
        np.asarray(group["time"], dtype=np.float64),
        np.asarray(group["plasma_current_c"], dtype=np.float64),
    )


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


def _load_companion(
    path: Path, slices: Sequence[Mapping[str, Any]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    written = [row for row in slices if bool(row.get("written", False))]
    with np.load(path, allow_pickle=False) as companion:
        required = {"row", "time", "conditioned", "conditioned_branch_guard_ok"}
        missing = required.difference(companion.files)
        if missing:
            raise ValueError(f"{path} is missing companion fields {sorted(missing)}")
        rows = np.asarray(companion["row"], dtype=np.int64).reshape(-1)
        times = np.asarray(companion["time"], dtype=np.float64).reshape(-1)
        conditioned = np.asarray(companion["conditioned"], dtype=bool).reshape(-1)
        guard_ok = np.asarray(
            companion["conditioned_branch_guard_ok"], dtype=bool
        ).reshape(-1)
    shapes = {array.shape for array in (rows, times, conditioned, guard_ok)}
    if len(shapes) != 1:
        raise ValueError(f"{path} companion arrays do not align")
    expected_rows = np.asarray([int(row["row"]) for row in written], dtype=np.int64)
    expected_times = np.asarray([float(row["time"]) for row in written])
    if not np.array_equal(rows, expected_rows) or not np.allclose(
        times, expected_times, rtol=0.0, atol=1.0e-9
    ):
        raise ValueError(f"{path} is not aligned to the manifest's written rows")
    return rows, times, conditioned, guard_ok


def _slice_array(session: Any, name: str, index: int) -> np.ndarray:
    value = session[name]
    if "time" in value.dims:
        value = value.isel(time=index)
    return np.asarray(value, dtype=np.float64)


def _outer_surface(session: Any, index: int) -> tuple[np.ndarray, np.ndarray]:
    levels = _slice_array(session, "flux_surface_psi_norm", index).reshape(-1)
    if not np.isfinite(levels).any():
        raise ValueError("Nova flux-surface levels are entirely non-finite")
    outer = int(np.nanargmin(np.abs(levels - 1.0)))
    if not np.isclose(levels[outer], 1.0, rtol=0.0, atol=1.0e-6):
        raise ValueError("Nova session has no flux surface at psi_norm 1.0")
    surface_r = _slice_array(session, "flux_surface_r", index)
    surface_z = _slice_array(session, "flux_surface_z", index)
    if surface_r.shape != surface_z.shape or surface_r.shape[0] != levels.size:
        raise ValueError("Nova flux surfaces and psi_norm levels do not align")
    return surface_r[outer], surface_z[outer]


def _summary(values: Sequence[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if not finite.size:
        return {"count": 0, "mean": None, "median": None, "p90": None, "max": None}
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p90": float(np.quantile(finite, 0.90)),
        "max": float(np.max(finite)),
    }


def _exclusion_counts(
    slices: Sequence[SliceFidelity], attribute: str
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in slices:
        reason = getattr(item, attribute)
        if reason is not None:
            counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def _aggregate_slices(slices: Sequence[SliceFidelity]) -> dict[str, Any]:
    boundary_evidence = [item for item in slices if item.boundary_evidence_eligible]
    axis_evidence = [item for item in slices if item.axis_evidence_eligible]
    joint_evidence = [item for item in slices if item.joint_evidence_eligible]
    flat_top_times = [item.time_s for item in slices if item.flat_top]
    boundary = [
        float(item.boundary_rms_cm)
        for item in boundary_evidence
        if item.boundary_rms_cm is not None
    ]
    axis = [
        float(item.axis_offset_cm)
        for item in axis_evidence
        if item.axis_offset_cm is not None
    ]
    boundary_passes = sum(
        item.boundary_within_limit is True for item in boundary_evidence
    )
    axis_passes = sum(item.axis_within_limit is True for item in axis_evidence)
    joint_passes = sum(item.joint_within_limits is True for item in joint_evidence)
    boundary_denominator = len(boundary_evidence)
    axis_denominator = len(axis_evidence)
    joint_denominator = len(joint_evidence)
    boundary_fraction = (
        boundary_passes / boundary_denominator if boundary_denominator else 0.0
    )
    axis_fraction = axis_passes / axis_denominator if axis_denominator else 0.0
    joint_fraction = joint_passes / joint_denominator if joint_denominator else 0.0
    return {
        "converged_slice_count": sum(item.recorded_converged for item in slices),
        "semantically_converged_slice_count": len(slices),
        "recorded_converged_slice_count": sum(
            item.recorded_converged for item in slices
        ),
        "conditioned_slice_count": sum(item.conditioned for item in slices),
        "recorded_conditioned_slice_count": sum(
            item.recorded_conditioned for item in slices
        ),
        "reclassified_as_free_count": sum(item.reclassified_as_free for item in slices),
        "reclassified_semantically_converged_count": sum(
            item.reclassified_as_free and item.semantically_converged for item in slices
        ),
        "reclassified_nova_axis_non_finite_count": sum(
            item.reclassified_as_free
            and item.semantically_converged
            and not item.nova_axis_finite
            for item in slices
        ),
        "free_slice_count": sum(not item.conditioned for item in slices),
        "flat_top_slice_count": sum(item.flat_top for item in slices),
        "flat_top_time_start_s": min(flat_top_times) if flat_top_times else None,
        "flat_top_time_end_s": max(flat_top_times) if flat_top_times else None,
        "evidence_slice_count": joint_denominator,
        "boundary_evidence_slice_count": boundary_denominator,
        "axis_evidence_slice_count": axis_denominator,
        "joint_evidence_slice_count": joint_denominator,
        "excluded_conditioned_flat_top_count": sum(
            item.conditioned and item.flat_top for item in slices
        ),
        "exclusions": {
            "boundary": _exclusion_counts(slices, "boundary_exclusion_reason"),
            "axis": _exclusion_counts(slices, "axis_exclusion_reason"),
            "joint": _exclusion_counts(slices, "joint_exclusion_reason"),
        },
        "boundary_rms_cm": _summary(boundary),
        "axis_offset_cm": _summary(axis),
        "boundary_pass_count": boundary_passes,
        "axis_pass_count": axis_passes,
        "joint_pass_count": joint_passes,
        "boundary_pass_fraction": boundary_fraction,
        "axis_pass_fraction": axis_fraction,
        "joint_pass_fraction": joint_fraction,
        "passed": joint_denominator > 0 and joint_fraction >= MIN_PASS_FRACTION,
    }


def score_shot(
    shot_id: int,
    *,
    session_root: Path = DEFAULT_SESSION_ROOT,
    level2_root: Path = equilibrium_labels.DEFAULT_LEVEL2_ROOT,
    level1_root: Path = LEVEL1_DIR,
) -> dict[str, Any]:
    """Score every converged slice of one complete carrier session."""
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
    semantically_converged: list[tuple[Mapping[str, Any], bool, bool, bool]] = []
    recorded_conditioned_rows = 0
    reclassified_rows = 0
    reclassified_converged_rows = 0
    for row in rows:
        if not bool(row.get("written", False)):
            continue
        session_index = row_to_session[int(row["row"])]
        recorded_conditioned = bool(conditioned[session_index])
        recorded_conditioned_rows += int(recorded_conditioned)
        is_converged, genuinely_conditioned, reclassified_as_free = _slice_semantics(
            row, recorded_conditioned=recorded_conditioned
        )
        reclassified_rows += int(reclassified_as_free)
        reclassified_converged_rows += int(reclassified_as_free and is_converged)
        if is_converged:
            semantically_converged.append(
                (
                    row,
                    recorded_conditioned,
                    genuinely_conditioned,
                    reclassified_as_free,
                )
            )
    session_indices = np.asarray(
        [row_to_session[int(row["row"])] for row, _, _, _ in semantically_converged],
        dtype=np.int64,
    )
    slice_times = companion_times[session_indices]
    current_times, plasma_current = _load_efit_current(shot_id, Path(level1_root))
    flat_top, current_receipt = flat_top_mask_from_current(
        current_times, plasma_current, slice_times
    )
    geometry = equilibrium_labels.load_equilibrium_geometry(
        shot_id,
        slice_times,
        level2_root=Path(level2_root),
        angles=equilibrium_labels.LCFS_ANGLES,
    )
    radius_start = 2 + 2 * equilibrium_labels.N_XPOINT_SLOTS
    efit_radii = geometry.target[:, radius_start:].astype(np.float64)
    efit_radius_mask = geometry.finite_mask[:, radius_start:]

    results: list[SliceFidelity] = []
    with xr.open_dataset(session_path, group="steering", engine="h5netcdf") as session:
        session_times = np.asarray(session["time"], dtype=np.float64).reshape(-1)
        if session_times.shape != companion_times.shape or not np.allclose(
            session_times, companion_times, rtol=0.0, atol=1.0e-9
        ):
            raise ValueError(f"{session_path} times do not align with its companion")
        for position, (
            row_semantics,
            index,
            is_flat_top,
        ) in enumerate(
            zip(semantically_converged, session_indices, flat_top, strict=True)
        ):
            row, recorded_conditioned, is_conditioned, reclassified_as_free = (
                row_semantics
            )
            session_index = int(index)
            branch_ok = bool(guard_ok[session_index])
            boundary_cm: float | None = None
            axis_cm: float | None = None
            base_exclusion: str | None = None
            if not is_flat_top:
                base_exclusion = "outside_flat_top"
            elif is_conditioned:
                base_exclusion = "conditioned_from_efit_centroid"
            nova_axis_r = float(
                _slice_array(session, "magnetic_axis_r", session_index).item()
            )
            nova_axis_z = float(
                _slice_array(session, "magnetic_axis_z", session_index).item()
            )
            nova_axis_finite = bool(np.isfinite((nova_axis_r, nova_axis_z)).all())
            boundary_metric_exclusion: str | None = None
            axis_metric_exclusion: str | None = None
            if not nova_axis_finite:
                boundary_metric_exclusion = "nova_magnetic_axis_non_finite"
                axis_metric_exclusion = "nova_magnetic_axis_non_finite"
            else:
                try:
                    nova_r, nova_z = _outer_surface(session, session_index)
                    nova_radii = equilibrium_labels.resample_lcfs_radii(
                        nova_r,
                        nova_z,
                        nova_axis_r,
                        nova_axis_z,
                        equilibrium_labels.LCFS_ANGLES,
                    )
                    boundary_cm = 100.0 * radius_rms_distance_m(
                        nova_radii,
                        efit_radii[position],
                        efit_radius_mask[position],
                    )
                except ValueError:
                    boundary_metric_exclusion = "boundary_metric_unavailable"
                try:
                    axis_cm = 100.0 * axis_offset_m(
                        nova_axis_r,
                        nova_axis_z,
                        float(geometry.target[position, 0]),
                        float(geometry.target[position, 1]),
                    )
                except ValueError:
                    axis_metric_exclusion = "axis_metric_unavailable"
            boundary_exclusion = base_exclusion or boundary_metric_exclusion
            axis_exclusion = base_exclusion or axis_metric_exclusion
            boundary_evidence_eligible = boundary_exclusion is None
            axis_evidence_eligible = axis_exclusion is None
            joint_evidence_eligible = (
                boundary_evidence_eligible and axis_evidence_eligible
            )
            if base_exclusion is not None:
                joint_exclusion = base_exclusion
            elif boundary_metric_exclusion == axis_metric_exclusion:
                joint_exclusion = boundary_metric_exclusion
            elif (
                boundary_metric_exclusion is not None
                and axis_metric_exclusion is not None
            ):
                joint_exclusion = "multiple_metrics_unavailable"
            else:
                joint_exclusion = boundary_metric_exclusion or axis_metric_exclusion
            solve_cost = float(
                _slice_array(session, "wall_seconds", session_index).item()
            )
            if not np.isfinite(solve_cost):
                solve_cost = None
            boundary_pass = (
                None if boundary_cm is None else boundary_cm <= BOUNDARY_LIMIT_CM
            )
            axis_pass = None if axis_cm is None else axis_cm <= AXIS_LIMIT_CM
            results.append(
                SliceFidelity(
                    manifest_row=int(row["row"]),
                    session_index=session_index,
                    time_s=float(slice_times[position]),
                    recorded_conditioned=recorded_conditioned,
                    conditioned=is_conditioned,
                    reclassified_as_free=reclassified_as_free,
                    recorded_converged=bool(row.get("converged", False)),
                    semantically_converged=True,
                    nova_axis_finite=nova_axis_finite,
                    conditioned_branch_guard_ok=branch_ok,
                    flat_top=bool(is_flat_top),
                    evidence_eligible=joint_evidence_eligible,
                    boundary_evidence_eligible=boundary_evidence_eligible,
                    axis_evidence_eligible=axis_evidence_eligible,
                    joint_evidence_eligible=joint_evidence_eligible,
                    boundary_rms_cm=boundary_cm,
                    axis_offset_cm=axis_cm,
                    boundary_within_limit=boundary_pass,
                    axis_within_limit=axis_pass,
                    joint_within_limits=(
                        None
                        if boundary_pass is None or axis_pass is None
                        else boundary_pass and axis_pass
                    ),
                    nova_solve_wall_seconds=solve_cost,
                    exclusion_reason=joint_exclusion,
                    boundary_exclusion_reason=boundary_exclusion,
                    axis_exclusion_reason=axis_exclusion,
                    joint_exclusion_reason=joint_exclusion,
                )
            )

    return {
        "shot_id": int(shot_id),
        "session_path": str(session_path.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "flat_top_current": current_receipt,
        "conditioned_predicate": {
            "recorded_conditioned_rows": recorded_conditioned_rows,
            "reclassified_as_free_rows": reclassified_rows,
            "reclassified_semantically_converged_rows": reclassified_converged_rows,
            "reclassified_semantically_unconverged_rows": (
                reclassified_rows - reclassified_converged_rows
            ),
        },
        "summary": _aggregate_slices(results),
        "slices": [asdict(item) for item in results],
    }


def score_carriers(
    shots: Sequence[int] = FROZEN_CARRIER_SHOTS,
    *,
    session_root: Path = DEFAULT_SESSION_ROOT,
    level2_root: Path = equilibrium_labels.DEFAULT_LEVEL2_ROOT,
    level1_root: Path = LEVEL1_DIR,
) -> dict[str, Any]:
    """Score the frozen carrier set and return a JSON-ready verdict."""
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
        SliceFidelity(**item) for shot in shot_results for item in shot["slices"]
    ]
    aggregate = _aggregate_slices(all_slices)
    predicate_totals = {
        key: sum(int(shot["conditioned_predicate"][key]) for shot in shot_results)
        for key in (
            "recorded_conditioned_rows",
            "reclassified_as_free_rows",
            "reclassified_semantically_converged_rows",
            "reclassified_semantically_unconverged_rows",
        )
    }
    aggregate["recorded_conditioned_row_count"] = predicate_totals[
        "recorded_conditioned_rows"
    ]
    aggregate["reclassified_as_free_count"] = predicate_totals[
        "reclassified_as_free_rows"
    ]
    aggregate["reclassified_semantically_converged_count"] = predicate_totals[
        "reclassified_semantically_converged_rows"
    ]
    aggregate["reclassified_semantically_unconverged_count"] = predicate_totals[
        "reclassified_semantically_unconverged_rows"
    ]
    corrected_counts = {
        "joint": {
            "pass_count": aggregate["joint_pass_count"],
            "denominator": aggregate["joint_evidence_slice_count"],
        },
        "boundary": {
            "pass_count": aggregate["boundary_pass_count"],
            "denominator": aggregate["boundary_evidence_slice_count"],
        },
        "axis": {
            "pass_count": aggregate["axis_pass_count"],
            "denominator": aggregate["axis_evidence_slice_count"],
        },
    }
    denominator_changes = {
        arm: int(corrected_counts[arm]["denominator"])
        - int(SUPERSEDED_PUBLISHED_COUNTS[arm]["denominator"])
        for arm in ("joint", "boundary", "axis")
    }
    reclassified = int(aggregate["reclassified_as_free_count"])
    recovered = int(aggregate["reclassified_semantically_converged_count"])
    recovered_without_axis = int(aggregate["reclassified_nova_axis_non_finite_count"])
    verdict_word = "PASS" if aggregate["passed"] else "FAIL"
    outcome_changed = bool(aggregate["passed"]) != SUPERSEDED_GATE_PASSED
    verdict_statement = (
        f"{verdict_word} — corrected conditioned predicate left the gate outcome "
        f"unchanged; denominator changes: joint {denominator_changes['joint']:+d}, "
        f"boundary {denominator_changes['boundary']:+d}, axis "
        f"{denominator_changes['axis']:+d}."
    )
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "gate": "physics_fidelity",
        "verdict": "pass" if aggregate["passed"] else "fail",
        "verdict_statement": verdict_statement,
        "passed": bool(aggregate["passed"]),
        "comparison": {
            "superseded_pre_correction": SUPERSEDED_PUBLISHED_COUNTS,
            "corrected_conditioned_predicate": corrected_counts,
            "denominator_changes": denominator_changes,
            "gate_outcome_changed": outcome_changed,
            "recovered_rows": {
                "reclassified_as_free": reclassified,
                "semantically_converged": recovered,
                "semantically_unconverged": reclassified - recovered,
                "nova_magnetic_axis_non_finite": recovered_without_axis,
                "classification_statement": (
                    f"The corrected predicate reclassifies {reclassified} rows as "
                    f"free; {recovered} are semantically converged and "
                    f"{reclassified - recovered} remain unconverged."
                ),
                "axis_denominator_statement": (
                    f"All {recovered_without_axis} of {recovered} recovered rows "
                    "carry non-finite Nova magnetic-axis R or Z, so they cannot "
                    "enlarge the axis denominator."
                ),
            },
        },
        "thresholds": {
            "boundary_rms_cm_max": BOUNDARY_LIMIT_CM,
            "axis_offset_cm_max": AXIS_LIMIT_CM,
            "minimum_joint_pass_fraction": MIN_PASS_FRACTION,
        },
        "flat_top_definition": {
            "signal": "level1 efm/plasma_current_c",
            "units": "A",
            "slice_rule": (
                "interpolated absolute current >= 80% of shot maximum absolute current"
            ),
        },
        "evidence_rule": (
            "semantically converged free Nova slices in flat top, where a row "
            "recorded conditioned is reclassified as free when its conditioning "
            "attempt has an exception and zero trips; genuinely conditioned rows "
            "are reported but excluded, and each metric requires its own finite "
            "EFIT and Nova geometry"
        ),
        "boundary_metric": (
            "RMS difference of eight LCFS radii at equilibrium_labels.LCFS_ANGLES; "
            "Nova psi_norm=1 surface is ray-cast about the Nova axis and only finite "
            "EFIT radius components are scored"
        ),
        "axis_metric": "Euclidean R-Z magnetic-axis offset",
        "sources": {
            "session_root": str(Path(session_root).resolve()),
            "level2_root": str(Path(level2_root).resolve()),
            "level1_root": str(Path(level1_root).resolve()),
            "efit_loader": (
                "imas_ambix.worldmodel.equilibrium_labels.load_equilibrium_geometry"
            ),
            "policy_digest": EXPECTED_POLICY_DIGEST,
            "carrier_identity": EXPECTED_CARRIER_IDENTITY,
        },
        "shots": shot_results,
        "aggregate": aggregate,
    }


def write_verdict_figure(verdict: Mapping[str, Any], path: Path) -> None:
    """Write a two-panel per-slice boundary and axis fidelity figure."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: PLC0415
    from matplotlib.figure import Figure  # noqa: PLC0415

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(11.5, 5.4), dpi=150, constrained_layout=True)
    FigureCanvasAgg(figure)
    boundary_axes, axis_axes = figure.subplots(1, 2, sharex=True)
    palette = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00")
    for color, shot in zip(palette, verdict["shots"], strict=True):
        slices = shot["slices"]
        times = np.asarray([item["time_s"] for item in slices])
        boundary = np.asarray(
            [
                np.nan if item["boundary_rms_cm"] is None else item["boundary_rms_cm"]
                for item in slices
            ]
        )
        axis = np.asarray(
            [
                np.nan if item["axis_offset_cm"] is None else item["axis_offset_cm"]
                for item in slices
            ]
        )
        eligible = np.asarray(
            [item["evidence_eligible"] for item in slices], dtype=bool
        )
        conditioned = np.asarray([item["conditioned"] for item in slices], dtype=bool)
        label = str(shot["shot_id"])
        boundary_axes.scatter(
            times[eligible], boundary[eligible], s=16, color=color, label=label
        )
        axis_axes.scatter(
            times[eligible], axis[eligible], s=16, color=color, label=label
        )
        boundary_axes.scatter(
            times[conditioned],
            boundary[conditioned],
            s=26,
            facecolors="none",
            edgecolors=color,
            marker="s",
        )
        axis_axes.scatter(
            times[conditioned],
            axis[conditioned],
            s=26,
            facecolors="none",
            edgecolors=color,
            marker="s",
        )
    for axes, title, limit in (
        (boundary_axes, "LCFS boundary RMS", BOUNDARY_LIMIT_CM),
        (axis_axes, "Magnetic-axis offset", AXIS_LIMIT_CM),
    ):
        axes.axhline(limit, color="#333333", linestyle="--", linewidth=1.0)
        axes.set_title(title)
        axes.set_xlabel("Time (s)")
        axes.set_ylabel("Distance (cm)")
        axes.grid(alpha=0.2)
    boundary_axes.legend(title="Shot", ncol=2, fontsize=8)
    comparison = verdict["comparison"]
    corrected = comparison["corrected_conditioned_predicate"]
    superseded = comparison["superseded_pre_correction"]
    figure.suptitle(
        f"Physics fidelity: {str(verdict['verdict']).upper()}  |  "
        f"joint {corrected['joint']['pass_count']}/"
        f"{corrected['joint']['denominator']} corrected vs "
        f"{superseded['joint']['pass_count']}/"
        f"{superseded['joint']['denominator']} superseded\n"
        f"boundary {corrected['boundary']['pass_count']}/"
        f"{corrected['boundary']['denominator']}; axis "
        f"{corrected['axis']['pass_count']}/{corrected['axis']['denominator']}; "
        "open squares are genuinely conditioned/excluded"
    )
    figure.savefig(output)


def write_verdict(verdict: Mapping[str, Any], output_dir: Path) -> tuple[Path, Path]:
    """Write the detailed JSON verdict and its figure."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "verdict.json"
    figure_path = directory / "verdict.png"
    json_path.write_text(
        json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_verdict_figure(verdict, figure_path)
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
    verdict = score_carriers(
        args.shots,
        session_root=args.session_root,
        level2_root=args.level2_root,
        level1_root=args.level1_root,
    )
    paths = write_verdict(verdict, args.output_dir)
    print(
        json.dumps(
            {
                "verdict": verdict["verdict"],
                "aggregate": verdict["aggregate"],
                "outputs": [str(path) for path in paths],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
