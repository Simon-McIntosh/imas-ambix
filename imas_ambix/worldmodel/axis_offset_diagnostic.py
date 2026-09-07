"""Diagnose signed magnetic-axis offsets against evaluator-only EFIT geometry.

The EFIT reconstruction enters only as evaluator evidence and is never exposed
as a training input.  Magnetic-axis geometry is read through
:mod:`imas_ambix.worldmodel.equilibrium_labels`; the EFIT current centroid is
read from its level-one ``efm`` signal.  The diagnostic keeps every converged
slice in its receipt, then reports the free flat-top subset used to interpret
the physics-fidelity axis miss.  Nova's current centroid is retained as a third
geometric referent, including signed radial and vertical components.
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

from imas_ambix.camdyn.dataset import level1_shot_path
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
CENTROID_AGREEMENT_CM = 3.0
CENTROID_LARGE_DISAGREEMENT_CM = 7.5
EFIT_CENTROID_R_SIGNAL_CANDIDATES = (
    "current_centrd_r",
    "current_centroid_r",
)


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
    efit_current_centroid_r_m: float | None
    d_r_cm: float | None
    d_z_cm: float | None
    axis_offset_cm: float | None
    nova_minus_efit_current_centroid_d_r_cm: float | None
    current_centroid_minus_nova_axis_d_r_cm: float | None
    current_centroid_minus_nova_axis_d_z_cm: float | None
    current_centroid_minus_efit_axis_d_r_cm: float | None
    current_centroid_minus_efit_axis_d_z_cm: float | None
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
    return signed_point_components_cm(
        nova_axis_r_m,
        nova_axis_z_m,
        efit_axis_r_m,
        efit_axis_z_m,
    )


def signed_point_components_cm(
    point_r_m: float,
    point_z_m: float,
    reference_r_m: float,
    reference_z_m: float,
) -> tuple[float, float]:
    """Return signed ``(point - reference)`` components in centimetres."""
    values = np.asarray(
        (point_r_m, point_z_m, reference_r_m, reference_z_m), dtype=np.float64
    )
    if not np.isfinite(values).all():
        raise ValueError("point coordinates must be finite")
    return (
        100.0 * float(point_r_m - reference_r_m),
        100.0 * float(point_z_m - reference_z_m),
    )


def signed_centroid_radial_offset_cm(
    nova_current_centroid_r_m: float,
    efit_current_centroid_r_m: float,
) -> float:
    """Return signed ``Nova - EFIT`` current-centroid radius in centimetres."""
    values = np.asarray(
        (nova_current_centroid_r_m, efit_current_centroid_r_m), dtype=np.float64
    )
    if not np.isfinite(values).all():
        raise ValueError("current-centroid radii must be finite")
    return 100.0 * float(nova_current_centroid_r_m - efit_current_centroid_r_m)


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


def _interpolate_finite_signal(
    native_times: np.ndarray,
    native_values: np.ndarray,
    query_times: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate a finite signal without extrapolation."""
    times = np.asarray(native_times, dtype=np.float64).reshape(-1)
    values = np.asarray(native_values, dtype=np.float64).reshape(-1)
    query = np.asarray(query_times, dtype=np.float64).reshape(-1)
    if times.shape != values.shape:
        raise ValueError("EFIT signal values need a matching time axis")
    finite = np.isfinite(times) & np.isfinite(values)
    if np.count_nonzero(finite) < 2:
        return np.full(query.shape, np.nan, dtype=np.float64)
    times = times[finite]
    values = values[finite]
    order = np.argsort(times, kind="stable")
    times = times[order]
    values = values[order]
    distinct = np.concatenate(([True], np.diff(times) > 0.0))
    times = times[distinct]
    values = values[distinct]
    result = np.full(query.shape, np.nan, dtype=np.float64)
    if times.size < 2:
        return result
    in_range = (query >= times[0]) & (query <= times[-1])
    result[in_range] = np.interp(query[in_range], times, values)
    return result


def load_efit_current_centroid_r(
    shot_id: int,
    frame_times: np.ndarray,
    *,
    level1_root: Path = LEVEL1_DIR,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load EFIT current-centroid R or return an explicit unavailable receipt.

    The magnetic axis is deliberately not a candidate: absence remains absence
    because substituting another physical quantity would invalidate the check.
    """
    import zarr  # noqa: PLC0415

    path = level1_shot_path(shot_id, level1_dir=Path(level1_root))
    store = zarr.open_group(str(path), mode="r")
    searched = [f"level1 efm/{name}" for name in EFIT_CENTROID_R_SIGNAL_CANDIDATES]
    if "efm" not in set(store.group_keys()):
        return np.full(np.asarray(frame_times).shape, np.nan), {
            "available": False,
            "signal": None,
            "searched_signals": searched,
            "reason": "level1 shot store has no efm group",
        }
    group = store["efm"]
    keys = set(group.array_keys())
    selected = next(
        (name for name in EFIT_CENTROID_R_SIGNAL_CANDIDATES if name in keys), None
    )
    if selected is None or "time" not in keys:
        reason = (
            "level1 efm group has no matching current-centroid major-radius signal"
            if selected is None
            else "level1 efm group has no time axis"
        )
        return np.full(np.asarray(frame_times).shape, np.nan), {
            "available": False,
            "signal": None,
            "searched_signals": searched,
            "reason": reason,
        }
    signal = group[selected]
    units = str(signal.attrs.get("units", ""))
    if units != "m":
        raise ValueError(
            f"shot {shot_id}: level1 efm/{selected} units are {units!r}, expected 'm'"
        )
    values = _interpolate_finite_signal(
        np.asarray(group["time"], dtype=np.float64),
        np.asarray(signal, dtype=np.float64),
        np.asarray(frame_times, dtype=np.float64),
    )
    return values, {
        "available": True,
        "signal": f"level1 efm/{selected}",
        "searched_signals": searched,
        "units": units,
        "description": str(signal.attrs.get("description", "")),
        "interpolation": "linear on finite native samples without extrapolation",
        "finite_interpolated_count": int(np.count_nonzero(np.isfinite(values))),
    }


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
    receipt["nova_minus_efit_current_centroid_dR_cm"] = receipt.pop(
        "nova_minus_efit_current_centroid_d_r_cm"
    )
    receipt["current_centroid_minus_nova_axis_dR_cm"] = receipt.pop(
        "current_centroid_minus_nova_axis_d_r_cm"
    )
    receipt["current_centroid_minus_nova_axis_dZ_cm"] = receipt.pop(
        "current_centroid_minus_nova_axis_d_z_cm"
    )
    receipt["current_centroid_minus_efit_axis_dR_cm"] = receipt.pop(
        "current_centroid_minus_efit_axis_d_r_cm"
    )
    receipt["current_centroid_minus_efit_axis_dZ_cm"] = receipt.pop(
        "current_centroid_minus_efit_axis_d_z_cm"
    )
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
        "nova_minus_efit_current_centroid_dR_cm": _signed_summary(
            [
                float(item.nova_minus_efit_current_centroid_d_r_cm)
                for item in evidence
                if item.nova_minus_efit_current_centroid_d_r_cm is not None
            ]
        ),
        "current_centroid_minus_nova_axis": {
            "dR_cm": _signed_summary(
                [
                    float(item.current_centroid_minus_nova_axis_d_r_cm)
                    for item in evidence
                    if item.current_centroid_minus_nova_axis_d_r_cm is not None
                ]
            ),
            "dZ_cm": _signed_summary(
                [
                    float(item.current_centroid_minus_nova_axis_d_z_cm)
                    for item in evidence
                    if item.current_centroid_minus_nova_axis_d_z_cm is not None
                ]
            ),
        },
        "current_centroid_minus_efit_axis": {
            "dR_cm": _signed_summary(
                [
                    float(item.current_centroid_minus_efit_axis_d_r_cm)
                    for item in evidence
                    if item.current_centroid_minus_efit_axis_d_r_cm is not None
                ]
            ),
            "dZ_cm": _signed_summary(
                [
                    float(item.current_centroid_minus_efit_axis_d_z_cm)
                    for item in evidence
                    if item.current_centroid_minus_efit_axis_d_z_cm is not None
                ]
            ),
        },
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


def classify_centroid_radial(
    summary: Mapping[str, Any], signal_receipt: Mapping[str, Any]
) -> tuple[str, str]:
    """Classify Nova-to-EFIT centroid-R agreement without a substitute signal."""
    radial = summary["nova_minus_efit_current_centroid_dR_cm"]
    if not bool(signal_receipt["available"]) or int(radial["count"]) == 0:
        searched = ", ".join(signal_receipt["searched_signals"])
        return (
            "efit_current_centroid_r_unavailable",
            "Unavailable: EFIT publishes no usable current-centroid major radius "
            f"for these evidence slices; searched {searched}, and the magnetic "
            "axis was not substituted.",
        )
    median = float(radial["median"])
    magnitude = abs(median)
    signal = str(signal_receipt["signals"][0])
    if magnitude <= CENTROID_AGREEMENT_CM:
        return (
            "agree_within_few_centimetres",
            f"Agreement: Nova and EFIT current-centroid R agree within a few "
            f"centimetres (median Nova - EFIT {median:+.2f} cm) using {signal}.",
        )
    if magnitude >= CENTROID_LARGE_DISAGREEMENT_CM:
        return (
            "disagree_by_about_ten_centimetres",
            f"Disagreement: Nova and EFIT current-centroid R differ by something "
            f"like ten centimetres (median Nova - EFIT {median:+.2f} cm) using "
            f"{signal}.",
        )
    return (
        "intermediate_radial_disagreement",
        f"Intermediate disagreement: Nova and EFIT current-centroid R differ by "
        f"more than a few but less than about ten centimetres (median Nova - EFIT "
        f"{median:+.2f} cm) using {signal}.",
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
    efit_centroid_r, centroid_r_receipt = load_efit_current_centroid_r(
        shot_id, slice_times, level1_root=Path(level1_root)
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
            referee_centroid_r = _optional_scalar(efit_centroid_r[position])
            d_r: float | None = None
            d_z: float | None = None
            axis_distance: float | None = None
            centroid_radial: float | None = None
            centroid_from_nova_axis_r: float | None = None
            centroid_from_nova_axis_z: float | None = None
            centroid_from_efit_axis_r: float | None = None
            centroid_from_efit_axis_z: float | None = None
            nova_centroid: float | None = None
            efit_centroid: float | None = None
            try:
                d_r, d_z = signed_axis_offset_cm(nova_r, nova_z, efit_r, efit_z)
                axis_distance = float(np.hypot(d_r, d_z))
            except TypeError, ValueError:
                exclusion = "non_finite_axis"
            with suppress(TypeError, ValueError):
                centroid_radial = signed_centroid_radial_offset_cm(
                    centroid_r, referee_centroid_r
                )
            with suppress(TypeError, ValueError):
                centroid_from_nova_axis_r, centroid_from_nova_axis_z = (
                    signed_point_components_cm(centroid_r, centroid_z, nova_r, nova_z)
                )
            with suppress(TypeError, ValueError):
                centroid_from_efit_axis_r, centroid_from_efit_axis_z = (
                    signed_point_components_cm(centroid_r, centroid_z, efit_r, efit_z)
                )
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
                    efit_current_centroid_r_m=referee_centroid_r,
                    d_r_cm=d_r,
                    d_z_cm=d_z,
                    axis_offset_cm=axis_distance,
                    nova_minus_efit_current_centroid_d_r_cm=centroid_radial,
                    current_centroid_minus_nova_axis_d_r_cm=(centroid_from_nova_axis_r),
                    current_centroid_minus_nova_axis_d_z_cm=(centroid_from_nova_axis_z),
                    current_centroid_minus_efit_axis_d_r_cm=(centroid_from_efit_axis_r),
                    current_centroid_minus_efit_axis_d_z_cm=(centroid_from_efit_axis_z),
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
        "efit_current_centroid_r": centroid_r_receipt,
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
    renamed_slice_keys = {
        "dR_cm": "d_r_cm",
        "dZ_cm": "d_z_cm",
        "nova_minus_efit_current_centroid_dR_cm": (
            "nova_minus_efit_current_centroid_d_r_cm"
        ),
        "current_centroid_minus_nova_axis_dR_cm": (
            "current_centroid_minus_nova_axis_d_r_cm"
        ),
        "current_centroid_minus_nova_axis_dZ_cm": (
            "current_centroid_minus_nova_axis_d_z_cm"
        ),
        "current_centroid_minus_efit_axis_dR_cm": (
            "current_centroid_minus_efit_axis_d_r_cm"
        ),
        "current_centroid_minus_efit_axis_dZ_cm": (
            "current_centroid_minus_efit_axis_d_z_cm"
        ),
    }
    all_slices = [
        AxisOffsetSlice(
            **{
                **{
                    key: value
                    for key, value in item.items()
                    if key not in renamed_slice_keys
                },
                **{
                    destination: item[source]
                    for source, destination in renamed_slice_keys.items()
                },
            }
        )
        for shot in shot_results
        for item in shot["slices"]
    ]
    aggregate = aggregate_slices(all_slices)
    classification, verdict = classify_offset(aggregate)
    centroid_signals = sorted(
        {
            str(shot["efit_current_centroid_r"]["signal"])
            for shot in shot_results
            if shot["efit_current_centroid_r"]["signal"] is not None
        }
    )
    centroid_signal_receipt = {
        "available": len(centroid_signals) > 0,
        "available_shot_count": sum(
            bool(shot["efit_current_centroid_r"]["available"]) for shot in shot_results
        ),
        "shot_count": len(shot_results),
        "signals": centroid_signals,
        "searched_signals": [
            f"level1 efm/{name}" for name in EFIT_CENTROID_R_SIGNAL_CANDIDATES
        ],
        "magnetic_axis_substituted": False,
    }
    centroid_classification, centroid_verdict = classify_centroid_radial(
        aggregate, centroid_signal_receipt
    )
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "diagnostic": "signed_axis_offset",
        "classification": classification,
        "verdict": verdict,
        "centroid_radial_classification": centroid_classification,
        "centroid_radial_verdict": centroid_verdict,
        "offset_definition": {
            "dR_cm": "100 * (Nova magnetic_axis_r - EFIT magnetic_axis_r)",
            "dZ_cm": "100 * (Nova magnetic_axis_z - EFIT magnetic_axis_z)",
            "axis_offset_cm": "hypot(dR_cm, dZ_cm)",
            "nova_minus_efit_current_centroid_dR_cm": (
                "100 * (Nova current_centroid_r - EFIT current-centroid R)"
            ),
            "current_centroid_minus_nova_axis": (
                "100 * (Nova current_centroid_r,z - Nova magnetic_axis_r,z)"
            ),
            "current_centroid_minus_efit_axis": (
                "100 * (Nova current_centroid_r,z - EFIT magnetic axis R,Z)"
            ),
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
            "efit_current_centroid_r": centroid_signal_receipt,
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


def _centroid_radial_artifact(diagnostic: Mapping[str, Any]) -> dict[str, Any]:
    """Promote the radial-centroid verdict while retaining full slice evidence."""
    return {
        **diagnostic,
        "diagnostic": "current_centroid_radial_offset",
        "classification": diagnostic["centroid_radial_classification"],
        "verdict": diagnostic["centroid_radial_verdict"],
        "axis_offset_interpretation": {
            "classification": diagnostic["classification"],
            "verdict": diagnostic["verdict"],
        },
    }


def write_centroid_radial_figure(diagnostic: Mapping[str, Any], path: Path) -> None:
    """Plot signed centroid-R evidence and both centroid-to-axis components."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: PLC0415
    from matplotlib.figure import Figure  # noqa: PLC0415

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(13.0, 7.5), dpi=150, constrained_layout=True)
    FigureCanvasAgg(figure)
    radial_axes, nova_axes, efit_axes = figure.subplots(1, 3)
    palette = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00")
    radial_point_count = 0
    for color, shot in zip(palette, diagnostic["shots"], strict=False):
        eligible = [item for item in shot["slices"] if item["evidence_eligible"]]
        label = str(shot["shot_id"])
        radial = [
            (item["time_s"], item["nova_minus_efit_current_centroid_dR_cm"])
            for item in eligible
            if item["nova_minus_efit_current_centroid_dR_cm"] is not None
        ]
        if radial:
            radial_point_count += len(radial)
            radial_axes.plot(
                [point[0] for point in radial],
                [point[1] for point in radial],
                "o-",
                color=color,
                linewidth=0.8,
                markersize=3,
                label=label,
            )
        nova_components = [
            (
                item["time_s"],
                item["current_centroid_minus_nova_axis_dR_cm"],
                item["current_centroid_minus_nova_axis_dZ_cm"],
            )
            for item in eligible
            if item["current_centroid_minus_nova_axis_dR_cm"] is not None
            and item["current_centroid_minus_nova_axis_dZ_cm"] is not None
        ]
        if nova_components:
            nova_axes.plot(
                [point[0] for point in nova_components],
                [point[1] for point in nova_components],
                "o",
                color=color,
                markersize=3,
                label=label,
            )
            nova_axes.plot(
                [point[0] for point in nova_components],
                [point[2] for point in nova_components],
                "x",
                color=color,
                markersize=4,
            )
        efit_components = [
            (
                item["time_s"],
                item["current_centroid_minus_efit_axis_dR_cm"],
                item["current_centroid_minus_efit_axis_dZ_cm"],
            )
            for item in eligible
            if item["current_centroid_minus_efit_axis_dR_cm"] is not None
            and item["current_centroid_minus_efit_axis_dZ_cm"] is not None
        ]
        if efit_components:
            efit_axes.plot(
                [point[0] for point in efit_components],
                [point[1] for point in efit_components],
                "o",
                color=color,
                markersize=3,
                label=label,
            )
            efit_axes.plot(
                [point[0] for point in efit_components],
                [point[2] for point in efit_components],
                "x",
                color=color,
                markersize=4,
            )
    if radial_point_count == 0:
        searched = ", ".join(
            diagnostic["sources"]["efit_current_centroid_r"]["searched_signals"]
        )
        radial_axes.text(
            0.5,
            0.5,
            f"EFIT current-centroid R unavailable\nSearched: {searched}\n"
            "No magnetic-axis substitution",
            ha="center",
            va="center",
            transform=radial_axes.transAxes,
        )
    for axes in (radial_axes, nova_axes, efit_axes):
        axes.axhline(0.0, color="#333333", linewidth=0.8)
        axes.set_xlabel("Time (s)")
        axes.set_ylabel("Signed component (cm)")
        axes.grid(alpha=0.2)
    radial_axes.set_title("Nova centroid R - EFIT centroid R")
    nova_axes.set_title("Nova centroid - Nova axis")
    efit_axes.set_title("Nova centroid - EFIT axis")
    if radial_point_count:
        radial_axes.legend(title="Shot", fontsize=7)
    nova_axes.legend(title="Shot; circles dR, crosses dZ", fontsize=7)
    aggregate = diagnostic["aggregate"]
    radial_summary = aggregate["nova_minus_efit_current_centroid_dR_cm"]
    median_text = (
        "unavailable"
        if radial_summary["median"] is None
        else f"{float(radial_summary['median']):+.2f} cm"
    )
    display_classification = str(diagnostic["classification"]).replace("_", " ")
    signals = diagnostic["sources"]["efit_current_centroid_r"].get("signals", [])
    signal_text = signals[0] if signals else "no EFIT centroid-R signal"
    figure.suptitle(
        f"Current-centroid radial check: {display_classification.upper()}\n"
        f"Median Nova - EFIT centroid R {median_text}; source {signal_text}"
    )
    figure.savefig(output)


def write_centroid_radial_diagnostic(
    diagnostic: Mapping[str, Any], output_dir: Path
) -> tuple[Path, Path]:
    """Write the radial-centroid JSON diagnostic and matching figure."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    artifact = _centroid_radial_artifact(diagnostic)
    json_path = directory / "centroid-radial.json"
    figure_path = directory / "centroid-radial.png"
    json_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_centroid_radial_figure(artifact, figure_path)
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
    parser.add_argument(
        "--artifact",
        choices=("axis", "centroid-radial", "both"),
        default="axis",
        help="select which diagnostic artifact pair to write",
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
    paths: tuple[Path, ...] = ()
    if args.artifact in {"axis", "both"}:
        paths += write_diagnostic(diagnostic, args.output_dir)
    if args.artifact in {"centroid-radial", "both"}:
        paths += write_centroid_radial_diagnostic(diagnostic, args.output_dir)
    print(
        json.dumps(
            {
                "classification": diagnostic["classification"],
                "verdict": diagnostic["verdict"],
                "centroid_radial_classification": diagnostic[
                    "centroid_radial_classification"
                ],
                "centroid_radial_verdict": diagnostic["centroid_radial_verdict"],
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
