"""Adjudicate low plasma-current values on labelled challenge frames.

The audit reproduces the corpus census contract exactly: the native plasma-
current series is linearly interpolated onto ``efit_times`` and a labelled
frame enters the audited population when its finite magnitude is below 50 kA.
Each such frame receives one mutually exclusive cause based on native time
support, whole-channel usability, and its position relative to labelled
frames with usable current.  The other recorded drive channels provide an
independent co-activity receipt; they do not change the cause assignment.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

DEFAULT_CORPUS = Path("/work/projects/imas_gpu/sophelio/raw/data/diii_d_train")
DEFAULT_ARTIFACT = Path("imas_ambix/challenge/artifacts/ip_channel_audit.json")
PLASMA_CURRENT_THRESHOLD_KA = 50.0
EXPECTED_SHOTS = 7_041
EXPECTED_LABELLED_FRAMES = 1_559_340
EXPECTED_AUDITED_FRAMES = 658_787

_DRIVE_COLUMNS = (
    "magnetics_ECOILA",
    *(f"magnetics_F{index}{side}" for index in range(1, 10) for side in "AB"),
)
_OTHER_ACTUATOR_COLUMNS = (*_DRIVE_COLUMNS, "magnetics_bcoil")
_READ_COLUMNS = (
    "efit_times",
    "magnetics_plasma_current_times",
    "magnetics_plasma_current",
)


class FrameCause(StrEnum):
    """Mutually exclusive explanations for an audited low-current frame."""

    OUTSIDE_NATIVE_SUPPORT = "outside_native_support"
    WHOLE_CHANNEL_BELOW_THRESHOLD = "whole_channel_below_threshold"
    PULSE_DISJOINT_FROM_LABEL_WINDOW = "pulse_disjoint_from_label_window"
    LABEL_GRID_MISSES_NATIVE_ACTIVITY = "label_grid_misses_native_activity"
    ALIGNED_LEADING_EDGE = "aligned_leading_edge"
    ALIGNED_TRAILING_EDGE = "aligned_trailing_edge"
    INTERIOR_LOW_CURRENT = "interior_low_current"


CAUSE_ORDER = tuple(FrameCause)


@dataclass(frozen=True)
class ShotAudit:
    """Compact per-shot receipt returned to the corpus aggregator."""

    labelled_frames: int
    audited_frames: int
    causes: dict[FrameCause, dict[str, Any]]


def _array(table: Any, name: str) -> np.ndarray:
    return np.asarray(table[name][0].as_py(), dtype=np.float64)


def interpolate_plasma_current(
    label_time_ms: np.ndarray,
    native_time_ms: np.ndarray,
    native_current_ka: np.ndarray,
) -> np.ndarray:
    """Interpolate the native plasma-current channel onto the label time base."""

    label_time = np.asarray(label_time_ms, dtype=np.float64)
    native_time = np.asarray(native_time_ms, dtype=np.float64)
    native_current = np.asarray(native_current_ka, dtype=np.float64)
    if native_time.ndim != 1 or native_current.shape != native_time.shape:
        raise ValueError("native plasma-current values must share one time base")
    if native_time.size < 2 or np.any(np.diff(native_time) <= 0.0):
        raise ValueError("native plasma-current time must be strictly increasing")
    return np.interp(label_time, native_time, native_current)


def classify_low_current_frames(
    label_time_ms: np.ndarray,
    native_time_ms: np.ndarray,
    native_current_ka: np.ndarray,
    *,
    threshold_ka: float = PLASMA_CURRENT_THRESHOLD_KA,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return interpolated current, audited mask, and a cause per audited frame.

    Cause precedence is deliberate.  Extrapolation beyond native support is
    identified before a whole-channel encoding, and whole-channel usability is
    decided before temporal position within a labelled pulse.
    """

    label_time = np.asarray(label_time_ms, dtype=np.float64)
    native_time = np.asarray(native_time_ms, dtype=np.float64)
    native_current = np.asarray(native_current_ka, dtype=np.float64)
    interpolated = interpolate_plasma_current(label_time, native_time, native_current)
    audited = np.isfinite(interpolated) & (np.abs(interpolated) < threshold_ka)
    causes = np.full(label_time.shape, "", dtype=object)

    outside = (label_time < native_time[0]) | (label_time > native_time[-1])
    causes[audited & outside] = FrameCause.OUTSIDE_NATIVE_SUPPORT.value
    unresolved = audited & (causes == "")

    finite_native = np.isfinite(native_current)
    native_active = finite_native & (np.abs(native_current) >= threshold_ka)
    if not np.any(native_active):
        causes[unresolved] = FrameCause.WHOLE_CHANNEL_BELOW_THRESHOLD.value
        return interpolated, audited, causes

    labelled_high = np.isfinite(interpolated) & ~audited
    if not np.any(labelled_high):
        active_indices = np.flatnonzero(native_active)
        pulse_disjoint = native_time[active_indices[-1]] < np.min(
            label_time
        ) or native_time[active_indices[0]] > np.max(label_time)
        cause = (
            FrameCause.PULSE_DISJOINT_FROM_LABEL_WINDOW
            if pulse_disjoint
            else FrameCause.LABEL_GRID_MISSES_NATIVE_ACTIVITY
        )
        causes[unresolved] = cause.value
        return interpolated, audited, causes

    high_indices = np.flatnonzero(labelled_high)
    frame_indices = np.arange(label_time.size)
    leading = frame_indices < high_indices[0]
    trailing = frame_indices > high_indices[-1]
    causes[unresolved & leading] = FrameCause.ALIGNED_LEADING_EDGE.value
    causes[unresolved & trailing] = FrameCause.ALIGNED_TRAILING_EDGE.value
    causes[unresolved & (causes == "")] = FrameCause.INTERIOR_LOW_CURRENT.value
    return interpolated, audited, causes


def _nearest_indices(source_time: np.ndarray, target_time: np.ndarray) -> np.ndarray:
    right = np.clip(np.searchsorted(source_time, target_time), 1, source_time.size - 1)
    choose_left = np.abs(target_time - source_time[right - 1]) <= np.abs(
        source_time[right] - target_time
    )
    return right - choose_left.astype(np.int64)


def _actuator_statistics(parquet_file: pq.ParquetFile) -> dict[str, Any]:
    row_group = parquet_file.metadata.row_group(0)
    statistics = {
        column.path_in_schema.removesuffix(".list.element"): column.statistics
        for column in (
            row_group.column(index) for index in range(row_group.num_columns)
        )
    }
    peaks = []
    nonconstant = 0
    null_values = 0
    channels_with_samples = 0
    channels_with_finite_bounds = 0
    for name in _OTHER_ACTUATOR_COLUMNS:
        channel = statistics[name]
        has_samples = channel is not None and channel.num_values > 0
        channels_with_samples += int(has_samples)
        if not has_samples or not channel.has_min_max:
            peaks.append(np.nan)
            continue
        minimum = float(channel.min)
        maximum = float(channel.max)
        finite_bounds = np.isfinite(minimum) and np.isfinite(maximum)
        channels_with_finite_bounds += int(finite_bounds)
        null_values += int(channel.null_count or 0)
        nonconstant += int(finite_bounds and minimum != maximum)
        peaks.append(max(abs(minimum), abs(maximum)))
    return {
        "peak_abs": np.asarray(peaks, dtype=np.float64),
        "channels_with_samples": channels_with_samples,
        "channels_with_finite_bounds": channels_with_finite_bounds,
        "nonconstant_channels": nonconstant,
        "null_values": null_values,
    }


def _scan_shot(path: Path) -> ShotAudit:
    parquet_file = pq.ParquetFile(path)
    table = parquet_file.read(columns=list(_READ_COLUMNS), use_threads=False)
    label_time = _array(table, "efit_times")
    native_time = _array(table, "magnetics_plasma_current_times")
    native_current = _array(table, "magnetics_plasma_current")
    interpolated, audited, causes = classify_low_current_frames(
        label_time, native_time, native_current
    )

    actuator_statistics = _actuator_statistics(parquet_file)

    nearest = _nearest_indices(native_time, label_time)
    nearest_current = native_current[nearest]
    nearest_delta_ms = np.abs(label_time - native_time[nearest])
    native_active = np.isfinite(native_current) & (
        np.abs(native_current) >= PLASMA_CURRENT_THRESHOLD_KA
    )
    if np.any(native_active):
        active_indices = np.flatnonzero(native_active)
        active_start = native_time[active_indices[0]]
        active_end = native_time[active_indices[-1]]
    else:
        active_start = np.inf
        active_end = -np.inf

    cause_receipts: dict[FrameCause, dict[str, Any]] = {}
    for cause in CAUSE_ORDER:
        mask = causes == cause.value
        if not np.any(mask):
            continue
        times = label_time[mask]
        support_relation = np.where(
            times < native_time[0], -1, np.where(times > native_time[-1], 1, 0)
        )
        activity_relation = np.where(
            times < active_start, -1, np.where(times > active_end, 1, 0)
        )
        count = int(np.count_nonzero(mask))
        all_channels = len(_OTHER_ACTUATOR_COLUMNS)
        cause_receipts[cause] = {
            "count": count,
            "within_native_support": int(np.count_nonzero(support_relation == 0)),
            "before_native_support": int(np.count_nonzero(support_relation == -1)),
            "after_native_support": int(np.count_nonzero(support_relation == 1)),
            "before_native_activity": int(np.count_nonzero(activity_relation == -1)),
            "during_native_activity_span": int(
                np.count_nonzero(activity_relation == 0)
            ),
            "after_native_activity": int(np.count_nonzero(activity_relation == 1)),
            "nearest_native_delta_ms": nearest_delta_ms[mask],
            "interpolated_exact_zero": int(np.count_nonzero(interpolated[mask] == 0.0)),
            "nearest_native_exact_zero": int(
                np.count_nonzero(nearest_current[mask] == 0.0)
            ),
            "nearest_native_nonfinite": int(
                np.count_nonzero(~np.isfinite(nearest_current[mask]))
            ),
            "whole_channel_below_threshold": int(not np.any(native_active)),
            "other_all_present_frames": int(
                count
                if actuator_statistics["channels_with_samples"] == all_channels
                else 0
            ),
            "other_all_finite_bounds_frames": int(
                count
                if actuator_statistics["channels_with_finite_bounds"] == all_channels
                else 0
            ),
            "other_all_nonconstant_frames": int(
                count
                if actuator_statistics["nonconstant_channels"] == all_channels
                else 0
            ),
            "other_nonconstant_channels": np.asarray(
                [actuator_statistics["nonconstant_channels"]], dtype=np.float64
            ),
            "other_null_values": actuator_statistics["null_values"],
            "other_peak_abs": actuator_statistics["peak_abs"][None, :],
            "ip_peak_abs": np.asarray(
                [float(np.nanmax(np.abs(native_current)))], dtype=np.float64
            ),
        }

    return ShotAudit(
        labelled_frames=int(label_time.size),
        audited_frames=int(np.count_nonzero(audited)),
        causes=cause_receipts,
    )


def _quantiles(values: list[np.ndarray]) -> dict[str, float | None]:
    if not values:
        return {"p10": None, "median": None, "p90": None}
    array = np.concatenate(values)
    if array.size == 0:
        return {"p10": None, "median": None, "p90": None}
    p10, median, p90 = np.nanpercentile(array, [10.0, 50.0, 90.0])
    return {
        "p10": round(float(p10), 9),
        "median": round(float(median), 9),
        "p90": round(float(p90), 9),
    }


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 9) if denominator else 0.0


def _peak_covariation(
    ip_peak_abs: np.ndarray, other_peak_abs: np.ndarray
) -> dict[str, Any]:
    correlations: dict[str, float | None] = {}
    finite_correlations: list[float] = []
    log_ip = np.log1p(ip_peak_abs)
    for index, name in enumerate(_OTHER_ACTUATOR_COLUMNS):
        other = other_peak_abs[:, index]
        valid = np.isfinite(log_ip) & np.isfinite(other)
        log_other = np.log1p(other[valid])
        usable = (
            np.count_nonzero(valid) >= 3
            and np.ptp(log_ip[valid]) > 0.0
            and np.ptp(log_other) > 0.0
        )
        if not usable:
            correlations[name] = None
            continue
        correlation = float(np.corrcoef(log_ip[valid], log_other)[0, 1])
        correlations[name] = round(correlation, 9)
        finite_correlations.append(correlation)
    values = np.asarray(finite_correlations, dtype=np.float64)
    return {
        "measure": "Pearson correlation of log1p native shot peak magnitudes",
        "channel_correlations": correlations,
        "finite_channel_count": int(values.size),
        "positive_channel_count": int(np.count_nonzero(values > 0.0)),
        "median_correlation": (
            round(float(np.median(values)), 9) if values.size else None
        ),
        "median_absolute_correlation": (
            round(float(np.median(np.abs(values))), 9) if values.size else None
        ),
    }


def audit_corpus(paths: list[Path], *, workers: int) -> dict[str, Any]:
    """Audit every supplied corpus object and return a JSON-ready receipt."""

    totals: Counter[str] = Counter()
    contributing_shots: Counter[str] = Counter()
    arrays: dict[str, dict[str, list[np.ndarray]]] = defaultdict(
        lambda: defaultdict(list)
    )
    labelled_frames = 0
    audited_frames = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for number, shot in enumerate(
            executor.map(_scan_shot, paths, chunksize=4), start=1
        ):
            labelled_frames += shot.labelled_frames
            audited_frames += shot.audited_frames
            for cause, receipt in shot.causes.items():
                name = cause.value
                contributing_shots[name] += 1
                for key, value in receipt.items():
                    if isinstance(value, np.ndarray):
                        arrays[name][key].append(value)
                    else:
                        totals[f"{name}:{key}"] += int(value)
            if number % 250 == 0:
                print(f"AUDITED {number}/{len(paths)}", flush=True)

    class_receipts: list[dict[str, Any]] = []
    for cause in CAUSE_ORDER:
        name = cause.value
        count = totals[f"{name}:count"]
        if count == 0:
            continue
        other_peaks = np.concatenate(arrays[name]["other_peak_abs"], axis=0)
        ip_peaks = np.concatenate(arrays[name]["ip_peak_abs"])
        within = totals[f"{name}:within_native_support"]
        exact_zero = totals[f"{name}:interpolated_exact_zero"]
        nearest_zero = totals[f"{name}:nearest_native_exact_zero"]
        other_present = totals[f"{name}:other_all_present_frames"]
        other_finite = totals[f"{name}:other_all_finite_bounds_frames"]
        other_nonconstant = totals[f"{name}:other_all_nonconstant_frames"]
        class_receipts.append(
            {
                "name": name,
                "frame_count": count,
                "shot_count": contributing_shots[name],
                "time_alignment": {
                    "within_native_support_count": within,
                    "within_native_support_rate": _rate(within, count),
                    "before_native_support_count": totals[
                        f"{name}:before_native_support"
                    ],
                    "after_native_support_count": totals[
                        f"{name}:after_native_support"
                    ],
                    "before_native_activity_count": totals[
                        f"{name}:before_native_activity"
                    ],
                    "during_native_activity_span_count": totals[
                        f"{name}:during_native_activity_span"
                    ],
                    "after_native_activity_count": totals[
                        f"{name}:after_native_activity"
                    ],
                    "nearest_native_delta_ms": _quantiles(
                        arrays[name]["nearest_native_delta_ms"]
                    ),
                },
                "sentinel_or_missing_encoding": {
                    "interpolated_exact_zero_count": exact_zero,
                    "interpolated_exact_zero_rate": _rate(exact_zero, count),
                    "nearest_native_exact_zero_count": nearest_zero,
                    "nearest_native_exact_zero_rate": _rate(nearest_zero, count),
                    "nearest_native_nonfinite_count": totals[
                        f"{name}:nearest_native_nonfinite"
                    ],
                    "whole_channel_below_threshold_shot_count": totals[
                        f"{name}:whole_channel_below_threshold"
                    ],
                },
                "other_actuator_covariation": {
                    "evidence_basis": (
                        "Parquet full-native-series statistics for all 20 other "
                        "actuator channels on every contributing shot"
                    ),
                    "all_twenty_channels_present_frame_count": other_present,
                    "all_twenty_channels_present_frame_rate": _rate(
                        other_present, count
                    ),
                    "all_twenty_channels_finite_bounds_frame_count": other_finite,
                    "all_twenty_channels_finite_bounds_frame_rate": _rate(
                        other_finite, count
                    ),
                    "all_twenty_channels_nonconstant_frame_count": other_nonconstant,
                    "all_twenty_channels_nonconstant_frame_rate": _rate(
                        other_nonconstant, count
                    ),
                    "nonconstant_channel_count_per_contributing_shot": _quantiles(
                        arrays[name]["other_nonconstant_channels"]
                    ),
                    "native_null_value_count": totals[f"{name}:other_null_values"],
                    "shot_peak_covariation_with_ip": _peak_covariation(
                        ip_peaks, other_peaks
                    ),
                },
                "downstream_verdict": _downstream_verdict(cause),
            }
        )

    return {
        "schema": "imas-ambix.challenge.ip-channel-audit",
        "population_contract": {
            "corpus": "full labelled DIII-D train corpus",
            "shot_count": len(paths),
            "labelled_frame_count": labelled_frames,
            "alignment": (
                "linear interpolation of native plasma current onto efit_times"
            ),
            "threshold": "finite abs(magnetics_plasma_current) < 50 kA",
            "audited_frame_count": audited_frames,
        },
        "class_precedence": [cause.value for cause in CAUSE_ORDER],
        "classes": class_receipts,
        "exclusion_policy": {
            "default_excluded_classes": [item["name"] for item in class_receipts],
            "ramp_only_classes": [
                item["name"]
                for item in class_receipts
                if item["name"]
                in {
                    FrameCause.ALIGNED_LEADING_EDGE.value,
                    FrameCause.ALIGNED_TRAILING_EDGE.value,
                }
            ],
            "rule": (
                "Every audited class is excluded from default confined-equilibrium "
                "training, calibration, scoring, and integral-constraint consumers. "
                "Aligned edge classes may be admitted only by explicitly ramp-aware "
                "limited-phase traversal code."
            ),
        },
    }


def _downstream_verdict(cause: FrameCause) -> dict[str, Any]:
    ramp_aware = cause in {
        FrameCause.ALIGNED_LEADING_EDGE,
        FrameCause.ALIGNED_TRAILING_EDGE,
    }
    return {
        "exclude_from_default_consumers": True,
        "scope": (
            "confined-equilibrium consumers; ramp-aware traversal may retain"
            if ramp_aware
            else "all plasma-current-dependent consumers"
        ),
    }


def validate_receipt(receipt: dict[str, Any]) -> None:
    """Reject incomplete scans and any non-partitioning class accounting."""

    population = receipt["population_contract"]
    if population["shot_count"] != EXPECTED_SHOTS:
        raise ValueError(
            f"expected {EXPECTED_SHOTS} shots, found {population['shot_count']}"
        )
    if population["labelled_frame_count"] != EXPECTED_LABELLED_FRAMES:
        raise ValueError(
            "labelled frame count differs from the pinned full-corpus census"
        )
    if population["audited_frame_count"] != EXPECTED_AUDITED_FRAMES:
        raise ValueError("audited frame count differs from the pinned population")
    class_total = sum(item["frame_count"] for item in receipt["classes"])
    if class_total != population["audited_frame_count"]:
        raise ValueError("cause classes do not partition the audited population")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    paths = sorted(args.corpus.glob("*.parquet"))
    receipt = audit_corpus(paths, workers=args.workers)
    validate_receipt(receipt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "shots": len(paths),
                "labelled_frames": receipt["population_contract"][
                    "labelled_frame_count"
                ],
                "audited_frames": receipt["population_contract"]["audited_frame_count"],
                "class_counts": {
                    item["name"]: item["frame_count"] for item in receipt["classes"]
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
