"""Measure the actual cadence of admitted Nova steering slices.

The decoder consumes histories on a fixed nominal cadence, while its
persistence comparator advances between consecutive converged session frames.
This audit measures those two notions of "previous frame" from each session's
recorded time coordinate.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import textwrap
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from imas_ambix.worldmodel.flux_label_dataset import (
    DEFAULT_HISTORY_SPACING_SECONDS,
    DEFAULT_SESSION_ROOT,
    EXPECTED_CARRIER_IDENTITY,
    EXPECTED_POLICY_DIGEST,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

FROZEN_CARRIER_SHOTS = (21978, 21983, 21985, 21986, 21989, 22086)
DEMO_CARRIER_SHOT = 22086
RELATIVE_GAP_TOLERANCE = 1.0e-3
HISTOGRAM_RESOLUTION_SECONDS = 1.0e-6
DEFAULT_OUTPUT_JSON = Path(
    "docs/figures/physics-carried-playable-plasma/alignment-audit/slice-spacing.json"
)
DEFAULT_OUTPUT_PNG = DEFAULT_OUTPUT_JSON.with_suffix(".png")


def _source_revision() -> str:
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _longest_true_run(values: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in np.asarray(values, dtype=bool).reshape(-1):
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _gap_histogram(gaps: np.ndarray) -> dict[str, Any]:
    rounded = (
        np.rint(np.asarray(gaps, dtype=np.float64) / HISTOGRAM_RESOLUTION_SECONDS)
        * HISTOGRAM_RESOLUTION_SECONDS
    )
    values, counts = np.unique(rounded, return_counts=True)
    return {
        "resolution_s": HISTOGRAM_RESOLUTION_SECONDS,
        "bins": [
            {"gap_s": float(value), "count": int(count)}
            for value, count in zip(values, counts, strict=True)
        ],
    }


def summarize_admitted_times(
    admitted_times_s: np.ndarray,
    *,
    nominal_gap_s: float = DEFAULT_HISTORY_SPACING_SECONDS,
    relative_tolerance: float = RELATIVE_GAP_TOLERANCE,
) -> dict[str, Any]:
    """Summarize intervals between consecutive admitted session slices."""
    times = np.asarray(admitted_times_s, dtype=np.float64).reshape(-1)
    if times.size and not np.isfinite(times).all():
        raise ValueError("admitted slice times must be finite")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("admitted slice times must be strictly increasing")
    if not np.isfinite(nominal_gap_s) or nominal_gap_s <= 0.0:
        raise ValueError("nominal gap must be finite and positive")
    if not np.isfinite(relative_tolerance) or relative_tolerance < 0.0:
        raise ValueError("relative tolerance must be finite and non-negative")

    gaps = np.diff(times)
    absolute_tolerance_s = nominal_gap_s * relative_tolerance
    nominal = np.abs(gaps - nominal_gap_s) <= absolute_tolerance_s
    comparison_count = int(gaps.size)
    non_nominal_count = int(np.count_nonzero(~nominal))
    fraction = non_nominal_count / comparison_count if comparison_count else 0.0
    distribution = {
        "minimum_s": float(np.min(gaps)) if comparison_count else None,
        "median_s": float(np.median(gaps)) if comparison_count else None,
        "maximum_s": float(np.max(gaps)) if comparison_count else None,
        "histogram": _gap_histogram(gaps),
    }
    return {
        "admitted_slice_count": int(times.size),
        "persistence_comparison_count": comparison_count,
        "nominal_gap_s": float(nominal_gap_s),
        "relative_tolerance": float(relative_tolerance),
        "absolute_tolerance_s": float(absolute_tolerance_s),
        "gap_distribution": distribution,
        "non_nominal_gap_count": non_nominal_count,
        "non_nominal_gap_fraction": float(fraction),
        "longest_uniform_gap_run": _longest_true_run(nominal),
        "all_gaps_nominal": bool(np.all(nominal)),
        "admitted_times_s": times.tolist(),
        "gaps_s": gaps.tolist(),
    }


def _load_manifest(path: Path, shot_id: int) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise ValueError(f"{path} is not an atomically complete session")
    if int(payload.get("shot", -1)) != shot_id:
        raise ValueError(f"{path} shot identity does not match {shot_id}")
    if str(payload.get("policy_digest", "")) != EXPECTED_POLICY_DIGEST:
        raise ValueError(f"{path} does not carry the pinned policy digest")
    if str(payload.get("carrier_identity", "")) != EXPECTED_CARRIER_IDENTITY:
        raise ValueError(f"{path} does not carry the pinned carrier identity")
    if not isinstance(payload.get("slices"), list):
        raise ValueError(f"{path} has no slice-row list")
    return payload


def _admitted_session_indices(
    slices: Sequence[Mapping[str, Any]], session_count: int
) -> tuple[np.ndarray, int]:
    admitted: list[int] = []
    session_index = 0
    for row in slices:
        if not isinstance(row, Mapping):
            raise ValueError("manifest contains a non-object slice row")
        if not bool(row.get("written", False)):
            continue
        if bool(row.get("converged", False)):
            admitted.append(session_index)
        session_index += 1
    if session_index != session_count:
        raise ValueError(
            f"manifest has {session_index} written rows but the session contains "
            f"{session_count} frames"
        )
    return np.asarray(admitted, dtype=np.int64), session_index


def score_session(
    shot_id: int,
    *,
    session_root: Path = DEFAULT_SESSION_ROOT,
) -> dict[str, Any]:
    """Measure admitted-slice spacing for one pinned carrier session."""
    import xarray as xr  # noqa: PLC0415

    root = Path(session_root)
    session_path = root / f"{shot_id}.nc"
    manifest_path = root / f"{shot_id}.manifest.json"
    manifest = _load_manifest(manifest_path, shot_id)
    slices = manifest["slices"]
    with xr.open_dataset(session_path, group="steering", engine="h5netcdf") as data:
        times = np.asarray(data["time"].values, dtype=np.float64).reshape(-1)
    admitted_indices, written_count = _admitted_session_indices(slices, times.size)
    summary = summarize_admitted_times(times[admitted_indices])
    summary.update(
        {
            "shot_id": int(shot_id),
            "demo_carrier": shot_id == DEMO_CARRIER_SHOT,
            "manifest_slice_count": len(slices),
            "written_slice_count": written_count,
            "unwritten_slice_count": len(slices) - written_count,
            "written_but_unconverged_count": written_count - int(admitted_indices.size),
            "session_path": str(session_path),
            "manifest_path": str(manifest_path),
        }
    )
    return summary


def _aggregate(shots: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    gaps = np.concatenate(
        [np.asarray(shot["gaps_s"], dtype=np.float64) for shot in shots]
    )
    aggregate = summarize_admitted_times_from_gaps(gaps)
    aggregate["carrier_count"] = len(shots)
    aggregate["admitted_slice_count"] = sum(
        int(shot["admitted_slice_count"]) for shot in shots
    )
    aggregate["longest_uniform_gap_run"] = max(
        (int(shot["longest_uniform_gap_run"]) for shot in shots), default=0
    )
    return aggregate


def summarize_admitted_times_from_gaps(
    gaps_s: np.ndarray,
    *,
    nominal_gap_s: float = DEFAULT_HISTORY_SPACING_SECONDS,
    relative_tolerance: float = RELATIVE_GAP_TOLERANCE,
) -> dict[str, Any]:
    """Summarize independent gaps without introducing cross-shot intervals."""
    gaps = np.asarray(gaps_s, dtype=np.float64).reshape(-1)
    if gaps.size and (not np.isfinite(gaps).all() or np.any(gaps <= 0.0)):
        raise ValueError("gaps must be finite and positive")
    absolute_tolerance_s = nominal_gap_s * relative_tolerance
    nominal = np.abs(gaps - nominal_gap_s) <= absolute_tolerance_s
    comparison_count = int(gaps.size)
    non_nominal_count = int(np.count_nonzero(~nominal))
    return {
        "persistence_comparison_count": comparison_count,
        "nominal_gap_s": float(nominal_gap_s),
        "relative_tolerance": float(relative_tolerance),
        "absolute_tolerance_s": float(absolute_tolerance_s),
        "gap_distribution": {
            "minimum_s": float(np.min(gaps)) if comparison_count else None,
            "median_s": float(np.median(gaps)) if comparison_count else None,
            "maximum_s": float(np.max(gaps)) if comparison_count else None,
            "histogram": _gap_histogram(gaps),
        },
        "non_nominal_gap_count": non_nominal_count,
        "non_nominal_gap_fraction": (
            float(non_nominal_count / comparison_count) if comparison_count else 0.0
        ),
        "all_gaps_nominal": bool(np.all(nominal)),
    }


def build_report(
    *,
    session_root: Path = DEFAULT_SESSION_ROOT,
    shot_ids: Sequence[int] = FROZEN_CARRIER_SHOTS,
) -> dict[str, Any]:
    """Score every requested carrier and assemble a publication receipt."""
    shots = [score_session(shot_id, session_root=session_root) for shot_id in shot_ids]
    aggregate = _aggregate(shots)
    if aggregate["all_gaps_nominal"]:
        verdict = (
            "SAME SPACING: every persistence comparison advances by 0.005 s "
            "within the registered tolerance, matching the model history cadence."
        )
    else:
        count = aggregate["non_nominal_gap_count"]
        total = aggregate["persistence_comparison_count"]
        fraction = aggregate["non_nominal_gap_fraction"]
        verdict = (
            "DIFFERENT SPACING: persistence crosses a non-nominal admitted-slice "
            f"gap on {count}/{total} comparisons ({fraction:.1%}), while model "
            "history assumes 0.005 s."
        )
    demo = next((shot for shot in shots if shot["shot_id"] == DEMO_CARRIER_SHOT), None)
    return {
        "schema": "admitted-slice-spacing",
        "generated_at": datetime.now(UTC).isoformat(),
        "source_revision": _source_revision(),
        "session_root": str(Path(session_root)),
        "time_source": "steering/time coordinate in each NetCDF session",
        "admission_rule": (
            "manifest rows with written=true and converged=true; session indices "
            "advance only for written rows"
        ),
        "comparison_definition": (
            "one persistence comparison per gap between consecutive admitted "
            "real frames"
        ),
        "model_history_gap_s": DEFAULT_HISTORY_SPACING_SECONDS,
        "non_nominal_definition": (
            "absolute gap error greater than 0.1 percent of 0.005 s"
        ),
        "shots": shots,
        "aggregate": aggregate,
        "demo_carrier": demo,
        "verdict": verdict,
    }


def write_figure(report: Mapping[str, Any], output: Path) -> None:
    """Draw per-carrier admitted gaps and non-nominal comparison shares."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: PLC0415
    from matplotlib.figure import Figure  # noqa: PLC0415

    shots = report["shots"]
    figure = Figure(figsize=(11.0, 7.2), constrained_layout=True)
    FigureCanvasAgg(figure)
    grid = figure.add_gridspec(2, 1, height_ratios=(3.0, 1.2))
    gap_axis = figure.add_subplot(grid[0])
    share_axis = figure.add_subplot(grid[1])

    labels: list[str] = []
    for position, shot in enumerate(shots):
        shot_id = int(shot["shot_id"])
        labels.append(str(shot_id))
        gaps_ms = np.asarray(shot["gaps_s"], dtype=np.float64) * 1000.0
        x = np.full(gaps_ms.shape, position, dtype=np.float64)
        if gaps_ms.size:
            x += np.linspace(-0.16, 0.16, gaps_ms.size)
        gap_axis.scatter(x, gaps_ms, s=23, alpha=0.78, edgecolors="none")
    gap_axis.axhline(
        DEFAULT_HISTORY_SPACING_SECONDS * 1000.0,
        color="black",
        linewidth=1.2,
        linestyle="--",
        label="model history: 5 ms",
    )
    gap_axis.set_xticks(range(len(labels)), labels)
    gap_axis.set_ylabel("Consecutive admitted gap (ms)")
    gap_axis.set_title("Actual gaps used by persistence comparisons")
    gap_axis.grid(axis="y", alpha=0.25)
    gap_axis.legend(loc="upper left")

    shares = [float(shot["non_nominal_gap_fraction"]) for shot in shots]
    share_axis.bar(range(len(labels)), np.asarray(shares) * 100.0, color="#bb4d3a")
    share_axis.set_xticks(range(len(labels)), labels)
    share_axis.set_xlabel("Carrier shot")
    share_axis.set_ylabel("Non-nominal (%)")
    share_axis.set_ylim(0.0, max(5.0, 105.0 * max(shares, default=0.0)))
    share_axis.grid(axis="y", alpha=0.25)
    figure.suptitle(textwrap.fill(str(report["verdict"]), width=105), fontsize=11)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)


def write_report(
    report: Mapping[str, Any], output_json: Path, output_png: Path
) -> None:
    """Write the receipt and its matching evidence figure."""
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_figure(report, output_png)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-root", type=Path, default=DEFAULT_SESSION_ROOT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-png", type=Path, default=DEFAULT_OUTPUT_PNG)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the admitted-slice cadence audit."""
    args = _parser().parse_args(argv)
    report = build_report(session_root=args.session_root)
    write_report(report, args.output_json, args.output_png)
    print(report["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
