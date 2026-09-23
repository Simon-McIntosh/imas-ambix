"""Measure aggregate decode throughput against running-request width on the local lane.

The engine's scheduler logs one ``Decode batch`` line per step carrying the
running request width and that step's aggregate generation throughput. This
script groups those intervals by exact width, takes the per-width median, and
puts a percentile bootstrap interval around it, so the throughput knee -- the
width past which adding a stream buys no aggregate tokens -- is read from the
running serve rather than from a cost model.

Parsing is delegated to :mod:`imas_ambix.agent.decode_bins`; this module adds
grouping, the bootstrap interval, the peak and operating-band reductions, and
the figure. Each width also records its own time span and median speculative
accept length, because a width bin is not a controlled condition: a throughput
difference between two widths can be a different workload happening at a
different hour rather than an effect of width.

A synthetic log with a planted peak is run through the same reduction and
asserted, so a pipeline that would report a peak regardless of its input is
caught rather than trusted.

Usage::

    python knee_measure.py --job 1273253=<log> --job 1276246=<log> \
        --json <knee.json> --png <knee.png> --control <control.json>
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from imas_ambix.agent.decode_bins import (  # noqa: E402
    DecodeInterval,
    parse_decode_lines,
    physical_lines,
)

MIN_INTERVALS = 20
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260923
BOOTSTRAP_LEVEL = 0.95
OPERATING_BAND = (29, 31)


def width_intervals(
    intervals: tuple[DecodeInterval, ...],
) -> dict[int, list[DecodeInterval]]:
    """Group decode intervals by exact running width."""
    by_width: dict[int, list[DecodeInterval]] = defaultdict(list)
    for interval in intervals:
        by_width[interval.width].append(interval)
    return dict(by_width)


def median(values: list[float]) -> float:
    """Median of ``values``."""
    ordered = sorted(values)
    count = len(ordered)
    midpoint = count // 2
    if count % 2:
        return ordered[midpoint]
    return 0.5 * (ordered[midpoint - 1] + ordered[midpoint])


def percentile(ordered: list[float], fraction: float) -> float:
    """Linear-interpolated percentile of an already-sorted, non-empty sequence."""
    if not ordered:
        raise ValueError("percentile of an empty sequence")
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_median_ci(
    values: list[float],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    level: float = BOOTSTRAP_LEVEL,
    chunk: int = 300,
) -> tuple[float, float]:
    """Percentile bootstrap interval for the median of ``values``.

    Resampling is deterministic from the seed; it is drawn in fixed-size chunks
    so the widest bins do not need one dense draw matrix.
    """
    if not values:
        raise ValueError("bootstrap of an empty sample")
    sample = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws: list[np.ndarray] = []
    done = 0
    while done < resamples:
        size = min(chunk, resamples - done)
        index = rng.integers(0, sample.size, size=(size, sample.size))
        draws.append(np.median(sample[index], axis=1))
        done += size
    medians = np.sort(np.concatenate(draws)).tolist()
    alpha = (1.0 - level) / 2.0
    return percentile(medians, alpha), percentile(medians, 1.0 - alpha)


def summarise_widths(
    by_width: dict[int, list[DecodeInterval]],
) -> dict[int, dict[str, object]]:
    """Per-width median, bootstrap interval, accept length and time span."""
    summary: dict[int, dict[str, object]] = {}
    for width in sorted(by_width):
        group = by_width[width]
        if len(group) < MIN_INTERVALS:
            continue
        rates = [interval.generation_rate for interval in group]
        accepts = [interval.accept_length for interval in group]
        stamps = [interval.timestamp for interval in group]
        low, high = bootstrap_median_ci(rates)
        summary[width] = {
            "intervals": len(group),
            "median_generation_tok_s": median(rates),
            "ci95_low": low,
            "ci95_high": high,
            "accept_length_median": median(accepts),
            "first_seen": min(stamps),
            "last_seen": max(stamps),
        }
    return summary


def peak_width(summary: dict[int, dict[str, object]]) -> tuple[int, float]:
    """The width with the highest median throughput among the qualifying bins."""
    if not summary:
        raise ValueError("no width reached the interval threshold")
    best = max(summary, key=lambda width: summary[width]["median_generation_tok_s"])
    return best, summary[best]["median_generation_tok_s"]


def operating_band_median(
    by_width: dict[int, list[DecodeInterval]],
) -> tuple[float, int]:
    """Median throughput across the observed operating widths, with its sample size."""
    low, high = OPERATING_BAND
    pooled = [
        interval.generation_rate
        for width, group in by_width.items()
        if low <= width <= high
        for interval in group
    ]
    if not pooled:
        raise ValueError(f"no intervals in the operating band {OPERATING_BAND}")
    return median(pooled), len(pooled)


def plateau_median(
    summary: dict[int, dict[str, object]], low: int, high: int
) -> tuple[float, int]:
    """Median of the per-width medians across a width range, with the range's size."""
    widths = [width for width in summary if low <= width <= high]
    if not widths:
        raise ValueError(f"no qualifying widths in [{low}, {high}]")
    medians = [summary[width]["median_generation_tok_s"] for width in widths]
    return median(medians), len(widths)


def analyse_log(path: str, label: str) -> dict[str, object]:
    """Summarise one serve log: per-width medians, the peak, and the band ratio."""
    raw = Path(path).read_bytes()
    text = raw.decode("utf-8", errors="replace")
    lines = physical_lines(text)
    parsed = parse_decode_lines(lines)
    by_width = width_intervals(parsed.intervals)
    summary = summarise_widths(by_width)
    peak, peak_median = peak_width(summary)
    band_median, band_intervals = operating_band_median(by_width)
    plateau, plateau_widths = plateau_median(summary, 17, 32)
    timestamps = [interval.timestamp for interval in parsed.intervals]
    return {
        "label": label,
        "source": path,
        "source_bytes": len(raw),
        "source_sha256": _digest(raw),
        "lines_read": len(lines),
        "intervals_total": len(parsed.intervals),
        "malformed_decode_lines": len(parsed.malformed),
        "span": {
            "first": min(timestamps) if timestamps else None,
            "last": max(timestamps) if timestamps else None,
        },
        "widths": {
            str(width): summary[width] for width in sorted(summary)
        },
        "peak": {"width": peak, "median_generation_tok_s": peak_median},
        "operating_band": {
            "widths": list(range(OPERATING_BAND[0], OPERATING_BAND[1] + 1)),
            "intervals": band_intervals,
            "median_generation_tok_s": band_median,
        },
        "plateau_17_32": {
            "widths": plateau_widths,
            "median_of_width_medians": plateau,
        },
        "ratio_operating_to_peak": band_median / peak_median,
        "ratio_operating_to_plateau": band_median / plateau,
    }


def _digest(raw: bytes) -> str:
    """SHA-256 of the exact bytes read, so a live-log reading is verifiable."""
    import hashlib

    return hashlib.sha256(raw).hexdigest()


def synthetic_log(path: str, *, peak_at: int, widths: range, per_width: int) -> str:
    """Write a log whose planted peak is at ``peak_at``, for the pipeline control."""
    lines: list[str] = []
    for width in widths:
        throughput = 900.0 if width == peak_at else 120.0
        for second in range(per_width):
            lines.append(
                f"[2026-09-17 12:{width % 60:02d}:{second % 60:02d} TP0 EP0] "
                f"Decode batch, #running-req: {width}, #full token: 7936, "
                f"full token usage: 0.00, #swa token: 512, swa token usage: 0.00, "
                f"accept len: 3.50, accept rate: 0.50, cuda graph: True, "
                f"gen throughput (token/s): {throughput}, #queue-req: 0"
            )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_control(workdir: Path) -> dict[str, object]:
    """Prove the peak reduction follows its input: plant peaks and read them back."""
    results: dict[str, object] = {}
    for peak_at in (12, 25):
        log_path = workdir / f"synthetic_peak_{peak_at}.log"
        synthetic_log(
            str(log_path), peak_at=peak_at, widths=range(1, 31), per_width=MIN_INTERVALS
        )
        report = analyse_log(str(log_path), f"synthetic-peak-{peak_at}")
        detected = report["peak"]["width"]
        results[str(peak_at)] = {
            "planted_peak_width": peak_at,
            "detected_peak_width": detected,
            "matched": detected == peak_at,
        }
    return results


def plot_jobs(job_reports: list[dict[str, object]], path: str) -> None:
    """Median throughput and its interval against width, one panel per job."""
    figure, axes = plt.subplots(
        1, len(job_reports), figsize=(6.6 * len(job_reports), 4.8), squeeze=False
    )
    for axis, report in zip(axes[0], job_reports, strict=True):
        widths = sorted(int(width) for width in report["widths"])
        medians = [report["widths"][str(w)]["median_generation_tok_s"] for w in widths]
        lows = [report["widths"][str(w)]["ci95_low"] for w in widths]
        highs = [report["widths"][str(w)]["ci95_high"] for w in widths]
        lower = [value - low for value, low in zip(medians, lows, strict=True)]
        upper = [high - value for value, high in zip(medians, highs, strict=True)]
        axis.errorbar(
            widths,
            medians,
            yerr=[lower, upper],
            marker="o",
            markersize=3,
            linewidth=1.2,
            capsize=2,
            color="#1f5c8a",
            ecolor="#9bb8cd",
        )
        axis.axvline(
            report["peak"]["width"],
            color="#8a1c1c",
            linewidth=1.25,
            linestyle="--",
            label=f"peak width {report['peak']['width']}",
        )
        axis.axvspan(
            OPERATING_BAND[0],
            OPERATING_BAND[1],
            color="#e8d9a0",
            alpha=0.35,
            label=f"operating band {OPERATING_BAND[0]}-{OPERATING_BAND[1]}",
        )
        axis.set_title(f"job {report['label']}")
        axis.set_xlabel("running requests (width)")
        axis.set_ylabel("aggregate generation tok/s (median)")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    figure.suptitle(
        "DSv4.1 lane: aggregate decode throughput against running-request width"
    )
    figure.tight_layout()
    figure.savefig(Path(path), dpi=140)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", action="append", default=[], help="LABEL=LOG_PATH")
    parser.add_argument("--json", required=True, help="output measurement document")
    parser.add_argument("--png", required=True, help="output figure")
    parser.add_argument("--control", required=True, help="output pipeline-control doc")
    args = parser.parse_args()

    job_reports: list[dict[str, object]] = []
    for spec in args.job:
        label, _, log_path = spec.partition("=")
        job_reports.append(analyse_log(log_path, label))
    if not job_reports:
        parser.error("at least one --job LABEL=LOG_PATH is required")

    control = run_control(Path(args.json).parent)
    mismatches = [name for name, result in control.items() if not result["matched"]]

    document = {
        "min_intervals": MIN_INTERVALS,
        "bootstrap": {
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "level": BOOTSTRAP_LEVEL,
        },
        "operating_band": list(OPERATING_BAND),
        "jobs": {report["label"]: report for report in job_reports},
        "control": control,
        "control_matched": not mismatches,
    }
    Path(args.json).write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    plot_jobs(job_reports, args.png)
    Path(args.control).write_text(
        json.dumps(
            {"min_intervals": MIN_INTERVALS, "control": control}, indent=2
        )
        + "\n",
        encoding="utf-8",
    )

    for report in job_reports:
        print(
            f"{report['label']}: lines={report['lines_read']} "
            f"intervals={report['intervals_total']} "
            f"malformed={report['malformed_decode_lines']} "
            f"widths={len(report['widths'])} "
            f"peak={report['peak']['width']}@"
            f"{report['peak']['median_generation_tok_s']:.1f} "
            f"band={report['operating_band']['median_generation_tok_s']:.1f} "
            f"plateau={report['plateau_17_32']['median_of_width_medians']:.1f} "
            f"ratio_peak={report['ratio_operating_to_peak']:.3f} "
            f"ratio_plateau={report['ratio_operating_to_plateau']:.3f}"
        )
    print(f"control matched: {not mismatches}")
    if mismatches:
        print(f"CONTROL FAILED: {mismatches}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
