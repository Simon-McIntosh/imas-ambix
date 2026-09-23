"""Measure decode throughput and prefix retention against KV occupancy.

The engine log supplies two complementary observations. ``Decode batch`` lines
carry aggregate generation throughput, running width, and full-token usage.
``Prefill batch`` lines carry cached and newly computed tokens. Chunked prefill
prints several lines for one admitted sequence, so those chunks are accumulated
until ``#pending-token`` reaches zero and become one token-weighted cache-hit
observation. The observation is assigned to the full-token usage on its first
chunk, when the prefix lookup happened.

The gate target is the upper edge of the highest occupancy bin whose median
aggregate rate remains at least 90 percent of the best qualifying bin and whose
median prefix-cache hit rate is at least 0.97. A synthetic log plants the first
degraded bin at occupancy 0.80; the reducer must report 0.80 as the safe upper
boundary before real logs are accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

BIN_WIDTH = 0.05
MIN_INTERVALS = 20
MIN_THROUGHPUT_FRACTION = 0.90
MIN_PREFIX_HIT_RATE = 0.97
COLLAPSE_RATE_TOK_S = 200.0
BOOTSTRAP_RESAMPLES = 2_000
BOOTSTRAP_SEED = 20260923
BOOTSTRAP_LEVEL = 0.95

_TIMESTAMP = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
_RUNNING = re.compile(r"#running-req:\s*(\d+)")
_FULL_USAGE = re.compile(r"full token usage:\s*(\d+(?:\.\d+)?)")
_GENERATION = re.compile(r"gen throughput \(token/s\):\s*(\d+(?:\.\d+)?)")
_NEW_TOKENS = re.compile(r"#new-token:\s*(\d+)")
_CACHED_TOKENS = re.compile(r"#cached-token:\s*(\d+)")
_PENDING_TOKENS = re.compile(r"#pending-token:\s*(\d+)")


@dataclass(frozen=True)
class DecodeSample:
    """One readable decode interval."""

    timestamp: str
    occupancy: float
    generation_rate: float
    running_width: int


@dataclass(frozen=True)
class PrefixSample:
    """One completed prefill sequence, reconstructed across chunks."""

    timestamp: str
    occupancy: float
    hit_rate: float
    running_width: int
    cached_tokens: int
    new_tokens: int


@dataclass
class PendingPrefill:
    """Chunks belonging to the currently admitted prefill sequence."""

    timestamp: str
    occupancy: float
    running_width: int
    cached_tokens: int = 0
    new_tokens: int = 0


def physical_lines(raw: bytes) -> list[str]:
    """Split on newlines without treating progress-bar carriage returns as rows."""
    text = raw.decode("utf-8", errors="replace")
    lines = [line.removesuffix("\r") for line in text.split("\n")]
    if lines and not lines[-1]:
        lines.pop()
    return lines


def _field(pattern: re.Pattern[str], line: str, name: str) -> str:
    match = pattern.search(line)
    if match is None:
        raise ValueError(f"missing {name}")
    return match.group(1)


def parse_log(
    raw: bytes,
) -> tuple[list[DecodeSample], list[PrefixSample], dict[str, int]]:
    """Read decode intervals and completed prefill sequences from one log."""
    decode: list[DecodeSample] = []
    prefix: list[PrefixSample] = []
    malformed_decode = 0
    malformed_prefill = 0
    incomplete_prefill = 0
    pending: PendingPrefill | None = None

    for line in physical_lines(raw):
        if "Decode batch" in line:
            try:
                decode.append(
                    DecodeSample(
                        timestamp=_field(_TIMESTAMP, line, "timestamp"),
                        occupancy=float(_field(_FULL_USAGE, line, "full token usage")),
                        generation_rate=float(
                            _field(_GENERATION, line, "generation throughput")
                        ),
                        running_width=int(_field(_RUNNING, line, "running width")),
                    )
                )
            except ValueError:
                malformed_decode += 1
            continue

        if "Prefill batch" not in line:
            continue
        try:
            timestamp = _field(_TIMESTAMP, line, "timestamp")
            occupancy = float(_field(_FULL_USAGE, line, "full token usage"))
            running = int(_field(_RUNNING, line, "running width"))
            new_tokens = int(_field(_NEW_TOKENS, line, "new tokens"))
            cached_tokens = int(_field(_CACHED_TOKENS, line, "cached tokens"))
            remaining = int(_field(_PENDING_TOKENS, line, "pending tokens"))
        except ValueError:
            malformed_prefill += 1
            continue

        if pending is None:
            pending = PendingPrefill(
                timestamp=timestamp,
                occupancy=occupancy,
                running_width=running,
            )
        pending.cached_tokens += cached_tokens
        pending.new_tokens += new_tokens
        if remaining:
            continue

        total = pending.cached_tokens + pending.new_tokens
        if total:
            prefix.append(
                PrefixSample(
                    timestamp=pending.timestamp,
                    occupancy=pending.occupancy,
                    hit_rate=pending.cached_tokens / total,
                    running_width=pending.running_width,
                    cached_tokens=pending.cached_tokens,
                    new_tokens=pending.new_tokens,
                )
            )
        pending = None

    if pending is not None:
        incomplete_prefill = 1
    return decode, prefix, {
        "malformed_decode_lines": malformed_decode,
        "malformed_prefill_lines": malformed_prefill,
        "incomplete_prefill_sequences": incomplete_prefill,
    }


def bin_lower(value: float) -> float:
    """Return the stable lower edge of a fixed-width occupancy bin."""
    bounded = min(max(value, 0.0), 1.0)
    if math.isclose(bounded, 1.0):
        return round(1.0 - BIN_WIDTH, 2)
    return round(math.floor((bounded + 1e-12) / BIN_WIDTH) * BIN_WIDTH, 2)


def median(values: list[float] | list[int]) -> float:
    """Return a float median for a non-empty sample."""
    return float(np.median(np.asarray(values, dtype=float)))


def bootstrap_median_ci(values: list[float], seed_offset: int) -> tuple[float, float]:
    """Deterministic percentile bootstrap interval for a sample median."""
    sample = np.asarray(values, dtype=float)
    if not sample.size:
        raise ValueError("bootstrap of an empty sample")
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    medians: list[np.ndarray] = []
    remaining = BOOTSTRAP_RESAMPLES
    chunk = max(1, min(250, 2_000_000 // sample.size))
    while remaining:
        count = min(chunk, remaining)
        indices = rng.integers(0, sample.size, size=(count, sample.size))
        medians.append(np.median(sample[indices], axis=1))
        remaining -= count
    draws = np.concatenate(medians)
    alpha = (1.0 - BOOTSTRAP_LEVEL) / 2.0
    low, high = np.quantile(draws, [alpha, 1.0 - alpha])
    return float(low), float(high)


def summarise_bins(
    decode: list[DecodeSample], prefix: list[PrefixSample]
) -> dict[str, dict[str, object]]:
    """Summarise each sufficiently populated occupancy bin."""
    decode_by_bin: dict[float, list[DecodeSample]] = defaultdict(list)
    prefix_by_bin: dict[float, list[PrefixSample]] = defaultdict(list)
    for sample in decode:
        decode_by_bin[bin_lower(sample.occupancy)].append(sample)
    for sample in prefix:
        prefix_by_bin[bin_lower(sample.occupancy)].append(sample)

    summary: dict[str, dict[str, object]] = {}
    for index, lower in enumerate(sorted(decode_by_bin)):
        intervals = decode_by_bin[lower]
        if len(intervals) < MIN_INTERVALS:
            continue
        rates = [sample.generation_rate for sample in intervals]
        widths = [sample.running_width for sample in intervals]
        hits = [sample.hit_rate for sample in prefix_by_bin.get(lower, [])]
        low, high = bootstrap_median_ci(rates, index)
        upper = round(lower + BIN_WIDTH, 2)
        label = f"{lower:.2f}-{upper:.2f}"
        collapsed = [rate for rate in rates if rate < COLLAPSE_RATE_TOK_S]
        summary[label] = {
            "occupancy_lower": lower,
            "occupancy_upper": upper,
            "decode_intervals": len(intervals),
            "median_aggregate_generation_tok_s": median(rates),
            "ci95_low": low,
            "ci95_high": high,
            "p10_aggregate_generation_tok_s": float(np.quantile(rates, 0.10)),
            "collapsed_intervals_below_200_tok_s": len(collapsed),
            "collapsed_interval_fraction": len(collapsed) / len(rates),
            "median_running_width": median(widths),
            "prefix_cache_observations": len(hits),
            "median_prefix_cache_hit_rate": median(hits) if hits else None,
            "first_decode": min(sample.timestamp for sample in intervals),
            "last_decode": max(sample.timestamp for sample in intervals),
        }
    return summary


def select_gate(bins: dict[str, dict[str, object]]) -> dict[str, object]:
    """Select the highest safe bin and report its upper edge as the target."""
    if not bins:
        raise ValueError("no occupancy bin reached the interval threshold")
    best = max(
        float(row["median_aggregate_generation_tok_s"]) for row in bins.values()
    )
    candidates = [
        (label, row)
        for label, row in bins.items()
        if float(row["median_aggregate_generation_tok_s"])
        >= MIN_THROUGHPUT_FRACTION * best
        and row["median_prefix_cache_hit_rate"] is not None
        and float(row["median_prefix_cache_hit_rate"]) >= MIN_PREFIX_HIT_RATE
    ]
    if not candidates:
        return {
            "best_median_aggregate_generation_tok_s": best,
            "qualifying_bin": None,
            "occupancy_target": None,
            "reason": "no bin passed both throughput and prefix-hit thresholds",
        }
    label, row = max(candidates, key=lambda item: float(item[1]["occupancy_lower"]))
    return {
        "best_median_aggregate_generation_tok_s": best,
        "minimum_acceptable_generation_tok_s": MIN_THROUGHPUT_FRACTION * best,
        "qualifying_bin": label,
        "qualifying_bin_median_generation_tok_s": row[
            "median_aggregate_generation_tok_s"
        ],
        "qualifying_bin_median_prefix_cache_hit_rate": row[
            "median_prefix_cache_hit_rate"
        ],
        "occupancy_target": row["occupancy_upper"],
    }


def analyse(
    path: str, label: str
) -> tuple[dict[str, object], list[DecodeSample], list[PrefixSample]]:
    """Analyse one exact log snapshot and anchor it by digest."""
    raw = Path(path).read_bytes()
    decode, prefix, parse_counts = parse_log(raw)
    bins = summarise_bins(decode, prefix)
    report: dict[str, object] = {
        "label": label,
        "source": path,
        "source_bytes": len(raw),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "decode_intervals_total": len(decode),
        "prefix_cache_observations_total": len(prefix),
        **parse_counts,
        "bin_width": BIN_WIDTH,
        "minimum_decode_intervals_per_bin": MIN_INTERVALS,
        "bins": bins,
        "gate": select_gate(bins),
    }
    return report, decode, prefix


def synthetic_log(path: Path) -> None:
    """Write a control whose first degraded occupancy bin begins at 0.80."""
    lines: list[str] = []
    for lower in (0.60, 0.65, 0.70, 0.75, 0.80, 0.85):
        degraded = lower >= 0.80
        generation = 340.0 if degraded else 500.0
        cached = 9000 if degraded else 9900
        new = 1000 if degraded else 100
        width = int(round(lower * 40))
        for interval in range(30):
            second = interval % 60
            lines.append(
                f"[2026-09-23 12:{int(lower * 10):02d}:{second:02d} TP0 EP0] "
                f"Decode batch, #running-req: {width}, #full token: 1, "
                f"full token usage: {lower:.2f}, #swa token: 1, "
                f"swa token usage: 0.00, accept len: 3.5, accept rate: 0.5, "
                f"cuda graph: True, gen throughput (token/s): {generation:.2f}, "
                "#queue-req: 0"
            )
            lines.append(
                f"[2026-09-23 12:{int(lower * 10):02d}:{second:02d} TP0 EP0] "
                f"Prefill batch, #new-seq: 1, #new-token: {new}, "
                f"#cached-token: {cached}, full token usage: {lower:.2f}, "
                f"swa token usage: 0.00, #running-req: {width}, #queue-req: 0, "
                "#pending-token: 0, cuda graph: False, "
                "input throughput (token/s): 1000.0"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_control(directory: Path) -> dict[str, object]:
    """Run the complete reducer against the planted-boundary control."""
    path = directory / "synthetic_cliff.log"
    synthetic_log(path)
    report, _, _ = analyse(str(path), "synthetic")
    detected = report["gate"]["occupancy_target"]
    return {
        "source": str(path),
        "planted_cliff_occupancy": 0.80,
        "detected_safe_upper_boundary": detected,
        "matched": math.isclose(float(detected), 0.80),
        "gate": report["gate"],
    }


def plot(job_reports: list[dict[str, object]], path: str) -> None:
    """Plot throughput intervals and prefix-hit medians by occupancy and job."""
    figure, axes = plt.subplots(
        2,
        len(job_reports),
        figsize=(6.4 * len(job_reports), 7.6),
        squeeze=False,
        sharex="col",
    )
    for column, report in enumerate(job_reports):
        rows = sorted(
            report["bins"].values(), key=lambda row: float(row["occupancy_lower"])
        )
        x = [0.5 * (row["occupancy_lower"] + row["occupancy_upper"]) for row in rows]
        y = [row["median_aggregate_generation_tok_s"] for row in rows]
        low = [row["ci95_low"] for row in rows]
        high = [row["ci95_high"] for row in rows]
        lower_error = [value - bound for value, bound in zip(y, low, strict=True)]
        upper_error = [bound - value for value, bound in zip(y, high, strict=True)]
        throughput_axis = axes[0][column]
        throughput_axis.errorbar(
            x,
            y,
            yerr=[lower_error, upper_error],
            marker="o",
            markersize=4,
            capsize=2,
            linewidth=1.4,
            color="#1f5c8a",
            ecolor="#9bb8cd",
        )
        throughput_axis.axhline(
            MIN_THROUGHPUT_FRACTION
            * report["gate"]["best_median_aggregate_generation_tok_s"],
            color="#8a1c1c",
            linestyle="--",
            linewidth=1.0,
            label="90% of best bin",
        )
        target = report["gate"]["occupancy_target"]
        if target is not None:
            throughput_axis.axvline(
                target,
                color="#704214",
                linestyle=":",
                linewidth=1.4,
                label=f"target {target:.2f}",
            )
        throughput_axis.set_title(f"job {report['label']}")
        throughput_axis.set_ylabel("aggregate generation tok/s")
        throughput_axis.grid(True, alpha=0.25)
        throughput_axis.legend(fontsize=8)

        hit_axis = axes[1][column]
        hit_x = [
            midpoint
            for midpoint, row in zip(x, rows, strict=True)
            if row["median_prefix_cache_hit_rate"] is not None
        ]
        hit_y = [
            row["median_prefix_cache_hit_rate"]
            for row in rows
            if row["median_prefix_cache_hit_rate"] is not None
        ]
        hit_axis.plot(hit_x, hit_y, marker="o", linewidth=1.4, color="#1b7f5a")
        hit_axis.axhline(
            MIN_PREFIX_HIT_RATE,
            color="#8a1c1c",
            linestyle="--",
            linewidth=1.0,
            label="minimum 0.97",
        )
        if target is not None:
            hit_axis.axvline(
                target,
                color="#704214",
                linestyle=":",
                linewidth=1.4,
                label=f"target {target:.2f}",
            )
        hit_axis.set_xlabel("KV full-token pool occupancy")
        hit_axis.set_ylabel("median prefix-cache hit rate")
        hit_axis.set_ylim(0.0, 1.02)
        hit_axis.grid(True, alpha=0.25)
        hit_axis.legend(fontsize=8)
    figure.suptitle("DSv4.1 lane throughput and prefix retention against KV occupancy")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", action="append", default=[], help="LABEL=LOG_PATH")
    parser.add_argument("--json", required=True, help="output measurement document")
    parser.add_argument("--png", required=True, help="output figure")
    parser.add_argument("--control", required=True, help="output control document")
    args = parser.parse_args()
    if not args.job:
        parser.error("at least one --job LABEL=LOG_PATH is required")

    output_directory = Path(args.json).parent
    control = run_control(output_directory)
    Path(args.control).write_text(
        json.dumps(control, indent=2) + "\n", encoding="utf-8"
    )
    if not control["matched"]:
        print(
            "CONTROL FAILED: planted cliff 0.80, detected "
            f"{control['detected_safe_upper_boundary']}"
        )
        return 1

    reports: list[dict[str, object]] = []
    for specification in args.job:
        label, separator, log_path = specification.partition("=")
        if not separator:
            parser.error(f"invalid --job value: {specification}")
        report, _, _ = analyse(log_path, label)
        reports.append(report)

    supported_targets = [
        float(report["gate"]["occupancy_target"])
        for report in reports
        if report["gate"]["occupancy_target"] is not None
    ]
    selected_target = min(supported_targets) if supported_targets else None
    document = {
        "bin_width": BIN_WIDTH,
        "minimum_decode_intervals_per_bin": MIN_INTERVALS,
        "bootstrap": {
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "level": BOOTSTRAP_LEVEL,
        },
        "gate_rule": {
            "minimum_throughput_fraction_of_best": MIN_THROUGHPUT_FRACTION,
            "minimum_median_prefix_cache_hit_rate": MIN_PREFIX_HIT_RATE,
            "selection": (
                "highest qualifying bin per job; conservative minimum upper "
                "boundary across jobs"
            ),
        },
        "jobs": {report["label"]: report for report in reports},
        "selected_occupancy_target": selected_target,
        "implied_gate_width_at_100000_mean_context": (
            math.floor(4_000_000 * selected_target / 100_000)
            if selected_target is not None
            else None
        ),
        "pool_tokens": 4_000_000,
        "mean_context_tokens_for_projection": 100_000,
        "control": control,
    }
    Path(args.json).write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    plot(reports, args.png)

    for report in reports:
        gate = report["gate"]
        print(
            f"job {report['label']}: decode={report['decode_intervals_total']} "
            f"prefix={report['prefix_cache_observations_total']} "
            f"bins={len(report['bins'])} qualifying={gate['qualifying_bin']} "
            f"target={gate['occupancy_target']}"
        )
    print(
        f"control: planted=0.80 detected={control['detected_safe_upper_boundary']} "
        f"matched={control['matched']}"
    )
    print(
        f"selected target={selected_target} implied_width="
        f"{document['implied_gate_width_at_100000_mean_context']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
