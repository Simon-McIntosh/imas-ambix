"""Name what separates collapsed decode intervals from healthy ones.

The engine's median decode throughput holds through high KV occupancy, but a
minority of intervals collapses. This reducer isolates those intervals from the
decode log and compares them, at matched occupancy bin and running width,
against intervals that held throughput. For every interval it gathers the
prefill load in a surrounding time window (new tokens and completed prefill
batches), the token-weighted prefix-cache hit rate in that window, the count of
HiCache load/backup lines, the count of retraction or eviction lines, and the
running width. It then ranks the features by how well their distributions
separate the two groups, using Cliff's delta as the observation and a
label-permutation p-value.

A control log plants a known separation and known retraction and HiCache lines,
so the reducer must both detect the planted feature and count the planted lines
before either absence in the real logs is believed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

BIN_WIDTH = 0.05
COLLAPSE_OCCUPANCY = 0.80
COLLAPSE_RATE_TOK_S = 200.0
WINDOW_SECONDS = 60
PERMUTATION_RESAMPLES = 20_000
PERMUTATION_SEED = 20260923
ALPHA = 0.05
MINIMUM_EFFECT = 0.33

_TIMESTAMP = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
_RUNNING = re.compile(r"#running-req:\s*(\d+)")
_FULL_USAGE = re.compile(r"full token usage:\s*(\d+(?:\.\d+)?)")
_GENERATION = re.compile(r"gen throughput \(token/s\):\s*(\d+(?:\.\d+)?)")
_NEW_TOKENS = re.compile(r"#new-token:\s*(\d+)")
_CACHED_TOKENS = re.compile(r"#cached-token:\s*(\d+)")
_PENDING_TOKENS = re.compile(r"#pending-token:\s*(\d+)")
_ACCEPT_RATE = re.compile(r"accept rate:\s*(\d+(?:\.\d+)?)")
_QUEUE_REQ = re.compile(r"#queue-req:\s*(\d+)")
_SWA_USAGE = re.compile(r"swa token usage:\s*(\d+(?:\.\d+)?)")

_HICACHE_LOAD = re.compile(r"hicache[^\n]*\b(load|prefetch|hit)", re.I)
_HICACHE_BACKUP = re.compile(r"hicache[^\n]*\b(backup|write)", re.I)
_RETRACT = re.compile(r"\bretract(?:ion|ed|s)?\b", re.I)
_EVICT = re.compile(r"\bevict(?:ion|ed|s)?\b", re.I)

EVENT_KEYS = ("hicache_load", "hicache_backup", "retract_evict")

FEATURES = (
    "prefill_new_tokens_window",
    "prefill_batches_window",
    "prefix_hit_rate_window",
    "hicache_load_window",
    "hicache_backup_window",
    "retract_evict_window",
    "running_width",
    "accept_rate",
)


@dataclass(frozen=True)
class DecodeSample:
    """One readable decode interval."""

    ts: float
    timestamp: str
    occupancy: float
    generation_rate: float
    running_width: int
    accept_rate: float
    queue_req: int
    swa_usage: float


@dataclass(frozen=True)
class PrefixSample:
    """One completed prefill sequence, reconstructed across chunks."""

    ts: float
    timestamp: str
    occupancy: float
    running_width: int
    cached_tokens: int
    new_tokens: int


@dataclass
class PendingPrefill:
    ts: float
    timestamp: str
    occupancy: float
    running_width: int
    cached_tokens: int = 0
    new_tokens: int = 0


def _epoch(timestamp: str) -> float:
    return datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S").timestamp()


def physical_lines(raw: bytes) -> list[str]:
    """Split on newlines without treating progress-bar carriage returns as rows."""
    text = raw.decode("utf-8", errors="replace")
    lines = [line.removesuffix("\r") for line in text.split("\n")]
    if lines and not lines[-1]:
        lines.pop()
    return lines


def _float(pattern: re.Pattern[str], line: str, default: float = 0.0) -> float:
    match = pattern.search(line)
    return float(match.group(1)) if match else default


def classify_event(line: str) -> str | None:
    """Return the cache-event key a runtime log line belongs to, if any."""
    if _RETRACT.search(line) or _EVICT.search(line):
        return "retract_evict"
    if _HICACHE_BACKUP.search(line):
        return "hicache_backup"
    if _HICACHE_LOAD.search(line):
        return "hicache_load"
    return None


def parse_log(
    raw: bytes,
) -> tuple[list[DecodeSample], list[PrefixSample], dict[str, object]]:
    """Read decode intervals, completed prefixes and cache-event timestamps."""
    decode: list[DecodeSample] = []
    prefix: list[PrefixSample] = []
    malformed_decode = 0
    malformed_prefill = 0
    pending: PendingPrefill | None = None
    events: dict[str, list[float]] = {key: [] for key in EVENT_KEYS}

    for line in physical_lines(raw):
        if "Decode batch" in line:
            ts_match = _TIMESTAMP.search(line)
            if ts_match is None:
                malformed_decode += 1
                continue
            timestamp = ts_match.group(1)
            decode.append(
                DecodeSample(
                    ts=_epoch(timestamp),
                    timestamp=timestamp,
                    occupancy=_float(_FULL_USAGE, line),
                    generation_rate=_float(_GENERATION, line),
                    running_width=int(_float(_RUNNING, line)),
                    accept_rate=_float(_ACCEPT_RATE, line),
                    queue_req=int(_float(_QUEUE_REQ, line)),
                    swa_usage=_float(_SWA_USAGE, line),
                )
            )
            continue

        if "Prefill batch" in line:
            ts_match = _TIMESTAMP.search(line)
            if ts_match is None:
                malformed_prefill += 1
                continue
            timestamp = ts_match.group(1)
            ts = _epoch(timestamp)
            new_tokens = int(_float(_NEW_TOKENS, line))
            cached_tokens = int(_float(_CACHED_TOKENS, line))
            remaining = int(_float(_PENDING_TOKENS, line))
            if pending is None:
                pending = PendingPrefill(
                    ts=ts,
                    timestamp=timestamp,
                    occupancy=_float(_FULL_USAGE, line),
                    running_width=int(_float(_RUNNING, line)),
                )
            pending.cached_tokens += cached_tokens
            pending.new_tokens += new_tokens
            if remaining:
                continue
            if pending.cached_tokens + pending.new_tokens:
                prefix.append(
                    PrefixSample(
                        ts=pending.ts,
                        timestamp=pending.timestamp,
                        occupancy=pending.occupancy,
                        running_width=pending.running_width,
                        cached_tokens=pending.cached_tokens,
                        new_tokens=pending.new_tokens,
                    )
                )
            pending = None
            continue

        # Cache-management events are counted wherever they appear. The config
        # banner and the init banner are excluded so only runtime cache activity
        # is counted, never the startup strings.
        if "server_args" in line or "Tree cache initialized" in line:
            continue
        ts_match = _TIMESTAMP.search(line)
        if ts_match is None:
            continue
        key = classify_event(line)
        if key is not None:
            events[key].append(_epoch(ts_match.group(1)))

    counts = {
        "malformed_decode_lines": malformed_decode,
        "malformed_prefill_lines": malformed_prefill,
        "incomplete_prefill_sequences": int(pending is not None),
        "runtime_hicache_load_lines": len(events["hicache_load"]),
        "runtime_hicache_backup_lines": len(events["hicache_backup"]),
        "runtime_retract_evict_lines": len(events["retract_evict"]),
    }
    return decode, prefix, {"counts": counts, "events": events}


def bin_lower(value: float) -> float:
    """Return the stable lower edge of a fixed-width occupancy bin."""
    bounded = min(max(value, 0.0), 1.0)
    if math.isclose(bounded, 1.0):
        return round(1.0 - BIN_WIDTH, 2)
    return round(math.floor((bounded + 1e-12) / BIN_WIDTH) * BIN_WIDTH, 2)


def window_features(
    decode: DecodeSample,
    prefill_sorted: list[PrefixSample],
    prefill_ts: list[float],
    events: dict[str, list[float]],
    window: int,
) -> dict[str, float]:
    """Aggregate the surrounding-window features for one decode interval."""
    low = bisect_left(prefill_ts, decode.ts - window)
    high = bisect_right(prefill_ts, decode.ts + window)
    in_window = prefill_sorted[low:high]
    new_tokens = sum(sample.new_tokens for sample in in_window)
    cached_tokens = sum(sample.cached_tokens for sample in in_window)
    total = new_tokens + cached_tokens

    def event_count(key: str) -> int:
        stamps = events[key]
        return bisect_right(stamps, decode.ts + window) - bisect_left(
            stamps, decode.ts - window
        )

    return {
        "prefill_new_tokens_window": float(new_tokens),
        "prefill_batches_window": float(len(in_window)),
        "prefix_hit_rate_window": (cached_tokens / total) if total else float("nan"),
        "hicache_load_window": float(event_count("hicache_load")),
        "hicache_backup_window": float(event_count("hicache_backup")),
        "retract_evict_window": float(event_count("retract_evict")),
        "running_width": float(decode.running_width),
        "accept_rate": decode.accept_rate,
    }


def cliff_delta(a: list[float], b: list[float]) -> float:
    """Cliff's delta: P(a>b) - P(a<b) over all pairs, in [-1, 1]."""
    return _cliff_delta_array(*_finite_pair(a, b))


def _finite_pair(
    a: list[float], b: list[float]
) -> tuple[np.ndarray, np.ndarray]:
    finite_a = np.asarray([v for v in a if not math.isnan(v)], dtype=float)
    finite_b = np.asarray([v for v in b if not math.isnan(v)], dtype=float)
    return finite_a, finite_b


def _cliff_delta_array(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    difference = a[:, None] - b[None, :]
    wins = float(np.count_nonzero(difference > 0))
    losses = float(np.count_nonzero(difference < 0))
    return (wins - losses) / (a.size * b.size)


def permutation_p(
    a: list[float], b: list[float], observed: float, rng: np.random.Generator
) -> float:
    """Two-sided label-permutation p-value for the Cliff's-delta statistic."""
    arr_a, arr_b = _finite_pair(a, b)
    if arr_a.size == 0 or arr_b.size == 0 or math.isnan(observed):
        return float("nan")
    pooled = np.concatenate([arr_a, arr_b])
    n_a = arr_a.size
    n_b = arr_b.size
    n_total = pooled.size
    # Chunk the permutations so the pairwise comparison stays small; each chunk
    # draws its own independent permutations and is tallied immediately.
    extreme = 0
    chunk = max(1, min(500, 4_000_000 // max(1, n_a * n_b)))
    remaining = PERMUTATION_RESAMPLES
    while remaining:
        count = min(chunk, remaining)
        order = rng.random((count, n_total)).argsort(axis=1)
        shuffled = pooled[order]
        differences = shuffled[:, :n_a, None] - shuffled[:, None, n_a:]
        wins = np.count_nonzero(differences > 0, axis=(1, 2))
        losses = np.count_nonzero(differences < 0, axis=(1, 2))
        deltas = (wins - losses) / (n_a * n_b)
        extreme += int(np.count_nonzero(np.abs(deltas) >= abs(observed) - 1e-12))
        remaining -= count
    return (extreme + 1) / (PERMUTATION_RESAMPLES + 1)


def median(values: list[float]) -> float | None:
    finite = [v for v in values if not math.isnan(v)]
    return float(np.median(finite)) if finite else None


def analyse(path: str, label: str) -> tuple[dict[str, object], dict[str, list[float]]]:
    """Classify collapse intervals and compare them to matched healthy ones."""
    raw = Path(path).read_bytes()
    decode, prefill, parsed = parse_log(raw)
    counts = parsed["counts"]
    events = parsed["events"]

    prefill_sorted = sorted(prefill, key=lambda sample: sample.ts)
    prefill_ts = [sample.ts for sample in prefill_sorted]

    high = [sample for sample in decode if sample.occupancy >= COLLAPSE_OCCUPANCY]
    collapsed = [s for s in high if s.generation_rate < COLLAPSE_RATE_TOK_S]
    healthy = [s for s in high if s.generation_rate >= COLLAPSE_RATE_TOK_S]

    healthy_by_key: dict[tuple[float, int], list[DecodeSample]] = defaultdict(list)
    for sample in healthy:
        healthy_by_key[(bin_lower(sample.occupancy), sample.running_width)].append(
            sample
        )

    matched_pool: dict[str, DecodeSample] = {}
    unmatched: list[dict[str, object]] = []
    for sample in collapsed:
        key = (bin_lower(sample.occupancy), sample.running_width)
        controls = healthy_by_key.get(key, [])
        if not controls:
            unmatched.append(
                {
                    "timestamp": sample.timestamp,
                    "occupancy": sample.occupancy,
                    "running_width": sample.running_width,
                    "generation_rate": sample.generation_rate,
                }
            )
            continue
        for control in controls:
            matched_pool.setdefault(control.timestamp, control)

    collapsed_features: dict[str, list[float]] = defaultdict(list)
    healthy_features: dict[str, list[float]] = defaultdict(list)
    collapsed_rows: list[dict[str, object]] = []
    healthy_rows: list[dict[str, object]] = []

    for sample in collapsed:
        features = window_features(
            sample, prefill_sorted, prefill_ts, events, WINDOW_SECONDS
        )
        collapsed_rows.append(
            {
                "timestamp": sample.timestamp,
                "occupancy": sample.occupancy,
                "generation_rate": sample.generation_rate,
                "running_width": sample.running_width,
                **features,
            }
        )
        for name, value in features.items():
            collapsed_features[name].append(value)

    for control in matched_pool.values():
        features = window_features(
            control, prefill_sorted, prefill_ts, events, WINDOW_SECONDS
        )
        healthy_rows.append(
            {
                "timestamp": control.timestamp,
                "occupancy": control.occupancy,
                "generation_rate": control.generation_rate,
                "running_width": control.running_width,
                **features,
            }
        )
        for name, value in features.items():
            healthy_features[name].append(value)

    rng = np.random.default_rng(PERMUTATION_SEED)
    separations = []
    for name in FEATURES:
        a = collapsed_features.get(name, [])
        b = healthy_features.get(name, [])
        delta = cliff_delta(a, b)
        p = permutation_p(a, b, delta, rng)
        separations.append(
            {
                "feature": name,
                "cliff_delta": delta,
                "abs_cliff_delta": (
                    abs(delta) if not math.isnan(delta) else float("nan")
                ),
                "permutation_p": p,
                "collapsed_median": median(a),
                "healthy_median": median(b),
                "collapsed_n": len(a),
                "healthy_n": len(b),
                "significant": bool(
                    not math.isnan(p) and p < ALPHA and abs(delta) >= MINIMUM_EFFECT
                ),
            }
        )
    ranked = sorted(
        separations,
        key=lambda row: (
            -1.0 if math.isnan(row["abs_cliff_delta"]) else -row["abs_cliff_delta"]
        ),
    )
    best = ranked[0] if ranked else None
    separating = [row for row in ranked if row["significant"]]

    report: dict[str, object] = {
        "label": label,
        "source": path,
        "source_bytes": len(raw),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "decode_intervals_total": len(decode),
        "prefix_cache_observations_total": len(prefill),
        **counts,
        "parameters": {
            "collapse_occupancy_at_or_above": COLLAPSE_OCCUPANCY,
            "collapse_rate_below_tok_s": COLLAPSE_RATE_TOK_S,
            "window_seconds": WINDOW_SECONDS,
            "occupancy_bin_width": BIN_WIDTH,
            "permutation_resamples": PERMUTATION_RESAMPLES,
            "permutation_seed": PERMUTATION_SEED,
            "alpha": ALPHA,
            "minimum_effect_for_significance": MINIMUM_EFFECT,
            "effect_size": "Cliff's delta",
        },
        "high_occupancy_intervals": len(high),
        "collapsed_intervals": len(collapsed),
        "healthy_intervals_at_high_occupancy": len(healthy),
        "matched_healthy_pool": len(matched_pool),
        "unmatched_collapsed": unmatched,
        "collapsed_intervals_detail": collapsed_rows,
        "matched_healthy_detail": healthy_rows,
        "separations": separations,
        "best_separating_feature": best,
        "features_significantly_separating": [row["feature"] for row in separating],
        "conclusions": _conclusions(len(collapsed), bool(separating), best),
    }
    return report, {"collapsed": collapsed_features, "healthy": healthy_features}


def _conclusions(collapsed_n: int, separated: bool, best: dict | None) -> str:
    if collapsed_n == 0:
        return "no collapsed interval at the occupancy floor; nothing to separate"
    if not separated or best is None:
        return (
            "no feature's distribution separates the collapsed intervals from "
            "matched healthy ones at the declared effect and significance"
        )
    direction = "higher" if best["cliff_delta"] > 0 else "lower"
    return (
        f"{best['feature']} separates the groups (Cliff's delta "
        f"{best['cliff_delta']:.3f}, p={best['permutation_p']:.4g}); collapsed "
        f"intervals carry {direction} values"
    )


def synthetic_log(path: Path) -> None:
    """Plant a known separation and known retraction and HiCache lines."""
    lines: list[str] = []
    for index in range(30):
        lines.append(
            f"[2026-09-23 12:00:{index:02d} TP0 EP0] Decode batch, #running-req: 32, "
            "#full token: 1, full token usage: 0.85, #swa token: 1, "
            "swa token usage: 0.10, accept len: 4.0, accept rate: 0.60, "
            "cuda graph: True, gen throughput (token/s): 500.00, #queue-req: 0"
        )
        lines.append(
            f"[2026-09-23 12:00:{index:02d} TP0 EP0] Prefill batch, #new-seq: 1, "
            "#new-token: 100, #cached-token: 9900, full token usage: 0.85, "
            "swa token usage: 0.10, #running-req: 32, #queue-req: 0, "
            "#pending-token: 0, cuda graph: False, input throughput (token/s): 900.0"
        )
    for index in range(30):
        lines.append(
            f"[2026-09-23 12:01:{index:02d} TP0 EP0] Decode batch, #running-req: 32, "
            "#full token: 1, full token usage: 0.85, #swa token: 1, "
            "swa token usage: 0.10, accept len: 3.8, accept rate: 0.60, "
            "cuda graph: True, gen throughput (token/s): 103.00, #queue-req: 0"
        )
        lines.append(
            f"[2026-09-23 12:01:{index:02d} TP0 EP0] Prefill batch, #new-seq: 4, "
            "#new-token: 8000, #cached-token: 2000, full token usage: 0.85, "
            "swa token usage: 0.10, #running-req: 32, #queue-req: 32, "
            "#pending-token: 0, cuda graph: False, input throughput (token/s): 900.0"
        )
    lines.append(
        "[2026-09-23 12:01:05 TP0 EP0] retract decode requests: 2 sequences "
        "preempted while the pool is pinned"
    )
    lines.append("[2026-09-23 12:01:06 TP0 EP0] HiCache load started for 1 node")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_control(directory: Path) -> dict[str, object]:
    """Prove the reducer detects a planted separation and counts planted lines."""
    path = directory / "synthetic_tail.log"
    synthetic_log(path)
    report, _ = analyse(str(path), "synthetic")
    best = report["best_separating_feature"] or {}
    return {
        "source": str(path),
        "planted_prefill_new_tokens_collapsed": 8000,
        "planted_prefill_new_tokens_healthy": 100,
        "planted_retract_lines": 1,
        "planted_hicache_load_lines": 1,
        "detected_best_feature": best.get("feature"),
        "detected_best_cliff_delta": best.get("cliff_delta"),
        "detected_best_p": best.get("permutation_p"),
        "features_significantly_separating": report[
            "features_significantly_separating"
        ],
        "collapsed_intervals": report["collapsed_intervals"],
        "runtime_retract_evict_lines": report["runtime_retract_evict_lines"],
        "runtime_hicache_load_lines": report["runtime_hicache_load_lines"],
        "separation_detected": bool(
            not math.isnan(best.get("abs_cliff_delta", float("nan")))
            and best.get("abs_cliff_delta", 0.0) >= 0.9
            and best.get("permutation_p", 1.0) < ALPHA
        ),
        "planted_feature_separates": (
            "prefill_new_tokens_window" in report["features_significantly_separating"]
        ),
        "retraction_line_counted": report["runtime_retract_evict_lines"] == 1,
        "hicache_line_counted": report["runtime_hicache_load_lines"] == 1,
    }


def plot(
    feature: str, collapsed: list[float], healthy: list[float], path: str
) -> None:
    """Plot the best separating feature for both groups."""
    collapsed = [v for v in collapsed if not math.isnan(v)]
    healthy = [v for v in healthy if not math.isnan(v)]
    figure, axis = plt.subplots(figsize=(7.4, 5.4))
    positions = [1, 2]
    data = [collapsed, healthy]
    box = axis.boxplot(
        data,
        positions=positions,
        widths=0.45,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#1b1b1b", "linewidth": 1.6},
    )
    for patch, colour in zip(box["boxes"], ["#8a1c1c", "#1f5c8a"], strict=True):
        patch.set_facecolor(colour)
        patch.set_alpha(0.30)
    rng = np.random.default_rng(PERMUTATION_SEED)
    for position, values in zip(positions, data, strict=True):
        jitter = rng.uniform(-0.07, 0.07, size=len(values)) if values else []
        axis.scatter(
            position + np.asarray(jitter, dtype=float),
            values,
            s=24,
            color="#1b1b1b",
            alpha=0.40,
            zorder=3,
        )
    axis.set_xticks(positions)
    axis.set_xticklabels(
        [f"collapsed (n={len(collapsed)})", f"healthy (n={len(healthy)})"]
    )
    axis.set_ylabel(feature)
    axis.set_title(
        f"{feature} by group at matched occupancy and width\n"
        f"median collapsed {median(collapsed)} vs healthy {median(healthy)}"
    )
    axis.grid(True, axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", action="append", default=[], help="LABEL=LOG_PATH")
    parser.add_argument("--json", dest="json_path", required=True)
    parser.add_argument("--png", required=True)
    parser.add_argument("--control", required=True)
    args = parser.parse_args()
    if not args.job:
        parser.error("at least one --job LABEL=LOG_PATH is required")

    output_directory = Path(args.json_path).parent
    control = run_control(output_directory)
    Path(args.control).write_text(
        json.dumps(control, indent=2) + "\n", encoding="utf-8"
    )
    if not (
        control["separation_detected"]
        and control["planted_feature_separates"]
        and control["retraction_line_counted"]
        and control["hicache_line_counted"]
    ):
        print("CONTROL FAILED:", json.dumps(control))
        return 1

    reports: list[dict[str, object]] = []
    plot_collapsed: list[float] = []
    plot_healthy: list[float] = []
    chosen_feature: str | None = None
    for specification in args.job:
        label, separator, log_path = specification.partition("=")
        if not separator:
            parser.error(f"invalid --job value: {specification}")
        report, groups = analyse(log_path, label)
        reports.append(report)
        best = report["best_separating_feature"]
        if best is not None and chosen_feature is None:
            chosen_feature = best["feature"]
        if best is not None and chosen_feature == best["feature"]:
            plot_collapsed.extend(groups["collapsed"][best["feature"]])
            plot_healthy.extend(groups["healthy"][best["feature"]])

    document = {
        "parameters": {
            "collapse_occupancy_at_or_above": COLLAPSE_OCCUPANCY,
            "collapse_rate_below_tok_s": COLLAPSE_RATE_TOK_S,
            "occupancy_bin_width": BIN_WIDTH,
            "window_seconds": WINDOW_SECONDS,
            "permutation_resamples": PERMUTATION_RESAMPLES,
            "permutation_seen": PERMUTATION_SEED,
            "alpha": ALPHA,
            "effect_size": "Cliff's delta",
        },
        "jobs": {report["label"]: report for report in reports},
        "control": control,
    }
    Path(args.json_path).write_text(
        json.dumps(document, indent=2) + "\n", encoding="utf-8"
    )
    if chosen_feature is not None:
        plot(chosen_feature, plot_collapsed, plot_healthy, args.png)

    for report in reports:
        best = report["best_separating_feature"]
        print(
            f"job {report['label']}: decode={report['decode_intervals_total']} "
            f"occ>=.80={report['high_occupancy_intervals']} "
            f"collapsed={report['collapsed_intervals']} "
            f"matched_healthy={report['matched_healthy_pool']} "
            f"best={best['feature'] if best else None} "
            f"delta={best['cliff_delta'] if best else None} "
            f"p={best['permutation_p'] if best else None}"
        )
        print("  conclusions:", report["conclusions"])
    print("control:", json.dumps(control))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
