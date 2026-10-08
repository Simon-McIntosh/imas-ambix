"""Measure decode and prefill behaviour against local-lane request occupancy.

The engine log supplies exact decode width; receipts supply individual request
rates and worker identity; the telemetry index supplies host-cache reads. All
inputs are read-only snapshots. Run ``--control`` before ``--measure``.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import heapq
import json
import math
import re
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from imas_ambix.agent.decode_bins import (  # noqa: E402
    parse_decode_lines,
    physical_lines,
)
from imas_ambix.agent.telemetry_index import TelemetryIndex  # noqa: E402

ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = Path("/home/ITER/mcintos/Code/imas-ambix")
JOBS = ("1280470", "1273253", "1276246", "1278105")
INDEX = Path("/home/ITER/mcintos/.cache/ambix/watch-index.sqlite3")
RECEIPTS = Path("/home/ITER/mcintos/public/imas-ambix/requests.jsonl")
CHANGE_AT = datetime.fromisoformat("2026-10-05T14:33:00+00:00").timestamp()
ENGINE_TIMEZONE = ZoneInfo("Europe/Paris")
MIN_STEPS = 20
RESAMPLES = 400
SEED = 20261008
AGGREGATE_WITHIN_PEAK = 0.95
MIN_RISE_FOR_KNEE = 0.15
CACHE_DROP = 0.05
UNCACHED_MULTIPLIER = 1.5
WORKER_IDLE_S = 120
FALL_DROP = 0.25
WINDOW_S = 60
CONFIG_FIELDS = (
    "max_total_tokens",
    "max_running_requests",
    "chunked_prefill_size",
    "max_prefill_tokens",
    "tp_size",
    "hicache_ratio",
    "hicache_mem_layout",
    "enable_mixed_chunk",
    "disable_radix_cache",
    "disable_chunked_prefix_cache",
)
PREFILL = re.compile(
    r"#new-token:\s*(\d+).*?#cached-token:\s*(\d+).*?"
    r"full token usage:\s*([\d.]+).*?#running-req:\s*(\d+)"
)
FULL_TOKEN = re.compile(r"#full token:\s*(\d+)")
KV_USAGE = re.compile(r"full token usage:\s*([\d.]+)")
STAMP = re.compile(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")


def epoch(stamp: str) -> int:
    """Convert the serve host's Europe/Paris wall clock to Unix seconds."""
    return int(
        datetime.fromisoformat(stamp).replace(tzinfo=ENGINE_TIMEZONE).timestamp()
    )


def med(values: list[float] | np.ndarray) -> float | None:
    return float(np.median(values)) if len(values) else None


def pct(values: list[float] | np.ndarray, q: float) -> float | None:
    return float(np.percentile(values, q)) if len(values) else None


def ci(values: list[float], *, seed: int = SEED) -> list[float] | None:
    """Percentile bootstrap for a bin median; cap huge bins without bias."""
    if not values:
        return None
    rng = np.random.default_rng(seed)
    sample = np.asarray(values, dtype=float)
    if len(sample) > 3000:
        sample = sample[rng.choice(len(sample), 3000, replace=False)]
    draws = [
        np.median(sample[rng.integers(len(sample), size=(50, len(sample)))], axis=1)
        for _ in range(RESAMPLES // 50)
    ]
    return [float(x) for x in np.percentile(np.concatenate(draws), [2.5, 97.5])]


def aggregate_knee(curve: dict[int, float]) -> int | None:
    """First width within 5% of peak, provided an actual rise precedes it."""
    if len(curve) < 4:
        return None
    peak = max(curve.values())
    first = curve[min(curve)]
    if peak <= 0 or (peak - first) / peak < MIN_RISE_FOR_KNEE:
        return None
    candidates = [
        w for w, value in sorted(curve.items()) if value >= AGGREGATE_WITHIN_PEAK * peak
    ]
    return min(candidates) if candidates else None


def piecewise_break(curve: dict[int, float]) -> dict[str, float | int] | None:
    """Two linear segments of per-request rate; require excess sharing loss."""
    widths = np.asarray(sorted(curve), dtype=float)
    if len(widths) < 7:
        return None
    rates = np.asarray([curve[int(w)] for w in widths])
    best = None
    for index in range(2, len(widths) - 3):
        left = np.polyfit(widths[: index + 1], rates[: index + 1], 1)
        right = np.polyfit(widths[index:], rates[index:], 1)
        predicted = np.r_[
            np.polyval(left, widths[:index]), np.polyval(right, widths[index:])
        ]
        error = float(np.sum((rates - predicted) ** 2))
        if best is None or error < best[0]:
            best = (error, int(widths[index]), float(left[0]), float(right[0]))
    assert best is not None
    _, width, before, after = best
    sharing_slope = -curve[width] / width
    if after >= sharing_slope or after >= before:
        return None
    return {
        "width": width,
        "slope_before": before,
        "slope_after": after,
        "sharing_slope_at_break": sharing_slope,
    }


def bootstrap_width(groups: dict[int, list[float]], reducer) -> list[int] | None:
    rng = np.random.default_rng(SEED)
    draws = []
    arrays = {
        w: np.asarray(v, dtype=float) for w, v in groups.items() if len(v) >= MIN_STEPS
    }
    for _ in range(RESAMPLES):
        curve = {
            w: float(np.median(v[rng.integers(len(v), size=len(v))]))
            for w, v in arrays.items()
        }
        result = reducer(curve)
        if isinstance(result, dict):
            result = result["width"]
        if result is not None:
            draws.append(result)
    return [int(x) for x in np.percentile(draws, [2.5, 97.5])] if draws else None


def control() -> None:
    rng = np.random.default_rng(SEED)
    curves = {}
    for label in ("planted", "flat"):
        groups = {}
        for width in range(1, 25):
            level = 80 + min(width, 12) * 25 if label == "planted" else 380
            groups[width] = (level + rng.normal(0, 2, 40)).tolist()
        curve = {w: med(v) for w, v in groups.items()}
        detected = aggregate_knee(curve)
        curves[label] = {
            "detected_width": detected,
            "bootstrap_ci95": bootstrap_width(groups, aggregate_knee),
        }
    print(json.dumps(curves, sort_keys=True))
    assert abs(curves["planted"]["detected_width"] - 12) <= 1
    assert curves["flat"]["detected_width"] is None


def read_prefix(path: Path) -> tuple[bytes, dict]:
    """Snapshot a growing file at its opening size and digest only those bytes."""
    with path.open("rb") as stream:
        length = path.stat().st_size
        raw = stream.read(length)
    return raw, {
        "path": str(path),
        "bytes_consumed": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def load_log(job: str) -> tuple[list[dict], list[dict], dict, dict]:
    path = SOURCE_ROOT / f"deepseek-v4-1-flash-{job}.log"
    raw, source = read_prefix(path)
    lines = physical_lines(raw.decode("utf-8", errors="replace"))
    args = next(
        (
            ast.literal_eval(line.split("server_args=", 1)[1])
            for line in lines
            if "server_args=" in line
        ),
        None,
    )
    config = {key: args.get(key) for key in CONFIG_FIELDS} if args else {}
    parsed = parse_decode_lines(lines)
    malformed_lines = {row.line_number for row in parsed.malformed}
    details = []
    prefills = []
    prefill_announced = 0
    for number, line in enumerate(lines, start=1):
        if "Decode batch" in line:
            if number in malformed_lines:
                continue
            tokens = FULL_TOKEN.search(line)
            kv = KV_USAGE.search(line)
            details.append(
                (int(tokens[1]) if tokens else None, float(kv[1]) if kv else None)
            )
        elif "Prefill batch" in line:
            prefill_announced += 1
            stamp = STAMP.search(line)
            match = PREFILL.search(line)
            if stamp and match:
                prefills.append(
                    {
                        "time": epoch(stamp[1]),
                        "new": int(match[1]),
                        "cached": int(match[2]),
                        "kv": float(match[3]),
                        "width": int(match[4]),
                        "job": job,
                    }
                )
    if len(details) != len(parsed.intervals):
        raise ValueError("decode parser/detail count mismatch")
    decodes = []
    for row, (tokens, kv) in zip(parsed.intervals, details, strict=True):
        when = epoch(row.timestamp)
        decodes.append(
            {
                "time": when,
                "width": row.width,
                "rate": row.generation_rate,
                "per_request_rate": row.generation_rate / row.width
                if row.width
                else None,
                "accept": row.accept_length,
                "full_tokens": tokens,
                "kv": kv,
                "job": job,
            }
        )
    source.update(
        {
            "timestamp_timezone": "Europe/Paris",
            "rows": len(lines),
            "decode_rows": len(decodes),
            "malformed_decode_rows": len(parsed.malformed),
            "prefill_rows": len(prefills),
            "prefill_announced": prefill_announced,
            "malformed_prefill_rows": prefill_announced - len(prefills),
            "span": [
                min((r["time"] for r in decodes), default=None),
                max((r["time"] for r in decodes), default=None),
            ],
        }
    )
    if (
        source["span"][1] is not None
        and source["span"][1] > datetime.now(UTC).timestamp() + 120
    ):
        raise ValueError("serve log lies in the future; timezone alignment failed")
    return decodes, prefills, config, source


def load_receipts(start: int, end: int) -> tuple[list[dict], dict]:
    raw, source = read_prefix(RECEIPTS)
    rows = []
    total = 0
    for line in raw.splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # possibly an incomplete live tail
        total += 1
        if row.get("status") != "completed" or not row.get("model"):
            continue
        try:
            when = datetime.fromisoformat(
                row["timestamp"].replace("Z", "+00:00")
            ).timestamp()
            duration = float(row["duration_s"])
        except KeyError, TypeError, ValueError:
            continue
        if when + duration < start or when > end:
            continue
        row["start_epoch"] = when
        row["end_epoch"] = when + duration
        rows.append(row)
    source.update(
        {
            "rows": total,
            "included_rows": len(rows),
            "span": [
                min((r["start_epoch"] for r in rows), default=None),
                max((r["end_epoch"] for r in rows), default=None),
            ],
        }
    )
    return rows, source


def load_telemetry(start: int, end: int) -> tuple[list[dict], dict]:
    names = (
        "engine.requests_running",
        "engine.uncached_prompt_tokens",
        "engine.cached_prompt_tokens.device",
        "engine.cached_prompt_tokens.host",
        "engine.kv_pool_occupancy",
        "engine.spec_decode.accept_length",
        "engine.prefix_cache_hit_rate",
        "engine.generation_tokens",
        "engine.prompt_tokens",
        "engine.prefix_cache_hits",
    )
    index = TelemetryIndex.readonly(INDEX)
    conn = index._conn
    placeholders = ",".join("?" for _ in names)
    query = (
        "SELECT s.id,s.ts_epoch,s.host,s.boot_id,s.job_id,m.name,m.kind,m.value "
        "FROM sample s JOIN measurement m ON m.sample_id=s.id "
        "WHERE s.profile_slug=? AND s.ts_epoch BETWEEN ? AND ? "
        f"AND m.name IN ({placeholders}) "
        "ORDER BY s.ts_epoch,s.id"
    )
    rows = {}
    digest = hashlib.sha256()
    bytes_read = 0
    count = 0
    for item in conn.execute(query, ("deepseek-v4-1-flash", start, end, *names)):
        record = tuple(item)
        encoded = json.dumps(record, separators=(",", ":")).encode() + b"\n"
        digest.update(encoded)
        bytes_read += len(encoded)
        count += 1
        key = item[0]
        row = rows.setdefault(
            key, {"time": item[1], "host": item[2], "boot": item[3], "job": item[4]}
        )
        row[item[5]] = item[7]
    index.close()
    points = list(rows.values())
    source = {
        "path": str(INDEX),
        "sha256": digest.hexdigest(),
        "sha256_scope": "canonical queried rows, not the live database file",
        "bytes_consumed": bytes_read,
        "measurement_rows": count,
        "samples": len(points),
        "span": [
            min((p["time"] for p in points), default=None),
            max((p["time"] for p in points), default=None),
        ],
    }
    return points, source


def receipt_metrics(
    rows: list[dict],
) -> tuple[dict[int, list[dict]], dict[int, list[float]], dict[int, int], dict]:
    """Sweep request starts and ends; count active and recently completed runs."""
    starts = sorted(rows, key=lambda row: row["start_epoch"])
    active: list[tuple[float, int]] = []
    active_runs: dict[str, int] = defaultdict(int)
    recent_end: dict[str, float] = {}
    by_width: dict[int, list[dict]] = defaultdict(list)
    worker_rates: dict[int, list[float]] = defaultdict(list)
    completions: dict[int, list[dict]] = defaultdict(list)
    for serial, row in enumerate(starts):
        now = row["start_epoch"]
        while active and active[0][0] <= now:
            ended, prior = heapq.heappop(active)
            run = starts[prior].get("run_id")
            if run:
                active_runs[run] -= 1
                recent_end[run] = max(recent_end.get(run, 0), ended)
        for run, ended in list(recent_end.items()):
            if ended < now - WORKER_IDLE_S and active_runs[run] <= 0:
                del recent_end[run]
        width = len(active) + 1
        run = row.get("run_id")
        if run:
            active_runs[run] += 1
        heapq.heappush(active, (row["end_epoch"], serial))
        workers = len(
            {key for key, n in active_runs.items() if n > 0} | set(recent_end)
        )
        row["request_occupancy"] = width
        row["worker_occupancy"] = workers
        completion = row.get("completion_tokens")
        ttft = row.get("time_to_first_token_s")
        duration = row.get("duration_s")
        try:
            decode_s = float(duration) - float(ttft)
            row["decode_tok_s"] = float(completion) / decode_s if decode_s > 0 else None
        except TypeError, ValueError:
            row["decode_tok_s"] = None
        by_width[width].append(row)
        completions[int(row["end_epoch"] // WINDOW_S) * WINDOW_S].append(row)
    # Sample every minute, including tool-call idle minutes with zero output.
    events = []
    for row in starts:
        if row.get("run_id"):
            events.append((row["start_epoch"], 1, row["run_id"]))
            events.append((row["end_epoch"], -1, row["run_id"]))
    events.sort()
    active_by_run: dict[str, int] = defaultdict(int)
    last_end: dict[str, float] = {}
    worker_minutes: dict[int, int] = {}
    event_index = 0
    first_minute = int(starts[0]["start_epoch"] // WINDOW_S) * WINDOW_S
    last_minute = int(max(r["end_epoch"] for r in starts) // WINDOW_S) * WINDOW_S
    for when in range(first_minute, last_minute + WORKER_IDLE_S + WINDOW_S, WINDOW_S):
        midpoint = when + WINDOW_S / 2
        while event_index < len(events) and events[event_index][0] <= midpoint:
            event_time, direction, run = events[event_index]
            active_by_run[run] += direction
            if direction < 0:
                last_end[run] = event_time
            event_index += 1
        for run in list(active_by_run):
            if (
                active_by_run[run] <= 0
                and last_end.get(run, -math.inf) < midpoint - WORKER_IDLE_S
            ):
                del active_by_run[run]
                last_end.pop(run, None)
        live = len(active_by_run)
        worker_minutes[when] = live
        if live == 0:
            continue
        tokens = sum(
            (
                float(row.get("completion_tokens") or 0)
                / max(float(row.get("sample_fraction") or 1), 1e-6)
            )
            for row in completions[when]
        )
        worker_rates[live].append(tokens / WINDOW_S / live)
    by_run = defaultdict(list)
    for row in starts:
        if row.get("run_id"):
            by_run[row["run_id"]].append(row)
    overlap_start = 0
    overlap_end = 0
    adjacent_pairs = 0
    for rows_of_run in by_run.values():
        for earlier, later in zip(rows_of_run, rows_of_run[1:], strict=False):
            adjacent_pairs += 1
            overlap_start += later["start_epoch"] < earlier["end_epoch"]
            overlap_end += (
                later["start_epoch"] - float(later["duration_s"])
                < earlier["start_epoch"]
            )
    check = {
        "timestamp_interpretation": "request start",
        "basis": (
            "router _record_receipt passes started_at and duration_s to sink; "
            "per-run adjacent intervals overlap less under start than "
            "end interpretation"
        ),
        "adjacent_per_run_pairs": adjacent_pairs,
        "overlaps_if_start": overlap_start,
        "overlaps_if_end": overlap_end,
        "positive_duration_rows": sum(
            row["end_epoch"] > row["start_epoch"] for row in rows
        ),
        "request_start_span": [starts[0]["start_epoch"], starts[-1]["start_epoch"]]
        if starts
        else None,
        "sample_fraction_below_one": sum(
            float(r.get("sample_fraction") or 1) < 1 for r in rows
        ),
    }
    return by_width, worker_rates, worker_minutes, check


def telemetry_bins(points: list[dict]) -> dict[int, dict]:
    """Reduce sampled cache and host-tier metrics by engine occupancy."""
    by_width: dict[int, list[dict]] = defaultdict(list)
    previous: dict[tuple[str, str], dict] = {}
    for row in points:
        width = row.get("engine.requests_running")
        if width is None:
            continue
        width = int(round(width))
        prior = previous.get((row["host"], row["boot"]))
        if prior and 0 < row["time"] - prior["time"] <= 300:
            for name in (
                "engine.generation_tokens",
                "engine.prompt_tokens",
                "engine.prefix_cache_hits",
            ):
                if name in row and name in prior:
                    delta = row[name] - prior[name]
                    if delta >= 0:
                        row[name + ".rate"] = delta / (row["time"] - prior["time"])
        previous[(row["host"], row["boot"])] = row
        by_width[width].append(row)
    result = {}
    fields = (
        "engine.cached_prompt_tokens.host",
        "engine.cached_prompt_tokens.device",
        "engine.uncached_prompt_tokens",
        "engine.kv_pool_occupancy",
        "engine.prefix_cache_hit_rate",
        "engine.generation_tokens.rate",
        "engine.prompt_tokens.rate",
        "engine.prefix_cache_hits.rate",
    )
    for width, rows in by_width.items():
        result[width] = {
            name: med([row[name] for row in rows if name in row]) for name in fields
        }
        result[width]["samples"] = len(rows)
    return result


def per_width(
    decodes: list[dict],
    prefills: list[dict],
    receipts: dict[int, list[dict]],
    telemetry: dict[int, dict],
) -> dict:
    by_width: dict[int, list[dict]] = defaultdict(list)
    by_prefill: dict[int, list[dict]] = defaultdict(list)
    time_at_width: dict[int, set[tuple[str, int]]] = defaultdict(set)
    decode_at_width: dict[int, set[tuple[str, int]]] = defaultdict(set)
    for row in decodes:
        if row["width"] > 0:
            by_width[row["width"]].append(row)
            time_at_width[row["width"]].add((row["job"], row["time"]))
            decode_at_width[row["width"]].add((row["job"], row["time"]))
    for row in prefills:
        by_prefill[row["width"]].append(row)
        time_at_width[row["width"]].add((row["job"], row["time"]))
    all_seconds = sum(len(v) for width, v in time_at_width.items() if width > 0)
    table = {}
    for width, rows in sorted(by_width.items()):
        if len(rows) < MIN_STEPS:
            continue
        rates = [row["rate"] for row in rows]
        request_rates = [
            row["per_request_rate"]
            for row in rows
            if row["per_request_rate"] is not None
        ]
        prs = by_prefill.get(width, [])
        cached = sum(row["cached"] for row in prs)
        new = sum(row["new"] for row in prs)
        receipt_rows = receipts.get(width, [])
        receipt_rates = [
            row["decode_tok_s"]
            for row in receipt_rows
            if row["decode_tok_s"] is not None
        ]
        contexts = [
            row["full_tokens"] / width for row in rows if row["full_tokens"] is not None
        ]
        stamps = [row["time"] for row in rows]
        wall_s = len(time_at_width[width])
        kv = [row["kv"] for row in rows if row["kv"] is not None]
        prefill_seconds = {(row["job"], row["time"]) for row in prs}
        ttft = [
            float(row["time_to_first_token_s"])
            for row in receipt_rows
            if row.get("time_to_first_token_s") is not None
        ]
        table[str(width)] = {
            "decode_steps": len(rows),
            "observed_seconds": wall_s,
            "time_share": wall_s / all_seconds if all_seconds else 0,
            "aggregate_tok_s_median": med(rates),
            "aggregate_ci95": ci(rates),
            "per_request_tok_s_median": med(request_rates),
            "per_request_ci95": ci(request_rates),
            "receipt_per_request_tok_s_median": med(receipt_rates),
            "receipt_per_request_ci95": ci(receipt_rates),
            "receipt_count": len(receipt_rows),
            "uncached_prefill_tok_s": new / wall_s if wall_s else None,
            "cache_hit_fraction": cached / (cached + new) if cached + new else None,
            "prefill_active_second_fraction": len(prefill_seconds) / wall_s
            if wall_s
            else None,
            "prefill_only_second_fraction": (
                len(prefill_seconds - decode_at_width[width]) / wall_s
                if wall_s
                else None
            ),
            "prefill_new_tokens": new,
            "prefill_cached_tokens": cached,
            "prefill_batches": len(prs),
            "kv_median": med(kv),
            "kv_p90": pct(kv, 90),
            "ttft_p50_s": med(ttft),
            "ttft_p90_s": pct(ttft, 90),
            "context_tokens_median": med(contexts),
            "accept_length_median": med([r["accept"] for r in rows]),
            "first_seen": min(stamps),
            "last_seen": max(stamps),
            "hours_spanned": (max(stamps) - min(stamps)) / 3600,
            "observed_hour_count": len({stamp // 3600 for stamp in stamps}),
            "jobs": sorted({r["job"] for r in rows}),
            "telemetry": telemetry.get(width),
        }
    widths = sorted(int(w) for w in table)
    for width in widths:
        prior = width - 1
        table[str(width)]["marginal_aggregate_tok_s"] = (
            table[str(width)]["aggregate_tok_s_median"]
            - table[str(prior)]["aggregate_tok_s_median"]
            if str(prior) in table
            else None
        )
    return table


def worker_axis_table(
    decodes: list[dict],
    prefills: list[dict],
    receipt_rows: list[dict],
    worker_rates: dict[int, list[float]],
    worker_minutes: dict[int, int],
    telemetry_rows: list[dict],
) -> dict:
    """Join worker-minute occupancy to engine and receipt observations."""
    by_decode = defaultdict(list)
    by_prefill = defaultdict(list)
    by_receipt = defaultdict(list)
    by_telemetry = defaultdict(list)

    def count_at(when: float) -> int | None:
        return worker_minutes.get(int(when // WINDOW_S) * WINDOW_S)

    for row in decodes:
        workers = count_at(row["time"])
        if workers is not None:
            by_decode[workers].append(row)
    for row in prefills:
        workers = count_at(row["time"])
        if workers is not None:
            by_prefill[workers].append(row)
    for row in receipt_rows:
        workers = count_at(row["start_epoch"])
        if workers is not None:
            by_receipt[workers].append(row)
    for row in telemetry_rows:
        workers = count_at(row["time"])
        if workers is not None:
            by_telemetry[workers].append(row)
    total_minutes = len(worker_minutes)
    table = {}
    for workers in sorted(set(worker_minutes.values())):
        if workers <= 0:
            continue
        minutes = sum(count == workers for count in worker_minutes.values())
        dec = by_decode[workers]
        pre = by_prefill[workers]
        rec = by_receipt[workers]
        tel = by_telemetry[workers]
        new = sum(r["new"] for r in pre)
        cached = sum(r["cached"] for r in pre)
        stamps = [r["time"] for r in dec]
        effective = worker_rates.get(workers, [])
        receipt_rates = [
            r["decode_tok_s"] for r in rec if r["decode_tok_s"] is not None
        ]
        ttft = [
            float(r["time_to_first_token_s"])
            for r in rec
            if r.get("time_to_first_token_s") is not None
        ]
        table[str(workers)] = {
            "minutes": minutes,
            "time_share": minutes / total_minutes,
            "aggregate_tok_s_median": med([r["rate"] for r in dec]),
            "aggregate_ci95": ci([r["rate"] for r in dec]),
            "per_request_tok_s_median": med(
                [
                    r["per_request_rate"]
                    for r in dec
                    if r["per_request_rate"] is not None
                ]
            ),
            "receipt_per_request_tok_s_median": med(receipt_rates),
            "receipt_per_request_ci95": ci(receipt_rates),
            "per_worker_effective_tok_s_median": med(effective),
            "per_worker_effective_ci95": ci(effective),
            "uncached_prefill_tok_s": new / (minutes * WINDOW_S),
            "cache_hit_fraction": cached / (cached + new) if cached + new else None,
            "prefill_active_minute_fraction": (
                len({int(r["time"] // WINDOW_S) * WINDOW_S for r in pre}) / minutes
            ),
            "kv_median": med([r["kv"] for r in dec if r["kv"] is not None]),
            "kv_p90": pct([r["kv"] for r in dec if r["kv"] is not None], 90),
            "ttft_p50_s": med(ttft),
            "ttft_p90_s": pct(ttft, 90),
            "context_tokens_median": med(
                [
                    float(r["prompt_tokens"])
                    for r in rec
                    if r.get("prompt_tokens") is not None
                ]
            ),
            "accept_length_median": med([r["accept"] for r in dec]),
            "host_tier_reads_median": med(
                [
                    r["engine.cached_prompt_tokens.host"]
                    for r in tel
                    if "engine.cached_prompt_tokens.host" in r
                ]
            ),
            "first_seen": min(stamps) if stamps else None,
            "last_seen": max(stamps) if stamps else None,
            "hours_spanned": (max(stamps) - min(stamps)) / 3600 if stamps else None,
            "observed_hour_count": len({stamp // 3600 for stamp in stamps}),
            "decode_steps": len(dec),
            "receipts": len(rec),
        }
    return table


def onset(table: dict) -> dict:
    low = [
        row
        for key, row in table.items()
        if 2 <= int(key) <= 8
        and row["cache_hit_fraction"] is not None
        and row["uncached_prefill_tok_s"] is not None
    ]
    if not low:
        return {"width": None, "kv_median": None, "reason": "no low-occupancy baseline"}
    baseline_hit = med([row["cache_hit_fraction"] for row in low])
    baseline_new = med([row["uncached_prefill_tok_s"] for row in low])
    found = next(
        (
            (int(w), row)
            for w, row in sorted(table.items(), key=lambda pair: int(pair[0]))
            if row["cache_hit_fraction"] is not None
            and row["uncached_prefill_tok_s"] is not None
            and row["cache_hit_fraction"] <= baseline_hit - CACHE_DROP
            and row["uncached_prefill_tok_s"] >= baseline_new * UNCACHED_MULTIPLIER
        ),
        None,
    )
    return {
        "width": found[0] if found else None,
        "kv_median": found[1]["kv_median"] if found else None,
        "baseline_hit": baseline_hit,
        "baseline_uncached_tok_s": baseline_new,
        "reason": None
        if found
        else "joint cache-drop and prefill-rise threshold never crossed",
    }


def onset_bootstrap(prefills: list[dict], table: dict) -> dict:
    """Resample prefill batches within width, preserving observed wall seconds."""
    grouped = defaultdict(list)
    for row in prefills:
        if str(row["width"]) in table:
            grouped[row["width"]].append((row["new"], row["cached"]))
    arrays = {w: np.asarray(rows, dtype=float) for w, rows in grouped.items() if rows}
    rng = np.random.default_rng(SEED)
    detected = []
    for _ in range(RESAMPLES):
        drawn = {}
        for width, values in arrays.items():
            sample = values[rng.integers(len(values), size=len(values))]
            new, cached = sample.sum(axis=0)
            seconds = table[str(width)]["observed_seconds"]
            drawn[width] = (cached / (cached + new), new / seconds)
        low = [drawn[w] for w in range(2, 9) if w in drawn]
        if not low:
            continue
        baseline_hit = med([pair[0] for pair in low])
        baseline_new = med([pair[1] for pair in low])
        candidates = [
            width
            for width, (hit, new) in sorted(drawn.items())
            if hit <= baseline_hit - CACHE_DROP
            and new >= baseline_new * UNCACHED_MULTIPLIER
        ]
        if candidates:
            detected.append(candidates[0])
    return {
        "conditional_ci95": [int(x) for x in np.percentile(detected, [2.5, 97.5])]
        if detected
        else None,
        "detection_fraction": len(detected) / RESAMPLES,
        "resamples": RESAMPLES,
    }


def episodes(
    decodes: list[dict], prefills: list[dict], baseline_hit: float
) -> tuple[list[dict], dict]:
    seconds: dict[tuple[str, int], dict[str, list[float] | float]] = defaultdict(
        lambda: {"rate": [], "width": [], "kv": [], "new": 0.0, "cached": 0.0}
    )
    for row in decodes:
        key = (row["job"], row["time"])
        for field, target in (("rate", "rate"), ("width", "width"), ("kv", "kv")):
            if row[field] is not None:
                seconds[key][target].append(row[field])
    for row in prefills:
        record = seconds[(row["job"], row["time"])]
        record["new"] += row["new"]
        record["cached"] += row["cached"]
        record["width"].append(row["width"])
        record["kv"].append(row["kv"])
    minute: dict[tuple[str, int], list[dict]] = defaultdict(list)
    series = []
    for (job, when), record in sorted(seconds.items()):
        point = {
            "job": job,
            "time": when,
            "rate": med(record["rate"]) if record["rate"] else 0.0,
            "width": med(record["width"]),
            "kv": med(record["kv"]),
            "new": record["new"],
            "cached": record["cached"],
        }
        series.append(point)
        minute[(job, when // WINDOW_S * WINDOW_S)].append(point)
    windows = {}
    for key, points in minute.items():
        if len(points) < 15:
            continue
        new = sum(p["new"] for p in points)
        cached = sum(p["cached"] for p in points)
        windows[key] = {
            "rate": float(np.mean([p["rate"] for p in points])),
            "width": med([p["width"] for p in points]),
            "new": new / WINDOW_S,
            "cache_hit_fraction": cached / (cached + new) if cached + new else None,
            "kv": med([p["kv"] for p in points if p["kv"] is not None]),
            "seconds": len(points),
        }
    found = []
    last_episode: dict[str, int] = {}
    for (job, when), current in sorted(windows.items(), key=lambda pair: pair[0]):
        prior = windows.get((job, when - WINDOW_S))
        if not prior or when - last_episode.get(job, -(10**12)) < 3 * WINDOW_S:
            continue
        if (
            prior["rate"] >= 100
            and current["width"] >= prior["width"]
            and current["rate"] <= (1 - FALL_DROP) * prior["rate"]
        ):
            surge = current["new"] >= UNCACHED_MULTIPLIER * max(prior["new"], 1)
            cache_drop = (
                prior["cache_hit_fraction"] is not None
                and current["cache_hit_fraction"] is not None
                and current["cache_hit_fraction"]
                <= prior["cache_hit_fraction"] - CACHE_DROP
            )
            cache_low = (
                current["cache_hit_fraction"] is not None
                and current["cache_hit_fraction"] <= baseline_hit - CACHE_DROP
            )
            found.append(
                {
                    "job": job,
                    "time": when,
                    "prior": prior,
                    "current": current,
                    "prefill_surge": surge,
                    "cache_hit_drop": cache_drop,
                    "cache_hit_below_baseline": cache_low,
                }
            )
            last_episode[job] = when
    return found, {"series": series, "windows": len(windows)}


def style_axis(axis, xlabel: str, ylabel: str) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(False)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(ROOT / f"{name}.svg", dpi=100)
    plt.close(fig)


def plot_histogram(
    decodes: list[dict], prefills: list[dict], worker_minutes: dict[int, int]
) -> None:
    plt.style.use("data-ink")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for index, (axis, source, label) in enumerate(
        (
            (axes[0], [*decodes, *prefills], "running requests"),
            (axes[1], worker_minutes.items(), "live workers"),
        )
    ):
        counts = defaultdict(lambda: [0, 0])
        seen = set()
        for row in source:
            when = row["time"] if index == 0 else row[0]
            count = row["width"] if index == 0 else row[1]
            identity = (row.get("job") if index == 0 else "router", when, count)
            if identity in seen:
                continue
            seen.add(identity)
            counts[count][int(when >= CHANGE_AT)] += 1
        widths = sorted(counts)
        if widths:
            before = np.array([counts[w][0] for w in widths], dtype=float)
            after = np.array([counts[w][1] for w in widths], dtype=float)
            before /= max(before.sum(), 1)
            after /= max(after.sum(), 1)
            axis.bar(
                np.array(widths) - 0.2,
                before,
                width=0.4,
                color="#77848b",
                label="before",
            )
            axis.bar(
                np.array(widths) + 0.2, after, width=0.4, color="#2a6f91", label="after"
            )
            axis.text(
                widths[-1],
                max(after.max(), before.max()) * 0.92,
                "after",
                color="#2a6f91",
                ha="right",
            )
            axis.text(
                widths[-1],
                max(after.max(), before.max()) * 0.72,
                "before",
                color="#77848b",
                ha="right",
            )
        style_axis(axis, label, "share of observed time")
    save(fig, "occupancy-histogram")


def plot_aggregate(table: dict, knee: int | None) -> None:
    plt.style.use("data-ink")
    fig, axis = plt.subplots(figsize=(14, 5))
    widths = sorted(int(w) for w in table)
    values = [table[str(w)]["aggregate_tok_s_median"] for w in widths]
    lows = [table[str(w)]["aggregate_ci95"][0] for w in widths]
    highs = [table[str(w)]["aggregate_ci95"][1] for w in widths]
    axis.bar(widths, values, color="#2a6f91", width=0.78)
    axis.errorbar(
        widths,
        values,
        yerr=[np.array(values) - lows, np.array(highs) - values],
        fmt="none",
        ecolor="#253540",
        capsize=2,
        linewidth=1.2,
    )
    axis.axvline(16, color="#777777", ls=":", lw=1.2)
    axis.text(
        16,
        max(values) * 0.95,
        "gate 16",
        rotation=90,
        va="top",
        ha="right",
        color="#777777",
    )
    if knee is not None:
        axis.axvline(knee, color="#8b5b31", ls="--", lw=1.2)
        axis.text(
            knee,
            max(values) * 0.75,
            f"knee {knee}",
            rotation=90,
            va="top",
            ha="right",
            color="#8b5b31",
        )
    style_axis(axis, "running requests", "aggregate decode [tok/s]")
    save(fig, "aggregate-throughput")


def plot_worker(table: dict, workers: dict, knee: int | None) -> None:
    plt.style.use("data-ink")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    widths = sorted(int(w) for w in table)
    rates = [table[str(w)]["per_request_tok_s_median"] for w in widths]
    axes[0].plot(widths, rates, color="#2a6f91", lw=3, marker="o", ms=4)
    lower = [table[str(w)]["per_request_ci95"][0] for w in widths]
    upper = [table[str(w)]["per_request_ci95"][1] for w in widths]
    axes[0].errorbar(
        widths,
        rates,
        yerr=[np.array(rates) - lower, np.array(upper) - rates],
        fmt="none",
        ecolor="#2a6f91",
        capsize=2,
    )
    if knee is not None:
        axes[0].axvline(knee, color="#8b5b31", ls="--", lw=1.2)
        axes[0].text(
            knee,
            max(rates) * 0.75,
            f"break {knee}",
            rotation=90,
            ha="right",
            color="#8b5b31",
        )
    style_axis(axes[0], "running requests", "per-request decode [tok/s]")
    worker_counts = sorted(
        int(w)
        for w in workers
        if workers[w]["per_worker_effective_tok_s_median"] is not None
    )
    if worker_counts:
        yield_rates = [
            workers[str(w)]["per_worker_effective_tok_s_median"] for w in worker_counts
        ]
        axes[1].plot(
            worker_counts, yield_rates, color="#4a805d", lw=3, marker="o", ms=4
        )
        lower = [workers[str(w)]["per_worker_effective_ci95"][0] for w in worker_counts]
        upper = [workers[str(w)]["per_worker_effective_ci95"][1] for w in worker_counts]
        axes[1].errorbar(
            worker_counts,
            yield_rates,
            yerr=[np.array(yield_rates) - lower, np.array(upper) - yield_rates],
            fmt="none",
            ecolor="#4a805d",
            capsize=2,
        )
    style_axis(axes[1], "live workers", "completed tokens per worker [tok/s]")
    save(fig, "per-worker-throughput")


def plot_prefill(table: dict, thrash: dict) -> None:
    plt.style.use("data-ink")
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    widths = sorted(int(w) for w in table)
    series = (
        ("uncached_prefill_tok_s", "uncached prefill [tok/s]", "#2a6f91"),
        ("cache_hit_fraction", "cache-hit fraction", "#4a805d"),
        ("kv_median", "KV pool usage", "#8b5b31"),
    )
    for axis, (field, label, color) in zip(axes, series, strict=True):
        axis.plot(
            widths,
            [table[str(w)][field] for w in widths],
            color=color,
            lw=3,
            marker="o",
            ms=4,
        )
        if thrash["width"] is not None:
            axis.axvline(thrash["width"], color="#777777", ls=":", lw=1.2)
        style_axis(axis, "running requests", label)
    if thrash["width"] is not None:
        axes[0].text(
            thrash["width"],
            axes[0].get_ylim()[1] * 0.85,
            f"onset {thrash['width']}",
            rotation=90,
            color="#777777",
            ha="right",
        )
    save(fig, "prefill-thrash")


def plot_episodes(episode_rows: list[dict], series: list[dict]) -> None:
    plt.style.use("data-ink")
    chosen = sorted(
        episode_rows,
        key=lambda row: (
            row["current"]["width"] >= 16,
            row["prefill_surge"],
            row["cache_hit_below_baseline"],
            row["cache_hit_drop"],
            row["current"]["width"],
        ),
        reverse=True,
    )[:3]
    fig, axes = plt.subplots(
        4, max(1, len(chosen)), figsize=(14, 10), squeeze=False, sharex="col"
    )
    fields = (
        ("width", "running requests", "#2a6f91"),
        ("rate", "decode [tok/s]", "#4a805d"),
        ("new", "uncached [tok/s]", "#8b5b31"),
        ("kv", "KV pool usage", "#253540"),
    )
    for col, episode in enumerate(chosen):
        points = [
            p
            for p in series
            if p["job"] == episode["job"] and abs(p["time"] - episode["time"]) <= 180
        ]
        elapsed = [(p["time"] - episode["time"]) / 60 for p in points]
        for row, (field, label, color) in enumerate(fields):
            axis = axes[row, col]
            axis.plot(elapsed, [p[field] for p in points], color=color, lw=2.6)
            axis.axvline(0, color="#777777", ls=":", lw=1.2)
            style_axis(axis, "minutes from fall-over", label if col == 0 else "")
            if row == 0:
                axis.text(
                    0.98,
                    0.9,
                    f"job {episode['job']}",
                    transform=axis.transAxes,
                    ha="right",
                    color="#253540",
                )
    if not chosen:
        for row, (_, label, _) in enumerate(fields):
            style_axis(axes[row, 0], "minutes from fall-over", label)
            if row == 0:
                axes[row, 0].text(
                    0.5,
                    0.5,
                    "No qualifying episode",
                    transform=axes[row, 0].transAxes,
                    ha="center",
                )
    save(fig, "fall-over-episodes")


def analyse() -> dict:
    all_decodes = []
    all_prefills = []
    configs = {}
    sources = {}
    reference = None
    for job in JOBS:
        decodes, prefills, config, source = load_log(job)
        same = reference is None or config == reference
        configs[job] = {
            "values": config,
            "matches_live_serve": same,
            "included": same,
            "decode_rows": len(decodes),
            "prefill_rows": len(prefills),
        }
        sources[f"serve_{job}"] = source
        if reference is None:
            reference = config
        if same:
            all_decodes.extend(decodes)
            all_prefills.extend(prefills)
    if not all_decodes:
        raise ValueError("no decode intervals; instrument failed positive control")
    start = min(row["time"] for row in all_decodes)
    end = max(row["time"] for row in all_decodes)
    receipt_rows, receipt_source = load_receipts(start, end)
    sources["router_receipts"] = receipt_source
    if not receipt_rows:
        raise ValueError(
            "no completed model receipts; instrument failed positive control"
        )
    telemetry_rows, telemetry_source = load_telemetry(start, end)
    sources["telemetry_index"] = telemetry_source
    if not telemetry_rows or not any(
        "engine.requests_running" in row for row in telemetry_rows
    ):
        raise ValueError("no running-request telemetry despite known serve logs")
    receipts, workers, worker_minutes, receipt_check = receipt_metrics(receipt_rows)
    telemetry = telemetry_bins(telemetry_rows)
    table = per_width(all_decodes, all_prefills, receipts, telemetry)
    worker_table = worker_axis_table(
        all_decodes, all_prefills, receipt_rows, workers, worker_minutes, telemetry_rows
    )
    if "16" not in table:
        raise ValueError("width 16 absent; cannot compare gate recommendation")
    aggregate_groups = defaultdict(list)
    request_groups = defaultdict(list)
    for row in all_decodes:
        if str(row["width"]) in table:
            aggregate_groups[row["width"]].append(row["rate"])
            request_groups[row["width"]].append(row["per_request_rate"])
    aggregate_curve = {w: med(values) for w, values in aggregate_groups.items()}
    request_curve = {w: med(values) for w, values in request_groups.items()}
    aggregate = aggregate_knee(aggregate_curve)
    request_break = piecewise_break(request_curve)
    thrash = onset(table)
    episode_rows, episode_context = episodes(
        all_decodes, all_prefills, thrash["baseline_hit"]
    )
    coincident = sum(row["prefill_surge"] for row in episode_rows)
    cache_drop_count = sum(row["cache_hit_drop"] for row in episode_rows)
    both_count = sum(
        row["prefill_surge"] and row["cache_hit_drop"] for row in episode_rows
    )
    surge_low_count = sum(
        row["prefill_surge"] and row["cache_hit_below_baseline"] for row in episode_rows
    )
    high_occupancy = [row for row in episode_rows if row["current"]["width"] >= 16]
    high_coincident = sum(row["prefill_surge"] for row in high_occupancy)
    high_both = sum(
        row["prefill_surge"] and row["cache_hit_drop"] for row in high_occupancy
    )
    high_surge_low = sum(
        row["prefill_surge"] and row["cache_hit_below_baseline"]
        for row in high_occupancy
    )
    aggregate_ci = bootstrap_width(aggregate_groups, aggregate_knee)
    request_ci = bootstrap_width(request_groups, piecewise_break)
    thrash["bootstrap"] = onset_bootstrap(all_prefills, table)
    # A lower gate needs all three independently observed boundaries below 16.
    # One broad aggregate plateau alone does not justify reducing capacity.
    lower_supported = (
        aggregate is not None
        and aggregate_ci is not None
        and aggregate_ci[1] < 16
        and request_break is not None
        and request_ci is not None
        and request_ci[1] < 16
        and thrash["width"] is not None
        and thrash["width"] <= 16
    )
    proposed = (
        min(aggregate, request_break["width"], thrash["width"] - 1)
        if lower_supported
        else 16
    )
    candidate = (
        aggregate if aggregate is not None and aggregate in aggregate_curve else 16
    )
    rate_samples = [
        row["request_occupancy"] / row["worker_occupancy"]
        for row in receipt_rows
        if row["start_epoch"] >= CHANGE_AT
        and row["worker_occupancy"] >= 3
        and 1 <= row["request_occupancy"] <= 16
    ]
    requests_per_worker = med(rate_samples)
    eligible_worker_counts = [
        int(count)
        for count, values in worker_table.items()
        if values["minutes"] >= 20
        and values["per_worker_effective_tok_s_median"] is not None
    ]
    prediction = {
        str(w): {
            "aggregate_tok_s": table[str(w)]["aggregate_tok_s_median"],
            "per_request_tok_s": table[str(w)]["per_request_tok_s_median"],
            "estimated_live_workers": round(w / requests_per_worker)
            if requests_per_worker
            else None,
        }
        for w in {candidate, proposed, 16}
    }
    for values in prediction.values():
        expected = values["estimated_live_workers"]
        observed_count = min(
            eligible_worker_counts, key=lambda count: abs(count - expected)
        )
        values["worker_bin_used"] = observed_count
        values["per_worker_effective_tok_s"] = worker_table[str(observed_count)][
            "per_worker_effective_tok_s_median"
        ]
    result = {
        "declared_thresholds": {
            "minimum_decode_steps_per_bin": MIN_STEPS,
            "bootstrap_resamples": RESAMPLES,
            "aggregate_within_peak_fraction": AGGREGATE_WITHIN_PEAK,
            "minimum_rise_for_aggregate_knee": MIN_RISE_FOR_KNEE,
            "cache_hit_drop_absolute": CACHE_DROP,
            "uncached_prefill_multiplier": UNCACHED_MULTIPLIER,
            "worker_idle_seconds": WORKER_IDLE_S,
            "fall_over_rate_drop_fraction": FALL_DROP,
            "fall_over_window_seconds": WINDOW_S,
            "lower_gate_requires_all_three_boundaries_below_current": True,
            "request_time_share_basis": (
                "distinct job-seconds with decode or prefill at width; a second can "
                "contribute to two widths during a transition"
            ),
            "uncached_prefill_rate_basis": (
                "new tokens at width divided by its distinct decode-or-prefill seconds"
            ),
        },
        "sources": sources,
        "serve_configs": configs,
        "positive_controls": {
            "decode_intervals": len(all_decodes),
            "prefill_batches": len(all_prefills),
            "completed_model_receipts": len(receipt_rows),
            "telemetry_samples": len(telemetry_rows),
            "receipt_timestamp": receipt_check,
        },
        "request_occupancy_bins": table,
        "worker_occupancy_bins": worker_table,
        "aggregate_knee": {
            "width": aggregate,
            "bootstrap_ci95": aggregate_ci,
            "definition": (
                "smallest median within 5% of maximum; at least 15% "
                "rise above first bin"
            ),
        },
        "per_worker_knee": {
            "piecewise_request_rate": request_break,
            "bootstrap_ci95": request_ci,
            "local_share_ratio_at_17": (
                request_curve[17] / (request_curve[16] * 16 / 17)
                if 16 in request_curve and 17 in request_curve
                else None
            ),
            "definition": (
                "two-line per-request fit; post-break slope steeper than "
                "sharing slope -rate/width"
            ),
        },
        "thrash_onset": thrash,
        "fall_over": {
            "episodes": episode_rows,
            "count": len(episode_rows),
            "prefill_surge_count": coincident,
            "cache_hit_drop_count": cache_drop_count,
            "prefill_surge_and_cache_hit_drop_count": both_count,
            "prefill_surge_and_low_cache_hit_count": surge_low_count,
            "at_or_above_gate_count": len(high_occupancy),
            "at_or_above_gate_prefill_surge_count": high_coincident,
            "at_or_above_gate_surge_and_cache_drop_count": high_both,
            "at_or_above_gate_surge_and_low_cache_hit_count": high_surge_low,
            "window_count": episode_context["windows"],
            "definition": (
                "adjacent 60-second windows: occupancy held/rising, mean logged-second "
                "decode rate drops >=25%, prior >=100 tok/s; prefill-only seconds "
                "carry zero decode; uncached surge >=1.5x prior"
            ),
        },
        "recommendation": {
            "gate_width": proposed,
            "current_gate_width": 16,
            "basis": (
                "lower gate only if aggregate knee upper CI, per-request break "
                "upper CI, and "
                "thrash onset all lie below 16; otherwise stay at 16"
            ),
            "candidate_aggregate_knee_width": candidate,
            "observed_requests_per_worker_post_change": requests_per_worker,
            "predicted_observed_medians": prediction,
            "causal_limit": (
                "occupancy bins pool different hours, contexts, and accept "
                "lengths; this is an observational recommendation"
            ),
        },
    }
    captions = {
        "occupancy-histogram": (
            "Share of seconds carrying decode or prefill by running-request width and "
            "all router minutes "
            "by live worker count, split at the 2026-10-05 14:33 UTC gate change. "
            "Worker activity includes 120 seconds after a completed request."
        ),
        "aggregate-throughput": (
            "Median SGLang decode generation rate at each exact running width, "
            "with percentile bootstrap 95% intervals. The vertical references "
            "mark the current gate of 16 and the first width within 5% of the "
            "observed maximum; workloads and serve hours differ across widths."
        ),
        "per-worker-throughput": (
            "Left: per-request decode rate from engine throughput divided by "
            "active requests. Right: completed tokens per worker per wall-clock "
            "second in router minutes, including tool-call idle time. Whiskers "
            "show percentile bootstrap intervals; "
            "the break requires a post-break slope steeper than proportional sharing."
        ),
        "prefill-thrash": (
            "Uncached prefill, cache-hit fraction, and KV-pool usage against exact "
            "request occupancy. Prefill is counted from SGLang batches and "
            "divided by seconds with decode or prefill at that width; joint onset "
            "requires a "
            "five-point cache-hit fall and a 1.5-fold uncached rise above widths 2–8."
        ),
        "fall-over-episodes": (
            "Representative 60-second fall-over episodes in the serve logs: "
            "decode rate falls at held or rising occupancy. Time zero marks "
            "the first lower-rate window; each column keeps the same job "
            "and shows uncached prefill and KV usage beside the rate."
        ),
    }
    result["captions"] = captions
    (ROOT / "occupancy_stats.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    plot_histogram(all_decodes, all_prefills, worker_minutes)
    plot_aggregate(table, aggregate)
    plot_worker(table, worker_table, request_break["width"] if request_break else None)
    plot_prefill(table, thrash)
    plot_episodes(episode_rows, episode_context["series"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", action="store_true")
    parser.add_argument("--measure", action="store_true")
    args = parser.parse_args()
    if args.control:
        control()
    if args.measure:
        result = analyse()
        print(
            json.dumps(
                {
                    "aggregate_knee": result["aggregate_knee"],
                    "per_worker_knee": result["per_worker_knee"],
                    "thrash_onset": result["thrash_onset"],
                    "fall_over": {
                        key: result["fall_over"][key]
                        for key in (
                            "count",
                            "prefill_surge_count",
                            "cache_hit_drop_count",
                            "prefill_surge_and_cache_hit_drop_count",
                            "prefill_surge_and_low_cache_hit_count",
                            "at_or_above_gate_count",
                            "at_or_above_gate_prefill_surge_count",
                            "at_or_above_gate_surge_and_cache_drop_count",
                            "at_or_above_gate_surge_and_low_cache_hit_count",
                        )
                    },
                    "recommendation": result["recommendation"],
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
