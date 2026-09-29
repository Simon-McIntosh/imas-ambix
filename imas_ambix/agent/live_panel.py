"""Live lane and receipt blocks for the watch document, with no index read.

Two published sources already carry every live serving figure, and neither
needs the telemetry index the full ``agent watch`` document builds. The lane
document the router rewrites every 30 s says what the shared lane is offering
and what the generation gate is holding; the recorder's raw receipt file is
appended once per serve, one row every 5 s, and its counters describe what the
engine actually did over a short trailing window.

This module turns those two into the ``lane`` and ``live`` blocks. It reads
only the lane document and the tail of one receipt file, so a caller can run it
every five seconds while the index work is unfinished. Nothing here imports
:mod:`imas_ambix.agent.telemetry_index`, and nothing here constructs an index.

The lane block copies the document's fields rather than recomputing them, and
preserves their presence: a withheld ``headroom`` stays withheld rather than
becoming a null a reader would take for zero, and ``headroom_is_upper_bound``
travels with the figure it qualifies, because a copy that drops it reports a
bound as a value.

The live block differences the receipt counters over a trailing window. A
window that does not hold two rows of one run cannot be differenced, and every
rate is then ``None`` with a stated reason rather than a zero -- a zero would be
a measurement claim the record cannot support.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from imas_ambix.agent.lane import classify_reading
from imas_ambix.agent.serving_receipts import tier_paths

#: The trailing window the ``live`` block describes. The recorder appends one
#: row per serve every 5 s, so 45 s is nine rows: long enough for a differenced
#: rate to mean something and short enough to describe the serve as it is now.
#: This is not :data:`imas_ambix.agent.watch.SAMPLE_SPEAKS_FOR`, which bounds
#: how long a stored row may stand in for a gap in the record; this window is
#: the fresh sample the live block is computed over.
WINDOW_SECONDS = 45.0

#: The shelf life assumed when a lane document carries none of its own.
DEFAULT_SHELF_LIFE_SECONDS = 120

#: The lane document's own name, placed beside the endpoint document the same
#: way every existing reader derives it. A caller has one spelling to reuse.
LANE_FILENAME = "lane.json"

#: The router generation-gate fields the lane block copies. ``width`` is the
#: configured width and ``effective_width`` the width in force; both are copied
#: so a reader that wants the width the gate is actually enforcing can see it.
GATE_FIELDS = (
    "width",
    "effective_width",
    "in_flight",
    "waiting",
    "paused",
    "width_mode",
    "reason",
)

#: The admission fields the lane block copies. These four are the queue's
#: demand signal; the verdict and the oldest wait are published nowhere else.
ADMISSION_FIELDS = ("headroom", "verdict", "waiting", "oldest_wait_seconds")

#: The worker-slot fields the lane block copies when the lane publishes them.
#: A lane that has not published them leaves them absent, which is a different
#: fact from a lane that has published zeros.
WORKER_SLOT_FIELDS = ("live_runs", "requests_per_run", "worker_slots")

#: The engine counters differenced over the window. A run ends when the job
#: changes or any of these falls below its predecessor.
COUNTER_NAMES = ("generation_tokens", "prompt_tokens", "uncached_prompt_tokens")

#: Lines read from the tail of a receipt file when selecting the window. The
#: window is nine rows; this is generous enough that a burst of faster rows is
#: still covered, and small enough that a large file is never read whole.
TAIL_LINES = 1024


def _now(now: datetime | None) -> datetime:
    return now if now is not None else datetime.now(UTC)


def _parse_stamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _tail_lines(path: Path, limit: int) -> list[str]:
    """The last *limit* lines of *path*, oldest first."""
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            block = b""
            position = size
            chunk = 65536
            while position > 0 and block.count(b"\n") <= limit:
                step = min(chunk, position)
                position -= step
                handle.seek(position)
                block = handle.read(step) + block
    except OSError:
        return []
    lines = block.decode("utf-8", errors="replace").splitlines()
    return lines[-limit:] if limit < len(lines) else lines


def _read_rows(path: Path, limit: int = TAIL_LINES) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in _tail_lines(path, limit):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _tier_names(directory: Path) -> set[Path]:
    """Every raw file's minute and hour sibling in *directory*.

    A file whose name is one of these is a compacted tier rather than a raw
    receipt file, and the ``live`` block reads only raw files.
    """
    names: set[Path] = set()
    for candidate in directory.glob("*.jsonl"):
        names.update(tier_paths(candidate))
    return names


def newest_raw_receipt(directory: str | Path) -> Path | None:
    """The raw receipt file whose last row carries the latest timestamp.

    The tiers a receipts directory also holds are excluded by their names,
    which :func:`~imas_ambix.agent.serving_receipts.tier_paths` derives from the
    raw files' own suffixes. Ties are broken by path name so the choice is
    deterministic rather than dependent on directory order.
    """
    base = Path(directory)
    if not base.is_dir():
        return None
    tiers = _tier_names(base)
    chosen: tuple[datetime, str, Path] | None = None
    for candidate in base.glob("*.jsonl"):
        if candidate in tiers:
            continue
        rows = _read_rows(candidate, limit=1)
        if not rows:
            continue
        stamp = _parse_stamp(rows[-1].get("timestamp"))
        if stamp is None:
            continue
        if chosen is None or (stamp, candidate.name) > (chosen[0], chosen[1]):
            chosen = (stamp, candidate.name, candidate)
    return chosen[2] if chosen is not None else None


def _counter(row: dict[str, Any], name: str) -> float | None:
    engine = row.get("engine")
    if not isinstance(engine, dict):
        return None
    value = engine.get(name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _gauge(row: dict[str, Any], name: str) -> float | None:
    value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _present(values: list[float | None]) -> list[float]:
    return [value for value in values if value is not None]


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _advance(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return values[-1] - values[0]


def _monotonic(rows: list[dict[str, Any]], name: str) -> bool | None:
    """Whether a counter never falls below its predecessor across *rows*.

    ``None`` when fewer than two rows carry the counter: absence is not a
    violation, it is an untakeable difference.
    """
    values = _present([_counter(row, name) for row in rows])
    if len(values) < 2:
        return None
    return all(
        later >= earlier for earlier, later in zip(values, values[1:], strict=False)
    )


def _histogram(row: dict[str, Any]) -> dict[str, Any] | None:
    engine = row.get("engine")
    if not isinstance(engine, dict):
        return None
    histograms = engine.get("histograms")
    if not isinstance(histograms, dict):
        return None
    ttft = histograms.get("time_to_first_token")
    if not isinstance(ttft, dict):
        return None
    buckets = ttft.get("buckets")
    if not isinstance(buckets, dict):
        return None
    return {
        "buckets": {
            str(key): float(value)
            for key, value in buckets.items()
            if isinstance(value, int | float) and not isinstance(value, bool)
        },
        "count": float(ttft.get("count") or 0.0),
        "total": float(ttft.get("sum") or 0.0),
    }


def _bound(key: str) -> float:
    return math.inf if key == "inf" else float(key)


def _median_from_histogram(rows: list[dict[str, Any]]) -> float | None:
    """The median of the window's time-to-first-token histogram advance.

    The histogram is cumulative since the engine started, so the window's
    distribution is the difference of the last and first rows' buckets. The
    median is the upper bound of the bucket whose cumulative advance first
    reaches half the total; when that bucket is the open tail the mean of the
    advance is returned instead, because an infinite bound is not a median.
    """
    first = _histogram(rows[0])
    last = _histogram(rows[-1])
    if first is None or last is None:
        return None
    keys = sorted({*first["buckets"], *last["buckets"]}, key=_bound)
    advances: list[tuple[float, float]] = []
    for key in keys:
        delta = last["buckets"].get(key, 0.0) - first["buckets"].get(key, 0.0)
        if delta > 0:
            advances.append((_bound(key), delta))
    total = sum(value for _, value in advances)
    if total <= 0:
        return None
    target = total / 2.0
    cumulative = 0.0
    for bound, value in advances:
        cumulative += value
        if cumulative >= target:
            if math.isinf(bound):
                count = last["count"] - first["count"]
                return (last["total"] - first["total"]) / count if count > 0 else None
            return bound
    return None


def _window_rows(rows: list[dict[str, Any]], now: datetime) -> list[dict[str, Any]]:
    floor = now.timestamp() - WINDOW_SECONDS
    ceiling = now.timestamp()
    selected: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        stamp = _parse_stamp(row.get("timestamp"))
        if stamp is None:
            continue
        epoch = stamp.timestamp()
        if floor <= epoch <= ceiling:
            selected.append((epoch, row))
    selected.sort(key=lambda item: item[0])
    return [row for _, row in selected]


def live_block(
    receipts_dir: str | Path, *, now: datetime | None = None
) -> dict[str, Any]:
    """The ``live`` block: per-stream rates over the newest raw receipt tail."""
    moment = _now(now)
    block: dict[str, Any] = {
        "file": None,
        "job_id": None,
        "served_name": None,
        "window_seconds": WINDOW_SECONDS,
        "rows": 0,
        "state": "unavailable",
        "mean_running": None,
        "engine_queue": None,
        "pool_occupancy": None,
        "generation_toks_per_s": None,
        "per_stream_toks_per_s": None,
        "prefill_toks_per_s": None,
        "cache_hit_rate": None,
        "median_time_to_first_token_s": None,
        "reason": None,
    }
    chosen = newest_raw_receipt(receipts_dir)
    if chosen is None:
        block["reason"] = "no raw receipt file with a readable row"
        return block
    block["file"] = chosen.name
    all_rows = _read_rows(chosen)
    if not all_rows:
        block["reason"] = "the chosen receipt file holds no readable row"
        return block
    last_row = all_rows[-1]
    block["job_id"] = last_row.get("job_id")
    block["served_name"] = last_row.get("served_name")
    rows = _window_rows(all_rows, moment)
    block["rows"] = len(rows)
    if not rows:
        block["reason"] = "no rows in the trailing window"
        return block
    block["state"] = "measured"
    block["mean_running"] = _mean(
        _present([_gauge(row, "num_requests_running") for row in rows])
    )
    block["engine_queue"] = _gauge(rows[-1], "num_requests_waiting")
    block["pool_occupancy"] = _gauge(rows[-1], "kv_cache_usage_perc")
    if len(rows) < 2:
        block["reason"] = "fewer than two rows in the window"
        return block
    jobs = {row.get("job_id") for row in rows}
    monotonic = {name: _monotonic(rows, name) for name in COUNTER_NAMES}
    if len(jobs) > 1 or any(flag is False for flag in monotonic.values()):
        block["reason"] = "the window spans more than one run"
        return block
    first_at = _parse_stamp(rows[0].get("timestamp"))
    last_at = _parse_stamp(rows[-1].get("timestamp"))
    elapsed = (
        (last_at - first_at).total_seconds()
        if first_at is not None and last_at is not None
        else 0.0
    )
    if elapsed <= 0:
        block["reason"] = "the window's rows carry no positive elapsed time"
        return block
    generation = _advance(
        _present([_counter(row, "generation_tokens") for row in rows])
    )
    prompt = _advance(_present([_counter(row, "prompt_tokens") for row in rows]))
    uncached = _advance(
        _present([_counter(row, "uncached_prompt_tokens") for row in rows])
    )
    if generation is not None:
        aggregate = generation / elapsed
        block["generation_toks_per_s"] = aggregate
        if block["mean_running"]:
            block["per_stream_toks_per_s"] = aggregate / block["mean_running"]
    if uncached is not None:
        block["prefill_toks_per_s"] = uncached / elapsed
    if prompt is not None and uncached is not None and prompt > 0:
        block["cache_hit_rate"] = (prompt - uncached) / prompt
    block["median_time_to_first_token_s"] = _median_from_histogram(rows)
    return block


def lane_block(
    document: dict[str, Any], *, now: datetime | None = None
) -> dict[str, Any]:
    """The ``lane`` block: a faithful copy of the published lane reading."""
    moment = _now(now)
    block: dict[str, Any] = {
        "observed_at": document.get("observed_at"),
        "state": "unavailable",
        "stale": None,
        "age_seconds": None,
        "suggested_shelf_life_seconds": document.get("suggested_shelf_life_seconds"),
    }
    observed = _parse_stamp(document.get("observed_at"))
    if observed is not None:
        block["age_seconds"] = max(0.0, (moment - observed).total_seconds())
    shelf_life = DEFAULT_SHELF_LIFE_SECONDS
    published = document.get("suggested_shelf_life_seconds")
    if isinstance(published, int | float) and not isinstance(published, bool):
        shelf_life = int(published)
    state = classify_reading(document, now=moment, shelf_life_seconds=shelf_life)
    block["state"] = state
    block["stale"] = state == "stale"
    if "headroom" in document:
        block["headroom"] = document["headroom"]
    if "engine_headroom" in document:
        block["engine_headroom"] = document["engine_headroom"]
    if "headroom_is_upper_bound" in document:
        block["headroom_is_upper_bound"] = document["headroom_is_upper_bound"]
    # A withheld figure is passed through as the document published it, so a
    # reader sees the refusal rather than a null it would read as zero.
    if "withheld" in document:
        block["withheld"] = document["withheld"]
    admission = document.get("admission")
    if isinstance(admission, dict):
        block["admission"] = {key: admission.get(key) for key in ADMISSION_FIELDS}
        slots = {
            key: admission.get(key) for key in WORKER_SLOT_FIELDS if key in admission
        }
        if slots:
            block["worker_slots"] = slots
    gate = document.get("router_generation_gate")
    if isinstance(gate, dict):
        block["router_generation_gate"] = {key: gate.get(key) for key in GATE_FIELDS}
    return block


def read_lane_block(path: str | Path, *, now: datetime | None = None) -> dict[str, Any]:
    """Read the lane document at *path* and return its ``lane`` block."""
    target = Path(path)
    try:
        document = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return lane_block({"state": "unavailable"}, now=now)
    if not isinstance(document, dict):
        return lane_block({"state": "unavailable"}, now=now)
    return lane_block(document, now=now)


def live_panel(
    *,
    receipts_dir: str | Path | None = None,
    lane_path: str | Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Both live blocks, resolved from the site when no path is supplied.

    A caller on the tail of the watch document supplies the paths it already
    has; a standalone caller lets the site configuration resolve them, which is
    the same resolution the recorder writes through.
    """
    if receipts_dir is None:
        from imas_ambix.agent.watch import default_record_dir

        receipts_dir = default_record_dir()
    if lane_path is None:
        from imas_ambix.agent.profile import SiteConfig

        lane_path = SiteConfig.from_env().endpoint_document.with_name(LANE_FILENAME)
    return {
        "lane": read_lane_block(lane_path, now=now),
        "live": live_block(receipts_dir, now=now),
    }
