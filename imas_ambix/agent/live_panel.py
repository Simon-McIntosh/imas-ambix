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

A third block, ``callers``, says whose requests fill the gate. It reads the
tail of the router's per-request record beside the lane document, and is built
in :mod:`imas_ambix.agent.lane_callers`, which also builds the ``history``
strip and the ``outcomes`` line from the same rows. Those two reach back
further than the callers window, so the record is read once over the longest
of the three and every level is cut from that one pass.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from imas_ambix.agent.lane import availability_percent, classify_reading, lane_headroom
from imas_ambix.agent.lane_callers import record_blocks
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
#: fact from a lane that has published zeros. ``observed_seconds`` travels with
#: them: it is the span the router actually observed this process, and
#: ``requests_per_run`` divides by it rather than by ``window_seconds``, so
#: without it a reader cannot reconcile the ratio with the busy request-seconds
#: behind it. ``unkeyed_share`` likewise travels with them: it is the share of
#: busy request-seconds whose requests carry no usable run id, and the run-keyed
#: figures beside it are computed over the remainder, so without it those
#: figures describe an unstated fraction of the lane. ``worker_slots_basis``
#: travels with them too: it says whether the slots were measured or assumed at
#: one request per run, which a reader needs to know whether the figure is a
#: level or an upper bound.
WORKER_SLOT_FIELDS = (
    "live_runs",
    "observed_seconds",
    "requests_per_run",
    "worker_slots",
    "worker_slots_basis",
    "unkeyed_share",
)

#: The per-session share fields the lane block copies when the lane publishes
#: them. They answer a dispatcher's second question: not how many slots the lane
#: has, but how many belong to the session dispatching. ``sessions`` maps each
#: working session to its run count and its slot figure, and
#: ``new_session_worker_slots`` is the figure for a session the router has not
#: yet seen. They are copied beside the worker-slot fields rather than in place
#: of them, and only when the lane published them, so a lane that has not
#: published a share leaves it absent rather than null.
SESSION_SLOT_FIELDS = (
    "active_sessions",
    "fair_share",
    "borrow_reserve",
    "new_session_worker_slots",
    "sessions",
)

#: The engine's capacity figures the lane block copies when the lane publishes
#: them. They describe the pool the engine holds and the load it carries --
#: ``hicache_host`` carries ``used_fraction``, the share of the host-tier store
#: in use. Each is copied under the name the lane document publishes it by,
#: because a reader comparing the block against the document it came from must
#: find the same field named the same way. A lane that has not published one
#: leaves it absent rather than null, which is a different fact from a lane that
#: measured a zero.
ENGINE_CAPACITY_FIELDS = (
    "hicache_host",
    "mean_context",
    "concurrent_requests",
    "preemptions",
)

#: The engine counters differenced over the window. A run ends when the job
#: changes or any of these falls below its predecessor.
COUNTER_NAMES = ("generation_tokens", "prompt_tokens", "uncached_prompt_tokens")

#: Lines read from the tail of a receipt file when selecting the window. The
#: window is nine rows; this is generous enough that a burst of faster rows is
#: still covered, and small enough that a large file is never read whole.
TAIL_LINES = 1024

#: Lines read from a candidate's tail when choosing the newest receipt file.
#: The recorder appends one row per atomic write, so a file caught mid-append
#: holds at most one incomplete final line; eight puts the last parseable row
#: well inside the read even if a few trailing lines are unreadable.
SELECTION_TAIL_LINES = 8

#: How many trailing unreadable lines a receipt file may carry before it is
#: read as a file that stopped writing rows rather than as one caught
#: mid-write. The recorder appends one row per atomic write, so a single torn
#: final line is ordinary; two or more mean the newest samples were never
#: written, and the file speaks for an engine whose recorder is failing rather
#: than for the shelf the engine is serving now.
TORN_TAIL_LINES = 1

#: The components that name one serving engine. A recorder writes one file per
#: serve on the machine it runs on, so the host, the model the serve answers as
#: and the job holding it separate two serves from each other and a serve from
#: its own later runs. Every live figure belongs to one of these.
ENGINE_KEY_FIELDS = ("hostname", "served_name", "job_id")


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
    return _parse_rows(_tail_lines(path, limit))


def _parse_rows(lines: list[str]) -> list[dict[str, Any]]:
    """Every line that parses as a JSON object, in the order given."""
    rows: list[dict[str, Any]] = []
    for line in lines:
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
    """The raw receipt file whose last parseable row carries the latest stamp.

    A file caught mid-append holds an incomplete final line, so the choice is
    made on the last row that parses rather than on the final line: a writer
    part-way through a row must not make its file invisible while every row
    before it is intact. The tiers a receipts directory also holds are excluded
    by their own names, which
    :func:`~imas_ambix.agent.serving_receipts.tier_paths` derives from the raw
    files' suffixes. Ties are broken by path name so the choice is
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
        rows = _read_rows(candidate, limit=SELECTION_TAIL_LINES)
        if not rows:
            continue
        stamp = _parse_stamp(rows[-1].get("timestamp"))
        if stamp is None:
            continue
        if chosen is None or (stamp, candidate.name) > (chosen[0], chosen[1]):
            chosen = (stamp, candidate.name, candidate)
    return chosen[2] if chosen is not None else None


def engine_identity(row: dict[str, Any]) -> tuple[Any, Any, Any]:
    """The serve one receipt row belongs to: host, served model and job.

    A serve is named by all three, because any one of them can move without the
    others: a second model joins the same host, a serve restarts under a new
    job, and the distinct names keep each engine's figures its own.
    """
    return tuple(row.get(name) for name in ENGINE_KEY_FIELDS)


def _trailing_breaks(lines: list[str]) -> int:
    """Unreadable non-blank lines after the last one that parses."""
    breaks = 0
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            json.loads(line)
        except json.JSONDecodeError:
            breaks += 1
            continue
        break
    return breaks


def receipt_serves(
    directory: str | Path,
) -> list[tuple[tuple[Any, Any, Any], Path, datetime | None, int]]:
    """One raw receipt file per engine, newest first.

    Each engine is the ``(hostname, served model, job)`` triple its newest raw
    row names, and the file chosen for it is the one whose last *parseable* row
    carries the latest stamp -- a file caught mid-append must not disappear
    while every row before the torn one is intact. The triple comes from that
    row rather than from the file's name, because a name carries no host and a
    restarted serve is a different engine under a new job. Ties break on path
    name so the choice is deterministic.

    Each entry carries the chosen file, the stamp that chose it, and how many
    trailing lines of it do not parse, which is what tells a row caught
    mid-write from a file whose newest samples were never written.
    """
    base = Path(directory)
    chosen: dict[tuple[Any, Any, Any], tuple[datetime, str, Path, int]] = {}
    for candidate in _raw_receipt_files(base):
        tail = _tail_lines(candidate, SELECTION_TAIL_LINES)
        rows = _parse_rows(tail)
        if not rows:
            continue
        stamp = _parse_stamp(rows[-1].get("timestamp"))
        if stamp is None:
            continue
        identity = engine_identity(rows[-1])
        if all(part in (None, "") for part in identity):
            # A row that names no serve cannot be attributed to an engine, and
            # guessing one would publish another engine's counters under it.
            continue
        breaks = _trailing_breaks(tail)
        current = chosen.get(identity)
        if current is None or (stamp, candidate.name) > (current[0], current[1]):
            chosen[identity] = (stamp, candidate.name, candidate, breaks)
    ordered = sorted(
        chosen.items(), key=lambda item: (item[1][0], item[1][1]), reverse=True
    )
    return [
        (identity, entry[2], entry[0], entry[3]) for identity, entry in ordered
    ]


def _raw_receipt_files(base: Path) -> list[Path]:
    """Every raw receipt file under *base*, the compacted tiers excluded."""
    if not base.is_dir():
        return []
    tiers = _tier_names(base)
    return [candidate for candidate in base.glob("*.jsonl") if candidate not in tiers]


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

    The histogram is cumulative since the engine started: each bucket counts
    every request at or below its upper bound, so the window's total is the
    advance of the topmost bucket, not the sum of the bucket advances. The
    median is interpolated inside the bucket whose cumulative advance first
    reaches half that total; when that bucket is the open tail the mean of the
    advance is returned instead, because an infinite bound is not a median.
    """
    first = _histogram(rows[0])
    last = _histogram(rows[-1])
    if first is None or last is None:
        return None
    keys = sorted({*first["buckets"], *last["buckets"]}, key=_bound)
    if not keys:
        return None
    advances: list[tuple[float, float]] = [
        (
            _bound(key),
            last["buckets"].get(key, 0.0) - first["buckets"].get(key, 0.0),
        )
        for key in keys
    ]
    total = advances[-1][1]
    if total <= 0:
        return None
    target = total / 2.0
    lower_bound = 0.0
    lower_count = 0.0
    for bound, count in advances:
        if math.isinf(bound):
            count_delta = last["count"] - first["count"]
            return (
                (last["total"] - first["total"]) / count_delta
                if count_delta > 0
                else None
            )
        if count >= target:
            return lower_bound + (target - lower_count) / (count - lower_count) * (
                bound - lower_bound
            )
        lower_bound = bound
        lower_count = count
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
    receipts_dir: str | Path,
    *,
    now: datetime | None = None,
    shelf_life_seconds: int = DEFAULT_SHELF_LIFE_SECONDS,
) -> dict[str, Any]:
    """The ``live`` block: per-stream rates over the newest raw receipt tail.

    The block is the newest engine's entry, with every engine's entry beneath it
    under ``engines``. A reader of the top-level figures sees the engine that
    last wrote a row, which is what a single-serve lane has always shown; a
    reader watching a lane of several serves reads each one's own counters out
    of the list rather than taking one serve for the whole machine.
    """
    entries = live_entries(
        receipts_dir, now=now, shelf_life_seconds=shelf_life_seconds
    )
    block = dict(entries[0]) if entries else _single_block(receipts_dir, now=now)
    block["engines"] = entries
    return block


def live_entries(
    receipts_dir: str | Path,
    *,
    now: datetime | None = None,
    shelf_life_seconds: int = DEFAULT_SHELF_LIFE_SECONDS,
) -> list[dict[str, Any]]:
    """One live entry per engine, newest first, and none for a silent engine.

    Each engine is the serve its own newest raw receipt names, so a lane of two
    serves yields two entries with their own counters rather than one entry
    chosen from whichever file happened to be written last. An engine whose
    newest readable row is older than the shelf life yields no entry at all: the
    shelf life is how long a reading stands for the serve's present state, and a
    row past it describes a serve that may no longer be there.

    An engine whose file has stopped carrying readable rows keeps an entry and
    reports itself unavailable with the reason. The recorder appends one row per
    atomic write, so one torn final line is a file caught mid-write; more than
    that means the newest samples were never written, and silence about the
    engine would read as a serve that is not running rather than as a recorder
    that cannot say.
    """
    moment = _now(now)
    entries: list[dict[str, Any]] = []
    for identity, path, stamp, breaks in receipt_serves(receipts_dir):
        age = max(0.0, (moment - stamp).total_seconds()) if stamp else None
        if age is not None and age > shelf_life_seconds:
            continue
        if breaks > TORN_TAIL_LINES:
            block = _figures()
            block.update(
                {
                    "file": path.name,
                    "state": "unavailable",
                    "reason": (
                        f"{breaks} trailing lines of the receipt file do not "
                        "parse, so its newest samples were never written"
                    ),
                    "tail_breaks": breaks,
                }
            )
        else:
            block = _file_block(path, moment)
        block.update(dict(zip(ENGINE_KEY_FIELDS, identity, strict=True)))
        block["age_seconds"] = age
        block["shelf_life_seconds"] = float(shelf_life_seconds)
        entries.append(block)
    return entries


def _figures() -> dict[str, Any]:
    """The live block's fields, carrying no measurement."""
    return {
        "file": None,
        "job_id": None,
        "served_name": None,
        "hostname": None,
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


def _single_block(
    receipts_dir: str | Path, *, now: datetime | None = None
) -> dict[str, Any]:
    """The newest raw receipt's figures, whether or not it is inside the shelf life.

    This is the whole block when no engine has written within the shelf life, so
    a reader is told *why* there is no live entry -- no file at all, no readable
    row, or no row in the trailing window -- rather than a bare absence.
    """
    chosen = newest_raw_receipt(receipts_dir)
    if chosen is None:
        block = _figures()
        block["reason"] = "no raw receipt file with a readable row"
        return block
    return _file_block(chosen, _now(now))


def _file_block(path: Path, moment: datetime) -> dict[str, Any]:
    """The live figures one receipt file supports."""
    block = _figures()
    block["file"] = path.name
    all_rows = _read_rows(path)
    if not all_rows:
        block["reason"] = "the chosen receipt file holds no readable row"
        return block
    last_row = all_rows[-1]
    block["job_id"] = last_row.get("job_id")
    block["served_name"] = last_row.get("served_name")
    block["hostname"] = last_row.get("hostname")
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
            key: admission.get(key)
            for key in (*WORKER_SLOT_FIELDS, *SESSION_SLOT_FIELDS)
            if key in admission
        }
        if slots:
            block["worker_slots"] = slots
    gate = document.get("router_generation_gate")
    if isinstance(gate, dict):
        block["router_generation_gate"] = {key: gate.get(key) for key in GATE_FIELDS}
    for key in ENGINE_CAPACITY_FIELDS:
        if key in document:
            block[key] = document[key]
    # Copy the published percentage, or derive it with the same functions the
    # router published it with when the document predates the field. A reader
    # sees the figure before the router is next restarted; an omitted figure
    # stays omitted rather than becoming a null read as zero. The headroom the
    # percentage divides is the shared gate-bounded selection, so a document
    # written before the rule -- one that withheld its sizing and published no
    # top-level headroom -- still derives the figure the router would publish.
    availability = document.get("availability_percent")
    if availability is None and isinstance(gate, dict):
        availability = availability_percent(
            lane_headroom(document),
            gate.get("effective_width"),
            paused=gate.get("paused") is True,
        )
    if availability is not None:
        block["availability_percent"] = availability
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


def panel_engines(
    lane_document: dict[str, Any], live_entries: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """One entry per engine, its live reading joined to its own gate figures.

    The live block counts one entry per serve it read. The pool document the
    router publishes carries, when it spans several engines, an ``engines``
    array holding each engine's own reading and gate. This joins the two by
    engine identity -- the registration id's host and port -- so an entry
    carries both what the engine did (from the recorder) and what its gate
    holds (from the router). An engine the pool document names but the recorder
    has no fresh row for keeps an entry with no live figures, and one the
    recorder saw but the pool document does not name keeps its live figures
    with no gate, so neither is silently dropped.

    A receipt row names its host and model but carries no port, so two serves
    on one node cannot be told apart by port from the record alone. They are
    told apart by the model each serves, and a pair that neither identity nor
    model separates is left unjoined rather than paired by list position: two
    serves on one node, reported in the order the recorder happened to write
    them, would otherwise each carry the other's figures and read as though
    they had been measured.
    """
    pool = [
        entry for entry in (lane_document.get("engines") or [])
        if isinstance(entry, dict)
    ]
    used: set[int] = set()
    entries: list[dict[str, Any]] = []
    # A pool document that spans one engine is written from that engine's own
    # reading and carries no engine id, so the model is the only key it offers.
    # Both sides are counted first: the model joins such an entry only when one
    # pool entry and one live reading carry it, which is a fact rather than a
    # guess.
    live_models: Counter[str] = Counter(
        str(live.get("served_name")) for live in live_entries
        if live.get("served_name")
    )
    unnamed_models: Counter[str] = Counter(
        model for entry in pool
        if (model := _pool_model(entry)) and _pool_identity(entry)[0] is None
    )
    for live in live_entries:
        match = _matching_engine(pool, used, live, live_models, unnamed_models)
        if match is not None:
            used.add(match)
        entries.append(_engine_entry(pool[match] if match is not None
                                     else {}, live))
    for index, engine in enumerate(pool):
        if index not in used:
            entries.append(_engine_entry(engine, None))
    return entries


def _matching_engine(
    pool: list[dict[str, Any]],
    used: set[int],
    live: dict[str, Any],
    live_models: Counter[str],
    unnamed_models: Counter[str],
) -> int | None:
    """The pool entry a live reading belongs to, or None when nothing tells.

    Identity is ``host:port``, which the pool entry carries and the receipt row
    does not. So the host narrows the pool to the engines on that node and the
    served model separates them when the host carries more than one engine. A
    host with two engines serving two models resolves on the model; a host with
    two engines serving one model has nothing left to separate them and yields
    no match, because pairing them by position is what gives an engine another
    engine's figures.

    The model is also a cross-check on a single-candidate match: a host whose
    only pool engine serves a different model is a reading the pool does not
    describe, and it is left unjoined rather than pinned to the wrong engine.

    A pool document spanning one engine carries no engine id at all -- the
    router composes it from that engine's own reading -- so it arrives with no
    host. That engine is joined by its model, and only when the model is unique
    on both sides: a fact about the record rather than a guess about order.
    """
    host = live.get("hostname")
    model = live.get("served_name")
    if isinstance(host, str) and host:
        index = _match_on_host(pool, used, _host_name(host), model)
        if index is not None:
            return index
    return _match_by_model_only(pool, used, model, live_models, unnamed_models)


def _match_on_host(
    pool: list[dict[str, Any]], used: set[int], host: str, model: Any
) -> int | None:
    """The entry on *host*, separated from its neighbours by *model*."""
    on_host = [
        index for index, engine in enumerate(pool)
        if index not in used and _pool_identity(engine)[0] == host
    ]
    if not on_host:
        return None
    if len(on_host) > 1:
        named = [
            index for index in on_host
            if model and _pool_model(pool[index]) == model
        ]
        if len(named) != 1:
            return None
        on_host = named
    index = on_host[0]
    entry_model = _pool_model(pool[index])
    if model and entry_model and entry_model != model:
        return None
    return index


def _match_by_model_only(
    pool: list[dict[str, Any]],
    used: set[int],
    model: Any,
    live_models: Counter[str],
    unnamed_models: Counter[str],
) -> int | None:
    """The one host-less entry serving *model*, when the model names it alone."""
    if not model:
        return None
    name = str(model)
    if live_models.get(name) != 1 or unnamed_models.get(name) != 1:
        return None
    for index, engine in enumerate(pool):
        if index in used or _pool_identity(engine)[0] is not None:
            continue
        if _pool_model(engine) == name:
            return index
    return None


def _pool_identity(engine: dict[str, Any]) -> tuple[str | None, str | None]:
    """A pool entry's ``(host, port)``, from its engine id where it has one."""
    engine_id = engine.get("engine_id")
    if isinstance(engine_id, str) and engine_id:
        host, _, port = engine_id.rpartition(":")
        if host:
            return (_host_name(host), port or None)
        return (_host_name(engine_id), None)
    host = engine.get("hostname") or engine.get("host")
    return (_host_name(host) if isinstance(host, str) and host else None, None)


def _pool_model(engine: dict[str, Any]) -> str | None:
    """The model a pool entry serves, under either name a pool gives it."""
    model = engine.get("model_id") or engine.get("model")
    return str(model) if model else None


def _host_name(host: str) -> str:
    """A host reduced to the node name the registry's engine ids carry."""
    from imas_ambix.agent.registry import _host_name as registry_host_name

    return registry_host_name(host)


def _engine_host(engine_id: str) -> str:
    """The node name out of an engine id, which is ``<host>:<port>``."""
    return _host_name(engine_id.rsplit(":", 1)[0]) if ":" in engine_id else _host_name(
        engine_id
    )


def _engine_entry(
    gate: dict[str, Any], live: dict[str, Any] | None
) -> dict[str, Any]:
    """One engine's row: its live figures and the gate reading beside them."""
    gate_block = gate.get("router_generation_gate")
    gate_block = gate_block if isinstance(gate_block, dict) else {}
    reading = live if isinstance(live, dict) else {}
    engine_id = gate.get("engine_id")
    host = reading.get("hostname") or (
        engine_id.rsplit(":", 1)[0] if isinstance(engine_id, str) else None
    )
    model = reading.get("served_name") or gate.get("model_id")
    available = gate.get("availability_percent")
    return {
        "engine": engine_id if isinstance(engine_id, str) and engine_id else host,
        "model": model,
        "host": host,
        "engine_id": engine_id,
        "live": reading,
        "gate": gate,
        "running": reading.get("mean_running"),
        "router_q": gate_block.get("waiting"),
        "tok_s": reading.get("generation_toks_per_s"),
        "per_stream": reading.get("per_stream_toks_per_s"),
        "ttft": reading.get("median_time_to_first_token_s"),
        "kv_pool": (
            reading.get("pool_occupancy")
            if reading.get("pool_occupancy") is not None
            else gate.get("kv_occupancy")
        ),
        "available": available,
    }


def live_panel(
    *,
    receipts_dir: str | Path | None = None,
    lane_path: str | Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Every live block, resolved from the site when no path is supplied.

    A caller on the tail of the watch document supplies the paths it already
    has; a standalone caller lets the site configuration resolve them, which is
    the same resolution the recorder writes through. The request-record blocks
    -- ``callers``, ``history`` and ``outcomes`` -- come from one bounded read
    of the record beside the lane document.
    """
    if receipts_dir is None:
        from imas_ambix.agent.watch import default_record_dir

        receipts_dir = default_record_dir()
    if lane_path is None:
        from imas_ambix.agent.profile import SiteConfig

        lane_path = SiteConfig.from_env().endpoint_document.with_name(LANE_FILENAME)
    # The router writes its request record beside the lane document, so the
    # caller breakdown, the history strip and the outcomes line all read the
    # same directory the lane block came from -- and read it once between them,
    # because each poll of this document would otherwise walk the same growing
    # file three times. All three hang off ``callers``, the block this document
    # already gives to the request record, so the document's own top-level keys
    # stay the three the live path has always published.
    blocks = record_blocks(Path(lane_path).parent, now=now)
    callers = dict(blocks["callers"])
    callers["history"] = blocks["history"]
    callers["outcomes"] = blocks["outcomes"]
    lane_document = _lane_document(lane_path, now=now)
    live = live_block(
        receipts_dir,
        now=now,
        shelf_life_seconds=published_shelf_life(lane_path),
    )
    document = {
        "lane": lane_block(lane_document, now=now),
        "live": live,
        "callers": callers,
    }
    # The engine dimension appears only when there is more than one engine, so
    # a one-engine lane's document carries exactly the keys it carried before
    # it existed. That is what makes the dimension cost a single-engine reader
    # nothing: its document and its render are unchanged.
    engines = panel_engines(lane_document, live.get("engines") or [])
    if len(engines) > 1:
        document["engines"] = engines
    return document


def _lane_document(path: str | Path, *, now: datetime | None = None) -> dict[str, Any]:
    """The published lane document, with every pool's engines gathered into it.

    The lane block copies only the fields it publishes; the ``engines`` array
    the pool document carries is read here, from the same file and the same
    pass, so the per-engine entries describe the document the lane block was
    built from.

    The router composes one document per model, though, and a document carries
    an ``engines`` array only when its own pool spans several engines. Every
    other model's pool is written beside the lane document under ``lanes/`` and
    named by the ``lanes`` index the lane document itself carries, so reading
    the lane document alone loses a second model entirely. Each named document
    therefore contributes its engines too, and a pool document that spans a
    single engine contributes itself, because that is what a one-engine pool's
    document is.
    """
    target = Path(path)
    document = _read_document(target)
    index = document.get("lanes")
    # One document per model, so an index naming more than one document is a
    # lane serving more than one pool -- and this document is then the default
    # pool's own, whatever it spans. A lane carrying a single pool keeps the
    # document it has always had.
    several = isinstance(index, dict) and len(
        {name for name in index.values() if isinstance(name, str) and name}
    ) > 1
    engines = _document_engines(document, standalone=several)
    if isinstance(index, dict):
        for model, name in index.items():
            if not isinstance(name, str) or not name:
                continue
            other = target.parent / name
            if other == target:
                continue
            document_of_other = _read_document(other)
            if isinstance(model, str) and model and not document_of_other.get(
                "model_id"
            ):
                # The index is keyed by model, so a document that does not name
                # its own still has one, and the entry needs it to be joined.
                document_of_other = {**document_of_other, "model_id": model}
            engines.extend(
                _document_engines(document_of_other, standalone=True)
            )
    if not engines:
        return document
    merged = dict(document)
    merged["engines"] = engines
    return merged


def _read_document(path: Path) -> dict[str, Any]:
    """One published document, or an empty one when it is unreadable."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {"state": "unavailable"}
    return document if isinstance(document, dict) else {"state": "unavailable"}


def _document_engines(
    document: dict[str, Any], *, standalone: bool
) -> list[dict[str, Any]]:
    """A pool document's engine entries.

    A pool spanning several engines lists them under ``engines``. A pool
    spanning one is that engine's own document and carries no list, so it is an
    entry only where the document is known to stand for one engine on its own:
    a pool named by a ``lanes`` index that names several is, and so is the lane
    document when it is one of those pools. A lane document holding the only
    pool is not, because it is then the whole lane and the recorder already
    names the engine it serves.
    """
    entries = [entry for entry in (document.get("engines") or [])
               if isinstance(entry, dict)]
    if entries:
        return entries
    if standalone and document.get("state") != "unavailable":
        return [document]
    return []


def published_shelf_life(path: str | Path) -> int:
    """The shelf life the lane document publishes, or the default.

    The lane document is the one place a deployment declares how long its
    readings stand for, and the live block answers to the same figure the lane
    block does -- a receipt file is chosen for an engine by the same rule that
    decides whether the lane itself is stale.
    """
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return DEFAULT_SHELF_LIFE_SECONDS
    published = document.get("suggested_shelf_life_seconds") if isinstance(
        document, dict
    ) else None
    if isinstance(published, int | float) and not isinstance(published, bool):
        return int(published)
    return DEFAULT_SHELF_LIFE_SECONDS
