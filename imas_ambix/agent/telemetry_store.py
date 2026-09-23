"""Tiered rotation and compaction for the serving telemetry record.

The recorder appends one JSON row per five-second tick to an append-only,
lock-free JSONL file. That is the right durable shape on a network filesystem,
but it is not a retention policy: at the measured rate a job grows 8.7 MB per
day and nothing ever rolls. This module compacts that record down through two
coarser resolutions so the near past stays exact and the long trend stays
affordable.

Three tiers, each a compaction of the one above:

===========  =============  ============
Tier         Resolution     Retained
===========  =============  ============
``raw``      5 s            ~2 days
``minute``   1-minute mean  ~5 weeks
``hour``     1-hour mean    indefinitely
===========  =============  ============

**What it is correct about is what it averages, and the two are not the same
operation.**

*Cumulative counters compact by their endpoint, never by a mean.* A mean of a
monotonically rising counter is meaningless, and every integral derived
downstream differences two endpoints. A minute row therefore carries the last
counter value its window observed, so the difference between two compacted rows
equals the difference between the two raw rows those endpoints came from -- this
is the exact property the tests assert.

*Gauges and utilisations compact by a time-weighted mean, and the weight is
carried.* A compacted row records how many source samples and how many seconds
of observation stand behind its means, because a mean that cannot say what it is
built from cannot be integrated honestly at the next tier. Compacting the minute
tier into the hour tier is then just the same weighted mean one level up, which
is why the operation is associative.

**Classification is by leaf name, not by position.** A numeric leaf is a counter
when its name ends in ``_total``, when it is a known cumulative quantity
(:data:`COUNTER_LEAF_NAMES`), or when it lives under a ``histograms`` section
(whose count, sum and buckets are all cumulative). Everything else numeric is a
gauge. Strings, booleans and ragged lists are identity and are carried from the
last row in the window.

**A null leaf is an absent observation, not an observation of zero.** It is
recorded only where its window holds no value yet, so it can never discard what
the window accumulated before it. A leaf null in every row of a window therefore
compacts to null -- the record's way of saying *not observed* rather than
*zero* -- while a gauge that is null in some rows and numeric in others compacts
to the weighted mean of the rows that did observe it, with the window's own
sample and second counts still declared in its ``obs`` block. A leaf written
after a null is a fresh observation, so a null that opens a window does not
suppress the values that follow it.

**A compaction never deletes its source.** The successor file is written and
re-read before anything is considered retired, so a crash mid-compaction loses
no source data. Re-running a compaction over the same source produces
byte-identical output: the transform is a pure function of the rows.

**A window belongs to one recording host, because a mean over two machines is a
reading of neither.** A row is grouped by the host it says it was recorded on
(:func:`~imas_ambix.agent.telemetry_index.row_host`), and a row carrying none is
attributed to the host doing the compacting -- its own nodename unless one is
named. The host is carried on the compacted row, so a later tier buckets by it
without having to be told again, and the window's duration and weight are
measured between rows of the same host: gaps taken across two interleaved
machines would halve every weight they touch.

**A window belongs to one boot of that host as well, because a window that
straddles a reboot differences two counters that never met.** Cumulative leaves
compact by endpoint, and the endpoint a row carries is the last reading of its
window, so a window holding rows from both sides of a reboot carries a
post-reboot endpoint beside a pre-reboot window's and the difference between
them is a figure about the reboot rather than about the machine. A row is
grouped by the boot it names (:func:`~imas_ambix.agent.telemetry_index.row_boot_id`),
and a row naming none is attributed to the boot doing the compacting; the boot
is carried on the compacted row exactly as the host is. A boot identity the
producer's parser refuses is stored as the empty string, so a row whose boot is
unknown groups with other unknown-boot rows and not with a keyed one.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from imas_ambix.agent.telemetry_index import (
    local_boot_id,
    resolve_boot_id,
    row_boot_id,
    row_host,
)

TIER_RAW = "raw"
TIER_MINUTE = "minute"
TIER_HOUR = "hour"

#: Resolution of each tier, in seconds. ``raw`` has no window of its own.
TIER_WINDOW_SECONDS: dict[str, int] = {
    TIER_MINUTE: 60,
    TIER_HOUR: 3600,
}

#: The tier a compaction of *this* tier produces.
NEXT_TIER: dict[str, str] = {
    TIER_RAW: TIER_MINUTE,
    TIER_MINUTE: TIER_HOUR,
}

#: Reserved keys a compacted row carries beside its compacted payload.
_TIER_KEY = "tier"
_WINDOW_START_KEY = "window_start"
_OBS_KEY = "obs"
_HOST_KEY = "host"
_BOOT_KEY = "boot_id"

#: Canonical cumulative quantities whose leaf name does not end in ``_total``.
#: The engine section spells cumulative token counters bare (``prompt_tokens``),
#: so a suffix test alone would average them.
COUNTER_LEAF_NAMES: frozenset[str] = frozenset(
    {
        "prompt_tokens",
        "generation_tokens",
        "prefix_cache_queries",
        "prefix_cache_hits",
        "cached_prompt_tokens",
        "uncached_prompt_tokens",
        "draft_tokens_total",
        "accepted_tokens_total",
        "num_accepted_per_pos",
    }
)

#: Decimal places a gauge mean and an observation weight are rounded to. The
#: rounding is what makes repeated compaction byte-identical rather than merely
#: near-identical on floats that differ in their last bit.
_GAUGE_PLACES = 9
_WEIGHT_PLACES = 6


class TelemetryStoreError(ValueError):
    """A row or path the store cannot compact honestly."""


@dataclass
class _Mean:
    """One gauge leaf's running time-weighted mean while a window accumulates."""

    weighted: float = 0.0
    weight: float = 0.0

    def add(self, value: float, weight: float) -> None:
        self.weighted += value * weight
        self.weight += weight

    def value(self) -> float:
        if self.weight == 0:
            return 0.0
        return round(self.weighted / self.weight, _GAUGE_PLACES)


@dataclass
class _Window:
    """Accumulated state for one time bucket."""

    start: _dt.datetime
    samples: int = 0
    seconds: float = 0.0
    endpoint: _dt.datetime | None = None
    payload: dict[str, Any] = field(default_factory=dict)


def _is_counter(path: tuple[str, ...]) -> bool:
    """Whether a leaf at *path* is a cumulative counter rather than a gauge."""
    if "histograms" in path:
        return True
    leaf = path[-1]
    return leaf.endswith("_total") or leaf in COUNTER_LEAF_NAMES


def _merge(
    acc: dict[str, Any],
    node: dict[str, Any],
    weight: float,
    path: tuple[str, ...],
) -> None:
    """Fold one row's payload *node* into the window accumulator *acc*.

    A counter leaf keeps the *latest* value (its endpoint); a gauge leaf folds
    into a running weighted mean; an identity leaf is overwritten so the window
    reports the last one seen; a null leaf is recorded only where the window has
    no value yet, so it cannot discard observations taken earlier in the window.
    Dicts recurse, and a list of equal-length dicts recurses element-wise so a
    per-card section is averaged per card rather than replaced.
    """
    for key, value in node.items():
        leaf_path = (*path, key)
        if value is None:
            # A null leaf is an absent observation, not an observation of zero,
            # so it is recorded only where the window holds nothing yet. Letting
            # it overwrite the accumulator would discard every observation taken
            # before it while still declaring the whole window's weight.
            acc.setdefault(key, None)
        elif isinstance(value, (bool, str)):
            acc[key] = value
        elif isinstance(value, (int, float)):
            if _is_counter(leaf_path):
                acc[key] = value
            else:
                entry = acc.get(key)
                if not isinstance(entry, _Mean):
                    entry = _Mean()
                    acc[key] = entry
                entry.add(float(value), weight)
        elif isinstance(value, dict):
            sub = acc.get(key)
            if not isinstance(sub, dict):
                sub = {}
                acc[key] = sub
            _merge(sub, value, weight, leaf_path)
        elif isinstance(value, list):
            acc[key] = _merge_list(acc.get(key), value, weight, leaf_path)
        else:  # pragma: no cover - a payload the recorder does not produce
            acc[key] = value


def _merge_list(
    acc: Any, values: list[Any], weight: float, path: tuple[str, ...]
) -> Any:
    """Compact a list leaf, element-wise when it is a list of dicts.

    A list of numeric values is a vector gauge and is averaged per index; a list
    of dicts (a per-card or per-position section) recurses by index. A ragged or
    otherwise unusual list is identity and the latest value is kept -- the store
    reports what it cannot average rather than inventing an alignment.
    """
    all_numbers = bool(values) and all(
        isinstance(v, (int, float)) and not isinstance(v, bool) for v in values
    )
    if all_numbers:
        if _is_counter(path):
            return values
        element = acc
        if not isinstance(element, list) or len(element) != len(values):
            element = [_Mean() for _ in values]
        for slot, value in zip(element, values, strict=True):
            slot.add(float(value), weight)
        return element
    all_dicts = bool(values) and all(isinstance(v, dict) for v in values)
    if all_dicts:
        element = acc
        if not isinstance(element, list) or len(element) != len(values):
            element = [{} for _ in values]
        for slot, value in zip(element, values, strict=True):
            _merge(slot, value, weight, path)
        return element
    return values


def _finalise(node: Any) -> Any:
    """Resolve accumulator objects into the plain JSON the record stores."""
    if isinstance(node, _Mean):
        return node.value()
    if isinstance(node, dict):
        return {key: _finalise(value) for key, value in node.items()}
    if isinstance(node, list):
        return [_finalise(value) for value in node]
    return node


def _parse_timestamp(row: dict[str, Any]) -> _dt.datetime:
    raw = row.get("timestamp")
    if not isinstance(raw, str):
        raise TelemetryStoreError(f"row has no ISO timestamp: {row!r}")
    try:
        stamp = _dt.datetime.fromisoformat(raw)
    except ValueError as exc:
        raise TelemetryStoreError(f"row timestamp is not ISO 8601: {raw!r}") from exc
    if stamp.tzinfo is None:
        raise TelemetryStoreError(f"row timestamp has no timezone: {raw!r}")
    return stamp


def _row_payload(row: dict[str, Any]) -> dict[str, Any]:
    """A row's own readings, without the keys that describe the row's grouping.

    The host and the boot are the window's identity rather than readings of it,
    so they are excluded here and written back by the window that owns them;
    merged as leaves they would be overwritten by whichever row of the window
    happened to be last.
    """
    return {
        key: value
        for key, value in row.items()
        if key
        not in (
            "timestamp",
            _TIER_KEY,
            _WINDOW_START_KEY,
            _OBS_KEY,
            _HOST_KEY,
            _BOOT_KEY,
        )
    }


def _row_weights(rows: list[dict[str, Any]], window_seconds: int) -> list[float]:
    """The observation duration each row stands for.

    A row already carrying an ``obs`` block is a compacted row and declares its
    own duration, which is what makes a second compaction level the same
    weighted mean one tier up. A raw row takes the gap to the next row, with the
    final row inheriting the last positive gap -- a forward-fill so the weights
    sum to the covered interval rather than dropping the tail.
    """
    declared: list[float] = []
    has_declared = True
    for row in rows:
        obs = row.get(_OBS_KEY)
        seconds = obs.get("seconds") if isinstance(obs, dict) else None
        if isinstance(seconds, (int, float)) and not isinstance(seconds, bool):
            declared.append(float(seconds))
        else:
            has_declared = False
            declared.append(0.0)
    if has_declared and any(declared):
        return declared

    stamps = [_parse_timestamp(row) for row in rows]
    weights: list[float] = []
    nominal = float(window_seconds)
    for index, stamp in enumerate(stamps):
        if index + 1 < len(stamps):
            gap = (stamps[index + 1] - stamp).total_seconds()
        elif weights:
            gap = weights[-1]
        else:
            gap = nominal
        if not (gap > 0):
            gap = nominal
        weights.append(gap)
        nominal = gap
    return weights


def _row_samples(rows: list[dict[str, Any]]) -> list[int]:
    """The number of source samples each row stands for.

    A raw row is one sample; a compacted row declares how many it represents, so
    a second compaction level accumulates the underlying count rather than
    counting its own coarser rows -- an hour built from sixty minute rows stands
    for every five-second sample those minutes stood for.
    """
    counts: list[int] = []
    for row in rows:
        obs = row.get(_OBS_KEY)
        declared = obs.get("samples") if isinstance(obs, dict) else None
        usable = (
            isinstance(declared, int)
            and not isinstance(declared, bool)
            and declared > 0
        )
        if usable:
            counts.append(declared)
        else:
            counts.append(1)
    return counts


def compact_rows(
    rows: list[dict[str, Any]],
    *,
    tier: str,
    host: str | None = None,
    boot_id: str | None = None,
) -> list[dict[str, Any]]:
    """Compact *rows* into one window per tier, per recording host and boot.

    *tier* is the resolution being produced (``minute`` or ``hour``) and fixes
    the window: a row belongs to the window its timestamp floors into, measured
    from the epoch so windows are stable across runs and machines. Rows must be
    ordered by timestamp; the store compacts the sequence it is given rather
    than reordering a record whose order is itself evidence.

    A window is per host, so two machines' readings in one window compact to two
    rows rather than to a mean belonging to neither, and per boot, so a window
    holding both sides of a reboot compacts to two rows rather than to a
    difference taken across counters that never met. A row is grouped under the
    host and boot it names
    (:func:`~imas_ambix.agent.telemetry_index.row_host`,
    :func:`~imas_ambix.agent.telemetry_index.row_boot_id`); *host* and *boot_id*
    are what a row naming neither is attributed to, defaulting to this node's
    own nodename and boot.

    Returns the compacted rows, each carrying ``tier``, its recording ``host``
    and ``boot_id``, the window's ``timestamp`` (its endpoint observation),
    ``window_start``, an ``obs`` block naming the samples and seconds behind its
    means, and the compacted payload under the same keys a raw row uses.
    """
    if tier not in TIER_WINDOW_SECONDS:
        raise TelemetryStoreError(f"no compaction window for tier {tier!r}")
    if not rows:
        return []
    fallback_host = host or os.uname().nodename
    fallback_boot, _ = resolve_boot_id(local_boot_id() if boot_id is None else boot_id)
    strides: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        recorded = row_host(row) or fallback_host
        boot, _ = resolve_boot_id(row_boot_id(row) or fallback_boot)
        strides.setdefault((recorded, boot), []).append(row)
    compacted: list[dict[str, Any]] = []
    for (recorded, boot), stride in strides.items():
        compacted.extend(
            _compact_stride(stride, tier, recorded, boot, TIER_WINDOW_SECONDS[tier])
        )
    return compacted


def _compact_stride(
    rows: list[dict[str, Any]],
    tier: str,
    host: str,
    boot_id: str,
    window_seconds: int,
) -> list[dict[str, Any]]:
    """Compact one host's boot's rows, in time order, into one row per window.

    The rows are one recording machine's over one boot, which is what lets a raw
    row's duration be read as the cadence gap it is: a raw row stands for the
    interval running to the next row of the same source, so durations inferred
    across two interleaved machines -- or across a reboot, where the cadence
    genuinely stops -- would be a fraction of the truth for both.
    """
    weights = _row_weights(rows, window_seconds)
    sample_counts = _row_samples(rows)
    windows: dict[int, _Window] = {}
    order: list[int] = []
    for row, weight, samples in zip(rows, weights, sample_counts, strict=True):
        stamp = _parse_timestamp(row)
        epoch = stamp.timestamp()
        bucket = math.floor(epoch / window_seconds) * window_seconds
        window = windows.get(bucket)
        if window is None:
            window = _Window(start=_dt.datetime.fromtimestamp(bucket, tz=_dt.UTC))
            windows[bucket] = window
            order.append(bucket)
        window.samples += samples
        window.seconds += weight
        window.endpoint = stamp
        _merge(window.payload, _row_payload(row), weight, ())

    compacted: list[dict[str, Any]] = []
    for bucket in order:
        window = windows[bucket]
        row: dict[str, Any] = {
            _TIER_KEY: tier,
            _HOST_KEY: host,
            _BOOT_KEY: boot_id,
            "timestamp": (window.endpoint or window.start).isoformat(),
            _WINDOW_START_KEY: window.start.isoformat(),
            _OBS_KEY: {
                "samples": window.samples,
                "seconds": round(window.seconds, _WEIGHT_PLACES),
            },
        }
        row.update(_finalise(window.payload))
        compacted.append(row)
    return compacted


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read one JSONL telemetry file into rows, one object per line."""
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TelemetryStoreError(f"{path}:{number} is not JSON") from exc
        if not isinstance(row, dict):
            raise TelemetryStoreError(f"{path}:{number} is not a JSON object")
        rows.append(row)
    return rows


def write_rows(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write *rows* as JSONL, deterministically.

    Keys are sorted and floats carry the rounding applied during compaction, so
    writing the same rows twice is byte-identical -- which is what lets a
    re-run of a compaction be recognised as a no-op rather than a rewrite.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows
    )
    destination.write_text(text, encoding="utf-8")


def compact_file(
    source: str | Path,
    destination: str | Path,
    *,
    tier: str,
    host: str | None = None,
    boot_id: str | None = None,
) -> int:
    """Compact *source* JSONL into *destination* at *tier*; return the row count.

    *host* and *boot_id* name the machine and boot rows that carry neither were
    recorded on; omitted, they are this node's own.

    The source is read and left untouched: a compaction writes its successor and
    never retires what it read, so a crash between the two loses nothing and the
    source remains readable. The successor is re-read and compared against the
    rows that were meant for it before this call returns, so a write that landed
    short or corrupted is reported as a failed compaction rather than as a
    completed one -- on a network filesystem a silent short write is exactly the
    failure that would otherwise be indistinguishable from success.
    """
    rows = compact_rows(read_rows(source), tier=tier, host=host, boot_id=boot_id)
    write_rows(destination, rows)
    readback = read_rows(destination)
    if readback != rows:
        raise TelemetryStoreError(
            f"{destination} does not hold what was written: "
            f"{len(readback)} rows read back, {len(rows)} written"
        )
    return len(rows)


def next_tier(tier: str) -> str:
    """The tier a compaction of *tier* produces."""
    try:
        return NEXT_TIER[tier]
    except KeyError as exc:
        raise TelemetryStoreError(f"no successor tier for {tier!r}") from exc


def run_compaction(
    raw_path: str | Path,
    minute_path: str | Path,
    hour_path: str | Path,
    *,
    host: str | None = None,
    boot_id: str | None = None,
) -> dict[str, int]:
    """Compact raw to minute to hour, leaving both sources in place.

    *host* and *boot_id* name the machine and boot raw rows that carry neither
    were recorded on; a compacted row carries both already, so the hour tier
    reads the hosts and boots the minute tier recorded rather than reapplying
    these.

    Returns the number of rows written to each compacted tier. Every source is
    read fully before its successor is written, so the hour tier never depends
    on a partial minute file.
    """
    minute_rows = compact_file(
        raw_path, minute_path, tier=TIER_MINUTE, host=host, boot_id=boot_id
    )
    hour_rows = compact_file(
        minute_path, hour_path, tier=TIER_HOUR, host=host, boot_id=boot_id
    )
    return {TIER_MINUTE: minute_rows, TIER_HOUR: hour_rows}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compact a telemetry record through its resolution tiers"
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--tier", required=True, choices=sorted(TIER_WINDOW_SECONDS))
    parser.add_argument(
        "--host",
        default=None,
        help="host rows that carry none were recorded on (default: this nodename)",
    )
    parser.add_argument(
        "--boot-id",
        default=None,
        help="boot rows that carry none were recorded on (default: this boot)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Rebuild one tier from one source file.

    This is the entry point a ``serving_receipts`` subcommand delegates to so a
    tier can be caught up or rebuilt while the serve is down.
    """
    args = _parser().parse_args(argv)
    compact_file(
        args.source,
        args.destination,
        tier=args.tier,
        host=args.host,
        boot_id=args.boot_id,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(main())
