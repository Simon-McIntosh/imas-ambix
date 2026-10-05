"""A long period's tokens equal the sum of its hours, never more.

A read can hold one job's raw rows beside its compacted tier rows -- the raw
file names no boot while a compacted row names the boot that wrote it, and the
two are read together before the coarser tier is dropped. A run boundary taken
on the boot alone then splits the job's single run at every alternation, and
each compacted row contributes its own window's advance a second time.

The identity below is the measure of that. A window's total is additive: the
total over a day equals the sum of the totals over the day's twenty-four hours,
for every counter the ledger reports. Draw the day's reading from one carrier
and the hours' from the same record, and the two must agree exactly -- the
failure this file pins is the day drifting above the sum of its parts.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import pytest

from imas_ambix.agent.telemetry_index import TelemetryIndex

_BASE = _dt.datetime(2026, 10, 1, 0, 0, 0, tzinfo=_dt.UTC)
_BOOT = "58cc3b8e-66b3-4b4f-baf3-e9e8b8f7c406"
_JOB = "1273253"
_HOUR = 3600
_DAY = 24 * _HOUR

# The counters the ledger reports: prompt and generation tokens, and the cached
# prompt tokens split by where the cache lived.
_COUNTERS = (
    "engine.prompt_tokens",
    "engine.generation_tokens",
    "engine.cached_prompt_tokens.device",
    "engine.cached_prompt_tokens.external",
)

# One distinct advancing quantity per counter, so a counter's total is
# recognisable rather than a copy of its neighbour's.
_SCALE = {
    "engine.prompt_tokens": 1000.0,
    "engine.generation_tokens": 100.0,
    "engine.cached_prompt_tokens.device": 250.0,
    "engine.cached_prompt_tokens.external": 40.0,
}


def _reading(seconds: int) -> dict[str, float]:
    """Every counter's value at one reading, as a flat dotted name -> value.

    The value is level across each hour boundary and advances within the hour:
    the reading at a boundary is one tick behind a plain linear ramp, so the
    reading just before the boundary and the reading at it carry the same value.
    That makes the day's readings and its hours' readings partition the same
    advance with nothing lost at a join, which is what lets the two totals be
    compared exactly rather than to within a boundary step.
    """
    boundary = seconds % _HOUR == 0 and seconds > 0
    ramp = 1000.0 * (seconds - 1 if boundary else seconds)
    return {name: ramp * scale / 1000.0 for name, scale in _SCALE.items()}


def _nested(flat: dict[str, float]) -> dict:
    """One reading's counters as the nested sections a record carries."""
    out: dict = {}
    for name, value in flat.items():
        node = out
        parts = name.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return out


def _raw_row(seconds: int) -> dict:
    """A raw reading of one job's counters, carrying no boot of its own."""
    return {
        "timestamp": (_BASE + _dt.timedelta(seconds=seconds)).isoformat(),
        "host": "node-a",
        "job_id": _JOB,
        **_nested(_reading(seconds)),
    }


def _minute_row(seconds: int) -> dict:
    """A compacted minute row ending at *seconds*, carrying a real boot.

    Its declared ``open`` is the counters one minute earlier, so the row spans
    a window of its own exactly as a compaction of the record would produce.
    """
    return {
        "tier": "minute",
        "timestamp": (_BASE + _dt.timedelta(seconds=seconds)).isoformat(),
        "window_start": (_BASE + _dt.timedelta(seconds=seconds - 60)).isoformat(),
        "host": "node-a",
        "job_id": _JOB,
        "boot_id": _BOOT,
        "open": _reading(seconds - 60),
        **_nested(_reading(seconds)),
    }


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _readings() -> list[int]:
    """The raw reading times: the day's origin and each hour's join.

    One reading one second before each boundary and one at it, so each hour
    window has a reading at or before its start and a reading inside it, and the
    two share the boundary's value.
    """
    times = [0]
    for hour in range(1, 25):
        times.append(hour * _HOUR - 1)
        times.append(hour * _HOUR)
    return times


def _totals(index: TelemetryIndex, name: str) -> tuple[float, float]:
    """``(day total, sum of the twenty-four hourly totals)`` for *name*."""
    origin = _BASE.timestamp()
    day = index.partitioned_total(name, origin, origin + _DAY).total or 0.0
    hourly = sum(
        index.partitioned_total(
            name, origin + hour * _HOUR, origin + (hour + 1) * _HOUR
        ).total
        or 0.0
        for hour in range(24)
    )
    return day, hourly


@pytest.mark.parametrize("name", _COUNTERS)
def test_a_day_of_one_jobs_raw_and_minute_rows_sums_to_its_hours(tmp_path, name):
    """One job's raw rows beside its minute rows, and the day equals its hours.

    Both carriers are held in one record file, so a read over it sees the raw
    readings and the compacted minute readings of the one job together. The
    day's total over that pair must equal the sum of the twenty-four hourly
    totals: the compacted rows describe the same advance the raw rows do, and a
    read that totals each once cannot make the day exceed the sum of its hours.
    """
    record_path = tmp_path / "records.jsonl"
    rows = [_raw_row(seconds) for seconds in _readings()]
    rows += [_minute_row(hour * _HOUR) for hour in range(1, 25)]
    _write(record_path, rows)

    with TelemetryIndex(tmp_path / "index.db") as index:
        index.ingest([record_path])
        day, hourly = _totals(index, name)

    assert day == pytest.approx(hourly), (
        f"{name}: the day total {day} is not the sum of its hours {hourly}"
    )
