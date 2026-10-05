"""Every commit mid-ingest leaves the ledger additive, not only the settled one.

The index reads a job's finest tier and drops the coarser one it read earlier
only once the pass is over, so between the two moments the index holds two
carriers of one job's counters. A publisher reading the index on its own tick
can land inside that window, and each such reading must still satisfy the
ledger's additive identity: the total over a day equals the sum of the totals
over its hours. This file drives the real ingest over a job's tiered files --
an hour tier, then a minute tier, then the raw file -- in small batches, and
checks the identity after every batch the pass commits, so the transient is
exercised at each step and not only at the end.
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
_MINUTE = 60
_HOUR = 3600
_DAY = 24 * _HOUR

_COUNTERS = (
    "engine.prompt_tokens",
    "engine.generation_tokens",
    "engine.cached_prompt_tokens.device",
    "engine.cached_prompt_tokens.external",
)

_SCALE = {
    "engine.prompt_tokens": 1000.0,
    "engine.generation_tokens": 100.0,
    "engine.cached_prompt_tokens.device": 250.0,
    "engine.cached_prompt_tokens.external": 40.0,
}


def _reading(seconds: int) -> dict[str, float]:
    """Every counter's value at one reading, level across each hour boundary.

    The reading one minute before an hour boundary carries the boundary's own
    value, so the counter is level across the join of any partition that breaks
    at an hour. That makes a day and its twenty-four hours partition the same
    advance with nothing lost at a join, so the two totals can be compared
    exactly rather than to within a boundary step.
    """
    level = seconds % _HOUR == _HOUR - _MINUTE
    ramp = 1000.0 * (seconds + _MINUTE if level else seconds)
    return {name: ramp * scale / 1000.0 for name, scale in _SCALE.items()}


def _nested(flat: dict[str, float]) -> dict:
    out: dict = {}
    for name, value in flat.items():
        node = out
        parts = name.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return out


def _row(seconds: int, *, tier: str | None, boot: str | None, span: int) -> dict:
    row = {
        "timestamp": (_BASE + _dt.timedelta(seconds=seconds)).isoformat(),
        "host": "node-a",
        "job_id": _JOB,
        **_nested(_reading(seconds)),
    }
    if tier is not None:
        row["tier"] = tier
        row["boot_id"] = boot
        row["window_start"] = (
            _BASE + _dt.timedelta(seconds=seconds - span)
        ).isoformat()
        row["open"] = _reading(seconds - span)
    return row


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _day_against_hours(index: TelemetryIndex) -> dict[str, tuple[float, float]]:
    """For every counter, ``(day total, sum of the twenty-four hourly totals)``."""
    origin = _BASE.timestamp()
    pairs: dict[str, tuple[float, float]] = {}
    for name in _COUNTERS:
        day = index.partitioned_total(name, origin, origin + _DAY).total or 0.0
        hourly = sum(
            index.partitioned_total(
                name, origin + hour * _HOUR, origin + (hour + 1) * _HOUR
            ).total
            or 0.0
            for hour in range(24)
        )
        pairs[name] = (day, hourly)
    return pairs


def test_the_ledger_stays_additive_after_every_commit_of_an_overlapping_ingest(
    tmp_path, monkeypatch
):
    """A read taken after each committed batch still equals the sum of its hours.

    Each tier is ingested on its own pass, coarsest first, so the index holds
    one carrier of the job when the next, finer one is offered beside it. With
    only one carrier present, a read satisfies the identity exactly; with two
    present -- the hour rows beside the minute rows, then the minute rows beside
    the raw rows) the read must satisfy it too. The check runs after every batch
    the pass commits, so any state the ingest writes is measured.
    """
    hour_path = tmp_path / f"serve-{_JOB}.hour.jsonl"
    minute_path = tmp_path / f"serve-{_JOB}.minute.jsonl"
    raw_path = tmp_path / f"serve-{_JOB}.jsonl"
    _write(
        hour_path,
        [
            _row(s, tier="hour", boot=_BOOT, span=_HOUR)
            for s in range(_HOUR, _DAY + 1, _HOUR)
        ],
    )
    _write(
        minute_path,
        [
            _row(s, tier="minute", boot=_BOOT, span=2 * _MINUTE)
            for s in range(_MINUTE, _DAY + 1, _MINUTE)
        ],
    )
    _write(
        raw_path,
        [_row(s, tier=None, boot=None, span=0) for s in range(0, _DAY + 1, _MINUTE)],
    )

    reads: list[dict[str, tuple[float, float]]] = []
    original = TelemetryIndex._record_source

    def checking_record_source(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        reads.append(_day_against_hours(self))
        return result

    monkeypatch.setattr(TelemetryIndex, "_record_source", checking_record_source)

    with TelemetryIndex(tmp_path / "index.db") as index:
        # First pass: the minute tier is the finest offered, so it is read and
        # held. Second pass: the raw file is offered beside it and is now the
        # finest, so the raw readings are committed while the minute rows are
        # still held, until the pass drops them.
        index.ingest([minute_path, hour_path], batch_rows=240)
        index.ingest([raw_path, minute_path, hour_path], batch_rows=240)

    assert reads, "the ingest committed no batch to check"
    for read in reads:
        for name, (day, hourly) in read.items():
            assert day == pytest.approx(hourly), (
                f"{name}: a mid-ingest read gives day {day} against hours {hourly}"
            )
