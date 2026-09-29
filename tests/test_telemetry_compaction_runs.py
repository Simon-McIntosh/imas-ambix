"""Compacted rows and the query totals agree on where a counter run ends.

A cumulative counter restarts whenever the serving process it belongs to is
replaced, which happens inside one window, on one host, over one boot. A
compaction that groups only by host and boot reads the join as a fall and
carries one serve's endpoint across it, so the row a window compacts to holds a
figure no counter ever reached -- and the query layer, which does partition at
the restart, then disagrees with the record it is meant to be a view of.

These tests hold the two layers to one run boundary: both call the same
predicate, and a window with a serve restart inside it compacts to one row per
serving job, each carrying its own opening and closing reading.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import pytest

import imas_ambix.agent.telemetry_index as telemetry_index
from imas_ambix.agent.telemetry_index import TelemetryIndex
from imas_ambix.agent.telemetry_store import (
    TIER_HOUR,
    TIER_MINUTE,
    compact_rows,
    run_compaction,
)

_BASE = _dt.datetime(2026, 9, 20, 6, 0, 0, tzinfo=_dt.UTC)
_COUNTER = "requests_served_total"


def _at(seconds: float) -> float:
    return _BASE.timestamp() + seconds


def _row(seconds: float, *, job: str, counter: float) -> dict:
    """One recorder sample: a cumulative counter read by one serving job."""
    return {
        "timestamp": (_BASE + _dt.timedelta(seconds=seconds)).isoformat(),
        "job_id": job,
        _COUNTER: counter,
        "num_requests_running": 8,
    }


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _compacted_run_total(rows: list[dict], name: str) -> float:
    """Total compacted rows run by run, each run contributing close - open.

    The runs are grouped with the predicate both layers call, so the figure is
    read the way the query layer reads the record: a run opens at its first
    window's opening reading and closes at its last window's endpoint.
    """
    runs: list[list[dict]] = []
    previous = None
    for row in sorted(rows, key=lambda entry: entry["timestamp"]):
        current = (
            (row.get("host"), row.get("boot_id"), row.get("job_id")),
            {name: float(row[name])},
        )
        if not telemetry_index.counter_run_continues(previous, current):
            runs.append([])
        runs[-1].append(row)
        previous = current
    return sum(float(run[-1][name]) - float(run[0]["open"][name]) for run in runs)


def test_a_serve_restart_inside_one_window_compacts_to_one_row_per_job():
    """A restart within one minute leaves one row per serve, not one merged row."""
    rows = [
        _row(0, job="1001", counter=1_000_000),
        _row(30, job="1001", counter=1_000_000 + 500_000),
        _row(40, job="1002", counter=0),
        _row(55, job="1002", counter=20_000),
    ]

    compacted = compact_rows(rows, tier=TIER_MINUTE)

    assert len(compacted) == 2
    assert {row["job_id"] for row in compacted} == {"1001", "1002"}
    by_job = {row["job_id"]: row for row in compacted}
    assert by_job["1001"][_COUNTER] == 1_500_000
    assert by_job["1001"]["open"][_COUNTER] == 1_000_000
    assert by_job["1002"][_COUNTER] == 20_000
    assert by_job["1002"]["open"][_COUNTER] == 0
    # Both rows belong to the one window the four samples fall into.
    assert by_job["1001"]["window_start"] == by_job["1002"]["window_start"]


def test_run_partitioned_totals_agree_between_raw_and_compacted_rows(tmp_path):
    """The compacted total equals the total the raw rows partition to.

    With a restart inside the window the two serves contribute their own
    differences, 500,000 and 20,000; without one the window is a single
    difference. In both cases the compacted rows must reproduce the figure the
    query layer gets from the raw rows they replace.
    """
    restart_rows = [
        _row(0, job="1001", counter=1_000_000),
        _row(30, job="1001", counter=1_000_000 + 500_000),
        _row(40, job="1002", counter=0),
        _row(55, job="1002", counter=20_000),
    ]
    steady_rows = [
        _row(0, job="2001", counter=100),
        _row(50, job="2001", counter=300),
    ]

    restart_compacted = compact_rows(restart_rows, tier=TIER_MINUTE)
    steady_compacted = compact_rows(steady_rows, tier=TIER_MINUTE)
    assert _compacted_run_total(restart_compacted, _COUNTER) == pytest.approx(520_000.0)
    assert _compacted_run_total(steady_compacted, _COUNTER) == pytest.approx(200.0)

    restart_path = tmp_path / "restart.jsonl"
    steady_path = tmp_path / "steady.jsonl"
    _write(restart_path, restart_rows)
    _write(steady_path, steady_rows)

    with TelemetryIndex(tmp_path / "restart.db") as index:
        index.ingest([restart_path])
        restart = index.partitioned_total(_COUNTER, _at(0), _at(60))
    assert restart.runs == 2
    assert restart.total == pytest.approx(520_000.0)
    assert restart.total == pytest.approx(
        _compacted_run_total(restart_compacted, _COUNTER)
    )

    with TelemetryIndex(tmp_path / "steady.db") as index:
        index.ingest([steady_path])
        steady = index.partitioned_total(_COUNTER, _at(0), _at(60))
    assert steady.runs == 1
    assert steady.total == pytest.approx(200.0)
    assert steady.total == pytest.approx(
        _compacted_run_total(steady_compacted, _COUNTER)
    )


def test_compaction_and_query_share_one_run_boundary_predicate(tmp_path, monkeypatch):
    """Both layers consult the same predicate, so one patch sees both callers."""
    calls: list[object] = []
    original = telemetry_index.counter_run_continues

    def spy(previous, current):
        calls.append(current)
        return original(previous, current)

    monkeypatch.setattr(telemetry_index, "counter_run_continues", spy)

    compact_rows(
        [
            _row(0, job="3001", counter=100),
            _row(30, job="3001", counter=400),
        ],
        tier=TIER_MINUTE,
    )
    after_compaction = len(calls)
    assert after_compaction > 0, "compact_rows did not consult the run boundary"

    path = tmp_path / "serve.jsonl"
    _write(
        path,
        [
            _row(0, job="3001", counter=100),
            _row(30, job="3001", counter=400),
        ],
    )
    with TelemetryIndex(tmp_path / "index.db") as index:
        index.ingest([path])
        index.partitioned_total(_COUNTER, _at(0), _at(60))

    assert len(calls) > after_compaction, (
        "partitioned_total did not consult the run boundary"
    )


def test_an_hour_opening_carries_the_runs_opening_through_the_tier_chain():
    """An hour takes its opening from the tier below's opening, not its closing.

    One job counting 0, 1000, 1000, 2000 inside one hour must give the same hour
    opening and closing whether the raw rows are compacted straight to the hour
    or raw to minute and then to the hour. A second level that took the hour's
    margin from the first minute row's closing would drop the entire first
    window of the run -- the loss the opening block exists to prevent.
    """
    rows = [
        _row(0, job="4001", counter=0),
        _row(15, job="4001", counter=1_000),
        _row(30, job="4001", counter=1_000),
        _row(45, job="4001", counter=2_000),
    ]

    direct = compact_rows(rows, tier=TIER_HOUR)
    chained = compact_rows(compact_rows(rows, tier=TIER_MINUTE), tier=TIER_HOUR)

    assert len(direct) == 1
    assert len(chained) == 1
    assert direct[0]["open"][_COUNTER] == 0
    assert direct[0][_COUNTER] == 2_000
    assert chained[0]["open"] == direct[0]["open"]
    assert chained[0][_COUNTER] == direct[0][_COUNTER]


def _tier_total(tmp_path, rows: list[dict], name: str, tag: str) -> float:
    """A window's total read through the index of each tier's own file alone."""
    raw_path = tmp_path / f"{tag}-raw.jsonl"
    minute_path = tmp_path / f"{tag}-minute.jsonl"
    hour_path = tmp_path / f"{tag}-hour.jsonl"
    _write(raw_path, rows)
    run_compaction(raw_path, minute_path, hour_path)

    totals: list[float] = []
    for index_name, source in (
        ("raw", raw_path),
        ("minute", minute_path),
        ("hour", hour_path),
    ):
        with TelemetryIndex(tmp_path / f"{tag}-{index_name}.db") as index:
            index.ingest([source])
            total = index.partitioned_total(name, _at(0), _at(60))
        totals.append(total.total)
    return totals


def test_every_tier_file_totals_a_restart_window_the_same(tmp_path):
    """Raw, minute and hour files each total each run as closing minus opening.

    Read through the index built from one tier's file alone, no neighbour row is
    available to difference against, so a compacted row has to carry the two
    endpoints its window spans. The two serves contribute 500,000 and 20,000,
    and every tier must reproduce 520,000.
    """
    rows = [
        _row(0, job="1001", counter=1_000_000),
        _row(30, job="1002", counter=0),
        _row(55, job="1002", counter=20_000),
    ]
    rows.insert(1, _row(15, job="1001", counter=1_500_000))

    raw, minute, hour = _tier_total(tmp_path, rows, _COUNTER, "restart")

    assert raw == pytest.approx(520_000.0)
    assert minute == pytest.approx(520_000.0)
    assert hour == pytest.approx(520_000.0)


def test_every_tier_file_totals_a_null_rich_window_the_same(tmp_path):
    """An absent observation is neither an endpoint nor a zero at any tier.

    A counter null in part of a window keeps its last observed value, so the
    window's advance is measured between observed readings alone -- through the
    index of each tier's own file, the raw, minute and hour totals must agree.
    """
    rows = [
        _row(0, job="5001", counter=100),
        _row(10, job="5001", counter=None),
        _row(25, job="5001", counter=300),
        _row(35, job="5001", counter=None),
        _row(50, job="5001", counter=700),
    ]

    raw, minute, hour = _tier_total(tmp_path, rows, _COUNTER, "nulls")

    assert raw == pytest.approx(600.0)
    assert minute == pytest.approx(600.0)
    assert hour == pytest.approx(600.0)
