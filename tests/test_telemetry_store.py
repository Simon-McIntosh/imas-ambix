"""A compacted telemetry record preserves what it must not average.

The store exists to make a multi-day record affordable without changing the
quantities a reader derives from it. Two operations matter and they are not the
same: cumulative counters must survive as endpoints so a difference across a
tier boundary equals the raw difference, and gauges must compact as a
time-weighted mean that carries the weight behind it. These tests drive a
synthetic multi-day record through both tiers and assert both, plus the two
durability properties the plan requires -- a compaction is idempotent, and it
never retires the tier it read.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta

import pytest

from imas_ambix.agent.telemetry_store import (
    TIER_HOUR,
    TIER_MINUTE,
    TelemetryStoreError,
    compact_file,
    compact_rows,
    read_rows,
    run_compaction,
)

_CADENCE_S = 20
_DAYS = 3
_START = datetime(2026, 9, 1, tzinfo=UTC)
_SAMPLES = _DAYS * 24 * 60 * 60 // _CADENCE_S


def _raw_rows() -> list[dict]:
    """A three-day record at twenty-second cadence, with every leaf shape.

    The counters rise monotonically, so a mean over them is a different number
    from their endpoint and the endpoint test can tell the two apart. The gauges
    oscillate, so a mean over them is not equal to any single sample.
    """
    rows = []
    for index in range(_SAMPLES):
        stamp = _START + timedelta(seconds=index * _CADENCE_S)
        rows.append(
            {
                "timestamp": stamp.isoformat(),
                "job_id": "1273253",
                "profile_slug": "deepseek-v4-1-flash",
                "served_name": "deepseek-v4.1-flash",
                "gpus": 4,
                "generation_throughput_toks_per_s": 100.0 + index % 50,
                "num_requests_running": index % 17,
                "kv_cache_usage_perc": 30.0 + 5.0 * math.sin(index / 40.0),
                "prefix_cache_queries_total": 1000 + 7 * index,
                "prefix_cache_hits_total": 500 + 3 * index,
                "engine": {
                    "family": "sglang",
                    "requests_running": index % 17,
                    "prompt_tokens": 5000 + 13 * index,
                    "generation_tokens": 2000 + 11 * index,
                    "cached_prompt_tokens": {"device": 100 + index},
                },
            }
        )
    return rows


def _write(path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _window_start(stamp: datetime, window_s: int) -> int:
    return math.floor(stamp.timestamp() / window_s) * window_s


def _raw_endpoints(rows: list[dict], window_s: int, key: str) -> dict[int, float]:
    """The last raw counter value in each window -- the endpoint a compacted row
    must carry. Rows are ascending, so the last write per window wins."""
    endpoints: dict[int, float] = {}
    for row in rows:
        stamp = datetime.fromisoformat(row["timestamp"])
        endpoints[_window_start(stamp, window_s)] = row[key]
    return endpoints


def _bucket_of(row: dict) -> int:
    return int(datetime.fromisoformat(row["window_start"]).timestamp())


@pytest.fixture
def record(tmp_path):
    raw_path = tmp_path / "raw.jsonl"
    rows = _raw_rows()
    _write(raw_path, rows)
    return raw_path, rows


def test_multi_day_record_compacts_through_both_tiers(tmp_path, record):
    raw_path, rows = record
    counts = run_compaction(
        raw_path, tmp_path / "minute.jsonl", tmp_path / "hour.jsonl"
    )

    minute = read_rows(tmp_path / "minute.jsonl")
    hour = read_rows(tmp_path / "hour.jsonl")

    assert counts == {TIER_MINUTE: len(minute), TIER_HOUR: len(hour)}
    assert len(minute) == _DAYS * 24 * 60  # one row per minute
    assert len(hour) == _DAYS * 24  # one row per hour
    for row in (*minute, *hour):
        assert row["obs"]["samples"] > 0
        assert row["obs"]["seconds"] > 0
        assert row["window_start"] <= row["timestamp"]


def test_counter_difference_across_a_tier_boundary_equals_the_raw_difference(
    tmp_path, record
):
    """The load-bearing property: endpoints are kept, never averaged.

    A mean of a rising counter is a number no reader can difference, so the
    compacted value must equal the raw endpoint exactly, and a difference taken
    between two compacted rows must equal the same difference taken on the raw
    rows those endpoints came from.
    """
    raw_path, rows = record
    run_compaction(raw_path, tmp_path / "minute.jsonl", tmp_path / "hour.jsonl")
    minute = read_rows(tmp_path / "minute.jsonl")
    hour = read_rows(tmp_path / "hour.jsonl")

    for window_s, tier_rows in ((60, minute), (3600, hour)):
        endpoints = _raw_endpoints(rows, window_s, "prefix_cache_queries_total")
        for row in tier_rows:
            assert row["prefix_cache_queries_total"] == endpoints[_bucket_of(row)]

        # A difference between two compacted rows must equal the raw difference.
        first, last = tier_rows[5], tier_rows[len(tier_rows) // 2]
        compacted_delta = (
            last["prefix_cache_queries_total"] - first["prefix_cache_queries_total"]
        )
        raw_delta = (
            endpoints[_bucket_of(last)] - endpoints[_bucket_of(first)]
        )
        assert compacted_delta == raw_delta > 0

        # A mean would have produced a different number; show it is not that.
        window = [r for r in rows if _window_start(
            datetime.fromisoformat(r["timestamp"]), window_s
        ) == _bucket_of(first)]
        mean = sum(r["prefix_cache_queries_total"] for r in window) / len(window)
        assert first["prefix_cache_queries_total"] > mean

    # Nested engine counters compact by endpoint too.
    raw_gen = {
        _window_start(datetime.fromisoformat(r["timestamp"]), 60): r["engine"][
            "generation_tokens"
        ]
        for r in rows
    }
    for row in minute:
        assert row["engine"]["generation_tokens"] == raw_gen[_bucket_of(row)]


def test_compacted_gauge_rows_carry_the_observation_weight_behind_them(
    tmp_path, record
):
    raw_path, rows = record
    run_compaction(raw_path, tmp_path / "minute.jsonl", tmp_path / "hour.jsonl")
    minute = read_rows(tmp_path / "minute.jsonl")
    hour = read_rows(tmp_path / "hour.jsonl")

    # Every minute of a three-day record is complete, so each carries its window.
    for row in minute:
        assert row["obs"]["samples"] == 60 // _CADENCE_S
        assert row["obs"]["seconds"] == pytest.approx(60, abs=1e-6)

    # The hour tier integrates the minute weights it is built from.
    assert sum(row["obs"]["samples"] for row in minute) == len(rows)
    for row in hour:
        assert row["obs"]["samples"] == (60 // _CADENCE_S) * 60
        assert row["obs"]["seconds"] == pytest.approx(3600, abs=1e-6)

    # A gauge that carried no weight could not be integrated at the next tier;
    # the hour mean must equal the sample-count-weighted mean of the minute rows.
    minute_by_hour: dict[int, list[dict]] = {}
    for row in minute:
        minute_by_hour.setdefault(_bucket_of(row) // 3600 * 3600, []).append(row)
    for row in hour:
        group = minute_by_hour[_bucket_of(row)]
        weighted = sum(
            m["kv_cache_usage_perc"] * m["obs"]["samples"] for m in group
        ) / sum(m["obs"]["samples"] for m in group)
        assert row["kv_cache_usage_perc"] == pytest.approx(weighted, abs=1e-6)


def test_gauge_means_are_time_weighted_not_single_samples(tmp_path, record):
    raw_path, rows = record
    run_compaction(raw_path, tmp_path / "minute.jsonl", tmp_path / "hour.jsonl")
    minute = read_rows(tmp_path / "minute.jsonl")

    target = minute[len(minute) // 3]
    window = [
        r
        for r in rows
        if _window_start(datetime.fromisoformat(r["timestamp"]), 60)
        == _bucket_of(target)
    ]
    expected_running = sum(r["num_requests_running"] for r in window) / len(window)
    assert target["num_requests_running"] == pytest.approx(expected_running, abs=1e-9)

    # An identity leaf is carried from the last row, never averaged.
    assert target["served_name"] == "deepseek-v4.1-flash"
    assert target["job_id"] == "1273253"


def test_compaction_is_idempotent(tmp_path, record):
    raw_path, _rows = record
    compact_file(raw_path, tmp_path / "a.jsonl", tier=TIER_MINUTE)
    compact_file(raw_path, tmp_path / "b.jsonl", tier=TIER_MINUTE)

    first = (tmp_path / "a.jsonl").read_bytes()
    second = (tmp_path / "b.jsonl").read_bytes()
    assert first == second

    run_compaction(raw_path, tmp_path / "m1.jsonl", tmp_path / "h1.jsonl")
    run_compaction(raw_path, tmp_path / "m2.jsonl", tmp_path / "h2.jsonl")
    assert (tmp_path / "m1.jsonl").read_bytes() == (tmp_path / "m2.jsonl").read_bytes()
    assert (tmp_path / "h1.jsonl").read_bytes() == (tmp_path / "h2.jsonl").read_bytes()


def test_a_source_tier_survives_its_successor(tmp_path, record):
    raw_path, rows = record
    run_compaction(raw_path, tmp_path / "minute.jsonl", tmp_path / "hour.jsonl")

    # Both sources are still present and readable after their successors landed.
    assert raw_path.exists()
    assert (tmp_path / "minute.jsonl").exists()
    assert len(read_rows(raw_path)) == len(rows)
    assert len(read_rows(tmp_path / "minute.jsonl")) == _DAYS * 60 * 24


def test_a_row_without_a_usable_timestamp_is_refused():
    with pytest.raises(TelemetryStoreError):
        compact_rows([{"prefix_cache_queries_total": 1}], tier=TIER_MINUTE)
    with pytest.raises(TelemetryStoreError):
        compact_rows(
            [{"timestamp": "2026-09-01T00:00:00"}], tier=TIER_MINUTE
        )
