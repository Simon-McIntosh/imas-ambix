"""The lane document publishes the rate its generating population achieved.

The consumer of the published document draws a distinction the producer must
honour: an absent ``throughput`` block reads as "not measured", while a block
present with a null mean reads as a measured zero per run. These cases pin both
sides of that distinction against a window of the odometer readings the
publisher already samples.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from imas_ambix.agent.lane import LaneCapacity, LaneWindow, write_lane_document

_POOL = 2_200_283
_START = datetime(2026, 10, 4, 12, 0, 0, tzinfo=UTC)


def _reading(*, tokens: int, running: int, at: datetime) -> LaneCapacity:
    """One lane sample carrying an odometer value, a run count and its stamp."""
    return LaneCapacity(
        model_id="deepseek-v4-flash",
        pool_tokens=_POOL,
        running=running,
        waiting=0,
        kv_occupancy=0.25,
        preemptions=0,
        prefix_hit_rate=0.3,
        generation_tokens=tokens,
        observed_at=at,
    )


def _published(tmp_path: Path, readings: list[LaneCapacity]) -> dict:
    """Compose the document the publisher writes for a window of readings."""
    path = tmp_path / "lane.json"
    write_lane_document(readings[-1], path, window=LaneWindow(readings=tuple(readings)))
    return json.loads(path.read_text(encoding="utf-8"))


def test_two_readings_publish_aggregate_runs_and_mean(tmp_path):
    """1000 then 7000 tokens over 60 s, 4 then 2 running: 100/3 tok/s per run."""
    readings = [
        _reading(tokens=1000, running=4, at=_START),
        _reading(tokens=7000, running=2, at=_START + timedelta(seconds=60)),
    ]

    block = _published(tmp_path, readings)["throughput"]

    assert block["aggregate_tokens_per_second"] == 100.0
    assert block["runs"] == 3.0
    assert block["mean_tokens_per_second"] == 33.333
    assert block["observed_at"] == "2026-10-04T12:01:00Z"


def test_single_reading_publishes_no_throughput_key(tmp_path):
    """One reading carries no interval, so the block is absent rather than null."""
    document = _published(tmp_path, [_reading(tokens=1000, running=4, at=_START)])

    assert "throughput" not in document


def test_counter_that_went_backwards_publishes_no_throughput_key(tmp_path):
    """A decrease means a new engine answered; the interval measures nothing."""
    readings = [
        _reading(tokens=7000, running=4, at=_START),
        _reading(tokens=1000, running=2, at=_START + timedelta(seconds=60)),
    ]

    assert "throughput" not in _published(tmp_path, readings)


def test_idle_window_publishes_runs_zero_with_null_mean(tmp_path):
    """An idle window is a measured zero per run, not an unpublished figure."""
    readings = [
        _reading(tokens=1000, running=0, at=_START),
        _reading(tokens=7000, running=0, at=_START + timedelta(seconds=60)),
    ]

    block = _published(tmp_path, readings)["throughput"]

    assert block["runs"] == 0.0
    assert block["mean_tokens_per_second"] is None
    assert block["aggregate_tokens_per_second"] == 100.0
