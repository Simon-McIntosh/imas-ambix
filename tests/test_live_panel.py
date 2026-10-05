"""Tests for the lane and live blocks of the watch document.

Every published figure is differenced from a fixture receipt tail rather than
read from a running engine, so the arithmetic is checked against numbers a
reader can recompute by hand. The fixture is written as the recorder writes it:
one JSON object per line, the counters under ``engine`` and the gauges flat.
"""

from __future__ import annotations

import ast
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from imas_ambix.agent import live_panel
from imas_ambix.agent.live_panel import (
    lane_block,
    live_block,
    newest_raw_receipt,
    read_lane_block,
)

BASE = datetime(2026, 9, 29, 12, 0, 0, tzinfo=UTC)


def _stamp(offset_seconds: float) -> str:
    """A timestamp in the form the lane document and receipt rows carry."""
    return (BASE + timedelta(seconds=offset_seconds)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _histogram(buckets: dict, count: float, total: float) -> dict:
    return {"buckets": dict(buckets), "count": count, "sum": total}


_TTFT_START = _histogram(
    {"0.1": 0.0, "0.2": 0.0, "0.4": 0.0, "0.6": 0.0, "0.8": 0.0, "1.0": 0.0}, 0.0, 0.0
)
# Cumulative since engine start: 0/2/6/14/18/20 requests at or below each
# bound. The window total is the advance of the topmost bucket (20), and half
# of it (10) is first reached inside the 0.6 bucket, so the median interpolates
# between 0.4 (6 requests) and 0.6 (14): 0.4 + (10-6)/(14-6) * 0.2 = 0.5.
_TTFT_END = _histogram(
    {"0.1": 0.0, "0.2": 2.0, "0.4": 6.0, "0.6": 14.0, "0.8": 18.0, "1.0": 20.0},
    20.0,
    12.0,
)
_ITL_START = _histogram({"0.01": 0.0, "0.05": 0.0, "0.1": 0.0, "inf": 0.0}, 0.0, 0.0)
_ITL_END = _histogram({"0.01": 1.0, "0.05": 1.0, "0.1": 3.0, "inf": 0.0}, 5.0, 0.42)


def _row(
    offset: float,
    *,
    job_id: int,
    running: float,
    generation: float,
    prompt: float,
    uncached: float,
    ttft: dict,
    itl: dict,
) -> dict:
    return {
        "timestamp": _stamp(offset),
        "job_id": job_id,
        "served_name": "deepseek-v4-1-flash",
        "num_requests_running": running,
        "num_requests_waiting": 1.0,
        "kv_cache_usage_perc": 0.42,
        "engine": {
            "generation_tokens": generation,
            "prompt_tokens": prompt,
            "uncached_prompt_tokens": uncached,
            "prefix_cache_hit_rate": 0.5,
            "histograms": {
                "time_to_first_token": ttft,
                "inter_token_latency": itl,
            },
        },
    }


def _one_run_rows() -> list[dict]:
    """Three rows 5 s apart over one run.

    The running count is 4, 5, 6 (mean 5). Generation advances 1000 tokens over
    10 s, so the aggregate is 100 tok/s and one stream's share is 20 tok/s.
    Uncached prompt tokens advance 1000 over the same window and prompt tokens
    2000, so the cache hit rate is 0.5.
    """
    first = _row(
        0.0,
        job_id=4811,
        running=4.0,
        generation=1_000_000,
        prompt=2_000_000,
        uncached=500_000,
        ttft=_TTFT_START,
        itl=_ITL_START,
    )
    middle = _row(
        5.0,
        job_id=4811,
        running=5.0,
        generation=1_000_500,
        prompt=2_001_000,
        uncached=500_500,
        ttft=_TTFT_START,
        itl=_ITL_START,
    )
    last = _row(
        10.0,
        job_id=4811,
        running=6.0,
        generation=1_001_000,
        prompt=2_002_000,
        uncached=501_000,
        ttft=_TTFT_END,
        itl=_ITL_END,
    )
    return [first, middle, last]


def _write(directory: Path, name: str, rows: list[dict]) -> Path:
    path = directory / name
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return path


def test_live_block_publishes_every_windowed_figure(tmp_path: Path):
    _write(tmp_path, "deepseek-v4-1-flash-4811.jsonl", _one_run_rows())
    block = live_block(tmp_path, now=BASE + timedelta(seconds=10))

    # The block names the file, job and served model it read.
    assert block["file"] == "deepseek-v4-1-flash-4811.jsonl"
    assert block["job_id"] == 4811
    assert block["served_name"] == "deepseek-v4-1-flash"
    assert block["state"] == "measured"
    assert block["rows"] == 3
    assert block["window_seconds"] == 45.0
    assert block["reason"] is None

    # Gauges come from the last row in the window.
    assert block["engine_queue"] == 1.0
    assert block["pool_occupancy"] == 0.42

    # The window aggregate divided by the window's mean running count.
    assert block["mean_running"] == pytest.approx(5.0)
    assert block["generation_toks_per_s"] == pytest.approx(100.0)
    assert block["per_stream_toks_per_s"] == pytest.approx(20.0)

    assert block["prefill_toks_per_s"] == pytest.approx(100.0)
    assert block["cache_hit_rate"] == pytest.approx(0.5)
    assert block["median_time_to_first_token_s"] == pytest.approx(0.5)


_REAL_TTFT_FIRST = _histogram(
    {
        "0.4": 46.0,
        "0.6": 9386.0,
        "0.8": 22300.0,
        "1.0": 26210.0,
        "2.0": 28895.0,
        "inf": 29940.0,
    },
    29940.0,
    0.0,
)
# A second recorder row of the same run, 100 requests later, each of them above
# 0.4 s. Every bucket advance is therefore the same 100: the window total is
# that 100 and the median lies inside the 0.4-0.6 bucket.
_REAL_TTFT_SECOND = _histogram(
    {
        "0.4": 46.0,
        "0.6": 9486.0,
        "0.8": 22400.0,
        "1.0": 26310.0,
        "2.0": 28995.0,
        "inf": 30040.0,
    },
    30040.0,
    0.0,
)


def test_median_interpolates_inside_a_cumulative_bucket(tmp_path: Path):
    """A real cumulative bucket pair, read as cumulative rather than disjoint."""
    rows = _one_run_rows()
    rows[0]["engine"]["histograms"]["time_to_first_token"] = _REAL_TTFT_FIRST
    rows[-1]["engine"]["histograms"]["time_to_first_token"] = _REAL_TTFT_SECOND
    _write(tmp_path, "deepseek-v4-1-flash-4811.jsonl", rows)
    block = live_block(tmp_path, now=BASE + timedelta(seconds=10))
    median = block["median_time_to_first_token_s"]
    # The window holds 100 requests, all in (0.4, 0.6]; half the total is first
    # reached there, so the median is 0.4 + (50-0)/(100-0) * (0.6-0.4).
    assert median == pytest.approx(0.5)
    assert 0.4 < median < 0.6


def test_per_stream_rate_is_not_a_median_inter_token_latency_reciprocal(
    tmp_path: Path,
):
    """The per-stream figure is an aggregate, not a latency reciprocal."""
    _write(tmp_path, "deepseek-v4-1-flash-4811.jsonl", _one_run_rows())
    block = live_block(tmp_path, now=BASE + timedelta(seconds=10))
    # The fixture's window median inter-token latency is 0.1 s, so a reconciler
    # that inverted it would publish 10.0 tok/s here.
    assert block["per_stream_toks_per_s"] == pytest.approx(20.0)
    assert block["per_stream_toks_per_s"] != pytest.approx(1.0 / 0.1)


def test_live_block_reports_null_rates_for_a_single_row_window(tmp_path: Path):
    _write(tmp_path, "deepseek-v4-1-flash-4811.jsonl", _one_run_rows()[2:])
    block = live_block(tmp_path, now=BASE + timedelta(seconds=10))
    assert block["rows"] == 1
    assert block["mean_running"] == 6.0
    assert block["generation_toks_per_s"] is None
    assert block["per_stream_toks_per_s"] is None
    assert block["prefill_toks_per_s"] is None
    assert block["cache_hit_rate"] is None
    assert block["reason"] == "fewer than two rows in the window"


def test_live_block_reports_null_rates_when_the_window_spans_two_runs(tmp_path: Path):
    rows = _one_run_rows()
    rows[2]["job_id"] = 4812
    _write(tmp_path, "deepseek-v4-1-flash-4811.jsonl", rows)
    block = live_block(tmp_path, now=BASE + timedelta(seconds=10))
    assert block["rows"] == 3
    assert block["generation_toks_per_s"] is None
    assert block["per_stream_toks_per_s"] is None
    assert block["prefill_toks_per_s"] is None
    assert block["cache_hit_rate"] is None
    assert block["reason"] == "the window spans more than one run"


def test_live_block_reports_null_rates_when_a_counter_falls(tmp_path: Path):
    rows = _one_run_rows()
    # The job id never changes, but the generation counter resets: a second run
    # began under the same job.
    rows[2]["engine"]["generation_tokens"] = 999_000
    _write(tmp_path, "deepseek-v4-1-flash-4811.jsonl", rows)
    block = live_block(tmp_path, now=BASE + timedelta(seconds=10))
    assert block["generation_toks_per_s"] is None
    assert block["per_stream_toks_per_s"] is None
    assert block["reason"] == "the window spans more than one run"


def test_newest_raw_receipt_ignores_tiers_and_picks_the_latest_tail(tmp_path: Path):
    _write(tmp_path, "deepseek-v4-1-flash-4811.jsonl", _one_run_rows())
    newer = tmp_path / "deepseek-v4-1-flash-4812.jsonl"
    last = {**(_one_run_rows()[-1]), "timestamp": _stamp(11.0)}
    newer.write_text("not json\n" + json.dumps(last) + "\n", encoding="utf-8")
    # The minute and hour tiers are named as the recorder names them and must be
    # skipped even though their contents are newer.
    for tier in ("minute", "hour"):
        tiered = {**(_one_run_rows()[-1]), "timestamp": _stamp(99.0)}
        (tmp_path / f"deepseek-v4-1-flash-4812.{tier}.jsonl").write_text(
            json.dumps(tiered) + "\n", encoding="utf-8"
        )
    assert newest_raw_receipt(tmp_path) == newer


def test_newest_raw_receipt_survives_a_half_written_final_line(tmp_path: Path):
    """A file caught mid-append is chosen on its last parseable row.

    The recorder writes one row per append, so a reader that judges a file by
    its final line alone loses the whole file while the writer is part-way
    through a row, even though every row before that one is intact. The final
    line below is a real row truncated mid-write.
    """
    path = tmp_path / "deepseek-v4-1-flash-4811.jsonl"
    torn = '{"timestamp": "2026-09-29T12:00:15Z", "job_'
    path.write_text(json.dumps(_one_run_rows()[-1]) + "\n" + torn, encoding="utf-8")
    assert newest_raw_receipt(tmp_path) == path
    block = live_block(tmp_path, now=BASE + timedelta(seconds=10))
    assert block["file"] == "deepseek-v4-1-flash-4811.jsonl"
    assert block["job_id"] == 4811
    assert block["served_name"] == "deepseek-v4-1-flash"


def test_live_block_reads_only_the_chosen_file(tmp_path: Path):
    _write(tmp_path, "deepseek-v4-1-flash-0001.jsonl", _one_run_rows())
    _write(tmp_path, "deepseek-v4-1-flash-0002.jsonl", _one_run_rows())
    block = live_block(tmp_path, now=BASE + timedelta(seconds=10))
    assert block["file"] == "deepseek-v4-1-flash-0002.jsonl"


def test_live_block_says_what_it_could_not_read_when_the_directory_is_empty(
    tmp_path: Path,
):
    block = live_block(tmp_path, now=BASE)
    assert block["file"] is None
    assert block["state"] == "unavailable"
    assert block["reason"] == "no raw receipt file with a readable row"


def test_lane_block_copies_the_published_fields():
    document = {
        "observed_at": _stamp(0.0),
        "state": "measured",
        "headroom": 7,
        "engine_headroom": 9,
        "headroom_is_upper_bound": False,
        "suggested_shelf_life_seconds": 30,
        "admission": {
            "headroom": 7,
            "verdict": "admit",
            "waiting": 0,
            "oldest_wait_seconds": 0.0,
            "live_runs": 5,
            "observed_seconds": 320.0,
            "requests_per_run": 0.8,
            "worker_slots": 18,
            "unkeyed_share": 0.25,
        },
        "router_generation_gate": {
            "width": 4,
            "effective_width": 3,
            "in_flight": 2,
            "waiting": 0,
            "paused": False,
            "width_mode": "auto",
            "reason": "within width",
        },
    }
    block = lane_block(document, now=BASE)
    assert block["observed_at"] == _stamp(0.0)
    assert block["state"] == "measured"
    assert block["stale"] is False
    assert block["suggested_shelf_life_seconds"] == 30
    assert block["headroom"] == 7
    assert block["engine_headroom"] == 9
    assert block["headroom_is_upper_bound"] is False
    assert block["admission"]["headroom"] == 7
    assert block["admission"]["verdict"] == "admit"
    assert block["admission"]["waiting"] == 0
    assert block["admission"]["oldest_wait_seconds"] == 0.0
    # The worker-slot fields travel with the admission block when the lane
    # publishes them, so a reader gets the slot count and the two figures it
    # divides from without recomputing either.
    assert block["worker_slots"]["live_runs"] == 5
    # The observed span travels beside the ratio it divides: a reader that wants
    # to reconcile requests_per_run with the busy seconds behind it needs the
    # span the router divided by, which is not the full window on a young router.
    assert block["worker_slots"]["observed_seconds"] == 320.0
    assert block["worker_slots"]["requests_per_run"] == 0.8
    assert block["worker_slots"]["worker_slots"] == 18
    # The unkeyed share rides with the run-keyed figures it bounds: the ratio
    # beside it is computed over the keyed remainder, so the reader needs both.
    assert block["worker_slots"]["unkeyed_share"] == 0.25
    assert block["router_generation_gate"]["width"] == 4
    assert block["router_generation_gate"]["effective_width"] == 3
    assert block["router_generation_gate"]["in_flight"] == 2
    assert block["router_generation_gate"]["waiting"] == 0
    assert block["router_generation_gate"]["width_mode"] == "auto"
    assert block["router_generation_gate"]["paused"] is False
    assert block["router_generation_gate"]["reason"] == "within width"


def test_lane_block_copies_a_published_availability_unchanged():
    document = {
        "observed_at": _stamp(0.0),
        "state": "measured",
        "headroom": 4,
        "availability_percent": 81.25,
        "router_generation_gate": {
            "width": 16,
            "effective_width": 8,
            "in_flight": 4,
            "waiting": 0,
            "paused": False,
        },
    }
    block = lane_block(document, now=BASE)
    assert block["availability_percent"] == 81.25


def test_lane_block_derives_availability_when_the_document_lacks_it():
    """A document predating the field draws the same figure, from the width in force.

    The router publishes the percentage only after a restart, so the reader
    derives it with the same function. Here the configured width is 16 but the
    width in force is 8 with headroom 4, so the gate is half open: 50.0.
    """
    document = {
        "observed_at": _stamp(0.0),
        "state": "measured",
        "headroom": 4,
        "router_generation_gate": {
            "width": 16,
            "effective_width": 8,
            "in_flight": 4,
            "waiting": 0,
            "paused": False,
        },
    }
    block = lane_block(document, now=BASE)
    assert block["availability_percent"] == 50.0


def test_lane_block_derives_availability_from_a_withheld_document():
    """A document predating the rule draws availability from its withheld estimate.

    The document carries no top-level headroom -- it predates the rule that
    publishes one bounded by the gate -- but it does carry the withheld estimate
    of 7 against an effective width of 16, so the shared selection derives 43.75
    rather than leaving the figure omitted.
    """
    document = {
        "observed_at": _stamp(0.0),
        "state": "measured",
        "sizing_verdict": "do-not-size",
        "withheld": {"headroom": 7, "concurrent_requests": 9, "why": "idle"},
        "router_generation_gate": {
            "width": 16,
            "effective_width": 16,
            "in_flight": 0,
            "waiting": 0,
            "paused": False,
        },
    }
    block = lane_block(document, now=BASE)
    assert block["availability_percent"] == 43.75
    assert "headroom" not in block
    assert block["withheld"]["headroom"] == 7


def test_lane_block_omits_availability_when_the_document_is_unreadable():
    """An unreadable lane has no figure to divide, so no percentage is derived."""
    document = {
        "observed_at": _stamp(0.0),
        "state": "unavailable",
        "reason": "metrics unreachable",
        "router_generation_gate": {
            "width": 16,
            "effective_width": 16,
            "in_flight": 0,
            "waiting": 0,
            "paused": False,
        },
    }
    assert "availability_percent" not in lane_block(document, now=BASE)


def test_lane_block_omits_unkeyed_share_when_the_lane_does_not_publish_it():
    """A lane that has not published the share leaves it absent, not null.

    The other worker-slot fields are present here, so this pins the share
    itself: a lane publishing slots but no unkeyed share is not the same as a
    lane publishing a share of zero.
    """
    document = {
        "observed_at": _stamp(0.0),
        "admission": {
            "headroom": 7,
            "verdict": "admit",
            "waiting": 0,
            "oldest_wait_seconds": 0.0,
            "live_runs": 5,
            "requests_per_run": 0.8,
            "worker_slots": 18,
        },
    }
    block = lane_block(document, now=BASE)
    assert set(block["worker_slots"]) == {
        "live_runs",
        "requests_per_run",
        "worker_slots",
    }
    assert "unkeyed_share" not in block["worker_slots"]


def test_lane_block_omits_observed_seconds_when_the_lane_does_not_publish_it():
    """A lane that has not published the observed span leaves it absent, not null.

    The other four worker-slot fields are present here, so this pins the span
    itself: a lane publishing slots but no observed span is not the same as a
    lane publishing a span of zero, and the ratio beside it must be copied
    rather than recomputed from a window the lane never named.
    """
    document = {
        "observed_at": _stamp(0.0),
        "admission": {
            "headroom": 7,
            "verdict": "admit",
            "waiting": 0,
            "oldest_wait_seconds": 0.0,
            "live_runs": 5,
            "requests_per_run": 0.8,
            "worker_slots": 18,
            "unkeyed_share": 0.25,
        },
    }
    block = lane_block(document, now=BASE)
    assert "observed_seconds" not in block["worker_slots"]
    assert set(block["worker_slots"]) == {
        "live_runs",
        "requests_per_run",
        "worker_slots",
        "unkeyed_share",
    }
    assert block["worker_slots"]["live_runs"] == 5
    assert block["worker_slots"]["requests_per_run"] == 0.8
    assert block["worker_slots"]["worker_slots"] == 18
    assert block["worker_slots"]["unkeyed_share"] == 0.25


def test_lane_block_copies_the_per_session_share_fields():
    """The per-session share rides beside the worker-slot fields it divides.

    A dispatcher needs its own session's figure, not only the lane's total, so
    the lane block copies the share fields into the same object. Each is copied
    as the lane published it, so a lane that published no sessions leaves the
    object without them -- which the omission tests above pin.
    """
    document = {
        "observed_at": _stamp(0.0),
        "admission": {
            "headroom": 7,
            "verdict": "open",
            "waiting": 0,
            "oldest_wait_seconds": 0.0,
            "live_runs": 14,
            "observed_seconds": 900.0,
            "requests_per_run": 0.5,
            "worker_slots": 10,
            "unkeyed_share": 0.0,
            "active_sessions": 2,
            "fair_share": 12,
            "borrow_reserve": 2,
            "new_session_worker_slots": 8,
            "sessions": {"A": {"live_runs": 2, "worker_slots": 10}},
        },
    }
    slots = lane_block(document, now=BASE)["worker_slots"]
    assert slots["active_sessions"] == 2
    assert slots["fair_share"] == 12
    assert slots["borrow_reserve"] == 2
    assert slots["new_session_worker_slots"] == 8
    assert slots["sessions"] == {"A": {"live_runs": 2, "worker_slots": 10}}
    assert slots["worker_slots"] == 10


def test_lane_block_copies_the_worker_slot_basis():
    """The basis travels with the slots it qualifies.

    ``worker_slots_basis`` says whether the figure was measured or assumed at
    one request per run, so a reader can tell a level from an upper bound; a
    copy that dropped it would report one as the other.
    """
    document = {
        "observed_at": _stamp(0.0),
        "admission": {
            "headroom": 6,
            "verdict": "open",
            "waiting": 0,
            "oldest_wait_seconds": 0.0,
            "live_runs": 2,
            "observed_seconds": 100.0,
            "requests_per_run": None,
            "worker_slots": 14,
            "worker_slots_basis": "assumed",
            "unkeyed_share": 0.0,
        },
    }
    slots = lane_block(document, now=BASE)["worker_slots"]
    assert slots["worker_slots_basis"] == "assumed"
    assert slots["requests_per_run"] is None
    assert slots["worker_slots"] == 14


def test_lane_block_keeps_headroom_is_upper_bound_with_its_figure():
    document = {
        "observed_at": _stamp(0.0),
        "headroom": 11,
        "headroom_is_upper_bound": True,
    }
    block = lane_block(document, now=BASE)
    assert block["headroom"] == 11
    assert block["headroom_is_upper_bound"] is True


def test_lane_block_reports_a_document_past_its_shelf_life_as_stale_with_its_age():
    document = {
        "observed_at": _stamp(-90.0),
        "headroom": 4,
        "suggested_shelf_life_seconds": 45,
    }
    block = lane_block(document, now=BASE)
    assert block["state"] == "stale"
    assert block["stale"] is True
    assert block["age_seconds"] == pytest.approx(90.0)


def test_lane_block_keeps_a_withheld_headroom_withheld():
    document = {
        "observed_at": _stamp(0.0),
        "withheld": "no measurement",
        "headroom_is_upper_bound": True,
    }
    block = lane_block(document, now=BASE)
    assert "headroom" not in block
    assert block["withheld"] == "no measurement"
    assert block["headroom_is_upper_bound"] is True


def test_lane_block_reports_an_absent_document_as_unavailable():
    block = lane_block({"state": "unavailable"}, now=BASE)
    assert block["state"] == "unavailable"
    assert block["stale"] is False
    assert "headroom" not in block


def test_read_lane_block_survives_a_missing_file(tmp_path: Path):
    block = read_lane_block(tmp_path / "lane.json", now=BASE)
    assert block["state"] == "unavailable"


def test_read_lane_block_reads_a_written_document(tmp_path: Path):
    path = tmp_path / "lane.json"
    path.write_text(
        json.dumps({"observed_at": _stamp(0.0), "headroom": 3}), encoding="utf-8"
    )
    block = read_lane_block(path, now=BASE)
    assert block["state"] == "measured"
    assert block["headroom"] == 3


def test_live_panel_imports_nothing_from_the_telemetry_index():
    source = Path(live_panel.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert not any("telemetry_index" in name for name in imported)
    assert "TelemetryIndex" not in source
