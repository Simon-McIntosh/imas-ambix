"""Tests for the shared lane's derived concurrency budget."""

from __future__ import annotations

import json

import pytest

from imas_ambix.agent.lane import (
    classify_reading,
    parse_lane_capacity,
    write_lane_document,
    write_unavailable_document,
)

_POOL = 2_200_283


def _metrics(
    *,
    running: float,
    waiting: float = 0.0,
    occupancy: float,
    preemptions: float = 0.0,
    queries: float = 1000.0,
    hits: float = 300.0,
    pool: int = _POOL,
) -> str:
    model = 'engine="0",model_name="deepseek-v4-flash"'
    return "\n".join(
        [
            "# HELP vllm:num_requests_running running requests",
            f"vllm:num_requests_running{{{model}}} {running}",
            f"vllm:num_requests_waiting{{{model}}} {waiting}",
            f'vllm:num_requests_waiting_by_reason{{{model},reason="capacity"}} 0.0',
            f"vllm:kv_cache_usage_perc{{{model}}} {occupancy}",
            f"vllm:num_preemptions_total{{{model}}} {preemptions}",
            f"vllm:prefix_cache_queries_total{{{model}}} {queries}",
            f"vllm:prefix_cache_hits_total{{{model}}} {hits}",
            f'vllm:cache_config_info{{{model},kv_cache_size_tokens="{pool}"}} 1.0',
        ]
    )


def test_budget_is_derived_from_the_engines_own_pool_size():
    """The pool must come from the engine, never from a profile or a constant."""
    capacity = parse_lane_capacity(_metrics(running=10, occupancy=0.267))

    assert capacity.pool_tokens == _POOL
    assert capacity.model_id == "deepseek-v4-flash"
    assert capacity.mean_context == pytest.approx(58_747, abs=50)
    assert capacity.concurrent_requests == 37
    assert capacity.headroom == 27


def test_a_heavier_workload_yields_a_smaller_budget_from_the_same_pool():
    """Tokens resident is the binding quantity, so the budget must move with it.

    Two readings of one serve within an hour gave roughly 59k and 97k tokens per
    request. A single written-down seat figure would have been wrong for one of
    them; deriving it on every read is what keeps both correct.
    """
    light = parse_lane_capacity(_metrics(running=10, occupancy=0.267))
    heavy = parse_lane_capacity(_metrics(running=11, occupancy=0.484))

    assert heavy.mean_context is not None
    assert light.mean_context is not None
    assert heavy.mean_context > light.mean_context
    assert heavy.concurrent_requests == 22
    assert light.concurrent_requests == 37


def test_an_idle_lane_reports_no_budget_rather_than_a_fabricated_one():
    """No traffic cannot be converted into a ceiling, so refuse to invent one."""
    capacity = parse_lane_capacity(_metrics(running=0, occupancy=0.0))

    assert capacity.mean_context is None
    assert capacity.concurrent_requests is None
    assert capacity.headroom is None
    assert "undefined" in capacity.summary()


def test_nothing_binding_is_reported_as_unmeasured_not_as_headroom():
    """An untested ceiling must announce that it is untested.

    Treating "not binding at the loads we could produce" as "not binding" is how
    a limit gets defended that nobody has ever reached.
    """
    quiet = parse_lane_capacity(_metrics(running=10, occupancy=0.267))

    assert quiet.binding_observed is False
    assert "extrapolation" in quiet.summary()


@pytest.mark.parametrize(
    ("waiting", "preemptions"),
    [(3.0, 0.0), (0.0, 5.0)],
)
def test_waiting_or_preemption_marks_the_lane_as_actually_binding(
    waiting, preemptions
):
    """Either signal is the lane speaking for itself rather than being modelled."""
    capacity = parse_lane_capacity(
        _metrics(
            running=10, occupancy=0.9, waiting=waiting, preemptions=preemptions
        )
    )

    assert capacity.binding_observed is True
    assert "binding" in capacity.summary()


def test_metrics_without_a_pool_size_are_refused():
    """A budget derived from a missing pool would be silently meaningless."""
    body = 'vllm:num_requests_running{engine="0",model_name="m"} 1.0'

    with pytest.raises(ValueError, match="KV pool size"):
        parse_lane_capacity(body)


def test_published_document_carries_its_observation_time(tmp_path):
    """A record with no stamp cannot be told apart from a current one."""
    capacity = parse_lane_capacity(_metrics(running=10, occupancy=0.267))

    path = write_lane_document(capacity, tmp_path / "lane.json")
    document = json.loads(path.read_text(encoding="utf-8"))

    assert document["observed_at"].endswith("Z")
    assert document["concurrent_requests"] == 37
    assert document["binding_observed"] is False
    assert document["pool_tokens"] == _POOL


def test_published_document_names_its_own_denominators(tmp_path):
    """A derived figure must say what it was divided by.

    A utilisation percentage computed against the wrong context window is
    arithmetically perfect and cannot be caught by inspecting the number; only
    the denominator travelling with it makes the error findable.
    """
    capacity = parse_lane_capacity(_metrics(running=10, occupancy=0.267))

    path = write_lane_document(capacity, tmp_path / "lane.json")
    document = json.loads(path.read_text(encoding="utf-8"))
    derived = document["derived_from"]

    assert derived["pool_tokens"] == _POOL
    assert derived["mean_context"] == document["mean_context"]
    assert derived["running_at_observation"] == 10
    assert "pool_tokens // mean_context" in derived["formula"]


def test_shelf_life_is_declared_for_the_reader_not_enforced_by_the_producer():
    """The producer must not bake a constant that is right at one fleet age."""
    capacity = parse_lane_capacity(_metrics(running=10, occupancy=0.267))

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as scratch:
        path = write_lane_document(capacity, Path(scratch) / "lane.json")
        document = json.loads(path.read_text(encoding="utf-8"))

    assert document["suggested_shelf_life_seconds"] == 120
    # Nothing in the reading refuses on age; staleness is the reader's call.
    assert not hasattr(capacity, "expired")


def test_zero_headroom_and_unmeasurable_are_distinguishable(tmp_path):
    """The figure must never carry its own validity.

    `0` and `null` are both falsy, so a reader writing `if not headroom: hold`
    collapses "the lane is full" into "we could not measure" -- opposite facts
    calling for opposite responses.
    """
    full = parse_lane_capacity(_metrics(running=37, occupancy=1.0, pool=_POOL))
    measured = json.loads(
        write_lane_document(full, tmp_path / "full.json").read_text(encoding="utf-8")
    )
    unavailable = json.loads(
        write_unavailable_document(
            "router unreachable", tmp_path / "gone.json"
        ).read_text(encoding="utf-8")
    )

    assert measured["state"] == "measured"
    assert measured["headroom"] == 0
    assert unavailable["state"] == "unavailable"
    assert "headroom" not in unavailable, "a missing key must raise, not read false"
    assert unavailable["reason"] == "router unreachable"


def test_a_stale_reading_keeps_its_figure_rather_than_becoming_unavailable(tmp_path):
    """"Measured 15 four minutes ago" and "could not measure" differ."""
    from datetime import UTC, datetime, timedelta

    capacity = parse_lane_capacity(_metrics(running=10, occupancy=0.267))
    document = json.loads(
        write_lane_document(capacity, tmp_path / "lane.json").read_text(
            encoding="utf-8"
        )
    )

    fresh = classify_reading(document, now=datetime.now(UTC))
    old = classify_reading(document, now=datetime.now(UTC) + timedelta(minutes=4))

    assert fresh == "measured"
    assert old == "stale"
    assert document["concurrent_requests"] == 37, "the figure survives staleness"


def test_an_unreadable_stamp_is_unavailable_not_silently_fresh():
    """A record whose age cannot be established must not pass as current."""
    assert classify_reading({"observed_at": "not-a-timestamp"}) == "unavailable"
    assert classify_reading({}) == "unavailable"
