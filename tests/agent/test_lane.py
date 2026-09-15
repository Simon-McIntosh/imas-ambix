"""Tests for the shared lane's derived concurrency budget."""

from __future__ import annotations

import json

import pytest

from imas_ambix.agent.lane import (
    classify_reading,
    detect_settling,
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
    # Half the pool, because the other half must hold the prefixes these
    # requests will reuse on their next turn.
    assert capacity.concurrent_requests == 18
    assert capacity.headroom == 8


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
    assert heavy.concurrent_requests == 11
    assert light.concurrent_requests == 18


def test_an_idle_lane_reports_no_MEASUREMENT_but_still_answers():
    """No traffic cannot be converted into a working-context figure.

    The mean context stays undefined, because there is nothing to measure. The
    advertised capacity does not: with nothing resident, what bounds a dispatch
    is the memory configuration rather than the traffic, and answering None
    there forced every reader to guess.
    """
    capacity = parse_lane_capacity(_metrics(running=0, occupancy=0.0))

    assert capacity.mean_context is None
    assert "undefined" in capacity.summary()
    assert capacity.concurrent_requests == capacity.max_concurrent


def test_nothing_binding_is_reported_as_unmeasured_not_as_headroom():
    """An untested ceiling must announce that it is untested.

    Treating "not binding at the loads we could produce" as "not binding" is how
    a limit gets defended that nobody has ever reached.
    """
    quiet = parse_lane_capacity(_metrics(running=10, occupancy=0.267))

    assert quiet.binding_observed is False
    assert "extrapolation" in quiet.summary()


def test_only_preemption_marks_the_lane_as_saturated():
    """Durable evidence only. A queue depth is an instantaneous reading.

    Preemption is cumulative and monotonic, so a non-zero count cannot be a
    sampling artefact. Waiting is routinely non-zero for a single poll under
    normal scheduling -- measured on this lane at four waiting against two
    running, cleared within one poll, no preemption.
    """
    preempting = parse_lane_capacity(
        _metrics(running=10, occupancy=0.9, preemptions=5.0)
    )

    assert preempting.binding_observed is True
    assert "SATURATED" in preempting.summary()


def test_a_momentary_queue_is_not_reported_as_saturation():
    """The first version of this field went true at waiting=2 on a healthy lane.

    A consumer gating on that refuses work the engine would have taken, which is
    the defect the deleted admission filter embodied.
    """
    queued = parse_lane_capacity(
        _metrics(running=16, occupancy=0.514, waiting=2.0, preemptions=0.0)
    )

    assert queued.binding_observed is False, "a queue depth is not saturation"
    assert "granularity, not pressure" in queued.summary()
    assert "SATURATED" not in queued.summary()


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
    assert document["concurrent_requests"] == 18
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
    assert document["concurrent_requests"] == 18, "the figure survives staleness"


def test_an_unreadable_stamp_is_unavailable_not_silently_fresh():
    """A record whose age cannot be established must not pass as current."""
    assert classify_reading({"observed_at": "not-a-timestamp"}) == "unavailable"
    assert classify_reading({}) == "unavailable"


def test_settling_is_detected_from_consecutive_samples_not_from_age():
    """A reader holding one sample cannot tell mid-settle from steady state.

    Reproduces the measured sequence at unchanged running of 19: mean context
    15,586 then ~62,865 tokens, a fourfold move with nothing joining or leaving.
    """
    first = parse_lane_capacity(_metrics(running=19, occupancy=0.1346))
    second = parse_lane_capacity(_metrics(running=19, occupancy=0.5430))

    assert detect_settling(None, first) is None, "no baseline is unknown"
    assert detect_settling(first, second) is True
    assert detect_settling(second, second) is False, "a steady lane has settled"


def test_a_changed_running_count_makes_settling_unknowable():
    """A moving mix explains a moving mean, so do not call it settling."""
    before = parse_lane_capacity(_metrics(running=8, occupancy=0.20))
    after = parse_lane_capacity(_metrics(running=19, occupancy=0.55))

    assert detect_settling(before, after) is None


def test_a_settling_reading_publishes_headroom_as_an_upper_bound(tmp_path):
    """The measured error runs one way: always generous, by up to eightfold."""
    capacity = parse_lane_capacity(_metrics(running=19, occupancy=0.1346))

    settling = json.loads(
        write_lane_document(
            capacity, tmp_path / "s.json", settling=True
        ).read_text(encoding="utf-8")
    )
    settled = json.loads(
        write_lane_document(
            capacity, tmp_path / "q.json", settling=False
        ).read_text(encoding="utf-8")
    )

    assert settling["settling"] is True
    assert settling["headroom_is_upper_bound"] is True
    assert settled["headroom_is_upper_bound"] is False


def test_unknown_settling_publishes_headroom_as_a_bound_not_a_figure(tmp_path):
    """The most dangerous reading is the one where settling cannot be decided.

    Measured: a sample published `headroom 134` unflagged because the running
    count had moved by 3 between polls, making settling unknown rather than
    true. The next sample read 17. Unknown must resolve toward the direction
    every measured error on this lane already runs -- apparent headroom.
    """
    capacity = parse_lane_capacity(_metrics(running=22, occupancy=0.141))

    unknown = json.loads(
        write_lane_document(
            capacity, tmp_path / "u.json", settling=None
        ).read_text(encoding="utf-8")
    )
    settled = json.loads(
        write_lane_document(
            capacity, tmp_path / "s.json", settling=False
        ).read_text(encoding="utf-8")
    )

    assert unknown["settling"] is None
    assert unknown["headroom_is_upper_bound"] is True, "unknown is not settled"
    assert settled["headroom_is_upper_bound"] is False


def test_advertised_capacity_reserves_room_for_the_prefixes_it_will_reuse():
    """Sizing to the whole pool leaves nothing to cache.

    Reproduces a reading taken 2026-09-15: pool 2,557,835, four running at 38%
    occupancy, mean context 243k. Published against the whole pool that was 10
    concurrent; against the half that keeps prefixes resident it is 5, which is
    what an independent re-derivation from the same advice produced.
    """
    capacity = parse_lane_capacity(
        _metrics(running=4, occupancy=0.3804, pool=2_557_835)
    )

    assert capacity.mean_context == pytest.approx(243_250, abs=500)
    assert capacity.concurrent_requests == 5, "half the pool, not all of it"
    assert capacity.headroom == 1


def test_a_shrinking_context_cannot_advertise_past_the_safety_ceiling():
    """The pool term rises without bound as the working context shrinks.

    Measured: a cold lane with short prompts advertised 131 concurrent, beyond
    anything this serve has survived. The crash edge is a property of the memory
    configuration rather than the traffic, so it caps what is published.
    """
    tiny = parse_lane_capacity(_metrics(running=1, occupancy=0.0061))

    assert tiny.concurrent_requests <= tiny.max_concurrent
    assert tiny.concurrent_requests == 81


def test_an_idle_lane_answers_with_the_ceiling_rather_than_nothing():
    """With nothing resident, the configuration bounds a dispatch, not traffic.

    Returning None forced every reader to guess: one treating it as a hold
    stalls a ramp permanently, one treating it as a green light is right only by
    luck. The first ramp step is exactly where the signal is most wanted.
    """
    idle = parse_lane_capacity(_metrics(running=0, occupancy=0.0))

    assert idle.mean_context is None, "still no measurement to report"
    assert idle.concurrent_requests == idle.max_concurrent
    assert idle.headroom == idle.max_concurrent
