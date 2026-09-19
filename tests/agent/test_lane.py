"""Tests for the shared lane's derived concurrency budget."""

from __future__ import annotations

import json

import pytest

from imas_ambix.agent.lane import (
    LaneCapacity,
    LaneWindow,
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


def test_an_idle_lane_reports_no_measurement_but_still_answers():
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
    assert "(pool_tokens * occupancy_target) // mean_context" in derived["formula"]
    # The occupancy target is a denominator too, and it was the one the recorded
    # formula omitted: the published figure has always been halved by it, so a
    # reader re-deriving from the stated arithmetic got twice the real budget.
    # Naming pool and context while silently dropping the factor between them is
    # the exact failure this test exists to catch.
    assert derived["occupancy_target"] == LaneCapacity.OCCUPANCY_TARGET
    assert "MEDIAN over the window" in derived["formula"]


def test_shelf_life_is_declared_for_the_reader_not_enforced_by_the_producer():
    """The producer must not bake a constant that is right at one fleet age."""
    capacity = parse_lane_capacity(_metrics(running=10, occupancy=0.267))

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as scratch:
        path = write_lane_document(capacity, Path(scratch) / "lane.json")
        document = json.loads(path.read_text(encoding="utf-8"))

    assert document["suggested_shelf_life_seconds"] == 45
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
    """ "Measured 15 four minutes ago" and "could not measure" differ."""
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
        write_lane_document(capacity, tmp_path / "s.json", settling=True).read_text(
            encoding="utf-8"
        )
    )
    settled = json.loads(
        write_lane_document(capacity, tmp_path / "q.json", settling=False).read_text(
            encoding="utf-8"
        )
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
        write_lane_document(capacity, tmp_path / "u.json", settling=None).read_text(
            encoding="utf-8"
        )
    )
    settled = json.loads(
        write_lane_document(capacity, tmp_path / "s.json", settling=False).read_text(
            encoding="utf-8"
        )
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


def test_the_shelf_life_follows_the_publishing_cadence():
    """A shelf life several times the volatility clears a stale figure for use.

    Measured 2026-09-15: a peer sampling every twenty seconds watched
    lane_headroom read 21, 11 then 14 across forty seconds while the document
    claimed two minutes of freshness. Deriving it from the cadence keeps the two
    in step if the cadence ever changes.
    """
    import tempfile
    from pathlib import Path

    capacity = parse_lane_capacity(_metrics(running=10, occupancy=0.267))
    with tempfile.TemporaryDirectory() as scratch:
        for interval, expected in ((30, 45), (10, 15), (60, 90)):
            written = write_lane_document(
                capacity,
                Path(scratch) / f"{interval}.json",
                refresh_interval=interval,
            )
            document = json.loads(written.read_text(encoding="utf-8"))
            assert document["suggested_shelf_life_seconds"] == expected


def test_windowed_budget_survives_an_excursion_a_single_sample_cannot():
    """One sample of this ratio is not a level.

    The four readings below are what two sessions independently measured on
    2026-09-15 inside a single 120-second shelf life, with every one of them
    current. Sized from any single sample the budget reads 57, 4, 78 or 21 --
    a seventeenfold swing in a figure a coordinator uses to size a wave, where
    too low stalls a fleet and too high is the failure the module exists to
    prevent.
    """
    readings = tuple(
        parse_lane_capacity(_metrics(running=10, occupancy=mean * 10 / _POOL))
        for mean in (19_000, 247_000, 14_000, 51_000)
    )
    window = LaneWindow(readings=readings)

    instant = [r.concurrent_requests for r in readings]
    assert max(instant) / min(instant) > 10, instant

    # The median lands inside the samples, not at either extreme.
    assert window.mean_context == 51_000
    assert min(instant) < window.concurrent_requests < max(instant)

    # And it declares that it is summarising noise rather than reporting a level.
    assert window.is_volatile is True
    assert window.spread == (14_000, 247_000)


def test_a_steady_lane_is_not_flagged_volatile():
    """The flag must discriminate, or it is decoration.

    A threshold that fires on ordinary variation trains its reader to ignore
    it, which is worse than not publishing it.
    """
    window = LaneWindow(
        readings=tuple(
            parse_lane_capacity(_metrics(running=10, occupancy=mean * 10 / _POOL))
            for mean in (80_000, 84_000, 79_000, 82_000)
        )
    )

    assert window.is_volatile is False
    assert window.spread == (79_000, 84_000)


def test_idle_readings_do_not_drag_the_window_toward_a_large_budget():
    """An idle sample contributes nothing rather than a zero context.

    Folding "no traffic" in as a small mean would inflate the budget, which is
    the generous direction every measurement error on this lane has run in.
    """
    idle = parse_lane_capacity(_metrics(running=0, occupancy=0.0))
    busy = parse_lane_capacity(_metrics(running=10, occupancy=90_000 * 10 / _POOL))

    assert LaneWindow(readings=(idle, busy)).mean_context == 90_000
    # All-idle stays undefined, and the budget falls back to the safety ceiling
    # rather than to an extrapolation from nothing.
    all_idle = LaneWindow(readings=(idle,))
    assert all_idle.mean_context is None
    assert all_idle.concurrent_requests == idle.max_concurrent


def test_document_publishes_the_instantaneous_figures_beside_the_smoothed_one():
    """Smoothing must be auditable, not invisible.

    A reader debugging one moment still needs the number that moment produced;
    a reader checking the smoothing needs both to compare.
    """
    import tempfile
    from pathlib import Path

    spike = parse_lane_capacity(_metrics(running=10, occupancy=247_000 * 10 / _POOL))
    window = LaneWindow(
        readings=tuple(
            parse_lane_capacity(_metrics(running=10, occupancy=mean * 10 / _POOL))
            for mean in (19_000, 247_000, 14_000, 51_000)
        )
    )

    with tempfile.TemporaryDirectory() as scratch:
        path = write_lane_document(
            spike, Path(scratch) / "lane.json", window=window, refresh_interval=30
        )
        document = json.loads(path.read_text(encoding="utf-8"))

    assert document["mean_context"] == 51_000
    assert document["mean_context_instant"] == 247_000
    assert document["mean_context_spread"] == [14_000, 247_000]
    assert document["window_samples"] == 4
    assert document["window_seconds"] == 120
    assert document["volatile"] is True
    # Volatility alone marks the figure a bound, independent of settling.
    assert document["headroom_is_upper_bound"] is True
    # This window cannot answer, so the two sizing figures are WITHHELD from the
    # top level rather than published beside a caveat. The instantaneous
    # figures stay, because they describe the sample rather than advise a
    # dispatch.
    assert "concurrent_requests" not in document
    assert "headroom" not in document
    assert document["withheld"]["headroom"] == window.headroom
    assert document["concurrent_requests_instant"] is not None


def test_a_volatile_window_says_what_to_do_not_only_that_it_is_volatile():
    """A flag a reader must interpret is re-derived differently by each reader.

    Measured across four sessions in one afternoon, the published headroom read
    125, 1, 82, 7, 2, 62, 1, 53 and 0 -- oscillating faster than the interval
    between two coordinators consulting it. Publishing the spread lets a
    careful reader reach the right conclusion; publishing the conclusion means
    every reader reaches it.
    """
    import tempfile
    from pathlib import Path

    window = LaneWindow(
        readings=tuple(
            parse_lane_capacity(_metrics(running=10, occupancy=mean * 10 / _POOL))
            for mean in (19_000, 247_000, 14_000, 51_000)
        )
    )
    steady = LaneWindow(
        readings=tuple(
            parse_lane_capacity(_metrics(running=10, occupancy=mean * 10 / _POOL))
            for mean in (80_000, 84_000, 79_000, 82_000)
        )
    )

    with tempfile.TemporaryDirectory() as scratch:
        noisy = json.loads(
            write_lane_document(
                window.latest, Path(scratch) / "a.json", window=window
            ).read_text(encoding="utf-8")
        )
        calm = json.loads(
            write_lane_document(
                steady.latest, Path(scratch) / "b.json", window=steady
            ).read_text(encoding="utf-8")
        )

    assert noisy["sizing_verdict"] == "do-not-size"
    # The reason names the measured spread, so the verdict is checkable rather
    # than something the reader must take on trust.
    assert "14,000-247,000" in noisy["sizing_reason"]
    # And it names the fields that DID stay stable, so the reader is redirected
    # rather than merely blocked.
    assert "waiting" in noisy["sizing_reason"]

    assert calm["sizing_verdict"] == "usable"


def test_the_published_hit_rate_and_offload_health_reach_a_consumer():
    """The signal this lane turns on was parsed, rendered locally, never published.

    Prefix-cache eviction is the first symptom of KV pressure and appears long
    before preemption, so a consumer watching only the preemption counter
    learns nothing until far too late.
    """
    import tempfile
    from pathlib import Path

    capacity = parse_lane_capacity(_metrics(running=10, occupancy=0.267))

    with tempfile.TemporaryDirectory() as scratch:
        document = json.loads(
            write_lane_document(capacity, Path(scratch) / "lane.json").read_text(
                encoding="utf-8"
            )
        )

    assert document["prefix_hit_rate"] == round(capacity.prefix_hit_rate, 4)
    # No connector in this exposition, so the block is ABSENT rather than
    # present and zero: a missing key raises on a reader that assumed it, while
    # a zeroed one reads as a measured verdict of "nothing restored".
    assert "offload" not in document


def test_an_idle_lane_refuses_instead_of_publishing_the_ceiling():
    """The quiet lane is the dangerous case, because of WHEN it is read.

    A coordinator consults this field to decide how large a wave to resume,
    which is exactly when the lane is quiet. Publishing the safety ceiling
    there returns the MAXIMUM figure at the moment of the largest dispatch
    decision. Measured 2026-09-15: the highest value any session recorded, 96,
    was taken at zero occupancy four minutes after a restart, against a real
    budget of 35 a few minutes later.
    """
    import tempfile
    from pathlib import Path

    idle = parse_lane_capacity(_metrics(running=0, occupancy=0.0))

    with tempfile.TemporaryDirectory() as scratch:
        document = json.loads(
            write_lane_document(idle, Path(scratch) / "lane.json").read_text(
                encoding="utf-8"
            )
        )

    assert document["sizing_verdict"] == "do-not-size"
    assert "concurrent_requests" not in document
    assert "headroom" not in document
    # The ceiling is still recorded, one level down, so a reader debugging the
    # lane can see what the arithmetic would have said.
    assert document["withheld"]["concurrent_requests"] == idle.max_concurrent
    assert "quiet lane" in document["sizing_reason"]
    # And the reading itself is still `measured` -- the lane WAS read
    # successfully; what is withheld is advice, not data.
    assert document["state"] == "measured"
