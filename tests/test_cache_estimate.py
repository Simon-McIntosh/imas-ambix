"""The receipt-gap cache hit rate: chaining rules and the lifetime boundary."""

from __future__ import annotations

from datetime import UTC, datetime

from imas_ambix.agent.cache_estimate import (
    CacheEstimate,
    arrival_of,
    caller_of,
    caller_share,
    estimate_hit_rate,
)

MODEL = "deepseek-v4.1-flash"
OTHER_MODEL = "glm-5.3"


def row(
    timestamp: str,
    prompt_tokens: int,
    *,
    run_id: str | None = None,
    coordinator_session: str | None = None,
    model: str = MODEL,
    duration_s: float = 0.0,
    status: str = "completed",
) -> dict:
    """One request receipt as the record carries them, with only what we read."""
    return {
        "timestamp": timestamp,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "duration_s": duration_s,
        "status": status,
        "run_id": run_id,
        "coordinator_session": coordinator_session,
    }


def test_hit_rate_over_a_fixture_is_pinned() -> None:
    """A caller's re-send inside the lifetime is the whole of the hit share.

    Caller A sends 100 tokens, then 200 ten seconds later (a hit), then 50
    after a gap past the lifetime (a miss). Caller B's first request, caller
    A's first request to a second model, and a caller-less request all miss.
    Hit tokens are 200 of 1000, so h is 0.2.
    """
    rows = [
        row("2026-09-01T00:00:00Z", 100, run_id="A"),
        row("2026-09-01T00:00:10Z", 200, run_id="A"),
        row("2026-09-01T00:06:00Z", 50, run_id="A"),
        row("2026-09-01T00:01:00Z", 300, run_id="B"),
        row("2026-09-01T00:00:05Z", 100, run_id="A", model=OTHER_MODEL),
        row("2026-09-01T00:00:20Z", 250),
    ]
    estimate = estimate_hit_rate(rows, 300.0)

    assert isinstance(estimate, CacheEstimate)
    assert estimate.requests == 6
    assert estimate.hits == 1
    assert estimate.prompt_tokens == 1000
    assert estimate.hit_prompt_tokens == 200
    assert estimate.hit_rate == 0.2


def test_a_gap_exactly_equal_to_the_lifetime_is_a_miss() -> None:
    """A gap of exactly the lifetime counts as a miss, so the compare is strict."""
    rows = [
        row("2026-09-02T00:00:00Z", 500, run_id="X"),
        row("2026-09-02T00:05:00Z", 500, run_id="X"),
    ]
    estimate = estimate_hit_rate(rows, 300.0)

    assert estimate.requests == 2
    assert estimate.hits == 0
    assert estimate.hit_rate == 0.0


def test_a_request_with_no_caller_is_never_a_hit() -> None:
    """Two requests that would chain by time still miss with no caller."""
    rows = [
        row("2026-09-03T00:00:00Z", 400),
        row("2026-09-03T00:00:05Z", 600, coordinator_session=""),
    ]
    estimate = estimate_hit_rate(rows, 300.0)

    assert estimate.hits == 0
    assert estimate.prompt_tokens == 1000
    assert estimate.hit_rate == 0.0


def test_two_models_do_not_chain() -> None:
    """The same caller's request to a second model starts a fresh chain."""
    rows = [
        row("2026-09-04T00:00:00Z", 300, run_id="A", model=MODEL),
        row("2026-09-04T00:00:05Z", 300, run_id="A", model=OTHER_MODEL),
    ]
    estimate = estimate_hit_rate(rows, 300.0)

    assert estimate.hits == 0
    assert estimate.hit_rate == 0.0


def test_arrival_is_the_timestamp() -> None:
    """The arrival is the row's own timestamp; the duration does not move it.

    The router stamps a receipt at admission, so the second request below
    arrived 350 s after the first -- past the first request's 300 s lifetime --
    however long its own exchange then took. Subtracting its 100 s duration
    would place it 250 s after the first and manufacture a hit, so the direct
    arrival assertion and the miss together separate the two readings.
    """
    first = row("2026-09-05T00:00:00Z", 300, run_id="A", duration_s=0.0)
    second = row("2026-09-05T00:05:50Z", 300, run_id="A", duration_s=100.0)

    assert arrival_of(first) == datetime(2026, 9, 5, 0, 0, 0, tzinfo=UTC)
    assert arrival_of(second) == datetime(2026, 9, 5, 0, 5, 50, tzinfo=UTC)

    estimate = estimate_hit_rate([first, second], 300.0)

    assert estimate.hits == 0
    assert estimate.hit_rate == 0.0


def test_caller_prefers_a_non_empty_run_id() -> None:
    stamp = "2026-09-01T00:00:00Z"
    assert caller_of(row(stamp, 1, run_id="A", coordinator_session="S")) == "A"
    assert caller_of(row(stamp, 1, run_id="", coordinator_session="S")) == "S"
    assert caller_of(row(stamp, 1, run_id=None, coordinator_session=None)) is None
    assert caller_of(row(stamp, 1)) is None


def test_caller_share_counts_attributed_rows() -> None:
    rows = [
        row("2026-09-01T00:00:00Z", 1, run_id="A"),
        row("2026-09-01T00:00:01Z", 1, coordinator_session="S"),
        row("2026-09-01T00:00:02Z", 1),
        row("2026-09-01T00:00:03Z", 1),
    ]
    assert caller_share(rows) == 0.5


def test_hit_rate_is_zero_over_an_empty_set() -> None:
    estimate = estimate_hit_rate([], 300.0)
    assert estimate.requests == 0
    assert estimate.hit_rate == 0.0


def test_a_row_without_a_timestamp_cannot_be_timed() -> None:
    assert arrival_of(row("", 1)) is None
    assert arrival_of({"timestamp": "not-a-time", "duration_s": 0.0}) is None


def test_a_non_completed_row_is_never_scored_as_a_hit() -> None:
    """An aborted re-send is not a hit, though its prompt tokens still count.

    Caller A's second request arrived 50 s after its first -- inside the
    lifetime -- but its status is ``aborted``, so it neither scores nor chains.
    Its 500 prompt tokens remain in the denominator, so the figure is over all
    the traffic rather than only the completed part of it.
    """
    rows = [
        row("2026-09-06T00:00:00Z", 500, run_id="A"),
        row("2026-09-06T00:00:50Z", 500, run_id="A", status="aborted"),
    ]
    estimate = estimate_hit_rate(rows, 300.0)

    assert estimate.hits == 0
    assert estimate.hit_rate == 0.0
    assert estimate.prompt_tokens == 1000


def test_a_non_completed_row_does_not_break_the_chain() -> None:
    """A completed request chains past an intervening aborted one to the last completed.

    Caller A's third request is 200 s after its first -- inside the lifetime --
    and 100 s after a ``failed`` second request between them. The failed row is
    not a chain anchor, so the third measures against the first and is the one
    and only hit; were the failed request to chain, both it (100 s after the
    first) and the third (100 s after it) would score, for two hits.
    """
    rows = [
        row("2026-09-07T00:00:00Z", 300, run_id="A"),
        row("2026-09-07T00:01:40Z", 300, run_id="A", status="failed"),
        row("2026-09-07T00:03:20Z", 300, run_id="A"),
    ]
    estimate = estimate_hit_rate(rows, 300.0)

    assert estimate.hits == 1
    assert estimate.hit_prompt_tokens == 300


def test_a_row_naming_no_status_is_not_completed() -> None:
    """A row whose status the record does not state cannot score or chain.

    Two same-caller requests 10 s apart would chain if either were completed,
    but neither states a status, so nothing chains and h is zero.
    """
    rows = [
        {
            "timestamp": "2026-09-08T00:00:00Z",
            "model": MODEL,
            "prompt_tokens": 400,
            "run_id": "A",
        },
        {
            "timestamp": "2026-09-08T00:00:10Z",
            "model": MODEL,
            "prompt_tokens": 400,
            "run_id": "A",
        },
    ]
    estimate = estimate_hit_rate(rows, 300.0)

    assert estimate.hits == 0
    assert estimate.hit_rate == 0.0
