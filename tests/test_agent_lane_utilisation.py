"""Window utilisation and the below-width split the lane document publishes.

A count in use and a queue depth describe one instant. These figures describe a
window: how many requests were in service, how much of the time the gate ran at
its width, what the lane delivered and what it tops out at, and -- when it ran
below its width -- whether work was absent or the gate offered nothing.

The receipt-replay tests compare the router's own computation against the
figures `docs/evidence/fragments/the-local-lane-delivers-its-plateau/` records
for the same rows, so the document is checked against a measurement it did not
produce:

  occupancy-fed.log       window 04:45..06:40 UTC: mean 11.10, at width 68.2%
  rate-by-load.log        the same window: 434 tok/s delivered
  occupancy-starved.log   02:00..04:45 UTC: mean 1.66, at width 0.3%
                          (rate-by-load.log: 113 tok/s delivered)

`tests/data/lane_receipts_2026-11-18.jsonl.gz` holds the receipt excerpts for
the three windows of 18 November (00:00-02:00, 02:00-04:45, 04:45-06:40 UTC),
taken from the router's `requests-2026-11-18.jsonl` with the caller address and
every field a utilisation figure does not read removed.
"""

from __future__ import annotations

import datetime as dt
import gzip
import json
import time
from functools import lru_cache
from pathlib import Path

import pytest

from imas_ambix.agent.lane_utilisation import (
    RequestInterval,
    intervals_from_rows,
    utilisation_block,
)
from imas_ambix.agent.router import _AdmissionLedger, _GenerationGate

WIDTH = 12
FIXTURE = Path(__file__).with_name("data") / "lane_receipts_2026-11-18.jsonl.gz"
FED = ("2026-11-18T04:45:00Z", "2026-11-18T06:40:00Z", 11.10, 0.682, 434.0)
STARVED = ("2026-11-18T02:00:00Z", "2026-11-18T04:45:00Z", 1.66, 0.003, 113.0)


def _stamp(text: str) -> float:
    return dt.datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


@lru_cache(maxsize=1)
def _intervals() -> tuple[RequestInterval, ...]:
    with gzip.open(FIXTURE, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    return tuple(intervals_from_rows(rows, model="deepseek-v4.1-flash"))


def _window(start: str, end: str) -> dict[str, object]:
    t0, t1 = _stamp(start), _stamp(end)
    block = utilisation_block(
        _intervals(),
        width=WIDTH,
        now=t1,
        window_seconds=t1 - t0,
        horizon=t0,
    )
    return block["window"]


@pytest.mark.parametrize(
    ("start", "end", "mean", "share", "rate"),
    [FED, STARVED],
    ids=["fed", "starved"],
)
def test_document_fields_match_the_recorded_windows(
    start: str, end: str, mean: float, share: float, rate: float
) -> None:
    """The document's figures are the ones the evidence scripts record.

    The tolerances are the section's: the mean within 0.1 request, the share of
    time at width within two points, the delivered rate within 2%. A document
    that reported the instantaneous count, or divided the tokens by the wrong
    span, lands outside all three.
    """
    window = _window(start, end)
    span = _stamp(end) - _stamp(start)

    assert window["mean_in_service"] == pytest.approx(mean, abs=0.1)
    assert window["share_at_width"] == pytest.approx(share, abs=0.02)
    assert window["delivered_tokens_per_second"] == pytest.approx(rate, rel=0.02)
    # The plateau estimate is the lane's own: it must hold the delivered rate,
    # because a lane cannot deliver more than the rate its own receipts top out
    # at, and it must not run away from it either.
    assert window["delivered_tokens_per_second"] <= window["plateau_tokens_per_second"]
    # The two attributed parts are a partition of the below-width time, so they
    # sum to it within the section's one second per hour.
    parts = (
        window["slots_offered_not_taken_seconds"] + window["no_slots_offered_seconds"]
    )
    assert parts == pytest.approx(window["below_width_seconds"], abs=span / 3600)


def test_the_starved_window_shortfall_is_slots_offered_and_not_taken() -> None:
    """02:00 to 04:45 was underfed: the gate had slots and nobody took them.

    This is the split's whole point -- an underfed lane and a throttled one
    produce the same in-service count, and only the split tells them apart. At
    least 90% of the time below width must be attributed to offered slots, as
    the measured window was.
    """
    window = _window(*STARVED[:2])
    below = window["below_width_seconds"]
    assert below > 0
    assert window["slots_offered_not_taken_seconds"] / below >= 0.90


def test_the_split_moves_the_other_way_when_the_gate_offers_nothing() -> None:
    """The fed window is not attributed the same way as the starved one.

    A split that reported one part for both windows would pass the starved
    window's threshold without telling anything apart. The fed window ran with
    its runs close to the capacity its own load implies, so a real share of its
    below-width time is the gate offering nothing.
    """
    fed = _window(*FED[:2])
    starved = _window(*STARVED[:2])
    fed_none = fed["no_slots_offered_seconds"] / fed["below_width_seconds"]
    starved_none = starved["no_slots_offered_seconds"] / starved["below_width_seconds"]
    assert fed_none > starved_none


def test_a_paused_gate_attributes_its_shortfall_to_no_slots_offered() -> None:
    """Negative control: a paused gate offers nothing, whatever its occupancy.

    The gate is paused by an operator precisely so that no new run starts, so
    the time it spends below its width while paused is not an underfed lane. The
    control reads as offered exactly when the pause is ignored -- the guard this
    test exists to fail.
    """
    now = time.monotonic()
    ledger = _AdmissionLedger(window_seconds=600.0, hour_seconds=600.0)
    for index in range(2):
        busy = ledger.admit(f"r-{index}", now=now - 600.0, arrived=now - 600.0)
        ledger.release(busy, now=now, completion_tokens=500)
    ledger.note_paused(True, now=now - 600.0)

    window = ledger.utilisation(now=now, effective_width=WIDTH, paused=True)["window"]

    assert window["below_width_seconds"] == pytest.approx(600.0, abs=1.0)
    assert window["slots_offered_not_taken_seconds"] == 0.0
    assert window["no_slots_offered_seconds"] == window["below_width_seconds"]


def test_an_open_gate_attributes_the_same_shortfall_to_offered_slots() -> None:
    """The same ledger, unpaused, attributes the shortfall to offered slots.

    Without this, the control above passes on a split that answers "no slots"
    to everything.
    """
    now = time.monotonic()
    ledger = _AdmissionLedger(window_seconds=600.0, hour_seconds=600.0)
    for index in range(2):
        busy = ledger.admit(f"r-{index}", now=now - 600.0, arrived=now - 600.0)
        ledger.release(busy, now=now, completion_tokens=500)

    window = ledger.utilisation(now=now, effective_width=WIDTH)["window"]

    assert window["below_width_seconds"] == pytest.approx(600.0, abs=1.0)
    assert window["slots_offered_not_taken_seconds"] == window["below_width_seconds"]
    assert window["no_slots_offered_seconds"] == 0.0


def test_a_pause_that_ended_still_marks_the_seconds_the_gate_was_shut() -> None:
    """A pause is recorded when it happens, not read back from the latest state.

    Half of the window ran shut and half ran open; publishing the state at the
    end of the window would label all of it with whichever half came last.
    """
    now = time.monotonic()
    ledger = _AdmissionLedger(window_seconds=600.0, hour_seconds=600.0)
    busy = ledger.admit("r-0", now=now - 600.0, arrived=now - 600.0)
    ledger.release(busy, now=now, completion_tokens=500)
    ledger.note_paused(True, now=now - 600.0)
    ledger.note_paused(True, now=now - 500.0)
    ledger.note_paused(False, now=now - 300.0)

    window = ledger.utilisation(now=now, effective_width=WIDTH)["window"]

    assert window["paused_seconds"] == pytest.approx(300.0, abs=1.0)
    assert window["no_slots_offered_seconds"] == pytest.approx(300.0, abs=1.0)
    assert window["slots_offered_not_taken_seconds"] == pytest.approx(300.0, abs=1.0)


def test_the_lane_document_publishes_the_block_and_one_mean_in_service() -> None:
    """The §4 admission block and the §6 block report one record's mean.

    A reader comparing the mean in service with the runs the slots were sized
    from must be comparing one record. The gate's block carries the mean beside
    the utilisation block, and the two are the same figure from the same
    intervals rather than two measurements that happen to agree.
    """
    gate = _GenerationGate(None, default_width=WIDTH)
    now = time.monotonic()
    # Admitted inside the working horizon, so these are the runs the slots are
    # sized from: a run whose last admission is older than the horizon holds no
    # slot, and a document counting it would size from a run that is gone.
    for index in range(2):
        busy = gate._admissions.admit(
            f"r-{index}", session="s", now=now - 100.0, arrived=now - 120.0
        )
        gate._admissions.release(busy, now=now, completion_tokens=1000)

    document = gate.admission_document()
    window = document["utilisation"]["window"]

    assert document["live_runs"] == 2
    assert window["mean_in_service"] == document["mean_in_service"]
    assert window["mean_in_service"] == pytest.approx(2.0, abs=0.01)
    assert window["below_width_seconds"] > 0
    assert document["utilisation"]["last_hour"]["mean_in_service"] == (
        pytest.approx(2.0, abs=0.01)
    )


def test_the_last_hour_reaches_past_the_slot_window() -> None:
    """The hour block reads intervals the slot window has already dropped.

    Capacity is sized over the averaging window, so the ledger keeps intervals
    for the longer of the two spans. A ledger that pruned at the slot window
    would publish the last hour from the averaging window's tail alone, and the
    delivered figure would silently be a shorter span's.
    """
    now = time.monotonic()
    ledger = _AdmissionLedger(window_seconds=900.0, hour_seconds=3600.0)
    old = ledger.admit("r-old", now=now - 3000.0, arrived=now - 3000.0)
    ledger.release(old, now=now - 2990.0, completion_tokens=600)
    recent = ledger.admit("r-new", now=now - 60.0, arrived=now - 60.0)
    ledger.release(recent, now=now - 1.0, completion_tokens=100)

    block = ledger.utilisation(now=now, effective_width=WIDTH)

    assert block["window"]["completion_tokens"] == 100
    assert block["last_hour"]["completion_tokens"] == 700


def test_capacity_is_withheld_when_the_record_offers_no_load() -> None:
    """A record with no keyed run states no offered load, so no capacity.

    Sizing from a guessed load is how a slot figure becomes the largest
    available dispatch suggestion, so the refusal is the answer here.
    """
    now = time.monotonic()
    ledger = _AdmissionLedger(window_seconds=600.0)
    busy = ledger.admit(None, now=now - 100.0, arrived=now - 100.0)
    ledger.release(busy, now=now - 1.0, completion_tokens=10)

    block = ledger.utilisation(now=now, effective_width=WIDTH)["window"]

    assert block["capacity"] is None
    assert block["slots_offered_not_taken_seconds"] is None
    assert block["no_slots_offered_seconds"] is None
    assert block["below_width_seconds"] > 0


def test_the_window_reports_the_median_gate_wait_it_admitted() -> None:
    """The window carries the wait a caller ordinarily saw at the gate.

    The median is the estimator because one request held back through a long
    admission moves a mean far more than it moves a median, and the panel
    draws this figure beside the count waiting now. The waits are taken over
    the requests admitted in the window, so a request still queued -- which
    has no wait yet -- cannot pull the figure down.
    """
    now = 10_000.0
    waits = [0.0, 2.0, 7.0]
    intervals = [
        RequestInterval(
            arrived=now - 600.0 + index,
            admitted=now - 600.0 + index + wait,
            ended=now - 600.0 + index + wait + 1.0,
            completion_tokens=1,
            run_id=f"r-{index}",
        )
        for index, wait in enumerate(waits)
    ]
    # A request that arrived but has not been admitted yet has waited for no
    # measurable span: it is not part of the window's admitted population.
    intervals.append(
        RequestInterval(arrived=now - 5.0, admitted=now + 300.0, ended=now + 400.0)
    )

    block = utilisation_block(intervals, width=WIDTH, now=now, window_seconds=900.0)

    assert block["window"]["median_gate_wait_seconds"] == pytest.approx(2.0)


def test_an_even_count_of_waits_averages_the_two_middles() -> None:
    """The median is the estimator the name means, at an even count too.

    With four admitted requests the median is the mean of the two middles, not
    either one of them and not a nearest-rank pick: 1.0 and 2.0 average to 1.5.
    The window figures take their median from the standard library, as the
    decode bins do, so this pins the rule a reader is promised rather than the
    odd-count case alone, which almost every pick agrees on.
    """
    now = 10_000.0
    waits = [0.0, 1.0, 2.0, 9.0]
    intervals = [
        RequestInterval(
            arrived=now - 600.0 + index,
            admitted=now - 600.0 + index + wait,
            ended=now - 600.0 + index + wait + 1.0,
            completion_tokens=1,
            run_id=f"r-{index}",
        )
        for index, wait in enumerate(waits)
    ]

    block = utilisation_block(intervals, width=WIDTH, now=now, window_seconds=900.0)

    assert block["window"]["median_gate_wait_seconds"] == pytest.approx(1.5)
