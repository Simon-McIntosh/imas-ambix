"""Worker-slot figures published in the router's admission block.

The lane's admission block answers a dispatcher's question in the unit the
dispatcher works in: how many runs hold the lane now, how many the gate's width
can hold at the load those runs offer, and how many more it can take.

These tests drive the ledger the figures are computed from with stub
admissions, and two drive the real request path so the run identity is the one
the caller declared in its header rather than anything the router infers from
the connection.
"""

from __future__ import annotations

import asyncio
import math
from typing import Any

from imas_ambix.agent.lane_sim import per_session_reserve_offers
from imas_ambix.agent.request_receipts import (
    COORDINATOR_SESSION_HEADER,
    RUN_ID_HEADER,
)
from imas_ambix.agent.router import (
    DEFAULT_MARGIN,
    Upstream,
    _AdmissionLedger,
    _GenerationGate,
)
from tests.test_agent_router import _invoke, _server, _status
from tests.test_request_receipts import (
    _request_body,
    _router_with_receipts,
    _sse_engine,
)

_NOW = 10_000.0
# The working horizon the live count and the offered load both use.
HORIZON = 180.0


def _in_service_ledger(
    now: float,
    runs: int,
    *,
    elapsed: float = 300.0,
    session: str | None = None,
) -> _AdmissionLedger:
    """``runs`` requests still in service, each admitted ``elapsed`` ago.

    A continuously serving run's live span is its service time, so it offers a
    load of one: only a run that spends part of its span between requests
    offers less. These runs are for the ratio and horizon figures, not for a
    capacity that is meant to fall short of the gate.
    """
    ledger = _AdmissionLedger()
    for index in range(runs):
        ledger.admit(f"r-{index}", session=session, now=now - elapsed)
    return ledger


def _live_ledger(
    now: float,
    runs: int,
    *,
    session: str | None = None,
    wait: float = 0.0,
) -> _AdmissionLedger:
    """``runs`` steady runs: two 150 s requests 100 s apart.

    Each run asked 400 s ago, was served for 150 s, asked again 100 s later and
    was served for another 150 s, so its observed span is 400 s and it offered
    300 s of service: the load 0.75 that width 12 and the default margin turn
    into a capacity of exactly 24. ``wait`` is the time each request spent at the
    gate before admission.
    """
    ledger = _AdmissionLedger()
    for index in range(runs):
        first = ledger.admit(
            f"r-{index}",
            session=session,
            now=now - 400.0,
            arrived=now - 400.0 - wait,
        )
        ledger.release(first, now=now - 250.0)
        second = ledger.admit(
            f"r-{index}",
            session=session,
            now=now - 150.0,
            arrived=now - 150.0 - wait,
        )
        ledger.release(second, now=now)
    return ledger


#: The load each ``_live_ledger`` run offers, and the capacity it sizes.
LOAD = 300.0 / 400.0


def _capacity_at(margin: float, width: int = 12) -> int:
    return math.floor(margin * width / LOAD)


def test_ledger_sizes_capacity_from_the_load_live_runs_offer() -> None:
    """The done-when ledger: fifteen live runs and nine ended long ago.

    Each live run arrived 405 s before the snapshot, was served for 150 s and
    holds another request in service. The nine old runs ended more than the
    working horizon ago, so they hold no slot and contribute no load. Hand
    arithmetic on the same ledger: service is 15 * 150 s, the observed span runs
    from the first arrival to the last end, 405 s per run, the gate wait is
    15 * 10 s, so the load is 150 / 395 and the capacity at width 12 and the
    default margin is floor(1.5 * 15 * 395 / 150) = 47.
    """
    now = _NOW
    ledger = _AdmissionLedger()
    for index in range(9):
        busy = ledger.admit(f"old-{index}", now=now - 5000.0, arrived=now - 5000.0)
        ledger.release(busy, now=now - 4500.0)
    for index in range(15):
        busy = ledger.admit(f"live-{index}", now=now - 400.0, arrived=now - 405.0)
        ledger.release(busy, now=now - 350.0)
        ledger.admit(f"live-{index}", now=now - 100.0, arrived=now - 105.0)
    snapshot = ledger.snapshot(now=now, effective_width=12, verdict="open")

    service = 15 * 150.0
    # First arrival 405 s ago, last request ended at the snapshot.
    live_span = 15 * 405.0
    waited = 15 * 10.0
    by_hand = math.floor(DEFAULT_MARGIN * 12 / (service / (live_span - waited)))

    assert snapshot["live_runs"] == 15
    assert snapshot["capacity"] == by_hand
    by_hand_exact = DEFAULT_MARGIN * 12 / (service / (live_span - waited))
    assert abs(snapshot["capacity"] - by_hand_exact) <= 1
    assert snapshot["worker_slots"] == snapshot["capacity"] - 15


def test_a_run_ended_before_the_horizon_holds_no_slot() -> None:
    """The horizon boundary: 181 s old is out, 179 s old is in.

    Both runs released their only request before the snapshot, so the horizon is
    the only thing that keeps them live. A count that took every run seen in the
    averaging window would report both, which is the defect the horizon fixes.
    """
    now = _NOW
    expired = _AdmissionLedger()
    for index in range(4):
        busy = expired.admit(f"r-{index}", now=now - 181.0)
        expired.release(busy, now=now - 181.0)
    assert (
        expired.snapshot(now=now, effective_width=12, verdict="open")["live_runs"] == 0
    )

    live = _AdmissionLedger()
    for index in range(4):
        busy = live.admit(f"r-{index}", now=now - 179.0)
        live.release(busy, now=now - 179.0)
    snapshot = live.snapshot(now=now, effective_width=12, verdict="open")
    assert snapshot["live_runs"] == 4


def test_a_run_waiting_at_the_gate_holds_a_slot_without_an_interval() -> None:
    """A request still queued has no admission, so the queue is its only record.

    Without the gate's queue the run would be invisible until its turn came, and
    a dispatcher would be told the lane had more room than it has.
    """
    now = _NOW
    ledger = _live_ledger(now, 5)
    snapshot = ledger.snapshot(
        now=now,
        effective_width=12,
        verdict="open",
        waiting_runs={"queued-0", "queued-1", "r-0"},
    )
    assert snapshot["live_runs"] == 7
    assert snapshot["worker_slots"] == snapshot["capacity"] - 7


def test_published_slots_are_capacity_less_the_live_runs() -> None:
    """The figure a dispatch reads is the spare runs, not the capacity."""
    now = _NOW
    ledger = _live_ledger(now, 10)
    snapshot = ledger.snapshot(now=now, effective_width=12, verdict="open")
    assert snapshot["worker_slots_basis"] == "measured"
    assert snapshot["capacity"] == 24
    assert snapshot["worker_slots"] == 14
    assert snapshot["worker_slots"] == snapshot["capacity"] - snapshot["live_runs"]


def test_capacity_is_the_margin_times_the_width_over_the_load() -> None:
    """A dispatcher's own arithmetic on the published margin is the figure."""
    now = _NOW
    ledger = _live_ledger(now, 10)
    open_snapshot = ledger.snapshot(now=now, effective_width=12, verdict="open")
    for margin in (1.0, 1.25, 1.5):
        snapshot = ledger.snapshot(
            now=now, effective_width=12, verdict="open", margin=margin
        )
        assert snapshot["capacity"] == _capacity_at(margin)
        assert snapshot["worker_slots"] == snapshot["capacity"] - 10
    # Below margin 1.0 the ten live runs exceed the capacity and the lane offers
    # no room rather than a negative number.
    tight = ledger.snapshot(now=now, effective_width=12, verdict="open", margin=0.2)
    assert tight["capacity"] < 10
    assert tight["worker_slots"] == 0
    assert open_snapshot["margin"] == DEFAULT_MARGIN


def test_gate_wait_is_left_out_of_the_offered_load() -> None:
    """A lane whose runs wait does not read as a lane of lighter runs.

    Twelve steady runs each spent 60 s at the gate before each of their two
    admissions. Counting that 120 s out of the run's live span raises the load
    and drops the capacity from 27 to 20: waiting costs the lane workers.
    Counting the wait in would hold the capacity at 27 and admit more into a
    gate that is already making requests wait, which is the runaway the rule
    exists to avoid.
    """
    now = _NOW
    ledger = _live_ledger(now, 12, wait=60.0)
    snapshot = ledger.snapshot(now=now, effective_width=12, verdict="open")
    including_wait = ledger.snapshot(
        now=now, effective_width=12, verdict="open", duty_includes_wait=True
    )
    assert snapshot["capacity"] == 20
    assert including_wait["capacity"] == 27
    assert snapshot["capacity"] < including_wait["capacity"]


def test_measuring_duty_including_gate_wait_runs_away() -> None:
    """Negative control: the failing rule this section rules out.

    Fifteen runs wait 170 s at the gate for a 10 s request, so their whole live
    span is mostly queue time. Read as a duty over that span the lane looks
    almost idle and the rule admits a worker count past 200 at the default
    margin; read with the wait removed, the same runs offer a load of one and
    the capacity collapses to the width. The two readings are the same ledger
    and the same margin, so a test that could only ever see one of them is not
    testing the distinction.
    """
    now = _NOW
    ledger = _AdmissionLedger(min_history_seconds=0.0)
    for index in range(15):
        ledger.admit(f"q-{index}", now=now - 10.0, arrived=now - 180.0)
    measured = ledger.snapshot(now=now, effective_width=12, verdict="open")
    runaway = ledger.snapshot(
        now=now, effective_width=12, verdict="open", duty_includes_wait=True
    )
    assert runaway["capacity"] > 200
    assert measured["capacity"] < runaway["capacity"] / 10


def test_a_thin_record_publishes_the_gate_width_as_an_assumed_capacity() -> None:
    """With no load to state the gate's own width is the capacity, as a bound."""
    ledger = _AdmissionLedger()
    snapshot = ledger.snapshot(now=_NOW, effective_width=16, verdict="open")
    assert snapshot["live_runs"] == 0
    assert snapshot["requests_per_run"] is None
    assert snapshot["worker_slots_basis"] == "assumed"
    assert snapshot["capacity"] == 16
    assert snapshot["worker_slots"] == 16


def test_two_live_runs_assume_the_capacity_and_keep_it_when_congested() -> None:
    """Below the live-run floor, two runs still get the gate's width as capacity."""
    now = _NOW
    ledger = _live_ledger(now, 2)
    snapshot = ledger.snapshot(now=now, effective_width=16, verdict="congested")
    assert snapshot["live_runs"] == 2
    assert snapshot["requests_per_run"] is None
    assert snapshot["worker_slots_basis"] == "assumed"
    assert snapshot["capacity"] == 16
    assert snapshot["worker_slots"] == 16 - 2


def test_fresh_ledger_assumes_capacity_then_measures_once_history_elapses() -> None:
    start = _NOW
    ledger = _in_service_ledger(start, 4, elapsed=0.0)
    early = ledger.snapshot(now=start + 299.0, effective_width=16, verdict="open")
    assert early["worker_slots_basis"] == "assumed"
    ready = ledger.snapshot(now=start + 300.0, effective_width=16, verdict="open")
    assert ready["worker_slots_basis"] == "measured"
    assert ready["live_runs"] == 4
    # Four runs continuously in service offer a load of one, so the capacity is
    # the margin times the width: floor(1.5 * 16).
    assert ready["capacity"] == 24


def test_ratio_averages_over_the_observed_span_not_the_full_window() -> None:
    """A ledger younger than the window divides by the span it has observed.

    Fourteen in-service runs admitted 320 s before the snapshot hold 320 busy
    seconds each, so the ratio over the observed span is 1.0. Dividing by the
    full 900 s window would publish 0.356, and a reader reconciling the ratio
    with the busy time behind it would be reading the wrong denominator.
    """
    now = _NOW
    ledger = _in_service_ledger(now, 14, elapsed=320.0)
    snapshot = ledger.snapshot(now=now, effective_width=16, verdict="open")
    assert snapshot["window_seconds"] == 900.0
    assert snapshot["observed_seconds"] == 320.0
    assert snapshot["live_runs"] == 14
    assert snapshot["requests_per_run"] == 1.0
    assert snapshot["requests_per_run"] != round(14 * 320.0 / 900.0 / 14, 3)


def test_observed_span_is_capped_at_the_window_from_above() -> None:
    """A router older than its window publishes the window, not its age.

    Fourteen in-service runs admitted 1200 s ago contribute only their 900 s
    inside the window, so the ratio is 900 / 900 per run. Dividing by the 1200 s
    of process age would publish 0.75 and overstate the slots the lane can take.
    """
    now = _NOW
    ledger = _in_service_ledger(now, 14, elapsed=1200.0)
    snapshot = ledger.snapshot(now=now, effective_width=16, verdict="open")
    assert snapshot["observed_seconds"] == snapshot["window_seconds"] == 900.0
    assert snapshot["live_runs"] == 14
    assert snapshot["requests_per_run"] == 1.0
    assert snapshot["requests_per_run"] != 0.75


def test_requests_opened_before_the_window_contribute_only_their_in_window_time() -> (
    None
):
    """The clip at the window start bounds a request's contribution to the ratio.

    Each request opened 1000 s before the snapshot, so 900 s of it lies inside
    the 900 s window and 100 s lies before it. Without the clip the ratio would
    be 1000 / 900 = 1.111.
    """
    now = _NOW
    ledger = _in_service_ledger(now, 3, elapsed=1000.0)
    snapshot = ledger.snapshot(now=now, effective_width=16, verdict="open")
    assert snapshot["live_runs"] == 3
    assert snapshot["requests_per_run"] == round(3 * 900.0 / 900.0 / 3, 3)


def test_a_queueing_gate_still_offers_its_capacity_and_a_paused_one_none() -> None:
    """A congested or full gate holds its backlog; only a pause stops dispatch.

    Requests past the width wait at the gate rather than on the engine, so the
    slots stay the capacity less the live runs. A pause is the one verdict that
    admits nothing new.
    """
    now = _NOW
    ledger = _live_ledger(now, 10)
    open_snapshot = ledger.snapshot(now=now, effective_width=12, verdict="open")
    assert open_snapshot["worker_slots"] == 14
    for verdict in ("congested", "full"):
        snapshot = ledger.snapshot(now=now, effective_width=12, verdict=verdict)
        assert snapshot["worker_slots"] == open_snapshot["worker_slots"]
    paused = ledger.snapshot(now=now, effective_width=12, verdict="paused")
    assert paused["worker_slots"] == 0
    assert paused["capacity"] == open_snapshot["capacity"]


def test_unkeyed_share_rides_beside_the_run_keyed_figures() -> None:
    """An unkeyed consumer is reported as its share of busy time, not as a run."""
    now = _NOW
    ledger = _live_ledger(now, 14)
    unkeyed = ledger.admit(None, now=now - 60.0)
    ledger.release(unkeyed, now=now)
    snapshot = ledger.snapshot(now=now, effective_width=16, verdict="open")
    assert snapshot["live_runs"] == 14
    # 300 keyed busy seconds per run over a 400 s observed span.
    assert snapshot["requests_per_run"] == 0.75
    assert snapshot["unkeyed_share"] == 60.0 / (14 * 300.0 + 60.0)


def test_live_runs_counts_run_ids_from_the_request_header() -> None:
    """The run identity is the caller's declared header, never the connection.

    Every request here arrives from the same synthetic caller, so a figure keyed
    on the connection-level hint would report one run rather than fourteen.
    """

    async def exercise() -> None:
        engine = _sse_engine()
        async with (
            _server(engine) as engine_url,
            _router_with_receipts([Upstream(engine_url)], None) as app,
        ):
            for index in range(14):
                response = await _invoke(
                    app,
                    "POST",
                    "/v1/chat/completions",
                    _request_body(),
                    headers=[
                        (b"content-type", b"application/json"),
                        (RUN_ID_HEADER.encode(), f"r-{index}".encode()),
                    ],
                )
                assert _status(response) == 200
            snapshot: dict[str, Any] = app._generation_gate.admission_document()

        assert snapshot["live_runs"] == 14
        assert snapshot["samples"] == 14

    asyncio.run(exercise())


def test_unreadable_run_id_header_sequence_is_recorded_unkeyed() -> None:
    """A header the reader cannot use leaves the request in the unkeyed share.

    Every request carries a run id, but it is over-long, so it is not a usable
    identity. Each must land in ``unkeyed_share`` rather than being attributed
    to a run the caller never correctly declared, and with no keyed run the
    ratio stays null and no run holds a slot.
    """
    unreadable = b"r-" + b"9" * 200

    async def exercise() -> None:
        engine = _sse_engine()
        async with (
            _server(engine) as engine_url,
            _router_with_receipts([Upstream(engine_url)], None) as app,
        ):
            for _ in range(3):
                response = await _invoke(
                    app,
                    "POST",
                    "/v1/chat/completions",
                    _request_body(),
                    headers=[
                        (b"content-type", b"application/json"),
                        (RUN_ID_HEADER.encode(), unreadable),
                    ],
                )
                assert _status(response) == 200
            snapshot: dict[str, Any] = app._generation_gate.admission_document()

        assert snapshot["live_runs"] == 0
        assert snapshot["samples"] == 3
        assert snapshot["unkeyed_share"] == 1.0
        assert snapshot["requests_per_run"] is None

    asyncio.run(exercise())


# --- The margin is a setting in the gate's control file --------------------


def test_the_gate_publishes_the_margin_from_its_control_file(tmp_path) -> None:
    gate_file = tmp_path / "router-gate.json"
    gate_file.write_text('{"width": 12, "margin": 1.5}', encoding="utf-8")
    gate = _GenerationGate(gate_file)
    assert gate.settings().margin == 1.5
    document = gate.admission_document()
    assert document["margin"] == 1.5


def test_a_malformed_margin_does_not_leave_a_silently_different_gate(tmp_path) -> None:
    """A bad margin is a launch error, and the gate says so in the document.

    An unreadable control file falls back to the defaults rather than to a
    guessed margin, so the published margin is the one actually in force.
    """
    gate_file = tmp_path / "router-gate.json"
    gate_file.write_text('{"width": 12, "margin": -1}', encoding="utf-8")
    gate = _GenerationGate(gate_file)
    assert gate.settings().margin == DEFAULT_MARGIN
    assert gate.admission_document()["margin"] == DEFAULT_MARGIN


# --- The per-session fair share -------------------------------------------
#
# Slots are partitioned only when more than one session is working, so a session
# working alone takes the whole global figure. Each session's runs are the steady
# two-request runs of ``_live_ledger``: a 400 s span carrying 300 s of service,
# so the offered load is 0.75 and the capacity at width 12 and the default
# margin is exactly 24.
_CAPACITY = 24


def _session_ledger(
    now: float,
    *,
    workers: dict[str | None, int],
    idle: float | None = None,
) -> _AdmissionLedger:
    """A ledger holding ``workers[session]`` steady runs on the lane.

    The zero-duration filler admission 900 s before ``now`` starts the observed
    span at the full window. When ``idle`` is given, one extra run is admitted
    and released ``idle`` seconds before ``now``, so its session claims a share
    in proportion to the time its request was inside the working horizon.
    """
    ledger = _AdmissionLedger()
    filler = ledger.admit("r-intro", session="filler", now=now - 900.0)
    ledger.release(filler, now=now - 900.0)
    index = 0
    for session, count in workers.items():
        for _ in range(count):
            first = ledger.admit(f"r{index}", session=session, now=now - 400.0)
            ledger.release(first, now=now - 250.0)
            second = ledger.admit(f"r{index}", session=session, now=now - 150.0)
            ledger.release(second, now=now)
            index += 1
    if idle is not None:
        released = ledger.admit("idle-0", session="idle", now=now - idle)
        ledger.release(released, now=now - idle)
    return ledger


def test_one_working_session_is_offered_the_whole_global_figure() -> None:
    """A session working alone takes the server, and the fair share is capacity."""
    ledger = _session_ledger(_NOW, workers={"A": 10})
    snapshot = ledger.snapshot(now=_NOW, effective_width=12, verdict="open")
    assert snapshot["capacity"] == _CAPACITY
    assert snapshot["worker_slots"] == 14
    assert snapshot["active_sessions"] == 1
    assert snapshot["fair_share"] == _CAPACITY
    assert snapshot["borrow_reserve"] == 2
    assert snapshot["sessions"] == {"A": {"live_runs": 10, "worker_slots": 14}}


def test_two_working_sessions_divide_the_capacity_and_borrow_around_it() -> None:
    """With two sessions the fair share is floor(24 / 2) = 12.

    Session A holds 3 workers and B holds 5, so the global figure is
    24 - 8 = 16. The global figure less the one borrow reserve the lane holds
    offers each of them 14. A new session is computed as a third with no
    workers of its own, so its share is floor(24 / 3), and the global figure
    less that one reserve offers it 14 -- not the 12 a reserve held for each
    session short of its share would have counted.
    """
    ledger = _session_ledger(_NOW, workers={"A": 3, "B": 5})
    snapshot = ledger.snapshot(now=_NOW, effective_width=12, verdict="open")
    assert snapshot["worker_slots"] == 16
    assert snapshot["active_sessions"] == 2
    assert snapshot["fair_share"] == 12
    assert snapshot["borrow_reserve"] == 2
    assert snapshot["sessions"]["A"] == {"live_runs": 3, "worker_slots": 14}
    assert snapshot["sessions"]["B"] == {"live_runs": 5, "worker_slots": 14}
    assert snapshot["new_session_worker_slots"] == 14


def test_session_idle_past_the_horizon_stops_claiming_a_share() -> None:
    """A session whose last admission was 200 s ago stops counting.

    The idle session is 200 s old at the snapshot, and at 180 s the working
    horizon is short of the window, so only B is working and it is offered the
    global figure. The control at 100 s shows the same session inside the
    horizon is counted, so the boundary is the horizon and not the request.
    """
    expired = _session_ledger(_NOW, workers={"B": 4}, idle=200.0)
    snapshot = expired.snapshot(now=_NOW, effective_width=12, verdict="open")
    assert snapshot["active_sessions"] == 1
    assert set(snapshot["sessions"]) == {"B"}
    assert snapshot["sessions"]["B"]["worker_slots"] == snapshot["worker_slots"]

    live = _session_ledger(_NOW, workers={"B": 4}, idle=100.0)
    assert (
        live.snapshot(now=_NOW, effective_width=12, verdict="open")["active_sessions"]
        == 2
    )


def test_session_less_runs_form_one_unattributed_session() -> None:
    """Runs declared with no session are grouped under one session.

    Three runs with no session header still share the lane with any session
    that has one, so they are grouped together as ``unattributed`` rather than
    left out of the share. They are working alone here, so they take the whole
    global figure of 24 - 3 = 21.
    """
    ledger = _session_ledger(_NOW, workers={None: 3})
    snapshot = ledger.snapshot(now=_NOW, effective_width=12, verdict="open")
    assert snapshot["active_sessions"] == 1
    assert snapshot["sessions"] == {
        "unattributed": {"live_runs": 3, "worker_slots": 21}
    }
    assert snapshot["worker_slots"] == 21


def test_paused_verdict_zeroes_every_per_session_figure() -> None:
    """A paused verdict clamps the global figure, and each share with it."""
    ledger = _session_ledger(_NOW, workers={"A": 2, "B": 12})
    snapshot = ledger.snapshot(now=_NOW, effective_width=12, verdict="paused")
    assert snapshot["worker_slots"] == 0
    for session in snapshot["sessions"].values():
        assert session["worker_slots"] == 0
    assert snapshot["sessions"]["A"]["live_runs"] == 2
    assert snapshot["new_session_worker_slots"] == 0


def test_session_figures_divide_the_assumed_capacity_when_the_load_is_unstated() -> (
    None
):
    """A thin record still offers a session its share of the assumed capacity.

    Two live runs are below the minimum the load rests on, so capacity is the
    gate's own width, 12, and the per-session figures divide that same capacity:
    one session working alone takes the whole figure of 12 - 2 = 10.
    """
    ledger = _session_ledger(_NOW, workers={"A": 2})
    assumed = ledger.snapshot(now=_NOW, effective_width=12, verdict="open")
    assert assumed["live_runs"] == 2
    assert assumed["requests_per_run"] is None
    assert assumed["worker_slots_basis"] == "assumed"
    assert assumed["capacity"] == 12
    assert assumed["worker_slots"] == 10
    assert assumed["active_sessions"] == 1
    assert assumed["fair_share"] == 12
    assert assumed["borrow_reserve"] == 2
    assert assumed["new_session_worker_slots"] == 8
    assert assumed["sessions"] == {"A": {"live_runs": 2, "worker_slots": 10}}


def test_session_share_is_keyed_from_the_caller_declared_headers() -> None:
    """The session a request declared is the session its run is grouped under.

    The request path reads both identities from the caller's own headers, so
    the session that shares the lane is the one the caller named rather than
    anything the router infers from the connection. The history floor is
    lowered only so a router a second old still publishes a share to read.
    """

    async def exercise() -> None:
        engine = _sse_engine()
        async with (
            _server(engine) as engine_url,
            _router_with_receipts([Upstream(engine_url)], None) as app,
        ):
            for index in range(3):
                response = await _invoke(
                    app,
                    "POST",
                    "/v1/chat/completions",
                    _request_body(),
                    headers=[
                        (b"content-type", b"application/json"),
                        (RUN_ID_HEADER.encode(), f"r-{index}".encode()),
                        (COORDINATOR_SESSION_HEADER.encode(), b"s-alpha"),
                    ],
                )
                assert _status(response) == 200
            app._generation_gate._admissions._min_history = 0.0
            snapshot: dict[str, Any] = app._generation_gate.admission_document()

        assert snapshot["requests_per_run"] is not None
        assert set(snapshot["sessions"]) == {"s-alpha"}
        assert snapshot["sessions"]["s-alpha"]["live_runs"] == 3
        assert snapshot["active_sessions"] == 1

    asyncio.run(exercise())


def test_a_session_with_the_work_is_offered_the_slots_idle_sessions_leave() -> None:
    """Five sessions, 11 live runs and 5 free slots: the busy session gets 3.

    At capacity 16 the fair share is floor(16 / 5) = 3. Four of the five
    sessions are below the share, and the lane holds one reserve of 2 for them
    in total, so the session already past its share is offered the 3 slots left
    beyond that reserve rather than the 0 a reserve per session left it. Each
    session below its share is still offered its room to grow to its share: B,
    D and E hold 1 run and need 2 more, C holds 2 and needs 1.

    The capacity is 16 at width 8 because the steady runs offer the load 0.75
    the default margin sizes against: floor(1.5 * 8 / 0.75) = 16.
    """

    ledger = _session_ledger(
        _NOW, workers={"A": 6, "B": 1, "C": 2, "D": 1, "E": 1}
    )
    document = ledger.snapshot(now=_NOW, effective_width=8, verdict="open")

    assert document["capacity"] == 16
    assert document["live_runs"] == 11
    assert document["worker_slots"] == 5
    assert document["active_sessions"] == 5
    assert document["fair_share"] == 3
    assert document["sessions"]["A"]["live_runs"] == 6

    offered = {
        name: entry["worker_slots"] for name, entry in document["sessions"].items()
    }
    assert offered == {"A": 3, "B": 3, "C": 3, "D": 3, "E": 3}, offered
    # B, D and E need 2 runs to reach the share and C needs 1; each is offered
    # 3, so the reserve holds none of them below its room.
    for name, runs in (("B", 1), ("C", 2), ("D", 1), ("E", 1)):
        assert offered[name] >= 3 - runs, (name, offered)


def test_a_reserve_per_session_leaving_nothing_for_the_queued_session() -> None:
    """Negative control: the reserve held once per session starves the work.

    The same document under the rule the lane ran, which held
    ``BORROW_RESERVE`` slots for each session below its share: B, D and E each
    reserve 2 and C reserves 1, 7 slots against the 5 that are free, so the
    session holding the 6 runs is offered nothing. The two rules read the same
    document's capacity, share and live runs, so a check that could only ever
    see one offer is not measuring the reserve.
    """
    ledger = _session_ledger(
        _NOW, workers={"A": 6, "B": 1, "C": 2, "D": 1, "E": 1}
    )
    document = ledger.snapshot(now=_NOW, effective_width=8, verdict="open")
    assert document["fair_share"] == 3
    assert document["worker_slots"] == 5

    per_session = per_session_reserve_offers(document)
    # Hand arithmetic on the same runs: B and D and E reserve min(3-1, 2) = 2
    # each and C reserves min(3-2, 2) = 1.
    assert 3 * 2 + 1 == 7
    assert document["worker_slots"] < 3 * 2 + 1
    assert per_session["A"] == 0
    # The lane's own rule offers what the reserves took: the offer is not an
    # artefact of having been passed a different document.
    assert document["sessions"]["A"]["worker_slots"] >= 3
    assert per_session["A"] < document["sessions"]["A"]["worker_slots"]
