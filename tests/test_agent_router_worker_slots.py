"""Worker-slot figures published in the router's admission block.

The lane's admission block answers a dispatcher's question in the unit the
dispatcher works in: how many runs share the lane, how many requests one of them
holds, and how many more it can take. These tests drive the ledger the figures
are computed from with stub admissions, and one test drives the real request path
so the run identity is the one the caller declared in its header rather than
anything the router infers from the connection.
"""

from __future__ import annotations

import asyncio
import math
from typing import Any

from imas_ambix.agent.request_receipts import (
    COORDINATOR_SESSION_HEADER,
    RUN_ID_HEADER,
)
from imas_ambix.agent.router import Upstream, _AdmissionLedger
from tests.test_agent_router import _invoke, _server, _status
from tests.test_request_receipts import (
    _request_body,
    _router_with_receipts,
    _sse_engine,
)

_NOW = 10_000.0
# Fourteen run ids whose admitted request-seconds sum to 5,940, the fixture the
# section's done-when names: 13 x 424 + 428.
_FLEET_LENGTHS = [424.0] * 13 + [428.0]


def _fleet(now: float) -> _AdmissionLedger:
    ledger = _AdmissionLedger()
    for index, length in enumerate(_FLEET_LENGTHS):
        # Each interval opens exactly one window before ``now`` so the ledger
        # has observed the full window and the ratio divides by 900 s.
        busy = ledger.admit(f"r-{index}", now=now - 900.0)
        ledger.release(busy, now=now - 900.0 + length)
    return ledger


def test_fleet_publishes_live_runs_ratio_and_worker_slots() -> None:
    ledger = _fleet(_NOW)
    snapshot = ledger.snapshot(now=_NOW, effective_width=16, verdict="open")
    assert snapshot["live_runs"] == 14
    assert snapshot["requests_per_run"] == 0.471
    assert snapshot["worker_slots_basis"] == "measured"
    assert snapshot["worker_slots"] == 19
    assert snapshot["window_seconds"] == 900.0
    assert snapshot["samples"] == 14
    assert snapshot["unkeyed_share"] == 0.0


def test_ratio_averages_over_the_observed_span_not_the_full_window() -> None:
    """A ledger younger than the window divides by the span it has observed.

    The first admission is 320 s before the snapshot, so the divisor is the
    320 s observed, not the 900 s window: fourteen runs holding 2822.4 busy
    request-seconds publish 2822.4 / 320 / 14 = 0.63 requests per run and
    floor(16 / 0.63) - 14 = 11 worker slots. Dividing by the full window would
    publish 0.224 and 57, understating the per-run load and overstating the
    lane's capacity on every router younger than its window.
    """
    now = _NOW
    ledger = _AdmissionLedger()
    for index in range(14):
        busy = ledger.admit(f"r-{index}", now=now - 320.0)
        ledger.release(busy, now=now - 320.0 + 201.6)
    snapshot = ledger.snapshot(now=now, effective_width=16, verdict="open")
    assert snapshot["window_seconds"] == 900.0
    assert snapshot["observed_seconds"] == 320.0
    assert snapshot["live_runs"] == 14
    assert snapshot["requests_per_run"] == 0.63
    assert snapshot["worker_slots"] == 11


def test_observed_span_is_capped_at_the_window_from_above() -> None:
    """A router older than its window publishes the window, not its age.

    The first admission is 1200 s before the snapshot, so the process has
    observed more than its 900 s window. The published span is the window, and
    the ratio divides keyed busy request-seconds by that 900 s span: fourteen
    in-flight runs each holding the full window are 12600 busy seconds, so the
    ratio is round(12600 / 900 / 14, 3) = 1.0. Dividing by the 1200 s of process
    age instead would publish 0.75 and overstate the slots the lane can take,
    which is the defect the cap guards against, so the cap is pinned from above
    by the 1200 s first admission.
    """
    now = _NOW
    ledger = _AdmissionLedger()
    keyed_busy = 0.0
    for index in range(14):
        ledger.admit(f"r-{index}", now=now - 1200.0)
        keyed_busy += 900.0
    snapshot = ledger.snapshot(now=now, effective_width=16, verdict="open")
    assert snapshot["observed_seconds"] == snapshot["window_seconds"] == 900.0
    assert snapshot["live_runs"] == 14
    assert snapshot["requests_per_run"] == round(keyed_busy / 900.0 / 14, 3)
    assert snapshot["requests_per_run"] != round(keyed_busy / 1200.0 / 14, 3)


def test_published_slots_follow_from_the_published_ratio() -> None:
    """A dispatcher's own arithmetic on the published figures is the figure.

    Three runs each hold 720.009 s in the window. The busy seconds divide to
    0.80001 requests per run, published as 0.8; at width 16 the formula applied
    to the published ratio gives 17 slots, where the unrounded ratio would give
    16. So the count a reader derives from the two published figures is exactly
    the count the router published.
    """
    now = _NOW
    ledger = _AdmissionLedger()
    for index in range(3):
        busy = ledger.admit(f"r-{index}", now=now - 900.0)
        ledger.release(busy, now=now - 900.0 + 720.009)
    snapshot = ledger.snapshot(now=now, effective_width=16, verdict="open")
    assert snapshot["live_runs"] == 3
    assert snapshot["requests_per_run"] == 0.8
    assert snapshot["worker_slots"] == 17
    published = snapshot["requests_per_run"]
    assert snapshot["worker_slots"] == max(
        0, math.floor(16 / published) - snapshot["live_runs"]
    )


def test_request_opened_before_the_window_contributes_only_its_in_window_time() -> None:
    """The clip at the window start bounds a request's contribution.

    Each request opens 100 s before the window and closes 50 s inside it, so
    exactly 50 s falls within the window. Without the clip each would carry the
    full 150 s and the ratio would be three times larger, so the published
    ratio is the evidence that the clip is in force.
    """
    now = _NOW
    ledger = _AdmissionLedger()
    for index in range(3):
        busy = ledger.admit(f"r-{index}", now=now - 1000.0)
        ledger.release(busy, now=now - 850.0)
    snapshot = ledger.snapshot(now=now, effective_width=16, verdict="open")
    assert snapshot["live_runs"] == 3
    assert snapshot["requests_per_run"] == round(3 * 50.0 / 900.0 / 3, 3)
    assert snapshot["worker_slots"] == max(
        0,
        math.floor(16 / snapshot["requests_per_run"]) - snapshot["live_runs"],
    )


def test_congested_verdict_clamps_worker_slots_to_zero() -> None:
    ledger = _fleet(_NOW)
    snapshot = ledger.snapshot(now=_NOW, effective_width=16, verdict="congested")
    assert snapshot["requests_per_run"] == 0.471
    assert snapshot["worker_slots"] == 0
    for verdict in ("full", "paused"):
        assert (
            ledger.snapshot(now=_NOW, effective_width=16, verdict=verdict)[
                "worker_slots"
            ]
            == 0
        )


def test_two_live_runs_assume_the_ratio_and_clamp_to_zero_when_congested() -> None:
    """Below the ratio floor the slots are assumed, and a congested gate is 0.

    Two runs are fewer than the ratio rests on, so no measured ratio is formed,
    but the slots are still published from the assumed one request per run --
    and a congested verdict clamps them to zero whatever the assumed capacity.
    """
    now = _NOW
    ledger = _AdmissionLedger()
    for index in range(2):
        busy = ledger.admit(f"r-{index}", now=now - 600.0)
        ledger.release(busy, now=now - 600.0 + 300.0)
    snapshot = ledger.snapshot(now=now, effective_width=16, verdict="congested")
    assert snapshot["live_runs"] == 2
    assert snapshot["requests_per_run"] is None
    assert snapshot["worker_slots_basis"] == "assumed"
    assert snapshot["worker_slots"] == 0


def test_request_before_window_contributes_only_its_in_window_time() -> None:
    now = _NOW
    ledger = _AdmissionLedger()
    busy = ledger.admit(None, now=now - 1000.0)
    ledger.release(busy, now=now - 500.0)
    snapshot = ledger.snapshot(now=now, effective_width=16, verdict="open")
    # The interval ran 500 s before the window opened and 400 s inside it, so
    # only the 400 s inside counts -- and with no keyed run the share is all of
    # the busy time.
    assert snapshot["samples"] == 1
    assert snapshot["unkeyed_share"] == 1.0


def test_request_in_flight_counts_its_elapsed_time_to_publication() -> None:
    now = _NOW
    ledger = _AdmissionLedger()
    for index in range(3):
        ledger.admit(f"r-{index}", now=now - 600.0)
    snapshot = ledger.snapshot(now=now, effective_width=16, verdict="open")
    assert snapshot["live_runs"] == 3
    # The ledger is 600 s old, so each in-flight request counts its full elapsed
    # 600 s into an observed span of 600 s: three full-span occupancies for three
    # runs is 1.0.
    assert snapshot["observed_seconds"] == 600.0
    assert snapshot["requests_per_run"] == round(3 * 600.0 / 600.0 / 3, 3)


def test_unkeyed_request_raises_share_and_leaves_ratio_unchanged() -> None:
    now = _NOW
    ledger = _fleet(now)
    unkeyed = ledger.admit(None, now=now - 900.0)
    ledger.release(unkeyed, now=now)
    snapshot = ledger.snapshot(now=now, effective_width=16, verdict="open")
    assert snapshot["live_runs"] == 14
    assert snapshot["requests_per_run"] == 0.471
    assert snapshot["unkeyed_share"] == 900.0 / (5940.0 + 900.0)


def test_ratio_rounding_to_zero_withholds_the_ratio_but_assumes_the_slots() -> None:
    """A ratio that rounds to zero is withheld while the slots are assumed.

    Three live runs each hold 0.1 s inside the window, so keyed busy time is
    0.3 s over a 900 s window for three runs -- 0.0001, which rounds to 0.0 at
    three decimals. A published 0.0 would divide by zero in the slot
    arithmetic, so ``requests_per_run`` is withheld and the slots are published
    from the assumed one request per run: 16 - 3 = 13.
    """
    now = _NOW
    ledger = _AdmissionLedger()
    for index in range(3):
        busy = ledger.admit(f"r-{index}", now=now - 600.0)
        ledger.release(busy, now=now - 600.0 + 0.1)
    snapshot = ledger.snapshot(now=now, effective_width=16, verdict="open")
    assert snapshot["window_seconds"] == 900.0
    assert round(3 * 0.1 / snapshot["window_seconds"] / 3, 3) == 0.0
    assert snapshot["live_runs"] == 3
    assert snapshot["requests_per_run"] is None
    assert snapshot["worker_slots_basis"] == "assumed"
    assert snapshot["worker_slots"] == 13


def test_empty_window_publishes_zero_runs_and_the_whole_open_width() -> None:
    ledger = _AdmissionLedger()
    snapshot = ledger.snapshot(now=_NOW, effective_width=16, verdict="open")
    assert snapshot["live_runs"] == 0
    assert snapshot["unkeyed_share"] is None
    assert snapshot["requests_per_run"] is None
    assert snapshot["worker_slots_basis"] == "assumed"
    assert snapshot["worker_slots"] == 16


def test_fresh_ledger_assumes_slots_then_measures_once_history_elapses() -> None:
    start = _NOW
    ledger = _AdmissionLedger()
    for index in range(4):
        busy = ledger.admit(f"r-{index}", now=start)
        ledger.release(busy, now=start + 10.0)
    early = ledger.snapshot(now=start + 299.0, effective_width=16, verdict="open")
    assert early["live_runs"] == 4
    assert early["requests_per_run"] is None
    assert early["worker_slots_basis"] == "assumed"
    assert early["worker_slots"] == 12
    ready = ledger.snapshot(now=start + 300.0, effective_width=16, verdict="open")
    assert ready["requests_per_run"] is not None
    assert ready["worker_slots_basis"] == "measured"
    assert ready["worker_slots"] is not None


def test_assumed_slots_follow_from_one_request_per_run() -> None:
    """Two live runs with 100 s of history publish the assumed slots.

    Below both the ratio's run floor and its history floor, the ledger assumes
    one request per run, so at width 16 it offers 16 - 2 = 14 slots, with the
    ratio still null and the basis reading ``assumed``.
    """
    now = _NOW
    ledger = _AdmissionLedger()
    for index in range(2):
        busy = ledger.admit(f"r-{index}", now=now - 100.0)
        ledger.release(busy, now=now)
    snapshot = ledger.snapshot(now=now, effective_width=16, verdict="open")
    assert snapshot["live_runs"] == 2
    assert snapshot["requests_per_run"] is None
    assert snapshot["worker_slots_basis"] == "assumed"
    assert snapshot["worker_slots"] == 14


def test_paused_verdict_zeroes_the_assumed_slots() -> None:
    """A paused gate admits nothing, assumed basis or not."""
    now = _NOW
    ledger = _AdmissionLedger()
    busy = ledger.admit("r-0", now=now - 100.0)
    ledger.release(busy, now=now)
    snapshot = ledger.snapshot(now=now, effective_width=16, verdict="paused")
    assert snapshot["requests_per_run"] is None
    assert snapshot["worker_slots_basis"] == "assumed"
    assert snapshot["worker_slots"] == 0


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
    derived ratio stays null.
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


# --- The per-session fair share -------------------------------------------
#
# Slots are partitioned only when more than one session is working, so a session
# working alone takes the whole global figure. These ledgers are built directly,
# at effective width 12 and a published ratio of 0.5, so capacity is 24. Each
# session's in-flight runs contribute 450 busy seconds into a 900 s observed
# window, which is what makes the logistics work out to a clean ratio.
_CAPACITY = 24


def _session_ledger(
    now: float,
    *,
    workers: dict[str | None, int],
    idle: float | None = None,
) -> _AdmissionLedger:
    """A ledger holding ``workers[session]`` in-flight runs started 450 s ago.

    The zero-duration filler admission 900 s before ``now`` starts the observed
    span at the full window. When ``idle`` is given, one extra run is admitted
    and released ``idle`` seconds before ``now``, so its session claims a share
    only while that age is inside the working horizon.
    """
    ledger = _AdmissionLedger()
    filler = ledger.admit("r-intro", session="filler", now=now - 900.0)
    ledger.release(filler, now=now - 900.0)
    index = 0
    for session, count in workers.items():
        for _ in range(count):
            ledger.admit(f"r{index}", session=session, now=now - 450.0)
            index += 1
    if idle is not None:
        released = ledger.admit("idle-0", session="idle", now=now - idle)
        ledger.release(released, now=now - idle)
    return ledger


def test_one_working_session_is_offered_the_whole_global_figure() -> None:
    """A session working alone takes the server, and the fair share is capacity.

    Ten in-flight runs at width 12 and ratio 0.5 give the global figure
    24 - 10 = 14. One working session is not partitioned, so the session's
    figure is the global figure rather than a fixed fraction of capacity.
    """
    ledger = _session_ledger(_NOW, workers={"A": 10})
    snapshot = ledger.snapshot(now=_NOW, effective_width=12, verdict="open")
    assert snapshot["worker_slots"] == 14
    assert snapshot["active_sessions"] == 1
    assert snapshot["fair_share"] == _CAPACITY
    assert snapshot["borrow_reserve"] == 2
    assert snapshot["sessions"] == {"A": {"live_runs": 10, "worker_slots": 14}}
    assert snapshot["sessions"]["A"]["worker_slots"] == snapshot["worker_slots"]


def test_two_working_sessions_divide_the_capacity_and_borrow_around_it() -> None:
    """With two sessions the fair share is floor(24 / 2) = 12.

    Session A holds 2 workers and B holds 12, so the global figure is
    24 - 14 = 10. A is below its share and grows to the share, which the global
    figure caps at 10. B is above its share and may borrow beyond it, but must
    leave A room to add 2 workers at once, which caps B at 10 - 2 = 8. A new
    session is computed as a third with no workers, so its share is 8 and it is
    offered 8.
    """
    ledger = _session_ledger(_NOW, workers={"A": 2, "B": 12})
    snapshot = ledger.snapshot(now=_NOW, effective_width=12, verdict="open")
    assert snapshot["worker_slots"] == 10
    assert snapshot["active_sessions"] == 2
    assert snapshot["fair_share"] == 12
    assert snapshot["borrow_reserve"] == 2
    assert snapshot["sessions"]["A"] == {"live_runs": 2, "worker_slots": 10}
    assert snapshot["sessions"]["B"] == {"live_runs": 12, "worker_slots": 8}
    assert snapshot["new_session_worker_slots"] == 8


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
    assert live.snapshot(now=_NOW, effective_width=12, verdict="open")[
        "active_sessions"
    ] == 2


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


def test_congested_verdict_zeroes_every_per_session_figure() -> None:
    """A congested verdict clamps the global figure, and each share with it."""
    ledger = _session_ledger(_NOW, workers={"A": 2, "B": 12})
    snapshot = ledger.snapshot(now=_NOW, effective_width=12, verdict="congested")
    assert snapshot["worker_slots"] == 0
    for session in snapshot["sessions"].values():
        assert session["worker_slots"] == 0
    assert snapshot["sessions"]["A"]["live_runs"] == 2
    assert snapshot["new_session_worker_slots"] == 0


def test_session_figures_divide_the_assumed_capacity_when_the_ratio_is_withheld() -> (
    None
):
    """A withheld ratio still offers the session its share of the assumed one.

    Two live runs are below the minimum the ratio rests on, so the ratio is
    withheld and the slots are assumed at one request per run. The per-session
    figures divide that same assumed capacity: one session working alone takes
    the whole figure of 12 - 2 = 10.
    """
    ledger = _session_ledger(_NOW, workers={"A": 2})
    assumed = ledger.snapshot(now=_NOW, effective_width=12, verdict="open")
    assert assumed["live_runs"] == 2
    assert assumed["requests_per_run"] is None
    assert assumed["worker_slots_basis"] == "assumed"
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
