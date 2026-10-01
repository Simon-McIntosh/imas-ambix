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

from imas_ambix.agent.request_receipts import RUN_ID_HEADER
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


def test_two_live_runs_publish_null_even_while_congested() -> None:
    now = _NOW
    ledger = _AdmissionLedger()
    for index in range(2):
        busy = ledger.admit(f"r-{index}", now=now - 600.0)
        ledger.release(busy, now=now - 600.0 + 300.0)
    snapshot = ledger.snapshot(now=now, effective_width=16, verdict="congested")
    assert snapshot["live_runs"] == 2
    assert snapshot["requests_per_run"] is None
    assert snapshot["worker_slots"] is None


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


def test_ratio_rounding_to_zero_withholds_both_derived_figures() -> None:
    """A ratio that rounds to zero is withheld, withholds both derived figures.

    Three live runs each hold 0.1 s inside the window, so keyed busy time is
    0.3 s over a 900 s window for three runs -- 0.0001, which rounds to 0.0 at
    three decimals. A published 0.0 would divide by zero in the slot
    arithmetic, so ``requests_per_run`` and ``worker_slots`` are both withheld
    while ``live_runs`` still reports the three runs sharing the lane.
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
    assert snapshot["worker_slots"] is None


def test_empty_window_publishes_zero_runs_and_null_share() -> None:
    ledger = _AdmissionLedger()
    snapshot = ledger.snapshot(now=_NOW, effective_width=16, verdict="open")
    assert snapshot["live_runs"] == 0
    assert snapshot["unkeyed_share"] is None
    assert snapshot["requests_per_run"] is None
    assert snapshot["worker_slots"] is None


def test_fresh_ledger_withholds_derived_figures_until_history_elapses() -> None:
    start = _NOW
    ledger = _AdmissionLedger()
    for index in range(4):
        busy = ledger.admit(f"r-{index}", now=start)
        ledger.release(busy, now=start + 10.0)
    early = ledger.snapshot(now=start + 299.0, effective_width=16, verdict="open")
    assert early["live_runs"] == 4
    assert early["requests_per_run"] is None
    assert early["worker_slots"] is None
    ready = ledger.snapshot(now=start + 300.0, effective_width=16, verdict="open")
    assert ready["requests_per_run"] is not None
    assert ready["worker_slots"] is not None


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
