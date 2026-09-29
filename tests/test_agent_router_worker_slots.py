"""Worker-slot figures published in the router's admission block.

The lane's admission block answers a dispatcher's question in the unit the
dispatcher works in: how many runs share the lane, how many requests one of the
requests one of them holds, and how many more it can take. These tests drive the
ledger the figures are computed from with stub admissions, and one test drives
the real request path so the run identity is the one the caller declared in its
header rather than anything the router infers from the connection.
"""

from __future__ import annotations

import asyncio
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
        busy = ledger.admit(f"r-{index}", now=now - 600.0)
        ledger.release(busy, now=now - 600.0 + length)
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
    # Three 600 s occupancies over a 900 s window, one per run.
    assert snapshot["requests_per_run"] == round(3 * 600.0 / 900.0 / 3, 3)


def test_unkeyed_request_raises_share_and_leaves_ratio_unchanged() -> None:
    now = _NOW
    ledger = _fleet(now)
    unkeyed = ledger.admit(None, now=now - 900.0)
    ledger.release(unkeyed, now=now)
    snapshot = ledger.snapshot(now=now, effective_width=16, verdict="open")
    assert snapshot["live_runs"] == 14
    assert snapshot["requests_per_run"] == 0.471
    assert snapshot["unkeyed_share"] == 900.0 / (5940.0 + 900.0)


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
