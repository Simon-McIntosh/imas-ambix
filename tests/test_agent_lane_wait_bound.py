"""The gate closes its slots while its own gate wait is above the bound."""

from __future__ import annotations

import logging
from pathlib import Path

from imas_ambix.agent.lane_sim import LaneSim, calibrate_rows, read_receipts
from imas_ambix.agent.router import (
    DEFAULT_WAIT_BOUND_SECONDS,
    _AdmissionLedger,
    _GenerationGate,
)

_RECEIPTS = Path(__file__).parent / "data" / "lane_receipts"
_NOW = 10_000.0


def _waits_ledger(now: float, waits: list[float]) -> _AdmissionLedger:
    """Requests whose gate waits are ``waits``, admitted inside the window."""
    ledger = _AdmissionLedger()
    ledger.admit("old", now=now - 400.0, arrived=now - 400.0)
    for index, wait in enumerate(waits):
        ledger.admit(
            f"r-{index}",
            now=now - 40.0,
            arrived=now - 40.0 - wait,
        )
    return ledger


def _document(ledger, now=_NOW, *, bound=30.0, verdict="open"):
    return ledger.snapshot(
        now=now,
        effective_width=12,
        verdict=verdict,
        wait_bound_seconds=bound,
    )


def test_the_bound_publishes_no_slots_and_just_under_it_does_not() -> None:
    """At the bound the gate shuts; one tenth below it the gate stays open."""
    at_bound = _document(_waits_ledger(_NOW, [30.0] * 10))
    under = _document(_waits_ledger(_NOW, [29.9] * 10))

    assert at_bound["gate_wait_p90_seconds"] == 30.0
    assert at_bound["wait_bound_seconds"] == 30.0
    assert at_bound["worker_slots_reason"] is None
    assert at_bound["worker_slots"] > 0

    assert under["gate_wait_p90_seconds"] == 29.9
    assert under["worker_slots_reason"] is None
    assert under["worker_slots"] > 0

    over = _document(_waits_ledger(_NOW, [30.1] * 10))
    assert over["gate_wait_p90_seconds"] == 30.1
    assert over["worker_slots_reason"] == "gate_wait"
    assert over["worker_slots"] == 0
    assert over["capacity"] == at_bound["capacity"]


def test_paused_and_wait_closed_documents_differ_only_in_the_reason() -> None:
    """Both closures zero the slots and state the one reason field."""
    wait_closed = _document(_waits_ledger(_NOW, [40.0] * 10))
    paused = _document(_waits_ledger(_NOW, [40.0] * 10), verdict="paused")

    assert wait_closed["worker_slots"] == 0
    assert paused["worker_slots"] == 0
    assert wait_closed["worker_slots_reason"] == "gate_wait"
    assert paused["worker_slots_reason"] == "paused"
    differences = {
        key: (wait_closed[key], paused[key])
        for key in wait_closed
        if wait_closed[key] != paused[key]
    }
    assert set(differences) == {"worker_slots_reason"}
    assert differences["worker_slots_reason"] == ("gate_wait", "paused")


def test_waits_older_than_the_window_do_not_keep_the_gate_shut() -> None:
    """A wait that has eased out of the five-minute window reopens the slots."""
    ledger = _AdmissionLedger()
    for index in range(10):
        ledger.admit(
            f"r-{index}",
            now=_NOW - 320.0,
            arrived=_NOW - 320.0 - 40.0,
        )
    document = _document(ledger)
    assert document["gate_wait_p90_seconds"] is None
    assert document["worker_slots_reason"] is None
    assert document["worker_slots"] > 0


def test_the_bound_is_read_from_the_gates_control_file(tmp_path) -> None:
    gate_file = tmp_path / "router-gate.json"
    gate_file.write_text('{"width": 12, "wait_bound_seconds": 12.0}', encoding="utf-8")
    gate = _GenerationGate(gate_file)
    assert gate.settings().wait_bound_seconds == 12.0
    document = gate.admission_document()
    assert document["wait_bound_seconds"] == 12.0
    assert document["worker_slots_reason"] is None


def test_a_malformed_bound_falls_back_to_the_default(tmp_path) -> None:
    gate_file = tmp_path / "router-gate.json"
    gate_file.write_text("", encoding="utf-8")
    gate_file.write_text('{"width": 12, "wait_bound_seconds": 0}', encoding="utf-8")
    gate = _GenerationGate(gate_file)
    assert gate.settings().wait_bound_seconds == DEFAULT_WAIT_BOUND_SECONDS


# --- the simulation: a margin the bound is needed at -----------------------

#: The margin the bound is demonstrated at: sized from the fed window's own
#: receipts it settles a pool where the bound-disabled run carries a five-minute
#: p90 wait over 45 s and the bound holds every settled window under it. The
#: margin is evidence, not a default, and it is re-measured when the rule the
#: simulation calls changes.
_SIM_MARGIN = 2.6
_SIM_WARM_UP_S = 1800.0
_SIM_SETTLED_S = _SIM_WARM_UP_S + 600.0


def _simulated(bound):
    rows = read_receipts(_RECEIPTS / "2026-11-18T0445-0640.jsonl.gz")
    calibration = calibrate_rows(rows, ("2026-11-18T04:45:00Z", "2026-11-18T06:40:00Z"))
    sim = LaneSim(
        calibration,
        seed=0,
        hours=4.5,
        margin=_SIM_MARGIN,
        wait_bound_seconds=bound,
    )
    sim.run()
    return sim


def _rank_p90(waits):
    """The window percentile, cut independently of the code under test.

    The guard decides closure with the shared estimator, so a window measured
    by that same function would move with a regression in it; the assertion
    below cuts its own nearest-rank figure instead.
    """
    ordered = sorted(waits)
    if not ordered:
        return None
    return ordered[int(0.9 * (len(ordered) - 1))]


def _window_p90s(sim):
    windows = []
    lo = _SIM_WARM_UP_S
    while lo + 300.0 <= sim.end:
        waits = [wait for at, wait in sim.wait_events if lo <= at < lo + 300.0]
        windows.append((lo, _rank_p90(waits)))
        lo += 300.0
    return windows


def test_the_bound_holds_a_high_margin_lane_under_committed_wait() -> None:
    """A margin whose unbounded run crosses 45 s is held under it by the bound."""
    unbounded = _simulated(None)
    bounded = _simulated(DEFAULT_WAIT_BOUND_SECONDS)

    assert unbounded.documents
    assert bounded.documents
    unbounded_p90 = [p for _lo, p in _window_p90s(unbounded) if p is not None]
    assert max(unbounded_p90) > 45.0

    settled = [
        p for lo, p in _window_p90s(bounded) if lo >= _SIM_SETTLED_S and p is not None
    ]
    assert settled and max(settled) < 45.0

    closed = [d for d in bounded.documents if d["worker_slots_reason"] == "gate_wait"]
    assert closed
    assert all(d["worker_slots"] == 0 for d in closed)
    assert not [d for d in unbounded.documents if d["worker_slots_reason"] is not None]


def test_the_gate_logs_its_settings_once_per_control_file_read(
    tmp_path, caplog
) -> None:
    """A re-read that finds the settings unchanged does not log them again."""
    config = tmp_path / "generation-gate.json"
    config.write_text('{"width": 12}', encoding="utf-8")
    gate = _GenerationGate(config)
    with caplog.at_level(logging.INFO, logger="imas_ambix.agent.router"):
        gate.settings()
        gate._next_config_refresh_at = 0.0
        gate.settings()
    logged = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("generation gate config")
    ]
    assert len(logged) == 1
