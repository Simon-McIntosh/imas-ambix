"""The gate closes its slots while its own gate wait is above the bound."""

from __future__ import annotations

from imas_ambix.agent.router import (
    DEFAULT_WAIT_BOUND_SECONDS,
    _AdmissionLedger,
    _GenerationGate,
)

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
