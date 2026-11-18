"""The lane simulation reproduces the lane its receipts measured.

The simulation is an instrument, so it is checked against the lane rather than
against itself: it is calibrated from committed receipt rows for the fed window
of 18 November 2026 -- 04:45 to 06:40 UTC, when a deep queue held the gate
nearly full -- and run with the worker count held at the count the router's own
ledger published over that window. It must then deliver the rate the excerpt
records and make requests wait the time it records.

The negative control is the same run with every think time removed: workers ask
again the instant a request completes, so a lane that cannot see its own wait
returns the plateau at a queueing delay far above the measured one, and the
reproduction's wait criterion fails on it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from imas_ambix.agent.lane_sim import (
    LANE_WIDTH,
    LaneSim,
    calibrate_rows,
    mean_live_runs,
    read_receipts,
    simulate_fixed,
)

RECEIPTS = Path(__file__).parent / "data" / "lane_receipts"

#: The window the plan's excerpt measures, and the two beside it that the same
#: extraction committed: the underfed hours before it.
WINDOWS = {
    "fed": (
        "2026-11-18T0445-0640.jsonl.gz",
        "2026-11-18T04:45:00Z",
        "2026-11-18T06:40:00Z",
    ),
    "early": (
        "2026-11-18T0000-0200.jsonl.gz",
        "2026-11-18T00:00:00Z",
        "2026-11-18T02:00:00Z",
    ),
    "starved": (
        "2026-11-18T0200-0445.jsonl.gz",
        "2026-11-18T02:00:00Z",
        "2026-11-18T04:45:00Z",
    ),
}

#: What the plan's excerpt records for the fed window.
MEASURED_TOK_S = 434.0
MEASURED_WAIT_P50_S = 2.5
#: What the done-when allows: the two figures are reported to these tolerances.
RATE_TOLERANCE = 0.05
WAIT_TOLERANCE_S = 2.0


def _fed_calibration():
    name, start, end = WINDOWS["fed"]
    rows = read_receipts(RECEIPTS / name)
    return rows, calibrate_rows(rows, (start, end))


def test_the_committed_excerpt_records_the_rate_the_plan_quotes():
    """The rows themselves deliver the excerpt's figure.

    Checked before the simulation is believed: a curve or think-time extraction
    that read the wrong rows would otherwise be indistinguishable from a model
    that reproduces the lane.
    """
    rows, _ = _fed_calibration()
    span = max(row.at + row.duration for row in rows) - min(row.at for row in rows)
    delivered = sum(row.tokens for row in rows) / span
    assert delivered == pytest.approx(MEASURED_TOK_S, rel=0.02)
    assert len({row.run for row in rows}) > 1


def test_simulation_reproduces_the_measured_lane():
    rows, calibration = _fed_calibration()
    workers = round(mean_live_runs(rows, width=LANE_WIDTH))
    result = simulate_fixed(calibration, workers)
    assert result.tok_s == pytest.approx(MEASURED_TOK_S, rel=RATE_TOLERANCE)
    assert abs(result.wait_p50 - MEASURED_WAIT_P50_S) <= WAIT_TOLERANCE_S


def test_zero_think_times_starve_the_gate_and_fail_the_reproduction():
    """Negative control: with no think time the lane cannot keep up.

    The plateau is still delivered, but only across a queueing delay far above
    the measured one, so the run is not a reproduction of the lane and the
    reproduction's own wait criterion fails on it.
    """
    rows, calibration = _fed_calibration()
    workers = round(mean_live_runs(rows, width=LANE_WIDTH))
    result = simulate_fixed(calibration.with_zero_think(), workers)
    assert result.wait_p50 > 10.0
    assert abs(result.wait_p50 - MEASURED_WAIT_P50_S) > WAIT_TOLERANCE_S


@pytest.mark.parametrize("name", sorted(WINDOWS))
def test_every_committed_window_calibrates(name):
    """Each committed excerpt is a whole workload the instrument can run.

    The three windows differ by half in think time, so a calibration that only
    worked for the fed window would have baked that window's load into the
    instrument.
    """
    filename, start, end = WINDOWS[name]
    rows = read_receipts(RECEIPTS / filename)
    calibration = calibrate_rows(rows, (start, end))
    assert calibration.think
    assert calibration.per_run
    assert calibration.work
    assert calibration.rate[LANE_WIDTH] > 0


def test_the_slot_count_is_the_router_ledgers_own():
    """The simulation publishes the router's slots, admitted and released here.

    The ledger the rule reads is the router's ``_AdmissionLedger``, fed with the
    simulated requests at the times the simulation served them, so the rule
    cannot drift into a copy that the router does not run.
    """
    rows, calibration = _fed_calibration()
    sim = LaneSim(calibration, seed=0, hours=0.5)
    sim.run(LANE_WIDTH)
    snapshot = sim.ledger.snapshot(
        now=sim.t, effective_width=LANE_WIDTH, verdict="open"
    )
    assert snapshot["live_runs"] > 0
    assert sim.publish_slots() == snapshot["worker_slots"]
    assert sim.rate_at(LANE_WIDTH) == calibration.rate[LANE_WIDTH]
