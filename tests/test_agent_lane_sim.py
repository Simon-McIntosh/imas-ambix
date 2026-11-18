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

from functools import cache
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
from imas_ambix.agent.lane_utilisation import DEFAULT_MARGIN

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
    published = sim.publish_slots()
    snapshot = sim.ledger.snapshot(
        now=sim.t,
        effective_width=LANE_WIDTH,
        verdict="open",
        waiting_runs={str(worker) for worker in sim.gate},
        waiting_count=len(sim.launching),
    )
    assert snapshot["live_runs"] > 0
    assert published == snapshot["worker_slots"]
    assert sim.rate_at(LANE_WIDTH) == calibration.rate[LANE_WIDTH]


# --- The offered-load rule across four hours and both workloads -----------
#
# The rule is the router's own, called through the ledger the simulation feeds,
# so these runs move the lane with the code the lane runs. The plan's measure:
#
#   * at margins 1.0, 1.25 and 1.5, the pool's ten-minute means after the first
#     hour stay within 15% of its mean over the last hour, on both committed
#     think-time regimes;
#   * at margin 1.25 the rule delivers at least 97% of the simulated plateau
#     with a 90th-percentile gate wait under 15 s;
#   * negative control: with duty measured over the live span including the
#     gate wait, the same run and margin exceeds 200 workers.

MARGINS = (1.0, 1.25, 1.5)
#: The two think-time regimes the committed excerpts bracket: 6.8 s in the fed
#: window, 16.5 s in the starved window before it.
WORKLOADS = ("fed", "starved")
RULE_HOURS = 4.0
STABILITY_TOLERANCE = 0.15
PLATEAU_SHARE = 0.97
WAIT_BOUND_S = 15.0
RUNAWAY_WORKERS = 200.0


@cache
def _workload(name: str):
    filename, start, end = WINDOWS[name]
    return calibrate_rows(read_receipts(RECEIPTS / filename), (start, end))


@cache
def _simulated_plateau(name: str) -> float:
    """The best rate a fixed pool delivers on this workload: the plateau.

    Measured from the instrument rather than quoted, because the plateau a rule
    is held to must be one the same model can reach. A fixed pool is the
    strongest delivery this lane model offers, so the best of a spread of fixed
    counts is the ceiling the rule is compared against.
    """
    return max(
        simulate_fixed(_workload(name), workers, seeds=1, hours=RULE_HOURS).tok_s
        for workers in (8, 12, 16, 20, 24, 28, 32)
    )


def _rule_run(
    name: str, margin: float, *, seed: int = 0, hours: float = RULE_HOURS, **kwargs
):
    sim = LaneSim(
        _workload(name),
        seed=seed,
        hours=hours,
        margin=margin,
        **kwargs,
    )
    return sim, sim.run()


@pytest.mark.parametrize("name", WORKLOADS)
@pytest.mark.parametrize("margin", MARGINS)
def test_the_pool_holds_its_level_through_the_run(name: str, margin: float) -> None:
    """The rule's count settles instead of drifting, at every margin and workload.

    The count is read in ten-minute buckets after the first hour, so a rule still
    ramping at hour three shows as a bucket far from the last hour's mean rather
    than averaging away inside one figure. The reference is the run's own mean
    over the last hour, because that is the level the rule settled on.
    """
    sim, _ = _rule_run(name, margin)
    means = sim.worker_ten_minute_means()
    assert len(means) > 6
    last_hour = means[-6:]
    reference = sum(last_hour) / len(last_hour)
    assert reference > 0
    for reading in means:
        assert abs(reading - reference) <= STABILITY_TOLERANCE * reference, (
            name,
            margin,
            means,
        )


@pytest.mark.parametrize("name", WORKLOADS)
def test_the_rule_delivers_the_plateau_at_the_default_margin(name: str) -> None:
    """At the default margin the rule fills the lane without making it queue.

    Delivery is quoted against the best a fixed pool of the same model reaches,
    so a rule that settles low fails even if its wait is short, and one that
    fills the gate by queueing fails the wait bound. The margin is the gate's
    own default, so the figure measured here is the one a deployment gets
    without setting a margin at all.
    """
    _, result = _rule_run(name, DEFAULT_MARGIN)
    assert result.tok_s >= PLATEAU_SHARE * _simulated_plateau(name), (
        name,
        result.tok_s,
        _simulated_plateau(name),
    )
    assert result.wait_p90 < WAIT_BOUND_S, (name, result.wait_p90)


def test_measuring_duty_including_the_gate_wait_runs_the_pool_away() -> None:
    """Negative control: the failing rule admits without bound.

    Duty over the live span including the gate wait falls as the gate fills, so
    the rule reads a queueing lane as an underused one and answers with more
    workers. The lane is run for eight hours so the drift is measured past
    where the rule without the wait settles.
    """
    sim, _ = _rule_run("fed", 1.25, hours=RULE_HOURS * 2, duty_includes_wait=True)
    last_hour = sim.worker_ten_minute_means()[-6:]
    assert sum(last_hour) / len(last_hour) > RUNAWAY_WORKERS, last_hour
