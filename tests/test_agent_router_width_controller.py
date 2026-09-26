"""The gate's automatic width seeks the throughput knee rather than the pool.

The lane these tests simulate is not the production lane but its measured shape:
aggregate decoded throughput rises linearly with the number of concurrent
streams up to a knee and is flat past it, while the per-stream rate falls as the
extra streams share one engine. That shape is what makes the pool arithmetic a
ceiling rather than an answer -- the engine can hold far more streams than it can
decode at full speed -- so the controller has to find the knee by measuring.

Every surface below is driven through ``observe_lane`` with a cumulative decoded
token counter, exactly as the router's lane publisher feeds it.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest

from imas_ambix.agent import router as router_mod

# The synthetic surface. Aggregate throughput reaches PLATEAU_TOK_S at
# KNEE_WIDTH streams: below the knee one more stream adds its own
# PLATEAU_PER_STREAM tokens/s, and above it the same total is shared out, so the
# per-stream rate falls as width / KNEE_WIDTH.
PLATEAU_TOK_S = 800.0
KNEE_WIDTH = 12
PLATEAU_PER_STREAM: float = PLATEAU_TOK_S / KNEE_WIDTH
WINDOW_SECONDS = 30.0
NOISE_TOK_S = 60.0
SEED = 20260925


def _write_auto_gate(
    path: Path,
    *,
    occupancy_target: float = 0.90,
    width_floor: int = 16,
    width_cap: int = 36,
    wait_seconds: float = 1.0,
) -> None:
    path.write_text(
        json.dumps(
            {
                "width": "auto",
                "occupancy_target": occupancy_target,
                "width_floor": width_floor,
                "width_cap": width_cap,
                "wait_seconds": wait_seconds,
            }
        ),
        encoding="utf-8",
    )


def _aggregate_rate(width: int, rng: random.Random, per_stream: float | None) -> float:
    """One window's decoded tokens per second at a width, with its own noise."""
    if per_stream is None:
        per_stream = (
            PLATEAU_PER_STREAM if width <= KNEE_WIDTH else PLATEAU_TOK_S / width
        )
    return max(0.0, width * per_stream + rng.gauss(0.0, NOISE_TOK_S))


def _drive(
    gate: Any,
    clock: list[float],
    rng: random.Random,
    *,
    scrapes: int,
    start_width: int | None = None,
    per_stream: float | None = None,
    prefix_hit_rate: float = 0.99,
    prefix_sequence: list[float] | None = None,
    kv_occupancy: float = 0.30,
    waiting: int = 3,
    running: int | None = None,
) -> list[int]:
    """Feed the controller a run of scrapes and record the width in force.

    The width is read through ``settings()`` on every scrape, which is also what
    refreshes the pool ceiling the controller sizes under, so the recorded
    sequence is what the lane document would have published.
    """
    if start_width is not None:
        gate.settings()
        gate._controller_width = start_width
        gate._effective_width = start_width
        gate._next_config_refresh_at = 0.0
    tokens = 10_000_000
    widths: list[int] = []
    for _ in range(scrapes):
        width = gate.settings().width
        widths.append(width)
        tokens += int(
            round(
                _aggregate_rate(width, rng, per_stream) * WINDOW_SECONDS
            )
        )
        clock[0] += WINDOW_SECONDS
        gate.observe_lane(
            4_000_000,
            20_000,
            now=clock[0],
            generation_tokens=tokens,
            running=max(width, 1) if running is None else running,
            waiting=waiting,
            prefix_hit_rate=(
                prefix_sequence[len(widths) - 1]
                if prefix_sequence is not None
                else prefix_hit_rate
            ),
            kv_occupancy=kv_occupancy,
        )
    return widths


def _drive_repeating(
    gate: Any,
    clock: list[float],
    *,
    scrapes: int,
    start_width: int,
    aggregate_tok_s: float,
    running: int,
    advance_pattern: tuple[bool, ...] = (True, False),
    waiting: int = 0,
    prefix_hit_rate: float = 0.99,
    kv_occupancy: float = 0.30,
) -> list[int]:
    """Feed scrapes where the decoded-token counter moves only on some turns.

    ``aggregate_tok_s`` is the rate the engine really generates at, so a scrape
    that finally reads the odometer sees the whole accrual since the previous
    reading that moved. A turn whose slot in ``advance_pattern`` is False
    repeats its predecessor's value, which is what a read that lands before the
    counter advances returns.
    """
    gate.settings()
    gate._controller_width = start_width
    gate._effective_width = start_width
    gate._next_config_refresh_at = 0.0
    tokens = 10_000_000
    pending = 0
    widths: list[int] = []
    for index in range(scrapes):
        widths.append(gate.settings().width)
        clock[0] += WINDOW_SECONDS
        pending += 1
        if advance_pattern[index % len(advance_pattern)]:
            tokens += int(round(aggregate_tok_s * pending * WINDOW_SECONDS))
            pending = 0
        gate.observe_lane(
            4_000_000,
            20_000,
            now=clock[0],
            generation_tokens=tokens,
            running=running,
            waiting=waiting,
            prefix_hit_rate=prefix_hit_rate,
            kv_occupancy=kv_occupancy,
        )
    return widths


def _fake_clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    clock = [1000.0]
    monkeypatch.setattr(router_mod.time, "monotonic", lambda: clock[0])
    return clock


@pytest.mark.parametrize("start_width", [None, 4])
def test_width_controller_converges_to_the_plateau_start(
    tmp_path, monkeypatch, start_width
) -> None:
    """The knee is found from either side of it, on a noisy flat plateau.

    ``start_width=None`` begins where the pool arithmetic puts the width (the
    cap, 36); 4 begins below the knee. Both must arrive at the knee and stay
    there, which is the whole point of measuring rather than computing: the pool
    arithmetic cannot tell 12 from 36.
    """
    gate_file = tmp_path / "router-gate.json"
    # The floor is deliberately below the knee so the search is free to walk
    # through it, and the cap is the pool arithmetic's own ceiling.
    _write_auto_gate(gate_file, width_floor=4, width_cap=36)
    clock = _fake_clock(monkeypatch)
    gate = router_mod._GenerationGate(gate_file)

    widths = _drive(
        gate, clock, random.Random(SEED), scrapes=400, start_width=start_width
    )

    assert abs(widths[-1] - KNEE_WIDTH) <= 2, widths[-40:]


def test_width_controller_reaches_below_the_memory_rules_own_floor(
    tmp_path, monkeypatch
) -> None:
    """The pool floor sits above the knee, so it cannot bound the search.

    A gate left at its documented defaults floors the pool arithmetic at 16 --
    more streams than the engine decodes at full speed, measured. Under the
    throughput rule that figure is a property of the pool, not an opinion about
    the knee, so the controller must be able to resolve a width below it.
    """
    assert router_mod.GATE_CONTROLLER_WIDTH_FLOOR <= KNEE_WIDTH
    gate_file = tmp_path / "router-gate.json"
    _write_auto_gate(gate_file)
    clock = _fake_clock(monkeypatch)
    gate = router_mod._GenerationGate(gate_file)
    # Before any reading the documented floor stands, and the pool arithmetic
    # takes over as soon as one arrives.
    assert gate.settings().width == 16
    assert gate._controller_lower <= KNEE_WIDTH

    widths = _drive(gate, clock, random.Random(SEED), scrapes=400)

    assert abs(widths[-1] - KNEE_WIDTH) <= 2, widths[-40:]
    assert widths[-1] < 16


def test_width_controller_never_exceeds_the_memory_ceiling(
    tmp_path, monkeypatch
) -> None:
    """The pool arithmetic still says how much the engine can hold.

    With the ceiling below the knee the plateau is never reachable, and the
    controller must sit against the cap rather than climb past it.
    """
    gate_file = tmp_path / "router-gate.json"
    _write_auto_gate(gate_file, width_floor=2, width_cap=8)
    clock = _fake_clock(monkeypatch)
    gate = router_mod._GenerationGate(gate_file)

    widths = _drive(gate, clock, random.Random(SEED), scrapes=200)

    assert max(widths) == 8
    assert widths[-1] == 8


def test_width_controller_holds_while_nothing_waits(tmp_path, monkeypatch) -> None:
    """A width can only be scored while demand exceeds it.

    With nothing waiting on the gate, the reading describes the demand rather
    than the width, so every width would look alike and the search would wander.
    """
    gate_file = tmp_path / "router-gate.json"
    _write_auto_gate(gate_file, width_floor=4, width_cap=36)
    clock = _fake_clock(monkeypatch)
    gate = router_mod._GenerationGate(gate_file)

    widths = _drive(gate, clock, random.Random(SEED), scrapes=80, waiting=0)

    # The first scrape only establishes the pool arithmetic; from the second the
    # width is 36 and stays there, and nothing below moves it.
    assert set(widths[1:]) == {36}
    assert gate._controller_note == "holding: no demand above the current width"


def test_width_controller_steps_down_when_the_prefix_cache_stops_reusing(
    tmp_path, monkeypatch
) -> None:
    """A collapsed hit rate means the width has evicted the shared context."""
    gate_file = tmp_path / "router-gate.json"
    _write_auto_gate(gate_file, width_floor=4, width_cap=36)
    clock = _fake_clock(monkeypatch)
    gate = router_mod._GenerationGate(gate_file)
    gate.settings()

    _drive(
        gate,
        clock,
        random.Random(SEED),
        scrapes=router_mod.GATE_CONTROLLER_WINDOWS + 1,
        prefix_hit_rate=0.40,
    )

    assert gate._controller_width == 34
    assert gate._controller_note is not None
    assert gate._controller_note.startswith("stepping down: prefix hit rate")
    assert "0.95 floor" in gate._controller_note


def test_width_controller_steps_down_at_the_kv_ceiling(tmp_path, monkeypatch) -> None:
    """The pool ceiling is the blunter of the two health readings."""
    gate_file = tmp_path / "router-gate.json"
    _write_auto_gate(gate_file, width_floor=4, width_cap=36)
    clock = _fake_clock(monkeypatch)
    gate = router_mod._GenerationGate(gate_file)
    gate.settings()

    _drive(
        gate,
        clock,
        random.Random(SEED),
        scrapes=router_mod.GATE_CONTROLLER_WINDOWS + 1,
        kv_occupancy=0.75,
    )

    assert gate._controller_width == 34
    assert gate._controller_note is not None
    assert gate._controller_note.startswith("stepping down: KV occupancy")


def test_width_controller_never_keeps_a_width_one_stream_cannot_use(
    tmp_path, monkeypatch
) -> None:
    """Aggregate throughput holding is not enough; each stream has a floor."""
    gate_file = tmp_path / "router-gate.json"
    _write_auto_gate(gate_file, width_floor=4, width_cap=36)
    clock = _fake_clock(monkeypatch)
    gate = router_mod._GenerationGate(gate_file)
    gate.settings()

    _drive(
        gate,
        clock,
        random.Random(SEED),
        scrapes=router_mod.GATE_CONTROLLER_WINDOWS + 1,
        per_stream=8.0,
    )

    assert gate._controller_width == 34
    assert gate._controller_note is not None
    assert gate._controller_note.startswith("stepping down: ")
    assert "per stream" in gate._controller_note


def test_width_controller_ignores_idle_prefix_sentinel(tmp_path, monkeypatch) -> None:
    """Idle SGLang scrapes report zero hits without a cold prefix cache."""
    gate_file = tmp_path / "router-gate.json"
    _write_auto_gate(gate_file, width_floor=4, width_cap=36)
    clock = _fake_clock(monkeypatch)
    gate = router_mod._GenerationGate(gate_file)

    widths = _drive(
        gate,
        clock,
        random.Random(SEED),
        scrapes=100,
        start_width=36,
        running=0,
        waiting=0,
        prefix_hit_rate=0.0,
    )

    assert set(widths) == {36}


def test_width_controller_ignores_one_noisy_health_reading(
    tmp_path, monkeypatch
) -> None:
    """One loaded-minute prefix dip does not change the width."""
    monkeypatch.setattr(router_mod, "GATE_CONTROLLER_WINDOWS", 6)
    gate_file = tmp_path / "router-gate.json"
    _write_auto_gate(gate_file, width_floor=4, width_cap=36)
    clock = _fake_clock(monkeypatch)
    gate = router_mod._GenerationGate(gate_file)

    widths = _drive(
        gate,
        clock,
        random.Random(SEED),
        scrapes=7,
        start_width=20,
        prefix_sequence=[0.99, 0.99, 0.99, 0.80, 0.99, 0.99, 0.99],
    )

    assert widths == [20] * 7
    assert gate._controller_width >= 20
    assert not (gate._controller_note or "").startswith("stepping down:")


def test_width_controller_steps_down_once_per_sustained_health_window(
    tmp_path, monkeypatch
) -> None:
    """A sustained prefix collapse gets one step for each judged window."""
    monkeypatch.setattr(router_mod, "GATE_CONTROLLER_WINDOWS", 6)
    gate_file = tmp_path / "router-gate.json"
    _write_auto_gate(gate_file, width_floor=4, width_cap=36)
    clock = _fake_clock(monkeypatch)
    gate = router_mod._GenerationGate(gate_file)

    widths = _drive(
        gate,
        clock,
        random.Random(SEED),
        scrapes=13,
        start_width=20,
        prefix_hit_rate=0.80,
    )

    assert widths[:6] == [20] * 6
    assert widths[6:12] == [18] * 6
    assert widths[12:] == [16]
    assert gate._controller_width == 16


def test_width_controller_guards_loaded_lane_without_waiters(
    tmp_path, monkeypatch
) -> None:
    """A loaded lane can exceed its queue demand and still need protection."""
    monkeypatch.setattr(router_mod, "GATE_CONTROLLER_WINDOWS", 6)
    gate_file = tmp_path / "router-gate.json"
    _write_auto_gate(gate_file, width_floor=4, width_cap=36)
    clock = _fake_clock(monkeypatch)
    gate = router_mod._GenerationGate(gate_file)

    widths = _drive(
        gate,
        clock,
        random.Random(SEED),
        scrapes=18,
        start_width=36,
        running=30,
        waiting=0,
        prefix_sequence=[0.70] * 12 + [0.99] * 6,
    )

    assert widths[:6] == [36] * 6
    assert widths[6:12] == [34] * 6
    assert widths[12:] == [32] * 6
    assert gate._controller_width == 32


def test_switching_from_integer_width_to_auto_preserves_current_width(
    tmp_path, monkeypatch
) -> None:
    """An auto switch starts at the operator width, below the memory ceiling."""
    gate_file = tmp_path / "router-gate.json"
    gate_file.write_text(
        json.dumps(
            {
                "width": 14,
                "width_floor": 4,
                "width_cap": 36,
                "wait_seconds": 1.0,
            }
        ),
        encoding="utf-8",
    )
    clock = _fake_clock(monkeypatch)
    gate = router_mod._GenerationGate(gate_file)

    assert gate.settings().width == 14
    gate.observe_lane(
        4_000_000,
        20_000,
        now=clock[0],
        running=30,
        waiting=0,
    )
    gate_file.write_text(
        json.dumps(
            {
                "width": "auto",
                "width_floor": 4,
                "width_cap": 36,
                "wait_seconds": 1.0,
            }
        ),
        encoding="utf-8",
    )
    clock[0] += 2.0

    assert gate.settings().width == 14
    assert gate._controller_width == 14


def test_width_controller_discards_the_window_spanning_a_counter_reset(
    tmp_path, monkeypatch
) -> None:
    """A counter below its own previous value is a new engine, not a rate."""
    gate_file = tmp_path / "router-gate.json"
    _write_auto_gate(gate_file, width_floor=4, width_cap=36)
    clock = _fake_clock(monkeypatch)
    gate = router_mod._GenerationGate(gate_file)
    rng = random.Random(SEED)

    widths = _drive(gate, clock, rng, scrapes=300)
    settled = widths[-1]
    assert abs(settled - KNEE_WIDTH) <= 2

    # Mid-accrual the engine is replaced: the odometer restarts far below where
    # it was. The interval spanning the restart measures two processes and must
    # not enter the window.
    gate._controller_rates.append(999.0)
    clock[0] += WINDOW_SECONDS
    gate.observe_lane(
        4_000_000,
        20_000,
        now=clock[0],
        generation_tokens=500,
        running=settled,
        waiting=3,
        prefix_hit_rate=0.99,
        kv_occupancy=0.30,
    )

    assert gate._controller_rates == []
    assert gate._controller_note == "memory rule: generation counter reset"
    assert gate.settings().width == settled


def test_published_gate_block_carries_the_controller_mode_and_reason(
    tmp_path, monkeypatch
) -> None:
    """A reader of lane.json sees which rule chose the width, and why."""
    gate_file = tmp_path / "router-gate.json"
    _write_auto_gate(gate_file, width_floor=4, width_cap=36)
    clock = _fake_clock(monkeypatch)
    gate = router_mod._GenerationGate(gate_file)

    before = gate.snapshot()
    assert before["width_mode"] == "memory"
    assert isinstance(before["width_reason"], str)

    widths = _drive(gate, clock, random.Random(SEED), scrapes=400)
    assert abs(widths[-1] - KNEE_WIDTH) <= 2

    after = gate.snapshot()
    assert after["width_mode"] == "throughput"
    assert isinstance(after["width_reason"], str)

    # An operator-set integer explains itself, and the extra keys would change
    # the shape of the published block for a reader that never asked for them.
    gate_file.write_text(
        json.dumps({"width": 3, "wait_seconds": 1.0}), encoding="utf-8"
    )
    clock[0] += 2.0
    fixed = router_mod._GenerationGate(gate_file).snapshot()
    assert "width_mode" not in fixed
    assert "width_reason" not in fixed


def test_published_gate_block_reports_the_converged_width(
    tmp_path, monkeypatch
) -> None:
    """The width in the published block is the one the controller settled on."""
    gate_file = tmp_path / "router-gate.json"
    _write_auto_gate(gate_file, width_floor=4, width_cap=36)
    clock = _fake_clock(monkeypatch)
    gate = router_mod._GenerationGate(gate_file)

    _drive(gate, clock, random.Random(SEED), scrapes=400)

    assert gate.snapshot()["width"] == gate._controller_width


def test_width_controller_ignores_a_repeated_counter_reading(
    tmp_path, monkeypatch
) -> None:
    """A scrape that repeats its predecessor's odometer value is not a stall.

    The lane publisher reads the decoded-token counter on a fixed interval, and
    a read that lands before the counter advances returns its predecessor's
    value. Read as the difference over the interval, that is a real number --
    zero -- so scored as a per-stream rate it says every stream has stopped while
    the lane is decoding normally, and the floor watch steps the width down on
    it. The lane here generates at about 25 tokens/s per stream with ten streams
    running, well clear of the floor, and every other scrape repeats the counter.

    The two arms differ only in the rate the counter accrues at, which is what
    separates a repeated reading from a slow lane: at five tokens/s per stream
    the same reading shape still steps the width down.
    """
    monkeypatch.setattr(router_mod, "GATE_CONTROLLER_WINDOWS", 6)
    clock = _fake_clock(monkeypatch)

    # The gate holds no waiters, so nothing but the health watch can move the
    # width and the assertion ranges over that watch alone.
    fast_file = tmp_path / "fast.json"
    _write_auto_gate(fast_file, width_floor=4, width_cap=36)
    fast = router_mod._GenerationGate(fast_file)
    fast_widths = _drive_repeating(
        fast,
        clock,
        scrapes=2 * router_mod.GATE_CONTROLLER_WINDOWS + 1,
        start_width=20,
        aggregate_tok_s=250.0,
        running=10,
    )

    assert fast_widths == [20] * (2 * router_mod.GATE_CONTROLLER_WINDOWS + 1)
    assert fast._controller_width == 20
    assert not (fast._controller_note or "").startswith("stepping down:")

    # A repeated reading carries no per-stream rate, so the floor watch needs
    # enough advancing scrapes inside its window to reach its judgement; two
    # full windows covers the first reading, which has no predecessor to
    # difference against.
    slow_file = tmp_path / "slow.json"
    _write_auto_gate(slow_file, width_floor=4, width_cap=36)
    slow = router_mod._GenerationGate(slow_file)
    _drive_repeating(
        slow,
        clock,
        scrapes=2 * router_mod.GATE_CONTROLLER_WINDOWS,
        start_width=20,
        aggregate_tok_s=50.0,
        running=10,
    )

    assert slow._controller_width == 18
    assert slow._controller_note is not None
    assert slow._controller_note.startswith("stepping down: ")
    assert "per stream" in slow._controller_note


def test_width_controller_backs_off_when_the_counter_is_frozen(
    tmp_path, monkeypatch
) -> None:
    """A counter that stops moving while the lane runs is a stalled engine.

    A scrape that reads the counter before it advances repeats one reading, and
    the controller must ignore it; a lane that has stopped decoding repeats
    every reading, and the same silence must not hold the width open for ever.
    The two are separated by how long the silence lasts, so a counter that goes
    out once and then never moves still reaches the per-stream floor watch
    while ten streams are running.
    """
    monkeypatch.setattr(router_mod, "GATE_CONTROLLER_WINDOWS", 6)
    gate_file = tmp_path / "router-gate.json"
    _write_auto_gate(gate_file, width_floor=4, width_cap=36)
    clock = _fake_clock(monkeypatch)
    gate = router_mod._GenerationGate(gate_file)
    # Fourteen is the width the lane ran at when a frozen counter held it.
    gate.settings()
    gate._controller_width = 14
    gate._effective_width = 14
    gate._next_config_refresh_at = 0.0

    tokens = 10_000_000
    widths: list[int] = []
    for _ in range(2 * router_mod.GATE_CONTROLLER_WINDOWS + 1):
        widths.append(gate.settings().width)
        clock[0] += WINDOW_SECONDS
        # The odometer goes out once, then never moves while ten streams run.
        gate.observe_lane(
            4_000_000,
            20_000,
            now=clock[0],
            generation_tokens=tokens,
            running=10,
            waiting=4,
            prefix_hit_rate=0.99,
            kv_occupancy=0.28,
        )

    assert gate._controller_width < 14, widths
    assert gate._controller_note is not None
    assert gate._controller_note.startswith("stepping down: ")
    assert "per stream" in gate._controller_note


def test_repeated_readings_keep_the_health_watches_cadence(
    tmp_path, monkeypatch
) -> None:
    """A repeated counter reading still carries a fresh health sample.

    The prefix-hit and KV watches read the lane's demand, cache and pool, none
    of which the counter describes, so skipping a repeated reading whole would
    halve their cadence. A sustained prefix collapse must therefore step the
    width on the same scrape whether or not every other reading repeats.
    """
    monkeypatch.setattr(router_mod, "GATE_CONTROLLER_WINDOWS", 6)
    window = router_mod.GATE_CONTROLLER_WINDOWS
    clock = _fake_clock(monkeypatch)

    steady_file = tmp_path / "steady.json"
    _write_auto_gate(steady_file, width_floor=4, width_cap=36)
    steady = router_mod._GenerationGate(steady_file)
    steady_widths = _drive(
        steady,
        clock,
        random.Random(SEED),
        scrapes=window,
        start_width=20,
        waiting=0,
        prefix_hit_rate=0.90,
    )

    repeating_file = tmp_path / "repeating.json"
    _write_auto_gate(repeating_file, width_floor=4, width_cap=36)
    repeating = router_mod._GenerationGate(repeating_file)
    repeating_widths = _drive_repeating(
        repeating,
        clock,
        scrapes=window,
        start_width=20,
        aggregate_tok_s=250.0,
        running=10,
        waiting=0,
        prefix_hit_rate=0.90,
    )

    assert steady_widths == [20] * window
    assert repeating_widths == [20] * window
    assert steady._controller_width == 18
    assert repeating._controller_width == 18
    assert repeating._controller_note is not None
    assert repeating._controller_note.startswith("stepping down: prefix hit rate")
