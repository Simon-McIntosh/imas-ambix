"""Closed simulation of the local lane: workers, the router gate, and its slots.

The instrument exists so a change to the slot rule is tried against the lane's
measured behaviour before it reaches the router. Two things are therefore taken
from the lane rather than re-derived here, because a second copy of the worker
behaviour or of the slot arithmetic would drift from the first and the
simulation would then be measuring its own model:

* :func:`calibrate_rows` reads the router's own receipt rows. The row shape is
  the one :mod:`imas_ambix.agent.request_receipts` writes, so a committed
  excerpt is read by the same code that reads the live file.
* :class:`LaneSim` in its rule mode publishes slots by calling the router's
  ``_AdmissionLedger`` -- admitted and released on the simulated requests at
  the times the simulation serves them -- and never by copying its arithmetic.

A worker alternates between thinking (tools) and one request. A request waits
at the gate while ``width`` requests are in service, then generates its tokens
at the per-stream rate measured for the number in service. A worker ends after
its sampled number of requests; the slot rule decides how many replacements
launch, and a launch takes 45 to 120 s to reach its first request.

Every sampled quantity -- think time, completion tokens, time to first token,
requests per run, and the rate curve -- is a value the lane was observed to
produce, not a chosen constant.
"""

from __future__ import annotations

import collections
import gzip
import json
import random
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from imas_ambix.agent.router import ADMISSION_WINDOW_SECONDS, _AdmissionLedger

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from pathlib import Path

#: The step the simulation advances by. A gate wait is reported to this
#: resolution and no finer, so a wait shorter than it reads as no wait.
DT = 0.25
#: The lane document's publication period and the admission pass's period.
PUBLISH_EVERY = 30.0
ADMIT_EVERY = 30.0
#: The launches one admission pass can start.
LAUNCH_POOL = 8
#: A launched worker's range of times to its first request, in seconds.
LAUNCH_RAMP_S = (45.0, 120.0)
#: Simulated time discarded before figures are taken, so the pool's start-up
#: transient is not reported as the lane's steady state.
WARM_UP_S = 1800.0
#: The gate width the lane ran at over the measured window.
LANE_WIDTH = 12

#: The model's own default for the model name. Every committed excerpt carries
#: it, and a row naming another model describes a different lane.
LANE_MODEL = "deepseek-v4.1-flash"


@dataclass(frozen=True, slots=True)
class ReceiptRow:
    """One request as the router recorded it: when it was admitted, how long it
    was served, how long it waited at the gate, and what it produced."""

    run: str
    at: float
    duration: float
    wait: float
    ttft: float
    tokens: int


@dataclass(frozen=True, slots=True)
class Calibration:
    """The worker behaviour a simulation run draws on.

    ``think`` is the gap between one request's completion and the next one's
    admission at the gate, so it excludes the gate wait; ``work`` is a request's
    completion tokens and its time to first token; ``per_run`` is how many
    requests a run made; and ``rate`` is the per-stream completion rate at each
    number of requests in service.
    """

    think: tuple[float, ...]
    work: tuple[tuple[int, float], ...]
    per_run: tuple[int, ...]
    rate: Mapping[int, float]
    window: tuple[str, str]

    def with_zero_think(self) -> Calibration:
        """The same lane with every think time removed.

        This is the negative control's workload: workers ask again the instant
        a request completes, so the gate is fed as hard as the worker count
        allows and a slot rule that cannot see the wait floods it.
        """
        return Calibration(
            think=(0.0,),
            work=self.work,
            per_run=self.per_run,
            rate=self.rate,
            window=self.window,
        )


def parse_timestamp(value: str) -> float:
    """Seconds since the epoch for a receipt's ISO-8601 timestamp."""
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def read_receipts(path: str | Path) -> list[ReceiptRow]:
    """The lane's rows from one receipt file, plain or gzipped.

    Rows for another model, or without a duration, describe a request this
    simulation cannot place in time and are skipped; a request that declared no
    run id is grouped under one shared name, because the receipts record the
    caller's silence rather than inventing an identity.
    """
    opener = gzip.open if str(path).endswith(".gz") else open
    rows: list[ReceiptRow] = []
    with opener(path, "rt") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("model") != LANE_MODEL or row.get("duration_s") is None:
                continue
            rows.append(
                ReceiptRow(
                    run=row.get("run_id") or "unkeyed",
                    at=parse_timestamp(row["timestamp"]),
                    duration=float(row["duration_s"]),
                    wait=float(row.get("gate_wait_s") or 0.0),
                    ttft=float(row.get("time_to_first_token_s") or 0.0),
                    tokens=int(row.get("completion_tokens") or 0),
                )
            )
    return sorted(rows, key=lambda row: row.at)


def _quantile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[int(p * (len(ordered) - 1))]


def calibrate_rows(rows: Iterable[ReceiptRow], window: tuple[str, str]) -> Calibration:
    """Extract the worker behaviour from one window's receipt rows.

    Think times come from the gap between one request's completion and the next
    request's admission at the gate, so the gate wait already inside that gap is
    taken out. Requests per run and the per-stream rate are taken only from rows
    a run made inside the window: a run seen at an edge is truncated, and its
    length would understate a full run's.
    """
    rows = list(rows)
    by_run: dict[str, list[ReceiptRow]] = collections.defaultdict(list)
    for row in rows:
        by_run[row.run].append(row)
    think: list[float] = []
    per_run: list[int] = []
    for group in by_run.values():
        group.sort(key=lambda row: row.at)
        per_run.append(len(group))
        for previous, following in zip(group, group[1:], strict=False):
            gap = (following.at - following.wait) - (previous.at + previous.duration)
            if gap >= 0:
                think.append(round(gap, 2))

    # The in-service count a request saw, from the lane's own admission and
    # release times, so the rate curve is measured against observed occupancy.
    events = sorted(
        [(row.at, 1) for row in rows] + [(row.at + row.duration, -1) for row in rows]
    )
    stamps: list[float] = []
    levels: list[int] = []
    current = 0
    for at, delta in events:
        current += delta
        stamps.append(at)
        levels.append(current)

    rate: dict[int, list[float]] = collections.defaultdict(lambda: [0.0, 0.0])
    work: list[tuple[int, float]] = []
    for row in rows:
        generating = row.duration - row.ttft
        if row.tokens > 0:
            work.append((row.tokens, round(row.ttft, 2)))
        if row.tokens >= 50 and generating >= 1.0:
            import bisect

            index = max(0, bisect.bisect_right(stamps, row.at + row.duration / 2) - 1)
            level = min(levels[index], LANE_WIDTH)
            rate[level][0] += row.tokens
            rate[level][1] += generating
    curve = {
        level: round(total / seconds, 1)
        for level, (total, seconds) in sorted(rate.items())
        if seconds > 300
    }
    if not curve or not think or not work:
        raise ValueError("a window with no usable rows cannot calibrate the lane")
    return Calibration(
        think=tuple(think),
        work=tuple(work),
        per_run=tuple(per_run),
        rate=curve,
        window=window,
    )


def mean_live_runs(
    rows: Iterable[ReceiptRow],
    *,
    width: int = LANE_WIDTH,
    period: float = PUBLISH_EVERY,
) -> float:
    """The lane's mean live run count over the window, as the router publishes it.

    The rows are replayed through the router's own ``_AdmissionLedger`` -- each
    request admitted at the timestamp the receipt records and released when
    the request completed -- and the ledger is asked, once per publication
    period, how many runs are live. That is the figure the router would have
    published to a dispatcher for this window, so the worker count a
    reproduction is held at is the count the lane actually ran at, and it moves
    with any change to what the ledger calls live.
    """
    ordered = sorted(rows, key=lambda row: row.at)
    if not ordered:
        raise ValueError("a window with no rows has no live run count")
    start = ordered[0].at
    end = max(row.at + row.duration for row in ordered)
    ledger = _AdmissionLedger(window_seconds=ADMISSION_WINDOW_SECONDS)
    published: list[int] = []
    index = 0
    at = start
    while at <= end:
        while index < len(ordered) and ordered[index].run is None:
            index += 1
        while index < len(ordered) and ordered[index].at <= at:
            row = ordered[index]
            interval = ledger.admit(row.run, now=row.at)
            ledger.release(interval, now=row.at + row.duration)
            index += 1
        snapshot = ledger.snapshot(now=at, effective_width=width, verdict="open")
        published.append(int(snapshot["live_runs"]))
        at += period
    return sum(published) / len(published)


@dataclass(frozen=True, slots=True)
class RunResult:
    """What one simulated run delivered, after the warm-up is discarded."""

    tok_s: float
    mean_in_service: float
    at_width: float
    wait_p50: float
    wait_p90: float
    wait_p99: float
    workers: float


class LaneSim:
    """One run of the lane: workers, the gate, and a slot rule.

    ``workers`` holds the pool at a fixed count, which is how the model is
    checked against the measured lane. ``workers=None`` instead runs the
    router's rule: every publication period the ledger is asked how many slots
    the lane can take, and the admission pass launches up to that many.
    """

    def __init__(
        self,
        calibration: Calibration,
        *,
        width: int = LANE_WIDTH,
        seed: int = 0,
        hours: float = 4.5,
        workers: int | None = None,
        queue: int = 10**9,
    ) -> None:
        if width < 1:
            raise ValueError("a gate is at least one request wide")
        curve = {int(k): v for k, v in calibration.rate.items()}
        top = max(curve)
        low = min(curve)
        self.rng = random.Random(seed)
        self.calibration = calibration
        self.width = width
        self.fixed_workers = workers
        self.queue = queue
        self.t = 0.0
        self.end = hours * 3600.0
        self.workers: dict[int, dict[str, float | str]] = {}
        self.next_id = 0
        self.gate: collections.deque[int] = collections.deque()
        self.in_service: set[int] = set()
        self.launching: list[float] = []
        self.ledger = _AdmissionLedger(window_seconds=ADMISSION_WINDOW_SECONDS)
        self.open_intervals: dict[int, object] = {}
        self.slots = 0
        self.waits: list[float] = []
        self.level_time: collections.Counter[int] = collections.Counter()
        self.live_time = 0.0

        def rate_at(k: int) -> float:
            """The measured per-stream rate at ``k`` in service.

            A load the window never showed is interpolated towards the nearest
            measured one: above the top, the curve is held to the constant total
            output the plateau was measured at; below the bottom, the lowest
            measured rate is used, which is conservative for a lane that is
            rarely that empty.
            """
            if k in curve:
                return curve[k]
            if k > top:
                return curve[top] * top / k
            return curve[low]

        self.rate_at = rate_at

    # -- workers ---------------------------------------------------------
    def _launch(self) -> None:
        ramp = self.rng.uniform(*LAUNCH_RAMP_S)
        self.launching.append(self.t + ramp)

    def _start_worker(self) -> None:
        worker = self.next_id
        self.next_id += 1
        self.workers[worker] = {
            "left": float(self.rng.choice(self.calibration.per_run)),
            "state": "think",
            "until": self.t,
        }

    def _new_request(self, worker: int) -> None:
        tokens, ttft = self.rng.choice(self.calibration.work)
        self.workers[worker].update(
            state="wait",
            tokens=float(max(tokens, 1)),
            ttft=ttft,
            arrived=self.t,
        )
        self.gate.append(worker)

    # -- the router's slot rule ------------------------------------------
    def publish_slots(self) -> int:
        """The slots this instant offers, from the router's own ledger.

        The fixed-count mode is the reproduction check, not a rule: it holds
        the pool at the count it was asked for and offers exactly the room left
        in it.
        """
        if self.fixed_workers is not None:
            held = len(self.workers) + len(self.launching)
            self.slots = max(0, self.fixed_workers - held)
            return self.slots
        snapshot = self.ledger.snapshot(
            now=self.t, effective_width=self.width, verdict="open"
        )
        self.slots = int(snapshot["worker_slots"])
        return self.slots

    # -- one run ---------------------------------------------------------
    def run(self, start_workers: int | None = None) -> RunResult:
        start = start_workers if start_workers is not None else self.width
        for _ in range(start):
            self._start_worker()
        next_publish = next_admit = 0.0
        tokens_out = 0.0
        while self.t < self.end:
            if self.t >= next_publish:
                self.publish_slots()
                next_publish += PUBLISH_EVERY
            if self.t >= next_admit:
                for _ in range(min(self.slots, LAUNCH_POOL, self.queue)):
                    self._launch()
                    self.queue -= 1
                    self.slots -= 1
                next_admit += ADMIT_EVERY
            due = [at for at in self.launching if at <= self.t]
            self.launching = [at for at in self.launching if at > self.t]
            for _ in due:
                self._start_worker()
            for worker, state in list(self.workers.items()):
                if state["state"] == "think" and float(state["until"]) <= self.t:
                    if float(state["left"]) <= 0:
                        del self.workers[worker]
                        continue
                    state["left"] = float(state["left"]) - 1
                    self._new_request(worker)
            while self.gate and len(self.in_service) < self.width:
                worker = self.gate.popleft()
                state = self.workers[worker]
                state["state"] = "serve"
                self.in_service.add(worker)
                self.open_intervals[worker] = self.ledger.admit(str(worker), now=self.t)
                if self.t > WARM_UP_S:
                    self.waits.append(self.t - float(state["arrived"]))
            rate = self.rate_at(len(self.in_service)) if self.in_service else 0.0
            for worker in list(self.in_service):
                state = self.workers[worker]
                if float(state["ttft"]) > 0:
                    state["ttft"] = float(state["ttft"]) - DT
                    continue
                state["tokens"] = float(state["tokens"]) - rate * DT
                if self.t > WARM_UP_S:
                    tokens_out += rate * DT
                if float(state["tokens"]) <= 0:
                    self.in_service.discard(worker)
                    interval = self.open_intervals.pop(worker, None)
                    if interval is not None:
                        self.ledger.release(interval, now=self.t)
                    state["state"] = "think"
                    state["until"] = self.t + self.rng.choice(self.calibration.think)
            if self.t > WARM_UP_S:
                self.level_time[len(self.in_service)] += DT
                self.live_time += len(self.workers) * DT
            self.t += DT
        span = max(self.end - WARM_UP_S, DT)
        return RunResult(
            tok_s=tokens_out / span,
            mean_in_service=sum(k * v for k, v in self.level_time.items()) / span,
            at_width=self.level_time[self.width] / span,
            wait_p50=_quantile(self.waits, 0.5),
            wait_p90=_quantile(self.waits, 0.9),
            wait_p99=_quantile(self.waits, 0.99),
            workers=self.live_time / span,
        )


def mean_result(results: Iterable[RunResult]) -> RunResult:
    """The mean of several seeds' runs, field by field."""
    gathered = list(results)
    if not gathered:
        raise ValueError("no runs to average")
    fields = RunResult.__dataclass_fields__
    return RunResult(
        **{
            name: sum(getattr(result, name) for result in gathered) / len(gathered)
            for name in fields
        }
    )


def simulate_fixed(
    calibration: Calibration,
    workers: int,
    *,
    seeds: int = 3,
    hours: float = 4.5,
    width: int = LANE_WIDTH,
) -> RunResult:
    """Hold the pool at ``workers`` and average the run over ``seeds`` seeds."""
    return mean_result(
        LaneSim(calibration, width=width, seed=seed, hours=hours, workers=workers).run(
            workers
        )
        for seed in range(seeds)
    )
