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

from imas_ambix.agent.lane_utilisation import DEFAULT_MARGIN, percentile_or_none
from imas_ambix.agent.router import (
    ADMISSION_WINDOW_SECONDS,
    BORROW_RESERVE,
    DEFAULT_WAIT_BOUND_SECONDS,
    _AdmissionLedger,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping
    from pathlib import Path
    from typing import Any

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
#: The bucket the rule's worker count is read in, after the run's first hour.
WORKER_BUCKET_S = 600.0
#: The gate width the lane ran at over the measured window.
LANE_WIDTH = 12

#: The model's own default for the model name. Every committed excerpt carries
#: it, and a row naming another model describes a different lane.
LANE_MODEL = "deepseek-v4.1-flash"

#: The session a lane run without declared sessions belongs to: one session
#: holding every worker.
DEFAULT_SESSION = "lane"
#: The key the superseded rule's offers carry the figure for a session the
#: document has not yet seen working.
NEW_SESSION = "<new>"


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
    """The p-quantile of *values*, an empty series reading as no wait at all.

    The figure is the gate's own wait percentile, so this defers to the one
    estimator rather than holding a second copy of the nearest-rank rule.
    """
    value = percentile_or_none(values, p)
    return 0.0 if value is None else value


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
    """The lane's mean live run count over the window, for the reproduction.

    The rows are replayed through the router's own ``_AdmissionLedger`` -- each
    request admitted at the timestamp the receipt records and released when the
    request completed -- and the ledger is asked, once per publication period,
    how many runs are live.

    The claim is the averaging window, not the router's shorter working horizon:
    a run counts while a request of it is in service or ended inside the window,
    which is the population of runs whose requests the window's records show.
    That is the pool a reproduction of the lane is held at. The lane publishes
    the working-horizon count, which is smaller, because a run that stopped
    asking for longer than that holds no slot; the two readings are deliberately
    different and neither is a measurement of the other.
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
        snapshot = ledger.snapshot(
            now=at,
            effective_width=width,
            verdict="open",
            claim_seconds=ADMISSION_WINDOW_SECONDS,
        )
        published.append(int(snapshot["live_runs"]))
        at += period
    return sum(published) / len(published)


@dataclass(frozen=True, slots=True)
class Session:
    """One session's share of a simulated lane.

    ``workers`` is how many it holds when the run starts; ``queue`` is how many
    launches it has waiting behind them, and ``None`` is the unbounded queue of
    a session with more work than the lane can take. ``hold`` keeps the starting
    workers working for the whole run, which is what a session with runs that
    never end and nothing queued looks like: it holds its runs and takes none of
    the slots it is offered.

    A session launches only from the offer the published document makes it, so
    the rule's per-session figures are what the run is measured with rather
    than an accounting the run keeps for itself.
    """

    workers: int
    queue: int | None = None
    hold: bool = False


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


def _steady(_t: float) -> float:
    """The calibration's own workload, unchanged for the whole run."""
    return 1.0


class LaneSim:
    """One run of the lane: workers, the gate, and a slot rule.

    ``workers`` holds the pool at a fixed count, which is how the model is
    checked against the measured lane. ``workers=None`` instead runs the
    router's rule: every publication period the ledger is asked how many slots
    the lane can take, and the admission pass launches up to that many.

    ``sessions`` divides the rule's pool between sessions, each named with the
    workers it holds and the launches it has queued. A session launches only up
    to the offer the published document makes it, so a lane of several sessions
    is measured with the same arithmetic the router publishes. Without
    ``sessions`` every worker belongs to one unnamed session, which is the lane
    run as a single session.
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
        margin: float = DEFAULT_MARGIN,
        duty_includes_wait: bool = False,
        wait_bound_seconds: float | None = DEFAULT_WAIT_BOUND_SECONDS,
        think_scale: Callable[[float], float] | None = None,
        sessions: Mapping[str, Session] | None = None,
        per_session_reserve: bool = False,
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
        self.margin = margin
        # The counterfactual the rule's negative control runs: duty measured
        # over the live span including the gate wait, so a queued lane reads as
        # a light one and the rule admits without bound.
        self.duty_includes_wait = duty_includes_wait
        # The gate's own wait bound, so the simulation exercises the router's
        # closure rather than a copy of it. ``None`` is the negative control.
        self.wait_bound_seconds = wait_bound_seconds
        # How hard the workers ask at instant t: a fleet that starts running
        # shorter turns offers more load, which is what a dropped think time
        # models. The default is the calibration's own workload.
        self.think_scale = think_scale if think_scale is not None else _steady
        # The counterfactual this section measures: the reserve held once for
        # every session below its share instead of once for the lane.
        self.per_session_reserve = per_session_reserve
        self.sessions: dict[str, Session] = (
            {DEFAULT_SESSION: Session(workers=0, queue=queue)}
            if sessions is None
            else dict(sessions)
        )
        self.explicit_sessions = sessions is not None
        # What each session still has queued. ``None`` is unbounded and 0 means
        # the session launches nothing more: the runs it holds keep asking, but
        # nothing replaces them.
        self.queues: dict[str, int | None] = {
            name: (10**9 if self.fixed_workers is not None else session.queue)
            for name, session in self.sessions.items()
        }
        # Each session's offer from the last published document.
        self.offers: dict[str, int] = {}
        self.t = 0.0
        self.end = hours * 3600.0
        self.workers: dict[int, dict[str, float | str]] = {}
        self.next_id = 0
        self.gate: collections.deque[int] = collections.deque()
        self.in_service: set[int] = set()
        self.launching: list[tuple[float, str]] = []
        self.ledger = _AdmissionLedger(window_seconds=ADMISSION_WINDOW_SECONDS)
        self.open_intervals: dict[int, object] = {}
        self.slots = 0
        self.capacity = 0
        self.waits: list[float] = []
        # Each published admission document, so a window's slots and the reason
        # they were shut can be read after the run rather than only at its end.
        self.documents: list[dict[str, object]] = []
        # Every admitted request's wait with the instant it was admitted, which
        # is the record a five-minute window's percentile is cut from.
        self.wait_events: list[tuple[float, float]] = []
        self.level_time: collections.Counter[int] = collections.Counter()
        self.live_time = 0.0
        # The rule's worker count in ten-minute buckets after the warm-up, so a
        # run's settling can be read as a trend rather than only as one mean: a
        # rule still ramping at hour three is not a rule that holds its count.
        self.worker_bucket_time: collections.Counter[int] = collections.Counter()
        self.worker_bucket_area: collections.Counter[int] = collections.Counter()

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
    def _launch(self, session: str) -> None:
        ramp = self.rng.uniform(*LAUNCH_RAMP_S)
        self.launching.append((self.t + ramp, session))

    def _start_worker(self, session: str = DEFAULT_SESSION) -> None:
        worker = self.next_id
        self.next_id += 1
        self.workers[worker] = {
            "left": float(self.rng.choice(self.calibration.per_run)),
            "state": "think",
            "until": self.t,
            "session": session,
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
        in it. In the rule mode the gate's queue is passed to the ledger as the
        waiting runs it is: a worker whose request waits has not been admitted,
        so the record alone would count the lane as emptier than it is and the
        rule would answer a queue with more workers.
        """
        if self.fixed_workers is not None:
            held = len(self.workers) + len(self.launching)
            self.slots = max(0, self.fixed_workers - held)
            # The fixed pool is its own rule: every session is offered the room
            # left in it, which the one session holding the pool takes. It is
            # not shared out per session, because a fixed pool is a reproduction
            # of a lane held at a count and not a lane moving under the rule.
            self.offers = {name: self.slots for name in self.sessions}
            return self.slots
        snapshot = self.ledger.snapshot(
            now=self.t,
            effective_width=self.width,
            verdict="open",
            margin=self.margin,
            waiting_runs={str(worker) for worker in self.gate},
            # A worker launched and not yet asking holds a slot the record
            # cannot show: it has no id and no request, so only its count can be
            # claimed. Without it the rule admits that slot again on the next
            # publication period and the pool overshoots the capacity.
            waiting_count=len(self.launching),
            # The launched workers' sessions, so the sessions the document
            # publishes count the runs the lane-wide figure counts.
            waiting_sessions=[session for _, session in self.launching],
            duty_includes_wait=self.duty_includes_wait,
            wait_bound_seconds=self.wait_bound_seconds,
        )
        self.capacity = int(snapshot["capacity"] or 0)
        self.slots = int(snapshot["worker_slots"])
        self.documents.append(
            {
                "t": self.t,
                "worker_slots": self.slots,
                "capacity": self.capacity,
                "live_runs": snapshot["live_runs"],
                "gate_wait_p90_seconds": snapshot["gate_wait_p90_seconds"],
                "worker_slots_reason": snapshot["worker_slots_reason"],
            }
        )
        # Each session launches only from the offer the document makes it: the
        # per-session figure when the document names the session, and the
        # new-session figure when it does not yet see it working.
        live = snapshot.get("sessions") or {}
        unseen = int(snapshot.get("new_session_worker_slots") or 0)
        if self.per_session_reserve:
            counters = per_session_reserve_offers(snapshot)
            self.offers = {
                name: counters.get(name, counters[NEW_SESSION])
                for name in self.sessions
            }
        else:
            self.offers = {
                name: int((live.get(name) or {}).get("worker_slots", unseen))
                for name in self.sessions
            }
        return self.slots

    def worker_ten_minute_means(self, *, first_hour: float = 3600.0) -> list[float]:
        """The pool's mean size in each ten-minute bucket after the first hour.

        Read from the same run the rule produced, so a count that is still
        settling shows as buckets that differ from the last hour's rather than
        as a mean that hides it. ``first_hour`` is measured from the end of the
        warm-up, so the count is read only after the pool has had an hour to
        find its level.
        """
        means: list[float] = []
        for bucket in sorted(self.worker_bucket_time):
            if (bucket + 1) * WORKER_BUCKET_S <= first_hour:
                continue
            span = self.worker_bucket_time[bucket]
            if span > 0:
                means.append(self.worker_bucket_area[bucket] / span)
        return means

    # -- one run ---------------------------------------------------------
    def run(self, start_workers: int | None = None) -> RunResult:
        if self.explicit_sessions:
            start = {
                name: session.workers for name, session in self.sessions.items()
            }
        else:
            start = {
                DEFAULT_SESSION: (
                    start_workers if start_workers is not None else self.width
                )
                # The single-session lane is started by the caller's count, as
                # the fixed-pool reproduction is.
            }
        for name, count in start.items():
            for _ in range(count):
                self._start_worker(name)
        next_publish = next_admit = 0.0
        tokens_out = 0.0
        while self.t < self.end:
            if self.t >= next_publish:
                self.publish_slots()
                next_publish += PUBLISH_EVERY
            if self.t >= next_admit:
                # The dispatcher's own count is a floor on the room left: a run
                # the record cannot see -- one thinking longer than the working
                # horizon -- is still a worker this run holds, and admitting
                # past the capacity it published would make that figure mean
                # nothing. The gate's own arithmetic is untouched; this is what
                # the consumer knows about its pool.
                held = len(self.workers) + len(self.launching)
                room = max(0, self.capacity - held) if self.capacity else self.slots
                for name in self.sessions:
                    # A session launches only up to the offer the document made
                    # the session, and only up to what it still has queued: a
                    # session offered slots with no work behind them leaves them
                    # for a session that has work.
                    queued = self.queues[name]
                    if queued is not None and queued <= 0:
                        continue
                    take = min(self.offers.get(name, 0), room, LAUNCH_POOL)
                    if queued is not None:
                        take = min(take, queued)
                    for _ in range(max(0, take)):
                        self._launch(name)
                        if queued is not None:
                            queued = int(queued) - 1
                            self.queues[name] = queued
                        room -= 1
                next_admit += ADMIT_EVERY
            due = [session for at, session in self.launching if at <= self.t]
            self.launching = [
                (at, session) for at, session in self.launching if at > self.t
            ]
            for session in due:
                self._start_worker(session)
            for worker, state in list(self.workers.items()):
                if state["state"] == "think" and float(state["until"]) <= self.t:
                    if float(state["left"]) <= 0:
                        session = self.sessions.get(str(state["session"]))
                        if session is not None and session.hold:
                            # A holding session's worker keeps asking: it is
                            # holding runs the lane must work around, not a run
                            # that ends.
                            state["left"] = float(
                                self.rng.choice(self.calibration.per_run)
                            )
                        else:
                            del self.workers[worker]
                            continue
                    state["left"] = float(state["left"]) - 1
                    self._new_request(worker)
            while self.gate and len(self.in_service) < self.width:
                worker = self.gate.popleft()
                state = self.workers[worker]
                state["state"] = "serve"
                self.in_service.add(worker)
                self.open_intervals[worker] = self.ledger.admit(
                    str(worker),
                    session=str(state["session"]),
                    now=self.t,
                    arrived=float(state["arrived"]),
                )
                if self.t > WARM_UP_S:
                    waited = self.t - float(state["arrived"])
                    self.waits.append(waited)
                    self.wait_events.append((self.t, waited))
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
                    factor = self.think_scale(self.t)
                    think = self.rng.choice(self.calibration.think) * factor
                    state["until"] = self.t + think
            if self.t > WARM_UP_S:
                self.level_time[len(self.in_service)] += DT
                self.live_time += len(self.workers) * DT
                bucket = int((self.t - WARM_UP_S) // WORKER_BUCKET_S)
                self.worker_bucket_time[bucket] += DT
                self.worker_bucket_area[bucket] += len(self.workers) * DT
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


def _borrowed_slots_per_session(
    *, u: int, fair_share: int, others: Iterable[int], global_slots: int
) -> int:
    """One session's offer under the superseded reserve-per-session rule.

    The lane used to hold ``BORROW_RESERVE`` slots for *each* working session
    below its share, so four quiet sessions reserved 8 slots against 7 free and
    left the session holding the queued work none. The rule has no
    implementation left to call, because the router holds the reserve once; it
    is stated here, where a superseded rule is kept so a check can be run
    against it, and nowhere in the router.
    """
    own_room = max(0, fair_share - u)
    reserve = sum(
        min(max(0, fair_share - other), BORROW_RESERVE) for other in others
    )
    return max(0, min(global_slots, max(own_room, global_slots - reserve)))


def per_session_reserve_offers(document: Mapping[str, Any]) -> dict[str, int]:
    """Each session's offer under the superseded reserve-per-session rule.

    Read from a published admission document's own fields -- its capacity, its
    lane-wide slots and each session's live runs -- so the check reads the same
    figures a reader of the document does. ``NEW_SESSION`` carries the figure
    for a session the document does not yet name. The offers of a paused
    document are all zero, because its lane-wide figure is.
    """
    capacity = int(document.get("capacity") or 0)
    global_slots = int(document.get("worker_slots") or 0)
    live = {
        str(name): int((entry or {}).get("live_runs") or 0)
        for name, entry in (document.get("sessions") or {}).items()
    }
    offers: dict[str, int] = {}
    if not live:
        offers[NEW_SESSION] = global_slots
        return offers
    if len(live) <= 1:
        offers = {name: global_slots for name in live}
    else:
        fair_share = capacity // len(live)
        for name, runs in live.items():
            offers[name] = _borrowed_slots_per_session(
                u=runs,
                fair_share=fair_share,
                others=[count for other, count in live.items() if other != name],
                global_slots=global_slots,
            )
    offers[NEW_SESSION] = _borrowed_slots_per_session(
        u=0,
        fair_share=capacity // (len(live) + 1),
        others=list(live.values()),
        global_slots=global_slots,
    )
    return offers
