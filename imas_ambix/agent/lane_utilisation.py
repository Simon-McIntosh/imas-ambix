"""Window utilisation of the generation gate, from its own request record.

An instant reading cannot answer two questions a reader of the lane asks
constantly: how full the lane actually ran over a window, and -- when it ran
below its width -- whether that was because no work was offered or because the
gate offered nothing. Both are computed here from the same record of request
intervals, so the lane document carries them rather than leaving every reader to
replay receipts.

The record is the intervals the gate already keeps: when a request arrived, when
it was admitted, when it ended, how many completion tokens it produced and which
run it belonged to. ``intervals_from_rows`` builds the same intervals from
receipt rows, which is how the computation is checked against the receipts that
produced the plan's figures.

Two figures need explaining because a reader will otherwise assume a different
denominator:

``share_at_width`` is the share of the window with at least ``width`` requests in
service. The sweep counts a request from admission to release, so the count is
the one the gate's own admission compares against its width.

``slots_offered_not_taken_seconds`` and ``no_slots_offered_seconds`` partition
the time the gate was below its width. A worker slot is offered while the gate
would admit another run: the published capacity less the runs live at that
instant. A run is live while one of its requests is in service or it made one
within the working horizon the per-session share already uses. Time below width
is attributed to slots offered when a slot was free and nobody took it -- an
underfed lane -- and to no slots offered when the gate offered none, which is
what a paused gate, a closed wait bound or a full run count does. A paused gate
therefore attributes every such second to no slots offered.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from imas_ambix.agent.median import median_or_none

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

# The margin the capacity is sized at, and the horizon a run keeps claiming a
# slot after its last admission. Both are published beside the figures; the
# defaults match the gate's own so a caller that passes neither still computes
# the block the gate publishes.
DEFAULT_MARGIN = 1.5
DEFAULT_WORKING_HORIZON_SECONDS = 180.0
DEFAULT_HOUR_SECONDS = 3600.0
# A level's rate is the mean of its requests, so a bin holding a handful of them
# is a single turn's noise. The evidence scripts apply the same floor.
PLATEAU_MIN_REQUESTS = 20


@dataclass(frozen=True, slots=True)
class RequestInterval:
    """One request's occupancy of the lane, as the record states it.

    ``arrived`` is when the caller reached the gate, ``admitted`` when the gate
    let it through and ``ended`` when its relay ended. ``completion_tokens`` is
    ``None`` when the row measured none -- a cut request, an engine that
    reported no usage -- because a zero there would read as a completed request
    that produced nothing.
    """

    arrived: float
    admitted: float
    ended: float
    completion_tokens: int | None = None
    run_id: str | None = None

    @property
    def service_seconds(self) -> float:
        return max(0.0, self.ended - self.admitted)

    @property
    def gate_wait_seconds(self) -> float:
        return max(0.0, self.admitted - self.arrived)


def intervals_from_rows(
    rows: Iterable[Mapping[str, object]], *, model: str | None = None
) -> list[RequestInterval]:
    """The request intervals a set of receipt rows records.

    A row's ``timestamp`` is the moment its relay began, after the gate admitted
    it, so the request arrived ``gate_wait_s`` earlier and ended ``duration_s``
    later -- the reading :func:`~imas_ambix.agent.request_receipts.receipt_interval`
    already makes for the rest of the repository. A row carrying no readable
    timestamp or duration is skipped rather than guessed at.
    """
    from imas_ambix.agent.request_receipts import receipt_interval

    intervals: list[RequestInterval] = []
    for row in rows:
        if model is not None and row.get("model") != model:
            continue
        span = receipt_interval(row)
        if span is None:
            continue
        arrived, admitted, ended = span
        tokens: int | None = None
        if row.get("status") == "completed":
            raw = row.get("completion_tokens")
            if isinstance(raw, int | float) and not isinstance(raw, bool) and raw >= 0:
                tokens = int(raw)
        run_id = row.get("run_id")
        intervals.append(
            RequestInterval(
                arrived=arrived,
                admitted=admitted,
                ended=ended,
                completion_tokens=tokens,
                run_id=run_id if isinstance(run_id, str) and run_id else None,
            )
        )
    intervals.sort(key=lambda interval: interval.admitted)
    return intervals


def _clip(interval: RequestInterval, start: float, end: float) -> tuple[float, float]:
    return max(interval.admitted, start), min(interval.ended, end)


def _live_spans(
    intervals: Sequence[RequestInterval],
    *,
    working_horizon: float = 0.0,
    starts_at: Literal["admitted", "arrived"] = "admitted",
) -> dict[str, tuple[float, float]]:
    """Each run's span: its first request to its last request's end.

    ``starts_at="admitted"`` is the live span the lane claims a slot over: a run
    claims from its first admission and for ``working_horizon`` after its last
    one -- the same claim the per-session share makes, so a run that stopped
    working stops counting as a live run.

    ``starts_at="arrived"`` with no horizon is the span the run offered its load
    over: the record's own first and last times, so the horizon is not stretched
    into a run that has stopped asking, and the wait between asking and being
    served is inside the span where the caller can take it out.
    """
    first: dict[str, float] = {}
    last_start: dict[str, float] = {}
    last_end: dict[str, float] = {}
    for interval in intervals:
        run = interval.run_id
        if run is None:
            continue
        opening = interval.arrived if starts_at == "arrived" else interval.admitted
        first[run] = min(first.get(run, opening), opening)
        last_start[run] = max(last_start.get(run, opening), opening)
        last_end[run] = max(last_end.get(run, interval.ended), interval.ended)
    return {
        run: (first[run], max(last_end[run], last_start[run] + working_horizon))
        for run in first
    }


def offered_load(
    intervals: Sequence[RequestInterval], *, include_gate_wait: bool = False
) -> float | None:
    """The load one live run offers the gate when it does not wait.

    A worker's share of the lane is its service time over its live span, with
    the time it spent waiting at the gate left OUT. Waiting is what the gate
    does to a run, not what the run brings, so a rule sized on this load does
    not read a fuller lane as room for more workers -- the failure a duty
    measured over the whole live span has.

    The span is the one the record observed, first arrival to last request end,
    not the working horizon a run keeps claiming a slot for after it last asked:
    that horizon is the slot claim, and stretching the span by it would read a
    run that has stopped asking as a lighter run.

    ``include_gate_wait`` states that failing rule instead: the duty is then
    the service time over the whole live span, so a run that waits looks like a
    lighter run and the rule admits more the longer the queue gets. Nothing in
    the lane sets it; the simulation's negative control is its only caller, and
    it exists so that control exercises this code rather than a copy of it.

    The figure is the mean over the live runs, each run one vote: a run's own
    duty is the load one live run offers, and averaging it over the lane is what
    keeps the estimate the section asks for. Weighting the average by span
    instead would let the longest-lived run set it -- a pooled ratio moves as a
    cohort ages, which is a change in who has been alive longest, not in the
    load the lane's runs offer. A median would give each run one vote too, but a
    few dozen runs make a noisy median when the think times are as heavy-tailed
    as these are, and the rule would size a dispatch from that noise.

    ``None`` when no keyed run offers a live span at all: a lane whose runs are
    all unkeyed cannot state the load, and a zero would divide.
    """
    service: dict[str, float] = {}
    waited: dict[str, float] = {}
    for interval in intervals:
        run = interval.run_id
        if run is None:
            continue
        service[run] = service.get(run, 0.0) + interval.service_seconds
        waited[run] = waited.get(run, 0.0) + interval.gate_wait_seconds
    duties: list[float] = []
    for run, (start, end) in _live_spans(intervals, starts_at="arrived").items():
        live = end - start
        available = live if include_gate_wait else live - waited[run]
        if available <= 0:
            continue
        duties.append(service[run] / available)
    if not duties:
        return None
    return sum(duties) / len(duties)


def capacity_from_offered_load(
    intervals: Sequence[RequestInterval],
    *,
    width: int,
    margin: float = DEFAULT_MARGIN,
    include_gate_wait: bool = False,
) -> int | None:
    """How many runs the gate's width can hold at the load they offer.

    The count is the margin times the width divided by the offered load, floored
    to a whole run: a run is admitted whole. ``None`` when the load cannot be
    stated, because a capacity guessed there would size a dispatch.
    """
    load = offered_load(intervals, include_gate_wait=include_gate_wait)
    if load is None or width <= 0 or margin <= 0:
        return None
    return max(0, math.floor(margin * width / load))


def _paused_seconds(
    spans: Sequence[tuple[float, float]], start: float, end: float
) -> float:
    total = 0.0
    for open_at, close_at in spans:
        total += max(0.0, min(close_at, end) - max(open_at, start))
    return total


def _sweep(
    *,
    intervals: Sequence[RequestInterval],
    start: float,
    end: float,
    width: int,
    capacity: int | None,
    paused_spans: Sequence[tuple[float, float]],
    working_horizon: float,
) -> dict[str, float | None]:
    """Accumulate the window's time-weighted figures over its own segments."""
    events: dict[float, list[tuple[int, int]]] = defaultdict(list)
    for interval in intervals:
        first, last = _clip(interval, start, end)
        if last <= first:
            continue
        events[first].append((0, 1))
        events[last].append((0, -1))
    for run_start, run_end in _live_spans(
        intervals, working_horizon=working_horizon
    ).values():
        first, last = max(run_start, start), min(run_end, end)
        if last <= first:
            continue
        events[first].append((1, 1))
        events[last].append((1, -1))
    for open_at, close_at in paused_spans:
        first, last = max(open_at, start), min(close_at, end)
        if last <= first:
            continue
        events[first].append((2, 1))
        events[last].append((2, -1))

    times = sorted(events)
    in_service = 0
    live_runs = 0
    paused = 0
    total = area = at_width = below = offered = none_offered = 0.0
    split_known = capacity is not None

    def accumulate(span: float) -> None:
        nonlocal total, area, at_width, below, offered, none_offered
        if span <= 0:
            return
        total += span
        area += in_service * span
        if in_service >= width:
            at_width += span
            return
        below += span
        if not split_known:
            return
        if paused > 0 or live_runs >= capacity:
            none_offered += span
        else:
            offered += span

    # The state before the first boundary holds for [start, first) as well, so
    # the segments are walked from the window's own start rather than from the
    # first event: a window whose record begins late is not a window that was
    # empty everywhere before it.
    cursor = start
    for moment in times:
        accumulate(moment - cursor)
        cursor = moment
        for kind, delta in events[moment]:
            if kind == 0:
                in_service += delta
            elif kind == 1:
                live_runs += delta
            else:
                paused += delta
    accumulate(end - cursor)
    return {
        "seconds": total,
        "mean_in_service": (area / total) if total > 0 else None,
        "share_at_width": (at_width / total) if total > 0 else None,
        "below_width_seconds": below if total > 0 else None,
        "slots_offered_not_taken_seconds": (
            offered if split_known and total > 0 else None
        ),
        "no_slots_offered_seconds": (
            none_offered if split_known and total > 0 else None
        ),
        "paused_seconds": _paused_seconds(paused_spans, start, end),
    }


def _level_means(
    intervals: Sequence[RequestInterval],
    *,
    start: float,
    end: float,
    width: int,
) -> list[tuple[int, float, int]]:
    """Each in-window request's mean in-service level, with its tokens and span.

    The level is the time-weighted count over the request's own span, which is
    what the evidence scripts bin on: a request's rate depends on how many
    others were in service beside it, not on the count at its first token.
    """
    events: dict[float, int] = defaultdict(int)
    for interval in intervals:
        first, last = _clip(interval, start, end)
        if last <= first:
            continue
        events[first] += 1
        events[last] -= 1
    times = sorted(events)
    cumulative: list[float] = []
    levels: list[int] = []
    area = 0.0
    cur = 0
    last = times[0] if times else start
    for moment in times:
        area += cur * (moment - last)
        last = moment
        cumulative.append(area)
        cur += events[moment]
        levels.append(cur)

    def mean_level(a: float, b: float) -> float:
        if b <= a:
            return 0.0
        left = bisect_right(times, a) - 1
        right = bisect_right(times, b) - 1
        before = (
            cumulative[left] + levels[left] * (a - times[left]) if left >= 0 else 0.0
        )
        upto = (
            cumulative[right] + levels[right] * (b - times[right])
            if right >= 0
            else 0.0
        )
        return max(0.0, (upto - before) / (b - a))

    out: list[tuple[int, float, int]] = []
    for interval in intervals:
        if interval.completion_tokens is None or interval.completion_tokens < 50:
            continue
        first, last = _clip(interval, start, end)
        span = interval.service_seconds
        if span <= 0:
            continue
        level = min(width, round(mean_level(first, last)))
        out.append(
            (level, interval.completion_tokens / span, interval.completion_tokens)
        )
    return out


def plateau_tokens_per_second(
    intervals: Sequence[RequestInterval], *, start: float, end: float, width: int
) -> float | None:
    """The lane's own estimate of the rate it tops out at, from these requests.

    The aggregate rate a level achieves is that level's mean per-stream rate
    times the level, so the plateau is the best bin the record actually holds --
    not the rate at the widest level seen, which a single fast request can set.
    """
    bins: dict[int, list[float]] = defaultdict(list)
    for level, per_stream, _tokens in _level_means(
        intervals, start=start, end=end, width=width
    ):
        bins[level].append(per_stream)
    best: float | None = None
    for level, rates in bins.items():
        if len(rates) < PLATEAU_MIN_REQUESTS:
            continue
        aggregate = level * (sum(rates) / len(rates))
        if best is None or aggregate > best:
            best = aggregate
    return best


def window_figures(
    intervals: Sequence[RequestInterval],
    *,
    width: int,
    start: float,
    end: float,
    capacity: int | None,
    paused_spans: Sequence[tuple[float, float]] = (),
    working_horizon: float = DEFAULT_WORKING_HORIZON_SECONDS,
) -> dict[str, object]:
    """The utilisation figures for one window of the record."""
    sweep = _sweep(
        intervals=intervals,
        start=start,
        end=end,
        width=width,
        capacity=capacity,
        paused_spans=paused_spans,
        working_horizon=working_horizon,
    )
    in_window = [interval for interval in intervals if start <= interval.admitted < end]
    span = end - start
    delivered = sum(
        interval.completion_tokens
        for interval in in_window
        if interval.completion_tokens is not None
    )
    plateau = plateau_tokens_per_second(intervals, start=start, end=end, width=width)
    # The gate wait is a median for the same reason the context window is: one
    # request held back through a long admission drags a mean far more than it
    # moves a median, and the median is the figure that says how long a caller
    # ordinarily waited at the gate. It is taken over the requests admitted in
    # the window, which is the population that actually waited at the gate --
    # requests still queued have no wait to measure yet and a request admitted
    # earlier belongs to an earlier window. The estimator is the package's one
    # median, which averages the two middles of an even count: a reader who
    # compares this figure against a benchmark's median is promised that both
    # are the same estimator.
    waits = [interval.gate_wait_seconds for interval in in_window]
    median_wait = median_or_none(waits)
    mean = sweep["mean_in_service"]
    share = sweep["share_at_width"]
    below = sweep["below_width_seconds"]
    return {
        "seconds": round(span, 3),
        "observed_seconds": round(float(sweep["seconds"] or 0.0), 3),
        "capacity": capacity,
        "mean_in_service": None if mean is None else round(float(mean), 2),
        "median_gate_wait_seconds": _rounded(median_wait),
        "share_at_width": None if share is None else round(float(share), 3),
        "below_width_seconds": None if below is None else round(float(below), 1),
        "slots_offered_not_taken_seconds": _rounded(
            sweep["slots_offered_not_taken_seconds"]
        ),
        "no_slots_offered_seconds": _rounded(sweep["no_slots_offered_seconds"]),
        "paused_seconds": round(float(sweep["paused_seconds"] or 0.0), 1),
        "delivered_tokens_per_second": (
            round(delivered / span, 1) if span > 0 else None
        ),
        "completion_tokens": delivered,
        "requests": len(in_window),
        "runs": len(
            {interval.run_id for interval in in_window if interval.run_id is not None}
        ),
        "plateau_tokens_per_second": None if plateau is None else round(plateau, 1),
    }


def percentile_or_none(values: Iterable[float], percentile: float) -> float | None:
    """The nearest-rank percentile of *values*, or ``None`` when there are none.

    The gate's wait bound is a percentile of the waits its own record holds, and
    a window that admitted nothing has no figure to compare -- a zero there
    would read as a lane that made nobody wait. The estimator is the one the
    lane simulation uses for the same figure, so an evidence script and the gate
    reading each other's windows agree on the number.
    """
    ordered = sorted(values)
    if not ordered:
        return None
    index = int(percentile * (len(ordered) - 1))
    return ordered[min(max(index, 0), len(ordered) - 1)]


def _rounded(value: object) -> float | None:
    return None if value is None else round(float(value), 1)


def utilisation_block(
    intervals: Sequence[RequestInterval],
    *,
    width: int,
    now: float,
    window_seconds: float,
    margin: float = DEFAULT_MARGIN,
    hour_seconds: float = DEFAULT_HOUR_SECONDS,
    paused_spans: Sequence[tuple[float, float]] = (),
    working_horizon: float = DEFAULT_WORKING_HORIZON_SECONDS,
    horizon: float | None = None,
) -> dict[str, object]:
    """The averaging window and the last hour of one record of intervals.

    Both windows are cut from the same intervals, so a reader comparing them is
    comparing two spans of one record rather than two measurements. ``horizon``
    is the first moment the record covers; before it the windows are shorter,
    and ``observed_seconds`` says by how much so a short history is not read as
    a quiet lane.
    """
    start_of_record = min((interval.admitted for interval in intervals), default=now)
    if horizon is not None:
        start_of_record = min(start_of_record, horizon)
    for_span = max(0.0, window_seconds)
    hour_span = max(0.0, hour_seconds)

    def block(span: float) -> dict[str, object]:
        start = max(start_of_record, now - span)
        # The capacity a window is attributed against is sized from the runs it
        # holds, so a starved window's spare slots are the ones its own load
        # implies rather than a figure carried in from a busier hour.
        inside = [interval for interval in intervals if interval.ended > start]
        capacity = capacity_from_offered_load(inside, width=width, margin=margin)
        if capacity is None:
            capacity = capacity_from_offered_load(intervals, width=width, margin=margin)
        return window_figures(
            intervals,
            width=width,
            start=start,
            end=now,
            capacity=capacity,
            paused_spans=paused_spans,
            working_horizon=working_horizon,
        )

    return {
        "width": width,
        "margin": round(margin, 3),
        "window": block(for_span),
        "last_hour": block(hour_span),
    }
