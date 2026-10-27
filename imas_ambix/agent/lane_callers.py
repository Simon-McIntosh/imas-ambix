"""Who is holding the shared lane, from the router's per-request record.

The lane document says how full the generation gate is; it does not say whose
requests fill it. The router's request record does: every row names the
connection it came from (``caller_hint``, the peer address and user agent) and,
when the caller reported them, the crew run and the coordinator session behind
it. This module groups a trailing window of those rows by caller and reports
how much of the gate each one held.

**The unit is held requests, not request count.** A request holds the lane from
the moment it reaches the gate until its relay ends -- waiting first, then
generating -- so ``held`` is the time-averaged number of one caller's requests
in that state over the window. It is the quantity the gate's width is measured
in, so the callers' figures sum to the gate's own occupancy and a caller holding
twelve of a twelve-wide gate is visible as exactly that. A count of finished
requests is not: a hundred short catalog reads and one long generation are not
the same load.

**A caller is a coordinator session, or a host when no session was reported.**
Sessions are what dispatch fleets, so they are the unit a share is argued over;
a request with no session is grouped by the address it came from, which is the
only identity the record holds for it. The address is the router's peer, so a
request relayed through an SSH tunnel carries the tunnel's host, not the machine
that issued it. No user name is published, because the record holds none and
inventing one from an address would read exactly like a reported one.

**``held_per_run`` is the fan-out a run-counted share cannot see.** The router's
fair share divides capacity in runs, at the lane-wide mean of requests per run.
A run that issues several requests at once -- parallel subagents inside one
worker -- is counted as one run while holding several slots, so a session's held
requests per run is published beside its run count, and ``heaviest_run_held``
names the worst single run.

**The same rows are totalled by account and by project beside the sessions.**
Each request row may carry an ``account`` and a ``project``, which the caller
reports about the identity behind it; the block's ``accounts`` level totals the
window by account and its ``projects`` level by ``(account, project)``. The
builder copies those two keys and infers nothing: a row that reports no account
is totalled under a single ``unattributed`` account, and a row that reports an
account but no project is totalled under ``unattributed`` within that account.
Because every row belongs to exactly one account and to exactly one project of
that account, the accounts sum to the block's ``held`` and an account's projects
sum to that account's own ``held``, so the levels reconcile with the per-session
list rather than restating it.

**Only finished requests are in the record.** A row is written when its request
ends, so a request still waiting or generating at the end of the window is
absent until it finishes; the figures lag the gate by about one request's
lifetime and understate a caller whose requests are all long. Rows the sink
sampled away are scaled back by each kept row's ``sample_fraction``; the peak is
a count of overlapping kept rows and is not scaled.

**The same rows carry a history strip and an outcomes line.** The ``history``
block bins the last half hour into one-minute bins and reports, per bin, the
time-averaged requests held at the gate and the p50 and p95 of the waits of the
requests admitted in it, so a reader sees whether the lane is filling rather
than only where it stands. The ``outcomes`` block counts the last five minutes'
requests by how they ended. The strip reaches back further than the callers'
window, so the three levels are cut from one bounded read of the record's tail
rather than three polls of a file that grows under them; the read and its byte
ceiling are
:func:`~imas_ambix.agent.request_receipts.recent_receipt_rows`. The read opens
over the wider of the callers window and the strip, so a caller asking for a
shorter window than the strip still gets the strip from the same pass, and each
block states whether that read covered the start of its **own** window -- a
five-minute callers block is not left claiming the strip's coverage, nor a
one-minute caller the coverage of half an hour.

Four outcomes partition the window: ``completed``, ``aborted`` and ``failed``
are the record's own status labels, and ``gate_expired`` is a request the router
answered itself after its generation-gate wait ran out. The record carries no
field naming the expiry, so it is read from the two facts that hold together on
such a row and on no other: it was answered by the router (the self-answered
upstream sentinel), and it ended ``failed`` after spending a positive time at
the gate. A request the router refused before any engine was chosen is
self-answered with no gate wait, and one whose caller left while uploading is
``aborted``, so neither is counted as an expiry.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from imas_ambix.agent.receipt_bins import inclusive_quantile
from imas_ambix.agent.request_receipts import (
    SELF_ANSWERED_UPSTREAM,
    STATUS_ABORTED,
    STATUS_COMPLETED,
    STATUS_FAILED,
    receipt_interval,
    recent_receipt_rows,
)

#: The trailing window the ``callers`` block describes. Long enough that a
#: session working in bursts is seen at its mean rather than at one burst, and
#: short enough that a wave dispatched a few minutes ago dominates it.
WINDOW_SECONDS = 300.0

#: The trailing window the ``history`` strip covers, in one-minute bins. Long
#: enough that a lane filling up is visible as a trend rather than as the
#: single number the lane document already publishes.
HISTORY_WINDOW_SECONDS = 1800.0

#: The width of one ``history`` bin. One minute is the shortest bin over which
#: the gate's own fair-share window means anything, and thirty of them fit the
#: strip a view can draw on one line.
HISTORY_BIN_SECONDS = 60.0

#: The trailing window the ``outcomes`` line counts over. Short enough to
#: describe the serve as it is now, and the same span as the callers window, so
#: a reader can hold a caller's share beside the outcomes of the same minutes.
OUTCOMES_WINDOW_SECONDS = 300.0

#: The three statuses the record labels a finished request with, and the
#: fourth outcome the record states no label for.
OUTCOME_STATUSES = (STATUS_COMPLETED, STATUS_ABORTED, STATUS_FAILED)
GATE_EXPIRED = "gate_expired"

#: The label the account and project levels carry for a request that reported no
#: account, or an account with no project. Every row written before the request
#: record carried the two keys is unattributed, and the label says what the
#: record holds rather than dropping the request from the totals.
UNATTRIBUTED = "unattributed"

#: What the figures are built from, stated on the block so a reader does not
#: have to know the record's write discipline to read the lag correctly.
BASIS = "finished requests; one still waiting or generating is absent until it ends"


def _isoformat(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _caller_parts(row: dict[str, Any]) -> tuple[str, str]:
    """The peer host and the user agent's product name from ``caller_hint``."""
    hint = str(row.get("caller_hint") or "")
    address, _, agent = hint.partition("|")
    host = address.rsplit(":", 1)[0] if ":" in address else address
    product = agent.split("/", 1)[0].strip()
    return host or "unknown", product or "unknown"


def _label(value: object) -> str:
    """A row's account or project, or the unattributed label when it holds none."""
    return value.strip() if isinstance(value, str) and value.strip() else UNATTRIBUTED


def _dominant(counts: Counter[str]) -> str:
    """The label a group's rows most often reported, or unattributed for none."""
    return counts.most_common(1)[0][0] if counts else UNATTRIBUTED


def _served_engine(row: dict[str, Any]) -> str | None:
    """The engine that relayed one row, or None for a row the router answered.

    The record names the upstream origin a request was relayed to, and the
    self-answered sentinel for the requests the router answered itself -- a
    catalog read, a refusal, or an expired gate wait. Only the former names an
    engine, so the sentinel is excluded rather than counted as one of the lane's
    engines, which would report a one-engine lane as two.
    """
    upstream = row.get("upstream")
    if not isinstance(upstream, str) or not upstream.strip():
        return None
    if upstream.strip() == SELF_ANSWERED_UPSTREAM:
        return None
    return upstream.strip()


def _engine_label(row: dict[str, Any]) -> str:
    """The engine bucket one row's held time belongs to.

    Every row lands in exactly one bucket so the buckets sum to the caller's
    held figure: a relayed row goes to its upstream, and a row the router
    answered itself goes to the gate engine that held it when the record names
    one and to the self-answered label when it does not.
    """
    engine = _served_engine(row)
    if engine is not None:
        return engine
    gate_engine = row.get("gate_engine")
    if isinstance(gate_engine, str) and gate_engine.strip():
        return gate_engine.strip()
    return SELF_ANSWERED_UPSTREAM


def _served_engines(rows: list[dict[str, Any]]) -> set[str]:
    """The distinct engines *rows* were relayed to."""
    return {
        engine for row in rows if (engine := _served_engine(row)) is not None
    }


def _engine_labels(rows: list[dict[str, Any]]) -> set[str]:
    """Every bucket a row falls in, including the ones no engine served.

    A request the router answered itself belongs to no engine, so it is neither
    dropped from the split nor attributed to one: the split has to account for
    every row, and a request given to an engine it never reached would make the
    split disagree with the total it is a split of.
    """
    return {_engine_label(row) for row in rows}


def _by_engine(counts: dict[str, float]) -> dict[str, float]:
    """One group's held time per engine, heaviest first."""
    return {
        engine: round(value, 2)
        for engine, value in sorted(counts.items(), key=lambda item: -item[1])
    }


def _tally(entry: dict[str, Any], session: str | None, run_id: str | None) -> None:
    """Count one row's session and run into a level's totals."""
    if session:
        entry["sessions"].add(session)
    if run_id:
        entry["runs"].add(run_id)


def _mean_gate_wait(entry: dict[str, Any]) -> float | None:
    return (
        round(entry["gate_wait_sum"] / entry["gate_wait_count"], 1)
        if entry["gate_wait_count"]
        else None
    )


def _share(held: float, total_held: float) -> float | None:
    return round(held / total_held, 3) if total_held > 0 else None


def _account_rows(
    accounts: dict[str, dict[str, Any]], total_held: float
) -> list[dict[str, Any]]:
    """One entry per account: its sessions, runs, held and share of the gate."""
    rows: list[dict[str, Any]] = []
    for entry in accounts.values():
        runs = len(entry["runs"])
        row: dict[str, Any] = {
            "account": entry["account"],
            "sessions": len(entry["sessions"]),
            "runs": runs,
            "held": round(entry["held"], 2),
            "held_per_run": round(entry["held"] / runs, 2) if runs else None,
            "mean_gate_wait_s": _mean_gate_wait(entry),
            "share": _share(entry["held"], total_held),
        }
        if entry["account"] == UNATTRIBUTED:
            row["hosts"] = sorted(entry["hosts"])
        rows.append(row)
    rows.sort(key=lambda row: (-row["held"], row["account"]))
    return rows


def _project_rows(
    projects: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    """One entry per ``(account, project)``: its sessions, runs, held and wait."""
    rows: list[dict[str, Any]] = []
    for entry in projects.values():
        runs = len(entry["runs"])
        rows.append(
            {
                "account": entry["account"],
                "project": entry["project"],
                "sessions": len(entry["sessions"]),
                "runs": runs,
                "held": round(entry["held"], 2),
                "held_per_run": round(entry["held"] / runs, 2) if runs else None,
                "mean_gate_wait_s": _mean_gate_wait(entry),
            }
        )
    rows.sort(key=lambda row: (-row["held"], row["account"], row["project"]))
    return rows


def _weight(row: dict[str, Any]) -> float:
    """How many offered requests one kept row stands for."""
    fraction = row.get("sample_fraction")
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
        return 1.0
    return 1.0 / fraction if 0.0 < fraction <= 1.0 else 1.0


def _overlap(start: float, end: float, low: float, high: float) -> float:
    return max(0.0, min(end, high) - max(start, low))


def _peak(spans: list[tuple[float, float]]) -> int:
    """The most spans open at once; an end and a start that touch do not overlap."""
    events = sorted(
        [(start, 1) for start, _ in spans] + [(end, -1) for _, end in spans]
    )
    current = peak = 0
    for _, step in events:
        current += step
        peak = max(peak, current)
    return peak


def callers_block(
    directory: str | Path,
    *,
    now: datetime | None = None,
    window_seconds: float = WINDOW_SECONDS,
) -> dict[str, Any]:
    """The ``callers`` block: each caller's share of the gate over the window.

    *directory* holds the router's request record -- the legacy
    ``requests.jsonl`` and any day files beside the lane document. A directory
    with no record, or a record that cannot be parsed, yields an ``unavailable``
    block naming why rather than an empty list a reader would take for an idle
    lane.
    """
    moment = (now or datetime.now(UTC)).astimezone(UTC)
    span_seconds = _read_span(window_seconds)
    try:
        rows, reached = _record_tail(directory, moment, span_seconds)
    except (OSError, ValueError) as error:
        return _unavailable_block(moment, window_seconds, detail=str(error))
    return _callers_from_rows(
        rows,
        covered=_window_covered(rows, moment, window_seconds, reached=reached),
        moment=moment,
        window_seconds=window_seconds,
    )


def record_blocks(
    directory: str | Path, *, now: datetime | None = None
) -> dict[str, Any]:
    """The ``callers``, ``history`` and ``outcomes`` blocks from one read.

    The three describe one stretch of the router's request record at three
    grains -- who held the gate over five minutes, what the gate held minute by
    minute over thirty, and how many requests ended each way over five -- so
    they are built together from a single bounded read of the record's tail. A
    document builder calls this rather than the blocks separately, so one poll
    costs one pass over a file the router is appending to.
    """
    moment = (now or datetime.now(UTC)).astimezone(UTC)
    span_seconds = _read_span(WINDOW_SECONDS)
    try:
        rows, reached = _record_tail(directory, moment, span_seconds)
    except (OSError, ValueError) as error:
        detail = str(error)
        return {
            "callers": _unavailable_block(moment, WINDOW_SECONDS, detail=detail),
            "history": _history_unavailable(moment, detail),
            "outcomes": _unavailable_block(
                moment, OUTCOMES_WINDOW_SECONDS, detail=detail
            ),
        }
    return {
        "callers": _callers_from_rows(
            rows,
            covered=_window_covered(
                rows, moment, WINDOW_SECONDS, reached=reached
            ),
            moment=moment,
            window_seconds=WINDOW_SECONDS,
        ),
        "history": _history_from_rows(
            rows,
            moment,
            covered=_window_covered(
                rows, moment, HISTORY_WINDOW_SECONDS, reached=reached
            ),
        ),
        "outcomes": _outcomes_from_rows(
            rows,
            moment,
            covered=_window_covered(
                rows, moment, OUTCOMES_WINDOW_SECONDS, reached=reached
            ),
        ),
    }


def _record_tail(
    directory: str | Path, moment: datetime, span_seconds: float
) -> tuple[list[dict[str, Any]], bool]:
    """One bounded read of the record's tail, open over *span_seconds*.

    The read is bounded by
    :data:`~imas_ambix.agent.request_receipts.RECENT_TAIL_MAX_BYTES`, because
    the document is polled while the record grows. The flag it returns says
    whether this read reached its own start, which is a different question from
    whether it covered any one level's window: see :func:`_window_covered`.
    """
    return recent_receipt_rows(
        Path(directory), since=moment - timedelta(seconds=span_seconds)
    )


def _read_span(window_seconds: float) -> float:
    """How far back a read must reach to serve a caller asking for its window.

    The history strip is cut from the same read as the callers level, so the
    read opens over whichever of the two windows is wider, and a caller asking
    for a window narrower than the strip still gets the strip from the same
    single pass.
    """
    return max(window_seconds, HISTORY_WINDOW_SECONDS)


def _window_covered(
    rows: list[dict[str, Any]],
    moment: datetime,
    window_seconds: float,
    *,
    reached: bool,
) -> bool:
    """Whether the read covered the whole of *window_seconds*, not another's.

    A level's own window decides its flag, so a caller asking for a minute is
    not told its figures describe half an hour. The read's *reached* flag
    answers the question for the span it was opened over, which covers every
    window no wider than that span; a cut read still covers a narrower window
    when the rows it did return reach back past that window's start, because
    the rows of a cut read are a contiguous suffix of the record.
    """
    if reached:
        return True
    opened = (moment - timedelta(seconds=window_seconds)).timestamp()
    ended = [interval[2] for row in rows if (interval := receipt_interval(row))]
    return bool(ended) and min(ended) <= opened


def _unavailable_block(
    moment: datetime, window_seconds: float, *, detail: str
) -> dict[str, Any]:
    """A block that could not be built, naming what stopped it."""
    opened = moment - timedelta(seconds=window_seconds)
    return {
        "state": "unavailable",
        "window_seconds": window_seconds,
        "window_start": _isoformat(opened),
        "window_end": _isoformat(moment),
        "detail": f"request record unreadable: {detail}",
    }


def _history_unavailable(moment: datetime, detail: str) -> dict[str, Any]:
    block = _unavailable_block(moment, HISTORY_WINDOW_SECONDS, detail=detail)
    block["bin_seconds"] = HISTORY_BIN_SECONDS
    block["bins"] = []
    return block


def _callers_from_rows(
    rows: list[dict[str, Any]],
    *,
    covered: bool,
    moment: datetime,
    window_seconds: float
) -> dict[str, Any]:
    """The callers level, from rows already read."""
    opened = moment - timedelta(seconds=window_seconds)
    low, high = opened.timestamp(), moment.timestamp()
    block: dict[str, Any] = {
        "state": "unavailable",
        "window_seconds": window_seconds,
        "window_start": _isoformat(opened),
        "window_end": _isoformat(moment),
        "basis": BASIS,
    }
    block["covered"] = covered
    # The per-engine split is published only when the record shows more than
    # one engine, so a one-engine lane's block carries exactly the keys it
    # carried before the engine dimension existed. A reader of a single-engine
    # document sees no new field, which is what makes the dimension cost it
    # nothing.
    multi = len(_served_engines(rows)) > 1

    groups: dict[tuple[str, str], dict[str, Any]] = {}
    spans: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    run_held: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    accounts: dict[str, dict[str, Any]] = {}
    projects: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        interval = receipt_interval(row)
        if interval is None:
            continue
        arrived, admitted, ended = interval
        if ended < low or arrived > high:
            continue
        session = row.get("coordinator_session")
        run_id = row.get("run_id")
        account = _label(row.get("account"))
        project = _label(row.get("project"))
        host, product = _caller_parts(row)
        key = ("session", session) if session else ("host", host)
        group = groups.setdefault(
            key,
            {
                "session": session or None,
                "hosts": set(),
                "clients": set(),
                "runs": set(),
                "account_counts": Counter(),
                "project_counts": Counter(),
                "requests": 0.0,
                "held": 0.0,
                "waiting": 0.0,
                "gate_wait_sum": 0.0,
                "gate_wait_count": 0.0,
                "prompt_tokens": 0.0,
                "completion_tokens": 0.0,
                "held_by_engine": defaultdict(float),
            },
        )
        weight = _weight(row)
        group["hosts"].add(host)
        group["clients"].add(product)
        group["account_counts"][account] += weight
        group["project_counts"][project] += weight
        held = weight * _overlap(arrived, ended, low, high) / window_seconds
        group["held"] += held
        engine = _engine_label(row)
        group["held_by_engine"][engine] += held
        group["waiting"] += (
            weight * _overlap(arrived, admitted, low, high) / window_seconds
        )
        if run_id:
            group["runs"].add(run_id)
            run_held[key][run_id] += held
        spans[key].append((max(arrived, low), min(ended, high)))
        account_entry = accounts.setdefault(
            account,
            {
                "account": account,
                "hosts": set(),
                "sessions": set(),
                "runs": set(),
                "held": 0.0,
                "gate_wait_sum": 0.0,
                "gate_wait_count": 0.0,
                "held_by_engine": defaultdict(float),
            },
        )
        project_entry = projects.setdefault(
            (account, project),
            {
                "account": account,
                "project": project,
                "sessions": set(),
                "runs": set(),
                "held": 0.0,
                "gate_wait_sum": 0.0,
                "gate_wait_count": 0.0,
                "held_by_engine": defaultdict(float),
            },
        )
        for entry in (account_entry, project_entry):
            entry["held"] += held
            entry["held_by_engine"][engine] += held
            _tally(entry, session, run_id)
        if account == UNATTRIBUTED:
            account_entry["hosts"].add(host)
        if low <= admitted <= high:
            group["requests"] += weight
            group["gate_wait_sum"] += weight * (admitted - arrived)
            group["gate_wait_count"] += weight
            for entry in (account_entry, project_entry):
                entry["gate_wait_sum"] += weight * (admitted - arrived)
                entry["gate_wait_count"] += weight
            for name in ("prompt_tokens", "completion_tokens"):
                value = row.get(name)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    group[name] += weight * value

    total_held = sum(group["held"] for group in groups.values())
    callers = []
    for key, group in groups.items():
        runs = len(group["runs"])
        heaviest = max(run_held[key].values(), default=None)
        callers.append(
            {
                "session": group["session"],
                "hosts": sorted(group["hosts"]),
                "clients": sorted(group["clients"]),
                "account": _dominant(group["account_counts"]),
                "project": _dominant(group["project_counts"]),
                "runs": runs,
                "requests": round(group["requests"], 1),
                "held": round(group["held"], 2),
                "waiting": round(group["waiting"], 2),
                "held_per_run": round(group["held"] / runs, 2) if runs else None,
                "heaviest_run_held": round(heaviest, 2)
                if heaviest is not None
                else None,
                "peak": _peak(spans[key]),
                "mean_gate_wait_s": (
                    round(group["gate_wait_sum"] / group["gate_wait_count"], 1)
                    if group["gate_wait_count"]
                    else None
                ),
                "prompt_tokens": round(group["prompt_tokens"]),
                "completion_tokens": round(group["completion_tokens"]),
                "share": round(group["held"] / total_held, 3)
                if total_held > 0
                else None,
            }
        )
        if multi:
            callers[-1]["held_by_engine"] = _by_engine(group["held_by_engine"])
            callers[-1]["engine"] = next(
                iter(callers[-1]["held_by_engine"]), None
            )
    callers.sort(key=lambda caller: (-caller["held"], -caller["requests"]))
    accounts_rows = _account_rows(accounts, total_held)
    projects_rows = _project_rows(projects)
    if multi:
        for row in accounts_rows:
            row["held_by_engine"] = _by_engine(
                accounts[row["account"]]["held_by_engine"]
            )
        for row in projects_rows:
            row["held_by_engine"] = _by_engine(
                projects[(row["account"], row["project"])]["held_by_engine"]
            )
    block.update(
        state="measured",
        held=round(total_held, 2),
        accounts=accounts_rows,
        projects=projects_rows,
        callers=callers,
    )
    if multi:
        block["engines"] = sorted(_engine_labels(rows))
    return block


def _gate_expired(row: dict[str, Any]) -> bool:
    """Whether one row records a request whose generation-gate wait ran out.

    The router answers an expired wait itself, so the row carries the
    self-answered upstream sentinel, and it ends ``failed`` after a positive
    wait. Those three together separate it from the other requests the router
    answers without relaying: a refusal before any engine was chosen is
    self-answered with no gate wait, and a caller that left while its request
    waited or uploaded records as ``aborted``.
    """
    if row.get("upstream") != SELF_ANSWERED_UPSTREAM:
        return False
    if row.get("status") != STATUS_FAILED:
        return False
    wait = row.get("gate_wait_s")
    if isinstance(wait, bool) or not isinstance(wait, (int, float)):
        return False
    return wait > 0.0


def _bin_percentile(values: list[float], fraction: float) -> float | None:
    """The *fraction* quantile of a bin's waits, in seconds, or None for none.

    The interpolation is :func:`~imas_ambix.agent.receipt_bins.inclusive_quantile`,
    which is the repository's one quantile; a bin waits report in tenths of a
    second, and a bin that admitted nobody reports no figure at all rather than
    zero, because a bin nobody had to wait in is a different fact from a bin
    nobody used.
    """
    if not values:
        return None
    return round(inclusive_quantile(tuple(values), fraction), 1)


def _history_from_rows(
    rows: list[dict[str, Any]], moment: datetime, covered: bool
) -> dict[str, Any]:
    """The ``history`` strip: one-minute bins over the last half hour.

    A bin's ``occupancy`` is the same quantity the callers block publishes --
    the time-averaged requests held at the gate, waiting or generating -- so a
    bin is directly comparable with the lane's own width and with a caller's
    held figure. ``gate_wait_p50_s`` and ``gate_wait_p95_s`` describe the waits
    of the requests admitted inside that bin, and are None for a bin that
    admitted none: a bin nobody had to wait in is a different fact from a bin
    nobody used.
    """
    block: dict[str, Any] = {
        "state": "unavailable",
        "window_seconds": HISTORY_WINDOW_SECONDS,
        "window_start": _isoformat(moment - timedelta(seconds=HISTORY_WINDOW_SECONDS)),
        "window_end": _isoformat(moment),
        "bin_seconds": HISTORY_BIN_SECONDS,
        "covered": covered,
        "bins": [],
    }
    if not rows:
        block["reason"] = "no rows in the trailing window"
        return block
    end = moment.timestamp()
    count = int(round(HISTORY_WINDOW_SECONDS / HISTORY_BIN_SECONDS))
    edges = [end - (count - index) * HISTORY_BIN_SECONDS for index in range(count + 1)]
    held = [0.0] * count
    admitted = [0.0] * count
    waits: list[list[float]] = [[] for _ in range(count)]
    multi = len(_served_engines(rows)) > 1
    engine_held: dict[str, list[float]] = defaultdict(lambda: [0.0] * count)
    engine_waits: dict[str, list[list[float]]] = defaultdict(
        lambda: [[] for _ in range(count)]
    )
    for row in rows:
        interval = receipt_interval(row)
        if interval is None:
            continue
        arrived, admitted_at, ended = interval
        weight = _weight(row)
        engine = _engine_label(row) if multi else None
        for index in range(count):
            low, high = edges[index], edges[index + 1]
            share = weight * _overlap(arrived, ended, low, high) / HISTORY_BIN_SECONDS
            held[index] += share
            if engine is not None:
                engine_held[engine][index] += share
            if low <= admitted_at < high:
                admitted[index] += weight
                waits[index].append(admitted_at - arrived)
                if engine is not None:
                    engine_waits[engine][index].append(admitted_at - arrived)
    block["bins"] = _bin_rows(edges, held, admitted, waits)
    block["state"] = "measured"
    if multi:
        block["engines"] = [
            {
                "engine": engine,
                "bins": _bin_rows(
                    edges, engine_held[engine], [0.0] * count,
                    engine_waits[engine],
                ),
            }
            for engine in sorted(
                engine_held, key=lambda name: -sum(engine_held[name])
            )
        ]
    if not covered:
        block["reason"] = "the read stopped at the record's byte ceiling"
    return block


def _bin_rows(
    edges: list[float],
    held: list[float],
    admitted: list[float],
    waits: list[list[float]],
) -> list[dict[str, Any]]:
    """One entry per history bin, from its accumulated figures."""
    return [
        {
            "start": _isoformat(datetime.fromtimestamp(edges[index], UTC)),
            "end": _isoformat(datetime.fromtimestamp(edges[index + 1], UTC)),
            "occupancy": round(0.0 if held[index] is None else held[index], 2),
            "requests": round(admitted[index], 1),
            "gate_wait_p50_s": _bin_percentile(waits[index], 0.5),
            "gate_wait_p95_s": _bin_percentile(waits[index], 0.95),
        }
        for index in range(len(edges) - 1)
    ]


def _outcomes_from_rows(
    rows: list[dict[str, Any]], moment: datetime, covered: bool
) -> dict[str, Any]:
    """The ``outcomes`` line: how the last five minutes' requests ended.

    Counted by when each request ended, which is when its row was written, so
    the line describes work that finished rather than work still in flight. The
    four figures partition the rows: a request the router answered itself after
    its gate wait ran out is counted as a gate expiry and not also as a
    failure, which would count one request twice.
    """
    opened = moment - timedelta(seconds=OUTCOMES_WINDOW_SECONDS)
    low, high = opened.timestamp(), moment.timestamp()
    counts = dict.fromkeys((*OUTCOME_STATUSES, GATE_EXPIRED), 0.0)
    per_engine: dict[str, dict[str, float]] = defaultdict(
        lambda: dict.fromkeys((*OUTCOME_STATUSES, GATE_EXPIRED), 0.0)
    )
    multi = len(_served_engines(rows)) > 1
    seen = False
    for row in rows:
        interval = receipt_interval(row)
        if interval is None:
            continue
        ended = interval[2]
        if ended < low or ended > high:
            continue
        seen = True
        weight = _weight(row)
        engine = _engine_label(row) if multi else None
        if _gate_expired(row):
            outcome = GATE_EXPIRED
        elif row.get("status") in counts:
            outcome = row["status"]
        else:
            continue
        counts[outcome] += weight
        if engine is not None:
            per_engine[engine][outcome] += weight
    block: dict[str, Any] = {
        "state": "unavailable",
        "window_seconds": OUTCOMES_WINDOW_SECONDS,
        "window_start": _isoformat(opened),
        "window_end": _isoformat(moment),
        "covered": covered,
    }
    if not seen:
        block["reason"] = "no request ended in the trailing window"
        return block
    block["state"] = "measured"
    block.update({name: round(total, 1) for name, total in counts.items()})
    if multi:
        block["by_engine"] = [
            {
                "engine": engine,
                **{name: round(total, 1) for name, total in per_engine[engine].items()},
            }
            for engine in sorted(
                per_engine, key=lambda name: -sum(per_engine[name].values())
            )
        ]
    if not covered:
        block["reason"] = "the read stopped at the record's byte ceiling"
    return block
