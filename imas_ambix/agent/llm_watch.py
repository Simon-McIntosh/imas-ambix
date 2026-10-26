"""The llm-watch view: one engine's live serving document, rendered as text.

The view renders the live serving document and nothing else. It reads the
published lane document, the tail of the recorder's newest raw receipt file,
the tail of the router's request record and the published watch document for
the period ledger -- every figure the serve panel draws is already published
somewhere, so nothing here probes the engine, opens the telemetry index or
fetches a price.

Three bands carry the readings, then the tables and the strips below them:

===========  ==========================================================
band         carries
===========  ==========================================================
throughput   the engine's own rates, differenced over the receipt window
capacity     the pool the engine holds and the context it serves
gate         the router's generation gate and the room to dispatch into
===========  ==========================================================

The throughput and gate bands are the serve panel's own, on the same column
grid and with the same figures, so a reader moving between the two tools reads
one set of numbers. The capacity band is what replaces the card conditions
there: temperature, power and percentage of capable draw belong to the machine
panel, and their slot here holds the engine's capacity instead.

**The running-workers view has two levels.** ``accounts`` is one row per
``account`` that held the gate, heaviest first, with its share drawn as a bar.
The table under it is the detail for one account -- the viewer's login by
default, the account ``--account`` names otherwise -- one row per project,
heaviest first, at most six, with the remainder folded into one ``+N more`` row
carrying their summed in-use. ``--sessions`` replaces the project rows with one
row per coordinator session. No other account is detailed unless
``--account`` names it, so the default view stays short however many sessions
hold the gate.

**A figure the document does not carry is a dash, never a zero.** An absence is
not a measurement, and a band of zeros reads as an idle lane when the truth is
that nothing was read. The one exception is deliberate and marked: when the
live entry itself is unavailable the view prints one line naming the reason in
place of the throughput band, because a band of dashes under an explanation
says less than the explanation alone.
"""

from __future__ import annotations

import json
import os
import sys
import time as _time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich import box
from rich.console import Console, Group
from rich.table import Table
from rich.text import Text

from imas_ambix.agent import watch as watch_mod
from imas_ambix.agent import watch_publish
from imas_ambix.agent.registry import _engine_id
from imas_ambix.agent.watch_style import (
    CELL_WIDTH,
    COLUMNS,
    DASH,
    LABEL_WIDTH,
    cell,
    fmt_ctx,
    fmt_rate,
    fmt_span,
    fmt_tokens,
    fmt_usd,
    pct,
    ramp,
    spark,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

#: How long the ``--live`` view waits between refreshes.
LIVE_REFRESH_SECONDS = 5.0

#: The label a row carries when the caller reported no account or no project.
UNATTRIBUTED = "unattributed"

#: Projects listed in the detail table before the rest fold into one row.
DETAIL_ROWS = 6

#: Per-worker decode reference, tok/s: the best single-stream rate measured on
#: this deployment. The per-worker cell is coloured against this rather than
#: against an absolute rate, so the figure answers whether each worker is
#: getting a fair share of what one worker can have. The value is the serve
#: panel's own, so the same rate takes the same colour on both.
PER_WORKER_REF = 40.0

#: The width a golden is rendered at. The narrow width is the tighter of the
#: two terminals an operator may be reading in, and no line may exceed either.
GOLDEN_WIDTH = 120

#: The exit status for an ``--engine`` id this document cannot show. It is a
#: refusal rather than a failure of the render, so it is separable from the
#: status a crash leaves.
REFUSED_ENGINE_STATUS = 2


# ── figures pulled off the document ──────────────────────────────────


def _number(source: dict[str, Any] | None, key: str) -> float | None:
    """One numeric figure out of a block, or None when it is absent or bool."""
    if not isinstance(source, dict):
        return None
    value = source.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _nested(source: dict[str, Any] | None, outer: str, inner: str) -> float | None:
    if not isinstance(source, dict):
        return None
    return _number(source.get(outer), inner)


def _viewer() -> str:
    """The login the detail table defaults to."""
    return os.environ.get("USER") or os.environ.get("LOGNAME") or "unknown"


def engine_endpoint(document: dict[str, Any], model_id: str | None) -> dict[str, Any]:
    """The endpoint entry describing the served engine, by model id.

    The head line's accelerator count, family and context ceiling are constants
    for the life of a deployment rather than readings, and they are published
    where every other consumer reads them: the endpoint document the router
    writes. An entry that names no model matches the document's only endpoint
    when there is exactly one, so a deployment that has not yet published a
    model id is still described rather than left blank.
    """
    endpoints = [
        entry for entry in (document.get("endpoints") or [])
        if isinstance(entry, dict)
    ]
    if model_id:
        for entry in endpoints:
            if entry.get("model_id") == model_id:
                return entry
    return endpoints[0] if len(endpoints) == 1 else {}


def served_engine(document: dict[str, Any]) -> str | None:
    """The id of the engine this document describes, or None when it names none.

    Read in the order the document states it: the served name the receipt block
    carries is what a client sends, and the lane and endpoint ids are the same
    release published by the two documents that route and describe it.
    """
    live = document.get("live") or {}
    lane = document.get("lane") or {}
    endpoint = engine_endpoint(document.get("engine") or {}, lane.get("model_id"))
    for value in (live.get("served_name"), lane.get("model_id"),
                  endpoint.get("model_id")):
        if isinstance(value, str) and value:
            return value
    return None


def engine_refusal(document: dict[str, Any], engine_id: str) -> str | None:
    """Why *engine_id* cannot be shown, naming the engine that is served.

    The view renders one engine: the lane the record, the pool figures and the
    gate all came from. An id naming a different engine is therefore a request
    the document cannot answer, and refusing it is the only honest reply --
    rendering anyway would label one engine's figures with the id the caller
    asked for. A document that names no engine at all refuses every id and says
    so, because there is no served engine to name.
    """
    served = served_engine(document)
    if engine_id == served:
        return None
    if served is None:
        return (f"llm-watch: this document names no served engine, so "
                f"{engine_id!r} cannot be selected")
    return f"llm-watch: this view serves {served} only, not {engine_id!r}"


def context_ceiling(lane: dict[str, Any], endpoint: dict[str, Any]) -> object:
    """The context window the lane is served at.

    The lane document's own pool figure is the number of tokens the engine
    holds; the endpoint document declares the per-request window. The window is
    what the head line states, because it is the ceiling every prompt below is
    measured against, and it is preferred in that order: a declared window the
    deployment published, else the context the pool document reports.
    """
    for source, key in ((endpoint, "max_context"), (lane, "max_context"),
                        (lane, "context_window")):
        value = source.get(key) if isinstance(source, dict) else None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
    return None


# ── the head line ────────────────────────────────────────────────────

#: What the head line says when the endpoint document records no checkpoint for
#: the served engine. A blank would read as "not applicable" rather than "not
#: recorded", and the served name is the same across a weights-only rotation.
CHECKPOINT_UNRECORDED = "checkpoint unrecorded"


def checkpoint_label(document: dict[str, Any]) -> str | None:
    """The checkpoint the served engine loaded, or None when none is recorded.

    The served name is the same across a weights-only rotation, so it cannot say
    which weights answered; the endpoint document's ``checkpoints`` map can. The
    key is the one the publishing side wrote, so it is built with the registry's
    own engine-id function rather than from the host the entry carries: the
    registry shortens the cluster component out of the host before writing, so a
    key built here from the raw host would miss every entry.

    The value is ``<hf_repo>@<revision>``; the label keeps the repository's last
    path part and a short commit, so a rotation shows as a different label under
    an unchanged served name.
    """
    lane = document.get("lane") or {}
    endpoint = engine_endpoint(document.get("engine") or {}, lane.get("model_id"))
    host = endpoint.get("host")
    port = endpoint.get("port")
    checkpoints = document.get("checkpoints")
    if not isinstance(host, str) or not host:
        return None
    if isinstance(port, bool) or not isinstance(port, int):
        return None
    if not isinstance(checkpoints, dict):
        return None
    recorded = checkpoints.get(_engine_id(host, port))
    if not isinstance(recorded, str) or not recorded:
        return None
    repository, _, revision = recorded.partition("@")
    name = repository.rsplit("/", 1)[-1]
    # ``unpinned`` is how a profile whose weights are not pinned to a revision
    # reports itself, so it is carried as written rather than shortened.
    short = revision[:8] if revision and revision != "unpinned" else revision
    return f"{name}@{short}" if short else name


def head_line(document: dict[str, Any]) -> str:
    """The engine in one line: what is served, on what, and at what context.

    Everything here is constant for the life of the deployment, which is why it
    sits above the bands rather than in them: a figure that cannot change
    during a session is not a reading, and repeating it every refresh costs ink
    that never says anything new.
    """
    lane = document.get("lane") or {}
    live = document.get("live") or {}
    model = live.get("served_name") or lane.get("model_id") or "?"
    endpoint = engine_endpoint(
        document.get("engine") or {}, lane.get("model_id")
    )
    host = str(endpoint.get("host") or "").split(".", 1)[0]
    bits = [f"[bold]{model}[/]"]
    checkpoint = checkpoint_label(document)
    bits.append(f"[cyan]{checkpoint}[/]" if checkpoint
                else f"[yellow]{CHECKPOINT_UNRECORDED}[/]")
    count = endpoint.get("accelerator_count")
    family = endpoint.get("accelerator_family")
    if count and family:
        bits.append(f"[dim]{count}×{family}[/]")
    elif family:
        bits.append(f"[dim]{family}[/]")
    if host:
        bits.append(f"[dim]{host.removeprefix('98dci4-')}[/]")
    ceiling = context_ceiling(lane, endpoint)
    if ceiling is not None:
        bits.append(f"[dim]{fmt_ctx(ceiling)} ctx[/]")
    return " [dim]·[/] ".join(bits)


# ── the three bands ──────────────────────────────────────────────────


def _grid() -> Table:
    """The shared column grid: a label column and six figure columns.

    The padding and the box are the ledger's own, so the bands and the ledger
    share one column pitch and a reader scanning down never has to re-find the
    columns. ``show_header`` is off because the bands label their own rows.
    """
    table = Table(box=box.SIMPLE_HEAD, padding=(0, 1), expand=False,
                  pad_edge=False, show_header=False)
    for index in range(COLUMNS):
        table.add_column(
            "",
            justify="right",
            width=LABEL_WIDTH if index == 0 else CELL_WIDTH,
            no_wrap=True,
        )
    return table


def _band(table: Table, labels: list[str], values: list[str], style: str) -> None:
    """Label row over value row, with a blank line between bands only."""
    if table.row_count:
        table.add_row(*[""] * COLUMNS)
    pad = [""] * (COLUMNS - len(labels))
    table.add_row(*(labels + pad), style="dim")
    table.add_row(*(values + pad), style=style)


def throughput_band(live: dict[str, Any]) -> Table:
    """The engine's own rates, from the receipt block.

    The figures are the serve panel's, read from the same block by the same
    names: ``running`` is the count of requests the engine holds, a measured
    quantity, and ``prefill/s`` is shortened from ``prefill tok/s`` because it
    is a character wider than the grid's figure columns. Every rate is shown
    only while the block says it measured one -- a window that could not be
    differenced carries a reason rather than a rate, and a rate drawn from it
    would be a claim the record does not support.
    """
    table = _grid()
    running = _number(live, "mean_running")
    queued = _number(live, "engine_queue")
    aggregate = _number(live, "generation_toks_per_s")
    per_stream = _number(live, "per_stream_toks_per_s")
    prefill = _number(live, "prefill_toks_per_s")
    ttft = _number(live, "median_time_to_first_token_s")
    hit = _number(live, "cache_hit_rate")
    measured = live.get("state") == "measured"
    queued_s = (
        f"[red]{queued:.0f}[/]" if queued is not None and queued >= 1
        else f"[dim]{queued:.0f}[/]" if queued is not None else DASH
    )
    _band(
        table,
        ["running", "queued", "tok/s out", "tok/s each", "prefill/s",
         "ttft s", "cache %"],
        [
            f"[cyan]{running:.1f}[/]" if running is not None else DASH,
            queued_s,
            fmt_rate(aggregate) if measured and aggregate else DASH,
            (f"[{ramp(100.0 * per_stream / PER_WORKER_REF, 60.0, 30.0)}]"
             f"{per_stream:.1f}[/]"
             if measured and per_stream else DASH),
            fmt_rate(prefill) if measured and prefill is not None else DASH,
            f"{ttft:.1f}" if measured and ttft is not None else DASH,
            pct(100.0 * hit, "good") if measured and hit is not None else DASH,
        ],
        "bold",
    )
    return table


def capacity_band(lane: dict[str, Any], live: dict[str, Any]) -> Table:
    """The pool the engine holds and the context it is serving.

    The card conditions the serve panel drew here -- temperature, power and
    draw against cap -- belong to the machine panel, so this band holds the
    engine's capacity instead. Each figure is read under the name the lane
    document publishes it by, and a figure the document does not carry is a
    dash: the band is drawn whether or not the deployment has published the
    reading yet, so a reader sees the column and knows whether it is filled.
    """
    table = _grid()
    # Pool occupancy is the one capacity figure the live block already carries,
    # because it is read from the engine's own counters rather than from the
    # lane document's window.
    pool = _number(live, "pool_occupancy")
    host_tier = _nested(lane, "hicache_host", "used_fraction")
    host_tier_pct = host_tier * 100.0 if host_tier is not None else None
    mean_ctx = _number(lane, "mean_context") or _number(lane, "mean_context_instant")
    ceiling = _number(lane, "concurrent_requests")
    headroom = _number(lane, "engine_headroom")
    preemptions = _number(lane, "preemptions")
    _band(
        table,
        ["kv pool %", "host tier %", "mean ctx", "ceiling", "eng headrm",
         "preemptions"],
        [
            pct(pool * 100.0 if pool is not None and pool <= 1.0 else pool)
            if pool is not None else DASH,
            pct(host_tier_pct) if host_tier_pct is not None else DASH,
            fmt_ctx(mean_ctx),
            cell(ceiling, 1.0, 1.0),
            cell(headroom, 1.0, 0.0, digits=0),
            cell(preemptions, 0.0, 0.0, inverted=True),
        ],
        "",
    )
    return table


def gate_band(lane: dict[str, Any]) -> Table:
    """The router's generation gate, and the room left to dispatch into.

    A request held here never reaches the engine, so the engine's queue reads
    zero while the gate is holding work: reading only the engine's queue cannot
    tell a drained lane from a full one. In flight is shown over the width in
    force rather than the configured width the gate may be shrinking from, and
    the available percentage is the lane's own published figure -- the view
    computes nothing from headroom or width itself, so its figure and the lane
    document's can never disagree.
    """
    table = _grid()
    gate = lane.get("router_generation_gate") or {}
    admission = lane.get("admission") or {}
    slots = lane.get("worker_slots") or {}
    width = _number(gate, "effective_width")
    in_flight = _number(gate, "in_flight")
    held = _number(gate, "waiting")
    oldest = _number(admission, "oldest_wait_seconds")
    verdict = admission.get("verdict")
    headroom = _number(lane, "headroom")
    worker_slots = _number(slots, "worker_slots")
    availability = _number(lane, "availability_percent")
    in_flight_s = (
        (f"{in_flight:.0f}" if in_flight is not None else DASH)
        + "/"
        + (f"{width:.0f}" if width is not None else DASH)
    )
    held_s = (
        f"[red]{held:.0f}[/]" if held is not None and held >= 1
        else f"[dim]{held:.0f}[/]" if held is not None else DASH
    )
    _band(
        table,
        ["in flight", "router q", "oldest wait", "verdict", "headroom",
         "slots", "available %"],
        [
            in_flight_s,
            held_s,
            fmt_span(oldest) if oldest is not None else DASH,
            f"[bold]{verdict}[/]" if verdict else DASH,
            f"[bold]{headroom:.0f}[/]" if headroom is not None else DASH,
            f"{worker_slots:.0f}" if worker_slots is not None else DASH,
            pct(availability, "good") if availability is not None else DASH,
        ],
        "",
    )
    return table


def bands(document: dict[str, Any]) -> list[object]:
    """The three bands, or the one line that stands in for an unread engine's.

    When the live entry is unavailable the throughput band is replaced by one
    line naming the reason. It is not drawn as a row of dashes or of zeros:
    a zero is a measurement the record never made, and a band of them under a
    heading reads as an idle engine rather than as a failed read. The capacity
    and gate bands are read from the lane document, a different published
    source, so those still appear when the receipt tail cannot be read.
    """
    live = document.get("live") or {}
    lane = document.get("lane") or {}
    parts: list[object] = []
    if live.get("state") == "measured":
        parts.append(throughput_band(live))
    else:
        reason = live.get("reason") or "no reading"
        parts.append(Text.from_markup(
            f"[yellow]live entry unavailable[/][dim] — {reason}[/]"
        ))
    parts.append(capacity_band(lane, live))
    parts.append(gate_band(lane))
    return parts


# ── the two levels of the running-workers view ───────────────────────


def _heading(label: str, window_seconds: object) -> str:
    """A table's first column heading, carrying the window it measured."""
    if isinstance(window_seconds, (int, float)) and not isinstance(
        window_seconds, bool
    ):
        return f"{label} {window_seconds / 60:.0f}m"
    return label


def _table(heads: Iterable[str]) -> Table:
    table = Table(box=box.SIMPLE_HEAD, padding=(0, 1), expand=False,
                  pad_edge=False, header_style="dim")
    for index, name in enumerate(heads):
        table.add_column(
            name,
            justify="left" if index == 0 else "right",
            width=LABEL_WIDTH if index == 0 else CELL_WIDTH,
            no_wrap=True,
        )
    return table


def _held(value: object) -> str:
    return f"{float(value):.1f}" if isinstance(value, (int, float)) else DASH


def _per_run(value: object) -> str:
    if not isinstance(value, (int, float)):
        return DASH
    # A fan-out at or below one is a run holding a single request, which is the
    # ordinary case; above it a run is issuing several at once and is marked,
    # because the share divides capacity in runs and cannot see it.
    return f"[{ramp(float(value), 1.0, 2.0, inverted=True)}]{float(value):.1f}[/]"


def accounts_table(callers: dict[str, Any], *, account: str | None = None) -> object:
    """One row per account that held the gate, heaviest first.

    The rows come from the block's ``accounts`` level, which totals the window
    by the account each request reported -- not from the per-session list. A
    session's account is the one its rows most often named, so a table built
    from the sessions would split one account across its sessions, count a
    session that reported two accounts under whichever dominated, and drop
    every request whose row carried no session at all. The level exists
    precisely so the table cannot do that.
    """
    if callers.get("state") != "measured":
        detail = callers.get("detail") or "no reading"
        return Text.from_markup(
            f"[yellow]accounts unavailable[/][dim] — {detail}[/]"
        )
    rows = [row for row in (callers.get("accounts") or []) if isinstance(row, dict)]
    table = _table((_heading("accounts", callers.get("window_seconds")),
                    "sessions", "runs", "in use", "per run",
                    "wait s", "share %"))
    for row in rows:
        name = str(row.get("account") or UNATTRIBUTED)
        style = "bold" if account and name == account else ""
        share = row.get("share")
        wait = row.get("mean_gate_wait_s")
        table.add_row(
            name,
            f"{row.get('sessions') or 0}",
            f"{row.get('runs') or 0}",
            _held(row.get("held")),
            _per_run(row.get("held_per_run")),
            (f"[{ramp(float(wait), 5.0, 30.0, inverted=True)}]{float(wait):.0f}[/]"
             if isinstance(wait, (int, float)) else DASH),
            (pct(100.0 * float(share), "good")
             if isinstance(share, (int, float)) else DASH),
            style=style,
        )
    return table


def _detail_rows(
    callers: dict[str, Any], account: str, sessions: bool
) -> list[dict[str, Any]]:
    """The detail table's rows: one account's projects, or one per session.

    A project row aggregates that account's ``projects`` entries; a session row
    is one entry of the per-session list, labelled by the project it named
    rather than by the session, which is an opaque identifier. Both are sorted
    heaviest first, so the fold below drops the smallest.
    """
    if sessions:
        rows = [
            {
                "label": str(caller.get("project") or UNATTRIBUTED),
                "session": str(caller.get("session") or "")[:4] or DASH,
                "sessions": 1,
                "runs": caller.get("runs"),
                "held": caller.get("held"),
                "held_per_run": caller.get("held_per_run"),
                "mean_gate_wait_s": caller.get("mean_gate_wait_s"),
            }
            for caller in (callers.get("callers") or [])
            if caller.get("account") == account
        ]
    else:
        rows = [
            {
                "label": str(row.get("project") or UNATTRIBUTED),
                "sessions": row.get("sessions"),
                "runs": row.get("runs"),
                "held": row.get("held"),
                "held_per_run": row.get("held_per_run"),
                "mean_gate_wait_s": row.get("mean_gate_wait_s"),
            }
            for row in (callers.get("projects") or [])
            if row.get("account") == account
        ]
    rows.sort(key=lambda row: (-(row.get("held") or 0.0), row["label"]))
    return rows


def detail_table(
    callers: dict[str, Any], *, account: str | None, sessions: bool = False
) -> object:
    """One account's projects, or its sessions, heaviest first.

    At most :data:`DETAIL_ROWS` rows are listed; the rest fold into one
    ``+N more`` row carrying their summed in-use, so the table says the same
    thing about an account with thirty projects as about one with three while
    keeping the folded share visible in the column the reader is comparing. An
    account that held nothing in the window gets one dim line rather than an
    empty table: an empty table is indistinguishable from a table that failed
    to build.
    """
    if callers.get("state") != "measured":
        detail = callers.get("detail") or "no reading"
        return Text.from_markup(
            f"[yellow]account detail unavailable[/][dim] — {detail}[/]"
        )
    name = account or _viewer()
    rows = _detail_rows(callers, name, sessions)
    if not rows:
        return Text.from_markup(
            f"[dim]{name} holds nothing in the window — no finished request "
            f"in the record named it[/]"
        )
    heads = (
        (_heading(name, callers.get("window_seconds")),
         "session" if sessions else "sessions",
         "runs", "in use", "per run", "wait s")
    )
    table = _table(heads)
    for row in rows[:DETAIL_ROWS]:
        wait = row.get("mean_gate_wait_s")
        table.add_row(
            row["label"],
            (row.get("session") or DASH) if sessions
            else f"{row.get('sessions') or 0}",
            f"{row.get('runs') or 0}" if row.get("runs") is not None else DASH,
            _held(row.get("held")),
            _per_run(row.get("held_per_run")),
            (f"[{ramp(float(wait), 5.0, 30.0, inverted=True)}]{float(wait):.0f}[/]"
             if isinstance(wait, (int, float)) else DASH),
        )
    folded = rows[DETAIL_ROWS:]
    if folded:
        held = sum(float(row.get("held") or 0.0) for row in folded)
        table.add_row(f"+{len(folded)} more", "", "", f"{held:.1f}", "", "",
                      style="dim")
    return table


def history_strip(callers: dict[str, Any], lane: dict[str, Any]) -> object:
    """Occupancy and gate wait, minute by minute over the last half hour.

    In flight is drawn against the gate's own width when the lane publishes
    one, so a strip read beside the bands means the same thing on every
    document: a full strip is a full gate, not merely the busiest minute of a
    quiet half hour. The wait strip carries the median wait of the requests
    admitted in each bin, which is the half of the distribution a reader acts
    on; a bin that admitted nobody takes the bottom level, because a gap would
    read as a missing bin rather than a quiet one.
    """
    history = callers.get("history") or {}
    bins = [bin_ for bin_ in (history.get("bins") or []) if isinstance(bin_, dict)]
    if history.get("state") != "measured" or not bins:
        reason = history.get("reason") or "no reading"
        return Text.from_markup(
            f"[yellow]history unavailable[/][dim] — {reason}[/]"
        )
    occupancy = [float(bin_.get("occupancy") or 0.0) for bin_ in bins]
    waits = [float(bin_.get("gate_wait_p50_s") or 0.0) for bin_ in bins]
    gate = lane.get("router_generation_gate") or {}
    width = _number(gate, "effective_width")
    window = history.get("window_seconds")
    label = _heading("history", window)
    return Text.from_markup(
        f"{label}   in flight [bold]{spark(occupancy, width)}[/]"
        f"   wait [cyan]{spark(waits)}[/]"
    )


def outcomes_line(callers: dict[str, Any]) -> object:
    """How the window's requests ended, in one line.

    The four figures partition the window: a request the router answered itself
    after its gate wait ran out is an expiry and not also a failure, so the
    counts never double-count one request. A window in which nothing ended says
    so rather than printing four zeros, which would read as a window nobody
    used.
    """
    outcomes = callers.get("outcomes") or {}
    if outcomes.get("state") != "measured":
        reason = outcomes.get("reason") or "no reading"
        return Text.from_markup(
            f"[yellow]outcomes unavailable[/][dim] — {reason}[/]"
        )
    label = _heading("outcomes", outcomes.get("window_seconds"))
    completed = outcomes.get("completed") or 0
    aborted = outcomes.get("aborted") or 0
    failed = outcomes.get("failed") or 0
    expired = outcomes.get("gate_expired") or 0
    failed_s = f"[red]{failed:.0f} failed[/]" if failed else "0 failed"
    return Text.from_markup(
        f"{label}   {completed:.0f} completed · {aborted:.0f} aborted · "
        f"{failed_s} · {expired:.0f} expired at the end"
    )


def ledger_table(
    ledger: list[dict[str, Any]] | None, note: str | None = None
) -> object:
    """Tokens and their cost, one row per published period.

    Every figure comes from the published watch document, which the recorder
    derives from the lane's own receipts, so a period's tokens and cost cannot
    disagree with the document the rest of the fleet reads. The ``window``
    column is not decoration: a day's row backed by six hours of record is a
    real answer to a different question, and without that column it is
    indistinguishable from a quiet day.
    """
    rows = [row for row in (ledger or []) if isinstance(row, dict)
            and row.get("label")]
    if not rows:
        return Text.from_markup(
            "[yellow]ledger unavailable[/][dim] — the published watch "
            "document carries no period rows[/]"
        )
    table = _table(("ledger", "tok in", "cached %", "tok out", "util %",
                    "cost $", "window"))
    for row in rows:
        tokens_in = float(row.get("tokens_in") or 0.0)
        cached = float(row.get("cached") or 0.0)
        tokens_out = float(row.get("tokens_out") or 0.0)
        utilisation = row.get("utilisation")
        period = float(row.get("period") or 0.0)
        covered = float(row.get("covered") or 0.0)
        share = 100.0 * cached / tokens_in if tokens_in > 0 else None
        cost = row.get("cost")
        table.add_row(
            str(row.get("label")),
            fmt_tokens(tokens_in),
            pct(share, "good") if share is not None else DASH,
            fmt_tokens(tokens_out),
            (pct(float(utilisation))
             if isinstance(utilisation, (int, float))
             and not isinstance(utilisation, bool) else DASH),
            fmt_usd(cost).lstrip("$") if isinstance(cost, (int, float)) else DASH,
            "full" if period and covered >= 0.98 * period else fmt_span(covered),
            style="dim",
        )
    if note:
        return Group(table, Text.from_markup(note))
    return table


# ── the document, and the text ───────────────────────────────────────


def build_document(
    *,
    record_dir: str | Path | None = None,
    lane_path: str | Path | None = None,
    endpoints_path: str | Path | None = None,
    watch_path: str | Path | None = None,
    now: Any = None,
) -> dict[str, Any]:
    """The view's whole input: the live document, the ledger, the endpoint.

    The live document is built on every call, because it is the part that
    changes: it reads the tail of the newest receipt file and the tail of the
    router's request record. The ledger and the endpoint entry describe a
    deployment rather than a moment, so they are read from the published
    documents the router and the recorder already write. The endpoint document
    is read once per build, which is once per refresh, because the checkpoint it
    records changes only when the lane is rotated.
    """
    document = watch_mod.live_document(
        record_dir=record_dir, lane_path=lane_path, now=now
    )
    endpoints = _read_json(
        endpoints_path if endpoints_path is not None else _endpoint_document()
    ) or {}
    document["engine"] = endpoints
    document["checkpoints"] = endpoints.get("checkpoints") or {}
    published = _read_json(watch_path or watch_publish.DEFAULT_OUTPUT) or {}
    document["ledger"] = published.get("ledger") or []
    document["ledger_published_at"] = published.get("published_at")
    return document


def _endpoint_document() -> Path:
    """The published endpoint document, at the path the site declares.

    Imported here rather than at module load because the render path does not
    need the site configuration at all: a caller that hands its own document in
    reads no environment, so a view of a fixture costs no configuration.
    """
    from imas_ambix.agent.profile import SiteConfig

    return SiteConfig.from_env().endpoint_document


def _read_json(path: str | Path | None) -> dict[str, Any] | None:
    """A published JSON document, or None when it is absent or unreadable."""
    if path is None:
        return None
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def render(
    document: dict[str, Any],
    *,
    account: str | None = None,
    sessions: bool = False,
) -> Group:
    """The whole view as one group of rich renderables."""
    callers = document.get("callers") or {}
    parts: list[object] = [Text.from_markup(head_line(document))]
    parts.extend(bands(document))
    parts.append(accounts_table(callers, account=account))
    parts.append(detail_table(callers, account=account, sessions=sessions))
    parts.append(history_strip(callers, document.get("lane") or {}))
    parts.append(outcomes_line(callers))
    parts.append(ledger_table(document.get("ledger")))
    return Group(*parts)


def render_text(
    document: dict[str, Any],
    *,
    account: str | None = None,
    sessions: bool = False,
    width: int = GOLDEN_WIDTH,
) -> str:
    """The view as plain text at *width*, so the output is comparable."""
    import io

    buffer = io.StringIO()
    Console(file=buffer, no_color=True, width=width, highlight=False).print(
        render(document, account=account, sessions=sessions), soft_wrap=False
    )
    return buffer.getvalue()


def main(argv: list[str] | None = None) -> int:
    """The ``llm-watch`` console script."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="llm-watch",
        description="Render the live serving document for one engine.",
    )
    parser.add_argument("--record", dest="record_dir", default=None,
                        help="receipts directory to read")
    parser.add_argument("--lane", dest="lane_path", default=None,
                        help="the router's published lane document")
    parser.add_argument("--endpoints", dest="endpoints_path", default=None,
                        help="the published endpoint document")
    parser.add_argument("--watch", dest="watch_path", default=None,
                        help="the published watch document the ledger is read from")
    parser.add_argument("--engine", dest="engine_id", default=None,
                        help="the engine whose bands are shown; this view serves "
                             "one engine, so any other id is refused")
    parser.add_argument("--account", dest="account", default=None,
                        help="the account whose detail table is shown")
    parser.add_argument("--sessions", action="store_true",
                        help="detail table lists coordinator sessions")
    parser.add_argument("--json", dest="as_json", action="store_true",
                        help="emit the document as JSON")
    parser.add_argument("--live", action="store_true",
                        help=f"refresh every {int(LIVE_REFRESH_SECONDS)}s")
    parser.add_argument("--width", type=int, default=GOLDEN_WIDTH)
    args = parser.parse_args(argv)

    def once() -> None:
        document = build_document(
            record_dir=args.record_dir,
            lane_path=args.lane_path,
            endpoints_path=args.endpoints_path,
            watch_path=args.watch_path,
        )
        if args.engine_id:
            refusal = engine_refusal(document, args.engine_id)
            if refusal is not None:
                sys.stderr.write(refusal + "\n")
                raise SystemExit(REFUSED_ENGINE_STATUS)
            document["selected_engine"] = args.engine_id
        if args.as_json:
            sys.stdout.write(json.dumps(document, indent=2, default=str) + "\n")
        else:
            sys.stdout.write(
                render_text(
                    document,
                    account=args.account,
                    sessions=args.sessions,
                    width=args.width,
                )
            )
        sys.stdout.flush()

    if not args.live:
        once()
        return 0
    for _ in refresh_loop(once):
        pass
    return 0


def refresh_loop(
    once: Callable[[], None],
    *,
    interval: float = LIVE_REFRESH_SECONDS,
    sleep: Callable[[float], None] = _time.sleep,
    ticks: int | None = None,
) -> Iterable[int]:
    """Call *once* now and every *interval* seconds until stopped.

    The sleep and the tick count are parameters so both a terminal run and a
    test drive the same loop: the interval is the measured cadence rather than
    a number the test restates, and a bounded run is the same code path as an
    unbounded one.
    """
    tick = 0
    while ticks is None or tick < ticks:
        once()
        yield tick
        tick += 1
        if ticks is not None and tick >= ticks:
            return
        sleep(interval)


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
