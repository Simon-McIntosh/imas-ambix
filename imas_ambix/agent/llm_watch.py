"""The llm-watch view: one engine's live serving document, rendered as text.

The view renders the live serving document and nothing else. It reads the
published lane document, the tail of the recorder's newest raw receipt file,
the tail of the router's request record and the published watch document for
the period ledger -- every figure the serve panel draws is already published
somewhere, so nothing here probes the engine, opens the telemetry index or
fetches a price.

The view is one column of titled sections as wide as the terminal, in the
order a reader asks about a lane:

===========  ==========================================================
section      answers
===========  ==========================================================
lane         three bands. live: requests in use on the cards, those
             waiting at the gate, the aggregate and per-stream rates,
             the prefill rate and the time to a first token. queue:
             the work lined up for the lane, launched and not yet
             launched. memory: whether memory is the limit
accounts     who is using the lane
projects     what one account is running on it
activity     how the last half hour went, and how requests ended
ledger       what the lane has cost
===========  ==========================================================

The sections and the tables inside them are ``ambix-watch``'s own, and that
tool draws its lanes with the LANE section from here, so a reader moving
between the two reads one set of figures laid out one way. Each band holds six
figures. The ledger's coverage is a column only when a period is short of its
record.

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
that nothing was read. When the live entry itself is unavailable, one line
under the serving band names the reason.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import sys
import time as _time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.console import Console, Group
from rich.markup import escape
from rich.text import Text

from imas_ambix.agent import watch as watch_mod
from imas_ambix.agent import watch_publish, work_queue
from imas_ambix.agent.registry import _engine_id
from imas_ambix.agent.watch_style import (
    DASH,
    DEFAULT_WIDTH,
    NARROW_WIDTH,
    bar_width,
    cell,
    figure_bar,
    flex_table,
    fmt_count,
    fmt_ctx,
    fmt_rate,
    fmt_span,
    fmt_tokens,
    fmt_usd,
    head,
    page,
    page_width,
    pct,
    ramp,
    section,
    spark,
    split_line,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from rich.constrain import Constrain
    from rich.table import Table

#: How long the ``--live`` view waits between refreshes.
LIVE_REFRESH_SECONDS = 5.0

#: The label a row carries when the caller reported no account or no project.
UNATTRIBUTED = "unattributed"

#: Projects listed in the detail table before the rest fold into one row.
DETAIL_ROWS = 6

#: The naming columns of the engines table: a model id and an engine id, each
#: sized to survive whole, so two engines on one node are told apart by port.
ENGINE_NAME_WIDTHS = (20, 16)

#: The width of an engine's name wherever one labels a row: a node and a port.
ENGINE_CELL_WIDTH = 16
ENGINE_STRIP_WIDTH = 16

#: The naming columns of the other tables.
BAND_NAME_WIDTH = 8
ACCOUNT_NAME_WIDTH = 14
PROJECT_NAME_WIDTH = 20
PERIOD_WIDTH = 8
MODEL_WIDTH = 20

#: Seconds after which the published ledger is labelled with its age: twice
#: the publisher's cadence.
LEDGER_STALE_SECONDS = 600.0

#: Per-worker decode reference, tok/s: the best single-stream rate measured on
#: this deployment. The per-worker cell is coloured against this rather than
#: against an absolute rate, so the figure answers whether each worker is
#: getting a fair share of what one worker can have. The value is the serve
#: panel's own, so the same rate takes the same colour on both.
PER_WORKER_REF = 40.0

#: The width a golden is rendered at.
GOLDEN_WIDTH = DEFAULT_WIDTH

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
    for entry in document_engines(document):
        if engine_id in _engine_names(entry):
            return None
    served = served_engine(document)
    if engine_id == served:
        return None
    if served is None and not document_engines(document):
        return (f"llm-watch: this document names no served engine, so "
                f"{engine_id!r} cannot be selected")
    known = sorted(
        {_engine_display(entry.get("engine") or entry.get("engine_id"))
         for entry in document_engines(document)}
    )
    if known:
        return (f"llm-watch: this document describes "
                f"{', '.join(known)}, not {engine_id!r}")
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


def _lane_title(document: dict[str, Any]) -> tuple[str, str]:
    """The lane section's title and note: what is served, and from what weights.

    Everything here is constant for the life of the deployment, which is why it
    sits above the bands and not in them: a figure that cannot change during a
    session is not a reading.
    """
    lane = document.get("lane") or {}
    live = document.get("live") or {}
    model = live.get("served_name") or lane.get("model_id") or "?"
    endpoint = engine_endpoint(
        document.get("engine") or {}, lane.get("model_id")
    )
    host = str(endpoint.get("host") or "").split(".", 1)[0]
    bits = []
    count = endpoint.get("accelerator_count")
    family = endpoint.get("accelerator_family")
    if count and family:
        bits.append(f"{count}×{family}")
    elif family:
        bits.append(str(family))
    if host:
        bits.append(host.removeprefix("98dci4-"))
    ceiling = context_ceiling(lane, endpoint)
    if ceiling is not None:
        bits.append(f"{fmt_ctx(ceiling)} ctx")
    checkpoint = checkpoint_label(document)
    return lane_title(str(model), bits), checkpoint_note(checkpoint)


def lane_title(model: str, facts: list[str]) -> str:
    """A lane section's title: the served name, then what it is served on."""
    detail = f" [dim]· {' · '.join(facts)}[/]" if facts else ""
    return f"[bold]lane {escape(model)}[/]{detail}"


def checkpoint_note(checkpoint: str | None) -> str:
    """The checkpoint as a lane section's note, or that none is recorded."""
    if checkpoint:
        return f"[cyan]{escape(checkpoint)}[/]"
    return f"[yellow]{CHECKPOINT_UNRECORDED}[/]"


def head_line(document: dict[str, Any]) -> str:
    """The lane's heading as one string: the served engine, then its checkpoint."""
    title, note = _lane_title(document)
    return f"{title} [dim]·[/] {note}"


# ── the lane's bands ──────────────────────────────────────────────────


def _band(name: str, labels: list[str], values: list[str]) -> Table:
    """One named band: a row of labels over a row of figures.

    The name stands in a short column at the left, which keeps the first
    figure clear of the margin. A band's figure columns are its own: the live
    row carries one figure more than the memory row, and each is spread over
    the page's whole width.
    """
    table = flex_table((BAND_NAME_WIDTH,), len(labels))
    table.add_row(f"[bold]{name}[/]", *[f"[dim]{label}[/]" for label in labels])
    table.add_row("", *values)
    return table


def alarm_count(value: float | None) -> str:
    """A count that is ordinary at zero and worth seeing above it."""
    if value is None:
        return DASH
    return f"[red]{value:.0f}[/]" if value >= 1 else f"[dim]{value:.0f}[/]"


def backlog_count(value: float | None, width: float | None = None) -> str:
    """A queue at the gate, coloured by how well it keeps the lane fed.

    Work waiting is what keeps the cards full, so the queue is green once it
    holds at least a gate's width of requests, yellow from half of that, and
    red below. Without a width to measure against, the count is uncoloured.
    """
    if value is None:
        return DASH
    if not width:
        return f"{value:.0f}"
    return f"[{ramp(100.0 * value / width, 100.0, 50.0)}]{value:.0f}[/]"


def serving_band(lane: dict[str, Any], live: dict[str, Any]) -> Table:
    """The lane's one row of live figures: its use, its queues and its rates.

    ``in use`` is the requests generating now over the gate's width in force:
    what the lane is carrying, never below zero and never above the width. It
    is coloured as a utilisation, green when the gate is full. ``gate wait``
    is the requests the gate is holding back, which have reached no card.
    They come from workers already running and wait seconds, so the figure
    says how hard the gate is pressed and not how much work is lined up: that
    is the queue band. It is green once at least a gate's width of requests
    wait. ``held`` is the requests the gate admitted
    that the engine has not scheduled, which is zero in ordinary running and
    the first sign the engine, and not the gate, is the limit. The lane
    document's ``headroom`` is drawn nowhere, because it subtracts the queue
    from the room and so goes negative whenever requests wait. ``tok/s`` is
    every stream together, ``/stream`` is what one request sees,
    ``prefill/s`` is prompt tokens read per second and ``ttft s`` the median
    time to a first token. A figure the receipt record did not measure is a
    dash.
    """
    gate = lane.get("router_generation_gate") or {}
    slots = _number(gate, "effective_width")
    in_flight = _number(gate, "in_flight")
    in_use = (
        (f"{in_flight:.0f}" if in_flight is not None else DASH)
        + "/"
        + (f"{slots:.0f}" if slots is not None else DASH)
    )
    share = 100.0 * in_flight / slots if in_flight is not None and slots else None
    measured = live.get("state") == "measured"
    aggregate = _number(live, "generation_toks_per_s")
    per_stream = _number(live, "per_stream_toks_per_s")
    prefill = _number(live, "prefill_toks_per_s")
    ttft = _number(live, "median_time_to_first_token_s")
    return _band(
        "live",
        ["in use", "gate wait", "held", "tok/s", "/stream", "prefill/s", "ttft s"],
        [
            (f"[bold {ramp(share, 80.0, 40.0)}]{in_use}[/]" if share is not None
             else f"[bold]{in_use}[/]"),
            backlog_count(_number(gate, "waiting"), slots),
            alarm_count(_number(live, "engine_queue")),
            (f"[bold]{fmt_rate(aggregate)}[/]" if measured and aggregate
             else DASH),
            (f"[bold {ramp(100.0 * per_stream / PER_WORKER_REF, 60.0, 30.0)}]"
             f"{per_stream:.1f}[/]"
             if measured and per_stream else DASH),
            (f"[bold]{fmt_rate(prefill)}[/]"
             if measured and prefill is not None else DASH),
            f"{ttft:.1f}" if measured and ttft is not None else DASH,
        ],
    )


def queue_band(queue: dict[str, Any], lane: dict[str, Any]) -> Table:
    """The work lined up for the lane, set under the live row's columns.

    Each figure stands under the live figure it is read against. ``workers``
    are runs launched and still going, under the requests in use: a worker
    holds a gate slot only while it generates, so workers outnumber requests.
    ``ready`` are requests waiting only for a worker slot, under the requests
    waiting at the gate; they are what keeps the cards full, so the count is
    green from a gate's width of them, yellow from half, red below.
    ``paused`` are set aside by a decision or a failed launch. ``waiting``
    wait on a plan section that has not landed. ``slots`` is how many more
    workers the router would admit now. ``oldest ready`` is how long the
    longest-waiting ready request has queued, at the right edge under the
    time to a first token.
    """
    slots = _number(lane.get("router_generation_gate") or {}, "effective_width")
    return _band(
        "queue",
        ["workers", "ready", "paused", "waiting", "slots", "", "oldest ready"],
        [
            fmt_count(queue.get("workers")),
            backlog_count(_number(queue, "ready"), slots),
            fmt_count(queue.get("paused")),
            fmt_count(queue.get("waiting")),
            fmt_count(lane.get("worker_slots")),
            "",
            fmt_span(queue.get("oldest_ready_seconds")),
        ],
    )


def memory_band(
    lane: dict[str, Any], live: dict[str, Any], width: int = DEFAULT_WIDTH
) -> Table:
    """Whether memory is the limit: the pool the engine holds, and what fits.

    ``kv pool`` is the share of the device pool that requests hold, which is
    the figure that binds. ``host tier`` is the host-memory tier evicted
    prefixes are kept in: it fills and stays full by design, so it is drawn
    without a judgement. ``cache hit`` is the share of prompt tokens served
    from cache, ``max reqs`` is how many requests of the mean working context
    the pool holds. Each figure is read under the name its document publishes
    it by, and one the document does not carry is a dash.
    """
    # Pool occupancy is the one capacity figure the live block carries, because
    # it is read from the engine's own counters and not from the lane window.
    pool = _number(live, "pool_occupancy")
    host_tier = _nested(lane, "hicache_host", "used_fraction")
    hit = _number(live, "cache_hit_rate")
    measured = live.get("state") == "measured"
    mean_ctx = _number(lane, "mean_context") or _number(lane, "mean_context_instant")
    labels = ["kv pool %", "host tier %", "cache hit %", "mean ctx", "max reqs",
              "preempted"]
    bar = figure_bar(width, BAND_NAME_WIDTH, len(labels))
    return _band(
        "memory",
        labels,
        [
            (pct(pool * 100.0 if pool <= 1.0 else pool, width=bar)
             if pool is not None else DASH),
            (pct(host_tier * 100.0, "plain", width=bar)
             if host_tier is not None else DASH),
            (pct(100.0 * hit, "good", width=bar)
             if measured and hit is not None else DASH),
            fmt_ctx(mean_ctx),
            cell(_number(lane, "concurrent_requests"), 1.0, 1.0),
            cell(_number(lane, "preemptions"), 0.0, 0.0, inverted=True),
        ],
    )


def bands(
    document: dict[str, Any], *, capacity: bool = True,
    width: int = DEFAULT_WIDTH,
) -> list[object]:
    """The bands in reading order: the live figures, the queue, then memory.

    The queue band is drawn when the document carries the dispatcher's count
    of the work lined up for the lane.

    When the live entry is unavailable its figures are dashes and one line
    names the reason: a zero is a measurement the record never made. The
    requests in use, the gate's wait and the memory figures are read from the
    lane document, a different published source, so those still appear when
    the receipt tail cannot be read.
    """
    live = document.get("live") or {}
    lane = document.get("lane") or {}
    parts: list[object] = [serving_band(lane, live)]
    if live.get("state") != "measured":
        reason = live.get("reason") or "no reading"
        parts.append(Text.from_markup(
            f"[yellow]live entry unavailable[/]"
            f"[dim] — {escape(str(reason))}[/]"
        ))
    queue = document.get("queue")
    if isinstance(queue, dict):
        parts += [Text(""), queue_band(queue, lane)]
    if capacity:
        parts += [Text(""), memory_band(lane, live, width)]
    return parts


def lane_section(
    document: dict[str, Any],
    width: int,
    *,
    capacity: bool = True,
    title: str | None = None,
    note: str | None = None,
) -> list[object]:
    """One lane as a section: named in the title, its checkpoint beside it.

    ``title`` and ``note`` replace the ones read from the document, for a
    caller that knows more about where the lane runs than the document says.
    """
    own_title, own_note = _lane_title(document)
    return section(
        title or own_title,
        bands(document, capacity=capacity, width=width),
        width,
        note=own_note if note is None else note,
    )


# ── several engines ──────────────────────────────────────────────────


def document_engines(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Every engine entry the document carries, in the order it lists them."""
    return [
        entry for entry in (document.get("engines") or [])
        if isinstance(entry, dict)
    ]


def multi_engine(document: dict[str, Any]) -> bool:
    """Whether the document describes more than one engine.

    One engine is the whole of today's view: no router line, no engines table,
    no engine column. The engine dimension is drawn only when there is a second
    engine to tell the first apart from, so a single-engine reader's document
    and render are unchanged by it.
    """
    return len(document_engines(document)) > 1


def engine_document(
    document: dict[str, Any], entry: dict[str, Any]
) -> dict[str, Any]:
    """The whole view document, narrowed to one engine's own figures.

    The bands read a ``live`` block and a lane document. Both exist per engine
    -- the live entry from the recorder and the pool document's own engine
    document -- so the engine's two are put where the bands look and the rest
    of the document is carried through, keeping the endpoint and the ledger.
    """
    lane = dict(document.get("lane") or {})
    gate = entry.get("gate") or {}
    for key, value in gate.items():
        if key != "engines":
            lane[key] = value
    if entry.get("model"):
        lane["model_id"] = entry["model"]
    return {**document, "lane": lane, "live": entry.get("live") or {}}


def _pressure(entry: dict[str, Any]) -> float:
    """How hard an engine is working, for choosing which bands to draw."""
    running = _number(entry, "running")
    available = _number(entry, "available")
    if running is None:
        return -1.0
    return running if available is None else running + 100.0 - available


def selected_engine(
    document: dict[str, Any], engine_id: str | None = None
) -> dict[str, Any] | None:
    """The engine whose bands are drawn: the named one, else the busiest.

    With several engines the terminal cannot hold every engine's bands,
    so one engine's are drawn. ``--engine`` chooses; with no choice, the engine
    under most pressure is the one a reader needs first.
    """
    engines = document_engines(document)
    if not engines:
        return None
    if engine_id:
        for entry in engines:
            if engine_id in _engine_names(entry):
                return entry
        return None
    return max(engines, key=_pressure)


def _engine_names(entry: dict[str, Any]) -> set[str]:
    """Every id an engine entry answers to on the command line."""
    names = {
        value for value in (entry.get("engine"), entry.get("engine_id"))
        if isinstance(value, str) and value
    }
    return names | {_engine_display(name) for name in names}


def _engine_display(engine_id: object) -> str:
    """An engine id as the tables print it: short node name and port.

    The record names an upstream by its origin, and a registration names the
    host as the node reported it. Both spellings reduce to the node name the
    registry's engine ids carry, so one engine reads the same in the engines
    table, the account detail and the history strip.
    """
    if not isinstance(engine_id, str) or not engine_id.strip():
        return DASH
    text = engine_id.strip()
    if "://" in text:
        text = text.split("://", 1)[1]
    text = text.split("/", 1)[0]
    host, separator, port = text.rpartition(":")
    if not separator:
        return text
    from imas_ambix.agent.registry import host_name

    return f"{host_name(host)}:{port}"


def _count(total: int, noun: str) -> str:
    return f"{total} {noun}" if total == 1 else f"{total} {noun}s"


def router_line(document: dict[str, Any], width: int = DEFAULT_WIDTH) -> object:
    """One line naming the router, how much it spans, and when it was read.

    The engine tables below are all of one document, so the line states how
    much that document spans -- servers, engines and models -- before a reader
    starts attributing rows to one of them.
    """
    engines = document_engines(document)
    models = {str(entry["model"]) for entry in engines if entry.get("model")}
    servers = {
        name for entry in engines if (name := server_name(entry)) is not None
    }
    bits = []
    address = document.get("router")
    if isinstance(address, str) and address:
        bits.append(f"[cyan]{_engine_display(address)}[/]")
    bits.append(_count(len(servers), "server"))
    bits.append(_count(len(engines), "engine"))
    bits.append(_count(len(models), "model"))
    stamp = (document.get("lane") or {}).get("observed_at")
    when = stamp[11:19] if isinstance(stamp, str) and len(stamp) >= 19 else ""
    return split_line(
        "[bold]router[/]  " + " [dim]·[/] ".join(bits), f"[dim]{when}[/]", width
    )


def server_name(entry: dict[str, Any]) -> str | None:
    """The node an engine entry runs on, without its port or cluster prefix."""
    host = entry.get("host") or entry.get("engine")
    if not isinstance(host, str) or not host:
        return None
    text = host.split("://", 1)[-1].split("/", 1)[0].split(":", 1)[0]
    from imas_ambix.agent.registry import host_name

    return host_name(text)


def engines_table(
    document: dict[str, Any], *, engine: str | None = None,
    width: int = DEFAULT_WIDTH,
) -> Table:
    """One row per engine: its load, its rate and the room its gate has left.

    The rate and the time to first token are the recorder's, the queue and the
    room are the router's, and the KV pool is the engine's own counter. A
    figure a source did not publish is a dash, never a zero. The row whose
    bands are drawn below is bold. A narrow page keeps the load, the rate and
    the room, and leaves the time to first token and the pool to the bands.
    """
    narrow = width < DEFAULT_WIDTH
    labels = ["model", "engine", "running", "waiting", "tok/s",
              "/stream",
              *(() if narrow else ("ttft s", "kv pool %")), "room %"]
    table = flex_table(ENGINE_NAME_WIDTHS, len(labels) - 2)
    head(table, labels)
    chosen = selected_engine(document, engine)
    for entry in document_engines(document):
        kv_pool = _number(entry, "kv_pool")
        running = _number(entry, "running")
        per_stream = _number(entry, "per_stream")
        ttft = _number(entry, "ttft")
        table.add_row(
            escape(str(entry.get("model") or DASH)),
            _engine_display(entry.get("engine") or entry.get("engine_id")),
            f"[cyan]{running:.1f}[/]" if running is not None else DASH,
            backlog_count(_number(entry, "router_q")),
            fmt_rate(_number(entry, "tok_s")),
            f"{per_stream:.1f}" if per_stream is not None else DASH,
            *(() if narrow else (
                f"{ttft:.1f}" if ttft is not None else DASH,
                f"{100.0 * kv_pool:.0f}" if kv_pool is not None else DASH,
            )),
            cell(_number(entry, "available"), 40.0, 10.0),
            style="bold" if entry is chosen else "",
        )
    return table


# ── the two levels of the running-workers view ───────────────────────


def _heading(label: str, window_seconds: object) -> str:
    """A panel's name, carrying the window it measured."""
    if isinstance(window_seconds, (int, float)) and not isinstance(
        window_seconds, bool
    ):
        return f"{label} {window_seconds / 60:.0f}m"
    return label


def _held(value: object) -> str:
    return f"{float(value):.1f}" if isinstance(value, (int, float)) else DASH


def _per_run(value: object) -> str:
    if not isinstance(value, (int, float)):
        return DASH
    # A fan-out at or below one is a run holding a single request, which is the
    # ordinary case; above it a run is issuing several at once and is marked,
    # because the share divides capacity in runs and cannot see it.
    return f"[{ramp(float(value), 1.0, 2.0, inverted=True)}]{float(value):.1f}[/]"


def _wait(value: object) -> str:
    if not isinstance(value, (int, float)):
        return DASH
    return f"[{ramp(float(value), 5.0, 30.0, inverted=True)}]{float(value):.0f}[/]"


def accounts_table(
    callers: dict[str, Any], *, account: str | None = None,
    width: int = DEFAULT_WIDTH,
) -> object:
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
            f"[yellow]unavailable[/][dim] — {escape(str(detail))}[/]"
        )
    rows = [row for row in (callers.get("accounts") or []) if isinstance(row, dict)]
    table = flex_table((ACCOUNT_NAME_WIDTH,), 6)
    head(table, ["account", "sessions", "runs", "in use", "per run", "wait s",
                 "share %"])
    for row in rows:
        name = str(row.get("account") or UNATTRIBUTED)
        share = row.get("share")
        table.add_row(
            escape(name),
            f"{row.get('sessions') or 0}",
            f"{row.get('runs') or 0}",
            _held(row.get("held")),
            _per_run(row.get("held_per_run")),
            _wait(row.get("mean_gate_wait_s")),
            (pct(100.0 * float(share), "good", width=bar_width(width))
             if isinstance(share, (int, float)) else DASH),
            style="bold" if account and name == account else "",
        )
    return table


def _dominant_engine(row: dict[str, Any]) -> str | None:
    """The engine one row's held time went to most, when a split was published."""
    by_engine = row.get("held_by_engine")
    if not isinstance(by_engine, dict) or not by_engine:
        return None
    return max(by_engine, key=lambda name: by_engine[name])


def _detail_rows(
    callers: dict[str, Any], account: str, sessions: bool
) -> list[dict[str, Any]]:
    """The detail table's rows: one account's projects, or one per session.

    A project row aggregates that account's ``projects`` entries; a session row
    is one entry of the per-session list, labelled by the project it named
    and not by the session, which is an opaque identifier. Both are sorted
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
                "engine": _dominant_engine(caller),
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
                "engine": _dominant_engine(row),
            }
            for row in (callers.get("projects") or [])
            if row.get("account") == account
        ]
    rows.sort(key=lambda row: (-(row.get("held") or 0.0), row["label"]))
    return rows


def detail_table(
    callers: dict[str, Any],
    *,
    account: str | None,
    sessions: bool = False,
    engines: bool = False,
) -> object:
    """One account's projects, or its sessions, heaviest first.

    At most :data:`DETAIL_ROWS` rows are listed; the rest fold into one
    ``+N more`` row carrying their summed in-use, so the table says the same
    thing about an account with thirty projects as about one with three while
    keeping the folded share visible in the column the reader is comparing. An
    account that held nothing in the window gets one dim line and no table: an
    empty table is indistinguishable from a table that failed to build.
    """
    if callers.get("state") != "measured":
        detail = callers.get("detail") or "no reading"
        return Text.from_markup(
            f"[yellow]unavailable[/][dim] — {escape(str(detail))}[/]"
        )
    name = account or _viewer()
    rows = _detail_rows(callers, name, sessions)
    if not rows:
        return Text.from_markup(
            f"[dim]{escape(name)} holds nothing in the window — no finished "
            f"request named it[/]"
        )
    labels = ["project", *(("engine",) if engines else ()),
              "session" if sessions else "sessions", "runs", "in use",
              "per run", "wait s"]
    names = (PROJECT_NAME_WIDTH, ENGINE_CELL_WIDTH) if engines else (
        PROJECT_NAME_WIDTH,)
    table = flex_table(names, 5)
    head(table, labels)
    for row in rows[:DETAIL_ROWS]:
        table.add_row(
            escape(row["label"]),
            *((_engine_display(row.get("engine")),) if engines else ()),
            (row.get("session") or DASH) if sessions
            else f"{row.get('sessions') or 0}",
            f"{row.get('runs') or 0}" if row.get("runs") is not None else DASH,
            _held(row.get("held")),
            _per_run(row.get("held_per_run")),
            _wait(row.get("mean_gate_wait_s")),
        )
    folded = rows[DETAIL_ROWS:]
    if folded:
        held = sum(float(row.get("held") or 0.0) for row in folded)
        table.add_row(f"+{len(folded)} more",
                      *((DASH,) if engines else ()),
                      "", "", f"{held:.1f}", "", "", style="dim")
    return table


#: The width of a history strip's column: one character per bin.
STRIP_WIDTH = 30

#: The narrowest an outcome count's column may be beside the strips.
COUNT_WIDTH = 8


def _history_rows(
    callers: dict[str, Any], lane: dict[str, Any]
) -> list[tuple[str, str, str]] | str:
    """Occupancy and gate wait strips per row, or the reason there are none.

    In flight is drawn against the gate's own width when the lane publishes
    one, so a strip read beside the bands means the same thing on every
    document: a full strip is a full gate, not merely the busiest minute of a
    quiet half hour. The wait strip carries the median wait of the requests
    admitted in each bin; a bin that admitted nobody takes the bottom level,
    because a gap would read as a missing bin and not a quiet one. With
    several engines each has its own row under the total's, so a reader sees
    which engine is filling and not only that the lane is.
    """
    history = callers.get("history") or {}
    bins = [bin_ for bin_ in (history.get("bins") or []) if isinstance(bin_, dict)]
    if history.get("state") != "measured" or not bins:
        return str(history.get("reason") or "no reading")
    width = _number(lane.get("router_generation_gate") or {}, "effective_width")

    def strips(who: str, bins: list[dict[str, Any]]) -> tuple[str, str, str]:
        occupancy = [float(bin_.get("occupancy") or 0.0) for bin_ in bins]
        waits = [float(bin_.get("gate_wait_p50_s") or 0.0) for bin_ in bins]
        return (who, f"[bold]{spark(occupancy, width)}[/]",
                f"[cyan]{spark(waits)}[/]")

    rows = [strips("", bins)]
    for entry in history.get("engines") or []:
        if not isinstance(entry, dict):
            continue
        engine_bins = [
            bin_ for bin_ in (entry.get("bins") or []) if isinstance(bin_, dict)
        ]
        if engine_bins:
            rows.append(strips(_engine_display(entry.get("engine")), engine_bins))
    return rows


def _outcome_rows(callers: dict[str, Any]) -> list[tuple[str, ...]] | str:
    """How the window's requests ended per row, or the reason nothing is known.

    The four figures partition the window: a request the router answered itself
    after its gate wait ran out is an expiry and not also a failure, so the
    counts never double-count one request. With several engines each has its
    own row under the total, because a lane in which one engine failed every
    request it accepted and another completed them all reads as healthy in
    the total alone.
    """
    outcomes = callers.get("outcomes") or {}
    if outcomes.get("state") != "measured":
        return str(outcomes.get("reason") or "no reading")

    def row(who: str, entry: dict[str, Any]) -> tuple[str, ...]:
        failed = entry.get("failed") or 0
        expired = entry.get("gate_expired") or 0
        return (
            who,
            f"{entry.get('completed') or 0:.0f}",
            f"{entry.get('aborted') or 0:.0f}",
            f"[red]{failed:.0f}[/]" if failed else "[dim]0[/]",
            f"[red]{expired:.0f}[/]" if expired else "[dim]0[/]",
        )

    rows = [row("", outcomes)]
    for entry in outcomes.get("by_engine") or []:
        if isinstance(entry, dict):
            rows.append(row(_engine_display(entry.get("engine")), entry))
    return rows


def _named(rows: list[tuple[str, ...]]) -> list[tuple[str, ...]]:
    """Rows with the first one named, when engine rows follow it."""
    if len(rows) > 1:
        return [("[dim]all engines[/]", *rows[0][1:]), *rows[1:]]
    return rows


def _strip_table(counts: int, width: int) -> Table:
    """A name column, two strip columns, then ``counts`` figure columns.

    The strips are never cut: on a body ``width`` columns wide too narrow for
    a whole engine name beside them, the name column gives way.
    """
    name = max(1, min(ENGINE_STRIP_WIDTH, width - 2 * (STRIP_WIDTH + 1)))
    table = flex_table((name, STRIP_WIDTH, STRIP_WIDTH), counts)
    if not counts:
        table.expand = False
    return table


def history_strip(
    callers: dict[str, Any], lane: dict[str, Any], width: int = DEFAULT_WIDTH
) -> object:
    """The half hour's occupancy and gate wait, one row and one per engine."""
    rows = _history_rows(callers, lane)
    if isinstance(rows, str):
        return Text.from_markup(
            f"[yellow]history unavailable[/][dim] — {escape(rows)}[/]"
        )
    table = _strip_table(0, width)
    head(table, ["", "in flight", "gate wait"])
    for row in _named(rows):
        table.add_row(*row)
    return table


def outcomes_line(callers: dict[str, Any]) -> object:
    """How the window's requests ended, one row and one per engine."""
    rows = _outcome_rows(callers)
    if isinstance(rows, str):
        return Text.from_markup(
            f"[yellow]outcomes unavailable[/][dim] — {escape(rows)}[/]"
        )
    table = flex_table((ENGINE_STRIP_WIDTH,), 4)
    head(table, ["", *_outcome_labels(callers)])
    for row in _named(rows):
        table.add_row(*row)
    return table


def _history_window(callers: dict[str, Any]) -> object:
    return (callers.get("history") or {}).get("window_seconds")


def _outcome_labels(_callers: dict[str, Any]) -> list[str]:
    return ["done", "aborted", "failed", "expired"]


def activity_title(callers: dict[str, Any]) -> str:
    """The activity section's title, naming the window each half covers."""
    spans = [
        _heading(name, (callers.get(key) or {}).get("window_seconds"))
        for name, key in (("strips", "history"), ("counts", "outcomes"))
    ]
    return f"[bold]activity[/]  [dim]{' · '.join(spans)}[/]"


def activity(
    callers: dict[str, Any], lane: dict[str, Any], width: int
) -> list[object]:
    """What the lane has been doing: the half hour's strips and how requests ended.

    On a body ``width`` columns wide enough, each row is one line: the two
    strips, then the four outcome counts beside them. Narrower, or when the
    two are not known for the same rows, the strips sit above the counts.
    """
    history, outcomes = _history_rows(callers, lane), _outcome_rows(callers)
    needed = ENGINE_STRIP_WIDTH + 2 * (STRIP_WIDTH + 1) + 4 * COUNT_WIDTH
    joined = (
        not isinstance(history, str)
        and not isinstance(outcomes, str)
        and [row[0] for row in history] == [row[0] for row in outcomes]
        and width >= needed
    )
    if not joined:
        return [history_strip(callers, lane, width), Text(""),
                outcomes_line(callers)]
    table = _strip_table(4, width)
    head(table, ["", "in flight", "gate wait", *_outcome_labels(callers)])
    for strips, counts in zip(_named(history), _named(outcomes), strict=True):
        table.add_row(*strips, *counts[1:])
    return [table]


def _published_age(stamp: object) -> float | None:
    """Seconds since a ``YYYY-MM-DDTHH:MM:SSZ`` stamp, or None when unreadable."""
    import calendar

    if not isinstance(stamp, str):
        return None
    try:
        moment = _time.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return None
    return max(0.0, _time.time() - calendar.timegm(moment))


def ledger_table(
    ledger: list[dict[str, Any]] | None,
    note: str | None = None,
    published_at: object = None,
    width: int = DEFAULT_WIDTH,
) -> object:
    """Tokens and their cost, one row per published period.

    Every figure comes from the published watch document, which the recorder
    derives from the lane's own receipts, so a period's tokens and cost cannot
    disagree with the document the rest of the fleet reads. A ``covered``
    column is drawn only when some period is backed by less than its whole
    record: a day's row backed by six hours answers a different question, and
    a column that reads ``full`` on every row says nothing. A ledger of
    several models is one table with a model column, because stacking one
    model's periods under another's would put two rows of one period together
    with nothing saying which model priced either. A narrow page that needs
    both extra columns gives up the cached share, which the lane's own cache
    figure already reports.
    """
    rows = [row for row in (ledger or []) if isinstance(row, dict)
            and row.get("label")]
    if not rows:
        return Text.from_markup(
            "[yellow]unavailable[/][dim] — the published watch document "
            "carries no period rows[/]"
        )

    def covered(row: dict[str, Any]) -> bool:
        period = float(row.get("period") or 0.0)
        return bool(period) and float(row.get("covered") or 0.0) >= 0.98 * period

    grouped = len({str(row.get("model")) for row in rows if row.get("model")}) > 1
    partial = not all(covered(row) for row in rows)
    cached_shown = not (grouped and partial and width < NARROW_WIDTH)
    labels = ["period", *(("model",) if grouped else ()),
              "tok in", *(("cached %",) if cached_shown else ()),
              "tok out", "util %", "cost $",
              *(("covered",) if partial else ())]
    names = (PERIOD_WIDTH, MODEL_WIDTH) if grouped else (PERIOD_WIDTH,)
    bar = figure_bar(width, sum(names), len(labels) - len(names))
    table = flex_table(names, len(labels) - len(names))
    head(table, labels)
    for row in rows:
        tokens_in = float(row.get("tokens_in") or 0.0)
        cached = float(row.get("cached") or 0.0)
        utilisation = _number(row, "utilisation")
        cost = row.get("cost")
        cells = [
            str(row.get("label")),
            *((escape(str(row.get("model") or DASH)),) if grouped else ()),
            fmt_tokens(tokens_in),
            *(((pct(100.0 * cached / tokens_in, "good", width=bar)
                if tokens_in > 0 else DASH),) if cached_shown else ()),
            fmt_tokens(float(row.get("tokens_out") or 0.0)),
            # A lane is paid for whether or not it is used, so a full period is
            # the healthy one.
            (pct(utilisation, "good", width=bar)
             if utilisation is not None else DASH),
            fmt_usd(cost).lstrip("$") if isinstance(cost, (int, float)) else DASH,
        ]
        if partial:
            cells.append(
                "full" if covered(row)
                else f"[yellow]{fmt_span(float(row.get('covered') or 0.0))}[/]"
            )
        table.add_row(*cells)
    parts: list[object] = [table]
    age = _published_age(published_at)
    if age is not None and age > LEDGER_STALE_SECONDS:
        parts.append(Text.from_markup(
            f"[yellow]ledger published {fmt_span(age)} ago[/]"
        ))
    if note:
        parts.append(Text.from_markup(note))
    return parts[0] if len(parts) == 1 else Group(*parts)



# ── the document, and the text ───────────────────────────────────────


def build_document(
    *,
    record_dir: str | Path | None = None,
    lane_path: str | Path | None = None,
    endpoints_path: str | Path | None = None,
    watch_path: str | Path | None = None,
    queue_dir: str | Path | None = None,
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
    published = _read_json(watch_path or watch_publish.default_output_path()) or {}
    document["ledger"] = published.get("ledger") or []
    document["ledger_published_at"] = published.get("published_at")
    queue = work_queue.read(queue_dir)
    if queue is not None:
        document["queue"] = queue
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
    engine: str | None = None,
    width: int | None = None,
) -> Constrain:
    """The whole view as one column of sections, ``width`` columns wide.

    In reading order: the lane and whether it can take work, who is using it,
    how the last half hour went, and what it has cost. With several engines
    the router line and the engines table come first, saying how much the
    document spans, and the lane below is one engine's: every engine's bands
    do not fit a terminal, the table carries every engine's figures, and
    ``--engine`` chooses whose bands are drawn.
    """
    width = page_width(width)
    callers = document.get("callers") or {}
    parts: list[object] = []
    multi = multi_engine(document)
    scoped = document
    if multi:
        parts += [router_line(document, width), Text("")]
        parts += section(
            "[bold]engines[/]",
            [engines_table(document, engine=engine, width=width)], width,
        )
        entry = selected_engine(document, engine)
        if entry is not None:
            scoped = engine_document(document, entry)
    parts += lane_section(scoped, width)

    window = callers.get("window_seconds")
    name = account or _viewer()
    parts += section(
        f"[bold]{_heading('accounts', window)}[/]",
        [accounts_table(callers, account=account, width=width)], width,
    )
    parts += section(
        f"[bold]{_heading('projects', window)}[/]  [dim]{escape(name)}[/]",
        [detail_table(callers, account=account, sessions=sessions,
                      engines=multi)], width,
    )
    parts += section(
        activity_title(callers),
        activity(callers, scoped.get("lane") or {}, width), width,
    )
    parts += section(
        "[bold]ledger[/]",
        [ledger_table(document.get("ledger"),
                      published_at=document.get("ledger_published_at"),
                      width=width)], width,
    )
    return page(parts[:-1], width)


def render_text(
    document: dict[str, Any],
    *,
    account: str | None = None,
    sessions: bool = False,
    engine: str | None = None,
    width: int = GOLDEN_WIDTH,
) -> str:
    """The view as plain text at *width*, so the output is comparable."""
    import io

    buffer = io.StringIO()
    Console(file=buffer, no_color=True, width=width, highlight=False).print(
        render(document, account=account, sessions=sessions, engine=engine,
               width=width),
        soft_wrap=False,
    )
    return buffer.getvalue()


def refresh_stamp(moment: _dt.datetime | None = None) -> str:
    """The line a live pass prints above its view: when that pass was taken.

    A refresh cadence can only be checked against what the view itself says, so
    the stamp travels with the output rather than being inferred from a
    reader's clock. *moment* is for a caller that already knows the time.
    """
    when = _dt.datetime.now(_dt.UTC) if moment is None else moment
    return f"# refreshed {when.astimezone(_dt.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')}"


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
    parser.add_argument(
        "--engine", dest="engine_id", default=None,
        help="the engine whose bands are shown, by its id or node:port; with "
             "several engines the busiest one is drawn unless this is given",
    )
    parser.add_argument("--account", dest="account", default=None,
                        help="the account whose detail table is shown")
    parser.add_argument("--sessions", action="store_true",
                        help="detail table lists coordinator sessions")
    parser.add_argument("--json", dest="as_json", action="store_true",
                        help="emit the document as JSON")
    parser.add_argument("--live", action="store_true",
                        help=f"refresh every {int(LIVE_REFRESH_SECONDS)}s; the "
                             "default in a terminal")
    parser.add_argument("--once", action="store_true",
                        help="print one reading and exit; the default when "
                             "the output is not a terminal")
    parser.add_argument("--width", type=int, default=None,
                        help="page width; the terminal's own by default")
    args = parser.parse_args(argv)
    import shutil

    width = page_width(args.width or shutil.get_terminal_size((0, 0)).columns)

    def read() -> dict[str, Any]:
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
        return document

    terminal = sys.stdout.isatty()
    live = live_by_default(args.live, args.once, args.as_json, terminal)

    def once() -> None:
        document = read()
        if args.as_json:
            sys.stdout.write(json.dumps(document, indent=2, default=str) + "\n")
        else:
            if live:
                sys.stdout.write(refresh_stamp() + "\n")
            sys.stdout.write(
                render_text(
                    document,
                    account=args.account,
                    sessions=args.sessions,
                    engine=args.engine_id,
                    width=width,
                )
            )
        sys.stdout.flush()

    if not live:
        once()
        return 0
    try:
        if terminal and not args.as_json:
            return _redraw_in_place(read, args)
        for _ in refresh_loop(once):
            pass
    except KeyboardInterrupt:
        return 130
    return 0


def live_by_default(live: bool, once: bool, as_json: bool, terminal: bool) -> bool:
    """Whether the view keeps refreshing.

    A reader at a terminal is watching, so the view refreshes unless ``--once``
    asks for a single reading. Output that goes to a pipe or a file is one
    reading unless ``--live`` asks for a stream, and so is JSON.
    """
    if once:
        return False
    return live or (terminal and not as_json)


def _redraw_in_place(read: Callable[[], dict[str, Any]], args: Any) -> int:
    """Redraw the view in place, in colour, until interrupted.

    The page takes the terminal's width on every pass, so resizing the window
    reflows it at the next refresh.
    """
    from rich.live import Live

    console = Console(highlight=False)

    def page() -> Group:
        width = page_width(args.width or console.size.width)
        return Group(
            Text(refresh_stamp(), style="dim"),
            render(read(), account=args.account, sessions=args.sessions,
                   engine=args.engine_id, width=width),
        )

    with Live(console=console, auto_refresh=False, screen=True) as view:
        for _ in refresh_loop(lambda: view.update(page(), refresh=True)):
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
