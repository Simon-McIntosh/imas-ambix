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

import datetime as _dt
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

#: The width of the engines table's first column, which carries a model and the
#: node it runs on: wider than a plain label, because it names two things.
ENGINE_LABEL_WIDTH = 24

#: The width of the detail table's engine column: a node name and a port.
ENGINE_CELL_WIDTH = 20

#: The width of an engine's name on its own history line. The strips are thirty
#: bins wide, so an engine's name is a column the total line does not spend: it
#: is sized to what a node name and a port need, no wider, so the line still
#: fits a hundred-column terminal.
ENGINE_STRIP_WIDTH = 16

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

    With several engines the terminal cannot hold every engine's three bands,
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
    from imas_ambix.agent.registry import _host_name

    return f"{_host_name(host)}:{port}"


def _count(total: int, noun: str) -> str:
    return f"{total} {noun}" if total == 1 else f"{total} {noun}s"


def router_line(document: dict[str, Any]) -> str:
    """One line naming the router, when the reading was taken, and its engines.

    The engine tables below are all of one document, so the line states how
    much that document spans -- servers, engines and models -- before a reader
    starts attributing rows to one of them.
    """
    engines = document_engines(document)
    models = {str(entry["model"]) for entry in engines if entry.get("model")}
    servers = {
        name for entry in engines if (name := _server_name(entry)) is not None
    }
    bits = ["[bold]router[/]"]
    address = document.get("router")
    if isinstance(address, str) and address:
        bits.append(f"[cyan]{address}[/]")
    stamp = (document.get("lane") or {}).get("observed_at")
    if isinstance(stamp, str) and len(stamp) >= 19:
        bits.append(stamp[11:19])
    bits.append(_count(len(servers), "server"))
    bits.append(_count(len(engines), "engine"))
    bits.append(_count(len(models), "model"))
    return " [dim]·[/] ".join(bits)


def _server_name(entry: dict[str, Any]) -> str | None:
    """The node an engine entry runs on, without its port or cluster prefix."""
    host = entry.get("host") or entry.get("engine")
    if not isinstance(host, str) or not host:
        return None
    text = host.split("://", 1)[-1].split("/", 1)[0].split(":", 1)[0]
    from imas_ambix.agent.registry import _host_name

    return _host_name(text)


def engines_table(
    document: dict[str, Any], *, engine: str | None = None
) -> Table:
    """One row per engine: its throughput and its gate's room, on one grid.

    The figures are the engines table's own: the rate and the time to first
    token are the recorder's, the router queue and the available percentage are
    the router's, and the KV pool is the engine's own counter. A figure a
    source did not publish is a dash, never a zero.
    """
    table = _table(
        ("engines", "running", "router q", "tok/s out", "tok/s each", "ttft s",
         "kv pool %", "available %"),
        label_width=ENGINE_LABEL_WIDTH,
    )
    chosen = selected_engine(document, engine)
    for entry in document_engines(document):
        name = " ".join(
            part for part in (
                str(entry.get("model") or ""),
                _server_name(entry) or "",
            ) if part
        ) or DASH
        queue = _number(entry, "router_q")
        kv_pool = _number(entry, "kv_pool")
        available = _number(entry, "available")
        table.add_row(
            name,
            f"[cyan]{_number(entry, 'running'):.1f}[/]"
            if _number(entry, "running") is not None else DASH,
            f"[red]{queue:.0f}[/]" if queue is not None and queue >= 1
            else f"[dim]{queue:.0f}[/]" if queue is not None else DASH,
            fmt_rate(_number(entry, "tok_s")),
            f"{_number(entry, 'per_stream'):.1f}"
            if _number(entry, "per_stream") is not None else DASH,
            f"{_number(entry, 'ttft'):.1f}"
            if _number(entry, "ttft") is not None else DASH,
            pct(100.0 * kv_pool) if kv_pool is not None else DASH,
            pct(available, "good") if available is not None else DASH,
            style="bold" if entry is chosen else "",
        )
    return table


# ── the two levels of the running-workers view ───────────────────────


def _heading(label: str, window_seconds: object) -> str:
    """A table's first column heading, carrying the window it measured."""
    if isinstance(window_seconds, (int, float)) and not isinstance(
        window_seconds, bool
    ):
        return f"{label} {window_seconds / 60:.0f}m"
    return label


def _table(
    heads: Iterable[str],
    *,
    label_width: int = LABEL_WIDTH,
    cell_width: int = CELL_WIDTH,
) -> Table:
    table = Table(box=box.SIMPLE_HEAD, padding=(0, 1), expand=False,
                  pad_edge=False, header_style="dim")
    for index, name in enumerate(heads):
        table.add_column(
            name,
            justify="left" if index == 0 else "right",
            width=label_width if index == 0 else cell_width,
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
    head = [_heading(name, callers.get("window_seconds"))]
    if engines:
        head.append("engine")
    head.extend(("session" if sessions else "sessions",
                 "runs", "in use", "per run", "wait s"))
    table = _table(
        head,
        label_width=LABEL_WIDTH,
        cell_width=ENGINE_CELL_WIDTH if engines else CELL_WIDTH,
    )
    blank = DASH if engines else ""
    for row in rows[:DETAIL_ROWS]:
        wait = row.get("mean_gate_wait_s")
        table.add_row(
            row["label"],
            *((_engine_display(row.get("engine")),) if engines else ()),
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
        table.add_row(f"+{len(folded)} more",
                      *((blank,) if engines else ()),
                      "", "", f"{held:.1f}", "", "", style="dim")
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
    lines = [
        f"{label}   in flight [bold]{spark(occupancy, width)}[/]"
        f"   wait [cyan]{spark(waits)}[/]"
    ]
    # One line per engine, on the same two strips, so a reader of a lane of
    # several engines can see which one is filling rather than only that the
    # lane is. The engine's own bins are the same quantity as the total's, so
    # two draws are comparable; the label column is matched to the total's.
    for entry in history.get("engines") or []:
        if not isinstance(entry, dict):
            continue
        engine_bins = [
            bin_ for bin_ in (entry.get("bins") or []) if isinstance(bin_, dict)
        ]
        if not engine_bins:
            continue
        name = _engine_display(entry.get("engine"))
        engine_occupancy = [float(b.get("occupancy") or 0.0) for b in engine_bins]
        engine_waits = [float(b.get("gate_wait_p50_s") or 0.0) for b in engine_bins]
        lines.append(
            f"{' ' * len(label)}   {name:<{ENGINE_STRIP_WIDTH}}"
            f" [bold]{spark(engine_occupancy, width)}[/]"
            f"   wait [cyan]{spark(engine_waits)}[/]"
        )
    if len(lines) == 1:
        return Text.from_markup(lines[0])
    return Group(*(Text.from_markup(line) for line in lines))


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

    def _line(completed: int, aborted: int, failed: int, expired: int) -> str:
        failed_s = f"[red]{failed:.0f} failed[/]" if failed else "0 failed"
        return (
            f"{completed:.0f} completed · {aborted:.0f} aborted · "
            f"{failed_s} · {expired:.0f} expired at the end"
        )

    lines = [
        f"{label}   " + _line(
            outcomes.get("completed") or 0,
            outcomes.get("aborted") or 0,
            outcomes.get("failed") or 0,
            outcomes.get("gate_expired") or 0,
        )
    ]
    # One line per engine under the total, for the same reason the history
    # carries one per engine: a lane in which one engine failed every request it
    # accepted and another completed them all reads as a healthy lane in the
    # total alone.
    for entry in outcomes.get("by_engine") or []:
        name = _engine_display(entry.get("engine"))
        lines.append(
            f"{' ' * len(label)}   {name:<{ENGINE_STRIP_WIDTH}} "
            + _line(
                entry.get("completed") or 0,
                entry.get("aborted") or 0,
                entry.get("failed") or 0,
                entry.get("gate_expired") or 0,
            )
        )
    if len(lines) == 1:
        return Text.from_markup(lines[0])
    return Group(*(Text.from_markup(line) for line in lines))


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
    # A ledger of several models is one table with a model column rather than
    # several tables: the rows are the same periods, and stacking one model's
    # periods under another's would put two rows labelled "3d" in one table with
    # nothing between them saying which model priced either.
    models = {str(row.get("model")) for row in rows if row.get("model")}
    grouped = len(models) > 1
    head = ("ledger", "model", "tok in", "cached %", "tok out", "util %",
            "cost $", "window") if grouped else (
        "ledger", "tok in", "cached %", "tok out", "util %", "cost $", "window")
    table = _table(head)
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
            *((str(row.get("model") or DASH),) if grouped else ()),
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
    published = _read_json(watch_path or watch_publish.default_output_path()) or {}
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
    engine: str | None = None,
) -> Group:
    """The whole view as one group of rich renderables.

    With one engine this is the view as it has always been drawn. With several,
    two things are added above it and one inside it: the router line and the
    engines table, which say how much the document spans, and the engine column
    in the account detail. The bands stay one engine's, because three bands per
    engine do not fit a terminal -- the table above them carries every engine's
    figures, and ``--engine`` chooses whose bands are drawn.
    """
    callers = document.get("callers") or {}
    parts: list[object] = []
    multi = multi_engine(document)
    scoped = document
    if multi:
        parts.append(Text.from_markup(router_line(document)))
        parts.append(engines_table(document, engine=engine))
        entry = selected_engine(document, engine)
        if entry is not None:
            scoped = engine_document(document, entry)
    parts.append(Text.from_markup(head_line(scoped)))
    parts.extend(bands(scoped))
    parts.append(accounts_table(callers, account=account))
    parts.append(detail_table(callers, account=account, sessions=sessions,
                              engines=multi))
    parts.append(history_strip(callers, scoped.get("lane") or {}))
    parts.append(outcomes_line(callers))
    parts.append(ledger_table(document.get("ledger")))
    return Group(*parts)


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
        render(document, account=account, sessions=sessions, engine=engine),
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
            if args.live:
                sys.stdout.write(refresh_stamp() + "\n")
            sys.stdout.write(
                render_text(
                    document,
                    account=args.account,
                    sessions=args.sessions,
                    engine=args.engine_id,
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
