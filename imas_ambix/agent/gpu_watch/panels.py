"""The pages, drawn on one ruled grid.

Every panel is a borderless table whose columns add up to :data:`WIDTH`, with
its last column right-aligned, and panels are separated by a rule of that same
width. That is the whole layout rule: nothing is sized to its own content, so
every panel ends on the same column whatever it holds.

Two pages are built from the panels. The overview is one row per server and
one row per lane, each with a total, then the period ledger. A server page is
that server's cards with their own temperature and power and a totals row, the
lanes served from it, and its queue, which takes every line the screen has
left.

A figure nothing measured is a dash, never a zero.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from rich.console import Console, Group
from rich.markup import escape
from rich.table import Table
from rich.text import Text

from imas_ambix.agent.watch_style import (
    DASH,
    fmt_ctx,
    fmt_span,
    fmt_tokens,
    fmt_usd,
    pct,
    ramp,
)

if TYPE_CHECKING:
    from imas_ambix.agent.gpu_watch.fleet import Job, Reading
    from imas_ambix.agent.gpu_watch.lane import Lane

#: Columns every panel is drawn to.
WIDTH = 80

#: Queue rows a server page keeps whenever there are that many jobs to show.
MIN_QUEUE_ROWS = 10

#: Per-stream decode reference, tok/s; the per-stream cell is coloured against
#: it so the figure says whether each stream gets a fair share of one stream's
#: best rate. The value is the one ``llm-watch`` colours against.
PER_STREAM_REF = 40.0

#: Die temperature at which the cards start to throttle, °C.
THROTTLE_C = 84.0

#: Seconds after which the published ledger is labelled with its age: twice
#: the publisher's cadence.
LEDGER_STALE_SECONDS = 600.0

MACHINE_WIDTHS = (14, 11, 11, 11, 11, 11, 11)
BAND_WIDTHS = (10,) * 8
SERVER_WIDTHS = (14, 10, 8, 11, 11, 9, 8, 9)
LANES_WIDTHS = (20, 10, 9, 8, 10, 6, 9, 8)
QUEUE_WIDTHS = (9, 10, 6, 29, 9, 12, 5)


# ── the grid ─────────────────────────────────────────────────────────


def grid(widths: tuple[int, ...], left: int = 1) -> Table:
    """A borderless table of fixed columns, the first ``left`` left-aligned."""
    table = Table(box=None, padding=0, pad_edge=False, show_header=False, expand=False)
    for index, width in enumerate(widths):
        table.add_column(
            "",
            width=width,
            no_wrap=True,
            overflow="ellipsis",
            justify="left" if index < left else "right",
        )
    return table


def rule() -> Text:
    return Text("─" * WIDTH, style="dim")


def head(table: Table, name: str, labels: list[str]) -> None:
    """The panel's name over its first column, and a label over each figure."""
    table.add_row(f"[bold]{name}[/]", *[f"[dim]{label}[/]" for label in labels])


def split_line(left: str, right: str, right_width: int) -> Table:
    """One line with text at each edge, so it too ends on the last column."""
    table = grid((WIDTH - right_width, right_width))
    table.add_row(left, right)
    return table


def bar(value: float, kind: str) -> str:
    """A four-cell bar and its percentage, nine characters wide.

    ``util`` is green when busy and red when idle; ``mem`` is an occupancy,
    coloured by how full it is.
    """
    value = max(0.0, min(100.0, value))
    fill = round(value / 100 * 4)
    if kind == "util":
        colour = "green" if value >= 70 else "yellow" if value >= 30 else "red"
    else:
        colour = "red" if value >= 90 else "magenta" if value >= 70 else "cyan"
    return f"[{colour}]{'█' * fill}{'░' * (4 - fill)}[/]{value:4.0f}%"


def ranges(indices: list[int]) -> str:
    """Card indices as ``0-1 3``, so a row naming many cards stays short."""
    out: list[str] = []
    run: list[int] = []
    for index in [*sorted(indices), None]:
        if run and index == run[-1] + 1:
            run.append(index)
            continue
        if run:
            out.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else f"{run[0]}")
        run = [index] if index is not None else []
    return " ".join(out)


def _number(source: object, key: str) -> float | None:
    value = source.get(key) if isinstance(source, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _count(value: float | None, alarm: bool = False) -> str:
    """A count that is ordinary at zero and worth seeing above it."""
    if value is None:
        return DASH
    if alarm and value >= 1:
        return f"[red]{value:.0f}[/]"
    return f"[dim]{value:.0f}[/]" if alarm else f"{value:.0f}"


# ── machine ──────────────────────────────────────────────────────────


def _power(cards: list) -> tuple[float | None, float | None]:
    """Total draw in watts and its share of the total cap, over reporting cards."""
    draws = [card.watts for card in cards if card.watts is not None]
    caps = [
        card.cap_watts for card in cards if card.watts is not None and card.cap_watts
    ]
    if not draws:
        return None, None
    total = sum(draws)
    return total, (100.0 * total / sum(caps) if len(caps) == len(draws) else None)


def _temp(value: float | None, suffix: str = "") -> str:
    if value is None:
        return DASH
    return f"[{ramp(value, 75.0, THROTTLE_C, inverted=True)}]{value:.0f}[/]{suffix}"


def server_title(reading: Reading) -> list[object]:
    """The server in one line: what it is, its state, and the cards held."""
    server = reading.server
    word = (reading.state or "unknown").split()[0]
    low = word.lower()
    down = "drain" in low or "down" in low
    if down:
        state = f"[bold red]{word.upper()}[/]"
    elif "alloc" in low or "mix" in low:
        state = f"[green]{word}[/]"
    else:
        state = f"[dim]{word}[/]"
    bits = [f"[bold]{server.name}[/]", server.hardware, state]
    if reading.group:
        bits.append(f"Group {reading.group}")
    if server.cards:
        bits.append(f"{reading.allocated_count}/{server.cards} alloc")
    elif reading.allocated_count:
        bits.append(f"{reading.allocated_count} alloc")
    out: list[object] = [
        split_line(" · ".join(bits), f"[dim]{time.strftime('%H:%M:%S')}[/]", 8)
    ]
    if down:
        # A drained or down node is the usual reason there are no card rows.
        detail = f" — {escape(reading.reason)}" if reading.reason else ""
        out.append(
            Text.from_markup(
                f"[red]no job can start here{detail}; an administrator must "
                f"resume the node[/]"
            )
        )
    return out


def machine_panel(reading: Reading, window: int) -> list[object]:
    """One row per readable card, one for the unreadable ones, then totals.

    Temperature and power are each card's own. The totals row is the mean
    utilisation and memory, the hottest card and the summed draw, over the
    cards that were read: a card another job holds reports nothing here, so it
    is named on its own row and left out of every total.
    """
    table = grid(MACHINE_WIDTHS)
    head(
        table,
        "MACHINE",
        ["util now", f"util {window}s", "mem", "°C", "kW", "% cap"],
    )
    cards = reading.cards
    for card in cards:
        share = (
            100.0 * card.watts / card.cap_watts
            if card.watts is not None and card.cap_watts
            else None
        )
        table.add_row(
            f"gpu {card.index}",
            bar(card.util, "util"),
            bar(card.util_mean, "util"),
            bar(card.mem, "mem"),
            _temp(card.temp),
            f"{card.watts / 1000.0:.2f}" if card.watts is not None else DASH,
            f"{share:.0f}" if share is not None else DASH,
        )
    hidden = reading.unreadable
    if hidden:
        table.add_row(
            f"gpu {ranges(hidden)}",
            "[yellow]unsampled[/]",
            "",
            "[dim]allocated[/]",
            DASH,
            DASH,
            DASH,
        )
    if len(cards) > 1:
        draw, share = _power(cards)
        temps = [card.temp for card in cards if card.temp is not None]
        count = len(cards)
        table.add_row(
            f"[bold]{count} cards[/]",
            f"[bold]{sum(c.util for c in cards) / count:.0f}%[/]",
            f"[bold]{sum(c.util_mean for c in cards) / count:.0f}%[/]",
            f"[bold]{sum(c.mem for c in cards) / count:.0f}%[/]",
            _temp(max(temps), " max") if temps else DASH,
            f"[bold]{draw / 1000.0:.2f}[/]" if draw is not None else DASH,
            f"[bold]{share:.0f}[/]" if share is not None else DASH,
        )
    if reading.host is not None:
        cpu, cpu_mean, ram = reading.host
        table.add_row(
            "[dim]host[/]",
            f"[dim]cpu[/] {cpu:4.0f}%",
            f"[dim]cpu[/] {cpu_mean:4.0f}%",
            f"[dim]ram[/] {ram:4.0f}%",
            "",
            "",
            "",
        )
    out: list[object] = [table]
    if not cards and not hidden:
        out.append(
            Text.from_markup(
                "[dim]no card readings — cards are read through a running "
                "job of this account[/]"
            )
        )
    return out


# ── lane ─────────────────────────────────────────────────────────────


def _stale(lane: dict) -> str:
    """A mark for a lane document past the shelf life the router gave it."""
    age = _number(lane, "age_seconds")
    shelf = _number(lane, "suggested_shelf_life_seconds")
    if lane.get("stale") is True or (age is not None and shelf and age > shelf):
        when = f"{age:,.0f}s" if age is not None else "age unknown"
        return f" · [bold yellow]stale {when}[/]"
    return ""


def lane_panel(lane: Lane, cards: str = "") -> list[object]:
    """One lane: a line saying what is served, then two bands of readings.

    The first band is the engine's own rates and the pool it holds; the second
    is the router's gate and the room left to dispatch into. A request held at
    the gate never reaches the engine, so the engine's ``queued`` reads zero
    while ``rt queue`` holds work, and both are shown.
    """
    bits = [f"[bold]LANE[/]  [bold]{escape(lane.model)}[/]"]
    if cards:
        bits.append(cards)
    elif lane.hardware:
        bits.append(lane.hardware)
    if lane.context is not None:
        bits.append(f"{fmt_ctx(lane.context)} ctx")
    left = " · ".join(bits) + _stale(lane.lane)
    right = (
        f"[cyan]{escape(lane.checkpoint)}[/]"
        if lane.checkpoint
        else "[yellow]checkpoint unrecorded[/]"
    )
    live, doc = lane.live, lane.lane
    measured = live.get("state") == "measured"

    def rate(key: str) -> str:
        value = _number(live, key)
        return fmt_tokens(value) if measured and value is not None else DASH

    running = _number(live, "mean_running")
    each = _number(live, "per_stream_toks_per_s")
    ttft = _number(live, "median_time_to_first_token_s")
    hit = _number(live, "cache_hit_rate")
    pool = _number(live, "pool_occupancy")
    table = grid(BAND_WIDTHS, left=0)
    table.add_row(
        *[
            f"[dim]{label}[/]"
            for label in (
                "running",
                "queued",
                "tok/s",
                "each",
                "prefill",
                "ttft s",
                "cache %",
                "kv pool %",
            )
        ]
    )
    table.add_row(
        f"[cyan]{running:.1f}[/]" if running is not None else DASH,
        _count(_number(live, "engine_queue"), alarm=True),
        rate("generation_toks_per_s"),
        (
            f"[{ramp(100.0 * each / PER_STREAM_REF, 60.0, 30.0)}]{each:.1f}[/]"
            if measured and each
            else DASH
        ),
        rate("prefill_toks_per_s"),
        f"{ttft:.1f}" if measured and ttft is not None else DASH,
        pct(100.0 * hit, "good", width=4) if measured and hit is not None else DASH,
        pct(100.0 * pool if pool <= 1.0 else pool, width=4)
        if pool is not None
        else DASH,
        style="bold",
    )
    gate = doc.get("router_generation_gate") or {}
    admission = doc.get("admission") or {}
    slots = doc.get("worker_slots") or {}
    in_flight, width = _number(gate, "in_flight"), _number(gate, "effective_width")
    oldest = _number(admission, "oldest_wait_seconds")
    verdict = admission.get("verdict")
    headroom = _number(doc, "headroom")
    availability = _number(doc, "availability_percent")
    table.add_row(
        *[
            f"[dim]{label}[/]"
            for label in (
                "in flight",
                "rt queue",
                "wait",
                "verdict",
                "headroom",
                "runs",
                "slots",
                "avail %",
            )
        ]
    )
    table.add_row(
        (f"{in_flight:.0f}" if in_flight is not None else DASH)
        + "/"
        + (f"{width:.0f}" if width is not None else DASH),
        _count(_number(gate, "waiting"), alarm=True),
        fmt_span(oldest) if oldest is not None else DASH,
        f"[bold]{escape(str(verdict))}[/]" if verdict else DASH,
        f"[bold]{headroom:.0f}[/]" if headroom is not None else DASH,
        _count(_number(slots, "live_runs")),
        _count(_number(slots, "worker_slots")),
        pct(availability, "good", width=4) if availability is not None else DASH,
    )
    return [split_line(left, right, 34), table]


# ── ledger ───────────────────────────────────────────────────────────


def ledger_panel(ledger: list | None, published_at: object = None) -> list[object]:
    """Tokens and their cost, one row per published period.

    A coverage column is drawn only when some period is backed by less than
    its whole record: a day's row backed by six hours answers a different
    question, and a column of ``full`` says nothing.
    """
    rows = [row for row in (ledger or []) if isinstance(row, dict) and row.get("label")]
    if not rows:
        return [
            Text.from_markup(
                "[bold]LEDGER[/]  [yellow]unavailable[/][dim] — the published "
                "watch document carries no period rows[/]"
            )
        ]

    def covered(row: dict) -> bool:
        period = float(row.get("period") or 0.0)
        return bool(period) and float(row.get("covered") or 0.0) >= 0.98 * period

    partial = not all(covered(row) for row in rows)
    labels = ["tok in", "cached %", "tok out", "util %", "cost $"]
    widths = (15, 13, 13, 13, 13, 13)
    if partial:
        labels.append("covered")
        widths = (14, 11, 11, 11, 11, 11, 11)
    table = grid(widths)
    head(table, "LEDGER", labels)
    for row in rows:
        tokens_in = float(row.get("tokens_in") or 0.0)
        cached = float(row.get("cached") or 0.0)
        utilisation = _number(row, "utilisation")
        cost = _number(row, "cost")
        cells = [
            str(row["label"]),
            fmt_tokens(tokens_in),
            pct(100.0 * cached / tokens_in, "good", width=4) if tokens_in > 0 else DASH,
            fmt_tokens(float(row.get("tokens_out") or 0.0)),
            pct(utilisation, width=4) if utilisation is not None else DASH,
            fmt_usd(cost).lstrip("$") if cost is not None else DASH,
        ]
        if partial:
            cells.append(
                "full" if covered(row) else fmt_span(float(row.get("covered") or 0.0))
            )
        table.add_row(*cells)
    out: list[object] = [table]
    age = _published_age(published_at)
    if age is not None and age > LEDGER_STALE_SECONDS:
        out.append(Text.from_markup(f"[yellow]ledger published {fmt_span(age)} ago[/]"))
    return out


def _published_age(stamp: object) -> float | None:
    """Seconds since a ``YYYY-MM-DDTHH:MM:SSZ`` stamp, or None when unreadable."""
    import calendar

    if not isinstance(stamp, str):
        return None
    try:
        moment = time.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return None
    return max(0.0, time.time() - calendar.timegm(moment))


# ── queue ────────────────────────────────────────────────────────────


def ordered_jobs(jobs: list[Job]) -> list[Job]:
    """Running jobs first, then pending; within each, group A, B, the rest."""
    return sorted(
        jobs,
        key=lambda job: (1 if job.pending else 0, job.group or "Z"),
    )


def queue_panel(
    jobs: list[Job], rows: int | None = None, gpus: dict[str, int] | None = None
) -> list[object]:
    """The server's queue, any user: running jobs, then pending ones.

    This is where another group's load shows, since its cards cannot be read.
    ``rows`` bounds the table to the lines the screen has left; jobs that do
    not fit are counted on a last line, never cut silently. ``gpus`` is the
    cards each job holds or asked for, by job id.
    """
    table = grid(QUEUE_WIDTHS, left=5)
    head(table, "QUEUE", ["user", "acct", "name", "state", "elapsed", "gpu"])
    if not jobs:
        table.add_row("[dim]idle[/]", "", "", "", "", "", "")
        return [table]
    ordered = ordered_jobs(jobs)
    shown = ordered
    if rows is not None and len(ordered) > rows:
        shown = ordered[: max(1, rows - 1)]
    for job in shown:
        colour = (
            "dim"
            if job.pending
            else {"A": "green", "B": "cyan"}.get(job.group or "", "default")
        )
        cells = (
            job.id,
            job.user,
            job.account,
            job.name,
            job.state,
            job.elapsed,
            f"{held}" if (held := (gpus or {}).get(job.id, job.gpus)) else DASH,
        )
        table.add_row(*[f"[{colour}]{escape(cell)}[/]" for cell in cells])
    rest = ordered[len(shown) :]
    if rest:
        waiting = sum(1 for job in rest if job.pending)
        table.add_row(
            f"[dim]+{len(rest)} more[/]",
            "",
            "",
            "",
            f"[dim]{waiting} pend[/]",
            "",
            "",
        )
    return [table]


# ── overview ─────────────────────────────────────────────────────────


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _percent(value: float | None) -> str:
    return f"{value:.0f}%" if value is not None else DASH


def servers_panel(readings: list[Reading]) -> list[object]:
    """One row per server and a total: cards held, load, heat, power, queue.

    Utilisation, memory, temperature and power describe the cards that were
    read, which may be fewer than the server holds; ``queue`` is running jobs
    then pending ones.
    """
    table = grid(SERVER_WIDTHS)
    head(
        table,
        "SERVERS",
        ["cards", "alloc", "util", "mem", "°C max", "kW", "queue"],
    )
    every: list = []
    pending: set[str] = set()
    running = allocated = total_cards = 0
    for reading in readings:
        server, cards = reading.server, reading.cards
        every += cards
        draw, _ = _power(cards)
        temps = [card.temp for card in cards if card.temp is not None]
        util = _mean([card.util_mean for card in cards])
        mem = _mean([card.mem for card in cards])
        low = reading.state.lower()
        down = "drain" in low or "down" in low
        running += len(reading.running)
        allocated += reading.allocated_count
        total_cards += server.cards or 0
        pending |= {job.id for job in reading.pending}
        table.add_row(
            f"[bold red]{server.name}[/]" if down else f"[bold]{server.name}[/]",
            server.hardware,
            f"{reading.allocated_count}/{server.cards or DASH}",
            bar(util, "util") if util is not None else DASH,
            bar(mem, "mem") if mem is not None else DASH,
            _temp(max(temps)) if temps else DASH,
            f"{draw / 1000.0:.2f}" if draw is not None else DASH,
            f"{len(reading.running)}+{len(reading.pending)}",
        )
    if len(readings) > 1:
        draw, _ = _power(every)
        temps = [card.temp for card in every if card.temp is not None]
        table.add_row(
            "[bold]total[/]",
            f"[bold]{total_cards}[/]",
            f"[bold]{allocated}/{total_cards}[/]",
            f"[bold]{_percent(_mean([card.util_mean for card in every]))}[/]",
            f"[bold]{_percent(_mean([card.mem for card in every]))}[/]",
            _temp(max(temps)) if temps else DASH,
            f"[bold]{draw / 1000.0:.2f}[/]" if draw is not None else DASH,
            f"[bold]{running}+{len(pending)}[/]",
        )
    if not readings:
        return [table, Text.from_markup("[yellow]no server found[/]")]
    return [table]


def no_document(reason: str = "") -> Text:
    """The line that stands where the lanes would be, with why when known."""
    detail = f"[dim] — {escape(reason)}[/]" if reason else ""
    return Text.from_markup(f"[yellow]no lane document read[/]{detail}")


def lanes_panel(lanes: list[Lane], reason: str = "") -> list[object]:
    """One row per lane and a total: load, rate, and the gate's room."""
    table = grid(LANES_WIDTHS)
    head(
        table,
        "LANES",
        ["server", "running", "tok/s", "in flight", "queue", "verdict", "avail %"],
    )
    totals = {"running": 0.0, "rate": 0.0, "in_flight": 0.0, "width": 0.0, "queue": 0.0}
    for lane in lanes:
        live, doc = lane.live, lane.lane
        measured = live.get("state") == "measured"
        gate = doc.get("router_generation_gate") or {}
        running = _number(live, "mean_running")
        rate = _number(live, "generation_toks_per_s") if measured else None
        in_flight, width = _number(gate, "in_flight"), _number(gate, "effective_width")
        queue = _number(gate, "waiting")
        verdict = (doc.get("admission") or {}).get("verdict")
        availability = _number(doc, "availability_percent")
        for key, value in (
            ("running", running),
            ("rate", rate),
            ("in_flight", in_flight),
            ("width", width),
            ("queue", queue),
        ):
            totals[key] += value or 0.0
        table.add_row(
            f"[bold]{escape(lane.model)}[/]",
            lane.server or DASH,
            f"[cyan]{running:.1f}[/]" if running is not None else DASH,
            fmt_tokens(rate) if rate is not None else DASH,
            (f"{in_flight:.0f}" if in_flight is not None else DASH)
            + "/"
            + (f"{width:.0f}" if width is not None else DASH),
            _count(queue, alarm=True),
            escape(str(verdict)) if verdict else DASH,
            pct(availability, "good", width=4) if availability is not None else DASH,
        )
    if len(lanes) > 1:
        table.add_row(
            "[bold]total[/]",
            "",
            f"[bold]{totals['running']:.1f}[/]",
            f"[bold]{fmt_tokens(totals['rate'])}[/]",
            f"[bold]{totals['in_flight']:.0f}/{totals['width']:.0f}[/]",
            _count(totals["queue"], alarm=True),
            "",
            "",
        )
    if not lanes:
        return [table, no_document(reason)]
    return [table]


# ── pages ────────────────────────────────────────────────────────────


def _join(sections: list[list[object]]) -> list[object]:
    """Sections one after another with a rule between each pair."""
    out: list[object] = []
    for section in sections:
        if out:
            out.append(rule())
        out += section
    return out


def line_count(parts: list[object], width: int) -> int:
    """Lines ``parts`` take when drawn ``width`` columns wide."""
    import io

    probe = Console(
        width=width, file=io.StringIO(), force_terminal=False, color_system=None
    )
    return len(probe.render_lines(Group(*parts), probe.options, pad=False))


def overview_page(
    readings: list[Reading],
    lanes: list[Lane],
    ledger: list | None,
    published_at: object = None,
    footer: str = "",
    reason: str = "",
) -> Group:
    """Every server, every lane, and what the lanes have cost."""
    count = len(readings)
    title = split_line(
        f"[bold]overview[/] · {count} server{'s' if count != 1 else ''}"
        f" · {len(lanes)} lane{'s' if len(lanes) != 1 else ''}",
        f"[dim]{time.strftime('%H:%M:%S')}[/]",
        8,
    )
    parts = _join(
        [
            [title],
            servers_panel(readings),
            lanes_panel(lanes, reason),
            ledger_panel(ledger, published_at),
        ]
    )
    if footer:
        parts += [rule(), Text.from_markup(footer)]
    return Group(*parts)


def server_page(
    reading: Reading,
    lanes: list[Lane],
    window: int,
    *,
    ledger: list | None = None,
    published_at: object = None,
    size: tuple[int, int] | None = None,
    footer: str = "",
) -> Group:
    """One server: its cards, the lanes served from it, and its queue.

    ``ledger`` is passed when this is the only server, so the single-server
    screen carries the ledger the overview would otherwise hold. ``size`` is
    the screen as ``(columns, lines)``; with it the queue takes every line the
    panels above leave, and when that is fewer than :data:`MIN_QUEUE_ROWS` the
    ledger gives up its lines, since the overview still holds it.
    """
    sections = [server_title(reading), machine_panel(reading, window)]
    for lane in lanes:
        held = reading.job_cards.get(lane.job_id or "")
        sections.append(lane_panel(lane, f"gpu {ranges(sorted(held))}" if held else ""))
    if ledger is not None:
        sections.append(ledger_panel(ledger, published_at))
    tail: list[object] = [rule(), Text.from_markup(footer)] if footer else []

    def fit(sections: list[list[object]]) -> tuple[list[object], int]:
        parts = _join(sections)
        # The rule above the queue and its header are the queue's own two lines.
        return parts, size[1] - line_count(parts + tail, size[0]) - 2

    parts, rows = _join(sections), None
    if size is not None:
        wanted = min(MIN_QUEUE_ROWS, len(reading.jobs))
        parts, rows = fit(sections)
        if rows < wanted and ledger is not None:
            parts, rows = fit(sections[:-1])
        rows = max(1, rows)
    gpus = {job.id: reading.gpus(job) for job in reading.jobs}
    return Group(*parts, rule(), *queue_panel(reading.jobs, rows, gpus), *tail)
