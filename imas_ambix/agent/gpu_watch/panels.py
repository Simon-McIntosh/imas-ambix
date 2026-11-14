"""The pages, built from titled boxes as wide as the terminal.

Every panel is a box from :mod:`imas_ambix.agent.watch_style`: a title on its
top edge, naming columns on the left and figure columns sharing the rest of
its width. A lane's box is ``llm-watch``'s own, so a lane on these pages and
the same lane there are one set of figures drawn one way.

Three pages are built from the boxes. The overview is one row per server and
one row per lane, each with a total, then the period ledger. A server page is
that server's cards with their own temperature and power and a totals row,
its queue, and the lanes served from it; on a terminal wide enough the cards
and the queue stand beside the lanes and the ledger. The lanes page is every
lane in full.

A figure nothing measured is a dash, never a zero.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from rich.console import Group
from rich.markup import escape
from rich.table import Table
from rich.text import Text

from imas_ambix.agent import llm_watch
from imas_ambix.agent.watch_style import (
    DASH,
    DEFAULT_WIDTH,
    TWO_COLUMN_WIDTH,
    boxed,
    column_widths,
    flex_table,
    fmt_ctx,
    fmt_tokens,
    head,
    line_count,
    ramp,
    side_by_side,
    split_line,
)

if TYPE_CHECKING:
    from rich.panel import Panel

    from imas_ambix.agent.gpu_watch.fleet import Job, Reading
    from imas_ambix.agent.gpu_watch.lane import Lane

#: Queue rows a server page keeps whenever there are that many jobs to show.
MIN_QUEUE_ROWS = 10

#: Die temperature at which the cards start to throttle, °C.
THROTTLE_C = 84.0

#: Lines a queue box spends on its two edges and its row of labels.
QUEUE_FRAME_LINES = 3


def bar(value: float, kind: str) -> str:
    """A four-cell bar and its percentage.

    ``util`` is green when busy and red when idle, because an idle card is
    the waste; ``mem`` is an occupancy, coloured by how full it is.
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


def _clock() -> str:
    return f"[dim]{time.strftime('%H:%M:%S')}[/]"


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


def _down(reading: Reading) -> bool:
    low = (reading.state or "").lower()
    return "drain" in low or "down" in low


def server_title(reading: Reading, width: int) -> list[object]:
    """The server in one line: what it is, its state, and the cards held."""
    server = reading.server
    word = (reading.state or "unknown").split()[0]
    low = word.lower()
    if _down(reading):
        state = f"[bold red]{word.upper()}[/]"
    elif "alloc" in low or "mix" in low:
        state = f"[green]{word}[/]"
    else:
        state = f"[dim]{word}[/]"
    bits = [f"[bold]{server.name}[/]", server.hardware, state]
    if reading.group:
        bits.append(f"Group {reading.group}")
    if server.cards:
        bits.append(f"{reading.allocated_count}/{server.cards} cards allocated")
    elif reading.allocated_count:
        bits.append(f"{reading.allocated_count} cards allocated")
    out: list[object] = [split_line(" [dim]·[/] ".join(bits), _clock(), width)]
    if _down(reading):
        # A drained or down node is the usual reason there are no card rows.
        detail = f" — {escape(reading.reason)}" if reading.reason else ""
        out.append(
            Text.from_markup(
                f"[red]no job can start here{detail}; an administrator must "
                f"resume the node[/]"
            )
        )
    return out


def machine_box(reading: Reading, window: int, width: int) -> Panel:
    """One row per readable card, one for the unreadable ones, then totals.

    Temperature and power are each card's own. The totals row is the mean
    utilisation and memory, the hottest card and the summed draw, over the
    cards that were read: a card another job holds reports nothing here, so it
    is named on its own row and left out of every total.
    """
    table = flex_table((10,), 6)
    head(table, ["card", "util now", f"util {window}s", "mem", "°C", "kW", "% cap"])
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
    body: list[object] = [table]
    if not cards and not hidden:
        body.append(
            Text.from_markup(
                "[dim]no card readings — cards are read through a running "
                "job of this account[/]"
            )
        )
    return boxed("[bold]MACHINE[/]", body, width)


# ── lane and ledger ──────────────────────────────────────────────────


def lane_box(lane: Lane, width: int, where: str = "", capacity: bool = True) -> Panel:
    """One lane as ``llm-watch`` draws it, titled with where it is served.

    ``where`` is the server or the cards the lane runs on, when the page
    knows them; without it the lane's own hardware stands there.
    """
    facts = [where or lane.hardware or ""]
    if lane.context is not None:
        facts.append(f"{fmt_ctx(lane.context)} ctx")
    return llm_watch.lane_box(
        {"lane": lane.lane, "live": lane.live},
        width,
        capacity=capacity,
        title=llm_watch.lane_title(lane.model, [fact for fact in facts if fact]),
        note=llm_watch.checkpoint_note(lane.checkpoint),
    )


def ledger_box(ledger: list | None, published_at: object, width: int) -> Panel:
    """Tokens and their cost per period, as ``llm-watch`` draws them."""
    return boxed(
        "[bold]LEDGER[/]",
        [llm_watch.ledger_table(ledger, published_at=published_at)],
        width,
    )


# ── queue ────────────────────────────────────────────────────────────


def ordered_jobs(jobs: list[Job]) -> list[Job]:
    """Running jobs first, then pending; within each, group A, B, the rest."""
    return sorted(
        jobs,
        key=lambda job: (1 if job.pending else 0, job.group or "Z"),
    )


def queue_box(
    jobs: list[Job],
    width: int,
    rows: int | None = None,
    gpus: dict[str, int] | None = None,
) -> Panel:
    """The server's queue, any user: running jobs, then pending ones.

    This is where another group's load shows, since its cards cannot be read.
    ``rows`` bounds the table to the lines the screen has left; jobs that do
    not fit are counted on a last line, never cut silently. ``gpus`` is the
    cards each job holds or asked for, by job id. The title counts the jobs,
    so the totals are read without counting rows.
    """
    table = Table(
        box=None, padding=(0, 0, 0, 1), pad_edge=False, show_header=False, expand=True
    )
    for name_width in (8, 9, 5):
        table.add_column("", width=name_width, no_wrap=True, overflow="ellipsis")
    table.add_column("", ratio=1, no_wrap=True, overflow="ellipsis")
    table.add_column("", width=8, no_wrap=True)
    table.add_column("", width=11, justify="right", no_wrap=True)
    table.add_column("", width=5, justify="right", no_wrap=True)
    head(table, ["job", "user", "acct", "name", "state", "elapsed", "cards"])
    running = sum(1 for job in jobs if job.running)
    waiting = sum(1 for job in jobs if job.pending)
    title = f"[bold]QUEUE[/]  [dim]{running} running · {waiting} pending[/]"
    if not jobs:
        table.add_row("[dim]idle[/]", "", "", "", "", "", "")
        return boxed(title, [table], width)
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
        held = (gpus or {}).get(job.id, job.gpus)
        cells = (
            job.id,
            job.user,
            job.account,
            job.name,
            job.state.lower(),
            job.elapsed,
            f"{held}" if held else DASH,
        )
        table.add_row(*[f"[{colour}]{escape(cell)}[/]" for cell in cells])
    rest = ordered[len(shown) :]
    if rest:
        pending = sum(1 for job in rest if job.pending)
        table.add_row(
            f"[dim]+{len(rest)} more[/]",
            "",
            "",
            f"[dim]{pending} of them pending[/]",
            "",
            "",
            "",
        )
    return boxed(title, [table], width)


# ── overview ─────────────────────────────────────────────────────────


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _percent(value: float | None) -> str:
    return f"{value:.0f}%" if value is not None else DASH


def servers_box(readings: list[Reading], width: int) -> Panel:
    """One row per server and a total: cards held, load, heat, power, queue.

    Utilisation, memory, temperature and power describe the cards that were
    read, which may be fewer than the server holds; ``jobs`` is running jobs
    then pending ones.
    """
    table = flex_table((9, 6), 6)
    head(
        table,
        ["server", "cards", "alloc", "util", "mem", "°C max", "kW", "jobs"],
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
        running += len(reading.running)
        allocated += reading.allocated_count
        total_cards += server.cards or 0
        pending |= {job.id for job in reading.pending}
        name = f"[bold red]{server.name}[/]" if _down(reading) else server.name
        table.add_row(
            f"[bold]{name}[/]",
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
    body: list[object] = [table]
    if not readings:
        body.append(Text.from_markup("[yellow]no server found[/]"))
    return boxed("[bold]SERVERS[/]", body, width)


def no_document(reason: str = "") -> Text:
    """The line that stands where the lanes would be, with why when known."""
    detail = f"[dim] — {escape(reason)}[/]" if reason else ""
    return Text.from_markup(f"[yellow]no lane document read[/]{detail}")


def lanes_box(lanes: list[Lane], width: int, reason: str = "") -> Panel:
    """One row per lane and a total: load, the gate, the rate and the verdict.

    The verdict has a column of its own width, since it is a word and the
    figures beside it are not; how many more workers a lane admits is in the
    lane's own box.
    """
    table = flex_table((19, 8), 4)
    table.add_column("", width=9, justify="right", no_wrap=True)
    head(
        table,
        ["lane", "server", "running", "in flight", "tok/s", "waiting", "verdict"],
    )
    totals = {key: 0.0 for key in ("running", "rate", "in_flight", "width", "queue")}
    for lane in lanes:
        live, doc = lane.live, lane.lane
        measured = live.get("state") == "measured"
        gate = doc.get("router_generation_gate") or {}
        running = _number(live, "mean_running")
        rate = _number(live, "generation_toks_per_s") if measured else None
        in_flight, gate_width = (
            _number(gate, "in_flight"),
            _number(gate, "effective_width"),
        )
        queue = _number(gate, "waiting")
        for key, value in (
            ("running", running),
            ("rate", rate),
            ("in_flight", in_flight),
            ("width", gate_width),
            ("queue", queue),
        ):
            totals[key] += value or 0.0
        table.add_row(
            f"[bold]{escape(lane.model)}[/]",
            lane.server or DASH,
            f"[cyan]{running:.1f}[/]" if running is not None else DASH,
            (f"{in_flight:.0f}" if in_flight is not None else DASH)
            + "/"
            + (f"{gate_width:.0f}" if gate_width is not None else DASH),
            fmt_tokens(rate) if rate is not None else DASH,
            llm_watch.alarm_count(queue),
            llm_watch.verdict_cell((doc.get("admission") or {}).get("verdict")),
        )
    if len(lanes) > 1:
        table.add_row(
            "[bold]total[/]",
            "",
            f"[bold]{totals['running']:.1f}[/]",
            f"[bold]{totals['in_flight']:.0f}/{totals['width']:.0f}[/]",
            f"[bold]{fmt_tokens(totals['rate'])}[/]",
            llm_watch.alarm_count(totals["queue"]),
            "",
        )
    body: list[object] = [table]
    if not lanes:
        body.append(no_document(reason))
    return boxed("[bold]LANES[/]", body, width)


# ── pages ────────────────────────────────────────────────────────────


def _foot(footer: str) -> list[object]:
    return [Text.from_markup(footer)] if footer else []


def overview_page(
    readings: list[Reading],
    lanes: list[Lane],
    ledger: list | None,
    published_at: object = None,
    footer: str = "",
    reason: str = "",
    width: int = DEFAULT_WIDTH,
) -> Group:
    """Every server, every lane, and what the lanes have cost."""
    count = len(readings)
    title = split_line(
        f"[bold]overview[/] [dim]·[/] {count} server{'s' if count != 1 else ''}"
        f" [dim]·[/] {len(lanes)} lane{'s' if len(lanes) != 1 else ''}",
        _clock(),
        width,
    )
    return Group(
        title,
        servers_box(readings, width),
        lanes_box(lanes, width, reason),
        ledger_box(ledger, published_at, width),
        *_foot(footer),
    )


def lanes_page(
    readings: list[Reading],
    lanes: list[Lane],
    ledger: list | None,
    published_at: object = None,
    footer: str = "",
    reason: str = "",
    width: int = DEFAULT_WIDTH,
) -> Group:
    """Every lane in full: the totals, then each lane's own box.

    Each lane is titled with the server and the cards it is served from, read
    from the server's job that holds it, so a lane and the cards under it can
    be matched across pages.
    """
    count = len(lanes)
    title = split_line(
        f"[bold]lanes[/] [dim]·[/] {count} lane{'s' if count != 1 else ''}",
        _clock(),
        width,
    )
    held = {
        job_id: (reading.server.name, cards)
        for reading in readings
        for job_id, cards in reading.job_cards.items()
    }
    parts: list[object] = [title, lanes_box(lanes, width, reason)]
    for lane in lanes:
        server, cards = held.get(lane.job_id or "", (lane.server, set()))
        where = " ".join(
            bit
            for bit in (server, f"gpu {ranges(sorted(cards))}" if cards else "")
            if bit
        )
        parts.append(lane_box(lane, width, where))
    parts.append(ledger_box(ledger, published_at, width))
    return Group(*parts, *_foot(footer))


def server_page(
    reading: Reading,
    lanes: list[Lane],
    window: int,
    *,
    ledger: list | None = None,
    published_at: object = None,
    size: tuple[int, int] | None = None,
    footer: str = "",
    width: int = DEFAULT_WIDTH,
) -> Group:
    """One server: its cards, its queue, and the lanes served from it.

    On a page wide enough the cards and the queue stand on the left and the
    lanes, with the ledger under them, on the right; narrower, the boxes are
    stacked with the queue last. ``ledger`` is passed when this is the only
    server, so the single-server screen carries the ledger the overview would
    otherwise hold. ``size`` is the screen as ``(columns, lines)``, with zero
    lines for a page that is printed and not fitted; fitted, the queue takes
    every line left to it, and when a stacked page leaves it
    fewer than :data:`MIN_QUEUE_ROWS` the ledger gives up its lines, since the
    overview still holds it.
    """
    title = server_title(reading, width)
    tail = _foot(footer)
    gpus = {job.id: reading.gpus(job) for job in reading.jobs}
    wide = width >= TWO_COLUMN_WIDTH and bool(lanes or ledger is not None)
    left_width, right_width = column_widths(width) if wide else (width, width)

    def lane_boxes(box_width: int) -> list[object]:
        boxes: list[object] = []
        for lane in lanes:
            held = reading.job_cards.get(lane.job_id or "")
            where = f"gpu {ranges(sorted(held))}" if held else ""
            boxes.append(lane_box(lane, box_width, where, capacity=wide))
        return boxes

    def rows_left(above: list[object]) -> int | None:
        if size is None or not size[1]:
            return None
        return size[1] - line_count(above + tail, width) - QUEUE_FRAME_LINES

    machine = machine_box(reading, window, left_width)
    if wide:
        right = lane_boxes(right_width)
        if ledger is not None:
            right.append(ledger_box(ledger, published_at, right_width))
        rows = rows_left([*title, machine])
        queue = queue_box(
            reading.jobs, left_width, None if rows is None else max(1, rows), gpus
        )
        return Group(*title, side_by_side([machine, queue], right, width), *tail)
    above = [*title, machine, *lane_boxes(width)]
    extra = [ledger_box(ledger, published_at, width)] if ledger is not None else []
    rows = rows_left(above + extra)
    if rows is not None and extra and rows < min(MIN_QUEUE_ROWS, len(reading.jobs)):
        extra, rows = [], rows_left(above)
    queue = queue_box(reading.jobs, width, None if rows is None else max(1, rows), gpus)
    return Group(*above, *extra, queue, *tail)
