"""The view: every server, every lane, the ledger and the queue in one column.

The page is a stack of sections from :mod:`imas_ambix.agent.watch_style`, each
a bold title over a table as wide as the terminal; nothing is framed. In
reading order:

* one section per server: its state and totals on the title line, then one
  row per card with its utilisation now, its rolling mean and its memory as
  bars that fill their columns;
* one section per lane, which is ``llm-watch``'s own, so a lane here and the
  same lane there are one set of figures drawn one way;
* the period ledger;
* the queue of every watched server, which takes the lines that are left.

Temperature and power are a server's, not a card's, on this page: the hottest
card and the summed draw sit on the server's title line.

A figure nothing measured is a dash, never a zero.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from rich.markup import escape
from rich.table import Table
from rich.text import Text

from imas_ambix.agent import llm_watch
from imas_ambix.agent.watch_style import (
    DASH,
    DEFAULT_WIDTH,
    fmt_ctx,
    head,
    line_count,
    meter,
    page,
    ramp,
    section,
)

if TYPE_CHECKING:
    from rich.constrain import Constrain

    from imas_ambix.agent.gpu_watch.fleet import Job, Reading
    from imas_ambix.agent.gpu_watch.lane import Lane

#: Queue rows the page keeps whenever there are that many jobs to show.
MIN_QUEUE_ROWS = 10

#: Die temperature at which the cards start to throttle, °C.
THROTTLE_C = 84.0

#: Lines the queue spends on its title and its row of labels.
QUEUE_HEAD_LINES = 2

#: The width of the column that names a card.
CARD_NAME_WIDTH = 12

#: The longest a card's bar is drawn, in cells.
MAX_BAR_CELLS = 40


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


def clock() -> str:
    return f"[dim]{time.strftime('%H:%M:%S')}[/]"


# ── servers ──────────────────────────────────────────────────────────


def _down(reading: Reading) -> bool:
    low = (reading.state or "").lower()
    return "drain" in low or "down" in low


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def server_title(reading: Reading) -> str:
    """The server in one line: what it is, its state, the cards held, its totals.

    The hottest card and the summed draw are over the cards that were read,
    which may be fewer than the server holds.
    """
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
    temps = [card.temp for card in reading.cards if card.temp is not None]
    if temps:
        hottest = max(temps)
        colour = ramp(hottest, 75.0, THROTTLE_C, inverted=True)
        bits.append(f"[{colour}]{hottest:.0f} °C[/] max")
    draws = [card.watts for card in reading.cards if card.watts is not None]
    if draws:
        bits.append(f"{sum(draws) / 1000.0:.2f} kW")
    return " [dim]·[/] ".join(bits)


def _card_table() -> Table:
    table = Table(
        box=None, padding=(0, 0, 0, 2), pad_edge=False, show_header=False, expand=True
    )
    table.add_column("", width=CARD_NAME_WIDTH, no_wrap=True, overflow="ellipsis")
    for _ in range(3):
        table.add_column("", ratio=1, no_wrap=True, overflow="ellipsis")
    return table


def bar_cells(width: int) -> int:
    """Cells a card's bar takes on a page ``width`` wide: most of its column."""
    column = (width - CARD_NAME_WIDTH) // 3
    return max(6, min(MAX_BAR_CELLS, column - 10))


def server_section(
    reading: Reading, window: int, width: int, cards: bool = True, note: str = ""
) -> list[object]:
    """One server: its title line, then its cards.

    Each readable card is a row of three bars: utilisation now, its mean over
    the window, and memory. Utilisation is green when busy and red when idle,
    because an idle card is the waste. Cards another account's job holds
    cannot be read, so they are named together on one row; a mean row follows
    when there is more than one card to average, and the host's own load
    closes the table. With ``cards`` false the per-card rows are left out and
    the mean row stands for them. ``note`` is set at the right of the title.
    """
    cells = bar_cells(width)
    table = _card_table()
    head(table, ["card", "util now", f"util {window}s", "memory"])
    read = reading.cards
    if cards:
        for card in read:
            table.add_row(
                f"gpu {card.index}",
                meter(card.util, "good", cells),
                meter(card.util_mean, "good", cells),
                meter(card.mem, "fill", cells),
            )
    hidden = reading.unreadable
    if hidden:
        table.add_row(
            f"gpu {ranges(hidden)}",
            "[yellow]another account's[/]",
            "[dim]not readable[/]",
            "",
        )
    if len(read) > 1 or (read and not cards):
        name = "mean" if cards else f"gpu {ranges([c.index for c in read])}"
        table.add_row(
            f"[bold]{name}[/]",
            meter(_mean([c.util for c in read]), "good", cells),
            meter(_mean([c.util_mean for c in read]), "good", cells),
            meter(_mean([c.mem for c in read]), "fill", cells),
        )
    if reading.host is not None:
        cpu, cpu_mean, ram = reading.host
        table.add_row(
            "[dim]host[/]",
            f"[dim]cpu {cpu:.0f}%[/]",
            f"[dim]cpu {cpu_mean:.0f}%[/]",
            f"[dim]ram {ram:.0f}%[/]",
        )
    body: list[object] = []
    if _down(reading):
        # A drained or down node is the usual reason there are no card rows.
        detail = f" — {escape(reading.reason)}" if reading.reason else ""
        body.append(
            Text.from_markup(
                f"[red]no job can start here{detail}; an administrator must "
                f"resume the node[/]"
            )
        )
    if read or hidden or reading.host is not None:
        body.append(table)
    else:
        body.append(
            Text.from_markup(
                "[dim]no card readings — cards are read through a running "
                "job of this account[/]"
            )
        )
    return section(server_title(reading), body, width, note=note)


# ── lanes and ledger ─────────────────────────────────────────────────


def lane_section(lane: Lane, width: int, where: str = "") -> list[object]:
    """One lane as ``llm-watch`` draws it, titled with where it is served.

    ``where`` is the server and the cards the lane runs on, when the page
    knows them; without it the lane's own hardware stands there.
    """
    facts = [where or lane.hardware or ""]
    if lane.context is not None:
        facts.append(f"{fmt_ctx(lane.context)} ctx")
    return llm_watch.lane_section(
        {"lane": lane.lane, "live": lane.live},
        width,
        title=llm_watch.lane_title(lane.model, [fact for fact in facts if fact]),
        note=llm_watch.checkpoint_note(lane.checkpoint),
    )


def no_document(reason: str = "") -> Text:
    """The line that stands where the lanes would be, with why when known."""
    detail = f"[dim] — {escape(reason)}[/]" if reason else ""
    return Text.from_markup(f"[yellow]no lane document read[/]{detail}")


def lane_sections(
    readings: list[Reading], lanes: list[Lane], width: int, reason: str = ""
) -> list[object]:
    """Every lane, each titled with the server and the cards it is served from.

    The cards are read from the server's job that holds the lane, so a lane
    and the card rows above it can be matched.
    """
    if not lanes:
        return section("[bold]LANE[/]", [no_document(reason)], width)
    held = {
        job_id: (reading.server.name, cards)
        for reading in readings
        for job_id, cards in reading.job_cards.items()
    }
    several = len(readings) > 1
    parts: list[object] = []
    for lane in lanes:
        server, cards = held.get(lane.job_id or "", (lane.server, set()))
        bits = [
            server if several or not cards else "",
            f"gpu {ranges(sorted(cards))}" if cards else "",
        ]
        parts += lane_section(lane, width, " ".join(bit for bit in bits if bit))
    return parts


def ledger_section(ledger: list | None, published_at: object, width: int) -> list:
    """Tokens and their cost per period, as ``llm-watch`` draws them."""
    return section(
        "[bold]LEDGER[/]",
        [llm_watch.ledger_table(ledger, published_at=published_at, width=width)],
        width,
    )


# ── queue ────────────────────────────────────────────────────────────


def ordered_jobs(jobs: list[Job]) -> list[Job]:
    """Running jobs first, then pending; within each, group A, B, the rest."""
    return sorted(
        jobs,
        key=lambda job: (1 if job.pending else 0, job.group or "Z"),
    )


def queue_section(
    readings: list[Reading], width: int, rows: int | None = None
) -> list[object]:
    """The queue of every watched server, any user: running, then pending.

    This is where another account's load shows, since its cards cannot be
    read. ``rows`` bounds the table to the lines the screen has left; jobs
    that do not fit are counted on a last line, never cut silently. The title
    counts the jobs, so the totals are read without counting rows, and a
    server column appears when more than one server is watched.
    """
    several = len(readings) > 1
    jobs: dict[str, tuple[Job, str, int]] = {}
    for reading in readings:
        for job in reading.jobs:
            where = reading.server.name if job.running or not several else ""
            jobs.setdefault(job.id, (job, where, reading.gpus(job)))
    table = Table(
        box=None, padding=(0, 0, 0, 2), pad_edge=False, show_header=False, expand=True
    )
    for name_width in (8, 10, 6):
        table.add_column("", width=name_width, no_wrap=True, overflow="ellipsis")
    table.add_column("", ratio=1, no_wrap=True, overflow="ellipsis")
    if several:
        table.add_column("", width=9, no_wrap=True, overflow="ellipsis")
    table.add_column("", width=8, no_wrap=True)
    table.add_column("", width=11, justify="right", no_wrap=True)
    table.add_column("", width=5, justify="right", no_wrap=True)
    spare = [""] if several else []
    head(
        table,
        [
            "job",
            "user",
            "acct",
            "name",
            *(["server"] if several else []),
            "state",
            "elapsed",
            "cards",
        ],
    )
    every = [job for job, _, _ in jobs.values()]
    running = sum(1 for job in every if job.running)
    waiting = sum(1 for job in every if job.pending)
    title = f"[bold]QUEUE[/]  [dim]{running} running · {waiting} pending[/]"
    if not every:
        table.add_row("[dim]idle[/]", "", "", "", *spare, "", "", "")
        return section(title, [table], width)
    ordered = ordered_jobs(every)
    shown = ordered
    if rows is not None and len(ordered) > rows:
        shown = ordered[: max(1, rows - 1)]
    for job in shown:
        _, where, held = jobs[job.id]
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
            *([where or DASH] if several else []),
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
            *spare,
            "",
            "",
            "",
        )
    return section(title, [table], width)


# ── the page ─────────────────────────────────────────────────────────


def view(
    readings: list[Reading],
    lanes: list[Lane],
    ledger: list | None,
    window: int,
    *,
    published_at: object = None,
    size: tuple[int, int] | None = None,
    footer: str = "",
    reason: str = "",
    width: int = DEFAULT_WIDTH,
) -> Constrain:
    """The whole page: servers, lanes, the ledger, then the queue.

    ``size`` is the screen as ``(columns, lines)``, with zero lines for a
    page that is printed and not fitted. Fitted, the queue takes every line
    left to it. When that is fewer than :data:`MIN_QUEUE_ROWS` for the jobs
    there are, the page gives lines back in the order it can best spare them:
    each server's card rows fold into its mean row, then the ledger goes.
    ``reason`` is why there is no lane document, shown where the lanes would
    be.
    """
    tail: list[object] = [Text.from_markup(footer)] if footer else []
    jobs = len({job.id for reading in readings for job in reading.jobs})
    wanted = min(MIN_QUEUE_ROWS, jobs)
    fitted = bool(size and size[1])

    def above(cards: bool, with_ledger: bool) -> list[object]:
        parts: list[object] = []
        for at, reading in enumerate(readings):
            parts += server_section(
                reading, window, width, cards, note="" if at else clock()
            )
        if not readings:
            parts += section(
                "[bold]SERVERS[/]",
                [Text.from_markup("[yellow]no server found[/]")],
                width,
            )
        parts += lane_sections(readings, lanes, width, reason)
        if with_ledger:
            parts += ledger_section(ledger, published_at, width)
        return parts

    def rows_left(parts: list[object]) -> int | None:
        if not fitted:
            return None
        # The footer stands one blank line below the queue.
        foot = line_count(tail, width) + 1 if tail else 0
        return size[1] - line_count(parts, width) - QUEUE_HEAD_LINES - foot

    parts = above(True, True)
    rows = rows_left(parts)
    for cards, with_ledger in ((False, True), (False, False)):
        if rows is None or rows >= wanted:
            break
        parts = above(cards, with_ledger)
        rows = rows_left(parts)
    queue = queue_section(readings, width, None if rows is None else max(1, rows))
    if tail:
        return page([*parts, *queue, *tail], width)
    return page([*parts, *queue[:-1]], width)
