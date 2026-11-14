"""Layout, colour and figure formatting shared by the watch views.

``llm-watch`` and ``ambix-watch`` are drawn from the same parts, so a reader
moving between them finds one layout: a single column as wide as the page,
which stops at :data:`MAX_WIDTH` however wide the terminal is,
made of sections. A section is a bold title on a line of its own, its table
beneath, and a blank line after it; nothing is framed or ruled. A table has
fixed-width naming columns on the left and figure columns that share the
remaining width evenly, right-aligned, so the last figure always sits against
the page's right edge and a wider terminal gives the figures more room without
changing where they are.
"""

from __future__ import annotations

from rich.console import Group
from rich.constrain import Constrain
from rich.table import Table
from rich.text import Text

from imas_ambix.agent.watch import fmt_span, fmt_tokens, fmt_usd

__all__ = [
    "DASH", "DEFAULT_WIDTH", "MAX_WIDTH", "NARROW_WIDTH", "WIDTHS",
    "blank", "cell", "flex_table", "fmt_count", "fmt_ctx", "fmt_rate",
    "fmt_span", "fmt_tokens", "fmt_usd", "head", "line_count", "meter",
    "page", "page_width", "pct", "bar_width", "ramp", "section", "spark", "split_line",
]

#: The page width when nothing says how wide the terminal is.
DEFAULT_WIDTH = 110

#: The narrowest and the widest a page is drawn. Below the first the figure
#: columns no longer hold their figures; above the second the eye has too far
#: to travel between a row's name and its last figure.
MIN_WIDTH = 80
MAX_WIDTH = 110

#: The page width below which a table drops its least important columns, so
#: the ones it keeps are never cut.
NARROW_WIDTH = 100

#: The widths the views are checked at: the narrowest page, the default, and
#: the widest.
WIDTHS = (80, 100, 110)


def page_width(columns: int | None = None) -> int:
    """The width a page is drawn at for a terminal ``columns`` wide."""
    if not columns:
        return DEFAULT_WIDTH
    return max(MIN_WIDTH, min(MAX_WIDTH, int(columns)))


def page(parts: list[object], width: int) -> Constrain:
    """``parts`` stacked as one page ``width`` columns wide.

    The width holds on a wider terminal, where a table would otherwise
    stretch its figures across the whole screen.
    """
    return Constrain(Group(*parts), width)


def bar_width(width: int) -> int:
    """Cells a percentage's bar takes in a figure column of a page ``width`` wide."""
    return 4 if width < NARROW_WIDTH else 8


def flex_table(names: tuple[int, ...], figures: int) -> Table:
    """A borderless table: naming columns of fixed width, then figure columns.

    The naming columns are left-aligned and as wide as ``names`` says. The
    ``figures`` columns are right-aligned and share what is left evenly. Every
    column after the first keeps one blank column to its left, so a figure
    that fills its column is cut and never run into its neighbour.
    """
    table = Table(box=None, padding=(0, 0, 0, 1), pad_edge=False,
                  show_header=False, expand=True)
    for width in names:
        table.add_column("", width=width, no_wrap=True, overflow="ellipsis")
    for _ in range(figures):
        table.add_column("", ratio=1, justify="right", no_wrap=True,
                         overflow="ellipsis")
    return table


def head(table: Table, labels: list[str]) -> None:
    """The dim row of labels over a table's columns."""
    table.add_row(*[f"[dim]{label}[/]" for label in labels])


def split_line(left: str, right: str, width: int | None = None) -> Table:
    """One line with text at each edge of a page ``width`` wide."""
    table = Table.grid(expand=True)
    table.width = width
    table.add_column(ratio=1, no_wrap=True, overflow="ellipsis")
    table.add_column(justify="right", no_wrap=True)
    table.add_row(left, right)
    return table


def section(
    title: str, body: list[object], width: int | None = None, *, note: str = ""
) -> list[object]:
    """A titled section: ``title`` as markup, the body, then a blank line.

    ``note`` is set at the right end of the title line: the place for a fact
    about the whole section that is not one of its readings.
    """
    return [split_line(title, note, width), *body, Text("")]


def line_count(parts: list[object], width: int) -> int:
    """Lines ``parts`` take when drawn ``width`` columns wide."""
    import io

    from rich.console import Console

    probe = Console(width=width, file=io.StringIO(), force_terminal=False,
                    color_system=None)
    return len(probe.render_lines(Group(*parts), probe.options, pad=False))


def blank() -> Text:
    return Text("")


#: What a figure the document did not carry renders as. A dash is a statement
#: that nothing was measured; a zero is a statement that zero was measured.
DASH = "—"

#: The percentage levels the sparkline draws, lowest first. Eight levels is the
#: most a single character cell can carry while staying legible in a terminal.
_SPARK = "▁▂▃▄▅▆▇█"


def ramp(value: float | None, good: float, fair: float,
         inverted: bool = False) -> str:
    """Health as a colour name, which is the only mark here that is not a digit.

    ``inverted`` is for a quantity that is healthy when small -- a wait, a
    preemption count -- so the thresholds keep their meaning rather than being
    written backwards at every call site.
    """
    if value is None:
        return "dim"
    if inverted:
        return "green" if value <= good else "yellow" if value <= fair else "red"
    return "green" if value >= good else "yellow" if value >= fair else "red"


def pct(value: float | None, mode: str = "fill", places: int = 0,
        width: int = 8) -> str:
    """A proportion as a filled bar and its figure, the same way every time.

    ``mode`` sets what the colour means. ``good`` is high-is-healthy, ``warn``
    is high-is-trouble, ``fill`` is an occupancy coloured by how close to full
    it is, and ``plain`` is a level that carries no judgement at all -- a
    cache, which is meant to sit full.
    """
    if value is None:
        return DASH
    v = max(0.0, min(100.0, float(value)))
    fill = round(v / 100 * width)
    if mode == "good":
        colour = ramp(v, 80, 40)
    elif mode == "warn":
        colour = ramp(v, 80, 95, inverted=True)
    elif mode == "plain":
        colour = "cyan"
    else:
        colour = "red" if v >= 90 else "magenta" if v >= 70 else "cyan"
    bars = "█" * fill + "░" * (width - fill)
    return f"[{colour}]{bars}[/] {v:.{places}f}"


def meter(value: float | None, mode: str, width: int) -> str:
    """A bar ``width`` cells long with its percentage after it, left-aligned.

    The long form of :func:`pct`, for a column whose whole point is the level:
    the figure is padded so a stack of meters ends on one column.
    """
    if value is None:
        return DASH
    bar, _, figure = pct(value, mode, width=width).rpartition(" ")
    return f"{bar} {figure:>3}%"


def fmt_count(value: object) -> str:
    """A whole count, or a dash when the document carried no figure for it."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return DASH
    return f"{value:.0f}"


def fmt_ctx(tokens: object) -> str:
    """A context length as a reader says it: ``144k``, ``1.05M``, ``262k``."""
    if isinstance(tokens, bool) or not isinstance(tokens, (int, float)):
        return DASH
    value = float(tokens)
    if abs(value) >= 1e6:
        return f"{value / 1e6:.3g}M"
    if abs(value) >= 1e3:
        return f"{value / 1e3:.3g}k"
    return f"{value:.0f}"


def fmt_rate(value: object) -> str:
    """A per-second figure, on the same SI ladder as a token count."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return DASH
    return fmt_tokens(float(value))


def cell(value: float | None, good: float, fair: float,
         inverted: bool = False, *, digits: int = 0) -> str:
    """One figure, coloured by where it sits against its own thresholds."""
    if value is None:
        return DASH
    return f"[{ramp(value, good, fair, inverted=inverted)}]{value:.{digits}f}[/]"


def spark(values: list[float], ceiling: float | None = None) -> str:
    """A run of occupancy bins as a one-character-per-bin strip.

    Each bin is drawn against *ceiling* when one is supplied -- the gate's own
    width, so a strip read beside the lane's figures means the same thing on
    every document -- and against the highest bin otherwise. A bin at or above
    the ceiling takes the top level, and a bin that measured nothing takes the
    bottom one rather than dropping out of the strip, because a gap would read
    as a missing bin rather than a quiet one.
    """
    if not values:
        return ""
    top = ceiling if isinstance(ceiling, (int, float)) and ceiling > 0 else max(
        values + [0.0]
    )
    if top <= 0:
        return _SPARK[0] * len(values)
    out = []
    for value in values:
        level = int(max(0.0, min(0.999, value / top)) * len(_SPARK))
        out.append(_SPARK[level])
    return "".join(out)
