"""Layout, colour and figure formatting shared by the watch views.

``llm-watch`` and ``ambix-watch`` are drawn from the same parts, so a reader
moving between them finds one layout. Every panel is a titled box as wide as
the page. Inside a box, a table has fixed-width naming columns on the left and
figure columns that share the remaining width evenly, right-aligned, so the
last figure always sits against the box's right edge and a wider terminal
gives the figures more room without changing where they are. Where the page is
wide enough, two stacks of boxes sit side by side.
"""

from __future__ import annotations

from rich import box as _box
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from imas_ambix.agent.watch import fmt_span, fmt_tokens, fmt_usd

__all__ = [
    "DASH", "DEFAULT_WIDTH", "MAX_WIDTH", "TWO_COLUMN_WIDTH", "WIDTHS",
    "blank", "boxed", "cell", "column_widths", "flex_table", "fmt_count",
    "fmt_ctx", "fmt_rate", "fmt_span", "fmt_tokens", "fmt_usd", "head",
    "inner_width", "line_count", "page_width", "pair", "pct", "ramp",
    "side_by_side", "spark", "split_line",
]

#: The page width when nothing says how wide the terminal is.
DEFAULT_WIDTH = 120

#: The narrowest and the widest a page is drawn. Below the first the figure
#: columns no longer hold their figures; above the second the eye has too far
#: to travel between a row's name and its last figure.
MIN_WIDTH = 80
MAX_WIDTH = 160

#: The page width from which two stacks of boxes sit side by side.
TWO_COLUMN_WIDTH = 132

#: The widths the views are checked at: the narrowest page, the default, and
#: one wide enough for two columns.
WIDTHS = (80, 120, 150)


def page_width(columns: int | None = None) -> int:
    """The width a page is drawn at for a terminal ``columns`` wide."""
    if not columns:
        return DEFAULT_WIDTH
    return max(MIN_WIDTH, min(MAX_WIDTH, int(columns)))


def inner_width(width: int) -> int:
    """The columns a box ``width`` wide leaves for its body."""
    return width - 4


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


def boxed(
    title: str, body: list[object], width: int, *, note: str = ""
) -> Panel:
    """A box ``width`` wide holding ``body``, titled with ``title`` as markup.

    ``note`` is set into the bottom edge on the right: the place for a fact
    about the whole box that is not one of its readings.
    """
    return Panel(
        Group(*body),
        title=title,
        title_align="left",
        subtitle=note or None,
        subtitle_align="right",
        box=_box.ROUNDED,
        border_style="dim",
        padding=(0, 1),
        width=width,
    )


def column_widths(width: int) -> tuple[int, int]:
    """The widths of the two stacks a page ``width`` wide divides into."""
    left = (width - 1) // 2
    return left, width - 1 - left


def side_by_side(left: list[object], right: list[object], width: int) -> Table:
    """Two stacks of boxes next to each other, one column apart."""
    left_width, right_width = column_widths(width)
    table = Table.grid(padding=0)
    table.add_column(width=left_width)
    table.add_column(width=1)
    table.add_column(width=right_width)
    table.add_row(Group(*left), "", Group(*right))
    return table


def line_count(parts: list[object], width: int) -> int:
    """Lines ``parts`` take when drawn ``width`` columns wide."""
    import io

    from rich.console import Console

    probe = Console(width=width, file=io.StringIO(), force_terminal=False,
                    color_system=None)
    return len(probe.render_lines(Group(*parts), probe.options, pad=False))


def pair(
    left: tuple[str, list[object]], right: tuple[str, list[object]], width: int
) -> list[object]:
    """Two boxes, each a ``(title, body)``: side by side where the page allows.

    Side by side they are drawn the same height, so the row they form ends on
    one line; on a narrower page they are stacked at full width.
    """
    if width < TWO_COLUMN_WIDTH:
        return [boxed(left[0], left[1], width), boxed(right[0], right[1], width)]
    widths = column_widths(width)
    boxes = [boxed(title, body, w) for (title, body), w in
             zip((left, right), widths, strict=True)]
    height = max(line_count([box], w) for box, w in zip(boxes, widths, strict=True))
    for box in boxes:
        box.height = height
    return [side_by_side([boxes[0]], [boxes[1]], width)]


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
        width: int = 4) -> str:
    """A proportion as a filled bar and its figure, the same way every time.

    ``mode`` sets what the colour means. ``good`` is high-is-healthy, ``warn``
    is high-is-trouble, and ``fill`` is neither -- an occupancy that is merely
    a fact, coloured by how full it is rather than by whether it should be.
    """
    if value is None:
        return DASH
    v = max(0.0, min(100.0, float(value)))
    fill = round(v / 100 * width)
    if mode == "good":
        colour = ramp(v, 80, 40)
    elif mode == "warn":
        colour = ramp(v, 80, 95, inverted=True)
    else:
        colour = "red" if v >= 90 else "magenta" if v >= 70 else "cyan"
    bars = "█" * fill + "░" * (width - fill)
    return f"[{colour}]{bars}[/] {v:.{places}f}"


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
