"""Geometry, colour and figure formatting shared by the llm-watch view.

One module owns the column grid, the terminal widths the view is rendered at,
and the two formatters that turn a proportion or a count into text. The bands,
the three tables and the ledger all read them from here, so a change to the
pitch or to a colour is one edit rather than a search for every call site.

The grid is the ledger's own, as the serve panel it replaces had it: a 13
character label column followed by 12 character figure columns, right
justified, one space of padding, no vertical rules. A reader scanning down the
document re-finds every column, and a figure that moves between bands does not
change columns under it.
"""

from __future__ import annotations

from imas_ambix.agent.watch import bar, fmt_span, fmt_tokens, fmt_usd

__all__ = [
    "CELL_WIDTH", "COLUMNS", "DASH", "GRID_WIDTH", "LABEL_WIDTH", "WIDTHS",
    "bar", "cell", "fmt_count", "fmt_ctx", "fmt_rate", "fmt_span", "fmt_tokens",
    "fmt_usd", "pct", "ramp", "spark",
]

#: The label column's width, and every figure column's width beside it. These
#: are one character narrower than the serve panel's, with one character of
#: padding either side of each column: seven columns at the panel's own pitch
#: render 105 characters wide, which does not fit the 100-column terminal the
#: view is read in, so the pitch is set to the widest grid that does.
LABEL_WIDTH = 12
CELL_WIDTH = 11

#: How many columns the grid holds. A band with fewer figures pads the rest,
#: so every band in the document shares one label column and one figure pitch.
COLUMNS = 7

#: The plain-text width of one grid row: the label column, the padding after
#: each column, and the figure columns. The view measures its own output
#: against the terminal width, and this is the figure that must fit.
GRID_WIDTH = LABEL_WIDTH + COLUMNS * CELL_WIDTH + COLUMNS

#: The two terminal widths the view is checked at. The wide one is the width a
#: golden is rendered at; the narrow one is the tighter of the two an operator
#: may be reading in, and every line must fit both.
WIDTHS = (100, 120)

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
        width: int = 5) -> str:
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