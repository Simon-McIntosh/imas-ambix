"""Workstation plot style and the named colour roles a figure draws with.

Every figure applies one base sheet — the installed ``data-ink`` style — through
:func:`apply_data_ink`, and colours each component through a role name rather
than a literal hue, so a component keeps one colour across every figure of a
study.  The role names are the contract: :data:`COMPONENT_ROLES` lists the drawn
components in the order a legend or direct label reads them, and
:data:`REFERENCE_ROLES` holds the neutrals a study reserves for setpoints,
limits and modelled series.

The default colours live in :data:`DEFAULT_PALETTE`; a study that is choosing
its colours renders candidates by overriding individual roles rather than by
editing this module.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

# The base sheet.  ``data-ink`` is installed under the user's matplotlib
# stylelib; when it is absent (a bare checkout, CI) the same parameters are
# applied from the mapping below so a figure never renders unstyled.
DATA_INK = "data-ink"

_DATA_INK_RCPARAMS: dict[str, object] = {
    "figure.figsize": (14.0, 8.0),
    "figure.dpi": 100,
    "savefig.dpi": 100,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.facecolor": "white",
    "font.family": "sans-serif",
    "font.size": 20,
    "axes.labelsize": 22,
    "axes.titlesize": 22,
    "figure.titlesize": 22,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,
    "text.color": "#0b0b0b",
    "axes.labelcolor": "#0b0b0b",
    "lines.linewidth": 2.6,
    "lines.markersize": 9,
    "axes.linewidth": 1.2,
    "axes.edgecolor": "#52514e",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "axes.titlelocation": "left",
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "legend.frameon": False,
    "legend.handlelength": 1.6,
}

# Linewidth roles from the sheet: a headline series is heavier than a companion,
# a setpoint is a neutral step, a limit is a thin dotted grey.
HEADLINE_LINEWIDTH = 3.0
SERIES_LINEWIDTH = 2.6
REFERENCE_LINEWIDTH = 2.4
LIMIT_LINEWIDTH = 1.2

COMPONENT_ROLES: tuple[str, ...] = (
    "pf_coils",
    "vessel_op1",
    "vessel_op2",
    "wall",
    "flux_loops",
    "pickup_probes",
)
"""Drawn components, each of which keeps one colour across every figure."""

REFERENCE_ROLES: tuple[str, ...] = ("reference", "limit", "model")
"""Neutrals reserved for setpoints, limits and modelled series."""

ALL_ROLES: tuple[str, ...] = COMPONENT_ROLES + REFERENCE_ROLES

#: The repository default colour roles.  The two vessel roles are lightness
#: steps of one blue hue, because the OP1 and OP2 vessels are the same structure
#: at the two phase geometries; the remaining components take distinct hues.
#: Red is reserved for a limit being broken, so no role here is red.  Every
#: colour holds at least 3:1 contrast against the white chart surface.
DEFAULT_PALETTE: dict[str, str] = {
    "pf_coils": "#872b6d",
    "vessel_op1": "#588cfe",
    "vessel_op2": "#345dba",
    "wall": "#016900",
    "flux_loops": "#a08d00",
    "pickup_probes": "#1e9ea7",
    "reference": "#52514e",
    "limit": "#8a8a8a",
    "model": "#0b0b0b",
}


def apply_data_ink() -> None:
    """Apply the workstation base sheet, vendoring its parameters if absent."""
    try:
        plt.style.use(DATA_INK)
    except OSError:
        plt.style.use(_DATA_INK_RCPARAMS)


def palette(overrides: dict[str, str] | None = None) -> dict[str, str]:
    """Return the role colours, with ``overrides`` applied per role."""
    colours = dict(DEFAULT_PALETTE)
    if overrides:
        unknown = set(overrides) - set(ALL_ROLES)
        if unknown:
            raise KeyError(f"unknown colour role(s): {sorted(unknown)}")
        colours.update(overrides)
    return colours


def direct_label(
    ax,
    x: float,
    y: float,
    text: str,
    colour: str,
    *,
    dx: float = 0.0,
    dy: float = 0.0,
    ha: str = "left",
    va: str = "center",
    weight: str = "normal",
) -> None:
    """Write ``text`` at ``(x, y)`` in the series colour, offset by ``dx``/``dy``.

    A direct label carries the series identity where a legend box would; it is
    drawn in the series colour so the reader binds the word to the mark without
    a lookup.
    """
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(x + dx, y + dy),
        color=colour,
        fontsize=20,
        ha=ha,
        va=va,
        fontweight=weight,
    )
