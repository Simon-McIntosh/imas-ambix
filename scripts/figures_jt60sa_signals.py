"""Draw the cached JT-60SA pulses through imas-ink.

Three figures, one SVG each under ``docs/figures/jt60sa-signals/``:

1. ``jt60sa-E101154-pf-coil-currents.svg`` — the PF coil currents the
   ``pf_active`` map serves for E101154, one panel per coil, and the plasma
   current the ``magnetics`` map serves.
2. ``jt60sa-E101154-flux-loops.svg`` — one panel per flux-loop index: the
   processed PSRC ``magFluxLp`` channel then the raw MDAC ``magFlxLp`` channel
   of the same index, the raw one dashed, the raw-to-processed correspondence
   drawn rather than assumed.
3. ``jt60sa-C060033-pf-coil-currents.svg`` — the same PF coil currents for the
   no-plasma commissioning record C060033.

Which channel each panel draws, and at what factor, comes from the packaged
values through the engine path they feed, taking the bound channel's unit from
its ``VirtualArray`` attributes; every time base, and every channel the maps
leave unbound, is read whole through :func:`read_channel`.  Each panel's first
series names its channel and unit on the y-axis, so no series carries a legend
label; the stacked panels share one time axis and share its label too.  The
script draws nothing itself and holds no colour, label or style constant: the
figures are imas-ink's default style, and the only style named here is the
replace that dashes the raw flux loop.  Figures are written through
``imas_ink.io.render_to_bytes`` so the drawn paths take the resolution bound
the style sets.

Run: ``uv run python scripts/figures_jt60sa_signals.py``.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import zarr
from imas_ink.components import TimeSeries
from imas_ink.figures import DEFAULT_STYLE, time_trace_figure_mpl
from imas_ink.io import render_to_bytes

from imas_ambix.data.eddb import normalised_shot, read_channel
from imas_ambix.data.paths import JT60SA_ROOT
from imas_ambix.data.signal_map import SignalRule, load_packaged_signal_map
from imas_ambix.data.virtual_zarr import VirtualZarrView

FIGURES = Path(__file__).resolve().parents[1] / "docs" / "figures" / "jt60sa-signals"

#: The bound channel's unit is carried on the virtual array's attributes under
#: this key, set from the signal map's target unit.
_UNITS_ATTR = "units"


def _store_path(root: Path, shot: str) -> Path:
    return Path(root) / f"{normalised_shot(shot)}.zarr"


def _bound_panel(
    root: Path, shot: str, rule: SignalRule, view: VirtualZarrView
) -> list[tuple[TimeSeries, str, np.ndarray]]:
    """One panel's first series for a map-bound channel, with its name and time.

    The values come through the virtual view, which applies the signal's
    factors; the unit is taken from the virtual array's attributes; the time
    base comes from the cache through :func:`read_channel`.
    """

    array = view[rule.semantic_id]
    values = np.asarray(array[...])
    if values.ndim > 1:
        values = values[0]
    time = read_channel(root, shot, rule.source_group, rule.source_array).time
    series = TimeSeries(
        time,
        values,
        ylabel=rule.semantic_id,
        units=str(array.attrs[_UNITS_ATTR]),
    )
    return [(series, rule.semantic_id, time)]


def _flux_panel(
    root: Path, shot: str, index: int, raw_style
) -> list[tuple[TimeSeries, str, np.ndarray]]:
    """One flux-loop panel: the processed PSRC channel then the raw MDAC one."""

    processed = read_channel(root, shot, "PSRC", f"magFluxLp{index}")
    raw = read_channel(root, shot, "MDAC", f"magFlxLp{index}")
    panels = [
        TimeSeries(
            processed.time,
            processed.data[0],
            ylabel=processed.dname,
            units=processed.unit,
        ),
        TimeSeries(
            raw.time,
            raw.data[0],
            ylabel=raw.dname,
            units=raw.unit,
            style=raw_style,
        ),
    ]
    return [
        (panels[0], processed.dname, processed.time),
        (panels[1], raw.dname, raw.time),
    ]


def _flux_loop_indices(root: Path, shot: str) -> list[int]:
    """The flux-loop indices whose processed and raw channels are both cached."""

    store = zarr.open_group(_store_path(root, shot), mode="r")

    def found(category: str, prefix: str) -> set[int]:
        indices = set()
        group = store[category]
        for name in group.array_keys():
            if name.endswith("_time") or not name.startswith(prefix):
                continue
            suffix = name[len(prefix) :]
            if suffix.isdigit():
                indices.add(int(suffix))
        return indices

    return sorted(found("PSRC", "magFluxLp") & found("MDAC", "magFlxLp"))


def _report(title: str, panels: list[list[tuple[TimeSeries, str, np.ndarray]]]) -> None:
    """Print the channel names and units drawn, and where a time base differs."""

    drawn = [entry for panel in panels for entry in panel]
    print(f"{title}:")
    for series, name, _time in drawn:
        print(f"  {name} [{series.units}]")

    units = {series.units for series, _name, _time in drawn}
    if len(units) > 1:
        print(f"  units differ across channels: {', '.join(sorted(units))}")

    for (_prev_series, prev_name, prev_time), (_series, name, time) in zip(
        drawn, drawn[1:], strict=False
    ):
        if time.shape != prev_time.shape or not np.array_equal(time, prev_time):
            print(
                f"  {name}: time base differs from its neighbour {prev_name} "
                f"({time.shape[-1]} vs {prev_time.shape[-1]} samples)"
            )


def _render(
    filename: str, panels: list[list[tuple[TimeSeries, str, np.ndarray]]]
) -> None:
    figure, _axes = time_trace_figure_mpl(
        [[series for series, _name, _time in panel] for panel in panels],
        style=DEFAULT_STYLE,
    )
    (FIGURES / filename).write_bytes(render_to_bytes(figure, format="svg"))
    _report(filename, panels)


def _main() -> None:
    root = Path(JT60SA_ROOT)
    FIGURES.mkdir(parents=True, exist_ok=True)

    pf_map = load_packaged_signal_map("jt-60sa", "pf_active")
    mag_map = load_packaged_signal_map("jt-60sa", "magnetics")

    first = "E101154"
    pf_view = VirtualZarrView.open(
        str(_store_path(root, first)), pf_map, shot=int(normalised_shot(first))
    )
    mag_view = VirtualZarrView.open(
        str(_store_path(root, first)), mag_map, shot=int(normalised_shot(first))
    )
    pf_panels = [_bound_panel(root, first, rule, pf_view) for rule in pf_map.signals]
    current_panels = [
        _bound_panel(root, first, rule, mag_view) for rule in mag_map.signals
    ]
    _render("jt60sa-E101154-pf-coil-currents.svg", pf_panels + current_panels)

    raw_style = replace(
        DEFAULT_STYLE, trace_linestyle=DEFAULT_STYLE.ref_lcfs_linestyle
    )
    flux_panels = [
        _flux_panel(root, first, index, raw_style)
        for index in _flux_loop_indices(root, first)
    ]
    _render("jt60sa-E101154-flux-loops.svg", flux_panels)

    commissioning = "C060033"
    commissioning_view = VirtualZarrView.open(
        str(_store_path(root, commissioning)),
        pf_map,
        shot=int(normalised_shot(commissioning)),
    )
    _render(
        "jt60sa-C060033-pf-coil-currents.svg",
        [_bound_panel(root, commissioning, rule, commissioning_view)
         for rule in pf_map.signals],
    )


if __name__ == "__main__":
    _main()
