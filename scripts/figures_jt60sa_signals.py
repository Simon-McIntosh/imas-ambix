"""Draw E101154's coil currents and plasma current from its IDS.

One figure, ``docs/figures/jt60sa-signals/jt60sa-E101154-pf-coil-currents.svg``:
one panel per series, the ten ``pf_active/coil/current`` traces CS1-EF6 and the
``magnetics/ip`` trace.  Every name and unit on the figure is read from the IDS
by imas-ink's :func:`~imas_ink.extract_signal_traces` and drawn by
:func:`~imas_ink.figures.time_trace_figure_mpl`; the script constructs no
:class:`~imas_ink.components.TimeSeries` and passes no label or unit string of
its own.  The figure is written through ``imas_ink.io.render_to_bytes`` as SVG.

The IDS file is one netCDF per pulse and run under the IDS root.  Its name is
built by the writer's :func:`~imas_alambic.pulse_writer.pulse_path`, so this
script holds no copy of the layout.

Run: ``uv run python scripts/figures_jt60sa_signals.py``.
"""

from __future__ import annotations

from pathlib import Path

import imas
import imas_ink
from imas_ink.figures import time_trace_figure_mpl
from imas_ink.io import render_to_bytes

from imas_alambic.pulse_writer import pulse_path
from imas_ambix.data.paths import JT60SA_ROOT

FIGURES = Path(__file__).resolve().parents[1] / "docs" / "figures" / "jt60sa-signals"

SHOT = "101154"
RUN = 0
FIGURE = "jt60sa-E101154-pf-coil-currents.svg"
PULSE_FILE = pulse_path(Path(JT60SA_ROOT) / "ids", SHOT, RUN)


def _read_pulse_ids(pulse_file: Path, ids_name: str):
    """Read one IDS from the pulse's run file."""

    with imas.DBEntry(str(pulse_file), "r") as entry:
        return entry.get(ids_name, autoconvert=False)


def _main() -> None:
    pf_active = _read_pulse_ids(PULSE_FILE, "pf_active")
    magnetics = _read_pulse_ids(PULSE_FILE, "magnetics")

    series = imas_ink.extract_signal_traces(
        pf_active, "coil/current"
    ) + imas_ink.extract_signal_traces(magnetics, "ip")

    figure, _axes = time_trace_figure_mpl(list(series))
    (FIGURES / FIGURE).write_bytes(render_to_bytes(figure, format="svg"))

    for trace in series:
        print(f"{trace.ylabel} [{trace.units}]")


if __name__ == "__main__":
    _main()
