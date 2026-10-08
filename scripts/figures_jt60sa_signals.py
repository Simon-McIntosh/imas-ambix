"""Draw JT-60SA pulse signals from written pulse IDSs.

Each figure has one panel per series. The coil-current figure draws the ten
``pf_active/coil/current`` traces CS1-EF6 and the ``magnetics/ip`` trace; the
flux-loop figure draws the ``magnetics/flux_loop/flux`` traces FL7 and the 26
differential entries FL1-FL7 to FL27-FL7; the probe figure draws the 17
``magnetics/b_field_pol_probe/field`` traces MP1-MP17.
Every name and unit is read from the IDS by imas-ink's
:func:`~imas_ink.extract_signal_traces` and drawn by
:func:`~imas_ink.figures.time_trace_figure_mpl`; the script constructs no
:class:`~imas_ink.components.TimeSeries` and passes no label or unit string of
its own. The figures are written through ``imas_ink.io.render_to_bytes`` as SVG.

FL7 is absolute. Each of the other 26 panels is the data dictionary's
differential flux psi_L - psi_7 for apparatus loop L, drawn as stored: the
stored value is the negative of the raw measured difference relative to FL7, and
the script never re-signs data. The flux-loop figure skips the 27 type-1 loop
entries that carry geometry and no flux, so the 26 differential entries appear
in the order the run file stores them.

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
from imas_alambic.settings import (
    ENV_IDS_ROOT,
    SettingsFlags,
    require_setting,
    resolve_settings,
)

FIGURES = Path(__file__).resolve().parents[1] / "docs" / "figures" / "jt60sa-signals"
PULSES = ("101154", "101031")
FIGURE_CALLS = (
    ("pf-coil-currents", (("pf_active", "coil/current"), ("magnetics", "ip"))),
    ("flux-loops", (("magnetics", "flux_loop/flux"),)),
    ("pf-probes", (("magnetics", "b_field_pol_probe/field"),)),
)


def _read_pulse_ids(pulse_file: Path, ids_name: str):
    """Read one IDS from the pulse's run file."""

    with imas.DBEntry(str(pulse_file), "r") as entry:
        return entry.get(ids_name, autoconvert=False)


def _pulse_figure(
    root: Path,
    pulse: str,
    run: int,
    series_pairs: tuple[tuple[str, str], ...],
):
    """Return a figure of IDS signals from one written pulse run."""

    pulse_file = pulse_path(root, pulse, run)
    series = [
        trace
        for ids_name, signal_path in series_pairs
        for trace in imas_ink.extract_signal_traces(
            _read_pulse_ids(pulse_file, ids_name), signal_path
        )
    ]
    figure, _axes = time_trace_figure_mpl(series)
    return figure


def _main() -> None:
    settings = resolve_settings(SettingsFlags(machine="jt-60sa"))
    root = Path(require_setting("ids_root", settings.ids_root, ENV_IDS_ROOT))
    for pulse in PULSES:
        for suffix, series_pairs in FIGURE_CALLS:
            figure = _pulse_figure(root, pulse, 0, series_pairs)
            output = FIGURES / f"jt60sa-E{pulse}-{suffix}.svg"
            output.write_bytes(render_to_bytes(figure, format="svg"))


if __name__ == "__main__":
    _main()
