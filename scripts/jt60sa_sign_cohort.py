"""Summarise the JT-60SA magnetics sign cohort against the FAME equilibrium scalars.

Reads the raw EDDB cache for the reference plasma shot and every surveyed shot
that carries a FAME equilibrium, scores each row's signs against the sixteen
COCOS candidates, and writes the resulting sign report — candidates, per-shot
current envelope, unscored relations and the per-channel vacuum slope — into a
landing fragment under ``docs/evidence``.

Run with the staged map bundle first on ``IMAS_ALAMBIC_MAP_PATH`` so the reads
resolve through ``maps/magnetics.json``, ``maps/tf.json`` and
``maps/equilibrium.json``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path

import numpy as np

from imas_alambic.eddb import normalised_shot
from imas_alambic.signal_map import load_packaged_signal_map
from imas_alambic.virtual_zarr import VirtualZarrView
from imas_ambix.data.cocos_convention import (
    COCOS_CANDIDATES,
    format_sign_report,
    read_signal_map_observation,
    surviving_conventions,
)

MACHINE = "jt-60sa"

#: JT-60SA's own current floors, passed explicitly: the plasma current that
#: counts as plasma-on, and the level below which a sample is pre-plasma.
JT60SA_MINIMUM_CURRENT_A = 50_000.0
JT60SA_BASELINE_CURRENT_A = 10_000.0

JT60SA_ROOT = Path("/work/projects/imas_gpu/jt60sa")
REPOSITORY = Path(__file__).resolve().parent.parent
EVIDENCE = REPOSITORY / "docs" / "evidence" / "fragments" / "jt60sa-machine-map"
SURVEY_PATH = EVIDENCE / "jtmm-op1-shot-survey.json"
VACUUM_PATH = EVIDENCE / "jtmm-vacuum-adjudication.json"
FRAGMENT_PATH = EVIDENCE / "jtmm-sign-cohort.html"
FIGURE_DIR = REPOSITORY / "docs" / "figures" / "jt60sa-machine-map" / "jtmm-sign-cohort"
FIGURE_NAME = "jt60sa-vacuum-channel-slope.svg"

#: The catalogue's reference plasma shot, read beside the surveyed list because
#: it carries the FAME equilibrium the equilibrium map was built from.
REFERENCE_SHOT = 101154

_PLASMA_CURRENT_TARGETS = ("magnetics/ip",)

REFERENCE_COLOUR = "#b07d2b"
IDENTITY_COLOUR = "#3b6ea5"
UNDECIDED_COLOUR = "#8a8f98"
NEUTRAL = "#5a5f66"

_VERDICT_COLOURS = {
    "negate": REFERENCE_COLOUR,
    "identity": IDENTITY_COLOUR,
    "undecided": UNDECIDED_COLOUR,
}


def cohort_shots(survey_path: Path = SURVEY_PATH) -> tuple[int, ...]:
    """Return the reference shot followed by every FAME-bearing surveyed shot."""

    survey = json.loads(survey_path.read_text())
    shots = [REFERENCE_SHOT]
    for row in survey:
        if not row.get("fame_present"):
            continue
        shot = int(str(row["shot"]).lstrip("Ee"))
        if shot not in shots:
            shots.append(shot)
    return tuple(shots)


def plasma_current_envelope(
    shot: int,
    *,
    root: Path | str = JT60SA_ROOT,
    minimum_current_a: float = JT60SA_MINIMUM_CURRENT_A,
) -> tuple[float, float]:
    """Return the peak absolute plasma current and the pre-plasma baseline.

    The baseline is the median of the samples before the current first rises
    past the plasma-on floor, so it measures the pre-plasma offset of the
    channel rather than the shot's average.
    """

    magnetics_map = load_packaged_signal_map(MACHINE, "magnetics")
    rule = next(
        signal
        for signal in magnetics_map.signals
        if signal.target_path == _PLASMA_CURRENT_TARGETS[0]
        or signal.target_path.startswith(f"{_PLASMA_CURRENT_TARGETS[0]}/")
    )
    pulse = Path(root) / f"{normalised_shot(shot)}.zarr"
    view = VirtualZarrView.open(str(pulse), magnetics_map, shot=shot)
    values, _ = view.raw_series(rule.semantic_id)
    series = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = np.isfinite(series)
    if not np.any(finite):
        raise ValueError(f"shot {shot} has no finite plasma-current samples")
    series = series[finite]
    peak = float(np.max(np.abs(series)))
    onsets = np.flatnonzero(np.abs(series) >= minimum_current_a)
    prefix = series[: onsets[0]] if onsets.size else series
    return peak, float(np.median(np.abs(prefix))) if prefix.size else 0.0


def vacuum_channels(vacuum_path: Path = VACUUM_PATH) -> dict:
    """Return the vacuum adjudication's per-channel slope and correlation."""

    adjudication = json.loads(vacuum_path.read_text())
    return {
        "rule": adjudication["rule"],
        "order": adjudication["channel_order"],
        "channels": adjudication["channels"],
        "verdict_counts": adjudication["verdict_counts"],
        "swap_control": adjudication["swap_control"],
    }


def _apply_style() -> None:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    with contextlib.suppress(OSError):
        plt.style.use("data-ink")
    plt.rcParams.update(
        {
            "figure.dpi": 100,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 22,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 20,
            "axes.linewidth": 1.2,
            "svg.hashsalt": "jt60sa-vacuum-channel-slope",
        }
    )


def write_vacuum_figure(vacuum: dict, path: Path) -> Path:
    """Draw each channel's vacuum slope against its correlation.

    The channels the vacuum verdict could not decide cluster near zero slope
    and low correlation; the reference loop stands apart with a slope of about
    minus one and a correlation of about one, which is what makes it the one
    channel a sign product may use.
    """

    import matplotlib.pyplot as plt  # noqa: PLC0415

    _apply_style()
    channels = vacuum["channels"]
    rule = vacuum["rule"]
    figure, axes = plt.subplots(figsize=(14, 6))
    for verdict, colour in _VERDICT_COLOURS.items():
        names = [
            name for name, entry in channels.items() if entry["verdict"] == verdict
        ]
        if not names:
            continue
        axes.scatter(
            [channels[name]["median_abs_r"] for name in names],
            [channels[name]["median_slope"] for name in names],
            s=45,
            color=colour,
            label=f"{verdict} ({len(names)})",
            zorder=3,
        )
    floor = rule["slope_tolerance"]
    axes.axhline(floor, color=NEUTRAL, linewidth=1.2, linestyle=":", zorder=1)
    axes.axhline(-floor, color=NEUTRAL, linewidth=1.2, linestyle=":", zorder=1)
    axes.axvline(
        rule["correlation_floor"], color=NEUTRAL, linewidth=1.2, linestyle=":", zorder=1
    )
    axes.axhline(0.0, color=NEUTRAL, linewidth=1.2, zorder=1)
    axes.text(
        rule["correlation_floor"] + 0.005,
        -0.62,
        "correlation floor",
        color=NEUTRAL,
        fontsize=20,
        rotation=90,
        va="bottom",
    )
    for name in vacuum["swap_control"]:
        if name not in channels:
            continue
        entry = channels[name]
        axes.annotate(
            name,
            (entry["median_abs_r"], entry["median_slope"]),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=20,
            color=_VERDICT_COLOURS.get(entry["verdict"], NEUTRAL),
        )
    axes.set_xlabel("median |r [-]")
    axes.set_ylabel("median vacuum slope [-]")
    axes.set_xlim(0.0, 1.03)
    axes.legend(frameon=False, loc="upper left")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, format="svg", metadata={"Date": None})
    plt.close(figure)
    return path


def _escape(text: object) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_fragment(
    observations,
    envelopes,
    vacuum: dict,
    report: str,
    *,
    figure_src: str | None,
) -> str:
    """Compose the landing fragment carrying the sign report."""

    survivors = surviving_conventions(observations)
    survivor_text = ", ".join(str(candidate) for candidate in survivors) or "none"
    rows = []
    for observation in observations:
        peak, baseline = envelopes[observation.shot]
        rows.append(
            "<tr>"
            f"<td>{observation.shot}</td>"
            f"<td>{peak:,.0f}</td>"
            f"<td>{baseline:,.0f}</td>"
            f"<td>{observation.retained_slices}</td>"
            f"<td>{observation.raw_flux_loop_channels}</td>"
            f"<td>{observation.safety_factor:.4g}</td>"
            f"<td>{observation.toroidal_field_t:.4g}</td>"
            "</tr>"
        )
    channel_rows = []
    for name in vacuum["order"]:
        entry = vacuum["channels"].get(name)
        if entry is None:
            continue
        channel_rows.append(
            "<tr>"
            f"<td>{_escape(name)}</td>"
            f"<td>{_escape(entry['verdict'])}</td>"
            f"<td>{entry['median_slope']:+.4f}</td>"
            f"<td>{entry['median_abs_r']:.4f}</td>"
            "</tr>"
        )
    figure_block = ""
    if figure_src:
        figure_block = (
            f'<figure id="jtmm-sign-cohort-figure">\n'
            f'<img src="{figure_src}" alt="Per-channel vacuum slope against '
            'correlation for the JT-60SA loop and probe channels">\n'
            "<figcaption>Each channel's median vacuum slope against its median "
            "absolute correlation over the three vacuum shots. The reference "
            "loop magFlxLp7 sits at slope near minus one and correlation near "
            "one; the channels the verdict could not decide cluster at low "
            "correlation.</figcaption>\n</figure>"
        )
    heading = "JT-60SA magnetics sign cohort: the source COCOS the raw signs select"
    return f"""<section id="jtmm-sign-cohort" data-reckon="fragment">
<h2 id="jtmm-sign-cohort-h">{heading}</h2>

<p id="jtmm-sign-cohort-scope">
Read-only against the EDDB zarr cache under <code>{_escape(JT60SA_ROOT)}</code>,
through the staged signal-map bundle mounted at <code>IMAS_ALAMBIC_MAP_PATH</code>.
Each row is one shot: the plasma current and the one absolute flux loop from
<code>maps/magnetics.json</code>, the coil current from <code>maps/tf.json</code>,
and the safety factor and field scalar from <code>maps/equilibrium.json</code>,
all through <code>VirtualZarrView.raw_series</code>, so no assumed sign enters.
No file outside this repository was written.
</p>

<h3 id="jtmm-sign-cohort-answer-h">Surviving candidates</h3>
<p id="jtmm-sign-cohort-answer">
Over {len(observations)} rows the surviving COCOS candidates are
<strong>{_escape(survivor_text)}</strong>, out of {len(COCOS_CANDIDATES)} candidates.
The raw absolute flux-loop response and the q relation over the FAME signs fix
the pair; the poloidal-flux relation is unscored on every row because the FAME
equilibrium carries no psi grid.
</p>

<h3 id="jtmm-sign-cohort-cohort-h">The cohort</h3>
<table id="jtmm-sign-cohort-cohort-table">
<thead><tr><th>shot</th><th>peak |Ip| [A]</th><th>pre-plasma baseline [A]</th>
<th>retained slices</th><th>flux channels</th><th>q95</th>
<th>F [T&middot;m]</th></tr></thead>
<tbody>
{chr(10).join(rows)}
</tbody>
</table>

{figure_block}

<h3 id="jtmm-sign-cohort-report-h">The sign report</h3>
<pre id="jtmm-sign-cohort-report"><code>{_escape(report)}</code></pre>

<h3 id="jtmm-sign-cohort-vacuum-h">Per-channel vacuum slope and correlation</h3>
<table id="jtmm-sign-cohort-vacuum-table">
<thead><tr><th>channel</th><th>verdict</th><th>median slope</th>
<th>median |r|</th></tr></thead>
<tbody>
{chr(10).join(channel_rows)}
</tbody>
</table>
<p id="jtmm-sign-cohort-vacuum-note">
Slope and correlation are read from <code>jtmm-vacuum-adjudication.json</code>
over the same three vacuum shots; the verdicts and the swap control are that
node's, not re-derived here.
</p>
</section>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=JT60SA_ROOT)
    parser.add_argument("--fragment", type=Path, default=FRAGMENT_PATH)
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    parser.add_argument("--no-figure", action="store_true")
    arguments = parser.parse_args(argv)

    shots = cohort_shots()
    observations = tuple(
        read_signal_map_observation(
            shot,
            MACHINE,
            root=arguments.root,
            minimum_current_a=JT60SA_MINIMUM_CURRENT_A,
            baseline_current_a=JT60SA_BASELINE_CURRENT_A,
        )
        for shot in shots
    )
    envelopes = {
        observation.shot: plasma_current_envelope(
            observation.shot,
            root=arguments.root,
            minimum_current_a=JT60SA_MINIMUM_CURRENT_A,
        )
        for observation in observations
    }
    vacuum = vacuum_channels()
    report = format_sign_report(observations)

    figure_src = None
    if not arguments.no_figure:
        figure_path = write_vacuum_figure(vacuum, arguments.figure_dir / FIGURE_NAME)
        figure_src = (
            "/imas-ambix/figures/jt60sa-machine-map/jtmm-sign-cohort/"
            f"{figure_path.name}"
        )

    fragment = build_fragment(
        observations, envelopes, vacuum, report, figure_src=figure_src
    )
    arguments.fragment.parent.mkdir(parents=True, exist_ok=True)
    arguments.fragment.write_text(fragment)

    survivors = surviving_conventions(observations)
    print(f"shots read: {len(observations)}")
    for observation in observations:
        peak, baseline = envelopes[observation.shot]
        print(
            f"  {observation.shot}: peak |Ip| {peak:,.0f} A, "
            f"baseline {baseline:,.0f} A, flux channels "
            f"{observation.raw_flux_loop_channels}, q95 "
            f"{observation.safety_factor:.4g}"
        )
    print(f"surviving COCOS candidates: {survivors}")
    print(report)
    print(f"fragment written: {arguments.fragment}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
