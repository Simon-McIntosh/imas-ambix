"""Per-era coil-circuit semantic ledger for the MAST amc pseudo-current channels.

Answers, for every G_pf column the geometry operator can build, WHAT each amc
channel physically means (feed vs coil vs case), which physical circuit drives
it, whether that circuit is independently energised or structurally passive,
and — crucially — whether the channel actually EXISTS per campaign era.  This
is the reference that lets a named flux-loop drive-column anomaly be
interpreted: is the amc channel the operator consumes the coil winding current,
the raw supply feed current, or the induced structural case current?

Two facts drive the ledger: the resolved geometry table declares which circuits
are active windings and the measured channel and weight for every driven
circuit; its acquisition channel list records which of those declarations is
available in each campaign era.  The geometry classifier consumes those same
declarations, so the ledger and operator share one machine-description source.

Measured scale precedents attached to the relevant rows as ``expected_scale``:

* solenoid (ohmic, circuit 1): ``k_sol = 1.0825`` — a UNIFORM 8.25%
  under-prediction of the solenoid response (a single scale, not a hidden
  circuit; SOLENOID_RESPONSE_SCALE in operator.py).
* the 8 measured case circuits (ids 14-21): the "case-k class" measured in an
  earlier 85-shot audit — scale factors of 0.66-0.71 for most, and ~-0.04 for
  the near-null / sign-ambiguous one.

Firewall: raw amb/amc geometry-only operator; NO EFIT reconstruction, NO amm
computed currents (those are wall-model outputs, never read here).

Usage::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \\
        python3 scripts/coil_circuit_semantic_ledger.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from imas_ambix.data.description_reader import read_geometry_table
from imas_ambix.gs.operator import build_operator

_REPO = Path(__file__).resolve().parents[1]
_ARTIFACT = (
    _REPO / "imas_ambix/latent/artifacts/patch_gate/coil_circuit_semantic_ledger.json"
)
_FIGURE = (
    _REPO / "docs/figures/nonaxisymmetric-field-subtraction/fig-coil-circuit-ledger.png"
)

# --- Measured scale precedents ----------------------------------------------
# The solenoid uniform under-prediction scale (SOLENOID_RESPONSE_SCALE): the
# vacuum-measured response is 8.25% low, corrected by a single scalar.
K_SOL = 1.0825
# The "case-k class": scale factors an earlier 85-shot vacuum audit measured
# for the coil-case circuits (case-current channel vs modelled case response).
CASE_K_MAIN = "0.66-0.71"
CASE_K_NULL = "-0.04"

# --- Campaign eras ----------------------------------------------------------
# Boundaries from the authoritative campaign map.  The amc "error_field"
# channels relabel from error_field_a/b to error_field_02/05 at ~shot 18500
# (mid-M6); the in-vessel RMP coil set grows (12 at >=19031, 18 at >=25404) but
# the efm PF geometry signature is frozen across all of them.  Each era carries
# a candidate shot list; the first that loads becomes its representative.
_ERA_CANDIDATES: tuple[tuple[str, str, tuple[int, ...]], ...] = (
    ("M5", "2004-12 to 2005-12", (13000, 13001, 12500, 13500, 12000)),
    ("M6-early", "2006-01 to ~2007 (error_field_a/b)", (15500, 15501, 16000, 15000)),
    ("M6-late", "~2007-08 (error_field_02/05 relabel)", (18700, 18701, 18800, 18600)),
    ("M7-early", "2008-04 (12-coil RMP install)", (19100, 19101, 19200, 19300)),
    ("M7-late", "2010-2011 (late M7)", (24800, 24801, 25000, 24500)),
    ("M8", "2011 (18-coil RMP)", (26000, 26001, 26500, 27000)),
    ("M9", "2012-2013 (final campaign)", (29000, 29001, 29500, 28900)),
)


def _classify_channel_semantics(channel: str) -> str:
    """FEED / COIL / CASE / (bare) COIL for a raw amc current channel name."""
    if channel.endswith("_feed_current"):
        return "feed"
    if channel.endswith("_case_current"):
        return "case"
    if channel.endswith("_coil_current"):
        return "coil"
    # sol_current / p6u_current / p6l_current — single bare winding channel
    return "coil"


def _expected_scale_for(circuit_id: int, is_case: bool, constrained_zero: bool) -> str:
    """Attach the measured scale precedent to the circuit, else empty."""
    if circuit_id == 1:  # ohmic / solenoid
        return f"k_sol={K_SOL} (uniform +8.25% response)"
    if is_case and not constrained_zero:
        return f"case-k class ({CASE_K_MAIN}; {CASE_K_NULL} for the near-null)"
    if is_case and constrained_zero:
        return "0 (the resolved source declares no drive)"
    return ""  # active P2-P6 coils: no measured scale anomaly on the coil column


def _build_era(era: str, window: str, candidates: tuple[int, ...]) -> dict | None:
    """Build the ledger rows for one era from the first candidate shot that loads."""
    shot = None
    table = None
    for cand in candidates:
        try:
            table = read_geometry_table(int(cand))
            shot = int(cand)
            break
        except Exception as exc:  # noqa: BLE001
            print(f"  [{era}] shot {cand} failed to load: {exc}", file=sys.stderr)
            continue
    if table is None or shot is None:
        print(f"  [{era}] NO candidate shot loaded", file=sys.stderr)
        return None

    amc_present = set(table.amc_current_channels)
    op = build_operator(table)
    by_circ = {c.circuit: c for c in op.circuit_classes}
    # which G_pf column (amc channel) each classified circuit landed in
    consumed_channels = set(op.pf_amc_channels)

    active_ids = set(table.active_circuits)
    drives = {drive.circuit: drive for drive in table.circuit_drives}

    rows: list[dict] = []

    # --- source-declared active circuits ------------------------------------
    for cid in sorted(active_ids):
        drive = drives.get(cid)
        cls = by_circ.get(cid)
        channel = drive.channel if drive is not None else ""
        coil_label = drive.conductor if drive is not None else ""
        exists = channel in amc_present
        consumed = channel if channel in consumed_channels else ""
        note_parts = []
        if cls is not None:
            note_parts.append(f"declared-table role={cls.role}")
            if cls.flag:
                note_parts.append(f"flag={cls.flag}")
        else:
            note_parts.append("circuit absent from this era's geometry")
        if drive is not None:
            note_parts.append(
                f"declared ampere-turns per ampere={drive.ampere_turns_per_ampere:.17g}"
            )
        rows.append(
            {
                "circuit": cid,
                "circuit_name": coil_label or f"circuit-{cid}",
                "family": "active",
                "role": cls.role if cls is not None else "absent",
                "coil_label": coil_label,
                "amc_channel": consumed or channel,
                "semantics": _classify_channel_semantics(consumed or channel),
                "exists": exists,
                "constrained_zero": False,
                "expected_scale": _expected_scale_for(cid, False, False),
                "feed_channel": "",
                "feed_exists": False,
                "note": "; ".join(note_parts),
            }
        )

    # --- source-declared driven structural circuits -------------------------
    for cid in sorted(set(drives) - active_ids):
        drive = drives[cid]
        cls = by_circ.get(cid)
        case_chan = drive.channel
        case_exists = case_chan in amc_present
        note_parts = [f"declared conductor {drive.conductor}"]
        if cls is not None:
            note_parts.append(f"declared-table role={cls.role}")
            if cls.flag:
                note_parts.append(f"flag={cls.flag}")
        else:
            note_parts.append("circuit absent from this era's geometry")
        rows.append(
            {
                "circuit": cid,
                "circuit_name": drive.conductor,
                "family": "case",
                "role": cls.role if cls is not None else "absent",
                "coil_label": drive.conductor,
                "amc_channel": case_chan,
                "semantics": "case",
                "exists": case_exists,
                "constrained_zero": False,
                "expected_scale": _expected_scale_for(cid, True, False),
                "feed_channel": "",
                "feed_exists": False,
                "note": "; ".join(note_parts),
            }
        )

    # error-field / RMP drive channels present (non-PF, non-plasma) — recorded
    # for provenance so a reader sees the era's relabel (a/b -> 02/05) and RMP
    # channel inventory; these are NOT PF/case columns.
    ef_channels = sorted(
        ch
        for ch in amc_present
        if ("error_field" in ch or ch.startswith("rmp") or "sad" in ch or "elm" in ch)
    )

    return {
        "era": era,
        "window": window,
        "shot": shot,
        "n_amc_channels": len(amc_present),
        "amc_channels": sorted(amc_present),
        "error_field_channels": ef_channels,
        "n_g_pf_columns": int(op.g_pf.shape[1]),
        "g_pf_amc_channels": list(op.pf_amc_channels),
        "rows": rows,
    }


def build_ledger() -> dict:
    """Build the full per-era ledger."""
    eras: list[dict] = []
    for era, window, cands in _ERA_CANDIDATES:
        print(f"[era {era}] candidates {cands}", file=sys.stderr, flush=True)
        e = _build_era(era, window, cands)
        if e is not None:
            eras.append(e)
            print(
                f"  -> shot {e['shot']}: {e['n_amc_channels']} amc chans, "
                f"{e['n_g_pf_columns']} G_pf cols",
                file=sys.stderr,
            )

    # presence matrix: circuit x era -> "coil"/"feed"/"case"/"absent"/"zero"
    all_circuits = sorted(
        {(row["circuit"], row["circuit_name"]) for era in eras for row in era["rows"]}
    )
    era_keys = [e["era"] for e in eras]
    matrix: dict[str, dict[str, str]] = {}
    for cid, name in all_circuits:
        row: dict[str, str] = {}
        for e in eras:
            r = next((x for x in e["rows"] if x["circuit"] == cid), None)
            if r is None:
                row[e["era"]] = "absent"
            elif r["constrained_zero"]:
                row[e["era"]] = "zero"
            elif not r["exists"]:
                row[e["era"]] = "no-channel"
            else:
                row[e["era"]] = r["semantics"]
        matrix[f"{cid:02d}:{name}"] = row

    return {
        "schema": "coil-circuit-semantic-ledger-v0",
        "firewall": "raw amb/amc + efm geometry only; NO EFIT, NO amm currents",
        "scale_precedents": {
            "solenoid_k_sol": K_SOL,
            "solenoid_note": "uniform +8.25% response scale (SOLENOID_RESPONSE_SCALE)",
            "case_k_main": CASE_K_MAIN,
            "case_k_null": CASE_K_NULL,
            "case_k_note": (
                "85-shot vacuum audit case-circuit scale factors; 0.66-0.71 for"
                " the 8 measured case circuits, ~-0.04 for the near-null one"
            ),
        },
        "semantics_key": {
            "feed": "raw per-turn supply current (kA); *_feed_current",
            "coil": "winding current = feed x turns (amp-turns); *_coil_current",
            "case": "induced structural coil-case current; *_case_current",
        },
        "era_keys": era_keys,
        "presence_matrix": matrix,
        "eras": eras,
    }


def _plot(ledger: dict, out: Path) -> None:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.patches import Patch  # noqa: PLC0415

    era_keys = ledger["era_keys"]
    matrix = ledger["presence_matrix"]
    circ_labels = list(matrix.keys())

    # colour code per cell semantic
    cmap = {
        "coil": "#1b7837",  # green — winding current (the GS-path actuator)
        "feed": "#7fbf7b",  # light green — supply feed (sibling)
        "case": "#2166ac",  # blue — induced structural case current
        "zero": "#b2182b",  # red — constrained to zero (no channel)
        "no-channel": "#f4a582",  # orange — channel absent this era
        "absent": "#d9d9d9",  # grey — circuit not in this era's geometry
    }
    order = ["coil", "feed", "case", "zero", "no-channel", "absent"]
    code = {k: i for i, k in enumerate(order)}

    n_c = len(circ_labels)
    n_e = len(era_keys)
    grid = np.full((n_c, n_e), np.nan)
    for i, cl in enumerate(circ_labels):
        for j, ek in enumerate(era_keys):
            grid[i, j] = code.get(matrix[cl][ek], code["absent"])

    fig, ax = plt.subplots(figsize=(max(9, 1.5 * n_e + 5), max(7, 0.36 * n_c)))
    listed = matplotlib.colors.ListedColormap([cmap[k] for k in order])
    ax.imshow(
        grid,
        aspect="auto",
        cmap=listed,
        vmin=-0.5,
        vmax=len(order) - 0.5,
        interpolation="nearest",
    )
    # per-cell separators so the block boundaries read clearly
    for x in range(n_e + 1):
        ax.axvline(x - 0.5, color="white", lw=0.6)
    for y in range(n_c + 1):
        ax.axhline(y - 0.5, color="white", lw=0.6)

    ax.set_xticks(range(n_e))
    ax.set_xticklabels(era_keys, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(n_c))
    ax.set_yticklabels(circ_labels, fontsize=7, family="monospace")
    ax.set_title(
        "MAST amc coil-circuit semantic ledger — channel present per era\n"
        f"solenoid k_sol={K_SOL} (uniform +8.25%);  case-k class "
        f"{CASE_K_MAIN} / {CASE_K_NULL}",
        fontsize=10,
    )
    # divider between active (1-13) and case (14-23)
    ax.axhline(12.5, color="k", lw=1.6)
    ax.text(
        -0.6, 6, "active", rotation=90, va="center", ha="right", fontsize=8, color="k"
    )
    ax.text(
        -0.6, 17, "case", rotation=90, va="center", ha="right", fontsize=8, color="k"
    )

    # right-margin scale-precedent tags (past the last era column)
    for i, cl in enumerate(circ_labels):
        cid = int(cl.split(":")[0])
        if 14 <= cid <= 21:
            ax.text(
                n_e - 0.35,
                i,
                "case-k",
                va="center",
                ha="left",
                fontsize=6,
                color="#08306b",
            )
        elif cid == 1:
            ax.text(
                n_e - 0.35,
                i,
                "k_sol",
                va="center",
                ha="left",
                fontsize=6,
                color="#00441b",
            )

    # bottom strip: the ONE real per-era amc change — error-field channel relabel
    ef_map = {e["era"]: e["error_field_channels"] for e in ledger["eras"]}
    ef_line = "error-field drive channels:   " + "   ".join(
        f"{k}={'/'.join(c.replace('error_field_', '') for c in ef_map.get(k, []))}"
        for k in era_keys
    )
    fig.text(
        0.5,
        0.005,
        ef_line + "     (a/b -> 02/05 relabel mid-M6; efm PF geometry frozen)",
        ha="center",
        fontsize=7,
        family="monospace",
        color="#555555",
    )

    handles = [Patch(facecolor=cmap[k], label=k) for k in order]
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.08, 1.0),
        fontsize=8,
        frameon=False,
        title="channel semantics",
    )
    fig.subplots_adjust(bottom=0.16, right=0.82)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifact", type=Path, default=_ARTIFACT)
    ap.add_argument("--figure", type=Path, default=_FIGURE)
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    ledger = build_ledger()
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    args.artifact.write_text(json.dumps(ledger, indent=2))
    print(f"wrote ledger -> {args.artifact}")

    if not args.no_figure:
        _plot(ledger, args.figure)
        print(f"wrote figure -> {args.figure}")

    # concise stdout summary
    print("\n=== per-era summary ===")
    for e in ledger["eras"]:
        print(
            f"{e['era']:9s} shot {e['shot']}  amc={e['n_amc_channels']:3d}  "
            f"G_pf cols={e['n_g_pf_columns']}  "
            f"ef_chans={len(e['error_field_channels'])}"
        )


if __name__ == "__main__":
    main()
