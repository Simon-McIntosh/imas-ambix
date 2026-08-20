"""Identity-independent audit of the per-coil vertical field at the plasma.

WHAT
----
For one campaign shot, this compares the GS forward operator's per-coil,
per-ampere vertical field ``B_z`` at the measured plasma location against an
independent re-tabulation built directly from the declared table's filaments
(centroids, extents, turns, circuit assignment, and current-share ``xmult``).
It reports, per coil, the signed percentage discrepancy between
the two field values at a gate point ``(R, Z) = (0.9, 0)`` plus two robustness
points, a turns cross-check against the machine description, the fcoil circuits
merged into each operator column, and the sign the coil's field takes both
per-ampere and at its measured flat-top current.

WHY
---
A force-balance diagnosis found the operator's vacuum vertical field at the
plasma ~15-27% weak on the ramp, waterfall-localised to the P4/P5 groups.  Part
of that could be a Shafranov-identity artefact at MAST's low aspect ratio and
part could be a genuine winding-representation error in the forward operator.
This audit isolates the second possibility with NO reference to any equilibrium
identity: it only measures the forward operator's coil columns against a
direct sum over the declared filaments.  If the two agree, the field deficit
is not a per-coil
geometry / turns / merge error and must be sought elsewhere; if they disagree
above a pre-declared 2% gate, the discrepancy is a candidate mechanism that
must be named as a geometry / turns / merge fix — never absorbed by a fitted
gain.  There are no fitted gains anywhere in this script.

The two tabulations use the SAME finite-area cylinder Green's kernel
(:func:`imas_ambix.gs.cylinder.hybrid_greens`); they differ only in
  * the source calculation — a direct declared-filament sum versus the
    operator's classified and merged columns; and
  * the solenoid response scale the operator applies to its P1 column.
The solenoid scale is a known, vacuum-measured machine-description correction
(:data:`imas_ambix.gs.operator.SOLENOID_RESPONSE_SCALE`); it is EXPECTED on the
sol column vs the raw geometric tabulation and is reported as a known mechanism,
never a new finding.  The scale is divided back out for a scale-free geometric
comparison of the solenoid winding representation.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from imas_ambix.gs.cylinder import hybrid_greens  # noqa: E402
from imas_ambix.gs.force_balance import known_coil_bz  # noqa: E402
from imas_ambix.gs.machine_geometry import MachineGeometryService  # noqa: E402
from imas_ambix.gs.operator import (  # noqa: E402
    COIL_MODEL_VERSION,
    SOLENOID_RESPONSE_SCALE,
    build_operator,
    classify_circuits,
    read_amc_currents_at_index,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from imas_ambix.gs.machine_geometry import OperatorGeometry

logger = logging.getLogger("coil_column_audit")

# Section-floor for the finite-area cylinder kernel (matches the operator's
# G_pf column build and force_balance._filament_field).
_SECTION_FLOOR = 0.01

# Pre-declared gate: any |discrepancy| beyond this at the gate point is a
# candidate mechanism that must be named (geometry / turns / merge), never a gain.
GATE_PCT = 2.0

# Evaluation points [m]; the gate is decided at the first (the measured axis
# neighbourhood), the others probe field-map robustness inboard / outboard.
GATE_POINT = (0.9, 0.0)
ROBUST_INNER = (0.75, 0.0)
ROBUST_OUTER = (1.1, 0.0)

# kA*turn -> A for the raw amc coil currents (turns folded into the channel).
_KA_TURN_TO_A = 1.0e3
# Flat-top selection: an index is usable when |Ip| clears this [kA].
_FLAT_TOP_IP_KA = 400.0


@dataclass(frozen=True)
class EvalPoint:
    """One field-evaluation location and its role in the audit."""

    r: float
    z: float
    role: str


def _circuit_bz_per_amp(
    filaments: Sequence[Any], r: np.ndarray, z: np.ndarray
) -> np.ndarray:
    """B_z [T per A] at ``(r, z)`` from one circuit's declared filaments.

    xmult-weighted sum of the finite-area cylinder kernel (index 2 of the
    ``(psi, br, bz)`` tuple), with the same section floor the operator uses.
    This mirrors :func:`imas_ambix.gs.force_balance._filament_field` but is
    evaluated independently of the operator's column assembly.
    """
    out = np.zeros(np.asarray(r).shape, dtype=np.float64)
    for f in filaments:
        w = float(f.xmult)
        if w == 0.0:
            continue
        da = max(abs(float(f.width)), _SECTION_FLOOR)
        dz = max(abs(float(f.height)), _SECTION_FLOOR)
        out = out + w * hybrid_greens(r, z, float(f.r), float(f.z), da, dz)[2]
    return out


def _declared_bz(
    table: OperatorGeometry,
    r: np.ndarray,
    z: np.ndarray,
) -> tuple[dict[str, float], dict[str, list[int]], list]:
    """Per-coil B_z [T per A] from a direct declared-filament sum.

    Classifies the declared filaments exactly as the operator does, groups the known
    active-PF circuits by their amc channel, and AVERAGES redundant same-channel
    circuits — mirroring :func:`imas_ambix.gs.operator.build_operator`'s merge
    rule — but without the solenoid response
    scale.  Returns ``(coil -> B_z, coil -> circuit ids, circuit classes)``.
    """
    declared_filaments = list(table.conductors)
    classes = classify_circuits(
        declared_filaments,
        table.available_current_channels,
        table.active_circuits,
        table.drive_map,
    )
    by_circ: dict[int, list[Any]] = {}
    for f in declared_filaments:
        by_circ.setdefault(f.circuit, []).append(f)

    # group KNOWN active-PF circuits by their coil label (unique amc channel).
    label_circs: dict[str, list[int]] = {}
    for cc in classes:
        if cc.role == "known_pf":
            label_circs.setdefault(cc.coil_label, []).append(cc.circuit)

    bz: dict[str, float] = {}
    circs_by_coil: dict[str, list[int]] = {}
    for label, circs in label_circs.items():
        circs = sorted(circs)
        cols = [_circuit_bz_per_amp(by_circ[c], r, z) for c in circs]
        bz[label] = float(np.mean(cols, axis=0)[0])
        circs_by_coil[label] = circs
    return bz, circs_by_coil, classes


def _operator_bz(table, r: np.ndarray, z: np.ndarray) -> dict[str, float]:
    """Per-amc-channel operator B_z [T per A] via the consumed force-balance path.

    This is exactly what the force-balance diagnosis evaluated: the collapsed
    table run through :func:`imas_ambix.gs.force_balance.known_coil_bz`, which
    applies the operator merge rule and the solenoid response scale.
    """
    channels, cols = known_coil_bz(table, r, z)
    arr = np.asarray(cols)
    return {channels[i]: float(arr[..., i][0]) for i in range(len(channels))}


def _turns_cross_check(
    coil: str, circs: list[int], table: OperatorGeometry
) -> dict[str, object]:
    """Compare the declared drive weight against the filament current shares."""
    by_circ: dict[int, list[Any]] = {}
    for f in table.conductors:
        by_circ.setdefault(f.circuit, []).append(f)
    fils = [f for c in circs for f in by_circ.get(c, [])]
    sum_xmult = float(sum(f.xmult for f in fils))
    sum_turns = float(sum(f.turns for f in fils))
    n_fil = len(fils)
    drive = next(
        (
            item
            for item in table.drive_map
            if item.circuit in circs and item.conductor == coil
        ),
        None,
    )
    drive_weight = drive.ampere_turns_per_ampere if drive is not None else None
    matches = drive_weight is not None and np.isclose(
        drive_weight, sum_xmult, rtol=0.0, atol=1.0e-12
    )
    return {
        "declared_drive_ampere_turns_per_ampere": drive_weight,
        "declared_sum_xmult": round(sum_xmult, 4),
        "declared_sum_turns": round(sum_turns, 4),
        "declared_n_filament": n_fil,
        "drive_weight_matches_sum_xmult": bool(matches),
    }


def _merge_case_check(
    merged: list[int], table: OperatorGeometry
) -> dict[str, object]:
    """Verify no case circuit was folded into this active coil's column.

    The table's active membership distinguishes winding drives from measured
    structural drives without reconstructing an identifier correspondence.
    """
    active = set(table.active_circuits)
    structural_drives = {
        drive.circuit for drive in table.drive_map if drive.circuit not in active
    }
    folded_cases = [cid for cid in merged if cid in structural_drives]
    return {
        "merged_circuits": merged,
        "clean": len(folded_cases) == 0,
        "wrongly_folded_case_circuits": folded_cases,
    }


def _find_flat_top_index(shot_id: int) -> tuple[int, float]:
    """Return a flat-top time index and its |Ip| [kA] (|Ip| clears the floor).

    Uses the RAW ``amc`` plasma-current channel only to LOCATE a stable
    high-current slice — the plasma current is never a source in this audit.
    """
    import zarr  # noqa: PLC0415

    from imas_ambix.data.paths import local_shot_path  # noqa: PLC0415

    store = zarr.open(str(local_shot_path(shot_id, tier="level1")), mode="r")
    ip = np.asarray(store["amc"]["plasma_current"][:], dtype=np.float64)
    above = np.isfinite(ip) & (np.abs(ip) > _FLAT_TOP_IP_KA)
    idxs = np.where(above)[0]
    if idxs.size == 0:
        raise RuntimeError(
            f"shot {shot_id}: no slice with |Ip| > {_FLAT_TOP_IP_KA} kA found"
        )
    mid = int(idxs[idxs.size // 2])
    return mid, float(ip[mid])


def run_audit(shot_id: int) -> dict[str, object]:
    """Build the full per-coil audit payload for one shot."""
    logger.info("reading declared winding table for shot %d", shot_id)
    table = MachineGeometryService().operator(shot_id)
    drives = {drive.circuit: drive for drive in table.drive_map}
    active_drives = [
        drives[circuit] for circuit in table.active_circuits if circuit in drives
    ]
    channels_by_coil = {drive.conductor: drive.channel for drive in active_drives}
    coils = list(channels_by_coil)

    points = [
        EvalPoint(*GATE_POINT, role="gate"),
        EvalPoint(*ROBUST_INNER, role="robust_inner"),
        EvalPoint(*ROBUST_OUTER, role="robust_outer"),
    ]

    logger.info("building operator columns from the declared table")
    op = build_operator(table)
    merged_by_chan = dict(zip(op.pf_amc_channels, op.pf_merged_circuits, strict=True))

    # Direct declared-filament and operator B_z at every evaluation point.
    declared_by_point: dict[str, dict[str, float]] = {}
    op_by_point: dict[str, dict[str, float]] = {}
    circs_by_coil: dict[str, list[int]] = {}
    for p in points:
        r = np.array([p.r], dtype=np.float64)
        z = np.array([p.z], dtype=np.float64)
        declared, circs, _ = _declared_bz(table, r, z)
        declared_by_point[p.role] = declared
        op_by_point[p.role] = _operator_bz(table, r, z)
        circs_by_coil = circs  # coil->circuit ids (point-independent)

    flat_idx, flat_ip_ka = _find_flat_top_index(shot_id)
    logger.info("flat-top index %d (|Ip| = %.1f kA)", flat_idx, flat_ip_ka)
    measured = read_amc_currents_at_index(shot_id, flat_idx)

    def _disc(op_v: float, auth_v: float) -> float:
        return (op_v - auth_v) / auth_v * 100.0 if auth_v != 0.0 else float("nan")

    rows: list[dict[str, object]] = []
    signs: dict[str, dict[str, object]] = {}
    worst_coil, worst_unexplained = "", 0.0
    n_above = 0

    for coil in coils:
        chan = channels_by_coil[coil]
        row: dict[str, object] = {"coil": coil, "amc_channel": chan}

        per_point: dict[str, dict[str, float]] = {}
        for p in points:
            auth_v = declared_by_point[p.role].get(coil, float("nan"))
            op_v = op_by_point[p.role].get(chan, float("nan"))
            per_point[p.role] = {
                "r": p.r,
                "z": p.z,
                "declared_bz_per_amp": auth_v,
                "operator_bz_per_amp": op_v,
                "discrepancy_pct": _disc(op_v, auth_v),
            }
        row["by_point"] = per_point

        gate = per_point["gate"]
        full_disc = gate["discrepancy_pct"]

        # scale-free view: the operator applies SOLENOID_RESPONSE_SCALE only to
        # the sol column, so divide it back out for a pure-geometry comparison.
        if coil == "sol":
            op_sf = gate["operator_bz_per_amp"] / SOLENOID_RESPONSE_SCALE
            sf_disc = _disc(op_sf, gate["declared_bz_per_amp"])
            row["operator_bz_per_amp_scale_free"] = op_sf
            row["scale_free_discrepancy_pct"] = sf_disc
            unexplained = sf_disc
        else:
            unexplained = full_disc

        row["turns_cross_check"] = _turns_cross_check(
            coil, circs_by_coil.get(coil, []), table
        )
        row["merge"] = _merge_case_check(merged_by_chan.get(chan, []), table)

        # name the mechanism for anything beyond the gate.
        mechanism = ""
        if coil == "sol" and abs(full_disc) > GATE_PCT:
            mechanism = (
                "known_mechanism: solenoid-response-scale "
                f"(x{SOLENOID_RESPONSE_SCALE} applied to the G_pf sol column; "
                "scale-free winding discrepancy "
                f"{row['scale_free_discrepancy_pct']:+.2f}%)"
            )
        if abs(unexplained) > GATE_PCT:
            n_above += 1
            if coil == "sol":
                mechanism += (
                    "; UNEXPLAINED collapse discrepancy on the solenoid winding "
                    "representation exceeds the gate — candidate geometry fix"
                )
            else:
                mechanism = (
                    "candidate geometry/merge fix: rectangular-collapse of the "
                    f"{coil} winding pack shifts its per-ampere B_z at the plasma "
                    f"by {unexplained:+.2f}% vs the direct declared-filament sum"
                )
        row["named_mechanism"] = mechanism

        if abs(unexplained) > abs(worst_unexplained):
            worst_coil, worst_unexplained = coil, unexplained
        rows.append(row)

        # signs block: per-ampere sign, measured current, field at that current.
        bz_pa = gate["operator_bz_per_amp"]
        meas_a = float(measured.get(chan, 0.0)) * _KA_TURN_TO_A
        bz_meas = bz_pa * meas_a
        signs[coil] = {
            "bz_per_amp": bz_pa,
            "bz_per_amp_sign": int(np.sign(bz_pa)),
            "measured_current_kA_turn": round(float(measured.get(chan, 0.0)), 4),
            "measured_current_A": meas_a,
            "bz_at_measured_current": bz_meas,
            "bz_at_measured_current_sign": int(np.sign(bz_meas)),
        }

    summary = {
        "n_coils": len(coils),
        "n_above_2pct_excluding_known": n_above,
        "worst_coil": worst_coil,
        "worst_pct": round(worst_unexplained, 4),
        "audit_clean": n_above == 0,
    }

    return {
        "schema": "coil-column-audit-v0",
        "shot": shot_id,
        "coil_model_version": COIL_MODEL_VERSION,
        "solenoid_response_scale": SOLENOID_RESPONSE_SCALE,
        "gate_pct": GATE_PCT,
        "gate_rule": (
            "any |discrepancy| > 2% at the gate point (0.9, 0) is a candidate "
            "mechanism and must be named as a geometry/turns/merge fix, never a "
            "fitted gain; the solenoid-response-scale on the sol column is a "
            "known, vacuum-measured machine-description correction and is "
            "reported as a known mechanism (excluded from the unexplained count)"
        ),
        "eval_points": {
            "gate": list(GATE_POINT),
            "robust_inner": list(ROBUST_INNER),
            "robust_outer": list(ROBUST_OUTER),
        },
        "flat_top_index": flat_idx,
        "flat_top_plasma_current_kA": round(flat_ip_ka, 2),
        "table": rows,
        "signs": signs,
        "summary": summary,
    }


def write_figure(payload: dict[str, object], out_path: Path) -> None:
    """Horizontal-bar figure of per-coil signed % discrepancy at the gate point.

    The ±2% gate band is shaded; the solenoid bar (its discrepancy is the known
    solenoid-response-scale) is drawn in a muted distinct colour and labelled.
    """
    rows: list[dict] = payload["table"]  # type: ignore[assignment]
    coils = [r["coil"] for r in rows]
    disc = [r["by_point"]["gate"]["discrepancy_pct"] for r in rows]

    y = np.arange(len(coils))[::-1]  # sol at top, machine order downward
    colors = ["#b0894f" if c == "sol" else "#3d6b9c" for c in coils]

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.axvspan(-GATE_PCT, GATE_PCT, color="#cfcfcf", alpha=0.35, zorder=0)
    ax.axvline(0.0, color="#555555", lw=0.8, zorder=1)
    ax.barh(y, disc, color=colors, height=0.66, zorder=2)

    ax.set_yticks(y)
    ax.set_yticklabels(coils, fontsize=9)
    ax.set_xlabel(
        "signed discrepancy  (operator − declared sum) / declared sum  [%]",
        fontsize=10,
    )
    ax.set_title(
        "Per-coil vertical-field winding audit at (R,Z)=(0.9,0), "
        f"shot {payload['shot']}",
        fontsize=10.5,
    )

    # annotate the sol known-mechanism bar.
    sol_idx = coils.index("sol")
    ax.annotate(
        f"known: solenoid-response-scale (×{payload['solenoid_response_scale']:.4g})",
        xy=(disc[sol_idx], y[sol_idx]),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
        fontsize=8,
        color="#6b5223",
    )
    ax.text(
        GATE_PCT,
        y[-1] - 0.7,
        "±2% gate",
        fontsize=8,
        color="#555555",
        ha="left",
        va="top",
    )

    xmax = max(3.0, max(abs(d) for d in disc) * 1.25)
    ax.set_xlim(-xmax, xmax)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _print_report(payload: dict[str, object]) -> None:
    rows: list[dict] = payload["table"]  # type: ignore[assignment]
    print(
        f"\n=== per-coil winding audit — shot {payload['shot']} "
        f"({payload['coil_model_version']}) ==="
    )
    hdr = (
        f"{'coil':6s} {'auth Bz/A':>12s} {'op Bz/A':>12s} {'disc%':>8s} "
        f"{'r0.75%':>8s} {'r1.10%':>8s} {'turns':>6s} {'merge':>6s}  mechanism"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        g = r["by_point"]["gate"]
        ri = r["by_point"]["robust_inner"]["discrepancy_pct"]
        ro = r["by_point"]["robust_outer"]["discrepancy_pct"]
        tc = r["turns_cross_check"]
        pf_turns = tc["pfsystems_turns"]
        turns_str = "-" if pf_turns is None else f"{pf_turns:g}"
        print(
            f"{r['coil']:6s} {g['declared_bz_per_amp']:12.4e} "
            f"{g['operator_bz_per_amp']:12.4e} {g['discrepancy_pct']:8.3f} "
            f"{ri:8.3f} {ro:8.3f} {turns_str:>6s} "
            f"{'ok' if r['merge']['clean'] else 'FOLD':>6s}  {r['named_mechanism']}"
        )

    print("\n=== signs block (per-coil B_z at the plasma) ===")
    print(
        f"{'coil':6s} {'Bz/A sign':>10s} {'I_meas [A]':>13s} "
        f"{'Bz@Imeas':>13s} {'sign':>5s}"
    )
    for coil, s in payload["signs"].items():  # type: ignore[union-attr]
        print(
            f"{coil:6s} {s['bz_per_amp_sign']:10d} {s['measured_current_A']:13.2f} "
            f"{s['bz_at_measured_current']:13.4e} {s['bz_at_measured_current_sign']:5d}"
        )

    s = payload["summary"]  # type: ignore[assignment]
    print("\n=== summary ===")
    print(f"  n_coils                        {s['n_coils']}")
    print(f"  n_above_2pct_excluding_known   {s['n_above_2pct_excluding_known']}")
    print(
        f"  worst_coil / worst_pct         {s['worst_coil']} / {s['worst_pct']:+.3f}%"
    )
    print(f"  audit_clean                    {s['audit_clean']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shot", type=int, default=11766, help="campaign shot id")
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "imas_ambix"
        / "gs"
        / "artifacts"
        / "coil_column_audit.json",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "docs"
        / "figures"
        / "equilibrium-realism"
        / "fig-coil-audit.png",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(
        level=args.log_level, format="%(levelname)s %(name)s: %(message)s"
    )

    payload = run_audit(args.shot)

    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    args.artifact.write_text(json.dumps(payload, indent=2))
    logger.info("wrote artifact %s", args.artifact)

    write_figure(payload, args.figure)
    logger.info("wrote figure %s", args.figure)

    _print_report(payload)


if __name__ == "__main__":
    main()
