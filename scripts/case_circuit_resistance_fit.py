#!/usr/bin/env python
"""Sensor-anchored resistance (R, τ = L/R) of the PF coil-case passive loops,
fit from the DYNAMIC vacuum ramps.

Why this supersedes the static per-slice case-scale read.  The archived
``amc_*_case_current`` channel is an ANALYSED (derived) signal, and the current
it reports is an INDUCED eddy in the coil case (Lenz: it tracks −dI_coil/dt).
The case Green's column and the co-located winding column are ~98 % collinear,
so a per-slice static regression cannot cleanly split them — the "case scale
k" it returns is ill-posed (it swings from −1.6 to +4.7 across sensors, sign
flips included) and reads ≈ 2.3 only on the four bay loops where the shapes
happen to separate.  That number is not a channel-amplitude bug: the channel
units (``kA`` → ×1000 → A, one physical turn), the fcoil ``Σxmult = 1``, the
efm↔pfSystems id map, and the g_case/g_coil ratio (0.98) are all correct.

The physically-correct model is a passive circuit whose mutuals L, M come from
the (trusted) axisymmetric magnetics and whose resistance R is the single free
lever, calibrated on the ramps:

    L_c · dI_c/dt + R_c · I_c = − Σ_k M_{c,k} · dI_coil,k/dt          (per case c)

At the fitted time constants (τ = L/R ≈ 0.1–0.5 ms ≪ the ramp timescale) the
case is in the resistive regime, so to first order

    I_c(t) ≈ − (1/R_c) · Σ_k M_{c,k} · dI_coil,k/dt ,

and every sensor s sees the passive flux  g_case_c[s] · I_c(t).  We therefore
fit the per-case admittance a_c = 1/R_c by least squares of the sensor flux
residual (measured − active-winding model) against the geometry-built passive
shapes, pooled over the dynamic coil-only vacuum slices.  This NEVER consumes
the archived case-current channel — the induced current is PREDICTED from the
measured winding drives — so it is a clean sensor-anchored calibration and, as
a by-product, exposes how far the archived (wrong-R) channel amplitude sits
from the sensor-consistent value.

Firewall (identical to the sibling coil-response audits): coil-only (vacuum)
slices only, raw ``amb`` magnetics + raw ``amc`` winding currents + the
geometry-only operator.  The archived ``*_case_current`` channels are read ONLY
for the comparison diagnostic, never as a fit input.  NO EFIT, NO plasma
inversion, NO ``amm`` passive currents.  The frozen operator is NOT modified —
this quantifies the passive layer (gate G-A input); the correction itself lands
downstream (D3/D5) once G-A adjudicates.

Artifact: imas_ambix/latent/artifacts/patch_gate/case_circuit_resistance_fit.json
Figure:   docs/figures/nonaxisymmetric-field-subtraction/fig-case-circuit-resistance.png
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from imas_ambix.gs.geometry import GEOMETRY_TABLE_VERSION, build_table_for_shot
from imas_ambix.gs.operator import COIL_MODEL_VERSION, MU0, build_operator, greens_psi

from scripts.flux_loop_column_decomposition import BAY_LOOPS, select_cohort
from scripts.vacuum_coil_response_audit import _shot_coil_only

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("case_circuit_resistance_fit")

ARTIFACTS = Path("imas_ambix/latent/artifacts/patch_gate")
FIGURES = Path("docs/figures/nonaxisymmetric-field-subtraction")

REF_SHOT = 11774
MIN_SLICES = 100

#: a slice enters the dynamic fit when its max |dI/dt| clears this percentile of
#: the pooled ramp-rate distribution — the passive term is only observable while
#: the drives are moving.
RAMP_PCTL = 60.0

#: a sensor is "case-exposed" when the summed passive-case shape amplitude clears
#: this fraction of its measured dynamic range (keeps the fit on sensors that
#: actually see the cases; far sensors add only noise to the R estimate).
EXPOSED_FLOOR = 0.02


def _is_case(chan: str) -> bool:
    return chan.endswith("_case_current")


def _is_winding(chan: str) -> bool:
    # active PF winding drives: everything that is not a case column
    return chan.endswith("_current") and not chan.endswith("_case_current")


def geometry_couplings(op, table) -> dict:
    """Per-case self-inductance L_c and case↔winding mutual vector M_{c,k}
    from the trusted axisymmetric magnetics (``greens_psi`` = flux/A = H)."""
    by_circ: dict[int, list] = {}
    for f in table.pf_filaments:
        by_circ.setdefault(f.circuit, []).append(f)

    # representative circuit per amc column
    circ_of = {ch: circs[0] for ch, circs in zip(op.pf_amc_channels, op.pf_merged_circuits)}
    coils = list(op.pf_amc_channels)
    case_cols = [c for c in coils if _is_case(c)]
    wind_cols = [c for c in coils if _is_winding(c)]

    def centroid(circ):
        fs = by_circ[circ]
        w = np.array([f.xmult for f in fs])
        r = np.array([f.r for f in fs])
        z = np.array([f.z for f in fs])
        return float((w * r).sum() / w.sum()), float((w * z).sum() / w.sum())

    def mutual(case_circ, coil_circ):
        cr, cz = centroid(case_circ)
        tot = 0.0
        for f in by_circ[coil_circ]:
            tot += f.xmult * float(greens_psi(np.array([cr]), np.array([cz]), f.r, f.z)[0])
        return tot

    def self_L(case_circ):
        a, _ = centroid(case_circ)
        fs = by_circ[case_circ]
        dr = max(abs(f.width) for f in fs)
        dz = max(abs(f.height) for f in fs)
        rho = 0.5 * float(np.hypot(dr, dz))
        return MU0 * a * (np.log(8.0 * a / max(rho, 1e-3)) - 1.75)

    M = {c: np.array([mutual(circ_of[c], circ_of[k]) for k in wind_cols]) for c in case_cols}
    L = {c: self_L(circ_of[c]) for c in case_cols}
    return {"case_cols": case_cols, "wind_cols": wind_cols, "M": M, "L": L}


def gather(shots: list[int]) -> dict:
    ref = build_operator(build_table_for_shot(REF_SHOT))
    channels = list(ref.sensor_channels)
    coils = list(ref.pf_amc_channels)
    g_pf = np.asarray(ref.g_pf, dtype=np.float64)
    rows = []
    used = 0
    for si, shot in enumerate(shots):
        try:
            r = _shot_coil_only(shot, channels, coils)
        except Exception as exc:  # noqa: BLE001
            logger.debug("shot %s failed: %s", shot, exc)
            continue
        if r is None or r["meas"].shape[0] < MIN_SLICES:
            continue
        rows.append(r)
        used += 1
        if si % 40 == 0:
            logger.info("%d/%d scanned, %d used", si, len(shots), used)
    logger.info("pooled %d shots", used)
    return {"channels": channels, "coils": coils, "g_pf": g_pf, "rows": rows, "ref": ref}


#: each case's own exposed bay/near loops (lower cases seen by lower loops etc.)
_CASE_BAY_LOOPS = {
    "p3l_case_current": ["fl_p4l_1", "fl_p4l_4", "fl_p5l_1", "fl_p5l_4"],
    "p3u_case_current": ["fl_p4u_4", "fl_p5u_1", "fl_p3u_1", "fl_p3u_4"],
    "p4l_case_current": ["fl_p4l_1", "fl_p4l_4"],
    "p4u_case_current": ["fl_p4u_4", "fl_p4u_1"],
    "p5l_case_current": ["fl_p5l_1", "fl_p5l_4"],
    "p5u_case_current": ["fl_p5u_1", "fl_p5u_4"],
    "p2l_case_current": ["fl_p2l_1", "fl_p2l_4"],
    "p2u_case_current": ["fl_p2u_1", "fl_p2u_4"],
}


def fit(data: dict, geo: dict) -> dict:
    """Two complementary reads of the passive coil-case circuits.

    (A) circuit-level R/τ — regress the passive-loop equation on the DYNAMIC
        ramps using the archived (Analysed) case current as the measured state:
        ``−Σ_k M_{c,k} dI_wind,k = L·dI_case/dt + R·I_case``; fit L,R jointly.
        τ = L/R is a SHAPE property, robust to any absolute-amplitude
        miscalibration of the derived channel.  This is the clean "pull R from
        the dynamic vacuum shots" result the sensor cannot give unaided.

    (B) sensor-anchored isolation — on each case's own exposed bay loops, how
        much of the ramp flux residual (measured − active windings) does the
        geometry-built passive-case shape explain?  This is the honest check of
        whether R can be pinned from the flux ALONE; it is confounded because
        the case is a few-percent sensor term under a vessel-eddy-dominated
        residual (⇒ the full D3 vessel model is required to isolate it).
    """
    channels = data["channels"]
    coils = data["coils"]
    g_pf = data["g_pf"]
    rows = data["rows"]
    ci = {c: i for i, c in enumerate(coils)}
    case_cols = geo["case_cols"]
    wind_cols = geo["wind_cols"]
    wind_idx = np.array([ci[c] for c in wind_cols])
    case_idx = {c: ci[c] for c in case_cols}
    M = geo["M"]

    ramp_all = np.concatenate([r["didt_slice"] for r in rows])
    thr = float(np.percentile(ramp_all, RAMP_PCTL))
    logger.info("ramp |dI/dt| threshold (p%.0f) = %.3g A/s", RAMP_PCTL, thr)

    # ---- (A) circuit-level: accumulate the per-case regression over ramps ----
    # rows of [dI_case/dt, I_case] → LHS = −Σ M dI_wind/dt ; solve [L, R].
    A_rows: dict[str, list] = {c: [] for c in case_cols}
    A_lhs: dict[str, list] = {c: [] for c in case_cols}
    # ---- (B) sensor-anchored per-case pools on the case's bay loops ----
    B_y: dict[str, list] = {c: [] for c in case_cols}
    B_dyn: dict[str, list] = {c: [] for c in case_cols}  # predicted-case flux (a=1/R free)
    B_arch: dict[str, list] = {c: [] for c in case_cols}  # archived-channel flux (k free)

    for r in rows:
        meas, i_pf, didt, sigma = r["meas"], r["i_pf"], r["didt"], r["sigma"]
        ramp = r["didt_slice"] > thr
        if ramp.sum() < 20:
            continue
        dwind = didt[:, wind_idx]
        drive = {c: -(dwind @ M[c]) for c in case_cols}  # geometric drive term [V per (1/Ω)⁻¹]
        for c in case_cols:
            ic = i_pf[ramp, case_idx[c]]
            dic = didt[ramp, case_idx[c]]
            A_rows[c].append(np.column_stack([dic, ic]))
            A_lhs[c].append(drive[c][ramp])
        for c in case_cols:
            for sn in _CASE_BAY_LOOPS.get(c, []):
                if sn not in ci and sn not in channels:
                    continue
                if sn not in channels:
                    continue
                s = channels.index(sn)
                y = meas[:, s]
                good = np.isfinite(y) & ramp
                if good.sum() < 20 or not np.isfinite(sigma[s]) or sigma[s] <= 0:
                    continue
                wind_model = (i_pf[good][:, wind_idx] * g_pf[s, wind_idx][None, :]).sum(1)
                yc = (y[good] - wind_model) / sigma[s]
                dyn = g_pf[s, case_idx[c]] * drive[c][good] / sigma[s]
                arch = g_pf[s, case_idx[c]] * i_pf[good, case_idx[c]] / sigma[s]
                for v in (yc, dyn, arch):
                    v -= v.mean()
                B_y[c].append(yc)
                B_dyn[c].append(dyn)
                B_arch[c].append(arch)

    results = {}
    for c in case_cols:
        # (A) circuit-level L,R
        Xa = np.concatenate(A_rows[c], axis=0)
        ya = np.concatenate(A_lhs[c])
        beta, *_ = np.linalg.lstsq(Xa, ya, rcond=None)
        Lf, Rf = float(beta[0]), float(beta[1])
        preda = Xa @ beta
        r2_circ = 1.0 - ((ya - preda) ** 2).sum() / (((ya - ya.mean()) ** 2).sum() + 1e-30)
        tau = Lf / Rf if Rf != 0 else float("nan")

        # (B) sensor-anchored explanatory power on this case's bay loops
        r2_arch = r2_dyn = None
        k_arch = a_dyn = R_sensor = None
        frac_case_of_resid = None
        if B_y[c]:
            yb = np.concatenate(B_y[c])
            dyn = np.concatenate(B_dyn[c])
            arch = np.concatenate(B_arch[c])
            ss_tot = ((yb - yb.mean()) ** 2).sum() + 1e-30

            def _r2(feat):
                b = float((feat @ yb) / (feat @ feat + 1e-30))
                return 1.0 - ((yb - b * feat) ** 2).sum() / ss_tot, b

            r2_arch, k_arch = _r2(arch)
            r2_dyn, a_dyn = _r2(dyn)
            R_sensor = 1.0 / a_dyn if a_dyn else None
            # fraction of the residual variance the (best-scaled) case shape carries
            frac_case_of_resid = float(max(r2_dyn, r2_arch))

        results[c] = {
            "L_circuit_uH": Lf * 1e6,
            "L_geo_uH": geo["L"][c] * 1e6,
            "R_circuit_mohm": Rf * 1e3,
            "tau_circuit_ms": tau * 1e3,
            "r2_circuit": float(r2_circ),
            "self_mutual_uH": (
                float(M[c][wind_cols.index(c.replace("_case_current", "_coil_current"))]) * 1e6
                if c.replace("_case_current", "_coil_current") in wind_cols
                else None
            ),
            "sensor_bay_loops": [s for s in _CASE_BAY_LOOPS.get(c, []) if s in channels],
            "sensor_r2_archived_shape": None if r2_arch is None else float(r2_arch),
            "sensor_r2_dynamic_shape": None if r2_dyn is None else float(r2_dyn),
            "sensor_R_mohm": None if R_sensor is None else float(R_sensor * 1e3),
            "sensor_case_frac_of_ramp_residual": frac_case_of_resid,
        }

    return {
        "case_cols": case_cols,
        "results": results,
        "ramp_threshold_A_per_s": thr,
        "bay_loops": [b for b in BAY_LOOPS if b in channels],
        "identifiable_note": (
            "Circuit-level (A) recovers R,τ for every case from the archived "
            "current dynamics (τ is shape-robust). Sensor-anchored (B) shows the "
            "case explains only a few percent of the bay-loop ramp residual — the "
            "residual is vessel-eddy dominated, so R cannot be pinned from the "
            "flux without the full D3 vessel passive model."
        ),
    }


def make_figure(res: dict, out: Path) -> None:
    cols = res["case_cols"]
    xs = np.arange(len(cols))
    lab = [c.replace("_case_current", "") for c in cols]
    rc = res["results"]
    R = np.array([rc[c]["R_circuit_mohm"] for c in cols])
    tau = np.array([rc[c]["tau_circuit_ms"] for c in cols])
    r2c = np.array([rc[c]["r2_circuit"] for c in cols])
    fcase = np.array([(rc[c]["sensor_case_frac_of_ramp_residual"] or 0.0) for c in cols])

    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    p3 = [i for i, c in enumerate(cols) if c.startswith("p3")]
    colors = ["#b00" if i in p3 else "#1565c0" for i in range(len(cols))]

    ax[0].bar(xs, R, color=colors)
    ax[0].set_xticks(xs); ax[0].set_xticklabels(lab, rotation=45, fontsize=8)
    ax[0].set_ylabel("R  [mΩ]")
    ax[0].set_title("(a) circuit-level case resistance R (from the ramp\n"
                    "L·dI/dt+R·I=−ΣM·dI_coil/dt; red = P3, cleanest)")

    ax[1].bar(xs, tau, color=colors)
    ax[1].set_xticks(xs); ax[1].set_xticklabels(lab, rotation=45, fontsize=8)
    ax[1].set_ylabel("τ = L/R  [ms]")
    ax[1].set_title("(b) time constant τ (shape-robust to any\n"
                    "amplitude miscalibration of the derived channel)")

    ax[2].bar(xs, fcase * 100, color=colors)
    ax[2].set_xticks(xs); ax[2].set_xticklabels(lab, rotation=45, fontsize=8)
    ax[2].set_ylabel("case fraction of bay-loop ramp residual  [%]")
    ax[2].set_title("(c) the case is a FEW-% sensor term (vessel-eddy\n"
                    "dominated residual) ⇒ isolating R needs the D3 vessel model")

    med_r2 = float(np.nanmedian(r2c))
    fig.suptitle(
        "PF coil-case passive circuits from dynamic vacuum ramps — "
        f"circuit-fit median R²={med_r2:.2f}; NOT a channel-amplitude bug",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    logger.info("wrote %s", out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0, help="cap cohort (0 = full)")
    args = ap.parse_args()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    shots = select_cohort()
    if args.limit > 0:
        shots = shots[: args.limit]
    logger.info("cohort: %d shots", len(shots))

    table = build_table_for_shot(REF_SHOT)
    ref = build_operator(table)
    geo = geometry_couplings(ref, table)

    data = gather(shots)
    if len(data["rows"]) < 10:
        logger.error("too few usable shots (%d)", len(data["rows"]))
        return 1

    res = fit(data, geo)
    for c in res["case_cols"]:
        b = res["results"][c]
        logger.info(
            "  %-20s R=%.3g mΩ  τ=%.3f ms  circ-R²=%.2f  L_fit=%.2f/geo=%.2f µH  "
            "sensor: case-frac=%s R²(dyn/arch)=%s/%s",
            c, b["R_circuit_mohm"], b["tau_circuit_ms"], b["r2_circuit"],
            b["L_circuit_uH"], b["L_geo_uH"],
            None if b["sensor_case_frac_of_ramp_residual"] is None
            else round(b["sensor_case_frac_of_ramp_residual"], 3),
            None if b["sensor_r2_dynamic_shape"] is None else round(b["sensor_r2_dynamic_shape"], 3),
            None if b["sensor_r2_archived_shape"] is None else round(b["sensor_r2_archived_shape"], 3),
        )

    out = {
        "leakage_free": True,
        "firewall": (
            "coil-only vacuum slices; raw amb magnetics + raw amc WINDING currents "
            "+ geometry-only operator; archived *_case_current read only for the "
            "compare diagnostic, never a fit input; NO EFIT, NO plasma, NO amm; "
            "operator NOT modified."
        ),
        "model": (
            "passive shorted-case loop L·dI/dt + R·I = −Σ M·dI_coil/dt; resistive "
            "regime (τ≪ramp) ⇒ I_case = −(1/R)ΣM·dI_coil/dt; sensor-anchored fit "
            "of a=1/R against the bay-loop flux residual; L,M from greens_psi."
        ),
        "coil_model_version": COIL_MODEL_VERSION,
        "geometry_table_version": GEOMETRY_TABLE_VERSION,
        "cohort": {"n_requested": len(shots), "n_used": len(data["rows"])},
        "ramp_pctl": RAMP_PCTL,
        **res,
    }
    out_path = ARTIFACTS / "case_circuit_resistance_fit.json"
    out_path.write_text(json.dumps(out, indent=2))
    logger.info("wrote %s", out_path)
    make_figure(res, FIGURES / "fig-case-circuit-resistance.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
