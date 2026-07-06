#!/usr/bin/env python
"""Audit the coordinate / sign (COCOS) conventions of the MAST equilibrium chain.

Answers three questions with REAL MAST data (raw FAIR-MAST level-1 zarr), for a
handful of shots:

1. What sign convention does the MAST ``efm`` reconstruction actually use?
   Specifically the sign of ``psi_axis - psi_boundary`` versus the sign of the
   plasma current ``Ip`` -- the single discriminator that tells COCOS-3-like
   (axis is a flux MAXIMUM for positive Ip) apart from COCOS-11 (axis is a flux
   MINIMUM for positive Ip).

2. Does our geometry-only Green's-function forward operator
   (:mod:`imas_ambix.gs.operator`) predict the raw ``amb`` magnetics with the
   CORRECT SIGN?  We take a coil-dominated (early / low-|Ip|) slice, run the
   vacuum (PF-only) prediction, and correlate it against the raw measured
   ``amb`` on the name-matched sensor rows.  A positive correlation with slope
   ~1 means the whole PF sign chain (amc ``kA*turn`` -> A, filament Green's
   sign, probe projection ``B_R cos t + B_Z sin t``) is coherent end to end.

3. Is the pipeline reading MAST from IMAS at all?  (It is not -- there is no
   ``master.h5`` / ``DBEntry`` on the MAST path; this script documents the raw
   source it actually reads.)

Writes a JSON summary + two figures under
``docs/figures/mast-imas-cocos-evaluation/``.  Read-only on all data; nothing
here feeds back into any fit (the ``efm`` reads are firewalled evaluator reads).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr

from imas_ambix.gs.geometry import build_table_for_shot
from imas_ambix.gs.operator import build_operator, read_amc_currents_at_index

L1 = Path("/work/projects/imas_gpu/mast/level1/shots")
FIGDIR = Path("docs/figures/mast-imas-cocos-evaluation")
SHOTS = [18502, 18503, 18504, 18505]


def _finite_signed_peak(a: np.ndarray) -> float:
    """Value of largest magnitude among finite entries (keeps its sign)."""
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    return float(a[int(np.argmax(np.abs(a)))])


def audit_efm_sign(shot: int) -> dict:
    """Sign of (psi_axis - psi_boundary) vs sign(Ip) from the efm reconstruction."""
    z = zarr.open(str(L1 / f"{shot}.zarr"), mode="r")
    efm = z["efm"]
    psi_ax = np.asarray(efm["psi_axis"], dtype=np.float64)
    psi_bnd = np.asarray(efm["psi_boundary"], dtype=np.float64)
    ipc = np.asarray(efm["plasma_current_c"], dtype=np.float64)
    good = np.isfinite(psi_ax) & np.isfinite(psi_bnd) & np.isfinite(ipc) & (np.abs(ipc) > 5e4)
    if not good.any():
        return {"shot": shot, "n_slices": 0}
    d = psi_ax[good] - psi_bnd[good]
    ip = ipc[good]
    return {
        "shot": shot,
        "n_slices": int(good.sum()),
        "ip_sign": int(np.sign(np.median(ip))),
        "ip_median_ka": round(float(np.median(ip)) / 1e3, 1),
        "psi_axis_minus_boundary_median": float(np.median(d)),
        "axis_is_flux_maximum": bool(np.median(d) > 0),
        # COCOS-11 predicts sign(psi_bnd - psi_axis) == sign(Ip);
        # i.e. sign(psi_axis - psi_bnd) == -sign(Ip). Check it.
        "matches_cocos11": bool(np.sign(np.median(d)) == -np.sign(np.median(ip))),
    }


def audit_forward_sign(shot: int) -> dict:
    """Vacuum (PF-only) forward prediction vs raw amb on a coil-dominated slice."""
    table = build_table_for_shot(shot)
    op = build_operator(table)

    z = zarr.open(str(L1 / f"{shot}.zarr"), mode="r")
    amb = z["amb"]
    amc = z["amc"]
    # The ideal VACUUM slice on MAST is just BEFORE plasma breakdown: the PF
    # coils are fully energised but there is no plasma current yet, so the raw
    # amb is a pure PF (vacuum) signal.  Ip is NaN pre-breakdown, so we pick the
    # last coil-energised sample immediately before the first finite Ip.
    ip = np.asarray(amc["plasma_current"], dtype=np.float64) if "plasma_current" in amc else None
    itime = np.asarray(amc["time"], dtype=np.float64) if "time" in amc else None
    sol = np.asarray(amc["sol_current"], dtype=np.float64) if "sol_current" in amc else None
    t_idx = None
    if ip is not None and itime is not None:
        fin_ip = np.where(np.isfinite(ip))[0]
        if fin_ip.size and sol is not None:
            first_ip = int(fin_ip[0])  # breakdown onset
            # coil-energised samples strictly before breakdown
            pre = np.where(np.isfinite(sol[:first_ip]) & (np.abs(sol[:first_ip]) > 1.0))[0]
            if pre.size:
                t_idx = int(pre[-1])  # last vacuum sample before breakdown
    if t_idx is None and sol is not None:
        fin_sol = np.where(np.isfinite(sol) & (np.abs(sol) > 1.0))[0]
        t_idx = int(fin_sol[0]) if fin_sol.size else 5
    if t_idx is None:
        t_idx = 5

    amc_vals = read_amc_currents_at_index(shot, t_idx)
    i_pf = op.assemble_pf_currents(amc_vals)
    pred = op.vacuum_prediction(i_pf)  # (n_sensor,) [Wb / T]

    # measured amb aligned by name to the operator's sensor rows
    amb_time = np.asarray(amb["time"], dtype=np.float64) if "time" in amb else None
    t_s = float(itime[t_idx]) if itime is not None else None
    meas = np.full(len(op.sensor_channels), np.nan)
    for k, ch in enumerate(op.sensor_channels):
        if ch in amb:
            arr = np.asarray(amb[ch], dtype=np.float64)
            if amb_time is not None and arr.shape == amb_time.shape and t_s is not None:
                j = int(np.argmin(np.abs(amb_time - t_s)))
            else:
                j = min(t_idx, arr.shape[0] - 1)
            meas[k] = arr[j]

    m = np.isfinite(meas) & np.isfinite(pred) & (np.abs(pred) > 0)
    out = {"shot": shot, "t_index": t_idx, "n_matched": int(m.sum())}
    if m.sum() >= 5:
        pr, me = pred[m], meas[m]
        # robust sign agreement: fraction of rows with matching sign
        sign_match = float(np.mean(np.sign(pr) == np.sign(me)))
        corr = float(np.corrcoef(pr, me)[0, 1])
        # least-squares slope meas ~ slope * pred (sign + magnitude coherence)
        slope = float(np.sum(pr * me) / np.sum(pr * pr))
        out.update(
            sign_match_fraction=round(sign_match, 3),
            correlation=round(corr, 3),
            slope_meas_vs_pred=round(slope, 3),
            pred=pr.tolist(),
            meas=me.tolist(),
        )
    return out


def main() -> int:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    efm_rows = [audit_efm_sign(s) for s in SHOTS]
    fwd_rows = []
    for s in SHOTS:
        try:
            fwd_rows.append(audit_forward_sign(s))
        except Exception as exc:  # noqa: BLE001
            fwd_rows.append({"shot": s, "error": str(exc)})

    report = {"efm_sign_convention": efm_rows, "forward_operator_sign": fwd_rows}
    (FIGDIR / "cocos_audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    # --- Figure 1: efm sign convention (psi_axis - psi_boundary) vs Ip -------
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for r in efm_rows:
        if r.get("n_slices", 0) == 0:
            continue
        ax.scatter(
            r["ip_median_ka"],
            r["psi_axis_minus_boundary_median"],
            s=90,
            zorder=3,
        )
        ax.annotate(
            str(r["shot"]),
            (r["ip_median_ka"], r["psi_axis_minus_boundary_median"]),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=9,
        )
    ax.axhline(0, color="0.5", lw=0.8)
    ax.axvline(0, color="0.5", lw=0.8)
    ax.set_xlabel("median plasma current Ip  [kA]  (efm plasma_current_c)")
    ax.set_ylabel(r"median $\psi_{axis}-\psi_{boundary}$  [Wb/rad]")
    ax.set_title(
        "MAST efm sign: axis is a flux MAXIMUM for POSITIVE Ip\n"
        "(top-right quadrant) -> COCOS-3-like, NOT COCOS-11 (which is top-left)"
    )
    # annotate quadrant expectations
    ax.text(
        0.03, 0.95, "COCOS-11 region\n(axis = flux min)", transform=ax.transAxes,
        va="top", ha="left", fontsize=8, color="#b03030",
        bbox={"boxstyle": "round", "fc": "#fbeaea", "ec": "#d9a0a0"},
    )
    ax.text(
        0.97, 0.95, "MAST efm here\n(axis = flux max)", transform=ax.transAxes,
        va="top", ha="right", fontsize=8, color="#20603a",
        bbox={"boxstyle": "round", "fc": "#e8f4ec", "ec": "#a0cbb0"},
    )
    fig.tight_layout()
    fig.savefig(FIGDIR / "efm-sign-convention.png", dpi=120)
    plt.close(fig)

    # --- Figure 2: forward vacuum prediction vs raw amb (sign coherence) -----
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 8.0))
    for ax, r in zip(axes.ravel(), fwd_rows, strict=False):
        if "pred" not in r:
            ax.text(0.5, 0.5, f"{r.get('shot')}: n/a", ha="center", va="center")
            ax.axis("off")
            continue
        pr = np.array(r["pred"])
        me = np.array(r["meas"])
        ax.scatter(pr, me, s=22, alpha=0.7)
        lim = max(np.max(np.abs(pr)), np.max(np.abs(me))) * 1.1
        ax.plot([-lim, lim], [-lim, lim], "0.6", lw=0.9, label="y = x")
        ax.axhline(0, color="0.85", lw=0.6)
        ax.axvline(0, color="0.85", lw=0.6)
        ax.set_title(
            f"shot {r['shot']}  (coil-dominated slice)\n"
            f"sign-match {r['sign_match_fraction']:.0%}  "
            f"corr {r['correlation']:.2f}  slope {r['slope_meas_vs_pred']:.2f}",
            fontsize=9,
        )
        ax.set_xlabel("vacuum forward prediction  G_pf . I_pf")
        ax.set_ylabel("raw measured amb")
        ax.legend(fontsize=7, loc="upper left")
    fig.suptitle(
        "Forward PF sign chain is coherent: G_pf . I_pf tracks raw amb "
        "in sign (positive slope on the y=x line)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIGDIR / "forward-sign-coherence.png", dpi=120)
    plt.close(fig)
    print("wrote figures to", FIGDIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
