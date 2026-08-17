#!/usr/bin/env python
"""Cohort comparison: moment vs toroidal-harmonic boundary, both via the push-out
reader, overlaid on the firewalled EFIT LCFS (scoring-only).

For each held-out shot's ramp-up and flat-top slice, reconstruct psi two ways —
the source-free TH read (``_harmonic_read_for_slice``) and the confined
low-order current-moment read (``fit_moment_currents`` -> ``psi_grid_2d_np``) —
read the LCFS off each with the SAME push-out (``lcfs_contour``, leg-clipped),
and overlay EFIT.  Reports the RMS LCFS radius error vs EFIT per read.  EFIT is
firewalled (scoring/plot only, never an input).
"""

from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scripts.closure_gate_eval as cg
from imas_ambix.latent.boundary_moment import MomentFitConfig, fit_moment_currents
from imas_ambix.latent.data import read_split_shot_lists
from imas_ambix.latent.topology import lcfs_contour
from scripts.closure_gate_eval import _harmonic_read_for_slice, load_frozen_lookup
from scripts.patch_flux_map_report import select_slices

FIGDIR = "docs/figures/th-boundary-robustness"
FROZEN = "imas_ambix/latent/artifacts/patch_gate/harmonic_prior_frozen.json"


def _rms_vs_efit(ring, efit):
    if ring is None or efit is None:
        return float("nan")
    ec = np.array([efit["lcfs_r"].mean(), efit["lcfs_z"].mean()])
    the = np.arctan2(efit["lcfs_z"] - ec[1], efit["lcfs_r"] - ec[0])
    re = np.hypot(efit["lcfs_r"] - ec[0], efit["lcfs_z"] - ec[1])
    thr = np.arctan2(ring[:, 1] - ec[1], ring[:, 0] - ec[0])
    rr = np.hypot(ring[:, 0] - ec[0], ring[:, 1] - ec[1])
    o = np.argsort(thr)
    ri = np.interp(the, thr[o], rr[o], period=2 * np.pi)
    return float(np.sqrt(np.mean((ri - re) ** 2)))


def _moment_ring(grid, basis, p):
    mom = fit_moment_currents(basis, p, MomentFitConfig(order=3))
    psi = np.asarray(basis.psi_grid_2d_np(mom.i_cell, p.i_pf), np.float64)
    psi = psi.reshape(grid.nz, grid.nr)
    ctr = (float(mom.centroid_r), float(mom.centroid_z))
    lc = lcfs_contour(
        psi,
        grid.rg,
        grid.zg,
        ctr,
        clip_legs=True,
        limiter_r=grid.limiter_r,
        limiter_z=grid.limiter_z,
    )
    return (lc.ring if lc.found else None), float(mom.misfit)


def main():
    _lookup, meta = load_frozen_lookup(FROZEN)
    _train, held = read_split_shot_lists(40, 8)
    panels = []  # (title, efit, moment_ring, th_ring, rms_m, rms_t)
    for shot in held:
        try:
            pay = cg.shot_payloads(
                shot, nr=65, nz=97, max_slices=20, min_ip_ka=300.0, split="eval"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"{shot}: load failed {exc!r}"[:90])
            continue
        grid, table, basis, pls = (
            pay["grid"],
            pay["table"],
            pay["basis"],
            pay["payloads"],
        )
        try:
            picks = select_slices(pls, shot)
        except Exception as exc:  # noqa: BLE001
            print(f"{shot}: efit select failed {exc!r}"[:90])
            picks = []
        efit_by_k = {k: ef for _kd, k, ef in picks}
        kinds_by_k = {k: kd for kd, k, _ef in picks}
        if not picks:  # still show reads without EFIT — pick flat-top + earliest
            ip = np.array([abs(p.ip_amperes) for p in pls])
            picks_k = [int(np.argmax(ip)), 0]
        else:
            picks_k = list(efit_by_k)
        for k in picks_k:
            p = pls[k]
            efit = efit_by_k.get(k)
            mring, mmis = _moment_ring(grid, basis, p)
            th = _harmonic_read_for_slice(p, grid, table, basis, meta or {})
            tring = th[6] if th is not None else None
            rms_m = 100.0 * _rms_vs_efit(mring, efit)
            rms_t = 100.0 * _rms_vs_efit(tring, efit)
            kind = kinds_by_k.get(k, "flattop")
            panels.append((f"{shot} {kind}", grid, efit, mring, tring, rms_m, rms_t))
            th_mis = th[2] if th else float("nan")
            print(
                f"{shot} {kind}: RMS[cm] moment={rms_m:.1f} TH={rms_t:.1f} "
                f"(moment misfit {mmis:.2f}, TH misfit {th_mis:.2f})"
            )

    ncol = 4
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 4.0 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, (title, grid, efit, mring, tring, rms_m, rms_t) in zip(
        axes, panels, strict=False
    ):
        ax.axis("on")
        ax.plot(grid.limiter_r, grid.limiter_z, "k-", lw=0.5)
        ax.set_aspect("equal")
        if efit is not None:
            ax.plot(
                efit["lcfs_r"], efit["lcfs_z"], "r-", lw=2.2, label="EFIT", zorder=5
            )
        if mring is not None:
            ax.plot(
                mring[:, 0], mring[:, 1], "b-", lw=1.2, label=f"moment {rms_m:.0f}cm"
            )
        if tring is not None:
            ax.plot(
                tring[:, 0],
                tring[:, 1],
                "-",
                color="orange",
                lw=1.2,
                label=f"TH {rms_t:.0f}cm",
            )
        ax.set_title(title, fontsize=8)
        ax.legend(fontsize=6, loc="upper right")
    fig.suptitle(
        "Moment (blue) vs toroidal-harmonic (orange) boundary — push-out reader, "
        "vs firewalled EFIT (red)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = f"{FIGDIR}/moment-vs-th-cohort.png"
    fig.savefig(out, dpi=110)
    print("wrote", out)


if __name__ == "__main__":
    raise SystemExit(main())
