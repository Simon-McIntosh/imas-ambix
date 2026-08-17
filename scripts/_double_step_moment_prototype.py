#!/usr/bin/env python
"""Prototype: two-pass (double-step) moment boundary read.

The single-pass moment read lets current fill the WHOLE in-limiter candidate set,
so current spills outside the true LCFS and the external field is that of an
OVER-SIZED plasma (confirmed: the moment boundary is systematically larger than
EFIT).  Two-pass fixes it self-consistently:

  pass 1  fill the vessel (all candidate cells) -> fit moments -> psi -> LCFS_1
  mask    keep only candidate cells INSIDE LCFS_1  (j = 0 outside the plasma)
  pass 2  refit the moments on the masked cell set -> psi -> LCFS_2

The cell->grid / cell->sensor interaction matrices are precomputed for ALL cells,
so restricting to the inside-LCFS_1 cells is a COLUMN MASK — no Green's-function
recompute.  Optionally iterate; here one extra pass.  Compares pass-1 vs pass-2
vs firewalled EFIT (scoring-only).
"""

from __future__ import annotations

import matplotlib
import numpy as np
from matplotlib.path import Path as MplPath

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scripts.closure_gate_eval as cg
from imas_ambix.latent import boundary_moment as bm
from imas_ambix.latent.boundary_moment import (
    MomentFitConfig,
    build_moment_basis,
    fit_moment_currents,
)
from imas_ambix.latent.topology import lcfs_contour
from scripts.patch_flux_map_report import select_slices


def _push(psi, grid, ctr):
    lc = lcfs_contour(
        psi,
        grid.rg,
        grid.zg,
        ctr,
        clip_legs=True,
        limiter_r=grid.limiter_r,
        limiter_z=grid.limiter_z,
    )
    return lc.ring if lc.found else None


def _rms(ring, efit):
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


def double_step(grid, basis, p, order=3, n_extra=1):
    """Return [(ring, misfit)] for pass 1 .. pass 1+n_extra (masked refits)."""
    r_cells = np.asarray(basis.r_cells.detach().cpu().numpy(), np.float64)
    z_cells = np.asarray(basis.z_cells.detach().cpu().numpy(), np.float64)
    cand = np.asarray(basis.candidate_mask.detach().cpu().numpy(), np.float64)
    m_sens = np.asarray(basis.m_sens.detach().cpu().numpy(), np.float64)
    r0 = float(basis.r0)
    cfg = MomentFitConfig(order=order)
    cell_rz = np.column_stack([r_cells, z_cells])

    out = []
    # pass 1 — fill the vessel (the project's stock single-pass read)
    mom = fit_moment_currents(basis, p, cfg)
    ctr = (float(mom.centroid_r), float(mom.centroid_z))
    psi = np.asarray(basis.psi_grid_2d_np(mom.i_cell, p.i_pf), np.float64).reshape(
        grid.nz, grid.nr
    )
    ring = _push(psi, grid, ctr)
    out.append((ring, float(mom.misfit)))

    mask = cand.copy()
    for _ in range(n_extra):
        if ring is None or ring.shape[0] < 4:
            break
        # keep candidate cells INSIDE the current LCFS (j = 0 outside)
        inside = MplPath(ring).contains_points(cell_rz)
        mask = np.where((cand > 0) & inside, 1.0, 0.0)
        if mask.sum() < (order + 1) * (order + 2) / 2:  # too few cells to fit
            break
        m_basis, _lab, _sc = build_moment_basis(  # noqa: N806
            r_cells, z_cells, mask, r0, order=order
        )
        a_sens = m_sens @ m_basis  # COLUMN-MASKED — no Green's recompute
        # the confined moment basis is near-collinear (collinear monomials on a
        # smaller region) — an un-regularised refit blows up into a wildly
        # oscillating, hollow/reversed current that destroys the plasma O-point.
        # escalate the ridge until the refit current is essentially single-signed
        # (a physical peaked plasma), the smallest regularisation that keeps it so.
        c = i_cell = None
        for ridge in (1e-6, 1e-4, 1e-2, 1e-1, 1e0, 1e1, 1e2):
            cfg_r = MomentFitConfig(order=order, ridge=ridge)
            c, mis, _cov = bm._fit_one(
                m_basis,
                a_sens,
                p.measured,
                p.vacuum,
                p.mask,
                p.scale,
                float(p.ip_amperes),
                cfg_r,
            )
            i_cell = m_basis @ c
            frac_neg = float(np.mean(i_cell[mask > 0] < 0.0))
            if frac_neg < 0.05:  # single-signed -> physical plasma current
                break
        psi = np.asarray(basis.psi_grid_2d_np(i_cell, p.i_pf), np.float64).reshape(
            grid.nz, grid.nr
        )
        # push from the plasma O-point (psi MAX inside the confined region) — the
        # current centroid is not the flux maximum, and lcfs_contour sweeps from
        # psi(origin), so a non-O-point origin corrupts the sweep
        wsum = float(i_cell.sum()) or 1.0
        cen = (
            float((i_cell * r_cells).sum() / wsum),
            float((i_cell * z_cells).sum() / wsum),
        )
        rr, zz = np.meshgrid(grid.rg, grid.zg)
        near = (np.hypot(rr - cen[0], zz - cen[1]) < 0.5) & MplPath(
            ring
        ).contains_points(np.column_stack([rr.ravel(), zz.ravel()])).reshape(
            grid.nz, grid.nr
        )
        psi_o = np.where(near, psi, -np.inf)
        io = np.unravel_index(int(np.argmax(psi_o)), psi.shape)
        ctr = (float(grid.rg[io[1]]), float(grid.zg[io[0]]))
        ring = _push(psi, grid, ctr)
        out.append((ring, float(mis)))
    return out


def main():
    slices = [
        (18502, "flattop"),
        (12143, "rampup"),
        (18505, "rampup"),
        (12189, "rampup"),
        (12143, "flattop"),
        (12190, "rampup"),
    ]
    # group by shot to load each payload once
    by_shot: dict[int, list[str]] = {}
    for shot, kind in slices:
        by_shot.setdefault(shot, []).append(kind)

    panels = []
    for shot, kinds in by_shot.items():
        pay = cg.shot_payloads(
            shot, nr=65, nz=97, max_slices=20, min_ip_ka=300.0, split="eval"
        )
        grid, basis, pls = pay["grid"], pay["basis"], pay["payloads"]
        picks = select_slices(pls, shot)
        by_kind = {kd: (k, ef) for kd, k, ef in picks}
        for kind in kinds:
            if kind not in by_kind:
                print(f"{shot} {kind}: no efit-snapped slice")
                continue
            k, efit = by_kind[kind]
            passes = double_step(grid, basis, pls[k], n_extra=1)
            rms = [100.0 * _rms(r, efit) for r, _m in passes]
            panels.append((f"{shot} {kind}", grid, efit, passes, rms))
            print(
                f"{shot} {kind}: RMS[cm] per pass = "
                + " -> ".join(f"{x:.1f}" for x in rms)
            )

    ncol = 3
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 4.2 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    colors = ["#1f77b4", "#2ca02c", "#9467bd"]
    for ax, (title, grid, efit, passes, rms) in zip(axes, panels, strict=False):
        ax.axis("on")
        ax.plot(grid.limiter_r, grid.limiter_z, "k-", lw=0.5)
        ax.set_aspect("equal")
        if efit is not None:
            ax.plot(
                efit["lcfs_r"], efit["lcfs_z"], "r-", lw=2.4, label="EFIT", zorder=6
            )
        for i, (ring, _m) in enumerate(passes):
            if ring is not None:
                ax.plot(
                    ring[:, 0],
                    ring[:, 1],
                    "-",
                    color=colors[i % 3],
                    lw=1.3,
                    label=f"pass{i + 1} {rms[i]:.0f}cm",
                )
        ax.set_title(title, fontsize=8)
        ax.legend(fontsize=6, loc="upper right")
    fig.suptitle(
        "Double-step moment read: pass 1 (fill vessel) -> pass 2/3 (j=0 outside "
        "LCFS, refit) vs EFIT (red)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = "docs/figures/th-boundary-robustness/double-step-moment.png"
    fig.savefig(out, dpi=115)
    print("wrote", out)


if __name__ == "__main__":
    raise SystemExit(main())
