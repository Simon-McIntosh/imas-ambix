#!/usr/bin/env python
"""Prototype: combined homogeneous-GS boundary read = moment (current) basis
+ source-free toroidal-harmonic (TH) basis, joint-fit in one whitened LSQ.

Both are Delta*-harmonic in the vacuum region, so their sum is too.  The
low-order moment current carries the smooth centred "fat blob" of the ST (where
a single TH focal ring conditions badly); the TH n>=1 harmonics carry the
annulus/edge structure (where TH is best-in-class).  Ip is pinned on the moment
monopole; the redundant TH n=0 column is dropped; a graded Sobolev ridge damps
high TH modes; column-normalised ridge keeps the near-redundant joint matrix
stable.  The reconstructed psi is ABSOLUTELY GAUGED —
``psi = psi_grid_2d_np(M@c_moment, i_pf)`` (moment current + coil) + the
source-free TH columns — so the push-out recovers the boundary flux.

Untracked helper; renders combined vs moment-only vs TH-only for a
poorly-conditioned slice (12143 ramp-up) and a well-conditioned one (18502
flat-top).
"""

from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from boundary_harmonic_gate_eval import (
    _adaptive_radii,
    hybrid_target_harmonic,
    sensor_arrays,
)

import scripts.closure_gate_eval as cg
from imas_ambix.latent.boundary_harmonic import (
    HarmonicFitConfig,
    harmonic_columns,
    harmonic_mode_penalty,
    harmonic_sensor_matrix,
)
from imas_ambix.latent.boundary_moment import MomentFitConfig, build_moment_basis
from imas_ambix.latent.topology import lcfs_contour


def _payload(shot, split, regime):
    pay = cg.shot_payloads(
        shot, nr=65, nz=97, max_slices=20, min_ip_ka=300.0, split=split
    )
    ips = np.array([abs(p.ip_amperes) for p in pay["payloads"]])
    k = (
        int(np.argmax(ips))
        if regime != "rampup"
        else int(np.argmin(np.where(ips > 3e5, ips, np.inf)))
    )
    return pay["grid"], pay["table"], pay["basis"], pay["payloads"][k]


def _moment_sensor(basis, p, order):
    r_cells = np.asarray(basis.r_cells.detach().cpu().numpy(), np.float64)
    z_cells = np.asarray(basis.z_cells.detach().cpu().numpy(), np.float64)
    cand = np.asarray(basis.candidate_mask.detach().cpu().numpy(), np.float64)
    m_sens = np.asarray(basis.m_sens.detach().cpu().numpy(), np.float64)
    M, _lab, _sc = build_moment_basis(  # noqa: N806
        r_cells, z_cells, cand, float(basis.r0), order=order
    )
    return M, m_sens @ M  # M (n_cells,Km), A_moment (S,Km)


def combined_read(
    grid,
    table,
    basis,
    p,
    *,
    mom_order=3,
    th_order=4,
    pole=None,
    sobolev_p=1.0,
    ridge=1e-6,
):
    """Joint moment+TH read → gauged psi + boundary ring."""
    sr, sz, sang, is_flux = sensor_arrays(table)
    ctr = None
    # current centroid = moment monopole/centroid; use it for the TH pole too
    from imas_ambix.latent.boundary_moment import fit_moment_currents  # noqa: PLC0415

    mom = fit_moment_currents(basis, p, MomentFitConfig(order=mom_order))
    ctr = (float(mom.centroid_r), float(mom.centroid_z))
    if pole is None:
        # offset the TH focal ring inboard of the centroid so |origin-pole| > 0
        # keeps the near-pole mask SMALL (the d=0 fallback is a fixed 0.25 m disk
        # that would swallow the plasma centre); the moment part carries the
        # centred blob, so the TH pole need not sit at the centre.
        pole = (ctr[0] * (1.0 - 0.41), ctr[1])
    cfg = HarmonicFitConfig(
        pole_r=pole[0],
        pole_z=pole[1],
        order=th_order,
        ridge=ridge,
        sobolev_p=sobolev_p,
    )

    M, a_mom = _moment_sensor(basis, p, mom_order)  # noqa: N806
    a_th = harmonic_sensor_matrix(sr, sz, sang, is_flux, cfg)  # (S,Kth)

    # whitening + Ip anchor on the moment monopole (col 0)
    meas = np.nan_to_num(np.asarray(p.measured, np.float64))
    vac = np.nan_to_num(np.asarray(p.vacuum, np.float64))
    keep = np.asarray(p.mask, bool)
    sc = np.asarray(p.scale, np.float64)
    w = np.zeros_like(meas)
    w[keep] = 1.0 / np.maximum(sc[keep], 1e-12)
    n_cand = float(M[:, 0].sum())
    c0 = float(p.ip_amperes) / max(n_cand, 1.0)
    b = meas - vac - a_mom[:, 0] * c0  # pin Ip on moment monopole

    # fit block: moment SHAPE moments (drop col 0) + TH n>=1 (drop TH monopole)
    a_fit = np.hstack([a_mom[:, 1:], a_th[:, 1:]])
    k_mom = a_mom.shape[1] - 1
    # graded penalty: flat on moment shape moments, (1+deg)^p on TH modes
    th_pen = harmonic_mode_penalty(th_order, sobolev_p)[1:]
    pen = np.concatenate([np.ones(k_mom), th_pen])

    aw = a_fit * w[:, None]
    bw = b * w
    col_norm = np.linalg.norm(aw, axis=0)
    col_norm = np.where(col_norm > 0, col_norm, 1.0)
    a_n = aw / col_norm[None, :]
    gram = a_n.T @ a_n + ridge * np.diag(pen)
    c_fit = np.linalg.solve(gram, a_n.T @ bw) / col_norm
    resid = (vac + a_mom[:, :1] @ [c0] + a_fit @ c_fit - meas) * w
    misfit = float((resid[keep] ** 2).sum() / max(int(keep.sum()), 1))

    c_mom = np.concatenate([[c0], c_fit[:k_mom]])
    c_th = c_fit[k_mom:]  # TH n>=1 coefficients

    # --- reconstruct ABSOLUTELY GAUGED psi -----------------------------------
    i_cell = M @ c_mom
    psi_gauged = np.asarray(basis.psi_grid_2d_np(i_cell, p.i_pf), np.float64)
    if psi_gauged.shape != (grid.nz, grid.nr):
        psi_gauged = psi_gauged.reshape(grid.nz, grid.nr)
    rr, zz = np.meshgrid(grid.rg, grid.zg)
    th_cols, _ = harmonic_columns(rr.ravel(), zz.ravel(), cfg)  # (N, Kth)
    psi_th = (th_cols[:, 1:] @ c_th).reshape(grid.nz, grid.nr)  # source-free n>=1
    psi_tot = psi_gauged + psi_th

    mask_r, excl_r = _adaptive_radii(ctr, pole, _R())
    _t, psi_ax, psi_b, field, _d = hybrid_target_harmonic(
        psi_tot, grid, ctr, pole, mask_r, excl_r, clip_legs=True
    )
    lc = lcfs_contour(
        field,
        grid.rg,
        grid.zg,
        ctr,
        clip_legs=True,
        limiter_r=grid.limiter_r,
        limiter_z=grid.limiter_z,
    )
    ring = lc.ring if lc.found else None
    return ring, float(misfit), float(psi_b), ctr, pole


class _R:
    mask_frac = 0.5
    exclude_frac = 1.1
    mask_radius = 0.25
    exclude_radius = 0.55


def main():
    slices = [
        (18502, "eval", "flattop"),
        (12143, "eval", "rampup"),
        (11767, "train", None),
    ]
    fig, axs = plt.subplots(1, len(slices), figsize=(3.2 * len(slices), 4.2))
    for ax, (shot, split, reg) in zip(axs, slices, strict=False):
        grid, table, basis, p = _payload(shot, split, reg)
        ring, mis, psib, ctr, pole = combined_read(grid, table, basis, p)
        ax.plot(grid.limiter_r, grid.limiter_z, "k-", lw=0.6)
        ax.plot([ctr[0]], [ctr[1]], "m+", ms=9)
        ax.set_aspect("equal")
        if ring is not None:
            ax.plot(ring[:, 0], ring[:, 1], "g-", lw=1.4)
            s = f"Rin={ring[:, 0].min():.3f} Rout={ring[:, 0].max():.3f}"
        else:
            s = "None"
        ax.set_title(f"{shot} {reg or 'train'}\ncombined mis={mis:.2f}", fontsize=9)
        print(f"{shot} {reg}: combined misfit={mis:.3f} psi_b={psib:.3e} {s}")
    fig.suptitle(
        "Combined moment + TH read (gauged) — poorly- vs well-conditioned", fontsize=11
    )
    fig.tight_layout()
    out = (
        "/run/user/39486/claude-39486/-home-ITER-mcintos-Code-imas-ambix/"
        "70fed06e-29b3-4204-8860-4ce583d5a0d8/scratchpad/combined_basis.png"
    )
    fig.savefig(out, dpi=115)
    print("wrote", out)


if __name__ == "__main__":
    raise SystemExit(main())
