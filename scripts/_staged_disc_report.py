#!/usr/bin/env python
"""Cohort figure: staged disc boundary read (uniform + gated quadrupole) vs EFIT.

For every held-out shot's ramp-up + flat-top slice, overlay:
  - firewalled EFIT LCFS (red, scoring/plot only)
  - uniform dynamic disc boundary (green)
  - staged read: uniform disc, skip dipole (centroid already fixed by the 2-DOF
    filament fit), quadrupole fitted to the RESIDUAL sensor signature (purple)

Prints the uniform-disc whitened misfit per slice — the conditioning-gate signal that
decides whether the quadrupole stage fires in production.  Visual confirmation
that boundaries are good, not just RMS metrics.
"""

from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scripts.closure_gate_eval as cg
from imas_ambix.latent.boundary_moment import build_moment_basis
from imas_ambix.latent.data import read_split_shot_lists
from imas_ambix.latent.topology import lcfs_contour
from scripts._dynamic_disc_prototype import _rms, dynamic_disc
from scripts.patch_flux_map_report import select_slices

FIGDIR = "docs/figures/th-boundary-robustness"


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


def staged_disc(grid, basis, table, p):
    """Uniform disc + quadrupole-on-residual.  Returns
    (ring_uniform, ring_staged, centroid, misfit0, misfit_quad)."""
    _r, r0, z0, rad = dynamic_disc(grid, basis, table, p)
    cr = grid.flat_r[grid.cells]
    cz = grid.flat_z[grid.cells]
    s = np.hypot(cr - r0, cz - z0) < rad
    ip = float(p.ip_amperes)
    ic0 = np.zeros(grid.cells.size)
    ic0[s] = ip / s.sum()

    m_sens = np.asarray(basis.m_sens.detach().cpu().numpy(), np.float64)
    r_cells = np.asarray(basis.r_cells.detach().cpu().numpy(), np.float64)
    z_cells = np.asarray(basis.z_cells.detach().cpu().numpy(), np.float64)
    keep = np.asarray(p.mask, bool)
    w = np.where(keep, 1.0 / np.maximum(np.asarray(p.scale), 1e-12), 0.0)
    b = np.nan_to_num(np.asarray(p.measured)) - np.nan_to_num(np.asarray(p.vacuum))
    resid = b - m_sens @ ic0
    mis0 = float(np.sum((w * resid)[keep] ** 2) / max(int(keep.sum()), 1))

    # quadrupole stage on the residual (skip dipole: position DOFs already fixed)
    mb, _lab, _sc = build_moment_basis(
        r_cells, z_cells, np.where(s, 1.0, 0.0), r0, order=2, z0=z0
    )
    cols = mb[:, 3:6]  # the three degree-2 zero-sum moments {u^2, uv, v^2}
    a = (m_sens @ cols) * w[:, None]
    cn = np.linalg.norm(a, axis=0)
    cn = np.where(cn > 0, cn, 1.0)
    cfit = (
        np.linalg.solve(
            (a / cn).T @ (a / cn) + 1e-3 * np.eye(3), (a / cn).T @ (resid * w)
        )
        / cn
    )
    resid_q = resid - (m_sens @ cols) @ cfit
    mis_q = float(np.sum((w * resid_q)[keep] ** 2) / max(int(keep.sum()), 1))

    psi0 = np.asarray(basis.psi_grid_2d_np(ic0, p.i_pf), np.float64).reshape(
        grid.nz, grid.nr
    )
    psi_q = np.asarray(
        basis.psi_grid_2d_np(ic0 + cols @ cfit, p.i_pf), np.float64
    ).reshape(grid.nz, grid.nr)
    return (
        _push(psi0, grid, (r0, z0)),
        _push(psi_q, grid, (r0, z0)),
        (r0, z0),
        mis0,
        mis_q,
    )


def main():
    _train, held = read_split_shot_lists(40, 8)
    panels = []
    for shot in held:
        try:
            pay = cg.shot_payloads(
                shot, nr=65, nz=97, max_slices=20, min_ip_ka=300.0, split="eval"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"{shot}: load failed {exc!r}"[:80])
            continue
        grid, basis, table, pls = (
            pay["grid"],
            pay["basis"],
            pay["table"],
            pay["payloads"],
        )
        try:
            picks = select_slices(pls, shot)
        except Exception as exc:  # noqa: BLE001
            print(f"{shot}: efit failed {exc!r}"[:80])
            continue
        for kind, k, efit in picks:
            p = pls[k]
            ring_u, ring_q, ctr, mis0, mis_q = staged_disc(grid, basis, table, p)
            rms_u = _rms(ring_u, efit)
            rms_q = _rms(ring_q, efit)
            panels.append(
                (f"{shot} {kind}", grid, efit, ring_u, ring_q, rms_u, rms_q, ctr)
            )
            print(
                f"{shot} {kind}: uniform={rms_u:.1f}cm staged={rms_q:.1f}cm  "
                f"misfit0={mis0:.3f} misfit_q={mis_q:.3f}"
            )

    ncol = 4
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 4.0 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, (title, grid, efit, ring_u, ring_q, rms_u, rms_q, ctr) in zip(
        axes, panels, strict=False
    ):
        ax.axis("on")
        ax.plot(grid.limiter_r, grid.limiter_z, "k-", lw=0.5)
        ax.set_aspect("equal")
        if efit is not None:
            ax.plot(
                efit["lcfs_r"], efit["lcfs_z"], "r-", lw=2.2, label="EFIT", zorder=5
            )
        if ring_u is not None:
            ax.plot(
                ring_u[:, 0],
                ring_u[:, 1],
                "g-",
                lw=1.2,
                label=f"uniform {rms_u:.0f}cm",
            )
        if ring_q is not None:
            ax.plot(
                ring_q[:, 0],
                ring_q[:, 1],
                "-",
                color="#9467bd",
                lw=1.2,
                label=f"+quad {rms_q:.0f}cm",
            )
        ax.plot([ctr[0]], [ctr[1]], "g+", ms=7)
        ax.set_title(title, fontsize=8)
        ax.legend(fontsize=6, loc="upper right")
    fig.suptitle(
        "Staged disc read: uniform (green) vs +quadrupole-on-residual (purple) "
        "vs firewalled EFIT (red)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = f"{FIGDIR}/staged-disc-cohort.png"
    fig.savefig(out, dpi=110)
    print("wrote", out)


if __name__ == "__main__":
    raise SystemExit(main())
