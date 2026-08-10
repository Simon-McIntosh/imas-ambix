#!/usr/bin/env python
"""Trials: add MAST passive-structure filaments to the staged-disc boundary read.

Hypothesis (lead): the ramp-up boundary residual is a MODELLING limitation, not
a signal defect — during ramp-up the vessel eddy currents are large and the
coil-subtracted "plasma signature" still contains their field.  The plasma
carries Ip, but the higher-order shape signal must be SHARED between the plasma
and the passive structure.  MAST passive geometry (the ``inferred_passive``
circuits in the geometry table) sets the filaments; all couplings use the
finite-area cylinder Biot-Savart kernel.

Each variant ends with the same push-out on the gauged total flux (plasma +
coil + passive) and the same boundary-shift over-fit gate: the production
baseline has no passive term; the staged fit removes passive modes before the
quadrupole fit; the joint fit shares the higher-order signature between both;
and the passive-only ablation omits the quadrupole.

The 80 passive circuits are compressed to their top-k whitened sensor-SVD
modes (k swept) so the read stays well-conditioned.  Scored vs firewalled EFIT
(scoring-only) on the ramp-ups + flat-top no-regress checks.
"""

from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scripts.closure_gate_eval as cg
from imas_ambix.cocos import project_poloidal_field
from imas_ambix.gs import operator as op
from imas_ambix.gs.cylinder import hybrid_greens
from imas_ambix.latent.boundary_disc import (
    DiscReadConfig,
    fit_current_centroid,
    limiter_radial_extent_at_z,
    ring_shift_rms,
    sensor_signature_arrays,
)
from imas_ambix.latent.boundary_moment import build_moment_basis
from imas_ambix.latent.topology import lcfs_contour
from scripts._dynamic_disc_prototype import _rms
from scripts.patch_flux_map_report import select_slices

FIGDIR = "docs/figures/th-boundary-robustness"


def passive_matrices(grid, table):
    """Sensor + grid couplings of every ``inferred_passive`` circuit.

    Returns ``(a_sens (S, P), g_grid (nz*nr, P))`` — per-ampere sensor
    signatures and grid flux columns, finite-area cylinder kernel throughout.
    """
    sr, sz, sang, is_flux = sensor_signature_arrays(table)
    classes = op.classify_circuits(table.pf_filaments, table.amc_current_channels)
    passive_circuits = sorted(
        c.circuit for c in classes if c.role == "inferred_passive"
    )
    by_circ: dict[int, list] = {}
    for f in table.pf_filaments:
        by_circ.setdefault(f.circuit, []).append(f)
    a_cols, g_cols = [], []
    for circ in passive_circuits:
        a = np.zeros(sr.size)
        g = np.zeros(grid.flat_r.size)
        for f in by_circ[circ]:
            w = max(abs(f.width), 0.01)
            h = max(abs(f.height), 0.01)
            psi_s, br_s, bz_s = hybrid_greens(sr, sz, f.r, f.z, w, h)
            a += f.xmult * np.where(
                is_flux, psi_s, project_poloidal_field(br_s, bz_s, sang)
            )
            g += f.xmult * hybrid_greens(grid.flat_r, grid.flat_z, f.r, f.z, w, h)[0]
        a_cols.append(a)
        g_cols.append(g)
    return np.column_stack(a_cols), np.column_stack(g_cols)


def uniform_disc(grid, basis, table, p, cfg):
    """Uniform disc at the robust centroid, self-consistently sized."""
    r0, z0 = fit_current_centroid(p, table, basis, cfg)
    cr = grid.flat_r[grid.cells]
    cz = grid.flat_z[grid.cells]
    rh, rl = limiter_radial_extent_at_z(
        np.asarray(grid.limiter_r), np.asarray(grid.limiter_z), z0
    )
    ip = float(p.ip_amperes)

    def disc(radius):
        s = np.hypot(cr - r0, cz - z0) < radius
        if s.sum() < cfg.min_cells:
            return None, None
        ic = np.zeros(grid.cells.size)
        ic[s] = ip / s.sum()
        return ic, s

    def push(psi):
        lc = lcfs_contour(
            psi,
            grid.rg,
            grid.zg,
            (r0, z0),
            clip_legs=True,
            limiter_r=grid.limiter_r,
            limiter_z=grid.limiter_z,
        )
        return lc

    rad = cfg.rad_init_frac * min(r0 - rh, rl - r0)
    ic = s = None
    for _ in range(cfg.max_radius_iter):
        ic, s = disc(rad)
        if ic is None:
            return None
        psi = np.asarray(basis.psi_grid_2d_np(ic, p.i_pf), np.float64).reshape(
            grid.nz, grid.nr
        )
        lc = push(psi)
        if not lc.found:
            return None
        bmin = float(np.hypot(lc.ring[:, 0] - r0, lc.ring[:, 1] - z0).mean())
        new = 0.5 * rad + 0.5 * bmin
        if abs(new - rad) < cfg.rad_tol:
            rad = new
            break
        rad = new
    ic, s = disc(rad)
    return ic, s, (r0, z0), rad, push


def trial(
    grid,
    basis,
    table,
    p,
    a_pas,
    g_pas,
    *,
    k_pas=8,
    variant="joint_passive",
    cfg=None,
):
    """One passive-augmented read.  Returns (ring, extras) or None."""
    cfg = cfg or DiscReadConfig()
    st = uniform_disc(grid, basis, table, p, cfg)
    if st is None:
        return None
    ic0, s, (r0, z0), rad, push = st
    keep = np.asarray(p.mask, bool)
    w = np.zeros(keep.size)
    w[keep] = 1.0 / np.maximum(np.asarray(p.scale, np.float64)[keep], 1e-12)
    b = np.nan_to_num(np.asarray(p.measured)) - np.nan_to_num(np.asarray(p.vacuum))
    m_sens = np.asarray(basis.m_sens.detach().cpu().numpy(), np.float64)
    resid = b - m_sens @ ic0

    # passive modes: top-k SVD of the whitened passive sensor block
    u, sv, vt = np.linalg.svd(a_pas * w[:, None], full_matrices=False)
    k = min(k_pas, int(np.sum(sv > 1e-10 * sv[0])))
    modes = vt[:k].T  # (P, k): circuit-current patterns of the top-k modes
    a_modes = a_pas @ modes  # (S, k) sensor signature per mode

    # quadrupole columns on the disc
    r_cells = np.asarray(basis.r_cells.detach().cpu().numpy(), np.float64)
    z_cells = np.asarray(basis.z_cells.detach().cpu().numpy(), np.float64)
    mb, _l, _s = build_moment_basis(
        r_cells, z_cells, np.where(s, 1.0, 0.0), r0, order=2, z0=z0
    )
    quad = mb[:, 3:6]
    a_quad = m_sens @ quad

    def ridge_fit(a_fit, rhs, lam):
        aw = a_fit * w[:, None]
        cn = np.linalg.norm(aw, axis=0)
        cn = np.where(cn > 0, cn, 1.0)
        an = aw / cn
        return (
            np.linalg.solve(an.T @ an + lam * np.eye(a_fit.shape[1]), an.T @ (rhs * w))
            / cn
        )

    c_pas = np.zeros(k)
    c_quad = np.zeros(3)
    if variant == "staged_passive":
        c_pas = ridge_fit(a_modes, resid, 1e-3)
        resid2 = resid - a_modes @ c_pas
        c_quad = ridge_fit(a_quad, resid2, 1e-3)
    elif variant == "joint_passive":
        a_joint = np.hstack([a_quad, a_modes])
        c = ridge_fit(a_joint, resid, 1e-3)
        c_quad, c_pas = c[:3], c[3:]
    elif variant == "passive_only":
        c_pas = ridge_fit(a_modes, resid, 1e-3)
    else:
        raise ValueError(f"unknown passive-fit variant {variant!r}")

    i_pas = modes @ c_pas  # per-circuit passive currents [A]
    psi_pas = (g_pas @ i_pas).reshape(grid.nz, grid.nr)

    def total_psi(ic):
        return (
            np.asarray(basis.psi_grid_2d_np(ic, p.i_pf), np.float64).reshape(
                grid.nz, grid.nr
            )
            + psi_pas
        )

    # boundary-shift over-fit gate, referenced to the passive-consistent stage
    lc_base = push(total_psi(ic0))
    if not lc_base.found:
        return None
    lc_full = push(total_psi(ic0 + quad @ c_quad))
    shift = (
        ring_shift_rms(lc_base.ring, lc_full.ring if lc_full.found else None, (r0, z0))
        / rad
    )
    if variant != "passive_only" and lc_full.found and shift < cfg.gate_shift_frac:
        lc, quad_on = lc_full, True
    else:
        lc, quad_on = lc_base, False
    return lc.ring, {
        "quad_on": quad_on,
        "shift": float(shift),
        "i_pas_max": float(np.abs(i_pas).max()) if i_pas.size else 0.0,
    }


def main():
    cases = [
        (12143, "rampup"),
        (12145, "rampup"),
        (12189, "rampup"),
        (12190, "rampup"),
        (18504, "rampup"),
        (18505, "rampup"),
        (18502, "flattop"),
        (12143, "flattop"),
    ]
    by_shot: dict[int, list[str]] = {}
    for shot, kind in cases:
        by_shot.setdefault(shot, []).append(kind)
    panels = []
    for shot, kinds in by_shot.items():
        pay = cg.shot_payloads(
            shot, nr=65, nz=97, max_slices=20, min_ip_ka=300.0, split="eval"
        )
        grid, basis, table, pls = (
            pay["grid"],
            pay["basis"],
            pay["table"],
            pay["payloads"],
        )
        a_pas, g_pas = passive_matrices(grid, table)
        picks = {kd: (k, ef) for kd, k, ef in select_slices(pls, shot)}
        for kind in kinds:
            if kind not in picks:
                continue
            k, efit = picks[kind]
            p = pls[k]
            from imas_ambix.latent.boundary_disc import disc_read

            baseline = disc_read(p, grid, table, basis)
            row = {"baseline": _rms(baseline.ring if baseline else None, efit)}
            rings = {"baseline": baseline.ring if baseline else None}
            for var in ("staged_passive", "joint_passive", "passive_only"):
                out = trial(grid, basis, table, p, a_pas, g_pas, variant=var)
                ring = out[0] if out else None
                row[var] = _rms(ring, efit)
                rings[var] = ring
                if var == "joint_passive" and out:
                    row["quad_on"] = out[1]["quad_on"]
                    row["ipas_kA"] = out[1]["i_pas_max"] / 1e3
            panels.append((f"{shot} {kind}", grid, efit, rings, row))
            print(
                f"{shot} {kind}: baseline={row['baseline']:.1f} "
                f"staged={row['staged_passive']:.1f} "
                f"joint={row['joint_passive']:.1f} "
                f"passive-only={row['passive_only']:.1f}cm "
                f"joint-quad={'ON' if row.get('quad_on') else 'off'} "
                f"max|i_pas|={row.get('ipas_kA', 0):.1f}kA"
            )

    ncol = 4
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 4.0 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    colors = {
        "baseline": "g",
        "staged_passive": "#1f77b4",
        "joint_passive": "#9467bd",
        "passive_only": "#8c564b",
    }
    for ax, (title, grid, efit, rings, row) in zip(axes, panels, strict=False):
        ax.axis("on")
        ax.plot(grid.limiter_r, grid.limiter_z, "k-", lw=0.5)
        ax.set_aspect("equal")
        if efit is not None:
            ax.plot(
                efit["lcfs_r"], efit["lcfs_z"], "r-", lw=2.2, label="EFIT", zorder=5
            )
        for var, ring in rings.items():
            if ring is not None:
                ax.plot(
                    ring[:, 0],
                    ring[:, 1],
                    "-",
                    color=colors[var],
                    lw=1.1,
                    label=f"{var} {row[var]:.0f}cm",
                )
        ax.set_title(title, fontsize=8)
        ax.legend(fontsize=5.5, loc="upper right")
    fig.suptitle(
        "Passive-augmented disc read: baseline, staged passive, joint "
        "quadrupole plus passive, and passive-only — vs EFIT (red)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = f"{FIGDIR}/passive-disc-trials.png"
    fig.savefig(out, dpi=110)
    print("wrote", out)


if __name__ == "__main__":
    raise SystemExit(main())
