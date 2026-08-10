#!/usr/bin/env python
"""Prototype: dynamic-sizing uniform-current disc boundary read.

The cell-current moment read is ill-conditioned (its centroid swings 0.3-0.7 m
across order); TH collapses on poorly-conditioned ramp-ups.  This read drops both
in favour of the simplest physical current consistent with the magnetics:

  1. Ip           — Rogowski (pinned).
  2. centroid     — a well-conditioned 2-DOF filament-position fit to the
                    coil-subtracted sensors (B-probes AND flux loops).
  3. disc radius  — self-consistently sized so the disc's own push-out boundary
                    minor radius is a fixed point (DYNAMIC sizing).
  4. current      — UNIFORM Ip over the disc (no shape moments needed — the
                    plasma elongation is supplied by the real coil field shaping
                    the psi_N=1 contour around the disc).
  5. boundary     — push-out of (disc psi + coil psi).

Scored against firewalled EFIT (scoring-only) over the held-out cohort, head to
head with the single-pass cell-current moment read.
"""

from __future__ import annotations

import matplotlib
import numpy as np
from scipy.optimize import minimize

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from boundary_harmonic_gate_eval import sensor_arrays

import scripts.closure_gate_eval as cg
from imas_ambix.cocos import project_poloidal_field
from imas_ambix.gs.cylinder import hybrid_greens
from imas_ambix.latent.boundary_moment import MomentFitConfig, fit_moment_currents
from imas_ambix.latent.data import read_split_shot_lists
from imas_ambix.latent.topology import lcfs_contour
from scripts.patch_flux_map_report import select_slices


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
    return 100.0 * float(np.sqrt(np.mean((ri - re) ** 2)))


def _fsens(sr, sz, sang, isf, rc, zc):
    psi, br, bz = hybrid_greens(sr, sz, rc, zc, 0.05, 0.05)
    return np.where(isf, psi, project_poloidal_field(br, bz, sang))


def _lim_at_z(lr, lz, z0):
    rs = []
    for i in range(len(lr)):
        a, b = lz[i], lz[(i + 1) % len(lr)]
        r0, r1 = lr[i], lr[(i + 1) % len(lr)]
        if (a - z0) * (b - z0) <= 0 and a != b:
            rs.append(r0 + (z0 - a) / (b - a) * (r1 - r0))
    return (min(rs), max(rs)) if rs else (float(lr.min()), float(lr.max()))


def dynamic_disc(grid, basis, table, p):
    """Return (ring, R0, Z0, radius) for the dynamic-sizing uniform-disc read."""
    sr, sz, sang, isf = sensor_arrays(table)
    w = np.where(
        np.asarray(p.mask, bool), 1.0 / np.maximum(np.asarray(p.scale), 1e-12), 0.0
    )
    b = np.nan_to_num(np.asarray(p.measured)) - np.nan_to_num(np.asarray(p.vacuum))
    ip = float(p.ip_amperes)
    m1 = fit_moment_currents(basis, p, MomentFitConfig(order=1))
    res = minimize(
        lambda x: float(
            np.sum((w * (ip * _fsens(sr, sz, sang, isf, x[0], x[1]) - b)) ** 2)
        ),  # noqa: E501
        [float(m1.centroid_r), float(m1.centroid_z)],
        method="Nelder-Mead",
        options={"xatol": 1e-3, "fatol": 1e-6},
    )
    r0, z0 = float(res.x[0]), float(res.x[1])
    cr = grid.flat_r[grid.cells]
    cz = grid.flat_z[grid.cells]
    rhfs, rlfs = _lim_at_z(np.array(grid.limiter_r), np.array(grid.limiter_z), z0)
    dminor = min(r0 - rhfs, rlfs - r0)

    def boundary(rad):
        s = np.hypot(cr - r0, cz - z0) < rad
        if s.sum() < 5:
            return None
        ic = np.zeros(grid.cells.size)
        ic[s] = ip / s.sum()
        psi = np.asarray(basis.psi_grid_2d_np(ic, p.i_pf), np.float64).reshape(
            grid.nz, grid.nr
        )
        lc = lcfs_contour(
            psi,
            grid.rg,
            grid.zg,
            (r0, z0),
            clip_legs=True,
            limiter_r=grid.limiter_r,
            limiter_z=grid.limiter_z,
        )
        return lc.ring if lc.found else None

    rad = 0.9 * dminor
    ring = None
    for _ in range(8):  # self-consistent: rad -> boundary minor radius
        ring = boundary(rad)
        if ring is None:
            break
        bmin = float(np.hypot(ring[:, 0] - r0, ring[:, 1] - z0).mean())
        new = 0.5 * rad + 0.5 * bmin
        if abs(new - rad) < 5e-3:
            rad = new
            ring = boundary(rad)
            break
        rad = new
    return ring, r0, z0, rad


def _moment_ring(grid, basis, p):
    mom = fit_moment_currents(basis, p, MomentFitConfig(order=3))
    psi = np.asarray(basis.psi_grid_2d_np(mom.i_cell, p.i_pf), np.float64).reshape(
        grid.nz, grid.nr
    )
    lc = lcfs_contour(
        psi,
        grid.rg,
        grid.zg,
        (float(mom.centroid_r), float(mom.centroid_z)),
        clip_legs=True,
        limiter_r=grid.limiter_r,
        limiter_z=grid.limiter_z,
    )
    return lc.ring if lc.found else None


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
            ring, r0, z0, rad = dynamic_disc(grid, basis, table, p)
            mring = _moment_ring(grid, basis, p)
            rms_d = _rms(ring, efit)
            rms_m = _rms(mring, efit)
            panels.append(
                (f"{shot} {kind}", grid, efit, ring, mring, rms_d, rms_m, (r0, z0), rad)
            )  # noqa: E501
            print(
                f"{shot} {kind}: dynamic-disc RMS={rms_d:.1f}cm "
                f"moment RMS={rms_m:.1f}cm (R0={r0:.2f} rad={rad:.2f})"
            )

    ncol = 4
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 4.0 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, (title, grid, efit, ring, mring, rms_d, rms_m, ctr, rad) in zip(
        axes, panels, strict=False
    ):
        ax.axis("on")
        ax.plot(grid.limiter_r, grid.limiter_z, "k-", lw=0.5)
        ax.set_aspect("equal")
        if efit is not None:
            ax.plot(
                efit["lcfs_r"], efit["lcfs_z"], "r-", lw=2.2, label="EFIT", zorder=5
            )  # noqa: E501
        if mring is not None:
            ax.plot(mring[:, 0], mring[:, 1], "b-", lw=1.0, label=f"moment {rms_m:.0f}")
        if ring is not None:
            ax.plot(ring[:, 0], ring[:, 1], "g-", lw=1.4, label=f"disc {rms_d:.0f}cm")
        th = np.linspace(0, 2 * np.pi, 60)
        ax.plot(ctr[0] + rad * np.cos(th), ctr[1] + rad * np.sin(th), "g:", lw=0.7)
        ax.plot([ctr[0]], [ctr[1]], "g+", ms=7)
        ax.set_title(title, fontsize=8)
        ax.legend(fontsize=6, loc="upper right")
    fig.suptitle(
        "Dynamic-sizing uniform disc (green) vs cell-current moment (blue) vs "
        "firewalled EFIT (red); dotted = disc",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = "docs/figures/th-boundary-robustness/dynamic-disc-cohort.png"
    fig.savefig(out, dpi=110)
    print("wrote", out)


if __name__ == "__main__":
    raise SystemExit(main())
