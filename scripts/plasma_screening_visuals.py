#!/usr/bin/env python
"""Visual evidence for the breakdown-start truth chain (imas-ink).

Three figure families over one regenerated skin-truth sequence, emitted
from the breakdown catch onward (``emit_pre``):

1. **Flux-surface / LCFS evolution** — nested ψ surfaces, the push-out
   LCFS, axis and X-candidates, from the 5%-Ip catch through the ramp to
   the flat-top hold (imas-ink components on the machine wall).
2. **Profile evolution** — the p′(ψ_N) and FF′(ψ_N) source functions
   extracted from the SOLVED equilibria by per-bin least squares on the
   two-term GS form jφ = R·p′ + FF′/(μ0·R), plus the flux-surface-averaged
   ⟨jφ⟩(ψ_N), as families coloured by Ip fraction.
3. **Model verification** — the plasma IS a grid of rectangular-cross-
   section toroidal (cylinder-kernel) cells across the whole in-limiter
   region, with currents set by the force-balanced GS solve whose source
   functions are flux functions: jφ/g(R) collapses onto a single curve of
   ψ_N (exact, by construction) — contrasted with the PRE-projection
   circuit state whose poloidal structure the flux-function family
   annihilates before every solve.

Figures: docs/figures/plasma-screening-dynamics/fig-truth-*.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from imas_ink import (
    ContourExtractor,
    FluxContours,
    LcfsOutline,
    OPointMarker,
    WallOutline,
    XPointMarkers,
    render_mpl,
)
from matplotlib import cm

from imas_ambix.latent.topology import lcfs_contour
from scripts.plasma_screening_gate import (
    BETA_SPLIT,
    _campaign,
    _psi_n_state,
    generate_skin_sequence,
)

FIGURES = Path("docs/figures/plasma-screening-dynamics")


def _row_geometry(row, grid):
    """(psi2d, axis, psi_axis, ring, psi_bnd) via the push-out reader."""
    truth = row["truth"]
    psi_n, _core, axis, axis_psi = _psi_n_state(
        truth.psi, grid, float(truth.ip_amperes)
    )
    lc = lcfs_contour(
        np.asarray(truth.psi),
        grid.rg,
        grid.zg,
        (float(axis[0]), float(axis[1])),
        limiter_r=grid.limiter_r,
        limiter_z=grid.limiter_z,
        clip_legs=True,
    )
    return truth.psi, axis, float(axis_psi), lc


def figure_flux_evolution(seq, grid, tag=""):
    rows = seq["rows"]
    n_show = min(8, len(rows))
    idx = np.unique(np.linspace(0, len(rows) - 1, n_show).astype(int))
    fig, axes = plt.subplots(2, 4, figsize=(15, 9), sharex=True, sharey=True)
    ip_end = float(seq["ip_end"])
    for ax, j in zip(axes.ravel(), idx, strict=False):
        row = rows[j]
        psi2d, axis, psi_ax, lc = _row_geometry(row, grid)
        r2d, z2d = np.meshgrid(grid.rg, grid.zg)
        ex = ContourExtractor(r2d, z2d, np.asarray(psi2d))
        psi_bnd = float(lc.psi_bnd) if lc.found else float(row["truth"].boundary_psi)
        render_mpl(ax, WallOutline(grid.limiter_r, grid.limiter_z))
        render_mpl(ax, FluxContours(ex.flux_surfaces(psi_ax, psi_bnd, n=7)))
        if lc.found:
            render_mpl(ax, LcfsOutline(lc.ring[:, 0], lc.ring[:, 1]))
        render_mpl(ax, OPointMarker(float(axis[0]), float(axis[1])))
        xt = np.asarray(row.get("x_true", []), dtype=np.float64).reshape(-1, 2)
        if xt.size:
            render_mpl(ax, XPointMarkers([(float(r), float(z)) for r, z in xt]))
        ip = float(row["truth"].ip_amperes)
        ax.set_title(
            f"t={row['time_s'] * 1e3:.0f} ms  Ip={ip / 1e3:.0f} kA "
            f"({ip / ip_end:.0%})\n{row['regime']} · {row['class_true']}",
            fontsize=9,
        )
        ax.set_aspect("equal")
    for ax in axes.ravel()[len(idx) :]:
        ax.axis("off")
    fig.suptitle(
        "Truth-chain flux surfaces + push-out LCFS — breakdown catch → "
        "growth → ramp → hold (one force-balanced free-boundary solve per step)"
    )
    fig.tight_layout()
    fig.savefig(FIGURES / f"fig-truth-flux-evolution{tag}.png", dpi=120)
    plt.close(fig)


def _bin_profiles(truth, grid, n_bins=14):
    """Per-ψ_N-bin (p′, FF′/μ0, ⟨jφ⟩) least-squares split of the SOLVED
    equilibrium's cell currents on the two-term GS form jφ = R·p′ + (FF′/μ0)/R."""
    psi_n, core, _axis, _apsi = _psi_n_state(truth.psi, grid, float(truth.ip_amperes))
    cells = grid.cells
    in_core = np.asarray(core, dtype=bool).ravel()[cells]
    pn = np.clip(psi_n[cells], 0.0, 1.0)[in_core]
    rr = grid.flat_r[cells][in_core]
    jphi = (truth.cell_currents / (grid.dr * grid.dz))[in_core]
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    pprime = np.full(n_bins, np.nan)
    ffprime = np.full(n_bins, np.nan)
    jbar = np.full(n_bins, np.nan)
    for b in range(n_bins):
        m = (pn >= edges[b]) & (pn < edges[b + 1])
        if m.sum() < 4:
            continue
        a = np.column_stack([rr[m], 1.0 / rr[m]])
        coef, *_ = np.linalg.lstsq(a, jphi[m], rcond=None)
        pprime[b], ffprime[b] = coef
        jbar[b] = float(jphi[m].mean())
    return centers, pprime, ffprime, jbar


def figure_profile_evolution(seq, grid, tag=""):
    rows = seq["rows"]
    ip_end = float(seq["ip_end"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    cmap = cm.viridis
    ls = {"pre": "--", "ramp": "-", "hold": ":"}
    for row in rows:
        frac = float(row["truth"].ip_amperes) / ip_end
        centers, pp, ff, jb = _bin_profiles(row["truth"], grid)
        kw = dict(color=cmap(frac), ls=ls[row["regime"]], lw=1.4, alpha=0.85)
        axes[0].plot(centers, pp / 1e6, **kw)
        axes[1].plot(centers, ff / 1e6, **kw)
        axes[2].plot(centers, jb / 1e6, **kw)
    axes[0].set_ylabel("p′-term amplitude  jφ_p/R  [MA/m³]")
    axes[1].set_ylabel("FF′-term amplitude  (FF′/μ0)  [MA/m]")
    axes[2].set_ylabel("⟨jφ⟩ flux-surface average  [MA/m²]")
    for ax, title in zip(
        axes,
        [
            "p′(ψ_N) — R-term source function",
            "FF′(ψ_N)/μ0 — 1/R-term source function",
            "flux-surface-averaged current density",
        ],
        strict=True,
    ):
        ax.set_xlabel("ψ_N")
        ax.set_title(title, fontsize=10)
        ax.axhline(0.0, color="k", lw=0.6)
    sm = cm.ScalarMappable(cmap=cmap)
    sm.set_clim(0.0, 1.0)
    fig.colorbar(sm, ax=axes, label="Ip fraction", shrink=0.85)
    fig.suptitle(
        "Truth-chain source-function evolution (extracted from the SOLVED "
        "equilibria by per-bin lstsq on jφ = R·p′ + FF′/(μ0R)) — "
        "pre-phase dashed, ramp solid, hold dotted"
    )
    fig.savefig(FIGURES / f"fig-truth-profile-evolution{tag}.png", dpi=120)
    plt.close(fig)


def figure_flux_function_check(seq, grid, tag=""):
    """The model-form verification: solved jφ/g(R) collapses onto a single
    curve of ψ_N (GS force balance with flux-function sources on the cell
    grid); the pre-projection circuit state does not (the poloidal content
    the flux-function truth family annihilates)."""
    rows = [r for r in seq["rows"] if "jphi_circuit_cells" in r]
    row = rows[len(rows) // 2] if rows else seq["rows"][len(seq["rows"]) // 2]
    truth = row["truth"]
    psi_n, core, axis, _apsi = _psi_n_state(truth.psi, grid, float(truth.ip_amperes))
    cells = grid.cells
    in_core = np.asarray(core, dtype=bool).ravel()[cells]
    pn = np.clip(psi_n[cells], 0.0, 1.0)
    rr = grid.flat_r[cells]
    zz = grid.flat_z[cells]
    theta = np.arctan2(zz - axis[1], rr - axis[0])
    g_r = BETA_SPLIT * rr / grid.r0 + (1.0 - BETA_SPLIT) * grid.r0 / rr
    j_solved = truth.cell_currents / (grid.dr * grid.dz)
    j_circuit = row.get("jphi_circuit_cells")

    def _binned_scatter_stat(j):
        """Poloidal-scatter statistic: per-fine-bin RMS residual after a
        LINEAR detrend in ψ_N (removes the profile's own radial variation
        within the bin, leaving only the scatter a function of ψ_N cannot
        have) / global max."""
        x = pn[in_core]
        y = (j / g_r)[in_core]
        edges = np.linspace(0, 1, 61)
        dev, norm = [], max(np.abs(y).max(), 1e-30)
        for b in range(60):
            m = (x >= edges[b]) & (x < edges[b + 1])
            if m.sum() > 5:
                a = np.column_stack([np.ones(int(m.sum())), x[m]])
                coef, *_ = np.linalg.lstsq(a, y[m], rcond=None)
                dev.append(np.sqrt(np.mean((y[m] - a @ coef) ** 2)))
        return float(np.mean(dev)) / norm if dev else float("nan")

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, j, title in zip(
        axes[0],
        [j_solved, j_circuit],
        [
            "SOLVED truth: jφ on the cell grid (force-balanced)",
            "pre-projection CIRCUIT state (patch dynamics, before the\n"
            "flux-function projection + GS re-solve)",
        ],
        strict=True,
    ):
        if j is None:
            ax.axis("off")
            continue
        j2d = np.zeros(grid.flat_r.size)
        j2d[cells] = j
        pc = ax.pcolormesh(
            grid.rg,
            grid.zg,
            j2d.reshape(grid.nz, grid.nr) / 1e6,
            cmap="RdBu_r",
            shading="auto",
        )
        vmax = float(np.abs(j2d).max()) / 1e6
        pc.set_clim(-vmax, vmax)
        render_mpl(ax, WallOutline(grid.limiter_r, grid.limiter_z))
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=10)
        fig.colorbar(pc, ax=ax, label="jφ [MA/m²]", shrink=0.8)

    for ax, j, name in zip(
        axes[1],
        [j_solved, j_circuit],
        ["solved truth", "circuit state"],
        strict=True,
    ):
        if j is None:
            ax.axis("off")
            continue
        y = (j / g_r)[in_core]
        sc = ax.scatter(
            pn[in_core],
            y / 1e6,
            c=theta[in_core],
            cmap="twilight",
            s=6,
            alpha=0.7,
        )
        stat = _binned_scatter_stat(j)
        ax.set_xlabel("ψ_N")
        ax.set_ylabel("jφ / g(R)  [MA/m²]   g = βR/R0 + (1−β)R0/R")
        ax.set_title(
            f"{name}: poloidal scatter (detrended) = {stat:.2e} of max\n"
            + (
                "(a flux function — single curve)"
                if stat < 5e-3
                else "(NOT a flux function — poloidal structure)"
            ),
            fontsize=10,
        )
        fig.colorbar(sc, ax=ax, label="poloidal angle θ [rad]", shrink=0.8)
    ip = float(truth.ip_amperes)
    fig.suptitle(
        f"Model-form verification (t={row['time_s'] * 1e3:.0f} ms, "
        f"Ip={ip / 1e3:.0f} kA, {row['regime']}): cylinder-cell grid + "
        "GS force balance with flux-function sources",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / f"fig-truth-flux-function-check{tag}.png", dpi=120)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=4200)
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()
    cfg = {
        "eta_true": [-6.5, 1.5, 1.5],
        "n_ramp": 10,
        "n_hold": 4,
        "n_pre": 6,
        "n_vac": 4,
        "settle_s": 0.05,
        "frac_bd": 0.05,
        "n_sub": 40,
        "quad_max": 0.25,
        "div_max": 0.0,
        "boost_max": 0.0,
        "frac_sat": 0.85,
        "beta_split": BETA_SPLIT,
        "vf_scale": 0.9,
        "eta_bd_pow": 1.5,
        "eta_bd_cap": 30.0,
        "emit_pre": True,
    }
    campaign = _campaign()
    grid = campaign.grid
    seq = generate_skin_sequence((args.seed, cfg))
    if seq is None:
        raise SystemExit("sequence dropped — pick another seed")
    FIGURES.mkdir(parents=True, exist_ok=True)
    tag = f"-{args.tag}" if args.tag else ""
    figure_flux_evolution(seq, grid, tag)
    figure_profile_evolution(seq, grid, tag)
    figure_flux_function_check(seq, grid, tag)
    print("figures written:", sorted(p.name for p in FIGURES.glob("fig-truth-*")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
