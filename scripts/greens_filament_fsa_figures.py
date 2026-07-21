"""Figures for the contour-free flux-surface-averaging read (greens-filament §3).

Solves one representative flat-top slice through the frozen spine (the same path
the physics-spine benchmark uses), then compares the host coarea read against the
contour-free connectivity (JAX) read on the identical equilibrium:

* fig-fsa-d-roughness-resolution.png — the d = g2·g3/ρ̂ second-difference
  roughness vs n_ρ ∈ {16,32,64,96} for both reads (the G2a/G2b evidence);
* fig-fsa-profile-smoothness.png — d(ρ̂), V′(ρ̂) and ⟨1/R⟩(ρ̂) overlaid for both
  reads at a fixed n_ρ (what "smoother, same surfaces" looks like).

Run on a SLURM compute node (needs the campaign geometry + a solve):

    srun --partition=sun_debug --cpus-per-task=8 --mem=16G --time=01:00:00 \
      bash -lc 'export TMPDIR=/tmp OMP_NUM_THREADS=1; cd <repo>; \
        uv run python -m scripts.greens_filament_fsa_figures --shot 21983'
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

FIGDIR = Path("docs/figures/greens-filament-solver")
NRHO_SWEEP = (16, 32, 48, 64, 96)
MODES = ("coarea", "connectivity")
_COL = {"coarea": "#c1440e", "connectivity": "#1f6feb"}
_LBL = {
    "coarea": "coarea (host baseline)",
    "connectivity": "connectivity (JAX, contour-free)",
}


def _solve_flat_top_slice(shot: int, nr: int, nz: int):
    """Rich frozen-spine ladder fit of the highest-Ip slice of ``shot``."""
    from imas_ambix.latent.boundary_disc import disc_read
    from scripts.greens_filament_gate_eval import _fit_slice
    from scripts.heldout_mse_gate_eval import _campaign_table, shot_bt0
    from scripts.position_controlled_solve_gate import _disc_seed_flat
    from scripts.spine_label_factory import factory_shot_payloads, frozen_spine_config

    spine, _ = frozen_spine_config()
    iso = spine["interior_solve"]
    n_p, n_f = int(iso["n_p"]), int(iso["n_f"])
    nonneg = iso["profile_kind"] == "monomial-nonneg"
    smoothness = float(iso["smoothness"])
    boundary_read = iso["boundary_read_scoring"]

    table = _campaign_table(shot)
    payload = factory_shot_payloads(
        shot, nr=nr, nz=nz, max_slices=12, min_ip_ka=200.0, table=table
    )
    if payload is None:
        raise SystemExit(f"shot {shot}: no payload")
    grid = payload["grid"]
    tbl, basis = payload["table"], payload["basis"]
    bt0 = shot_bt0(shot)
    # the highest-Ip payload = a clean flat-top slice
    ps = payload["payloads"]
    p = max(ps, key=lambda q: abs(q.ip_amperes))
    inv = disc_read(p, grid, tbl, basis)
    if inv is None or inv.ring is None:
        raise SystemExit(f"shot {shot}: disc read failed")
    centroid = (float(inv.centroid_r), float(inv.centroid_z))
    disc_seed = _disc_seed_flat(grid, inv)
    kw = dict(smoothness=smoothness, boundary_read=boundary_read, sigma=0.02)
    f_basin = _fit_slice(
        grid,
        tbl,
        basis,
        p,
        substrate="grid-delstar",
        warm=disc_seed,
        centroid=centroid,
        n_p=1,
        n_f=1,
        nonneg=False,
        **kw,
    )
    scored = f_basin.scored and f_basin.jphi_flat is not None
    seed = f_basin.jphi_flat if scored else disc_seed
    fit = _fit_slice(
        grid,
        tbl,
        basis,
        p,
        substrate="grid-delstar",
        warm=seed,
        centroid=centroid,
        n_p=n_p,
        n_f=n_f,
        nonneg=nonneg,
        **kw,
    )
    if not fit.scored:
        raise SystemExit(f"shot {shot}: rich solve did not score")
    return fit, grid, bt0, n_p, n_f, nonneg


def main() -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from imas_ambix.latent.current_diffusion import flux_surface_geometry
    from imas_ambix.spine_bench.runner import _d_roughness

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shot", type=int, default=21983)
    ap.add_argument("--nr", type=int, default=65)
    ap.add_argument("--nz", type=int, default=97)
    args = ap.parse_args()

    fit, grid, bt0, n_p, n_f, nonneg = _solve_flat_top_slice(
        args.shot, args.nr, args.nz
    )
    geo_kw = dict(
        coeffs=np.asarray(fit.coeffs, dtype=np.float64),
        ip_amperes=abs(float(fit.ip_amperes)),
        n_p=n_p,
        n_f=n_f,
        nonneg=nonneg,
        b_phi0=bt0,
    )
    FIGDIR.mkdir(parents=True, exist_ok=True)

    # --- fig 1: d-roughness resolution curve, both reads ---------------------
    rough = {m: [] for m in MODES}
    for m in MODES:
        for nrho in NRHO_SWEEP:
            geo = flux_surface_geometry(fit.psi, grid, n_rho=nrho, fsa_mode=m, **geo_kw)
            rough[m].append(_d_roughness(np.asarray(geo.d_face)) if geo else np.nan)
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for m in MODES:
        ax.plot(NRHO_SWEEP, rough[m], "o-", color=_COL[m], label=_LBL[m], lw=2, ms=6)
    ax.set_xscale("log", base=2)
    ax.set_xticks(NRHO_SWEEP)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel(r"radial resolution $n_\rho$")
    ax.set_ylabel(r"relative 2nd-diff roughness of $d=g_2 g_3/\hat\rho$")
    ax.set_title(
        f"FSA d-roughness vs resolution (shot {args.shot} flat-top)\n"
        "lower = smoother; a rising curve = degrades with resolution"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    p1 = FIGDIR / "fig-fsa-d-roughness-resolution.png"
    fig.savefig(p1, dpi=130)
    plt.close(fig)

    # --- fig 2: profile smoothness overlay at a fixed n_rho ------------------
    nrho = 64
    geos = {
        m: flux_surface_geometry(fit.psi, grid, n_rho=nrho, fsa_mode=m, **geo_kw)
        for m in MODES
    }
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))
    for m in MODES:
        g = geos[m]
        axes[0].plot(g.rho_face, g.d_face, color=_COL[m], label=_LBL[m], lw=1.8)
        axes[1].plot(g.rho_face, g.vpr_face, color=_COL[m], lw=1.8)
        axes[2].plot(g.rho_cell, g.inv_r_cell, color=_COL[m], lw=1.8)
    axes[0].set_title(r"$d=g_2 g_3/\hat\rho$ (diffusion coefficient)")
    axes[1].set_title(r"$V'=dV/d\hat\rho$")
    axes[2].set_title(r"$\langle 1/R\rangle$")
    for a in axes:
        a.set_xlabel(r"$\hat\rho$")
        a.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(
        f"Flux-surface metrics, host vs contour-free reads on the SAME ψ "
        f"(shot {args.shot}, n_ρ={nrho})"
    )
    fig.tight_layout()
    p2 = FIGDIR / "fig-fsa-profile-smoothness.png"
    fig.savefig(p2, dpi=130)
    plt.close(fig)

    print("roughness by n_rho:")
    for m in MODES:
        print(
            f"  {m:13s}: "
            + "  ".join(
                f"n{nr}={r:.3f}" for nr, r in zip(NRHO_SWEEP, rough[m], strict=True)
            )
        )
    print(f"wrote {p1}\nwrote {p2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
