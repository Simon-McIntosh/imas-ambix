#!/usr/bin/env python
"""Scoping studies for the patch-current force-balance representation.

The proposal under study: represent the plasma as piecewise-constant currents on
the in-limiter grid cells ("patches"), compute flux/field/diagnostics everywhere
by precomputed finite-area Green's interaction matrices (exactly solenoidal, no
grid solve, no boundary conditions), and express Grad-Shafranov force balance as
a differentiable RESIDUAL instead of an inner Picard solve.  In this basis
Ampere's law and div B = 0 hold identically for ANY current vector — the only
physics content left in GS is the flux-function structure

    jphi(R, Z) = a(psi) * R + b(psi) / R      (a = p', b = FF'/mu0)

i.e. on each psi level-set jphi is affine in {R, 1/R}.  The structure residual
penalises departure from that form without parametrising a(psi), b(psi), without
normalised flux, and without any topology extraction.

Sub-commands (each writes artifacts + figures):

  assemble      build & cache the patch->grid flux interaction matrix
  validate      far-field limit, Delta*-consistency, FD cross-check, timings
  oracle        bootstrapped-Picard reference solves -> patch currents (held-out)
  discriminate  structure residual on oracle vs scrambled/shifted/blob/null-space
  identify      whitened SVD spectrum + axis-shift visibility + training-free
                residual-regularised variational inverse (axis error vs lambda)

Artifacts: imas_ambix/latent/artifacts/patch_scoping/
Figures:   docs/figures/patch-current-force-balance/
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from imas_ambix.gs.cylinder import hybrid_greens
from imas_ambix.gs.geometry import build_table_for_shot
from imas_ambix.gs.operator import build_operator, greens_psi
from imas_ambix.latent.data import (
    feature_schema,
    load_shot_windows,
    read_split_shot_lists,
)
from imas_ambix.latent.gs_solve import (
    MU0,
    EquilibriumGrid,
    _read_axis,
    fit_profile,
    solve_equilibrium_bootstrapped,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("patch_basis_studies")

ARTIFACTS = Path("imas_ambix/latent/artifacts/patch_scoping")
FIGURES = Path("docs/figures/patch-current-force-balance")

# fixed categorical order for the perturbation classes (dataviz: assign by
# entity, never cycle) — validated single-hue-per-class, no rainbow
CLASS_ORDER = ("oracle", "null-space", "radial-shift", "gaussian-blob", "permuted")
CLASS_COLOR = {
    "oracle": "#2166ac",
    "null-space": "#7b3294",
    "radial-shift": "#d95f02",
    "gaussian-blob": "#c51b7d",
    "permuted": "#636363",
}


# --------------------------------------------------------------------------
# shared geometry / matrix assembly
# --------------------------------------------------------------------------


def build_grid(shot: int, nr: int, nz: int):
    table = build_table_for_shot(shot)
    fwd = build_operator(table)
    grid = EquilibriumGrid.from_table(table, nr=nr, nz=nz)
    return table, fwd, grid


def patch_grid_matrix(grid: EquilibriumGrid, cache: Path | None) -> np.ndarray:
    """(n_grid, n_cells) total-flux interaction matrix [Wb per A].

    Column c = psi on the whole grid per ampere of uniformly distributed
    current in cell c (finite-area kernel near the cell, point filament far).
    Rows at the cell centroids themselves are smooth and finite — the
    finite-area kernel is regular inside the conductor, which is what makes
    the patch->patch self-consistent flux (and hence the structure residual)
    well-defined without any self-inductance regularisation.
    """
    if cache is not None and cache.exists():
        g = np.load(cache)["g_pg"]
        if g.shape == (grid.flat_r.size, grid.cells.size):
            logger.info("patch->grid matrix loaded from %s", cache)
            return g
    t0 = time.perf_counter()
    cols = np.empty((grid.flat_r.size, grid.cells.size))
    for k, c in enumerate(grid.cells):
        cols[:, k] = hybrid_greens(
            grid.flat_r,
            grid.flat_z,
            float(grid.flat_r[c]),
            float(grid.flat_z[c]),
            grid.dr,
            grid.dz,
        )[0]
    dt = time.perf_counter() - t0
    logger.info(
        "patch->grid matrix assembled: %d cells x %d grid pts in %.1f s",
        grid.cells.size,
        grid.flat_r.size,
        dt,
    )
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, g_pg=cols, assemble_seconds=dt)
    return cols


def delta_star_apply(psi2d: np.ndarray, rg: np.ndarray, zg: np.ndarray) -> np.ndarray:
    """5-point FD Delta* on the interior (same stencil as the Picard solver)."""
    dr = float(rg[1] - rg[0])
    dz = float(zg[1] - zg[0])
    out = np.full_like(psi2d, np.nan)
    r = rg[None, 1:-1]
    rp = 0.5 * (rg[1:-1] + rg[2:])[None, :]
    rm = 0.5 * (rg[1:-1] + rg[:-2])[None, :]
    ce = r / (rp * dr * dr)
    cw = r / (rm * dr * dr)
    cn = 1.0 / (dz * dz)
    out[1:-1, 1:-1] = (
        ce * psi2d[1:-1, 2:]
        + cw * psi2d[1:-1, :-2]
        + cn * (psi2d[2:, 1:-1] + psi2d[:-2, 1:-1])
        - (ce + cw + 2.0 * cn) * psi2d[1:-1, 1:-1]
    )
    return out


# --------------------------------------------------------------------------
# the structure residual (torch, differentiable)
# --------------------------------------------------------------------------


def structure_residual(
    psi_c: torch.Tensor,
    r_c: torch.Tensor,
    jphi_c: torch.Tensor,
    *,
    n_bins: int = 24,
    bandwidth_bins: float = 1.0,
) -> torch.Tensor:
    """Unexplained fraction of current-weighted jphi variance under GS structure.

    Soft-bins cells by their (total) psi, weighted by jphi^2 so zero-current
    vacuum cells never pollute a bin; per bin solves the closed-form weighted
    least squares jphi ~ a*R + b/R; returns  sum_b sum_c w_bc (jphi - fit)^2 /
    sum_b sum_c w_bc jphi^2  — dimensionless, 0 for exact GS structure, O(1)
    for structureless current.  Bin placement is detached (auxiliary geometry);
    everything else differentiates through psi_c and jphi_c.
    """
    w_amp = jphi_c * jphi_c
    total = w_amp.sum()
    if float(total) <= 0.0:
        return torch.zeros((), dtype=psi_c.dtype)
    w_amp = w_amp / total
    with torch.no_grad():
        mean = (w_amp * psi_c).sum()
        std = torch.sqrt((w_amp * (psi_c - mean) ** 2).sum()).clamp_min(1e-12)
        lo, hi = mean - 2.5 * std, mean + 2.5 * std
        mu = torch.linspace(
            float(lo), float(hi), n_bins, dtype=psi_c.dtype, device=psi_c.device
        )
        h = bandwidth_bins * (hi - lo) / n_bins
    w = torch.exp(-0.5 * ((psi_c[None, :] - mu[:, None]) / h) ** 2) * w_amp[None, :]
    x = torch.stack([r_c, 1.0 / r_c], dim=-1)  # (N, 2)
    xtwx = torch.einsum("bn,nk,nl->bkl", w, x, x)
    eye = torch.eye(2, dtype=psi_c.dtype, device=psi_c.device)
    ridge = 1e-9 * xtwx.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-30)
    xtwy = torch.einsum("bn,nk,n->bk", w, x, jphi_c)
    beta = torch.linalg.solve(xtwx + ridge[:, None, None] * eye, xtwy)  # (B, 2)
    fit = torch.einsum("nk,bk->bn", x, beta)  # (B, N)
    num = (w * (jphi_c[None, :] - fit) ** 2).sum()
    den = (w * (jphi_c[None, :] ** 2)).sum().clamp_min(1e-30)
    return num / den


def residual_of_currents(
    i_cell: np.ndarray,
    g_cc: np.ndarray,
    psi_coil_cells: np.ndarray,
    r_cells: np.ndarray,
    cell_area: float,
    **kw,
) -> float:
    """Structure residual of a numpy current vector against its OWN total flux."""
    i_t = torch.as_tensor(i_cell, dtype=torch.float64)
    psi = torch.as_tensor(g_cc, dtype=torch.float64) @ i_t + torch.as_tensor(
        psi_coil_cells, dtype=torch.float64
    )
    jphi = i_t / cell_area
    r = torch.as_tensor(r_cells, dtype=torch.float64)
    with torch.no_grad():
        return float(structure_residual(psi, r, jphi, **kw))


# --------------------------------------------------------------------------
# slice payloads (measured magnetics, oracle fits)
# --------------------------------------------------------------------------


def slice_payloads(shot: int, table, fwd, grid, *, max_slices: int, min_ip_ka: float):
    """Measured magnetics per selected slice, aligned to sensor_greens rows."""
    g_sens, channels = grid.sensor_greens(table)
    w = load_shot_windows(shot, fwd, "eval", feature_schema(), with_referee=True)
    if w is None or w.ref_target is None:
        return None
    row_of = {ch: i for i, ch in enumerate(fwd.sensor_channels)}
    ch_rows = np.array([row_of.get(ch, -1) for ch in channels])
    present = ch_rows >= 0
    valid = [
        t
        for t in range(w.times.size)
        if np.isfinite(w.ref_target[t, :2]).all()
        and abs(w.anchored[t, 0]) > min_ip_ka
    ]
    if len(valid) > max_slices:
        valid = valid[:: max(1, len(valid) // max_slices)][:max_slices]
    scale = np.nanstd(w.raw_mag, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
    scale_ch = np.where(present, scale[np.clip(ch_rows, 0, None)], 1.0)
    out = []
    for t in valid:
        vac = fwd.vacuum_prediction(w.i_pf[t])
        out.append(
            {
                "shot": shot,
                "t_index": t,
                "time_s": float(w.times[t]),
                "i_pf": w.i_pf[t],
                "ip_amperes": float(abs(w.anchored[t, 0])) * 1e3,
                "measured": np.where(
                    present, w.raw_mag[t][np.clip(ch_rows, 0, None)], np.nan
                ),
                "vacuum": np.where(
                    present, vac[np.clip(ch_rows, 0, None)], 0.0
                ),
                "mask": present & w.mag_mask[t][np.clip(ch_rows, 0, None)],
                "scale": scale_ch,
                "ref_target": w.ref_target[t],
            }
        )
    return out


def oracle_slice_path(shot: int, t_index: int) -> Path:
    return ARTIFACTS / f"oracle_shot{shot}_t{t_index}.npz"


# --------------------------------------------------------------------------
# sub-commands
# --------------------------------------------------------------------------


def cmd_assemble(args) -> None:
    table, fwd, grid = build_grid(args.shot, args.nr, args.nz)
    cache = ARTIFACTS / f"g_pg_{table.signature.key}_{args.nr}x{args.nz}.npz"
    patch_grid_matrix(grid, cache)
    g_sens, channels = grid.sensor_greens(table)
    logger.info(
        "cells=%d grid=%dx%d sensors=%d", grid.cells.size, args.nr, args.nz, len(channels)
    )


def cmd_validate(args) -> None:
    table, fwd, grid = build_grid(args.shot, args.nr, args.nz)
    cache = ARTIFACTS / f"g_pg_{table.signature.key}_{args.nr}x{args.nz}.npz"
    g_pg = patch_grid_matrix(grid, cache)
    checks: dict = {}

    # --- far-field limit: PURE finite-area kernel vs point filament --------
    # (the hybrid column is point-identical beyond the switch band by
    # construction — exercising cylinder_greens itself is the real check)
    from imas_ambix.gs.cylinder import cylinder_greens

    c_mid = grid.cells[np.argmin(
        np.hypot(grid.flat_r[grid.cells] - grid.r0, grid.flat_z[grid.cells])
    )]
    dist = np.hypot(
        grid.flat_r - grid.flat_r[c_mid], grid.flat_z - grid.flat_z[c_mid]
    )
    far = dist > 10.0 * max(grid.dr, grid.dz)
    fa = cylinder_greens(
        grid.flat_r[far], grid.flat_z[far],
        float(grid.flat_r[c_mid]), float(grid.flat_z[c_mid]), grid.dr, grid.dz,
    )[0]
    point = greens_psi(
        grid.flat_r[far], grid.flat_z[far],
        float(grid.flat_r[c_mid]), float(grid.flat_z[c_mid]),
    )
    rel = np.abs(fa - point) / np.maximum(np.abs(point), 1e-16)
    checks["far_field_rel_max"] = float(rel.max())
    checks["far_field_rel_median"] = float(np.median(rel))

    # --- reference currents: one oracle solve for the check field ----------
    payloads = slice_payloads(
        args.shot, table, fwd, grid, max_slices=6, min_ip_ka=args.min_ip_ka
    )
    assert payloads, "no valid slice for validation"
    payloads = sorted(payloads, key=lambda q: -q["ip_amperes"])  # flat-top first
    fit, p = None, None
    for p in payloads:
        fit = fit_profile(
            grid,
            table,
            i_pf=p["i_pf"],
            ip_amperes=p["ip_amperes"],
            measured=p["measured"],
            vacuum_prediction=p["vacuum"],
            sensor_scale=p["scale"],
            sensor_mask=p["mask"],
            beta0_grid=(0.1, 0.3, 0.5, 0.7, 0.9),
            alpha_grid=(1.0, 2.0),
        )
        if fit is not None and fit.result.converged:
            break
        fit = None
    assert fit is not None and p is not None, "reference Picard solve failed"
    i_cell = fit.result.cell_currents
    cell_area = grid.dr * grid.dz

    # --- Delta*-consistency: FD Delta* of the EXACT Green's psi vs source --
    psi_plasma = (g_pg @ i_cell).reshape(grid.nz, grid.nr)
    lhs = delta_star_apply(psi_plasma, grid.rg, grid.zg)
    rhs = np.zeros(grid.flat_r.size)
    # total-flux convention: the Green's columns carry Φ = 2π R A_φ, so
    # Δ*Φ = −2π μ0 R jφ
    rhs[grid.cells] = -2.0 * np.pi * MU0 * grid.flat_r[grid.cells] * (
        i_cell / cell_area
    )
    rhs2d = rhs.reshape(grid.nz, grid.nr)
    scale_j = MU0 * np.abs(rhs2d[np.isfinite(rhs2d)]).max()
    err = np.abs(lhs - rhs2d)
    interior = np.isfinite(lhs)
    checks["delta_star_rel_rms"] = float(
        np.sqrt(np.nanmean(err[interior] ** 2)) / max(np.abs(rhs2d).max(), 1e-30)
    )

    # --- FD cross-check: Dirichlet solve (edge BC from Green's) vs Green's --
    psi_b2d = np.zeros((grid.nz, grid.nr))
    psi_b2d.ravel()[grid.edge_idx] = (g_pg @ i_cell)[grid.edge_idx]
    psi_fd = grid.solve_dirichlet(rhs2d, psi_b2d)
    fd_err = np.abs(psi_fd - psi_plasma)
    span = float(psi_plasma.max() - psi_plasma.min())
    checks["fd_vs_greens_rel_max"] = float(fd_err.max() / max(span, 1e-30))
    checks["fd_vs_greens_rel_rms"] = float(
        np.sqrt(np.mean(fd_err**2)) / max(span, 1e-30)
    )

    # --- smooth manufactured current: isolates FD truncation from assembly --
    # broad Gaussian well inside the domain — FD and Green's must agree to
    # O(h^2); a large discrepancy HERE would be an assembly bug, while a large
    # discrepancy only at sharp real-current features is the FD grid's own
    # representation error (the Green's field is exact for the patch source).
    r_cells = grid.flat_r[grid.cells]
    z_cells = grid.flat_z[grid.cells]
    blob = np.exp(-(((r_cells - grid.r0) / 0.35) ** 2 + (z_cells / 0.5) ** 2))
    blob *= p["ip_amperes"] / blob.sum()
    psi_blob = (g_pg @ blob).reshape(grid.nz, grid.nr)
    rhs_b = np.zeros(grid.flat_r.size)
    rhs_b[grid.cells] = -2.0 * np.pi * MU0 * r_cells * (blob / cell_area)
    rhs_b2d = rhs_b.reshape(grid.nz, grid.nr)
    psi_bb = np.zeros((grid.nz, grid.nr))
    psi_bb.ravel()[grid.edge_idx] = (g_pg @ blob)[grid.edge_idx]
    psi_fd_b = grid.solve_dirichlet(rhs_b2d, psi_bb)
    span_b = float(psi_blob.max() - psi_blob.min())
    checks["fd_vs_greens_smooth_rel_rms"] = float(
        np.sqrt(np.mean((psi_fd_b - psi_blob) ** 2)) / max(span_b, 1e-30)
    )
    checks["fd_vs_greens_smooth_rel_max"] = float(
        np.abs(psi_fd_b - psi_blob).max() / max(span_b, 1e-30)
    )

    # --- timing: full forward (grid psi + sensors) vs one Picard solve -----
    g_sens, _ = grid.sensor_greens(table)
    t0 = time.perf_counter()
    for _ in range(100):
        _psi = g_pg @ i_cell
        _d = g_sens @ i_cell
    forward_ms = (time.perf_counter() - t0) / 100 * 1e3
    t0 = time.perf_counter()
    solve_equilibrium_bootstrapped(
        grid, p["i_pf"], p["ip_amperes"], beta0=fit.beta0, alpha=fit.alpha
    )
    picard_s = time.perf_counter() - t0
    checks["forward_ms"] = forward_ms
    checks["picard_bootstrapped_s"] = picard_s
    checks["speedup"] = picard_s / (forward_ms / 1e3)

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "validate.json").write_text(json.dumps(checks, indent=2))
    logger.info("validate: %s", json.dumps(checks, indent=2))

    # --- figure -------------------------------------------------------------
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), constrained_layout=True)
    ext = [grid.rg[0], grid.rg[-1], grid.zg[0], grid.zg[-1]]
    vmax = np.abs(psi_plasma).max()
    im0 = axes[0].imshow(
        psi_plasma, origin="lower", extent=ext, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
        aspect="auto",
    )
    axes[0].plot(grid.limiter_r, grid.limiter_z, color="#222222", lw=1.0)
    axes[0].set_title("(a) plasma psi from interaction matrix [Wb]")
    fig.colorbar(im0, ax=axes[0], shrink=0.85)
    src_scale = max(np.abs(rhs2d).max(), 1e-30)
    im1 = axes[1].imshow(
        np.abs(np.nan_to_num(lhs - rhs2d)) / src_scale,
        origin="lower", extent=ext, cmap="Blues", aspect="auto",
    )
    axes[1].set_title("(b) |FD Delta* psi - source| / max|source|")
    fig.colorbar(im1, ax=axes[1], shrink=0.85)
    im2 = axes[2].imshow(
        fd_err / max(span, 1e-30), origin="lower", extent=ext, cmap="Blues",
        aspect="auto",
    )
    axes[2].set_title("(c) |FD solve - Green's| / psi span")
    fig.colorbar(im2, ax=axes[2], shrink=0.85)
    for ax in axes:
        ax.set_xlabel("R [m]")
    axes[0].set_ylabel("Z [m]")
    fig.suptitle(
        f"Interaction-matrix validation - shot {args.shot} "
        f"(forward {forward_ms:.2f} ms vs Picard {picard_s:.1f} s, "
        f"x{checks['speedup']:.0f})"
    )
    fig.savefig(FIGURES / "fig-interaction-matrix-validation.png", dpi=150)
    logger.info("figure written: fig-interaction-matrix-validation.png")


def cmd_oracle(args) -> None:
    """Reference bootstrapped-Picard fits on held-out slices -> patch currents."""
    _train, held = read_split_shot_lists(40, 8)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    n_done = 0
    for shot in held[: args.n_shots]:
        try:
            table, fwd, grid = build_grid(int(shot), args.nr, args.nz)
            payloads = slice_payloads(
                int(shot), table, fwd, grid,
                max_slices=args.slices_per_shot, min_ip_ka=args.min_ip_ka,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("shot %s failed to load: %s", shot, exc)
            continue
        if not payloads:
            continue
        for p in payloads:
            out = oracle_slice_path(p["shot"], p["t_index"])
            if out.exists():
                n_done += 1
                continue
            fit = fit_profile(
                grid,
                table,
                i_pf=p["i_pf"],
                ip_amperes=p["ip_amperes"],
                measured=p["measured"],
                vacuum_prediction=p["vacuum"],
                sensor_scale=p["scale"],
                sensor_mask=p["mask"],
                beta0_grid=(0.1, 0.3, 0.5, 0.7, 0.9),
                alpha_grid=(1.0, 2.0),
            )
            if fit is None or fit.cost > args.cost_limit or not fit.result.converged:
                logger.info(
                    "shot %d t=%d: no converged low-cost fit (cost=%s)",
                    p["shot"], p["t_index"], None if fit is None else f"{fit.cost:.2f}",
                )
                continue
            np.savez_compressed(
                out,
                i_cell=fit.result.cell_currents,
                beta0=fit.beta0,
                alpha=fit.alpha,
                cost=fit.cost,
                axis=np.array(fit.result.axis),
                axis_psi=fit.result.axis_psi,
                boundary_psi=fit.result.boundary_psi,
                i_pf=p["i_pf"],
                ip_amperes=p["ip_amperes"],
                measured=p["measured"],
                vacuum=p["vacuum"],
                mask=p["mask"],
                scale=p["scale"],
                ref_target=p["ref_target"],
                time_s=p["time_s"],
            )
            n_done += 1
            logger.info(
                "shot %d t=%d: oracle saved (cost %.2f, axis %.3f,%.3f)",
                p["shot"], p["t_index"], fit.cost, *fit.result.axis,
            )
    logger.info("oracle slices available: %d", n_done)


def _load_oracles() -> list[dict]:
    out = []
    for f in sorted(ARTIFACTS.glob("oracle_shot*_t*.npz")):
        d = dict(np.load(f))
        name = f.stem.replace("oracle_shot", "")
        shot, t = name.split("_t")
        d["shot"], d["t_index"] = int(shot), int(t)
        out.append(d)
    return out


def cmd_discriminate(args) -> None:
    oracles = _load_oracles()
    assert oracles, "run the oracle sub-command first"
    by_shot: dict[int, tuple] = {}
    rows = []
    rng = np.random.default_rng(20260703)
    for d in oracles:
        shot = d["shot"]
        if shot not in by_shot:
            table, fwd, grid = build_grid(shot, args.nr, args.nz)
            cache = ARTIFACTS / f"g_pg_{table.signature.key}_{args.nr}x{args.nz}.npz"
            g_pg = patch_grid_matrix(grid, cache)
            g_cc = g_pg[grid.cells, :]
            g_sens, _ = grid.sensor_greens(table)
            by_shot[shot] = (table, fwd, grid, g_pg, g_cc, g_sens)
        table, fwd, grid, g_pg, g_cc, g_sens = by_shot[shot]
        cell_area = grid.dr * grid.dz
        r_cells = grid.flat_r[grid.cells]
        psi_coil_cells = grid.coil_psi(d["i_pf"])[grid.cells]
        i0 = d["i_cell"]

        def resid(i_vec: np.ndarray) -> float:
            return residual_of_currents(
                i_vec, g_cc, psi_coil_cells, r_cells, cell_area,
                n_bins=args.n_bins,
            )

        core = np.abs(i0) > 0
        rows.append((d["shot"], d["t_index"], "oracle", resid(i0)))

        # permuted: same currents, shuffled among the core cells
        for _ in range(3):
            i_p = i0.copy()
            idx = np.where(core)[0]
            i_p[idx] = i0[idx][rng.permutation(idx.size)]
            rows.append((d["shot"], d["t_index"], "permuted", resid(i_p)))

        # radial shift: same pattern moved outward by k cells (force balance
        # broken relative to the coil field it sits in)
        i2d = np.zeros(grid.flat_r.size)
        i2d[grid.cells] = i0
        i2d = i2d.reshape(grid.nz, grid.nr)
        for k in (2, 4):
            i_s2d = np.zeros_like(i2d)
            i_s2d[:, k:] = i2d[:, :-k]
            i_s = i_s2d.ravel()[grid.cells]
            if abs(i_s.sum()) > 0:
                i_s *= i0.sum() / i_s.sum()
            rows.append((d["shot"], d["t_index"], f"radial-shift", resid(i_s)))

        # gaussian blob: the Picard seed shape scaled to Ip — plausible-looking,
        # not force balanced
        blob = np.exp(
            -(
                ((r_cells - grid.r0) / 0.35) ** 2
                + (grid.flat_z[grid.cells] / 0.5) ** 2
            )
        )
        blob *= i0.sum() / blob.sum()
        rows.append((d["shot"], d["t_index"], "gaussian-blob", resid(blob)))

        # null-space perturbation: invisible to every sensor, so ONLY a physics
        # residual can penalise it
        a = g_sens / d["scale"][:, None]
        _u, _s, vt = np.linalg.svd(a, full_matrices=True)
        null = vt[a.shape[0]:, :]
        for _ in range(3):
            v = null.T @ rng.standard_normal(null.shape[0])
            v /= np.linalg.norm(v)
            amp = args.null_fraction * np.linalg.norm(i0)
            rows.append(
                (d["shot"], d["t_index"], "null-space", resid(i0 + amp * v))
            )

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with (ARTIFACTS / "discriminate.json").open("w") as fh:
        json.dump(
            [
                {"shot": s, "t_index": t, "class": c, "residual": r}
                for s, t, c, r in rows
            ],
            fh,
            indent=2,
        )
    for cls in CLASS_ORDER:
        vals = [r for _, _, c, r in rows if c == cls]
        if vals:
            logger.info(
                "%-14s n=%2d residual median %.4f (min %.4f max %.4f)",
                cls, len(vals), np.median(vals), min(vals), max(vals),
            )

    # figure: strip chart per class (log x)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 4.2), constrained_layout=True)
    for yi, cls in enumerate(CLASS_ORDER):
        vals = np.array([r for _, _, c, r in rows if c == cls])
        if not vals.size:
            continue
        jitter = (np.arange(vals.size) - vals.size / 2) * 0.05
        ax.scatter(
            vals, np.full(vals.size, yi) + jitter, s=42,
            color=CLASS_COLOR[cls], zorder=3,
        )
        ax.annotate(
            f"median {np.median(vals):.3f}",
            (np.median(vals), yi + 0.28),
            fontsize=9, color=CLASS_COLOR[cls], ha="center",
        )
    ax.set_yticks(range(len(CLASS_ORDER)), CLASS_ORDER)
    ax.set_xscale("log")
    ax.set_xlabel("structure residual (unexplained fraction of jphi^2)")
    ax.set_title("Force-balance structure residual discriminates real equilibria")
    ax.grid(True, axis="x", alpha=0.25)
    fig.savefig(FIGURES / "fig-structure-residual-discrimination.png", dpi=150)
    logger.info("figure written: fig-structure-residual-discrimination.png")


def cmd_identify(args) -> None:
    oracles = _load_oracles()
    assert oracles, "run the oracle sub-command first"
    torch.manual_seed(20260703)

    # baseline (train-mean geometry) from the recorded gate arrays
    gate = np.load("imas_ambix/latent/artifacts/gs_solve_gate_arrays.npz")
    baseline_axis = gate["baseline"][0, :2]

    by_shot: dict[int, tuple] = {}
    svd_out: dict = {}
    inverse_rows = []
    lambdas = [float(x) for x in args.lambdas.split(",")]
    for d in oracles:
        shot = d["shot"]
        if shot not in by_shot:
            table, fwd, grid = build_grid(shot, args.nr, args.nz)
            cache = ARTIFACTS / f"g_pg_{table.signature.key}_{args.nr}x{args.nz}.npz"
            g_pg = patch_grid_matrix(grid, cache)
            by_shot[shot] = (table, fwd, grid, g_pg, grid.sensor_greens(table)[0])
        table, fwd, grid, g_pg, g_sens = by_shot[shot]
        cell_area = grid.dr * grid.dz
        r_cells = grid.flat_r[grid.cells]
        z_cells = grid.flat_z[grid.cells]

        # ---- SVD identifiability (once per shot) ---------------------------
        if shot not in svd_out:
            a = g_sens / d["scale"][:, None]
            s = np.linalg.svd(a, compute_uv=False)
            # visible fraction of a rigid radial axis shift of the oracle
            i2d = np.zeros(grid.flat_r.size)
            i2d[grid.cells] = d["i_cell"]
            i2d = i2d.reshape(grid.nz, grid.nr)
            vis = []
            u, sv, vt = np.linalg.svd(a, full_matrices=False)
            row_space = vt  # (S, N)
            for k in (1, 2, 3, 4):
                sh = np.zeros_like(i2d)
                sh[:, k:] = i2d[:, :-k]
                dvec = sh.ravel()[grid.cells] - d["i_cell"]
                coef = row_space @ dvec
                vis.append(
                    (
                        k * grid.dr,
                        float(np.linalg.norm(coef) / max(np.linalg.norm(dvec), 1e-30)),
                        float(np.linalg.norm((row_space @ dvec) * sv)),
                    )
                )
            svd_out[shot] = {
                "singular_values": s.tolist(),
                "axis_shift_visibility": vis,
                "n_cells": int(grid.cells.size),
                "n_sensors": int(g_sens.shape[0]),
            }

        # ---- training-free variational inverse -----------------------------
        g_cc_t = torch.as_tensor(g_pg[grid.cells, :], dtype=torch.float64)
        g_sens_t = torch.as_tensor(g_sens, dtype=torch.float64)
        psi_coil_cells = torch.as_tensor(
            grid.coil_psi(d["i_pf"])[grid.cells], dtype=torch.float64
        )
        r_t = torch.as_tensor(r_cells, dtype=torch.float64)
        meas = torch.as_tensor(np.nan_to_num(d["measured"]), dtype=torch.float64)
        vac = torch.as_tensor(d["vacuum"], dtype=torch.float64)
        mask = torch.as_tensor(d["mask"].astype(np.float64), dtype=torch.float64)
        scale = torch.as_tensor(d["scale"], dtype=torch.float64)
        ip = float(d["ip_amperes"])
        candidate = torch.as_tensor(
            (grid.topology_candidate.ravel()[grid.cells]).astype(np.float64),
            dtype=torch.float64,
        )

        seed = np.exp(
            -(
                ((r_cells - grid.r0) / 0.35) ** 2
                + (z_cells / 0.5) ** 2
            )
        )
        seed = seed / seed.sum() * ip

        for lam in lambdas:
            i_var = torch.tensor(seed.copy(), dtype=torch.float64, requires_grad=True)
            opt = torch.optim.Adam([i_var], lr=args.lr * ip / seed.size)
            for _ in range(args.iters):
                opt.zero_grad()
                i_eff = i_var * candidate
                pred = vac + g_sens_t @ i_eff
                misfit = (mask * ((pred - meas) / scale) ** 2).sum() / mask.sum()
                ip_pen = ((i_eff.sum() - ip) / ip) ** 2
                psi_c = g_cc_t @ i_eff + psi_coil_cells
                fb = structure_residual(psi_c, r_t, i_eff / cell_area,
                                        n_bins=args.n_bins)
                loss = misfit + 10.0 * ip_pen + lam * fb
                loss.backward()
                opt.step()
            with torch.no_grad():
                i_fin = (i_var * candidate).numpy()
            psi2d = (g_pg @ i_fin + grid.coil_psi(d["i_pf"])).reshape(
                grid.nz, grid.nr
            )
            axis, _ = _read_axis(psi2d, grid, 1.0)
            ref_axis = d["ref_target"][:2]
            err = float(np.hypot(axis[0] - ref_axis[0], axis[1] - ref_axis[1]))
            inverse_rows.append(
                {
                    "shot": shot,
                    "t_index": int(d["t_index"]),
                    "lambda_fb": lam,
                    "axis_error_m": err,
                    "misfit": float(misfit),
                    "structure_residual": float(fb),
                    "oracle_axis_error_m": float(
                        np.hypot(*(np.asarray(d["axis"]) - ref_axis))
                    ),
                    "baseline_axis_error_m": float(
                        np.hypot(*(baseline_axis - ref_axis))
                    ),
                }
            )
            logger.info(
                "shot %d t=%d lam=%.2g: axis err %.3f m (oracle %.3f, baseline %.3f)"
                " misfit %.2f fb %.4f",
                shot, d["t_index"], lam, err,
                inverse_rows[-1]["oracle_axis_error_m"],
                inverse_rows[-1]["baseline_axis_error_m"],
                float(misfit), float(fb),
            )

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "identify.json").write_text(
        json.dumps({"svd": svd_out, "inverse": inverse_rows}, indent=2)
    )

    # ---- figures ------------------------------------------------------------
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    for shot, sv in svd_out.items():
        s = np.array(sv["singular_values"])
        axes[0].semilogy(np.arange(1, s.size + 1), s / s[0], lw=2,
                         label=f"shot {shot}")
    axes[0].axhline(1e-3, color="#888888", lw=1, ls="--")
    axes[0].annotate("~measurement floor", (30, 1.2e-3), fontsize=8, color="#666666")
    axes[0].set_xlabel("mode index")
    axes[0].set_ylabel("normalised singular value")
    axes[0].set_title("(a) whitened sensor-matrix spectrum")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.25)
    for shot, sv in svd_out.items():
        v = np.array(sv["axis_shift_visibility"])
        axes[1].plot(v[:, 0] * 100, v[:, 1], "o-", lw=2, label=f"shot {shot}")
    axes[1].set_xlabel("rigid radial shift of plasma current [cm]")
    axes[1].set_ylabel("fraction of ||dI|| in sensor row space")
    axes[1].set_title("(b) axis-shift visibility to magnetics")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.25)
    lam_arr = sorted({r["lambda_fb"] for r in inverse_rows})
    med = [
        np.median([r["axis_error_m"] for r in inverse_rows if r["lambda_fb"] == la])
        for la in lam_arr
    ]
    for r in inverse_rows:
        axes[2].scatter(
            max(r["lambda_fb"], args.lambda_floor), r["axis_error_m"],
            color="#2166ac", s=24, alpha=0.5, zorder=3,
        )
    axes[2].plot(
        [max(la, args.lambda_floor) for la in lam_arr], med, "-o",
        color="#2166ac", lw=2, zorder=4, label="median",
    )
    oracle_med = np.median([r["oracle_axis_error_m"] for r in inverse_rows])
    base_med = np.median([r["baseline_axis_error_m"] for r in inverse_rows])
    axes[2].axhline(oracle_med, color="#1b7837", lw=1.5, ls="--")
    axes[2].annotate(f"Picard oracle {oracle_med:.3f} m", (lam_arr[1] if len(lam_arr) > 1 else 1, oracle_med * 1.05),
                     fontsize=8, color="#1b7837")
    axes[2].axhline(base_med, color="#636363", lw=1.5, ls=":")
    axes[2].annotate(f"train-mean {base_med:.3f} m", (lam_arr[1] if len(lam_arr) > 1 else 1, base_med * 1.05),
                     fontsize=8, color="#636363")
    axes[2].set_xscale("log")
    axes[2].set_xlabel("force-balance weight lambda")
    axes[2].set_ylabel("axis error vs referee [m]")
    axes[2].set_title("(c) training-free inverse: physics weight vs axis error")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(fontsize=8)
    fig.savefig(FIGURES / "fig-identifiability-and-inverse.png", dpi=150)
    logger.info("figure written: fig-identifiability-and-inverse.png")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shot", type=int, default=18502)
    ap.add_argument("--nr", type=int, default=65)
    ap.add_argument("--nz", type=int, default=97)
    ap.add_argument("--min-ip-ka", type=float, default=300.0)
    ap.add_argument("--n-bins", type=int, default=24)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("assemble")
    sub.add_parser("validate")
    p_or = sub.add_parser("oracle")
    p_or.add_argument("--n-shots", type=int, default=4)
    p_or.add_argument("--slices-per-shot", type=int, default=3)
    p_or.add_argument("--cost-limit", type=float, default=3.0)
    p_di = sub.add_parser("discriminate")
    p_di.add_argument("--null-fraction", type=float, default=0.3)
    p_id = sub.add_parser("identify")
    p_id.add_argument("--lambdas", type=str, default="0,0.3,1,3,10")
    p_id.add_argument("--iters", type=int, default=800)
    p_id.add_argument("--lr", type=float, default=0.05)
    p_id.add_argument("--lambda-floor", type=float, default=0.03)
    args = ap.parse_args()
    {
        "assemble": cmd_assemble,
        "validate": cmd_validate,
        "oracle": cmd_oracle,
        "discriminate": cmd_discriminate,
        "identify": cmd_identify,
    }[args.cmd](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
