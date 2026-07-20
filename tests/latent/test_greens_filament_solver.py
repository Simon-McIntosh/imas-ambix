"""Tests for the grid-free Green's/filament ψ substrate.

The free-boundary Grad–Shafranov solve evaluates ψ from the filament currents
either by inverting the gridded 5-point Δ* operator (``grid-delstar``, the
historical solve) or by the analytic finite-area Green's matvec
(``greens-matvec``).  Both invert the SAME elliptic operator — one on the grid,
one analytically — so on the same current distribution they must agree to
discretisation error, and exactly on the domain edge (both read the same
finite-area Green's block there).  These tests pin:

* T1  — the analytic plasma→target ψ reproduces the point-loop flux far from a
        filament, and the solve renormalises the filament currents so the net
        equals the measured Ip (filament turns sum to 1).
* the two substrates agree on the same current distribution (edge exactly,
  interior to discretisation tolerance) — the go/no-go's substrate-equivalence
  claim, isolated from the fixed-point iteration;
* the default substrate is byte-unchanged; and
* the grid-free path never assembles or inverts the gridded Δ* operator (G1c).
"""

from __future__ import annotations

import numpy as np
import pytest

from imas_ambix.gs.operator import greens_psi
from imas_ambix.latent.gs_solve import (
    SUBSTRATE_GREENS,
    SUBSTRATE_GRID,
    EquilibriumGrid,
    _plasma_psi_field,
    solve_equilibrium,
)

MU0 = 4.0e-7 * np.pi


def _circular_grid(r0=1.0, rb=0.4, nr=65, nz=65, with_coils=True):
    """A synthetic grid: circular limiter + an optional vertical-field coil pair."""
    rg = np.linspace(r0 - rb - 0.2, r0 + rb + 0.2, nr)
    zg = np.linspace(-(rb + 0.3), rb + 0.3, nz)
    theta = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
    limiter_r = r0 + rb * np.cos(theta)
    limiter_z = rb * np.sin(theta)
    n = rg.size * zg.size
    if with_coils:
        mesh_r, mesh_z = np.meshgrid(rg, zg)
        fr, fz = mesh_r.ravel(), mesh_z.ravel()
        # a Helmholtz-like pair well outside the limiter → a smooth vertical well
        col = greens_psi(fr, fz, r0 + 0.9, 0.8) + greens_psi(fr, fz, r0 + 0.9, -0.8)
        coil_cols = col[:, None]
    else:
        coil_cols = np.zeros((n, 0))
    return EquilibriumGrid(
        rg=rg,
        zg=zg,
        limiter_r=limiter_r,
        limiter_z=limiter_z,
        coil_psi_columns=coil_cols,
        r0=r0,
    )


def _gaussian_cell_currents(grid, ip=8.0e5, width=0.12):
    """A compact Gaussian jφ on the in-limiter cells, scaled to net current Ip."""
    cr = grid.flat_r[grid.cells]
    cz = grid.flat_z[grid.cells]
    jphi_cells = np.exp(-(((cr - grid.r0) / width) ** 2 + (cz / width) ** 2))
    jphi_full = np.zeros(grid.flat_r.size)
    jphi_full[grid.cells] = jphi_cells
    cell_area = grid.dr * grid.dz
    i_cell = jphi_full[grid.cells] * cell_area
    i_cell = i_cell * (ip / i_cell.sum())
    # jphi_full must carry the SAME Ip scaling as i_cell for the grid RHS
    jphi_full = jphi_full * (ip / (jphi_full[grid.cells] * cell_area).sum())
    return jphi_full, i_cell


# --- T1: analytic plasma→target flux + Ip renormalisation ------------------


def test_plasma_grid_psi_matches_point_loop_far_field():
    """Far from a single filament, the analytic finite-area ψ → the point loop."""
    grid = _circular_grid(with_coils=False)
    cols = grid.plasma_grid_psi_columns()
    assert cols.shape == (grid.flat_r.size, grid.cells.size)
    # a mid-domain source cell, evaluated at every FAR grid point (> 6 cells away)
    k = int(grid.cells.size // 2)
    src = grid.cells[k]
    sr, sz = float(grid.flat_r[src]), float(grid.flat_z[src])
    dist = np.hypot(grid.flat_r - sr, grid.flat_z - sz)
    far = dist > 6.0 * max(grid.dr, grid.dz)
    loop = greens_psi(grid.flat_r[far], grid.flat_z[far], sr, sz)
    analytic = cols[far, k]
    # finite-area vs point loop differ only by the O((d/r)^2) second moment
    np.testing.assert_allclose(analytic, loop, rtol=2e-3, atol=1e-9)


def test_edge_block_equals_g_edge_exactly():
    """The plasma→grid ψ columns reproduce the Dirichlet ``g_edge`` block exactly
    (same kernel), so the two substrates share the boundary data by construction."""
    grid = _circular_grid(with_coils=False)
    cols = grid.plasma_grid_psi_columns()
    np.testing.assert_allclose(cols[grid.edge_idx, :], grid.g_edge, rtol=0, atol=0)


def test_solve_renormalises_net_current_to_ip():
    """The converged (or last) filament currents sum to the measured Ip exactly —
    the filament turns sum to 1 by construction (net = Ip)."""
    grid = _circular_grid()
    ip = 7.5e5
    for substrate in (SUBSTRATE_GRID, SUBSTRATE_GREENS):
        res = solve_equilibrium(
            grid, np.array([1.0e4]), ip, max_iterations=20, substrate=substrate
        )
        assert res.cell_currents.sum() == pytest.approx(ip, rel=1e-9)


# --- substrate equivalence (the go/no-go substrate claim, iteration-free) --


def test_substrates_agree_on_same_currents():
    """On one fixed current distribution the analytic matvec and the gridded Δ*
    solve give the same ψ — exactly on the edge, to discretisation error inside."""
    grid = _circular_grid()
    psi_coil = grid.coil_psi(np.array([1.0e4]))
    jphi_full, i_cell = _gaussian_cell_currents(grid)

    psi_grid = _plasma_psi_field(
        grid, jphi_full, i_cell, psi_coil, substrate=SUBSTRATE_GRID
    )
    psi_greens = _plasma_psi_field(
        grid, jphi_full, i_cell, psi_coil, substrate=SUBSTRATE_GREENS
    )

    # edge ring: identical (shared finite-area Green's block)
    np.testing.assert_allclose(
        psi_greens[grid.edge_idx], psi_grid[grid.edge_idx], rtol=0, atol=1e-9
    )
    # interior: agree to discretisation error on this coarse grid
    span = float(np.abs(psi_grid).max())
    rel = np.abs(psi_greens - psi_grid) / span
    assert float(rel.max()) < 0.05
    assert float(np.sqrt(np.mean(rel**2))) < 0.01


# --- default byte-identity + G1c (no gridded Δ* in the grid-free path) -----


def test_default_substrate_is_grid_and_byte_identical():
    """Omitting ``substrate`` equals passing the grid default, field-for-field."""
    grid_a = _circular_grid()
    grid_b = _circular_grid()
    ip = 6.0e5
    ra = solve_equilibrium(grid_a, np.array([1.0e4]), ip, max_iterations=15)
    rb = solve_equilibrium(
        grid_b, np.array([1.0e4]), ip, max_iterations=15, substrate=SUBSTRATE_GRID
    )
    np.testing.assert_array_equal(ra.psi, rb.psi)
    assert ra.axis == rb.axis


def test_grid_path_assembles_delstar_greens_path_does_not():
    """G1c: the grid-free path never builds the Δ* LU; the grid path does."""
    grid_free = _circular_grid()
    assert grid_free._lu is None
    solve_equilibrium(
        grid_free,
        np.array([1.0e4]),
        6.0e5,
        max_iterations=8,
        substrate=SUBSTRATE_GREENS,
    )
    assert grid_free._lu is None  # analytic matvec only — no elliptic operator

    grid_gs = _circular_grid()
    solve_equilibrium(
        grid_gs,
        np.array([1.0e4]),
        6.0e5,
        max_iterations=8,
        substrate=SUBSTRATE_GRID,
    )
    assert grid_gs._lu is not None  # the 5-point Δ* was factorised and solved


def test_greens_matvec_rejects_boundary_continuation():
    """The grid-free substrate has no legacy boundary-continuation form."""
    grid = _circular_grid()
    with pytest.raises(ValueError, match="analytic-add"):
        solve_equilibrium(
            grid,
            np.array([1.0e4]),
            6.0e5,
            coil_field_mode="boundary-continuation",
            substrate=SUBSTRATE_GREENS,
        )
