"""Discrete plasma Green's matrix + field matrix — the linear-in-DOF annulus map.

The soft-prior annulus penalty needs psi (and B_pol) at annulus points as a LINEAR
function of the per-cell plasma current, and that linear map must be the SAME operator
the Picard sweep actually applies (FD Dirichlet solve: interior source Delta*Phi =
-2 pi mu0 R jphi, edge BC = g_edge @ i_cell) -- not a free-space Green's sum, which
carries different domain-boundary behaviour.  These tests pin that consistency.
"""

from __future__ import annotations

import numpy as np

from imas_ambix.latent.gs_solve import MU0, EquilibriumGrid

from .test_gs_solve import _confining_table


def _grid():
    return EquilibriumGrid.from_table(_confining_table(), nr=49, nz=65)


def test_plasma_green_grid_reproduces_fd_dirichlet_solve():
    """plasma_green_grid @ i_cell == the Picard sweep's plasma psi, to ~1e-9."""
    grid = _grid()
    rng = np.random.default_rng(0)
    i_cell = rng.standard_normal(grid.cells.size) * 3.0e4  # [A]

    # the in-loop construction (gs_solve.solve_equilibrium_lsq lines ~1163-1173)
    cell_area = grid.dr * grid.dz
    jphi = np.zeros(grid.flat_r.size)
    jphi[grid.cells] = i_cell / cell_area
    rhs2d = (-(2.0 * np.pi * MU0) * grid.flat_r * jphi).reshape(grid.nz, grid.nr)
    psi_b2d = np.zeros((grid.nz, grid.nr))
    psi_b2d.ravel()[grid.edge_idx] = grid.g_edge @ i_cell
    psi_ref = grid.solve_dirichlet(rhs2d, psi_b2d).ravel()

    g_pl = grid.plasma_green_grid()
    psi_lin = g_pl @ i_cell

    span = float(psi_ref.max() - psi_ref.min()) or 1.0
    assert np.sqrt(np.mean((psi_lin - psi_ref) ** 2)) / span < 1e-9


def test_plasma_green_grid_cached_and_shaped():
    grid = _grid()
    g1 = grid.plasma_green_grid()
    g2 = grid.plasma_green_grid()
    assert g1 is g2  # cached (pure geometry)
    assert g1.shape == (grid.flat_r.size, grid.cells.size)


def test_plasma_bfield_green_grid_matches_finite_difference_of_psi():
    """B_R,B_Z green columns == curl of the psi green columns (total-flux convention
    B_R=-(1/2 pi R) dPsi/dZ, B_Z=+(1/2 pi R) dPsi/dR) on interior points."""
    grid = _grid()
    g_pl = grid.plasma_green_grid()
    gbr, gbz = grid.plasma_bfield_green_grid()
    assert gbr.shape == g_pl.shape and gbz.shape == g_pl.shape

    rng = np.random.default_rng(1)
    i_cell = rng.standard_normal(grid.cells.size) * 2.0e4
    psi = (g_pl @ i_cell).reshape(grid.nz, grid.nr)
    br = (gbr @ i_cell).reshape(grid.nz, grid.nr)
    bz = (gbz @ i_cell).reshape(grid.nz, grid.nr)

    # central differences on the interior, matching the module's own stencil
    rr = grid.mesh_r
    dpsi_dz = np.zeros_like(psi)
    dpsi_dr = np.zeros_like(psi)
    dpsi_dz[1:-1, :] = (psi[2:, :] - psi[:-2, :]) / (2.0 * grid.dz)
    dpsi_dr[:, 1:-1] = (psi[:, 2:] - psi[:, :-2]) / (2.0 * grid.dr)
    br_fd = -dpsi_dz / (2.0 * np.pi * rr)
    bz_fd = dpsi_dr / (2.0 * np.pi * rr)

    sel = np.zeros_like(psi, dtype=bool)
    sel[2:-2, 2:-2] = True
    for a, b in ((br, br_fd), (bz, bz_fd)):
        num = np.sqrt(np.mean((a[sel] - b[sel]) ** 2))
        den = np.sqrt(np.mean(b[sel] ** 2)) or 1.0
        assert num / den < 1e-9
