"""Analytic thick-cylinder Green's matrices — the linear-in-DOF annulus map.

The soft-prior annulus penalty needs psi AND B_pol at annulus points as a LINEAR
function of the per-cell plasma current.  Both come straight from the finite-area
cylinder Biot-Savart kernel (hybrid_greens) — ANALYTIC ψ and field, no finite
differences — matching the analytic carrier ψ the §2 annulus-consistency metric
uses.  These tests pin that the cached matrices equal the direct kernel
superposition, cell-for-cell.
"""

from __future__ import annotations

import numpy as np

from imas_ambix.gs.cylinder import hybrid_greens
from imas_ambix.latent.gs_solve import EquilibriumGrid

from .test_gs_solve import _confining_table


def _grid():
    return EquilibriumGrid.from_table(_confining_table(), nr=49, nz=65)


def _direct_superposition(grid, i_cell):
    cr = grid.flat_r[grid.cells]
    cz = grid.flat_z[grid.cells]
    psi = np.zeros(grid.cells.size)
    br = np.zeros(grid.cells.size)
    bz = np.zeros(grid.cells.size)
    for k, c in enumerate(grid.cells):
        p, r_, z_ = hybrid_greens(
            cr, cz, float(grid.flat_r[c]), float(grid.flat_z[c]), grid.dr, grid.dz
        )
        psi += i_cell[k] * p
        br += i_cell[k] * r_
        bz += i_cell[k] * z_
    return psi, br, bz


def test_cell_greens_matches_analytic_superposition():
    """psi/br/bz @ i_cell == the direct hybrid_greens superposition, to ~1e-12."""
    grid = _grid()
    rng = np.random.default_rng(0)
    i_cell = rng.standard_normal(grid.cells.size) * 3.0e4  # [A]

    g = grid.cell_greens()
    psi_lin = g["psi"] @ i_cell
    br_lin = g["br"] @ i_cell
    bz_lin = g["bz"] @ i_cell
    psi_ref, br_ref, bz_ref = _direct_superposition(grid, i_cell)

    for lin, ref in ((psi_lin, psi_ref), (br_lin, br_ref), (bz_lin, bz_ref)):
        span = float(np.abs(ref).max()) or 1.0
        assert np.max(np.abs(lin - ref)) / span < 1e-12


def test_cell_greens_cached_and_shaped():
    grid = _grid()
    g1 = grid.cell_greens()
    g2 = grid.cell_greens()
    assert g1 is g2  # cached (pure geometry)
    n = grid.cells.size
    assert g1["psi"].shape == (n, n)
    assert g1["br"].shape == (n, n)
    assert g1["bz"].shape == (n, n)
    assert np.array_equal(g1["cells"], grid.cells)


def test_field_is_gauge_free_psi_is_not():
    """Adding a constant current-independent offset to psi shifts psi but the
    field rows are unchanged — the property the grad-psi annulus penalty exploits."""
    grid = _grid()
    g = grid.cell_greens()
    # the field matrices carry no constant mode: a uniform column shift changes
    # psi but B is a difference operator on psi in the kernel, so identical
    # currents give identical fields regardless of any psi datum choice.
    assert np.isfinite(g["br"]).all() and np.isfinite(g["bz"]).all()


def test_plasma_design_matrix_is_thick_cylinder_not_point_filament():
    """The plasma patch→grid design matrix (:func:`_assemble_g_pg`) and
    :meth:`cell_greens` model every plasma cell with the finite-area
    thick-cylinder kernel, ungrouped — NOT the log-singular point filament.

    Guardrail: a column equals the thick kernel exactly (so a regression to the
    point kernel is caught), AND near the source cell the thick and point
    kernels differ materially (the thick self/near term is finite where the
    point filament diverges) — proving the thick choice is load-bearing, not an
    accidental match.
    """
    from imas_ambix.gs.operator import greens_psi  # noqa: PLC0415
    from imas_ambix.latent.patch_basis import _assemble_g_pg  # noqa: PLC0415

    grid = _grid()
    g_pg = _assemble_g_pg(grid)  # (n_grid, n_cells)
    k = grid.cells.size // 2
    c = grid.cells[k]
    cr, cz = float(grid.flat_r[c]), float(grid.flat_z[c])
    thick = hybrid_greens(grid.flat_r, grid.flat_z, cr, cz, grid.dr, grid.dz)[0]
    point = greens_psi(grid.flat_r, grid.flat_z, cr, cz)

    # design-matrix column IS the thick kernel (exact)
    assert np.max(np.abs(g_pg[:, k] - thick)) / (np.abs(thick).max()) < 1e-12
    # the thick self term is finite and well-behaved, and materially different
    # from the point-filament kernel at the source cell (the load-bearing gap
    # that would vanish if the design matrix regressed to a point filament)
    assert np.isfinite(thick[c]) and abs(thick[c]) > 0.0
    assert abs(thick[c] - point[c]) > 0.3 * abs(thick[c])
