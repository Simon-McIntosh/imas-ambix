"""The well-posed position-controlled solve: current-centroid moment constraint.

These pin the §2 behaviour of :func:`solve_equilibrium_lsq` when the profile
coefficients are determined by the position constraint alone (Ip + R/Z current
centroid, no magnetics fit — the firewall-minimal position lever):

  * an all-default / centroid-off :class:`SoftPriors` leaves the data-fit
    reconstruction byte-identical (the frozen recon is untouched);
  * with the magnetics masked OFF the solve still runs and holds a confined
    equilibrium, and its current centroid tracks the measured target;
  * with the magnetics masked OFF the result is INDEPENDENT of ``measured`` —
    the only magnetics consumed is the centroid moment (+ Ip), the firewall
    G2c contract.
"""

from __future__ import annotations

import numpy as np

from imas_ambix.latent.gs_solve import (
    EquilibriumGrid,
    SoftPriors,
    solve_equilibrium_lsq,
)

from .test_gs_solve import _confining_table, _synthetic_confining_slice


def _centroid(grid: EquilibriumGrid, cell_currents: np.ndarray) -> tuple[float, float]:
    """Current centroid (R, Z) [m] of a per-cell current vector (grid.cells order)."""
    ic = np.asarray(cell_currents, dtype=np.float64)
    tot = ic.sum()
    cr = grid.flat_r[grid.cells]
    cz = grid.flat_z[grid.cells]
    return float((cr * ic).sum() / tot), float((cz * ic).sum() / tot)


def _fixture(nr=49, nz=65, beta0=0.6, alpha=1.5):
    table = _confining_table()
    grid = EquilibriumGrid.from_table(table, nr=nr, nz=nz)
    ip = 4.0e5
    i_pf = np.array([-6.0e4, -6.0e4])
    meas, vac, res_true = _synthetic_confining_slice(
        grid, table, i_pf, ip, beta0, alpha
    )
    return table, grid, i_pf, ip, meas, vac, res_true


def test_centroid_off_is_byte_identical():
    """A SoftPriors with no centroid target leaves the data-fit solve unchanged."""
    table, grid, i_pf, ip, meas, vac, _ = _fixture()
    kw = dict(
        measured=meas,
        vacuum_prediction=vac,
        sensor_scale=np.abs(meas) + 1e-9,
        sensor_mask=np.ones(meas.size, dtype=bool),
        max_iterations=40,
    )
    base = solve_equilibrium_lsq(grid, table, i_pf, ip, **kw)
    off = solve_equilibrium_lsq(grid, table, i_pf, ip, soft_priors=SoftPriors(), **kw)
    np.testing.assert_array_equal(base.result.psi, off.result.psi)
    np.testing.assert_array_equal(base.coeffs, off.coeffs)


def test_position_solve_holds_confined_without_magnetics():
    """With the magnetics masked off, the centroid-constrained solve still
    converges to a confined interior equilibrium seeded from a compact current."""
    table, grid, i_pf, ip, meas, vac, res_true = _fixture()
    r_c, z_c = _centroid(grid, res_true.cell_currents)
    off_mask = np.zeros(meas.size, dtype=bool)
    fit = solve_equilibrium_lsq(
        grid,
        table,
        i_pf,
        ip,
        measured=meas,
        vacuum_prediction=vac,
        sensor_scale=np.abs(meas) + 1e-9,
        sensor_mask=off_mask,  # NO magnetics fit
        n_p=1,
        n_f=1,
        initial_jphi=res_true.jphi.ravel(),  # compact confined seed
        soft_priors=SoftPriors(
            centroid_r_target=r_c,
            centroid_z_target=z_c,
            centroid_sigma_r=0.02,
            centroid_sigma_z=0.02,
        ),
        max_iterations=80,
    )
    assert np.isfinite(fit.result.psi).all()
    # an interior O-point exists (axis strictly inside the R grid, not railed)
    assert grid.rg[0] < fit.result.axis[0] < grid.rg[-1]
    # the current centroid tracks the measured target to a few cm
    fr, fz = _centroid(grid, fit.result.cell_currents)
    assert abs(fr - r_c) < 0.05
    assert abs(fz - z_c) < 0.05


def test_position_solve_ignores_magnetics_values():
    """G2c firewall: with the mask off, the result depends ONLY on the centroid
    moment (+ Ip) — corrupting ``measured`` leaves the equilibrium unchanged."""
    table, grid, i_pf, ip, meas, vac, res_true = _fixture()
    r_c, z_c = _centroid(grid, res_true.cell_currents)
    off_mask = np.zeros(meas.size, dtype=bool)
    sp = SoftPriors(centroid_r_target=r_c, centroid_z_target=z_c)
    common = dict(
        vacuum_prediction=vac,
        sensor_scale=np.abs(meas) + 1e-9,
        sensor_mask=off_mask,
        n_p=1,
        n_f=1,
        initial_jphi=res_true.jphi.ravel(),
        soft_priors=sp,
        max_iterations=60,
    )
    a = solve_equilibrium_lsq(grid, table, i_pf, ip, measured=meas, **common)
    b = solve_equilibrium_lsq(
        grid, table, i_pf, ip, measured=meas * 3.0 + 7.0, **common
    )
    np.testing.assert_array_equal(a.result.psi, b.result.psi)


def test_centroid_target_moves_the_current():
    """The constraint BITES: shifting the R target inboard pulls the current
    centroid inboard relative to the self-consistent target."""
    table, grid, i_pf, ip, meas, vac, res_true = _fixture()
    r_c, z_c = _centroid(grid, res_true.cell_currents)
    off_mask = np.zeros(meas.size, dtype=bool)
    common = dict(
        measured=meas,
        vacuum_prediction=vac,
        sensor_scale=np.abs(meas) + 1e-9,
        sensor_mask=off_mask,
        n_p=1,
        n_f=1,
        initial_jphi=res_true.jphi.ravel(),
        max_iterations=80,
    )
    at_true = solve_equilibrium_lsq(
        grid,
        table,
        i_pf,
        ip,
        soft_priors=SoftPriors(centroid_r_target=r_c, centroid_z_target=z_c),
        **common,
    )
    inboard = solve_equilibrium_lsq(
        grid,
        table,
        i_pf,
        ip,
        soft_priors=SoftPriors(
            centroid_r_target=r_c - 0.10,
            centroid_z_target=z_c,
            centroid_sigma_r=0.01,
        ),
        **common,
    )
    r_at, _ = _centroid(grid, at_true.result.cell_currents)
    r_in, _ = _centroid(grid, inboard.result.cell_currents)
    assert r_in < r_at  # the inboard target pulled the centroid inboard
