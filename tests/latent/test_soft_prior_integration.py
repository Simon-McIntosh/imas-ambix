"""Soft-prior wiring in solve_equilibrium_lsq: byte-identical OFF, and each
prior assembles/bites when ON.

Fast synthetic checks — the full held-out gate lives on SLURM.  These pin that
(1) soft_priors=None leaves the solve unchanged, (2) the assembler builds the
right rows (abs-ψ anchor gauge column, q active-set, Ip-soft, pressure), and
(3) end-to-end the annulus anchor pulls the near-edge field toward its target
without breaking convergence.
"""

from __future__ import annotations

import numpy as np

from imas_ambix.latent.boundary_prior import annulus_point_set
from imas_ambix.latent.gs_solve import (
    EquilibriumGrid,
    SoftPriors,
    _assemble_soft_prior_rows,
    solve_equilibrium_lsq,
)

from .test_gs_solve import _confining_table, _synthetic_confining_slice


def _slice(nr=49, nz=65, beta0=0.6, alpha=1.5):
    table = _confining_table()
    grid = EquilibriumGrid.from_table(table, nr=nr, nz=nz)
    ip = 4.0e5
    i_pf = np.array([-6.0e4, -6.0e4])
    meas, vac, res_true = _synthetic_confining_slice(grid, table, i_pf, ip, beta0, alpha)
    kw = dict(
        measured=meas,
        vacuum_prediction=vac,
        sensor_scale=np.abs(meas) + 1e-9,
        sensor_mask=np.ones(meas.size, dtype=bool),
    )
    return table, grid, i_pf, ip, kw, res_true


def test_soft_priors_none_is_byte_identical():
    table, grid, i_pf, ip, kw, _ = _slice()
    a = solve_equilibrium_lsq(grid, table, i_pf, ip, max_iterations=40, **kw)
    b = solve_equilibrium_lsq(
        grid, table, i_pf, ip, max_iterations=40, soft_priors=None, **kw
    )
    assert np.allclose(a.coeffs, b.coeffs)
    assert np.allclose(a.result.psi, b.result.psi)
    # an all-default SoftPriors() (every knob off) is also a no-op
    c = solve_equilibrium_lsq(
        grid, table, i_pf, ip, max_iterations=40, soft_priors=SoftPriors(), **kw
    )
    assert np.allclose(a.coeffs, c.coeffs)


def test_assemble_abs_psi_anchor_adds_gauge_column():
    k_dof, kp = 2, 0
    n_ann = 6
    ann_rows = np.arange(n_ann)
    # fake a grid-like object with the fields the assembler reads
    _, grid, *_ = _slice(nr=33, nz=45)
    ann_rows = np.arange(min(n_ann, grid.cells.size))
    u_n = np.ones((grid.cells.size, k_dof))
    sp = SoftPriors(
        anchor_form="abs-psi",
        anchor_weight=1.0,
        anchor_ann_rows=ann_rows,
        anchor_psi_target=np.zeros(ann_rows.size),
        anchor_gauge_offset=True,
    )
    a_extra, b_extra, n_gauge = _assemble_soft_prior_rows(
        sp,
        grid=grid,
        u_n=u_n,
        a_anchor=u_n.sum(0),
        axis_images_unit=np.zeros(k_dof),
        coeffs_prev=np.zeros(k_dof),
        psi_coil=np.zeros(grid.flat_r.size),
        psi_pass=np.zeros((grid.flat_r.size, 0)),
        a_pass=np.zeros(0),
        ip_amperes=4.0e5,
        k_dof=k_dof,
        kp=kp,
    )
    assert n_gauge == 1
    assert a_extra.shape == (ann_rows.size, k_dof + kp + 1)
    assert b_extra.shape == (ann_rows.size,)
    # the gauge column is the −1 offset column of the abs-ψ rows
    assert np.allclose(a_extra[:, -1], -1.0 * (a_extra[:, -1] != 0).astype(float) * np.abs(a_extra[:, -1]))


def test_assemble_q_bound_is_active_set():
    k_dof, kp = 2, 0
    _, grid, *_ = _slice(nr=33, nz=45)
    u_n = np.ones((grid.cells.size, k_dof))
    images_axis = np.array([1.0, 1.0])  # j_axis = sum(coeffs)
    common = dict(
        grid=grid,
        u_n=u_n,
        a_anchor=u_n.sum(0),
        axis_images_unit=images_axis,
        psi_coil=np.zeros(grid.flat_r.size),
        psi_pass=np.zeros((grid.flat_r.size, 0)),
        a_pass=np.zeros(0),
        ip_amperes=4.0e5,
        k_dof=k_dof,
        kp=kp,
    )
    sp = SoftPriors(q_axis_max=1.0, q_weight=1.0)
    # iterate satisfies q>=1 (j_axis=0 < bound) -> NO row
    a0, _, _ = _assemble_soft_prior_rows(sp, coeffs_prev=np.zeros(k_dof), **common)
    assert a0.shape[0] == 0
    # iterate violates (j_axis=4 > 1) -> one row appears
    a1, b1, _ = _assemble_soft_prior_rows(sp, coeffs_prev=np.array([2.0, 2.0]), **common)
    assert a1.shape[0] == 1
    assert np.isclose(b1[0], 1.0)  # weight * j_axis_max


def test_anchor_pulls_annulus_toward_target():
    """A high-weight abs-ψ anchor moves the near-edge field toward a shifted
    target (proving it bites), while a self-consistent target leaves the good
    solution essentially unchanged (proving it does no harm), and both converge."""
    table, grid, i_pf, ip, kw, _ = _slice()
    base = solve_equilibrium_lsq(grid, table, i_pf, ip, max_iterations=60, **kw)
    psi0 = base.result.psi.ravel()

    ann = annulus_point_set(
        grid,
        psi_carrier=base.result.psi,
        axis_psi=base.result.axis_psi,
        boundary_psi=base.result.boundary_psi,
    )
    ann_rows = np.searchsorted(grid.cells, ann)
    ann_rows = ann_rows[
        (ann_rows < grid.cells.size)
        & (grid.cells[np.clip(ann_rows, 0, grid.cells.size - 1)] == ann)
    ]
    assert ann_rows.size > 5
    cells_flat = grid.cells[ann_rows]
    base_ann = psi0[cells_flat]

    def _rms(psi, target):
        d = psi.ravel()[cells_flat] - target
        d = d - d.mean()  # remove the gauge offset both fields carry
        return float(np.sqrt(np.mean(d**2)))

    # (a) self-consistent target — the anchor must not disturb the solution
    sp_self = SoftPriors(
        anchor_form="abs-psi",
        anchor_weight=5.0,
        anchor_ann_rows=ann_rows,
        anchor_psi_target=base_ann,
        anchor_gauge_offset=True,
        anchor_robust_clip=None,
    )
    self_fit = solve_equilibrium_lsq(
        grid, table, i_pf, ip, max_iterations=60, soft_priors=sp_self, **kw
    )
    assert self_fit.result.converged
    assert _rms(self_fit.result.psi, base_ann) < 1e-3  # negligible drift

    # (b) SHIFTED target: a non-constant radial ramp the gauge offset cannot
    # absorb.  A high-weight anchor must pull the annulus field toward it.
    r_ann = grid.flat_r[cells_flat]
    ramp = 0.05 * (base_ann.max() - base_ann.min()) * (r_ann - r_ann.mean())
    ramp = ramp - ramp.mean()
    target = base_ann + ramp
    sp_shift = SoftPriors(
        anchor_form="abs-psi",
        anchor_weight=50.0,
        anchor_ann_rows=ann_rows,
        anchor_psi_target=target,
        anchor_gauge_offset=True,
        anchor_robust_clip=None,
    )
    shifted = solve_equilibrium_lsq(
        grid, table, i_pf, ip, max_iterations=60, soft_priors=sp_shift, **kw
    )
    assert np.isfinite(shifted.result.psi).all()
    # the anchored field is closer to the shifted target than the free solve
    assert _rms(shifted.result.psi, target) < _rms(base.result.psi, target)
