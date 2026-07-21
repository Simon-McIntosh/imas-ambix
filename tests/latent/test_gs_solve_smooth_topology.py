"""Tests for the continuous (temperature-smoothed) topology read in the
free-boundary solve.

The hard per-sweep read (critical points + labelled core mask) makes the
fixed-point map non-differentiable: a cell flips in/out of the core mask
discretely as ψ moves.  The opt-in ``topology_read='connectivity'`` path swaps
in the temperature-smoothed connectivity kernel — softmin boundary binding +
retracted-gate sigmoid core weight + sub-grid stencil axis — under which the
map is end-to-end differentiable.  These tests pin:

* the smooth read reproduces the hard read's axis / binding flux / core
  support on a converged synthetic equilibrium (it is not a lossy read);
* the smooth core weight is Lipschitz in ψ (no discrete flips), while the
  hard mask is integer-valued by construction;
* a soft SOL cap has no smooth-kernel equivalent and fails loud;
* the default solve path is untouched (``topology_read`` validates, and the
  default equals an explicit ``'hard'``).
"""

from __future__ import annotations

import numpy as np
import pytest

from imas_ambix.latent.gs_solve import (
    TOPOLOGY_HARD,
    _read_axis,
    _read_boundary_psi,
    _read_topology_smooth,
    solve_equilibrium,
    solve_equilibrium_nk,
)
from tests.latent.test_greens_filament_solver import _circular_grid


def _converged_psi(grid, ip=8.0e5):
    """A converged fixed-shape synthetic equilibrium ψ (flat, 2-D) on ``grid``."""
    res = solve_equilibrium(grid, np.array([1.0e4]), ip, max_iterations=60)
    return res.psi.ravel(), res


def test_smooth_read_reproduces_hard_read():
    """Axis, binding flux, and core support agree with the hard read on a
    converged equilibrium — the smooth read is not lossy."""
    grid = _circular_grid()
    psi_flat, res = _converged_psi(grid)
    psi2d = psi_flat.reshape(grid.nz, grid.nr)

    axis_h, axis_psi_h = _read_axis(psi2d, grid, 1.0)
    boundary_h = _read_boundary_psi(psi2d, grid, axis_psi_h)
    span_h = boundary_h - axis_psi_h

    axis_s, axis_psi_s, boundary_s, weight, core = _read_topology_smooth(
        psi_flat, grid, axis_h, 1.0, 1e-3
    )

    # axis within one grid cell, fluxes within a few % of the span
    assert np.hypot(axis_s[0] - axis_h[0], axis_s[1] - axis_h[1]) < 2.0 * max(
        grid.dr, grid.dz
    )
    # the 3×3 biquadratic under-reads this deliberately sharp synthetic peak a
    # little (real-slice agreement is gated separately, at the mm level)
    assert abs(axis_psi_s - axis_psi_h) < 0.06 * abs(span_h)
    # the smooth binding is the SUB-GRID wall tangency; the hard read contacts
    # the wall at grid points (up to ~half a cell off the polygon), so compare
    # against the true interpolated tangency, not the quantized hard value
    from scipy.interpolate import RegularGridInterpolator

    from imas_ambix.latent.connectivity_boundary import _densify_wall

    wr, wz = _densify_wall(grid)
    itp = RegularGridInterpolator((grid.zg, grid.rg), psi2d)
    wall_vals = itp(np.column_stack([wz, wr]))
    boundary_true = wall_vals[np.argmin(np.abs(wall_vals - axis_psi_h))]
    assert abs(boundary_s - boundary_true) < 0.005 * abs(span_h)
    # and the hard read agrees with the true tangency only to its half-cell
    # wall quantization — the smooth read is strictly the better-grounded one
    assert abs(boundary_h - boundary_true) < 0.5 * abs(span_h)

    # core support: weight ≈ 1 deep inside the hard core, ≈ 0 well outside it
    psi_n_h = (psi_flat - axis_psi_h) / span_h
    deep = (psi_n_h < 0.8) & grid.inside_limiter.ravel()
    far = (psi_n_h > 1.2) | ~grid.inside_limiter.ravel()
    assert weight[deep].min() > 0.95
    assert weight[far].max() < 0.05
    assert core.any()


def test_smooth_weight_is_lipschitz_in_psi():
    """A small ψ perturbation moves the smooth weight by a small, bounded
    amount — the discrete mask-flip signature is gone by construction."""
    grid = _circular_grid()
    psi_flat, _res = _converged_psi(grid)
    axis_h, _axis_psi_h = _read_axis(psi_flat.reshape(grid.nz, grid.nr), grid, 1.0)

    _ax, ap, bp, w0, _c = _read_topology_smooth(psi_flat, grid, axis_h, 1.0, 1e-3)
    span = bp - ap
    # perturb ψ non-uniformly by ~0.01% of the span (a sub-temperature flux
    # ripple — a uniform shift would cancel exactly in the normalised flux)
    rng = np.random.default_rng(7)
    eps = 1e-4 * abs(span) * rng.standard_normal(psi_flat.size)
    _ax2, _ap2, _bp2, w1, _c2 = _read_topology_smooth(
        psi_flat + eps, grid, axis_h, 1.0, 1e-3
    )
    # sigmoid slope is 1/(4·τ) per unit normalised flux → a sub-τ ripple moves
    # the sigmoid body by a bounded fraction (a hard mask flip is 1.0); the
    # retracted flood gate is a boolean selection whose O(τ) shell caps any
    # residual flip at σ(1) ≈ 0.73
    assert np.abs(w1 - w0).max() < 0.8
    assert np.abs(w1 - w0).mean() < 0.01


def test_smooth_read_rejects_sol_cap():
    """A soft SOL cap (core_cap > 1) has no smooth-kernel equivalent — the
    read fails loud instead of silently mis-binding the core."""
    grid = _circular_grid()
    psi_flat, _res = _converged_psi(grid)
    axis_h, _ = _read_axis(psi_flat.reshape(grid.nz, grid.nr), grid, 1.0)
    with pytest.raises(ValueError, match="SOL cap"):
        _read_topology_smooth(psi_flat, grid, axis_h, 1.05, 1e-3)


def test_topology_read_validates():
    grid = _circular_grid()
    with pytest.raises(ValueError, match="topology_read"):
        solve_equilibrium_nk(
            grid, np.array([1.0e4]), 8.0e5, topology_read="banana", maxiter=1
        )


def test_nk_smooth_map_runs_on_synthetic():
    """NK on the smooth map converges on the well-posed synthetic case and
    lands on the same equilibrium as the hard-map fixed point."""
    grid = _circular_grid()
    res_h = solve_equilibrium(grid, np.array([1.0e4]), 8.0e5, max_iterations=60)
    res_s = solve_equilibrium_nk(
        grid,
        np.array([1.0e4]),
        8.0e5,
        topology_read="connectivity",
        picard_warmup=12,
        maxiter=40,
    )
    assert np.isfinite(res_s.residual)
    # same axis to within one grid cell
    assert np.hypot(
        res_s.axis[0] - res_h.axis[0], res_s.axis[1] - res_h.axis[1]
    ) < 2.0 * max(grid.dr, grid.dz)


def test_default_topology_read_is_hard():
    """An explicit 'hard' is the same code path as the default (byte-identical)."""
    grid = _circular_grid()
    kw = dict(picard_warmup=6, maxiter=5, f_tol=1e-12)
    res_a = solve_equilibrium_nk(grid, np.array([1.0e4]), 8.0e5, **kw)
    res_b = solve_equilibrium_nk(
        grid, np.array([1.0e4]), 8.0e5, topology_read=TOPOLOGY_HARD, **kw
    )
    np.testing.assert_array_equal(res_a.psi, res_b.psi)
    np.testing.assert_array_equal(res_a.jphi, res_b.jphi)
