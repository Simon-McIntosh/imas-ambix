"""Tests for the free-boundary Grad-Shafranov equilibrium solver.

The solver is the force-balanced ψ decoder: given the KNOWN coil currents, the
measured plasma current, and low-DOF profile parameters, it solves

    Δ*ψ = −μ0 R jφ(ψ_N; θ)   inside the limiter,   jφ = 0 outside the LCFS,

by under-relaxed Picard iteration with Dirichlet boundary values re-evaluated
each iteration from the current sources (Green's functions).  Correctness is
pinned analytically — no EFIT anywhere:

* the DISCRETE Δ* stencil is validated against a manufactured solution
  (plug an analytic ψ into Δ*, solve with the resulting RHS + exact BCs, and
  recover ψ to second-order accuracy);
* the VACUUM limit is validated against the Green's-function coil field (the
  interior solve of Δ*ψ = 0 with Green's BCs must reproduce the Green's ψ);
* the PICARD fixed point on a confining synthetic coil set must converge to an
  equilibrium whose axis is an interior O-point, whose total current equals
  the prescribed Ip, and whose current vanishes outside the core;
* the profile ansatz is FIREWALL-clean by construction (static check: the
  module imports nothing from the evaluator/EFIT side).
"""

from __future__ import annotations

import numpy as np

from imas_ambix.latent.gs_solve import (
    EquilibriumGrid,
    profile_jphi_shape,
    solve_equilibrium,
)


def _confining_table():
    """Synthetic machine: rectangular limiter + a vertical-field coil pair.

    Two coils above/below the midplane carrying opposite-sign current to the
    plasma produce the inward vertical field a positive-Ip plasma needs for
    radial force balance — the minimal confining configuration.
    """
    from imas_ambix.gs import geometry as gsg

    probes = [
        gsg.BProbe(index=i, r=1.35, z=-0.6 + 0.3 * i, angle_deg=90.0, length=0.02)
        for i in range(5)
    ]
    sensor_map = [
        gsg.SensorMapping(f"obv{i:02d}", "b_probe", i, p.r, p.z, p.angle_deg, 0.001, "")
        for i, p in enumerate(probes)
    ]
    pf = [
        gsg.PFFilament(
            r=1.1, z=1.0, turns=1.0, width=0.06, height=0.06, circuit=1, xmult=1.0
        ),
        gsg.PFFilament(
            r=1.1, z=-1.0, turns=1.0, width=0.06, height=0.06, circuit=2, xmult=1.0
        ),
    ]
    return gsg.GeometryTable(
        signature=gsg.SetupSignature(
            n_bprobe=5, n_fluxloop=0, n_pf_filament=2, n_limiter=5, digest="feed0000"
        ),
        shots=[1],
        b_probes=probes,
        flux_loops=[],
        pf_filaments=pf,
        limiter_r=[0.35, 1.45, 1.45, 0.35, 0.35],
        limiter_z=[-0.85, -0.85, 0.85, 0.85, -0.85],
        sensor_map=sensor_map,
        passive_structures=[],
        amc_current_channels=[],
        unmatched_amb=[],
    )


def test_delta_star_stencil_manufactured_solution():
    """Δ* of ψ = R⁴ is 8R² analytically; solve with that RHS + exact BCs and
    recover ψ to second order."""
    table = _confining_table()
    grid = EquilibriumGrid.from_table(table, nr=49, nz=65)
    rr = grid.mesh_r
    psi_exact = rr**4
    rhs_interior = 8.0 * rr**2  # Δ*(R^4) = R d/dR(1/R · 4R^3) = R·d/dR(4R^2) = 8R^2
    psi_num = grid.solve_dirichlet(rhs_interior, psi_exact)
    err = np.abs(psi_num - psi_exact).max() / np.abs(psi_exact).max()
    assert err < 5e-3  # second-order stencil on a modest grid


def test_vacuum_limit_matches_greens_coil_field():
    """Interior solve of Δ*ψ=0 with Green's BCs reproduces the Green's coil ψ."""
    table = _confining_table()
    grid = EquilibriumGrid.from_table(table, nr=41, nz=57)
    i_pf = np.array([-3.0e4, -3.0e4])
    psi_coil = grid.coil_psi(i_pf).reshape(grid.nz, grid.nr)
    psi_num = grid.solve_dirichlet(np.zeros_like(psi_coil), psi_coil)
    err = np.abs(psi_num - psi_coil).max() / max(np.abs(psi_coil).max(), 1e-12)
    assert err < 5e-3


def test_picard_converges_to_confined_equilibrium():
    """Full free-boundary solve: interior axis, Ip conserved, current confined."""
    table = _confining_table()
    grid = EquilibriumGrid.from_table(table, nr=49, nz=65)
    ip = 4.0e5  # A
    i_pf = np.array([-6.0e4, -6.0e4])  # confining vertical field
    res = solve_equilibrium(grid, i_pf, ip, beta0=0.5)
    assert res.converged, f"Picard did not converge (residual {res.residual:.2e})"
    ar, az = res.axis
    assert 0.35 < ar < 1.45 and -0.85 < az < 0.85  # interior axis
    assert abs(az) < 0.15  # up-down symmetric configuration → axis near midplane
    np.testing.assert_allclose(res.cell_currents.sum(), ip, rtol=1e-6)
    # current confined: nothing outside the core region
    outside = ~res.core_mask
    assert np.abs(res.jphi.ravel()[outside.ravel()]).max() == 0.0


def test_profile_shape_positive_and_normalised():
    """The jφ(ψ_N) ansatz is positive inside, zero at/beyond the boundary."""
    psin = np.linspace(0, 1.2, 100)
    r = np.full_like(psin, 0.9)
    shape = profile_jphi_shape(psin, r, r0=0.9, beta0=0.4)
    inside = psin < 1.0
    assert (shape[inside] > 0).all()
    assert (shape[~inside] == 0).all()


def test_firewall_static_no_evaluator_imports():
    """The solver module must not import anything from the EFIT/evaluator side."""
    from pathlib import Path

    import imas_ambix.latent.gs_solve as m

    src = Path(m.__file__).read_text()
    for banned in ("efit_referee", "equilibrium_labels", "worldmodel"):
        assert banned not in src, f"gs_solve imports the firewalled {banned}"
