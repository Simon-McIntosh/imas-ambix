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


def _table_with_interior_coil():
    """A confining exterior coil plus a genuinely in-vessel one (MAST P6U).

    ``_confining_table``'s two coils are placed near-but-not-quite (~14 cm)
    from a real MAST coil centroid, so :func:`classify_circuits` labels them
    ``inferred_passive`` and they never drive ``grid.coil_psi`` at all (the
    KNOWN-coil column list ends up empty) — the exclusion this test targets
    never fires in that fixture.  This table places one filament exactly at
    the real P2OU centroid (outside this small limiter, a smooth exterior
    source) and one exactly at the real P6U centroid — MAST's actual in-vessel
    divertor coil position, inside the limiter — each wired to its matching
    ``amc`` channel so both are classified ``known_pf`` and land in
    ``grid.coil_psi_columns``.
    """
    from imas_ambix.gs import geometry as gsg

    probes = [
        gsg.BProbe(index=i, r=1.9, z=-0.6 + 0.3 * i, angle_deg=90.0, length=0.02)
        for i in range(5)
    ]
    sensor_map = [
        gsg.SensorMapping(f"obv{i:02d}", "b_probe", i, p.r, p.z, p.angle_deg, 0.001, "")
        for i, p in enumerate(probes)
    ]
    pf = [
        gsg.PFFilament(  # P2OU: exterior, harmonic everywhere in-domain
            r=0.528, z=1.72, turns=1.0, width=0.10, height=0.10, circuit=2, xmult=1.0
        ),
        gsg.PFFilament(  # P6U: genuinely in-vessel, inside the limiter
            r=1.43, z=0.90, turns=1.0, width=0.10, height=0.10, circuit=1, xmult=1.0
        ),
    ]
    return gsg.GeometryTable(
        signature=gsg.SetupSignature(
            n_bprobe=5, n_fluxloop=0, n_pf_filament=2, n_limiter=5, digest="feed0002"
        ),
        shots=[1],
        b_probes=probes,
        flux_loops=[],
        pf_filaments=pf,
        limiter_r=[0.3, 1.65, 1.65, 0.3, 0.3],
        limiter_z=[-1.05, -1.05, 1.05, 1.05, -1.05],
        sensor_map=sensor_map,
        passive_structures=[],
        amc_current_channels=["p6u_current", "p2ou_coil_current"],
        unmatched_amb=[],
    )


def test_analytic_add_matches_greens_field_interior_coil_continuation_does_not():
    """``coil_field_mode`` arms for a coil sitting INSIDE the solve domain.

    With zero plasma current, ``"analytic-add"`` solves a trivial zero-source
    plasma problem and adds the exact finite-area coil field, so it must
    reproduce ``grid.coil_psi`` exactly everywhere, including at the coil.
    ``"boundary-continuation"`` instead solves Δ*ψ=0 with the coil's own
    boundary values as Dirichlet data — a harmonic continuation with no
    source term for the in-vessel coil — so it must drift from the true field
    near the coil (where the true field has a real, non-harmonic source)
    while agreeing far away, where the true field genuinely is harmonic (the
    exterior coil is unaffected by either arm's treatment of the interior
    one, by linearity of Δ*).
    """
    table = _table_with_interior_coil()
    grid = EquilibriumGrid.from_table(table, nr=49, nz=65)
    assert grid._coil_psi_columns.shape[1] == 2, "both coils must be KNOWN"
    i_pf = np.array([-1.2e5, 8.0e4])  # [P2OU exterior, P6U interior]
    psi_coil = grid.coil_psi(i_pf)
    zero_rhs = np.zeros((grid.nz, grid.nr))

    psi_b_add = np.zeros((grid.nz, grid.nr))  # zero plasma current -> zero BC
    psi_add = grid.solve_dirichlet(zero_rhs, psi_b_add).ravel() + psi_coil
    np.testing.assert_allclose(psi_add, psi_coil, atol=1e-9)

    psi_b_cont = np.zeros((grid.nz, grid.nr))
    psi_b_cont.ravel()[grid.edge_idx] = psi_coil[grid.edge_idx]
    psi_cont = grid.solve_dirichlet(zero_rhs, psi_b_cont).ravel()

    r0, r1, z0, z1 = grid.conductor_rects[-1]  # P6U's raw pack (last filament)
    cr, cz = 0.5 * (r0 + r1), 0.5 * (z0 + z1)
    dist = np.hypot(grid.flat_r - cr, grid.flat_z - cz)
    inside = grid.inside_limiter.ravel()
    near = inside & (dist <= 0.15)
    far = inside & (dist >= 0.5)
    assert near.any() and far.any()

    span = float(psi_coil.max() - psi_coil.min())
    rel_diff = np.abs(psi_cont - psi_coil) / span
    assert rel_diff[near].max() > 0.1, (
        "boundary-continuation must miss the in-vessel coil's real source"
    )
    assert rel_diff[far].max() < 0.08, (
        "far from the coil both arms must agree (the true field is harmonic there)"
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


def test_plasma_source_scale_matches_greens_superposition():
    """FD solve of the plasma part reproduces the Green's-function superposition.

    The Green's columns carry TOTAL flux Φ = 2π R A_φ, so the matching FD
    source is Δ*Φ = −2π μ0 R jφ.  A per-radian source (−μ0 R jφ) under-weights
    the plasma's own flux well by 2π against the coil field and boundary
    values — this test pins the scale by comparing the Dirichlet solve for a
    smooth compact current against the same current's direct Green's field.
    """
    from imas_ambix.gs.cylinder import hybrid_greens
    from imas_ambix.latent.gs_solve import MU0

    table = _confining_table()
    grid = EquilibriumGrid.from_table(table, nr=49, nz=65)
    r_c = grid.flat_r[grid.cells]
    z_c = grid.flat_z[grid.cells]
    blob = np.exp(-(((r_c - grid.r0) / 0.25) ** 2 + (z_c / 0.3) ** 2))
    i_cell = blob / blob.sum() * 4.0e5  # [A]
    psi_greens = np.zeros(grid.flat_r.size)
    for k, c in enumerate(grid.cells):
        psi_greens += i_cell[k] * hybrid_greens(
            grid.flat_r, grid.flat_z,
            float(grid.flat_r[c]), float(grid.flat_z[c]), grid.dr, grid.dz,
        )[0]
    psi_greens2d = psi_greens.reshape(grid.nz, grid.nr)
    jphi = np.zeros(grid.flat_r.size)
    jphi[grid.cells] = i_cell / (grid.dr * grid.dz)
    rhs2d = (-(2.0 * np.pi * MU0) * grid.flat_r * jphi).reshape(grid.nz, grid.nr)
    psi_b2d = np.zeros_like(rhs2d)
    psi_b2d.ravel()[grid.edge_idx] = psi_greens[grid.edge_idx]
    psi_fd = grid.solve_dirichlet(rhs2d, psi_b2d)
    span = psi_greens2d.max() - psi_greens2d.min()
    rel_rms = float(np.sqrt(np.mean((psi_fd - psi_greens2d) ** 2)) / span)
    assert rel_rms < 0.05  # a missing 2π shows up at ~0.2+


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


def test_sensor_greens_single_cell_matches_analytic():
    """The cell→sensor Green's matrix row for a B-probe equals the analytic
    projected field of a unit filament at that cell."""
    from imas_ambix.gs.operator import greens_bz_br

    table = _confining_table()
    grid = EquilibriumGrid.from_table(table, nr=33, nz=45)
    g_sens, channels = grid.sensor_greens(table)
    assert g_sens.shape == (len(channels), grid.cells.size)
    # probe 2 is vertical (angle 90 deg) at (1.35, 0.0): its response to a unit
    # current at any cell is Bz there
    probe_row = channels.index("obv02")
    c = grid.cells.size // 2
    cr, cz = grid.flat_r[grid.cells[c]], grid.flat_z[grid.cells[c]]
    bz, _br = greens_bz_br(np.array([1.35]), np.array([0.0]), float(cr), float(cz))
    np.testing.assert_allclose(g_sens[probe_row, c], bz[0], rtol=1e-9)


def test_fit_profile_recovers_generating_beta0():
    """Fitting the profile against magnetics synthesised from a KNOWN β0
    equilibrium recovers that β0 (self-consistency of the fit machinery)."""
    from imas_ambix.latent.gs_solve import fit_profile

    from imas_ambix.latent.gs_solve import solve_equilibrium_bootstrapped

    table = _confining_table()
    grid = EquilibriumGrid.from_table(table, nr=49, nz=65)
    ip = 4.0e5
    i_pf = np.array([-6.0e4, -6.0e4])
    true_beta0 = 0.7
    # synthesise through the SAME two-stage path fit_profile solves with —
    # isolates parameter recovery from Picard path-to-path variation
    res = solve_equilibrium_bootstrapped(grid, i_pf, ip, beta0=true_beta0)
    assert res.converged
    g_sens, channels = grid.sensor_greens(table)
    # synthetic "measured" magnetics = coil vacuum part + plasma part; the coil
    # part comes from the same subdivided-filament greens the solver uses,
    # evaluated at the sensors directly (the synthetic coils are not amc-mapped)
    from imas_ambix.gs.operator import greens_bz_br

    vac = np.zeros(len(channels))
    for k, m in enumerate(table.sensor_map):
        ang = np.deg2rad(m.angle_deg if m.angle_deg is not None else 90.0)
        for f, cur in zip(table.pf_filaments, i_pf, strict=True):
            bz, br = greens_bz_br(np.array([m.r]), np.array([m.z]), f.r, f.z)
            vac[k] += cur * (br[0] * np.cos(ang) + bz[0] * np.sin(ang))
    meas = vac + g_sens @ res.cell_currents
    scale = np.abs(meas) + 1e-9
    fit = fit_profile(
        grid,
        table,
        i_pf=i_pf,
        ip_amperes=ip,
        measured=meas,
        vacuum_prediction=vac,
        sensor_scale=scale,
        sensor_mask=np.ones(meas.size, dtype=bool),
        beta0_grid=(0.3, 0.5, 0.7, 0.9),
    )
    assert fit.result.converged
    assert fit.beta0 == true_beta0  # grid fit must pick the generating value
    assert fit.cost < 1e-3  # near-perfect match at the true parameters


def test_conductor_exclusion_in_topology_reads():
    """Critical points and the fallback axis read must skip winding packs.

    The exact finite-area coil field has genuine extrema at every conductor;
    a pack straddling the limiter contour must not capture the axis read.
    """
    table = _confining_table()
    grid = EquilibriumGrid.from_table(table, nr=41, nz=57)
    # pack rectangles registered, dilated by one grid cell
    assert grid.conductor_rects.shape == (2, 4)
    clear = grid.clear_of_conductors(
        np.array([1.1, 1.1, 0.9]), np.array([1.0, -1.0, 0.0])
    )
    assert not clear[0] and not clear[1]  # at the packs
    assert clear[2]  # plasma centre untouched
    # the grid-level topology-candidate mask stays inside the limiter
    assert grid.topology_candidate.sum() > 0
    assert not (grid.topology_candidate & ~grid.inside_limiter.ravel()).any()


def test_bootstrapped_solve_matches_direct_on_confining_config():
    """Where plain Picard already converges, the two-stage bootstrap must land
    on the same equilibrium (same axis, same Ip)."""
    from imas_ambix.latent.gs_solve import solve_equilibrium_bootstrapped

    table = _confining_table()
    grid = EquilibriumGrid.from_table(table, nr=49, nz=65)
    ip = 4.0e5
    i_pf = np.array([-6.0e4, -6.0e4])
    direct = solve_equilibrium(grid, i_pf, ip, beta0=0.5)
    boot = solve_equilibrium_bootstrapped(grid, i_pf, ip, beta0=0.5)
    assert direct.converged and boot.converged
    np.testing.assert_allclose(boot.axis, direct.axis, atol=2e-2)
    np.testing.assert_allclose(boot.cell_currents.sum(), ip, rtol=1e-6)


def test_boundary_continuation_arm_reproduces_legacy_structure():
    """The diagnostic arm solves the TOTAL field through the FD problem; on a
    configuration with all coils outside the solve domain both arms agree."""
    table = _confining_table()
    grid = EquilibriumGrid.from_table(table, nr=49, nz=65)
    ip = 4.0e5
    i_pf = np.array([-6.0e4, -6.0e4])
    add = solve_equilibrium(grid, i_pf, ip, beta0=0.5)
    cont = solve_equilibrium(
        grid, i_pf, ip, beta0=0.5, coil_field_mode="boundary-continuation"
    )
    assert add.converged and cont.converged
    # coils sit OUTSIDE the limiter here, where harmonic continuation is exact
    np.testing.assert_allclose(cont.axis, add.axis, atol=2e-2)
