"""Verification pins for the plasma screening circuit.

Four mandatory pins, all on analytic fixtures (no shot data):

1. **1D-limit identity** — on a shared large-aspect nested-circle case the
   flux-surface-averaged filament circuit reproduces ``diffuse_psi`` (itself
   TORAX-verified) to discretisation error, for the same η, geometry and
   emergent Ip(t): current dynamics and flux diffusion are ONE system.
2. **Analytic skin time** — a thin toroidal current shell built through the
   module's own tiling path decays at the textbook
   τ = μ0·(ln(8R0/r_s) − 2)·A_cross/(2πη), and a fast voltage drive puts the
   early current on the plasma surface (the skin) before resistive
   penetration relaxes it to the conductance-weighted steady state.
3. **Exact-ZOH trajectory** — the screening-mode integration is exact for
   piecewise-linear flux + piecewise-constant voltage drives (coarse ==
   dense), and second-order on smooth drives.
4. **Edge localisation + machine agnosticism** — screening modes are
   zero-net-current and edge-weighted beyond the uniform share, and a
   uniform geometric rescale leaves mode shapes invariant while mapping
   τ → τ·s² (L ∝ s, R ∝ 1/s — exact for the cylinder kernel).
"""

from __future__ import annotations

import numpy as np
import pytest

from imas_ambix.latent import plasma_screening as ps
from imas_ambix.latent.current_diffusion import EtaProfile, FluxSurfaceGeometry
from imas_ambix.latent.gs_solve import EquilibriumGrid

R0 = 8.0  # large-aspect major radius [m] — toroidal corrections O(a/R0)² ~ 0.4%
A_MINOR = 0.5
B0 = 1.0
ETA_FLAT = EtaProfile(eta0=3.0e-7, contrast=0.0, shape=1.0)


def _nested_circle_grid(r0: float = R0, a: float = A_MINOR, n: int = 57):
    """Analytic nested-circle machine: circular limiter, no coils."""
    lim_t = np.linspace(0.0, 2.0 * np.pi, 65)
    lim_r = r0 + 1.24 * a * np.cos(lim_t)
    lim_z = 1.24 * a * np.sin(lim_t)
    rg = np.linspace(r0 - 1.4 * a, r0 + 1.4 * a, n)
    zg = np.linspace(-1.4 * a, 1.4 * a, n)
    return EquilibriumGrid(
        rg=rg,
        zg=zg,
        limiter_r=lim_r,
        limiter_z=lim_z,
        coil_psi_columns=np.zeros((n * n, 0)),
        r0=r0,
    )


def _rho_map(grid, r0: float, a: float) -> np.ndarray:
    return np.hypot(grid.flat_r - r0, grid.flat_z) / a


def _fixture_circuit(grid, r0: float = R0, a: float = A_MINOR, n_rad=12, n_pol=8):
    rho = _rho_map(grid, r0, a)
    psi_n = np.clip(rho**2, 0.0, 1.0)
    core = (rho <= 1.0).reshape(grid.nz, grid.nr)
    return ps.build_plasma_circuit_from_state(
        grid, psi_n, core, (r0, 0.0), n_rad=n_rad, n_pol=n_pol
    )


def _analytic_geometry(n_rho: int = 24) -> FluxSurfaceGeometry:
    """Exact 1D metrics of concentric circular surfaces at large aspect ratio."""
    rho_face = np.linspace(0.0, 1.0, n_rho + 1)
    rho_cell = 0.5 * (rho_face[:-1] + rho_face[1:])
    return FluxSurfaceGeometry(
        rho_face=rho_face,
        rho_cell=rho_cell,
        psi_face=np.zeros(n_rho + 1),  # zero initial current — uniform flux
        psi_n_face=rho_face**2,
        psi_n_cell=rho_cell**2,
        vpr_face=4.0 * np.pi**2 * R0 * A_MINOR**2 * rho_face,
        vpr_cell=4.0 * np.pi**2 * R0 * A_MINOR**2 * rho_cell,
        g2_face=16.0 * np.pi**4 * A_MINOR**2 * rho_face**2,
        g3_face=np.full(n_rho + 1, 1.0 / R0**2),
        g3_cell=np.full(n_rho, 1.0 / R0**2),
        f_face=np.full(n_rho + 1, R0 * B0),
        f_cell=np.full(n_rho, R0 * B0),
        b2_cell=np.full(n_rho, B0**2),
        inv_r_cell=np.full(n_rho, 1.0 / R0),
        phi_b=B0 * np.pi * A_MINOR**2,
        r0=R0,
        ip_amperes=0.0,
        axis_psi=0.0,
        boundary_psi=0.0,
        volume=2.0 * np.pi**2 * R0 * A_MINOR**2,
        q_face=np.ones(n_rho + 1),
        # the cylinder kernel's physical ψ DECREASES outward for positive
        # current (the MAST-sign orientation) — the fixture must match it or
        # the raw flux comparison sees a mirror image (the current profile is
        # blind to this: the flat-η diffusion is odd-symmetric in ψ)
        flux_sign=-1.0,
    )


@pytest.fixture(scope="module")
def nested_grid():
    return _nested_circle_grid()


@pytest.fixture(scope="module")
def nested_circuit(nested_grid):
    return _fixture_circuit(nested_grid)


# ---------------------------------------------------------------------------
# pin 1 — the 1D-limit identity: circuit ≡ diffuse_psi on the shared case
# ---------------------------------------------------------------------------


def test_flux_surface_limit_matches_diffuse_psi(nested_grid, nested_circuit):
    from imas_ambix.latent.current_diffusion import diffuse_psi

    circuit = nested_circuit
    times = np.linspace(0.0, 0.06, 121)  # fast against τ_a = μ0 a²/η ≈ 1 s
    u = 1.0  # constant loop voltage [V] — exactly ZOH-representable
    i_patch = ps.evolve_patch_currents(
        circuit,
        ETA_FLAT,
        times,
        i0=np.zeros(circuit.n_patches),
        loop_voltage=np.full(times.size, u),
    )
    ip_t = i_patch.sum(axis=1)
    assert ip_t[-1] > 0.0

    geo = _analytic_geometry(n_rho=24)
    step = diffuse_psi(geo, ETA_FLAT, t_grid=times, ip_of_t=ip_t)
    i_1d = geo.enclosed_current(step["psi_face"][-1])

    # circuit cumulative enclosed current at the face radii (cell-resolved)
    t = circuit.tiling
    cell_cur = t.share * i_patch[-1][t.owner]
    rho_cell = np.sqrt(np.clip(circuit.cell_psi_n, 0.0, 1.0))
    order = np.argsort(rho_cell)
    cum = np.cumsum(cell_cur[order])
    i_circ = np.interp(geo.rho_face, rho_cell[order], cum, left=0.0)

    ip_end = float(ip_t[-1])
    err = (i_circ[2:-1] - i_1d[2:-1]) / ip_end
    assert float(np.sqrt(np.mean(err**2))) < 0.05

    # the pin must be non-trivial: a genuine skin formed (enclosed-current
    # profile hollow relative to the fully-penetrated steady state)
    i_ss = ps.steady_state_currents(circuit, ETA_FLAT, ip_end)
    cum_ss = np.cumsum((t.share * i_ss[t.owner])[order])
    i_half = float(np.interp(0.7, rho_cell[order], cum))
    i_half_ss = float(np.interp(0.7, rho_cell[order], cum_ss))
    assert i_half < 0.6 * i_half_ss  # interior strongly screened at t ≪ τ


def test_interior_flux_profile_matches_diffuse_psi(nested_grid, nested_circuit):
    from imas_ambix.latent.current_diffusion import diffuse_psi

    circuit = nested_circuit
    times = np.linspace(0.0, 0.06, 121)
    i_patch = ps.evolve_patch_currents(
        circuit,
        ETA_FLAT,
        times,
        i0=np.zeros(circuit.n_patches),
        loop_voltage=np.ones(times.size),
    )
    ip_t = i_patch.sum(axis=1)
    geo = _analytic_geometry(n_rho=24)
    step = diffuse_psi(geo, ETA_FLAT, t_grid=times, ip_of_t=ip_t)

    # circuit flux at the patches (the external uniform term cancels in the
    # difference to the outermost ring, as does the 1D gauge)
    psi_patch = circuit.lmat @ i_patch[-1]
    rho_patch = np.sqrt(np.clip(circuit.tiling.psi_n, 0.0, 1.0))
    outer = rho_patch > rho_patch.max() - 1.0 / 12.0
    psi_1d = step["psi_face"][-1]
    psi_1d_at = np.interp(rho_patch, geo.rho_face, psi_1d)
    # reference BOTH sides to the same outer-ring patch mean (kills the two
    # gauges without injecting the steep edge-bin offset into the residual)
    d_circ = psi_patch - psi_patch[outer].mean()
    d_1d = psi_1d_at - psi_1d_at[outer].mean()
    span = float(np.ptp(d_1d))
    assert span > 0.0
    resid = d_circ - d_1d
    rms = float(np.sqrt(np.mean(resid**2)))
    assert rms < 0.08 * span  # steep edge-bin averaging dominates the tail

    # the residual is DISCRETISATION-limited: a finer radial tiling shrinks it
    fine = _fixture_circuit(nested_grid, n_rad=18, n_pol=8)
    i_fine = ps.evolve_patch_currents(
        fine,
        ETA_FLAT,
        times,
        i0=np.zeros(fine.n_patches),
        loop_voltage=np.ones(times.size),
    )
    step_f = diffuse_psi(geo, ETA_FLAT, t_grid=times, ip_of_t=i_fine.sum(axis=1))
    psi_p_f = fine.lmat @ i_fine[-1]
    rho_p_f = np.sqrt(np.clip(fine.tiling.psi_n, 0.0, 1.0))
    outer_f = rho_p_f > rho_p_f.max() - 1.0 / 18.0
    psi_1d_f = np.interp(rho_p_f, geo.rho_face, step_f["psi_face"][-1])
    resid_f = (psi_p_f - psi_p_f[outer_f].mean()) - (
        psi_1d_f - psi_1d_f[outer_f].mean()
    )
    assert float(np.sqrt(np.mean(resid_f**2))) < rms


# ---------------------------------------------------------------------------
# pin 2 — analytic thin-shell skin time + surface-current screening
# ---------------------------------------------------------------------------


def test_thin_shell_decay_time_is_analytic(nested_grid):
    grid = nested_grid
    rho = _rho_map(grid, R0, A_MINOR)
    r_s = 0.4  # shell minor radius [m]
    dr_half = 0.55 * grid.dr
    shell = (np.abs(rho * A_MINOR - r_s) <= dr_half).reshape(grid.nz, grid.nr)
    shell &= grid.inside_limiter
    psi_n = np.clip(rho**2, 0.0, 1.0)
    # n_rad=1 → the whole selection is ONE patch (the uniform shell circuit)
    circuit = ps.build_plasma_circuit_from_state(
        grid, psi_n, shell, (R0, 0.0), n_rad=1, n_pol=1
    )
    assert circuit.n_patches == 1
    tau, _v = ps.circuit_eigensystem(circuit, ETA_FLAT)
    a_cross = circuit.tiling.cell_index.size * circuit.cell_area
    tau_analytic = (
        ps.MU0
        * (np.log(8.0 * R0 / r_s) - 2.0)
        * a_cross
        / (2.0 * np.pi * ETA_FLAT.eta0)
    )
    assert abs(float(tau[0]) - tau_analytic) / tau_analytic < 0.08


def test_fast_drive_puts_current_on_the_surface(nested_circuit):
    """The skin effect IS circuit screening: early current lives at the edge,
    late current relaxes to the conductance-weighted steady state."""
    circuit = nested_circuit
    tau_a = ps.MU0 * A_MINOR**2 / ETA_FLAT.eta0  # ≈ 1 s resistive scale
    rho_patch = np.sqrt(np.clip(circuit.tiling.psi_n, 0.0, 1.0))
    outer = rho_patch > 0.8

    # early (t ≪ τ): the drive's current is carried by the outermost shells
    times = np.linspace(0.0, 2e-3, 21)
    i_early = ps.evolve_patch_currents(
        circuit,
        ETA_FLAT,
        times,
        i0=np.zeros(circuit.n_patches),
        loop_voltage=np.ones(times.size),
    )[-1]
    frac_outer_early = float(i_early[outer].sum() / i_early.sum())
    assert frac_outer_early > 0.8

    # late (t ≫ τ): the distribution equals the resistive steady state
    times = np.linspace(0.0, 12.0 * tau_a, 400)
    i_late = ps.evolve_patch_currents(
        circuit,
        ETA_FLAT,
        times,
        i0=np.zeros(circuit.n_patches),
        loop_voltage=np.ones(times.size),
    )[-1]
    i_ss = ps.steady_state_currents(circuit, ETA_FLAT, float(i_late.sum()))
    assert float(np.abs(i_late - i_ss).max() / np.abs(i_ss).max()) < 0.05
    frac_outer_late = float(i_late[outer].sum() / i_late.sum())
    assert frac_outer_late < frac_outer_early - 0.3  # penetration happened


# ---------------------------------------------------------------------------
# pin 3 — exact-ZOH trajectory: coarse == dense for the ZOH drive classes
# ---------------------------------------------------------------------------


def test_zoh_trajectory_exact_for_piecewise_linear_drive(nested_grid, nested_circuit):
    eta = ETA_FLAT
    basis = ps.screening_eigenbasis(
        nested_grid,
        nested_circuit,
        eta,
        np.zeros((1, nested_grid.cells.size)),
        k=3,
    )
    t_coarse = np.linspace(0.0, 0.08, 9)
    t_dense = np.linspace(0.0, 0.08, 401)
    slope = np.linspace(1.0, 3.0, basis.n_modes)

    def psi_of(t):
        return np.outer(t, slope)  # piecewise-linear (globally linear) flux

    a_coarse = ps.screening_trajectory(basis, t_coarse)
    assert np.allclose(a_coarse, 0.0)  # no drive → no state

    from imas_ambix.latent.temporal_operator import integrate_eddy_ode

    ac, _ = integrate_eddy_ode(basis.tau, t_coarse, psi_of(t_coarse))
    ad, _ = integrate_eddy_ode(basis.tau, t_dense, psi_of(t_dense))
    ad_at = np.vstack([np.interp(t_coarse, t_dense, ad[:, m]) for m in range(3)]).T
    assert np.abs(ac - ad_at).max() < 1e-10 * max(np.abs(ad).max(), 1e-30)

    # smooth (quadratic) drive: halving the step shrinks the error ~4×
    def run(n):
        t = np.linspace(0.0, 0.08, n)
        a, _ = integrate_eddy_ode(basis.tau, t, np.outer(t**2, slope))
        return float(a[-1, 0])

    exact = run(4001)
    e1 = abs(run(11) - exact)
    e2 = abs(run(21) - exact)
    assert e1 / max(e2, 1e-30) > 3.0


# ---------------------------------------------------------------------------
# pin 4 — edge-localised, zero-net, machine-agnostic screening modes
# ---------------------------------------------------------------------------


def test_screening_modes_zero_net_and_edge_weighted(nested_grid, nested_circuit):
    basis = ps.screening_eigenbasis(
        nested_grid,
        nested_circuit,
        ETA_FLAT,
        np.zeros((1, nested_grid.cells.size)),
        k=3,
    )
    assert basis.n_modes == 3
    # zero net current — exact by construction (subspace deflation)
    net = np.abs(basis.i_cells.sum(axis=0))
    assert net.max() < 1e-9 * np.abs(basis.i_cells).sum(axis=0).max()

    # the leading (slowest) mode is edge-weighted beyond the uniform share
    rho_cell = np.sqrt(np.clip(nested_circuit.cell_psi_n, 0.0, 1.0))
    w_mode = np.abs(basis.i_cells[nested_circuit.tiling.cell_index, 0])
    mean_rho_mode = float((w_mode * rho_cell).sum() / w_mode.sum())
    mean_rho_area = float(rho_cell.mean())
    assert mean_rho_mode > mean_rho_area

    # sidecar contract: unit whitened norm columns, physical→fit scale carried
    scale = np.full(1, 0.5)
    side = ps.screening_sidecar(basis, scale)
    assert side["g_cols"].shape == (1, 3)
    assert side["psi_cols"].shape == (nested_grid.flat_r.size, 3)
    assert side["amp_scale"].shape == (3,)


# ---------------------------------------------------------------------------
# pin 5 — coupled plasma+external system: one circuit carries ALL the dΦ/dt
# terms, and the flux ledger closes from a zero (pre-breakdown) state
# ---------------------------------------------------------------------------


def _external_ring(grid, r_ring: float, z_ring: float, w: float = 0.04):
    """One vessel-like external ring: grid flux column + analytic self-L."""
    from imas_ambix.gs.cylinder import hybrid_greens

    psi_col, _br, _bz = hybrid_greens(grid.flat_r, grid.flat_z, r_ring, z_ring, w, w)
    psi_self, _b1, _b2 = hybrid_greens(
        np.array([r_ring]), np.array([z_ring]), r_ring, z_ring, w, w
    )
    return psi_col[:, np.newaxis], float(psi_self[0])


def test_coupled_matches_uncoupled_at_zero_cross_linkage(nested_grid, nested_circuit):
    circuit = nested_circuit
    _col, l_self = _external_ring(nested_grid, R0, 0.68)
    coupled = ps.build_coupled_circuit(
        circuit,
        l_ext=np.array([[l_self]]),
        r_ext=np.array([1.0e-4]),
        m_ext_coil=np.zeros((1, 0)),
        m_patch_ext=np.zeros((circuit.n_patches, 1)),
    )
    times = np.linspace(0.0, 0.05, 41)
    u = np.ones(times.size)
    i_plasma = ps.evolve_patch_currents(
        circuit, ETA_FLAT, times, i0=np.zeros(circuit.n_patches), loop_voltage=u
    )
    i_coupled = ps.evolve_coupled(
        coupled, ETA_FLAT, times, i0=np.zeros(coupled.n_total), loop_voltage=u
    )
    p = circuit.n_patches
    ref = np.abs(i_plasma[-1]).max()
    assert np.abs(i_coupled[:, :p] - i_plasma).max() < 1e-6 * ref
    # undriven, uncoupled external ring stays exactly quiescent
    assert np.abs(i_coupled[:, p:]).max() < 1e-12 * ref


def test_coupled_external_ring_screens_the_plasma_swing(nested_grid, nested_circuit):
    """Lenz: a fast plasma current swing induces an OPPOSING external eddy,
    which then decays on the ring's own L/R time once the drive stops."""
    circuit = nested_circuit
    col, l_self = _external_ring(nested_grid, R0, 0.68)
    m_pe = ps.patch_external_linkage(nested_grid, circuit.tiling, col)
    assert m_pe.shape == (circuit.n_patches, 1)
    assert (m_pe > 0).all()  # coaxial rings link positively
    coupled = ps.build_coupled_circuit(
        circuit,
        l_ext=np.array([[l_self]]),
        r_ext=np.array([2.0e-5]),
        m_ext_coil=np.zeros((1, 0)),
        m_patch_ext=m_pe,
    )
    p = circuit.n_patches
    times = np.linspace(0.0, 2e-3, 41)  # fast against both time scales
    i_state = ps.evolve_coupled(
        coupled,
        ETA_FLAT,
        times,
        i0=np.zeros(coupled.n_total),
        loop_voltage=np.ones(times.size),
    )
    ip_end = float(i_state[-1, :p].sum())
    i_ring = float(i_state[-1, p])
    assert ip_end > 0.0
    assert i_ring < 0.0  # opposing (screening) eddy
    # and the shoot-to-Ip drive lands the PLASMA total, not the grand total
    u, i_end = ps.coupled_loop_voltage_for_ip(
        coupled,
        ETA_FLAT,
        times,
        i0=np.zeros(coupled.n_total),
        ip_target=1.0e5,
    )
    assert abs(float(i_end[:p].sum()) - 1.0e5) < 1e-3 * 1.0e5
    assert float(i_end[p]) < 0.0


def test_flux_ledger_closes_from_zero_state(nested_grid, nested_circuit):
    """∫u dt = ΔΨ̄ + ∫mean(R·i) dt — exact when the state integrates from
    zero (the breakdown-start contract: no free integral constant)."""
    circuit = nested_circuit
    col, l_self = _external_ring(nested_grid, R0, 0.68)
    m_pe = ps.patch_external_linkage(nested_grid, circuit.tiling, col)
    coupled = ps.build_coupled_circuit(
        circuit,
        l_ext=np.array([[l_self]]),
        r_ext=np.array([2.0e-5]),
        m_ext_coil=np.zeros((1, 0)),
        m_patch_ext=m_pe,
    )
    times = np.linspace(0.0, 0.08, 401)
    u_const = 0.7
    i_state = ps.evolve_coupled(
        coupled,
        ETA_FLAT,
        times,
        i0=np.zeros(coupled.n_total),
        loop_voltage=np.full(times.size, u_const),
    )
    psi_bar_0 = ps.mean_plasma_linked_flux(coupled, i_state[0])
    psi_bar_t = ps.mean_plasma_linked_flux(coupled, i_state[-1])
    assert abs(psi_bar_0) < 1e-30  # zero state ⇒ zero linked flux, no constant
    p = circuit.n_patches
    r_p = circuit.r_diag(ETA_FLAT)
    resistive = float(
        np.trapezoid((i_state[:, :p] * r_p[np.newaxis, :]).mean(axis=1), times)
    )
    vs_drive = u_const * float(times[-1] - times[0])
    closure = abs(vs_drive - ((psi_bar_t - psi_bar_0) + resistive))
    assert closure < 5e-4 * abs(vs_drive)


def test_uniform_rescale_maps_tau_by_s_squared():
    s = 2.0
    grid1 = _nested_circle_grid(r0=R0, a=A_MINOR, n=33)
    grid2 = _nested_circle_grid(r0=s * R0, a=s * A_MINOR, n=33)
    c1 = _fixture_circuit(grid1, r0=R0, a=A_MINOR, n_rad=8, n_pol=6)
    c2 = _fixture_circuit(grid2, r0=s * R0, a=s * A_MINOR, n_rad=8, n_pol=6)
    assert c1.n_patches == c2.n_patches
    tau1, v1 = ps.circuit_eigensystem(c1, ETA_FLAT)
    tau2, v2 = ps.circuit_eigensystem(c2, ETA_FLAT)
    # L ∝ s (Biot–Savart), R ∝ 1/s (ring length / area) ⇒ τ ∝ s² exactly
    ratio = tau2 / tau1
    assert np.abs(ratio / s**2 - 1.0).max() < 1e-6
    # mode SHAPES are dimensionless — identical up to sign
    for m in range(3):
        c = abs(
            float(v1[:, m] @ c1.lmat @ v2[:, m])
            / np.sqrt(float(v1[:, m] @ c1.lmat @ v1[:, m]))
            / np.sqrt(float(v2[:, m] @ c1.lmat @ v2[:, m]))
        )
        assert c > 0.999
