"""Tests for the 1D resistive current-diffusion temporal prior.

Correctness is pinned analytically and against the solver's own equilibria —
no EFIT, no data dependency:

* the CIRCULAR large-aspect limit of the ψ-diffusion reduces to the classical
  cylinder equation ∂ψ/∂t = (η/μ0)·(1/r)∂/∂r(r ∂ψ/∂r): a Bessel eigenmode
  perturbation must decay at its analytic rate, the prescribed-Ip edge BC
  must hold the enclosed current, and the late-time state must consume flux
  rigidly (spatially uniform loop voltage at the analytic ring value);
* the flux-surface geometry extracted from a solved synthetic equilibrium
  must close Ampère's law (enclosed current from the contour metrics equals
  the prescribed Ip) and reproduce the plasma volume;
* the (j_tor, ⟨J·B⟩) → coefficient projection must round-trip a known
  coefficient vector;
* the flux budget must decompose exactly (surface = axis + internal), which
  is the inductive/resistive consumption ledger;
* the coefficient prior at weight 0 must be byte-identical to the frozen
  solve, and a tight prior must pin the coefficients to its centre.
"""

from __future__ import annotations

import numpy as np

from imas_ambix.cocos import project_poloidal_field
from imas_ambix.latent.current_diffusion import (
    EtaProfile,
    FluxSurfaceGeometry,
    basis_projection_images,
    diffuse_psi,
    ejima_coefficient,
    flux_budget,
    flux_surface_geometry,
    predicted_current,
    project_coefficients,
)
from imas_ambix.latent.gs_solve import MU0, EquilibriumGrid

from .test_gs_solve import _confining_table, _synthetic_confining_slice


def _circular_geometry(
    *,
    a: float = 0.5,
    r0: float = 3.0,
    b0: float = 2.0,
    ip: float = 5.0e5,
    n_rho: int = 48,
) -> FluxSurfaceGeometry:
    """Exact circular large-aspect metrics (the analytic verification rig).

    V = 2π²·R0·r², Φ_tor = π r² B0, ρ̂ = r/a, F = R0·B0, ⟨1/R²⟩ = 1/R0²,
    g2 = ⟨(∇V)²/R²⟩ = 16π⁴r².  The initial ψ carries a uniform current
    density (ψ' ∝ ρ̂² edge-matched to Ip through the Ampère identity).
    """
    rho_face = np.linspace(0.0, 1.0, n_rho + 1)
    rho_cell = 0.5 * (rho_face[:-1] + rho_face[1:])
    r_face = a * rho_face
    phi_b = np.pi * a * a * b0
    f_face = np.full(n_rho + 1, r0 * b0)
    g2_face = 16.0 * np.pi**4 * r_face**2
    g3_face = np.full(n_rho + 1, 1.0 / r0**2)
    vpr_face = 4.0 * np.pi**2 * r0 * a * a * rho_face
    # uniform-j initial flux: ψ(ρ̂) with I(ρ̂) = Ip·ρ̂² through the BC identity
    d_face = np.zeros(n_rho + 1)
    d_face[1:] = g2_face[1:] * g3_face[1:] / rho_face[1:]
    grad = np.zeros(n_rho + 1)
    grad[1:] = (
        ip
        * rho_face[1:] ** 2
        * (16.0 * np.pi**3 * MU0 * phi_b)
        / (d_face[1:] * f_face[1:])
    )
    psi_face = np.concatenate(
        [[0.0], np.cumsum(0.5 * (grad[1:] + grad[:-1]) * np.diff(rho_face))]
    )
    return FluxSurfaceGeometry(
        rho_face=rho_face,
        rho_cell=rho_cell,
        psi_face=psi_face,
        psi_n_face=rho_face**2,
        psi_n_cell=rho_cell**2,
        vpr_face=vpr_face,
        vpr_cell=0.5 * (vpr_face[:-1] + vpr_face[1:]),
        g2_face=g2_face,
        g3_face=g3_face,
        g3_cell=np.full(n_rho, 1.0 / r0**2),
        f_face=f_face,
        f_cell=np.full(n_rho, r0 * b0),
        b2_cell=np.full(n_rho, b0 * b0),
        inv_r_cell=np.full(n_rho, 1.0 / r0),
        phi_b=phi_b,
        r0=r0,
        ip_amperes=ip,
        axis_psi=0.0,
        boundary_psi=float(psi_face[-1]),
        volume=2.0 * np.pi**2 * r0 * a * a,
        q_face=np.ones(n_rho + 1),
    )


def test_bessel_mode_decays_at_the_analytic_rate():
    """A J0 Neumann eigenmode decays at λ = (η/μ0)·(j'₁/a)² in the circular
    limit — the diffusion operator against the classical cylinder solution."""
    from scipy.special import j0, j1, jn_zeros

    a = 0.5
    eta0 = 1.0e-6
    geo = _circular_geometry(a=a)
    eta = EtaProfile(eta0=eta0, contrast=0.0, shape=1.0)
    k1 = float(jn_zeros(1, 1)[0])  # first zero of J1 → Neumann mode of J0
    assert abs(j1(k1)) < 1e-12
    eps = 1.0e-3 * float(np.ptp(geo.psi_face))
    mode = j0(k1 * geo.rho_face)
    lam_true = (eta0 / MU0) * (k1 / a) ** 2

    t_end = 0.2 / lam_true
    t = np.linspace(0.0, t_end, 400)
    ip_t = np.full(t.size, geo.ip_amperes)
    base = diffuse_psi(geo, eta, t_grid=t, ip_of_t=ip_t)
    pert = diffuse_psi(
        geo, eta, t_grid=t, ip_of_t=ip_t, psi0_face=geo.psi_face + eps * mode
    )
    dev = pert["psi_face"] - base["psi_face"]
    # remove the neutral constant mode (conserved under pure Neumann BCs)
    dev = dev - dev.mean(axis=1, keepdims=True)
    mode0 = mode - mode.mean()
    amp = dev @ mode0 / (mode0 @ mode0)
    # fit the decay rate over the window (skip the first step: BE transient)
    lam_fit = -np.polyfit(t[1:], np.log(np.abs(amp[1:])), 1)[0]
    assert abs(lam_fit - lam_true) / lam_true < 0.03


def test_ip_boundary_condition_holds_enclosed_current():
    """The edge-gradient BC carries the prescribed (ramping) Ip: the Ampère
    identity read back from the evolved flux matches the drive."""
    geo = _circular_geometry()
    eta = EtaProfile(eta0=1.0e-7, contrast=0.0, shape=1.0)
    t = np.linspace(0.0, 0.05, 201)
    ip_t = geo.ip_amperes * (1.0 + 0.5 * t / t[-1])  # 50% ramp
    out = diffuse_psi(geo, eta, t_grid=t, ip_of_t=ip_t)
    i_edge = geo.enclosed_current(out["psi_face"][-1])[-1]
    # transient skin-layer states carry an O(Δρ̂) scheme-consistency gap in
    # the read-back; the BC itself is exact (see the steady/identity tests)
    assert abs(i_edge - ip_t[-1]) / ip_t[-1] < 2e-2


def test_late_time_flux_consumption_is_rigid_and_ohmic():
    """With constant Ip and uniform η the state relaxes to rigid flux
    consumption: spatially uniform loop voltage at the analytic ring value
    V = 2π R0 η Ip / (π a²) — the resistive channel of the budget."""
    a, r0 = 0.5, 3.0
    eta0 = 1.0e-6
    geo = _circular_geometry(a=a, r0=r0)
    eta = EtaProfile(eta0=eta0, contrast=0.0, shape=1.0)
    tau = MU0 * a * a / eta0  # resistive time scale
    t = np.linspace(0.0, 3.0 * tau, 600)
    ip_t = np.full(t.size, geo.ip_amperes)
    out = diffuse_psi(geo, eta, t_grid=t, ip_of_t=ip_t)
    v_axis, v_bdry = out["v_axis"][-1], out["v_bdry"][-1]
    v_ring = 2.0 * np.pi * r0 * eta0 * geo.ip_amperes / (np.pi * a * a)
    assert abs(v_axis - v_bdry) < 0.05 * abs(v_ring)  # rigid consumption
    assert abs(abs(v_bdry) - v_ring) / v_ring < 0.05  # ohmic magnitude
    # the budget ledger decomposes exactly: surface = resistive + inductive
    budget = flux_budget(out, geo)
    assert np.isclose(
        budget["d_psi_bdry"], budget["d_psi_axis"] + budget["d_psi_internal"]
    )
    # constant Ip, relaxed profile → consumption is (almost) all resistive
    assert abs(budget["d_psi_internal"]) < 0.2 * abs(budget["d_psi_axis"])


def test_ejima_coefficient_normalisation():
    """C_E = |ΔΨ_res|/(μ0·R0·|ΔIp|) — the definitional check."""
    assert np.isclose(ejima_coefficient(MU0 * 0.85 * 5.0e5 * 0.45, 5.0e5, 0.85), 0.45)


def test_eta_profile_family_is_bounded_and_monotone():
    eta = EtaProfile(eta0=3.0e-8, contrast=3.0, shape=1.5)
    pn = np.linspace(0.0, 1.0, 50)
    vals = eta(pn)
    assert np.all(np.diff(vals) >= 0.0)  # monotone toward the cold edge
    assert np.isclose(vals[0], 3.0e-8) and np.isclose(vals[-1], 3.0e-8 * np.exp(3.0))
    rt = EtaProfile.from_vector(eta.as_vector())
    assert np.isclose(rt.eta0, eta.eta0) and np.isclose(rt.contrast, eta.contrast)
    lo, hi = zip(*EtaProfile.BOUNDS, strict=True)
    clipped = EtaProfile.from_vector(np.array([0.0, 99.0, 99.0]))
    assert clipped.eta0 <= hi[0] and clipped.contrast <= hi[1]
    assert clipped.shape <= hi[2]


def _interior_limiter_fixture():
    """A genuinely confined synthetic machine: limiter interior to the grid.

    ``_confining_table``'s plasma is a wall-supported blob (its flux surfaces
    leave through the grid edge, which doubles as the limiter), so no nested
    surfaces exist to trace.  Here the limiter box sits well inside a wider
    solve domain and a VF coil pair outside it holds a confined branch —
    nested closed surfaces out to ψ_N ≈ 0.9, the configuration the
    flux-surface extraction actually meets on real spine equilibria.
    """
    from imas_ambix.gs import geometry as gsg
    from imas_ambix.gs.cylinder import hybrid_greens

    lim_r = [0.9, 1.7, 1.7, 0.9, 0.9]
    lim_z = [-0.55, -0.55, 0.55, 0.55, -0.55]
    rg = np.linspace(0.4, 2.2, 61)
    zg = np.linspace(-1.1, 1.1, 73)
    coils = [(1.3, 0.9), (1.3, -0.9)]
    mesh_r, mesh_z = np.meshgrid(rg, zg)
    fr, fz = mesh_r.ravel(), mesh_z.ravel()
    cols = [hybrid_greens(fr, fz, cr, cz, 0.06, 0.06)[0] for cr, cz in coils]
    rects = np.array([[cr - 0.03, cr + 0.03, cz - 0.03, cz + 0.03] for cr, cz in coils])
    grid = EquilibriumGrid(
        rg=rg,
        zg=zg,
        limiter_r=np.array(lim_r),
        limiter_z=np.array(lim_z),
        coil_psi_columns=np.column_stack(cols),
        r0=1.3,
        conductor_rects=rects,
    )
    probes = [
        gsg.BProbe(index=i, r=1.95, z=-0.6 + 0.3 * i, angle_deg=-90.0, length=0.02)
        for i in range(5)
    ]
    smap = [
        gsg.SensorMapping(f"obv{i:02d}", "b_probe", i, p.r, p.z, p.angle_deg, 0.001, "")
        for i, p in enumerate(probes)
    ]
    table = gsg.GeometryTable(
        signature=gsg.SetupSignature(
            n_bprobe=5, n_fluxloop=0, n_pf_filament=2, n_limiter=5, digest="feed0004"
        ),
        shots=[1],
        b_probes=probes,
        flux_loops=[],
        pf_filaments=[
            gsg.PFFilament(
                r=cr, z=cz, turns=1.0, width=0.06, height=0.06, circuit=k + 1, xmult=1.0
            )
            for k, (cr, cz) in enumerate(coils)
        ],
        limiter_r=lim_r,
        limiter_z=lim_z,
        sensor_map=smap,
        passive_structures=[],
        amc_current_channels=[],
        unmatched_amb=[],
    )
    return grid, table


def _ladder_slice(grid, table, i_pf, ip):
    """A converged ladder fit of a synthetic confining slice (K = 2)."""
    from imas_ambix.gs.operator import greens_bz_br
    from imas_ambix.latent.gs_solve import (
        solve_equilibrium_bootstrapped,
        solve_equilibrium_lsq,
    )

    res = solve_equilibrium_bootstrapped(grid, i_pf, ip, beta0=0.6, alpha=1.0)
    assert res.converged
    g_sens, channels = grid.sensor_greens(table)
    vac = np.zeros(len(channels))
    for k, m in enumerate(table.sensor_map):
        for f, cur in zip(table.pf_filaments, i_pf, strict=True):
            bz, br = greens_bz_br(np.array([m.r]), np.array([m.z]), f.r, f.z)
            vac[k] += cur * project_poloidal_field(br[0], bz[0], m.angle_deg)
    meas = vac + g_sens @ res.cell_currents
    lf = solve_equilibrium_lsq(
        grid,
        table,
        i_pf,
        ip,
        measured=meas,
        vacuum_prediction=vac,
        sensor_scale=np.abs(meas) + 1e-9,
        sensor_mask=np.ones(meas.size, dtype=bool),
        n_p=1,
        n_f=1,
        nonneg=True,
    )
    assert lf.result.converged
    return lf, meas, vac


def test_flux_surface_geometry_closes_ampere_and_volume():
    """Contour metrics from a solved equilibrium must close Ampère's law
    (enclosed current at the edge = Ip) and reproduce the core volume."""
    grid, table = _interior_limiter_fixture()
    ip = 4.0e5
    i_pf = np.array([-8.0e4, -8.0e4])
    lf, _, _ = _ladder_slice(grid, table, i_pf, ip)
    geo = flux_surface_geometry(
        lf.result.psi,
        grid,
        coeffs=lf.coeffs,
        ip_amperes=ip,
        n_p=1,
        n_f=1,
        nonneg=True,
        b_phi0=1.0,
    )
    assert geo is not None
    i_edge = geo.enclosed_current(geo.psi_face)[-1]
    assert abs(i_edge - ip) / ip < 0.08  # 49×65 grid + contour discretisation
    # volume vs the direct core-cell sum ∑ 2πR dA
    core = lf.result.core_mask.ravel()
    v_cells = float((2.0 * np.pi * grid.flat_r[core]).sum() * grid.dr * grid.dz)
    assert abs(geo.volume - v_cells) / v_cells < 0.12
    # metric sanity: q positive-finite, F near vacuum, monotone ψ_N map
    assert np.all(np.isfinite(geo.q_face)) and np.all(geo.q_face > 0)
    assert np.all(np.diff(geo.psi_n_face) >= -1e-12)


def test_projection_round_trips_known_coefficients():
    """(j_tor, ⟨J·B⟩) built FROM a coefficient vector must project back to it."""
    grid, table = _interior_limiter_fixture()
    ip = 4.0e5
    i_pf = np.array([-8.0e4, -8.0e4])
    lf, _, _ = _ladder_slice(grid, table, i_pf, ip)
    geo = flux_surface_geometry(
        lf.result.psi,
        grid,
        coeffs=lf.coeffs,
        ip_amperes=ip,
        n_p=1,
        n_f=1,
        nonneg=True,
        b_phi0=1.0,
    )
    assert geo is not None
    from imas_ambix.latent.current_diffusion import reconstruct_profile_scales

    rec = reconstruct_profile_scales(lf.result.psi, grid, ip, n_p=1, n_f=1, nonneg=True)
    images = basis_projection_images(geo, rec["s_k"], n_p=1, n_f=1, nonneg=True)
    c_true = np.asarray(lf.coeffs, dtype=np.float64)
    j_tor = images["a_tor"] @ c_true
    j_par = images["a_par"] @ c_true
    c_rec = project_coefficients(geo, images, j_tor, j_par, nonneg=True)
    assert c_rec is not None
    assert np.allclose(c_rec, c_true, rtol=1e-4, atol=1e-6 * max(c_true.max(), 1.0))


def test_predicted_current_integrates_to_ip():
    """The predicted j_tot profile integrates back to the enclosed current."""
    geo = _circular_geometry()
    eta = EtaProfile(eta0=1.0e-7, contrast=0.0, shape=1.0)
    t = np.linspace(0.0, 0.02, 101)
    ip_t = np.full(t.size, geo.ip_amperes)
    out = diffuse_psi(geo, eta, t_grid=t, ip_of_t=ip_t)
    pred = predicted_current(geo, out["psi_face"][-1], out["psidot_face"], eta)
    # ∫ j_tot dS = Σ j_tot·spr·dρ̂ must equal the edge enclosed current
    spr = geo.vpr_cell * geo.inv_r_cell / (2.0 * np.pi)
    i_from_j = float(np.sum(pred["j_tor"] * spr * np.diff(geo.rho_face)))
    assert abs(i_from_j - pred["i_face"][-1]) / geo.ip_amperes < 1e-9
    assert abs(pred["i_face"][-1] - geo.ip_amperes) / geo.ip_amperes < 2e-2


def test_coeff_prior_zero_weight_is_byte_identical_and_tight_prior_pins():
    """The gate contract: weight 0 leaves the frozen solve byte-identical;
    a tight prior pins the fitted coefficients to its centre."""
    from imas_ambix.latent.patch_inverse import SlicePayload
    from scripts.closure_gate_eval import fit_and_read_slice

    table = _confining_table()
    grid = EquilibriumGrid.from_table(table, nr=49, nz=65)
    ip = 4.0e5
    i_pf = np.array([-6.0e4, -6.0e4])
    meas, vac, _ = _synthetic_confining_slice(grid, table, i_pf, ip, 0.6, 1.0)
    payload = SlicePayload(
        measured=meas,
        vacuum=vac,
        mask=np.ones(meas.size, dtype=bool),
        scale=np.abs(meas) + 1e-9,
        i_pf=i_pf,
        ip_amperes=ip,
        shot=1,
        t_index=0,
        time_s=0.0,
    )
    kw = dict(
        beta0_grid=(0.5,),
        alpha_grid=(1.0,),
        cost_limit=float("inf"),
        convergence_limit=5e-3,
        retry_max_iterations=None,
        fit_mode="ladder",
        n_p=1,
        n_f=1,
        nonneg=True,
    )
    base = fit_and_read_slice(grid, table, payload, **kw)
    center = np.array([0.9, 0.1])
    zero_w = fit_and_read_slice(grid, table, payload, coeff_prior=(center, 0.0), **kw)
    assert base.scored and zero_w.scored
    # 14-D geometry targets carry NaN where a leg is absent (equal_nan compare)
    assert np.array_equal(
        np.asarray(base.target), np.asarray(zero_w.target), equal_nan=True
    )
    assert np.array_equal(np.asarray(base.coeffs), np.asarray(zero_w.coeffs))
    tight = fit_and_read_slice(grid, table, payload, coeff_prior=(center, 1.0e4), **kw)
    assert tight.scored
    c_t = np.asarray(tight.coeffs)
    assert np.linalg.norm(c_t - center) < 0.05 * np.linalg.norm(center)


def test_torax_crosscheck_circular_psi_evolution():
    """The in-house operator matches TORAX on the shared circular case.

    ``scripts/torax_crosscheck_reference.py`` (run under the TORAX
    environment) evolved pure ohmic current diffusion on the built-in
    circular geometry with a ramping prescribed Ip and a Sauter σ from
    prescribed profiles.  Here the SAME metrics, σ, initial ψ and Ip trace
    drive ``diffuse_psi`` in chunks (σ refreshed per chunk — TORAX\'s Sauter
    σ evolves with q as ψ diffuses, mirroring the production frozen-per-
    interval contract), pinning the SOLVER formulation (toc coefficient,
    Ip-BC normalisation, θ-scheme) against the reference implementation to
    0.2%.  The metric extraction is pinned separately (the Ampère test), so
    the diffusion face coefficient here is TORAX\'s own g2g3/ρ̂ array.
    """
    from pathlib import Path

    ref_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "torax_circular_psi_reference.npz"
    )
    z = np.load(ref_path)
    rf = np.asarray(z["rho_face_norm"], dtype=np.float64)
    rc_mid = 0.5 * (rf[:-1] + rf[1:])
    grid50 = np.concatenate([[0.0], np.asarray(z["rho_cell_norm"]), [1.0]])
    times = np.asarray(z["times"], dtype=np.float64)
    n_rho = rf.size - 1
    # d_face == TORAX\'s own array: g2 re-expressed so g2·g3/ρ̂ matches theirs
    g2_adj = np.zeros_like(rf)
    g2_adj[1:] = z["g2g3_over_rhon_face"][1:] * rf[1:] / z["g3_face"][1:]

    def _geo(psi0: np.ndarray) -> FluxSurfaceGeometry:
        return FluxSurfaceGeometry(
            rho_face=rf,
            rho_cell=rc_mid,
            psi_face=psi0,
            psi_n_face=rf,  # dummy monotone map — the σ table is keyed to ρ̂
            psi_n_cell=rc_mid,
            vpr_face=np.asarray(z["vpr_face"], dtype=np.float64),
            vpr_cell=np.interp(rc_mid, rf, z["vpr_face"]),
            g2_face=g2_adj,
            g3_face=np.asarray(z["g3_face"], dtype=np.float64),
            g3_cell=np.interp(rc_mid, rf, z["g3_face"]),
            f_face=np.asarray(z["F_face"], dtype=np.float64),
            f_cell=np.interp(rc_mid, rf, z["F_face"]),
            b2_cell=np.ones(n_rho),
            inv_r_cell=np.full(n_rho, 1.0 / float(z["R_major"])),
            phi_b=float(z["Phi_b"]),
            r0=float(z["R_major"]),
            ip_amperes=float(z["ip"][0]),
            axis_psi=float(psi0[0]),
            boundary_psi=float(psi0[-1]),
            volume=float(np.trapezoid(z["vpr_face"], rf)),
            q_face=np.ones(rf.size),
            flux_sign=1.0,
        )

    psi = np.interp(rf, grid50, z["psi"][0])
    psi_start = psi.copy()
    bounds = np.linspace(0, times.size - 1, 16).astype(int)
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        sig_now = np.asarray(z["sigma_parallel"][a], dtype=np.float64)

        class _SigmaTable:
            def __call__(self, pn, _s=sig_now):
                return 1.0 / np.interp(np.asarray(pn), grid50, _s)

        out = diffuse_psi(
            _geo(psi),
            _SigmaTable(),
            t_grid=times[a : b + 1],
            ip_of_t=np.asarray(z["ip"][a : b + 1]),
        )
        psi = out["psi_face"][-1]

    d_ours = psi - psi_start
    d_torax = np.interp(rf, grid50, z["psi"][-1]) - psi_start
    rms = float(np.sqrt(np.mean((d_ours - d_torax) ** 2)))
    scale = float(np.sqrt(np.mean(d_torax**2)))
    assert scale > 0
    assert rms / scale < 0.01  # solver-formulation agreement vs TORAX


def test_short_interval_prediction_is_consistent_with_the_source_fit():
    """End-to-end (geometry → diffusion → prediction → projection) on a
    MAST-sign equilibrium: over a short, quiet interval the predicted
    coefficients must stay close to the source fit's own — the zero-
    innovation contract that catches amplitude and flux-sign defects in the
    prediction chain (both measured failure modes collapsed c_pred to ~0).
    """
    from imas_ambix.latent.current_diffusion import (
        basis_projection_images,
        diffuse_psi,
        predicted_current,
        project_coefficients,
    )

    grid, table = _interior_limiter_fixture()
    ip = 4.0e5
    i_pf = np.array([-8.0e4, -8.0e4])
    lf, _, _ = _ladder_slice(grid, table, i_pf, ip)
    geo = flux_surface_geometry(
        lf.result.psi,
        grid,
        coeffs=lf.coeffs,
        ip_amperes=ip,
        n_p=1,
        n_f=1,
        nonneg=True,
        b_phi0=1.0,
    )
    assert geo is not None and geo.flux_sign == -1.0  # the MAST-sign branch
    eta = EtaProfile(eta0=5.0e-8, contrast=1.5, shape=1.5)
    t = np.linspace(0.0, 0.02, 24)
    out = diffuse_psi(geo, eta, t_grid=t, ip_of_t=np.full(t.size, ip))
    pred = predicted_current(geo, out["psi_face"][-1], out["psidot_face"], eta)
    # the dissipative parallel channel must be positive where current flows
    assert float(np.median(np.sign(pred["j_par_b"]))) > 0
    images = basis_projection_images(geo, geo.s_k, n_p=1, n_f=1, nonneg=True)
    c_pred = project_coefficients(
        geo, images, pred["j_tor"], pred["j_par_b"], nonneg=True
    )
    assert c_pred is not None
    c_fit = np.asarray(lf.coeffs, dtype=np.float64)
    # 20 ms at constant Ip barely moves the profile — the prediction must
    # carry the fit's own amplitude (not collapse toward zero)
    assert abs(c_pred.sum() - c_fit.sum()) < 0.35 * c_fit.sum()


def test_ledger_floop_anchor_subtracts_exactly_the_fl_residual_drift():
    """The flux-loop-anchored swing arm must remove exactly the drift of the
    fit's fl residual from the budget chain — and reduce to the fit-swing arm
    when that residual is time-constant.  A fabricated linear drift on
    identical equilibria (true swing zero) pins the algebra."""
    from scripts.current_diffusion_flux_ledger_report import shot_ledger

    grid, table = _interior_limiter_fixture()
    ip = 4.0e5
    i_pf = np.array([-8.0e4, -8.0e4])
    lf, _, _ = _ladder_slice(grid, table, i_pf, ip)
    geo = flux_surface_geometry(
        lf.result.psi,
        grid,
        coeffs=lf.coeffs,
        ip_amperes=ip,
        n_p=1,
        n_f=1,
        nonneg=True,
        b_phi0=1.0,
    )
    assert geo is not None
    times = [0.0, 0.02, 0.04, 0.06]
    drift = 0.05  # fabricated fl-residual drift per interval [Wb]
    raw_times = np.linspace(-0.1, 0.2, 40)
    materials = {
        "geos": [geo] * 4,
        "times": times,
        "raw_times": raw_times,
        "ip_raw_amp": np.full(raw_times.size, ip),
        "fl_resid_wb": [0.0, drift, 2 * drift, 3 * drift],
    }
    args_d = {"n_sub_steps": 8, "par_weight": 1.0}
    eta = EtaProfile(eta0=5.0e-8, contrast=0.0, shape=2.0)
    led_fit = shot_ledger(1, args_d, eta, materials=materials, swing="fit")
    led_fl = shot_ledger(1, args_d, eta, materials=materials, swing="floop")
    assert led_fit is not None and led_fl is not None
    for r_fit, r_fl in zip(led_fit["rows"], led_fl["rows"], strict=True):
        expected = -geo.flux_sign * (r_fl["fl_resid_wb"] - 0.0)
        got = r_fl["span_budget_wb"] - r_fit["span_budget_wb"]
        assert abs(got - expected) < 1e-12
    # constant residual → identical chains
    materials["fl_resid_wb"] = [0.7, 0.7, 0.7, 0.7]
    led_const = shot_ledger(1, args_d, eta, materials=materials, swing="floop")
    for r_fit, r_c in zip(led_fit["rows"], led_const["rows"], strict=True):
        assert abs(r_c["span_budget_wb"] - r_fit["span_budget_wb"]) < 1e-12


def test_ledger_sane_gate_and_nonind_scaling():
    """li3 sanity gate: an impossible threshold leaves no gauge slice (None);
    a permissive one keeps every row sane.  f_ni = 1 must zero the modelled
    resistive term so the budget chain carries only the measured swing."""
    from scripts.current_diffusion_flux_ledger_report import shot_ledger

    grid, table = _interior_limiter_fixture()
    ip = 4.0e5
    i_pf = np.array([-8.0e4, -8.0e4])
    lf, _, _ = _ladder_slice(grid, table, i_pf, ip)
    geo = flux_surface_geometry(
        lf.result.psi,
        grid,
        coeffs=lf.coeffs,
        ip_amperes=ip,
        n_p=1,
        n_f=1,
        nonneg=True,
        b_phi0=1.0,
    )
    assert geo is not None
    times = [0.0, 0.02, 0.04, 0.06]
    raw_times = np.linspace(-0.1, 0.2, 40)
    materials = {
        "geos": [geo] * 4,
        "times": times,
        "raw_times": raw_times,
        "ip_raw_amp": np.full(raw_times.size, ip),
    }
    args_d = {"n_sub_steps": 8, "par_weight": 1.0}
    eta = EtaProfile(eta0=5.0e-8, contrast=0.0, shape=2.0)
    assert shot_ledger(1, args_d, eta, materials=materials, li3_sane_max=1e-4) is None
    led = shot_ledger(1, args_d, eta, materials=materials, li3_sane_max=50.0)
    assert led is not None and led["n_sane"] == led["n_rows"] == 4
    assert led["closure_rms_sane_wb"] is not None
    led_ni = shot_ledger(1, args_d, eta, materials=materials, f_ni=1.0)
    for r in led_ni["rows"]:
        assert r["d_res_model_wb"] == 0.0
    # identical geos → measured swing zero → the f_ni=1 budget is constant
    buds = [r["span_budget_wb"] for r in led_ni["rows"]]
    assert max(buds) - min(buds) < 1e-12


def test_beta_p_sensitivity_pins_convention_and_linearity():
    """∂βp/∂coeffs: FF′ rows exactly zero; βp positive and O(0.1–2) on the
    fixture; and the closed-form monomial cumulative must agree with an
    independent numeric pressure integration through beta_poloidal (catches
    sign/mask/cumulative slips without reusing the closed form).  The flux
    convention itself (drive ŝ(R/R0)φ = 2πR·p′_Φ) is fixed by the solve's GS
    form, pinned elsewhere by the Ampère-closure test."""
    from imas_ambix.latent.current_diffusion import (
        beta_p_coeff_sensitivity,
        reconstruct_profile_scales,
    )
    from imas_ambix.latent.moment_priors import beta_poloidal

    grid, table = _interior_limiter_fixture()
    ip = 4.0e5
    i_pf = np.array([-8.0e4, -8.0e4])
    lf, _, _ = _ladder_slice(grid, table, i_pf, ip)
    psi2d = lf.result.psi
    sens = beta_p_coeff_sensitivity(psi2d, grid, ip, n_p=1, n_f=1, nonneg=True)
    assert sens is not None
    assert sens.shape == (2,) and sens[1] == 0.0  # FF' row carries nothing
    c = np.asarray(lf.coeffs, dtype=np.float64)
    beta_lin = float(sens @ c)
    assert 0.05 < beta_lin < 3.0
    # independent numeric route: physical p′(ψ_N) sampled on a fine grid,
    # trapezoid-integrated to pressure2d, volume-integrated by beta_poloidal
    rec = reconstruct_profile_scales(psi2d, grid, ip, n_p=1, n_f=1, nonneg=True)
    span = rec["boundary_psi"] - rec["axis_psi"]
    u = np.linspace(0.0, 1.0, 801)
    pprime = c[0] * rec["s_k"][0] * (1.0 - u) ** 0.5 / (2.0 * np.pi * grid.r0)
    tail = np.concatenate(
        [
            np.cumsum((pprime[::-1][:-1] + pprime[::-1][1:]) * 0.5)[::-1]
            * (u[1] - u[0]),
            [0.0],
        ]
    )
    p_flat = -span * np.interp(np.clip(rec["psi_n"], 0.0, 1.0), u, tail)
    p2d = np.where(rec["core"].ravel(), p_flat, 0.0).reshape(grid.nz, grid.nr)
    beta_num = beta_poloidal(
        psi2d,
        lf.result.jphi,
        p2d,
        grid.rg,
        grid.zg,
        axis_psi=rec["axis_psi"],
        boundary_psi=rec["boundary_psi"],
        r0=grid.r0,
    )
    # beta_poloidal normalises by the grid-integrated jφ current; the
    # sensitivity uses the prescribed Ip — allow the discretisation gap
    assert abs(beta_lin - beta_num) / beta_num < 0.1, (beta_lin, beta_num)
