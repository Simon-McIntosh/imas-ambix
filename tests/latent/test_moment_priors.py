"""Tests for the covariance-weighted global-moment soft priors.

Everything is analytic / synthetic — no EFIT, no corpus, no SLURM.  The
headline evidence is :func:`test_pressure_prior_separates_pprime_from_ffprime`:
on a deliberately degenerate p′/FF′ system the split is under-determined
without the pressure prior and recovered to the injected truth with it.
"""

from __future__ import annotations

import numpy as np
import pytest

from imas_ambix.latent.moment_priors import (
    MU0,
    TE_DATA_GATE,
    beta_p_li_over_2,
    beta_poloidal,
    centroid_moment_rows,
    density_pressure_proxy,
    internal_inductance_li,
    ip_soft_prior_row,
    kinetic_pressure_target,
    moment_consistency_rows,
    pressure_gradient_prior_rows,
)


# --------------------------------------------------------------------------- #
# 1. Ip soft prior
# --------------------------------------------------------------------------- #
def _toy_data_system(k_dof=3, n_data=5, seed=0):
    """A small over-determined data block A x ≈ y on the coeff sub-vector."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n_data, k_dof))
    x_true = rng.standard_normal(k_dof)
    y = a @ x_true
    return a, y, x_true


def test_ip_soft_row_whitening():
    """Weight is 1/(sigma_rel·|Ip|); passive columns are zero."""
    a_anchor = np.array([1.0, 2.0, 3.0])
    row, rhs = ip_soft_prior_row(
        a_anchor, ip_amperes=5.0e5, sigma_rel=0.02, k_dof=3, kp=2
    )
    w = 1.0 / (0.02 * 5.0e5)
    assert row.shape == (5,)
    np.testing.assert_allclose(row[:3], w * a_anchor)
    np.testing.assert_allclose(row[3:], 0.0)
    assert rhs == pytest.approx(w * 5.0e5)


def test_ip_soft_prior_approaches_hard_anchor():
    """As sigma_rel → 0 the soft solve reproduces the hard-constrained solve."""
    a, y, _ = _toy_data_system(k_dof=3, n_data=6)
    ip = 1.7  # O(1) so the whitening weight 1/(sigma_rel·|Ip|) can grow large
    a_anchor = np.array([0.7, -0.4, 1.1])

    # hard-constrained least squares: min ||a x - y|| s.t. a_anchor·x = ip
    # via KKT.
    k = 3
    h = a.T @ a
    kkt = np.zeros((k + 1, k + 1))
    kkt[:k, :k] = 2.0 * h
    kkt[:k, k] = a_anchor
    kkt[k, :k] = a_anchor
    rhs = np.concatenate([2.0 * (a.T @ y), [ip]])
    x_hard = np.linalg.solve(kkt, rhs)[:k]

    # soft solve at tiny sigma_rel (no passive block here)
    row, r = ip_soft_prior_row(a_anchor, ip, sigma_rel=1e-6, k_dof=3, kp=0)
    a_aug = np.vstack([a, row[np.newaxis, :]])
    y_aug = np.concatenate([y, [r]])
    x_soft, *_ = np.linalg.lstsq(a_aug, y_aug, rcond=None)

    assert a_anchor @ x_soft == pytest.approx(ip, rel=1e-4)
    np.testing.assert_allclose(x_soft, x_hard, rtol=1e-3, atol=1e-3)


# --------------------------------------------------------------------------- #
# 1b. Current-centroid moment rows (the position lever, like the Ip anchor)
# --------------------------------------------------------------------------- #
def _toy_current_system(k_dof=4, n_cells=37, seed=1):
    """A synthetic (cell_r, cell_z, U, coeffs) with an exactly-known centroid.

    U (n_cells, k_dof) are per-coefficient cell currents [A]; jφ_cell = U·coeffs,
    Ip = Σ U·coeffs, and the true centroid is Σ coord·jφ / Ip — all computed
    directly so the row residual can be checked against machine zero.
    """
    rng = np.random.default_rng(seed)
    cell_r = 0.6 + 0.8 * rng.random(n_cells)  # MAST-like R ∈ [0.6, 1.4] m
    cell_z = -0.4 + 0.8 * rng.random(n_cells)
    u = rng.random((n_cells, k_dof))  # ≥ 0 so the current is unidirectional
    coeffs = rng.random(k_dof)
    i_cell = u @ coeffs
    ip = float(i_cell.sum())
    r_c = float((cell_r * i_cell).sum() / ip)
    z_c = float((cell_z * i_cell).sum() / ip)
    return cell_r, cell_z, u, coeffs, ip, r_c, z_c


def test_centroid_rows_reproduce_known_centroid():
    """At the true centroid target the whitened moment residual is machine-zero.

    This is the T1 correctness check: the R and Z centroid moment rows encode
    ∫R jφ = R_c·Ip and ∫Z jφ = Z_c·Ip exactly (linear-homogeneous in coeffs,
    like the Ip anchor), so row·[coeffs, 0] − rhs = 0 to machine precision when
    the target equals the current's actual centroid.
    """
    cell_r, cell_z, u, coeffs, ip, r_c, z_c = _toy_current_system(k_dof=4)
    kp = 3
    rows, rhs = centroid_moment_rows(
        cell_r=cell_r,
        cell_z=cell_z,
        unit_cell_currents=u,
        r_target=r_c,
        z_target=z_c,
        ip_amperes=ip,
        sigma_r=0.02,
        sigma_z=0.02,
        k_dof=4,
        kp=kp,
    )
    assert rows.shape == (2, 4 + kp)
    x = np.concatenate([coeffs, np.zeros(kp)])
    np.testing.assert_allclose(rows @ x, rhs, atol=1e-9, rtol=0.0)
    # passive columns carry no centroid sensitivity
    np.testing.assert_allclose(rows[:, 4:], 0.0)


def test_centroid_rows_whitening_and_offset_centroid():
    """A centroid offset of σ metres produces unit whitened residual per axis."""
    cell_r, cell_z, u, coeffs, ip, r_c, z_c = _toy_current_system(k_dof=3)
    sigma = 0.03
    rows, rhs = centroid_moment_rows(
        cell_r=cell_r,
        cell_z=cell_z,
        unit_cell_currents=u,
        r_target=r_c + sigma,  # target 1σ outboard of the true centroid
        z_target=z_c - sigma,  # and 1σ below
        ip_amperes=ip,
        sigma_r=sigma,
        sigma_z=sigma,
        k_dof=3,
        kp=0,
    )
    resid = rows @ coeffs - rhs
    # residual = (moment(coeffs) − target·Ip)/(σ·|Ip|) = ∓(σ·Ip)/(σ·|Ip|) = ∓1
    np.testing.assert_allclose(np.abs(resid), 1.0, atol=1e-9)


def test_centroid_rows_skip_none_target():
    """A None coordinate target drops that row (R-only or Z-only pinning)."""
    cell_r, cell_z, u, coeffs, ip, r_c, z_c = _toy_current_system(k_dof=2)
    rows_r, _ = centroid_moment_rows(
        cell_r=cell_r,
        cell_z=cell_z,
        unit_cell_currents=u,
        r_target=r_c,
        z_target=None,
        ip_amperes=ip,
        sigma_r=0.02,
        sigma_z=0.02,
        k_dof=2,
        kp=0,
    )
    assert rows_r.shape == (1, 2)
    none_rows, none_rhs = centroid_moment_rows(
        cell_r=cell_r,
        cell_z=cell_z,
        unit_cell_currents=u,
        r_target=None,
        z_target=None,
        ip_amperes=ip,
        sigma_r=0.02,
        sigma_z=0.02,
        k_dof=2,
        kp=0,
    )
    assert none_rows.shape == (0, 2)
    assert none_rhs.shape == (0,)


# --------------------------------------------------------------------------- #
# 2. βp, li, βp + li/2 diagnostics on synthetic equilibria
# --------------------------------------------------------------------------- #
def _uniform_current_disk(r0=10.0, a=1.0, n=161, ip=1.0):
    """Large-aspect circular plasma, uniform toroidal current density.

    Builds the analytic per-radian poloidal flux of a uniform-current disk so
    that B_pol(ρ) = μ0 Ip ρ / (2π a²) inside (li(3) = 0.5), on a Cartesian
    (R, Z) grid.  Returns (psi2d, jphi2d, rg, zg, axis_psi, boundary_psi).
    """
    rg = np.linspace(r0 - 1.6 * a, r0 + 1.6 * a, n)
    zg = np.linspace(-1.6 * a, 1.6 * a, n)
    rr, zz = np.meshgrid(rg, zg)
    rho = np.hypot(rr - r0, zz)
    j0 = ip / (np.pi * a**2)  # uniform density [A/m²]
    jphi2d = np.where(rho <= a, j0, 0.0)
    # physical poloidal field Bp(ρ)=μ0 Ip ρ/(2π a²) inside.  _poloidal_field
    # reads Bp = |∇ψ_pr|/R, so ψ_pr must carry a toroidal R0 factor:
    # |∇ψ_pr| = R·Bp ≈ R0·Bp → ψ_pr = −0.5·R0·b_pref·ρ² inside, log outside.
    b_pref = MU0 * ip / (2.0 * np.pi * a**2)
    psi_pr = r0 * np.where(
        rho <= a,
        -0.5 * b_pref * rho**2,
        -0.5 * b_pref * a**2 - b_pref * a**2 * np.log(np.maximum(rho, a) / a),
    )
    psi2d = 2.0 * np.pi * psi_pr  # TOTAL flux Φ = 2π ψ_pr
    axis_psi = float(psi2d[np.argmin(np.abs(zg)), np.argmin(np.abs(rg - r0))])
    boundary_psi = 2.0 * np.pi * r0 * (-0.5 * b_pref * a**2)
    return psi2d, jphi2d, rg, zg, axis_psi, boundary_psi


def test_internal_inductance_uniform_disk_is_half():
    """li(3) = 0.5 for a uniform-current large-aspect circular plasma."""
    psi2d, jphi2d, rg, zg, ax, bd = _uniform_current_disk()
    li = internal_inductance_li(
        psi2d, jphi2d, rg, zg, axis_psi=ax, boundary_psi=bd, r0=10.0
    )
    assert li == pytest.approx(0.5, abs=0.05)


def test_beta_poloidal_uniform_pressure_disk():
    """βp of a uniform-pressure disk matches 8π² a² p0/(μ0 Ip²)."""
    r0, a, ip, p0 = 10.0, 1.0, 1.0, 3.0e3
    psi2d, jphi2d, rg, zg, ax, bd = _uniform_current_disk(r0=r0, a=a, ip=ip)
    rr, _ = np.meshgrid(rg, zg)
    rho = np.hypot(rr - r0, np.meshgrid(rg, zg)[1])
    pressure2d = np.where(rho <= a, p0, 0.0)
    bp = beta_poloidal(
        psi2d,
        jphi2d,
        pressure2d,
        rg,
        zg,
        axis_psi=ax,
        boundary_psi=bd,
        r0=r0,
    )
    expected = 8.0 * np.pi**2 * a**2 * p0 / (MU0 * ip**2)
    assert bp == pytest.approx(expected, rel=0.05)


def test_beta_p_li_over_2_recovers_injected_asymmetry():
    """A grid with a KNOWN boundary-field asymmetry recovers βp + li/2.

    Construct ψ_pr = −B_pa[ρ + (target/2R0) ρ² cosθ] whose ρ≈a poloidal field
    carries the Shafranov cosθ modulation for a chosen βp + li/2 = target; the
    boundary-asymmetry read must return target (large aspect, few-% identity).
    """
    r0, a, target, b_pa = 10.0, 1.0, 2.0, 1.0
    n = 401
    rg = np.linspace(r0 - 1.5 * a, r0 + 1.5 * a, n)
    zg = np.linspace(-1.5 * a, 1.5 * a, n)
    rr, zz = np.meshgrid(rg, zg)
    x, z = rr - r0, zz
    rho = np.hypot(x, z)
    theta = np.arctan2(z, x)
    psi_pr = -b_pa * (rho + (target / (2.0 * r0)) * rho**2 * np.cos(theta))
    psi2d = 2.0 * np.pi * psi_pr
    boundary_psi = 2.0 * np.pi * (-b_pa * a)
    got = beta_p_li_over_2(
        psi2d,
        np.zeros_like(psi2d),
        rg,
        zg,
        axis=(r0, 0.0),
        boundary_psi=boundary_psi,
        r0=r0,
        b_phi0=1.0,
    )
    assert got == pytest.approx(target, rel=0.15)


def test_boundary_asymmetry_extractor_exact_on_samples():
    """The extractor recovers target from analytic boundary samples exactly."""
    from imas_ambix.latent.moment_priors import _boundary_field_asymmetry

    r0, a, target, b_pa = 10.0, 1.0, 1.6, 2.0
    theta = np.linspace(-np.pi, np.pi, 200, endpoint=False)
    bp = b_pa * (1.0 + (a / r0) * (target - 1.0) * np.cos(theta))
    r_minor = np.full_like(theta, a)
    got = _boundary_field_asymmetry(theta, bp, r_minor, r0)
    assert got == pytest.approx(target, rel=1e-6)


# --------------------------------------------------------------------------- #
# 3. moment_consistency_rows
# --------------------------------------------------------------------------- #
def test_moment_consistency_row_whitening_and_zeros():
    s = np.array([1.0, -2.0])
    row, rhs = moment_consistency_rows(
        computed_moment_unit_sensitivity=s,
        target_moment=0.8,
        sigma=0.1,
        k_dof=2,
        kp=3,
    )
    assert row.shape == (5,)
    np.testing.assert_allclose(row[:2], s / 0.1)
    np.testing.assert_allclose(row[2:], 0.0)
    assert rhs == pytest.approx(0.8 / 0.1)


def test_moment_consistency_pins_linear_moment():
    """A soft moment row drives s·coeffs to the target in a toy solve."""
    a, y, _ = _toy_data_system(k_dof=2, n_data=4, seed=3)
    s = np.array([1.0, 1.0])
    row, rhs = moment_consistency_rows(
        computed_moment_unit_sensitivity=s,
        target_moment=2.5,
        sigma=1e-4,
        k_dof=2,
        kp=0,
    )
    a_aug = np.vstack([a, row[np.newaxis, :]])
    y_aug = np.concatenate([y, [rhs]])
    x, *_ = np.linalg.lstsq(a_aug, y_aug, rcond=None)
    assert s @ x == pytest.approx(2.5, rel=1e-3)


# --------------------------------------------------------------------------- #
# 4. Pressure-gradient prior — the p′/FF′ separation lever (headline)
# --------------------------------------------------------------------------- #
def test_pressure_prior_separates_pprime_from_ffprime():
    """The pressure prior resolves an under-determined p′/FF′ split.

    Build a degenerate 2-DOF system: the p′ (col 0) and FF′ (col 1) sensor
    signatures are nearly collinear (as at finite aspect, where R/R0 ≈ R0/R),
    so the data + Ip constraint fix the SUM c_p + c_f well but the split
    poorly.  Inject a true split, then show:
      * WITHOUT the pressure prior → recovered split is far from truth;
      * WITH a pressure prior pinning the p′ family → both recovered.
    """
    rng = np.random.default_rng(7)
    k_dof, kp = 2, 0
    cp_true, cf_true = 0.7, 0.3

    # two near-collinear sensor columns: the p′ and FF′ signatures differ only
    # by a component (eps) driven BELOW the measurement noise floor, so the
    # data + Ip anchor fix the SUM but leave the split under-determined.
    base = rng.standard_normal(8)
    base /= np.linalg.norm(base)
    perturb = rng.standard_normal(8)
    perturb -= (perturb @ base) * base
    perturb /= np.linalg.norm(perturb)
    eps, noise = 5.0e-4, 5.0e-3
    v_p = base
    v_f = base + eps * perturb  # independent component well under the noise
    g = np.column_stack([v_p, v_f])  # (8, 2) design
    y = g @ np.array([cp_true, cf_true])
    y = y + noise * rng.standard_normal(y.size)  # measurement noise

    # Ip soft anchor on the sum (a_anchor = [1, 1], Ip = cp+cf)
    ip_row, ip_rhs = ip_soft_prior_row(
        np.array([1.0, 1.0]), cp_true + cf_true, sigma_rel=1e-3, k_dof=k_dof, kp=kp
    )

    # WITHOUT pressure prior: data + Ip only
    a0 = np.vstack([g, ip_row[np.newaxis, :]])
    b0 = np.concatenate([y, [ip_rhs]])
    x0, *_ = np.linalg.lstsq(a0, b0, rcond=None)
    err_without = abs(x0[0] - cp_true)

    # WITH pressure prior pinning the p′ family (col 0) to its true value.
    # p_basis_slice = [[1.0]] maps coeffs[:1] → p′; target = cp_true.
    p_rows, p_rhs = pressure_gradient_prior_rows(
        p_basis_slice=np.array([[1.0]]),
        pprime_target=np.array([cp_true]),
        sigma=1e-3,
        k_dof=k_dof,
        kp=kp,
    )
    a1 = np.vstack([g, ip_row[np.newaxis, :], p_rows])
    b1 = np.concatenate([y, [ip_rhs], p_rhs])
    x1, *_ = np.linalg.lstsq(a1, b1, rcond=None)
    err_with = abs(x1[0] - cp_true)

    # the degeneracy: without the prior the split is badly determined ...
    assert err_without > 0.1
    # ... and the pressure prior resolves it to the injected truth.
    assert err_with < 1e-2
    assert x1[1] == pytest.approx(cf_true, abs=1e-2)
    # the prior is strictly better at recovering the split
    assert err_with < 0.1 * err_without


def test_pressure_prior_rows_shape_and_ffprime_zero():
    """Rows constrain ONLY the p′ family; FF′ + passive columns are zero."""
    p = np.array([[1.0, 0.5], [0.2, 0.9], [0.0, 1.0]])  # (3 samples, n_p=2)
    rows, rhs = pressure_gradient_prior_rows(
        p_basis_slice=p,
        pprime_target=np.array([1.0, 2.0, 3.0]),
        sigma=0.5,
        k_dof=5,  # n_p=2, n_f=3
        kp=4,
    )
    assert rows.shape == (3, 9)
    np.testing.assert_allclose(rows[:, :2], p / 0.5)
    np.testing.assert_allclose(rows[:, 2:], 0.0)  # FF′ + passive untouched
    np.testing.assert_allclose(rhs, np.array([1.0, 2.0, 3.0]) / 0.5)


def test_pressure_prior_per_sample_sigma():
    p = np.array([[1.0], [1.0]])
    rows, rhs = pressure_gradient_prior_rows(
        p_basis_slice=p,
        pprime_target=np.array([2.0, 4.0]),
        sigma=np.array([0.5, 2.0]),
        k_dof=1,
        kp=0,
    )
    np.testing.assert_allclose(rows[:, 0], [1.0 / 0.5, 1.0 / 2.0])
    np.testing.assert_allclose(rhs, [2.0 / 0.5, 4.0 / 2.0])


# --------------------------------------------------------------------------- #
# 5. Pressure target sources — proxy + the Te data gate
# --------------------------------------------------------------------------- #
def test_density_pressure_proxy_is_negative_and_falls_to_edge():
    psi_n = np.linspace(0.0, 1.0, 11)
    pprime = density_pressure_proxy(5.0e19, psi_n, gamma=1.5)
    assert np.all(pprime <= 0.0)  # pressure falls from axis to edge
    assert abs(pprime[0]) > abs(pprime[-1])  # magnitude largest on axis
    # magnitude scales linearly with line density (weak-lever sanity)
    pprime2 = density_pressure_proxy(1.0e20, psi_n, gamma=1.5)
    np.testing.assert_allclose(pprime2, pprime * (1.0e20 / 5.0e19), rtol=1e-12)


def test_density_pressure_proxy_rejects_unknown_shape():
    with pytest.raises(ValueError):
        density_pressure_proxy(1.0e19, np.linspace(0, 1, 5), shape="bananas")


def test_kinetic_pressure_target_is_data_gated():
    """The kinetic target raises, naming the missing Thomson Te data."""
    with pytest.raises(NotImplementedError) as exc:
        kinetic_pressure_target(np.ones(5), np.ones(5), np.linspace(0, 1, 5))
    assert "Thomson" in str(exc.value)
    assert str(exc.value) == TE_DATA_GATE


# --------------------------------------------------------------------------- #
# 6. Input validation
# --------------------------------------------------------------------------- #
def test_ip_soft_row_rejects_bad_sigma_and_length():
    with pytest.raises(ValueError):
        ip_soft_prior_row(np.array([1.0]), 1.0, sigma_rel=0.0, k_dof=1, kp=0)
    with pytest.raises(ValueError):
        ip_soft_prior_row(np.array([1.0, 2.0]), 1.0, sigma_rel=0.1, k_dof=1, kp=0)


def test_moment_rows_reject_bad_sensitivity_length():
    with pytest.raises(ValueError):
        moment_consistency_rows(
            computed_moment_unit_sensitivity=np.array([1.0, 2.0]),
            target_moment=1.0,
            sigma=0.1,
            k_dof=3,
            kp=0,
        )
