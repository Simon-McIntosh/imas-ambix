"""Tests for profile-DOF regularisation (soft SOL foot, q≥1 prior, smoothness).

All checks are analytic / synthetic — no data loads, no EFIT.  The soft SOL
edge is pinned to reproduce :func:`gs_solve.profile_basis` when its foot is
inactive and to stay C¹ and non-negative when active; the q machinery is pinned
against a known large-aspect circular equilibrium with a uniform current disc
(constant q = 2 B_φ0/(μ0 R_0 j_0)); the smoothness Gram is pinned equal to the
solver's own second-difference Gram.
"""

from __future__ import annotations

import numpy as np
import pytest

from imas_ambix.latent import gs_solve
from imas_ambix.latent.profile_regularization import (
    MU0,
    curvature_gram,
    edge_factor_with_foot,
    f_from_ffprime,
    monotonicity_penalty_rows,
    profile_basis_foot,
    q_axis_linear_bound,
    q_axis_penalty_row,
    q_profile,
)

EXPONENTS = (0.5, 1.0, 1.5, 2.0, 3.0)


# --- A. soft SOL edge + C¹ decay foot -------------------------------------


@pytest.mark.parametrize("exponent", EXPONENTS)
def test_edge_factor_inactive_matches_bare(exponent):
    """cap ≤ 1 reproduces the bare (1−ψ_N)^e edge factor and zeros past 1."""
    psi_n = np.linspace(-0.1, 1.4, 300)
    got = edge_factor_with_foot(psi_n, w=0.05, cap=1.0, exponent=exponent)
    bare = np.where(
        psi_n < 1.0, np.power(1.0 - np.clip(psi_n, 0.0, 1.0), exponent), 0.0
    )
    assert np.allclose(got, bare, atol=1e-12)


@pytest.mark.parametrize("exponent", EXPONENTS)
def test_edge_factor_c1_at_knot(exponent):
    """Value and one-sided slopes agree across the knot at ψ_N = 1 − w."""
    w = 0.05
    knot = 1.0 - w
    h = 1e-7

    def f(x):
        return float(
            edge_factor_with_foot(np.array([x]), w=w, cap=1.1, exponent=exponent)[0]
        )

    # value from both sides
    v_lo = f(knot - h)
    v_hi = f(knot + h)
    v_at = f(knot)
    assert abs(v_at - w**exponent) < 1e-9
    assert abs(v_hi - v_lo) < 1e-6
    # one-sided derivatives meet the analytic core slope −e·w^(e−1)
    slope_expected = -exponent * w ** (exponent - 1.0)
    d_lo = (v_at - v_lo) / h
    d_hi = (v_hi - v_at) / h
    assert abs(d_lo - slope_expected) < 1e-3 * (abs(slope_expected) + 1.0)
    assert abs(d_hi - slope_expected) < 1e-3 * (abs(slope_expected) + 1.0)


@pytest.mark.parametrize("exponent", EXPONENTS)
def test_edge_factor_nonneg_and_sol_current(exponent):
    """Non-negative through the band; positive in the SOL; zero beyond cap."""
    psi_n = np.linspace(0.0, 1.4, 500)
    fac = edge_factor_with_foot(psi_n, w=0.05, cap=1.1, exponent=exponent)
    assert np.all(fac >= -1e-15)
    sol = (psi_n > 1.0) & (psi_n <= 1.1)
    assert np.all(fac[sol] > 0.0)  # SOL current admitted, not pinned to zero
    assert np.all(fac[psi_n > 1.1] == 0.0)


def test_edge_factor_rejects_bad_width():
    with pytest.raises(ValueError):
        edge_factor_with_foot(np.array([0.5]), w=0.0, cap=1.1)
    with pytest.raises(ValueError):
        edge_factor_with_foot(np.array([0.5]), w=1.5, cap=1.1)


@pytest.mark.parametrize("kind", ["legendre", "monomial-nonneg"])
def test_profile_basis_foot_reproduces_gs_solve(kind):
    """With cap = 1 the footed basis equals gs_solve.profile_basis for ψ_N<1."""
    rng = np.random.default_rng(0)
    psi_n = np.clip(rng.uniform(0.0, 0.999, 200), 0.0, 0.999)
    r = rng.uniform(0.4, 1.4, 200)
    kw = dict(r0=0.85, n_p=3, n_f=2, kind=kind)
    bare = gs_solve.profile_basis(psi_n, r, **kw)
    footed = profile_basis_foot(psi_n, r, w=0.05, cap=1.0, **kw)
    assert footed.shape == bare.shape
    assert np.allclose(footed, bare, atol=1e-12)


@pytest.mark.parametrize("kind", ["legendre", "monomial-nonneg"])
def test_profile_basis_foot_admits_sol_band(kind):
    """cap = 1.1 leaves columns non-zero in the SOL band (ψ_N ∈ (1, 1.1])."""
    psi_n = np.linspace(0.0, 1.2, 400)
    r = np.full_like(psi_n, 0.85)
    cols = profile_basis_foot(
        psi_n, r, r0=0.85, n_p=2, n_f=1, kind=kind, w=0.05, cap=1.1
    )
    sol = (psi_n > 1.0) & (psi_n <= 1.1)
    assert np.any(np.abs(cols[sol, :]) > 0.0)
    assert np.allclose(cols[psi_n > 1.1, :], 0.0)
    if kind == "monomial-nonneg":
        assert np.all(cols >= -1e-15)


# --- B. safety factor and q≥1 prior ---------------------------------------


def test_f_from_ffprime_constant_gives_linear_f_squared():
    """Constant FF′ ⇒ F² linear in ψ_N; F(ψ_N=1) = f_boundary."""
    psi_n = np.linspace(0.0, 1.0, 50)
    ffp = np.full_like(psi_n, 0.02)  # small: keeps F² > 0 (no clipping) so linear
    f_boundary = 0.85 * 0.55
    f = f_from_ffprime(psi_n, ffp, f_boundary=f_boundary, dpsi_dpsin=1.7)
    assert np.all(f**2 > 0.0)  # unclipped regime
    assert abs(float(f[-1]) - f_boundary) < 1e-9  # anchored at boundary
    f2 = f**2
    # linear in ψ_N: second difference ≈ 0
    assert np.allclose(np.diff(f2, 2), 0.0, atol=1e-9)
    # sign follows f_boundary
    assert np.all(f > 0.0)


def test_f_from_ffprime_negative_boundary_sign():
    psi_n = np.linspace(0.0, 1.0, 20)
    f = f_from_ffprime(psi_n, np.zeros_like(psi_n), f_boundary=-0.4)
    assert np.all(f < 0.0)
    assert np.allclose(f, -0.4, atol=1e-9)  # FF′ = 0 ⇒ F constant


def test_q_axis_linear_bound_magnitude():
    """MAST-scale bound j_axis_max = 2 B_φ0/(μ0 R_0) is order 1e6 A/m²."""
    j_max = q_axis_linear_bound(b_phi0=0.55, r0=0.85)
    assert 5e5 < j_max < 2e6
    # exact relation
    assert abs(j_max - 2.0 * 0.55 / (MU0 * 0.85)) < 1.0


def test_q_axis_penalty_row_one_sided():
    """Penalty is zero at/below the bound, positive above it."""
    images = np.array([1.0, 0.5, 0.25])
    j_max = 1.0e6
    pen = q_axis_penalty_row(images_axis_unit=images, weight=2.0, j_axis_max=j_max)
    # coeffs giving j_axis below the bound
    lo = np.array([4.0e5, 2.0e5, 0.0])  # j_axis = 4e5 + 1e5 = 5e5 < 1e6
    assert pen.hinge(lo) == 0.0
    # coeffs giving j_axis above the bound
    hi = np.array([1.2e6, 2.0e5, 0.0])  # j_axis = 1.2e6 + 1e5 = 1.3e6 > 1e6
    assert pen.hinge(hi) > 0.0
    assert abs(pen.hinge(hi) - 2.0 * (1.3e6 - 1.0e6)) < 1.0
    # linear row for the LSQ assembly
    assert np.allclose(pen.row, 2.0 * images)
    assert abs(pen.rhs - 2.0 * j_max) < 1e-6


def test_q_profile_uniform_disc_circular():
    """q recovered on a large-aspect uniform-current disc (analytic q = 1)."""
    r0 = 3.0
    b_phi0 = 1.0
    f0 = r0 * b_phi0  # F = R·B_φ, constant (vacuum)
    q_target = 1.0
    # j0 chosen so 2 B_φ0/(μ0 R_0 j0) = q_target
    j0 = 2.0 * b_phi0 / (MU0 * r0 * q_target)
    a = 0.4  # minor radius → aspect ratio 7.5 (large)
    # Φ = −C r², C = π R0 μ0 j0 / 2  (TOTAL flux; axis is the maximum)
    cc = np.pi * r0 * MU0 * j0 / 2.0
    rg = np.linspace(r0 - 0.55, r0 + 0.55, 221)
    zg = np.linspace(-0.55, 0.55, 221)
    mesh_r, mesh_z = np.meshgrid(rg, zg)
    rr2 = (mesh_r - r0) ** 2 + mesh_z**2
    phi = -cc * rr2
    axis_psi = 0.0
    boundary_psi = -cc * a**2

    out = q_profile(
        phi,
        rg,
        zg,
        axis=(r0, 0.0),
        axis_psi=axis_psi,
        boundary_psi=boundary_psi,
        f_of_psin=lambda psin: f0,
        r0=r0,
        psi_n_surfaces=np.array([0.3, 0.5, 0.7, 0.9]),
    )
    q = out["q"]
    assert np.all(np.isfinite(q))
    # uniform current ⇒ constant q = q_target on every surface (few %)
    assert np.allclose(q, q_target, rtol=0.03)
    assert abs(out["q_axis"] - q_target) < 0.05


# --- C. higher-order regularisation ---------------------------------------


@pytest.mark.parametrize("n_p,n_f", [(3, 2), (5, 5), (2, 1), (4, 0), (0, 3)])
def test_curvature_gram_matches_solver(n_p, n_f):
    weight = 0.7
    got = curvature_gram(n_p, n_f, weight)
    ref = gs_solve._second_difference_gram(n_p, n_f, weight)
    assert np.allclose(got, ref, atol=1e-15)


def test_curvature_gram_zero_weight():
    assert np.allclose(curvature_gram(4, 3, 0.0), 0.0)


def test_monotonicity_penalty_direction():
    """A decreasing profile incurs no penalty; an increasing one does."""
    # sampled basis: two monomial edge columns for one family
    psi_n = np.linspace(0.05, 0.95, 8)
    edge = 1.0 - psi_n
    basis = np.column_stack([edge, edge**2])  # (8, 2) — one family, cols 0:2
    pen = monotonicity_penalty_rows(psi_n, basis, family_slices=[(0, 2)], weight=1.0)
    # positive coeffs → profile = c·(edge, edge²) DECREASES outward → no penalty
    decreasing = np.array([1.0, 0.5])
    assert pen.penalty(decreasing) == pytest.approx(0.0, abs=1e-12)
    # negative coeffs → profile INCREASES outward → penalised
    increasing = np.array([-1.0, -0.5])
    assert pen.penalty(increasing) > 0.0


def test_monotonicity_penalty_empty_families():
    psi_n = np.linspace(0.0, 1.0, 5)
    basis = np.zeros((5, 3))
    pen = monotonicity_penalty_rows(psi_n, basis, family_slices=[], weight=1.0)
    assert pen.rows.shape == (0, 3)
    assert pen.penalty(np.ones(3)) == 0.0


def test_module_is_firewall_clean():
    """No EFIT / evaluator import leaks into the pure-numpy regulariser."""
    import pathlib

    import imas_ambix.latent.profile_regularization as mod

    src = pathlib.Path(mod.__file__).read_text()
    import_lines = [
        ln.strip()
        for ln in src.splitlines()
        if ln.strip().startswith(("import ", "from "))
    ]
    banned = ("equilibrium_labels", "efit", "evaluate", "referee", "worldmodel")
    for ln in import_lines:
        assert not any(b in ln.lower() for b in banned), ln
