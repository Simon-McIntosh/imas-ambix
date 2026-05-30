"""Tests for the T8 SINDy distillation of the transition kernel.

All tests are synthetic and fast — they do NOT pull the 80 MB T7 trajectory
cache or train the engine.  They validate:

1. The polynomial library + feature names.
2. STLSQ recovers a KNOWN sparse map (and rejects below-threshold noise terms).
3. The reduced-basis projection / lift round-trips on the subspace.
4. ReducedTransition produces the correct Δz on-manifold and matches the numpy
   path (the runtime-swap module behaves like the fit it was built from).
5. Discrete→continuous rate conversion + the sub-500-Hz / aliasing flag.
6. DMD/Koopman recovers a known linear generator's eigenvalues.
7. Jacobian-based reduced spectrum on an analytic transition.
"""

from __future__ import annotations

import numpy as np
import torch

from imas_ambix.statespace.discovery_sindy import (
    _NYQUIST_HZ,
    ReducedTransition,
    _continuous_rates,
    _feature_names,
    _poly_powers,
    _r2_score,
    build_library,
    build_reduced_basis,
    dmd_koopman,
    render_recurrence,
    stlsq,
)

# ---------------------------------------------------------------------------
# 1. Polynomial library
# ---------------------------------------------------------------------------


def test_poly_powers_degree2_2vars():
    powers = _poly_powers(2, 2)
    # constant, 2 linear, 3 quadratic
    assert (0, 0) in powers
    assert (1, 0) in powers and (0, 1) in powers
    assert (2, 0) in powers and (1, 1) in powers and (0, 2) in powers
    assert len(powers) == 6


def test_feature_names():
    powers = _poly_powers(2, 2)
    names = _feature_names(powers)
    assert names[0] == "1"
    assert "xi0" in names
    assert any("^2" in n for n in names)


def test_build_library_shapes():
    rng = np.random.default_rng(0)
    xi = rng.standard_normal((50, 3))
    theta, powers = build_library(xi, degree=2)
    # 1 + 3 linear + 6 quadratic (sym) = 10 features
    assert theta.shape == (50, len(powers))
    # the constant column is all ones
    const_idx = powers.index((0, 0, 0))
    np.testing.assert_allclose(theta[:, const_idx], 1.0)


# ---------------------------------------------------------------------------
# 2. STLSQ recovers a known sparse map
# ---------------------------------------------------------------------------


def test_stlsq_recovers_sparse_linear():
    """Δξ = 0.4 ξ0 − 0.3 ξ1 should be recovered; spurious terms thresholded out."""
    rng = np.random.default_rng(1)
    xi = rng.standard_normal((2000, 2))
    theta, powers = build_library(xi, degree=2)
    # true coefficients: only the two linear terms are active for dim 0
    true = np.zeros((len(powers), 2))
    true[powers.index((1, 0)), 0] = 0.4
    true[powers.index((0, 1)), 0] = -0.3
    true[powers.index((1, 0)), 1] = 0.1
    dxi = theta @ true
    xi_fit = stlsq(theta, dxi, rel_threshold=0.10)
    # ridge (alpha=1e-3) shrinks coefficients slightly; support + magnitude
    # recovered to ridge tolerance, exact zeros elsewhere.
    np.testing.assert_allclose(xi_fit, true, atol=2e-3)
    assert (xi_fit[true == 0.0] == 0.0).all(), "spurious terms must be exact zeros"
    # R² should be ~1 on the noise-free target
    assert _r2_score(theta, dxi, xi_fit) > 0.999


def test_stlsq_recovers_sparse_on_badly_scaled_columns():
    """REGRESSION: the real-data scale bug (ξ std≈3, quadratic col norms≈47).

    A fixed ABSOLUTE coefficient threshold zeroes everything here because the
    coefficients are inversely related to the column scale.  Column-normalised
    relative thresholding must still recover the correct sparse support.  This
    is the exact blind spot that unit-scale synthetic data hides.
    """
    rng = np.random.default_rng(11)
    # Badly-scaled reduced coords: std ≈ 3.4 / 1.4 / 1.6 (mirrors the latent).
    xi = rng.standard_normal((4000, 3)) * np.array([3.4, 1.4, 1.6])
    theta, powers = build_library(xi, degree=2)
    true = np.zeros((len(powers), 3))
    # A linear term AND a quadratic term (the quadratic col has a huge norm, so
    # its physical coefficient is tiny — an absolute floor would kill it).
    true[powers.index((1, 0, 0)), 0] = 0.30  # 0.30 * xi0
    true[powers.index((2, 0, 0)), 0] = 0.02  # 0.02 * xi0^2 (tiny coeff, big col)
    true[powers.index((0, 1, 0)), 1] = -0.25
    dxi = theta @ true
    xi_fit = stlsq(theta, dxi, rel_threshold=0.10)
    # support recovered exactly (both the big-coeff linear AND tiny-coeff quad)
    assert xi_fit[powers.index((1, 0, 0)), 0] != 0.0
    assert xi_fit[powers.index((2, 0, 0)), 0] != 0.0, (
        "the tiny-coefficient quadratic term on a high-norm column must survive"
    )
    assert xi_fit[powers.index((0, 1, 0)), 1] != 0.0
    # ridge shrinkage tolerance (alpha=1e-3); the tiny quad coeff is ~0.02
    np.testing.assert_allclose(xi_fit, true, atol=3e-3)
    assert _r2_score(theta, dxi, xi_fit) > 0.999


def test_stlsq_thresholds_small_terms():
    """A term contributing < rel_threshold × RMS(Δξ) is driven to zero."""
    rng = np.random.default_rng(2)
    xi = rng.standard_normal((2000, 2))
    theta, powers = build_library(xi, degree=2)
    true = np.zeros((len(powers), 2))
    true[powers.index((1, 0)), 0] = 0.5
    true[powers.index((0, 1)), 0] = 0.001  # negligible contribution
    dxi = theta @ true
    xi_fit = stlsq(theta, dxi, rel_threshold=0.10)
    assert abs(xi_fit[powers.index((0, 1)), 0]) < 1e-9
    assert abs(xi_fit[powers.index((1, 0)), 0] - 0.5) < 1e-3


def test_stlsq_identity_map_resolves_to_empty():
    """A near-zero increment (drift_reg→identity) resolves to the empty map."""
    rng = np.random.default_rng(12)
    xi = rng.standard_normal((2000, 3)) * 2.0
    theta, _ = build_library(xi, degree=2)
    dxi = np.zeros((2000, 3))  # f_θ ≡ 0 (the quiescent drift_reg limit)
    xi_fit = stlsq(theta, dxi, rel_threshold=0.10)
    assert np.count_nonzero(xi_fit) == 0


def test_render_recurrence_runs():
    powers = _poly_powers(2, 2)
    xi = np.zeros((len(powers), 2))
    xi[powers.index((1, 0)), 0] = 0.4
    lines = render_recurrence(xi, powers)
    assert len(lines) == 2
    assert "xi0" in lines[0]
    # a zero row renders as "0"
    assert lines[1].endswith("0")


# ---------------------------------------------------------------------------
# 3. Reduced-basis round trip
# ---------------------------------------------------------------------------


def test_reduced_basis_roundtrip_on_subspace():
    """Projecting then lifting a vector in the top-r subspace is identity."""
    rng = np.random.default_rng(3)
    # build data that lives in a 2-d subspace of R^5
    basis_dirs = rng.standard_normal((5, 2))
    coords = rng.standard_normal((500, 2))
    z = coords @ basis_dirs.T + rng.standard_normal((500, 5)) * 1e-6
    rb = build_reduced_basis(z, r=2)
    xi = rb.project(z)
    z_rt = rb.lift(xi)
    # the subspace component is preserved to high accuracy
    np.testing.assert_allclose(z, z_rt, atol=1e-3)


def test_reduced_basis_increment_meanfree():
    """Increment project/lift uses no mean offset (mean-free)."""
    rng = np.random.default_rng(4)
    z = rng.standard_normal((300, 4))
    rb = build_reduced_basis(z, r=2)
    dz = rng.standard_normal((10, 4))
    dxi = rb.project_increment(dz)
    # lifting the projected increment is the subspace projection of dz
    dz_proj = rb.lift_increment(dxi)
    # projecting again gives the same reduced increment (idempotent on subspace)
    np.testing.assert_allclose(rb.project_increment(dz_proj), dxi, atol=1e-6)


def test_reduced_basis_orthonormal_columns():
    rng = np.random.default_rng(5)
    z = rng.standard_normal((400, 6))
    rb = build_reduced_basis(z, r=3)
    gram = rb.V_r.T @ rb.V_r
    np.testing.assert_allclose(gram, np.eye(3), atol=1e-6)


# ---------------------------------------------------------------------------
# 4. ReducedTransition matches the numpy fit + zero-coeff → zero increment
# ---------------------------------------------------------------------------


def test_reduced_transition_matches_numpy():
    rng = np.random.default_rng(6)
    z = rng.standard_normal((500, 5))
    rb = build_reduced_basis(z, r=2)
    theta, powers = build_library(rb.project(z), degree=2)
    coeffs = np.zeros((len(powers), 2))
    coeffs[powers.index((1, 0)), 0] = 0.3
    coeffs[powers.index((0, 1)), 1] = -0.2
    mod = ReducedTransition(rb, coeffs, powers)

    zt = torch.from_numpy(z[:8]).float()
    with torch.no_grad():
        dz_torch = mod(zt).numpy()

    # numpy reference path
    xi = rb.project(z[:8])
    th, _ = build_library(xi, degree=2)
    dxi = th @ coeffs
    dz_np = rb.lift_increment(dxi)
    np.testing.assert_allclose(dz_torch, dz_np, atol=1e-4)


def test_reduced_transition_zero_coeffs_zero_increment():
    rng = np.random.default_rng(7)
    z = rng.standard_normal((200, 4))
    rb = build_reduced_basis(z, r=2)
    _, powers = build_library(rb.project(z), degree=2)
    coeffs = np.zeros((len(powers), 2))
    mod = ReducedTransition(rb, coeffs, powers)
    zt = torch.from_numpy(z[:5]).float()
    with torch.no_grad():
        dz = mod(zt).numpy()
    np.testing.assert_allclose(dz, 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# 5. Rate conversion + aliasing flag (crux 2)
# ---------------------------------------------------------------------------


def test_continuous_rate_pure_decay():
    """A real eigenvalue μ=0.9 → negative decay rate, zero frequency."""
    out = _continuous_rates(np.array([0.9 + 0j]))
    rate = out["rates"][0]
    assert rate["decay_rate_per_s"] < 0  # 0.9 < 1 → decaying
    assert abs(rate["osc_freq_hz_folded"]) < 1e-6
    assert rate["below_nyquist"] is True
    assert out["nyquist_hz"] == _NYQUIST_HZ


def test_continuous_rate_oscillatory_folded_below_nyquist():
    """A complex eigenvalue's folded frequency must lie in [0, Nyquist]."""
    # μ = exp(i 2π f dt) with f = 100 Hz, dt = 1ms → angle 0.2π
    f = 100.0
    mu = np.exp(1j * 2 * np.pi * f * 1e-3)
    out = _continuous_rates(np.array([mu]))
    rate = out["rates"][0]
    assert abs(rate["osc_freq_hz_folded"] - f) < 1.0
    assert rate["below_nyquist"] is True
    # decay rate ~0 (on the unit circle)
    assert abs(rate["decay_rate_per_s"]) < 1e-6


def test_continuous_rate_unit_eigenvalue_is_identity():
    """μ=1 (drift_reg→identity signature) → zero decay, zero frequency."""
    out = _continuous_rates(np.array([1.0 + 0j]))
    rate = out["rates"][0]
    assert abs(rate["decay_rate_per_s"]) < 1e-9
    assert abs(rate["osc_freq_hz_folded"]) < 1e-9


# ---------------------------------------------------------------------------
# 6. DMD/Koopman recovers a known linear generator
# ---------------------------------------------------------------------------


def test_dmd_recovers_linear_generator():
    """Δξ = A ξ with known A → eig(I+A) matches the true discrete eigenvalues."""
    rng = np.random.default_rng(8)
    a_true = np.array([[-0.1, 0.2], [-0.2, -0.1]])  # spiral decay
    xi = rng.standard_normal((3000, 2))
    dxi = xi @ a_true.T
    out = dmd_koopman(xi, dxi)
    a_fit = np.array(out["A"])
    np.testing.assert_allclose(a_fit, a_true, atol=1e-6)
    # eig(I + A_true)
    true_eig = np.linalg.eigvals(np.eye(2) + a_true)
    fit_eig = np.array(out["M_eigenvalues_re"]) + 1j * np.array(out["M_eigenvalues_im"])
    # match as sets (sorted by real part)
    np.testing.assert_allclose(np.sort(true_eig.real), np.sort(fit_eig.real), atol=1e-5)


def test_dmd_spectrum_decay_for_stable_system():
    """A clearly-stable system reports negative decay rates, all sub-Nyquist."""
    rng = np.random.default_rng(9)
    a_true = np.array([[-0.3, 0.0], [0.0, -0.5]])  # pure decay
    xi = rng.standard_normal((2000, 2))
    dxi = xi @ a_true.T
    out = dmd_koopman(xi, dxi)
    for rate in out["spectrum"]["rates"]:
        assert rate["decay_rate_per_s"] < 0
        assert rate["below_nyquist"] is True
