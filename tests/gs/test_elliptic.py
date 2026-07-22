"""Validation of the incomplete elliptic integral of the third kind.

Π(n; φ, m) = ∫₀^φ dθ / [(1 − n sin²θ) √(1 − m sin²θ)]   (m = k²)

is absent from :mod:`scipy.special`.  This module builds it on scipy's Carlson
symmetric forms ``elliprf`` / ``elliprj`` and pins it — to 1e-12 relative — against
mpmath's arbitrary-precision ``ellippi`` across the parameter ranges the Urankar
Part V polygon field integrals exercise, including:

* the ordinary branch (characteristic ``n < 1``) at amplitudes φ ∈ [0, π] (so the
  half-period reduction past π/2 is covered);
* the Cauchy-principal-value branch (``n > 1``) both below and above the singular
  amplitude θ_s = arcsin(1/√n) where the integrand pole is crossed.
"""

from __future__ import annotations

import mpmath as mp
import numpy as np
import pytest

from imas_ambix.gs.elliptic import ellippi, ellippi_complete

mp.mp.dps = 40


def _ref_inc(n: float, phi: float, m: float) -> float:
    """mpmath incomplete Π(n; φ, m).

    For the characteristic ``n > 1`` past the singular amplitude θ_s the integrand
    pole is crossed and mpmath returns the *complex* analytic continuation.  The
    real-valued third-kind integral (DLMF/Carlson, and hence scipy ``elliprj`` with
    a negative fourth argument) is its **real part** — the Cauchy principal value —
    so that is the reference the pole-crossing branch is pinned against.
    """
    return float(mp.re(mp.ellippi(n, phi, m)))


def _ref_complete(n: float, m: float) -> float:
    return float(mp.ellippi(n, m))


# ---- ordinary branch: n < 1, amplitudes spanning past π/2 ------------------

_N_ORD = [-3.0, -0.7, 0.0, 0.25, 0.6, 0.9, 0.99]
_PHI = [0.05, 0.3, 0.7, np.pi / 4, 1.2, np.pi / 2, 2.0, 2.9]
_M = [0.0, 0.15, 0.4, 0.7, 0.9, 0.99]


@pytest.mark.parametrize("n", _N_ORD)
@pytest.mark.parametrize("m", _M)
def test_incomplete_ordinary_branch(n, m):
    phi = np.array(_PHI)
    got = ellippi(n, phi, m)
    want = np.array([_ref_inc(n, float(p), m) for p in phi])
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("m", _M)
def test_complete_matches_pi_over_two_limit(m):
    n = np.array(_N_ORD)
    comp = ellippi_complete(n, m)
    inc_half = ellippi(n, np.pi / 2, m)
    want = np.array([_ref_complete(float(v), m) for v in n])
    np.testing.assert_allclose(comp, want, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(comp, inc_half, rtol=1e-12, atol=1e-13)


# ---- Cauchy principal-value branch: n > 1 ----------------------------------

_N_PV = [1.3, 2.0, 5.0, 25.0]


@pytest.mark.parametrize("n", _N_PV)
def test_incomplete_pv_below_singular_amplitude(n):
    """Amplitudes short of θ_s = arcsin(1/√n): ordinary (non-PV) integral."""
    theta_s = np.arcsin(1.0 / np.sqrt(n))
    phi = np.linspace(0.02, 0.95 * theta_s, 6)
    for m in (0.0, 0.5, 0.9):
        got = ellippi(n, phi, m)
        want = np.array([_ref_inc(n, float(p), m) for p in phi])
        np.testing.assert_allclose(got, want, rtol=1e-11, atol=1e-13)


@pytest.mark.parametrize("n", _N_PV)
def test_incomplete_pv_above_singular_amplitude(n):
    """Amplitudes past θ_s: genuine Cauchy principal value."""
    theta_s = np.arcsin(1.0 / np.sqrt(n))
    phi = np.linspace(1.05 * theta_s, min(1.5 * theta_s, np.pi / 2 - 1e-3), 6)
    for m in (0.0, 0.5, 0.9):
        got = ellippi(n, phi, m)
        want = np.array([_ref_inc(n, float(p), m) for p in phi])
        np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-12)


# ---- vectorisation / broadcasting ------------------------------------------


def test_broadcasts_over_all_arguments():
    n = np.array([0.3, 0.6, 0.9])[:, None]
    phi = np.array([0.4, 0.8, 1.2, 1.5])[None, :]
    got = ellippi(n, phi, 0.5)
    assert got.shape == (3, 4)
    want = np.array(
        [
            [_ref_inc(float(nn), float(pp), 0.5) for pp in phi.ravel()]
            for nn in n.ravel()
        ]
    )
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-13)


def test_odd_in_amplitude():
    """Π is odd in φ: Π(n; −φ, m) = −Π(n; φ, m)."""
    phi = np.array([0.3, 0.9, 1.4])
    np.testing.assert_allclose(
        ellippi(0.6, -phi, 0.4), -ellippi(0.6, phi, 0.4), rtol=1e-13
    )
