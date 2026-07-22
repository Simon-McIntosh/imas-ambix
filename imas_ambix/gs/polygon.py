"""Finite-area Green's functions: complete toroidal conductors of POLYGON section.

Generalises :mod:`imas_ambix.gs.cylinder` (rectangular section, Urankar Part
III) to an arbitrary polygon cross-section, from L. K. Urankar, *"Vector
potential and magnetic field of current-carrying finite arc segment in
analytical form — Part V: polygon cross section,"* IEEE Trans. Magn. 26(3),
1171–1180 (1990) — implemented from the paper (nova's abandoned
``biot/polygon.py`` was consulted only as a fingerprint of the edge
parametrisation).

Why: a slanted (parallelogram) or trapezoidal conductor — the vessel end
crowns and P2 stability-plate arms, a non-rectangular coil pack — is neither a
filled axis-aligned box (the rectangle kernel's assumption) nor cheaply
represented by a multi-filament tiling (O(N) cost, Riemann-limited accuracy).
Urankar converts the cross-section surface integral into a CONTOUR sum over
the polygon's edges (Stokes), does the edge-parameter integral in closed form,
and leaves — for the axisymmetric full turn — a single smooth 1-D integral
over the arc angle φ per edge.

The vector potential.  Per edge ν with endpoints (r'ᵥ₁, z'ᵥ₁) → (r'ᵥ₂, z'ᵥ₂)
[paper eqs (7)–(9a)]: parametrise the edge r'(u) = r₁ + b₁u with u = z' − z,
slope b₁ = Δr/Δz, intercept r₁ = r'ᵥ₁ − b₁·(z'ᵥ₁ − z).  The u-integral of eq
(8) has the closed antiderivative (eq 9, with a₀² = 1 + b₁², G² = u² + r²sin²φ,
B² = (r₁ − r cosφ)² + a₀²r²sin²φ, D² = G² + (r' − r cosφ)², Γ = u + b₁(r'−r cosφ),
β₁ = (r'−r cosφ)/G, β₂ = Γ/B, β₃ = [u(r'−r cosφ) − b₁G²]/(r sinφ D)):

    g(u, φ) = Γ·D/(2a₀²) + u·r cosφ·arsinh β₁
            + [B² + 2a₀²·r cosφ·(r₁ − r cosφ)]/(2a₀³)·arsinh β₂
            − (r²/2)·sin 2φ·arctan β₃

NOTE the 1990 typesetting trap: the printed "Γ(φ)/2a₀²D(φ)" means Γ·D/(2a₀²)
— D(φ) is a NUMERATOR factor.  Fixed three ways: dimensional analysis, the
b₁ = 0 reduction of eq (9) back to the eq (8) integrand, and eq (12) where
D(φ) → a(1 − k²sin²α)^{1/2} appears explicitly as a factor.  The paper's
Appendix B errata correct Parts II–IV only; Part V's own equations stand as
printed.  ``g`` reproduces the raw cross-section integral ∫∫ r'/D dr'dz'
edge-by-edge to machine precision (regression-pinned against a dense 2-D
quadrature and against the rectangular kernel).

Then Â_φ(r, z) = −Σᵥ ∫ cosφ · [g]ᵤᵥ₁ᵘᵛ² dφ  (eqs 3b, 10a/b, j = φ), and the
axisymmetric flux ψ = 2π μ0 R · Â_φ / (4π A) per ampere of total current.

The field.  Rather than transcribe the paper's closed B integrands (eq 11b —
a longer, more error-prone form), the field is the EXACT curl of the verified
vector potential, B_Z = (1/2πR) ∂ψ/∂R and B_R = −(1/2πR) ∂ψ/∂Z, evaluated by
COMPLEX-STEP differentiation of ψ.  Complex-step (∂f/∂x = Im f(x + ih)/h with
h ~ 1e-30) is exact to machine precision — it has none of the subtractive
cancellation of a real finite difference — so this is analytic differentiation,
not a numerical approximation, and it guarantees ψ↔B consistency by
construction.  For a rectangle it reproduces ``cylinder_greens``' B to ~1e-15.

For the FULL TURN (axisymmetric ring, arc = 2π) the φ-integrand is even about
φ = π, so it is evaluated on [0, π] and doubled, with composite Gauss–Legendre
panels.  The integrand is analytic for every target off the section boundary —
including INSIDE the conductor — because D² ≥ r²sin²φ > 0 at the interior
quadrature nodes; convergence is spectral.  This mirrors the in-tree
precedent: the rectangle kernel itself carries a 785-point arcsinh (ζ)
quadrature inside its "closed" antiderivative, so a smooth bounded 1-D
quadrature per edge is the established cost model.  (A fully-elliptic closed
assembly via complete/incomplete K/E/Π — :mod:`imas_ambix.gs.elliptic` — remains
available as a later optimisation if the accuracy/cost benchmark asks for it.)

Sign/units conventions match :func:`imas_ambix.gs.cylinder.cylinder_greens`
(and hence the point-filament ``greens_psi``/``greens_bz_br``), per ampere of
TOTAL conductor current, with uniform azimuthal current density J_φ = 1/A:

    ψ  [Wb/A]  = 2π μ0 R · Â_φ / (4π A)
    B  [T/A]   = curl of ψ.
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss

MU0 = 4.0e-7 * np.pi

# Composite Gauss–Legendre rule on φ ∈ [0, π] (doubled by even symmetry): the
# integrand is analytic on the open interval, so a modest panel count converges
# past 1e-12.  16 panels × 48 nodes reproduce the rectangular kernel to ~1e-11.
_N_PANELS = 16
_N_NODES = 48
_CSTEP = 1e-30  # complex-step increment (∂ via Im part; no cancellation)


def _phi_rule(n_panels: int, n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    x, w = leggauss(n_nodes)
    edges = np.linspace(0.0, np.pi, n_panels + 1)
    lo, hi = edges[:-1, None], edges[1:, None]
    phi = (0.5 * (hi - lo) * x[None, :] + 0.5 * (hi + lo)).ravel()
    wts = (0.5 * (hi - lo) * w[None, :]).ravel()
    return phi, wts


def _psi_hat(
    r: np.ndarray,
    z: np.ndarray,
    v: np.ndarray,
    cosp: np.ndarray,
    sinp: np.ndarray,
    sin2p: np.ndarray,
    w_cos: np.ndarray,
    sign: float,
    area: float,
) -> np.ndarray:
    """Complex-analytic ψ(r, z) per ampere from the verified edge antiderivative.

    ``r, z`` are ``(T, 1)`` (possibly complex, for the complex-step curl); the φ
    node arrays are ``(Q,)``.  Returns ``(T,)`` — real for real inputs.
    """
    n = len(v)
    a_hat = np.zeros(r.shape[0], dtype=np.result_type(r.dtype, z.dtype))
    z_scale = max(float(np.ptp(v[:, 1])), 1e-6)
    for i in range(n):
        ra, za = v[i]
        rb, zb = v[(i + 1) % n]
        dz = zb - za
        if abs(dz) < 1e-12 * z_scale:
            continue  # horizontal edge: f_ν(φ) vanishes (paper eq 7a)
        b1 = (rb - ra) / dz
        a02 = 1.0 + b1 * b1
        a03 = a02 * np.sqrt(a02)
        r1 = ra - b1 * (za - z)  # (T, 1) — depends on z
        for u, s_lim in ((zb - z, 1.0), (za - z, -1.0)):
            rp = r1 + b1 * u
            rmc = rp - r * cosp
            r1mc = r1 - r * cosp
            g2 = u * u + (r * sinp) ** 2
            b2 = r1mc * r1mc + a02 * (r * sinp) ** 2
            d = np.sqrt(g2 + rmc * rmc)
            cap_gamma = u + b1 * rmc
            ash1 = np.arcsinh(rmc / np.sqrt(g2))
            ash2 = np.arcsinh(cap_gamma / np.sqrt(b2))
            at3 = np.arctan((u * rmc - b1 * g2) / (r * sinp * d))
            g = (
                cap_gamma * d / (2.0 * a02)
                + u * r * cosp * ash1
                + (b2 + 2.0 * a02 * r * cosp * r1mc) / (2.0 * a03) * ash2
                - 0.5 * r * r * sin2p * at3
            )
            # −[g]ᵤₐᵘᵇ  ⇒  −g(ub)·(+1) − g(ua)·(−1); fold the ±1 into s_lim.
            a_hat += -s_lim * (g @ w_cos)
    a_hat *= 2.0  # [0, π] half-turn ×2
    norm = sign * MU0 / (4.0 * np.pi * area)
    return 2.0 * np.pi * r[:, 0] * norm * a_hat


def polygon_greens(
    target_r: np.ndarray,
    target_z: np.ndarray,
    vertices: np.ndarray,
    *,
    n_panels: int = _N_PANELS,
    n_nodes: int = _N_NODES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(ψ, B_R, B_Z) per ampere at targets, from a polygon-section ring.

    ``vertices`` — (n, 2) array of the section's (r, z) corners, either
    orientation, no repeated closing vertex.  Returns arrays shaped like
    ``target_r``: total poloidal flux ψ [Wb/A] and field components [T/A],
    smooth everywhere including inside the conductor.  Horizontal edges
    (Δz = 0) contribute nothing (paper eq 7a) and are skipped.
    """
    v = np.asarray(vertices, dtype=np.float64)
    tr = np.asarray(target_r, dtype=np.float64)
    tz = np.asarray(target_z, dtype=np.float64)
    shape = tr.shape
    r = tr.ravel()[:, None]
    z = tz.ravel()[:, None]

    rolled = np.roll(v, -1, axis=0)
    signed_area2 = float(np.sum(v[:, 0] * rolled[:, 1] - rolled[:, 0] * v[:, 1]))
    area = 0.5 * abs(signed_area2)
    # the counter-clockwise edge sum yields −f(φ); one orientation sign fixes
    # all three components at once (pinned by the rectangle-reduction and
    # filament oracles in tests/gs/test_polygon.py).
    sign = -np.sign(signed_area2)

    phi, wts = _phi_rule(n_panels, n_nodes)
    cosp = np.cos(phi)
    sinp = np.sin(phi)
    sin2p = np.sin(2.0 * phi)
    w_cos = wts * cosp

    def psi_at(rr: np.ndarray, zz: np.ndarray) -> np.ndarray:
        return _psi_hat(rr, zz, v, cosp, sinp, sin2p, w_cos, sign, area)

    # one complex-step pass in r gives ψ (real part) and ∂ψ/∂R (imag/h → B_Z);
    # one in z gives ∂ψ/∂Z (imag/h → B_R).  Exact-to-machine-precision curl.
    h = _CSTEP
    psi_r = psi_at(r + 1j * h, z)
    dpsi_dz = psi_at(r, z + 1j * h).imag / h
    psi = psi_r.real
    dpsi_dr = psi_r.imag / h
    two_pi_r = 2.0 * np.pi * r[:, 0]
    bz = dpsi_dr / two_pi_r
    br = -dpsi_dz / two_pi_r
    return psi.reshape(shape), br.reshape(shape), bz.reshape(shape)


__all__ = ["polygon_greens", "MU0"]
