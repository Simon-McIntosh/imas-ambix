"""Finite-area Green's functions: complete toroidal conductors of rectangular section.

Extracted from the ``nova.biot`` cylinder formulation (author Simon McIntosh —
``nova/biot/{constants,cylinder,zeta}.py``), re-implemented here self-contained
(numpy + scipy.special only) so the equilibrium decoder stays lean and
machine-agnostic.  Validated against golden values generated from nova itself
(see ``tests/gs/test_cylinder.py``).

Why finite-area: a point-filament Green's function is log-singular at the
source, so any evaluation grid that approaches a conductor — the in-vessel PF
winding packs, or the plasma current cells the GS solve distributes current
over — inherits a spurious near-field spike.  The finite-area kernel spreads
unit current uniformly over the rectangular cross-section and is smooth
everywhere, *including inside the conductor*, which is what a ψ field read for
topology (axis / X-points / LCFS) requires.

Formulation: closed-form antiderivatives of the uniformly-distributed ring
current — complete elliptic integrals K, E, Π (Carlson forms) plus a 1-D
``zeta`` quadrature (midpoint rule on an arcsinh integrand) — evaluated at the
four cross-section corners and combined with alternating signs (the standard
definite-double-integral corner rule), normalised per ampere of TOTAL conductor
current:

    ψ  = 2π μ0 R · Aphi_corner / (2π A)          [Wb/A]
    B  = μ0 · {Br,Bz}_corner / (2π A)            [T/A]

with A the cross-section area.  Sign/units conventions match
:mod:`imas_ambix.gs.operator`'s point-filament ``greens_psi``/``greens_bz_br``
(the far-field limit — pinned by test).
"""

from __future__ import annotations

import numpy as np
import scipy.special  # type: ignore[import-untyped]

MU0 = 4.0e-7 * np.pi
_EPS = 2.0 * np.finfo(float).eps


def _sign(x: np.ndarray) -> np.ndarray:
    """Sign with a dead-band: 0 within numerical noise of zero."""
    return np.where(np.abs(x) > 1e4 * _EPS, np.sign(x), 0.0)


def _ellipp(n: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Complete elliptic integral of the 3rd kind via Carlson symmetric forms."""
    x = np.zeros_like(n)
    y = 1.0 - m
    z = np.ones_like(n)
    p = 1.0 - n
    rf = scipy.special.elliprf(x, y, z)
    rj = scipy.special.elliprj(x, y, z, p)
    return rf + rj * n / 3.0


def _zeta(
    rs: np.ndarray, r: np.ndarray, gamma: np.ndarray, *, points: int = 785
) -> np.ndarray:
    """The ζ integral: midpoint quadrature of the arcsinh integrand over α∈[0, π/2].

    ζ = ∫ arcsinh((rs − r·cos φ)/√(γ² + r² sin²φ)) dα with φ = π − 2α — the one
    non-closed-form piece of the cylinder antiderivative.  ``points`` matches
    the reference implementation's resolution (500 per unit α → 785 for π/2).
    """
    alpha_max = np.pi / 2.0
    dalpha = alpha_max / (points - 1)
    alpha = np.linspace(0.0, alpha_max, points)[:-1] + dalpha / 2.0  # midpoints
    phi = np.pi - 2.0 * alpha  # (Q,)
    sin2 = np.sin(phi) ** 2
    cosphi = np.cos(phi)
    # broadcast: (..., 1) against (Q,)
    g2 = gamma[..., None] ** 2 + r[..., None] ** 2 * sin2
    integrand = np.arcsinh((rs[..., None] - r[..., None] * cosphi) / np.sqrt(g2))
    return dalpha * integrand.sum(axis=-1)


def _corner_fields(
    rs: np.ndarray, zs: np.ndarray, r: np.ndarray, z: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Antiderivative coefficients (Aphi_hat, Br_hat, Bz_hat) at one corner set.

    All inputs broadcast to the same shape ``(..., 4)`` — target coordinates
    repeated over the four source-section corners.
    """
    gamma = zs - z
    a2 = gamma**2 + (rs + r) ** 2
    a = np.sqrt(a2)
    b = rs + r
    c2 = gamma**2 + r**2
    c = np.sqrt(c2)
    k2 = (1.0 - _EPS) * 4.0 * r * rs / a2
    v = 1.0 + k2 * (gamma**2 - b * r) / (2.0 * r * rs)
    ellip_k = scipy.special.ellipk(k2)
    ellip_e = scipy.special.ellipe(k2)
    u_coef = k2 * (4.0 * gamma**2 + 3.0 * rs**2 - 5.0 * r**2) / (4.0 * r)

    np2 = {
        1: 2.0 * r / (r - c - _EPS),
        2: (1.0 - _EPS) * 2.0 * r / (r + c),
        3: (1.0 - _EPS) * 4.0 * r * rs / b**2,
    }
    pi3 = {p: _ellipp(np2[p], k2) for p in (1, 2, 3)}

    qr = {p: (rs - (-1.0) ** p * c) * np2[p] * gamma**2 * c / r for p in (1, 2)}
    qr[3] = np.zeros_like(r)
    qz = {p: (rs - (-1.0) ** p * c) * -2.0 * gamma * c * np2[p] for p in (1, 2)}
    qz[3] = gamma * b * (rs - r) * np2[3]
    pphi = {
        p: (rs - (-1.0) ** p * c) * np2[p] * c * (3.0 * r**2 - c2) / (2.0 * r)
        for p in (1, 2)
    }
    pphi[3] = -rs / b * (rs - r) * (3.0 * r**2 - rs**2)

    def p_sum(coef: dict[int, np.ndarray]) -> np.ndarray:
        out = np.zeros_like(coef[1])
        for p in (1, 2, 3):
            out += (-1.0) ** p * coef[p] * pi3[p]
        return out

    cphi = -1.0 / 3.0 * r**2 * np.pi / 2.0 * _sign(gamma) * (_sign(rs - r) + 1.0)
    dz_coef = 3.0 / r * cphi
    zeta = _zeta(rs, r, gamma)

    aphi_hat = (
        cphi
        + gamma * r * zeta
        + gamma * a / (6.0 * r) * (u_coef * ellip_k - 2.0 * rs * ellip_e)
        + gamma / (6.0 * a * r) * p_sum(pphi)
    )
    br_hat = (
        r * zeta
        - a / (2.0 * r) * rs * (ellip_e - v * ellip_k)
        - 1.0 / (4.0 * a * r) * p_sum(qr)
    )
    bz_hat = (
        dz_coef
        + 2.0 * gamma * zeta
        - a / (2.0 * r) * 1.5 * gamma * k2 * ellip_k
        - 1.0 / (4.0 * a * r) * p_sum(qz)
    )
    return aphi_hat, br_hat, bz_hat


def cylinder_greens(
    target_r: np.ndarray,
    target_z: np.ndarray,
    a: float,
    z0: float,
    da: float,
    dz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(ψ, B_R, B_Z) per ampere at targets, from a rectangular-section ring.

    ``a, z0`` — section centroid [m]; ``da, dz`` — radial/vertical extents [m].
    Returns arrays shaped like ``target_r``: total poloidal flux ψ [Wb/A] and
    field components [T/A], smooth everywhere including inside the section.
    """
    tr = np.asarray(target_r, dtype=np.float64)
    tz = np.asarray(target_z, dtype=np.float64)
    # corner order (matching the reference): (−,−), (+,−), (+,+), (−,+)
    rs = np.stack(
        [np.full(tr.shape, a + d * da / 2.0) for d in (-1, 1, 1, -1)], axis=-1
    )
    zs = np.stack(
        [np.full(tr.shape, z0 + d * dz / 2.0) for d in (-1, -1, 1, 1)], axis=-1
    )
    r4 = np.repeat(tr[..., None], 4, axis=-1)
    z4 = np.repeat(tz[..., None], 4, axis=-1)

    aphi_hat, br_hat, bz_hat = _corner_fields(rs, zs, r4, z4)
    area = da * dz

    def corner(data: np.ndarray) -> np.ndarray:
        return (
            1.0
            / (2.0 * np.pi * area)
            * ((data[..., 2] - data[..., 3]) - (data[..., 1] - data[..., 0]))
        )

    aphi = corner(aphi_hat)
    psi = 2.0 * np.pi * MU0 * tr * aphi
    br = MU0 * corner(br_hat)
    bz = MU0 * corner(bz_hat)
    return psi, br, bz


__all__ = ["cylinder_greens", "MU0"]
