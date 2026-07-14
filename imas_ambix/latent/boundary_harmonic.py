"""Source-free toroidal-harmonic boundary read in the vacuum annulus.

In the vacuum annulus between the plasma and the sensors J_phi = 0, so the
poloidal flux solves the HOMOGENEOUS Grad-Shafranov operator exactly,

    Delta* psi  =  d2psi/dR2 - (1/R) dpsi/dR + d2psi/dZ2  =  0 .

Unlike a free interior patch current or a low-order current-moment fit -- both
of which reconstruct a current whose j_phi is non-zero in the vacuum region and
so VIOLATES the premise the boundary read rests on -- this module represents the
plasma-produced flux DIRECTLY in a basis that cannot commit any current to the
annulus: the toroidal harmonics (ring functions) that separate Delta* psi = 0
about a fixed pole placed near the magnetic axis.

Representation
--------------
About a toroidal-coordinate pole (focal ring of radius ``pole_r`` at height
``pole_z``) the homogeneous GS operator separates.  Its solutions carry the
order-1 half-integer-degree Legendre functions (order 1 -- not the scalar
Laplace order 0 -- because of the -(1/R) d/dR term) with a
``sqrt(cosh eta - cos theta)`` prefactor:

    psi_{n,c}(eta, theta) = sqrt(cosh eta - cos theta) * F_n(cosh eta) * cos(n theta)
    psi_{n,s}(eta, theta) = sqrt(cosh eta - cos theta) * F_n(cosh eta) * sin(n theta)

with F_n either P^1_{n-1/2} or Q^1_{n-1/2}.  The plasma current sits INSIDE the
pole and the flux is observed OUTWARD in the annulus toward the sensors, so the
physically correct set is the one that stays regular in the source-free region
out to infinity (eta -> 0); the numerical filament-recovery test selects it and
pins :data:`_DECAYING_KIND`.

Conventions (matching :mod:`imas_ambix.gs.operator` /
:mod:`imas_ambix.latent.patch_basis`): every psi column carries the TOTAL
poloidal flux ``Phi = 2 pi R A_phi`` [Wb], so a flux loop reads ``psi`` and a
B-probe reads the orientation-projected field with ``B_R = -(1/(2 pi R)) dPhi/dZ``,
``B_Z = +(1/(2 pi R)) dPhi/dR``.  The harmonic basis carries NO current on grid
cells, so ``PatchBasis.m_sens`` is NOT used -- the source-free premise is
structural, not merely low-order.  The KNOWN coil field is added separately by
the caller through the harness's thick-cylinder ``hybrid_greens`` coil columns
(``EquilibriumGrid._coil_psi_columns``); NO point-filament coil term is ever
introduced here (binding: all coil couplings are the finite-area cylinder
kernel).

Nothing here imports from ``eval`` or ``worldmodel`` -- the firewalled EFIT
referee only ever *scores* the psi this produces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache

import numpy as np

MU0 = 4.0e-7 * np.pi
"""Vacuum permeability [T*m/A]."""

_D_FLOOR = 1.0e-12
"""Numerical floor on the product of focal distances (the pole itself)."""

# The exterior-regular decaying set, pinned by the filament-recovery test
# (test_boundary_harmonic.py::test_recovers_filament_flux_in_annulus).  A field
# produced by sources INSIDE the pole and observed OUTWARD in the source-free
# annulus must stay finite out to infinity (eta -> 0, cosh eta -> 1); that fixes
# which of {P, Q} is physical.  Kept as a module constant so the gate and the
# tests agree on one representation.
_DECAYING_KIND = "P"


# --- toroidal coordinates ---------------------------------------------------


def toroidal_coords(
    r: np.ndarray, z: np.ndarray, pole_r: float, pole_z: float = 0.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Toroidal coordinates about a focal ring of radius ``pole_r`` at ``pole_z``.

    Returns ``(cosh_eta, cos_theta, sin_theta, theta, cosh_eta_minus_cos_theta)``
    for the forward transform

        R = a sinh(eta) / (cosh eta - cos theta),
        Z = pole_z + a sin(theta) / (cosh eta - cos theta),   a = pole_r,

    whose self-consistent inverse (derived, round-trip pinned by a test) is

        d1^2 = (R - a)^2 + (Z - pole_z)^2,  d2^2 = (R + a)^2 + (Z - pole_z)^2,
        D    = d1 d2,   P = R^2 + (Z - pole_z)^2,
        cosh eta = (P + a^2)/D,  cos theta = (P - a^2)/D,  sin theta = 2 a (Z-pole_z)/D,
        cosh eta - cos theta = 2 a^2 / D .

    The focal ring (R=a, Z=pole_z) is eta -> infinity (D -> 0); the symmetry
    axis R=0 and spatial infinity are eta -> 0 (cosh eta -> 1).
    """
    a = float(pole_r)
    dz = np.asarray(z, dtype=np.float64) - float(pole_z)
    r = np.asarray(r, dtype=np.float64)
    d1sq = (r - a) ** 2 + dz**2
    d2sq = (r + a) ** 2 + dz**2
    d = np.sqrt(np.maximum(d1sq * d2sq, _D_FLOOR**2))
    p = r**2 + dz**2
    cosh_eta = (p + a**2) / d
    cos_theta = (p - a**2) / d
    sin_theta = 2.0 * a * dz / d
    theta = np.arctan2(sin_theta, cos_theta)
    cmc = 2.0 * a**2 / d  # cosh eta - cos theta (always > 0)
    # cosh eta >= 1 by construction; clip tiny round-off below 1.
    cosh_eta = np.maximum(cosh_eta, 1.0)
    return cosh_eta, cos_theta, sin_theta, theta, cmc


# --- half-integer-degree order-1 Legendre (ring) functions ------------------


def ring_P1(order: int, x: np.ndarray) -> np.ndarray:
    """Vectorised ``P^1_{n-1/2}(x)`` for ``n = 0..order``, ``x = cosh eta >= 1``.

    The fast path that makes a PER-SLICE pole affordable (mpmath, at ~10 us a
    scalar, cannot be re-run per slice once the pole moves and the column cache
    is defeated).  Built from the order-0 ring-function elliptic-integral seeds

        P_{-1/2}(x) = (2/pi) sqrt(2/(x+1)) K(m),   m = (x-1)/(x+1)
        P_{ 1/2}(x) = (2/pi) [ sqrt(2(x+1)) E(m) - sqrt(2/(x+1)) K(m) ]

    (``scipy.special.ellipk/ellipe`` take the PARAMETER ``m = k^2``, not the
    modulus), climbed in DEGREE by the stable forward recurrence
    ``(nu+1)P_{nu+1} = (2nu+1) x P_nu - nu P_{nu-1}`` (P is the dominant, forward-
    stable solution -- Gil, Segura & Temme), then raised to order 1 by
    ``P^1_nu = (x^2-1)^{-1/2} nu (x P_nu - P_{nu-1})`` with the degree-reflection
    ``P_{-3/2} = P_{1/2}``.  Verified against mpmath to ~1e-10 by
    ``test_boundary_harmonic.py::test_fast_ring_matches_mpmath``.

    Returns ``(order+1, N)``.
    """
    from scipy.special import ellipe, ellipk  # noqa: PLC0415

    x = np.asarray(x, dtype=np.float64)
    # x = 1 is spatial infinity/axis (P^1 -> 0 there); clamp just off it so the
    # (x^2-1)^{-1/2} order-raise stays finite (the far field is masked anyway).
    x = np.maximum(x, 1.0 + 1e-12)
    m = (x - 1.0) / (x + 1.0)
    big_k = ellipk(m)
    big_e = ellipe(m)
    s_lo = np.sqrt(2.0 / (x + 1.0))
    s_hi = np.sqrt(2.0 * (x + 1.0))
    two_pi = 2.0 / np.pi
    # order-0 degree list d[j] = P_{j-1/2}
    d = [two_pi * s_lo * big_k, two_pi * (s_hi * big_e - s_lo * big_k)]
    for j in range(1, order + 1):
        nu = j - 0.5
        d.append(((2.0 * nu + 1.0) * x * d[j] - nu * d[j - 1]) / (nu + 1.0))
    inv_s = 1.0 / np.sqrt(x * x - 1.0)
    out = np.empty((order + 1, x.size), dtype=np.float64)
    for n in range(order + 1):
        nu = n - 0.5
        d_prev = d[1] if n == 0 else d[n - 1]  # P_{-3/2} = P_{1/2}
        out[n] = nu * (x * d[n] - d_prev) * inv_s
    return out


@cache
def _ring_fn(n: int, kind: str):
    """Return a vectorised evaluator x -> F^1_{n-1/2}(x) for x = cosh eta >= 1.

    Uses mpmath's Legendre functions of the SECOND-argument-region (type 3,
    the branch defined for real argument > 1 -- the ring/toroidal functions).
    Cached per (degree, kind); the returned callable vectorises over a numpy
    array.  mpmath is the correctness reference; the evaluation cost is
    negligible because the fit only touches ~10^2 sensor rows and the grid eval
    only the scored/plotted slices.
    """
    import mpmath  # noqa: PLC0415

    deg = n - 0.5
    legen = mpmath.legenp if kind == "P" else mpmath.legenq

    def scalar(x: float) -> float:
        # type=3 is the real-argument-(x>1) branch; Q picks up a spurious
        # imaginary part from the branch cut, so keep the real part (each of the
        # real/imag parts independently solves the Legendre ODE).
        return float(mpmath.re(legen(deg, 1, float(x), type=3)))

    vec = np.vectorize(scalar, otypes=[np.float64])

    def evaluate(x: np.ndarray) -> np.ndarray:
        return vec(np.asarray(x, dtype=np.float64))

    return evaluate


# --- harmonic basis ---------------------------------------------------------


@dataclass
class HarmonicFitConfig:
    """Configuration for the source-free toroidal-harmonic annulus fit.

    ``pole_r`` / ``pole_z`` place the toroidal-coordinate focal ring near the
    nominal magnetic axis (fixed campaign geometry -- the locked
    ``pole-and-truncation`` decision, no per-slice pole).  ``order`` is the max
    harmonic index ``n``; the CI-gated ladder sweeps it on train shots and
    freezes it.  ``kind`` is the radial set (defaults to the exterior-regular
    decaying set).  ``ridge`` is a tiny numerical floor in a column-normalised
    frame.  ``ip_anchor`` adds the poloidal-circulation Ip constraint.
    """

    pole_r: float = 0.9
    pole_z: float = 0.0
    order: int = 3
    kind: str = _DECAYING_KIND
    ridge: float = 1e-8
    ip_anchor: bool = False


def harmonic_labels(order: int) -> list[str]:
    """Column labels ``P<n>c`` / ``P<n>s`` for the harmonic basis of ``order``."""
    labels = ["h0"]
    for n in range(1, order + 1):
        labels += [f"h{n}c", f"h{n}s"]
    return labels


# Memoise the (mpmath-backed) column matrices by exact (points, config) key.
# The pole is FIXED and the grid + sensor geometry are FIXED per campaign, so
# every slice of a campaign asks for the SAME (r, z) arrays -- the gate and the
# figure runs would otherwise pay the mpmath grid evaluation (~30 s) per slice.
# This is pure caching (no new math); the ring-function values are unchanged.
_COLUMN_CACHE: dict[tuple, tuple[np.ndarray, list[str]]] = {}
_COLUMN_CACHE_MAX = 64


def harmonic_columns(
    r: np.ndarray, z: np.ndarray, cfg: HarmonicFitConfig
) -> tuple[np.ndarray, list[str]]:
    """Toroidal-harmonic flux columns ``(n_pts, K)`` at points ``(r, z)``.

    Column ``k`` is a source-free flux ``psi_k(R, Z)`` (Delta* psi_k = 0 away
    from the pole).  ``K = 2*order + 1``: one ``n=0`` term plus a cos/sin pair
    per ``n = 1..order``.  Each column carries the sqrt(cosh eta - cos theta)
    prefactor times the order-1 half-integer Legendre radial function times the
    angular factor.

    Results are memoised by the exact ``(r, z, cfg)`` key (see
    :data:`_COLUMN_CACHE`) -- repeated identical point sets (the fixed grid /
    sensor geometry across a campaign's slices) return instantly.
    """
    r_arr = np.asarray(r, dtype=np.float64)
    z_arr = np.asarray(z, dtype=np.float64)
    key = (
        r_arr.tobytes(),
        z_arr.tobytes(),
        r_arr.shape,
        float(cfg.pole_r),
        float(cfg.pole_z),
        int(cfg.order),
        cfg.kind,
    )
    hit = _COLUMN_CACHE.get(key)
    if hit is not None:
        return hit[0], list(hit[1])
    m, labels = _harmonic_columns_uncached(r_arr, z_arr, cfg)
    if len(_COLUMN_CACHE) >= _COLUMN_CACHE_MAX:
        _COLUMN_CACHE.pop(next(iter(_COLUMN_CACHE)))  # FIFO evict
    _COLUMN_CACHE[key] = (m, labels)
    return m, list(labels)


def _harmonic_columns_uncached(
    r: np.ndarray, z: np.ndarray, cfg: HarmonicFitConfig
) -> tuple[np.ndarray, list[str]]:
    r_arr = np.asarray(r, dtype=np.float64)
    cosh_eta, _, _, theta, cmc = toroidal_coords(r, z, cfg.pole_r, cfg.pole_z)
    # Total-flux convention: the order-1 toroidal harmonic is A_phi; the measured
    # flux is Phi = 2 pi R A_phi, so the flux column carries an explicit R factor
    # (the 2 pi is absorbed into the fit coefficient).  This makes each column a
    # homogeneous-GS solution Delta* Phi_k = 0 -- WITHOUT the R factor the columns
    # solve the A_phi equation, not the flux equation, and cannot represent a
    # flux-loop signature (confirmed: the R-less fit stalls ~12%, with-R ~1e-8).
    pref = r_arr * np.sqrt(np.maximum(cmc, 0.0))
    # Fast vectorised elliptic-integral path for the physical P set (all orders
    # in one call); mpmath only for the Q ablation set.  The fast path is what
    # makes the per-slice pole affordable.
    if cfg.kind == "P":
        radials = ring_P1(cfg.order, cosh_eta)  # (order+1, N)
    else:
        radials = np.stack(
            [_ring_fn(n, cfg.kind)(cosh_eta) for n in range(cfg.order + 1)], axis=0
        )
    cols: list[np.ndarray] = []
    labels: list[str] = []
    for n in range(cfg.order + 1):
        base = pref * radials[n]
        cols.append(base * np.cos(n * theta))
        labels.append("h0" if n == 0 else f"h{n}c")
        if n >= 1:
            cols.append(base * np.sin(n * theta))
            labels.append(f"h{n}s")
    m = np.stack(cols, axis=1) if cols else np.zeros((np.asarray(r).size, 0))
    return m, labels


def harmonic_field_columns(
    r: np.ndarray, z: np.ndarray, cfg: HarmonicFitConfig, *, h: float = 1.0e-4
) -> tuple[np.ndarray, np.ndarray]:
    """``(B_R, B_Z)`` columns ``(n_pts, K)`` of each harmonic flux column.

    Central finite differences of the analytic (smooth, source-free) flux
    columns give the field with the total-flux convention
    ``B_R = -(1/(2 pi R)) dPhi/dZ``, ``B_Z = +(1/(2 pi R)) dPhi/dR``.
    """
    r = np.asarray(r, dtype=np.float64)
    psi_rp, _ = harmonic_columns(r + h, z, cfg)
    psi_rm, _ = harmonic_columns(r - h, z, cfg)
    psi_zp, _ = harmonic_columns(r, np.asarray(z, dtype=np.float64) + h, cfg)
    psi_zm, _ = harmonic_columns(r, np.asarray(z, dtype=np.float64) - h, cfg)
    dpsi_dr = (psi_rp - psi_rm) / (2.0 * h)
    dpsi_dz = (psi_zp - psi_zm) / (2.0 * h)
    r_col = np.maximum(r[:, None], _D_FLOOR)
    b_r = -dpsi_dz / (2.0 * np.pi * r_col)
    b_z = dpsi_dr / (2.0 * np.pi * r_col)
    return b_r, b_z


def harmonic_sensor_matrix(
    sensor_r: np.ndarray,
    sensor_z: np.ndarray,
    sensor_angle_deg: np.ndarray,
    is_flux: np.ndarray,
    cfg: HarmonicFitConfig,
) -> np.ndarray:
    """Design matrix ``A`` ``(S, K)``: each harmonic's signature per sensor row.

    Flux-loop rows get ``psi_k``; B-probe rows get the orientation-projected
    field ``B_R,k cos(theta) + B_Z,k sin(theta)`` (theta = ``angle_deg``).  Row
    order must match the payload's ``measured`` / ``vacuum`` vectors (i.e.
    ``table.sensor_map`` order).
    """
    psi, _ = harmonic_columns(sensor_r, sensor_z, cfg)
    b_r, b_z = harmonic_field_columns(sensor_r, sensor_z, cfg)
    th = np.deg2rad(np.asarray(sensor_angle_deg, dtype=np.float64))[:, None]
    bproj = b_r * np.cos(th) + b_z * np.sin(th)
    flux = np.asarray(is_flux, dtype=bool)[:, None]
    return np.where(flux, psi, bproj)


# --- the fit ----------------------------------------------------------------


@dataclass
class HarmonicInversion:
    """One slice's toroidal-harmonic annulus fit."""

    coeffs: np.ndarray  # (K,) harmonic amplitudes
    labels: list[str]
    misfit: float  # whitened mean-square sensor residual (trusted rows)
    cfg: HarmonicFitConfig
    shot: int = 0
    t_index: int = 0
    time_s: float = float("nan")
    coeff_cov: np.ndarray | None = field(default=None, repr=False)

    def psi_on_grid(self, grid_r: np.ndarray, grid_z: np.ndarray) -> np.ndarray:
        """Plasma harmonic flux ``(nz, nr)`` [Wb] on the grid (NO coil term)."""
        return harmonic_psi_on_grid(self.cfg, self.coeffs, grid_r, grid_z)


def _fit_one(
    a_sens: np.ndarray,  # (S, K) harmonic sensor design matrix
    measured: np.ndarray,
    vacuum: np.ndarray,
    mask: np.ndarray,
    scale: np.ndarray,
    ridge: float,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Whitened, column-normalised least-squares fit of the K harmonic amplitudes.

    Fits the plasma sensor signature ``measured - vacuum`` on the trusted rows;
    absent channels (NaN in ``measured``) are zeroed before whitening so
    ``NaN * 0`` cannot poison the normal equations (as the moment/free inverses
    do).  Returns ``(coeffs, misfit, cov)``.
    """
    keep = np.asarray(mask, dtype=bool)
    meas = np.nan_to_num(np.asarray(measured, dtype=np.float64))
    vac = np.nan_to_num(np.asarray(vacuum, dtype=np.float64))
    sc = np.asarray(scale, dtype=np.float64)
    w = np.zeros_like(meas)
    w[keep] = 1.0 / np.maximum(sc[keep], 1e-12)

    b = meas - vac  # plasma sensor signature
    aw = a_sens * w[:, None]
    bw = b * w
    col_norm = np.linalg.norm(aw, axis=0)
    col_norm = np.where(col_norm > 0, col_norm, 1.0)
    a_n = aw / col_norm[None, :]
    n_k = a_n.shape[1]
    gram = a_n.T @ a_n + ridge * np.eye(n_k)
    rhs = a_n.T @ bw
    c = np.linalg.solve(gram, rhs) / col_norm if n_k else np.zeros(0)

    resid = (vac + a_sens @ c - meas) * w
    n_keep = int(keep.sum())
    misfit = float((resid[keep] ** 2).sum() / max(n_keep, 1))
    try:
        cov_n = np.linalg.pinv(gram)
        cov = cov_n / np.outer(col_norm, col_norm)
    except np.linalg.LinAlgError:  # pragma: no cover
        cov = np.full((n_k, n_k), np.nan)
    return c, misfit, cov


def fit_harmonic(
    sensors: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    payload,
    cfg: HarmonicFitConfig | None = None,
) -> HarmonicInversion:
    """Fit the toroidal-harmonic amplitudes for one slice payload.

    ``sensors`` is ``(sensor_r, sensor_z, sensor_angle_deg, is_flux)`` in the
    payload's sensor-row order.  ``payload`` is a
    :class:`imas_ambix.latent.patch_inverse.SlicePayload`.  Assemble the grid
    flux with :meth:`HarmonicInversion.psi_on_grid` (plasma) plus the harness's
    thick-cylinder coil term.
    """
    cfg = cfg or HarmonicFitConfig()
    sensor_r, sensor_z, sensor_ang, is_flux = sensors
    a_sens = harmonic_sensor_matrix(sensor_r, sensor_z, sensor_ang, is_flux, cfg)
    c, misfit, cov = _fit_one(
        a_sens,
        payload.measured,
        payload.vacuum,
        payload.mask,
        payload.scale,
        cfg.ridge,
    )
    return HarmonicInversion(
        coeffs=c,
        labels=harmonic_labels(cfg.order),
        misfit=misfit,
        cfg=cfg,
        shot=getattr(payload, "shot", 0),
        t_index=getattr(payload, "t_index", 0),
        time_s=getattr(payload, "time_s", float("nan")),
        coeff_cov=cov,
    )


def harmonic_psi_on_grid(
    cfg: HarmonicFitConfig, coeffs: np.ndarray, grid_r: np.ndarray, grid_z: np.ndarray
) -> np.ndarray:
    """Plasma harmonic flux ``(nz, nr)`` [Wb] on the ``(grid_z, grid_r)`` raster.

    Returns the PLASMA contribution only (no coil term).  Row index = Z, column
    index = R, matching :meth:`PatchBasis.psi_grid_2d_np`.
    """
    gr = np.asarray(grid_r, dtype=np.float64)
    gz = np.asarray(grid_z, dtype=np.float64)
    rr, zz = np.meshgrid(gr, gz)  # (nz, nr)
    cols, _ = harmonic_columns(rr.ravel(), zz.ravel(), cfg)
    psi = cols @ np.asarray(coeffs, dtype=np.float64)
    return psi.reshape(zz.shape)


def mask_invalid_interior(
    psi: np.ndarray,
    grid_r: np.ndarray,
    grid_z: np.ndarray,
    pole_r: float,
    pole_z: float,
    radius: float,
    *,
    axis_rz: tuple[float, float] | None = None,
    k: float = 20.0,
) -> np.ndarray:
    """Fill the near-pole disk (the INVALID interior) with the confined-side extreme.

    The toroidal harmonics are physical ONLY in the source-free annulus; toward
    the focal ring (the pole) the P ring functions DIVERGE, so the reconstructed
    flux blows up inside the plasma where the expansion does not hold (the plan's
    "validity domain is the annulus only").  Reading the boundary directly off
    that field puts spurious saddles and early ray-cast crossings in the invalid
    interior.  This replaces the disk of radius ``radius`` about the pole with a
    single value ``k`` standard deviations past the annulus median on the
    CONFINED side (toward ``axis_rz`` if given, else toward the near-pole
    extreme), so the interior reads as "deeper than any boundary" -- the annulus
    boundary read (X-point set, bounding flux, LCFS ray-cast from the carrier
    axis) then sees a clean confined plateau inside and the physical field
    outside.  The interior itself is owned by the carrier, not this field.

    Returns a copy; the annulus (outside ``radius``) is untouched.
    """
    psi = np.asarray(psi, dtype=np.float64)
    gr = np.asarray(grid_r, dtype=np.float64)
    gz = np.asarray(grid_z, dtype=np.float64)
    rr, zz = np.meshgrid(gr, gz)
    inside = np.hypot(rr - pole_r, zz - pole_z) < radius
    if not inside.any() or inside.all():
        return psi.copy()
    ann = psi[~inside]
    med = float(np.median(ann))
    spread = float(np.std(ann)) or 1.0
    if axis_rz is not None:
        ia = int(np.argmin(np.abs(gz - axis_rz[1])))
        ja = int(np.argmin(np.abs(gr - axis_rz[0])))
        sign = np.sign(psi[ia, ja] - med)
    else:
        near = psi[inside]
        sign = np.sign(near[int(np.argmax(np.abs(near - med)))] - med)
    if sign == 0:
        sign = 1.0
    out = psi.copy()
    out[inside] = med + sign * k * spread
    return out


def gs_operator(psi: np.ndarray, r_1d: np.ndarray, z_1d: np.ndarray) -> np.ndarray:
    """Numerical Delta* psi on a ``(nz, nr)`` field (the correctness oracle).

    ``Delta* psi = psi_RR - (1/R) psi_R + psi_ZZ`` via second-order central
    differences on the (assumed uniform) raster.  Interior points only (edges
    returned as NaN).  Used by the TDD test to verify each harmonic column is a
    homogeneous-GS solution independent of the literature formula's provenance.
    """
    psi = np.asarray(psi, dtype=np.float64)
    r = np.asarray(r_1d, dtype=np.float64)
    z = np.asarray(z_1d, dtype=np.float64)
    dr = float(r[1] - r[0])
    dz = float(z[1] - z[0])
    out = np.full_like(psi, np.nan)
    psi_rr = (psi[:, 2:] - 2.0 * psi[:, 1:-1] + psi[:, :-2]) / dr**2
    psi_r = (psi[:, 2:] - psi[:, :-2]) / (2.0 * dr)
    psi_zz = (psi[2:, :] - 2.0 * psi[1:-1, :] + psi[:-2, :]) / dz**2
    r_int = r[1:-1][None, :]  # (1, nr-2) interior R
    out[1:-1, 1:-1] = psi_rr[1:-1, :] - psi_r[1:-1, :] / r_int + psi_zz[:, 1:-1]
    return out


__all__ = [
    "MU0",
    "HarmonicFitConfig",
    "HarmonicInversion",
    "fit_harmonic",
    "gs_operator",
    "harmonic_columns",
    "harmonic_field_columns",
    "mask_invalid_interior",
    "ring_P1",
    "harmonic_labels",
    "harmonic_psi_on_grid",
    "harmonic_sensor_matrix",
    "toroidal_coords",
]
