"""Covariance-weighted soft priors on global equilibrium moments.

The soft-prior interior solve fits a low-DOF current profile to the external
magnetics.  The magnetics alone are information-poor: they pin the total
current (Ip, Rogowski) and a single field-shape combination (βp + li/2, the
Shafranov diamagnetic parameter) — but NOT the split between the pressure
gradient (p′) and the poloidal-current (FF′) drives, which produce nearly
collinear external signatures at finite aspect ratio.  This module assembles
extra weighted rows on the per-sweep LSQ variable vector

    x = [coeffs (k_dof = n_p + n_f), a_pass (kp)]

(the same vector the Ip hard KKT and the smoothness Gram act on in
:func:`imas_ambix.latent.gs_solve.solve_equilibrium_lsq`).  Every prior here is
a *soft* covariance-weighted row: residual r = (row·x − rhs) enters the LSQ
whitened so that ‖r‖ ~ N(0, 1) when the modelled quantity matches its target
to within the stated uncertainty.

Three families, in increasing data dependence:

1. **Ip soft prior** — the whitened soft form of the hard Rogowski KKT.  The
   default solve keeps the HARD anchor (Rogowski Ip is clean); this is the
   opt-in soft form for when the anchor should trade against the data.
2. **βp + li/2 consistency** — the diamagnetic field-shape combination is
   FIREWALL-SAFE: it is fixed by the external poloidal field alone (no
   pressure, no EFIT).  :func:`beta_p_li_over_2` reads it off the equilibrium
   iterate's boundary-field asymmetry; :func:`moment_consistency_rows` pins a
   linear-in-coeffs moment to a target with covariance.
3. **Pressure-gradient prior** — the lever that BREAKS the p′/FF′ degeneracy.
   Separating p′ from FF′ needs an independent pressure.  The machinery is
   built and proven on synthetic degenerate systems; two target sources are
   provided — a WEAK density-shape proxy (firewall-safe, low confidence) and a
   kinetic Te·ne target that is DATA-GATED (Thomson Te is absent from the
   corpus — see the data-availability audit below).

Data-availability audit (corpus + :mod:`imas_ambix.latent.data`)
----------------------------------------------------------------
The per-slice payload and the anchored raw scalars were audited directly.
Loadable, firewall-safe per slice:

* **Ip** — Rogowski plasma current [A] (``SlicePayload.ip_amperes``,
  ``data.ANCHORED_NAMES[0]``).
* **n_e** — line-averaged electron density [m⁻²] (``data.ANCHORED_NAMES[1]``).

NOT present in the corpus (confirmed: no such channel in ``data.py`` /
``SlicePayload``):

* diamagnetic-loop flux → would give βp directly; ABSENT.
* Thomson-scattering Te(ψ_N) → the temperature that separates p′ from FF′;
  ABSENT.  This is the data gate recorded by :func:`kinetic_pressure_target`.
* CXRS Ti/rotation → ABSENT (a separate rotation study tracks this).

Consequence: βp + li/2 is recoverable (external magnetics, firewall-safe);
full p′/FF′ separation from an independent temperature is DATA-GATED — the
density proxy is the only firewall-safe pressure lever and must carry a large
covariance.

All functions are pure numpy.  Physics conventions follow the shared
soft-prior contracts: TOTAL flux Φ = 2π R A_φ [Wb], MAST sign (positive Ip ⇒
ψ_axis > ψ_boundary), profile basis jφ = (R/R0)·Σ cᵖ φ_k + (R0/R)·Σ cᶠ φ_k
with columns 0..n_p−1 the p′ (pressure-gradient) family.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MU0 = 4.0e-7 * np.pi

#: The exact data missing from the corpus that would let the kinetic pressure
#: target separate p′ from FF′ without an assumed shape.  Recorded in code so
#: the gate is greppable, not buried in prose.
TE_DATA_GATE = (
    "Thomson-scattering Te(psi_N) is absent from the MAST corpus assembled by "
    "imas_ambix.latent.data (anchored scalars are Ip + line-averaged n_e only). "
    "Without an independent temperature profile the kinetic pressure "
    "p(psi_N) = n_e(psi_N)*(Te+Ti) cannot be formed, so p'/ff' separation is "
    "limited to the weak density-shape proxy (density_pressure_proxy)."
)

__all__ = [
    "MU0",
    "TE_DATA_GATE",
    "MomentTarget",
    "ip_soft_prior_row",
    "moment_consistency_rows",
    "beta_p_li_over_2",
    "internal_inductance_li",
    "beta_poloidal",
    "pressure_gradient_prior_rows",
    "density_pressure_proxy",
    "kinetic_pressure_target",
]


# --------------------------------------------------------------------------- #
# 1. Ip soft prior (covariance form of the hard Rogowski KKT)
# --------------------------------------------------------------------------- #
def ip_soft_prior_row(
    a_anchor: np.ndarray,
    ip_amperes: float,
    *,
    sigma_rel: float,
    k_dof: int,
    kp: int,
) -> tuple[np.ndarray, float]:
    """One whitened soft row pinning net current to the Rogowski Ip.

    The hard anchor is ``a_anchor · coeffs = Ip`` (net current per unit
    coefficient; passive block excluded — a_anchor has length ``k_dof``).  The
    soft form weights the whitened residual by ``w = 1/(sigma_rel·|Ip|)`` so
    that a fractional Ip error of ``sigma_rel`` contributes unit cost::

        r = w · (a_anchor · coeffs − Ip),   w = 1 / (sigma_rel · |Ip|)

    Returns ``(row, rhs)`` on the full variable vector
    ``x = [coeffs(k_dof), a_pass(kp)]`` (passive columns zero), suitable for
    stacking onto the LSQ design as ``row · x = rhs``.

    The DEFAULT solve keeps the hard KKT anchor (clean Rogowski current); this
    is the opt-in soft form the orchestrator selects when the anchor should
    trade against the sensor residual.  As ``sigma_rel → 0`` the weight → ∞ and
    the soft solution approaches the hard-constrained one (tested).
    """
    a_anchor = np.asarray(a_anchor, dtype=np.float64).ravel()
    if a_anchor.size != k_dof:
        raise ValueError(f"a_anchor length {a_anchor.size} != k_dof {k_dof}")
    if sigma_rel <= 0.0:
        raise ValueError("sigma_rel must be > 0 (use the hard KKT for an exact anchor)")
    w = 1.0 / (sigma_rel * max(abs(float(ip_amperes)), 1e-30))
    row = np.zeros(k_dof + kp, dtype=np.float64)
    row[:k_dof] = w * a_anchor
    rhs = w * float(ip_amperes)
    return row, rhs


# --------------------------------------------------------------------------- #
# 2. βp + li/2 from the equilibrium + the generic moment-consistency row
# --------------------------------------------------------------------------- #
@dataclass
class MomentTarget:
    """A scalar moment measured off the equilibrium iterate, with uncertainty.

    ``value`` is the diagnostic read (e.g. βp + li/2); ``sigma`` its 1σ
    covariance in the same units; ``available`` is False for a data-gated
    quantity, in which case ``reason`` names the missing measurement.
    """

    value: float
    sigma: float
    available: bool = True
    reason: str = ""


def _grid_spacing(rg: np.ndarray, zg: np.ndarray) -> tuple[float, float]:
    rg = np.asarray(rg, dtype=np.float64)
    zg = np.asarray(zg, dtype=np.float64)
    dr = float(np.mean(np.diff(rg))) if rg.size > 1 else 1.0
    dz = float(np.mean(np.diff(zg))) if zg.size > 1 else 1.0
    return dr, dz


def _plasma_mask(psi2d: np.ndarray, axis_psi: float, boundary_psi: float) -> np.ndarray:
    """Boolean (nz, nr) mask of the confined region 0 ≤ ψ_N < 1.

    ψ_N = (ψ − ψ_axis)/(ψ_bnd − ψ_axis); MAST sign makes ψ_axis the extremum,
    so a cell is inside iff ψ lies strictly between boundary_psi and axis_psi.
    """
    lo, hi = sorted((float(axis_psi), float(boundary_psi)))
    return (psi2d > lo) & (psi2d < hi)


def _poloidal_field(psi2d: np.ndarray, rg: np.ndarray, zg: np.ndarray) -> np.ndarray:
    """|B_pol| on the grid from the TOTAL flux Φ = 2π R A_φ.

    Per-radian poloidal flux ψ_pr = Φ/(2π); B_pol = |∇ψ_pr|/R = |∇Φ|/(2π R).
    """
    psi2d = np.asarray(psi2d, dtype=np.float64)
    rg = np.asarray(rg, dtype=np.float64)
    zg = np.asarray(zg, dtype=np.float64)
    # np.gradient with explicit coordinates handles non-unit spacing; axis 0
    # is Z (rows), axis 1 is R (cols).
    dphi_dz, dphi_dr = np.gradient(psi2d, zg, rg)
    rr = np.maximum(rg[np.newaxis, :], 1e-6)
    return np.hypot(dphi_dr, dphi_dz) / (2.0 * np.pi * rr)


def internal_inductance_li(
    psi2d: np.ndarray,
    jphi2d: np.ndarray,
    rg: np.ndarray,
    zg: np.ndarray,
    *,
    axis_psi: float,
    boundary_psi: float,
    r0: float,
    mu0: float = MU0,
) -> float:
    """Normalised internal inductance li(3) of the equilibrium iterate.

    li(3) is the ITER/EFIT convention

        li(3) = 2 ∫_V B_pol² dV / (μ0² Ip² R0)

    with dV = 2π R dR dZ over the confined region and Ip the enclosed toroidal
    current.  It needs only the poloidal field and Ip — no pressure, no
    perimeter — so it is a pure field diagnostic (firewall-safe).  For a
    uniform-current large-aspect circular plasma li(3) = 0.5 (tested).
    """
    dr, dz = _grid_spacing(rg, zg)
    mask = _plasma_mask(psi2d, axis_psi, boundary_psi)
    if not mask.any():
        return float("nan")
    bp = _poloidal_field(psi2d, rg, zg)
    rr = np.asarray(rg, dtype=np.float64)[np.newaxis, :] * np.ones_like(psi2d)
    dvol = 2.0 * np.pi * rr * dr * dz
    ip = float(np.sum(np.asarray(jphi2d)[mask]) * dr * dz)
    bp2_dv = float(np.sum((bp[mask] ** 2) * dvol[mask]))
    denom = (mu0**2) * (ip**2) * float(r0)
    if abs(denom) < 1e-30:
        return float("nan")
    return 2.0 * bp2_dv / denom


def beta_poloidal(
    psi2d: np.ndarray,
    jphi2d: np.ndarray,
    pressure2d: np.ndarray,
    rg: np.ndarray,
    zg: np.ndarray,
    *,
    axis_psi: float,
    boundary_psi: float,
    r0: float,
    mu0: float = MU0,
) -> float:
    """Poloidal beta βp of the equilibrium iterate — REQUIRES pressure.

        βp = (4 / (μ0 R0 Ip²)) ∫ p dV,   dV = 2π R dR dZ

    (equivalent to 2μ0 ⟨p⟩_V / B_pa² with B_pa = μ0 Ip / (2π a) at large
    aspect).  ``pressure2d`` [Pa] is the kinetic pressure on the grid; it is
    NOT derivable from the firewall-safe corpus (no Thomson Te — see
    :data:`TE_DATA_GATE`), so βp is a DIAGNOSTIC available only when an
    independent pressure is supplied.  For a uniform-pressure large-aspect
    disk this reduces to 8π² a² p0 / (μ0 Ip²) (tested).
    """
    dr, dz = _grid_spacing(rg, zg)
    mask = _plasma_mask(psi2d, axis_psi, boundary_psi)
    if not mask.any():
        return float("nan")
    rr = np.asarray(rg, dtype=np.float64)[np.newaxis, :] * np.ones_like(psi2d)
    dvol = 2.0 * np.pi * rr * dr * dz
    ip = float(np.sum(np.asarray(jphi2d)[mask]) * dr * dz)
    p_dv = float(np.sum(np.asarray(pressure2d)[mask] * dvol[mask]))
    denom = mu0 * float(r0) * (ip**2)
    if abs(denom) < 1e-30:
        return float("nan")
    return 4.0 * p_dv / denom


def _boundary_points(
    psi2d: np.ndarray, rg: np.ndarray, zg: np.ndarray, boundary_psi: float
) -> tuple[np.ndarray, np.ndarray]:
    """Linear-interpolated (R, Z) crossings of ψ = boundary_psi on grid edges.

    Pure-numpy contour sampling: scan every horizontal and vertical grid edge
    for a sign change of (ψ − boundary_psi) and place the crossing by linear
    interpolation.  Returns concatenated (r, z) crossing coordinates.
    """
    rg = np.asarray(rg, dtype=np.float64)
    zg = np.asarray(zg, dtype=np.float64)
    f = np.asarray(psi2d, dtype=np.float64) - float(boundary_psi)
    r_pts: list[float] = []
    z_pts: list[float] = []
    # horizontal edges (vary R at fixed Z)
    left, right = f[:, :-1], f[:, 1:]
    cross = (left * right) < 0.0
    ii, jj = np.where(cross)
    for i, j in zip(ii, jj, strict=True):
        t = left[i, j] / (left[i, j] - right[i, j])
        r_pts.append(rg[j] + t * (rg[j + 1] - rg[j]))
        z_pts.append(zg[i])
    # vertical edges (vary Z at fixed R)
    down, up = f[:-1, :], f[1:, :]
    cross = (down * up) < 0.0
    ii, jj = np.where(cross)
    for i, j in zip(ii, jj, strict=True):
        t = down[i, j] / (down[i, j] - up[i, j])
        r_pts.append(rg[j])
        z_pts.append(zg[i] + t * (zg[i + 1] - zg[i]))
    return np.asarray(r_pts), np.asarray(z_pts)


def _bilinear(
    field: np.ndarray, rg: np.ndarray, zg: np.ndarray, r: np.ndarray, z: np.ndarray
) -> np.ndarray:
    """Bilinear sample of a (nz, nr) field at scattered (r, z)."""
    rg = np.asarray(rg, dtype=np.float64)
    zg = np.asarray(zg, dtype=np.float64)
    jr = np.clip(np.searchsorted(rg, r) - 1, 0, rg.size - 2)
    iz = np.clip(np.searchsorted(zg, z) - 1, 0, zg.size - 2)
    tr = (r - rg[jr]) / (rg[jr + 1] - rg[jr])
    tz = (z - zg[iz]) / (zg[iz + 1] - zg[iz])
    f00 = field[iz, jr]
    f01 = field[iz, jr + 1]
    f10 = field[iz + 1, jr]
    f11 = field[iz + 1, jr + 1]
    return (
        f00 * (1 - tr) * (1 - tz)
        + f01 * tr * (1 - tz)
        + f10 * (1 - tr) * tz
        + f11 * tr * tz
    )


def _boundary_field_asymmetry(
    theta: np.ndarray, bp: np.ndarray, r_minor: np.ndarray, r0: float
) -> float:
    """βp + li/2 from the m=1 (cosθ) asymmetry of the boundary poloidal field.

    The Shafranov result for a large-aspect circular boundary is

        B_pol(θ) = B_pa · [1 + (a/R0)(βp + li/2 − 1) cosθ]

    so with the fitted m=0 mean ``c0`` and m=1 cosine amplitude ``c1``,
    ε = c1/c0 = (a/R0)(βp + li/2 − 1) and βp + li/2 = 1 + ε·R0/⟨a⟩.
    """
    theta = np.asarray(theta, dtype=np.float64)
    bp = np.asarray(bp, dtype=np.float64)
    # least-squares fit Bp ≈ c0 + c1 cosθ + s1 sinθ
    basis = np.column_stack([np.ones_like(theta), np.cos(theta), np.sin(theta)])
    coef, *_ = np.linalg.lstsq(basis, bp, rcond=None)
    c0, c1 = float(coef[0]), float(coef[1])
    if abs(c0) < 1e-30:
        return float("nan")
    a_mean = float(np.mean(r_minor))
    eps = c1 / c0
    return 1.0 + eps * float(r0) / max(a_mean, 1e-9)


def beta_p_li_over_2(
    psi2d: np.ndarray,
    jphi2d: np.ndarray,
    rg: np.ndarray,
    zg: np.ndarray,
    *,
    axis: tuple[float, float],
    boundary_psi: float,
    r0: float,
    b_phi0: float,
    mu0: float = MU0,
) -> float:
    """βp + li/2 of the equilibrium iterate from the boundary-field asymmetry.

    This is the FIREWALL-SAFE diamagnetic combination: the m=1 asymmetry of the
    poloidal field on the plasma boundary is fixed by the external field alone
    (Shafranov), independent of the internal p′/FF′ split — which is exactly
    why the magnetics can constrain this combination but not the split.  The
    boundary contour ψ = ``boundary_psi`` is sampled (pure-numpy edge
    crossings), B_pol = |∇Φ|/(2π R) is read there, and the cosθ modulation
    about the magnetic ``axis`` gives βp + li/2 (see
    :func:`_boundary_field_asymmetry`).

    ``b_phi0`` (vacuum toroidal field at R0) and ``jphi2d`` are accepted for
    interface symmetry with the diamagnetic/paramagnetic diagnostics and future
    virial cross-checks; the boundary-asymmetry read uses only ψ and geometry.
    At the spherical-tokamak aspect ratio of MAST the large-aspect identity is
    approximate — the orchestrator validates the absolute level against the
    firewalled EFIT referee (scoring only, never an input).
    """
    r_pts, z_pts = _boundary_points(psi2d, rg, zg, boundary_psi)
    if r_pts.size < 6:
        return float("nan")
    bp = _poloidal_field(psi2d, rg, zg)
    bp_bd = _bilinear(bp, rg, zg, r_pts, z_pts)
    r_ax, z_ax = float(axis[0]), float(axis[1])
    theta = np.arctan2(z_pts - z_ax, r_pts - r_ax)
    r_minor = np.hypot(r_pts - r_ax, z_pts - z_ax)
    return _boundary_field_asymmetry(theta, bp_bd, r_minor, r0)


def moment_consistency_rows(
    *,
    computed_moment_unit_sensitivity: np.ndarray,
    target_moment: float,
    sigma: float,
    k_dof: int,
    kp: int,
) -> tuple[np.ndarray, float]:
    """One whitened soft row pinning a linear-in-coeffs moment to a target.

    ``computed_moment_unit_sensitivity`` s is ∂(moment)/∂(coeffs), length
    ``k_dof`` — the moment linearised about the current iterate so that
    ``moment(x) ≈ s · coeffs``.  The soft row whitens the residual by 1/σ::

        r = (1/σ) · (s · coeffs − target_moment)

    For a genuinely linear-homogeneous moment (e.g. Ip = a_anchor·coeffs) pass
    s and the raw target directly.  For a NONLINEAR moment (βp + li/2) the
    orchestrator linearises about the iterate: s = ∂m/∂coeffs there and
    ``target_moment`` = m_target − (m(iterate) − s·coeffs_iterate), i.e. the
    target shifted into the linearised frame.  Returns ``(row, rhs)`` on
    ``x = [coeffs(k_dof), a_pass(kp)]`` (passive columns zero).
    """
    s = np.asarray(computed_moment_unit_sensitivity, dtype=np.float64).ravel()
    if s.size != k_dof:
        raise ValueError(f"sensitivity length {s.size} != k_dof {k_dof}")
    if sigma <= 0.0:
        raise ValueError("sigma must be > 0")
    row = np.zeros(k_dof + kp, dtype=np.float64)
    row[:k_dof] = s / sigma
    rhs = float(target_moment) / sigma
    return row, rhs


# --------------------------------------------------------------------------- #
# 3. Pressure-gradient prior — the p′/FF′ separation lever
# --------------------------------------------------------------------------- #
def pressure_gradient_prior_rows(
    *,
    p_basis_slice: np.ndarray,
    pprime_target: np.ndarray,
    sigma,
    k_dof: int,
    kp: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Soft rows pulling the p′-family coefficients toward a target p′(ψ_N).

    The p′ (pressure-gradient) drive occupies coefficient columns 0..n_p−1 of
    ``x = [coeffs(k_dof), a_pass(kp)]`` (contracts).  ``p_basis_slice`` P is
    (n_samples, n_p): the p′-family basis functions evaluated at the ψ_N
    sample points, so the modelled p′ at those points is ``P · coeffs[:n_p]``.
    Given a target ``pprime_target`` (n_samples,) with covariance ``sigma``
    (scalar or per-sample), the whitened rows are::

        r_i = (1/σ_i) · (Σ_k P_ik coeffs_k − pprime_target_i)

    Pinning the p′ family directly is the lever that BREAKS the p′/FF′
    degeneracy: the external magnetics fix only the p′+FF′ combination, leaving
    the split under-determined; a covariance-weighted p′ target resolves it
    (proven on a synthetic degenerate system in the tests).  Returns
    ``(rows, rhs)`` with ``rows`` shape (n_samples, k_dof + kp) — the FF′ and
    passive columns are zero, so this constrains ONLY the p′ family.
    """
    p = np.asarray(p_basis_slice, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError("p_basis_slice must be (n_samples, n_p)")
    n_samples, n_p = p.shape
    if n_p > k_dof:
        raise ValueError(f"p'-family width {n_p} exceeds k_dof {k_dof}")
    target = np.asarray(pprime_target, dtype=np.float64).ravel()
    if target.size != n_samples:
        raise ValueError("pprime_target length must match p_basis_slice rows")
    sig = np.asarray(sigma, dtype=np.float64)
    if sig.ndim == 0:
        sig = np.full(n_samples, float(sig))
    if np.any(sig <= 0.0):
        raise ValueError("sigma must be > 0")
    rows = np.zeros((n_samples, k_dof + kp), dtype=np.float64)
    rows[:, :n_p] = p / sig[:, np.newaxis]
    rhs = target / sig
    return rows, rhs


def density_pressure_proxy(
    n_e_line: float,
    psi_n_grid: np.ndarray,
    *,
    shape: str = "parabolic",
    gamma: float = 1.5,
    t_e_ref_ev: float = 100.0,
) -> np.ndarray:
    """WEAK, firewall-safe p′(ψ_N) proxy from line-averaged density.

    The corpus carries only line-averaged n_e — no temperature — so this proxy
    ASSUMES a pressure shape and a nominal temperature to give a low-confidence
    p′(ψ_N).  Kinetic pressure is modelled as

        p(ψ_N) = n_e(ψ_N) · k_B (Te + Ti),   n_e(ψ_N) = n̄_e · φ(ψ_N)

    with an assumed profile φ(ψ_N) = (1 − ψ_N)^γ ("parabolic": γ≈1) and a
    nominal ``t_e_ref_ev`` standing in for (Te + Ti).  Then

        p′(ψ_N) = dp/dψ_N = −γ · n̄_e · k_B · t_e_ref · (1 − ψ_N)^(γ−1)

    (negative: pressure falls from axis to edge).  This is DELIBERATELY weak —
    the magnitude rests on an assumed temperature and shape — so it MUST carry
    a large covariance when fed to :func:`pressure_gradient_prior_rows`.  It is
    a shape prior, not a measurement.  Returns p′(ψ_N) [Pa] on ``psi_n_grid``.
    """
    psi_n = np.clip(np.asarray(psi_n_grid, dtype=np.float64), 0.0, 1.0)
    if shape != "parabolic":
        raise ValueError(f"unknown proxy shape {shape!r} (only 'parabolic')")
    k_b = 1.602176634e-19  # J/eV (temperature given in eV → energy)
    p0 = float(n_e_line) * k_b * float(t_e_ref_ev)  # nominal on-axis pressure [Pa]
    edge = np.power(np.maximum(1.0 - psi_n, 0.0), max(gamma - 1.0, 0.0))
    return -gamma * p0 * edge


def kinetic_pressure_target(
    te_profile: np.ndarray,
    ne_profile: np.ndarray,
    psi_n: np.ndarray,
) -> np.ndarray:
    """DATA-GATED kinetic pressure target p(ψ_N) = n_e · k_B (Te + Ti).

    This is the REAL pressure that would separate p′ from FF′ without an
    assumed shape.  It is NOT implementable against the current corpus: Thomson
    Te(ψ_N) is absent (see :data:`TE_DATA_GATE` and the module data-audit).
    Raising here — rather than fabricating from an assumed temperature — keeps
    the data gate honest and greppable; use :func:`density_pressure_proxy` for
    the firewall-safe weak lever until Thomson Te is added to the corpus.
    """
    raise NotImplementedError(TE_DATA_GATE)
