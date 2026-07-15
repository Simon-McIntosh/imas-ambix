"""Profile-DOF regularisation for the soft-prior interior force-balance solve.

The interior spine (:mod:`imas_ambix.latent.gs_solve`) discovers the flux
functions p′(ψ_N) and FF′(ψ_N) by solving their basis coefficients against the
raw magnetics EVERY Picard sweep — force balance, not a fixed analytic form.
This module supplies the pieces that let that discovery run at HIGHER profile
DOF without over-fitting, and that let current live in the public scrape-off
layer (SOL) instead of being pinned to zero at the separatrix:

A. **Soft SOL edge + decay foot.**  The ``(1−ψ_N)^e`` boundary factor used by
   :func:`gs_solve.profile_basis` is exactly zero at ψ_N = 1 and goes negative /
   complex past it.  :func:`edge_factor_with_foot` continues the profile past a
   knot with a C¹ (value + slope matched) exponential foot that stays ≥ 0 out to
   a dimensionless cap (≈ 1.1), so SOL current is representable and the profile
   never dips negative.  :func:`profile_basis_foot` mirrors
   :func:`gs_solve.profile_basis` using that footed factor.

B. **Safety-factor q(ψ) diagnostic and a q ≥ 1 (sawtooth) prior.**
   :func:`q_profile` computes the flux-surface-averaged q on a handful of
   surfaces by contour integration; :func:`q_axis_linear_bound` gives the
   large-aspect circular on-axis bound j_φ,axis ≤ 2 B_φ0/(μ0 R_0) that is
   equivalent to q_0 ≥ 1; :func:`q_axis_penalty_row` turns that bound into a
   soft one-sided penalty row on the profile coefficients.

C. **Higher-order profile regularisation.**  :func:`curvature_gram` rebuilds
   the second-difference smoothness Gram (matching
   :func:`gs_solve._second_difference_gram`) and
   :func:`monotonicity_penalty_rows` softly discourages a non-monotone drive
   where one is physically expected.

All knobs are dimensionless or geometry-scaled (never fixed metres), and no
EFIT output enters any path — the module is machine-agnostic and firewall-safe.

Conventions (``docs/notes/soft-prior-contracts.md``): TOTAL flux Φ = 2π R A_φ,
Δ*Φ = −2π μ0 R jφ, MAST sign ψ_axis > ψ_bnd, ψ_N = (ψ−ψ_axis)/(ψ_bnd−ψ_axis),
and the two-family current basis jφ = (R/R0)·Σ cᵖ φ_k + (R0/R)·Σ cᶠ φ_k.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MU0 = 4.0e-7 * np.pi

# MAST nominal vacuum toroidal field at R0 [T].  The vacuum-field magnitude
# B_φ0 (equivalently F_vac = R0·B_φ0) is NOT stored in the campaign geometry
# table (imas_ambix.gs.geometry) nor in the firewall-safe anchored scalars
# (only Ip and n_e are anchored — data.py:ANCHORED_NAMES).  The codebase uses
# the constant F_VAC = R0·B0 = 0.85 m · 0.55 T (scripts/*_gate_eval.py), i.e.
# B_φ0 ≈ 0.55 T at R0 = 0.85 m.  A per-shot value is available from the raw
# `tf_current` amc channel (a machine measurement, not EFIT — firewall-safe)
# if the orchestrator chooses to expose it; callers pass ``b_phi0`` explicitly.
MAST_NOMINAL_B_PHI0 = 0.55


def mast_nominal_b_phi0() -> float:
    """Nominal MAST vacuum toroidal field at R0 [T] (≈ 0.55 T).

    Fallback when a per-shot ``tf_current``-derived value is not supplied.
    """
    return MAST_NOMINAL_B_PHI0


# ---------------------------------------------------------------------------
# A. Soft SOL edge + C¹ decay-foot closure
# ---------------------------------------------------------------------------


def edge_factor_with_foot(
    psi_n: np.ndarray, *, w: float, cap: float = 1.1, exponent: float = 1.0
) -> np.ndarray:
    """The ``(1−ψ_N)^exponent`` boundary factor continued past the separatrix.

    For ψ_N ≤ knot the factor is exactly ``(1−ψ_N)^exponent`` (the
    :func:`gs_solve.profile_basis` edge factor).  Past the knot it is a C¹
    exponential decay foot — matched in VALUE and SLOPE — that stays strictly
    positive out to ``cap`` and is zero beyond it.  This lets current decay
    smoothly into the public SOL rather than being pinned to zero at ψ_N = 1
    (where the bare factor vanishes) or going negative past it.

    Why the knot sits INSIDE ψ_N = 1 (knot = 1 − ``w``): the bare factor
    ``(1−ψ_N)^exponent`` is exactly ZERO at ψ_N = 1, so no positive,
    value-and-slope-matched continuation exists there — matching value 0 with a
    finite negative slope forces the foot negative.  A positive SOL foot
    therefore requires a pedestal: the C¹ blend knot is placed a dimensionless
    ``w`` inside the separatrix, where the factor value w^exponent > 0 and its
    slope are both finite, and the foot continues from there.  Only the outer
    ``w`` fraction of the core is modified.

    Parameters
    ----------
    psi_n : normalised flux (0 at axis, 1 at separatrix).
    w : dimensionless pedestal/foot width (start 0.03–0.05).  Also the knot
        inset (knot = 1 − w).  The decay length of the foot is w/exponent.
    cap : dimensionless SOL support limit (≈ 1.1).  ``cap ≤ 1`` disables the
        foot and reproduces :func:`gs_solve.profile_basis` exactly.
    exponent : the monomial exponent e (1.0 for the linear / Legendre boundary
        factor; the ``monomial-nonneg`` ladder passes 0.5, 1, 1.5, 2, 3).

    Returns
    -------
    array of the boundary factor, same shape as ``psi_n``, ≥ 0 everywhere for
    ``exponent`` in the physical (positive) range.
    """
    psi_n = np.asarray(psi_n, dtype=np.float64)
    exponent = float(exponent)
    cap = float(cap)

    # foot inactive: identical to the bare gs_solve.profile_basis edge factor
    if cap <= 1.0:
        base = np.power(1.0 - np.clip(psi_n, 0.0, 1.0), exponent)
        return np.where(psi_n < 1.0, base, 0.0)

    w = float(w)
    if not (0.0 < w < 1.0):
        raise ValueError(f"foot width w must be in (0, 1), got {w!r}")
    knot = 1.0 - w
    v0 = w**exponent  # factor value at the knot
    # d/dψ_N (1−ψ_N)^exponent = −exponent·(1−ψ_N)^(exponent−1); at the knot:
    slope_over_value = -exponent / w  # = s0 / v0, independent of exponent

    out = np.zeros_like(psi_n)
    core = psi_n <= knot
    foot = (psi_n > knot) & (psi_n <= cap)
    out[core] = np.power(1.0 - np.clip(psi_n[core], 0.0, 1.0), exponent)
    out[foot] = v0 * np.exp(slope_over_value * (psi_n[foot] - knot))
    return out


def profile_basis_foot(
    psi_n: np.ndarray,
    r: np.ndarray,
    *,
    r0: float,
    n_p: int,
    n_f: int,
    kind: str = "legendre",
    w: float = 0.05,
    cap: float = 1.1,
    centrifugal_gamma=None,
) -> np.ndarray:
    """(n_points, n_p + n_f) jφ basis images with the footed SOL edge.

    Mirrors :func:`gs_solve.profile_basis` — same two drive families (R/R0
    pressure-gradient, R0/R FF′), same ``kind`` semantics — but uses
    :func:`edge_factor_with_foot` so columns are admitted in the SOL band
    ψ_N ∈ (1, ``cap``] and decay smoothly instead of vanishing / going
    negative at the separatrix.  With ``cap ≤ 1`` it reproduces
    :func:`gs_solve.profile_basis` for ψ_N < 1 (the foot is inactive).

    ``kind="legendre"``: φ_k = P_{k−1}(2·clip(ψ_N,0,1)−1)·edge_foot(exponent=1)
    (in the SOL the Legendre argument saturates at 1, so all columns share the
    single decaying foot shape per drive — the SOL current has one shape DOF).
    ``kind="monomial-nonneg"``: φ_k = edge_foot(exponent=e_k) with the ladder
    e = (0.5, 1, 1.5, 2, 3) — every column ≥ 0, so non-negative coefficients
    give jφ·sign ≥ 0 pointwise through the SOL band.
    """
    from numpy.polynomial import legendre  # noqa: PLC0415

    psi_n = np.asarray(psi_n, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    in_band = psi_n < cap
    rr = np.maximum(r, 1e-3)
    x = 2.0 * np.clip(psi_n, 0.0, 1.0) - 1.0
    nonneg_exponents = (0.5, 1.0, 1.5, 2.0, 3.0)
    if centrifugal_gamma is not None:
        gam = np.asarray(centrifugal_gamma(np.clip(psi_n, 0.0, 1.0)))
        cent = np.exp(np.clip(gam * (rr**2 - r0**2), -3.0, 3.0))
    else:
        cent = None
    cols = []
    for family, (drive, n_k) in enumerate(((rr / r0, n_p), (r0 / rr, n_f))):
        for k in range(n_k):
            if kind == "monomial-nonneg":
                phi = edge_factor_with_foot(
                    psi_n, w=w, cap=cap, exponent=nonneg_exponents[k]
                )
            elif kind == "legendre":
                edge = edge_factor_with_foot(psi_n, w=w, cap=cap, exponent=1.0)
                phi = legendre.legval(x, [0.0] * k + [1.0]) * edge
            else:  # pragma: no cover — callers pass validated kinds
                raise ValueError(f"unknown basis kind {kind!r}")
            col = drive * phi
            if cent is not None and family == 0:  # pressure drive only
                col = col * cent
            cols.append(np.where(in_band, col, 0.0))
    return (
        np.column_stack(cols) if cols else np.zeros((psi_n.size, 0), dtype=np.float64)
    )


# ---------------------------------------------------------------------------
# B. Safety-factor q(ψ) diagnostic and q ≥ 1 (sawtooth) prior
# ---------------------------------------------------------------------------


def f_from_ffprime(
    psi_n_grid: np.ndarray,
    ffprime_of_psin: np.ndarray,
    *,
    f_boundary: float,
    dpsi_dpsin: float = 1.0,
) -> np.ndarray:
    """Integrate FF′ inward from the boundary to F(ψ_N).

    ``FF′ = ½ d(F²)/dψ`` ⇒ ``F²(ψ) = F²_bnd + 2 ∫_bnd^ψ FF′ dψ`` (same anchor
    convention as :func:`structure_residual.integrate_closures`).  ``F`` is
    single-signed (F = R·B_φ), so its sign follows ``f_boundary`` = R0·B_φ0.

    ``ffprime_of_psin`` is FF′ sampled on ``psi_n_grid``; ``dpsi_dpsin`` =
    (ψ_bnd − ψ_axis) converts the ψ_N integration variable to real ψ.  For a
    CONSTANT FF′ this returns F² linear in ψ_N regardless of ``dpsi_dpsin``.

    Returns F on ``psi_n_grid`` (ordered as given).
    """
    psi_n = np.asarray(psi_n_grid, dtype=np.float64)
    ffp = np.asarray(ffprime_of_psin, dtype=np.float64)
    if psi_n.shape != ffp.shape:
        raise ValueError("psi_n_grid and ffprime_of_psin must share a shape")
    order = np.argsort(psi_n)
    xs = psi_n[order]
    ys = ffp[order]
    # cumulative ∫ from the boundary (ψ_N = 1) inward, in real ψ
    i_bnd = int(np.argmin(np.abs(xs - 1.0)))
    integral = np.zeros_like(xs)
    # integrate outward from i_bnd in both directions along the sorted grid
    for i in range(i_bnd - 1, -1, -1):
        integral[i] = integral[i + 1] + 0.5 * (ys[i] + ys[i + 1]) * (xs[i] - xs[i + 1])
    for i in range(i_bnd + 1, len(xs)):
        integral[i] = integral[i - 1] + 0.5 * (ys[i] + ys[i - 1]) * (xs[i] - xs[i - 1])
    f_squared = f_boundary * f_boundary + 2.0 * dpsi_dpsin * integral
    f_sorted = np.sign(f_boundary) * np.sqrt(np.clip(f_squared, 0.0, None))
    out = np.empty_like(f_sorted)
    out[order] = f_sorted
    return out


def q_axis_linear_bound(*, b_phi0: float, r0: float) -> float:
    """On-axis current-density bound j_φ,axis ≤ 2 B_φ0/(μ0 R_0) ⟺ q_0 ≥ 1.

    Large-aspect circular limit: q_0 = 2 B_φ0/(μ0 R_0 j_φ,axis), so q_0 ≥ 1 is
    the LINEAR constraint j_φ,axis ≤ this returned maximum.  This is the
    per-sweep-usable form of the sawtooth (q ≥ 1) prior.  For MAST-scale
    B_φ0 ≈ 0.55 T, R_0 ≈ 0.85 m the bound is ≈ 1.0e6 A/m².
    """
    return 2.0 * float(b_phi0) / (MU0 * float(r0))


@dataclass
class SoftUpperBoundRow:
    """A soft one-sided (upper) bound ``images_axis_unit·coeffs ≤ j_axis_max``.

    ``row``/``rhs`` are the weighted linear row for the orchestrator's LSQ
    assembly (append ``row`` to the design matrix and ``rhs`` to the target);
    used one-sided, the row penalises only the EXCESS of j_φ,axis over the
    bound.  :meth:`hinge` gives that one-sided penalty for a coefficient vector
    (zero below the bound, positive above) — the diagnostic form.
    """

    row: np.ndarray  # (k_dof,) = weight · images_axis_unit
    rhs: float  # weight · j_axis_max
    j_axis_max: float
    weight: float
    images_axis_unit: np.ndarray

    def j_axis(self, coeffs: np.ndarray) -> float:
        """On-axis current density j_φ,axis for a coefficient vector [A/m²]."""
        return float(np.asarray(self.images_axis_unit) @ np.asarray(coeffs))

    def hinge(self, coeffs: np.ndarray) -> float:
        """One-sided penalty weight·max(0, j_φ,axis − j_axis_max)."""
        return self.weight * max(0.0, self.j_axis(coeffs) - self.j_axis_max)


def q_axis_penalty_row(
    *, images_axis_unit: np.ndarray, weight: float, j_axis_max: float
) -> SoftUpperBoundRow:
    """Soft one-sided penalty row enforcing q_0 ≥ 1 on the profile coeffs.

    ``images_axis_unit`` is the profile-basis value at the axis (ψ_N = 0,
    R = R0) per unit coefficient — a (k_dof,) vector the orchestrator supplies
    (e.g. one row of :func:`gs_solve.profile_basis` evaluated at the axis,
    scaled the same way the sweep normalises its columns).  The returned row,
    used as a soft UPPER bound, softly enforces j_φ,axis ≤ ``j_axis_max`` and
    hence q_0 ≥ 1 (:func:`q_axis_linear_bound`).
    """
    images = np.asarray(images_axis_unit, dtype=np.float64)
    weight = float(weight)
    return SoftUpperBoundRow(
        row=weight * images,
        rhs=weight * float(j_axis_max),
        j_axis_max=float(j_axis_max),
        weight=weight,
        images_axis_unit=images,
    )


def _axis_enclosing_ring(gen, level, axis):
    """Largest closed contour ring at ``level`` whose interior holds the axis."""
    from imas_ambix.latent.topology import _inside_polygon  # noqa: PLC0415

    closepoly = 79  # contourpy SeparateCode end-of-closed-polygon marker
    ar = np.array([axis[0]], dtype=np.float64)
    az = np.array([axis[1]], dtype=np.float64)
    best = None
    best_area = -1.0
    points_list, codes_list = gen.lines(float(level))
    for points, code in zip(points_list, codes_list, strict=True):
        if points.shape[0] < 6 or code[-1] != closepoly:
            continue
        if not _inside_polygon(ar, az, points[:, 0], points[:, 1])[0]:
            continue
        x, y = points[:, 0], points[:, 1]
        area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
        if area > best_area:
            best_area = area
            best = points
    return best


def q_profile(
    psi2d: np.ndarray,
    rg: np.ndarray,
    zg: np.ndarray,
    *,
    axis: tuple[float, float],
    axis_psi: float,
    boundary_psi: float,
    f_of_psin,
    r0: float,
    psi_n_surfaces: np.ndarray | None = None,
) -> dict:
    """Flux-surface safety factor q(ψ_N) by contour integration (diagnostic).

    In TOTAL flux Φ (= this module's ``psi2d``), B_pol = |∇Φ|/(2πR) and
    B_φ = F/R, so the safety factor of a closed surface is

        q(ψ) = (1/2π) ∮ B_φ/(R B_pol) dl_pol = |F| · ∮ dl_pol / (R |∇Φ|),

    with F constant on the surface.  For each requested ψ_N the
    axis-enclosing closed contour of Φ is traced (contourpy), |∇Φ| is taken by
    central differences and bilinearly sampled on the ring, and the line
    integral is evaluated on the polygon.  ``f_of_psin`` is a callable
    ψ_N ↦ F(ψ_N) = R·B_φ (integrate it from FF′ with :func:`f_from_ffprime`).

    Returns ``{"psi_n": surfaces, "q": q_on_surfaces, "q_axis": extrapolated}``.
    ``q_axis`` is a linear extrapolation of q(ψ_N) toward ψ_N = 0 (large-aspect
    circular limit q_0 = 2 B_φ0/(μ0 R_0 j_axis)).
    """
    import contourpy  # noqa: PLC0415
    from scipy.interpolate import RegularGridInterpolator  # noqa: PLC0415

    psi2d = np.asarray(psi2d, dtype=np.float64)
    rg = np.asarray(rg, dtype=np.float64)
    zg = np.asarray(zg, dtype=np.float64)
    if psi_n_surfaces is None:
        psi_n_surfaces = np.array([0.2, 0.4, 0.6, 0.8, 0.95])
    psi_n_surfaces = np.asarray(psi_n_surfaces, dtype=np.float64)

    span = boundary_psi - axis_psi
    if abs(span) < 1e-30:
        span = 1e-30

    # |∇Φ| by central differences (np.gradient takes coordinate axes)
    dphi_dz, dphi_dr = np.gradient(psi2d, zg, rg)
    grad_mag = np.hypot(dphi_dr, dphi_dz)
    grad_interp = RegularGridInterpolator(
        (zg, rg), grad_mag, bounds_error=False, fill_value=None
    )

    gen = contourpy.contour_generator(
        rg, zg, psi2d, line_type=contourpy.LineType.SeparateCode
    )

    q_vals = np.full(psi_n_surfaces.shape, np.nan)
    for i, psin in enumerate(psi_n_surfaces):
        level = axis_psi + psin * span
        ring = _axis_enclosing_ring(gen, level, axis)
        if ring is None or ring.shape[0] < 6:
            continue
        # midpoints of the closed polygon segments
        r_seg = ring[:, 0]
        z_seg = ring[:, 1]
        rm = 0.5 * (r_seg + np.roll(r_seg, -1))
        zm = 0.5 * (z_seg + np.roll(z_seg, -1))
        dl = np.hypot(np.roll(r_seg, -1) - r_seg, np.roll(z_seg, -1) - z_seg)
        gm = grad_interp(np.column_stack([zm, rm]))
        good = (gm > 1e-30) & (rm > 1e-6)
        if not good.any():
            continue
        loop = float(np.sum(dl[good] / (rm[good] * gm[good])))
        f_val = float(f_of_psin(float(psin)))
        q_vals[i] = abs(f_val) * loop

    q_axis = _extrapolate_to_axis(psi_n_surfaces, q_vals)
    return {"psi_n": psi_n_surfaces, "q": q_vals, "q_axis": q_axis}


def _extrapolate_to_axis(psi_n: np.ndarray, q: np.ndarray) -> float:
    """Linear extrapolation of q(ψ_N) to ψ_N = 0 over the innermost surfaces."""
    finite = np.isfinite(q)
    if finite.sum() == 0:
        return float("nan")
    xs = psi_n[finite]
    ys = q[finite]
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    if xs.size == 1:
        return float(ys[0])
    n = min(3, xs.size)
    a, b = np.polyfit(xs[:n], ys[:n], 1)
    return float(b)  # value at ψ_N = 0


# ---------------------------------------------------------------------------
# C. Higher-order profile regularisation (discovered, not analytic)
# ---------------------------------------------------------------------------


def curvature_gram(n_p: int, n_f: int, weight: float) -> np.ndarray:
    """Block-diagonal per-family second-difference smoothness Gram.

    Compatible builder for :func:`gs_solve._second_difference_gram`: the same
    curvature penalty on each drive family's coefficient ladder (the discrete
    ∂²/∂k² operator, normalised by the number of interior differences).  Used
    as a factor-row ridge in the per-sweep coefficient LSQ.
    """
    k = n_p + n_f
    s = np.zeros((k, k))
    if weight <= 0.0:
        return s
    for lo, n in ((0, n_p), (n_p, n_f)):
        if n < 3:
            continue
        d2 = np.zeros((n - 2, n))
        for i in range(n - 2):
            d2[i, i], d2[i, i + 1], d2[i, i + 2] = 1.0, -2.0, 1.0
        s[lo : lo + n, lo : lo + n] = weight * (d2.T @ d2) / max(n - 2, 1)
    return s


@dataclass
class MonotonicityPenalty:
    """Soft one-sided monotonicity penalty on the sampled drive profile.

    ``rows`` (m, k_dof) are the weighted forward-difference operators in ψ_N:
    for a profile expected to be NON-INCREASING outward (peaked at the axis),
    ``rows @ coeffs ≤ 0`` is satisfied and :meth:`penalty` is zero; an
    increasing (non-monotone) coefficient vector makes some ``rows @ coeffs``
    positive and incurs a penalty.  Append ``rows`` one-sided to the LSQ to
    softly discourage the non-physical direction.
    """

    rows: np.ndarray  # (m, k_dof), weight-scaled forward differences in ψ_N
    weight: float

    def penalty(self, coeffs: np.ndarray) -> float:
        """Σ relu(rows @ coeffs) — zero for a monotone-decreasing profile."""
        if self.rows.size == 0:
            return 0.0
        return float(np.clip(self.rows @ np.asarray(coeffs), 0.0, None).sum())


def monotonicity_penalty_rows(
    psi_n_samples: np.ndarray,
    basis_at_samples: np.ndarray,
    *,
    family_slices,
    weight: float,
) -> MonotonicityPenalty:
    """Build a soft monotonicity penalty on one or more drive families.

    ``basis_at_samples`` is the (n_samples, k_dof) profile-basis image matrix
    at ``psi_n_samples`` (sorted or not — sorted internally by ψ_N).  For each
    ``(start, stop)`` in ``family_slices`` the profile of that family is
    ``basis_at_samples[:, start:stop] @ coeffs[start:stop]``; the penalty rows
    are its consecutive forward differences in ψ_N (only that family's columns
    populated), scaled by ``weight``.  A positive ``rows @ coeffs`` means the
    profile INCREASES outward there — the physically discouraged direction.
    """
    psi_n = np.asarray(psi_n_samples, dtype=np.float64)
    basis = np.asarray(basis_at_samples, dtype=np.float64)
    weight = float(weight)
    order = np.argsort(psi_n)
    basis = basis[order]
    k_dof = basis.shape[1]
    rows = []
    for start, stop in family_slices:
        block = basis[:, start:stop]
        diffs = np.diff(block, axis=0)  # forward difference in ψ_N
        for d in diffs:
            row = np.zeros(k_dof)
            row[start:stop] = weight * d
            rows.append(row)
    rows_arr = np.array(rows) if rows else np.zeros((0, k_dof))
    return MonotonicityPenalty(rows=rows_arr, weight=weight)
