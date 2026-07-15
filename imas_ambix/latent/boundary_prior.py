"""Annulus soft-prior penalty rows for the free-boundary interior solve.

The interior force-balance solve (``gs_solve.solve_equilibrium_lsq``) is
anchored to the landed source-free toroidal-harmonic boundary read by a SOFT
prior: the interior-solve flux, extrapolated into the shared vacuum annulus,
must agree with the frozen harmonic flux there.  This module assembles the
weighted least-squares ROWS that penalise that disagreement, on the same
variable vector the solve already carries::

    x = [coeffs (k_dof), a_pass (kp)]              (+ 1 gauge offset g, optional)

ψ at an annulus point is LINEAR in the solve DOF::

    ψ_ann = ψ_basis_ann @ coeffs  +  ψ_pass_ann @ a_pass  +  ψ_fixed_ann

where ``ψ_basis_ann`` (n_ann, k_dof) is the cell→annulus total-flux Green's
image of the L1-normalised unit-coefficient basis currents, ``ψ_pass_ann``
(n_ann, kp) the passive images, and ``ψ_fixed_ann`` (n_ann,) the fixed
coil+vacuum flux at the annulus points.  The orchestrator builds these inside
the Picard loop (analogous to the ``g_edge`` boundary images) and passes them
in; the gradient forms use the analogous cell→annulus ∂ψ/∂R, ∂ψ/∂Z images.

Gauge (see equilibrium-boundary-closure §3.4).  The harmonic field is NOT
gauge-free -- 12 flux loops pin its absolute level -- so the penalty must KEEP
that gauge rather than blindly mean-project it away (the gate's consistency
metric mean-projects only to isolate SHAPE for a source-free-premise *check*).
Two admissible forms, both built here:

* ``form="grad-psi"`` -- match ∇ψ (equivalently B_pol) in the annulus.
  Manifestly gauge-independent (the gradient of an additive constant is zero),
  uses the best-measured quantity (the 69 B-probes).  No offset DOF.
* ``form="abs-psi"`` -- match absolute ψ with ONE shared rank-1 offset DOF ``g``
  that is FREE (unpenalised).  It absorbs only the inter-model DC nuisance --
  the weakly-pinned disagreement between two individually flux-loop-gauged
  reconstructions -- not the SHAPE.  With ``gauge_offset=False`` the offset is
  taken as pre-removed (an ablation control with no free DC DOF).

Robust weighting (§3.3e).  Per-slice consistency is heavy-tailed (median 0.066
vs mean ~4e4 on one recorded config), so a single global weight on raw RMS lets
outlier slices/points dominate.  Two levers, composed multiplicatively onto the
base ``weight``: ``per_slice_uncertainty`` whitens the whole slice by 1/σ, and
``robust_clip`` Huber-down-weights individual annulus points whose target-vs-
fixed mismatch is an outlier relative to the slice's robust bulk (the near-pole
harmonic divergences the smooth plasma basis cannot represent).

All functions are pure numpy -- no data access, no solver state.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "annulus_point_set",
    "annulus_penalty_rows",
    "harmonic_annulus_target",
    "load_frozen_harmonic_prior",
]


# --- annulus point set ------------------------------------------------------


def annulus_point_set(grid, *, psi_carrier, axis_psi, boundary_psi):
    """Flat grid indices of the shared vacuum annulus, matching the gate.

    Reproduces ``boundary_harmonic_gate_eval.annulus_consistency_rms``'s region:
    the annulus is the set of points INSIDE the limiter but OUTSIDE the confined
    region, with the confined region = flux deeper than ``boundary_psi`` toward
    ``axis_psi`` (``sign = sign(axis_psi − boundary_psi)``).

    ``boundary_psi`` and ``axis_psi`` MUST come from the §2 read's own boundary
    flux, FROZEN per slice -- never from the solve's evolving boundary -- so the
    penalty domain is fixed across the Picard loop (§3.3d).

    Parameters
    ----------
    grid
        An object exposing ``inside_limiter`` (a ``(nz, nr)`` bool mask).
    psi_carrier
        The flux field whose confined side defines the region.  Any shape whose
        ravel aligns with ``inside_limiter.ravel()`` (flat or ``(nz, nr)``).
    axis_psi, boundary_psi
        Frozen confined-side reference and boundary flux (§2 read).

    Returns
    -------
    numpy.ndarray
        Sorted flat indices (into the ravelled ``(nz, nr)`` grid) of the annulus.
    """
    psi = np.asarray(psi_carrier, dtype=np.float64).ravel()
    inside = np.asarray(grid.inside_limiter, dtype=bool).ravel()
    if psi.shape != inside.shape:
        raise ValueError(
            f"psi_carrier ravel {psi.shape} != inside_limiter ravel {inside.shape}"
        )
    sign = np.sign(axis_psi - boundary_psi)
    confined = (psi - boundary_psi) * sign > 0.0
    annulus = inside & ~confined
    return np.where(annulus)[0]


# --- robust weighting -------------------------------------------------------


def _robust_weights(residual_proxy, robust_clip):
    """Huber-style per-row weights from a residual proxy.

    Points within ``robust_clip`` robust-sigma of the (median-centred) bulk keep
    weight 1; beyond that a point of standardised deviation ``z`` gets weight
    ``robust_clip / z`` (the Huber down-weight).  The proxy is CENTRED by its
    median before scoring, so a uniform additive shift -- an absolute-ψ gauge
    change -- leaves the weights invariant.  The robust scale is the MAD
    (×1.4826, the Gaussian-consistent estimator); a degenerate (zero) scale
    returns uniform weights.
    """
    r = np.asarray(residual_proxy, dtype=np.float64)
    if r.size == 0 or robust_clip is None:
        return np.ones(r.shape, dtype=np.float64)
    centred = r - np.median(r)
    scale = 1.4826 * np.median(np.abs(centred))
    w = np.ones(r.shape, dtype=np.float64)
    if not (scale > 0.0):
        return w
    z = np.abs(centred) / scale
    hot = z > robust_clip
    w[hot] = robust_clip / z[hot]
    return w


# --- penalty rows -----------------------------------------------------------


def _as_design(basis, npts, ncol):
    """Coerce a possibly-``None`` design block to a ``(npts, ncol)`` array."""
    if ncol == 0:
        return np.zeros((npts, 0), dtype=np.float64)
    if basis is None:
        raise ValueError("design block is None but the DOF count is non-zero")
    arr = np.asarray(basis, dtype=np.float64)
    if arr.shape != (npts, ncol):
        raise ValueError(f"design block {arr.shape} != expected {(npts, ncol)}")
    return arr


def annulus_penalty_rows(
    *,
    form,
    psi_basis_ann,
    psi_pass_ann,
    psi_fixed_ann,
    psi_target_ann,
    grad_basis_ann=None,
    grad_pass_ann=None,
    grad_fixed_ann=None,
    grad_target_ann=None,
    k_dof,
    kp,
    weight,
    per_slice_uncertainty=None,
    robust_clip=None,
    gauge_offset=False,
):
    """Weighted least-squares rows penalising annulus flux disagreement.

    Returns ``(A_extra, b_extra)`` such that appending them to the solve's
    whitened design and stacking ``b_extra`` under ``y`` adds the annulus soft
    prior.  Rows act on ``x = [coeffs (k_dof), a_pass (kp)]``, extended by a
    trailing free offset column ``g`` when ``form="abs-psi"`` and
    ``gauge_offset=True``.  Each row ``i`` is the whitened residual

        grad-psi : w_i·(∇ψ_basis·coeffs + ∇ψ_pass·a_pass + ∇ψ_fixed − ∇ψ_target)
        abs-psi  : w_i·(ψ_basis·coeffs + ψ_pass·a_pass + ψ_fixed − ψ_target − g)

    driven to zero, with per-row weight ``w_i = weight / σ · h_i`` where ``σ`` is
    ``per_slice_uncertainty`` (default 1) and ``h_i`` the Huber robust weight
    (default 1; see :func:`_robust_weights`).

    Parameters
    ----------
    form : {"grad-psi", "abs-psi"}
        Gauge-keeping penalty form (see module docstring / §3.4).
    psi_basis_ann, psi_pass_ann, psi_fixed_ann, psi_target_ann
        Absolute-ψ blocks at the annulus points: basis ``(n_ann, k_dof)``,
        passive ``(n_ann, kp)``, fixed coil+vacuum ``(n_ann,)``, harmonic target
        ``(n_ann,)``.  Required for ``form="abs-psi"``; ignored for grad-ψ.
    grad_basis_ann, grad_pass_ann, grad_fixed_ann, grad_target_ann
        Gradient blocks, R and Z components stacked by the caller into
        ``n_grad`` rows (typically ``2·n_ann``): basis ``(n_grad, k_dof)``,
        passive ``(n_grad, kp)``, fixed ``(n_grad,)``, target ``(n_grad,)``.
        Same convention on target and basis (∂ψ/∂R, ∂ψ/∂Z of TOTAL flux, or
        both scaled point-wise to B_pol -- the module is agnostic as long as the
        two sides match).  Required for ``form="grad-psi"``.
    k_dof, kp : int
        Profile-coefficient and passive-mode counts (the layout of ``x``).
    weight : float
        Base prior weight (relative to the whitened magnetics rows).
    per_slice_uncertainty : float, optional
        Slice std-like scale σ; rows are scaled by 1/σ.  Feed the read's own
        uncertainty (e.g. from the frozen prior's ``misfit`` / ``coeff_cov``).
    robust_clip : float, optional
        Huber threshold (in robust-sigma) for per-point down-weighting; ``None``
        disables it.  The residual proxy is ``target − fixed`` (the mismatch the
        DOF must reproduce), centred and MAD-scaled.
    gauge_offset : bool
        ``form="abs-psi"`` only: add the free rank-1 offset column ``g``.  When
        ``False`` the offset is treated as pre-removed (ablation control).

    Returns
    -------
    (A_extra, b_extra) : (numpy.ndarray, numpy.ndarray)
        ``A_extra`` shape ``(n_rows, n_var)`` with ``n_var = k_dof + kp`` (+1 for
        the abs-ψ offset column); ``b_extra`` shape ``(n_rows,)``.
    """
    if form not in ("grad-psi", "abs-psi"):
        raise ValueError(
            f"form must be 'grad-psi' or 'abs-psi', got {form!r} "
            "(the gate's blind mean-projection is not an admissible penalty)"
        )
    sigma = 1.0 if per_slice_uncertainty is None else float(per_slice_uncertainty)
    if not (sigma > 0.0):
        raise ValueError("per_slice_uncertainty must be positive")
    base = float(weight) / sigma

    if form == "grad-psi":
        target = np.asarray(grad_target_ann, dtype=np.float64).ravel()
        fixed = np.asarray(grad_fixed_ann, dtype=np.float64).ravel()
        npts = target.shape[0]
        basis = _as_design(grad_basis_ann, npts, k_dof)
        passv = _as_design(grad_pass_ann, npts, kp)
        design = np.hstack([basis, passv])
        resid_target = target - fixed
        offset_col = None
    else:  # abs-psi
        target = np.asarray(psi_target_ann, dtype=np.float64).ravel()
        fixed = np.asarray(psi_fixed_ann, dtype=np.float64).ravel()
        npts = target.shape[0]
        basis = _as_design(psi_basis_ann, npts, k_dof)
        passv = _as_design(psi_pass_ann, npts, kp)
        design = np.hstack([basis, passv])
        resid_target = target - fixed
        # the free shared offset g enters as a −1 column; the residual is
        #   ψ_solve − ψ_target − g = 0  ⇒  design·x − g = target − fixed
        offset_col = -np.ones((npts, 1)) if gauge_offset else None

    if offset_col is not None:
        design = np.hstack([design, offset_col])

    h = _robust_weights(resid_target, robust_clip)
    w = base * h
    a_extra = w[:, np.newaxis] * design
    b_extra = w * resid_target
    return a_extra, b_extra


# --- frozen harmonic target -------------------------------------------------


def load_frozen_harmonic_prior(*args, **kwargs):
    """Re-export of the frozen-prior loader owned by the harmonic-freeze script.

    The frozen prior (per-slice coeffs / labels / cfg / misfit / coeff_cov) is
    written by ``scripts/harmonic_prior_freeze.py``; this thin re-export lets the
    orchestrator import the loader from one place next to the penalty rows.
    Imported lazily so this module has no hard dependency on the freeze script.
    """
    from imas_ambix.latent.harmonic_prior_freeze import (  # type: ignore
        load_frozen_harmonic_prior as _loader,
    )

    return _loader(*args, **kwargs)


def _slice_field(frozen_slice, name):
    """Read ``name`` from a frozen-slice object (attribute) or mapping (key)."""
    if hasattr(frozen_slice, name):
        return getattr(frozen_slice, name)
    if isinstance(frozen_slice, dict):
        return frozen_slice[name]
    raise AttributeError(f"frozen slice has no field {name!r}")


def harmonic_annulus_target(frozen_slice, grid, ann_idx, form, *, fd_step=None):
    """Evaluate the frozen harmonic ψ (or ∇ψ) at the annulus points.

    Uses the frozen slice's own harmonic ``cfg`` + ``coeffs`` and evaluates the
    plasma harmonic flux directly at the annulus ``(R, Z)`` via
    :func:`imas_ambix.latent.boundary_harmonic.harmonic_columns` (arbitrary-point
    evaluation, no full-grid raster).

    Parameters
    ----------
    frozen_slice
        Object/mapping exposing ``cfg`` (a ``HarmonicFitConfig``) and ``coeffs``.
    grid
        Grid exposing ``flat_r`` / ``flat_z``.
    ann_idx : array-like
        Flat annulus indices (from :func:`annulus_point_set`).
    form : {"grad-psi", "abs-psi"}
        ``"abs-psi"`` returns the ψ target ``(n_ann,)``; ``"grad-psi"`` returns
        the R- and Z-derivative components STACKED ``(2·n_ann,)`` (R block then
        Z block), matching the row layout the caller uses for the grad basis.
    fd_step : float, optional
        Central-difference step for the gradient (grad-ψ only).  Defaults to a
        small fraction of the grid spacing.  A finite-difference gradient of the
        source-free harmonic field is used when the harmonic module exposes no
        analytic point-wise gradient; it is O(h²) accurate on the smooth annulus
        field.

    Returns
    -------
    numpy.ndarray
        The harmonic target at the annulus points for the requested form.
    """
    from imas_ambix.latent import boundary_harmonic as bh

    cfg = _slice_field(frozen_slice, "cfg")
    coeffs = np.asarray(_slice_field(frozen_slice, "coeffs"), dtype=np.float64)
    idx = np.asarray(ann_idx, dtype=np.intp)
    ar = np.asarray(grid.flat_r, dtype=np.float64)[idx]
    az = np.asarray(grid.flat_z, dtype=np.float64)[idx]

    def psi_at(r, z):
        cols, _ = bh.harmonic_columns(r, z, cfg)
        return cols @ coeffs

    if form == "abs-psi":
        return psi_at(ar, az)
    if form != "grad-psi":
        raise ValueError(f"form must be 'grad-psi' or 'abs-psi', got {form!r}")

    # Prefer an analytic point-wise gradient if the harmonic module grows one;
    # otherwise central-difference the source-free field (smooth on the annulus).
    if hasattr(bh, "harmonic_grad_columns"):
        gr_cols, gz_cols, _ = bh.harmonic_grad_columns(ar, az, cfg)
        d_dr = gr_cols @ coeffs
        d_dz = gz_cols @ coeffs
        return np.concatenate([d_dr, d_dz])

    if fd_step is None:
        dr = (
            float(np.mean(np.diff(np.asarray(grid.rg)))) if hasattr(grid, "rg") else 0.0
        )
        dz = (
            float(np.mean(np.diff(np.asarray(grid.zg)))) if hasattr(grid, "zg") else 0.0
        )
        h = 0.25 * max(dr, dz)
        if not (h > 0.0):
            h = 1e-3
    else:
        h = float(fd_step)
    d_dr = (psi_at(ar + h, az) - psi_at(ar - h, az)) / (2.0 * h)
    d_dz = (psi_at(ar, az + h) - psi_at(ar, az - h)) / (2.0 * h)
    return np.concatenate([d_dr, d_dz])
