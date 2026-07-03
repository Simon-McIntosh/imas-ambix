"""Profile-free, differentiable Grad-Shafranov *structure* residual.

In the finite-area patch-current basis Ampère's law and ``div B = 0`` are
identities of the assembly matrices, so the only physics content left in
Grad-Shafranov is the **flux-function structure of the current**: on each
connected ψ level-set the toroidal current density obeys

    jφ = R·p′(ψ) + FF′(ψ)/(μ₀·R)          ⇔          R·jφ = p′(ψ)·R² + FF′(ψ)/μ₀ ,

i.e. ``R·jφ`` is an *affine* function of ``x = R²`` on a level set — slope
``a = p′`` [Pa/Wb], intercept ``b = FF′/μ₀``.  Neither profile shape nor
topology is prescribed; a state is GS-compliant iff, level-set by level-set,
this affine relation holds.  The residual below measures the *unexplained*
current-weighted variance of that relation and is therefore zero for **any**
force-balanced state regardless of its profiles, and O(1) for a structureless
current.

Design (mirrors the scoping validation, artifact
``imas_ambix/latent/artifacts/patch_scoping/discriminate-affine-r2.json`` —
real equilibria at the ~0.0036 soft-binning floor, sensor-null-space
perturbations ~5.7× floor, permuted currents ~20× floor):

* soft-bin cells by their **total** ψ (bin centres/bandwidth detached — auxiliary
  geometry — but the Gaussian kernel weights differentiate through ``psi_c``, so
  gradients flow through the level-set geometry as well as through ``jphi_c``);
* weight each cell by ``jφ²`` so zero-current vacuum cells never pollute a bin;
* per bin, a closed-form ridge-stabilised 2×2 weighted least squares eliminates
  ``(a, b)`` analytically — no inner optimisation, no profile basis;
* the readout is the current-weighted unexplained fraction ∈ [0, 1].

**Connectivity caveat (the honest form of "no topology").**  The flux-function
property holds *per connected component* of a level set.  In a diverted plasma
one ψ value can label a core surface, a private-flux arc, and an SOL contour;
naïve ψ-value grouping wrongly ties the same ``(a, b)`` to all three.  The
``jφ²`` weighting already makes the constraint vacuous where a level carries no
current; where disconnected components both carry current the ``connectivity``
arm adds a soft, label-free, still-differentiable (R, Z)-locality kernel factor
(``"locality"``), or splits each ψ-bin by no-grad connected-component labels
computed outside the gradient path (``"labels"``).  Topology enters *softly*,
never as a hard separatrix extraction in the loop.

**Discovery-signal framing.**  A persistent residual is *signal*, not an error
to fit away: genuine halo / SOL current or anomalous redistribution that breaks
axisymmetric force balance reports here honestly.

**Rejected alternative — pointwise tangential elimination.**  One could instead
enforce ``B·∇jφ`` structure pointwise by projecting out the flux-surface
tangential derivative.  That form is singular at the midplane (where ∇ψ is
purely radial) and at every field null (where ∇ψ → 0), so it injects spurious
large residuals exactly at the O-point and X-points; the level-set-binned
weighted-LSQ form above is well-conditioned everywhere current flows and is the
one used.

Firewall: this module imports nothing from the EFIT / evaluator side.  The only
optional dependency is ``scipy.ndimage`` for the explicitly no-grad label helper.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

MU0 = 4.0e-7 * 3.141592653589793
"""Vacuum permeability [T·m/A]."""

_FORMS = ("affine-r2", "jphi")
_RIDGE = 1e-9


# --------------------------------------------------------------------------
# internals — the shared binned weighted-LSQ machinery
# --------------------------------------------------------------------------


def _design(form: str, r_c: torch.Tensor, jphi_c: torch.Tensor):
    """Return (x, y) design for the per-bin 2×2 fit.

    ``affine-r2``:  y = R·jφ,  x = [R², 1]  (genuinely affine; a = p′, b = FF′/μ₀)
    ``jphi``:       y = jφ,    x = [R, 1/R] (the as-scoped conditioning; identical
                    physics up to an R² weighting).
    """
    if form == "affine-r2":
        x = torch.stack([r_c * r_c, torch.ones_like(r_c)], dim=-1)  # (N, 2)
        y = r_c * jphi_c
    elif form == "jphi":
        x = torch.stack([r_c, 1.0 / r_c], dim=-1)  # (N, 2)
        y = jphi_c
    else:
        raise ValueError(f"unknown structure-residual form: {form!r} (use {_FORMS})")
    return x, y


def _bin_grid(
    psi_c: torch.Tensor, w_amp: torch.Tensor, n_bins: int, bandwidth_bins: float
):
    """The detached ψ-bin centres ``mu`` and Gaussian bandwidth ``h``.

    Auxiliary geometry: placed at the jφ²-weighted ψ mean ± 2.5 σ.  Kept out of
    the gradient graph so the residual differentiates only through the kernel
    (and through jφ), not through where the bins sit.
    """
    with torch.no_grad():
        mean = (w_amp * psi_c).sum()
        std = torch.sqrt((w_amp * (psi_c - mean) ** 2).sum()).clamp_min(1e-12)
        lo, hi = mean - 2.5 * std, mean + 2.5 * std
        mu = torch.linspace(
            float(lo), float(hi), n_bins, dtype=psi_c.dtype, device=psi_c.device
        )
        h = bandwidth_bins * (hi - lo) / n_bins
    return mu, h


def _bin_kernels(
    psi_c: torch.Tensor,
    w_amp: torch.Tensor,
    n_bins: int,
    bandwidth_bins: float,
    bin_grid: tuple[torch.Tensor, torch.Tensor] | None = None,
):
    """Soft ψ-bin kernel weights ``w0[k, n]`` (Gaussian in ψ × jφ² amplitude).

    Bin centres ``mu`` and bandwidth ``h`` are detached (auxiliary geometry);
    the kernel itself differentiates through ``psi_c``.  Pass ``bin_grid`` to
    hold the binning fixed (see :func:`structure_residual`).
    """
    if bin_grid is None:
        mu, h = _bin_grid(psi_c, w_amp, n_bins, bandwidth_bins)
    else:
        mu, h = bin_grid
    kern = torch.exp(-0.5 * ((psi_c[None, :] - mu[:, None]) / h) ** 2)  # (B, N)
    w0 = kern * w_amp[None, :]
    return w0, mu, h


def _locality_factor(
    w0: torch.Tensor,
    r_c: torch.Tensor,
    z_c: torch.Tensor,
    locality_scale: float | None,
):
    """Per-bin (R, Z)-locality multiplier with DETACHED per-bin centroids.

    For each ψ-bin ``k`` the (jφ²·ψ-kernel)-weighted centroid ``(R_k, Z_k)`` is
    computed under no-grad; cells far from it (a disconnected component at the
    same ψ) are suppressed by ``exp(−((R−R_k)²+(Z−Z_k)²)/(2·scale²))``.  The
    factor does not depend on ``psi_c`` (centroid detached), so it is a fixed
    per-(k, n) multiplier — gradients still flow through ``w0``'s ψ-kernel.

    ``locality_scale`` default: twice the median per-bin weighted RMS spatial
    radius of the current cloud (adapts to grid resolution; documented sane
    default — pass an explicit scale to resolve components closer than this).
    """
    with torch.no_grad():
        mass = w0.sum(-1).clamp_min(1e-30)  # (B,)
        r_k = (w0 * r_c[None, :]).sum(-1) / mass  # (B,)
        z_k = (w0 * z_c[None, :]).sum(-1) / mass
        d2 = (r_c[None, :] - r_k[:, None]) ** 2 + (z_c[None, :] - z_k[:, None]) ** 2
        if locality_scale is None:
            rms = torch.sqrt((w0 * d2).sum(-1) / mass).clamp_min(1e-6)  # (B,)
            good = w0.sum(-1) > 1e-12 * w0.sum().clamp_min(1e-30)
            med = rms[good].median() if bool(good.any()) else rms.median()
            scale = 2.0 * float(med)
        else:
            scale = float(locality_scale)
        loc = torch.exp(-0.5 * d2 / (scale * scale))  # (B, N), detached
    return loc


def _label_expand(w0: torch.Tensor, component_labels: torch.Tensor) -> torch.Tensor:
    """Expand ψ-bins into (ψ-bin × connected-component) rows (weights zeroed
    across label mismatch).  Labels are treated as constants (no gradient)."""
    if component_labels is None:
        raise ValueError("connectivity='labels' requires component_labels")
    lab = component_labels.detach().to(torch.long).reshape(-1)
    uniq = torch.unique(lab)
    onehot = (lab[None, :] == uniq[:, None]).to(w0.dtype)  # (L, N)
    w = w0[:, None, :] * onehot[None, :, :]  # (B, L, N)
    return w.reshape(-1, w0.shape[-1])  # (B*L, N)


def _binned_weights(
    psi_c: torch.Tensor,
    r_c: torch.Tensor,
    jphi_c: torch.Tensor,
    *,
    n_bins: int,
    bandwidth_bins: float,
    z_c: torch.Tensor | None,
    connectivity: str | None,
    locality_scale: float | None,
    component_labels: torch.Tensor | None,
    bin_grid: tuple[torch.Tensor, torch.Tensor] | None = None,
):
    """Return (W, mu) — the (rows, N) weight matrix and the ψ-bin centres."""
    w_amp = jphi_c * jphi_c
    total = w_amp.sum()
    if float(total) <= 0.0:
        return None, None
    w_amp = w_amp / total
    w0, mu, _ = _bin_kernels(psi_c, w_amp, n_bins, bandwidth_bins, bin_grid=bin_grid)
    if connectivity is None:
        return w0, mu
    if connectivity == "locality":
        if z_c is None:
            raise ValueError("connectivity='locality' requires z_c")
        return w0 * _locality_factor(w0, r_c, z_c, locality_scale), mu
    if connectivity == "labels":
        return _label_expand(w0, component_labels), mu
    raise ValueError(
        f"unknown connectivity: {connectivity!r} (use None, 'locality', 'labels')"
    )


def _normal_equations(w: torch.Tensor, x: torch.Tensor, y: torch.Tensor):
    """Assemble the per-bin 2×2 normal equations (XᵀWX, XᵀWy) via matvecs.

    Avoids a 4-D ``(B, N, 2, 2)`` einsum intermediate — the products are formed
    once as length-N vectors and contracted by ``w @ vec`` (a (B, N)·(N,) matvec),
    which is far cheaper and much less thread-hostile on small tensors.
    """
    x0, x1 = x[:, 0], x[:, 1]
    a00 = w @ (x0 * x0)  # (B,)
    a01 = w @ (x0 * x1)
    a11 = w @ (x1 * x1)
    m = torch.stack(
        [torch.stack([a00, a01], -1), torch.stack([a01, a11], -1)], -2
    )  # (B, 2, 2)
    c = torch.stack([w @ (x0 * y), w @ (x1 * y)], -1)  # (B, 2)
    return m, c


def _ridge(m: torch.Tensor) -> torch.Tensor:
    """Ridge-stabilise a batch of 2×2 normal matrices in place-style (returns new)."""
    eye = torch.eye(2, dtype=m.dtype, device=m.device)
    r = _RIDGE * m.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-30)
    return m + r[:, None, None] * eye


def _solve_bins(w: torch.Tensor, x: torch.Tensor, y: torch.Tensor):
    """Batched ridge-stabilised weighted 2×2 solve.  Returns (beta, fit).

    beta[b] = argmin_β Σ_n w[b,n] (y[n] − x[n]·β)² ,  fit[b,n] = x[n]·beta[b].
    """
    m, c = _normal_equations(w, x, y)
    beta = torch.linalg.solve(_ridge(m), c)  # (B, 2)
    fit = beta[:, 0:1] * x[:, 0][None, :] + beta[:, 1:2] * x[:, 1][None, :]  # (B, N)
    return beta, fit


# --------------------------------------------------------------------------
# the residual
# --------------------------------------------------------------------------


def structure_residual(
    psi_c: torch.Tensor,
    r_c: torch.Tensor,
    jphi_c: torch.Tensor,
    *,
    n_bins: int = 24,
    bandwidth_bins: float = 1.0,
    form: str = "affine-r2",
    z_c: torch.Tensor | None = None,
    connectivity: str | None = None,
    locality_scale: float | None = None,
    component_labels: torch.Tensor | None = None,
    bin_grid: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Unexplained current-weighted fraction of the GS flux-function structure.

    Scalar in [0, 1]: 0 for an exactly GS-compliant current (any profile shape,
    any topology), O(1) for a structureless one.  Differentiable through both
    ``psi_c`` (the level-set geometry, via the ψ-kernel) and ``jphi_c`` (via the
    amplitude weighting and the response ``y``); bin placement is detached.

    Parameters
    ----------
    psi_c, r_c, jphi_c:
        Per-cell 1-D tensors: total poloidal flux [Wb], major radius [m], and
        toroidal current density [A/m²].
    n_bins, bandwidth_bins:
        Number of soft ψ-bins and the Gaussian bandwidth in bin-spacing units.
    form:
        ``"affine-r2"`` (default) or ``"jphi"`` — see :func:`_design`.
    z_c:
        Per-cell Z [m]; required for ``connectivity='locality'``.
    connectivity:
        ``None`` (naïve ψ-only binning, the ablation baseline), ``"locality"``
        (single-centroid (R, Z) kernel factor), or ``"labels"`` (per-component
        sub-bins from ``component_labels``).
    locality_scale:
        (R, Z) locality length [m] for ``connectivity='locality'``; ``None`` ⇒
        the documented adaptive default (see :func:`_locality_factor`).
    component_labels:
        Int tensor per cell (connected-component id) for ``connectivity='labels'``,
        computed OUTSIDE the gradient path (see :func:`connected_component_labels`).
    bin_grid:
        Optional precomputed ``(centres, bandwidth)`` (from :func:`_bin_grid`) that
        holds the ψ-binning fixed instead of re-deriving it from ``psi_c``.  Use to
        freeze bins across an optimisation, or to gradient-check the intended
        (bins-detached) sensitivity.
    """
    w, _ = _binned_weights(
        psi_c,
        r_c,
        jphi_c,
        n_bins=n_bins,
        bandwidth_bins=bandwidth_bins,
        z_c=z_c,
        connectivity=connectivity,
        locality_scale=locality_scale,
        component_labels=component_labels,
        bin_grid=bin_grid,
    )
    if w is None:
        return torch.zeros((), dtype=psi_c.dtype, device=psi_c.device)
    x, y = _design(form, r_c, jphi_c)
    _, fit = _solve_bins(w, x, y)
    num = (w * (y[None, :] - fit) ** 2).sum()
    den = (w * (y[None, :] ** 2)).sum().clamp_min(1e-30)
    return num / den


# --------------------------------------------------------------------------
# closure recovery (the fit exposed instead of eliminated)
# --------------------------------------------------------------------------


@dataclass
class ClosureFit:
    """Per-ψ-bin recovered flux-function coefficients with regression errors.

    For ``form='affine-r2'`` the coefficients ARE the closures: ``a_k = p′``
    [Pa/Wb], ``b_k = FF′/μ₀`` [A/m²·... ] per bin, with standard errors from the
    weighted-LSQ covariance and the per-bin weight mass.  All fields are 1-D
    length-``n_bins`` (detached) tensors.
    """

    psi_centers: torch.Tensor
    a_k: torch.Tensor
    b_k: torch.Tensor
    a_err: torch.Tensor
    b_err: torch.Tensor
    weight_mass: torch.Tensor


def fit_flux_functions(
    psi_c: torch.Tensor,
    r_c: torch.Tensor,
    jphi_c: torch.Tensor,
    *,
    n_bins: int = 24,
    bandwidth_bins: float = 1.0,
    form: str = "affine-r2",
    z_c: torch.Tensor | None = None,
    connectivity: str | None = None,
    locality_scale: float | None = None,
) -> ClosureFit:
    """Recover the per-ψ-bin flux-function coefficients and their uncertainties.

    Same weighted-LSQ as :func:`structure_residual`, but the ``(a, b)`` are
    exposed rather than eliminated.  Per-bin standard errors are
    ``σ̂²·(XᵀW̃X)⁻¹`` with ``σ̂²`` the weighted residual variance over an
    effective-sample-size (Kish) degrees-of-freedom, ``n_eff = (Σw)²/Σw²``.

    ``connectivity`` accepts ``None`` or ``'locality'`` (both length ``n_bins``);
    ``'labels'`` is a residual-only mode — the exposed coefficients are per ψ-bin.
    """
    if connectivity == "labels":
        raise ValueError(
            "fit_flux_functions exposes per-ψ-bin coefficients; connectivity="
            "'labels' expands rows and is residual-only"
        )
    w, mu = _binned_weights(
        psi_c,
        r_c,
        jphi_c,
        n_bins=n_bins,
        bandwidth_bins=bandwidth_bins,
        z_c=z_c,
        connectivity=connectivity,
        locality_scale=locality_scale,
        component_labels=None,
    )
    if w is None:
        z = torch.zeros(n_bins, dtype=psi_c.dtype, device=psi_c.device)
        return ClosureFit(*(z.clone() for _ in range(6)))
    x, y = _design(form, r_c, jphi_c)

    sw = w.sum(-1)  # (B,) raw mass
    sw2 = (w * w).sum(-1)
    n_eff = (sw * sw) / sw2.clamp_min(1e-300)  # (B,) Kish effective sample size
    # frequency-weight rescale: Σ w̃ = n_eff, so SEs behave like ordinary
    # regression on n_eff points
    wn = w * (n_eff / sw.clamp_min(1e-300))[:, None]

    m0, c = _normal_equations(wn, x, y)
    m = _ridge(m0)
    beta = torch.linalg.solve(m, c)  # (B, 2)
    fit = beta[:, 0:1] * x[:, 0][None, :] + beta[:, 1:2] * x[:, 1][None, :]  # (B, N)
    resid = y[None, :] - fit
    rss = (wn * resid**2).sum(-1)  # (B,)
    dof = (n_eff - 2.0).clamp_min(1e-6)
    sigma2 = rss / dof  # (B,)
    cov = sigma2[:, None, None] * torch.linalg.inv(m)  # (B, 2, 2)
    a_err = torch.sqrt(cov[:, 0, 0].clamp_min(0.0))
    b_err = torch.sqrt(cov[:, 1, 1].clamp_min(0.0))

    return ClosureFit(
        psi_centers=mu.detach(),
        a_k=beta[:, 0].detach(),
        b_k=beta[:, 1].detach(),
        a_err=a_err.detach(),
        b_err=b_err.detach(),
        weight_mass=sw.detach(),
    )


def integrate_closures(
    fit: ClosureFit,
    *,
    psi_axis: float,
    psi_boundary: float,
    f_vac: float,
    mass_frac_threshold: float = 1e-3,
) -> dict:
    """Integrate the recovered coefficients into p(ψ), F²(ψ), F(ψ).

    Orders the well-populated bins from boundary to axis and trapezoid-integrates
    from the boundary (p(boundary) = 0):

        p(ψ)   = ∫_ψb^ψ a dψ′                 (a = p′)
        F²(ψ)  = F²_vac + 2·μ₀·∫_ψb^ψ b dψ′   (b = FF′/μ₀)
        F(ψ)   = √(max(F², 0))

    Bins whose ``weight_mass`` is below ``mass_frac_threshold × max(weight_mass)``
    are dropped (empty / vacuum bins carry no closure information).  Returns a
    dict with numpy-array keys ``"psi"``, ``"p"``, ``"f_squared"``, ``"f"``.
    """
    import numpy as np

    psi = np.asarray(fit.psi_centers, dtype=np.float64)
    a_k = np.asarray(fit.a_k, dtype=np.float64)
    b_k = np.asarray(fit.b_k, dtype=np.float64)
    mass = np.asarray(fit.weight_mass, dtype=np.float64)

    keep = mass > mass_frac_threshold * (mass.max() if mass.size else 0.0)
    if keep.sum() < 2:
        empty = np.zeros(0)
        return {"psi": empty, "p": empty, "f_squared": empty, "f": empty}
    psi, a_k, b_k = psi[keep], a_k[keep], b_k[keep]

    # order from boundary → axis along the (signed) ψ direction
    direction = 1.0 if psi_axis >= psi_boundary else -1.0
    order = np.argsort(direction * (psi - psi_boundary))
    psi, a_k, b_k = psi[order], a_k[order], b_k[order]

    def _cumtrapz_from_boundary(vals: np.ndarray) -> np.ndarray:
        out = np.zeros_like(vals)
        # integrate from the boundary anchor ψ_b to the first retained centre,
        # then trapezoid across the ordered centres
        dpsi0 = psi[0] - psi_boundary
        out[0] = 0.5 * (vals[0] + vals[0]) * dpsi0  # anchor value ≈ vals[0]
        for i in range(1, len(vals)):
            out[i] = out[i - 1] + 0.5 * (vals[i] + vals[i - 1]) * (psi[i] - psi[i - 1])
        return out

    p = _cumtrapz_from_boundary(a_k)
    f_squared = f_vac * f_vac + 2.0 * MU0 * _cumtrapz_from_boundary(b_k)
    f = np.sqrt(np.clip(f_squared, 0.0, None))
    return {"psi": psi, "p": p, "f_squared": f_squared, "f": f}


# --------------------------------------------------------------------------
# ablatable soft rungs (OFF unless explicitly called)
# --------------------------------------------------------------------------


def f2_integrability_penalty(
    b_k: torch.Tensor, dpsi: torch.Tensor | float, f_vac: float
) -> torch.Tensor:
    """Penalise negative F²(ψ) along the cumulative-sum sequence.

    ``F²(ψ) = F²_vac + 2·μ₀·Σ b dψ`` must stay ≥ 0 for the recovered ``b(ψ) =
    FF′/μ₀`` to correspond to a real toroidal field.  Returns ``Σ relu(−F²)``
    over the cumulative sequence — zero for a compliant ``b_k``, positive for a
    pathologically negative one.  ``b_k`` must be ordered boundary → axis.
    """
    if not torch.is_tensor(dpsi):
        dpsi = torch.as_tensor(dpsi, dtype=b_k.dtype, device=b_k.device)
    f2_seq = f_vac * f_vac + 2.0 * MU0 * torch.cumsum(b_k * dpsi, dim=0)
    return torch.relu(-f2_seq).sum()


def coefficient_smoothness_penalty(coeffs: torch.Tensor) -> torch.Tensor:
    """Mean squared second difference of a coefficient sequence ({a_k} or {b_k}).

    An optional ψ-smoothness regulariser; OFF by default (call explicitly).
    """
    if coeffs.numel() < 3:
        return torch.zeros((), dtype=coeffs.dtype, device=coeffs.device)
    d2 = coeffs[2:] - 2.0 * coeffs[1:-1] + coeffs[:-2]
    return (d2 * d2).mean()


def edge_taper_penalty(jphi_c: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
    """Weighted mean-square edge current — a taper toward zero current at the
    plasma edge.

    OFF BY DEFAULT and deliberately not wired anywhere: halo / SOL edge currents
    are REAL and carry discovery signal, so tapering them away would suppress
    exactly the physics the residual is meant to surface.  Provided only so an
    ablation can measure the cost of assuming a current-free edge.
    """
    return (edge_weight * jphi_c * jphi_c).sum() / edge_weight.sum().clamp_min(1e-30)


# --------------------------------------------------------------------------
# optional no-grad connectivity helper
# --------------------------------------------------------------------------


def connected_component_labels(psi_2d, psi_lo: float, psi_hi: float):
    """No-grad connected-component labels of the ψ ∈ [lo, hi] band (numpy).

    A convenience wrapper around ``scipy.ndimage.label`` on a ψ-band mask, for
    building ``component_labels`` for ``connectivity='labels'``.  Purely numpy /
    no-grad by construction — call outside any autograd context and feed the
    result back as a constant int tensor.
    """
    import numpy as np
    from scipy import ndimage

    band = (np.asarray(psi_2d) >= psi_lo) & (np.asarray(psi_2d) <= psi_hi)
    labels, _ = ndimage.label(band)
    return labels


__all__ = [
    "MU0",
    "ClosureFit",
    "coefficient_smoothness_penalty",
    "connected_component_labels",
    "edge_taper_penalty",
    "f2_integrability_penalty",
    "fit_flux_functions",
    "integrate_closures",
    "structure_residual",
]
