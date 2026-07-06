"""Constrained external-field boundary read via a low-order current-moment basis.

The plasma boundary (separatrix / X-point / last-closed-flux-surface) is
**externally well-posed**: outside the plasma J_phi = 0, so the poloidal flux
solves the *homogeneous* Grad-Shafranov operator Delta* psi = 0, and the
external magnetics fix a finite set of low-order current moments -- total
current Ip, the current centroid (R, Z), and the low-order shape moments
(elongation, triangularity) -- that set the boundary shape.  Reading the
boundary off a *free* ~5000-DOF interior patch current lets interior null-space
slack (current arrangements that fit the magnetics equally) leak small-scale
lumpiness into the near-boundary field, which surfaces as spurious off-axis
saddles and an under-sized last-closed surface.

This module represents the plasma current with a small moment basis
(``K`` ~ 6-10 monomials in the normalised in-plasma coordinates), fits its
amplitudes to the plasma sensor signature (``measured - vacuum``) by whitened
least squares with a Rogowski-Ip anchor, and reconstructs a smooth *total* flux
psi(R, Z).  Because the current carries only a handful of moments, the external
field is a well-conditioned function of exactly the quantities the magnetics
constrain -- no interior slack, hence no spurious near-boundary saddles.

It reuses the existing Green's operator -- ``PatchBasis.m_sens`` (patch-current
-> sensor) and ``PatchBasis.g_pg`` (patch-current -> grid flux, via
:meth:`PatchBasis.psi_grid_2d_np`) -- so the reconstructed psi is directly
comparable to the free-current read and the same topology read + firewalled-EFIT
scoring apply unchanged.  The current lives on the *same* patch cells, so the
patch-psi and the moment-psi share a vacuum region where both are valid: their
agreement there is the checkable consistency condition of the plan.

Conventions preserved from the patch substrate: total poloidal flux
Phi = 2 pi R A_phi [Wb] (NOT the per-radian psi/2pi), mu0 explicit, raw SI, and
the MAST sign convention psi_axis > psi_boundary (the axis is the flux maximum).
Nothing here imports from ``eval`` or ``worldmodel`` -- the firewalled EFIT
referee only ever *scores* the psi this produces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from imas_ambix.latent.patch_basis import PatchBasis


def moment_terms(order: int) -> list[tuple[int, int]]:
    """The (p, q) monomial powers u^p v^q with p + q <= ``order``.

    ``order=1`` -> {1, u, v} (Ip + current centroid); ``order=2`` adds the
    quadrupoles {u^2, uv, v^2} (elongation / tilt); ``order=3`` adds the
    octupoles that carry triangularity.  Raw monomials span the same low-order
    moment space as the symmetrised {u^2 +/- v^2} combinations but are simpler
    and orthogonalise cleanly under the whitened least-squares fit.
    """
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")
    return [(p, q) for d in range(order + 1) for p in range(d + 1) for q in [d - p]]


def build_moment_basis(
    r_cells: np.ndarray,
    z_cells: np.ndarray,
    candidate_mask: np.ndarray,
    r0: float,
    *,
    order: int = 3,
    z0: float = 0.0,
    scale: float | None = None,
) -> tuple[np.ndarray, list[str], float]:
    """Low-order current-moment basis ``M`` (n_cells, K) on the plasma cells.

    Each column is a monomial u^p v^q of the normalised in-plasma coordinates
    ``u = (R - r0) / a``, ``v = (Z - z0) / a`` (``a`` = ``scale``), restricted to
    the conductor-clear in-limiter candidate cells (zero elsewhere).  A current
    vector ``i_cell = M @ c`` is therefore a smooth low-order distribution whose
    external field is fixed by its moments -- the object the external magnetics
    constrain well.

    Parameters
    ----------
    r_cells, z_cells : (n,) cell centre coordinates [m].
    candidate_mask : (n,) 1.0 on conductor-clear in-limiter cells, else 0.0.
    r0 : radial centre for the normalised coordinate [m] (machine / plasma R0).
    order : max total monomial degree (>= 1).
    z0 : vertical centre for the normalised coordinate [m].
    scale : normalising length ``a`` [m]; defaults to the RMS candidate-cell
        radius about (r0, z0), a geometry-derived scale (no per-shot tuning).

    Returns
    -------
    M : (n, K) basis, candidate-masked.
    labels : human-readable ``u^p v^q`` column labels.
    scale : the length scale actually used [m].
    """
    r = np.asarray(r_cells, dtype=np.float64)
    z = np.asarray(z_cells, dtype=np.float64)
    keep = np.asarray(candidate_mask, dtype=np.float64) > 0.0
    if scale is None:
        if keep.any():
            du = r[keep] - r0
            dv = z[keep] - z0
            scale = float(np.sqrt(np.mean(du * du + dv * dv)))
        else:  # pragma: no cover - degenerate geometry
            scale = 1.0
    scale = max(float(scale), 1e-9)

    u = (r - r0) / scale
    v = (z - z0) / scale
    terms = moment_terms(order)
    cols = []
    labels = []
    for p, q in terms:
        if p == 0 and q == 0:
            col = keep.astype(np.float64)  # monopole == candidate mask
        else:
            col = (u**p) * (v**q)
            # zero-sum over candidate cells: subtract the mean so every
            # higher moment carries NO net current -- Ip lands purely on the
            # monopole.  This does not change the span (the subtracted piece is
            # a multiple of the monopole) but decouples the total-current
            # (Rogowski Ip) constraint onto a single coefficient.
            if keep.any():
                col = col - col[keep].mean()
            col = np.where(keep, col, 0.0)
        cols.append(col)
        labels.append(_term_label(p, q))
    M = np.stack(cols, axis=1) if cols else np.zeros((r.size, 0))
    return M, labels, scale


def _term_label(p: int, q: int) -> str:
    if p == 0 and q == 0:
        return "1"
    parts = []
    if p:
        parts.append("u" if p == 1 else f"u^{p}")
    if q:
        parts.append("v" if q == 1 else f"v^{q}")
    return "".join(parts)


@dataclass
class MomentFitConfig:
    """Configuration for the current-moment boundary fit."""

    order: int = 3  # max monomial degree of the moment basis
    ip_anchor: bool = True  # hard-pin the total current to the Rogowski Ip
    ridge: float = (
        1e-8  # Tikhonov floor on the (whitened, column-normalised) normal eqns
    )
    z0: float = 0.0  # vertical centre of the normalised coordinate [m]
    scale: float | None = None  # override the geometry-derived length scale [m]


@dataclass
class MomentInversion:
    """One slice's current-moment fit: currents + moment diagnostics."""

    i_cell: np.ndarray  # (n_cells,) fitted patch currents [A]
    coeffs: np.ndarray  # (K,) moment amplitudes [A] (per normalised monomial)
    labels: list[str]
    misfit: float  # whitened mean-square sensor residual (trusted rows)
    ip_fit: float  # total fitted current Sum(i_cell) [A]
    ip_rel_err: float  # |ip_fit - Rogowski Ip| / |Ip|
    centroid_r: float  # current centroid R = Sum(i r) / Sum(i) [m]
    centroid_z: float  # current centroid Z [m]
    scale: float  # normalising length used [m]
    shot: int = 0
    t_index: int = 0
    time_s: float = float("nan")
    coeff_cov: np.ndarray | None = field(default=None, repr=False)


def _fit_one(
    M: np.ndarray,
    a_sens: np.ndarray,  # (S, K) = m_sens @ M, sensor signature of each moment
    measured: np.ndarray,
    vacuum: np.ndarray,
    mask: np.ndarray,
    scale: np.ndarray,
    ip: float,
    cfg: MomentFitConfig,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Whitened least-squares fit of the K moment amplitudes for one slice.

    The basis is built so the monopole column is the candidate mask and every
    higher moment is zero-sum, hence ``Sum(M @ c) = c[0] * n_candidate``.  With
    ``ip_anchor`` the monopole is pinned HARD to the Rogowski current
    ``c[0] = Ip / n_candidate`` (a trusted absolute measurement; it also removes
    the trivial zero-current solution) and only the K-1 shape coefficients are
    fit by whitened least squares to the Ip-subtracted plasma signature::

        min_{c[1:]}  Sum_trusted [ (vacuum + a_sens[:,0] c0 + a_sens[:,1:] c[1:]
                                    - measured) / sensor_scale ]^2

    Without ``ip_anchor`` all K coefficients are fit freely (ablation).  The
    ridge is a tiny numerical floor applied in a column-normalised frame so it
    does not bias the fit.  Returns (c, misfit, cov) in the raw coefficient
    frame.
    """
    keep = np.asarray(mask, dtype=bool)
    # absent channels carry NaN in ``measured``; they are masked out (w = 0), but
    # NaN * 0 = NaN would still poison the least squares, so zero them first
    # (exactly as the free-current inverse does with np.nan_to_num).
    meas = np.nan_to_num(np.asarray(measured, dtype=np.float64))
    vac = np.nan_to_num(np.asarray(vacuum, dtype=np.float64))
    sc = np.asarray(scale, dtype=np.float64)
    w = np.zeros_like(meas)
    w[keep] = 1.0 / np.maximum(sc[keep], 1e-12)

    b = meas - vac  # plasma sensor signature
    if cfg.ip_anchor:
        n_cand = float(M[:, 0].sum())  # candidate-cell count (monopole == mask)
        c0 = float(ip) / max(n_cand, 1.0)
        b = b - a_sens[:, 0] * c0  # subtract the pinned monopole signature
        a_fit = a_sens[:, 1:]  # fit only the shape moments
    else:
        c0 = None
        a_fit = a_sens

    aw = a_fit * w[:, None]
    bw = b * w
    col_norm = np.linalg.norm(aw, axis=0)
    col_norm = np.where(col_norm > 0, col_norm, 1.0)
    a_n = aw / col_norm[None, :]
    n_k = a_n.shape[1]
    gram = a_n.T @ a_n + cfg.ridge * np.eye(n_k)
    rhs = a_n.T @ bw
    c_fit = np.linalg.solve(gram, rhs) / col_norm if n_k else np.zeros(0)

    c = np.concatenate([[c0], c_fit]) if cfg.ip_anchor else c_fit

    resid = (vac + a_sens @ c - meas) * w
    n_keep = int(keep.sum())
    misfit = float((resid[keep] ** 2).sum() / max(n_keep, 1))

    try:
        cov_n = np.linalg.pinv(gram)
        cov = cov_n / np.outer(col_norm, col_norm)
    except np.linalg.LinAlgError:  # pragma: no cover
        cov = np.full((n_k, n_k), np.nan)
    return c, misfit, cov


def fit_moment_currents(
    basis: PatchBasis,
    payload,
    cfg: MomentFitConfig | None = None,
    *,
    r0: float | None = None,
) -> MomentInversion:
    """Fit the current-moment amplitudes for a single slice payload.

    ``payload`` is a :class:`imas_ambix.latent.patch_inverse.SlicePayload`
    (fields ``measured``, ``vacuum``, ``mask``, ``scale``, ``i_pf``,
    ``ip_amperes``).  Returns the fitted per-cell currents plus moment
    diagnostics; assemble psi with ``basis.psi_grid_2d_np(inv.i_cell, i_pf)``.
    """
    cfg = cfg or MomentFitConfig()
    r_cells = np.asarray(basis.r_cells.detach().cpu().numpy(), dtype=np.float64)
    z_cells = np.asarray(basis.z_cells.detach().cpu().numpy(), dtype=np.float64)
    cand = np.asarray(basis.candidate_mask.detach().cpu().numpy(), dtype=np.float64)
    m_sens = np.asarray(basis.m_sens.detach().cpu().numpy(), dtype=np.float64)
    r0v = float(basis.r0) if r0 is None else float(r0)

    M, labels, scale = build_moment_basis(
        r_cells, z_cells, cand, r0v, order=cfg.order, z0=cfg.z0, scale=cfg.scale
    )
    a_sens = m_sens @ M  # (S, K)

    c, misfit, cov = _fit_one(
        M,
        a_sens,
        payload.measured,
        payload.vacuum,
        payload.mask,
        payload.scale,
        float(payload.ip_amperes),
        cfg,
    )
    i_cell = M @ c
    ip_fit = float(i_cell.sum())
    ip_ref = float(payload.ip_amperes)
    denom = ip_fit if abs(ip_fit) > 1e-12 else 1.0
    return MomentInversion(
        i_cell=i_cell,
        coeffs=c,
        labels=labels,
        misfit=misfit,
        ip_fit=ip_fit,
        ip_rel_err=float(abs(ip_fit - ip_ref) / max(abs(ip_ref), 1.0)),
        centroid_r=float((i_cell * r_cells).sum() / denom),
        centroid_z=float((i_cell * z_cells).sum() / denom),
        scale=scale,
        shot=getattr(payload, "shot", 0),
        t_index=getattr(payload, "t_index", 0),
        time_s=getattr(payload, "time_s", float("nan")),
        coeff_cov=cov,
    )


def invert_slices_moment(
    basis: PatchBasis,
    payloads: list,
    cfg: MomentFitConfig | None = None,
    *,
    r0: float | None = None,
) -> list[MomentInversion]:
    """Fit the current-moment representation for a batch of slices.

    Mirrors :func:`imas_ambix.latent.patch_inverse.invert_slices` in signature
    so a gate script can swap the free-current inverse for the constrained
    moment read.  Each slice is an independent linear least-squares solve (no
    optimiser, no seed, deterministic), so this is far cheaper than the Adam
    inverse and returns one :class:`MomentInversion` per payload in order.
    """
    cfg = cfg or MomentFitConfig()
    return [fit_moment_currents(basis, p, cfg, r0=r0) for p in payloads]


__all__ = [
    "MomentFitConfig",
    "MomentInversion",
    "build_moment_basis",
    "fit_moment_currents",
    "invert_slices_moment",
    "moment_terms",
]
