"""Topology-aware training penalties for the learned equilibrium operator.

The self-supervised sensor objective has degenerate directions: corrections
that improve whitened sensor misfit while drifting the boundary and wiggling
the saddles.  These penalties collapse that degeneracy by injecting the
topology of a valid boundary — it terminates at a limiter tangency OR an
X-point — into the TRAINING SIGNAL, without ever supervising the boundary
class (the emitted class stays emergent in the push-out reader).

Every term is a **delta form about the classical spine solution**: it acts on
the CHANGE the correction induces (``δψ = G_pg·δi_cell + G_grid·δa_eddy``)
and is identically zero at zero correction.  The decode is a frozen-ψ_N
linearisation, so an absolute topological condition is not representable —
the delta form keeps the approximation honest and leaks no spine-emulation
pressure at ``dc = 0``.

Two penalties:

* **Terminator-consistency anchor** (spine-trusted slices only).  At the
  spine's own terminator candidates — X-point locations and the
  limiter-contact point — penalise (a) the induced gradient change
  ``∇δψ`` (full at an X-point, tangential-only at the limiter contact:
  the tangency condition constrains just the along-wall derivative), and
  (b) DIFFERENTIAL flux changes across candidates, weighted by the spine's
  soft terminator selection (MAST convention ψ_axis > ψ_boundary, so the
  binding candidate is the flux-largest — a softmax picks it smoothly).
  Uniform flux shifts are free (a gauge); differential ones can re-order
  which terminator binds and are penalised.

* **Critical-point integrity regulariser** (unconditional, label-free).
  A spurious interior O-point or saddle requires ``|∇ψ| → 0`` somewhere the
  spine field had a healthy gradient.  Penalise gradient EROSION below the
  spine's own (capped) margin: ``relu(min(|∇ψ_spine|, m·s) − |∇ψ|)²``.
  Near legitimate nulls (axis, X-points) the spine margin is already ~0, so
  the constraint fades out gracefully with no exclusion zones and no labels.

Cost: the anchor needs ψ and ∇ψ at 2–3 points per slice — interpolation
rows over the precomputed Green's matrices (built once at sequence-build
time); the integrity term needs δψ on the grid, one extra Green's matmul.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

#: fixed candidate slots per slice: two X-point targets + the limiter contact
MAX_TERMINATOR_CANDIDATES = 3

#: relative softmax temperature for the terminator flux selection (fraction
#: of the axis-to-boundary flux span)
SOFTPICK_REL_TEMP = 0.05

#: cap on the integrity gradient margin, as a fraction of the slice's median
#: gradient — the penalty never demands more than this floor
INTEGRITY_MARGIN_REL = 0.3


# ---------------------------------------------------------------------------
# pointwise interpolation rows over a flattened (nz, nr) grid field
# ---------------------------------------------------------------------------
def point_rows(
    grid_r: np.ndarray, grid_z: np.ndarray, points: np.ndarray
) -> np.ndarray:
    """(n_pts, 3, G) rows giving ``(ψ, ∂ψ/∂R, ∂ψ/∂Z)`` at interior points.

    Value = bilinear interpolation; gradients = bilinear interpolation of the
    central-difference gradient field (exact wherever the gradient field is
    piecewise linear — in particular exact for quadratic ψ).  Points are
    clamped one cell inside the grid so the FD stencil stays interior.
    Grids must be uniform (they are: :class:`EquilibriumGrid` linspaces).
    """
    grid_r = np.asarray(grid_r, dtype=np.float64)
    grid_z = np.asarray(grid_z, dtype=np.float64)
    nr, nz = grid_r.size, grid_z.size
    dr = float(grid_r[1] - grid_r[0])
    dz = float(grid_z[1] - grid_z[0])
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    rows = np.zeros((pts.shape[0], 3, nz * nr), dtype=np.float64)
    for q, (r, z) in enumerate(pts):
        # cell indices with a one-cell interior margin for the FD stencil
        ir = int(np.clip(np.floor((r - grid_r[0]) / dr), 1, nr - 3))
        iz = int(np.clip(np.floor((z - grid_z[0]) / dz), 1, nz - 3))
        fr = np.clip((r - grid_r[ir]) / dr, 0.0, 1.0)
        fz = np.clip((z - grid_z[iz]) / dz, 0.0, 1.0)
        corners = (
            (iz, ir, (1 - fz) * (1 - fr)),
            (iz, ir + 1, (1 - fz) * fr),
            (iz + 1, ir, fz * (1 - fr)),
            (iz + 1, ir + 1, fz * fr),
        )
        for cz, cr, w in corners:
            rows[q, 0, cz * nr + cr] += w
            # central FD of the grid field at the corner node, then bilinear
            rows[q, 1, cz * nr + (cr + 1)] += w / (2.0 * dr)
            rows[q, 1, cz * nr + (cr - 1)] -= w / (2.0 * dr)
            rows[q, 2, (cz + 1) * nr + cr] += w / (2.0 * dz)
            rows[q, 2, (cz - 1) * nr + cr] -= w / (2.0 * dz)
    return rows


def _bilinear(psi_2d: np.ndarray, grid_r, grid_z, r: float, z: float) -> float:
    """Bilinear ψ at one point (numpy, no rows needed)."""
    nr, nz = len(grid_r), len(grid_z)
    dr = float(grid_r[1] - grid_r[0])
    dz = float(grid_z[1] - grid_z[0])
    ir = int(np.clip(np.floor((r - grid_r[0]) / dr), 0, nr - 2))
    iz = int(np.clip(np.floor((z - grid_z[0]) / dz), 0, nz - 2))
    fr = np.clip((r - grid_r[ir]) / dr, 0.0, 1.0)
    fz = np.clip((z - grid_z[iz]) / dz, 0.0, 1.0)
    return float(
        (1 - fz) * (1 - fr) * psi_2d[iz, ir]
        + (1 - fz) * fr * psi_2d[iz, ir + 1]
        + fz * (1 - fr) * psi_2d[iz + 1, ir]
        + fz * fr * psi_2d[iz + 1, ir + 1]
    )


def median_gradient_scale(
    psi_2d: np.ndarray,
    grid_r: np.ndarray,
    grid_z: np.ndarray,
    region_2d: np.ndarray | None = None,
) -> float:
    """Median ``|∇ψ|`` [Wb/m] over the (interior of the) region — the
    per-slice normalisation for both penalties."""
    dr = float(grid_r[1] - grid_r[0])
    dz = float(grid_z[1] - grid_z[0])
    gz, gr = np.gradient(np.asarray(psi_2d, dtype=np.float64), dz, dr)
    gmag = np.hypot(gr, gz)
    if region_2d is not None:
        vals = gmag[np.asarray(region_2d, dtype=bool)]
        if vals.size == 0:
            vals = gmag.ravel()
    else:
        vals = gmag.ravel()
    return float(max(np.median(vals), 1e-30))


# ---------------------------------------------------------------------------
# terminator-consistency anchor — per-slice build (numpy) + batched penalty
# ---------------------------------------------------------------------------
@dataclass
class SliceAnchor:
    """Precomputed anchor payload for one spine-trusted slice.

    ``rows_cell`` (Q, 3, n_cells) and ``rows_mode`` (Q, 3, k) map the
    correction (cell-current change / eddy amplitudes) to ``(δψ, δψ_R, δψ_Z)``
    at each terminator candidate; invalid candidate slots carry zero rows.
    ``proj`` (Q, 2, 2) projects the gradient change (identity at X-points,
    tangent outer product at the limiter contact).  ``w_flux`` (Q,) are the
    spine's fixed softmax terminator weights.
    """

    rows_cell: np.ndarray
    rows_mode: np.ndarray
    proj: np.ndarray
    w_flux: np.ndarray
    cand_mask: np.ndarray
    grad_scale: float
    flux_scale: float


def build_slice_anchor(
    psi_2d: np.ndarray,
    grid_r: np.ndarray,
    grid_z: np.ndarray,
    target: np.ndarray,
    limiter_r: np.ndarray,
    limiter_z: np.ndarray,
    g_pg: np.ndarray,
    g_grid: np.ndarray,
    *,
    grad_scale: float,
    softpick_rel_temp: float = SOFTPICK_REL_TEMP,
) -> SliceAnchor | None:
    """Assemble the terminator anchor for one slice from the spine's own read.

    Candidates: the spine target's X-point slots (``target[2:4]``,
    ``target[4:6]``; NaN = absent) with identity projectors, plus the
    limiter-contact point — the limiter vertex whose spine flux is closest to
    the axis flux (MAST convention: flux decreases outward, so that is the
    max-flux vertex) — with a tangential projector.  Returns ``None`` when
    the slice has no finite axis (nothing to trust).
    """
    psi_2d = np.asarray(psi_2d, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    axis_r, axis_z = target[0], target[1]
    if not (np.isfinite(axis_r) and np.isfinite(axis_z)):
        return None
    psi_axis = _bilinear(psi_2d, grid_r, grid_z, axis_r, axis_z)

    r_lo, r_hi = grid_r[1], grid_r[-2]
    z_lo, z_hi = grid_z[1], grid_z[-2]

    points: list[tuple[float, float]] = []
    projs: list[np.ndarray] = []
    for sl in (target[2:4], target[4:6]):
        if np.all(np.isfinite(sl)) and r_lo <= sl[0] <= r_hi and z_lo <= sl[1] <= z_hi:
            points.append((float(sl[0]), float(sl[1])))
            projs.append(np.eye(2))

    lim_r = np.asarray(limiter_r, dtype=np.float64)
    lim_z = np.asarray(limiter_z, dtype=np.float64)
    if lim_r.size >= 3:
        lim_psi = np.array(
            [
                _bilinear(psi_2d, grid_r, grid_z, lr, lz)
                for lr, lz in zip(lim_r, lim_z, strict=True)
            ]
        )
        j = int(np.argmax(lim_psi))  # flux closest to the axis flux
        pr, pz = float(lim_r[j]), float(lim_z[j])
        if r_lo <= pr <= r_hi and z_lo <= pz <= z_hi:
            j_prev, j_next = (j - 1) % lim_r.size, (j + 1) % lim_r.size
            t_vec = np.array(
                [lim_r[j_next] - lim_r[j_prev], lim_z[j_next] - lim_z[j_prev]]
            )
            norm = np.linalg.norm(t_vec)
            if norm > 1e-12:
                t_hat = t_vec / norm
                points.append((pr, pz))
                projs.append(np.outer(t_hat, t_hat))

    if not points:
        return None

    q_used = min(len(points), MAX_TERMINATOR_CANDIDATES)
    points, projs = points[:q_used], projs[:q_used]
    cand_flux = np.array([_bilinear(psi_2d, grid_r, grid_z, r, z) for r, z in points])
    psi_b = float(cand_flux.max())
    flux_scale = max(abs(psi_axis - psi_b), 1e-6 * max(abs(psi_axis), 1.0))
    temp = max(softpick_rel_temp * flux_scale, 1e-30)
    w = np.exp((cand_flux - psi_b) / temp)
    w = w / w.sum()

    rows = point_rows(grid_r, grid_z, np.asarray(points))  # (q, 3, G)
    n_cells = g_pg.shape[1]
    k = g_grid.shape[1]
    q_max = MAX_TERMINATOR_CANDIDATES
    rows_cell = np.zeros((q_max, 3, n_cells))
    rows_mode = np.zeros((q_max, 3, k))
    proj = np.zeros((q_max, 2, 2))
    w_flux = np.zeros(q_max)
    cand_mask = np.zeros(q_max, dtype=bool)
    rows_cell[:q_used] = rows @ g_pg
    rows_mode[:q_used] = rows @ g_grid
    proj[:q_used] = np.stack(projs)
    w_flux[:q_used] = w
    cand_mask[:q_used] = True
    return SliceAnchor(
        rows_cell=rows_cell,
        rows_mode=rows_mode,
        proj=proj,
        w_flux=w_flux,
        cand_mask=cand_mask,
        grad_scale=float(max(grad_scale, 1e-30)),
        flux_scale=float(flux_scale),
    )


def terminator_penalty(
    di: torch.Tensor,  # (N, n_cells) cell-current change
    da: torch.Tensor,  # (N, k) eddy mode amplitudes
    rows_cell: torch.Tensor,  # (N, Q, 3, n_cells)
    rows_mode: torch.Tensor,  # (N, Q, 3, k)
    proj: torch.Tensor,  # (N, Q, 2, 2)
    w_flux: torch.Tensor,  # (N, Q) — zero at invalid slots, sums to 1 else
    cand_mask: torch.Tensor,  # (N, Q) bool
    grad_scale: torch.Tensor,  # (N,) [Wb/m]
    flux_scale: torch.Tensor,  # (N,) [Wb]
) -> torch.Tensor:
    """Per-step terminator-consistency penalty (dimensionless), (N,).

    Steps without an anchor carry all-zero rows/weights and clamped scales,
    so their penalty is exactly zero — no NaN can enter through a masked
    step (the padded-batch contract: never divide by an un-clamped zero).
    """
    dvals = torch.einsum("nqcm,nm->nqc", rows_cell, di) + torch.einsum(
        "nqck,nk->nqc", rows_mode, da
    )
    g = dvals[..., 1:]
    gs2 = grad_scale.clamp(min=1e-30).pow(2).unsqueeze(-1)
    pen_grad = torch.einsum("nqi,nqij,nqj->nq", g, proj, g) / gs2
    valid = cand_mask.to(pen_grad.dtype)
    pen_grad = (pen_grad * valid).sum(dim=-1) / valid.sum(dim=-1).clamp(min=1.0)

    dflux = dvals[..., 0]
    mean = (w_flux * dflux).sum(dim=-1, keepdim=True)
    fs2 = flux_scale.clamp(min=1e-30).pow(2)
    pen_flux = (w_flux * (dflux - mean) ** 2).sum(dim=-1) / fs2
    return pen_grad + pen_flux


# ---------------------------------------------------------------------------
# critical-point integrity regulariser — batched torch
# ---------------------------------------------------------------------------
def integrity_penalty(
    dpsi_grid: torch.Tensor,  # (N, G) correction flux on the grid
    psi_spine_grid: torch.Tensor,  # (N, G) spine flux (fixed)
    region: torch.Tensor,  # (G,) bool — in-limiter, conductor-clear
    s_med: torch.Tensor,  # (N,) median |∇ψ_spine| over the region
    *,
    nz: int,
    nr: int,
    dr: float,
    dz: float,
    margin_rel: float = INTEGRITY_MARGIN_REL,
) -> torch.Tensor:
    """Mean gradient-erosion penalty over the region interior, (N,).

    Zero exactly at ``dpsi = 0``: the margin is ``min(|∇ψ_spine|, m·s_med)``
    so the spine field itself always satisfies it, including at its own
    legitimate nulls (axis, X-points) where the margin degrades to ~0.
    """
    n = dpsi_grid.shape[0]
    psi = (psi_spine_grid + dpsi_grid).reshape(n, nz, nr)
    psi0 = psi_spine_grid.reshape(n, nz, nr)

    def grad_mag(p: torch.Tensor) -> torch.Tensor:
        g_r = (p[:, 1:-1, 2:] - p[:, 1:-1, :-2]) / (2.0 * dr)
        g_z = (p[:, 2:, 1:-1] - p[:, :-2, 1:-1]) / (2.0 * dz)
        return torch.sqrt(g_r**2 + g_z**2 + 1e-30)

    gmag = grad_mag(psi)
    gmag0 = grad_mag(psi0)
    s = s_med.clamp(min=1e-30).reshape(n, 1, 1)
    m0 = torch.minimum(gmag0, margin_rel * s)
    pen = torch.relu((m0 - gmag) / s) ** 2
    region_i = region.reshape(nz, nr)[1:-1, 1:-1]
    pen = torch.where(region_i.unsqueeze(0), pen, torch.zeros_like(pen))
    denom = region_i.to(pen.dtype).sum().clamp(min=1.0)
    return pen.sum(dim=(-2, -1)) / denom


__all__ = [
    "INTEGRITY_MARGIN_REL",
    "MAX_TERMINATOR_CANDIDATES",
    "SOFTPICK_REL_TEMP",
    "SliceAnchor",
    "build_slice_anchor",
    "integrity_penalty",
    "median_gradient_scale",
    "point_rows",
    "terminator_penalty",
]
