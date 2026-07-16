"""Differentiable profile→field decode through the exact Green's substrate.

Turns low-DOF profile-coefficient corrections about a converged classical
free-boundary solution into cell currents, sensor predictions, and ψ(R,Z),
entirely through the precomputed finite-area Green's matrices of
:class:`~imas_ambix.latent.patch_basis.PatchBasis` (thick-cylinder kernels —
never point-filament).  The physics layer is load-bearing: a network wired
through this decode can only redistribute current within the profile-DOF span
the classical solve itself uses, and its output becomes a field exclusively
through the exact operators — no freehand field emission is representable.

The normalised-flux map ψ_N(R, Z) is a FIXED input (the classical solve's own
converged map), so the decode is a linearisation about the classical solution:
exactly linear in the corrections up to the non-negativity clamp (the
unidirectional-current fact the classical solve also imposes) and the Ip
renormalisation (the Rogowski anchor).  Gradients through the Green's layer
are therefore analytic, and the zero-correction decode reproduces the
classical cell currents exactly.

Conventions match :mod:`imas_ambix.latent.gs_solve`: total poloidal flux
Φ = 2πR·A_φ [Wb], positive-Ip normalisation (callers pass |Ip|), MAST
psi_axis > psi_boundary.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from imas_ambix.latent.patch_basis import PatchBasis

#: exponent ladder of the non-negative monomial basis — MUST mirror
#: :func:`imas_ambix.latent.gs_solve.profile_basis` (pinned by test).
NONNEG_EXPONENTS = (0.5, 1.0, 1.5, 2.0, 3.0)


def _legendre_p(x: torch.Tensor, k: int) -> torch.Tensor:
    """Legendre polynomial P_k(x) by the three-term recurrence."""
    if k == 0:
        return torch.ones_like(x)
    p_prev = torch.ones_like(x)
    p = x
    for n in range(1, k):
        p_prev, p = p, ((2.0 * n + 1.0) * x * p - n * p_prev) / (n + 1.0)
    return p


def profile_basis_torch(
    psi_n: torch.Tensor,
    r: torch.Tensor,
    *,
    r0: float,
    n_p: int,
    n_f: int,
    kind: str = "monomial-nonneg",
) -> torch.Tensor:
    """Torch mirror of :func:`imas_ambix.latent.gs_solve.profile_basis`.

    ``psi_n`` and ``r`` broadcast to a common shape ``(...,)``; returns
    ``(..., n_p + n_f)`` jφ basis images, zero at and beyond the boundary.
    Semantics are pinned against the numpy implementation by test — the two
    must agree to fp64 round-off for both basis kinds.
    """
    psi_n, r = torch.broadcast_tensors(psi_n, r)
    inside = psi_n < 1.0
    rr = torch.clamp(r, min=1e-3)
    clipped = torch.clamp(psi_n, 0.0, 1.0)
    x = 2.0 * clipped - 1.0
    edge = torch.where(inside, 1.0 - clipped, torch.zeros_like(psi_n))
    cols: list[torch.Tensor] = []
    for drive, n_k in ((rr / r0, n_p), (r0 / rr, n_f)):
        for k in range(n_k):
            if kind == "monomial-nonneg":
                phi = edge ** NONNEG_EXPONENTS[k]
            elif kind == "legendre":
                phi = _legendre_p(x, k) * edge
            else:
                raise ValueError(f"unknown basis kind {kind!r}")
            col = drive * phi
            cols.append(torch.where(inside, col, torch.zeros_like(col)))
    if not cols:
        return psi_n.new_zeros((*psi_n.shape, 0))
    return torch.stack(cols, dim=-1)


class ProfileGreensDecoder(nn.Module):
    """Profile-DOF corrections → cell currents → sensors + ψ, differentiably.

    Wraps a per-campaign :class:`PatchBasis`.  The decode contract:

    * ``profile_columns`` evaluates the jφ basis at the (fixed) per-slice
      ψ_N map on the plasma cells and rescales every column to carry ``ip``
      of gross current — so corrections ``dc`` are dimensionless O(≲1)
      mixture weights directly comparable to the classical ladder
      coefficients across slices.
    * ``cell_currents`` applies the additive DOF-space correction about the
      classical cell currents, clamps to the unidirectional-current sign,
      and renormalises to the Rogowski ``ip`` (exact, differentiable).
      ``dc = 0`` returns the classical currents exactly.
    * ``sensors`` / ``psi_grid_2d`` are the exact Green's layer
      (:class:`PatchBasis` matmuls) — the only route from currents to fields.

    Shapes: ``B`` batch, ``n`` cells, ``K = n_p + n_f`` profile DOF.
    """

    def __init__(
        self,
        basis: PatchBasis,
        *,
        n_p: int = 3,
        n_f: int = 3,
        kind: str = "monomial-nonneg",
    ) -> None:
        super().__init__()
        self.basis = basis
        self.n_p = int(n_p)
        self.n_f = int(n_f)
        self.kind = str(kind)

    @property
    def n_dof(self) -> int:
        return self.n_p + self.n_f

    def profile_columns(
        self, psi_n_cells: torch.Tensor, ip: torch.Tensor
    ) -> torch.Tensor:
        """Ip-normalised basis columns ``(B, n, K)`` at the fixed ψ_N map.

        ``psi_n_cells`` is ``(B, n)`` (or ``(n,)``); ``ip`` is ``(B,)`` (or a
        scalar).  Every column is rescaled so its gross |current| equals
        ``ip`` (degenerate all-zero columns are left at zero, never inflated).
        """
        if psi_n_cells.dim() == 1:
            psi_n_cells = psi_n_cells.unsqueeze(0)
        ip = torch.as_tensor(
            ip, dtype=psi_n_cells.dtype, device=psi_n_cells.device
        ).reshape(-1)
        r = self.basis.r_cells.to(dtype=psi_n_cells.dtype).unsqueeze(0)
        cols = profile_basis_torch(
            psi_n_cells,
            r,
            r0=self.basis.r0,
            n_p=self.n_p,
            n_f=self.n_f,
            kind=self.kind,
        )  # (B, n, K)
        gross = cols.abs().sum(dim=1)  # (B, K)
        scale = torch.where(
            gross > 0.0, ip.unsqueeze(-1) / gross.clamp(min=1e-30), gross
        )
        return cols * scale.unsqueeze(1)

    @staticmethod
    def cell_currents(
        i_cell0: torch.Tensor,
        dc: torch.Tensor,
        columns: torch.Tensor,
        ip: torch.Tensor,
    ) -> torch.Tensor:
        """``relu(i_cell0 + columns @ dc)`` renormalised to ``ip`` — (B, n).

        ``i_cell0`` (B, n) are the classical solve's converged per-cell
        currents [A]; ``dc`` (B, K) the profile-DOF corrections; ``columns``
        from :meth:`profile_columns`.  The clamp keeps jφ unidirectional (the
        same fact the classical non-negative ladder imposes) and the rescale
        keeps the Rogowski Ip anchor exact.
        """
        if i_cell0.dim() == 1:
            i_cell0 = i_cell0.unsqueeze(0)
        ip = torch.as_tensor(ip, dtype=i_cell0.dtype, device=i_cell0.device).reshape(-1)
        u = torch.relu(i_cell0 + torch.einsum("bnk,bk->bn", columns, dc))
        total = u.sum(dim=-1).clamp(min=1e-30)
        return u * (ip / total).unsqueeze(-1)

    def sensors(self, i_cell: torch.Tensor, i_pf=None) -> torch.Tensor:
        """Exact Green's-layer sensor prediction ``(B, S)`` [Wb / T]."""
        return self.basis.sensors(i_cell, i_pf)

    def psi_grid_2d(self, i_cell: torch.Tensor, i_pf=None) -> torch.Tensor:
        """Exact Green's-layer total flux ``(B, nz, nr)`` [Wb]."""
        return self.basis.psi_grid_2d(i_cell, i_pf)

    def decode(
        self,
        i_cell0: torch.Tensor,
        dc: torch.Tensor,
        psi_n_cells: torch.Tensor,
        ip: torch.Tensor,
        i_pf: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Full decode: corrections → ``{"i_cell", "sensors"}``."""
        columns = self.profile_columns(psi_n_cells, ip)
        i_cell = self.cell_currents(i_cell0, dc, columns, ip)
        return {"i_cell": i_cell, "sensors": self.sensors(i_cell, i_pf)}


__all__ = [
    "NONNEG_EXPONENTS",
    "ProfileGreensDecoder",
    "profile_basis_torch",
]
