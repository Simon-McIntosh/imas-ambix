"""Flux-diffusion transport prior evaluated on a patch-current ψ profile.

The engine's temporal anchor —
:class:`~imas_ambix.latent.transport.FluxDiffusionPrior` — is
representation-agnostic: it consumes a radial flux profile ψ(ρ) and says
nothing about how that profile was produced.  The GS-grounded engine feeds it
the midplane slice of the reconstructed GS ψ field
(:meth:`~imas_ambix.latent.engine.GSGroundedLatentEngine.psi_profile`); the
patch-current substrate needs the same glue from a
:class:`~imas_ambix.latent.patch_basis.PatchBasis` ψ.

This module is that glue: a differentiable map from batched patch currents +
KNOWN coil currents to an outboard midplane ψ(ρ) profile, plus the two-time-slice
wrapper that mirrors the engine's transport-loss block.  The convention is
matched to :meth:`GSGroundedLatentEngine.psi_profile` EXACTLY — the midplane row
(Z nearest 0) of the 2-D flux field, with ρ the grid R coordinate — so the prior
sees the same kind of profile regardless of which substrate assembled it.

The v0 profile is the whole midplane row (ρ = grid R, not a flux-surface-mapped
ψ(ρN)); the magnetic axis position varies shot-to-shot, so a proper
outboard-half, axis-anchored ψ(ρN) is the natural follow-up.  Everything here is
a differentiable torch path (``basis.psi_grid_2d``), so the profile and the
transport terms carry gradients back to the patch currents.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from imas_ambix.latent.patch_basis import PatchBasis
    from imas_ambix.latent.transport import FluxDiffusionPrior


def patch_psi_profile(
    basis: PatchBasis,
    i_cell: torch.Tensor,
    i_pf: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Midplane flux profile ψ(R; Z≈0) ``(B, nr)`` + its ρ coordinate ``(1, nr)``.

    Assembles the total poloidal flux on the grid via the differentiable
    :meth:`PatchBasis.psi_grid_2d` forward and takes the midplane row (the Z grid
    line nearest 0), mirroring
    :meth:`~imas_ambix.latent.engine.GSGroundedLatentEngine.psi_profile`: a fixed
    linear slice of the flux field, with ρ the grid R coordinate, fed to the
    transport prior as ψ(ρ).  (A flux-surface-mapped ψ(ρN) is the follow-up.)
    """
    psi2d = basis.psi_grid_2d(i_cell, i_pf)  # (B, nz, nr), row = Z, col = R
    iz_mid = int(torch.argmin(basis.grid_z.abs()))  # Z nearest 0
    prof = psi2d[:, iz_mid, :]  # (B, nr)
    rho = basis.grid_r.to(prof.dtype).unsqueeze(0)  # (1, nr)
    return prof, rho


def transport_prior_terms(
    prior: FluxDiffusionPrior,
    basis: PatchBasis,
    i_cell_t: torch.Tensor,
    i_cell_tp1: torch.Tensor,
    i_pf_t: torch.Tensor | None,
    i_pf_tp1: torch.Tensor | None,
    *,
    dt: float,
    feat: torch.Tensor,
    cmd: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Temporal-anchor terms for one patch-current transition (t → t+1).

    Mirrors the transport block of
    :meth:`~imas_ambix.latent.engine.GSGroundedLatentEngine.losses`: build the
    midplane ψ(ρ) profile at both time slices from the patch currents, then run
    the flux-diffusion prior.  Returns the two soft guard-rail penalties
    (``dissipation``, ``volt_second``) plus the ``diffusivity_min`` diagnostic
    that verifies D≥0 (strictly positive) by construction.
    """
    prof_t, rho = patch_psi_profile(basis, i_cell_t, i_pf_t)
    prof_tp1, _ = patch_psi_profile(basis, i_cell_tp1, i_pf_tp1)
    return prior.priors(
        prof_t,
        prof_tp1,
        dt=float(dt),
        rho=rho,
        feat=feat,
        cmd=cmd,
    )


__all__ = ["patch_psi_profile", "transport_prior_terms"]
