"""The temporal anchor: a soft, learned flux-diffusion transport prior.

The GS operator anchors ψ *spatially* (force balance at an instant) but says
nothing about how ψ evolves in time.  The physically-correct temporal
counterpart is **transport**, and transport carries an **arrow of time**: the
plasma's resistive flux evolution is irreversible.  This module encodes that,
mostly through inequality / sign-definite constraints — the safest physics to
inject because they forbid the unphysical without prescribing a trajectory
(guard-rails, not rails).

The soft, learned flux/current-diffusion prior::

    ∂ψ/∂t = D·𝒟[ψ] + S ,   D = η∥/μ₀ > 0 ,   S = non-inductive + command-driven

with

* the parallel resistivity / diffusivity ``D`` a **learned closure**, made
  **strictly positive** (softplus + floor) so ``η∥>0 ⇒ D≥0`` *by construction*
  — a D≥0 diffusion is parabolic: forward-well-posed, backward-ill-posed.  That
  IS the arrow of time, baked into the operator (it cannot run the resistive
  evolution backward or spontaneously sharpen the profile);
* the sources ``S`` learned, with the **command entering through the source
  term** (loop-voltage / current-drive are in the plan), so the plan drives the
  *temporal* evolution of ψ — controllability by construction on the
  identifiable chain (:meth:`FluxDiffusionPrior.dpsi_dt` changes when the
  command is zeroed);
* two soft, falsifiable guard-rails — **non-negative resistive dissipation**
  (monotone magnetic-energy decay, :meth:`dissipation_penalty`) and the
  **Volt-second / flux budget** (:meth:`volt_second_penalty`).  A persistent
  residual is a *discovery signal* (anomalous transport — sawtooth/NTM
  redistribution), never an error to fit away.

Everything is differentiable; the diffusivity floor guarantees D≥0 without a
penalty.  The prior operates on a radial flux profile ψ(ρ) — the natural home
of current diffusion — which the engine exposes from the latent's ψ
representation.
"""

from __future__ import annotations

import torch
from torch import nn

from imas_ambix.physics import CurrentDiffusion, FluxSurfaceGeometry

MU0 = 4.0e-7 * 3.141592653589793
"""Vacuum permeability [T·m/A]."""


class FluxDiffusionPrior(nn.Module):
    """Soft, learned flux/current-diffusion prior on ∂ψ/∂t with sign guard-rails.

    Parameters
    ----------
    nrho:
        Number of radial (ρ) grid points the flux profile ψ(ρ) lives on.
    cmd_dim:
        Command (plan) dimension — the loop-voltage / current-drive channel that
        drives the inductive source term.
    feat_dim:
        Latent free-block feature dimension the closures (η∥, non-inductive
        sources) are learned from.
    hidden:
        Hidden width of the closure MLPs.
    diffusivity_floor:
        Strictly-positive floor added to the softplus diffusivity so ``D>0``
        holds for *any* input (η∥>0 ⇒ D≥0 by construction).
    current_diffusion:
        Optional Nova physical current-diffusion solver for the same interval.
        Its :class:`FluxSurfaceGeometry` is the immutable machine/equilibrium
        metric context; the learned closure remains Ambix-owned.
    """

    def __init__(
        self,
        *,
        nrho: int,
        cmd_dim: int,
        feat_dim: int,
        hidden: int = 64,
        diffusivity_floor: float = 1e-6,
        mu0: float = MU0,
        current_diffusion: CurrentDiffusion | None = None,
    ) -> None:
        super().__init__()
        if current_diffusion is not None and not isinstance(
            current_diffusion, CurrentDiffusion
        ):
            raise TypeError("current_diffusion must be a Nova CurrentDiffusion")
        self.nrho = int(nrho)
        self.cmd_dim = int(cmd_dim)
        self.feat_dim = int(feat_dim)
        self.mu0 = float(mu0)
        self.diffusivity_floor = float(diffusivity_floor)
        self.current_diffusion = current_diffusion

        if (
            current_diffusion is not None
            and current_diffusion.geometry.rho_cell.size != self.nrho
        ):
            raise ValueError(
                "current-diffusion geometry cell count "
                f"{current_diffusion.geometry.rho_cell.size} != nrho {self.nrho}"
            )

        # learned resistivity/diffusivity closure D = η∥/μ₀ (per-ρ, strictly > 0)
        self.eta_net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, nrho),
        )
        # learned NON-inductive source (bootstrap etc.) — feat only, no bias so
        # a zero latent yields a zero non-inductive source (a clean identity).
        self.noninductive = nn.Linear(feat_dim, nrho, bias=False)
        # INDUCTIVE source — the COMMAND channel (loop voltage / current drive);
        # no bias so a zero command yields a zero inductive source (load-bearing).
        self.inductive = nn.Linear(cmd_dim, nrho, bias=False)

    @property
    def physical_geometry(self) -> FluxSurfaceGeometry | None:
        """Nova-owned physical geometry associated with this learned prior."""
        return (
            None if self.current_diffusion is None else self.current_diffusion.geometry
        )

    # ---- learned closures ----

    def diffusivity(self, feat: torch.Tensor) -> torch.Tensor:
        """Learned diffusivity ``D = η∥/μ₀`` ``(B, nrho)`` — strictly positive.

        ``softplus`` maps ℝ → (0, ∞); the floor makes it strictly positive for
        any input, so ``D≥0`` (parabolic, forward-well-posed) holds by
        construction — never a penalty, never violable.
        """
        return nn.functional.softplus(self.eta_net(feat)) + self.diffusivity_floor

    def inductive_source(self, cmd: torch.Tensor) -> torch.Tensor:
        """Command-driven inductive source ``(B, nrho)`` (loop voltage / CD)."""
        return self.inductive(cmd)

    def source(self, feat: torch.Tensor, cmd: torch.Tensor) -> torch.Tensor:
        """Total source ``S = non-inductive(feat) + inductive(command)``, ``(B, nrho)``."""  # noqa: E501
        return self.noninductive(feat) + self.inductive_source(cmd)

    # ---- the flux-diffusion operator ----

    def diffusion_operator(self, psi: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        """Cylindrical flux-diffusion operator ``𝒟[ψ] = ψ'' + ψ'/ρ`` ``(B, nrho)``.

        Second-order finite differences on the (near-uniform) ρ grid with
        reflect (Neumann, ∂ψ/∂ρ=0) end conditions — axis regularity + a
        no-flux outer wall.  ρ is floored by one grid step to keep the
        cylindrical ``ψ'/ρ`` term finite at the magnetic axis.
        """
        rho = rho.to(psi.dtype)
        if rho.dim() == 1:
            rho = rho.unsqueeze(0)
        h = (rho[..., 1:] - rho[..., :-1]).mean()
        # reflect-pad ψ by one on each side → Neumann BCs
        pad = torch.cat([psi[..., 1:2], psi, psi[..., -2:-1]], dim=-1)
        d2 = (pad[..., 2:] - 2.0 * pad[..., 1:-1] + pad[..., :-2]) / (h * h)
        d1 = (pad[..., 2:] - pad[..., :-2]) / (2.0 * h)
        rho_safe = rho.clamp_min(h)
        return d2 + d1 / rho_safe

    def dpsi_dt(
        self,
        psi: torch.Tensor,
        rho: torch.Tensor,
        feat: torch.Tensor,
        cmd: torch.Tensor,
    ) -> torch.Tensor:
        """Modelled ∂ψ/∂t ``= D·𝒟[ψ] + S`` ``(B, nrho)`` (the transport tendency)."""
        return self.diffusivity(feat) * self.diffusion_operator(psi, rho) + self.source(
            feat, cmd
        )

    def resistive_rate(
        self, psi: torch.Tensor, rho: torch.Tensor, feat: torch.Tensor
    ) -> torch.Tensor:
        """The resistive-only tendency ``D·𝒟[ψ]`` ``(B, nrho)`` (no sources)."""
        return self.diffusivity(feat) * self.diffusion_operator(psi, rho)

    # ---- soft guard-rail priors ----

    @staticmethod
    def magnetic_energy(psi: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        """Poloidal magnetic energy proxy ``½∫(∂ψ/∂ρ)² dρ`` ``(B,)``."""
        rho = rho.to(psi.dtype)
        if rho.dim() == 1:
            rho = rho.unsqueeze(0)
        h = (rho[..., 1:] - rho[..., :-1]).mean()
        grad = (psi[..., 1:] - psi[..., :-1]) / h
        return 0.5 * (grad**2).sum(dim=-1) * h

    def dissipation_penalty(
        self,
        psi: torch.Tensor,
        resistive_rate: torch.Tensor,
        dt: float,
        rho: torch.Tensor,
    ) -> torch.Tensor:
        """Non-negative-dissipation guard-rail (monotone magnetic-energy decay).

        The resistive channel may only *lose* magnetic energy (entropy
        production ≥ 0).  Penalise any resistive step that raises the magnetic
        energy: ``mean relu(E(ψ + dt·resistive) − E(ψ))`` — zero for a genuinely
        dissipative (smoothing) step, positive for an anti-diffusive one.
        """
        e0 = self.magnetic_energy(psi, rho)
        e1 = self.magnetic_energy(psi + dt * resistive_rate, rho)
        return torch.relu(e1 - e0).mean()

    def volt_second_penalty(
        self,
        psi_t: torch.Tensor,
        psi_tp1: torch.Tensor,
        dt: float,
        inductive_source: torch.Tensor,
        transport_rate: torch.Tensor,
    ) -> torch.Tensor:
        """Volt-second / flux-budget residual (soft, falsifiable).

        Poloidal flux is supplied inductively (loop voltage / solenoid — in the
        plan) and consumed / redistributed resistively.  The budget identity
        ``Δψ/Δt = inductive supply + transport`` must balance; the squared
        residual is the soft prior (and a *discovery signal* when it persists).
        """
        net = (psi_tp1 - psi_t) / dt
        residual = net - (inductive_source + transport_rate)
        return (residual**2).mean()

    def priors(
        self,
        psi_t: torch.Tensor,
        psi_tp1: torch.Tensor,
        *,
        dt: float,
        rho: torch.Tensor,
        feat: torch.Tensor,
        cmd: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """All temporal-anchor terms for one transition (ψ_t → ψ_{t+1}).

        Returns the two soft penalties (``dissipation``, ``volt_second``) plus
        the ``diffusivity_min`` diagnostic that *verifies* D≥0 (strictly
        positive) — the gate's falsifiable, monitored quantities.
        """
        diff = self.diffusivity(feat)
        resistive = diff * self.diffusion_operator(psi_t, rho)
        s_ind = self.inductive_source(cmd)
        s_non = self.noninductive(feat)
        return {
            "dissipation": self.dissipation_penalty(psi_t, resistive, dt, rho),
            "volt_second": self.volt_second_penalty(
                psi_t, psi_tp1, dt, s_ind, resistive + s_non
            ),
            "diffusivity_min": diff.min().detach(),
        }


__all__ = ["FluxDiffusionPrior", "MU0"]
