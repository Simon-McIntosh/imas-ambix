"""The hybrid latent encoder (anchored ⊕ free ⊕ ψ), in dimensionless coordinates.

The shared latent has two parts (§6):

* an **anchored block** supervised *only* on genuinely raw measured scalars
  (plasma current Ip from the Rogowski coil, coil currents, line-averaged
  density) — the physics handle that gives the command a clean lever and
  transfer a dimensionless anchor;
* a **free block** for the closure physics the equations cannot pin (edge
  turbulence, fast transients, the pedestal) — where the data-led model earns
  its keep; it feeds the transport prior's learned η∥ / sources.

It also carries the **plasma-current profile amplitudes θ** — the ψ
representation the GS observation operator reads out to predicted magnetics and
to the flux field.  Boundary / X-point / axis are *never* encoder outputs; they
are a deterministic read of the solved ψ (:mod:`imas_ambix.latent.topology`).

Dimensionless coordinates — resolved LEARNED-SOFT (the open
``dimensionless-coordinate-set`` decision).  The encoder *learns* the map to
the latent and a soft standardisation regulariser
(:meth:`HybridLatentEncoder.dimensionless_regulariser`) nudges the anchored
block toward zero-mean / unit-variance dimensionless coordinates, rather than
hard-imposing a fixed ρ*, β, ν* basis (which would over-constrain and churn on
transfer).  This is the flexible choice the North Star extrapolation-coordinates
direction recommends.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class LatentConfig:
    """Shape + capacity of the hybrid latent encoder."""

    n_features: int  # input feature dim (absolute-calibrated, geometry-tagged)
    n_theta: int  # plasma-current profile DOF K (the ψ representation)
    n_anchored: int  # raw-supervised scalars (Ip, coil currents, n_e, ...)
    n_free: int  # free closure-physics block dim
    hidden: int = 128
    depth: int = 3
    probabilistic: bool = False  # emit a belief (mean+logvar) over the free block
    dropout: float = 0.0


@dataclass
class HybridLatent:
    """The encoded hybrid latent for one batch of time-slices."""

    theta: torch.Tensor  # (B, K) plasma-current profile amplitudes (ψ rep)
    anchored: torch.Tensor  # (B, n_anchored) raw-supervised dimensionless scalars
    free: torch.Tensor  # (B, n_free) free closure block
    free_logvar: torch.Tensor | None  # (B, n_free) belief log-variance, or None


class HybridLatentEncoder(nn.Module):
    """Absolute-calibrated features → hybrid latent (anchored ⊕ free ⊕ θ)."""

    def __init__(self, config: LatentConfig) -> None:
        super().__init__()
        self.config = config
        layers: list[nn.Module] = []
        d = config.n_features
        for _ in range(config.depth):
            layers += [nn.Linear(d, config.hidden), nn.SiLU()]
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            d = config.hidden
        self.backbone = nn.Sequential(*layers)
        self.theta_head = nn.Linear(d, config.n_theta)
        self.anchored_head = nn.Linear(d, config.n_anchored)
        self.free_head = nn.Linear(d, config.n_free)
        self.free_logvar_head = (
            nn.Linear(d, config.n_free) if config.probabilistic else None
        )

    def forward(self, x: torch.Tensor) -> HybridLatent:
        h = self.backbone(x)
        logvar = None
        if self.free_logvar_head is not None:
            logvar = self.free_logvar_head(h)
        return HybridLatent(
            theta=self.theta_head(h),
            anchored=self.anchored_head(h),
            free=self.free_head(h),
            free_logvar=logvar,
        )

    # ---- losses / regularisers ----

    def anchored_loss(
        self,
        latent: HybridLatent,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Masked smooth-L1 regression of the anchored block to raw scalars.

        ``target`` : ``(B, n_anchored)`` dimensionless raw-scalar targets.
        ``mask``   : ``(B, n_anchored)`` bool — True where the scalar is
        genuinely measured (missing scalars contribute nothing; a fully-masked
        batch yields exactly zero, never NaN).
        """
        pred = latent.anchored
        per = nn.functional.smooth_l1_loss(pred, target, reduction="none")
        if mask is None:
            return per.mean()
        m = mask.to(per.dtype)
        denom = m.sum()
        if denom.item() == 0:
            return per.new_zeros(())
        return (per * m).sum() / denom

    def dimensionless_regulariser(self, latent: HybridLatent) -> torch.Tensor:
        """Soft standardisation of the anchored block (the learned-soft resolution).

        Penalises departure of the anchored block from zero-mean / unit-variance
        across the batch — a *soft* nudge toward dimensionless coordinates that
        prescribes no fixed basis (so it cannot over-constrain).  Zero for a
        genuinely standardised block.
        """
        a = latent.anchored
        if a.shape[0] < 2:
            return a.new_zeros(())
        mean = a.mean(dim=0)
        var = a.var(dim=0, unbiased=False)
        return (mean**2).mean() + ((var - 1.0) ** 2).mean()

    def kl_free_bits(
        self, latent: HybridLatent, free_bits: float = 0.0
    ) -> torch.Tensor:
        """KL(belief ‖ 𝒩(0,1)) on the free block with a free-bits floor.

        Keeps the stochastic free block informative without collapse (§9).
        Returns zero when the encoder is deterministic (no belief emitted).
        """
        if latent.free_logvar is None:
            return latent.free.new_zeros(())
        mu = latent.free
        logvar = latent.free_logvar
        kl = 0.5 * (mu**2 + logvar.exp() - 1.0 - logvar)  # (B, n_free)
        if free_bits > 0:
            kl = torch.clamp(kl, min=free_bits)
        return kl.sum(dim=-1).mean()

    def reparameterise(self, latent: HybridLatent) -> torch.Tensor:
        """Sample the free block from its belief (identity if deterministic)."""
        if latent.free_logvar is None:
            return latent.free
        std = torch.exp(0.5 * latent.free_logvar)
        return latent.free + std * torch.randn_like(std)


__all__ = ["LatentConfig", "HybridLatent", "HybridLatentEncoder"]
