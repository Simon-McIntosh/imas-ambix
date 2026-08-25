"""The hybrid latent encoder (anchored ⊕ free ⊕ ψ), in dimensionless coordinates.

The shared latent has two parts:

* an **anchored block** supervised *only* on genuinely raw measured scalars
  (plasma current Ip from the Rogowski coil, coil currents, line-averaged
  density) — the physics handle that gives the command a clean lever and
  transfer a dimensionless anchor;
* a **free block** for the closure physics the equations cannot pin (edge
  turbulence, fast transients, the pedestal) — where the data-led model earns
  its keep; it feeds the transport prior's learned η∥ / sources.

It also carries the **plasma-current profile amplitudes θ** — the polynomial ψ
representation the (legacy) GS observation operator reads out to predicted
magnetics and to the flux field — and, for the patch-current carrier
(:class:`~imas_ambix.latent.patch_basis.PatchBasis`), a **patch-current head**
emitting a dimensionless per-cell shape ``x`` (``HybridLatent.i_cell_x``,
``(B, n_cells)``).  The head itself knows nothing about amperes: the convention
``I = x · Ip / n_cells`` (Ip = the measured Rogowski anchor) is applied
downstream by the engine, which also owns the campaign's candidate mask.
Boundary / X-point / axis are *never* encoder outputs; they are a deterministic
read of the solved ψ (:mod:`imas_ambix.latent.topology`).

Dimensionless coordinates — resolved LEARNED-SOFT (the open
``dimensionless-coordinate-set`` decision).  The encoder *learns* the map to
the latent and a soft standardisation regulariser
(:meth:`HybridLatentEncoder.dimensionless_regulariser`) nudges the anchored
block toward zero-mean / unit-variance dimensionless coordinates, rather than
hard-imposing a fixed ρ*, β, ν* basis (which would over-constrain and churn on
transfer).  This is the flexible choice the North Star extrapolation-coordinates
direction recommends.

A second realisation of that same idea is the **closure-coordinate head**
(``LatentConfig.n_closure_bins``): a linear readout of per-ψ-bin flux-function
coefficients ``(a_k = p′, b_k = FF′/μ₀)`` — the closures
:mod:`imas_ambix.latent.structure_residual` recovers from a force-balanced
current.  It is trained self-supervised, entirely inside the firewall
(:meth:`~imas_ambix.latent.engine.GSGroundedLatentEngine.closure_readout_loss`
matches it to the detached fit of the engine's OWN predicted currents — no
EFIT anywhere): the head learns to *read the closures straight out of the
latent* rather than requiring a downstream fit of the reconstructed current
every time, which is what makes ``(a_k, b_k)`` a genuine dimensionless latent
coordinate rather than a derived diagnostic.
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
    profile_head: bool = False  # emit (β0, α) — the amortized GS profile fit
    n_cells: int = 0  # patch-current head width (0 = disabled, no PatchBasis carrier)
    n_closure_bins: int = 0  # closure-coordinate head width (0 = disabled)


@dataclass
class HybridLatent:
    """The encoded hybrid latent for one batch of time-slices."""

    theta: torch.Tensor  # (B, K) plasma-current profile amplitudes (ψ rep)
    anchored: torch.Tensor  # (B, n_anchored) raw-supervised dimensionless scalars
    free: torch.Tensor  # (B, n_free) free closure block
    free_logvar: torch.Tensor | None  # (B, n_free) belief log-variance, or None
    profile: torch.Tensor | None = None  # (B, 2) amortized (β0, α), bounded
    i_cell_x: torch.Tensor | None = None  # (B, n_cells) dimensionless patch shape
    closure: torch.Tensor | None = None  # (B, n_closure_bins, 2) per-bin (a_k, b_k)


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
        self.profile_head = nn.Linear(d, 2) if config.profile_head else None
        self.patch_head = nn.Linear(d, config.n_cells) if config.n_cells else None
        self.closure_head = (
            nn.Linear(d, 2 * config.n_closure_bins) if config.n_closure_bins else None
        )

    # bounds of the amortized profile parameters: β0 ∈ (0, 1) by the ansatz's
    # pressure/FF′ split; α ∈ (0.5, 3) brackets the fit grid's peakedness range
    ALPHA_MIN = 0.5
    ALPHA_SPAN = 2.5

    def forward(self, x: torch.Tensor) -> HybridLatent:
        h = self.backbone(x)
        logvar = None
        if self.free_logvar_head is not None:
            logvar = self.free_logvar_head(h)
        profile = None
        if self.profile_head is not None:
            raw = self.profile_head(h)
            beta0 = torch.sigmoid(raw[:, :1])
            alpha = self.ALPHA_MIN + self.ALPHA_SPAN * torch.sigmoid(raw[:, 1:])
            profile = torch.cat([beta0, alpha], dim=-1)
        i_cell_x = self.patch_head(h) if self.patch_head is not None else None
        closure = None
        if self.closure_head is not None:
            # h.detach(): the closure head is a READ-ONLY probe on the shared
            # backbone, never a shaper of it. Without this, closure_readout_loss's
            # gradient into the SHARED h would still reach patch_head's output
            # i_cell on the NEXT forward pass (via the backbone weight update),
            # even though the loss never touches patch_head's own parameters --
            # exactly the indirect "closure loss reshapes the currents" coupling
            # so a diverging closure term cannot drag i_cell far beyond its
            # expected scale.
            closure = self.closure_head(h.detach()).view(
                -1, self.config.n_closure_bins, 2
            )
        return HybridLatent(
            theta=self.theta_head(h),
            anchored=self.anchored_head(h),
            free=self.free_head(h),
            free_logvar=logvar,
            profile=profile,
            i_cell_x=i_cell_x,
            closure=closure,
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

    def profile_loss(
        self,
        latent: HybridLatent,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Masked smooth-L1 regression of the profile head to precomputed fits.

        ``target`` : ``(B, 2)`` per-slice (β0, α) from the corpus fit-target
        precompute (label-free — derived purely from raw magnetics + measured
        Ip through the free-boundary solve).  ``mask`` : ``(B,)`` bool — True
        where a low-cost converged fit exists for the slice.
        """
        if latent.profile is None:
            return target.new_zeros(())
        per = nn.functional.smooth_l1_loss(
            latent.profile, target, reduction="none"
        ).mean(dim=-1)
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

        Keeps the stochastic free block informative without collapse.
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
