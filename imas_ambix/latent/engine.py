"""The assembled GS-grounded latent engine + composite raw-signal objective.

This wires the three stage-2 primitives into one model:

* :class:`~imas_ambix.latent.encoder.HybridLatentEncoder` — absolute features →
  hybrid latent (anchored ⊕ free ⊕ θ);
* :class:`~imas_ambix.latent.gs_observation.GSObservation` — the EFIT-free GS
  **spatial anchor** (θ → magnetics @ sensors + ψ field);
* :class:`~imas_ambix.latent.transport.FluxDiffusionPrior` — the flux-diffusion
  **temporal anchor** on ∂ψ/∂t (D≥0, command-driven, sign guard-rails).

and defines the training objective (§9): raw-signal self-supervision through the
physics-grounded bottleneck —

    L = w_gs · ‖(B̂ − B_raw)/σ‖²                (GS residual vs RAW magnetics)
      + w_anc · anchored-block regression        (Ip, coils, n_e — raw scalars)
      + w_diss · dissipation≥0 + w_vs · Volt-second   (temporal guard-rails)
      + w_dim · dimensionless regulariser + w_kl · KL(free-bits)

Topology (axis / X-point / LCFS / public-private) is read from the solved ψ
(:mod:`imas_ambix.latent.topology`), never trained — the firewalled EFIT
referee only scores it.  The GS residual is a *falsifiable* prior: a persistent
residual is a discovery signal (where force balance breaks), not an error to fit
away.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from imas_ambix.latent.encoder import HybridLatent, HybridLatentEncoder  # noqa: TC001
from imas_ambix.latent.gs_observation import GSObservation  # noqa: TC001
from imas_ambix.latent.topology import TopologyReadout, read_topology
from imas_ambix.latent.transport import FluxDiffusionPrior  # noqa: TC001


@dataclass
class LossWeights:
    """Composite-loss weights (§9).  Priors are soft — guard-rails, not rails."""

    gs_residual: float = 1.0
    anchored: float = 1.0
    amortization: float = 1.0  # profile head vs precomputed (β0, α) fits
    dissipation: float = 0.1
    volt_second: float = 0.1
    dimensionless: float = 1e-3
    kl: float = 1e-3
    free_bits: float = 0.1


@dataclass
class _Grid:
    r_1d: np.ndarray
    z_1d: np.ndarray
    iz_mid: int = field(default=0)


class GSGroundedLatentEngine(nn.Module):
    """Encoder + GS spatial anchor + flux-diffusion temporal anchor, one model."""

    def __init__(
        self,
        encoder: HybridLatentEncoder,
        gs_observation: GSObservation,
        transport: FluxDiffusionPrior,
        *,
        weights: LossWeights | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.gs = gs_observation
        self.transport = transport
        self.weights = weights or LossWeights()
        r_1d = self.gs.grid_r_1d.detach().cpu().numpy()
        z_1d = self.gs.grid_z_1d.detach().cpu().numpy()
        self._grid = _Grid(r_1d=r_1d, z_1d=z_1d, iz_mid=int(np.argmin(np.abs(z_1d))))

    # ---- forward maps ----

    def encode(self, x: torch.Tensor) -> HybridLatent:
        return self.encoder(x)

    def predict_magnetics(
        self, latent: HybridLatent, i_pf: torch.Tensor
    ) -> torch.Tensor:
        """Predicted sensor magnetics from the latent ψ representation θ."""
        return self.gs(latent.theta, i_pf)

    def psi_field_2d(self, latent: HybridLatent, i_pf: torch.Tensor) -> torch.Tensor:
        return self.gs.psi_field_2d(latent.theta, i_pf)

    def psi_profile(
        self, latent: HybridLatent, i_pf: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Midplane flux profile ψ(R; Z≈0) ``(B, nr)`` + its ρ coordinate ``(1, nr)``.

        The natural home of current diffusion: the poloidal flux along the
        outboard midplane, a fixed linear slice of the reconstructed ψ field, fed
        to the transport prior as ψ(ρ).
        """
        psi2d = self.psi_field_2d(latent, i_pf)  # (B, nz, nr)
        prof = psi2d[:, self._grid.iz_mid, :]  # (B, nr)
        rho = self.gs.grid_r_1d.to(prof.dtype).unsqueeze(0)  # (1, nr)
        return prof, rho

    # ---- losses ----

    def gs_residual_loss(
        self,
        latent: HybridLatent,
        i_pf: torch.Tensor,
        raw_mag: torch.Tensor,
        sensor_scale: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Whitened GS residual ‖(B̂ − B_raw)/σ‖² — the spatial anchor (§8).

        ``mask`` (``(B, S)`` bool) restricts the residual to sensors the shot
        actually measures (an operator predicts every campaign channel, but a
        given shot may not carry all of them) — unmeasured / NaN sensors
        contribute nothing.
        """
        pred = self.predict_magnetics(latent, i_pf)
        raw = torch.nan_to_num(raw_mag, nan=0.0)
        resid = (pred - raw) / sensor_scale.clamp_min(1e-12)
        if mask is None:
            return (resid**2).mean()
        m = mask.to(resid.dtype)
        return (resid**2 * m).sum() / m.sum().clamp_min(1.0)

    def losses(self, batch: dict) -> dict[str, torch.Tensor]:
        """Composite raw-signal objective for a consecutive (t, t+1) window.

        ``batch`` keys: ``x_t``, ``x_tp1`` (features); ``i_pf_t``, ``i_pf_tp1``
        (KNOWN coil currents); ``raw_mag_t``, ``sensor_scale`` (GS residual);
        ``cmd_t`` (command); ``anchored_target``/``anchored_mask`` (raw scalars);
        ``dt`` (timestep).  Returns each term + the weighted ``total`` and the
        ``diffusivity_min`` diagnostic (verifies D≥0).
        """
        w = self.weights
        lat_t = self.encode(batch["x_t"])
        lat_tp1 = self.encode(batch["x_tp1"])

        gs_res = self.gs_residual_loss(
            lat_t,
            batch["i_pf_t"],
            batch["raw_mag_t"],
            batch["sensor_scale"],
            batch.get("mag_mask"),
        )
        anchored = self.encoder.anchored_loss(
            lat_t, batch["anchored_target"], batch.get("anchored_mask")
        )
        profile_target = batch.get("profile_target")
        amortization = (
            self.encoder.profile_loss(lat_t, profile_target, batch.get("profile_mask"))
            if profile_target is not None
            else lat_t.theta.new_zeros(())
        )
        dim_reg = self.encoder.dimensionless_regulariser(lat_t)
        kl = self.encoder.kl_free_bits(lat_t, w.free_bits)

        prof_t, rho = self.psi_profile(lat_t, batch["i_pf_t"])
        prof_tp1, _ = self.psi_profile(lat_tp1, batch["i_pf_tp1"])
        priors = self.transport.priors(
            prof_t,
            prof_tp1,
            dt=float(batch["dt"]),
            rho=rho,
            feat=lat_t.free,
            cmd=batch["cmd_t"],
        )

        total = (
            w.gs_residual * gs_res
            + w.anchored * anchored
            + w.amortization * amortization
            + w.dissipation * priors["dissipation"]
            + w.volt_second * priors["volt_second"]
            + w.dimensionless * dim_reg
            + w.kl * kl
        )
        return {
            "gs_residual": gs_res,
            "anchored": anchored,
            "amortization": amortization,
            "dissipation": priors["dissipation"],
            "volt_second": priors["volt_second"],
            "dimensionless": dim_reg,
            "kl": kl,
            "diffusivity_min": priors["diffusivity_min"],
            "total": total,
        }

    # ---- topology read (numpy, downstream of the differentiable anchors) ----

    @torch.no_grad()
    def read_topology(
        self,
        theta: torch.Tensor,
        i_pf: torch.Tensor,
        *,
        limiter_r: np.ndarray | None = None,
        limiter_z: np.ndarray | None = None,
    ) -> list[TopologyReadout]:
        """Deterministic topology read of the solved ψ, per batch sample.

        Returns one :class:`~imas_ambix.latent.topology.TopologyReadout` per
        sample — the oracle-shaped 14-D geometry (axis / X-point set / LCFS) the
        firewalled referee scores.
        """
        psi2d = self.gs.psi_field_2d(theta, i_pf).detach().cpu().numpy()
        out: list[TopologyReadout] = []
        for b in range(psi2d.shape[0]):
            out.append(
                read_topology(
                    psi2d[b],
                    self._grid.r_1d,
                    self._grid.z_1d,
                    limiter_r=limiter_r,
                    limiter_z=limiter_z,
                )
            )
        return out


__all__ = ["GSGroundedLatentEngine", "LossWeights"]
