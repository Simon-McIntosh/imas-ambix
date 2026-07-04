"""The assembled GS-grounded latent engine + composite raw-signal objective.

This wires the stage-2 primitives into one model, on the **patch-current**
force-balance substrate (:mod:`imas_ambix.latent.patch_basis`) that replaced
the low-DOF polynomial-θ ψ carrier:

* :class:`~imas_ambix.latent.encoder.HybridLatentEncoder` — absolute features →
  hybrid latent (anchored ⊕ free ⊕ patch-current shape ``x``);
* :class:`~imas_ambix.latent.patch_basis.PatchBasis` — the EFIT-free finite-area
  Green's-function forward substrate (a per-campaign fixed-geometry matmul: no
  grid solve, Ampère + div B = 0 hold identically) that is the engine's
  **spatial anchor**: patch currents → sensor magnetics + the ψ field;
* :mod:`imas_ambix.latent.structure_residual` — the remaining GS physics
  content in this basis (the flux-function structure ``jφ = a(ψ)R + b(ψ)/R``),
  a profile-free, topology-free residual that is zero for any force-balanced
  current and O(1) for a structureless one;
* :class:`~imas_ambix.latent.transport.FluxDiffusionPrior` — the flux-diffusion
  **temporal anchor** on ∂ψ/∂t (D≥0, command-driven, sign guard-rails), fed the
  patch-current midplane ψ(ρ) profile via
  :mod:`imas_ambix.latent.patch_transport`.

and defines the training objective: raw-signal self-supervision through the
physics-grounded bottleneck —

    L = w_mag  · ‖(B̂ − B_raw)/σ‖²         (masked whitened misfit vs RAW magnetics)
      + w_ip   · ((ΣI − Ip)/Ip)²           (Rogowski anchor, kills zero-current)
      + w_fb   · λ · R_structure           (discrepancy-weighted GS structure residual)
      + w_anc  · anchored-block regression (Ip, coils, n_e — raw scalars)
      + w_diss · dissipation≥0 + w_vs · Volt-second  (temporal guard-rails)
      + w_dim  · dimensionless regulariser + w_kl · KL(free-bits)
      + w_clo  · closure-coordinate self-supervised readout (optional, off if
                 the encoder carries no closure head — see
                 :meth:`GSGroundedLatentEngine.closure_readout_loss`)

The per-example structure-residual weight ``λ`` follows the locked
bounded-discrepancy policy (:class:`~imas_ambix.latent.patch_encoder.DiscrepancyLambda`,
λ0=3, target-ratio 1.5, λ≤100): a training script owns one scheduler per
campaign and passes ``batch["structure_lam"] = schedule.get(example_ids)``; a
constant ``weights.structure_lam0`` fallback lets a batch without that
bookkeeping (e.g. a unit test) still exercise the term.

Topology (axis / X-point / LCFS / public-private) is read from the solved ψ
(:mod:`imas_ambix.latent.topology`), never trained — the firewalled EFIT
referee only scores it.  The magnetics misfit is a *falsifiable* prior: a
persistent residual is a discovery signal (where force balance breaks), not an
error to fit away.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from imas_ambix.latent.encoder import HybridLatent, HybridLatentEncoder  # noqa: TC001
from imas_ambix.latent.patch_basis import PatchBasis  # noqa: TC001
from imas_ambix.latent.patch_transport import patch_psi_profile, transport_prior_terms
from imas_ambix.latent.structure_residual import fit_flux_functions, structure_residual
from imas_ambix.latent.topology import TopologyReadout, read_topology
from imas_ambix.latent.transport import FluxDiffusionPrior  # noqa: TC001


@dataclass
class LossWeights:
    """Composite-loss weights.  Priors are soft — guard-rails, not rails."""

    magnetics: float = 1.0
    ip_anchor: float = 10.0
    structure_residual: float = 1.0
    structure_lam0: float = 3.0  # fallback per-example λ absent a batch schedule
    anchored: float = 1.0
    dissipation: float = 0.1
    volt_second: float = 0.1
    dimensionless: float = 1e-3
    kl: float = 1e-3
    free_bits: float = 0.1
    closure: float = 0.05  # modest by design — an auxiliary readout, ablatable


@dataclass
class _Grid:
    r_1d: np.ndarray
    z_1d: np.ndarray
    iz_mid: int = field(default=0)


class GSGroundedLatentEngine(nn.Module):
    """Encoder + patch-current spatial anchor + flux-diffusion temporal anchor."""

    def __init__(
        self,
        encoder: HybridLatentEncoder,
        basis: PatchBasis,
        transport: FluxDiffusionPrior,
        *,
        weights: LossWeights | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.basis = basis
        self.transport = transport
        self.weights = weights or LossWeights()
        r_1d = self.basis.grid_r.detach().cpu().numpy()
        z_1d = self.basis.grid_z.detach().cpu().numpy()
        self._grid = _Grid(r_1d=r_1d, z_1d=z_1d, iz_mid=int(np.argmin(np.abs(z_1d))))

    # ---- forward maps ----

    def encode(self, x: torch.Tensor) -> HybridLatent:
        return self.encoder(x)

    def i_cell_from_latent(
        self, latent: HybridLatent, ip: torch.Tensor
    ) -> torch.Tensor:
        """Latent → per-cell currents ``(B, n_cells)`` [A].

        ``I = x · Ip / n_cells · candidate_mask`` — the same convention the
        amortised patch encoder and the variational inverse use (see
        :mod:`imas_ambix.latent.patch_encoder`), so one head output serves any
        plasma current and the conductor-clear candidate mask (factual
        geometry) zeroes forbidden cells.
        """
        if latent.i_cell_x is None:
            raise ValueError(
                "latent carries no patch-current head output — build the "
                "encoder's LatentConfig with n_cells = the campaign's cell count"
            )
        x = latent.i_cell_x
        n = x.shape[-1]
        mask = self.basis.candidate_mask.to(dtype=x.dtype, device=x.device)
        ip = ip.to(dtype=x.dtype, device=x.device)
        return x * (ip[:, None] / n) * mask[None, :]

    def predict_magnetics(
        self, i_cell: torch.Tensor, i_pf: torch.Tensor
    ) -> torch.Tensor:
        """Predicted sensor magnetics from patch currents ``(B, S)``."""
        return self.basis.sensors(i_cell, i_pf)

    def psi_field_2d(self, i_cell: torch.Tensor, i_pf: torch.Tensor) -> torch.Tensor:
        return self.basis.psi_grid_2d(i_cell, i_pf)

    def psi_profile(
        self, i_cell: torch.Tensor, i_pf: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Midplane flux profile ψ(R; Z≈0) ``(B, nr)`` + its ρ coordinate ``(1, nr)``.

        Delegates to :func:`~imas_ambix.latent.patch_transport.patch_psi_profile`
        — the natural home of current diffusion: the poloidal flux along the
        outboard midplane, fed to the transport prior as ψ(ρ).
        """
        return patch_psi_profile(self.basis, i_cell, i_pf)

    # ---- losses ----

    def magnetics_loss(
        self,
        i_cell: torch.Tensor,
        i_pf: torch.Tensor,
        raw_mag: torch.Tensor,
        sensor_scale: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Whitened masked misfit ‖(B̂ − B_raw)/σ‖² — the spatial anchor.

        ``mask`` (``(B, S)`` bool) restricts the residual to sensors the shot
        actually measures (an operator predicts every campaign channel, but a
        given shot may not carry all of them) — unmeasured / NaN sensors
        contribute nothing.
        """
        pred = self.predict_magnetics(i_cell, i_pf)
        raw = torch.nan_to_num(raw_mag, nan=0.0)
        resid = (pred - raw) / sensor_scale.clamp_min(1e-12)
        if mask is None:
            return (resid**2).mean()
        m = mask.to(resid.dtype)
        return (resid**2 * m).sum() / m.sum().clamp_min(1.0)

    def structure_residual_loss(
        self, i_cell: torch.Tensor, i_pf: torch.Tensor, lam: torch.Tensor
    ) -> torch.Tensor:
        """Discrepancy-weighted profile-free GS structure residual (per example).

        Uses the shipped default recipe (``form="affine-r2"``,
        ``connectivity="locality"`` — see ``scripts/patch_gate_eval.py``).
        """
        psi_c = self.basis.psi_cells(i_cell, i_pf)  # (B, n) [Wb]
        jphi_c = i_cell / self.basis.cell_area  # (B, n) [A/m^2]
        r_c = self.basis.r_cells.to(dtype=i_cell.dtype, device=i_cell.device)
        z_c = self.basis.z_cells.to(dtype=i_cell.dtype, device=i_cell.device)
        rows = [
            structure_residual(
                psi_c[k], r_c, jphi_c[k], z_c=z_c, connectivity="locality"
            )
            for k in range(psi_c.shape[0])
        ]
        fb = torch.stack(rows)
        return (lam.to(fb.dtype) * fb).mean()

    def closure_readout_loss(
        self, latent: HybridLatent, i_cell: torch.Tensor, i_pf: torch.Tensor
    ) -> torch.Tensor:
        """Self-supervised aux loss for the closure-coordinate head (§B3).

        Per example, fits the flux-function coefficients
        (:func:`~imas_ambix.latent.structure_residual.fit_flux_functions`, same
        ``connectivity="locality"`` recipe as :meth:`structure_residual_loss`)
        from the engine's OWN predicted currents and matches the head's
        ``(a_k, b_k)`` readout to that fit — the fit is DETACHED, so this trains
        the head to read closures straight out of the latent, never the other
        way round.  Bins with negligible current-weighted mass (vacuum bins,
        see :func:`~imas_ambix.latent.structure_residual.integrate_closures`'s
        ``mass_frac_threshold`` convention) carry no closure information and are
        excluded.  Zero (no gradient) when the encoder carries no closure head.

        The residual is normalised by the fit's OWN per-bin standard error
        (a chi-squared-style ``((pred − target)/σ)²``), not the raw physical
        units: ``a_k``/``b_k`` span many orders of magnitude across the corpus
        (SI flux-function coefficients), so an unnormalised L2 term would
        dwarf every other loss and dominate the shared trunk's gradient
        purely from unit choice.  A convergent head reads ~1 here on average
        — directly the "within 1σ" criterion the closure gate reports.
        """
        if latent.closure is None:
            return i_cell.new_zeros(())
        n_bins = int(latent.closure.shape[1])
        psi_c = self.basis.psi_cells(i_cell, i_pf)
        jphi_c = i_cell / self.basis.cell_area
        r_c = self.basis.r_cells.to(dtype=i_cell.dtype, device=i_cell.device)
        z_c = self.basis.z_cells.to(dtype=i_cell.dtype, device=i_cell.device)
        per_example = []
        for k in range(psi_c.shape[0]):
            fit = fit_flux_functions(
                psi_c[k],
                r_c,
                jphi_c[k],
                n_bins=n_bins,
                z_c=z_c,
                connectivity="locality",
            )
            mass = fit.weight_mass
            keep = (mass > 1e-3 * mass.max().clamp_min(1e-30)).to(fit.a_k.dtype)
            denom = keep.sum().clamp_min(1.0)
            pred_a, pred_b = latent.closure[k, :, 0], latent.closure[k, :, 1]
            a_floor = 1e-3 * fit.a_k.abs().amax().clamp_min(1.0)
            b_floor = 1e-3 * fit.b_k.abs().amax().clamp_min(1.0)
            per_bin = ((pred_a - fit.a_k) / fit.a_err.clamp_min(a_floor)) ** 2 + (
                (pred_b - fit.b_k) / fit.b_err.clamp_min(b_floor)
            ) ** 2
            per_example.append((per_bin * keep).sum() / denom)
        return torch.stack(per_example).mean()

    def losses(self, batch: dict) -> dict[str, torch.Tensor]:
        """Composite raw-signal objective for a consecutive (t, t+1) window.

        ``batch`` keys: ``x_t``, ``x_tp1`` (features); ``i_pf_t``, ``i_pf_tp1``
        (KNOWN coil currents); ``ip_t`` (Rogowski plasma current [A], the
        patch-current normalisation anchor — ``ip_tp1`` optional, defaults to
        ``ip_t``); ``raw_mag_t``, ``sensor_scale`` (magnetics misfit);
        ``mag_mask`` (optional sensor validity mask); ``structure_lam``
        (optional per-example bounded-discrepancy weight — falls back to
        ``weights.structure_lam0``); ``cmd_t`` (command); ``anchored_target`` /
        ``anchored_mask`` (raw scalars); ``dt`` (timestep).  Returns each term +
        the weighted ``total`` and the ``diffusivity_min`` diagnostic (verifies
        D≥0).
        """
        w = self.weights
        lat_t = self.encode(batch["x_t"])
        lat_tp1 = self.encode(batch["x_tp1"])

        ip_t = batch["ip_t"]
        ip_tp1 = batch.get("ip_tp1", ip_t)
        i_cell_t = self.i_cell_from_latent(lat_t, ip_t)
        i_cell_tp1 = self.i_cell_from_latent(lat_tp1, ip_tp1)

        magnetics = self.magnetics_loss(
            i_cell_t,
            batch["i_pf_t"],
            batch["raw_mag_t"],
            batch["sensor_scale"],
            batch.get("mag_mask"),
        )
        ip_pen = ((i_cell_t.sum(-1) - ip_t) / ip_t) ** 2
        ip_anchor = ip_pen.mean()

        lam = batch.get("structure_lam")
        if lam is None:
            lam = i_cell_t.new_full((i_cell_t.shape[0],), w.structure_lam0)
        structure = self.structure_residual_loss(i_cell_t, batch["i_pf_t"], lam)

        anchored = self.encoder.anchored_loss(
            lat_t, batch["anchored_target"], batch.get("anchored_mask")
        )
        dim_reg = self.encoder.dimensionless_regulariser(lat_t)
        kl = self.encoder.kl_free_bits(lat_t, w.free_bits)
        closure = self.closure_readout_loss(lat_t, i_cell_t, batch["i_pf_t"])

        priors = transport_prior_terms(
            self.transport,
            self.basis,
            i_cell_t,
            i_cell_tp1,
            batch["i_pf_t"],
            batch["i_pf_tp1"],
            dt=float(batch["dt"]),
            feat=lat_t.free,
            cmd=batch["cmd_t"],
        )

        total = (
            w.magnetics * magnetics
            + w.ip_anchor * ip_anchor
            + w.structure_residual * structure
            + w.anchored * anchored
            + w.dissipation * priors["dissipation"]
            + w.volt_second * priors["volt_second"]
            + w.dimensionless * dim_reg
            + w.kl * kl
            + w.closure * closure
        )
        return {
            "magnetics": magnetics,
            "ip_anchor": ip_anchor,
            "structure_residual": structure,
            "anchored": anchored,
            "dissipation": priors["dissipation"],
            "volt_second": priors["volt_second"],
            "dimensionless": dim_reg,
            "kl": kl,
            "closure": closure,
            "diffusivity_min": priors["diffusivity_min"],
            "total": total,
        }

    # ---- topology read (numpy, downstream of the differentiable anchors) ----

    @torch.no_grad()
    def read_topology(
        self,
        i_cell: torch.Tensor,
        i_pf: torch.Tensor,
        *,
        limiter_r: np.ndarray | None = None,
        limiter_z: np.ndarray | None = None,
    ) -> list[TopologyReadout]:
        """Deterministic topology read of the solved ψ, per batch sample.

        Uses the fp64 numpy assembly (:meth:`PatchBasis.psi_grid_2d_np`) — the
        precision the axis-placement read needs — one slice at a time.  Returns
        one :class:`~imas_ambix.latent.topology.TopologyReadout` per sample —
        the oracle-shaped 14-D geometry (axis / X-point set / LCFS) the
        firewalled referee scores.
        """
        ic = i_cell.detach().cpu().numpy()
        ip = i_pf.detach().cpu().numpy()
        out: list[TopologyReadout] = []
        for b in range(ic.shape[0]):
            psi2d = self.basis.psi_grid_2d_np(ic[b], ip[b])
            out.append(
                read_topology(
                    psi2d,
                    self._grid.r_1d,
                    self._grid.z_1d,
                    limiter_r=limiter_r,
                    limiter_z=limiter_z,
                )
            )
        return out


__all__ = ["GSGroundedLatentEngine", "LossWeights"]
