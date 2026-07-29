"""Tests for the flux-diffusion transport prior.

The transport prior is the temporal counterpart of the GS spatial anchor: it
constrains ∂ψ/∂t through a soft, learned flux/current-diffusion operator with
the arrow of time baked in. The properties pinned here make the physical
contract verifiable:

* the learned diffusivity is **strictly positive** (η∥>0 ⇒ D≥0) by construction
  — parabolic, forward-well-posed — for ANY latent input, even adversarial ones;
* the diffusion operator is genuinely **smoothing** (a flux peak decays forward),
  i.e. it is not accidentally anti-diffusive;
* the **command is load-bearing** — zeroing the command source changes the ψ
  evolution (the causal-identifiability test the prior camera work failed);
* the **dissipation-≥0** guard-rail flags a magnetic-energy-increasing resistive
  step and vanishes for a genuinely dissipative one;
* the **Volt-second budget** residual vanishes when Δψ balances the inductive
  supply minus resistive consumption, and is positive otherwise.

Guard-rails, not rails: the priors forbid the unphysical (negative diffusivity,
anti-diffusion, spontaneous energy/flux growth) while prescribing no trajectory.
"""

from __future__ import annotations

import torch

from imas_ambix.latent.transport import FluxDiffusionPrior
from imas_ambix.physics import (
    CurrentDiffusion,
    FluxSurfaceGeometry,
    current_diffusion_from_mapping,
)


def _prior(nrho=32, cmd_dim=2, feat_dim=8):
    torch.manual_seed(0)
    return FluxDiffusionPrior(nrho=nrho, cmd_dim=cmd_dim, feat_dim=feat_dim)


def _physical_solver(nrho: int = 4) -> CurrentDiffusion:
    face = torch.linspace(0.0, 1.0, nrho + 1).numpy()
    cell = 0.5 * (face[:-1] + face[1:])
    return current_diffusion_from_mapping(
        {
            "rho_face": face,
            "rho_cell": cell,
            "psi_face": face.copy(),
            "psi_n_face": face,
            "psi_n_cell": cell,
            "vpr_face": face,
            "vpr_cell": cell,
            "g2_face": 1.0 + face,
            "g3_face": 1.0 + face,
            "g3_cell": 1.0 + cell,
            "f_face": 1.0 + face,
            "f_cell": 1.0 + cell,
            "b2_cell": 1.0 + cell,
            "inv_r_cell": 1.0 + cell,
            "phi_b": 0.8,
            "r0": 0.9,
            "ip_amperes": 5.0e5,
            "axis_psi": 0.0,
            "boundary_psi": 1.0,
            "volume": 8.0,
            "q_face": 1.0 + face,
            "flux_sign": 1.0,
        },
        {"eta0": 8.0e-8, "contrast": 1.5, "shape": 2.0},
    )


def test_prior_carries_nova_physical_transport_context():
    solver = _physical_solver()
    prior = FluxDiffusionPrior(
        nrho=solver.geometry.rho_cell.size,
        cmd_dim=2,
        feat_dim=8,
        current_diffusion=solver,
    )

    assert isinstance(prior.current_diffusion, CurrentDiffusion)
    assert isinstance(prior.physical_geometry, FluxSurfaceGeometry)
    assert prior.physical_geometry is solver.geometry


def test_diffusivity_is_strictly_positive_for_any_input():
    prior = _prior()
    feat = torch.randn(16, 8) * 50.0 - 100.0  # adversarially large & negative
    d = prior.diffusivity(feat)
    assert d.shape == (16, prior.nrho)
    assert (d > 0).all()  # η∥>0 ⇒ D≥0, strictly, by construction


def test_diffusion_operator_smooths_a_flux_peak():
    """Forward parabolic diffusion decays a local ψ peak (∂ψ/∂t < 0 at the peak)."""
    prior = _prior(nrho=64)
    rho = torch.linspace(0.0, 1.0, 64).unsqueeze(0)
    # a Gaussian bump peaked at the centre
    psi = torch.exp(-(((rho - 0.5) / 0.1) ** 2))
    feat = torch.zeros(1, 8)
    cmd = torch.zeros(1, 2)
    dpsi = prior.dpsi_dt(psi, rho, feat, cmd)
    peak = int(torch.argmax(psi[0]))
    assert dpsi[0, peak] < 0  # the peak decays — diffusion, not anti-diffusion


def test_command_source_is_load_bearing():
    """Zeroing the command source must change the ψ evolution."""
    prior = _prior()
    rho = torch.linspace(0.0, 1.0, prior.nrho).unsqueeze(0)
    psi = torch.sin(rho * 3.14159)
    feat = torch.randn(4, 8)
    cmd = torch.randn(4, 2)
    dpsi_cmd = prior.dpsi_dt(psi, rho, feat, cmd)
    dpsi_zero = prior.dpsi_dt(psi, rho, feat, torch.zeros_like(cmd))
    diff = (dpsi_cmd - dpsi_zero).abs().sum()
    assert diff > 0  # the command genuinely drives ∂ψ/∂t (identifiable chain)


def test_dissipation_penalty_flags_energy_increase():
    """The resistive channel must not increase magnetic energy."""
    prior = _prior(nrho=48)
    rho = torch.linspace(0.0, 1.0, 48).unsqueeze(0)
    psi = torch.exp(-(((rho - 0.5) / 0.12) ** 2))
    # a genuinely diffusive (smoothing) resistive rate → penalty ≈ 0
    diffusive = prior.diffusion_operator(psi, rho)  # D[ψ], smooths
    pen_ok = prior.dissipation_penalty(psi, diffusive, dt=1e-3, rho=rho)
    # an ANTI-diffusive rate (sharpening) → magnetic energy grows → penalty > 0
    pen_bad = prior.dissipation_penalty(psi, -diffusive, dt=1e-3, rho=rho)
    assert pen_ok.item() < 1e-6
    assert pen_bad.item() > 0.0


def test_volt_second_budget_vanishes_when_balanced():
    """Δψ = dt·(inductive supply + resistive term) ⇒ zero budget residual."""
    prior = _prior(nrho=40)
    rho = torch.linspace(0.0, 1.0, 40).unsqueeze(0)
    psi_t = torch.sin(rho * 3.0)
    dt = 2e-3
    s_ind = 0.3 * torch.ones_like(psi_t)  # inductive command supply
    res = prior.diffusion_operator(psi_t, rho)  # resistive redistribution
    psi_tp1 = psi_t + dt * (s_ind + res)  # exactly balanced
    pen0 = prior.volt_second_penalty(psi_t, psi_tp1, dt, s_ind, res)
    pen_bad = prior.volt_second_penalty(psi_t, psi_tp1 + 0.5, dt, s_ind, res)
    assert pen0.item() < 1e-8
    assert pen_bad.item() > 0.0


def test_priors_are_differentiable_and_report_d_nonneg():
    prior = _prior()
    rho = torch.linspace(0.0, 1.0, prior.nrho).unsqueeze(0)
    psi_t = torch.randn(3, prior.nrho, requires_grad=True)
    psi_tp1 = torch.randn(3, prior.nrho)
    feat = torch.randn(3, 8)
    cmd = torch.randn(3, 2)
    out = prior.priors(psi_t, psi_tp1, dt=1e-3, rho=rho, feat=feat, cmd=cmd)
    assert "dissipation" in out and "volt_second" in out and "diffusivity_min" in out
    assert out["diffusivity_min"] > 0  # D ≥ 0 verified (strictly positive)
    total = out["dissipation"] + out["volt_second"]
    total.backward()
    assert psi_t.grad is not None and torch.isfinite(psi_t.grad).all()
