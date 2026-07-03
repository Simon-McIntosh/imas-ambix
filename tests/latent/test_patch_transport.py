"""Tests for the flux-diffusion transport prior on a patch-current ψ profile.

The glue maps batched patch currents + KNOWN coil currents to the midplane ψ(ρ)
profile the representation-agnostic
:class:`~imas_ambix.latent.transport.FluxDiffusionPrior` consumes, matching the
GS-grounded engine's convention exactly.  Correctness is pinned on synthetic
geometry — no MAST data, no EFIT:

* shapes / dtype, and the profile is differentiable back to the patch currents;
* the profile IS the midplane row of ``basis.psi_grid_2d`` (pins the convention);
* the two-time-slice wrapper returns finite guard-rail terms and preserves the
  D≥0 (strictly positive diffusivity) guarantee;
* the module is firewall-clean by construction (static check).
"""

from __future__ import annotations

import numpy as np
import torch

from imas_ambix.latent.patch_basis import PatchBasis
from imas_ambix.latent.patch_transport import (
    patch_psi_profile,
    transport_prior_terms,
)
from imas_ambix.latent.transport import FluxDiffusionPrior
from tests.latent.test_patch_basis import _confining_table


def _basis(dtype: torch.dtype = torch.float64) -> PatchBasis:
    return PatchBasis.from_table(
        _confining_table(), nr=41, nz=57, cache_dir=None, dtype=dtype
    )


def _currents(basis: PatchBasis, batch: int, *, seed: int, dtype: torch.dtype):
    n = int(basis.r_cells.shape[0])
    n_coil = int(basis.psi_coil_grid.shape[1])
    rng = np.random.default_rng(seed)
    i_cell = torch.as_tensor(rng.standard_normal((batch, n)) * 1e4, dtype=dtype)
    i_pf = torch.as_tensor(rng.standard_normal((batch, n_coil)) * 1e4, dtype=dtype)
    return i_cell, i_pf


def test_profile_shapes_dtype_and_differentiable():
    """ψ(ρ) has the right shape / dtype and carries gradient to the currents."""
    basis = _basis()
    i_cell, i_pf = _currents(basis, 3, seed=0, dtype=torch.float64)
    i_cell.requires_grad_(True)

    prof, rho = patch_psi_profile(basis, i_cell, i_pf)
    assert prof.shape == (3, basis.nr)
    assert rho.shape == (1, basis.nr)
    assert prof.dtype == torch.float64
    assert rho.dtype == prof.dtype

    prof.sum().backward()
    grad = i_cell.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert float(grad.abs().max()) > 0.0


def test_profile_is_midplane_row_of_psi_grid_2d():
    """The profile equals the Z-nearest-0 row of ``basis.psi_grid_2d`` exactly.

    Trivially true by construction — asserted to pin the midplane convention
    (matching the GS-grounded engine's ``psi_profile``).
    """
    basis = _basis()
    i_cell, i_pf = _currents(basis, 4, seed=1, dtype=torch.float64)

    prof, rho = patch_psi_profile(basis, i_cell, i_pf)
    psi2d = basis.psi_grid_2d(i_cell, i_pf)
    iz_mid = int(torch.argmin(basis.grid_z.abs()))
    torch.testing.assert_close(prof, psi2d[:, iz_mid, :], rtol=0.0, atol=0.0)
    torch.testing.assert_close(rho[0], basis.grid_r.to(prof.dtype), rtol=0.0, atol=0.0)


def test_transport_terms_finite_and_diffusivity_positive():
    """The two-time-slice wrapper returns finite terms and keeps D > 0."""
    basis = _basis()
    i_cell_t, i_pf_t = _currents(basis, 5, seed=2, dtype=torch.float64)
    i_cell_tp1, i_pf_tp1 = _currents(basis, 5, seed=3, dtype=torch.float64)

    feat_dim, cmd_dim = 6, 3
    prior = FluxDiffusionPrior(
        nrho=basis.nr, cmd_dim=cmd_dim, feat_dim=feat_dim
    ).double()
    rng = np.random.default_rng(4)
    feat = torch.as_tensor(rng.standard_normal((5, feat_dim)), dtype=torch.float64)
    cmd = torch.as_tensor(rng.standard_normal((5, cmd_dim)), dtype=torch.float64)

    terms = transport_prior_terms(
        prior,
        basis,
        i_cell_t,
        i_cell_tp1,
        i_pf_t,
        i_pf_tp1,
        dt=1e-3,
        feat=feat,
        cmd=cmd,
    )
    assert set(terms) == {"dissipation", "volt_second", "diffusivity_min"}
    for name, val in terms.items():
        assert torch.isfinite(val).all(), f"{name} is not finite"
    # D≥0 by construction (softplus + floor) — strictly positive.
    assert float(terms["diffusivity_min"]) >= 0.0
    assert float(terms["diffusivity_min"]) >= prior.diffusivity_floor
    # dissipation is a non-negative guard-rail penalty.
    assert float(terms["dissipation"].detach()) >= 0.0


def test_firewall_static_no_evaluator_imports():
    """The glue module must not touch the EFIT/evaluator side."""
    from pathlib import Path

    import imas_ambix.latent.patch_transport as m

    src = Path(m.__file__).read_text()
    for banned in ("efit_referee", "equilibrium_labels", "worldmodel"):
        assert banned not in src, f"patch_transport imports the firewalled {banned}"
