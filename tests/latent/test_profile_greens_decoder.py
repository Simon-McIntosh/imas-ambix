"""Gradient and round-trip checks for the differentiable profile→field decode.

The decode is a linearisation about a classical solution, so its gradients
admit closed forms: the sensor map is linear in the cell currents (the exact
Green's matrix) and the current map is linear in the corrections up to the
sign clamp and the Ip renormalisation.  These tests pin

* the torch profile basis against the numpy solver basis (both kinds),
* autograd through the full stack against ``torch.autograd.gradcheck`` (fp64),
* autograd against the hand-derived analytic Jacobian of sensors w.r.t. the
  corrections,
* the exact zero-correction round-trip (``dc = 0`` reproduces the classical
  currents and their sensor prediction bit-for-bit).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from imas_ambix.latent.gs_solve import profile_basis
from imas_ambix.latent.patch_basis import PatchBasis
from imas_ambix.latent.profile_greens_decoder import (
    ProfileGreensDecoder,
    profile_basis_torch,
)

RNG = np.random.default_rng(7)


def _toy_basis(n_cells: int = 24, n_sensor: int = 9, n_coil: int = 3) -> PatchBasis:
    """A tiny synthetic PatchBasis (random matrices, consistent shapes).

    The decoder only consumes ``m_sens`` / ``m_coil`` / ``g_pg`` /
    ``psi_coil_grid`` / ``r_cells`` / ``r0`` — random values exercise the
    gradient paths without a full campaign geometry build.
    """
    nr, nz = 6, 8
    n_grid = nr * nz
    return PatchBasis(
        g_pg=RNG.normal(size=(n_grid, n_cells)),
        g_cc=RNG.normal(size=(n_cells, n_cells)),
        m_sens=RNG.normal(size=(n_sensor, n_cells)),
        m_coil=RNG.normal(size=(n_sensor, n_coil)),
        psi_coil_grid=RNG.normal(size=(n_grid, n_coil)),
        psi_coil_cells=RNG.normal(size=(n_cells, n_coil)),
        r_cells=RNG.uniform(0.3, 1.4, size=n_cells),
        z_cells=RNG.uniform(-1.0, 1.0, size=n_cells),
        candidate_mask=np.ones(n_cells),
        grid_r=np.linspace(0.2, 1.5, nr),
        grid_z=np.linspace(-1.2, 1.2, nz),
        nr=nr,
        nz=nz,
        cell_area=4e-4,
        r0=0.85,
        sensor_channels=[f"ch{i}" for i in range(n_sensor)],
        dtype=torch.float64,
    )


@pytest.mark.parametrize("kind", ["monomial-nonneg", "legendre"])
@pytest.mark.parametrize(("n_p", "n_f"), [(3, 3), (1, 1), (2, 4)])
def test_profile_basis_torch_matches_numpy(kind, n_p, n_f):
    psi_n = RNG.uniform(-0.1, 1.3, size=200)
    r = RNG.uniform(0.2, 1.5, size=200)
    want = profile_basis(psi_n, r, r0=0.85, n_p=n_p, n_f=n_f, kind=kind)
    got = profile_basis_torch(
        torch.tensor(psi_n, dtype=torch.float64),
        torch.tensor(r, dtype=torch.float64),
        r0=0.85,
        n_p=n_p,
        n_f=n_f,
        kind=kind,
    )
    np.testing.assert_allclose(got.numpy(), want, rtol=0.0, atol=1e-13)


def _slice_inputs(basis: PatchBasis, batch: int = 2):
    n = int(basis.r_cells.shape[0])
    n_coil = int(basis.m_coil.shape[1])
    psi_n = torch.tensor(RNG.uniform(0.05, 1.2, size=(batch, n)))
    ip = torch.tensor(RNG.uniform(4e5, 8e5, size=batch))
    i0 = torch.relu(torch.tensor(RNG.normal(size=(batch, n))) + 1.5)
    i0 = i0 * (ip / i0.sum(dim=-1)).unsqueeze(-1)  # classical currents sum to Ip
    i_pf = torch.tensor(RNG.normal(scale=1e3, size=(batch, n_coil)))
    return psi_n, ip, i0, i_pf


def test_zero_correction_round_trip_is_exact():
    basis = _toy_basis()
    dec = ProfileGreensDecoder(basis, n_p=3, n_f=3)
    psi_n, ip, i0, i_pf = _slice_inputs(basis)
    out = dec.decode(i0, torch.zeros((2, dec.n_dof), dtype=i0.dtype), psi_n, ip, i_pf)
    torch.testing.assert_close(out["i_cell"], i0, rtol=0.0, atol=1e-9)
    want = i0 @ basis.m_sens.T + i_pf @ basis.m_coil.T
    torch.testing.assert_close(out["sensors"], want, rtol=1e-12, atol=1e-12)


def test_gradcheck_through_full_stack():
    basis = _toy_basis(n_cells=12, n_sensor=5, n_coil=2)
    dec = ProfileGreensDecoder(basis, n_p=2, n_f=2)
    psi_n, ip, i0, i_pf = _slice_inputs(basis, batch=1)
    columns = dec.profile_columns(psi_n, ip)
    # keep the relu strictly in its linear branch: corrections small vs i0
    dc0 = 1e-3 * torch.tensor(RNG.normal(size=(1, dec.n_dof)), requires_grad=True)

    def sensors_of(dc: torch.Tensor) -> torch.Tensor:
        i_cell = dec.cell_currents(i0, dc, columns, ip)
        return dec.sensors(i_cell, i_pf)

    assert torch.autograd.gradcheck(sensors_of, (dc0,), atol=1e-6, rtol=1e-4)


def test_sensor_jacobian_matches_analytic():
    """d(sensors)/d(dc) against the closed form on the active (u > 0) branch.

    With u = i0 + B·dc all-positive, s(dc) = M·(ip·u/σ) + coil, σ = Σu, so
    J = (ip/σ)·M·(B − u ⊗ colsum(B)/σ) exactly.
    """
    basis = _toy_basis()
    dec = ProfileGreensDecoder(basis, n_p=3, n_f=3)
    psi_n, ip, i0, i_pf = _slice_inputs(basis, batch=1)
    columns = dec.profile_columns(psi_n, ip)
    dc = 1e-4 * torch.tensor(RNG.normal(size=(1, dec.n_dof)))

    def sensors_of(v: torch.Tensor) -> torch.Tensor:
        return dec.sensors(dec.cell_currents(i0, v, columns, ip), i_pf).squeeze(0)

    auto = torch.autograd.functional.jacobian(sensors_of, dc).squeeze(1)  # (S, K)

    b = columns[0]  # (n, K)
    u = i0[0] + b @ dc[0]
    assert bool((u > 0).all()), "test setup must stay on the active branch"
    sigma = u.sum()
    m = basis.m_sens
    inner = b - torch.outer(u, b.sum(dim=0)) / sigma
    analytic = (ip[0] / sigma) * (m @ inner)
    torch.testing.assert_close(auto, analytic, rtol=1e-9, atol=1e-9)


def test_greens_layer_gradient_is_the_exact_matrix():
    """d(sensors)/d(i_cell) IS the Green's matrix — the layer adds nothing."""
    basis = _toy_basis()
    dec = ProfileGreensDecoder(basis)
    n = int(basis.r_cells.shape[0])
    i_cell = torch.tensor(RNG.normal(size=(n,)), requires_grad=True)
    jac = torch.autograd.functional.jacobian(
        lambda ic: dec.sensors(ic).squeeze(0), i_cell
    )
    torch.testing.assert_close(jac, basis.m_sens, rtol=0.0, atol=0.0)


def test_ip_anchor_is_exact_for_any_correction():
    basis = _toy_basis()
    dec = ProfileGreensDecoder(basis, n_p=3, n_f=3)
    psi_n, ip, i0, i_pf = _slice_inputs(basis)
    dc = torch.tensor(RNG.normal(scale=0.3, size=(2, dec.n_dof)))
    i_cell = dec.cell_currents(i0, dc, dec.profile_columns(psi_n, ip), ip)
    torch.testing.assert_close(i_cell.sum(dim=-1), ip, rtol=1e-12, atol=1e-6)
    assert bool((i_cell >= 0).all())
