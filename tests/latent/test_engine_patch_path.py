"""Load-bearing training-behaviour pins for the patch-current engine carrier.

:mod:`test_engine` covers shapes, the composite loss, D≥0, command
load-bearing-ness and topology.  This file pins the specific claims that
motivated the patch-current carrier swap:

* gradients from the magnetics prediction AND the structure residual reach
  the encoder's patch-current head — both physics paths through the basis
  are genuinely load-bearing on the encoder, not dead ends;
* the Rogowski Ip anchor is strong enough to pull Σi_cell toward Ip on a
  trivial overfit (kills the zero-current minimum the misfit alone permits);
* a single-example overfit drives the whitened magnetics misfit down (the
  same claim as :func:`test_engine.test_engine_grounds_in_raw_magnetics`,
  pinned here through :meth:`GSGroundedLatentEngine.magnetics_loss` directly).
"""

from __future__ import annotations

import numpy as np
import torch

from imas_ambix.latent.encoder import HybridLatentEncoder, LatentConfig
from imas_ambix.latent.engine import GSGroundedLatentEngine
from imas_ambix.latent.patch_basis import PatchBasis
from imas_ambix.latent.transport import FluxDiffusionPrior
from tests.latent.test_patch_basis import _confining_table


def _engine(*, n_free: int = 6, n_anchored: int = 2):
    table = _confining_table()
    basis = PatchBasis.from_table(
        table, nr=33, nz=45, cache_dir=None, dtype=torch.float64
    )
    n = int(basis.r_cells.shape[0])
    n_coil = int(basis.psi_coil_grid.shape[1])
    n_sensor = len(basis.sensor_channels)
    cfg = LatentConfig(
        n_features=n_sensor,
        n_theta=1,
        n_anchored=n_anchored,
        n_free=n_free,
        n_cells=n,
        hidden=48,
        depth=2,
    )
    enc = HybridLatentEncoder(cfg).double()
    tr = FluxDiffusionPrior(
        nrho=basis.nr, cmd_dim=max(n_coil, 1), feat_dim=n_free
    ).double()
    eng = GSGroundedLatentEngine(enc, basis, tr)
    return eng, basis, n_coil, n_sensor


def test_gradients_flow_through_basis_and_structure_residual():
    """Both the magnetics prediction AND the structure residual are
    load-bearing on the encoder's patch-current head — .backward() through
    either reaches finite, non-zero gradient on the head's weights."""
    torch.manual_seed(0)
    eng, basis, n_coil, n_sensor = _engine()
    b = 4
    x = torch.randn(b, n_sensor, dtype=torch.float64, requires_grad=True)
    ip = torch.full((b,), 5.0e4, dtype=torch.float64)
    i_pf = torch.zeros(b, n_coil, dtype=torch.float64)
    lam = torch.full((b,), 3.0, dtype=torch.float64)

    lat = eng.encode(x)
    i_cell = eng.i_cell_from_latent(lat, ip)
    mag = eng.predict_magnetics(i_cell, i_pf)
    struct = eng.structure_residual_loss(i_cell, i_pf, lam)
    (mag.pow(2).sum() + struct).backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert float(x.grad.abs().max()) > 0.0

    head_grad = eng.encoder.patch_head.weight.grad
    assert head_grad is not None
    assert torch.isfinite(head_grad).all()
    assert float(head_grad.abs().max()) > 0.0


def test_ip_anchor_drives_sum_toward_ip():
    """The Rogowski Ip anchor alone (no misfit term) pulls Σi_cell -> Ip."""
    torch.manual_seed(1)
    eng, basis, n_coil, n_sensor = _engine()
    b = 4
    x = torch.randn(b, n_sensor, dtype=torch.float64)
    ip = torch.tensor([8.0e4, 3.0e4, -5.0e4, 1.0e5], dtype=torch.float64)
    opt = torch.optim.Adam(eng.encoder.parameters(), lr=1e-2)

    def ip_pen():
        lat = eng.encode(x)
        i_cell = eng.i_cell_from_latent(lat, ip)
        return (((i_cell.sum(-1) - ip) / ip) ** 2).mean()

    r0 = ip_pen().item()
    for _ in range(400):
        opt.zero_grad()
        loss = ip_pen()
        loss.backward()
        opt.step()
    r1 = ip_pen().item()
    assert r1 < 0.01 * r0


def test_single_example_whitened_misfit_decreases_on_overfit():
    """Overfitting one synthetic slice drives the whitened magnetics misfit
    (:meth:`GSGroundedLatentEngine.magnetics_loss`) sharply down."""
    torch.manual_seed(2)
    eng, basis, n_coil, n_sensor = _engine()
    rng = np.random.default_rng(0)
    n = int(basis.r_cells.shape[0])
    i_cell_true = torch.as_tensor(
        rng.standard_normal((1, n)) * 1e4, dtype=torch.float64
    )
    i_pf = torch.zeros(1, n_coil, dtype=torch.float64)
    raw_mag = basis.sensors(i_cell_true, i_pf).detach()
    scale = raw_mag.abs().clamp_min(1e-6)
    x = raw_mag / scale
    ip = i_cell_true.sum(-1).detach()
    opt = torch.optim.Adam(eng.encoder.parameters(), lr=5e-3)

    def misfit():
        lat = eng.encode(x)
        i_cell = eng.i_cell_from_latent(lat, ip)
        return eng.magnetics_loss(i_cell, i_pf, raw_mag, scale)

    m0 = misfit().item()
    for _ in range(300):
        opt.zero_grad()
        loss = misfit()
        loss.backward()
        opt.step()
    m1 = misfit().item()
    assert m1 < 0.5 * m0
