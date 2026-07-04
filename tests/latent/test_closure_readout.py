"""Tests for the closure-coordinate head (§B3 — closures as dimensionless
latent coordinates).

Pins: the head's output shape; the self-supervised aux loss
(:meth:`GSGroundedLatentEngine.closure_readout_loss`) decreases on a trivial
overfit; the closures it targets are finite; and the ``F² >= 0`` integrability
penalty (:func:`imas_ambix.latent.structure_residual.f2_integrability_penalty`)
still runs cleanly against head-predicted coefficients — the closure-recovery
path this head is meant to short-circuit stays intact.
"""

from __future__ import annotations

import torch

from imas_ambix.latent.encoder import HybridLatentEncoder, LatentConfig
from imas_ambix.latent.engine import GSGroundedLatentEngine, LossWeights
from imas_ambix.latent.patch_basis import PatchBasis
from imas_ambix.latent.structure_residual import f2_integrability_penalty
from imas_ambix.latent.transport import FluxDiffusionPrior
from tests.latent.test_patch_basis import _confining_table

N_BINS = 8


def _engine(*, n_closure_bins: int = N_BINS, n_free: int = 6):
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
        n_anchored=2,
        n_free=n_free,
        n_cells=n,
        n_closure_bins=n_closure_bins,
        hidden=48,
        depth=2,
    )
    enc = HybridLatentEncoder(cfg).double()
    tr = FluxDiffusionPrior(
        nrho=basis.nr, cmd_dim=max(n_coil, 1), feat_dim=n_free
    ).double()
    eng = GSGroundedLatentEngine(enc, basis, tr)
    return eng, basis, n_coil, n_sensor


def test_closure_head_shapes():
    eng, basis, n_coil, n_sensor = _engine()
    b = 5
    x = torch.randn(b, n_sensor, dtype=torch.float64)
    lat = eng.encode(x)
    assert lat.closure is not None
    assert lat.closure.shape == (b, N_BINS, 2)


def test_closure_head_disabled_yields_none_and_zero_loss():
    eng, basis, n_coil, n_sensor = _engine(n_closure_bins=0)
    b = 3
    x = torch.randn(b, n_sensor, dtype=torch.float64)
    ip = torch.full((b,), 5.0e4, dtype=torch.float64)
    i_pf = torch.zeros(b, n_coil, dtype=torch.float64)
    lat = eng.encode(x)
    assert lat.closure is None
    i_cell = eng.i_cell_from_latent(lat, ip)
    loss = eng.closure_readout_loss(lat, i_cell, i_pf)
    assert float(loss) == 0.0


def test_closure_aux_loss_decreases_on_overfit():
    """Overfitting the closure head alone (currents held fixed) drives the
    self-supervised aux loss toward its own detached target — the head learns
    to read closures straight out of the latent."""
    torch.manual_seed(3)
    eng, basis, n_coil, n_sensor = _engine()
    b = 4
    x = torch.randn(b, n_sensor, dtype=torch.float64)
    # a modest plasma current keeps j_phi = I/area (and hence the a_k, b_k
    # closures it implies) in a numerically tractable range for this smoke
    # test — the physical current scale plays no role in what is being pinned
    ip = torch.full((b,), 1.0e3, dtype=torch.float64)
    i_pf = torch.zeros(b, n_coil, dtype=torch.float64)

    # freeze everything except the closure head so the aux loss is measured in
    # isolation from the (co-adapting) current head
    for name, p in eng.encoder.named_parameters():
        p.requires_grad_(name.startswith("closure_head"))
    opt = torch.optim.Adam(eng.encoder.closure_head.parameters(), lr=0.5)

    def aux():
        lat = eng.encode(x)
        i_cell = eng.i_cell_from_latent(lat, ip)
        return eng.closure_readout_loss(lat, i_cell, i_pf)

    a0 = aux().item()
    for _ in range(300):
        opt.zero_grad()
        loss = aux()
        loss.backward()
        opt.step()
    a1 = aux().item()
    assert a1 < 0.5 * a0


def test_closures_finite_and_f2_integrability_penalty_path_intact():
    """The fit the closure head targets is finite, and the F² >= 0
    integrability penalty (the closure-recovery ablation this head
    short-circuits) still evaluates cleanly on head-predicted coefficients."""
    torch.manual_seed(4)
    eng, basis, n_coil, n_sensor = _engine()
    b = 2
    x = torch.randn(b, n_sensor, dtype=torch.float64)
    ip = torch.full((b,), 5.0e4, dtype=torch.float64)
    i_pf = torch.zeros(b, n_coil, dtype=torch.float64)
    lat = eng.encode(x)
    i_cell = eng.i_cell_from_latent(lat, ip)

    # the detached target the aux loss trains against
    from imas_ambix.latent.structure_residual import fit_flux_functions

    psi_c = eng.basis.psi_cells(i_cell, i_pf)
    jphi_c = i_cell / eng.basis.cell_area
    r_c, z_c = eng.basis.r_cells, eng.basis.z_cells
    fit = fit_flux_functions(
        psi_c[0], r_c, jphi_c[0], n_bins=N_BINS, z_c=z_c, connectivity="locality"
    )
    assert torch.isfinite(fit.a_k).all()
    assert torch.isfinite(fit.b_k).all()

    # F^2 >= 0 penalty against the HEAD's own (untrained) b_k prediction — the
    # path this head is meant to short-circuit must keep working end to end
    b_k_head = lat.closure[0, :, 1]
    dpsi = torch.full((N_BINS,), 1e-3, dtype=torch.float64)
    penalty = f2_integrability_penalty(b_k_head, dpsi, f_vac=1.0)
    assert torch.isfinite(penalty)
    assert penalty >= 0
