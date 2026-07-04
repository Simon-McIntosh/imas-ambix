"""Tests for the Gaussian ("gaussian-direct") patch-encoder head.

The default heads (``"direct"``, ``"lowrank"``) MUST stay byte-identical to
their pre-Gaussian-head behaviour — a corpus worker has staged retrains in
flight against them.  This file therefore splits into two concerns:

* regression: the Gaussian addition changes nothing about the existing head
  construction or forward path when it is not selected;
* new behaviour: the mean + log-σ head, exact linear sensor-variance
  propagation (``pred_var = i_var @ (m_sens²)ᵀ``, never a full covariance),
  the whitened Gaussian NLL in :func:`amortised_losses`, the log-σ clamp, and
  an overfit smoke test showing the NLL can be driven down on one slice.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from imas_ambix.latent.gs_solve import EquilibriumGrid, solve_equilibrium
from imas_ambix.latent.patch_basis import PatchBasis
from imas_ambix.latent.patch_encoder import (
    GAUSSIAN_LOG_SIGMA_MAX,
    GAUSSIAN_LOG_SIGMA_MIN,
    PatchCurrentEncoder,
    PatchEncoderConfig,
    amortised_losses,
    sensor_geometry_from_records,
)


def _confining_table():
    """Synthetic machine: rectangular limiter + a vertical-field coil pair."""
    from imas_ambix.gs import geometry as gsg

    probes = [
        gsg.BProbe(index=i, r=1.35, z=-0.6 + 0.3 * i, angle_deg=90.0, length=0.02)
        for i in range(5)
    ]
    sensor_map = [
        gsg.SensorMapping(f"obv{i:02d}", "b_probe", i, p.r, p.z, p.angle_deg, 0.001, "")
        for i, p in enumerate(probes)
    ]
    pf = [
        gsg.PFFilament(
            r=1.1, z=1.0, turns=1.0, width=0.06, height=0.06, circuit=1, xmult=1.0
        ),
        gsg.PFFilament(
            r=1.1, z=-1.0, turns=1.0, width=0.06, height=0.06, circuit=2, xmult=1.0
        ),
    ]
    return gsg.GeometryTable(
        signature=gsg.SetupSignature(
            n_bprobe=5, n_fluxloop=0, n_pf_filament=2, n_limiter=5, digest="feed0000"
        ),
        shots=[1],
        b_probes=probes,
        flux_loops=[],
        pf_filaments=pf,
        limiter_r=[0.35, 1.45, 1.45, 0.35, 0.35],
        limiter_z=[-0.85, -0.85, 0.85, 0.85, -0.85],
        sensor_map=sensor_map,
        passive_structures=[],
        amc_current_channels=[],
        unmatched_amb=[],
    )


def _small_encoder(*, n_coil: int, d_model=32, n_layers=1, n_time=4, head="direct"):
    """A tiny encoder + a candidate mask, no basis needed (pure shape tests)."""
    rng = np.random.default_rng(0)
    n_sensor = 5
    geom = sensor_geometry_from_records(
        r=rng.uniform(1.0, 1.5, n_sensor),
        z=rng.uniform(-0.6, 0.6, n_sensor),
        angle_deg=np.full(n_sensor, 90.0),
        kind=["b_probe"] * n_sensor,
    )
    coils = rng.uniform(0.8, 1.2, (n_coil, 2)) if n_coil else None
    n_cells = 40
    cm = np.ones(n_cells)
    cm[:5] = 0.0  # a forbidden region
    cfg = PatchEncoderConfig(
        d_model=d_model,
        n_heads=4,
        n_layers=n_layers,
        dim_feedforward=64,
        dropout=0.15,
        n_time=n_time,
        head=head,
    )
    enc = PatchCurrentEncoder(
        cfg, sensor_geometry=geom, coil_centroids=coils, candidate_mask=cm
    )
    return enc


def _rand_inputs(enc, b=3, *, seed=1):
    rng = np.random.default_rng(seed)
    t, s = enc.n_time, enc.n_sensor
    values = torch.as_tensor(rng.standard_normal((b, t, s)), dtype=torch.float32)
    finite = torch.ones(b, t, s, dtype=torch.bool)
    i_pf = torch.as_tensor(rng.standard_normal((b, enc.n_coil)), dtype=torch.float32)
    ip = torch.as_tensor(rng.uniform(1e5, 5e5, b), dtype=torch.float32)
    return values, finite, i_pf, ip


# --------------------------------------------------------------------------
# regression: the default heads are untouched by the Gaussian addition
# --------------------------------------------------------------------------


def test_direct_and_lowrank_heads_gain_no_gaussian_only_params():
    """The default heads must not pick up ``log_sigma_head`` or any other
    Gaussian-only parameter — construction is exactly what it was before."""
    for head in ("direct", "lowrank"):
        enc = _small_encoder(n_coil=2, head=head)
        assert not hasattr(enc, "log_sigma_head")


def test_direct_head_return_variance_flag_is_a_no_op():
    """``return_variance=True`` must not change the default heads' output —
    they ignore the flag and still return the single mean tensor, byte-for-
    byte identical to calling without it."""
    for head in ("direct", "lowrank"):
        enc = _small_encoder(n_coil=2, head=head).eval()
        values, finite, i_pf, ip = _rand_inputs(enc)
        with torch.no_grad():
            out_plain = enc(values, finite, i_pf, ip)
            out_flagged = enc(values, finite, i_pf, ip, return_variance=True)
        assert isinstance(out_plain, torch.Tensor)
        assert isinstance(out_flagged, torch.Tensor)  # not a tuple
        torch.testing.assert_close(out_plain, out_flagged)


def test_amortised_losses_i_var_none_keeps_original_keys():
    """``i_var=None`` (the default) must not add an ``nll`` key or otherwise
    change the returned dict's shape — this is the pre-Gaussian-head path."""
    table = _confining_table()
    basis = PatchBasis.from_table(table, nr=25, nz=33, cache_dir=None)
    grid = EquilibriumGrid.from_table(table, nr=25, nz=33)
    ip = 4.0e5
    res = solve_equilibrium(grid, np.array([-6.0e4, -6.0e4]), ip, beta0=0.5)
    truth = torch.as_tensor(res.cell_currents, dtype=torch.float64)[None]

    n_sensor = int(basis.m_sens.shape[0])
    measured = basis.sensors(truth)
    kwargs = dict(
        measured=measured,
        vacuum=torch.zeros(1, n_sensor, dtype=torch.float64),
        mask=torch.ones(1, n_sensor, dtype=torch.float64),
        scale=measured.abs() + 1e-9,
        i_pf_amperes=torch.zeros(1, 0, dtype=torch.float64),
        ip=torch.tensor([ip], dtype=torch.float64),
        lam=torch.tensor([0.0], dtype=torch.float64),
    )
    out = amortised_losses(basis, truth, **kwargs)
    assert set(out.keys()) == {"misfit", "ip_pen", "fb", "total"}


# --------------------------------------------------------------------------
# new behaviour: the gaussian-direct head
# --------------------------------------------------------------------------


def test_gaussian_head_forward_shapes_and_masking():
    """Mean/variance both ``(B, n_cells)``; forbidden cells zeroed in both;
    the mean-only call matches the mean half of the variance-returning call."""
    enc = _small_encoder(n_coil=2, head="gaussian-direct").eval()
    values, finite, i_pf, ip = _rand_inputs(enc)
    with torch.no_grad():
        mean_only = enc(values, finite, i_pf, ip)
        mean, var = enc(values, finite, i_pf, ip, return_variance=True)
    assert mean_only.shape == (3, enc.n_cells)
    assert mean.shape == (3, enc.n_cells)
    assert var.shape == (3, enc.n_cells)
    torch.testing.assert_close(mean, mean_only)
    assert torch.all(var >= 0)
    forbidden = enc.candidate_mask == 0
    assert torch.all(mean[:, forbidden] == 0.0)
    assert torch.all(var[:, forbidden] == 0.0)


def test_gaussian_head_backward_populates_grad():
    """Both the mean arm and the log-σ arm receive gradient."""
    enc = _small_encoder(n_coil=2, head="gaussian-direct")
    values, finite, i_pf, ip = _rand_inputs(enc)
    mean, var = enc(values, finite, i_pf, ip, return_variance=True)
    (mean.pow(2).mean() + var.mean()).backward()
    assert enc.head.weight.grad is not None
    assert enc.log_sigma_head.weight.grad is not None


def test_gaussian_log_sigma_clamp():
    """log-σ is clamped into [GAUSSIAN_LOG_SIGMA_MIN, GAUSSIAN_LOG_SIGMA_MAX]
    regardless of what the linear head produces."""
    enc = _small_encoder(n_coil=0, head="gaussian-direct")
    d = enc.config.d_model
    pooled = torch.zeros(2, d)

    with torch.no_grad():
        enc.log_sigma_head.weight.zero_()
        enc.log_sigma_head.bias.fill_(1000.0)
    _, log_sigma = enc._decode(pooled)
    assert torch.all(log_sigma == GAUSSIAN_LOG_SIGMA_MAX)

    with torch.no_grad():
        enc.log_sigma_head.bias.fill_(-1000.0)
    _, log_sigma = enc._decode(pooled)
    assert torch.all(log_sigma == GAUSSIAN_LOG_SIGMA_MIN)


def test_exact_linear_variance_propagation_matches_monte_carlo():
    """``pred_var = i_var @ (m_sens²)ᵀ`` (the diagonal matvec the encoder/loss
    use) matches Monte-Carlo sampling of the same diagonal Gaussian through
    the real linear sensor forward, within sampling tolerance."""
    table = _confining_table()
    basis = PatchBasis.from_table(table, nr=25, nz=33, cache_dir=None)
    n = int(basis.m_sens.shape[1])
    rng = np.random.default_rng(3)

    mean = torch.as_tensor(rng.uniform(-1.0e4, 1.0e4, n), dtype=torch.float64)
    var = torch.as_tensor(rng.uniform(1.0e6, 1.0e8, n), dtype=torch.float64)
    std = var.sqrt()

    m_sens = basis.m_sens.to(torch.float64)  # (S, n)
    analytic_mean = mean @ m_sens.T
    analytic_var = var @ (m_sens**2).T

    torch.manual_seed(0)
    k = 60_000
    samples = mean[None, :] + std[None, :] * torch.randn(k, n, dtype=torch.float64)
    pred_samples = samples @ m_sens.T  # (K, S)
    mc_mean = pred_samples.mean(0)
    mc_var = pred_samples.var(0, unbiased=True)

    torch.testing.assert_close(
        analytic_mean, mc_mean, rtol=0.02, atol=0.02 * analytic_mean.abs().max()
    )
    torch.testing.assert_close(
        analytic_var, mc_var, rtol=0.05, atol=0.05 * analytic_var.abs().max()
    )


def test_amortised_losses_gaussian_nll_matches_closed_form():
    """The ``nll`` term returned by :func:`amortised_losses` matches the
    closed-form whitened Gaussian NLL computed independently from the same
    ``m_sens`` — a direct check of the implementation, not just the physics
    claim (covered separately by the Monte-Carlo test above)."""
    table = _confining_table()
    basis = PatchBasis.from_table(table, nr=25, nz=33, cache_dir=None)
    n = int(basis.m_sens.shape[1])
    n_sensor = int(basis.m_sens.shape[0])
    rng = np.random.default_rng(4)

    ic = torch.as_tensor(rng.uniform(-1.0e4, 1.0e4, (1, n)), dtype=torch.float64)
    i_var = torch.as_tensor(rng.uniform(1.0e4, 1.0e6, (1, n)), dtype=torch.float64)
    vacuum = torch.zeros(1, n_sensor, dtype=torch.float64)
    measured = torch.as_tensor(
        rng.uniform(-1.0, 1.0, (1, n_sensor)), dtype=torch.float64
    )
    mask = torch.ones(1, n_sensor, dtype=torch.float64)
    scale = torch.as_tensor(rng.uniform(0.5, 2.0, (1, n_sensor)), dtype=torch.float64)

    out = amortised_losses(
        basis,
        ic,
        measured=measured,
        vacuum=vacuum,
        mask=mask,
        scale=scale,
        i_pf_amperes=torch.zeros(1, 0, dtype=torch.float64),
        ip=torch.tensor([1.0], dtype=torch.float64),
        lam=torch.tensor([0.0], dtype=torch.float64),
        i_var=i_var,
    )
    assert "nll" in out
    assert "misfit" in out  # still reported, for comparability

    m_sens = basis.m_sens.to(torch.float64)
    pred = vacuum + ic @ m_sens.T
    pred_var = i_var @ (m_sens**2).T
    resid = (pred - measured) / scale
    pred_var_wh = pred_var / scale**2
    nll_terms = 0.5 * (torch.log(2.0 * math.pi * pred_var_wh) + resid**2 / pred_var_wh)
    expected = nll_terms.mean(-1)
    torch.testing.assert_close(out["nll"], expected, rtol=1e-9, atol=1e-9)


def test_gaussian_overfit_single_example_smoke():
    """The go/no-go for the distributional path: a tiny gaussian-direct
    encoder drives the NLL down substantially on ONE synthetic slice."""
    torch.manual_seed(0)
    table = _confining_table()
    basis = PatchBasis.from_table(table, nr=25, nz=33, cache_dir=None)
    grid = EquilibriumGrid.from_table(table, nr=25, nz=33)
    ip = 4.0e5
    res = solve_equilibrium(grid, np.array([-6.0e4, -6.0e4]), ip, beta0=0.5)
    truth = torch.as_tensor(res.cell_currents, dtype=torch.float64)[None]

    n_sensor = int(basis.m_sens.shape[0])
    measured = basis.sensors(truth)
    scale = measured.abs() + 1e-9

    smap = table.sensor_map
    geom = sensor_geometry_from_records(
        r=[m.r for m in smap],
        z=[m.z for m in smap],
        angle_deg=[m.angle_deg for m in smap],
        kind=[m.kind for m in smap],
    )
    cfg = PatchEncoderConfig(
        d_model=32,
        n_heads=4,
        n_layers=1,
        dim_feedforward=64,
        dropout=0.0,
        n_time=4,
        head="gaussian-direct",
    )
    enc = PatchCurrentEncoder(
        cfg,
        sensor_geometry=geom,
        coil_centroids=None,
        candidate_mask=basis.candidate_mask.numpy(),
    )
    enc.train()

    vals = (measured.to(torch.float32) / scale.to(torch.float32)).reshape(
        1, 1, n_sensor
    )
    values = vals.expand(1, 4, n_sensor).contiguous()
    finite = torch.ones(1, 4, n_sensor, dtype=torch.bool)
    i_pf = torch.zeros(1, 0, dtype=torch.float32)
    ip_t = torch.tensor([ip], dtype=torch.float32)

    loss_kwargs = dict(
        measured=measured,
        vacuum=torch.zeros(1, n_sensor, dtype=torch.float64),
        mask=torch.ones(1, n_sensor, dtype=torch.float64),
        scale=scale,
        i_pf_amperes=torch.zeros(1, 0, dtype=torch.float64),
        ip=torch.tensor([ip], dtype=torch.float64),
        lam=torch.tensor([0.0], dtype=torch.float64),
    )

    opt = torch.optim.Adam(enc.parameters(), lr=1e-3)
    init_nll = None
    final_nll = None
    init_misfit = None
    final_misfit = None
    for step in range(300):
        opt.zero_grad()
        i_mean, i_var = enc(values, finite, i_pf, ip_t, return_variance=True)
        out = amortised_losses(basis, i_mean, i_var=i_var, **loss_kwargs)
        out["total"].backward()
        opt.step()
        if step == 0:
            init_nll = float(out["nll"][0])
            init_misfit = float(out["misfit"][0])
        final_nll = float(out["nll"][0])
        final_misfit = float(out["misfit"][0])

    assert init_nll is not None and final_nll is not None
    assert final_nll < init_nll - 5.0, (init_nll, final_nll)
    assert final_misfit < init_misfit / 10.0, (init_misfit, final_misfit)


def test_firewall_static_no_evaluator_imports():
    """The encoder module must not touch the EFIT / evaluator / world-model
    side — re-asserted here since this file exercises the new head path."""
    from pathlib import Path

    import imas_ambix.latent.patch_encoder as m

    src = Path(m.__file__).read_text()
    for banned in ("efit_referee", "equilibrium_labels", "worldmodel"):
        assert banned not in src, f"patch_encoder references the firewalled {banned}"
