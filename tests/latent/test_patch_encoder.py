"""Tests for the amortised patch-current encoder.

The encoder maps a temporal window of raw magnetics + known coil currents to a
per-cell patch-current vector in one forward pass, trained with the same
self-supervised objective the variational inverse minimises.  Correctness is
pinned on synthetic geometry (CPU, no MAST data, no EFIT):

* shapes + the dimensionless ``I = x·Ip/n·candidate`` current convention;
* both head arms construct, forward, and backprop;
* the (sensor, step) tokens are set-like — permuting sensors + their geometry
  leaves the output unchanged;
* non-finite entries are handled by the has-value flag (no NaNs, mild effect);
* the batched self-supervised losses on a real Picard equilibrium — misfit ≈ 0
  at the truth, structure residual small at the truth and larger when permuted;
* the discrepancy-λ schedule warm-up / freeze / adapt / clamp behaviour;
* a single-example overfit smoke test (the go/no-go: the encoder can fit);
* a static firewall check.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from imas_ambix.gs.machine_geometry import GeometryIdentity, OperatorGeometry
from imas_ambix.latent.gs_solve import EquilibriumGrid, solve_equilibrium
from imas_ambix.latent.patch_basis import PatchBasis
from imas_ambix.latent.patch_encoder import (
    DiscrepancyLambda,
    PatchCurrentEncoder,
    PatchEncoderConfig,
    amortised_losses,
    sensor_geometry_from_records,
)


def _operator_geometry() -> OperatorGeometry:
    probes = [
        SimpleNamespace(
            index=i,
            r=1.35,
            z=-0.6 + 0.3 * i,
            angle_deg=-90.0,
            length=0.02,
        )
        for i in range(5)
    ]
    sensors = tuple(
        SimpleNamespace(
            amb_channel=f"obv{i:02d}",
            kind="b_probe",
            efm_index=i,
            r=probe.r,
            z=probe.z,
            angle_deg=probe.angle_deg,
            residual_m=0.001,
            flag="",
        )
        for i, probe in enumerate(probes)
    )
    conductors = (
        SimpleNamespace(
            r=1.1,
            z=1.0,
            turns=1.0,
            width=0.06,
            height=0.06,
            circuit=1,
            xmult=1.0,
        ),
        SimpleNamespace(
            r=1.1,
            z=-1.0,
            turns=1.0,
            width=0.06,
            height=0.06,
            circuit=2,
            xmult=1.0,
        ),
    )
    return OperatorGeometry(
        identity=GeometryIdentity(
            representation_key="mp5-fl0-fc2-lim5-feed0000",
            representation_digest="feed0000",
            derivation_id="synthetic-encoder",
            physical_digest="",
            registry_digest="",
        ),
        probes=tuple(probes),
        loops=(),
        conductors=conductors,
        passives=(),
        limiter_r=(0.35, 1.45, 1.45, 0.35, 0.35),
        limiter_z=(-0.85, -0.85, 0.85, 0.85, -0.85),
        polygon_sections=(),
        drive_map=(),
        sensor_map=sensors,
        unmatched_channels=(),
        active_circuits=(),
        available_current_channels=(),
        r0=0.85,
        minor_radius=0.65,
        unresolved_turns={},
        coil_channels=(),
        coil_column_matrix=np.zeros((len(sensors), 0), dtype=np.float64),
    )


def _confining_table():
    """Synthetic machine: rectangular limiter + a vertical-field coil pair."""
    return _operator_geometry()


def _small_encoder(*, n_coil: int, d_model=32, n_layers=1, n_time=4, head="direct"):
    """A tiny encoder + a candidate mask, no basis needed (pure shape tests)."""
    rng = np.random.default_rng(0)
    n_sensor = 5
    geom = sensor_geometry_from_records(
        r=rng.uniform(1.0, 1.5, n_sensor),
        z=rng.uniform(-0.6, 0.6, n_sensor),
        angle_deg=np.full(n_sensor, -90.0),
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


def test_forward_shapes_and_current_convention():
    """Forward returns (B, n_cells); currents zero on the forbidden mask; ΣI is
    within a factor of the Ip scale (untrained, so only order-of-magnitude)."""
    for n_coil in (0, 2):
        enc = _small_encoder(n_coil=n_coil).eval()
        values, finite, i_pf, ip = _rand_inputs(enc)
        with torch.no_grad():
            i_cell = enc(values, finite, i_pf, ip)
        assert i_cell.shape == (3, enc.n_cells)
        forbidden = enc.candidate_mask == 0
        assert torch.all(i_cell[:, forbidden] == 0.0)
        # ΣI is not exact (untrained) but is on the Ip scale, not wildly off
        ratio = i_cell.sum(-1).abs() / ip
        assert torch.all(ratio < 50.0)


def test_both_head_arms_forward_and_backward():
    """direct + lowrank arms construct, forward, and populate grads."""
    for head in ("direct", "lowrank"):
        enc = _small_encoder(n_coil=2, head=head)
        values, finite, i_pf, ip = _rand_inputs(enc)
        i_cell = enc(values, finite, i_pf, ip)
        loss = (i_cell**2).mean()
        loss.backward()
        # a trunk parameter and a head parameter both receive gradient
        assert enc.value_proj.weight.grad is not None
        if head == "direct":
            assert enc.head.weight.grad is not None
        else:
            assert enc.basis_u.grad is not None
            assert enc.residual_alpha.grad is not None


def test_permutation_equivariance_over_sensors():
    """Permuting sensors together with their geometry leaves the output fixed."""
    enc = _small_encoder(n_coil=2).eval()
    values, finite, i_pf, ip = _rand_inputs(enc)
    with torch.no_grad():
        i1 = enc(values, finite, i_pf, ip)
    perm = torch.randperm(enc.n_sensor)
    # permute the construction geometry buffers to match the permuted inputs
    enc.register_buffer("sensor_geom", enc.sensor_geom[perm].clone())
    enc.register_buffer("sensor_kind", enc.sensor_kind[perm].clone())
    with torch.no_grad():
        i2 = enc(values[:, :, perm], finite[:, :, perm], i_pf, ip)
    torch.testing.assert_close(i1, i2, rtol=1e-4, atol=1e-5)


def test_finite_mask_is_nan_safe_and_mild():
    """A non-finite value with finite=False produces no NaNs and a bounded change."""
    enc = _small_encoder(n_coil=2).eval()
    values, finite, i_pf, ip = _rand_inputs(enc)
    with torch.no_grad():
        i_clean = enc(values, finite, i_pf, ip)
    v2 = values.clone()
    f2 = finite.clone()
    v2[0, 0, 0] = float("nan")
    f2[0, 0, 0] = False
    with torch.no_grad():
        i_masked = enc(v2, f2, i_pf, ip)
    assert torch.isfinite(i_masked).all()
    assert not torch.equal(i_clean, i_masked)  # the flag does change the token
    delta = (i_masked - i_clean).norm()
    assert torch.isfinite(delta)
    assert delta < 3.0 * i_clean.norm() + 1e-6  # bounded, not a blow-up


def test_amortised_losses_on_picard_truth():
    """On a real Picard equilibrium: misfit ≈ 0 at the truth, structure residual
    small at the truth and larger for a permuted current."""
    table = _confining_table()
    basis = PatchBasis.from_table(table, nr=25, nz=33, cache_dir=None)
    grid = EquilibriumGrid.from_table(table, nr=25, nz=33)
    ip = 4.0e5
    res = solve_equilibrium(grid, np.array([-6.0e4, -6.0e4]), ip, beta0=0.5)
    assert res.converged
    truth = torch.as_tensor(res.cell_currents, dtype=torch.float64)[None]  # (1, n)

    n_sensor = int(basis.m_sens.shape[0])
    measured = basis.sensors(truth)  # (1, S) — no coil columns for this table
    vacuum = torch.zeros(1, n_sensor, dtype=torch.float64)
    mask = torch.ones(1, n_sensor, dtype=torch.float64)
    scale = measured.abs() + 1e-9
    kwargs = dict(
        measured=measured,
        vacuum=vacuum,
        mask=mask,
        scale=scale,
        i_pf_amperes=torch.zeros(1, 0, dtype=torch.float64),
        ip=torch.tensor([ip], dtype=torch.float64),
        lam=torch.tensor([0.0], dtype=torch.float64),
    )
    out = amortised_losses(basis, truth, **kwargs)
    for v in out.values():
        assert torch.isfinite(v).all()
    assert float(out["misfit"][0]) < 1e-8  # pred == measured at the truth

    perm = torch.as_tensor(np.random.default_rng(0).permutation(truth.shape[1]))
    permuted = truth[:, perm]
    out_perm = amortised_losses(basis, permuted, **kwargs)
    assert float(out["fb"][0]) < float(out_perm["fb"][0])  # structure favours truth


def test_discrepancy_lambda_schedule():
    """Warm-up returns 0; boundary freezes the target; then λ adapts and clamps."""
    sched = DiscrepancyLambda(
        4, warmup_epochs=2, lam0=3.0, lam_max=100.0, ratio=1.5, adapt_factor=1.5
    )
    ids = torch.tensor([0, 1, 2, 3])
    warm_misfit = torch.tensor([1.0, 1.0, 1.0, 1.0])

    assert torch.all(sched.get(ids) == 0.0)  # epoch 0 < warm-up
    sched.update(ids, warm_misfit, epoch=0)
    sched.update(ids, warm_misfit, epoch=1)
    assert torch.all(sched.get(ids) == 0.0)  # still in warm-up

    sched.update(ids, warm_misfit, epoch=2)  # boundary: target = 1.5 × 1.0
    assert torch.allclose(sched.get(ids), torch.full((4,), 3.0, dtype=torch.float64))
    assert torch.allclose(sched.target[ids], torch.full((4,), 1.5, dtype=torch.float64))

    # misfit below target → λ rises; well above 1.2×target → λ falls
    sched.update(ids, torch.tensor([0.1, 0.1, 10.0, 10.0]), epoch=3)
    lam = sched.get(ids)
    assert float(lam[0]) > 3.0 and float(lam[2]) < 3.0

    # clamp: hammer one example below target for many epochs → hits lam_max
    for e in range(4, 40):
        sched.update(torch.tensor([0]), torch.tensor([0.0]), epoch=e)
    assert float(sched.get(torch.tensor([0]))[0]) <= 100.0 + 1e-9
    assert float(sched.get(torch.tensor([0]))[0]) >= 100.0 - 1e-6  # saturated


def test_overfit_single_example_smoke():
    """The go/no-go: a tiny encoder fits ONE synthetic slice — misfit drops ≥10×."""
    torch.manual_seed(0)
    table = _confining_table()
    basis = PatchBasis.from_table(table, nr=25, nz=33, cache_dir=None)
    grid = EquilibriumGrid.from_table(table, nr=25, nz=33)
    ip = 4.0e5
    res = solve_equilibrium(grid, np.array([-6.0e4, -6.0e4]), ip, beta0=0.5)
    truth = torch.as_tensor(res.cell_currents, dtype=torch.float64)[None]

    n_sensor = int(basis.m_sens.shape[0])
    measured = basis.sensors(truth)  # (1, S)
    scale = measured.abs() + 1e-9

    # geometry aligned to the operator sensor rows (b-probe, vertical)
    smap = table.sensor_map
    geom = sensor_geometry_from_records(
        r=[m.r for m in smap],
        z=[m.z for m in smap],
        angle_deg=[m.angle_deg for m in smap],
        kind=[m.kind for m in smap],
    )
    cfg = PatchEncoderConfig(
        d_model=32, n_heads=4, n_layers=1, dim_feedforward=64, dropout=0.0, n_time=4
    )
    enc = PatchCurrentEncoder(
        cfg,
        sensor_geometry=geom,
        coil_centroids=None,
        candidate_mask=basis.candidate_mask.numpy(),
    )
    enc.train()

    # constant single-example window (standardised magnetics broadcast over time)
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
    init_misfit = None
    final_misfit = None
    for step in range(300):
        opt.zero_grad()
        i_cell = enc(values, finite, i_pf, ip_t)
        out = amortised_losses(basis, i_cell, **loss_kwargs)
        out["total"].backward()
        opt.step()
        if step == 0:
            init_misfit = float(out["misfit"][0])
        final_misfit = float(out["misfit"][0])
    assert init_misfit is not None and final_misfit is not None
    assert final_misfit < init_misfit / 10.0, (init_misfit, final_misfit)


def test_firewall_static_no_evaluator_imports():
    """The encoder module must not touch the EFIT / evaluator / world-model side."""
    from pathlib import Path

    import imas_ambix.latent.patch_encoder as m

    src = Path(m.__file__).read_text()
    for banned in ("efit_referee", "equilibrium_labels", "worldmodel"):
        assert banned not in src, f"patch_encoder references the firewalled {banned}"
