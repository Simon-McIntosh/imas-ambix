"""Tests for the assembled GS-grounded latent engine + composite loss.

The engine wires the hybrid-latent encoder → patch-current basis (spatial
anchor) → flux-diffusion transport prior (temporal anchor) and defines the
raw-signal self-supervised objective. The load-bearing behaviour pinned here:

* the whole chain (encoder → patch-current shape → sensors) can **ground in
  raw magnetics** — training drives the whitened misfit sharply down (no EFIT
  labels);
* the composite loss returns every term, is differentiable, and reports D≥0;
* the command is **load-bearing** through the transport prior at the engine
  level (zeroing it changes the transport tendency);
* topology is read from the engine's solved ψ (axis recovered).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from imas_ambix.gs.machine_geometry import GeometryIdentity, OperatorGeometry
from imas_ambix.latent.encoder import HybridLatentEncoder, LatentConfig
from imas_ambix.latent.engine import GSGroundedLatentEngine
from imas_ambix.latent.patch_basis import PatchBasis
from imas_ambix.latent.transport import FluxDiffusionPrior


def _synthetic_table() -> OperatorGeometry:
    bp_v = SimpleNamespace(index=0, r=1.5, z=0.0, angle_deg=-90.0, length=0.025)
    bp_r = SimpleNamespace(index=1, r=1.4, z=0.3, angle_deg=0.0, length=0.025)
    bp_v2 = SimpleNamespace(index=2, r=1.45, z=-0.3, angle_deg=-90.0, length=0.025)
    fl = SimpleNamespace(index=0, r=1.3, z=0.5)
    pf_known = [
        SimpleNamespace(
            r=1.50, z=1.10, turns=1.0, width=0.01, height=0.01, circuit=1, xmult=1.0
        ),
    ]
    pf_passive = [
        SimpleNamespace(
            r=2.0, z=0.0, turns=1.0, width=0.01, height=0.01, circuit=2, xmult=1.0
        ),
    ]
    identity = GeometryIdentity(
        representation_key="mp3-fl1-fc2-lim4-abcd12340000",
        representation_digest="abcd12340000",
        derivation_id="synthetic-engine",
        physical_digest="",
        registry_digest="",
    )
    sensor_map = [
        SimpleNamespace(
            amb_channel="obv01",
            kind="b_probe",
            efm_index=0,
            r=1.5,
            z=0.0,
            angle_deg=-90.0,
            residual_m=0.001,
            flag="",
        ),
        SimpleNamespace(
            amb_channel="obr01",
            kind="b_probe",
            efm_index=1,
            r=1.4,
            z=0.3,
            angle_deg=0.0,
            residual_m=0.001,
            flag="",
        ),
        SimpleNamespace(
            amb_channel="obv02",
            kind="b_probe",
            efm_index=2,
            r=1.45,
            z=-0.3,
            angle_deg=-90.0,
            residual_m=0.001,
            flag="",
        ),
        SimpleNamespace(
            amb_channel="fl_p4u_1",
            kind="flux_loop",
            efm_index=0,
            r=1.3,
            z=0.5,
            angle_deg=None,
            residual_m=0.001,
            flag="",
        ),
    ]
    return OperatorGeometry(
        identity=identity,
        probes=(bp_v, bp_r, bp_v2),
        loops=(fl,),
        conductors=tuple(pf_known + pf_passive),
        passives=(SimpleNamespace(name="wall_a", r=2.0, z=0.0, obsolete=False),),
        limiter_r=(0.3, 1.6, 1.6, 0.3),
        limiter_z=(-1.0, -1.0, 1.0, 1.0),
        polygon_sections=(),
        drive_map=(),
        sensor_map=tuple(sensor_map),
        unmatched_channels=(),
        active_circuits=(),
        available_current_channels=("p4u_coil_current", "plasma_current"),
        r0=0.85,
        minor_radius=0.65,
        unresolved_turns={},
        coil_channels=(),
        coil_column_matrix=np.zeros((len(sensor_map), 0), dtype=np.float64),
    )


def _engine(table, *, probabilistic=False, n_free=6, nr=49, nz=65):
    basis = PatchBasis.from_table(
        table, nr=nr, nz=nz, cache_dir=None, dtype=torch.float64
    )
    n_sensor = len(basis.sensor_channels)
    n_coil = int(basis.psi_coil_grid.shape[1])
    n_cells = int(basis.r_cells.shape[0])
    cfg = LatentConfig(
        n_features=n_sensor,
        n_theta=1,
        n_anchored=3,
        n_free=n_free,
        n_cells=n_cells,
        hidden=64,
        depth=2,
        probabilistic=probabilistic,
    )
    enc = HybridLatentEncoder(cfg)
    tr = FluxDiffusionPrior(nrho=basis.nr, cmd_dim=max(n_coil, 1), feat_dim=n_free)
    return GSGroundedLatentEngine(enc, basis, tr), basis, n_coil


def test_engine_forward_shapes():
    table = _synthetic_table()
    eng, basis, n_coil = _engine(table)
    eng = eng.double()
    b = 4
    x = torch.randn(b, len(basis.sensor_channels), dtype=torch.float64)
    i_pf = torch.randn(b, n_coil, dtype=torch.float64)
    ip = torch.full((b,), 5.0e4, dtype=torch.float64)
    lat = eng.encode(x)
    i_cell = eng.i_cell_from_latent(lat, ip)
    assert i_cell.shape == (b, int(basis.r_cells.shape[0]))
    pred = eng.predict_magnetics(i_cell, i_pf)
    assert pred.shape == (b, len(basis.sensor_channels))
    prof, rho = eng.psi_profile(i_cell, i_pf)
    assert prof.shape == (b, basis.nr)
    assert rho.shape[-1] == basis.nr


def test_engine_grounds_in_raw_magnetics():
    """Training the encoder→patch-current→sensors chain drives the whitened
    magnetics misfit sharply down."""
    torch.manual_seed(0)
    table = _synthetic_table()
    eng, basis, n_coil = _engine(table)
    eng = eng.double()
    rng = np.random.default_rng(0)
    n = int(basis.r_cells.shape[0])
    # a known 'true' patch-current vector + zero coil current → synthetic raw
    # magnetics.  The whitened residual is scale-free, so the amplitude is
    # irrelevant — O(1e4) currents for clean optimiser conditioning.
    i_cell_true = torch.as_tensor(
        rng.standard_normal((1, n)) * 1e4, dtype=torch.float64
    )
    i_pf = torch.zeros(1, n_coil, dtype=torch.float64)
    raw_mag = basis.sensors(i_cell_true, i_pf).detach()
    scale = raw_mag.abs().clamp_min(1e-6)
    ip = i_cell_true.sum(-1).detach()
    x = (raw_mag / scale).repeat(32, 1)  # features = normalised raw magnetics
    rm = raw_mag.repeat(32, 1)
    ipf = i_pf.repeat(32, 1)
    sc = scale.repeat(32, 1)
    ip_batch = ip.repeat(32)
    opt = torch.optim.Adam(eng.parameters(), lr=5e-3)

    def misfit():
        lat = eng.encode(x)
        i_cell = eng.i_cell_from_latent(lat, ip_batch)
        return eng.magnetics_loss(i_cell, ipf, rm, sc)

    r0 = misfit().item()
    for _ in range(400):
        opt.zero_grad()
        loss = misfit()
        loss.backward()
        opt.step()
    r1 = misfit().item()
    assert r1 < 0.05 * r0  # the readout learns to explain the raw magnetics


def test_composite_losses_differentiable_and_report_d_nonneg():
    torch.manual_seed(1)
    table = _synthetic_table()
    eng, basis, n_coil = _engine(table)
    eng = eng.double()
    n_s = len(basis.sensor_channels)
    b = 6
    batch = {
        "x_t": torch.randn(b, n_s, dtype=torch.float64, requires_grad=True),
        "x_tp1": torch.randn(b, n_s, dtype=torch.float64),
        "i_pf_t": torch.randn(b, n_coil, dtype=torch.float64),
        "i_pf_tp1": torch.randn(b, n_coil, dtype=torch.float64),
        "ip_t": torch.full((b,), 5.0e4, dtype=torch.float64),
        "raw_mag_t": torch.randn(b, n_s, dtype=torch.float64),
        "sensor_scale": torch.ones(b, n_s, dtype=torch.float64),
        "cmd_t": torch.randn(b, max(n_coil, 1), dtype=torch.float64),
        "anchored_target": torch.randn(b, 3, dtype=torch.float64),
        "anchored_mask": torch.ones(b, 3, dtype=torch.bool),
        "dt": 1e-3,
    }
    out = eng.losses(batch)
    for k in (
        "magnetics",
        "ip_anchor",
        "structure_residual",
        "anchored",
        "dissipation",
        "volt_second",
        "dimensionless",
        "closure",
        "total",
    ):
        assert k in out, k
        assert torch.isfinite(out[k]).all(), k
    assert out["diffusivity_min"] > 0  # D ≥ 0 verified
    out["total"].backward()
    assert batch["x_t"].grad is not None and torch.isfinite(batch["x_t"].grad).all()


def test_engine_command_is_load_bearing():
    table = _synthetic_table()
    eng, basis, n_coil = _engine(table)
    eng = eng.double()
    n_s = len(basis.sensor_channels)
    b = 3
    x = torch.randn(b, n_s, dtype=torch.float64)
    i_pf = torch.randn(b, n_coil, dtype=torch.float64)
    ip = torch.full((b,), 5.0e4, dtype=torch.float64)
    lat = eng.encode(x)
    i_cell = eng.i_cell_from_latent(lat, ip)
    prof, rho = eng.psi_profile(i_cell, i_pf)
    cmd = torch.randn(b, max(n_coil, 1), dtype=torch.float64)
    d_cmd = eng.transport.dpsi_dt(prof, rho, lat.free, cmd)
    d_zero = eng.transport.dpsi_dt(prof, rho, lat.free, torch.zeros_like(cmd))
    assert (d_cmd - d_zero).abs().sum() > 0


def test_engine_reads_topology_axis():
    table = _synthetic_table()
    eng, basis, n_coil = _engine(table)
    eng = eng.double()
    # a compact interior current blob → an O-point (axis) inside the vessel
    r_c = basis.r_cells.numpy()
    z_c = basis.z_cells.numpy()
    blob = np.exp(-(((r_c - basis.r0) / 0.35) ** 2 + (z_c / 0.5) ** 2))
    i_cell = torch.as_tensor((blob / blob.sum() * 4.0e5)[None], dtype=torch.float64)
    i_pf = torch.zeros(1, n_coil, dtype=torch.float64)
    read = eng.read_topology(i_cell, i_pf)[0]
    assert read.axis is not None
    assert np.isfinite(read.target[:2]).all()
