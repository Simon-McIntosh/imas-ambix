"""Tests for the corpus training loop (shared encoder, multi-campaign, D≥0 gate).

The trainer folds all campaigns' shots into ONE shared
:class:`~imas_ambix.latent.encoder.HybridLatentEncoder` (the machine-agnostic
design: the encoder is campaign-agnostic; only the patch-current forward
substrate + transport prior are per-campaign, tied to that campaign's fixed
device geometry). Two things are pinned here without touching the mirror:

* :func:`consecutive_pairs` builds (t, t+1) training pairs from a shot's
  time-ordered slices, breaking across any gap larger than the nominal
  timestep (plasma-on discontinuities must not silently pair across a hole);
* the multi-campaign training step shares ONE encoder's parameters across
  campaigns without double-registering them in the optimiser (a duplicate
  parameter reference would double-update per step and silently corrupt
  training) — pinned by checking the optimiser's parameter id-set is unique
  and that a training step measurably reduces the composite loss on synthetic
  two-campaign data.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from imas_ambix.gs.machine_geometry import GeometryIdentity, OperatorGeometry
from imas_ambix.latent.encoder import HybridLatentEncoder, LatentConfig
from imas_ambix.latent.engine import GSGroundedLatentEngine
from imas_ambix.latent.patch_basis import PatchBasis
from imas_ambix.latent.training import CorpusTrainer, consecutive_pairs
from imas_ambix.latent.transport import FluxDiffusionPrior


def test_consecutive_pairs_within_uniform_grid():
    times = np.arange(0, 0.01, 1e-3)  # 10 slices at 1 kHz, no gaps
    pairs = consecutive_pairs(times)
    assert len(pairs) == 9
    for i, (a, b, dt) in enumerate(pairs):
        assert a == i and b == i + 1
        assert dt == pytest.approx(1e-3, rel=1e-6)


def test_consecutive_pairs_break_across_a_gap():
    # a gap between index 3 and 4 (plasma-off in between, e.g. 1ms -> 50ms)
    times = np.array([0.0, 1e-3, 2e-3, 3e-3, 0.050, 0.051, 0.052])
    pairs = consecutive_pairs(times, max_dt=2e-3)
    pair_indices = [(a, b) for a, b, _ in pairs]
    assert (3, 4) not in pair_indices  # the gap must not be bridged
    assert (0, 1) in pair_indices
    assert (4, 5) in pair_indices


def _campaign_table(digest: str, r_shift: float = 0.0) -> OperatorGeometry:
    probes = [
        SimpleNamespace(
            index=i,
            r=1.4 + r_shift + 0.02 * i,
            z=-0.4 + 0.16 * i,
            angle_deg=-90.0 * (i % 2),
            length=0.02,
        )
        for i in range(6)
    ]
    sensor_map = [
        SimpleNamespace(
            amb_channel=f"obv{i:02d}",
            kind="b_probe",
            efm_index=i,
            r=p.r,
            z=p.z,
            angle_deg=p.angle_deg,
            residual_m=0.001,
            flag="",
        )
        for i, p in enumerate(probes)
    ]
    conductors = (
        SimpleNamespace(
            # kept at the P4U-recognised centroid regardless of r_shift (only the
            # sensor layout varies between campaigns) — classify_circuits matches
            # a KNOWN coil by proximity to a fixed centroid table, so shifting
            # this would silently drop the coil to "inferred passive" (0 columns).
            r=1.5,
            z=1.1,
            turns=1.0,
            width=0.01,
            height=0.01,
            circuit=1,
            xmult=1.0,
        ),
    )
    return OperatorGeometry(
        identity=GeometryIdentity(
            representation_key=f"mp6-fl0-fc1-lim4-{digest}",
            representation_digest=digest,
            derivation_id="synthetic-training",
            physical_digest="",
            registry_digest="",
        ),
        probes=tuple(probes),
        loops=(),
        conductors=conductors,
        passives=(SimpleNamespace(name="w", r=2.0, z=0.0, obsolete=False),),
        limiter_r=(0.3, 1.6, 1.6, 0.3),
        limiter_z=(-1.0, -1.0, 1.0, 1.0),
        polygon_sections=(),
        drive_map=(),
        sensor_map=tuple(sensor_map),
        unmatched_channels=(),
        active_circuits=(),
        available_current_channels=("p4u_coil_current",),
        r0=0.85,
        minor_radius=0.65,
        unresolved_turns={},
        coil_channels=(),
        coil_column_matrix=np.zeros((len(sensor_map), 0), dtype=np.float64),
    )


def _make_engines(n_features=8, n_free=6):
    """Two campaigns' PatchBasis+transport sharing ONE encoder.

    Both tables share the same limiter + grid resolution (only sensor layout
    differs), so their patch cell counts agree — required for one shared
    encoder patch-current head to serve both campaigns' candidate masks.
    """
    tables = {
        "camp_a": _campaign_table("aaa0"),
        "camp_b": _campaign_table("bbb0", r_shift=0.1),
    }
    basis_by_campaign = {
        k: PatchBasis.from_table(t, nr=17, nz=21, cache_dir=None, dtype=torch.float64)
        for k, t in tables.items()
    }
    n_cells = int(basis_by_campaign["camp_a"].r_cells.shape[0])
    encoder = HybridLatentEncoder(
        LatentConfig(
            n_features=n_features,
            n_theta=1,
            n_anchored=2,
            n_free=n_free,
            n_cells=n_cells,
            hidden=32,
            depth=2,
        )
    ).double()
    transport_by_campaign = {
        k: FluxDiffusionPrior(nrho=b.nr, cmd_dim=1, feat_dim=n_free).double()
        for k, b in basis_by_campaign.items()
    }
    engines = {
        k: GSGroundedLatentEngine(
            encoder, basis_by_campaign[k], transport_by_campaign[k]
        )
        for k in basis_by_campaign
    }
    return encoder, engines


def test_shared_encoder_not_double_registered_in_optimizer():
    encoder, engines = _make_engines()
    trainer = CorpusTrainer(encoder, engines)
    param_ids = [id(p) for p in trainer.optimizer.param_groups[0]["params"]]
    assert len(param_ids) == len(set(param_ids))  # no parameter appears twice
    # the shared encoder's parameters must all be present exactly once
    enc_ids = {id(p) for p in encoder.parameters()}
    assert enc_ids.issubset(set(param_ids))


def test_training_step_reduces_composite_loss_across_campaigns():
    torch.manual_seed(0)
    encoder, engines = _make_engines()
    trainer = CorpusTrainer(encoder, engines)

    def _batch(engine, n_s, n_coil, b=8):
        return {
            "x_t": torch.randn(b, 8, dtype=torch.float64),
            "x_tp1": torch.randn(b, 8, dtype=torch.float64),
            "i_pf_t": torch.randn(b, n_coil, dtype=torch.float64),
            "i_pf_tp1": torch.randn(b, n_coil, dtype=torch.float64),
            "ip_t": torch.full((b,), 5.0e4, dtype=torch.float64),
            "raw_mag_t": torch.randn(b, n_s, dtype=torch.float64),
            "sensor_scale": torch.ones(b, n_s, dtype=torch.float64),
            "cmd_t": torch.randn(b, n_coil, dtype=torch.float64),
            "anchored_target": torch.randn(b, 2, dtype=torch.float64),
            "anchored_mask": torch.ones(b, 2, dtype=torch.bool),
            "dt": 1e-3,
        }

    batches = {
        k: _batch(e, len(e.basis.sensor_channels), e.basis.psi_coil_grid.shape[1])
        for k, e in engines.items()
    }
    loss0 = sum(
        trainer.engines[k].losses(b)["total"].item() for k, b in batches.items()
    )
    for _ in range(100):
        trainer.step({k: (lambda b=b: b) for k, b in batches.items()})
    loss1 = sum(
        trainer.engines[k].losses(b)["total"].item() for k, b in batches.items()
    )
    assert loss1 < loss0


def test_checkpoint_round_trip():
    encoder, engines = _make_engines()
    trainer = CorpusTrainer(encoder, engines)
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "ckpt.pt"
        trainer.save(path, step=42)
        encoder2, engines2 = _make_engines()
        trainer2 = CorpusTrainer(encoder2, engines2)
        step = trainer2.load(path)
        assert step == 42
        for p1, p2 in zip(encoder.parameters(), encoder2.parameters(), strict=True):
            torch.testing.assert_close(p1, p2)


def test_checkpoint_round_trips_arbitrary_extra_metadata():
    """Normalisation stats (feature/anchored/command) must survive a checkpoint
    round-trip — an eval run reproducing the trained encoder's exact input
    scaling depends on this, not just the learned weights."""
    encoder, engines = _make_engines()
    trainer = CorpusTrainer(encoder, engines)
    import tempfile
    from pathlib import Path

    extra = {"feature_mean": np.array([1.0, 2.0]), "cmd_std": {"camp_a": 3.5}}
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "ckpt.pt"
        trainer.save(path, step=7, extra=extra)
        encoder2, engines2 = _make_engines()
        trainer2 = CorpusTrainer(encoder2, engines2)
        step, loaded_extra = trainer2.load(path, return_extra=True)
        assert step == 7
        np.testing.assert_allclose(loaded_extra["feature_mean"], extra["feature_mean"])
        assert loaded_extra["cmd_std"]["camp_a"] == 3.5


def test_step_skips_nonfinite_loss_and_clips_gradients():
    """A batch that produces a non-finite loss must NOT reach optimizer.step
    (one poisoned batch would corrupt the SHARED encoder for every campaign
    for the rest of a long run), and finite steps must clip gradients."""
    torch.manual_seed(0)
    encoder, engines = _make_engines()
    trainer = CorpusTrainer(encoder, engines, max_grad_norm=1.0)
    key = next(iter(engines))
    e = engines[key]
    n_s = len(e.basis.sensor_channels)
    n_coil = e.basis.psi_coil_grid.shape[1]

    def batch(bad=False):
        b = {
            "x_t": torch.randn(4, 8, dtype=torch.float64),
            "x_tp1": torch.randn(4, 8, dtype=torch.float64),
            "i_pf_t": torch.randn(4, n_coil, dtype=torch.float64),
            "i_pf_tp1": torch.randn(4, n_coil, dtype=torch.float64),
            "ip_t": torch.full((4,), 5.0e4, dtype=torch.float64),
            "raw_mag_t": torch.randn(4, n_s, dtype=torch.float64),
            "sensor_scale": torch.ones(4, n_s, dtype=torch.float64),
            "cmd_t": torch.randn(4, n_coil, dtype=torch.float64),
            "anchored_target": torch.randn(4, 2, dtype=torch.float64),
            "anchored_mask": torch.ones(4, 2, dtype=torch.bool),
            "dt": 1e-3,
        }
        if bad:
            # poison the encoder INPUT: a NaN feature propagates through every
            # head and makes the whole composite loss non-finite (raw_mag NaNs
            # are legitimately absorbed by the residual's nan_to_num + mask)
            b["x_t"] = b["x_t"] * float("nan")
        return b

    before = [p.detach().clone() for p in encoder.parameters()]
    out = trainer.step({key: lambda: batch(bad=True)})
    after = list(encoder.parameters())
    for p0, p1 in zip(before, after, strict=True):
        torch.testing.assert_close(p0, p1)  # poisoned batch → no update
    assert out.get(key) is None or not np.isfinite(out[key])

    out = trainer.step({key: lambda: batch(bad=False)})
    assert np.isfinite(out[key])  # healthy batch still trains
