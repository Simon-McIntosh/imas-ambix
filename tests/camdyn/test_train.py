"""Tests for the camdyn training loop + W1 eval.

The heavy end-to-end test runs a 3-step CPU smoke against the synthetic
corpus fixture (a tiny model, a tiny split) so it catches shape/wiring
bugs without any GPU.  Lighter tests cover config (de)serialisation and
the batch-assembly path.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from imas_ambix.camdyn.dataset import discover_token_shots
from imas_ambix.camdyn.masking import ClipMaskConfig
from imas_ambix.camdyn.model import N_COND_CHANNELS, CamdynConfig
from imas_ambix.camdyn.splits import CamdynSplit
from imas_ambix.camdyn.train import (
    TrainConfig,
    Trainer,
    _agg,
    _assemble_batch,
    _bootstrap_sanity,
)

# ---------------------------------------------------------------------------
# TrainConfig (de)serialisation
# ---------------------------------------------------------------------------


def test_train_config_roundtrip():
    cfg = TrainConfig(
        model=CamdynConfig(temporal_attention=True, dim=64),
        batch_size=8,
        max_steps=10,
    )
    d = cfg.to_dict()
    cfg2 = TrainConfig.from_dict(d)
    assert cfg2.model.temporal_attention is True
    assert cfg2.model.dim == 64
    assert cfg2.batch_size == 8
    assert cfg2.betas == (0.9, 0.95)


def test_train_config_loads_shipped_yaml():
    yaml = pytest.importorskip("yaml")  # noqa: F841
    cfgdir = Path(__file__).resolve().parents[2] / "imas_ambix" / "camdyn" / "configs"
    base = TrainConfig.load(cfgdir / "baseline_w1_v0.yaml")
    assert base.model.temporal_attention is False  # D1 baseline
    assert base.run_name == "baseline_w1_v0"
    smoke = TrainConfig.load(cfgdir / "smoke_cpu.yaml")
    assert smoke.device == "cpu"
    assert smoke.max_steps == 3


def test_baseline_config_is_matched_to_a_d2_arm():
    """The D1 config must differ from a D2 arm ONLY in the toggle."""
    yaml = pytest.importorskip("yaml")  # noqa: F841
    cfgdir = Path(__file__).resolve().parents[2] / "imas_ambix" / "camdyn" / "configs"
    d1 = TrainConfig.load(cfgdir / "baseline_w1_v0.yaml").model
    d2 = CamdynConfig.from_dict({**d1.to_dict(), "temporal_attention": True})
    a, b = d1.to_dict(), d2.to_dict()
    diff = {k for k in a if a[k] != b[k]}
    assert diff == {"temporal_attention"}


# ---------------------------------------------------------------------------
# Batch assembly
# ---------------------------------------------------------------------------


def test_assemble_batch_shapes_and_mask_complement(synthetic_corpus):
    sc = synthetic_corpus
    specs = discover_token_shots(
        token_root=sc["token_root"],
        level1_dir=sc["level1_dir"],
        shot_ids=sc["shot_ids"],
        read_n_frames=True,
    )
    from imas_ambix.camdyn.dataset import FrameTokenDataset, FrameWindowConfig
    from imas_ambix.camdyn.train import _level1_for

    ds = FrameTokenDataset(specs, FrameWindowConfig(n_frames=6, stride=4), as_dict=True)
    windows = []
    for i in range(min(3, len(ds))):
        w = ds[i]
        w["level1_path"] = _level1_for(specs, int(w["shot_id"]))
        windows.append(w)
    rng = np.random.default_rng(0)
    arr = _assemble_batch(windows, ClipMaskConfig(), rng, progress=None)
    b = len(windows)
    assert arr["tokens"].shape == (b, 6, 16, 16)
    assert arr["cond_values"].shape == (b, 6, N_COND_CHANNELS)
    assert arr["cond_missing"].shape == (b, 6, N_COND_CHANNELS)
    # loss_mask must be the exact complement of visible
    np.testing.assert_array_equal(arr["loss_mask"], ~arr["visible"])


# ---------------------------------------------------------------------------
# Aggregation / bootstrap helpers
# ---------------------------------------------------------------------------


def test_agg_handles_empty_and_nonempty():
    assert _agg(np.array([]))["n"] == 0
    a = _agg(np.array([1.0, 2.0, 3.0]))
    assert a["mean"] == pytest.approx(2.0)
    assert a["n"] == 3


def test_bootstrap_sanity_within_arm_is_not_significant():
    """Two halves of the SAME arm must not favour either side."""
    rng = np.random.default_rng(0)
    nll = rng.standard_normal(2000) + 5.0
    ci = _bootstrap_sanity(nll)
    assert ci["favours_dynamics"] is False
    assert ci["lo"] < 0.0 < ci["hi"]  # straddles zero


# ---------------------------------------------------------------------------
# End-to-end CPU smoke — the real wiring test
# ---------------------------------------------------------------------------


def _write_split(tmp_path: Path, sc) -> Path:
    """A tiny split over the synthetic shots: train/val/held_out."""
    ids = sc["shot_ids"]
    split = CamdynSplit(
        train=[ids[0]],
        val=[ids[1]],
        held_out=[ids[2]] if len(ids) > 2 else [ids[1]],
        n_token_shots=len(ids),
    )
    path = tmp_path / "smoke_split.json"
    path.write_text(json.dumps(split.to_dict()), encoding="utf-8")
    return path


def test_cpu_smoke_train_and_w1_artifact(synthetic_corpus, tmp_path, monkeypatch):
    """3-step CPU train + held-out W1 eval against the synthetic corpus.

    Patches discovery + the default split so the trainer runs entirely on
    the fixture's tiny synthetic stores (no real corpus, no GPU).
    """
    pytest.importorskip("torch")
    sc = synthetic_corpus
    split_path = _write_split(tmp_path, sc)

    # Route discover_token_shots at the synthetic roots.
    import imas_ambix.camdyn.train as trainmod

    real_discover = discover_token_shots

    def _patched_discover(*, shot_ids=None, read_n_frames=False, **_kw):
        return real_discover(
            token_root=sc["token_root"],
            level1_dir=sc["level1_dir"],
            shot_ids=shot_ids,
            read_n_frames=read_n_frames,
        )

    monkeypatch.setattr(trainmod, "discover_token_shots", _patched_discover)

    cfg = TrainConfig(
        model=CamdynConfig(
            temporal_attention=False,
            dim=32,
            n_layers=2,
            n_heads=4,
            mlp_ratio=2.0,
            n_frames=6,
            cond_channels=N_COND_CHANNELS,
        ),
        n_frames=6,
        stride=4,
        batch_size=2,
        max_steps=3,
        warmup_frac=0.0,
        curriculum=False,
        num_workers=0,
        val_windows=4,
        eval_windows=4,
        log_every=1,
        val_every=1000,
        ckpt_every=1000,
        device="cpu",
        split_path=str(split_path),
        ckpt_root=str(tmp_path / "ckpt"),
        run_name="smoke",
        artifact_out=str(tmp_path / "w1.json"),
    )

    trainer = Trainer(cfg)
    w1 = trainer.train()

    # --- artifact written + structurally complete ---
    art = json.loads((tmp_path / "w1.json").read_text())
    assert art["arm"] == "D1-baseline"
    assert art["temporal_attention"] is False
    assert "held_out" in art
    assert "masked_nll" in art["held_out"]
    assert "masked_top1" in art["held_out"]
    assert "motion_weighted" in art["held_out"]
    # every frozen named geometry was scored
    from imas_ambix.camdyn.masking import NAMED_GEOMETRIES

    assert set(art["named_geometry"]) == set(NAMED_GEOMETRIES)
    # bootstrap CI helper was exercised
    assert "bootstrap_ci_sanity" in art
    # a checkpoint exists on disk
    assert Path(art["checkpoint"]).exists()
    # finite scores
    assert np.isfinite(w1["held_out"]["masked_nll"]["mean"])
    assert 0.0 <= w1["held_out"]["masked_top1"]["mean"] <= 1.0
