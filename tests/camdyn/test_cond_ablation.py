"""Tests for the conditioning ablation.

Covers the suppression contract (the channel-keep mask + the
"actuator absent" zero-value/raise-missing rewrite), the
``CondMaskedTrainer`` chokepoint override, and the matched-budget
three-arm comparison end-to-end on tiny CPU checkpoints.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from imas_ambix.camdyn.cond_ablation import (
    KEEP_FULL,
    KEEP_IP_NE,
    KEEP_NONE,
    REGIME_KEEP,
    CondMaskedTrainer,
    channel_keep_mask,
    suppress_conditioning,
)
from imas_ambix.camdyn.conditioning import CONDITIONING_CHANNELS
from imas_ambix.camdyn.dataset import discover_token_shots
from imas_ambix.camdyn.model import N_COND_CHANNELS, CamdynConfig
from imas_ambix.camdyn.splits import CamdynSplit
from imas_ambix.camdyn.train import TrainConfig


def _write_split(tmp_path: Path, sc) -> Path:
    ids = sc["shot_ids"]
    split = CamdynSplit(
        train=[ids[0]],
        val=[ids[1]],
        held_out=[ids[0], ids[1]],
        n_token_shots=len(ids),
    )
    path = tmp_path / "ablation_split.json"
    path.write_text(json.dumps(split.to_dict()), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Keep-mask + suppression contract
# ---------------------------------------------------------------------------


def test_keep_mask_regimes():
    full = channel_keep_mask(KEEP_FULL)
    none = channel_keep_mask(KEEP_NONE)
    ipne = channel_keep_mask(KEEP_IP_NE)
    assert full.shape == (N_COND_CHANNELS,)
    assert full.all()
    assert not none.any()
    assert ipne.sum() == 2  # exactly plasma_current + ne_line_integrated
    keys = [c.key for c in CONDITIONING_CHANNELS]
    kept = {keys[i] for i in np.where(ipne)[0]}
    assert kept == {"plasma_current", "ne_line_integrated"}


def test_regime_keep_table_matches_constants():
    assert REGIME_KEEP["full"] == KEEP_FULL
    assert REGIME_KEEP["ip_ne"] == KEEP_IP_NE
    assert REGIME_KEEP["none"] == KEEP_NONE


def test_suppress_conditioning_zeroes_values_and_raises_missing():
    b, f, c = 2, 6, N_COND_CHANNELS
    cv = np.random.default_rng(0).standard_normal((b, f, c)).astype(np.float32)
    cm = np.zeros((b, f, c), dtype=np.float32)
    keep = channel_keep_mask(KEEP_IP_NE)
    cv2, cm2 = suppress_conditioning(cv, cm, keep)
    drop = ~keep
    # suppressed channels: value zeroed, missing raised to 1
    assert np.all(cv2[..., drop] == 0.0)
    assert np.all(cm2[..., drop] == 1.0)
    # kept channels untouched
    np.testing.assert_array_equal(cv2[..., keep], cv[..., keep])
    np.testing.assert_array_equal(cm2[..., keep], cm[..., keep])
    # inputs not mutated
    assert not np.all(cv[..., drop] == 0.0)


def test_suppress_none_zeroes_everything():
    cv = np.ones((1, 3, N_COND_CHANNELS), dtype=np.float32)
    cm = np.zeros((1, 3, N_COND_CHANNELS), dtype=np.float32)
    cv2, cm2 = suppress_conditioning(cv, cm, channel_keep_mask(KEEP_NONE))
    assert np.all(cv2 == 0.0)
    assert np.all(cm2 == 1.0)


def test_cond_masked_trainer_overrides_chokepoint():
    """The subclass must suppress RAW conditioning before the parent z-scores."""
    torch = pytest.importorskip("torch")
    cfg = TrainConfig(
        model=CamdynConfig(temporal_attention=True, dim=16, n_layers=1, n_heads=2),
        num_workers=0,
        device="cpu",
    )
    tr = CondMaskedTrainer(cfg, KEEP_IP_NE)
    # identity stats so z-score is a no-op and we can read suppression directly
    tr._cond_stats = (
        np.zeros(N_COND_CHANNELS, dtype=np.float32),
        np.ones(N_COND_CHANNELS, dtype=np.float32),
    )
    nf = 4
    arr = {
        "tokens": np.zeros((1, nf, 16, 16), dtype=np.int64),
        "visible": np.ones((1, nf, 16, 16), dtype=bool),
        "loss_mask": np.zeros((1, nf, 16, 16), dtype=bool),
        "cond_values": np.full((1, nf, N_COND_CHANNELS), 3.0, dtype=np.float32),
        "cond_missing": np.zeros((1, nf, N_COND_CHANNELS), dtype=np.float32),
        "dt": np.zeros((1, nf), dtype=np.float32),
        "valid": np.ones((1, nf), dtype=bool),
    }
    t = tr._batch_to_tensors(arr, torch, torch.device("cpu"))
    cv = t["cond_values"].cpu().numpy()
    cm = t["cond_missing"].cpu().numpy()
    keep = channel_keep_mask(KEEP_IP_NE)
    drop = ~keep
    assert np.all(cv[..., drop] == 0.0)  # suppressed → 0 after z-score (mu=0)
    assert np.all(cm[..., drop] == 1.0)
    assert np.all(cv[..., keep] == 3.0)  # kept channel survives


# ---------------------------------------------------------------------------
# End-to-end matched-budget comparison (CPU, tiny arms)
# ---------------------------------------------------------------------------


def test_compare_conditioning_end_to_end_cpu(synthetic_corpus, tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from imas_ambix.camdyn import cond_ablation as ca
    from imas_ambix.camdyn.masking import NAMED_GEOMETRIES
    from imas_ambix.camdyn.model import CamdynModel

    sc = synthetic_corpus
    split_path = _write_split(tmp_path, sc)

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
            temporal_attention=True,
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
        num_workers=0,
        eval_windows=4,
        max_heldout_shots=None,
        seed=0,
        split_path=str(split_path),
        device="cpu",
    )
    cond_stats = [[0.0] * N_COND_CHANNELS, [1.0] * N_COND_CHANNELS]
    # all three arms share one tiny dynamics checkpoint (the comparison only
    # varies the RUNTIME suppression mask per arm — exactly the contract)
    ckpts = {}
    for arm in ("full", "ip_ne", "none"):
        model = CamdynModel.from_config(cfg.model)
        p = tmp_path / f"{arm}.pt"
        torch.save(
            {
                "config": cfg.to_dict(),
                "model_state": model.module.state_dict(),
                "cond_stats": cond_stats,
            },
            p,
        )
        ckpts[arm] = p

    res = ca.compare_conditioning(ckpts, split_path=str(split_path), device="cpu")

    assert set(res["regimes"]) == {"full", "ip_ne", "none"}
    ho = res["held_out"]
    for arm in ("full", "ip_ne", "none"):
        assert "masked_nll" in ho[arm]
        assert "masked_top1" in ho[arm]
    # paired-vs-none CIs present with aligned pair counts
    assert "full_vs_none_top1" in ho and "ip_ne_vs_none_top1" in ho
    assert ho["full_vs_none_top1"]["n_pairs"] > 0
    # the named-geometry suite (incl. frontier_half) all scored
    assert set(res["named_geometry"]) == set(NAMED_GEOMETRIES)
    assert "frontier_half" in res["named_geometry"]
    # verdict flags are booleans
    for k in (
        "full_beats_none_nll",
        "ip_ne_beats_none_nll",
        "frontier_full_beats_none_nll",
    ):
        assert isinstance(res["verdict"][k], bool)
