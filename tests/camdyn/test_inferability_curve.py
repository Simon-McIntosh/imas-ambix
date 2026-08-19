"""Tests for held-out quality versus visible camera area."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from imas_ambix.camdyn.dataset import discover_token_shots
from imas_ambix.camdyn.inferability_curve import (
    DEFAULT_VISIBLE_FRACTIONS,
    carry_forward_scores,
    clip_masks_for_batch,
    compact_clip_shape,
    evaluate_inferability,
    mean_bootstrap_ci,
)
from imas_ambix.camdyn.model import LFQ_BITS, N_COND_CHANNELS, CamdynConfig
from imas_ambix.camdyn.splits import CamdynSplit
from imas_ambix.camdyn.train import TrainConfig


def _write_split(tmp_path: Path, corpus) -> Path:
    ids = corpus["shot_ids"]
    split = CamdynSplit(
        train=[ids[0]],
        val=[ids[1]],
        held_out=[ids[0], ids[1]],
        n_token_shots=len(ids),
    )
    path = tmp_path / "curve_split.json"
    path.write_text(json.dumps(split.to_dict()), encoding="utf-8")
    return path


def test_compact_shapes_span_requested_range_with_small_discretisation_error():
    realised = []
    for fraction in DEFAULT_VISIBLE_FRACTIONS:
        height, width = compact_clip_shape(fraction)
        assert 1 <= height <= 16
        assert 1 <= width <= 16
        actual = height * width / 256.0
        assert abs(actual - fraction) <= 0.025
        realised.append(actual)
    assert realised == sorted(realised)
    assert realised[0] <= 0.06
    assert realised[-1] >= 0.48


def test_clip_masks_are_reproducible_static_and_position_varied():
    shot_ids = np.array([1001, 1002, 1003])
    first = clip_masks_for_batch(shot_ids, 6, 0.2, seed=9, sample_offset=0)
    second = clip_masks_for_batch(shot_ids, 6, 0.2, seed=9, sample_offset=0)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (3, 6, 16, 16)
    for sample in first:
        np.testing.assert_array_equal(sample[0], sample[-1])
    assert not np.array_equal(first[0], first[1])


def test_carry_forward_uniform_fallback_has_defined_nll_and_zero_top1():
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, 1 << LFQ_BITS, size=(4, 16, 16), dtype=np.int64)
    visible = np.zeros_like(tokens, dtype=bool)
    visible[:, 4:8, 4:8] = True
    loss_mask = ~visible
    nll, top1 = carry_forward_scores(
        tokens, visible, loss_mask, np.ones(tokens.shape[0], dtype=bool)
    )
    assert nll.size == int(loss_mask.sum())
    assert np.allclose(nll, LFQ_BITS * np.log(2.0))
    assert np.all(top1 == 0.0)


def test_mean_bootstrap_ci_is_reproducible_and_contains_mean():
    values = np.arange(1.0, 9.0)
    first = mean_bootstrap_ci(values, n_boot=500, seed=4)
    second = mean_bootstrap_ci(values, n_boot=500, seed=4)
    assert first == second
    assert first["lo"] < first["mean"] < first["hi"]
    assert first["n_shots"] == 8


def test_curve_end_to_end_on_synthetic_heldout(synthetic_corpus, tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from imas_ambix.camdyn.model import CamdynModel

    corpus = synthetic_corpus
    split_path = _write_split(tmp_path, corpus)
    import imas_ambix.camdyn.train as trainmod

    real_discover = discover_token_shots

    def patched_discover(*, shot_ids=None, read_n_frames=False, **_kwargs):
        return real_discover(
            token_root=corpus["token_root"],
            level1_dir=corpus["level1_dir"],
            shot_ids=shot_ids,
            read_n_frames=read_n_frames,
        )

    monkeypatch.setattr(trainmod, "discover_token_shots", patched_discover)

    def config(temporal):
        return TrainConfig(
            model=CamdynConfig(
                temporal_attention=temporal,
                dim=8,
                n_layers=1,
                n_heads=1,
                mlp_ratio=2.0,
                n_frames=4,
                cond_channels=N_COND_CHANNELS,
            ),
            n_frames=4,
            stride=8,
            batch_size=2,
            num_workers=0,
            eval_windows=2,
            max_heldout_shots=None,
            seed=0,
            split_path=str(split_path),
            device="cpu",
        )

    stats = [[0.0] * N_COND_CHANNELS, [1.0] * N_COND_CHANNELS]
    checkpoints = {}
    for arm, temporal in (("baseline", False), ("dynamics", True)):
        cfg = config(temporal)
        model = CamdynModel.from_config(cfg.model)
        path = tmp_path / f"{arm}.pt"
        torch.save(
            {
                "config": cfg.to_dict(),
                "model_state": model.module.state_dict(),
                "cond_stats": stats,
            },
            path,
        )
        checkpoints[arm] = path

    artifact = evaluate_inferability(
        checkpoints["baseline"],
        checkpoints["dynamics"],
        fractions=DEFAULT_VISIBLE_FRACTIONS,
        split_path=split_path,
        device="cpu",
        max_windows=2,
        n_boot=100,
    )
    assert len(artifact["fractions"]) == 5
    assert artifact["bootstrap"]["unit"] == "held-out shot"
    assert artifact["n_materialised_windows"] > 0
    for row in artifact["fractions"]:
        for arm in ("dynamics", "per_frame_baseline", "carry_forward"):
            assert row[arm]["masked_nll"]["n_shots"] > 0
            assert row[arm]["masked_top1"]["n_shots"] > 0
            assert np.isfinite(row[arm]["masked_nll"]["mean"])
        assert row["dynamics_vs_per_frame_baseline"]["masked_nll"]["n_shots"] > 0
