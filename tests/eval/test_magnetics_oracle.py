"""Unit tests for the absolute-magnetics equilibrium oracle.

All synthetic — no /work, no GPU, no staged stores.  They check the pieces a
caller relies on: the corpus-level standardisation PRESERVES absolute inter-shot
scale (the whole point of the oracle), the skill / verdict math is correct on
known arrays, the held-out cohorts are wired into the forced-test set, and a tiny
probe trains a few steps on CPU and produces well-formed predictions.
"""

from __future__ import annotations

import numpy as np
import pytest

from imas_ambix.eval import magnetics_oracle as mo

# --- synthetic example builders ---------------------------------------------


def _example(shot_id, raw, target, mask):
    return {
        "shot_id": int(shot_id),
        "raw": np.asarray(raw, np.float64),
        "target": np.asarray(target, np.float32),
        "mask": np.asarray(mask, bool),
    }


def _synthetic_examples(n, n_steps, n_channels, target_dim, seed=0):
    """A small batch where the target is a clean linear function of the raw mean.

    The probe should be able to fit this, so post-train RMSE beats the baseline.
    """
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        raw = rng.normal(size=(n_steps, n_channels)) + rng.normal() * 3.0
        # target is a deterministic linear readout of channel means (recoverable).
        feat = raw.mean(axis=0)
        w = np.linspace(0.1, 0.5, n_channels)
        base = float(feat @ w)
        target = np.array([base + 0.5 * d for d in range(target_dim)], np.float32)
        mask = np.ones(target_dim, bool)
        out.append(_example(1000 + i, raw, target, mask))
    return out


# --- forced-test cohort wiring ----------------------------------------------


def test_forced_test_shots_include_gate_and_standing_held_out():
    forced = set(mo.FORCED_TEST_SHOTS)
    assert set(mo.GATE_COHORT) <= forced
    assert set(mo.STANDING_HELD_OUT) <= forced
    # sorted, de-duplicated
    assert list(mo.FORCED_TEST_SHOTS) == sorted(forced)


# --- corpus-level standardisation preserves ABSOLUTE inter-shot scale --------


def test_fit_channel_stats_preserves_absolute_inter_shot_scale():
    """Two shots offset by a constant must KEEP their relative offset after the
    SHARED corpus-level standardisation (a per-shot z-score would erase it)."""
    n_steps, n_channels = 6, 4
    rng = np.random.default_rng(1)
    base = rng.normal(size=(n_steps, n_channels))
    low = _example(1, base + 0.0, np.zeros(4), np.ones(4, bool))
    high = _example(2, base + 10.0, np.zeros(4), np.ones(4, bool))
    examples = [low, high]

    stats = mo.fit_channel_stats(examples, n_channels)
    x, valid = mo.batch_raw_arrays(examples, stats, n_steps)

    assert x.shape == (2, n_steps, n_channels)
    assert valid.dtype == np.float32 and valid.all()
    # the high shot stays strictly above the low shot on every channel mean —
    # absolute scale survived standardisation.
    low_mean = x[0].mean(axis=0)
    high_mean = x[1].mean(axis=0)
    assert np.all(high_mean > low_mean)
    # and the gap reflects the original +10 offset scaled by 1/std (not collapsed)
    assert np.all((high_mean - low_mean) > 1e-3)


def test_fit_channel_stats_nan_safe_and_masked():
    n_steps, n_channels = 5, 3
    raw = np.full((n_steps, n_channels), np.nan)
    raw[:, 0] = np.arange(n_steps)  # only channel 0 has finite data
    ex = _example(7, raw, np.zeros(2), np.ones(2, bool))
    stats = mo.fit_channel_stats([ex], n_channels)
    # channel with no finite TRAIN data -> mean 0, std 1
    assert stats.mean[1] == 0.0 and stats.std[1] == 1.0
    assert stats.mean[2] == 0.0 and stats.std[2] == 1.0
    # finite channel got a real mean
    assert abs(stats.mean[0] - np.arange(n_steps).mean()) < 1e-9

    x, valid = mo.batch_raw_arrays([ex], stats, n_steps)
    # NaN steps map to the channel centre (0 after standardisation) and mask 0
    assert np.isfinite(x).all()
    assert valid[0, :, 0].all()  # channel 0 finite
    assert not valid[0, :, 1].any()  # channel 1 all-NaN -> mask zero


def test_batch_raw_arrays_handles_ragged_channels():
    """An example with fewer channels than the corpus max is zero-padded."""
    stats = mo.ChannelStats(mean=np.zeros(5), std=np.ones(5), n_channels=5)
    ex = _example(1, np.ones((4, 3)), np.zeros(2), np.ones(2, bool))
    x, valid = mo.batch_raw_arrays([ex], stats, n_steps=4)
    assert x.shape == (1, 4, 5)
    assert valid[0, :, :3].all()  # present channels valid
    assert not valid[0, :, 3:].any()  # padded channels invalid


# --- skill / baseline / verdict math ----------------------------------------


def test_per_component_and_baseline_rmse():
    pred = np.array([[1.0, 2.0], [1.0, 2.0]])
    y = np.array([[1.5, 2.0], [0.5, 2.0]])  # err 0.5 on comp 0, 0 on comp 1
    mask = np.ones((2, 2), bool)
    rmse = mo.per_component_rmse(pred, y, mask)
    assert abs(rmse[0] - 0.5) < 1e-9
    assert abs(rmse[1] - 0.0) < 1e-9

    ytr = np.array([[0.0], [2.0]])  # train mean 1.0
    mtr = np.ones((2, 1), bool)
    yte = np.array([[1.0], [1.0]])  # baseline err 0
    mte = np.ones((2, 1), bool)
    base = mo.mean_predictor_rmse(ytr, mtr, yte, mte)
    assert abs(base[0] - 0.0) < 1e-9


def test_oracle_skill_and_verdict_structure():
    names = ("axis_R", "axis_Z", "xpt_R", "xpt_Z")
    # probe half the baseline error everywhere -> skill +0.5, beats at 1.3x
    rmse_probe = np.array([1.0, 1.0, 1.0, 1.0])
    rmse_base = np.array([2.0, 2.0, 2.0, 2.0])
    skill = mo.oracle_skill(rmse_probe, rmse_base, names, names)
    assert abs(skill - 0.5) < 1e-9

    verd = mo.verdict(rmse_probe, rmse_base, names, ratio_threshold=1.3)
    assert verd.feasible is True
    assert abs(verd.headline_skill - 0.5) < 1e-9
    assert abs(verd.axis_skill - 0.5) < 1e-9
    assert abs(verd.xpt_skill - 0.5) < 1e-9
    assert len(verd.components) == 4
    for row in verd.components:
        assert row["beats_baseline"] is True
        assert abs(row["skill"] - 0.5) < 1e-9
    d = verd.to_dict()
    assert d["feasible"] is True and d["headline_skill"] is not None


def test_verdict_infeasible_when_one_component_misses():
    names = ("axis_R", "axis_Z", "xpt_R", "xpt_Z")
    rmse_probe = np.array([1.0, 1.0, 1.0, 1.9])  # xpt_Z barely beats -> < /1.3 fails
    rmse_base = np.array([2.0, 2.0, 2.0, 2.0])
    verd = mo.verdict(rmse_probe, rmse_base, names, ratio_threshold=1.3)
    assert verd.feasible is False
    # the failing component is the X-point Z
    failing = [r["component"] for r in verd.components if r["beats_baseline"] is False]
    assert failing == ["xpt_Z"]


def test_oracle_skill_nan_safe_returns_none():
    names = ("axis_R", "axis_Z")
    rmse_probe = np.array([np.nan, np.nan])
    rmse_base = np.array([np.nan, 0.0])
    assert mo.oracle_skill(rmse_probe, rmse_base, names, names) is None


# --- tiny end-to-end probe train on CPU (torch) -----------------------------


def test_train_and_evaluate_tiny_probe_cpu():
    pytest.importorskip("torch")
    n_steps, n_channels, target_dim = 6, 4, 4
    tr = _synthetic_examples(24, n_steps, n_channels, target_dim, seed=0)
    te = _synthetic_examples(8, n_steps, n_channels, target_dim, seed=99)

    stats = mo.fit_channel_stats(tr, n_channels)
    model, tstats = mo.train_probe(
        tr,
        stats,
        n_steps=n_steps,
        target_dim=target_dim,
        epochs=5,
        batch_size=8,
        lr=1e-3,
        device="cpu",
        seed=0,
        d_model=32,
        n_layers=1,
        n_heads=4,
        dropout=0.0,
    )
    assert model.n_parameters() > 0
    assert tstats.mean.shape == (target_dim,)

    pred, yte, mte = mo.evaluate(
        model, te, stats, tstats, n_steps=n_steps, device="cpu", batch_size=8
    )
    assert pred.shape == (len(te), target_dim)
    assert yte.shape == pred.shape
    assert np.isfinite(pred).all()

    # the probe should beat the mean-predictor on this recoverable synthetic task
    ytr = np.stack([ex["target"] for ex in tr]).astype(np.float32)
    mtr = np.stack([ex["mask"] for ex in tr]).astype(bool)
    rmse_probe = mo.per_component_rmse(pred, yte, mte)
    rmse_base = mo.mean_predictor_rmse(ytr, mtr, yte, mte)
    skill = mo.oracle_skill(
        rmse_probe,
        rmse_base,
        [f"c{d}" for d in range(target_dim)],
        [f"c{d}" for d in range(target_dim)],
    )
    assert skill is not None and skill > 0.0
