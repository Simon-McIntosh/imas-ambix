"""Conditioning loaders: resample alignment, missingness, leakage ban."""

from __future__ import annotations

import numpy as np
import pytest

from imas_ambix.camdyn.conditioning import (
    BANNED_CONDITIONING_SOURCES,
    CONDITIONING_CHANNELS,
    DALPHA_BURST_CHANNEL,
    ConditioningChannel,
    assert_fast_inputs_are_not_probe_targets,
    assert_no_leakage_sources,
    load_conditioning,
    resample_to_frames,
)
from imas_ambix.camdyn.dataset import level1_shot_path


def test_resample_zero_order_hold_is_causal():
    sig_t = np.array([0.0, 1.0, 2.0, 3.0])
    sig_v = np.array([10.0, 20.0, 30.0, 40.0])
    # frame at 1.5 should hold the sample at t=1.0 (causal — no future leak)
    frame_t = np.array([0.5, 1.0, 1.5, 2.9, 3.0])
    held, missing = resample_to_frames(sig_t, sig_v, frame_t)
    assert held.tolist() == [10.0, 20.0, 20.0, 30.0, 40.0]
    assert missing.tolist() == [0.0, 0.0, 0.0, 0.0, 0.0]


def test_resample_flags_frames_before_record_start():
    sig_t = np.array([1.0, 2.0])
    sig_v = np.array([5.0, 6.0])
    frame_t = np.array([0.0, 0.5, 1.0, 2.0])
    held, missing = resample_to_frames(sig_t, sig_v, frame_t)
    assert missing.tolist() == [1.0, 1.0, 0.0, 0.0]
    # before-record frames hold the first sample value
    assert held[0] == 5.0


def test_resample_empty_signal_all_missing():
    held, missing = resample_to_frames(np.array([]), np.array([]), np.array([0.0, 1.0]))
    assert missing.tolist() == [1.0, 1.0]
    assert held.tolist() == [0.0, 0.0]


def test_resample_nan_samples_flagged_missing():
    sig_t = np.array([0.0, 1.0, 2.0])
    sig_v = np.array([1.0, np.nan, 3.0])
    frame_t = np.array([0.0, 1.0, 2.0])
    held, missing = resample_to_frames(sig_t, sig_v, frame_t)
    assert missing.tolist() == [0.0, 1.0, 0.0]
    assert held[1] == 0.0  # NaN replaced by fill


def test_leakage_ban_rejects_efit_esm_xdc():
    assert frozenset({"efm", "esm", "xdc"}) == BANNED_CONDITIONING_SOURCES
    for bad in ("efm", "esm", "xdc"):
        with pytest.raises(ValueError, match="leakage"):
            assert_no_leakage_sources(["amc", bad])


def test_default_channel_set_has_no_banned_sources():
    sources = {c.source for c in CONDITIONING_CHANNELS}
    assert not (sources & BANNED_CONDITIONING_SOURCES)
    # and it covers the actuator pillars + scalars
    assert sources == {"amc", "anb", "aga", "ane", "xim"}


def test_default_channel_set_excludes_dalpha_probe_target():
    keys = {c.key for c in CONDITIONING_CHANNELS}
    assert "dalpha_integrated" not in keys
    assert DALPHA_BURST_CHANNEL.key in keys
    assert not any(c.is_probe_target for c in CONDITIONING_CHANNELS)


def test_load_conditioning_aligns_to_frames(synthetic_corpus):
    sid = 90001
    lpath = level1_shot_path(sid, level1_dir=synthetic_corpus["level1_dir"])
    n = synthetic_corpus["n_frames"][sid]
    dt = 1.0 / 600.0
    frame_time = 0.01 + dt * np.arange(n)
    cond = load_conditioning(lpath, frame_time, sid)
    assert cond.values.shape == (n, len(CONDITIONING_CHANNELS))
    assert cond.missing.shape == (n, len(CONDITIONING_CHANNELS))
    assert len(cond.channel_keys) == len(CONDITIONING_CHANNELS)
    # plasma current present (amc spans the whole window) → not missing
    j = cond.channel_keys.index("plasma_current")
    assert cond.missing[:, j].sum() == 0
    # value is in physical Amps (kA fixture × 1e3 scale) → O(1e5)
    assert np.median(np.abs(cond.values[:, j])) > 1e4


def test_load_conditioning_missingness_for_late_starting_beam(synthetic_corpus):
    # anb (beam) starts at frame 5 in the fixture → first frames flagged missing
    sid = 90001
    lpath = level1_shot_path(sid, level1_dir=synthetic_corpus["level1_dir"])
    n = synthetic_corpus["n_frames"][sid]
    dt = 1.0 / 600.0
    frame_time = 0.01 + dt * np.arange(n)
    cond = load_conditioning(lpath, frame_time, sid)
    j = cond.channel_keys.index("nbi_tot_sum_power")
    # at least the first frame should be missing (beam record starts later)
    assert cond.missing[0, j] == 1.0


def test_load_conditioning_absent_shot_all_missing():
    frame_time = np.array([0.0, 0.001, 0.002])
    cond = load_conditioning(None, frame_time, 12345)
    assert cond.missing.all()
    assert (cond.values == 0).all()


def test_include_dalpha_adds_separable_probe_channel(synthetic_corpus):
    sid = 90001
    lpath = level1_shot_path(sid, level1_dir=synthetic_corpus["level1_dir"])
    n = synthetic_corpus["n_frames"][sid]
    dt = 1.0 / 600.0
    frame_time = 0.01 + dt * np.arange(n)
    cond = load_conditioning(lpath, frame_time, sid, include_dalpha=True)
    assert "dalpha_integrated" in cond.channel_keys
    assert cond.values.shape[1] == len(CONDITIONING_CHANNELS) + 1


def test_load_conditioning_rejects_banned_channel(synthetic_corpus):
    sid = 90001
    lpath = level1_shot_path(sid, level1_dir=synthetic_corpus["level1_dir"])
    frame_time = np.array([0.01, 0.02])
    bad_channels = (
        *CONDITIONING_CHANNELS,
        ConditioningChannel("psi", "efm", "psi", ""),
    )
    with pytest.raises(ValueError, match="leakage"):
        load_conditioning(lpath, frame_time, sid, channels=bad_channels)


def test_fast_input_cannot_also_be_probe_target():
    bad = ConditioningChannel(
        "ambiguous_dalpha",
        "xim",
        "da_hm10_t",
        "a.u.",
        is_probe_target=True,
        is_fast_input=True,
    )
    with pytest.raises(ValueError, match="cannot also be diagnostic probe"):
        assert_fast_inputs_are_not_probe_targets((bad,))
