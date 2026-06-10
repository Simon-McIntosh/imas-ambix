"""Frame-grid token dataset: shape/dtype, timestamps, Δt, short-shot padding."""

from __future__ import annotations

import numpy as np
import pytest

from imas_ambix.camdyn.dataset import (
    FRAME_GRID,
    VOCAB_SIZE,
    FrameTokenDataset,
    FrameWindowConfig,
    discover_token_shots,
)
from tests.camdyn.conftest import (
    REAL_LEVEL1_PATH,
    REAL_TOKEN_PATH,
    REAL_TOKEN_SHOT,
)


def _dataset(corpus, n_frames=16, stride=8, drop_short=True, as_dict=False):
    specs = discover_token_shots(
        camera=corpus["camera"],
        vocab_version=corpus["vocab_version"],
        token_root=corpus["token_root"],
        level1_dir=corpus["level1_dir"],
        shot_ids=corpus["shot_ids"],
        read_n_frames=True,
    )
    cfg = FrameWindowConfig(n_frames=n_frames, stride=stride, drop_short=drop_short)
    return FrameTokenDataset(specs, cfg, as_dict=as_dict), specs


def test_window_shape_and_dtype(synthetic_corpus):
    ds, _ = _dataset(synthetic_corpus, n_frames=16)
    assert len(ds) > 0
    win = ds[0]
    assert win.tokens.shape == (16, *FRAME_GRID)
    assert win.tokens.dtype == np.int32
    assert win.tokens.min() >= 0
    assert win.tokens.max() < VOCAB_SIZE


def test_timestamps_and_dt_present_and_consistent(synthetic_corpus):
    ds, _ = _dataset(synthetic_corpus, n_frames=16)
    win = ds[0]
    assert win.frame_time.shape == (16,)
    assert win.dt.shape == (16,)
    assert not win.time_is_synthetic  # level-1 time exists in the fixture
    # forward Δt matches the diff of frame_time (last repeats)
    assert np.allclose(win.dt[:-1], np.diff(win.frame_time))
    assert win.dt[-1] == pytest.approx(win.dt[-2])
    # monotone time, positive Δt
    assert np.all(np.diff(win.frame_time) > 0)


def test_valid_frames_all_true_for_full_window(synthetic_corpus):
    ds, _ = _dataset(synthetic_corpus, n_frames=16)
    win = ds[0]
    assert win.valid_frames.all()


def test_short_shot_padding_when_not_dropped(synthetic_corpus):
    # shot 90003 has only 8 frames; with n_frames=16 and drop_short=False it
    # yields one padded window with a valid mask.
    ds, _ = _dataset(synthetic_corpus, n_frames=16, drop_short=False)
    shorts = [w for w in (ds[i] for i in range(len(ds))) if w.shot_id == 90003]
    assert len(shorts) == 1
    w = shorts[0]
    assert w.tokens.shape == (16, *FRAME_GRID)
    assert w.valid_frames.sum() == 8
    assert not w.valid_frames[8:].any()
    # padded tokens are zero
    assert np.all(w.tokens[8:] == 0)


def test_drop_short_excludes_short_shot(synthetic_corpus):
    ds, _ = _dataset(synthetic_corpus, n_frames=16, drop_short=True)
    sids = {ds[i].shot_id for i in range(len(ds))}
    assert 90003 not in sids
    assert 90001 in sids and 90002 in sids


def test_iter_is_shuffled_and_covers_all_windows(synthetic_corpus):
    ds, _ = _dataset(synthetic_corpus, n_frames=16, stride=8)
    via_iter = [(w.shot_id, w.start) for w in ds]
    via_map = [(ds[i].shot_id, ds[i].start) for i in range(len(ds))]
    assert sorted(via_iter) == sorted(via_map)
    assert len(via_iter) == len(ds)


def test_as_dict_emission(synthetic_corpus):
    ds, _ = _dataset(synthetic_corpus, n_frames=16, as_dict=True)
    d = ds[0]
    assert set(d) >= {"tokens", "frame_time", "dt", "valid_frames", "shot_id"}
    assert d["tokens"].shape == (16, *FRAME_GRID)


def test_tail_window_always_covered(synthetic_corpus):
    # n_frames=16, shot 90001 has 40 frames → last start must be 24 (40-16)
    ds, _ = _dataset(synthetic_corpus, n_frames=16, stride=8)
    starts_90001 = sorted(ds[i].start for i in range(len(ds)) if ds[i].shot_id == 90001)
    assert starts_90001[-1] == 24


@pytest.mark.skipif(not REAL_TOKEN_PATH.exists(), reason="rbb token corpus not mounted")
def test_real_shot_smoke():
    specs = discover_token_shots(shot_ids=[REAL_TOKEN_SHOT], read_n_frames=True)
    assert len(specs) == 1
    assert specs[0].n_frames > 100
    cfg = FrameWindowConfig(n_frames=12, stride=64)
    ds = FrameTokenDataset(specs, cfg)
    win = ds[0]
    assert win.tokens.shape == (12, *FRAME_GRID)
    assert win.tokens.dtype == np.int32
    if REAL_LEVEL1_PATH.exists():
        assert not win.time_is_synthetic
        assert np.all(np.diff(win.frame_time) > 0)
