"""Tests for the bounded DataLoader composition layer (loader.py).

These run against the synthetic-corpus fixture (CPU-fast, no real corpus,
no GPU).  They assert the loader yields exactly the batch dict the trainer
consumes — right keys / shapes / dtypes, mask complement, conditioning
alignment with the locked D0 loader, and that per-shot caching does not
change the produced values (num_workers == 0 path).
"""

from __future__ import annotations

import numpy as np
import pytest

from imas_ambix.camdyn.conditioning import (
    CONDITIONING_CHANNELS,
    load_conditioning,
)
from imas_ambix.camdyn.dataset import (
    FrameTokenDataset,
    FrameWindowConfig,
    discover_token_shots,
)
from imas_ambix.camdyn.loader import (
    BATCH_KEYS,
    CamdynWindowStream,
    _hold_traces_to_frames,
    _read_shot_cond_traces,
    collate_windows,
)
from imas_ambix.camdyn.masking import ClipMaskConfig, MaskMode

N_COND = len(CONDITIONING_CHANNELS)


def _specs(sc):
    return discover_token_shots(
        token_root=sc["token_root"],
        level1_dir=sc["level1_dir"],
        shot_ids=sc["shot_ids"],
        read_n_frames=True,
    )


# ---------------------------------------------------------------------------
# Per-window stream (the worker body) — shapes / dtypes / mask complement
# ---------------------------------------------------------------------------


def test_stream_yields_expected_fields_and_shapes(synthetic_corpus):
    sc = synthetic_corpus
    specs = _specs(sc)
    nf = 6
    stream = CamdynWindowStream(
        specs,
        FrameWindowConfig(n_frames=nf, stride=4),
        ClipMaskConfig(),
        seed=0,
    )
    items = list(stream)
    assert items, "stream produced no windows"
    for it in items:
        assert it["tokens"].shape == (nf, 16, 16)
        assert it["tokens"].dtype == np.int64
        assert it["visible"].shape == (nf, 16, 16)
        assert it["visible"].dtype == np.bool_
        assert it["loss_mask"].dtype == np.bool_
        # loss_mask is the exact complement of visible
        np.testing.assert_array_equal(it["loss_mask"], ~it["visible"])
        assert it["cond_values"].shape == (nf, N_COND)
        assert it["cond_missing"].shape == (nf, N_COND)
        assert it["dt"].shape == (nf,)
        assert it["valid"].shape == (nf,)
        assert it["valid"].dtype == np.bool_
        assert it["frame_time"].shape == (nf,)


def test_stream_window_count_matches_dataset_enum(synthetic_corpus):
    """The stream covers exactly the dataset's enumerated windows."""
    sc = synthetic_corpus
    specs = _specs(sc)
    cfg = FrameWindowConfig(n_frames=6, stride=4)
    ds = FrameTokenDataset(specs, cfg, as_dict=True)
    stream = CamdynWindowStream(specs, cfg, ClipMaskConfig(), seed=0)
    assert len(list(stream)) == len(ds)
    assert len(stream) == len(ds)


# ---------------------------------------------------------------------------
# Collate — batch dict matches the train.py contract
# ---------------------------------------------------------------------------


def test_collate_stacks_into_batch_dict(synthetic_corpus):
    sc = synthetic_corpus
    specs = _specs(sc)
    nf = 6
    stream = CamdynWindowStream(
        specs, FrameWindowConfig(n_frames=nf, stride=4), ClipMaskConfig(), seed=1
    )
    items = list(stream)[:3]
    batch = collate_windows(items)
    b = len(items)
    assert set(BATCH_KEYS) <= set(batch)
    assert batch["tokens"].shape == (b, nf, 16, 16)
    assert batch["tokens"].dtype == np.int64
    assert batch["visible"].shape == (b, nf, 16, 16)
    assert batch["cond_values"].shape == (b, nf, N_COND)
    assert batch["cond_missing"].shape == (b, nf, N_COND)
    assert batch["dt"].shape == (b, nf)
    assert batch["valid"].shape == (b, nf)
    assert batch["frame_time"].shape == (b, nf)
    assert batch["shot_id"].shape == (b,)
    assert batch["shot_id"].dtype == np.int64
    # complement preserved through the stack
    np.testing.assert_array_equal(batch["loss_mask"], ~batch["visible"])


# ---------------------------------------------------------------------------
# Conditioning alignment — cached path == locked D0 loader
# ---------------------------------------------------------------------------


def test_cached_conditioning_matches_locked_loader(synthetic_corpus):
    """The per-shot RAW-trace cache reproduces load_conditioning exactly."""
    sc = synthetic_corpus
    specs = _specs(sc)
    spec = specs[0]
    # arbitrary frame times spanning the shot record
    ft = np.array([0.012, 0.02, 0.03, 0.05], dtype=np.float64)

    traces = _read_shot_cond_traces(spec.level1_path, CONDITIONING_CHANNELS)
    cv, cm = _hold_traces_to_frames(traces, ft, CONDITIONING_CHANNELS)

    ref = load_conditioning(
        spec.level1_path, ft, int(spec.shot_id), channels=CONDITIONING_CHANNELS
    )
    np.testing.assert_allclose(cv, ref.values, rtol=0, atol=0)
    np.testing.assert_array_equal(cm, ref.missing)


# ---------------------------------------------------------------------------
# Forced mode + bounded epoch
# ---------------------------------------------------------------------------


def test_forced_mode_full_masks_everything(synthetic_corpus):
    sc = synthetic_corpus
    specs = _specs(sc)
    stream = CamdynWindowStream(
        specs,
        FrameWindowConfig(n_frames=6, stride=4),
        ClipMaskConfig(),
        seed=0,
        mode=MaskMode.FULL,
    )
    for it in stream:
        assert not it["visible"].any()  # FULL → nothing visible
        assert it["loss_mask"].all()


def test_max_windows_caps_the_epoch(synthetic_corpus):
    sc = synthetic_corpus
    specs = _specs(sc)
    cap = 2
    stream = CamdynWindowStream(
        specs,
        FrameWindowConfig(n_frames=6, stride=4),
        ClipMaskConfig(),
        seed=0,
        max_windows=cap,
    )
    assert len(list(stream)) == cap
    assert len(stream) == cap


# ---------------------------------------------------------------------------
# Torch DataLoader factory (multi-worker) — only when torch is present
# ---------------------------------------------------------------------------


def test_make_loader_num_workers_yields_full_epoch(synthetic_corpus):
    pytest.importorskip("torch")
    from imas_ambix.camdyn.loader import make_loader

    sc = synthetic_corpus
    specs = _specs(sc)
    cfg = FrameWindowConfig(n_frames=6, stride=4)
    ds = FrameTokenDataset(specs, cfg, as_dict=True)
    n_windows = len(ds)

    for nw in (0, 2):
        loader = make_loader(
            specs,
            cfg,
            ClipMaskConfig(),
            batch_size=2,
            num_workers=nw,
            seed=0,
        )
        seen = 0
        for batch in loader:
            assert batch["tokens"].ndim == 4
            assert batch["cond_values"].shape[-1] == N_COND
            np.testing.assert_array_equal(batch["loss_mask"], ~batch["visible"])
            seen += batch["tokens"].shape[0]
        # union over workers covers exactly the epoch (no dup, no drop)
        assert seen == n_windows
