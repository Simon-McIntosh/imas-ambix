"""Integration tests: block_kind encode → persist → load → window.

Covers:
- ShotTokenizer emits aligned tokens and block_kind arrays.
- save_shot_stream / load_shot_stream round-trip exactly.
- ShotTokenDataset uses block_kind to compute a non-uniform loss_mask.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import xarray as xr

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _fresh_registry():
    """Snapshot + restore the shared tokenizer registry singleton."""
    from imas_ambix.tokenizer import registry as singleton
    from imas_ambix.tokenizer.registry import CONTROL_RANGE

    saved_blocks = dict(singleton._blocks)
    saved_cursor = singleton._cursor
    singleton._blocks.clear()
    singleton._cursor = CONTROL_RANGE[1]
    return singleton, saved_blocks, saved_cursor


def _restore_registry(singleton, saved_blocks, saved_cursor):
    singleton._blocks.clear()
    singleton._blocks.update(saved_blocks)
    singleton._cursor = saved_cursor


@pytest.fixture
def fresh_registry():
    singleton, sb, sc = _fresh_registry()
    yield singleton
    _restore_registry(singleton, sb, sc)


def _toy_shot_tokenizer(fresh_registry):
    """Return a ShotTokenizer with lightweight placeholder implementations."""
    from imas_ambix.tokenizer.frames import PlaceholderFrameTokenizer
    from imas_ambix.tokenizer.multimodal import ShotTokenizer
    from imas_ambix.tokenizer.signals import UniformQuantizer

    ft = PlaceholderFrameTokenizer(
        spatial_compression=4, temporal_compression=1, intensity_levels=64
    )
    st = UniformQuantizer(n_bins=64)
    return ShotTokenizer(frame_tokenizer=ft, signal_tokenizer=st)


def _toy_frames(n_steps: int = 4) -> np.ndarray:
    """Synthetic ``(T, H, W)`` uint16 frame array."""
    return np.zeros((n_steps, 16, 16), dtype=np.uint16)


def _toy_signals(n_steps: int = 4) -> xr.Dataset:
    """Synthetic xr.Dataset with two signal channels."""
    t = np.arange(n_steps, dtype=np.float64) * 0.01
    return xr.Dataset(
        {"ip": (("time",), np.sin(t) * 1000), "ne": (("time",), np.cos(t) * 1e19)},
        coords={"time": t},
    )


# ---------------------------------------------------------------------------
# Test 1: aligned tokens and block_kind from ShotTokenizer
# ---------------------------------------------------------------------------


def test_shot_tokenizer_emits_aligned_block_kind(fresh_registry):
    """encode_shot_with_block_kind returns (tokens, block_kind) of equal length."""
    from imas_ambix.tokenizer.base import BlockKind
    from imas_ambix.tokenizer.registry import CONTROL_TOKENS

    n_steps = 4
    shot_tok = _toy_shot_tokenizer(fresh_registry)
    st = shot_tok.signal_tokenizer
    ds = _toy_signals(n_steps)
    st.fit([ds])

    frames = _toy_frames(n_steps)
    tokens, block_kind = shot_tok.encode_shot_with_block_kind(frames=frames, signals=ds)

    # Lengths match
    assert tokens.shape == block_kind.shape
    assert tokens.ndim == 1
    assert tokens.dtype == np.int32
    assert block_kind.dtype == np.uint8

    # <bos> is CONTROL
    assert tokens[0] == CONTROL_TOKENS["bos"], "first token must be <bos>"
    assert block_kind[0] == BlockKind.CONTROL, "<bos> must be CONTROL"

    # <eos> is CONTROL
    assert tokens[-1] == CONTROL_TOKENS["eos"], "last token must be <eos>"
    assert block_kind[-1] == BlockKind.CONTROL, "<eos> must be CONTROL"

    # FRAME tokens exist
    assert BlockKind.FRAME in block_kind.tolist(), "no FRAME tokens found"
    # SIGNAL tokens exist
    assert BlockKind.SIGNAL in block_kind.tolist(), "no SIGNAL tokens found"
    # CONTROL tokens beyond bos/eos (the <sep> tokens)
    n_control = int((block_kind == BlockKind.CONTROL).sum())
    expected_min = n_steps + 2
    assert n_control >= expected_min, (
        f"expected >= {expected_min} CONTROL positions "
        f"(bos + {n_steps} sep + eos), got {n_control}"
    )

    # Return-value shape is consistent with return_block_kind=True path
    tokens2, block_kind2 = shot_tok.encode_shot(
        frames=frames, signals=ds, return_block_kind=True
    )
    np.testing.assert_array_equal(tokens, tokens2)
    np.testing.assert_array_equal(block_kind, block_kind2)

    # Default (return_block_kind=False) still returns a plain ndarray
    tokens_plain = shot_tok.encode_shot(frames=frames, signals=ds)
    assert isinstance(tokens_plain, np.ndarray)
    np.testing.assert_array_equal(tokens, tokens_plain)


# ---------------------------------------------------------------------------
# Test 2: save_shot_stream / load_shot_stream round-trip
# ---------------------------------------------------------------------------


def test_persist_round_trip_with_block_kind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fresh_registry
):
    """save_shot_stream → load_shot_stream round-trips tokens and block_kind."""
    import imas_ambix.data.persist as persist_mod

    monkeypatch.setattr(persist_mod, "TOKEN_ROOT", tmp_path)

    n_steps = 6
    shot_tok = _toy_shot_tokenizer(fresh_registry)
    st = shot_tok.signal_tokenizer
    ds = _toy_signals(n_steps)
    st.fit([ds])

    frames = _toy_frames(n_steps)
    tokens, block_kind = shot_tok.encode_shot_with_block_kind(frames=frames, signals=ds)

    shot_id = 99901
    path = persist_mod.save_shot_stream(
        shot_id=shot_id,
        tokens=tokens,
        block_kind=block_kind,
        vocab_version="v1",
    )
    assert path.exists(), f"stream Zarr was not written to {path}"

    tokens_rt, block_kind_rt = persist_mod.load_shot_stream(
        shot_id=shot_id,
        vocab_version="v1",
    )

    np.testing.assert_array_equal(tokens_rt, tokens)
    np.testing.assert_array_equal(block_kind_rt, block_kind)
    assert tokens_rt.dtype == np.int32
    assert block_kind_rt.dtype == np.uint8


# ---------------------------------------------------------------------------
# Test 3: ShotTokenDataset uses block_kind for a non-uniform loss_mask
# ---------------------------------------------------------------------------


def test_loader_uses_block_kind_for_loss_mask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fresh_registry
):
    """Write a synthetic shot with block_kind; loader produces correct loss_mask."""
    import zarr

    import imas_ambix.data.persist as persist_mod
    from imas_ambix.data.loaders import (
        BLOCK_WEIGHTS,
        ShotTokenDataset,
        ShotTokenSpec,
        WindowSamplerConfig,
    )

    monkeypatch.setattr(persist_mod, "TOKEN_ROOT", tmp_path)

    # Build a deterministic token/block_kind pair
    n_steps = 4
    shot_tok = _toy_shot_tokenizer(fresh_registry)
    st = shot_tok.signal_tokenizer
    ds = _toy_signals(n_steps)
    st.fit([ds])

    frames = _toy_frames(n_steps)
    tokens, block_kind = shot_tok.encode_shot_with_block_kind(frames=frames, signals=ds)

    n = len(tokens)
    context = n  # window = entire shot

    # Write directly as a streams Zarr
    stream_path = tmp_path / "v1" / "streams" / "99902.zarr"
    stream_path.parent.mkdir(parents=True, exist_ok=True)
    store = zarr.open_group(str(stream_path), mode="w")
    store.create_array("tokens", data=tokens)
    store.create_array("block_kind", data=block_kind)

    specs = [ShotTokenSpec(shot_id=99902, n_tokens=n, path=stream_path)]
    config = WindowSamplerConfig(context_length=context, stride=context, seed=0)
    dataset = ShotTokenDataset(specs, config)

    windows = list(dataset)
    assert len(windows) == 1
    window = windows[0]

    loss_mask = window["loss_mask"]
    assert loss_mask.dtype == np.float32
    assert loss_mask.shape == (context,)

    # loss_mask must NOT be uniform ones everywhere — block_kind varies
    assert not np.all(loss_mask == 1.0), (
        "loss_mask should not be uniform 1.0 when block_kind is present"
    )

    # Verify each position individually against BLOCK_WEIGHTS
    for i, code in enumerate(block_kind):
        expected = BLOCK_WEIGHTS.get(int(code), 1.0)
        assert loss_mask[i] == pytest.approx(expected, abs=1e-6), (
            f"position {i}: code={code}, expected weight={expected}, got {loss_mask[i]}"
        )
