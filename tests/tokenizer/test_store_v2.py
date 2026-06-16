"""Schema tests for the multi-rate phase-preserving token store (v2)."""

from __future__ import annotations

import numpy as np
import pytest

from imas_ambix.tokenizer.registry import (
    BLOCK_XIM_PATCH,
    BLOCK_XMA_MODE,
    BLOCK_XMA_PATCH,
    VOCAB_VERSION,
)
from imas_ambix.tokenizer.store_v2 import (
    REQUIRED_ATTRS,
    STORE_GENERATION,
    SignalHFTokens,
    StoreV2Attrs,
    load_signal_hf_tokens,
    registry_v2_path,
    save_signal_hf_tokens,
    signal_hf_token_path,
)


def _make_attrs(**over) -> StoreV2Attrs:
    base = dict(
        tokenizer_name="signal_hf_xma_patch_v2",
        vocab_version=VOCAB_VERSION,
        native_rate_hz=100_000.0,
        token_rate_hz=1_562.5,
        n_channels=3,
        channel_names=("ccbv_01", "ccbv_02", "ccbv_03"),
        phase_preserving=True,
        original_window=(0.0, 0.32),
    )
    base.update(over)
    return StoreV2Attrs(**base)


def test_vocab_version_bumped():
    assert VOCAB_VERSION == "v2"


def test_block_names_distinct():
    names = {BLOCK_XMA_PATCH, BLOCK_XMA_MODE, BLOCK_XIM_PATCH}
    assert len(names) == 3
    assert all(n.endswith("_v2") for n in names)


def test_attrs_validate_channel_count():
    with pytest.raises(ValueError, match="n_channels"):
        _make_attrs(n_channels=5)  # 5 != len(channel_names)=3


def test_attrs_reject_nonpositive_rate():
    with pytest.raises(ValueError, match="rates"):
        _make_attrs(native_rate_hz=0.0)


def test_attrs_roundtrip():
    a = _make_attrs(metadata={"patch_size": 64, "stft": True})
    restored = StoreV2Attrs.from_attrs(a.to_attrs())
    assert restored.tokenizer_name == a.tokenizer_name
    assert restored.native_rate_hz == a.native_rate_hz
    assert restored.token_rate_hz == a.token_rate_hz
    assert restored.channel_names == a.channel_names
    assert restored.phase_preserving is True
    assert restored.original_window == a.original_window
    assert restored.metadata["patch_size"] == 64


def test_from_attrs_rejects_missing_keys():
    full = _make_attrs().to_attrs()
    for key in REQUIRED_ATTRS:
        partial = {k: v for k, v in full.items() if k != key}
        with pytest.raises(ValueError, match="missing required"):
            StoreV2Attrs.from_attrs(partial)


def test_paths_carry_store_generation():
    p = signal_hf_token_path(30460, "xma")
    assert p.parts[-4] == STORE_GENERATION
    assert p.parts[-3] == "signals_hf"
    assert p.parts[-2] == "30460"
    assert p.name == "xma.zarr"
    assert registry_v2_path().parts[-2] == STORE_GENERATION


def test_save_load_roundtrip(tmp_path, monkeypatch):
    import imas_ambix.tokenizer.store_v2 as mod

    monkeypatch.setattr(mod, "TOKEN_ROOT", tmp_path)

    n_tok, n_ch = 20, 3
    rng = np.random.default_rng(0)
    tokens = rng.integers(10, 50, size=(n_tok, n_ch)).astype(np.int32)
    token_time = np.linspace(0.0, 0.32, n_tok)
    valid = np.ones((n_tok, n_ch), dtype=bool)
    valid[5:8, 1] = False  # a channel dropout window — must round-trip
    attrs = _make_attrs()

    path = save_signal_hf_tokens(30460, "xma", tokens, token_time, valid, attrs)
    assert path.exists()

    loaded = load_signal_hf_tokens(30460, "xma")
    assert isinstance(loaded, SignalHFTokens)
    np.testing.assert_array_equal(loaded.tokens, tokens)
    np.testing.assert_allclose(loaded.token_time, token_time)
    np.testing.assert_array_equal(loaded.valid, valid)
    assert loaded.valid[5:8, 1].sum() == 0  # dropout preserved, not zero-filled
    assert loaded.attrs.phase_preserving is True
    assert loaded.attrs.native_rate_hz == 100_000.0
    assert loaded.n_tokens == n_tok
    assert loaded.n_channels == n_ch


def test_save_rejects_shape_mismatch(tmp_path, monkeypatch):
    import imas_ambix.tokenizer.store_v2 as mod

    monkeypatch.setattr(mod, "TOKEN_ROOT", tmp_path)
    attrs = _make_attrs()
    tokens = np.zeros((10, 3), dtype=np.int32)
    # token_time wrong length
    with pytest.raises(ValueError, match="token_time"):
        save_signal_hf_tokens(
            1, "xma", tokens, np.zeros(9), np.ones((10, 3), bool), attrs
        )
    # valid wrong shape
    with pytest.raises(ValueError, match="valid"):
        save_signal_hf_tokens(
            1, "xma", tokens, np.zeros(10), np.ones((10, 2), bool), attrs
        )
    # attrs channel-count disagrees with tokens
    bad = _make_attrs(n_channels=2, channel_names=("ccbv_01", "ccbv_02"))
    with pytest.raises(ValueError, match="n_channels"):
        save_signal_hf_tokens(
            1, "xma", tokens, np.zeros(10), np.ones((10, 3), bool), bad
        )
