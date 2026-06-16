"""Tests for the HF signal-token trainer/encoder driver logic."""

from __future__ import annotations

import numpy as np
import pytest

from imas_ambix.tokenizer import signal_hf_encode as enc
from imas_ambix.tokenizer.patch_transformer import (
    PatchTokenizerConfig,
    PatchTransformerTokenizer,
)
from imas_ambix.tokenizer.store_v2 import load_signal_hf_tokens


def test_canonical_ccbv_modern_legacy_align():
    assert enc._canonical_ccbv("ccbv_01") == "ccbv_01"
    assert enc._canonical_ccbv("ccbv01") == "ccbv_01"
    assert enc._canonical_ccbv("ccbv_40") == "ccbv_40"
    assert enc._canonical_ccbv("ccbv40") == "ccbv_40"
    # non-coil names rejected
    assert enc._canonical_ccbv("dia_loop") is None
    assert enc._canonical_ccbv("fl_cc01") is None


def test_parse_ids_comma_and_space():
    assert enc._parse_ids("1,2,3") == [1, 2, 3]
    assert enc._parse_ids("10 20 30") == [10, 20, 30]


def test_specs_have_distinct_blocks():
    assert enc.XMA_SPEC.is_coil_array is True
    assert enc.XMA_SPEC.mode_block is not None
    assert enc.XIM_SPEC.is_coil_array is False
    assert enc.XIM_SPEC.mode_block is None
    assert enc.XMA_SPEC.patch_block != enc.XIM_SPEC.patch_block


def _train_tiny_tokenizer(n_channels: int, bottleneck: str = "fsq"):
    rng = np.random.default_rng(0)
    T = 64 * 40  # 40 patches at patch_size 64
    windows = [
        rng.standard_normal((n_channels, T)).astype(np.float32) for _ in range(2)
    ]
    cfg = PatchTokenizerConfig(
        patch_size=64,
        seq_patches=16,
        d_model=32,
        n_layers=1,
        n_heads=2,
        bottleneck=bottleneck,
    )
    tok = PatchTransformerTokenizer(cfg=cfg, device="cpu")
    tok.fit(windows, epochs=2, seed=0)
    return tok


def test_encode_synthetic_window_to_store(tmp_path, monkeypatch):
    """encode_shots writes a valid v2 store group via a monkeypatched loader."""
    import imas_ambix.tokenizer.store_v2 as store_mod

    monkeypatch.setattr(store_mod, "TOKEN_ROOT", tmp_path)

    rng = np.random.default_rng(1)
    T, C = 64 * 30, 4
    data = rng.standard_normal((C, T)).astype(np.float32)
    chan = [f"da_ch{i}" for i in range(C)]
    valid = np.ones(C, dtype=bool)

    def fake_loader(shot_id, group):
        return data, chan, valid, 50_000.0, (0.0, T / 50_000.0)

    monkeypatch.setattr(enc, "load_shot_window", fake_loader)

    tok = _train_tiny_tokenizer(C, bottleneck="fsq")
    tok.name = enc.XIM_SPEC.patch_block
    summary = enc.encode_shots("xim", [12345], tok)
    assert len(summary["encoded"]) == 1
    assert summary["skipped"] == []

    g = load_signal_hf_tokens(12345, "xim")
    assert g.tokens.shape[1] == C
    assert g.token_time.shape[0] == g.tokens.shape[0]
    assert g.valid.shape == g.tokens.shape
    assert g.attrs.phase_preserving is True
    assert g.attrs.native_rate_hz == 50_000.0
    # token rate = native / patch_size
    assert g.attrs.token_rate_hz == pytest.approx(50_000.0 / 64)
    # per-token time is monotone and within the original window
    assert np.all(np.diff(g.token_time) > 0)
    assert g.token_time[0] >= g.attrs.original_window[0]


def test_encode_skips_missing_shot(tmp_path, monkeypatch):
    import imas_ambix.tokenizer.store_v2 as store_mod

    monkeypatch.setattr(store_mod, "TOKEN_ROOT", tmp_path)
    monkeypatch.setattr(enc, "load_shot_window", lambda s, g: None)

    tok = _train_tiny_tokenizer(4)
    tok.name = enc.XIM_SPEC.patch_block
    summary = enc.encode_shots("xim", [999], tok)
    assert summary["encoded"] == []
    assert summary["skipped"][0]["reason"] == "no_data"


def test_coil_array_emits_mode_block(tmp_path, monkeypatch):
    import imas_ambix.tokenizer.store_v2 as store_mod

    monkeypatch.setattr(store_mod, "TOKEN_ROOT", tmp_path)

    rng = np.random.default_rng(2)
    T, C = 64 * 30, 16  # enough channels for the spatial-DFT mode block
    data = rng.standard_normal((C, T)).astype(np.float32)
    chan = [f"ccbv_{i + 1:02d}" for i in range(C)]
    valid = np.ones(C, dtype=bool)
    monkeypatch.setattr(
        enc,
        "load_shot_window",
        lambda s, g: (data, chan, valid, 5_000.0, (0.0, T / 5_000.0)),
    )

    tok = _train_tiny_tokenizer(C)
    tok.name = enc.XMA_SPEC.patch_block
    enc.encode_shots("xma", [777], tok)

    mode = load_signal_hf_tokens(777, "xma_mode")
    assert mode.attrs.metadata["kind"] == "spatial_dft_mode_amplitudes"
    assert mode.attrs.n_channels == 2 * enc.N_MODES
    assert mode.attrs.phase_preserving is True
