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
    # xsx is a chord array → cross-channel profile latent, distinct blocks.
    assert enc.XSX_SPEC.is_coil_array is True
    assert enc.XSX_SPEC.mode_block is not None
    assert enc.XSX_SPEC.patch_block not in (
        enc.XMA_SPEC.patch_block,
        enc.XIM_SPEC.patch_block,
    )
    assert enc.XSX_SPEC.mode_block != enc.XMA_SPEC.mode_block


def test_xsx_blocks_append_after_existing_no_vocab_bump():
    """The xsx blocks must sit ABOVE the xma/xim blocks (append-only) and the
    vocab generation must NOT have been re-bumped."""
    from imas_ambix.tokenizer.registry import (
        VOCAB_VERSION,
        TokenRegistry,
    )

    assert VOCAB_VERSION == "v2"  # append-only — never re-bumped for xsx
    r = TokenRegistry()
    xma_s, xma_e = r.allocate(enc.BLOCK_XMA_PATCH, 12800)
    r.allocate(enc.BLOCK_XMA_MODE, 1)
    xim_s, xim_e = r.allocate(enc.BLOCK_XIM_PATCH, 12800)
    xsx_s, xsx_e = r.allocate(enc.BLOCK_XSX_PATCH, 12800)
    # xsx allocated last → its range starts at or above every prior block end.
    assert xsx_s >= xim_e
    assert xsx_s >= xma_e
    assert xsx_e > xsx_s


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


def test_continuous_encode_persists_embedding(tmp_path, monkeypatch):
    """A continuous bottleneck stores the latent (the discrete ids are vestigial)."""
    import imas_ambix.tokenizer.store_v2 as store_mod

    monkeypatch.setattr(store_mod, "TOKEN_ROOT", tmp_path)

    rng = np.random.default_rng(3)
    T, C = 64 * 30, 4
    data = rng.standard_normal((C, T)).astype(np.float32)
    chan = [f"da_ch{i}" for i in range(C)]
    monkeypatch.setattr(
        enc,
        "load_shot_window",
        lambda s, g: (data, chan, np.ones(C, bool), 50_000.0, (0.0, T / 50_000.0)),
    )

    tok = _train_tiny_tokenizer(C, bottleneck="continuous")
    tok.name = enc.XIM_SPEC.patch_block
    enc.encode_shots("xim", [555], tok)

    g = load_signal_hf_tokens(555, "xim")
    assert g.attrs.metadata["bottleneck"] == "continuous"
    assert g.embedding is not None  # phase-preserving payload stored
    assert g.embedding.shape[:2] == g.tokens.shape  # (P, C, d)


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


def test_xsx_chord_array_emits_profile_block(tmp_path, monkeypatch):
    """xsx encodes per-chord patch tokens AND a cross-chord profile latent."""
    import imas_ambix.tokenizer.store_v2 as store_mod

    monkeypatch.setattr(store_mod, "TOKEN_ROOT", tmp_path)

    rng = np.random.default_rng(7)
    T, C = 64 * 30, 18  # 18 hcam_l chords
    data = rng.standard_normal((C, T)).astype(np.float32)
    chan = [f"hcam_l_{i:02d}" for i in range(C)]
    valid = np.ones(C, dtype=bool)
    monkeypatch.setattr(
        enc,
        "load_shot_window",
        lambda s, g: (data, chan, valid, 500_000.0, (0.0, T / 500_000.0)),
    )

    tok = _train_tiny_tokenizer(C, bottleneck="continuous")
    tok.name = enc.XSX_SPEC.patch_block
    summary = enc.encode_shots("xsx", [4242], tok)
    assert len(summary["encoded"]) == 1

    g = load_signal_hf_tokens(4242, "xsx")
    assert g.tokens.shape[1] == C
    assert g.attrs.native_rate_hz == 500_000.0
    assert g.attrs.token_rate_hz == pytest.approx(500_000.0 / 64)

    prof = load_signal_hf_tokens(4242, "xsx_mode")
    assert prof.attrs.metadata["kind"] == "spatial_dft_mode_amplitudes"
    assert prof.attrs.n_channels == 2 * enc.N_MODES
    assert prof.attrs.phase_preserving is True


def test_xsx_loader_marks_stuck_chord_invalid(tmp_path, monkeypatch):
    """The known stuck hcam_l chord is masked invalid by load_shot_window."""
    from imas_ambix.statespace.fast_features import XSX_STUCK_CHANNEL

    class _Fake:
        rate_hz = 500_000.0
        time = np.linspace(0.0, 0.6, 64 * 10)
        hcam_l = (
            np.random.default_rng(0).standard_normal((18, 64 * 10)).astype(np.float32)
        )
        hcam_u = None
        hcam_l_r1 = np.full(18, np.nan)
        hcam_u_r1 = None
        avail_mask = np.array([True, False])

    # Point LEVEL1_DIR at tmp and create the shot dir so the existence guard
    # passes; stub the reader so no real Zarr is needed.
    monkeypatch.setattr(enc, "LEVEL1_DIR", tmp_path)
    (tmp_path / "123.zarr").mkdir()
    monkeypatch.setattr(enc, "read_xsx_shot", lambda p: _Fake())

    out = enc.load_shot_window(123, "xsx")
    assert out is not None
    _data, chan, valid, rate, _win = out
    assert rate == 500_000.0
    assert len(chan) == 18
    assert not valid[XSX_STUCK_CHANNEL]
    assert valid.sum() == 17  # all chords valid except the stuck one


def test_shot_window_dataset_reports_skip_reasons(monkeypatch):
    """ShotWindowDataset yields (sid, None, reason) for absent / unreadable shots."""
    monkeypatch.setattr(enc, "group_present", lambda s, g: s != 2)
    monkeypatch.setattr(
        enc, "load_shot_window", lambda s, g: None if s == 3 else ("w",)
    )
    ds = enc.ShotWindowDataset([1, 2, 3], "xim")
    assert len(ds) == 3
    assert ds[0] == (1, ("w",), None)
    assert ds[1] == (2, None, "group_absent")
    assert ds[2] == (3, None, "no_data")


def test_skip_existing_and_group_absent(tmp_path, monkeypatch):
    import imas_ambix.tokenizer.store_v2 as store_mod

    monkeypatch.setattr(store_mod, "TOKEN_ROOT", tmp_path)
    # group_present False → group_absent; load_shot_window never called.
    monkeypatch.setattr(enc, "group_present", lambda s, g: False)
    monkeypatch.setattr(
        enc,
        "load_shot_window",
        lambda s, g: (_ for _ in ()).throw(
            AssertionError("loader must not be called when group absent")
        ),
    )
    tok = _train_tiny_tokenizer(4)
    tok.name = enc.XIM_SPEC.patch_block
    summary = enc.encode_shots("xim", [888], tok)
    assert summary["encoded"] == []
    assert summary["skipped"][0]["reason"] == "group_absent"


def test_already_encoded_rejects_truncated_store(tmp_path, monkeypatch):
    """A store with arrays but no attrs (mid-write kill) is NOT 'already encoded'.

    Guarantees a corpus resume re-encodes a shot whose write was interrupted at
    the SLURM time limit, rather than skipping the truncated output forever.
    """
    import zarr

    import imas_ambix.tokenizer.store_v2 as store_mod

    monkeypatch.setattr(store_mod, "TOKEN_ROOT", tmp_path)
    # A complete store reads back as already-encoded.
    rng = np.random.default_rng(11)
    data = rng.standard_normal((4, 64 * 20)).astype(np.float32)
    chan = [f"da_ch{i}" for i in range(4)]
    monkeypatch.setattr(
        enc,
        "load_shot_window",
        lambda s, g: (data, chan, np.ones(4, bool), 50_000.0, (0.0, 0.4)),
    )
    tok = _train_tiny_tokenizer(4)
    tok.name = enc.XIM_SPEC.patch_block
    enc.encode_shots("xim", [4321], tok, skip_existing=False)
    assert enc.already_encoded(4321, "xim") is True

    # Now truncate it: arrays present, attrs stripped → must read as NOT done.
    path = store_mod.signal_hf_token_path(4321, "xim")
    g = zarr.open_group(str(path), mode="a")
    for k in list(g.attrs.keys()):
        del g.attrs[k]
    assert enc.already_encoded(4321, "xim") is False


def test_already_encoded_require_absolute_supersedes_per_shot(tmp_path, monkeypatch):
    """A per-shot store is NOT 'done' for an absolute encode → it is re-encoded.

    This is the supersede contract: re-running the corpus in absolute mode must
    replace legacy per-shot stores, while an already-absolute shot is skipped on
    resume.
    """
    import imas_ambix.tokenizer.store_v2 as store_mod

    monkeypatch.setattr(store_mod, "TOKEN_ROOT", tmp_path)
    rng = np.random.default_rng(12)
    data = rng.standard_normal((4, 64 * 20)).astype(np.float32)
    chan = [f"da_ch{i}" for i in range(4)]
    monkeypatch.setattr(
        enc,
        "load_shot_window",
        lambda s, g: (data, chan, np.ones(4, bool), 50_000.0, (0.0, 0.4)),
    )
    tok = _train_tiny_tokenizer(4)
    tok.name = enc.XIM_SPEC.patch_block
    # Write a per-shot (no calibration) store.
    enc.encode_shots("xim", [7777], tok, skip_existing=False)
    # Complete, and "done" under the legacy (mode-agnostic) check …
    assert enc.already_encoded(7777, "xim") is True
    # … but NOT done when absolute is required → the supersede re-encodes it.
    assert enc.already_encoded(7777, "xim", require_mode="absolute") is False
    assert enc.already_encoded(7777, "xim", require_mode="per_shot") is True
