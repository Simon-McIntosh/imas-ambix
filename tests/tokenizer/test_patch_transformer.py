"""Tests for the phase-aware patch-transformer 1-D signal tokenizer."""

from __future__ import annotations

import numpy as np
import pytest

from imas_ambix.tokenizer.patch_transformer import (
    PatchTokenizerConfig,
    PatchTransformerTokenizer,
    mode_decomposition,
    mode_number_recovery,
    patchify,
    phase_error,
    reconstruction_crps,
    stft_lift,
    stft_unlift,
    unpatchify,
)


def test_patchify_roundtrip_exact():
    x = np.random.default_rng(0).standard_normal((3, 200)).astype(np.float32)
    patches, n_pad = patchify(x, 64)
    assert patches.shape == (3, 4, 64)  # 200 -> 256 padded -> 4 patches
    assert n_pad == 56
    back = unpatchify(patches, n_pad)
    np.testing.assert_allclose(back, x)


def test_patchify_exact_multiple_no_pad():
    x = np.random.default_rng(1).standard_normal((2, 128)).astype(np.float32)
    patches, n_pad = patchify(x, 64)
    assert n_pad == 0
    np.testing.assert_allclose(unpatchify(patches, n_pad), x)


def test_stft_lift_preserves_phase_roundtrip():
    patches = np.random.default_rng(2).standard_normal((2, 3, 64)).astype(np.float32)
    feats = stft_lift(patches)
    assert feats.shape[-1] == 2 * (64 // 2 + 1)  # real + imag
    back = stft_unlift(feats, 64)
    np.testing.assert_allclose(back, patches, atol=1e-4)


def test_phase_error_zero_for_identity():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((4, 256))
    assert phase_error(x, x.copy(), dt=1.0) == pytest.approx(0.0, abs=1e-9)


def test_phase_error_detects_phase_flip():
    """A magnitude-preserving, phase-scrambling map must show large phase error."""
    t = np.linspace(0, 1, 256, endpoint=False)
    x = np.sin(2 * np.pi * 8 * t)[None, :]
    # Shift phase by quarter period — same magnitude spectrum, wrong phase.
    x_shift = np.sin(2 * np.pi * 8 * t + np.pi / 2)[None, :]
    assert phase_error(x, x_shift, dt=1.0) > 0.5
    # amplitude (CRPS) does not flag it as strongly — phase metric is needed.
    assert reconstruction_crps(x, x_shift) > 0.0


def test_mode_decomposition_recovers_known_mode():
    """A pure m=2 travelling wave across coils → energy in mode 2."""
    T, C = 64, 16
    ang = np.linspace(0, 2 * np.pi, C, endpoint=False)
    tt = np.arange(T)
    sig = np.cos(2 * ang[None, :] + 0.1 * tt[:, None])  # (T, C), m=2
    modes = mode_decomposition(sig, n_modes=4)  # (T, 8)
    amp = np.sqrt(modes[:, :4] ** 2 + modes[:, 4:] ** 2)  # per-mode amplitude
    dominant = amp.mean(0).argmax()
    assert dominant == 2


def test_mode_number_recovery_perfect_for_identity():
    T, C = 64, 16
    ang = np.linspace(0, 2 * np.pi, C, endpoint=False)
    tt = np.arange(T)
    sig = np.cos(2 * ang[None, :] + 0.1 * tt[:, None]) + 0.5 * np.cos(
        1 * ang[None, :] + 0.07 * tt[:, None]
    )
    rec = mode_number_recovery(sig, sig.copy())
    assert rec["mean_complex_corr"] == pytest.approx(1.0, abs=1e-6)
    assert rec["mean_mode_phase_err"] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("bottleneck", ["continuous", "fsq", "vq"])
def test_tokenizer_fit_encode_shapes(bottleneck):
    rng = np.random.default_rng(0)
    T, C = 256, 4
    windows = [rng.standard_normal((C, T)).astype(np.float32) for _ in range(3)]
    cfg = PatchTokenizerConfig(
        patch_size=64, d_model=32, n_layers=1, n_heads=2, bottleneck=bottleneck
    )
    tok = PatchTransformerTokenizer(cfg=cfg, device="cpu")
    hist = tok.fit(windows, epochs=2, batch_channels=16, seed=0)
    assert len(hist["recon"]) == 2

    ids, latent, recon = tok.encode_window(windows[0])
    assert ids.shape == (C, T // 64)
    assert latent.shape[:2] == (C, T // 64)
    assert latent.ndim == 3  # (C, P, embed_dim)
    assert recon.shape == (C, T)

    m = tok.roundtrip_metrics(windows[0], dt=1 / 5e4, is_coil_array=True)
    assert "recon_crps" in m and "phase_err" in m
    assert "mean_complex_corr" in m  # coil-array mode recovery present
    if bottleneck == "continuous":
        assert m["codebook_size"] == 0
    else:
        assert m["codebook_size"] > 0


def test_tokenizer_requires_fit_before_encode():
    tok = PatchTransformerTokenizer(device="cpu")
    with pytest.raises(RuntimeError, match="fit"):
        tok.encode_window(np.zeros((2, 128), dtype=np.float32))


def test_tokenizer_save_load(tmp_path):
    rng = np.random.default_rng(1)
    windows = [rng.standard_normal((3, 128)).astype(np.float32) for _ in range(2)]
    cfg = PatchTokenizerConfig(
        patch_size=64, d_model=32, n_layers=1, n_heads=2, bottleneck="fsq"
    )
    tok = PatchTransformerTokenizer(cfg=cfg, device="cpu")
    tok.fit(windows, epochs=1, seed=0)
    ids0, _lat0, recon0 = tok.encode_window(windows[0])

    path = tmp_path / "tok.pt"
    tok.save(path)

    tok2 = PatchTransformerTokenizer(device="cpu")
    tok2.load(path)
    ids1, _lat1, recon1 = tok2.encode_window(windows[0])
    np.testing.assert_array_equal(ids0, ids1)
    np.testing.assert_allclose(recon0, recon1, atol=1e-5)


def _calib(name, mean, std):
    from imas_ambix.calibration.signals import ChannelCalibration

    return ChannelCalibration(
        name=name,
        mean=float(mean),
        std=float(std),
        min_value=float(mean - 5 * std),
        max_value=float(mean + 5 * std),
        q01=float(mean - 2 * std),
        q50=float(mean),
        q99=float(mean + 2 * std),
        n_samples=1000,
        n_shots=10,
    )


def test_fit_records_calibration_mode_and_survives_save_load(tmp_path):
    rng = np.random.default_rng(2)
    windows = [rng.standard_normal((2, 128)).astype(np.float32) for _ in range(2)]
    names = ["a", "b"]
    cal = {"a": _calib("a", 0.0, 1.0), "b": _calib("b", 0.0, 1.0)}
    cfg = PatchTokenizerConfig(
        patch_size=64, d_model=32, n_layers=1, n_heads=2, bottleneck="fsq"
    )
    tok = PatchTransformerTokenizer(cfg=cfg, device="cpu")
    # per-window default before fit
    assert tok.calibration_mode == "per_window"
    tok.fit(windows, epochs=1, seed=0, channel_names=names, corpus_calibration=cal)
    assert tok.calibration_mode == "absolute"
    path = tmp_path / "abs.pt"
    tok.save(path)
    tok2 = PatchTransformerTokenizer(device="cpu")
    tok2.load(path)
    assert tok2.calibration_mode == "absolute"


def test_absolute_normalisation_uses_corpus_stats():
    # A channel with a large DC offset: per-window centres it on its own mean
    # (~10), absolute centres it on the corpus mean (0) → the normalised inputs
    # must differ, proving the calibration is actually applied in _normalise.
    rng = np.random.default_rng(3)
    x = (rng.standard_normal((1, 256)).astype(np.float32)) + 10.0
    cfg = PatchTokenizerConfig(patch_size=64, d_model=16, n_layers=1, n_heads=2)
    tok = PatchTransformerTokenizer(cfg=cfg, device="cpu")
    z_pw, m_pw, _ = tok._normalise(x, fit=False)
    z_abs, m_abs, _ = tok._normalise(
        x,
        fit=False,
        channel_names=["a"],
        corpus_calibration={"a": _calib("a", 0.0, 1.0)},
    )
    assert m_pw[0, 0] == pytest.approx(x.mean(), abs=1e-3)  # per-window mean ~10
    assert m_abs[0, 0] == pytest.approx(0.0, abs=1e-9)  # corpus mean 0
    assert not np.allclose(z_pw, z_abs)


def test_normalise_default_is_byte_identical_without_calibration():
    # The safety invariant: omitting calibration leaves behaviour unchanged.
    rng = np.random.default_rng(4)
    x = rng.standard_normal((3, 128)).astype(np.float32)
    tok = PatchTransformerTokenizer(device="cpu")
    z_a, m_a, s_a = tok._normalise(x, fit=False)
    z_b, m_b, s_b = tok._normalise(
        x, fit=False, channel_names=None, corpus_calibration=None
    )
    np.testing.assert_array_equal(z_a, z_b)
    np.testing.assert_array_equal(m_a, m_b)
    np.testing.assert_array_equal(s_a, s_b)
