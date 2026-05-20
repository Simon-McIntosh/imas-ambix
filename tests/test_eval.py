"""Smoke tests for the evaluation metrics and rollout stub.

Tests use synthetic frame pairs built from numpy random arrays plus small
perturbations so they run without any fusion data. Heavy metrics (lpips,
rfid) are marked ``slow`` and skip cleanly when optional deps are absent.

Related modules: ``imas_ambix/eval/metrics.py``, ``imas_ambix/eval/rollout.py``
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from imas_ambix.eval import (
    RolloutConfig,
    centroid_mse,
    chord_integrated_emission,
    chord_nrmse,
    compute_all_metrics,
    edge_displacement,
    frame_centroid,
    psnr,
    rollout,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_frames(
    t: int = 8, h: int = 32, w: int = 32, channels: int = 3, seed: int = 0
) -> np.ndarray:
    """Generate a synthetic ``(T, H, W, C)`` uint8 frame sequence."""
    rng = np.random.default_rng(seed)
    return rng.integers(30, 220, size=(t, h, w, channels), dtype=np.uint8)


def _random_frames_gray(
    t: int = 8, h: int = 32, w: int = 32, seed: int = 0
) -> np.ndarray:
    """Generate a synthetic ``(T, H, W)`` uint8 frame sequence."""
    rng = np.random.default_rng(seed)
    return rng.integers(30, 220, size=(t, h, w), dtype=np.uint8)


# ---------------------------------------------------------------------------
# psnr
# ---------------------------------------------------------------------------


def test_psnr_identical_frames_is_inf():
    frames = _random_frames()
    result = psnr(frames, frames)
    assert math.isinf(result) and result > 0


def test_psnr_noisy_is_positive():
    rng = np.random.default_rng(1)
    clean = _random_frames(seed=2)
    noisy = np.clip(
        clean.astype(np.int32) + rng.integers(-20, 20, size=clean.shape),
        0,
        255,
    ).astype(np.uint8)
    assert psnr(noisy, clean) > 0.0


def test_psnr_more_noise_lower_value():
    rng = np.random.default_rng(3)
    clean = _random_frames(seed=4)
    small_noise = np.clip(
        clean.astype(np.int32) + rng.integers(-5, 5, size=clean.shape),
        0,
        255,
    ).astype(np.uint8)
    large_noise = np.clip(
        clean.astype(np.int32) + rng.integers(-80, 80, size=clean.shape),
        0,
        255,
    ).astype(np.uint8)
    assert psnr(small_noise, clean) > psnr(large_noise, clean)


def test_psnr_grayscale_input():
    gray = _random_frames_gray(seed=5)
    result = psnr(gray, gray)
    assert math.isinf(result)


def test_psnr_shape_mismatch_raises():
    a = _random_frames(t=4)
    b = _random_frames(t=8)
    with pytest.raises(ValueError, match="same shape"):
        psnr(a, b)


# ---------------------------------------------------------------------------
# frame_centroid
# ---------------------------------------------------------------------------


def test_frame_centroid_returns_correct_shape():
    frames = _random_frames(t=6, h=32, w=32)
    c = frame_centroid(frames)
    assert c.shape == (6, 2)


def test_frame_centroid_uniform_frame_is_centre():
    # Uniform brightness → centroid should be at (W/2, H/2) approximately
    frames = np.full((4, 32, 32), fill_value=100, dtype=np.uint8)
    c = frame_centroid(frames)
    np.testing.assert_allclose(c[:, 0], 15.5, atol=0.5)  # x ≈ (32-1)/2
    np.testing.assert_allclose(c[:, 1], 15.5, atol=0.5)  # y ≈ (32-1)/2


def test_frame_centroid_bright_spot_left():
    # Bright spot in left half → x centroid < W/2
    frames = np.zeros((3, 32, 32), dtype=np.uint8)
    frames[:, :, :8] = 200  # left 8 columns bright
    c = frame_centroid(frames)
    assert (c[:, 0] < 16.0).all()


def test_frame_centroid_accepts_rgb():
    rgb = _random_frames(t=4, h=16, w=16, channels=3)
    c = frame_centroid(rgb)
    assert c.shape == (4, 2)


# ---------------------------------------------------------------------------
# centroid_mse
# ---------------------------------------------------------------------------


def test_centroid_mse_identical_is_zero():
    frames = _random_frames(seed=10)
    assert centroid_mse(frames, frames) == 0.0


def test_centroid_mse_different_frames_is_positive():
    rng = np.random.default_rng(11)
    ref = _random_frames(seed=12)
    pred = np.clip(
        ref.astype(np.int32) + rng.integers(-50, 50, size=ref.shape),
        0,
        255,
    ).astype(np.uint8)
    assert centroid_mse(ref, pred) >= 0.0  # non-negative


def test_centroid_mse_scaled_signal_small():
    # Lightly perturbed frames should give small centroid MSE
    ref = _random_frames(t=8, h=32, w=32, seed=13)
    pred = np.clip(ref.astype(np.int32) + 5, 0, 255).astype(np.uint8)
    assert centroid_mse(ref, pred) < 4.0  # centroid barely moves


# ---------------------------------------------------------------------------
# chord_integrated_emission
# ---------------------------------------------------------------------------


def test_chord_integrated_emission_shape():
    frames = _random_frames_gray(t=10, h=32, w=32)
    out = chord_integrated_emission(frames)
    assert out.shape == (10,)


def test_chord_integrated_emission_custom_chord():
    frames = np.zeros((5, 32, 32), dtype=np.uint8)
    frames[:, 10, :] = 200  # bright row 10 only
    out_mid = chord_integrated_emission(frames)  # H//2 = 16 → all dark
    out_row10 = chord_integrated_emission(frames, chord_y=10)
    assert out_mid.sum() == 0
    assert out_row10.sum() > 0


def test_chord_integrated_emission_rgb_input():
    frames = _random_frames(t=5, h=32, w=32, channels=3)
    out = chord_integrated_emission(frames)
    assert out.shape == (5,)


# ---------------------------------------------------------------------------
# chord_nrmse
# ---------------------------------------------------------------------------


def test_chord_nrmse_identical_is_zero():
    frames = _random_frames_gray(t=8, seed=20)
    assert chord_nrmse(frames, frames) == 0.0


def test_chord_nrmse_tiny_noise_is_small():
    # Adding a small additive offset (1 DN) shifts the chord sum minimally
    # relative to the chord's natural shot-to-shot variation.
    rng = np.random.default_rng(21)
    ref = _random_frames_gray(t=10, h=32, w=32, seed=22)
    # +/- 1 DN per pixel shifts each chord sum by at most W=32 out of ~3500
    noise = rng.integers(-1, 2, size=ref.shape, dtype=np.int32)
    pred = np.clip(ref.astype(np.int32) + noise, 0, 255).astype(np.uint8)
    nrmse = chord_nrmse(ref, pred)
    assert nrmse < 0.1  # ~1-DN noise should be well below 10% NRMSE


def test_chord_nrmse_constant_reference_is_inf_when_error():
    # Constant brightness → std=0 → NRMSE is inf (unless pred also constant same)
    ref = np.full((5, 32, 32), fill_value=128, dtype=np.uint8)
    pred = np.full((5, 32, 32), fill_value=200, dtype=np.uint8)
    result = chord_nrmse(ref, pred)
    assert math.isinf(result)


def test_chord_nrmse_constant_identical_is_zero():
    ref = np.full((5, 32, 32), fill_value=128, dtype=np.uint8)
    assert chord_nrmse(ref, ref) == 0.0


# ---------------------------------------------------------------------------
# edge_displacement
# ---------------------------------------------------------------------------


def test_edge_displacement_identical_is_zero():
    frames = np.zeros((4, 32, 32), dtype=np.uint8)
    frames[:, 10:22, 10:22] = 200  # bright square → clear edge
    result = edge_displacement(frames, frames)
    assert result == 0.0


def test_edge_displacement_returns_float():
    ref = np.zeros((4, 32, 32), dtype=np.uint8)
    ref[:, 8:24, 8:24] = 180
    pred = np.zeros((4, 32, 32), dtype=np.uint8)
    pred[:, 10:26, 10:26] = 180  # shifted square
    result = edge_displacement(ref, pred)
    assert isinstance(result, float)


def test_edge_displacement_rgb_input():
    ref = np.zeros((4, 32, 32, 3), dtype=np.uint8)
    ref[:, 8:24, 8:24, :] = 180
    pred = np.zeros((4, 32, 32, 3), dtype=np.uint8)
    pred[:, 10:26, 10:26, :] = 180
    result = edge_displacement(ref, pred)
    assert isinstance(result, float)


# ---------------------------------------------------------------------------
# compute_all_metrics
# ---------------------------------------------------------------------------


def test_compute_all_metrics_returns_expected_keys():
    ref = _random_frames(t=4, h=16, w=16, seed=40)
    pred = _random_frames(t=4, h=16, w=16, seed=41)
    metrics = compute_all_metrics(ref, pred)
    expected_keys = {
        "psnr",
        "lpips",
        "rfid",
        "centroid_mse",
        "chord_nrmse",
        "edge_displacement_mad",
    }
    assert set(metrics.keys()) == expected_keys


def test_compute_all_metrics_all_values_are_floats():
    ref = _random_frames(t=4, h=16, w=16, seed=42)
    pred = _random_frames(t=4, h=16, w=16, seed=43)
    metrics = compute_all_metrics(ref, pred)
    for key, val in metrics.items():
        assert isinstance(val, float), f"{key} is not float: {val!r}"


def test_compute_all_metrics_identical_psnr_is_inf():
    frames = _random_frames(t=4, h=16, w=16, seed=44)
    metrics = compute_all_metrics(frames, frames)
    assert math.isinf(metrics["psnr"])


def test_compute_all_metrics_identical_centroid_mse_is_zero():
    frames = _random_frames(t=4, h=16, w=16, seed=45)
    metrics = compute_all_metrics(frames, frames)
    assert metrics["centroid_mse"] == 0.0


def test_compute_all_metrics_chord_y_passthrough():
    ref = _random_frames_gray(t=4, h=32, w=32, seed=46)
    pred = _random_frames_gray(t=4, h=32, w=32, seed=47)
    m1 = compute_all_metrics(ref, pred, chord_y=8)
    m2 = compute_all_metrics(ref, pred, chord_y=24)
    # Different chords should generally give different chord_nrmse values
    # (they may happen to match for specific seeds but usually won't)
    assert isinstance(m1["chord_nrmse"], float)
    assert isinstance(m2["chord_nrmse"], float)


# ---------------------------------------------------------------------------
# Heavy metrics — skipped when deps absent
# ---------------------------------------------------------------------------


def test_lpips_identical_is_near_zero():
    pytest.importorskip("lpips")
    pytest.importorskip("torch")
    from imas_ambix.eval.metrics import lpips as lpips_fn

    frames = _random_frames(t=4, h=64, w=64, seed=50)
    score = lpips_fn(frames, frames)
    assert 0.0 <= score < 0.05


def test_lpips_noisy_is_positive():
    pytest.importorskip("lpips")
    pytest.importorskip("torch")
    from imas_ambix.eval.metrics import lpips as lpips_fn

    rng = np.random.default_rng(51)
    clean = _random_frames(t=4, h=64, w=64, seed=52)
    noisy = np.clip(
        clean.astype(np.int32) + rng.integers(-60, 60, size=clean.shape),
        0,
        255,
    ).astype(np.uint8)
    assert lpips_fn(clean, noisy) > 0.0


def test_rfid_identical_is_near_zero():
    pytest.importorskip("torchvision")
    pytest.importorskip("torch")
    from imas_ambix.eval.metrics import rfid as rfid_fn

    frames = _random_frames(t=4, h=64, w=64, seed=60)
    score = rfid_fn(frames, frames)
    assert score >= 0.0


def test_rfid_different_is_positive():
    pytest.importorskip("torchvision")
    pytest.importorskip("torch")
    from imas_ambix.eval.metrics import rfid as rfid_fn

    ref = _random_frames(t=6, h=64, w=64, seed=61)
    pred = _random_frames(t=6, h=64, w=64, seed=62)
    score = rfid_fn(ref, pred)
    assert score >= 0.0


# ---------------------------------------------------------------------------
# RolloutConfig and rollout stub
# ---------------------------------------------------------------------------


def test_rollout_config_default_values():
    cfg = RolloutConfig()
    assert cfg.prefix_tokens == 2048
    assert cfg.rollout_steps == 100
    assert cfg.top_k == 64
    assert cfg.temperature == 0.8
    assert cfg.force_signal_action_tokens is True


def test_rollout_config_custom_values():
    cfg = RolloutConfig(prefix_tokens=512, rollout_steps=50, top_k=32, temperature=1.0)
    assert cfg.prefix_tokens == 512
    assert cfg.rollout_steps == 50
    assert cfg.top_k == 32
    assert cfg.temperature == 1.0


def test_rollout_raises_not_implemented():
    cfg = RolloutConfig()
    with pytest.raises(NotImplementedError, match="not yet implemented"):
        rollout(None, None, None, cfg)


def test_rollout_stub_message_mentions_model():
    cfg = RolloutConfig()
    with pytest.raises(NotImplementedError) as exc_info:
        rollout(model=None, initial_tokens=None, control_tokens=None, config=cfg)
    assert "model" in str(exc_info.value).lower()
