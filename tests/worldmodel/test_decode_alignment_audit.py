from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from imas_ambix.worldmodel.decode_alignment_audit import (
    audit_alignment,
    brightness_centroid_displacements,
    least_squares_intensity_mapping,
    mean_absolute_error_matrix,
    split_gif_panels,
    temporal_lag_curve,
    translate_images,
    write_audit,
)


def _moving_frames(count: int = 12) -> np.ndarray:
    frames = np.zeros((count, 24, 24), dtype=np.float64)
    for index in range(count):
        y = 3 + index
        x = 4 + (3 * index) % 13
        frames[index, y : y + 4, x : x + 5] = 40.0 + 10.0 * index
    return frames


def test_temporal_curve_recovers_planted_three_frame_lag():
    real = _moving_frames()
    decoded = np.full_like(real, 255.0)
    decoded[:-3] = real[3:]

    matrix = mean_absolute_error_matrix(decoded, real)
    curve, best_lag = temporal_lag_curve(matrix)

    assert best_lag == 3
    best = next(point for point in curve if point["lag_frames"] == best_lag)
    assert best["mean_absolute_error_u8"] == pytest.approx(0.0)


def test_brightness_centroid_recovers_rightward_seven_pixel_translation():
    real = np.zeros((4, 32, 40), dtype=np.float64)
    real[:, 9:15, 8:14] = 100.0
    decoded = translate_images(real, dx_px=7, dy_px=0)

    displacement = brightness_centroid_displacements(decoded, real)

    assert np.median(displacement[:, 0]) == pytest.approx(7.0)
    assert np.median(displacement[:, 1]) == pytest.approx(0.0)
    assert np.hypot(*np.median(displacement, axis=0)) == pytest.approx(7.0)


def test_intensity_fit_recovers_planted_halving():
    decoded = np.arange(2 * 8 * 9, dtype=np.float64).reshape(2, 8, 9)
    real = 0.5 * decoded + 7.0

    mapping = least_squares_intensity_mapping(decoded, real)

    assert mapping["scale"] == pytest.approx(0.5)
    assert mapping["offset_u8"] == pytest.approx(7.0)
    assert mapping["residual_mean_absolute_error_u8"] == pytest.approx(0.0)


def test_split_and_writer_preserve_complete_evidence(tmp_path):
    rng = np.random.default_rng(42)
    real = rng.integers(0, 100, size=(8, 256, 256, 3), dtype=np.uint8)
    decoded = (real.astype(np.float64) * 2.0).clip(0, 255).astype(np.uint8)
    gif_frames = []
    for real_frame, decoded_frame in zip(real, decoded, strict=True):
        canvas = np.zeros((288, 512, 3), dtype=np.uint8)
        canvas[32:, :256] = real_frame
        canvas[32:, 256:] = decoded_frame
        gif_frames.append(Image.fromarray(canvas))
    gif_path = tmp_path / "paired.gif"
    gif_frames[0].save(
        gif_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=100,
        loop=0,
        optimize=False,
    )

    loaded_real, loaded_decoded = split_gif_panels(gif_path)
    audit = audit_alignment(loaded_real, loaded_decoded, reported_error_u8=12.0)
    json_path, figure_path = write_audit(audit, tmp_path / "audit")
    receipt = json.loads(json_path.read_text())

    assert loaded_real.shape == (8, 256, 256, 3)
    assert loaded_decoded.shape == loaded_real.shape
    assert np.asarray(receipt["mae_matrix_u8"]).shape == (8, 8)
    assert receipt["temporal"]["best_lag_frames"] == 0
    assert receipt["intensity_mapping"]["scale"] == pytest.approx(0.5, abs=0.02)
    assert set(receipt["error_decomposition_u8"]) == {
        "temporal_lag",
        "spatial_translation",
        "intensity_scale",
        "genuine_content_error",
    }
    assert figure_path.stat().st_size > 10_000
