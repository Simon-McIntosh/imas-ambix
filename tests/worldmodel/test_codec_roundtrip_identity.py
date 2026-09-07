from __future__ import annotations

import numpy as np

from imas_ambix.worldmodel.codec_roundtrip_identity import (
    analyse_roundtrip,
    fit_scale_and_offset,
)


def test_scale_and_offset_recovery_on_planted_rescaling() -> None:
    roundtrip = np.arange(3 * 4 * 5, dtype=np.float64).reshape(3, 4, 5)
    reference = 0.25 * roundtrip + 17.0

    scale, offset, residual = fit_scale_and_offset(roundtrip, reference)

    np.testing.assert_allclose(scale, 0.25, atol=1e-12)
    np.testing.assert_allclose(offset, 17.0, atol=1e-12)
    assert residual < 1e-12


def test_identity_roundtrip_is_reported_as_faithful() -> None:
    rng = np.random.default_rng(42)
    frames = rng.integers(12, 38, size=(4, 8, 8, 3), dtype=np.uint8)

    result = analyse_roundtrip(frames, frames.copy())

    assert result["mean_absolute_error_u8"] == 0.0
    assert result["dynamic_range_preserved"] is True
    assert "model is convicted" in str(result["verdict"])
    assert len(result["per_frame_intensity"]) == 4
