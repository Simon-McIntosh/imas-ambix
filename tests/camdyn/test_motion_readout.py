"""Tests for the phase-sensitive filament-motion readout."""

from __future__ import annotations

import json

import numpy as np
import pytest

from imas_ambix.camdyn.motion_readout import (
    _strict_json_data,
    dynamic_range_verdict,
    motion_readout,
)


def _moving_filament(n_frames: int = 18) -> np.ndarray:
    """Synthetic bright filament translating through a dim structured field."""
    yy, xx = np.mgrid[:256, :256]
    frames = []
    for frame in range(n_frames):
        centre_x = 64.0 + 3.0 * frame
        centre_y = 210.0 + 4.0 * np.sin(frame / 3.0)
        filament = 180.0 * np.exp(
            -((xx - centre_x) ** 2 / (2 * 5.0**2) + (yy - centre_y) ** 2 / (2 * 9.0**2))
        )
        background = 25.0 + 5.0 * np.sin(xx / 30.0) + 3.0 * np.cos(yy / 21.0)
        frames.append(background + filament)
    return np.stack(frames)


def _filament_cells() -> np.ndarray:
    cells = np.zeros((16, 16), dtype=bool)
    cells[11:16, 2:12] = True
    return cells


def test_oracle_and_persistence_anchor_dynamic_range():
    truth = _moving_filament()
    persistence = np.repeat(truth[:1], truth.shape[0], axis=0)
    post = list(range(truth.shape[0]))

    oracle = motion_readout(truth, truth, _filament_cells(), post)
    frozen = motion_readout(persistence, truth, _filament_cells(), post)

    assert oracle["score"] == pytest.approx(1.0, abs=1e-12)
    assert frozen["score"] == pytest.approx(0.0, abs=1e-12)
    assert oracle["n_transitions"] == truth.shape[0] - 1


def test_partial_motion_is_between_persistence_and_oracle():
    truth = _moving_filament()
    persistence = np.repeat(truth[:1], truth.shape[0], axis=0)
    partial = 0.55 * truth + 0.45 * persistence
    result = motion_readout(
        partial, truth, _filament_cells(), list(range(truth.shape[0]))
    )

    assert 0.55 < result["score"] < 0.9
    assert all(-1.0 <= row["score"] <= 1.0 for row in result["transitions"])


def test_wrong_phase_motion_is_rejected():
    truth = _moving_filament()
    wrong_direction = truth[::-1].copy()
    result = motion_readout(
        wrong_direction, truth, _filament_cells(), list(range(truth.shape[0]))
    )

    assert result["score"] < 0.1


def test_nonmeasurable_transition_is_encoded_as_strict_json_null():
    truth = _moving_filament(4)
    truth[1] = truth[0]
    result = motion_readout(truth, truth, _filament_cells(), [0, 1, 2, 3])

    assert result["transitions"][0]["measurable"] is False
    assert result["transitions"][0]["score"] is None
    assert result["transitions"][0]["displacement_skill"] is None
    assert result["n_nonmeasurable_transitions"] >= 1
    encoded = _strict_json_data(result)
    payload = json.dumps(encoded, allow_nan=False)
    assert '"score": null' in payload


def test_invalid_support_and_frame_contracts_fail_closed():
    truth = _moving_filament(4)
    with pytest.raises(ValueError, match="filament_cells"):
        motion_readout(truth, truth, np.ones((4, 4), dtype=bool), [0, 1])
    with pytest.raises(ValueError, match="at least two"):
        motion_readout(truth, truth, _filament_cells(), [0])
    with pytest.raises(ValueError, match="shapes differ"):
        motion_readout(truth[:-1], truth, _filament_cells(), [0, 1])


def test_dynamic_range_verdict_reports_headroom():
    verdict = dynamic_range_verdict(
        {"persistence": 0.0, "oracle": 1.0, "coloured_noise": 0.01, "map": 0.31}
    )

    assert verdict["validated"] is True
    assert all(verdict["checks"].values())
    assert verdict["headroom"] == "HEADROOM"
    assert verdict["map_minus_persistence"] == pytest.approx(0.31)


def test_dynamic_range_verdict_blocks_gameable_noise():
    verdict = dynamic_range_verdict(
        {"persistence": 0.0, "oracle": 1.0, "coloured_noise": 0.21, "map": 0.31}
    )

    assert verdict["validated"] is False
    assert verdict["checks"]["coloured_noise_near_zero"] is False
    assert verdict["headroom"] == "UNJUDGEABLE"
