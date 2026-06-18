"""Decoded-pixel-error vs persistence scoring for held-out spacetime decodes.

A held-out token-mismatch near the 2**18-vocab saturation point coexisted with
coherent-LOOKING but non-forecasting video on the last run; the honest signal
was decoded-pixel error vs a persistence baseline (freeze the last context frame
across the forecast window).  These tests pin that scorer so the held-out
verdict is reproducible, not measured by hand.
"""

from __future__ import annotations

import numpy as np

from imas_ambix.worldmodel.spacetime_dream import forecast_pixel_errors


def _ramp_stack(n_frames: int, value_per_frame, h: int = 8, w: int = 8) -> np.ndarray:
    """An (F, H, W, 3) uint8 stack where frame f is filled with value_per_frame[f]."""
    stack = np.zeros((n_frames, h, w, 3), dtype=np.uint8)
    for f in range(n_frames):
        stack[f] = value_per_frame[f]
    return stack


def test_perfect_prediction_beats_persistence_on_moving_target():
    """A prediction equal to GT has zero error; persistence (frozen frame) does not.

    GT moves every frame (10, 20, 30, 40); context is the first 2 frames so the
    forecast window is frames 2..3 (values 30, 40).  A perfect model matches GT
    exactly -> 0 error.  Persistence freezes gt[ctx-1] = frame 1 (value 20), so it
    is off by |30-20|=10 and |40-20|=20 -> mean 15.
    """
    gt = _ramp_stack(4, [10, 20, 30, 40])
    pred = gt.copy()  # perfect prediction
    out = forecast_pixel_errors(gt, pred, ctx=2)
    assert out["model_pixel_error"] == 0.0
    assert out["persistence_pixel_error"] == 15.0
    # ratio = model / persistence; 0 means the model is infinitely better.
    assert out["ratio"] == 0.0
    assert out["model_beats_persistence"] is True


def test_static_scene_persistence_is_unbeatable():
    """When GT does not move, persistence is perfect (0 error) and any drift loses.

    GT is constant at 50 across all 4 frames; ctx=2.  Persistence freezes frame 1
    (50) and is exactly right on the forecast (error 0).  A model that drifts to 60
    on every forecast frame is off by 10 -> loses to persistence (ratio = inf-like;
    we report model worse).
    """
    gt = _ramp_stack(4, [50, 50, 50, 50])
    pred = _ramp_stack(4, [50, 50, 60, 60])  # forecast frames drift +10
    out = forecast_pixel_errors(gt, pred, ctx=2)
    assert out["persistence_pixel_error"] == 0.0
    assert out["model_pixel_error"] == 10.0
    assert out["model_beats_persistence"] is False


def test_only_forecast_frames_are_scored():
    """Errors on context frames (fi < ctx) must NOT count — only fi >= ctx.

    GT = (10, 20, 30); ctx=2 so only frame 2 is forecast.  Make the model wildly
    wrong on the CONTEXT frames (which must be ignored) but exact on the forecast
    frame -> model_pixel_error must be 0.
    """
    gt = _ramp_stack(3, [10, 20, 30])
    pred = _ramp_stack(3, [99, 99, 30])  # garbage in context, exact on forecast
    out = forecast_pixel_errors(gt, pred, ctx=2)
    assert out["model_pixel_error"] == 0.0
