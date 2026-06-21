"""Unit tests for the horizon-spanning windowing (the persistence-trap fix).

The bug: with frame_stride=1 the n_frames window was n_frames CONSECUTIVE native
frames (~12-24ms at MAST's 1-2kHz), far shorter than the ~125ms ramp-up, so the
model trained on near-static clips and learned frame persistence, not actuator
cause-effect.  The fix derives a PER-SHOT frame_stride from the shot's cadence so
n_frames span ~target_horizon_s.  These assert that derivation.
"""

from __future__ import annotations

import numpy as np

from imas_ambix.worldmodel.spacetime_dataset import (
    SpacetimeWindowConfig,
    _fps_from_times,
    effective_frame_stride,
    effective_window_span,
)


def test_fps_from_times_median():
    # 2 kHz cadence (0.5 ms spacing) with one dropped frame -> median still 2000.
    t = np.arange(100, dtype=np.float64) * 5e-4
    t = np.delete(t, 50)  # drop a frame (a 1ms gap appears once)
    assert abs(_fps_from_times(t) - 2000.0) < 1.0
    # 1 kHz.
    assert abs(_fps_from_times(np.arange(50) * 1e-3) - 1000.0) < 1.0
    # degenerate axes -> None.
    assert _fps_from_times(None) is None
    assert _fps_from_times(np.array([1.0])) is None
    assert _fps_from_times(np.zeros(10)) is None


def test_effective_frame_stride_spans_horizon():
    cfg = SpacetimeWindowConfig(n_frames=24, target_horizon_s=0.25)
    # 2000 fps: stride = round(0.25*2000/24) = round(20.8) = 21 -> 24*21/2000=0.252s
    assert effective_frame_stride(cfg, 2000.0) == 21
    # 1000 fps: round(0.25*1000/24)=round(10.4)=10 -> 24*10/1000=0.24s
    assert effective_frame_stride(cfg, 1000.0) == 10
    # 600 fps: round(0.25*600/24)=round(6.25)=6 -> 24*6/600=0.24s
    assert effective_frame_stride(cfg, 600.0) == 6
    # the resulting physical span is within ~15% of the 0.25s target for each.
    for fps in (600.0, 1000.0, 2000.0):
        st = effective_frame_stride(cfg, fps)
        span_s = cfg.n_frames * st / fps
        assert abs(span_s - 0.25) / 0.25 < 0.15, (fps, st, span_s)


def test_stride_floor_and_horizon_off():
    cfg = SpacetimeWindowConfig(n_frames=24, target_horizon_s=0.25)
    # a very low fps can't fill the horizon -> stride floored at 1 (never 0).
    assert effective_frame_stride(cfg, 10.0) == 1
    # unknown fps -> falls back to the literal frame_stride.
    cfg2 = SpacetimeWindowConfig(n_frames=24, frame_stride=3, target_horizon_s=0.25)
    assert effective_frame_stride(cfg2, None) == 3
    # target_horizon_s=0 -> always the literal frame_stride (legacy path).
    cfg3 = SpacetimeWindowConfig(n_frames=24, frame_stride=2, target_horizon_s=0.0)
    assert effective_frame_stride(cfg3, 2000.0) == 2


def test_effective_window_span():
    cfg = SpacetimeWindowConfig(n_frames=24, target_horizon_s=0.25)
    # span = (n_frames-1)*stride + 1; at 2000 fps stride=21 -> 23*21+1 = 484.
    assert effective_window_span(cfg, 2000.0) == 23 * 21 + 1
    # at 1000 fps stride=10 -> 23*10+1 = 231.
    assert effective_window_span(cfg, 1000.0) == 23 * 10 + 1
    # the old bug (24 consecutive native frames) was span=24; the fix is ~20x more.
    assert effective_window_span(cfg, 2000.0) > 20 * 24


def test_horizon_default_is_quarter_second():
    # the config default must enable the horizon window (not the legacy 15ms).
    cfg = SpacetimeWindowConfig()
    assert cfg.target_horizon_s == 0.25
    assert effective_frame_stride(cfg, 2000.0) > 1, (
        "default config still produces stride 1 — the persistence-trap bug"
    )
