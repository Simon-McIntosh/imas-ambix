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


# ---------------------------------------------------------------------------
# manifest-window corpus path (option i): per-window start_frame + per-shot stride
# ---------------------------------------------------------------------------


def test_manifest_train_windows_derives_per_shot_stride(tmp_path):
    import json

    from imas_ambix.worldmodel.controllable_train import manifest_train_windows

    manifest = tmp_path / "m.json"
    manifest.write_text(
        json.dumps(
            {
                "horizon_s": 0.25,
                "windows": [
                    {"shot_id": 100, "start_frame": 506, "fps": 2000.0, "frame_stride": 11},  # noqa: E501
                    {"shot_id": 200, "start_frame": 126, "fps": 500.0, "frame_stride": 3},  # noqa: E501
                    {"shot_id": 18502, "start_frame": 9, "fps": 2000.0},  # held-out
                ],
            }
        )
    )
    ws = manifest_train_windows(
        manifest, held_out={18502}, target_horizon_s=0.25, n_frames=24
    )
    # held-out excluded; 2 train windows.
    assert {w.shot_id for w in ws} == {100, 200}
    by = {w.shot_id: w for w in ws}
    # 2000 fps -> round(0.25*2000/24)=21; 500 fps -> round(0.25*500/24)=5.
    assert by[100].frame_stride == 21 and by[100].start_frame == 506
    assert by[200].frame_stride == 5 and by[200].start_frame == 126
    # the recorded fps is carried for logging.
    assert by[100].fps == 2000.0


def test_manifest_train_windows_horizon_off_uses_manifest_stride(tmp_path):
    import json

    from imas_ambix.worldmodel.controllable_train import manifest_train_windows

    manifest = tmp_path / "m.json"
    win = {"shot_id": 1, "start_frame": 0, "fps": 2000.0, "frame_stride": 11}
    manifest.write_text(json.dumps({"windows": [win]}))
    # target_horizon_s=0 -> fall back to the manifest's own frame_stride.
    ws = manifest_train_windows(
        manifest, held_out=set(), target_horizon_s=0.0, n_frames=24
    )
    assert ws[0].frame_stride == 11


def test_manifest_window_dataset_assembles_at_start(monkeypatch):
    """ManifestWindowDataset assembles at B's start_frame; horizon config passed."""
    import imas_ambix.worldmodel.controllable_train as ct
    from imas_ambix.worldmodel.controllable_train import (
        ManifestWindowDataset,
        _ManifestWindow,
    )
    from imas_ambix.worldmodel.spacetime_dataset import SpacetimeWindowConfig

    captured = {}

    def _fake_assemble(shot_id, config, modalities, n_sig, n_act, **kw):
        captured["shot_id"] = shot_id
        captured["target_horizon_s"] = config.target_horizon_s
        captured["start_frame"] = kw.get("start_frame")
        return "SAMPLE"

    monkeypatch.setattr(ct, "assemble_controllable_window", _fake_assemble)
    ds = ManifestWindowDataset(
        [_ManifestWindow(shot_id=42, start_frame=506, frame_stride=21, fps=2000.0)],
        SpacetimeWindowConfig(n_frames=24, target_horizon_s=0.25),
        [],
        4,
        8,
    )
    assert len(ds) == 1
    assert ds[0] == "SAMPLE"
    # B's start_frame is passed through; the TIME-BASED path stays enabled
    # (target_horizon_s=0.25, NOT zeroed — assemble_window subsamples by time, so
    # the manifest's fixed stride is intentionally not used).
    assert captured["shot_id"] == 42
    assert captured["target_horizon_s"] == 0.25
    assert captured["start_frame"] == 506


# ---------------------------------------------------------------------------
# time-based subsample (cadence-robust): spans the horizon even when fps changes
# ---------------------------------------------------------------------------


def test_time_spanned_indices_uniform_cadence():
    from imas_ambix.worldmodel.spacetime_dataset import _time_spanned_indices

    t = np.arange(2000, dtype=np.float64) * 5e-4  # 2 kHz, 1.0s recording
    idx, span = _time_spanned_indices(
        t, t.size, n_frames=24, horizon_s=0.25, start_frame=0
    )
    assert idx.shape == (24,)
    assert idx[0] == 0
    # span is ~0.25s (within one frame dt).
    assert abs(span - 0.25) < 6e-4
    # strictly increasing.
    assert np.all(np.diff(idx) > 0)


def test_time_spanned_indices_variable_cadence_still_spans_horizon():
    """The killer case: cadence ACCELERATES mid-window — span must stay ~0.25s."""
    from imas_ambix.worldmodel.spacetime_dataset import _time_spanned_indices

    # first 0.20s at 2kHz (dt 0.5ms), then 0.30s at 8kHz (dt 0.125ms).
    t1 = np.arange(400) * 5e-4  # 0..0.1995
    t2 = t1[-1] + 1.25e-4 * (1 + np.arange(2400))  # accelerated tail
    t = np.concatenate([t1, t2])
    idx, span = _time_spanned_indices(
        t, t.size, n_frames=24, horizon_s=0.25, start_frame=0
    )
    # the fixed-stride bug undershot here; the time-based pick spans ~0.25s.
    assert abs(span - 0.25) < 1e-3, span
    assert np.all(np.diff(idx) > 0)
    # the index stride is NON-uniform (smaller in the fast region) — proof it's
    # time-driven, not a constant frame step.
    steps = np.diff(idx)
    assert steps.max() > steps.min(), "stride should vary with the cadence"


def test_time_spanned_indices_backs_off_start_past_end():
    from imas_ambix.worldmodel.spacetime_dataset import _time_spanned_indices

    t = np.arange(600, dtype=np.float64) * 5e-4  # 0.3s recording
    # start near the end: not enough room for 0.25s forward -> backs off.
    idx, span = _time_spanned_indices(
        t, t.size, n_frames=24, horizon_s=0.25, start_frame=550
    )
    assert idx[-1] <= t.size - 1
    assert abs(span - 0.25) < 6e-4


def test_time_spanned_indices_raises_when_recording_too_short():
    import pytest

    from imas_ambix.worldmodel.spacetime_dataset import _time_spanned_indices

    t = np.arange(100, dtype=np.float64) * 5e-4  # 0.05s — shorter than 0.25s
    with pytest.raises(ValueError, match="horizon"):
        _time_spanned_indices(t, t.size, n_frames=24, horizon_s=0.25, start_frame=0)
