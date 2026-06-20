"""Shot-level leakage guard for the overlapping-window training pipeline.

The world model holds out WHOLE pulses (18502 / 18503 / 18504 / 18505 — the
fixed eval set, incl. the bright shots 18502 / 18505) for evaluation.  The
overlapping-window pipeline tiles each TRAIN recording into many sliding windows
to maximise the training signal from the fixed corpus.  The binding safety
property is that NO window from a held-out shot can ever enter the training set:

* :func:`enumerate_windows` only ever emits ``(shot_id, start)`` pairs whose
  ``shot_id`` is one of the ids handed in — it never reaches outside the list.
  The trainer subtracts the held-out shots BEFORE calling it, so the guarantee
  is structural; this test asserts the emitted set carries ZERO held-out ids
  even when held-out shots are deliberately mixed into a candidate pool.
* The dataset built from the window list exposes only the train shots.

These tests monkeypatch ``camera_frame_count`` so they run without the on-disk
token corpus (the leakage property is independent of frame counts).
"""

from __future__ import annotations

import pytest

from imas_ambix.worldmodel import spacetime_dataset_v2 as ds_v2
from imas_ambix.worldmodel.spacetime_dataset import SpacetimeWindowConfig
from imas_ambix.worldmodel.spacetime_dataset_v2 import (
    OverlappingSignalWindowDataset,
    default_signal_modalities,
    enumerate_windows,
)

# The fixed held-out set for the camera world model (the M1/M2 eval shots).  The
# bright shots are 18502 / 18505.  This mirrors scripts/slurm/*v2*.sbatch
# WM_EVAL_SHOTS and imas_ambix/worldmodel/sampling_rescore.py.
HELD_OUT = (18502, 18503, 18504, 18505)


@pytest.fixture
def patched_frame_count(monkeypatch):
    """Make every shot report a long-enough recording (no disk needed)."""

    def _fake_count(shot_id, camera, *, token_root=None):  # noqa: ANN001, ARG001
        return 6000  # long recording -> many windows per shot

    monkeypatch.setattr(ds_v2, "camera_frame_count", _fake_count)
    return _fake_count


def _window_cfg() -> SpacetimeWindowConfig:
    # the M1/M2 window: 24 native-cadence frames, 8 context, stride-1 cadence.
    return SpacetimeWindowConfig(
        n_frames=24, n_plan=8, context_frames=8, frame_stride=1
    )


def test_enumerate_windows_emits_only_input_shots(patched_frame_count):
    cfg = _window_cfg()
    train_shots = [15085, 15086, 15087, 16000, 17000]
    windows = enumerate_windows(train_shots, cfg, window_stride=8)
    assert windows, "expected windows for long recordings"
    emitted = {s for s, _ in windows}
    assert emitted <= set(train_shots), emitted - set(train_shots)


def test_enumerate_windows_excludes_held_out(patched_frame_count):
    """The core leakage assertion: held-out shots subtracted upstream stay out.

    Build a candidate pool that INCLUDES the held-out shots, subtract them the
    way the trainer does, enumerate, and assert ZERO emitted windows belong to a
    held-out shot.
    """
    cfg = _window_cfg()
    # a pool with the held-out shots interleaved among train shots.
    pool = [15000, 18502, 15001, 18503, 15002, 18504, 15003, 18505, 15004]
    held = set(HELD_OUT)
    train_shots = [s for s in pool if s not in held]  # the trainer's subtraction

    windows = enumerate_windows(train_shots, cfg, window_stride=8)
    leaked = sorted({s for s, _ in windows} & held)
    assert not leaked, f"held-out shots leaked into the window list: {leaked}"
    # and the train shots ARE represented (the subtraction did not drop them).
    assert {s for s, _ in windows} == set(train_shots)


def test_enumerate_windows_overlap_yields_many_per_shot(patched_frame_count):
    """50%-overlap stride must yield many windows per long recording."""
    cfg = _window_cfg()
    span = (cfg.n_frames - 1) * cfg.frame_stride + 1  # 24
    windows = enumerate_windows([15085], cfg, window_stride=max(1, span // 2))
    # 6000-frame recording, span 24, stride 12 -> ~ (6000-24)/12 + 1 windows.
    assert len(windows) > 100, len(windows)
    # all from the one shot, strictly ascending starts, last window is the tail.
    starts = [f for _, f in windows]
    assert all(s == 15085 for s, _ in windows)
    assert starts == sorted(starts)
    assert starts[-1] == 6000 - span  # tail window always included


def test_enumerate_windows_default_stride_is_half_span(patched_frame_count):
    cfg = _window_cfg()
    span = (cfg.n_frames - 1) * cfg.frame_stride + 1
    auto = enumerate_windows([15085], cfg)  # default stride
    explicit = enumerate_windows([15085], cfg, window_stride=max(1, span // 2))
    assert auto == explicit


def test_enumerate_windows_respects_max_windows_per_shot(patched_frame_count):
    cfg = _window_cfg()
    capped = enumerate_windows(
        [15085, 15086], cfg, window_stride=8, max_windows_per_shot=5
    )
    per_shot: dict[int, int] = {}
    for s, _ in capped:
        per_shot[s] = per_shot.get(s, 0) + 1
    assert all(c <= 5 for c in per_shot.values()), per_shot


def test_enumerate_windows_skips_short_recordings(monkeypatch):
    cfg = _window_cfg()
    span = (cfg.n_frames - 1) * cfg.frame_stride + 1

    def _short(shot_id, camera, *, token_root=None):  # noqa: ANN001, ARG001
        # 15085 too short for even one window; 15086 long enough.
        return span - 1 if shot_id == 15085 else 6000

    monkeypatch.setattr(ds_v2, "camera_frame_count", _short)
    windows = enumerate_windows([15085, 15086], cfg, window_stride=8)
    assert {s for s, _ in windows} == {15086}


def test_enumerate_windows_deterministic_order(patched_frame_count):
    cfg = _window_cfg()
    shots = [17000, 15085, 16000]  # deliberately unsorted
    a = enumerate_windows(shots, cfg, window_stride=12)
    b = enumerate_windows(shots, cfg, window_stride=12)
    assert a == b
    # ascending shot id, then ascending start.
    assert a == sorted(a)


def test_enumerate_windows_rejects_bad_stride(patched_frame_count):
    cfg = _window_cfg()
    with pytest.raises(ValueError, match="window_stride"):
        enumerate_windows([15085], cfg, window_stride=-1)


def test_overlapping_dataset_exposes_only_train_shots(patched_frame_count):
    """The dataset built from the window list never surfaces a held-out shot."""
    cfg = _window_cfg()
    pool = [15000, 18502, 15001, 18505, 15002]
    train_shots = [s for s in pool if s not in set(HELD_OUT)]
    windows = enumerate_windows(train_shots, cfg, window_stride=12)
    dataset = OverlappingSignalWindowDataset(
        windows,
        cfg,
        default_signal_modalities(),
        n_signal_steps=4,
    )
    assert set(dataset.shot_ids) == set(train_shots)
    assert not (set(dataset.shot_ids) & set(HELD_OUT))
    assert len(dataset) == len(windows)
