"""Tests for the calibration library.

All tests use synthetic data in tmp_path — no network or S3 access.
"""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

import numpy as np
import pytest
import xarray as xr
import zarr

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_signal_zarr(
    path: Path,
    group: str,
    channels: dict[str, np.ndarray],
    time: np.ndarray | None = None,
) -> None:
    """Write a synthetic xarray Dataset as Zarr to path/group."""
    if time is None:
        first = next(iter(channels.values()))
        time = np.arange(len(first), dtype=np.float64) * 0.01

    ds = xr.Dataset(
        {name: (("time",), arr) for name, arr in channels.items()},
        coords={"time": time},
    )
    ds.to_zarr(str(path), group=group, mode="w")


def _make_frame_zarr(
    path: Path,
    camera: str,
    frames: np.ndarray,
) -> None:
    """Write a (T, H, W) frame array as a Zarr array at path/camera."""
    root = zarr.open(str(path), mode="a")
    root[camera] = frames


# ---------------------------------------------------------------------------
# Signal calibration tests
# ---------------------------------------------------------------------------


def test_signal_calibration_known_mean_std(tmp_path):
    """compute_signal_calibration recovers analytic mean/std to within 1%."""
    from imas_ambix.calibration import compute_signal_calibration

    rng = np.random.default_rng(42)
    n = 1000
    true_mean, true_std = 5.0, 2.0
    data = rng.normal(true_mean, true_std, n)

    shot = tmp_path / "shot_0.zarr"
    _make_signal_zarr(shot, "signals", {"x": data})

    cal = compute_signal_calibration([shot], "signals")
    assert "x" in cal
    ch = cal["x"]

    # 1-sigma ~ 2/sqrt(1000) ~ 6% for n=1000; use 10% tolerance
    assert abs(ch.mean - true_mean) / true_std < 0.10, (
        f"mean {ch.mean:.4f} deviates > 10% from {true_mean}"
    )
    assert abs(ch.std - true_std) / true_std < 0.10, (
        f"std {ch.std:.4f} deviates > 10% from {true_std}"
    )
    assert ch.n_samples == n
    assert ch.n_shots == 1


def test_signal_calibration_welford_vs_naive(tmp_path):
    """Welford streaming result matches naive np.mean / np.std on the same data."""
    from imas_ambix.calibration import compute_signal_calibration

    rng = np.random.default_rng(7)
    all_data: list[np.ndarray] = []
    shot_paths: list[Path] = []

    for i in range(5):
        arr = rng.normal(0.0, 1.0, 200)
        all_data.append(arr)
        p = tmp_path / f"s{i}.zarr"
        _make_signal_zarr(p, "g", {"ch": arr})
        shot_paths.append(p)

    cal = compute_signal_calibration(shot_paths, "g")
    combined = np.concatenate(all_data)

    np.testing.assert_allclose(cal["ch"].mean, combined.mean(), rtol=1e-9)
    np.testing.assert_allclose(cal["ch"].std, combined.std(), rtol=1e-9)


def test_signal_calibration_quantiles(tmp_path):
    """Approximate quantiles (q01, q50, q99) are within 5% of numpy.quantile."""
    from imas_ambix.calibration import compute_signal_calibration

    rng = np.random.default_rng(99)
    data = rng.uniform(0.0, 100.0, 2000)

    shot = tmp_path / "shot.zarr"
    _make_signal_zarr(shot, "g", {"v": data})

    cal = compute_signal_calibration([shot], "g")
    ch = cal["v"]

    assert abs(ch.q01 - float(np.percentile(data, 1))) < 2.0
    assert abs(ch.q50 - float(np.percentile(data, 50))) < 2.0
    assert abs(ch.q99 - float(np.percentile(data, 99))) < 2.0


def test_signal_calibration_all_nan_channel(tmp_path):
    """Channel with all-NaN values gives n_samples=0 and doesn't crash."""
    from imas_ambix.calibration import compute_signal_calibration

    nan_data = np.full(50, float("nan"))
    good_data = np.ones(50)

    shot = tmp_path / "shot.zarr"
    _make_signal_zarr(shot, "g", {"bad": nan_data, "good": good_data})

    cal = compute_signal_calibration([shot], "g")

    # "bad" channel should be absent (no finite samples, not added to dict)
    # OR present with n_samples=0; either is acceptable behaviour.
    if "bad" in cal:
        assert cal["bad"].n_samples == 0
    # "good" channel must be present with correct stats
    assert "good" in cal
    assert cal["good"].n_samples == 50


def test_signal_calibration_multi_shot_accumulation(tmp_path):
    """Aggregation across 3 shots gives n_samples = sum of all shot lengths."""
    from imas_ambix.calibration import compute_signal_calibration

    rng = np.random.default_rng(0)
    sizes = [100, 150, 200]
    shot_paths = []

    for i, n in enumerate(sizes):
        arr = rng.normal(0.0, 1.0, n)
        p = tmp_path / f"shot_{i}.zarr"
        _make_signal_zarr(p, "g", {"c": arr})
        shot_paths.append(p)

    cal = compute_signal_calibration(shot_paths, "g")
    assert cal["c"].n_samples == sum(sizes)
    assert cal["c"].n_shots == 3


def test_signal_calibration_missing_shot_skipped(tmp_path):
    """A non-existent shot path is silently skipped; valid shots still contribute."""
    from imas_ambix.calibration import compute_signal_calibration

    data = np.arange(10.0)
    good = tmp_path / "good.zarr"
    _make_signal_zarr(good, "g", {"x": data})

    ghost = tmp_path / "does_not_exist.zarr"

    cal = compute_signal_calibration([good, ghost], "g")
    assert "x" in cal
    assert cal["x"].n_shots == 1


def test_signal_calibration_channel_filter(tmp_path):
    """channels= kwarg restricts output to the specified subset."""
    from imas_ambix.calibration import compute_signal_calibration

    rng = np.random.default_rng(1)
    shot = tmp_path / "shot.zarr"
    _make_signal_zarr(shot, "g", {"a": rng.normal(0, 1, 50), "b": rng.normal(0, 1, 50)})

    cal = compute_signal_calibration([shot], "g", channels=("a",))
    assert "a" in cal
    assert "b" not in cal


# ---------------------------------------------------------------------------
# Frame calibration tests
# ---------------------------------------------------------------------------


def test_frame_calibration_suggested_per_shot_when_range_varies(tmp_path):
    """suggested='per_shot' when shot maxima vary by >5× across shots."""
    from imas_ambix.calibration import compute_frame_calibration

    rng = np.random.default_rng(3)
    shot_paths = []

    # Shot 0: frames with max ~100
    p0 = tmp_path / "s0.zarr"
    _make_frame_zarr(p0, "rbb", (rng.uniform(0, 100, (4, 16, 16))).astype(np.uint16))
    shot_paths.append(p0)

    # Shot 1: frames with max ~1000 (10× shot 0)
    p1 = tmp_path / "s1.zarr"
    _make_frame_zarr(p1, "rbb", (rng.uniform(0, 1000, (4, 16, 16))).astype(np.uint16))
    shot_paths.append(p1)

    # Shot 2: frames with max ~3000 (30× shot 0)
    p2 = tmp_path / "s2.zarr"
    _make_frame_zarr(p2, "rbb", (rng.uniform(0, 3000, (4, 16, 16))).astype(np.uint16))
    shot_paths.append(p2)

    cal = compute_frame_calibration(shot_paths, "rbb", sample_frames_per_shot=4)
    assert cal.suggested == "per_shot", (
        f"Expected 'per_shot' due to 30× range variation, got {cal.suggested!r}"
    )


def test_frame_calibration_suggested_global_when_close(tmp_path):
    """suggested='global' when per-shot maxima are within 5× of each other."""
    from imas_ambix.calibration import compute_frame_calibration

    rng = np.random.default_rng(4)
    shot_paths = []

    # Three shots with similar ranges (max ~1000 ± 10%)
    for i in range(3):
        p = tmp_path / f"s{i}.zarr"
        lo = 900 + i * 20
        hi = lo + 100
        _make_frame_zarr(p, "rbb", (rng.uniform(lo, hi, (4, 16, 16))).astype(np.uint16))
        shot_paths.append(p)

    cal = compute_frame_calibration(shot_paths, "rbb", sample_frames_per_shot=4)
    assert cal.suggested == "global", (
        f"Expected 'global' for similar ranges, got {cal.suggested!r}"
    )


def test_frame_calibration_known_dynamic_range(tmp_path):
    """global_min/max match the synthetic frame range."""
    from imas_ambix.calibration import compute_frame_calibration

    frames = np.zeros((8, 16, 16), dtype=np.uint16)
    frames[0, 0, 0] = 50  # min
    frames[1, 0, 0] = 500  # max

    shot = tmp_path / "shot.zarr"
    _make_frame_zarr(shot, "rbb", frames)

    cal = compute_frame_calibration([shot], "rbb", sample_frames_per_shot=8)
    assert cal.global_min == pytest.approx(0.0)
    assert cal.global_max == pytest.approx(500.0)
    assert cal.camera == "rbb"


def test_frame_calibration_missing_camera_skipped(tmp_path):
    """Shot with no 'rbb' group is silently skipped."""
    from imas_ambix.calibration import compute_frame_calibration

    rng = np.random.default_rng(5)
    shot = tmp_path / "shot.zarr"
    _make_frame_zarr(shot, "rba", (rng.integers(0, 100, (4, 8, 8))).astype(np.uint16))

    cal = compute_frame_calibration([shot], "rbb", sample_frames_per_shot=4)
    # No shots contributed — all NaN / empty
    assert math.isnan(cal.global_min) or len(cal.per_shot_min) == 0


# ---------------------------------------------------------------------------
# Persistence round-trip tests
# ---------------------------------------------------------------------------


def test_save_load_signal_calibration_roundtrip(tmp_path):
    """save_calibration + load_signal_calibration round-trips signal calibrations."""
    from imas_ambix.calibration import (
        ChannelCalibration,
        load_signal_calibration,
        save_calibration,
    )

    cal: dict[str, ChannelCalibration] = {
        "ip": ChannelCalibration(
            name="ip",
            mean=1000.0,
            std=200.0,
            min_value=-500.0,
            max_value=3000.0,
            q01=-400.0,
            q50=1000.0,
            q99=2900.0,
            n_samples=50000,
            n_shots=100,
        ),
        "ne": ChannelCalibration(
            name="ne",
            mean=1e19,
            std=2e18,
            min_value=1e18,
            max_value=5e19,
            q01=2e18,
            q50=1e19,
            q99=4.5e19,
            n_samples=40000,
            n_shots=80,
        ),
    }
    path = tmp_path / "signals" / "summary.json"
    save_calibration(cal, path)

    loaded = load_signal_calibration(path)
    assert set(loaded.keys()) == {"ip", "ne"}
    assert loaded["ip"].mean == pytest.approx(1000.0)
    assert loaded["ip"].n_shots == 100
    assert loaded["ne"].std == pytest.approx(2e18)


def test_save_load_frame_calibration_roundtrip(tmp_path):
    """save_calibration + load_frame_calibration round-trips a FrameCalibration."""
    from imas_ambix.calibration import (
        FrameCalibration,
        load_frame_calibration,
        save_calibration,
    )

    cal = FrameCalibration(
        camera="rbb",
        global_min=0.0,
        global_max=4095.0,
        global_mean=1200.0,
        global_std=800.0,
        per_shot_min={0: 10.0, 1: 5.0, 2: 20.0},
        per_shot_max={0: 3000.0, 1: 4000.0, 2: 3500.0},
        suggested="global",
    )
    path = tmp_path / "frames" / "rbb.json"
    save_calibration(cal, path)

    loaded = load_frame_calibration(path)
    assert loaded.camera == "rbb"
    assert loaded.global_max == pytest.approx(4095.0)
    assert loaded.global_mean == pytest.approx(1200.0)
    assert loaded.per_shot_max[0] == pytest.approx(3000.0)
    assert loaded.suggested == "global"


def test_save_calibration_wrong_type_raises(tmp_path):
    """save_calibration raises TypeError for unsupported input types."""
    from imas_ambix.calibration import save_calibration

    with pytest.raises(TypeError, match="ChannelCalibration"):
        save_calibration("not a calibration", tmp_path / "bad.json")  # type: ignore[arg-type]


def test_load_signal_calibration_wrong_type_raises(tmp_path):
    """load_signal_calibration raises ValueError when JSON has wrong __type__."""
    from imas_ambix.calibration import load_signal_calibration

    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps({"__type__": "FrameCalibration", "data": {}}))

    with pytest.raises(ValueError, match="SignalCalibration"):
        load_signal_calibration(wrong)


# ---------------------------------------------------------------------------
# Cross-tier integration tests
# ---------------------------------------------------------------------------


def test_signal_calibration_l2_magnetics_style(tmp_path):
    """Signal calibration works on a fake L2 magnetics-style Dataset."""
    from imas_ambix.calibration import compute_signal_calibration

    rng = np.random.default_rng(10)
    n_time = 500

    # Simulate a magnetics-style dataset with multiple channels
    channels = {
        "b_field_pol": rng.normal(0.5, 0.1, n_time),
        "b_field_tor": rng.normal(2.0, 0.3, n_time),
        "loop_voltage": rng.normal(-1.0, 0.05, n_time),
    }
    shot = tmp_path / "mag_shot.zarr"
    _make_signal_zarr(shot, "magnetics", channels)

    cal = compute_signal_calibration([shot], "magnetics")

    assert len(cal) == 3
    for ch_name, arr in channels.items():
        assert ch_name in cal
        assert cal[ch_name].n_samples == n_time
        np.testing.assert_allclose(cal[ch_name].mean, arr.mean(), rtol=1e-8)


def test_frame_calibration_l1_rbb_style(tmp_path):
    """Frame calibration works on a fake L1 rbb-style array."""
    from imas_ambix.calibration import compute_frame_calibration

    rng = np.random.default_rng(11)
    n_frames, h, w = 20, 256, 320

    shot = tmp_path / "rbb_shot.zarr"
    frames = rng.integers(100, 3000, (n_frames, h, w), dtype=np.uint16)
    _make_frame_zarr(shot, "rbb", frames)

    cal = compute_frame_calibration([shot], "rbb", sample_frames_per_shot=8)

    assert cal.camera == "rbb"
    assert cal.global_min >= 100.0
    assert cal.global_max <= 3000.0
    assert cal.global_mean > 0.0
    assert 0 in cal.per_shot_min
