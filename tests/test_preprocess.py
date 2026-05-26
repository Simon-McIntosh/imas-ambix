"""Tests for imas_ambix.data.preprocess.

All tests are fully offline (synthetic arrays, tmp_path).
No GPU, no real FAIR-MAST data, no network.

Key contracts verified:
  - _normalise_to_uint8 is bit-exact with tokenizer._normalise_frames_to_uint8
  - preprocess_rbb_shot produces (T, image_size, image_size, 3) uint8
  - skip-existing logic works correctly
  - bulk_preprocess returns one report per shot in input order
  - CLI subcommand preprocess-frames is wired up and accepts expected flags
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import zarr

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_UINT16 = np.dtype("uint16")


def make_frame_zarr(
    base_dir: Path,
    shot_id: int,
    camera: str = "rbb",
    *,
    t: int = 10,
    h: int = 32,
    w: int = 34,
    dtype: np.dtype = _UINT16,
    seed: int = 42,
) -> Path:
    """Create a minimal L1-style frame Zarr for shot_id under base_dir."""
    rng = np.random.default_rng(seed)
    shot_path = base_dir / f"{shot_id}.zarr"
    shot_path.mkdir(parents=True, exist_ok=True)
    g = zarr.open_group(str(shot_path), mode="w")
    cam = g.create_group(camera)
    data = rng.integers(100, 60000, (t, h, w), dtype=np.uint16)
    if dtype == np.uint8:
        data = data.astype(np.uint8)
    cam.create_array(
        "data",
        data=data,
        dimension_names=["time", "y", "x"],
    )
    cam.create_array(
        "time",
        data=np.arange(t, dtype=np.float64) * 0.01,
        dimension_names=["time"],
    )
    return shot_path


# ---------------------------------------------------------------------------
# _normalise_to_uint8 — bit-exact parity with tokenizer._normalise_frames_to_uint8
# ---------------------------------------------------------------------------


def test_normalise_to_uint8_bit_exact_vs_tokenizer() -> None:
    """_normalise_to_uint8 must match the tokenizer's implementation byte-for-byte."""
    from imas_ambix.data.preprocess import _normalise_to_uint8
    from imas_ambix.tokenizer.frames import _normalise_frames_to_uint8

    rng = np.random.default_rng(0)
    frames_u16 = rng.integers(500, 55000, (20, 64, 64), dtype=np.uint16)

    preprocess_result = _normalise_to_uint8(frames_u16)
    tokenizer_result = _normalise_frames_to_uint8(frames_u16)

    assert preprocess_result.shape == tokenizer_result.shape
    assert preprocess_result.dtype == tokenizer_result.dtype == np.uint8
    np.testing.assert_array_equal(
        preprocess_result,
        tokenizer_result,
        err_msg=(
            "_normalise_to_uint8 diverges from "
            "tokenizer._normalise_frames_to_uint8"
        ),
    )


def test_normalise_to_uint8_already_uint8() -> None:
    """uint8 input is returned unchanged (no-op)."""
    from imas_ambix.data.preprocess import _normalise_to_uint8

    x = np.array([[[0, 128, 255]], [[100, 200, 50]]], dtype=np.uint8)
    result = _normalise_to_uint8(x)
    assert result.dtype == np.uint8
    np.testing.assert_array_equal(result, x)


def test_normalise_to_uint8_flat_input_returns_zeros() -> None:
    """When hi == lo, output must be all zeros (matches tokenizer contract)."""
    from imas_ambix.data.preprocess import _normalise_to_uint8
    from imas_ambix.tokenizer.frames import _normalise_frames_to_uint8

    flat = np.full((5, 10, 10), 12345, dtype=np.uint16)
    preprocess_result = _normalise_to_uint8(flat)
    tokenizer_result = _normalise_frames_to_uint8(flat)

    assert preprocess_result.dtype == np.uint8
    np.testing.assert_array_equal(preprocess_result, tokenizer_result)
    np.testing.assert_array_equal(preprocess_result, np.zeros_like(preprocess_result))


def test_normalise_to_uint8_known_values() -> None:
    """Check specific known values: lo=0 hi=65535 → maps proportionally."""
    from imas_ambix.data.preprocess import _normalise_to_uint8

    # 1 frame, 1×4 pixel row
    x = np.array([[[0, 65535, 32767, 1000]]], dtype=np.uint16)
    result = _normalise_to_uint8(x)
    assert result[0, 0, 0] == 0     # lo maps to 0
    assert result[0, 0, 1] == 255   # hi maps to 255
    # 32767 / 65535 * 255 ≈ 127.498… → clips to uint8 → 127
    assert result[0, 0, 2] == 127


# ---------------------------------------------------------------------------
# _resize_frames
# ---------------------------------------------------------------------------


def test_resize_frames_cv2_shape() -> None:
    """cv2 backend resizes (T, H, W, 3) → (T, S, S, 3)."""
    pytest.importorskip("cv2")
    from imas_ambix.data.preprocess import _resize_frames

    rng = np.random.default_rng(1)
    rgb = rng.integers(0, 256, (5, 32, 34, 3), dtype=np.uint8)
    out = _resize_frames(rgb, 16, backend="cv2")
    assert out.shape == (5, 16, 16, 3)
    assert out.dtype == np.uint8


def test_resize_frames_noop_when_already_correct_size() -> None:
    """No-op when h == w == image_size."""
    from imas_ambix.data.preprocess import _resize_frames

    rng = np.random.default_rng(2)
    rgb = rng.integers(0, 256, (3, 16, 16, 3), dtype=np.uint8)
    out = _resize_frames(rgb, 16, backend="cv2")
    np.testing.assert_array_equal(out, rgb)


def test_resize_frames_invalid_backend_raises() -> None:
    from imas_ambix.data.preprocess import _resize_frames

    rgb = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="resize_backend"):
        _resize_frames(rgb, 4, backend="unknown_backend")


# ---------------------------------------------------------------------------
# preprocess_rbb_shot — synthetic data
# ---------------------------------------------------------------------------


def test_preprocess_rbb_shot_happy_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """preprocess_rbb_shot writes (T, S, S, 3) uint8 and returns clean report."""
    pytest.importorskip("cv2")
    from imas_ambix.data.preprocess import preprocess_rbb_shot

    src_root = tmp_path / "level1" / "shots"
    dst_root = tmp_path / "preprocessed" / "rbb-256"
    make_frame_zarr(src_root, 9901, camera="rbb", t=5, h=32, w=34)

    report = preprocess_rbb_shot(
        9901, src_root, dst_root, image_size=16, resize_backend="cv2"
    )

    assert report.error is None, f"unexpected error: {report.error}"
    assert report.skipped is False
    assert report.n_frames == 5
    assert report.image_size == 16
    assert report.shot_id == 9901

    # Verify on-disk shape
    out_zarr = zarr.open_array(str(dst_root / "9901.zarr" / "data"), mode="r")
    assert out_zarr.shape == (5, 16, 16, 3)
    assert out_zarr.dtype == np.uint8


def test_preprocess_rbb_shot_skip_existing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Second call with skip_existing=True returns skipped report."""
    pytest.importorskip("cv2")
    from imas_ambix.data.preprocess import preprocess_rbb_shot

    src_root = tmp_path / "level1" / "shots"
    dst_root = tmp_path / "preprocessed" / "rbb-256"
    make_frame_zarr(src_root, 9902, camera="rbb", t=4, h=20, w=20)

    r1 = preprocess_rbb_shot(9902, src_root, dst_root, image_size=8)
    assert r1.error is None
    assert r1.n_frames == 4

    r2 = preprocess_rbb_shot(9902, src_root, dst_root, image_size=8, skip_existing=True)
    assert r2.skipped is True
    assert r2.n_frames == 0


def test_preprocess_rbb_shot_no_skip_overwrites(
    tmp_path: Path,
) -> None:
    """skip_existing=False re-processes even when output already exists."""
    pytest.importorskip("cv2")
    from imas_ambix.data.preprocess import preprocess_rbb_shot

    src_root = tmp_path / "level1" / "shots"
    dst_root = tmp_path / "preprocessed" / "rbb-256"
    make_frame_zarr(src_root, 9903, camera="rbb", t=3, h=16, w=16)

    r1 = preprocess_rbb_shot(
        9903, src_root, dst_root, image_size=8, skip_existing=False
    )
    assert r1.n_frames == 3

    r2 = preprocess_rbb_shot(
        9903, src_root, dst_root, image_size=8, skip_existing=False
    )
    assert r2.skipped is False
    assert r2.n_frames == 3


def test_preprocess_rbb_shot_missing_shot_populates_error(
    tmp_path: Path,
) -> None:
    """Missing shot Zarr → error captured, no crash."""
    pytest.importorskip("cv2")
    from imas_ambix.data.preprocess import preprocess_rbb_shot

    src_root = tmp_path / "level1" / "shots"
    src_root.mkdir(parents=True, exist_ok=True)
    dst_root = tmp_path / "preprocessed" / "rbb-256"

    report = preprocess_rbb_shot(99999, src_root, dst_root, image_size=16)
    assert report.error is not None
    assert report.n_frames == 0


def test_preprocess_rbb_shot_missing_camera_populates_error(
    tmp_path: Path,
) -> None:
    """Missing camera group → error captured, no crash."""
    pytest.importorskip("cv2")
    from imas_ambix.data.preprocess import preprocess_rbb_shot

    src_root = tmp_path / "level1" / "shots"
    dst_root = tmp_path / "preprocessed" / "rbb-256"
    make_frame_zarr(src_root, 9904, camera="rbb", t=3, h=16, w=16)

    report = preprocess_rbb_shot(9904, src_root, dst_root, camera="rir", image_size=8)
    assert report.error is not None
    assert report.n_frames == 0


# ---------------------------------------------------------------------------
# Bit-exact normalise+RGB step vs tokenizer (without resize)
# ---------------------------------------------------------------------------


def test_preprocess_normalise_rgb_step_matches_tokenizer(
    tmp_path: Path,
) -> None:
    """The normalise+RGB step of preprocess is bit-exact vs the tokenizer's path.

    We preprocess with image_size == input_size (no resize effect), then
    compare the result against the tokenizer's _normalise + np.repeat path.
    This is the key invariant: the preprocess store, when consumed by the
    encode loop, must produce identical token input as the legacy path.
    """
    pytest.importorskip("cv2")
    import xarray as xr

    from imas_ambix.data.preprocess import preprocess_rbb_shot
    from imas_ambix.tokenizer.frames import _normalise_frames_to_uint8

    src_root = tmp_path / "level1" / "shots"
    dst_root = tmp_path / "preprocessed"
    t, h, w = 6, 16, 16  # use square to avoid any resize effect
    make_frame_zarr(src_root, 9910, camera="rbb", t=t, h=h, w=w, seed=77)

    # Run preprocess with no-op resize (image_size == h == w == 16)
    report = preprocess_rbb_shot(
        9910, src_root, dst_root, image_size=16, resize_backend="cv2"
    )
    assert report.error is None

    # Read back the preprocessed data
    out_zarr = zarr.open_array(str(dst_root / "9910.zarr" / "data"), mode="r")
    preprocessed = out_zarr[:]  # (T, 16, 16, 3) uint8

    # Compute the legacy normalise+RGB from the raw frames
    shot_zarr = src_root / "9910.zarr"
    ds = xr.open_zarr(str(shot_zarr / "rbb"))
    raw = np.asarray(ds[list(ds.data_vars)[0]].values)  # (T, H, W)
    legacy_u8 = _normalise_frames_to_uint8(raw)          # (T, H, W) uint8
    legacy_rgb = np.repeat(legacy_u8[..., None], 3, axis=-1)  # (T, H, W, 3)

    np.testing.assert_array_equal(
        preprocessed,
        legacy_rgb,
        err_msg=(
            "Preprocessed frames diverge from legacy normalise+RGB at "
            "equal-size (no-resize) image_size"
        ),
    )


# ---------------------------------------------------------------------------
# bulk_preprocess
# ---------------------------------------------------------------------------


def test_bulk_preprocess_returns_one_report_per_shot(
    tmp_path: Path,
) -> None:
    """bulk_preprocess returns reports in input order, all successful."""
    pytest.importorskip("cv2")
    from imas_ambix.data.preprocess import bulk_preprocess

    src_root = tmp_path / "level1" / "shots"
    dst_root = tmp_path / "preprocessed" / "rbb-256"
    shot_ids = [9920, 9921, 9922]
    for sid in shot_ids:
        make_frame_zarr(src_root, sid, camera="rbb", t=3, h=16, w=16, seed=sid)

    reports = bulk_preprocess(
        shot_ids, src_root, dst_root, image_size=8, workers=1
    )

    assert len(reports) == 3
    for r, sid in zip(reports, shot_ids, strict=True):
        assert r.shot_id == sid
        assert r.error is None
        assert r.n_frames == 3


def test_bulk_preprocess_skip_existing(
    tmp_path: Path,
) -> None:
    """Second bulk run skips shots that already exist."""
    pytest.importorskip("cv2")
    from imas_ambix.data.preprocess import bulk_preprocess

    src_root = tmp_path / "level1" / "shots"
    dst_root = tmp_path / "preprocessed" / "rbb-256"
    shot_ids = [9930, 9931]
    for sid in shot_ids:
        make_frame_zarr(src_root, sid, camera="rbb", t=4, h=16, w=16, seed=sid)

    bulk_preprocess(shot_ids, src_root, dst_root, image_size=8, skip_existing=False)
    reports2 = bulk_preprocess(
        shot_ids, src_root, dst_root, image_size=8, skip_existing=True
    )

    assert all(r.skipped for r in reports2)


# ---------------------------------------------------------------------------
# CLI — preprocess-frames subcommand
# ---------------------------------------------------------------------------


def test_preprocess_frames_cli_help() -> None:
    """preprocess-frames --help exits 0 and documents expected flags."""
    from click.testing import CliRunner

    from imas_ambix.data.cli import data

    runner = CliRunner()
    result = runner.invoke(data, ["preprocess-frames", "--help"])
    assert result.exit_code == 0, result.output
    assert "--camera" in result.output
    assert "--image-size" in result.output
    assert "--workers" in result.output
    assert "--skip-existing" in result.output
    assert "--output-dir" in result.output
    assert "--resize-backend" in result.output


def test_preprocess_frames_cli_shot_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """preprocess-frames --shot-ids processes specified shots."""
    pytest.importorskip("cv2")
    from click.testing import CliRunner

    import imas_ambix.data.paths as paths_mod
    from imas_ambix.data.cli import data

    src_root = tmp_path / "level1" / "shots"
    dst_root = tmp_path / "preprocessed" / "rbb-256"
    for sid in [9940, 9941]:
        make_frame_zarr(src_root, sid, camera="rbb", t=3, h=16, w=16, seed=sid)

    monkeypatch.setattr(paths_mod, "LEVEL1_DIR", src_root)

    runner = CliRunner()
    result = runner.invoke(
        data,
        [
            "preprocess-frames",
            "--shot-ids", "9940,9941",
            "--camera", "rbb",
            "--image-size", "8",
            "--output-dir", str(dst_root),
            "--workers", "1",
            "--no-skip-existing",
            "--resize-backend", "cv2",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (dst_root / "9940.zarr").exists()
    assert (dst_root / "9941.zarr").exists()


def test_preprocess_frames_cli_from_bucket_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--from-bucket-all enumerates all L1 shots in src_root dir."""
    pytest.importorskip("cv2")
    from click.testing import CliRunner

    import imas_ambix.data.paths as paths_mod
    from imas_ambix.data.cli import data

    src_root = tmp_path / "level1" / "shots"
    dst_root = tmp_path / "preprocessed" / "rbb-256"
    for sid in [9950, 9951, 9952]:
        make_frame_zarr(src_root, sid, camera="rbb", t=2, h=8, w=8, seed=sid)

    monkeypatch.setattr(paths_mod, "LEVEL1_DIR", src_root)

    runner = CliRunner()
    result = runner.invoke(
        data,
        [
            "preprocess-frames",
            "--from-bucket-all",
            "--camera", "rbb",
            "--image-size", "4",
            "--output-dir", str(dst_root),
            "--workers", "1",
            "--no-skip-existing",
        ],
    )
    assert result.exit_code == 0, result.output
    # All 3 shots should be present
    for sid in [9950, 9951, 9952]:
        assert (dst_root / f"{sid}.zarr").exists(), f"{sid}.zarr missing"


def test_preprocess_frames_cli_no_shots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No shot IDs → exits cleanly with a message."""
    from click.testing import CliRunner

    import imas_ambix.data.paths as paths_mod
    from imas_ambix.data.cli import data

    src_root = tmp_path / "level1" / "shots"
    src_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(paths_mod, "LEVEL1_DIR", src_root)

    runner = CliRunner()
    result = runner.invoke(data, ["preprocess-frames"])
    assert result.exit_code == 0
    assert "no shot" in result.output.lower() or "0 shots" in result.output.lower()
