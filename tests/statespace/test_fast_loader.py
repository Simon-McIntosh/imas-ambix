"""Unit tests for imas_ambix.statespace.fast_loader (D0 — fast/exotic panel).

Synthetic fixtures only for CI; GPFS-guarded integration tests against real
shots 30460 (modern xma schema) and 20631 (legacy xma schema).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from imas_ambix.statespace.fast_loader import (
    ActShot,
    AoeShot,
    XmaShot,
    XsxShot,
    align_to_mse_window,
    mse_eval_window,
    probe_fast_panel,
    read_act_shot,
    read_aoe_shot,
    read_xma_shot,
    read_xsx_shot,
)

# ---------------------------------------------------------------------------
# Synthetic Zarr fixture helpers
# ---------------------------------------------------------------------------

LEVEL1_DIR = Path("/work/projects/imas_gpu/mast/level1/shots")
GPFS_AVAILABLE = LEVEL1_DIR.exists()

pytestmark_gpfs = pytest.mark.skipif(
    not GPFS_AVAILABLE, reason="GPFS level-1 mirror not accessible"
)


def _make_zarr_array(group_path: Path, name: str, data: np.ndarray) -> None:
    """Write a minimal Zarr v2 array (single chunk, no compression)."""
    import json

    arr_dir = group_path / name
    arr_dir.mkdir(parents=True, exist_ok=True)

    flat = data.flatten(order="C")
    raw_bytes = flat.tobytes()

    dtype = data.dtype
    zarr_dtype = str(dtype)
    if zarr_dtype == "float32":
        zarr_dtype = "<f4"
    elif zarr_dtype == "float64":
        zarr_dtype = "<f8"
    elif zarr_dtype == "int32":
        zarr_dtype = "<i4"

    meta = {
        "chunks": list(data.shape),
        "compressor": None,
        "dtype": zarr_dtype,
        "fill_value": "NaN" if "f" in zarr_dtype else 0,
        "filters": None,
        "order": "C",
        "shape": list(data.shape),
        "zarr_format": 2,
    }
    (arr_dir / ".zarray").write_text(json.dumps(meta))

    # Chunk key: N-d arrays use "0.0..." format
    chunk_key = ".".join(["0"] * len(data.shape))
    (arr_dir / chunk_key).write_bytes(raw_bytes)


def _make_zarr_group(group_path: Path) -> None:
    import json

    group_path.mkdir(parents=True, exist_ok=True)
    (group_path / ".zgroup").write_text(json.dumps({"zarr_format": 2}))


def _make_xma_modern_fixture(
    root: Path, shot_id: int = 99001, n_samples: int = 100
) -> Path:
    """Minimal modern xma fixture: 40 ccbv + 9 fl_cc + 2 dia at 5 kHz."""
    shot_path = root / f"{shot_id}.zarr"
    xma_path = shot_path / "xma"
    _make_zarr_group(xma_path)

    total = n_samples * 22  # 22× sparse: every 22nd sample is finite
    time_full = np.full(total, np.nan)
    time1 = np.full(total, np.nan)
    indices = np.arange(n_samples) * 22
    time_full[:] = np.nan
    t_values = np.arange(n_samples) * 0.0002  # 5 kHz
    time1[indices] = t_values
    # time (storage axis at 110 kHz)
    time_110 = np.linspace(-0.0, float(total - 1) / 110000, total)
    _make_zarr_array(xma_path, "time", time_110.astype(np.float32))
    _make_zarr_array(xma_path, "time1", time1.astype(np.float32))

    # ccbv channels
    rng = np.random.default_rng(42)
    for i in range(1, 41):
        ch = np.full(total, np.nan, dtype=np.float32)
        ch[indices] = rng.standard_normal(n_samples).astype(np.float32)
        _make_zarr_array(xma_path, f"ccbv_{i:02d}", ch)

    # fl_cc channels
    for i in range(1, 10):
        ch = np.full(total, np.nan, dtype=np.float32)
        ch[indices] = rng.standard_normal(n_samples).astype(np.float32)
        _make_zarr_array(xma_path, f"fl_cc{i:02d}", ch)

    # dia channels
    for name in ["dia_loop", "dia_loopdot"]:
        ch = np.full(total, np.nan, dtype=np.float32)
        ch[indices] = rng.standard_normal(n_samples).astype(np.float32)
        _make_zarr_array(xma_path, name, ch)

    return shot_path


def _make_xma_legacy_fixture(
    root: Path, shot_id: int = 99002, n_samples: int = 100
) -> Path:
    """Minimal legacy xma fixture: ccbv01..40 + sec time axis."""
    shot_path = root / f"{shot_id}.zarr"
    xma_path = shot_path / "xma"
    _make_zarr_group(xma_path)

    total = n_samples * 20
    indices = np.arange(n_samples) * 20
    sec = np.linspace(-0.5, 1.5, total, dtype=np.float32)
    _make_zarr_array(xma_path, "sec", sec)

    rng = np.random.default_rng(7)
    for i in range(1, 41):
        ch = np.full(total, np.nan, dtype=np.float32)
        ch[indices] = rng.standard_normal(n_samples).astype(np.float32)
        _make_zarr_array(xma_path, f"ccbv{i:02d}", ch)

    for i in range(1, 6):
        ch = np.full(total, np.nan, dtype=np.float32)
        ch[indices] = rng.standard_normal(n_samples).astype(np.float32)
        _make_zarr_array(xma_path, f"flcc{i:02d}", ch)

    return shot_path


def _make_xsx_fixture(root: Path, shot_id: int = 99003, n_time: int = 200) -> Path:
    shot_path = root / f"{shot_id}.zarr"
    xsx_path = shot_path / "xsx"
    _make_zarr_group(xsx_path)

    time = np.linspace(-0.01, 0.39, n_time, dtype=np.float32)
    _make_zarr_array(xsx_path, "time", time)
    rng = np.random.default_rng(3)
    hcam_l = rng.standard_normal((18, n_time)).astype(np.float32)
    hcam_u = rng.standard_normal((18, n_time)).astype(np.float32)
    _make_zarr_array(xsx_path, "hcam_l", hcam_l)
    _make_zarr_array(xsx_path, "hcam_u", hcam_u)
    r1_l = np.linspace(0.8, 1.3, 18, dtype=np.float32)
    r1_u = np.linspace(0.7, 1.2, 18, dtype=np.float32)
    _make_zarr_array(xsx_path, "hcam_l_r1", r1_l)
    _make_zarr_array(xsx_path, "hcam_u_r1", r1_u)
    return shot_path


def _make_aoe_fixture(
    root: Path, shot_id: int = 99004, n_total: int = 300, n_active: int = 100
) -> Path:
    shot_path = root / f"{shot_id}.zarr"
    aoe_path = shot_path / "aoe"
    _make_zarr_group(aoe_path)

    time = np.linspace(-0.5, 1.0, n_total, dtype=np.float32)
    _make_zarr_array(aoe_path, "time", time)
    indices = np.arange(n_active) + 100  # active window in the middle
    rng = np.random.default_rng(5)
    for bname in ["ka_band", "k_band", "u_band", "fast_k", "fast_ka"]:
        arr = np.full(n_total, np.nan, dtype=np.float32)
        arr[indices] = rng.standard_normal(n_active).astype(np.float32)
        _make_zarr_array(aoe_path, bname, arr)

    return shot_path


def _make_act_fixture(
    root: Path, shot_id: int = 99005, n_chords: int = 6, n_slices: int = 96
) -> Path:
    shot_path = root / f"{shot_id}.zarr"
    act_path = shot_path / "act"
    _make_zarr_group(act_path)

    mr = np.linspace(0.8, 1.3, n_slices, dtype=np.float32)
    _make_zarr_array(act_path, "majorradius", mr)

    rng = np.random.default_rng(11)
    # Beam on for slices [20, 80]; NaN otherwise
    beam_slices = slice(20, 80)
    for name in [
        "c_pla_temperature",
        "c_pla_temperature_error",
        "c_pla_velocity",
        "c_pla_velocity_error",
        "c_pla_cx_counts",
    ]:
        arr = np.full((n_chords, n_slices), np.nan, dtype=np.float32)
        arr[:, beam_slices] = rng.standard_normal((n_chords, 60)).astype(np.float32)
        _make_zarr_array(act_path, name, arr)

    return shot_path


# ---------------------------------------------------------------------------
# xma tests
# ---------------------------------------------------------------------------


class TestReadXmaShot:
    def test_modern_schema_basic(self, tmp_path: Path) -> None:
        shot = _make_xma_modern_fixture(tmp_path, shot_id=99001, n_samples=100)
        result = read_xma_shot(shot)
        assert result is not None
        assert isinstance(result, XmaShot)
        assert result.schema == "modern"
        assert result.shot_id == 99001
        assert result.n_slices == 100
        assert result.time.shape == (100,)
        assert result.time.dtype == np.float64
        assert np.all(np.isfinite(result.time))
        # 40 ccbv + 9 fl_cc + 2 dia = 51 channels
        assert result.n_channels == 51
        assert result.data.shape == (100, 51)
        assert result.avail_mask.shape == (51,)
        assert result.avail_mask.all()  # all channels have data

    def test_modern_rate_is_5khz(self, tmp_path: Path) -> None:
        shot = _make_xma_modern_fixture(tmp_path, shot_id=99001, n_samples=50)
        result = read_xma_shot(shot)
        assert result is not None
        assert abs(result.rate_hz - 5000.0) < 100  # within 100 Hz of 5 kHz

    def test_legacy_schema_basic(self, tmp_path: Path) -> None:
        shot = _make_xma_legacy_fixture(tmp_path, shot_id=99002, n_samples=80)
        result = read_xma_shot(shot)
        assert result is not None
        assert result.schema == "legacy"
        assert result.shot_id == 99002
        assert result.n_slices == 80
        # Should include ccbv01..40 = 40 channels (flcc only 5 in fixture)
        assert result.n_channels >= 40
        assert result.data.shape[0] == 80

    def test_missing_group_returns_none(self, tmp_path: Path) -> None:
        shot_path = tmp_path / "99999.zarr"
        shot_path.mkdir()
        assert read_xma_shot(shot_path) is None

    def test_time_is_strictly_increasing(self, tmp_path: Path) -> None:
        shot = _make_xma_modern_fixture(tmp_path, shot_id=99001, n_samples=50)
        result = read_xma_shot(shot)
        assert result is not None
        assert np.all(np.diff(result.time) > 0)


class TestXmaShot:
    def test_n_slices_n_channels_properties(self, tmp_path: Path) -> None:
        shot = _make_xma_modern_fixture(tmp_path, n_samples=30)
        result = read_xma_shot(shot)
        assert result is not None
        assert result.n_slices == result.time.shape[0]
        assert result.n_channels == len(result.channel_names)
        assert result.data.shape == (result.n_slices, result.n_channels)


# ---------------------------------------------------------------------------
# xsx tests
# ---------------------------------------------------------------------------


class TestReadXsxShot:
    def test_basic(self, tmp_path: Path) -> None:
        shot = _make_xsx_fixture(tmp_path, shot_id=99003, n_time=200)
        result = read_xsx_shot(shot)
        assert result is not None
        assert isinstance(result, XsxShot)
        assert result.shot_id == 99003
        assert result.time.shape == (200,)
        assert result.hcam_l.shape == (18, 200)
        assert result.hcam_u is not None
        assert result.hcam_u.shape == (18, 200)
        assert result.avail_mask[0]  # hcam_l present
        assert result.avail_mask[1]  # hcam_u present

    def test_r1_positions(self, tmp_path: Path) -> None:
        shot = _make_xsx_fixture(tmp_path, shot_id=99003)
        result = read_xsx_shot(shot)
        assert result is not None
        assert result.hcam_l_r1.shape == (18,)
        assert np.all(np.isfinite(result.hcam_l_r1))

    def test_rate_computed_from_time(self, tmp_path: Path) -> None:
        shot = _make_xsx_fixture(tmp_path, n_time=1000)
        result = read_xsx_shot(shot)
        assert result is not None
        assert result.rate_hz > 0

    def test_missing_group_returns_none(self, tmp_path: Path) -> None:
        shot_path = tmp_path / "99999.zarr"
        shot_path.mkdir()
        assert read_xsx_shot(shot_path) is None


# ---------------------------------------------------------------------------
# aoe tests
# ---------------------------------------------------------------------------


class TestReadAoeShot:
    def test_basic(self, tmp_path: Path) -> None:
        shot = _make_aoe_fixture(tmp_path, shot_id=99004, n_total=300, n_active=100)
        result = read_aoe_shot(shot)
        assert result is not None
        assert isinstance(result, AoeShot)
        assert result.shot_id == 99004
        assert result.n_slices == 100
        assert "ka_band" in result.bands
        assert result.bands["ka_band"].shape == (100,)
        assert result.avail_mask["ka_band"] is True

    def test_nan_outside_window_excluded(self, tmp_path: Path) -> None:
        shot = _make_aoe_fixture(tmp_path, n_total=500, n_active=50)
        result = read_aoe_shot(shot)
        assert result is not None
        assert result.n_slices == 50
        assert np.all(np.isfinite(result.bands["ka_band"]))  # finite window only

    def test_time_monotone(self, tmp_path: Path) -> None:
        shot = _make_aoe_fixture(tmp_path, n_total=300, n_active=100)
        result = read_aoe_shot(shot)
        assert result is not None
        assert np.all(np.diff(result.time) > 0)

    def test_missing_group_returns_none(self, tmp_path: Path) -> None:
        shot_path = tmp_path / "99999.zarr"
        shot_path.mkdir()
        assert read_aoe_shot(shot_path) is None


# ---------------------------------------------------------------------------
# act tests
# ---------------------------------------------------------------------------


class TestReadActShot:
    def test_basic(self, tmp_path: Path) -> None:
        shot = _make_act_fixture(tmp_path, shot_id=99005, n_chords=6, n_slices=96)
        result = read_act_shot(shot)
        assert result is not None
        assert isinstance(result, ActShot)
        assert result.shot_id == 99005
        assert result.n_chords == 6
        assert result.n_slices == 96
        assert result.temperature.shape == (6, 96)
        assert result.velocity.shape == (6, 96)
        assert result.beam_on_mask.shape == (96,)
        assert result.avail_mask["temperature"] is True

    def test_beam_on_mask_is_finite_temp(self, tmp_path: Path) -> None:
        """beam_on_mask should be True for slices [20..80)."""
        shot = _make_act_fixture(tmp_path, n_chords=6, n_slices=96)
        result = read_act_shot(shot)
        assert result is not None
        # Slices [20, 80) are beam-on in the fixture
        assert result.beam_on_mask[10] is np.bool_(False)  # before beam
        assert result.beam_on_mask[40] is np.bool_(True)  # during beam
        assert result.beam_on_mask[90] is np.bool_(False)  # after beam

    def test_major_radius_present(self, tmp_path: Path) -> None:
        shot = _make_act_fixture(tmp_path, n_slices=96)
        result = read_act_shot(shot)
        assert result is not None
        assert result.major_radius.shape == (96,)
        assert np.all(np.isfinite(result.major_radius))

    def test_missing_group_returns_none(self, tmp_path: Path) -> None:
        shot_path = tmp_path / "99999.zarr"
        shot_path.mkdir()
        assert read_act_shot(shot_path) is None


# ---------------------------------------------------------------------------
# probe_fast_panel tests
# ---------------------------------------------------------------------------


class TestProbeFastPanel:
    def test_all_present(self, tmp_path: Path) -> None:
        shot_path = tmp_path / "12345.zarr"
        shot_path.mkdir()
        for grp in ["xma", "xsx", "aoe", "act"]:
            (shot_path / grp).mkdir()
        # Mark xma as legacy
        (shot_path / "xma" / "ccbv01").mkdir()
        (shot_path / "xsx" / "hcam_l").mkdir()
        result = probe_fast_panel(shot_path)
        assert result.shot_id == 12345
        assert result.has_xma is True
        assert result.xma_schema == "legacy"
        assert result.has_xsx is True
        assert result.xsx_has_hcam is True
        assert result.has_aoe is True
        assert result.has_act is True

    def test_modern_xma_detected(self, tmp_path: Path) -> None:
        shot_path = tmp_path / "99999.zarr"
        (shot_path / "xma" / "time1").mkdir(parents=True)
        result = probe_fast_panel(shot_path)
        assert result.xma_schema == "modern"
        assert result.has_xma is True

    def test_all_missing(self, tmp_path: Path) -> None:
        shot_path = tmp_path / "00001.zarr"
        shot_path.mkdir()
        result = probe_fast_panel(shot_path)
        assert result.has_xma is False
        assert result.xma_schema == "missing"
        assert result.has_xsx is False
        assert result.has_aoe is False
        assert result.has_act is False


# ---------------------------------------------------------------------------
# Window alignment tests
# ---------------------------------------------------------------------------


class TestAlignToMseWindow:
    def test_restricts_to_window(self) -> None:
        time = np.linspace(0.0, 1.0, 100)
        data = np.ones((100, 5), dtype=np.float32)
        t_win, d_win = align_to_mse_window(time, data, t_min=0.3, t_max=0.7)
        assert t_win.min() >= 0.3
        assert t_win.max() <= 0.7
        assert t_win.shape[0] == d_win.shape[0]

    def test_empty_window(self) -> None:
        time = np.linspace(0.0, 0.5, 50)
        data = np.ones((50, 3))
        t_win, d_win = align_to_mse_window(time, data, t_min=0.6, t_max=1.0)
        assert t_win.shape[0] == 0
        assert d_win.shape[0] == 0

    def test_full_window(self) -> None:
        time = np.linspace(0.0, 1.0, 100)
        data = np.ones((100, 4))
        t_win, d_win = align_to_mse_window(time, data, t_min=0.0, t_max=1.0)
        assert t_win.shape[0] == 100


class TestMseEvalWindow:
    def test_extracts_min_max(self) -> None:
        entry = {"beam_on_slice_times": [0.1, 0.2, 0.3, 0.15, 0.25]}
        t_min, t_max = mse_eval_window(entry)
        assert t_min == pytest.approx(0.1)
        assert t_max == pytest.approx(0.3)

    def test_empty_times(self) -> None:
        entry = {"beam_on_slice_times": []}
        t_min, t_max = mse_eval_window(entry)
        assert t_min == float("-inf")
        assert t_max == float("inf")


# ---------------------------------------------------------------------------
# Integration tests against real MAST level-1 data (GPFS-guarded)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not GPFS_AVAILABLE, reason="GPFS not accessible")
class TestRealShotModernXma:
    """Shot 30460 — modern xma schema (ccbv_01, 5 kHz)."""

    SHOT_PATH = LEVEL1_DIR / "30460.zarr"

    def test_read_xma_modern(self) -> None:
        result = read_xma_shot(self.SHOT_PATH)
        assert result is not None
        assert result.schema == "modern"
        assert result.shot_id == 30460
        assert result.n_slices == 30000
        assert abs(result.rate_hz - 5000.0) < 200
        assert "ccbv_01" in result.channel_names
        assert "fl_cc01" in result.channel_names
        assert "dia_loop" in result.channel_names
        assert result.avail_mask.any()

    def test_fl_cc01_not_all_nan(self) -> None:
        """fl_cc01 clocks on time2 — verify it has finite data (not silently NaN)."""
        result = read_xma_shot(self.SHOT_PATH)
        assert result is not None
        fl01_idx = result.channel_names.index("fl_cc01")
        assert np.isfinite(result.data[:, fl01_idx]).sum() == result.n_slices, (
            "fl_cc01 should be 30000 finite (time2-clocked); all-NaN = time1-mask bug"
        )

    def test_read_xsx(self) -> None:
        result = read_xsx_shot(self.SHOT_PATH)
        assert result is not None
        assert result.shot_id == 30460
        assert result.hcam_l.shape[0] == 18  # 18 channels
        assert result.hcam_u is not None
        assert abs(result.rate_hz - 500_000.0) < 50_000

    def test_read_aoe(self) -> None:
        result = read_aoe_shot(self.SHOT_PATH)
        assert result is not None
        assert result.shot_id == 30460
        assert result.n_slices == 262144
        assert result.avail_mask.get("ka_band") is True
        assert result.avail_mask.get("k_band") is True
        assert np.all(np.isfinite(result.bands["ka_band"]))

    def test_read_act(self) -> None:
        result = read_act_shot(self.SHOT_PATH)
        assert result is not None
        assert result.shot_id == 30460
        assert result.n_chords == 6
        assert result.beam_on_mask.any()
        assert result.avail_mask["temperature"] is True


@pytest.mark.skipif(not GPFS_AVAILABLE, reason="GPFS not accessible")
class TestRealShotLegacyXma:
    """Shot 20631 — legacy xma schema (ccbv01, sec time axis)."""

    SHOT_PATH = LEVEL1_DIR / "20631.zarr"

    def test_read_xma_legacy(self) -> None:
        result = read_xma_shot(self.SHOT_PATH)
        assert result is not None
        assert result.schema == "legacy"
        assert result.shot_id == 20631
        assert result.n_slices == 7500
        assert "ccbv01" in result.channel_names
        assert result.data.shape[0] == 7500

    def test_read_xsx(self) -> None:
        result = read_xsx_shot(self.SHOT_PATH)
        assert result is not None
        assert result.hcam_l.shape[0] == 18
        # 100 kHz campaign era
        assert result.rate_hz < 200_000

    def test_read_act(self) -> None:
        result = read_act_shot(self.SHOT_PATH)
        assert result is not None
        assert result.beam_on_mask.any()


@pytest.mark.skipif(not GPFS_AVAILABLE, reason="GPFS not accessible")
class TestMseWindowAlignment:
    """Verify fast-signal windows span the MSE eval time range for held-out shots."""

    def test_xsx_spans_mse_window_on_30460(self) -> None:
        """xsx window covers ≥ 90 % of MSE slices for shot 30460.

        xsx acquires from ~ -0.01 s while the MSE window starts at ~ -0.047 s
        (pre-ramp slices before xsx trigger).  Full span is not guaranteed;
        ≥ 90 % overlap is the practical requirement for the study.
        """
        import json

        manifest_path = Path(
            "/work/projects/imas_gpu/mast/manifests/mse_heldout_split_v0.json"
        )
        if not manifest_path.exists():
            pytest.skip("MSE manifest not found")

        d = json.loads(manifest_path.read_text())
        shots = d.get("shots", {})
        if "30460" not in shots:
            pytest.skip("Shot 30460 not in MSE cohort")

        mse_times = np.array(shots["30460"].get("beam_on_slice_times", []))
        result = read_xsx_shot(LEVEL1_DIR / "30460.zarr")
        assert result is not None, "xsx read failed for shot 30460"
        t0, t1 = float(result.time[0]), float(result.time[-1])
        overlap_frac = float(((mse_times >= t0) & (mse_times <= t1)).mean())
        assert overlap_frac >= 0.90, (
            f"xsx window [{t0:.3f}, {t1:.3f}]s covers only "
            f"{overlap_frac:.0%} of MSE slices (need ≥ 90 %)"
        )

    def test_xma_spans_mse_window_on_30460(self) -> None:
        import json

        manifest_path = Path(
            "/work/projects/imas_gpu/mast/manifests/mse_heldout_split_v0.json"
        )
        if not manifest_path.exists():
            pytest.skip("MSE manifest not found")

        d = json.loads(manifest_path.read_text())
        shots = d.get("shots", {})
        if "30460" not in shots:
            pytest.skip("Shot 30460 not in MSE cohort")

        t_min, t_max = mse_eval_window(shots["30460"])
        result = read_xma_shot(LEVEL1_DIR / "30460.zarr")
        assert result is not None
        assert result.time[0] <= t_min
        assert result.time[-1] >= t_max
