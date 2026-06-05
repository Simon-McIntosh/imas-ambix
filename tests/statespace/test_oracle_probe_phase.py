from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from imas_ambix.statespace.oracle_probe import (
    ProbeConfig,
    _arms_for_config,
    _xma_phase_block,
    _xsx_phase_block,
)
from tests.statespace.test_fast_loader import _make_zarr_array, _make_zarr_group


def _make_xsx_phase_fixture(
    root: Path,
    *,
    shot_id: int,
    n_time: int = 30_000,
    phase_step: float = 0.0,
    freq_hz: float = 500.0,
) -> Path:
    shot_path = root / f"{shot_id}.zarr"
    _make_zarr_group(shot_path)
    xsx_path = shot_path / "xsx"
    _make_zarr_group(xsx_path)

    dt = 2e-6  # 500 kHz campaign-rate fixture
    time = np.arange(n_time, dtype=np.float64) * dt
    _make_zarr_array(xsx_path, "time", time.astype(np.float32))

    phase = np.arange(18, dtype=np.float64)[:, None] * phase_step
    carrier = 2.0 * np.pi * freq_hz * time[None, :]
    hcam_l = np.sin(carrier + phase).astype(np.float32)
    hcam_u = np.sin(carrier + phase + 0.5 * phase_step).astype(np.float32)
    _make_zarr_array(xsx_path, "hcam_l", hcam_l)
    _make_zarr_array(xsx_path, "hcam_u", hcam_u)
    _make_zarr_array(xsx_path, "hcam_l_r1", np.linspace(0.8, 1.3, 18, dtype=np.float32))
    _make_zarr_array(xsx_path, "hcam_u_r1", np.linspace(0.7, 1.2, 18, dtype=np.float32))
    return shot_path


def _make_xma_phase_fixture(
    root: Path,
    *,
    shot_id: int,
    n_samples: int = 400,
    mode_m: int = 1,
    freq_hz: float = 400.0,
) -> Path:
    shot_path = root / f"{shot_id}.zarr"
    _make_zarr_group(shot_path)
    xma_path = shot_path / "xma"
    _make_zarr_group(xma_path)

    total = n_samples * 22
    indices = np.arange(n_samples) * 22
    t_values = np.arange(n_samples, dtype=np.float64) * 2e-4  # 5 kHz compact axis
    time1 = np.full(total, np.nan, dtype=np.float32)
    time1[indices] = t_values.astype(np.float32)
    time_storage = np.arange(total, dtype=np.float64) / 110_000.0
    _make_zarr_array(xma_path, "time", time_storage.astype(np.float32))
    _make_zarr_array(xma_path, "time1", time1)

    theta = 2.0 * np.pi * np.arange(40, dtype=np.float64) / 40.0
    carrier = 2.0 * np.pi * freq_hz * t_values
    for i in range(1, 41):
        ch = np.full(total, np.nan, dtype=np.float32)
        waveform = np.sin(carrier + mode_m * theta[i - 1]).astype(np.float32)
        ch[indices] = waveform
        _make_zarr_array(xma_path, f"ccbv_{i:02d}", ch)

    for i in range(1, 10):
        ch = np.full(total, np.nan, dtype=np.float32)
        ch[indices] = np.cos(carrier).astype(np.float32)
        _make_zarr_array(xma_path, f"fl_cc{i:02d}", ch)

    for name in ("dia_loop", "dia_loopdot"):
        ch = np.full(total, np.nan, dtype=np.float32)
        ch[indices] = np.sin(carrier).astype(np.float32)
        _make_zarr_array(xma_path, name, ch)

    return shot_path


def test_arms_for_config_registers_phase_modalities() -> None:
    base = _arms_for_config(ProbeConfig())
    assert "E_mag_sxr_phase" not in base
    assert "E_mag_mirnov_phase" not in base

    phase = _arms_for_config(ProbeConfig(phase_features=True))
    assert phase["E_mag_sxr_phase"] == ["mag", "sxr_phase"]
    assert phase["E_mag_mirnov_phase"] == ["mag", "mirnov_phase"]
    assert phase["E_mag_sxr_mirnov_phase"] == ["mag", "sxr_phase", "mirnov_phase"]


def test_xsx_phase_block_is_phase_sensitive(tmp_path: Path) -> None:
    import zarr

    flat = _make_xsx_phase_fixture(tmp_path, shot_id=99101, phase_step=0.0)
    ramp = _make_xsx_phase_fixture(tmp_path, shot_id=99102, phase_step=0.20)
    slice_t = np.array([0.055], dtype=np.float64)

    flat_store = zarr.open_group(str(flat), mode="r")
    ramp_store = zarr.open_group(str(ramp), mode="r")
    flat_block, flat_ok = _xsx_phase_block(flat_store, slice_t)
    ramp_block, ramp_ok = _xsx_phase_block(ramp_store, slice_t)

    assert flat_ok is True
    assert ramp_ok is True
    assert flat_block.shape == (1, 110)
    assert ramp_block.shape == (1, 110)
    assert flat_block[0, 0] == pytest.approx(1.0, abs=0.1)
    assert abs(flat_block[0, 18]) < 0.1
    assert np.nanmean(np.abs(ramp_block[0, :36] - flat_block[0, :36])) > 0.05
    assert ramp_block[0, -2] == pytest.approx(500.0, rel=0.15)
    assert ramp_block[0, -1] == pytest.approx(1.0)


def test_xma_phase_block_recovers_m_mode_bias(tmp_path: Path) -> None:
    m1 = _make_xma_phase_fixture(tmp_path, shot_id=99201, mode_m=1)
    m2 = _make_xma_phase_fixture(tmp_path, shot_id=99202, mode_m=2)
    slice_t = np.array([0.079], dtype=np.float64)

    m1_block, m1_ok = _xma_phase_block(m1, slice_t)
    m2_block, m2_ok = _xma_phase_block(m2, slice_t)

    assert m1_ok is True
    assert m2_ok is True
    assert m1_block.shape == (1, 84)
    assert m2_block.shape == (1, 84)
    # Layout: 40 real, 40 imag, m1_amp, m2_amp, dom_freq, avail.
    assert m1_block[0, 80] > m1_block[0, 81]
    assert m2_block[0, 81] > m2_block[0, 80]
    assert m2_block[0, 82] == pytest.approx(400.0, rel=0.20)
    assert m2_block[0, 83] == pytest.approx(1.0)
