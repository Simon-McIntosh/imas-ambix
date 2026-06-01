"""Smoke tests for the multi-modal statespace dataset (D3, S9).

Tests that rbb (camera), xsx (chord SXR), abm (bolometer), and ayc (Thomson)
load correctly for shots that have all four modalities present, with the
correct shapes, present-flags, and finite alignment to the model grid.

Shot inventory: shots ≥ 23000 with {rbb, xsx, abm, ayc} all present.
Confirmed candidates: 23142, 23143, 23144.
"""

from __future__ import annotations

import numpy as np
import pytest

from imas_ambix.data.paths import LEVEL1_DIR
from imas_ambix.statespace.dataset import (
    DatasetConfig,
    ModalitySpec,
    StatespaceDataset,
)

# ---------------------------------------------------------------------------
# Smoke-test shots: confirmed to carry rbb + xsx + abm + ayc
# ---------------------------------------------------------------------------
SMOKE_SHOTS = [23142, 23143, 23144]

# These shots must exist or the test is skipped (CI without /work data)
LEVEL1_AVAILABLE = (LEVEL1_DIR / f"{SMOKE_SHOTS[0]}.zarr").exists()


def _multimodal_config(model_hz: float = 100.0) -> DatasetConfig:
    """Build a multi-modal DatasetConfig for the four new modalities + amc."""
    return DatasetConfig(
        input_groups=["amc", "rbb", "xsx", "abm", "ayc"],
        target_group="",  # no target for these tests
        model_hz=model_hz,
        modality_spec={
            "amc": ModalitySpec(kind="signal_1d"),
            "rbb": ModalitySpec(kind="camera", frame_hw=(64, 64)),
            "xsx": ModalitySpec(kind="chord_2d", n_channels=54),
            "abm": ModalitySpec(kind="chord_2d", n_channels=32),
            "ayc": ModalitySpec(kind="thomson"),
        },
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_samples(shots: list[int], model_hz: float = 100.0) -> list[dict]:
    cfg = _multimodal_config(model_hz)
    ds = StatespaceDataset(shots, cfg, skip_missing_target=False)
    return list(ds)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not LEVEL1_AVAILABLE, reason="Level-1 data not available")
def test_multimodal_loads_correct_count():
    """All three smoke shots should yield a sample (no silent drops)."""
    samples = _load_samples(SMOKE_SHOTS)
    assert len(samples) == len(SMOKE_SHOTS), (
        f"Expected {len(SMOKE_SHOTS)} samples, got {len(samples)}"
    )


@pytest.mark.skipif(not LEVEL1_AVAILABLE, reason="Level-1 data not available")
def test_shot_grid_shape():
    """Each sample's shot_grid must be 1-D and non-empty."""
    samples = _load_samples(SMOKE_SHOTS)
    for s in samples:
        grid = s["shot_grid"]
        assert isinstance(grid, np.ndarray), f"shot {s['shot_id']}: grid not ndarray"
        assert grid.ndim == 1, f"shot {s['shot_id']}: grid.ndim={grid.ndim}"
        assert grid.size >= 2, f"shot {s['shot_id']}: grid too short ({grid.size})"
        assert np.all(np.diff(grid) > 0), f"shot {s['shot_id']}: grid not monotone"


@pytest.mark.skipif(not LEVEL1_AVAILABLE, reason="Level-1 data not available")
def test_all_present_flags_true():
    """For the confirmed shots, all four modalities + amc must be present."""
    samples = _load_samples(SMOKE_SHOTS)
    for s in samples:
        sid = s["shot_id"]
        pf = s["present_flags"]
        for grp in ["amc", "rbb", "xsx", "abm", "ayc"]:
            assert pf[grp], f"shot {sid}: group '{grp}' expected present but got False"


@pytest.mark.skipif(not LEVEL1_AVAILABLE, reason="Level-1 data not available")
def test_rbb_camera_shape():
    """rbb: (T_grid, 64, 64) float32 — downsampled from 544×640."""
    samples = _load_samples(SMOKE_SHOTS)
    for s in samples:
        sid = s["shot_id"]
        T = s["shot_grid"].size
        rbb = s["inputs"]["rbb"]
        assert isinstance(rbb, np.ndarray), f"shot {sid}: rbb not ndarray"
        assert rbb.dtype == np.float32, f"shot {sid}: rbb dtype={rbb.dtype}"
        assert rbb.shape == (T, 64, 64), (
            f"shot {sid}: rbb shape={rbb.shape}, expected ({T}, 64, 64)"
        )
        # Should have non-trivial values (frames are real plasma images)
        assert np.any(rbb > 0), f"shot {sid}: rbb appears all-zero"


@pytest.mark.skipif(not LEVEL1_AVAILABLE, reason="Level-1 data not available")
def test_xsx_chord_shape():
    """xsx: (T_grid, 54) float32 — 3 cameras × 18 chords."""
    samples = _load_samples(SMOKE_SHOTS)
    for s in samples:
        sid = s["shot_id"]
        T = s["shot_grid"].size
        xsx = s["inputs"]["xsx"]
        assert isinstance(xsx, np.ndarray), f"shot {sid}: xsx not ndarray"
        assert xsx.dtype == np.float32, f"shot {sid}: xsx dtype={xsx.dtype}"
        assert xsx.shape == (T, 54), (
            f"shot {sid}: xsx shape={xsx.shape}, expected ({T}, 54)"
        )


@pytest.mark.skipif(not LEVEL1_AVAILABLE, reason="Level-1 data not available")
def test_abm_chord_shape():
    """abm: (T_grid, 32) float32 — 32 bolometer channels."""
    samples = _load_samples(SMOKE_SHOTS)
    for s in samples:
        sid = s["shot_id"]
        T = s["shot_grid"].size
        abm = s["inputs"]["abm"]
        assert isinstance(abm, np.ndarray), f"shot {sid}: abm not ndarray"
        assert abm.dtype == np.float32, f"shot {sid}: abm dtype={abm.dtype}"
        assert abm.shape == (T, 32), (
            f"shot {sid}: abm shape={abm.shape}, expected ({T}, 32)"
        )


@pytest.mark.skipif(not LEVEL1_AVAILABLE, reason="Level-1 data not available")
def test_thomson_shape_and_freshness():
    """ayc: (T_grid, 14) float32; freshness column ([-1]) in [0,1]."""
    from imas_ambix.statespace.integrated_inputs import N_THOMSON_FEATURES

    samples = _load_samples(SMOKE_SHOTS)
    for s in samples:
        sid = s["shot_id"]
        T = s["shot_grid"].size
        ts = s["inputs"]["ayc"]
        assert isinstance(ts, np.ndarray), f"shot {sid}: ayc not ndarray"
        assert ts.dtype == np.float32, f"shot {sid}: ayc dtype={ts.dtype}"
        assert ts.shape == (T, N_THOMSON_FEATURES), (
            f"shot {sid}: ayc shape={ts.shape}, expected ({T}, {N_THOMSON_FEATURES})"
        )
        # Freshness column should be in [0, 1]
        fresh = ts[:, -1]
        assert np.all(fresh >= 0.0), f"shot {sid}: freshness has negative values"
        assert np.all(fresh <= 1.0), f"shot {sid}: freshness > 1.0"
        # Some fresh measurements should exist (ayc is present for these shots)
        assert np.any(fresh > 0.0), f"shot {sid}: no fresh Thomson measurements"


@pytest.mark.skipif(not LEVEL1_AVAILABLE, reason="Level-1 data not available")
def test_model_grid_alignment():
    """All modalities aligned to the shot grid must have the same T dimension."""
    samples = _load_samples(SMOKE_SHOTS)
    for s in samples:
        sid = s["shot_id"]
        T = s["shot_grid"].size
        for grp in ["rbb", "xsx", "abm", "ayc"]:
            arr = s["inputs"][grp]
            assert arr.shape[0] == T, (
                f"shot {sid}: '{grp}' time axis {arr.shape[0]} != grid {T}"
            )


@pytest.mark.skipif(not LEVEL1_AVAILABLE, reason="Level-1 data not available")
def test_grid_spacing_matches_model_hz():
    """The shot grid should have uniform spacing ~1/model_hz."""
    model_hz = 100.0
    samples = _load_samples(SMOKE_SHOTS, model_hz=model_hz)
    for s in samples:
        sid = s["shot_id"]
        grid = s["shot_grid"]
        dt = np.diff(grid)
        expected_dt = 1.0 / model_hz
        np.testing.assert_allclose(
            dt,
            expected_dt,
            rtol=1e-6,
            err_msg=f"shot {sid}: grid spacing not uniform at {model_hz} Hz",
        )


@pytest.mark.skipif(not LEVEL1_AVAILABLE, reason="Level-1 data not available")
def test_absent_group_returns_zero_and_false():
    """When a group is absent, load must return a zero array + present=False.

    We test this with a shot that has all four modalities and ask for an
    additional fictitious group 'xxx_absent' to verify the missingness policy.
    """

    cfg = DatasetConfig(
        input_groups=["amc", "rbb", "xsx", "abm", "ayc", "xxx_absent"],
        target_group="",
        model_hz=100.0,
        modality_spec={
            "amc": ModalitySpec(kind="signal_1d"),
            "rbb": ModalitySpec(kind="camera", frame_hw=(64, 64)),
            "xsx": ModalitySpec(kind="chord_2d", n_channels=54),
            "abm": ModalitySpec(kind="chord_2d", n_channels=32),
            "ayc": ModalitySpec(kind="thomson"),
            "xxx_absent": ModalitySpec(kind="chord_2d", n_channels=10),
        },
    )
    ds = StatespaceDataset([SMOKE_SHOTS[0]], cfg, skip_missing_target=False)
    samples = list(ds)
    assert samples, "No samples produced"
    s = samples[0]

    # Absent group must be False
    assert not s["present_flags"]["xxx_absent"], "xxx_absent should be absent"
    assert "xxx_absent" in s["missing_inputs"]

    # Absent array must be zeros of the right shape
    T = s["shot_grid"].size
    arr = s["inputs"]["xxx_absent"]
    assert arr.shape == (T, 10), f"absent group shape {arr.shape} != ({T}, 10)"
    assert np.all(arr == 0.0), "absent group should be zero-filled"


@pytest.mark.skipif(not LEVEL1_AVAILABLE, reason="Level-1 data not available")
def test_excluded_groups_raise():
    """DatasetConfig must raise if excluded groups (efm, esm, xim, …) are requested."""
    for bad_grp in ["efm", "esm", "xim", "ada", "aim"]:
        with pytest.raises(ValueError, match="excluded"):
            StatespaceDataset(
                [SMOKE_SHOTS[0]],
                DatasetConfig(input_groups=[bad_grp]),
            )


@pytest.mark.skipif(not LEVEL1_AVAILABLE, reason="Level-1 data not available")
def test_backward_compat_signal1d_only():
    """Original signal_1d-only usage must still work (backward compat)."""
    import xarray as xr

    cfg = DatasetConfig(
        input_groups=["amc", "ama"],
        target_group="",
        model_hz=100.0,
        # No modality_spec → defaults to signal_1d for all
    )
    ds = StatespaceDataset([SMOKE_SHOTS[0]], cfg, skip_missing_target=False)
    samples = list(ds)
    assert samples, "No samples for backward-compat test"
    s = samples[0]
    # amc should be an xr.Dataset (signal_1d path)
    assert isinstance(s["inputs"].get("amc"), xr.Dataset), (
        "signal_1d group should return xr.Dataset"
    )
    # present_flags key should exist
    assert "amc" in s["present_flags"]


# ---------------------------------------------------------------------------
# Main: run as smoke script
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not LEVEL1_AVAILABLE:
        print(f"SKIP: LEVEL1_DIR={LEVEL1_DIR} not accessible from this host.")
        raise SystemExit(0)

    print(f"Level-1 dir: {LEVEL1_DIR}")
    print(f"Smoke shots: {SMOKE_SHOTS}")
    cfg = _multimodal_config(model_hz=100.0)
    ds = StatespaceDataset(SMOKE_SHOTS, cfg, skip_missing_target=False)

    for s in ds:
        sid = s["shot_id"]
        grid = s["shot_grid"]
        T = grid.size
        pf = s["present_flags"]
        print(f"\n=== Shot {sid} ===")
        print(f"  model grid: T={T}, t=[{grid[0]:.4f}, {grid[-1]:.4f}] s")
        print(f"  present_flags: {pf}")
        for grp in ["amc", "rbb", "xsx", "abm", "ayc"]:
            inp = s["inputs"].get(grp)
            if inp is not None:
                shape = getattr(inp, "shape", None) or {
                    k: v.shape for k, v in inp.data_vars.items()
                }
                print(f"  {grp}: shape={shape}")
        print(f"  missing_inputs: {s['missing_inputs']}")
