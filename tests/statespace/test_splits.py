"""Unit tests for imas_ambix.statespace.splits.

All tests use synthetic data — no GPFS access.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from imas_ambix.statespace.splits import (
    RegimeBox,
    ShotSplits,
    _compute_regime_scalars_one,
    _plasma_on_window,
    build_splits,
    propose_ood_box,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_regime_scalars(
    n: int = 200,
    seed: int = 0,
    ip_range: tuple[float, float] = (50.0, 250.0),
    ne_range: tuple[float, float] = (1e19, 2e20),
) -> dict[int, dict[str, float]]:
    """Generate synthetic regime scalars for n shots."""
    rng = np.random.default_rng(seed)
    shot_ids = list(range(10000, 10000 + n))
    return {
        sid: {
            "ip_mean": float(rng.uniform(*ip_range)),
            "ne_mean": float(rng.uniform(*ne_range)),
        }
        for sid in shot_ids
    }


def _make_shot_with_plasma(
    shot_dir: Path,
    ip_flat_top: float = 600.0,
    ne_flat_top: float = 5e19,
    n_pre: int = 500,
    n_on: int = 300,
    n_post: int = 500,
    ne_spike: float | None = None,
) -> None:
    """Write a synthetic shot Zarr with amc/ane: zero pre-window, flat-top,
    zero post-window.

    The amc and ane share an identical time base for simplicity, so the
    plasma-on window selection is exercised end-to-end. A diluted middle-80%
    mean would NOT recover ``ip_flat_top`` because the off-plasma zeros pull
    it down; the plasma-on mask must.
    """
    import xarray as xr  # noqa: PLC0415

    n = n_pre + n_on + n_post
    dt = 2e-4  # 250 µs grid
    time = (np.arange(n) * dt - n_pre * dt).astype(np.float64)

    ip = np.zeros(n, dtype=np.float64)
    ip[n_pre : n_pre + n_on] = ip_flat_top

    ne = np.zeros(n, dtype=np.float64)
    ne[n_pre : n_pre + n_on] = ne_flat_top
    if ne_spike is not None:
        # Inject a single fringe-jump spike inside the plasma-on window
        ne[n_pre + n_on // 2] = ne_spike

    shot_dir.mkdir(parents=True, exist_ok=True)
    ds_amc = xr.Dataset(
        {"plasma_current": (("time",), ip)},
        coords={"time": time},
    )
    ds_amc.to_zarr(str(shot_dir), group="amc", mode="w")
    ds_ane = xr.Dataset(
        {"density": (("time",), ne)},
        coords={"time": time},
    )
    ds_ane.to_zarr(str(shot_dir), group="ane", mode="a")


# ---------------------------------------------------------------------------
# Tests for RegimeBox
# ---------------------------------------------------------------------------


class TestRegimeBox:
    def test_contains_inside(self) -> None:
        box = RegimeBox(ip_min=150, ip_max=300, ne_min=10.0, ne_max=50.0)
        # ip=200 kA, ne=15 (×10¹⁹ m⁻², line-integrated) — inside
        assert box.contains(200.0, 15.0) is True

    def test_contains_outside_ip(self) -> None:
        box = RegimeBox(ip_min=150, ip_max=300, ne_min=10.0, ne_max=50.0)
        assert box.contains(100.0, 15.0) is False  # ip too low
        assert box.contains(350.0, 15.0) is False  # ip too high

    def test_contains_outside_ne(self) -> None:
        box = RegimeBox(ip_min=150, ip_max=300, ne_min=10.0, ne_max=50.0)
        assert box.contains(200.0, 5.0) is False  # ne too low
        assert box.contains(200.0, 60.0) is False  # ne too high

    def test_contains_on_boundary(self) -> None:
        box = RegimeBox(ip_min=150, ip_max=300, ne_min=10.0, ne_max=50.0)
        # Boundary points: inclusive
        assert box.contains(150.0, 10.0) is True
        assert box.contains(300.0, 50.0) is True

    def test_to_dict_serialisable(self) -> None:
        box = RegimeBox(
            ip_min=100, ip_max=200, ne_min=5.0, ne_max=20.0, description="test"
        )
        d = box.to_dict()
        json.dumps(d)  # must not raise

    def test_units_consistent_with_scalars(self) -> None:
        """box.contains expects ip in kA and ne in 1e19 units (pre-scaled)."""
        # Real MAST shot: ip=200 kA, ne=7e19 m^-2 (line-integrated) → ne_scaled=7.0
        box = RegimeBox(ip_min=150, ip_max=300, ne_min=5.0, ne_max=15.0)
        # Pass ip in kA and ne already divided by 1e19
        ne_raw = 7e19
        ne_scaled = ne_raw / 1e19  # = 7.0
        assert box.contains(200.0, ne_scaled) is True


# ---------------------------------------------------------------------------
# Tests for plasma-on masking (physical regime scalars)
# ---------------------------------------------------------------------------


class TestPlasmaOnWindow:
    def test_recovers_contiguous_span(self) -> None:
        """The plasma-on window should be the first→last over-threshold index."""
        n_pre, n_on, n_post = 100, 50, 100
        ip = np.zeros(n_pre + n_on + n_post)
        ip[n_pre : n_pre + n_on] = 600.0
        time = np.arange(ip.size, dtype=float) * 1e-3
        result = _plasma_on_window(ip, time)
        assert result is not None
        t_start, t_end, mask = result
        assert mask.sum() == n_on
        assert t_start == time[n_pre]
        assert t_end == time[n_pre + n_on - 1]

    def test_no_plasma_returns_none(self) -> None:
        """A record entirely below the floor returns None (no plasma)."""
        ip = np.full(1000, 10.0)  # below the 50 kA floor
        time = np.arange(1000, dtype=float)
        assert _plasma_on_window(ip, time) is None

    def test_handles_negative_current(self) -> None:
        """Plasma-on detection uses |Iₚ| (MAST Iₚ can be signed)."""
        n = 300
        ip = np.zeros(n)
        ip[100:200] = -600.0  # negative-signed plasma current
        time = np.arange(n, dtype=float) * 1e-3
        result = _plasma_on_window(ip, time)
        assert result is not None
        _, _, mask = result
        assert mask.sum() == 100

    def test_fraction_threshold(self) -> None:
        """Threshold = 0.2 × peak, so a ramp keeps only the high portion."""
        # Ramp from 0 to 1000 kA — only |Iₚ| > 200 kA is plasma-on
        ip = np.linspace(0, 1000, 1000)
        time = np.arange(1000, dtype=float)
        result = _plasma_on_window(ip, time)
        assert result is not None
        _, _, mask = result
        # Indices where ip > 200 → roughly the top 80% of the ramp
        first_on = np.where(mask)[0][0]
        assert ip[first_on] >= 200.0 * 0.99  # threshold ≈ 200 kA


class TestComputeRegimeScalarsOne:
    def test_recovers_flat_top_not_diluted_mean(self, tmp_path: Path) -> None:
        """The plasma-on mask must recover ip_flat_top, not the diluted mean.

        With 500 zeros + 300 flat-top(600) + 500 zeros, the middle-80% mean
        is heavily diluted (~138 kA), but the plasma-on mean must be ~600 kA.
        """
        shot_dir = tmp_path / "10001.zarr"
        _make_shot_with_plasma(
            shot_dir,
            ip_flat_top=600.0,
            ne_flat_top=5e19,
            n_pre=500,
            n_on=300,
            n_post=500,
        )
        s = _compute_regime_scalars_one(shot_dir)
        assert "ip_mean" in s
        # Must recover the flat-top, NOT the diluted middle-80% mean
        assert abs(s["ip_mean"] - 600.0) < 1.0, (
            f"Expected ~600 kA flat-top, got diluted {s['ip_mean']:.1f}"
        )
        # Sanity: the naive middle-80% mean would be far lower
        diluted = 600.0 * 300 / (0.8 * 1300)
        assert s["ip_mean"] > diluted * 2

    def test_ne_median_rejects_spike(self, tmp_path: Path) -> None:
        """A density fringe-jump spike must be rejected by median + clip."""
        shot_dir = tmp_path / "10002.zarr"
        _make_shot_with_plasma(
            shot_dir,
            ip_flat_top=600.0,
            ne_flat_top=5e19,
            n_pre=300,
            n_on=200,
            n_post=300,
            ne_spike=400e19,  # non-physical spike (> 50e19 clip)
        )
        s = _compute_regime_scalars_one(shot_dir)
        assert "ne_mean" in s
        # Median over the flat-top + clip rejects the 400e19 spike → ~5e19
        assert abs(s["ne_mean"] - 5e19) < 0.5e19, (
            f"Expected ~5e19 (spike rejected), got {s['ne_mean']:.2e}"
        )

    def test_no_amc_returns_empty(self, tmp_path: Path) -> None:
        """A shot without amc returns an empty dict."""
        shot_dir = tmp_path / "10003.zarr"
        shot_dir.mkdir()
        assert _compute_regime_scalars_one(shot_dir) == {}

    def test_no_plasma_returns_empty(self, tmp_path: Path) -> None:
        """A shot whose Iₚ never exceeds the floor is omitted."""
        import xarray as xr  # noqa: PLC0415

        shot_dir = tmp_path / "10004.zarr"
        shot_dir.mkdir()
        n = 1000
        time = np.arange(n, dtype=float) * 1e-3
        xr.Dataset(
            {"plasma_current": (("time",), np.full(n, 10.0))},  # below floor
            coords={"time": time},
        ).to_zarr(str(shot_dir), group="amc", mode="w")
        assert _compute_regime_scalars_one(shot_dir) == {}

    def test_ne_uses_own_time_axis(self, tmp_path: Path) -> None:
        """ne is selected by ITS OWN time axis within the Iₚ plasma window.

        amc and ane have different lengths / time grids in real data; the
        window bounds come from amc time, ne selection uses ane time.
        """
        import xarray as xr  # noqa: PLC0415

        shot_dir = tmp_path / "10005.zarr"
        shot_dir.mkdir()
        # amc: 800 samples at 250 µs, plasma-on in [100,300)
        n_amc = 800
        amc_time = np.arange(n_amc) * 2e-4 - 100 * 2e-4
        ip = np.zeros(n_amc)
        ip[100:300] = 600.0
        xr.Dataset(
            {"plasma_current": (("time",), ip)},
            coords={"time": amc_time},
        ).to_zarr(str(shot_dir), group="amc", mode="w")
        # ane: DIFFERENT length/grid (1600 samples at 100 µs)
        n_ane = 1600
        ane_time = np.arange(n_ane) * 1e-4 - 100 * 1e-4
        ne = np.zeros(n_ane)
        # Set ne=5e19 only inside the amc plasma-on time window
        t_start = amc_time[100]
        t_end = amc_time[299]
        ne[(ane_time >= t_start) & (ane_time <= t_end)] = 5e19
        xr.Dataset(
            {"density": (("time",), ne)},
            coords={"time": ane_time},
        ).to_zarr(str(shot_dir), group="ane", mode="a")

        s = _compute_regime_scalars_one(shot_dir)
        assert "ne_mean" in s
        assert abs(s["ne_mean"] - 5e19) < 0.5e19


# ---------------------------------------------------------------------------
# Tests for propose_ood_box
# ---------------------------------------------------------------------------


class TestProposeOodBox:
    def test_box_fraction_target(self) -> None:
        """Proposed box should capture a non-trivial fraction of shots.

        For uniformly-distributed synthetic data the joint-exceedance fraction
        is (~1-pct)^2.  With pct=0.84 that gives ~(0.16)^2=2.6%.  We just check
        that the box is non-empty and that its fraction is within a broad range
        consistent with an upper-right-quadrant box.
        """
        scalars = _make_regime_scalars(n=500, seed=42)
        box, ood_shots = propose_ood_box(scalars, ood_fraction_target=0.10)
        fraction = len(ood_shots) / len(scalars)
        # Must be non-zero and not capture everything
        assert len(ood_shots) > 0, "OOD box should capture some shots"
        assert fraction < 0.50, f"OOD fraction {fraction:.3f} unexpectedly large"

    def test_box_upper_right_quadrant(self) -> None:
        """Proposed box should have thresholds above 50th percentile."""
        scalars = _make_regime_scalars(n=300)
        box, _ = propose_ood_box(scalars)
        ips = [v["ip_mean"] for v in scalars.values()]
        nes = [v["ne_mean"] / 1e19 for v in scalars.values()]
        # Both thresholds should be in the upper half of the distribution
        assert box.ip_min > np.percentile(ips, 50)
        assert box.ne_min > np.percentile(nes, 50)

    def test_returns_non_empty_ood_shots(self) -> None:
        """propose_ood_box should return at least some OOD shots."""
        scalars = _make_regime_scalars(n=500)
        box, ood_shots = propose_ood_box(scalars, ood_fraction_target=0.08)
        assert len(ood_shots) > 0

    def test_raises_on_empty_scalars(self) -> None:
        with pytest.raises(ValueError, match="No shots"):
            propose_ood_box({})

    def test_ood_shots_are_in_scalars(self) -> None:
        """All returned OOD shot IDs should be keys in the scalars dict."""
        scalars = _make_regime_scalars(n=300)
        box, ood_shots = propose_ood_box(scalars)
        for sid in ood_shots:
            assert sid in scalars


# ---------------------------------------------------------------------------
# Tests for build_splits
# ---------------------------------------------------------------------------


class TestBuildSplits:
    def test_splits_disjoint(self) -> None:
        """train, calibration, and test_ood_regime must be disjoint."""
        scalars = _make_regime_scalars(n=300)
        all_shots = list(scalars.keys())
        splits = build_splits(
            co_available_shots=all_shots,
            regime_scalars=scalars,
            held_out_family="dalpha",
            input_groups=["ama", "ane"],
            cal_fraction=0.12,
        )
        train_set = set(splits.train)
        cal_set = set(splits.calibration)
        ood_set = set(splits.test_ood_regime)
        assert train_set.isdisjoint(cal_set), "train and cal overlap"
        assert train_set.isdisjoint(ood_set), "train and OOD overlap"
        assert cal_set.isdisjoint(ood_set), "cal and OOD overlap"

    def test_splits_cover_all_shots(self) -> None:
        """Union of all splits should equal the co_available_shots."""
        scalars = _make_regime_scalars(n=200)
        shots = list(scalars.keys())
        splits = build_splits(
            co_available_shots=shots,
            regime_scalars=scalars,
            held_out_family="dalpha",
            input_groups=["ama"],
            cal_fraction=0.10,
        )
        combined = (
            set(splits.train) | set(splits.calibration) | set(splits.test_ood_regime)
        )
        assert combined == set(shots), f"Missing: {set(shots) - combined}"

    def test_cal_fraction_respected(self) -> None:
        """Calibration set size should be approximately cal_fraction of non-OOD."""
        scalars = _make_regime_scalars(n=500)
        shots = list(scalars.keys())
        splits = build_splits(
            co_available_shots=shots,
            regime_scalars=scalars,
            held_out_family="dalpha",
            input_groups=["ama", "ane"],
            cal_fraction=0.15,
        )
        non_ood = splits.n_train + splits.n_cal
        actual_fraction = splits.n_cal / max(non_ood, 1)
        assert abs(actual_fraction - 0.15) < 0.02

    def test_reproducible_with_seed(self) -> None:
        """Same seed should produce identical splits."""
        scalars = _make_regime_scalars(n=200)
        shots = list(scalars.keys())
        s1 = build_splits(shots, scalars, "dalpha", ["ama"], seed=0)
        s2 = build_splits(shots, scalars, "dalpha", ["ama"], seed=0)
        assert s1.train == s2.train
        assert s1.calibration == s2.calibration

    def test_different_seeds_differ(self) -> None:
        """Different seeds should (usually) produce different cal splits."""
        scalars = _make_regime_scalars(n=200)
        shots = list(scalars.keys())
        s1 = build_splits(shots, scalars, "dalpha", ["ama"], seed=0)
        s2 = build_splits(shots, scalars, "dalpha", ["ama"], seed=999)
        # Very likely to differ for large enough N
        assert set(s1.calibration) != set(s2.calibration)

    def test_magnetics_target_circularity_warning(self) -> None:
        """Magnetics as target should trigger circularity warning."""
        scalars = _make_regime_scalars(n=200)
        splits = build_splits(
            co_available_shots=list(scalars.keys()),
            regime_scalars=scalars,
            held_out_family="magnetics",
            input_groups=["ane"],
        )
        assert splits.circularity_warning != ""
        assert "CIRCULARITY" in splits.circularity_warning

    def test_no_circularity_for_dalpha(self) -> None:
        """Dalpha as target should NOT trigger circularity warning."""
        scalars = _make_regime_scalars(n=100)
        splits = build_splits(
            co_available_shots=list(scalars.keys()),
            regime_scalars=scalars,
            held_out_family="dalpha",
            input_groups=["ama", "ane"],
        )
        assert splits.circularity_warning == ""

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        scalars = _make_regime_scalars(n=100)
        splits = build_splits(
            co_available_shots=list(scalars.keys()),
            regime_scalars=scalars,
            held_out_family="dalpha",
            input_groups=["ama", "ane"],
        )
        path = tmp_path / "splits.json"
        splits.save(path)
        s2 = ShotSplits.load(path)
        assert s2.n_train == splits.n_train
        assert s2.n_cal == splits.n_cal
        assert s2.held_out_family == splits.held_out_family

    def test_input_groups_stored(self) -> None:
        """Input groups should be stored in the split artifact."""
        scalars = _make_regime_scalars(n=100)
        input_groups = ["ama", "amb", "ane"]
        splits = build_splits(
            co_available_shots=list(scalars.keys()),
            regime_scalars=scalars,
            held_out_family="dalpha",
            input_groups=input_groups,
        )
        assert sorted(splits.input_groups) == sorted(input_groups)

    def test_ood_box_contained_in_artifact(self) -> None:
        """The OOD box should be saved in the split artifact."""
        scalars = _make_regime_scalars(n=200)
        splits = build_splits(
            co_available_shots=list(scalars.keys()),
            regime_scalars=scalars,
            held_out_family="dalpha",
            input_groups=["ama"],
        )
        assert splits.ood_box is not None
        d = splits.to_dict()
        assert d["ood_box"] is not None
        assert "ip_min_kA" in d["ood_box"]
