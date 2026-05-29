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


# ---------------------------------------------------------------------------
# Tests for RegimeBox
# ---------------------------------------------------------------------------


class TestRegimeBox:
    def test_contains_inside(self) -> None:
        box = RegimeBox(ip_min=150, ip_max=300, ne_min=10.0, ne_max=50.0)
        # ip=200 kA, ne=15 (×10¹⁹ m⁻³) — inside
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
        # Real MAST shot: ip=200 kA, ne=7e19 m^-3 → ne_scaled=7.0
        box = RegimeBox(ip_min=150, ip_max=300, ne_min=5.0, ne_max=15.0)
        # Pass ip in kA and ne already divided by 1e19
        ne_raw = 7e19
        ne_scaled = ne_raw / 1e19  # = 7.0
        assert box.contains(200.0, ne_scaled) is True


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
