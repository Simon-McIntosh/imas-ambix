"""Tests for the sensor-space field/flux residual and its schema registration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from imas_ambix.spine_bench.runner import _magnetics_residual
from imas_ambix.spine_bench.schema import METRICS, SCHEMA_VERSION

METRIC = "magnetics_residual_whitened_rms"


# --- registration ------------------------------------------------------------


def test_the_metric_is_registered_with_a_unit_and_a_direction():
    metric = METRICS[METRIC]

    assert metric.unit == "normalised"
    assert metric.direction == "lower_better"
    assert "whitened" in metric.description


def test_the_schema_version_moved_with_the_registry():
    """A metric may not appear without a version bump: 1.3 stamps lack it."""
    assert SCHEMA_VERSION == "spine-bench/1.4"


# --- the measurement ---------------------------------------------------------


@dataclass
class _Fit:
    scored: bool
    jphi_flat: np.ndarray | None


@dataclass
class _Payload:
    measured: np.ndarray
    vacuum: np.ndarray
    scale: np.ndarray
    mask: np.ndarray


class _Grid:
    """The three things the residual reads off a grid, and nothing else."""

    def __init__(self, cells: np.ndarray, greens: np.ndarray, dr: float, dz: float):
        self.cells = cells
        self.dr = dr
        self.dz = dz
        self._greens = greens

    def sensor_greens(self, table):
        assert table == "geometry"
        return self._greens, [f"ch{i}" for i in range(self._greens.shape[0])]


def _case(*, mask, measured_offset=0.0):
    """A three-sensor, two-cell forward problem with a known answer."""
    greens = np.array([[2.0, 0.0], [0.0, 4.0], [1.0, 1.0]])
    jphi = np.array([0.0, 3.0, 5.0, 0.0])  # cells 1 and 2 carry current
    cells = np.array([1, 2])
    dr = dz = 0.5  # cell area 0.25 -> cell currents 0.75 and 1.25
    vacuum = np.array([10.0, 20.0, 30.0])
    exact = vacuum + greens @ (jphi[cells] * dr * dz)
    payload = _Payload(
        measured=exact - measured_offset,
        vacuum=vacuum,
        scale=np.array([0.5, 2.0, 1.0]),
        mask=np.asarray(mask, dtype=bool),
    )
    return _Fit(True, jphi), _Grid(cells, greens, dr, dz), payload


def test_a_perfect_forward_prediction_scores_zero():
    fit, grid, payload = _case(mask=[True, True, True])

    assert _magnetics_residual(fit, grid, "geometry", payload) == pytest.approx(0.0)


def test_the_residual_is_whitened_by_the_per_channel_scale():
    """A uniform offset of 1.0 becomes 1/scale per channel before the rms."""
    fit, grid, payload = _case(mask=[True, True, True], measured_offset=1.0)

    expected = np.sqrt(np.mean((1.0 / np.array([0.5, 2.0, 1.0])) ** 2))
    assert _magnetics_residual(fit, grid, "geometry", payload) == pytest.approx(
        expected
    )


def test_only_the_payload_mask_contributes():
    fit, grid, payload = _case(mask=[True, False, False], measured_offset=1.0)

    assert _magnetics_residual(fit, grid, "geometry", payload) == pytest.approx(2.0)


def test_an_unscored_fit_measures_nothing():
    fit, grid, payload = _case(mask=[True, True, True])
    fit.scored = False

    assert np.isnan(_magnetics_residual(fit, grid, "geometry", payload))


def test_a_fully_masked_slice_measures_nothing():
    fit, grid, payload = _case(mask=[False, False, False])

    assert np.isnan(_magnetics_residual(fit, grid, "geometry", payload))


def test_a_channel_with_no_measurement_is_left_out_rather_than_poisoning_the_rms():
    fit, grid, payload = _case(mask=[True, True, True], measured_offset=1.0)
    payload.measured[1] = np.nan

    expected = np.sqrt(np.mean((1.0 / np.array([0.5, 1.0])) ** 2))
    assert _magnetics_residual(fit, grid, "geometry", payload) == pytest.approx(
        expected
    )


# --- the registered gate -----------------------------------------------------


def _tolerances():
    from imas_ambix.spine_bench import parity

    return [t for t in parity.PARITY_TOLERANCES if t.metric == METRIC]


def test_both_gated_arms_carry_the_misfit():
    from imas_ambix.spine_bench import parity

    assert sorted(t.arm for t in _tolerances()) == sorted(parity.GATED_ARMS)


def test_the_margin_is_fractional_rather_than_a_change_budget():
    """A change budget on an absolute misfit would admit an unrelated equilibrium."""
    from imas_ambix.spine_bench import parity

    for tolerance in _tolerances():
        assert tolerance.lower_better
        assert tolerance.bound == pytest.approx(
            tolerance.reference * (1.0 + parity.MAGNETICS_RESIDUAL_REGRESSION_BUDGET)
        )
        assert tolerance.bound < tolerance.reference * parity.REPRODUCTION_CHANGE_BUDGET


def test_the_margin_clears_the_measured_cross_substrate_spread():
    """The two arms differ by 7.3e-4; the budget must sit well above that floor."""
    from imas_ambix.spine_bench import parity

    arms = {t.arm: t.reference for t in _tolerances()}
    spread = abs(arms["greens-matvec"] - arms["grid-delstar"]) / min(arms.values())

    assert spread < 1e-3
    assert 10.0 * spread < parity.MAGNETICS_RESIDUAL_REGRESSION_BUDGET


def test_the_banked_before_path_clears_every_registered_tolerance():
    from pathlib import Path

    import yaml

    from imas_ambix.spine_bench import parity
    from imas_ambix.spine_bench.schema import SpineBenchmarkStamp

    path = Path("imas_ambix/spine_bench/results") / parity.BEFORE_PATH_STAMP
    stamp = SpineBenchmarkStamp(**yaml.safe_load(path.read_text()))

    assert stamp.schema_version == SCHEMA_VERSION
    assert not stamp.env.git_dirty
    report = parity.evaluate(stamp)
    assert report.ok, report.describe()
    assert report.checked == len(parity.PARITY_TOLERANCES)


def test_comparing_paths_scores_the_misfit_too():
    """An after-path that drifts past the budget must fail, not pass quietly."""
    import copy
    from pathlib import Path

    import yaml

    from imas_ambix.spine_bench import parity
    from imas_ambix.spine_bench.schema import SpineBenchmarkStamp

    path = Path("imas_ambix/spine_bench/results") / parity.BEFORE_PATH_STAMP
    before = SpineBenchmarkStamp(**yaml.safe_load(path.read_text()))

    assert parity.compare_paths(before, before).ok

    after = copy.deepcopy(before)
    for arm in parity.GATED_ARMS:
        after.aggregate[arm][METRIC] *= 1.10
    report = parity.compare_paths(before, after)
    assert not report.ok
    assert {f.metric for f in report.failures} == {METRIC}
