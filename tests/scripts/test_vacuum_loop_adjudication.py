"""Tests for the JT-60SA vacuum adjudication rule and its quasi-static filter.

``scripts/vacuum_loop_adjudication.py`` fits every bound flux loop and
tangential probe against its vacuum prediction and assigns a per-channel
verdict.  The verdict is the median over the shot cohort, fitted only on
quasi-static samples.  These tests drive the rule, the filter and the probe
angle supply directly on synthetic inputs, so no reference store is opened
except for the probe-angle check, which reads the packaged description.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

_MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "vacuum_loop_adjudication.py"
)
_spec = importlib.util.spec_from_file_location(
    "vacuum_loop_adjudication_under_test", _MODULE_PATH
)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)

_median_verdict = _module._jt60sa_median_verdict
_quasi_static_mask = _module._jt60sa_quasi_static_mask


def _fit(slope: float, pearson_r: float) -> dict[str, float]:
    return {
        "slope": slope,
        "offset": 0.0,
        "pearson_r": pearson_r,
        "n_samples": 1000,
    }


def _every_shot_identity(
    fits: list[tuple[int, dict[str, float]]],
) -> bool:
    """The superseded rule: identity only if every shot clears the threshold."""

    return all(
        abs(fit["slope"] - 1.0) <= _module.JT60SA_SLOPE_TOLERANCE
        and abs(fit["pearson_r"]) > 0.98
        for _, fit in fits
    )


def test_probe_angles_supplied_for_all_seventeen() -> None:
    from imas_ambix.data.description_reader import read_geometry_table

    table = read_geometry_table(100595, machine="jt-60sa")
    angles = {
        mapping.amb_channel: mapping.angle_deg
        for mapping in table.sensor_map
        if mapping.amb_channel.startswith("MP")
    }
    probes = {f"MP{index}": angles.get(f"MP{index}") for index in range(1, 18)}
    assert all(value is not None for value in probes.values()), probes
    assert len(probes) == 17


def test_median_rule_keeps_noisy_shot_every_shot_rule_drops() -> None:
    fits = [
        (100579, _fit(1.02, 0.99)),
        (100595, _fit(1.25, 0.97)),
        (100642, _fit(1.01, 0.99)),
    ]
    assert _every_shot_identity(fits) is False
    verdict, reason, outliers = _median_verdict(fits)
    assert verdict == "identity", reason
    assert outliers == [100595]


def test_sign_disagreement_is_undecided() -> None:
    fits = [
        (100579, _fit(1.01, 0.99)),
        (100595, _fit(1.00, 0.99)),
        (100642, _fit(-1.00, 0.99)),
    ]
    verdict, reason, _outliers = _median_verdict(fits)
    assert verdict == "undecided"
    assert "sign" in reason
    assert 100642 in _outliers


def test_median_correlation_below_floor_is_undecided() -> None:
    fits = [
        (100579, _fit(1.0, 0.99)),
        (100595, _fit(1.0, 0.80)),
        (100642, _fit(1.0, 0.85)),
    ]
    verdict, reason, _outliers = _median_verdict(fits)
    assert verdict == "undecided"
    assert "median |r|" in reason


def test_quasi_static_mask_excludes_fast_sample() -> None:
    # Two coils, ten samples. The first coil's fastest slew is 100 A/s, the
    # second's is 10 A/s; the fraction is 0.1, so a sample qualifies only when
    # coil 0 is at or below 10 A/s and coil 1 at or below 1 A/s.
    slew = np.array(
        [
            [1.0, 0.2],  # slow on both coils -> kept
            [100.0, 0.2],  # coil 0 ramps fast -> excluded
            [1.0, 10.0],  # coil 1 ramps fast -> excluded
        ]
    )
    mask = _quasi_static_mask(slew, 0.1)
    assert mask.tolist() == [True, False, False]


def test_quasi_static_mask_honours_shot_own_peak() -> None:
    # The same absolute slew is quasi-static in a fast shot and not in a slow
    # one: the reference is each coil's own peak across the shot.
    fast = np.array([[50.0], [100.0]])
    slow = np.array([[0.5], [1.0]])
    assert _quasi_static_mask(fast, 0.6).tolist() == [True, False]
    assert _quasi_static_mask(slow, 0.6).tolist() == [True, False]


def test_missing_fit_is_undecided() -> None:
    fits = [
        (100579, _fit(1.0, 0.99)),
        (100595, None),
        (100642, _fit(1.0, 0.99)),
    ]
    verdict, reason, _outliers = _median_verdict(fits)
    assert verdict == "undecided"
    assert "100595" in reason
