"""Tests for the T10 integrated-input Thomson pressure loader."""

from __future__ import annotations

import numpy as np

from imas_ambix.statespace.integrated_inputs import (
    N_THOMSON_FEATURES,
    TS_R_GRID,
    ThomsonStream,
    integrated_feature_schema,
    load_thomson_stream,
    thomson_features_at_time,
)


def test_feature_vector_shape_and_freshness():
    pe = np.array([100.0, 300.0, 500.0, 200.0])
    r = np.array([0.4, 0.7, 1.0, 1.3])
    f = thomson_features_at_time(pe, r)
    assert f.shape == (N_THOMSON_FEATURES,)
    assert f[-1] == 1.0  # fresh measurement
    # profile block is finite, scalars finite
    assert np.isfinite(f).all()
    # peak scalar matches
    assert abs(f[len(TS_R_GRID)] - 500.0) < 1e-6


def test_empty_profile_is_zero_with_no_freshness():
    f = thomson_features_at_time(np.array([np.nan, np.nan]), np.array([0.4, 0.7]))
    assert f.shape == (N_THOMSON_FEATURES,)
    assert f[-1] == 0.0  # not fresh
    assert np.allclose(f, 0.0)


def test_gridding_invariance():
    # Same underlying profile sampled on two different radial grids → the fixed
    # vessel-grid profile block should be close (the feature is gridding-invariant).
    r1 = np.linspace(0.3, 1.4, 35)
    r2 = np.linspace(0.3, 1.4, 131)
    prof = lambda rr: 800.0 * np.exp(-((rr - 0.6) ** 2) / 0.1)  # noqa: E731
    f1 = thomson_features_at_time(prof(r1), r1)
    f2 = thomson_features_at_time(prof(r2), r2)
    np_ = len(TS_R_GRID)
    # interior nodes (avoid flat-extrapolation edges) agree to a few %
    rel = np.abs(f1[1 : np_ - 1] - f2[1 : np_ - 1]) / (np.abs(f2[1 : np_ - 1]) + 1.0)
    assert rel.max() < 0.05


class _FakeArr:
    def __init__(self, a):
        self._a = np.asarray(a)

    def __array__(self, dtype=None):
        return self._a if dtype is None else self._a.astype(dtype)

    @property
    def shape(self):
        return self._a.shape

    @property
    def ndim(self):
        return self._a.ndim


class _FakeGroup(dict):
    pass


class _FakeStore:
    def __init__(self, groups):
        self._g = groups

    def __contains__(self, k):
        return k in self._g

    def __getitem__(self, k):
        return self._g[k]


def test_forward_fill_is_causal():
    # Two Thomson measurements at t=0.10 and 0.30; grid spans 0..0.5.
    t_meas = np.array([0.10, 0.30])
    r = np.array([0.4, 0.7, 1.0])
    pe = np.array([[100.0, 200.0, 50.0], [300.0, 400.0, 100.0]])
    g = _FakeGroup()
    g["pe"] = _FakeArr(pe)
    g["time"] = _FakeArr(t_meas)
    g["radius"] = _FakeArr(r)
    store = _FakeStore({"atm": g})
    grid = np.linspace(0.0, 0.5, 51)  # 0,0.01,...,0.5
    ts = load_thomson_stream(store, grid)
    assert isinstance(ts, ThomsonStream)
    assert ts.system == "atm"
    assert ts.n_measurements == 2
    # before the first measurement: zero features, freshness 0 (NO future leak)
    before = grid < 0.10
    assert np.allclose(ts.features[before], 0.0)
    # between t=0.10 and 0.30 the held profile is the FIRST measurement's peak
    mid = (grid >= 0.10) & (grid < 0.30)
    peak_idx = len(TS_R_GRID)
    assert np.allclose(ts.features[mid, peak_idx], 200.0)
    # after t=0.30 the held profile is the SECOND measurement's peak
    after = grid >= 0.30
    assert np.allclose(ts.features[after, peak_idx], 400.0)
    # freshness is 1.0 exactly at a measurement grid-point and decays after
    i010 = int(np.argmin(np.abs(grid - 0.10)))
    assert ts.features[i010, -1] > 0.99


def test_integrated_schema_appends_thomson_last():
    schema = integrated_feature_schema()
    keys = list(schema.keys())
    assert keys[-1] == "thomson_pe"
    assert keys[:4] == ["ama", "amb", "amc", "ane"]
    assert len(schema["thomson_pe"]) == N_THOMSON_FEATURES
