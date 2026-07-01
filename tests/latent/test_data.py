"""Offline tests for the real-data assembly helpers (no /work mirror needed).

The mirror-touching loaders (:func:`load_shot_windows`) run on the compute node;
here we pin the pure glue — the by-name sensor alignment and the corpus-level
(absolute) calibration stats.
"""

from __future__ import annotations

import numpy as np

from imas_ambix.latent.data import align_sensor_columns, fit_corpus_stats


def test_align_sensor_columns_matches_by_name():
    sensor_channels = ["obv01", "obr01", "fl_p4u_1", "obv99_missing"]
    amb_names = ["fl_p4u_1", "obr01", "obv01", "somethingelse"]
    op_rows, x_cols = align_sensor_columns(sensor_channels, amb_names)
    # obv99_missing is not in amb_names → excluded; the rest matched by name
    assert list(op_rows) == [0, 1, 2]
    # x_cols point at the amb-name index (added to the amb offset by the caller)
    assert list(x_cols) == [
        amb_names.index("obv01"),
        amb_names.index("obr01"),
        amb_names.index("fl_p4u_1"),
    ]


def test_align_sensor_columns_empty_when_no_overlap():
    op_rows, x_cols = align_sensor_columns(["a", "b"], ["c", "d"])
    assert op_rows.size == 0 and x_cols.size == 0


def test_corpus_stats_are_corpus_level_not_per_shot():
    # two shots with different offsets — corpus stats span BOTH (absolute scale)
    shot_a = np.ones((10, 3)) * 2.0
    shot_b = np.ones((10, 3)) * 4.0
    stats = fit_corpus_stats([shot_a, shot_b])
    np.testing.assert_allclose(stats.mean, [3.0, 3.0, 3.0])
    assert (stats.std > 0).all()  # inter-shot spread preserved (not per-shot zeroed)
    norm = stats.normalise(shot_a)
    assert np.all(norm < 0)  # shot_a (below corpus mean) maps below zero


def test_corpus_stats_normalise_roundtrip_scale():
    x = np.random.RandomState(0).randn(50, 4) * 5 + 1
    stats = fit_corpus_stats([x])
    norm = stats.normalise(x)
    np.testing.assert_allclose(norm.mean(axis=0), 0.0, atol=1e-9)
    np.testing.assert_allclose(norm.std(axis=0), 1.0, atol=1e-6)
