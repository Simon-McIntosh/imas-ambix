"""Offline tests for the real-data assembly helpers (no /work mirror needed).

The mirror-touching loaders (:func:`load_shot_windows`) run on the compute node;
here we pin the pure glue — the by-name sensor alignment and the corpus-level
(absolute) calibration stats.
"""

from __future__ import annotations

import numpy as np

from imas_ambix.latent.data import (
    align_sensor_columns,
    fit_corpus_stats,
    robust_channel_scale,
)


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


def test_anchored_columns_resolved_by_name_not_index():
    """Ip/n_e columns must be resolved from the schema BY NAME (the amc list is
    alphabetically sorted, so hard-coded indices silently hit the wrong channel
    — the original defect read tf_current as Ip)."""
    from imas_ambix.latent.data import anchored_columns

    schema = {
        "ama": ["a1", "a2"],
        "amb": ["b1", "b2", "b3"],
        "amc": ["p4u_current", "plasma_current", "tf_current"],
        "ane": ["density"],
    }
    ip_col, ne_col = anchored_columns(schema)
    assert ip_col == 2 + 3 + 1  # ama + amb + index of plasma_current in amc
    assert ne_col == 2 + 3 + 3 + 0  # ... + index of density in ane


def test_anchored_columns_raise_on_missing_channel():
    """A schema without plasma_current must fail LOUD, not fall back."""
    import pytest

    from imas_ambix.latent.data import anchored_columns

    schema = {"amc": ["tf_current"], "ane": ["density"]}
    with pytest.raises(KeyError):
        anchored_columns(schema)


# --- robust_channel_scale: kind-relative whitening floor ----------------
#
# The concrete pathology this fixes: a flux loop (fl_cc04) whose per-shot
# natural variability happened to be near-zero (a real data characteristic,
# not corruption) turned an unremarkable vacuum/measured mismatch into a
# whitened misfit of ~3.7e6 once divided by that near-zero scale — dominating
# any training batch it landed in and driving the amortised-encoder's ip_pen
# to actively diverge on the full 10,846-shot corpus.


def test_robust_channel_scale_floors_the_fl_cc04_pathology():
    """A tiny-scale flux loop among many otherwise-healthy flux loops (as in
    the real fc938 signature, ~46 flux loops) gets floored; the resulting
    whitened residual is bounded relative to its kind's typical scale, not
    free to explode."""
    fl_others = [f"fl_p{i}l_1" for i in range(8)]
    channels = ["obr01", "obv01", "fl_cc04", *fl_others]
    # fl_cc04's OWN natural variability this shot: ~3e-5 (a real near-zero
    # reading), while its many sibling flux loops sit around 0.02 (kind-typical).
    scale = np.array([0.01, 0.012, 3e-5, *([0.020] * len(fl_others))])
    floored = robust_channel_scale(scale, channels)

    fl_idx = channels.index("fl_cc04")
    kind_typical = 0.020  # the many OTHER flux loops dominate the median
    assert floored[fl_idx] > scale[fl_idx] * 30  # meaningfully raised
    assert floored[fl_idx] <= 0.05 * kind_typical * 1.5  # stays near the floor

    # the SAME vacuum/measured mismatch (the observed fl_cc04 gap) is tamed by
    # orders of magnitude once whitened against the floored scale instead of
    # the pathological raw one
    mismatch = 0.35  # the observed vacuum-vs-measured gap on fl_cc04
    unfloored_resid_sq = (mismatch / scale[fl_idx]) ** 2  # the raw pathology
    floored_resid_sq = (mismatch / floored[fl_idx]) ** 2
    assert floored_resid_sq < unfloored_resid_sq / 1000  # orders of magnitude tamed


def test_robust_channel_scale_leaves_healthy_channels_unchanged():
    """A channel already well within its kind's typical range is untouched —
    the floor only activates for the implausibly-small tail."""
    channels = ["obr01", "obr02", "obv01", "fl_a", "fl_b", "fl_c"]
    scale = np.array([0.010, 0.011, 0.009, 0.020, 0.021, 0.019])
    floored = robust_channel_scale(scale, channels)
    np.testing.assert_allclose(floored, scale)


def test_robust_channel_scale_all_degenerate_kind_falls_back_to_one():
    """If an entire kind has no finite-positive scale, fall back to the
    pre-existing absolute 1.0 floor for that kind (no median to floor from)."""
    channels = ["fl_a", "fl_b", "obr01"]
    scale = np.array([np.nan, 0.0, 0.01])
    floored = robust_channel_scale(scale, channels)
    assert floored[0] == 1.0
    assert floored[1] == 1.0
    assert floored[2] == 0.01  # the healthy b-probe kind is untouched


def test_robust_channel_scale_vectorises_over_examples():
    """(N, S) input floors each row independently against its OWN row's kind
    medians — matching a cached corpus's per-example scale array where every
    row is a different shot."""
    channels = ["obr01", "fl_cc04", "fl_p6l_1"]
    scale = np.array(
        [
            [0.01, 3e-5, 0.02],  # row 0: fl_cc04 pathological this shot
            [0.01, 0.018, 0.02],  # row 1: healthy shot, no flooring needed
        ]
    )
    floored = robust_channel_scale(scale, channels)
    assert floored.shape == scale.shape
    assert floored[0, 1] > scale[0, 1] * 10  # row 0's fl_cc04 raised
    np.testing.assert_allclose(floored[1], scale[1])  # row 1 untouched
