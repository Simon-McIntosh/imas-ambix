"""A compacted row carries each gauge leaf's own count, and peak leaves' extremes.

A row-wide sample count cannot say that one gauge was observed in three of
twelve samples and another in all twelve, so a mean resting on three is
indistinguishable from one resting on twelve -- the absent-versus-zero
distinction, one level down. And a mean alone cannot say whether an hour at 40%
occupancy was flat or one spike, which is the difference between a healthy serve
and a capacity problem.

These tests hold the per-leaf count and the peak extremes across the tier chain:
an hour compacted straight from raw rows must carry the same counts and extremes
as an hour compacted from minute rows, so retaining a coarser tier does not lose
what the finer one measured.
"""

from __future__ import annotations

import datetime as _dt

import pytest

from imas_ambix.agent.telemetry_store import (
    TIER_HOUR,
    TIER_MINUTE,
    compact_rows,
)

_BASE = _dt.datetime(2026, 9, 20, 6, 0, 0, tzinfo=_dt.UTC)
_RUNNING = "num_requests_running"
_KV = "kv_cache_usage_perc"
_IDLE_GAUGE = "gpu_memory_used_bytes"


def _row(seconds: float, **leaves: float | None) -> dict:
    """One recorder sample carrying the named gauge leaves."""
    row: dict = {"timestamp": (_BASE + _dt.timedelta(seconds=seconds)).isoformat()}
    row.update(leaves)
    return row


def _sample_times(count: int, step: float = 5.0) -> list[float]:
    return [index * step for index in range(count)]


def _counts(node: dict) -> dict:
    return node["obs"]["leaves"]


def test_a_gauge_null_in_most_of_a_window_reports_only_its_observed_count():
    """Twelve samples with the gauge null in nine carry an observed count of 3.

    The row's own sample count stays twelve, because twelve samples were taken;
    the leaf's count is three, because three observed it. A mean built on the
    three must be legible as such rather than read as a window-wide average.
    """
    observed = {2: 4.0, 6: 8.0, 10: 12.0}
    rows = [
        _row(second, **{_RUNNING: observed.get(index)})
        for index, second in enumerate(_sample_times(12))
    ]

    compacted = compact_rows(rows, tier=TIER_MINUTE)

    assert len(compacted) == 1
    row = compacted[0]
    assert row["obs"]["samples"] == 12
    entry = _counts(row)[_RUNNING]
    assert entry["n"] == 3
    assert row[_RUNNING] == pytest.approx((4.0 + 8.0 + 12.0) / 3.0)


def test_a_non_peak_gauge_carries_a_count_but_no_extremes():
    """A count belongs to every gauge leaf; extremes to the peak-read ones only."""
    rows = [
        _row(second, **{_IDLE_GAUGE: float(index)})
        for index, second in enumerate(_sample_times(4))
    ]

    compacted = compact_rows(rows, tier=TIER_MINUTE)

    entry = _counts(compacted[0])[_IDLE_GAUGE]
    assert entry["n"] == 4
    assert set(entry) == {"n"}


def test_a_kv_spike_survives_as_the_windows_maximum_through_the_tier_chain():
    """One spike to 1.0 in a 0.4 minute is the maximum of the hour either way.

    An hour compacted straight from the raw rows and an hour compacted from the
    minute rows must both report the spike: the extremes fold by min and max, so
    the coarser tier does not average the capacity signal away.
    """
    value_at = {7: 1.0}
    rows = [
        _row(second, **{_KV: value_at.get(index, 0.4)})
        for index, second in enumerate(_sample_times(12))
    ]

    direct = compact_rows(rows, tier=TIER_HOUR)
    chained = compact_rows(compact_rows(rows, tier=TIER_MINUTE), tier=TIER_HOUR)

    assert len(direct) == 1
    assert len(chained) == 1
    for row in (direct[0], chained[0]):
        entry = _counts(row)[_KV]
        assert entry["n"] == 12
        assert entry["min"] == pytest.approx(0.4)
        assert entry["max"] == pytest.approx(1.0)
    assert direct[0][_KV] == chained[0][_KV] == pytest.approx(0.45)


def test_re_compaction_sums_leaf_counts_rather_than_taking_the_latest():
    """An hour's per-leaf count is the sum of its minutes', not the last one's.

    Two minutes of six observations each must give an hour count of twelve.
    Taking the latest count instead would report six -- the number of
    observations in one minute -- while the row claims an hour of them.
    """
    rows = [
        _row(second, **{_RUNNING: float(index)})
        for index, second in enumerate(_sample_times(12, step=10.0))
    ]

    minute_rows = compact_rows(rows, tier=TIER_MINUTE)
    minute_counts = [_counts(row)[_RUNNING]["n"] for row in minute_rows]
    assert minute_counts == [6, 6]

    direct = compact_rows(rows, tier=TIER_HOUR)
    chained = compact_rows(minute_rows, tier=TIER_HOUR)

    assert _counts(direct[0])[_RUNNING]["n"] == 12
    assert _counts(chained[0])[_RUNNING]["n"] == 12
