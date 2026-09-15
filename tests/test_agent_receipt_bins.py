"""Tests for width-binned interval receipt summaries."""

from __future__ import annotations

import json

import pytest

from imas_ambix.agent.receipt_bins import collect_receipt_bins


def _interval(
    *,
    running: int,
    generation: float | None,
    prefill: float | None,
    prefix_hit_rate: float | None,
) -> dict[str, int | float | None]:
    return {
        "num_requests_running": running,
        "generation_throughput_toks_per_s": generation,
        "prompt_throughput_toks_per_s": prefill,
        "prefix_cache_hit_rate_interval": prefix_hit_rate,
    }


def test_collector_bins_intervals_and_keeps_prefix_hits_as_distribution(
    tmp_path,
) -> None:
    receipts = tmp_path / "receipts.jsonl"
    rows = [
        _interval(running=12, generation=None, prefill=None, prefix_hit_rate=None),
        _interval(running=10, generation=120.0, prefill=500.0, prefix_hit_rate=0.2),
        _interval(running=14, generation=280.0, prefill=700.0, prefix_hit_rate=0.8),
        _interval(running=15, generation=150.0, prefill=900.0, prefix_hit_rate=0.9),
    ]
    receipts.write_text(
        "".join(f"{json.dumps(row)}\n" for row in rows), encoding="utf-8"
    )

    report = collect_receipt_bins(receipts).to_dict()

    assert report["rows_read"] == 4
    assert report["excluded_intervals"] == 1
    assert report["unbinned_intervals"] == 0
    first_bin = report["bins"]["10-14"]
    assert first_bin["intervals"] == 2
    assert first_bin["decode_toks_per_s_per_worker_median"] == pytest.approx(16.0)
    assert first_bin["prefill_toks_per_s_median"] == pytest.approx(600.0)
    assert first_bin["prefix_hit_rate"]["median"] == pytest.approx(0.5)
    assert first_bin["prefix_hit_rate"]["deciles"] == {
        "p10": pytest.approx(0.26),
        "p20": pytest.approx(0.32),
        "p30": pytest.approx(0.38),
        "p40": pytest.approx(0.44),
        "p50": pytest.approx(0.5),
        "p60": pytest.approx(0.56),
        "p70": pytest.approx(0.62),
        "p80": pytest.approx(0.68),
        "p90": pytest.approx(0.74),
    }


def test_collector_reports_single_interval_distribution_without_meaningless_error(
    tmp_path,
) -> None:
    receipts = tmp_path / "receipts.jsonl"
    receipts.write_text(
        json.dumps(
            _interval(running=19, generation=190.0, prefill=950.0, prefix_hit_rate=0.9)
        )
        + "\n",
        encoding="utf-8",
    )

    report = collect_receipt_bins(receipts).to_dict()

    comparison_bin = report["bins"]["15-19"]
    assert comparison_bin["intervals"] == 1
    assert comparison_bin["decode_toks_per_s_per_worker_median"] == pytest.approx(10.0)
    assert comparison_bin["prefill_toks_per_s_median"] == pytest.approx(950.0)
    assert comparison_bin["prefix_hit_rate"] == {
        "median": pytest.approx(0.9),
        "deciles": {f"p{decile * 10}": pytest.approx(0.9) for decile in range(1, 10)},
    }
