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
) -> dict[str, object]:
    return {
        "num_requests_running": running,
        "generation_throughput_toks_per_s": generation,
        "prompt_throughput_toks_per_s": prefill,
        "engine": {"prefix_cache_hit_rate": prefix_hit_rate},
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


def test_collector_takes_prefix_hit_rate_from_engine_counters(tmp_path) -> None:
    """A family stating only cumulative hits and queries rates their advance."""
    receipts = tmp_path / "receipts.jsonl"
    rows = [
        {
            "num_requests_running": 10,
            "generation_throughput_toks_per_s": 100.0,
            "prompt_throughput_toks_per_s": 500.0,
            "engine": {"prefix_cache_queries": 1_000.0, "prefix_cache_hits": 400.0},
        },
        {
            "num_requests_running": 12,
            "generation_throughput_toks_per_s": 120.0,
            "prompt_throughput_toks_per_s": 600.0,
            "engine": {"prefix_cache_queries": 1_200.0, "prefix_cache_hits": 500.0},
        },
    ]
    receipts.write_text(
        "".join(f"{json.dumps(row)}\n" for row in rows), encoding="utf-8"
    )

    report = collect_receipt_bins(receipts).to_dict()

    assert report["rows_read"] == 2
    # The opening row has no earlier reading to difference its counters against.
    assert report["excluded_intervals"] == 1
    first_bin = report["bins"]["10-14"]
    assert first_bin["intervals"] == 1
    assert first_bin["prefix_hit_rate"]["median"] == pytest.approx(0.5)


def test_collector_declines_a_rate_spanning_two_serves(tmp_path) -> None:
    """Counters from two serves differenced would state a rate neither served.

    Both rows carry a valid interval and the counters advance, so nothing but
    the serve identity distinguishes this from the pair above.
    """
    receipts = tmp_path / "receipts.jsonl"

    def _write(job_ids: tuple[str, str]) -> dict[str, object]:
        rows = [
            {
                "num_requests_running": 10,
                "generation_throughput_toks_per_s": 100.0,
                "prompt_throughput_toks_per_s": 500.0,
                "hostname": "98dci4-gpu-0003",
                "job_id": job_ids[0],
                "engine": {
                    "prefix_cache_queries": 500_000.0,
                    "prefix_cache_hits": 400_000.0,
                },
            },
            {
                "num_requests_running": 12,
                "generation_throughput_toks_per_s": 120.0,
                "prompt_throughput_toks_per_s": 600.0,
                "hostname": "98dci4-gpu-0003",
                "job_id": job_ids[1],
                "engine": {
                    "prefix_cache_queries": 1_000_000.0,
                    "prefix_cache_hits": 900_000.0,
                },
            },
        ]
        receipts.write_text(
            "".join(f"{json.dumps(row)}\n" for row in rows), encoding="utf-8"
        )
        return collect_receipt_bins(receipts).to_dict()

    # Control: one serve across both rows, so the advance is its own and bins.
    same_serve = _write(("1001", "1001"))
    assert same_serve["excluded_intervals"] == 1
    assert same_serve["bins"]["10-14"]["intervals"] == 1
    assert same_serve["bins"]["10-14"]["prefix_hit_rate"]["median"] == pytest.approx(
        1.0
    )

    # Two serves: the same counters and the same positive advance, declined.
    across_serves = _write(("1001", "1002"))
    assert across_serves["excluded_intervals"] == 2
    assert across_serves["bins"]["10-14"]["intervals"] == 0
    assert across_serves["bins"]["10-14"]["prefix_hit_rate"] is None


def test_collector_declines_a_rate_whose_counters_fell(tmp_path) -> None:
    """A serve that restarted inside the interval resets both counters.

    The advance is non-positive rather than merely detached, and the row states
    nothing else missing, so only the refusal keeps it out of the bin.
    """
    receipts = tmp_path / "receipts.jsonl"
    rows = [
        {
            "num_requests_running": 10,
            "generation_throughput_toks_per_s": 100.0,
            "prompt_throughput_toks_per_s": 500.0,
            "hostname": "98dci4-gpu-0003",
            "job_id": "1001",
            "engine": {"prefix_cache_queries": 5_000.0, "prefix_cache_hits": 4_000.0},
        },
        {
            "num_requests_running": 12,
            "generation_throughput_toks_per_s": 120.0,
            "prompt_throughput_toks_per_s": 600.0,
            "hostname": "98dci4-gpu-0003",
            "job_id": "1001",
            "engine": {"prefix_cache_queries": 300.0, "prefix_cache_hits": 250.0},
        },
    ]
    receipts.write_text(
        "".join(f"{json.dumps(row)}\n" for row in rows), encoding="utf-8"
    )

    report = collect_receipt_bins(receipts).to_dict()

    assert report["rows_read"] == 2
    assert report["excluded_intervals"] == 2
    assert report["bins"]["10-14"]["intervals"] == 0
    assert report["bins"]["10-14"]["prefix_hit_rate"] is None
