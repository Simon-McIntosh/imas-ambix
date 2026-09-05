"""Tests for imas_ambix.agent.serving_receipts: continuous /metrics recording.

Coverage
--------
1.  ``_serving_snapshot`` reads gauges (running/waiting/KV usage), counters
    (prompt/generation tokens, prefix-cache queries and hits), and the
    speculative-decode sub-snapshot out of one recorded scrape.
2.  ``build_receipt_row`` on a lone snapshot: no throughput yet (needs a
    second sample), but a non-null prefix-cache hit rate and spec-decode
    fields are absent (no prior spec snapshot).
3.  ``build_receipt_row`` on two snapshots separated in time: generation and
    prompt throughput are the counter deltas over the elapsed wall time, and
    speculative-decode acceptance is the delta-derived rate.
4.  A zero-elapsed pair yields no throughput rather than dividing by zero.
5.  ``record_receipts`` samples on an interval, appends one JSON row per
    successful scrape to a durable file, and a failed scrape is skipped
    rather than recorded or raised.
6.  The written file is append-only JSONL: an existing row is never
    rewritten when the recorder is invoked again against the same path.

All HTTP is stubbed at ``urllib.request.urlopen`` — no network, no GPU.
"""

from __future__ import annotations

import datetime as _dt
import json
import urllib.error
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

from imas_ambix.agent import serving_receipts as sr

if TYPE_CHECKING:
    from collections.abc import Callable


# ---------------------------------------------------------------------------
# Fixtures — recorded shape of a real vLLM /metrics scrape, two samples 5s
# apart so counters visibly advance.
# ---------------------------------------------------------------------------


def _metrics_text(
    *,
    running: float,
    waiting: float,
    kv_usage: float,
    prompt_tokens: float,
    generation_tokens: float,
    prefix_queries: float,
    prefix_hits: float,
    draft_tokens: float,
    accepted_tokens: float,
    per_pos: tuple[float, ...] = (),
) -> bytes:
    lines = [
        "# HELP vllm:num_requests_running Number of running requests.",
        "# TYPE vllm:num_requests_running gauge",
        f'vllm:num_requests_running{{model_name="deepseek-v4-flash"}} {running}',
        "# HELP vllm:num_requests_waiting Number of waiting requests.",
        "# TYPE vllm:num_requests_waiting gauge",
        f'vllm:num_requests_waiting{{model_name="deepseek-v4-flash"}} {waiting}',
        "# HELP vllm:gpu_cache_usage_perc GPU KV-cache usage.",
        "# TYPE vllm:gpu_cache_usage_perc gauge",
        f'vllm:gpu_cache_usage_perc{{model_name="deepseek-v4-flash"}} {kv_usage}',
        "# HELP vllm:prompt_tokens_total Prefill tokens processed.",
        "# TYPE vllm:prompt_tokens_total counter",
        f'vllm:prompt_tokens_total{{model_name="deepseek-v4-flash"}} {prompt_tokens}',
        "# HELP vllm:generation_tokens_total Generation tokens processed.",
        "# TYPE vllm:generation_tokens_total counter",
        f'vllm:generation_tokens_total{{model_name="x"}} {generation_tokens}',
        "# HELP vllm:gpu_prefix_cache_queries_total Prefix cache queries.",
        "# TYPE vllm:gpu_prefix_cache_queries_total counter",
        f'vllm:gpu_prefix_cache_queries_total{{model_name="x"}} {prefix_queries}',
        "# HELP vllm:gpu_prefix_cache_hits_total Prefix cache hits.",
        "# TYPE vllm:gpu_prefix_cache_hits_total counter",
        f'vllm:gpu_prefix_cache_hits_total{{model_name="x"}} {prefix_hits}',
        "# HELP vllm:spec_decode_num_draft_tokens_total Draft tokens proposed.",
        "# TYPE vllm:spec_decode_num_draft_tokens_total counter",
        f'vllm:spec_decode_num_draft_tokens_total{{engine="0"}} {draft_tokens}',
        "# HELP vllm:spec_decode_num_accepted_tokens_total Tokens kept.",
        "# TYPE vllm:spec_decode_num_accepted_tokens_total counter",
        f'vllm:spec_decode_num_accepted_tokens_total{{engine="0"}} {accepted_tokens}',
        "# TYPE vllm:spec_decode_num_accepted_tokens_per_pos counter",
    ]
    lines += [
        f'vllm:spec_decode_num_accepted_tokens_per_pos{{engine="0",position="{i}"}} {v}'
        for i, v in enumerate(per_pos)
    ]
    lines.append("")
    return "\n".join(lines).encode()


SAMPLE_T0 = _metrics_text(
    running=3,
    waiting=0,
    kv_usage=0.12,
    prompt_tokens=128_301,
    generation_tokens=542_213,
    prefix_queries=984_123,
    prefix_hits=812_004,
    draft_tokens=40_000,
    accepted_tokens=24_000,
    per_pos=(9000.0, 6000.0, 4500.0, 2500.0, 2000.0),
)

SAMPLE_T1 = _metrics_text(
    running=2,
    waiting=1,
    kv_usage=0.15,
    prompt_tokens=128_301,
    generation_tokens=542_513,  # +300 generation tokens
    prefix_queries=986_123,  # +2000 queries
    prefix_hits=813_804,  # +1800 hits
    draft_tokens=40_500,  # +500 draft
    accepted_tokens=24_300,  # +300 accepted
    per_pos=(9090.0, 6060.0, 4545.0, 2525.0, 2020.0),
)

T0 = _dt.datetime(2026, 9, 5, 12, 0, 0, tzinfo=_dt.UTC)
T1 = T0 + _dt.timedelta(seconds=5)


class _FakeResponse:
    def __init__(self, body: bytes, status: int = 200) -> None:
        self._body = body
        self.status = status

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False


def _stub_urlopen(bodies: list[bytes | Exception]) -> Callable[..., _FakeResponse]:
    """Answer successive ``/metrics`` scrapes from *bodies*, in order."""
    pending = list(bodies)

    def _urlopen(req: Any, timeout: float | None = None, **_kwargs: Any) -> Any:
        outcome = pending.pop(0) if len(pending) > 1 else pending[0]
        if isinstance(outcome, Exception):
            raise outcome
        return _FakeResponse(outcome)

    return _urlopen


# ---------------------------------------------------------------------------
# Parse
# ---------------------------------------------------------------------------


def test_serving_snapshot_reads_gauges_counters_and_spec_decode() -> None:
    snapshot = sr._serving_snapshot(SAMPLE_T0.decode())

    assert snapshot["gauges"]["num_requests_running"] == 3.0
    assert snapshot["gauges"]["num_requests_waiting"] == 0.0
    assert snapshot["gauges"]["kv_cache_usage_perc"] == 0.12
    assert snapshot["counters"]["prompt_tokens_total"] == 128_301.0
    assert snapshot["counters"]["generation_tokens_total"] == 542_213.0
    assert snapshot["counters"]["prefix_cache_queries_total"] == 984_123.0
    assert snapshot["counters"]["prefix_cache_hits_total"] == 812_004.0
    assert snapshot["spec_decode"]["draft_tokens_total"] == 40_000.0
    assert snapshot["spec_decode"]["accepted_tokens_total"] == 24_000.0


def test_serving_snapshot_ignores_created_gauges_and_external_counters() -> None:
    """A recorded shape from the live two-card endpoint's own scrape.

    vLLM's OpenMetrics exposition pairs every counter with a same-named
    ``_created`` gauge (its creation timestamp, not a data point) and, where
    KV-connector cross-instance sharing exists, an ``external_`` variant of
    the prefix-cache counters. Both must be excluded rather than folded into
    the real counter, or a scrape carrying them silently reports a value that
    is not tokens.
    """
    text = "\n".join(
        [
            "# TYPE vllm:num_requests_running gauge",
            'vllm:num_requests_running{engine="0",model_name="x"} 0.0',
            "# TYPE vllm:kv_cache_usage_perc gauge",
            'vllm:kv_cache_usage_perc{engine="0",model_name="x"} 0.0',
            "# TYPE vllm:prompt_tokens_total counter",
            'vllm:prompt_tokens_total{engine="0",model_name="x"} 4.4481e+08',
            "# TYPE vllm:prompt_tokens_created gauge",
            'vllm:prompt_tokens_created{engine="0",model_name="x"} 1.7882e+09',
            "# TYPE vllm:prefix_cache_queries_total counter",
            'vllm:prefix_cache_queries_total{engine="0",model_name="x"} 4.4523e+08',
            "# TYPE vllm:prefix_cache_queries_created gauge",
            'vllm:prefix_cache_queries_created{engine="0",model_name="x"} 1.7882e+09',
            "# TYPE vllm:external_prefix_cache_queries_total counter",
            'vllm:external_prefix_cache_queries_total{engine="0",model_name="x"} 0.0',
            "",
        ]
    )
    snapshot = sr._serving_snapshot(text)

    assert snapshot["counters"]["prompt_tokens_total"] == 4.4481e08
    assert snapshot["counters"]["prefix_cache_queries_total"] == 4.4523e08


def test_sample_serving_metrics_returns_none_on_fetch_failure() -> None:
    with patch(
        "urllib.request.urlopen",
        _stub_urlopen([urllib.error.URLError("no route to host")]),
    ):
        snapshot, sampled_at = sr.sample_serving_metrics("http://gpu-node:18800")

    assert snapshot is None
    assert sampled_at.tzinfo is not None


# ---------------------------------------------------------------------------
# Receipt row shape
# ---------------------------------------------------------------------------


def test_receipt_row_on_lone_snapshot_has_no_throughput_yet() -> None:
    snapshot = sr._serving_snapshot(SAMPLE_T0.decode())
    row = sr.build_receipt_row(
        None,
        None,
        snapshot,
        T0,
        job_id="1234567",
        profile_slug="deepseek-v4-flash",
        served_name="deepseek-v4-flash",
        gpus=4,
    )

    assert row.timestamp == T0.isoformat()
    assert row.job_id == "1234567"
    assert row.profile_slug == "deepseek-v4-flash"
    assert row.served_name == "deepseek-v4-flash"
    assert row.gpus == 4
    assert row.generation_throughput_toks_per_s is None
    assert row.prompt_throughput_toks_per_s is None
    assert row.num_requests_running == 3
    assert row.num_requests_waiting == 0
    assert row.kv_cache_usage_perc == 0.12
    assert row.prefix_cache_queries_total == 984_123
    assert row.prefix_cache_hits_total == 812_004
    # Cumulative ratio needs no second sample.
    assert row.prefix_cache_hit_rate == round(812_004 / 984_123, 4)
    # No prior spec-decode snapshot to difference against.
    assert row.spec_draft_tokens is None
    assert row.spec_accepted_tokens is None
    assert row.spec_acceptance_rate is None


def test_receipt_row_over_a_window_computes_throughput_and_acceptance() -> None:
    prev = sr._serving_snapshot(SAMPLE_T0.decode())
    curr = sr._serving_snapshot(SAMPLE_T1.decode())
    row = sr.build_receipt_row(
        prev,
        T0,
        curr,
        T1,
        job_id="1234567",
        profile_slug="deepseek-v4-flash",
        served_name="deepseek-v4-flash",
        gpus=4,
    )

    # +300 generation tokens / 5s, +0 prompt tokens / 5s.
    assert row.generation_throughput_toks_per_s == 60.0
    assert row.prompt_throughput_toks_per_s == 0.0
    assert row.num_requests_running == 2
    assert row.num_requests_waiting == 1
    assert row.kv_cache_usage_perc == 0.15
    # +2000 queries, +1800 hits over the window.
    assert row.spec_draft_tokens == 500
    assert row.spec_accepted_tokens == 300
    assert row.spec_acceptance_rate == round(300 / 500, 4)
    assert row.spec_num_accepted_per_pos == [90, 60, 45, 25, 20]
    # Cumulative ratio at T1, not windowed.
    assert row.prefix_cache_hit_rate == round(813_804 / 986_123, 4)


def test_receipt_row_zero_elapsed_reports_no_throughput() -> None:
    prev = sr._serving_snapshot(SAMPLE_T0.decode())
    curr = sr._serving_snapshot(SAMPLE_T1.decode())
    row = sr.build_receipt_row(
        prev,
        T0,
        curr,
        T0,  # same instant as the previous sample
        job_id=None,
        profile_slug=None,
        served_name=None,
        gpus=None,
    )

    assert row.generation_throughput_toks_per_s is None
    assert row.prompt_throughput_toks_per_s is None


def test_receipt_row_serializes_to_json() -> None:
    snapshot = sr._serving_snapshot(SAMPLE_T0.decode())
    row = sr.build_receipt_row(
        None,
        None,
        snapshot,
        T0,
        job_id="1234567",
        profile_slug="deepseek-v4-flash",
        served_name="deepseek-v4-flash",
        gpus=4,
    )
    payload = json.loads(row.to_json())
    assert payload["profile_slug"] == "deepseek-v4-flash"
    assert payload["prefix_cache_hit_rate"] == row.prefix_cache_hit_rate


# ---------------------------------------------------------------------------
# Continuous recorder
# ---------------------------------------------------------------------------


def test_record_receipts_appends_one_row_per_successful_sample(
    tmp_path: Path,
) -> None:
    receipts_path = tmp_path / "receipts.jsonl"
    # start, then one elapsed-check per iteration: 5s (continue), 10s (stop).
    clock = iter([0.0, 5.0, 10.0])
    wall_clock = iter([T0, T1])
    slept: list[float] = []

    with patch("urllib.request.urlopen", _stub_urlopen([SAMPLE_T0, SAMPLE_T1])):
        rows_written = sr.record_receipts(
            "http://98dci4-gpu-0003:18800",
            receipts_path,
            interval_s=5.0,
            duration_s=10.0,
            sleep=slept.append,
            monotonic=lambda: next(clock),
            now=lambda: next(wall_clock),
        )

    assert rows_written == 2
    lines = receipts_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    second = json.loads(lines[1])
    assert first["generation_throughput_toks_per_s"] is None
    assert second["generation_throughput_toks_per_s"] is not None
    assert second["prefix_cache_hit_rate"] is not None
    assert slept == [5.0]


def test_record_receipts_skips_a_failed_scrape_without_raising(
    tmp_path: Path,
) -> None:
    receipts_path = tmp_path / "receipts.jsonl"
    clock = iter([0.0, 5.0])

    with patch(
        "urllib.request.urlopen",
        _stub_urlopen([urllib.error.URLError("engine not up yet")]),
    ):
        rows_written = sr.record_receipts(
            "http://98dci4-gpu-0003:18800",
            receipts_path,
            interval_s=5.0,
            duration_s=5.0,
            sleep=lambda _s: None,
            monotonic=lambda: next(clock),
        )

    assert rows_written == 0
    assert receipts_path.read_text(encoding="utf-8") == ""


def test_record_receipts_is_append_only_across_invocations(tmp_path: Path) -> None:
    receipts_path = tmp_path / "receipts.jsonl"
    clock_first = iter([0.0, 0.0])
    with patch("urllib.request.urlopen", _stub_urlopen([SAMPLE_T0])):
        sr.record_receipts(
            "http://98dci4-gpu-0003:18800",
            receipts_path,
            interval_s=5.0,
            duration_s=0.0,
            sleep=lambda _s: None,
            monotonic=lambda: next(clock_first),
            now=lambda: T0,
        )
    first_pass = receipts_path.read_text(encoding="utf-8")
    assert len(first_pass.splitlines()) == 1

    clock_second = iter([0.0, 0.0])
    with patch("urllib.request.urlopen", _stub_urlopen([SAMPLE_T1])):
        sr.record_receipts(
            "http://98dci4-gpu-0003:18800",
            receipts_path,
            interval_s=5.0,
            duration_s=0.0,
            sleep=lambda _s: None,
            monotonic=lambda: next(clock_second),
            now=lambda: T1,
        )
    second_pass = receipts_path.read_text(encoding="utf-8")
    lines = second_pass.splitlines()
    assert len(lines) == 2
    assert lines[0] == first_pass.strip()
