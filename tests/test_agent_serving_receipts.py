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
7.  A row carries the probe's card, host and job sections beside the engine
    section, and names the node whose readings they are; a section the probe
    did not read this tick is *absent* from the row rather than ``null``, so a
    reader never has to tell an unmeasured section from a measured zero.
8.  The recorder compacts its own record into the resolution tiers between
    samples, on its own cadence, and a compaction that fails is reported and
    skipped rather than allowed to end the recording.

All HTTP is stubbed at ``urllib.request.urlopen`` — no network, no GPU, and
the node probe is a stub of the caller's — the real one shells out to
``nvidia-smi``/``squeue`` on the machine running the suite.
"""

from __future__ import annotations

import datetime as _dt
import json
import urllib.error
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest

from imas_ambix.agent import node_probe
from imas_ambix.agent import serving_receipts as sr
from imas_ambix.agent.profile import SiteConfig, load_profile
from imas_ambix.agent.slurm import generate_serve_script

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
        "# TYPE vllm:spec_decode_num_accepted_tokens_per_pos_total counter",
    ]
    lines += [
        (
            "vllm:spec_decode_num_accepted_tokens_per_pos_total"
            f'{{engine="0",position="{i}"}} {v}'
        )
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

FIRST_SAMPLE_AT = _dt.datetime(2026, 9, 5, 12, 0, 0, tzinfo=_dt.UTC)
SECOND_SAMPLE_AT = FIRST_SAMPLE_AT + _dt.timedelta(seconds=5)


# ---------------------------------------------------------------------------
# Verbatim scrapes captured from the live four-card DSpark endpoint
# (http://98dci4-gpu-0003:18801/metrics, 2026-09-05T14:01-14:02Z), 15s apart.
# Unlike SAMPLE_T0/SAMPLE_T1 above, these are the engine's own bytes, not a
# hand-written approximation — the earlier hand-written fixture's per-position
# metric name (``..._per_pos``, no ``_total``) never matched what the engine
# actually publishes (``..._per_pos_total``), so it could not catch a naming
# defect a live scrape exposes immediately.
# ---------------------------------------------------------------------------

def _live_sample(name: str, labels: str, value: str) -> str:
    """One data line, in the engine's own exposition shape.

    The ``# HELP``/``# TYPE`` comment lines a real scrape also carries are
    pure prose that :func:`~imas_ambix.agent.bench._parse_prometheus_text`
    never reads, so they are omitted here rather than wrapped to fit the
    lint line-length gate; every value below is transcribed unchanged from
    the live scrape.
    """
    return f'vllm:{name}{{{labels}}} {value}'


_ENGINE_LABELS = 'engine="0",model_name="deepseek-v4-flash"'

LIVE_METRICS_T0 = "\n".join(
    [
        _live_sample("spec_decode_num_drafts_total", _ENGINE_LABELS, "2739.0"),
        _live_sample(
            "spec_decode_num_drafts_created",
            _ENGINE_LABELS,
            "1.7886160770691342e+09",
        ),
        _live_sample("spec_decode_num_draft_tokens_total", _ENGINE_LABELS, "13695.0"),
        _live_sample(
            "spec_decode_num_draft_tokens_created",
            _ENGINE_LABELS,
            "1.7886160770691533e+09",
        ),
        _live_sample(
            "spec_decode_num_accepted_tokens_total", _ENGINE_LABELS, "6425.0"
        ),
        _live_sample(
            "spec_decode_num_accepted_tokens_created",
            _ENGINE_LABELS,
            "1.7886160770691645e+09",
        ),
        _live_sample(
            "spec_decode_num_accepted_tokens_per_pos_total",
            f'{_ENGINE_LABELS},position="0"',
            "2048.0",
        ),
        _live_sample(
            "spec_decode_num_accepted_tokens_per_pos_total",
            f'{_ENGINE_LABELS},position="1"',
            "1588.0",
        ),
        _live_sample(
            "spec_decode_num_accepted_tokens_per_pos_total",
            f'{_ENGINE_LABELS},position="2"',
            "1235.0",
        ),
        _live_sample(
            "spec_decode_num_accepted_tokens_per_pos_total",
            f'{_ENGINE_LABELS},position="3"',
            "903.0",
        ),
        _live_sample(
            "spec_decode_num_accepted_tokens_per_pos_total",
            f'{_ENGINE_LABELS},position="4"',
            "651.0",
        ),
        _live_sample(
            "spec_decode_num_accepted_tokens_per_pos_created",
            f'{_ENGINE_LABELS},position="0"',
            "1.7886160770691786e+09",
        ),
        _live_sample(
            "spec_decode_num_accepted_tokens_per_pos_created",
            f'{_ENGINE_LABELS},position="1"',
            "1.7886160770691829e+09",
        ),
        _live_sample(
            "spec_decode_num_accepted_tokens_per_pos_created",
            f'{_ENGINE_LABELS},position="2"',
            "1.788616077069186e+09",
        ),
        _live_sample(
            "spec_decode_num_accepted_tokens_per_pos_created",
            f'{_ENGINE_LABELS},position="3"',
            "1.7886160770691888e+09",
        ),
        _live_sample(
            "spec_decode_num_accepted_tokens_per_pos_created",
            f'{_ENGINE_LABELS},position="4"',
            "1.788616077069193e+09",
        ),
        _live_sample("num_requests_running", _ENGINE_LABELS, "0.0"),
        _live_sample("num_requests_waiting", _ENGINE_LABELS, "0.0"),
        _live_sample("kv_cache_usage_perc", _ENGINE_LABELS, "0.0"),
        _live_sample("prefix_cache_queries_total", _ENGINE_LABELS, "3.153518e+06"),
        _live_sample(
            "prefix_cache_queries_created",
            _ENGINE_LABELS,
            "1.7886160770693438e+09",
        ),
        _live_sample("prefix_cache_hits_total", _ENGINE_LABELS, "619520.0"),
        _live_sample(
            "prefix_cache_hits_created", _ENGINE_LABELS, "1.788616077069351e+09"
        ),
        _live_sample("prompt_tokens_total", _ENGINE_LABELS, "3.153518e+06"),
        _live_sample(
            "prompt_tokens_created", _ENGINE_LABELS, "1.7886160770694067e+09"
        ),
        _live_sample("generation_tokens_total", _ENGINE_LABELS, "9170.0"),
        _live_sample(
            "generation_tokens_created", _ENGINE_LABELS, "1.7886160770694497e+09"
        ),
        "",
    ]
)

LIVE_METRICS_T1 = "\n".join(
    [
        _live_sample("spec_decode_num_drafts_total", _ENGINE_LABELS, "2780.0"),
        _live_sample(
            "spec_decode_num_drafts_created",
            _ENGINE_LABELS,
            "1.7886160770691342e+09",
        ),
        _live_sample("spec_decode_num_draft_tokens_total", _ENGINE_LABELS, "13900.0"),
        _live_sample(
            "spec_decode_num_draft_tokens_created",
            _ENGINE_LABELS,
            "1.7886160770691533e+09",
        ),
        _live_sample(
            "spec_decode_num_accepted_tokens_total", _ENGINE_LABELS, "6535.0"
        ),
        _live_sample(
            "spec_decode_num_accepted_tokens_created",
            _ENGINE_LABELS,
            "1.7886160770691645e+09",
        ),
        _live_sample(
            "spec_decode_num_accepted_tokens_per_pos_total",
            f'{_ENGINE_LABELS},position="0"',
            "2083.0",
        ),
        _live_sample(
            "spec_decode_num_accepted_tokens_per_pos_total",
            f'{_ENGINE_LABELS},position="1"',
            "1618.0",
        ),
        _live_sample(
            "spec_decode_num_accepted_tokens_per_pos_total",
            f'{_ENGINE_LABELS},position="2"',
            "1255.0",
        ),
        _live_sample(
            "spec_decode_num_accepted_tokens_per_pos_total",
            f'{_ENGINE_LABELS},position="3"',
            "918.0",
        ),
        _live_sample(
            "spec_decode_num_accepted_tokens_per_pos_total",
            f'{_ENGINE_LABELS},position="4"',
            "661.0",
        ),
        _live_sample(
            "spec_decode_num_accepted_tokens_per_pos_created",
            f'{_ENGINE_LABELS},position="0"',
            "1.7886160770691786e+09",
        ),
        _live_sample(
            "spec_decode_num_accepted_tokens_per_pos_created",
            f'{_ENGINE_LABELS},position="1"',
            "1.7886160770691829e+09",
        ),
        _live_sample(
            "spec_decode_num_accepted_tokens_per_pos_created",
            f'{_ENGINE_LABELS},position="2"',
            "1.788616077069186e+09",
        ),
        _live_sample(
            "spec_decode_num_accepted_tokens_per_pos_created",
            f'{_ENGINE_LABELS},position="3"',
            "1.7886160770691888e+09",
        ),
        _live_sample(
            "spec_decode_num_accepted_tokens_per_pos_created",
            f'{_ENGINE_LABELS},position="4"',
            "1.788616077069193e+09",
        ),
        _live_sample("num_requests_running", _ENGINE_LABELS, "1.0"),
        _live_sample("num_requests_waiting", _ENGINE_LABELS, "0.0"),
        _live_sample("kv_cache_usage_perc", _ENGINE_LABELS, "0.00593715239154613"),
        _live_sample("prefix_cache_queries_total", _ENGINE_LABELS, "3.254449e+06"),
        _live_sample(
            "prefix_cache_queries_created",
            _ENGINE_LABELS,
            "1.7886160770693438e+09",
        ),
        _live_sample("prefix_cache_hits_total", _ENGINE_LABELS, "650496.0"),
        _live_sample(
            "prefix_cache_hits_created", _ENGINE_LABELS, "1.788616077069351e+09"
        ),
        _live_sample("prompt_tokens_total", _ENGINE_LABELS, "3.254449e+06"),
        _live_sample(
            "prompt_tokens_created", _ENGINE_LABELS, "1.7886160770694067e+09"
        ),
        _live_sample("generation_tokens_total", _ENGINE_LABELS, "9322.0"),
        _live_sample(
            "generation_tokens_created", _ENGINE_LABELS, "1.7886160770694497e+09"
        ),
        "",
    ]
)


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


def test_serving_snapshot_reads_live_four_card_spec_decode_names() -> None:
    """Names verified against the live engine, not assumed from a fixture.

    ``spec_decode_num_draft_tokens_total``,
    ``spec_decode_num_accepted_tokens_total`` and
    ``spec_decode_num_accepted_tokens_per_pos_total`` are what the live
    four-card DSpark engine actually publishes; each also carries a
    same-named ``_created`` gauge sibling (a creation timestamp, ~1.79e9),
    which a correct reader must not fold into the token count.
    """
    snapshot = sr._serving_snapshot(LIVE_METRICS_T0)

    assert snapshot["spec_decode"]["draft_tokens_total"] == 13695.0
    assert snapshot["spec_decode"]["accepted_tokens_total"] == 6425.0
    assert snapshot["spec_decode"]["num_accepted_per_pos"] == [
        2048.0,
        1588.0,
        1235.0,
        903.0,
        651.0,
    ]


def test_receipt_row_over_live_four_card_window_is_nonzero_and_descending() -> None:
    """The delta-derived receipt fields on a real 15s window from the engine.

    This is the regression the plan named directly: reading these fields
    against a hand-written fixture cannot catch a name the engine does not
    actually publish, so this asserts against the engine's own bytes.
    """
    prev = sr._serving_snapshot(LIVE_METRICS_T0)
    curr = sr._serving_snapshot(LIVE_METRICS_T1)
    row = sr.build_receipt_row(
        prev,
        FIRST_SAMPLE_AT,
        curr,
        FIRST_SAMPLE_AT + _dt.timedelta(seconds=15),
        job_id="1262921",
        profile_slug="deepseek-v4-flash",
        served_name="deepseek-v4-flash",
        gpus=4,
    )

    assert row.spec_draft_tokens == 205
    assert row.spec_accepted_tokens == 110
    assert row.spec_acceptance_rate == round(110 / 205, 4)
    assert row.spec_num_accepted_per_pos == [35, 30, 20, 15, 10]
    assert row.spec_num_accepted_per_pos == sorted(
        row.spec_num_accepted_per_pos, reverse=True
    )


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
        FIRST_SAMPLE_AT,
        job_id="1234567",
        profile_slug="deepseek-v4-flash",
        served_name="deepseek-v4-flash",
        gpus=4,
    )

    assert row.timestamp == FIRST_SAMPLE_AT.isoformat()
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
    assert row.prefix_cache_query_delta is None
    assert row.prefix_cache_hit_delta is None
    assert row.prefix_cache_hit_rate_interval is None
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
        FIRST_SAMPLE_AT,
        curr,
        SECOND_SAMPLE_AT,
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
    assert row.prefix_cache_query_delta == 2_000
    assert row.prefix_cache_hit_delta == 1_800
    assert row.prefix_cache_hit_rate_interval == 0.9
    # Cumulative ratio at the later sample, not windowed.
    assert row.prefix_cache_hit_rate == round(813_804 / 986_123, 4)
    assert row.prefix_cache_hit_rate == 0.8253


def test_receipt_row_zero_elapsed_reports_no_throughput() -> None:
    prev = sr._serving_snapshot(SAMPLE_T0.decode())
    curr = sr._serving_snapshot(SAMPLE_T1.decode())
    row = sr.build_receipt_row(
        prev,
        FIRST_SAMPLE_AT,
        curr,
        FIRST_SAMPLE_AT,  # same instant as the previous sample
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
        FIRST_SAMPLE_AT,
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
    wall_clock = iter([FIRST_SAMPLE_AT, SECOND_SAMPLE_AT])
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
    assert first["prefix_cache_query_delta"] is None
    assert first["prefix_cache_hit_delta"] is None
    assert first["prefix_cache_hit_rate_interval"] is None
    assert second["generation_throughput_toks_per_s"] is not None
    assert second["prefix_cache_query_delta"] == 2_000
    assert second["prefix_cache_hit_delta"] == 1_800
    assert second["prefix_cache_hit_rate_interval"] == 0.9
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
            now=lambda: FIRST_SAMPLE_AT,
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
            now=lambda: SECOND_SAMPLE_AT,
        )
    second_pass = receipts_path.read_text(encoding="utf-8")
    lines = second_pass.splitlines()
    assert len(lines) == 2
    assert lines[0] == first_pass.strip()


# ---------------------------------------------------------------------------
# Node sections, the producing host, and the recorder's own compaction
# ---------------------------------------------------------------------------


def _read_rows(path: Path) -> list[dict[str, Any]]:
    """The JSONL rows written at *path*, one parsed object per line."""
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


CARD_SECTION: dict[str, Any] = {
    "index_source": "SLURM_JOB_GPUS",
    "count": 1,
    "cards": [{"index": 2, "utilization_percent": 91, "temperature_c": 61}],
}
HOST_SECTION: dict[str, Any] = {
    "memory_total_mib": 1_048_576,
    "memory_available_mib": 700_000,
    "cpu_busy_fraction": 0.31,
}
JOB_SECTION: dict[str, Any] = {
    "hostname": "98dci4-gpu-0003",
    "count": 1,
    "jobs": [{"job_id": "1262921", "state": "RUNNING"}],
}


class _StubProbe:
    """A node probe that answers from a per-tick list and records its ticks.

    The real probe owns its own cadence and shells out for each reading; this
    one answers ticks in order, with the last answer repeating, so a section
    that is due on the first tick and omitted afterwards is exercised without
    waiting on a clock or a scheduler.
    """

    def __init__(self, sections_by_tick: list[dict[str, Any]]) -> None:
        self.hostname = "98dci4-gpu-0003"
        self.ticks: list[float] = []
        self._by_tick = sections_by_tick

    def sample(self, now: float) -> dict[str, Any]:
        self.ticks.append(now)
        index = min(len(self.ticks), len(self._by_tick)) - 1
        return self._by_tick[index]


def test_receipt_row_carries_node_sections_and_the_producing_host() -> None:
    snapshot = sr._serving_snapshot(SAMPLE_T0.decode())
    row = sr.build_receipt_row(
        None,
        None,
        snapshot,
        FIRST_SAMPLE_AT,
        job_id="1262921",
        profile_slug="deepseek-v4-flash",
        served_name="deepseek-v4-flash",
        gpus=4,
        hostname="98dci4-gpu-0003",
        node_sections={
            "cards": CARD_SECTION,
            "host": HOST_SECTION,
            "jobs": JOB_SECTION,
        },
    )

    payload = json.loads(row.to_json())
    assert payload["schema_version"] == sr.ROW_SCHEMA_VERSION
    assert payload["hostname"] == "98dci4-gpu-0003"
    assert payload["cards"] == CARD_SECTION
    assert payload["host"] == HOST_SECTION
    assert payload["jobs"] == JOB_SECTION


def test_receipt_row_omits_a_section_its_probe_did_not_read() -> None:
    """A section absent this tick is absent from the row, never ``null``.

    That is what lets a reader treat a present key as a measurement. Writing
    the key with a null value would make an unread job table indistinguishable
    from a job table that was read and found empty.
    """
    snapshot = sr._serving_snapshot(SAMPLE_T0.decode())
    row = sr.build_receipt_row(
        None,
        None,
        snapshot,
        FIRST_SAMPLE_AT,
        job_id="1262921",
        profile_slug="deepseek-v4-flash",
        served_name="deepseek-v4-flash",
        gpus=4,
        hostname="98dci4-gpu-0003",
        # The job table is a SLURM RPC read on its own slower cadence, so this
        # tick simply does not have one.
        node_sections={"cards": CARD_SECTION, "host": HOST_SECTION},
    )

    payload = json.loads(row.to_json())
    assert set(sr.ROW_SECTIONS) - set(payload) == {"jobs"}
    assert all(payload[name] is not None for name in sr.ROW_SECTIONS if name in payload)


def test_receipt_row_without_a_probe_names_the_local_host() -> None:
    """With no probe the row still names its host, and carries no node section."""
    snapshot = sr._serving_snapshot(SAMPLE_T0.decode())
    row = sr.build_receipt_row(
        None,
        None,
        snapshot,
        FIRST_SAMPLE_AT,
        job_id="1262921",
        profile_slug="deepseek-v4-flash",
        served_name="deepseek-v4-flash",
        gpus=4,
    )

    payload = json.loads(row.to_json())
    assert payload["hostname"] == sr.local_hostname()
    for section in ("cards", "host", "jobs"):
        assert section not in payload


def test_record_receipts_attaches_probe_sections_and_names_the_host(
    tmp_path: Path,
) -> None:
    receipts_path = tmp_path / "receipts.jsonl"
    sections = {"cards": CARD_SECTION, "host": HOST_SECTION, "jobs": JOB_SECTION}
    probe = _StubProbe([sections, sections])
    clock = iter([0.0, 5.0, 10.0])
    wall_clock = iter([FIRST_SAMPLE_AT, SECOND_SAMPLE_AT])

    with patch("urllib.request.urlopen", _stub_urlopen([SAMPLE_T0, SAMPLE_T1])):
        rows_written = sr.record_receipts(
            "http://98dci4-gpu-0003:18800",
            receipts_path,
            interval_s=5.0,
            duration_s=10.0,
            probe=probe,
            sleep=lambda _s: None,
            monotonic=lambda: next(clock),
            now=lambda: next(wall_clock),
        )

    assert rows_written == 2
    written = _read_rows(receipts_path)
    assert [row["hostname"] for row in written] == ["98dci4-gpu-0003"] * 2
    assert [row["cards"] for row in written] == [CARD_SECTION] * 2
    assert [row["jobs"] for row in written] == [JOB_SECTION] * 2
    # The probe is offered the tick the recorder is already holding for its own
    # elapsed check, so the two cannot disagree about when the tick happened.
    assert probe.ticks == [5.0, 10.0]


def test_record_receipts_writes_a_staggered_section_as_absent(tmp_path: Path) -> None:
    """The omission survives serialization into the file, tick by tick."""
    receipts_path = tmp_path / "receipts.jsonl"
    probe = _StubProbe(
        [
            {"cards": CARD_SECTION, "host": HOST_SECTION, "jobs": JOB_SECTION},
            {"cards": CARD_SECTION, "host": HOST_SECTION},
        ]
    )
    clock = iter([0.0, 5.0, 10.0])
    wall_clock = iter([FIRST_SAMPLE_AT, SECOND_SAMPLE_AT])

    with patch("urllib.request.urlopen", _stub_urlopen([SAMPLE_T0, SAMPLE_T1])):
        sr.record_receipts(
            "http://98dci4-gpu-0003:18800",
            receipts_path,
            interval_s=5.0,
            duration_s=10.0,
            probe=probe,
            sleep=lambda _s: None,
            monotonic=lambda: next(clock),
            now=lambda: next(wall_clock),
        )

    first, second = _read_rows(receipts_path)
    assert "jobs" in first
    assert "jobs" not in second
    assert "cards" in second


def test_tier_paths_are_siblings_of_the_raw_record() -> None:
    minute, hour = sr.tier_paths("/receipts/deepseek-v4-flash-1262921.jsonl")
    assert minute == Path("/receipts/deepseek-v4-flash-1262921.minute.jsonl")
    assert hour == Path("/receipts/deepseek-v4-flash-1262921.hour.jsonl")


def test_record_receipts_compacts_its_record_between_samples(tmp_path: Path) -> None:
    receipts_path = tmp_path / "receipts.jsonl"
    minute_path, hour_path = sr.tier_paths(receipts_path)
    clock = iter([0.0, 5.0, 10.0])
    wall_clock = iter([FIRST_SAMPLE_AT, SECOND_SAMPLE_AT])

    with patch("urllib.request.urlopen", _stub_urlopen([SAMPLE_T0, SAMPLE_T1])):
        rows_written = sr.record_receipts(
            "http://98dci4-gpu-0003:18800",
            receipts_path,
            interval_s=5.0,
            duration_s=10.0,
            compaction_paths=(minute_path, hour_path),
            compaction_interval_s=6.0,
            sleep=lambda _s: None,
            monotonic=lambda: next(clock),
            now=lambda: next(wall_clock),
        )

    assert rows_written == 2
    # Both raw ticks fall in one minute window, so the tier is coarser than its
    # source by construction rather than by luck of the fixtures.
    assert len(receipts_path.read_text(encoding="utf-8").splitlines()) == 2
    minute_rows = _read_rows(minute_path)
    assert len(minute_rows) == 1
    assert minute_rows[0]["tier"] == "minute"
    assert minute_rows[0]["obs"] == {"samples": 2, "seconds": 10.0}
    hour_rows = _read_rows(hour_path)
    assert len(hour_rows) == 1
    assert hour_rows[0]["tier"] == "hour"


def test_record_receipts_does_not_compact_before_the_first_cadence(
    tmp_path: Path,
) -> None:
    """Compaction is on its own clock: a tick short of the cadence is untouched."""
    receipts_path = tmp_path / "receipts.jsonl"
    minute_path, hour_path = sr.tier_paths(receipts_path)
    clock = iter([0.0, 5.0, 10.0])

    with patch("urllib.request.urlopen", _stub_urlopen([SAMPLE_T0, SAMPLE_T1])):
        sr.record_receipts(
            "http://98dci4-gpu-0003:18800",
            receipts_path,
            interval_s=5.0,
            duration_s=10.0,
            compaction_paths=(minute_path, hour_path),
            compaction_interval_s=100.0,
            sleep=lambda _s: None,
            monotonic=lambda: next(clock),
            now=lambda: FIRST_SAMPLE_AT,
        )

    assert not minute_path.exists()
    assert not hour_path.exists()


def test_record_receipts_keeps_recording_when_a_compaction_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A compaction that cannot finish costs one attempt, not the record."""
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("occupied", encoding="utf-8")
    receipts_path = tmp_path / "receipts.jsonl"
    clock = iter([0.0, 5.0, 10.0])
    wall_clock = iter([FIRST_SAMPLE_AT, SECOND_SAMPLE_AT])

    with patch("urllib.request.urlopen", _stub_urlopen([SAMPLE_T0, SAMPLE_T1])):
        rows_written = sr.record_receipts(
            "http://98dci4-gpu-0003:18800",
            receipts_path,
            interval_s=5.0,
            duration_s=10.0,
            compaction_paths=(blocker / "minute.jsonl", blocker / "hour.jsonl"),
            compaction_interval_s=1.0,
            sleep=lambda _s: None,
            monotonic=lambda: next(clock),
            now=lambda: next(wall_clock),
        )

    assert rows_written == 2
    assert len(receipts_path.read_text(encoding="utf-8").splitlines()) == 2
    assert "compaction" in capsys.readouterr().err


def test_receipts_main_rebuilds_one_tier_and_exits(tmp_path: Path) -> None:
    """The other CLI mode: catch a tier up from a file and exit."""
    source = tmp_path / "receipts.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {
                    "timestamp": FIRST_SAMPLE_AT.isoformat(),
                    "schema_version": sr.ROW_SCHEMA_VERSION,
                    "hostname": "98dci4-gpu-0003",
                    "generation_throughput_toks_per_s": 10.0,
                },
                {
                    "timestamp": SECOND_SAMPLE_AT.isoformat(),
                    "schema_version": sr.ROW_SCHEMA_VERSION,
                    "hostname": "98dci4-gpu-0003",
                    "generation_throughput_toks_per_s": 30.0,
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )
    destination = tmp_path / "receipts.minute.jsonl"

    exit_code = sr.main(
        [
            "--compact-tier",
            "minute",
            "--compact-source",
            str(source),
            "--compact-destination",
            str(destination),
        ]
    )

    assert exit_code == 0
    rows = _read_rows(destination)
    assert len(rows) == 1
    assert rows[0]["tier"] == "minute"
    assert rows[0]["obs"]["samples"] == 2
    # A gauge compacts to the time-weighted mean of the two 5-second samples.
    assert rows[0]["generation_throughput_toks_per_s"] == 20.0


def test_receipts_main_refuses_a_tier_rebuild_without_its_paths(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit):
        sr.main(["--compact-tier", "minute"])


# ---------------------------------------------------------------------------
# Generated serve script — background sidecar wiring
# ---------------------------------------------------------------------------


def test_generated_serve_script_starts_receipts_sidecar_by_default(
    tmp_path: Path,
) -> None:
    profile = load_profile("deepseek-v4-flash").for_gpus(4)
    script = generate_serve_script(
        profile, SiteConfig(base_dir=str(tmp_path)), port=18801
    )

    assert "-m imas_ambix.agent.serving_receipts" in script
    assert '"$_RECEIPTS_DIR"' in script
    receipts_dir = str(Path(tmp_path) / "agents" / "receipts")
    assert receipts_dir in script
    assert f"{profile.slug}-$SLURM_JOB_ID.jsonl" in script
    assert "_RECEIPTS_PID=$!" in script
    # Started right after the engine process, before the receipts recorder
    # could be mistaken for the server itself.
    assert script.index("SERVER_PID=$!") < script.index(
        "-m imas_ambix.agent.serving_receipts"
    )
    # Killed on cleanup so it never outlives the job it is sampling.
    cleanup_start = script.index("cleanup_serve()")
    cleanup_end = script.index("terminate_serve()")
    assert '"$_RECEIPTS_PID"' in script[cleanup_start:cleanup_end]


def test_generated_serve_script_omits_receipts_sidecar_when_disabled(
    tmp_path: Path,
) -> None:
    profile = load_profile("deepseek-v4-flash").for_gpus(4)
    script = generate_serve_script(
        profile,
        SiteConfig(base_dir=str(tmp_path)),
        port=18801,
        receipts_enabled=False,
    )

    assert "-m imas_ambix.agent.serving_receipts" not in script
    assert "_RECEIPTS_PID=$!" not in script
    # The kill-on-cleanup guard stays present and harmless: _RECEIPTS_PID is
    # declared but never assigned, so the guard is a no-op rather than an
    # unbound-variable failure.
    assert '_RECEIPTS_PID=""' in script


def test_receipts_main_invokes_record_receipts_with_parsed_arguments(
    tmp_path: Path,
) -> None:
    receipts_path = tmp_path / "receipts.jsonl"
    captured: dict[str, Any] = {}

    def _fake_record_receipts(base_url: str, path: Any, **kwargs: Any) -> int:
        captured["base_url"] = base_url
        captured["path"] = path
        captured.update(kwargs)
        return 3

    with patch.object(sr, "record_receipts", _fake_record_receipts):
        exit_code = sr.main(
            [
                "--base-url",
                "http://98dci4-gpu-0003:18801",
                "--receipts-path",
                str(receipts_path),
                "--interval",
                "5",
                "--job-id",
                "1262921",
                "--profile-slug",
                "deepseek-v4-flash",
                "--served-name",
                "deepseek-v4-flash",
                "--gpus",
                "4",
            ]
        )

    assert exit_code == 0
    assert captured["base_url"] == "http://98dci4-gpu-0003:18801"
    assert captured["path"] == str(receipts_path)
    assert captured["interval_s"] == 5.0
    assert captured["serve_job_id"] == "1262921"
    assert captured["profile_slug"] == "deepseek-v4-flash"
    assert captured["served_name"] == "deepseek-v4-flash"
    assert captured["gpus"] == 4
    # The node probe is on by default, because the sidecar that invokes this is
    # the process holding the job's allocation — the only place the readings
    # are free — and the record's own tiers are derived from the raw path.
    assert isinstance(captured["probe"], node_probe.NodeProbe)
    assert captured["compaction_paths"] == sr.tier_paths(str(receipts_path))
    assert captured["compaction_interval_s"] == sr.DEFAULT_COMPACTION_INTERVAL_S


def test_receipts_main_can_record_without_the_probe_or_compaction(
    tmp_path: Path,
) -> None:
    receipts_path = tmp_path / "receipts.jsonl"
    captured: dict[str, Any] = {}

    def _fake_record_receipts(base_url: str, path: Any, **kwargs: Any) -> int:
        captured.update(kwargs)
        return 0

    with patch.object(sr, "record_receipts", _fake_record_receipts):
        exit_code = sr.main(
            [
                "--base-url",
                "http://98dci4-gpu-0003:18801",
                "--receipts-path",
                str(receipts_path),
                "--no-node-probe",
                "--no-compaction",
            ]
        )

    assert exit_code == 0
    assert captured["probe"] is None
    assert captured["compaction_paths"] is None
