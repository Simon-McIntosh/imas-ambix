"""Continuous serving-time receipts for ``imas-ambix agent receipts``.

:mod:`imas_ambix.agent.bench` already parses a Prometheus ``/metrics`` scrape
(:func:`~imas_ambix.agent.bench._parse_prometheus_text`) and differences
cumulative counters across a window
(:func:`~imas_ambix.agent.bench._counter_delta`,
:func:`~imas_ambix.agent.bench._per_position_delta`). This module reuses that
parsing and differencing to build a *continuous* recorder: it samples a live
engine's ``/metrics`` on an interval and appends one JSON row per sample to a
durable, append-only receipts file, so a serve's whole life is a readable
record rather than something reconstructed afterwards from a terminal
summary. Speculative-decode counters are read here by exact bare name rather
than reusing ``bench``'s substring-matching snapshot, because the substring
match folds each counter's ``_created`` timestamp companion into its token
total (see the alias-set comment below).
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from imas_ambix.agent.bench import (
    _counter_delta,
    _fetch_body,
    _parse_prometheus_text,
    _per_position_delta,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from imas_ambix.agent.profile import ModelProfile

# Matched on the bare metric name (after the ``vllm:`` namespace prefix),
# against every spelling an engine version is known to use. An exact-name set
# rather than a substring is deliberate here: vLLM's OpenMetrics exposition
# pairs every counter with a same-named ``_created`` gauge (its creation
# timestamp, not a data point) and a cross-instance ``external_`` variant, and
# a substring test would silently fold either into the real counter — measured
# on the live four-card engine, where a substring match on "draft"/"accept"
# summed each counter's ``_created`` companion straight into its token total.
_GAUGE_NAMES = {
    "num_requests_running": {"num_requests_running"},
    "num_requests_waiting": {"num_requests_waiting"},
    "kv_cache_usage_perc": {"kv_cache_usage_perc", "gpu_cache_usage_perc"},
}
_COUNTER_NAMES = {
    "prompt_tokens_total": {"prompt_tokens_total"},
    "generation_tokens_total": {"generation_tokens_total"},
    "prefix_cache_queries_total": {
        "prefix_cache_queries_total",
        "gpu_prefix_cache_queries_total",
    },
    "prefix_cache_hits_total": {
        "prefix_cache_hits_total",
        "gpu_prefix_cache_hits_total",
    },
}
_SPEC_DECODE_COUNTER_NAMES = {
    "draft_tokens_total": {"spec_decode_num_draft_tokens_total"},
    "accepted_tokens_total": {"spec_decode_num_accepted_tokens_total"},
}
_SPEC_DECODE_PER_POSITION_NAMES = {"spec_decode_num_accepted_tokens_per_pos_total"}


def _bare_metric_name(name: str) -> str:
    """Metric name with its ``vllm:``-style namespace prefix stripped."""
    return name.rpartition(":")[2]


def _serving_snapshot(text: str) -> dict[str, Any]:
    """One scrape's serving-lifecycle gauges, counters, and spec-decode state.

    Speculative-decode counters are matched by the same exact-bare-name
    discipline as the gauges and counters above, against the names the live
    vLLM 0.28.0 engine's own ``/metrics`` scrape was verified to publish:
    ``spec_decode_num_draft_tokens_total``,
    ``spec_decode_num_accepted_tokens_total``, and, labelled by draft
    position, ``spec_decode_num_accepted_tokens_per_pos_total``.
    """
    gauges: dict[str, float] = {}
    counters: dict[str, float] = {}
    spec_counters: dict[str, float] = {}
    spec_per_position: dict[int, float] = {}
    for name, labels, value in _parse_prometheus_text(text):
        bare = _bare_metric_name(name.lower())
        if bare in _SPEC_DECODE_PER_POSITION_NAMES:
            pos = _coerce_int(labels.get("position"))
            if pos is not None:
                spec_per_position[pos] = spec_per_position.get(pos, 0.0) + value
            continue
        matched_spec = False
        for key, names in _SPEC_DECODE_COUNTER_NAMES.items():
            if bare in names:
                spec_counters[key] = spec_counters.get(key, 0.0) + value
                matched_spec = True
                break
        if matched_spec:
            continue
        for key, names in _GAUGE_NAMES.items():
            if bare in names:
                gauges[key] = gauges.get(key, 0.0) + value
                break
        else:
            for key, names in _COUNTER_NAMES.items():
                if bare in names:
                    counters[key] = counters.get(key, 0.0) + value
                    break
    spec_decode = {
        "draft_tokens_total": spec_counters.get("draft_tokens_total"),
        "accepted_tokens_total": spec_counters.get("accepted_tokens_total"),
        "num_accepted_per_pos": (
            [spec_per_position[pos] for pos in sorted(spec_per_position)]
            if spec_per_position
            else None
        ),
    }
    return {
        "gauges": gauges,
        "counters": counters,
        "spec_decode": spec_decode,
    }


@dataclasses.dataclass(frozen=True)
class ReceiptRow:
    """One sample in the append-only serving receipts record."""

    timestamp: str
    job_id: str | None
    profile_slug: str | None
    served_name: str | None
    gpus: int | None
    generation_throughput_toks_per_s: float | None
    prompt_throughput_toks_per_s: float | None
    num_requests_running: int | None
    num_requests_waiting: int | None
    kv_cache_usage_perc: float | None
    prefix_cache_queries_total: int | None
    prefix_cache_hits_total: int | None
    prefix_cache_hit_rate: float | None
    spec_draft_tokens: int | None
    spec_accepted_tokens: int | None
    spec_acceptance_rate: float | None
    spec_num_accepted_per_pos: list[int] | None

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), sort_keys=False)


def _coerce_int(value: Any) -> int | None:
    """Convert *value* to ``int``, rounding a float or parsing a string.

    Used both on scraped gauge values (floats) and on a Prometheus label's
    ``position`` value (a string), so a bare ``round(value)`` is not enough —
    ``round`` rejects a string outright.
    """
    if value is None:
        return None
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _prefix_cache_hit_rate(counters: dict[str, float]) -> float | None:
    """Cumulative prefix-cache hit rate, available from a single scrape.

    Unlike throughput, a hit rate is a ratio of two lifetime counters, so it
    needs no second sample to be meaningful — which is what lets the very
    first row in a run already carry a non-null value.
    """
    queries = counters.get("prefix_cache_queries_total")
    hits = counters.get("prefix_cache_hits_total")
    if not queries or hits is None:
        return None
    return round(hits / queries, 4)


def build_receipt_row(
    previous: dict[str, Any] | None,
    previous_at: _dt.datetime | None,
    current: dict[str, Any],
    current_at: _dt.datetime,
    *,
    job_id: str | None,
    profile_slug: str | None,
    served_name: str | None,
    gpus: int | None,
) -> ReceiptRow:
    """One receipt row from two consecutive ``/metrics`` snapshots.

    Throughput is a rate, so it needs the wall-clock delta between two
    samples; a lone snapshot (the first sample of a run) reports it as
    ``None`` rather than a lifetime average passed off as an instantaneous
    rate. The prefix-cache hit rate carries no such requirement and is
    computed from *current* alone.
    """
    gauges = current["gauges"]
    counters = current["counters"]
    spec = current["spec_decode"]

    gen_tps: float | None = None
    prompt_tps: float | None = None
    if previous is not None and previous_at is not None:
        elapsed = (current_at - previous_at).total_seconds()
        if elapsed > 0:
            gen_delta = _counter_delta(
                previous["counters"], counters, "generation_tokens_total"
            )
            if gen_delta is not None:
                gen_tps = round(gen_delta / elapsed, 4)
            prompt_delta = _counter_delta(
                previous["counters"], counters, "prompt_tokens_total"
            )
            if prompt_delta is not None:
                prompt_tps = round(prompt_delta / elapsed, 4)

    prev_spec = previous["spec_decode"] if previous is not None else None
    draft_delta = (
        _counter_delta(prev_spec, spec, "draft_tokens_total")
        if prev_spec is not None
        else None
    )
    accepted_delta = (
        _counter_delta(prev_spec, spec, "accepted_tokens_total")
        if prev_spec is not None
        else None
    )
    per_pos_delta = (
        _per_position_delta(prev_spec, spec) if prev_spec is not None else None
    )
    acceptance_rate = (
        round(accepted_delta / draft_delta, 4)
        if draft_delta is not None and draft_delta > 0 and accepted_delta is not None
        else None
    )

    return ReceiptRow(
        timestamp=current_at.isoformat(),
        job_id=job_id,
        profile_slug=profile_slug,
        served_name=served_name,
        gpus=gpus,
        generation_throughput_toks_per_s=gen_tps,
        prompt_throughput_toks_per_s=prompt_tps,
        num_requests_running=_coerce_int(gauges.get("num_requests_running")),
        num_requests_waiting=_coerce_int(gauges.get("num_requests_waiting")),
        kv_cache_usage_perc=gauges.get("kv_cache_usage_perc"),
        prefix_cache_queries_total=_coerce_int(
            counters.get("prefix_cache_queries_total")
        ),
        prefix_cache_hits_total=_coerce_int(counters.get("prefix_cache_hits_total")),
        prefix_cache_hit_rate=_prefix_cache_hit_rate(counters),
        spec_draft_tokens=draft_delta,
        spec_accepted_tokens=accepted_delta,
        spec_acceptance_rate=acceptance_rate,
        spec_num_accepted_per_pos=per_pos_delta,
    )


def _utcnow() -> _dt.datetime:
    return _dt.datetime.now(_dt.UTC)


def sample_serving_metrics(
    base_url: str,
    *,
    api_key: str | None = None,
    now: Callable[[], _dt.datetime] = _utcnow,
) -> tuple[dict[str, Any] | None, _dt.datetime]:
    """One scrape of *base_url*'s ``/metrics``.

    The snapshot is ``None`` when the scrape itself failed (engine down,
    route missing, timeout) — a transient miss must not stop the recorder,
    so the caller skips the row rather than raising.
    """
    body = _fetch_body(f"{base_url}/metrics", api_key)
    sampled_at = now()
    if body is None:
        return None, sampled_at
    return _serving_snapshot(body), sampled_at


def record_receipts(
    base_url: str,
    receipts_path: str | Path,
    *,
    interval_s: float = 5.0,
    duration_s: float | None = None,
    api_key: str | None = None,
    profile: ModelProfile | None = None,
    serve_job_id: str | None = None,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    now: Callable[[], _dt.datetime] = _utcnow,
) -> int:
    """Sample *base_url* every *interval_s* and append one row per sample.

    Runs until *duration_s* elapses, or indefinitely when ``None`` — the
    caller (a time-bounded CLI invocation, or a serving job's own lifetime)
    owns the stopping condition either way. At least one sample is always
    taken before the duration is checked, so a ``duration_s`` of zero still
    records a single row rather than none. Rows are appended and flushed one
    at a time so the file is a durable record even if the recorder is killed
    mid-run. Returns the number of rows written.
    """
    path = Path(receipts_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    profile_slug = profile.slug if profile is not None else None
    served_name = profile.model.served_name if profile is not None else None
    gpus = profile.slurm.gpus if profile is not None else None

    previous: dict[str, Any] | None = None
    previous_at: _dt.datetime | None = None
    rows_written = 0
    start = monotonic()

    with path.open("a", encoding="utf-8") as fh:
        while True:
            snapshot, sampled_at = sample_serving_metrics(
                base_url, api_key=api_key, now=now
            )
            if snapshot is not None:
                row = build_receipt_row(
                    previous,
                    previous_at,
                    snapshot,
                    sampled_at,
                    job_id=serve_job_id,
                    profile_slug=profile_slug,
                    served_name=served_name,
                    gpus=gpus,
                )
                fh.write(row.to_json() + "\n")
                fh.flush()
                rows_written += 1
                previous, previous_at = snapshot, sampled_at
            if duration_s is not None and monotonic() - start >= duration_s:
                break
            sleep(interval_s)

    return rows_written
