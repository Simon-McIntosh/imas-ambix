"""Continuous serving-time receipts for ``imas-ambix agent receipts``.

:mod:`imas_ambix.agent.engine_metrics` owns which series carries which
quantity for each engine family, and :mod:`imas_ambix.agent.bench` differences
cumulative counters across a window
(:func:`~imas_ambix.agent.bench._counter_delta`,
:func:`~imas_ambix.agent.bench._per_position_delta`). This module reuses both
to build a *continuous* recorder: it samples a serve's ``/metrics`` on an
interval and appends one JSON row per sample to a durable, append-only
receipts file, so a serve's whole life is a readable record rather than
something reconstructed afterwards from a terminal summary.

The row carries two views of one scrape. The flat fields are the established
vLLM-shaped vocabulary, kept because saved runs and readers resolve through
them; the ``engine`` section is the canonical family-agnostic set, and it is
where the quantities that flat vocabulary has no home for live — the cached
prompt tokens split by the tier that answered them, the uncached remainder,
and both latency histograms. The ``engine`` section carries only what was
observed, so an absent quantity is absent rather than null or zero.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from imas_ambix.agent import engine_metrics
from imas_ambix.agent.bench import (
    _counter_delta,
    _fetch_body,
    _per_position_delta,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from imas_ambix.agent.profile import ModelProfile

# The flat row vocabulary is the vLLM spelling of each canonical quantity, and
# these maps are the whole of that correspondence: a reader of the flat fields
# wants the name it already knows, and the engine section beside it carries the
# canonical name. One direction only — the canonical set is the source, so a
# family that resolves a quantity under every engine spelling produces the same
# flat field the single-family reader always produced.
_LEGACY_GAUGE_NAMES: dict[str, str] = {
    "requests_running": "num_requests_running",
    "requests_queued": "num_requests_waiting",
    "kv_pool_occupancy": "kv_cache_usage_perc",
}
_LEGACY_COUNTER_NAMES: dict[str, str] = {
    "prompt_tokens": "prompt_tokens_total",
    "generation_tokens": "generation_tokens_total",
    "prefix_cache_queries": "prefix_cache_queries_total",
    "prefix_cache_hits": "prefix_cache_hits_total",
}


def _serving_snapshot(text: str) -> dict[str, Any]:
    """One scrape as flat serving fields plus the canonical engine section.

    Resolution is delegated to :mod:`imas_ambix.agent.engine_metrics`, so a
    scrape from any engine family populates the same fields; the flat keys are
    then filled from the canonical values under their established names.

    Speculative-decode counters keep their exact-name discipline there too:
    vLLM pairs every counter with a same-named ``_created`` gauge — its
    creation timestamp, not a data point — and a substring test folds that
    sibling into the token total (measured on the live four-card engine).
    """
    metrics = engine_metrics.read_metrics(text)
    gauges = {
        legacy: metrics.gauges[canon]
        for canon, legacy in _LEGACY_GAUGE_NAMES.items()
        if canon in metrics.gauges
    }
    counters = {
        legacy: metrics.counters[canon]
        for canon, legacy in _LEGACY_COUNTER_NAMES.items()
        if canon in metrics.counters
    }
    spec_decode = {
        "draft_tokens_total": metrics.spec_decode.get("draft_tokens_total"),
        "accepted_tokens_total": metrics.spec_decode.get("accepted_tokens_total"),
        "num_accepted_per_pos": metrics.spec_decode.get("num_accepted_per_pos"),
    }
    return {
        "gauges": gauges,
        "counters": counters,
        "spec_decode": spec_decode,
        "engine": metrics.row_section(),
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
    prefix_cache_query_delta: int | None
    prefix_cache_hit_delta: int | None
    prefix_cache_hit_rate_interval: float | None
    prefix_cache_hit_rate: float | None
    spec_draft_tokens: int | None
    spec_accepted_tokens: int | None
    spec_acceptance_rate: float | None
    spec_num_accepted_per_pos: list[int] | None
    engine: dict[str, Any] = dataclasses.field(default_factory=dict)

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
    computed from *current* alone. Prefix-cache interval fields, by contrast,
    require the preceding snapshot and remain ``None`` on the first row.
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

    prev_counters = previous["counters"] if previous is not None else None
    prefix_cache_query_delta = (
        _counter_delta(prev_counters, counters, "prefix_cache_queries_total")
        if prev_counters is not None
        else None
    )
    prefix_cache_hit_delta = (
        _counter_delta(prev_counters, counters, "prefix_cache_hits_total")
        if prev_counters is not None
        else None
    )
    prefix_cache_hit_rate_interval = (
        round(prefix_cache_hit_delta / prefix_cache_query_delta, 4)
        if prefix_cache_query_delta is not None
        and prefix_cache_query_delta > 0
        and prefix_cache_hit_delta is not None
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
        prefix_cache_query_delta=prefix_cache_query_delta,
        prefix_cache_hit_delta=prefix_cache_hit_delta,
        prefix_cache_hit_rate_interval=prefix_cache_hit_rate_interval,
        prefix_cache_hit_rate=_prefix_cache_hit_rate(counters),
        spec_draft_tokens=draft_delta,
        spec_accepted_tokens=accepted_delta,
        spec_acceptance_rate=acceptance_rate,
        spec_num_accepted_per_pos=per_pos_delta,
        engine=current.get("engine", {}),
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
    profile_slug: str | None = None,
    served_name: str | None = None,
    gpus: int | None = None,
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

    *profile_slug*, *served_name*, and *gpus* label each row when no
    :class:`~imas_ambix.agent.profile.ModelProfile` is available — a
    generated serve script's own sidecar invocation knows these values
    directly and has no reason to reconstruct a profile object for them.
    *profile*, when given, takes precedence over the discrete fields.
    """
    path = Path(receipts_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if profile is not None:
        profile_slug = profile.slug
        served_name = profile.model.served_name
        gpus = profile.slurm.gpus

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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sample a live engine's /metrics and append receipt rows"
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--receipts-path", required=True)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--profile-slug", default=None)
    parser.add_argument("--served-name", default=None)
    parser.add_argument("--gpus", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the recorder as a standalone process.

    This is what a generated serve script's background sidecar invokes
    (``python -m imas_ambix.agent.serving_receipts``) — it knows the
    engine's own URL, the job id, and the profile identity directly from
    the script that launched it, with no scheduler lookup of its own.
    """
    args = _parser().parse_args(argv)
    record_receipts(
        args.base_url,
        args.receipts_path,
        interval_s=args.interval,
        duration_s=args.duration,
        api_key=args.api_key,
        serve_job_id=args.job_id,
        profile_slug=args.profile_slug,
        served_name=args.served_name,
        gpus=args.gpus,
    )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
