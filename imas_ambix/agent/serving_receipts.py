"""Continuous serving-time receipts for ``imas-ambix agent receipts``.

:mod:`imas_ambix.agent.engine_metrics` owns which series carries which
quantity for each engine family, and :mod:`imas_ambix.agent.bench` differences
cumulative counters across a window
(:func:`~imas_ambix.agent.bench._counter_delta`). This module reuses it
to build a *continuous* recorder: it samples a serve's ``/metrics`` on an
interval and appends one JSON row per sample to a durable, append-only
receipts file, so a serve's whole life is a readable record rather than
something reconstructed afterwards from a terminal summary.

The row carries a small flat field set and the canonical ``engine`` section.
The flat fields are the established vLLM-shaped vocabulary a saved run and a
reader resolve through, and they are only the ones the engine section states
in a different *shape* rather than a different name: the recorded throughputs
are rates over the sampling interval, where the engine section carries the
cumulative token counters they were differenced from. Prefix-cache and
speculative-decode quantities are read from the engine section, which carries
them directly rather than only as a derived flat column. The ``engine``
section carries only what was observed, so an absent quantity is absent
rather than null or zero.

Beside it the row carries what the node itself measured:
:mod:`imas_ambix.agent.node_probe` supplies the ``cards``, ``host`` and
``jobs`` sections, so a slow hour can be explained from the record rather
than re-derived from SLURM accounting later. Every row also names the host
that produced it, since one receipts directory collects rows from every node
a job ran on.

The row also carries the serve's own configuration in ``serve_config``: the
engine-derived settings that decide what its numbers *mean*, so an occupancy
figure recorded under one topology can be read without guessing the settings
it was measured under. The settings are read from the engine rather than from
the launcher, because a value the engine reports is the one in force while a
launcher's is the one requested; a caller may pass values the engine did not
report, and they fill gaps only.

**A section is omitted rather than nulled.** The node sections and the serve
configuration are sparse in the same way the engine section is: a probe that
failed, is absent, or did not run this tick contributes no key at all, and a
setting the engine did not report is absent from ``serve_config`` rather than
present as a null. A reader therefore never has to tell a measured zero from a
section nothing measured. That rule is enforced at the point of serialization,
which is the only place it can be enforced for a fixed-schema row.

Between samples the recorder also compacts its own record down through
:mod:`imas_ambix.agent.telemetry_store`'s resolution tiers. The recorder is
idle there, the transform is idempotent and never deletes its source, and a
compaction that fails must not stop the recording — so a failed compaction is
dropped and the next tick proceeds, exactly as a failed probe is.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import json
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from imas_ambix.agent import engine_metrics, node_probe, telemetry_store
from imas_ambix.agent.bench import (
    _counter_delta,
    _fetch_body,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from imas_ambix.agent.profile import ModelProfile

# The flat row vocabulary is the vLLM spelling of each canonical quantity, and
# these maps are the whole of that correspondence: a reader of the flat fields
# wants the name it already knows, and the engine section beside it carries the
# canonical name. One direction only — the canonical set is the source, so a
# family that resolves a quantity under every engine spelling produces the same
# flat field the single-family reader always produced. Only the quantities the
# flat row still states a *shape* for appear here; a quantity whose canonical
# spelling the engine section already carries needs no flat mirror.
_LEGACY_GAUGE_NAMES: dict[str, str] = {
    "requests_running": "num_requests_running",
    "requests_queued": "num_requests_waiting",
    "kv_pool_occupancy": "kv_cache_usage_perc",
}
_LEGACY_COUNTER_NAMES: dict[str, str] = {
    "prompt_tokens": "prompt_tokens_total",
    "generation_tokens": "generation_tokens_total",
}

#: Version of the row shape. A reader that ingests old and new files together
#: needs the value on the row, because the flat vocabulary alone cannot say
#: whether the sparse sections below are absent or merely empty. The flat,
#: sectionless rows carry no version; ``2`` is the first shape that has them,
#: ``3`` adds the serve configuration beside them, and ``4`` drops the flat
#: prefix-cache and speculative-decode columns whose quantities the engine
#: section already carries.
ROW_SCHEMA_VERSION = 4

#: The row's sparse sections. A section whose probe did not run is *omitted*
#: from the serialized payload rather than written as ``null``, so a present
#: key always means a measurement and an absent one never reads as a zero.
ROW_SECTIONS: tuple[str, ...] = ("engine", "cards", "host", "jobs", "serve_config")

#: The serve settings a row records, under the names the engine reports them.
#: They are the inputs that give the row's own quantities their meaning: an
#: occupancy figure, a pool total or a decode rate is a different measurement
#: under a different context window, memory fraction or tensor-parallel width,
#: and all of those move between launches of one profile. Held as an ordered
#: tuple so the record states one fixed vocabulary rather than whatever a
#: given engine build happened to expose. The first six names are also the
#: launcher's flag names (``--context-length``, ``--max-total-tokens``, ...);
#: ``tp_size`` is the engine's own spelling of the setting the launch flag
#: calls ``--tensor-parallel-size``.
SERVE_CONFIG_OPTIONS: tuple[str, ...] = (
    "context_length",
    "max_total_tokens",
    "mem_fraction_static",
    "max_running_requests",
    "hicache_ratio",
    "speculative_algorithm",
    "tp_size",
)

#: How often the recorder compacts its own record between samples. The
#: compaction re-reads the whole raw file, so its cadence is minutes rather
#: than ticks; the transform is idempotent, so a missed or failed run costs
#: nothing but the wait until the next one.
DEFAULT_COMPACTION_INTERVAL_S = 300.0


def _serving_snapshot(text: str) -> dict[str, Any]:
    """One scrape as flat serving fields plus the canonical engine section.

    Resolution is delegated to :mod:`imas_ambix.agent.engine_metrics`, so a
    scrape from any engine family populates the same fields; the flat keys are
    then filled from the canonical values under their established names.

    Only the flat fields whose shape differs from the engine section are
    mirrored. The prefix-cache and speculative-decode quantities are read
    straight off the engine section, which carries them under their canonical
    names, and where vLLM's exact-name discipline (a same-named ``_created``
    gauge sibling is a creation timestamp, not a data point) is applied.
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
    return {
        "gauges": gauges,
        "counters": counters,
        "engine": metrics.row_section(),
    }


@dataclasses.dataclass(frozen=True)
class ReceiptRow:
    """One sample in the append-only serving receipts record.

    The sections at the end are sparse: each is ``None`` when its source did
    not produce a reading on this tick, and :meth:`to_json` omits it rather
    than writing a null. The flat fields above keep their established meaning,
    ``None`` included -- they are a fixed vocabulary where a null has always
    said *not observed*, and a reader resolving through them is entitled to
    see the key. Prefix-cache and speculative-decode quantities have no flat
    field: the ``engine`` section carries them under their canonical names,
    and its own rule (absent rather than null) then applies to them.
    """

    schema_version: int
    timestamp: str
    hostname: str
    job_id: str | None
    profile_slug: str | None
    served_name: str | None
    gpus: int | None
    generation_throughput_toks_per_s: float | None
    prompt_throughput_toks_per_s: float | None
    num_requests_running: int | None
    num_requests_waiting: int | None
    kv_cache_usage_perc: float | None
    engine: dict[str, Any] | None = None
    cards: dict[str, Any] | None = None
    host: dict[str, Any] | None = None
    jobs: dict[str, Any] | None = None
    serve_config: dict[str, Any] | None = None

    def to_json(self) -> str:
        payload = dataclasses.asdict(self)
        for section in ROW_SECTIONS:
            # An absent source costs no bytes and can never be mistaken for a
            # measurement, which is the whole point of a sparse section.
            if payload.get(section) is None:
                payload.pop(section, None)
        return json.dumps(payload, sort_keys=False)


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
    hostname: str | None = None,
    node_sections: Mapping[str, Any] | None = None,
    serve_config: Mapping[str, Any] | None = None,
) -> ReceiptRow:
    """One receipt row from two consecutive ``/metrics`` snapshots.

    Throughput is a rate, so it needs the wall-clock delta between two
    samples; a lone snapshot (the first sample of a run) reports it as
    ``None`` rather than a lifetime average passed off as an instantaneous
    rate. The prefix-cache and speculative-decode counters are not differenced
    here: the engine section carries the cumulative values themselves, and a
    reader that wants an interval difference takes it from two rows' engine
    sections.

    *hostname* names the host this row was taken on, defaulting to the local
    node because a recorder only ever writes rows about the machine it runs
    on. *node_sections* is the sparse mapping
    :meth:`~imas_ambix.agent.node_probe.NodeProbe.sample` returned, passed
    through unchanged: a section it did not produce stays absent from the row
    rather than arriving as an empty one. *serve_config* is the settings
    :func:`read_serve_config` resolved, likewise carried as a section that is
    absent rather than empty when nothing was read.
    """
    gauges = current["gauges"]
    counters = current["counters"]

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

    sections = node_sections or {}
    return ReceiptRow(
        schema_version=ROW_SCHEMA_VERSION,
        timestamp=current_at.isoformat(),
        hostname=hostname or local_hostname(),
        job_id=job_id,
        profile_slug=profile_slug,
        served_name=served_name,
        gpus=gpus,
        generation_throughput_toks_per_s=gen_tps,
        prompt_throughput_toks_per_s=prompt_tps,
        num_requests_running=_coerce_int(gauges.get("num_requests_running")),
        num_requests_waiting=_coerce_int(gauges.get("num_requests_waiting")),
        kv_cache_usage_perc=gauges.get("kv_cache_usage_perc"),
        engine=current.get("engine") or None,
        cards=sections.get("cards"),
        host=sections.get("host"),
        jobs=sections.get("jobs"),
        serve_config=dict(serve_config) if serve_config else None,
    )


def local_hostname() -> str:
    """The node this process is running on, as the row should name it."""
    return os.uname().nodename


def read_serve_config(base_url: str, *, api_key: str | None = None) -> dict[str, Any]:
    """The serve settings *base_url*'s engine reports about itself.

    The engine is asked rather than the launcher, because the two answer
    different questions. A launcher states what was *requested* -- and a
    request can be rescaled on the way to the engine, or silently replaced by
    an engine default when its flag is omitted -- while the engine reports
    what is *in force*, which is the value the row's own quantities were
    measured under. Only the settings in :data:`SERVE_CONFIG_OPTIONS` are
    kept, so an engine's full argument dump does not become the row's shape.

    Returns the subset the engine reported, and an empty mapping when the
    route is absent, the engine is down, the body is not JSON, or the engine
    reports none of the settings -- a serve on an engine family that does not
    publish its arguments simply carries no serve configuration. A setting
    the engine did not report is left out rather than written as a null, on
    the same rule the sparse sections follow.
    """
    body = _fetch_body(f"{base_url}/server_info", api_key)
    if body is None:
        return {}
    try:
        payload = json.loads(body)
    except ValueError:
        return {}
    if not isinstance(payload, dict):
        return {}
    return {
        option: payload[option]
        for option in SERVE_CONFIG_OPTIONS
        if payload.get(option) is not None
    }


def _merge_serve_config(
    engine_config: Mapping[str, Any] | None,
    caller_config: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Combine the engine's settings with the caller's, engine winning.

    A caller supplies only what the engine left out, so a value the engine
    reports is never overwritten by one a launcher requested. The result is
    ``None`` when neither source had anything, which is what keeps an unread
    serve configuration absent from the row instead of present and empty.
    """
    merged: dict[str, Any] = {
        option: caller_config[option]
        for option in SERVE_CONFIG_OPTIONS
        if caller_config and caller_config.get(option) is not None
    }
    if engine_config:
        merged.update(engine_config)
    return merged or None


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


def compact_record(
    source: str | Path,
    destination: str | Path,
    *,
    tier: str = telemetry_store.TIER_MINUTE,
) -> bool:
    """Compact *source* into *destination*, reporting failure instead of raising.

    The recorder compacts between samples, so a compaction that cannot finish —
    a path it may not write, a row the store will not parse, a short write on a
    busy filesystem — must cost one attempt and not the record. The sample loop
    is the durable thing here, so every such failure is reported on stderr and
    the next tick proceeds; the transform is idempotent, so the retry at the
    next cadence reaches the same result. Returns whether the tier was written.
    """
    try:
        telemetry_store.compact_file(source, destination, tier=tier)
    except (OSError, telemetry_store.TelemetryStoreError) as exc:
        print(
            f"serving_receipts: compaction of {source} -> {destination} "
            f"at tier {tier!r} failed: {exc}",
            file=sys.stderr,
        )
        return False
    return True


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
    serve_config: Mapping[str, Any] | None = None,
    probe: node_probe.NodeProbe | None = None,
    compaction_paths: tuple[str | Path, str | Path] | None = None,
    compaction_interval_s: float = DEFAULT_COMPACTION_INTERVAL_S,
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

    *serve_config* is the settings a caller passes about the serve, used only
    to fill what the engine did not report: the recorder asks the engine
    first, because a reported value is the one in force. The resolution is
    made once and carried on every later row, since these settings are fixed
    for the life of a serve; a serve whose engine answers but publishes none
    of them settles as carrying no configuration rather than asking again on
    every tick.

    *probe* is the node-side sampler: when given, each tick is offered to
    :meth:`~imas_ambix.agent.node_probe.NodeProbe.sample` and the sections it
    returns are attached to that row, and every row is named with the probe's
    own hostname, which is the node the readings came from. ``None`` records
    the engine and the local host name with no node sections — the reading is
    the caller's to opt into, since only a process actually holding the GPU
    allocation can make them.

    *compaction_paths*, when given, is the ``(minute, hour)`` pair the
    recorder rolls its own record into, at most once per
    *compaction_interval_s*; a failed compaction is reported and skipped
    rather than allowed to end the recording.
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
    next_compaction = start + compaction_interval_s
    resolved_serve_config: dict[str, Any] | None = None
    serve_config_settled = False

    with path.open("a", encoding="utf-8") as fh:
        while True:
            # One clock reading per tick: the probe stamps its own sampling
            # cadence from it, and the duration check below reads the same
            # instant rather than paying for a second call.
            tick = monotonic()
            hostname = local_hostname()
            sections: Mapping[str, Any] = {}
            if probe is not None:
                hostname = probe.hostname
                sections = probe.sample(tick)
            snapshot, sampled_at = sample_serving_metrics(
                base_url, api_key=api_key, now=now
            )
            if not serve_config_settled:
                # Retried until the engine has answered at all: the recorder
                # starts beside the engine and tolerates it not being up yet,
                # and an engine that is up but publishes no settings will
                # never start, so the ask stops there rather than repeating
                # on every tick of a long serve.
                resolved_serve_config = _merge_serve_config(
                    read_serve_config(base_url, api_key=api_key), serve_config
                )
                if resolved_serve_config is not None or snapshot is not None:
                    serve_config_settled = True
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
                    hostname=hostname,
                    node_sections=sections,
                    serve_config=resolved_serve_config,
                )
                fh.write(row.to_json() + "\n")
                fh.flush()
                rows_written += 1
                previous, previous_at = snapshot, sampled_at
            if compaction_paths is not None and tick >= next_compaction:
                next_compaction = tick + compaction_interval_s
                minute_path, hour_path = compaction_paths
                compact_record(path, minute_path, tier=telemetry_store.TIER_MINUTE)
                compact_record(minute_path, hour_path, tier=telemetry_store.TIER_HOUR)
            if duration_s is not None and tick - start >= duration_s:
                break
            sleep(interval_s)

    return rows_written


def tier_paths(receipts_path: str | Path) -> tuple[Path, Path]:
    """The ``(minute, hour)`` tiers a receipts file rolls into.

    The tiers are siblings of the raw file, named between its stem and its
    suffix, so one receipts directory holds a serve's whole retention ladder
    and a reader can find the coarser resolution from the file it already has.
    """
    path = Path(receipts_path)
    suffix = "".join(path.suffixes)
    stem = path.name[: -len(suffix)] if suffix else path.name
    return (
        path.with_name(f"{stem}.minute{suffix or '.jsonl'}"),
        path.with_name(f"{stem}.hour{suffix or '.jsonl'}"),
    )


def _coerce_setting(raw: str) -> Any:
    """A ``--serve-setting`` value as a number when it reads as one.

    The engine reports these as JSON numbers and strings; a caller stating the
    same setting on a command line states it as text, and a numeric setting
    written as a string would make two rows for one topology compare unequal.
    """
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def _parse_serve_settings(
    pairs: Sequence[str] | None, *, parser: argparse.ArgumentParser
) -> dict[str, Any] | None:
    """The ``--serve-setting`` arguments as a mapping, or ``None`` when unset.

    An unknown key is refused rather than dropped: a mistyped setting that
    silently contributes nothing would leave a row claiming a configuration
    it does not carry.
    """
    if not pairs:
        return None
    settings: dict[str, Any] = {}
    for pair in pairs:
        key, separator, raw = pair.partition("=")
        if not separator or key not in SERVE_CONFIG_OPTIONS:
            parser.error(
                "--serve-setting expects KEY=VALUE with KEY one of "
                f"{', '.join(SERVE_CONFIG_OPTIONS)}; got {pair!r}"
            )
        settings[key] = _coerce_setting(raw)
    return settings


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sample a live engine's /metrics and append receipt rows"
    )
    parser.add_argument("--base-url")
    parser.add_argument("--receipts-path")
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--profile-slug", default=None)
    parser.add_argument("--served-name", default=None)
    parser.add_argument("--gpus", type=int, default=None)
    parser.add_argument(
        "--serve-setting",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help=(
            "A serve setting the engine does not report about itself; "
            "repeatable, and used only where the engine reports nothing. "
            f"KEY is one of {', '.join(SERVE_CONFIG_OPTIONS)}"
        ),
    )
    parser.add_argument(
        "--no-node-probe",
        action="store_true",
        help="Record the engine alone, without the card, host and job sections",
    )
    parser.add_argument(
        "--no-compaction",
        action="store_true",
        help="Leave the raw record uncompacted for this run",
    )
    parser.add_argument(
        "--compaction-interval",
        type=float,
        default=DEFAULT_COMPACTION_INTERVAL_S,
    )
    parser.add_argument(
        "--compact-tier",
        choices=sorted(telemetry_store.TIER_WINDOW_SECONDS),
        default=None,
        help="Rebuild this tier from --compact-source and exit",
    )
    parser.add_argument("--compact-source", default=None)
    parser.add_argument("--compact-destination", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the recorder as a standalone process.

    This is what a generated serve script's background sidecar invokes
    (``python -m imas_ambix.agent.serving_receipts``) — it knows the
    engine's own URL, the job id, and the profile identity directly from
    the script that launched it, with no scheduler lookup of its own. It
    also owns the node probe, because the sidecar runs inside the job's own
    allocation: that is the only place the card, host and job readings are
    free, and the row names the host it got them from. Serve settings come
    from the engine's own report; ``--serve-setting`` fills only what that
    engine does not report about itself.

    ``--compact-tier`` selects the other mode: rebuild one resolution tier
    from a source file and exit, which is how a record is caught up after
    the serve has stopped. It delegates to
    :mod:`imas_ambix.agent.telemetry_store`, whose tier semantics
    (endpoint counters, weighted-mean gauges, source never retired) are
    defined there rather than restated here.
    """
    args = _parser().parse_args(argv)

    if args.compact_tier is not None:
        if not (args.compact_source and args.compact_destination):
            _parser().error(
                "--compact-tier needs --compact-source and --compact-destination"
            )
        count = telemetry_store.compact_file(
            args.compact_source, args.compact_destination, tier=args.compact_tier
        )
        print(f"{args.compact_destination}: {count} rows")
        return 0

    if not (args.base_url and args.receipts_path):
        _parser().error("--base-url and --receipts-path are required")

    probe = None if args.no_node_probe else node_probe.NodeProbe()
    compaction_paths = None if args.no_compaction else tier_paths(args.receipts_path)

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
        serve_config=_parse_serve_settings(args.serve_setting, parser=_parser()),
        probe=probe,
        compaction_paths=compaction_paths,
        compaction_interval_s=args.compaction_interval,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
