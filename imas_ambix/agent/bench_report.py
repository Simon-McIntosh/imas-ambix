"""Read, summarise and compare saved ``imas-ambix agent bench`` runs.

``agent bench`` writes one JSON document per run under
``~/.local/share/ambix/bench/<model>_<timestamp>.json``. This module
turns those documents into comparable headline figures so two node
sizes, two precisions, or two engine versions can be placed side by
side.

The module is deliberately inert: it performs no HTTP, runs no
benchmark, and owns no CLI entry point. Everything is a pure function
over already-saved reports, plus two ``rich`` renderers.

Older saved runs predate the serving-configuration provenance block, so
``server_info["provenance"]`` and ``server_info["spec_decode"]`` are
treated as optional throughout. A field that is not recorded renders as
:data:`UNKNOWN`; it is never guessed and never substituted with a
plausible-looking value. Likewise a percentage change against a zero or
absent baseline renders as :data:`NOT_COMPARABLE` rather than an
infinity or a NaN.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
import statistics
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterable, Sequence

    from rich.console import Console

__all__ = [
    "NOT_COMPARABLE",
    "PROVENANCE_FIELDS",
    "UNKNOWN",
    "BenchReportError",
    "compare_runs",
    "default_bench_dir",
    "describe_run",
    "discover_reports",
    "load_report",
    "render_comparison",
    "render_run",
]

# ── Markers and field registries ────────────────────────────────────

#: Rendered in place of a figure the saved run does not record.
UNKNOWN = "unknown"

#: Rendered in place of a percentage change with no usable baseline.
NOT_COMPARABLE = "n/a"

#: Serving-configuration fields, in the order they are displayed.
#: A run saved before provenance was recorded carries none of them.
PROVENANCE_FIELDS: tuple[str, ...] = (
    "profile_slug",
    "model_name",
    "served_name",
    "engine_type",
    "engine_version",
    "tensor_parallel",
    "gpus",
    "kv_cache_dtype",
    "max_model_len",
    "max_num_seqs",
    "speculative_method",
    "speculative_num_tokens",
    "quantization",
    "slurm_job_id",
    "gpu_host",
    "captured_at",
)

#: Draft-token accounting fields of the speculative-decoding block.
_SPEC_DECODE_FIELDS: tuple[str, ...] = (
    "draft_tokens_total",
    "accepted_tokens_total",
    "acceptance_rate",
    "num_accepted_per_pos",
    "source",
)

#: Display order for categories; anything else sorts after, by name.
_CATEGORY_ORDER: tuple[str, ...] = (
    "throughput",
    "prefill",
    "context",
    "tools",
    "reasoning",
    "concurrency",
)

_OK_STATUSES = frozenset({"passed", "skipped"})

#: Per-test metrics compared for each category: the key in the test
#: index, a display label, a format, whether larger is better, and the
#: factor taking the stored unit to the displayed one.
_TEST_METRICS: dict[str, tuple[tuple[str, str, str, bool, float], ...]] = {
    "prefill": (
        ("prefill_tps", "prefill tok/s", "{:.0f}", True, 1.0),
        ("ttft_s", "ttft ms", "{:.0f}", False, 1000.0),
    ),
    "concurrency": (
        ("aggregate_tps", "aggregate tok/s", "{:.1f}", True, 1.0),
        ("decode_tps", "per-stream tok/s", "{:.1f}", True, 1.0),
    ),
    # Tool calls are validated, not timed: the non-streaming request
    # records no decode rate, so the repeat count is the whole signal.
    "tools": (("passed", "repeats passed", "{:.0f}", True, 1.0),),
}

_DEFAULT_TEST_METRICS: tuple[tuple[str, str, str, bool, float], ...] = (
    ("decode_tps", "decode tok/s", "{:.1f}", True, 1.0),
)

#: Headline figures compared for the run as a whole.
_HEADLINE_METRICS: tuple[tuple[str, str, str, bool | None, float], ...] = (
    ("decode tok/s (median, single-stream)", "decode_tps", "{:.1f}", True, 1.0),
    ("ttft ms (median)", "ttft_median_s", "{:.0f}", False, 1000.0),
    ("ttft ms (p95)", "ttft_p95_s", "{:.0f}", False, 1000.0),
    ("peak aggregate tok/s", "peak_aggregate_tps", "{:.1f}", True, 1.0),
    ("peak aggregate at workers", "peak_aggregate_workers", "{:.0f}", None, 1.0),
    ("draft-token acceptance %", "acceptance_rate", "{:.1f}", True, 100.0),
    ("tests passed", "passed_count", "{:.0f}", True, 1.0),
    ("tests failed", "failed_count", "{:.0f}", False, 1.0),
)

_SIZE_PATTERN = re.compile(r"(\d+)([kKmM]?)")
_STAMP_PATTERN = re.compile(r"(\d{8})_(\d{6})")


class BenchReportError(ValueError):
    """A saved benchmark report is missing, unreadable, or malformed."""


# ── Loading and discovery ───────────────────────────────────────────


def default_bench_dir() -> Path:
    """Return the directory ``agent bench`` saves its runs into."""
    return Path.home() / ".local" / "share" / "ambix" / "bench"


def load_report(path: str | Path) -> dict[str, Any]:
    """Load and validate one saved benchmark run.

    Returns the parsed document with ``source_path`` stamped in, so a
    later summary can name where the figures came from. Every failure
    mode raises :class:`BenchReportError` naming the offending file.
    """
    report_path = Path(path)
    try:
        raw = report_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise BenchReportError(f"{report_path}: no such benchmark report") from exc
    except OSError as exc:
        raise BenchReportError(f"{report_path}: cannot be read ({exc})") from exc

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise BenchReportError(f"{report_path}: not valid JSON ({exc})") from exc

    if not isinstance(parsed, dict):
        kind = type(parsed).__name__
        raise BenchReportError(
            f"{report_path}: expected a JSON object at the top level, got {kind}"
        )

    results = parsed.get("results")
    if not isinstance(results, list):
        raise BenchReportError(
            f"{report_path}: missing the 'results' list — not a benchmark report"
        )
    for index, record in enumerate(results):
        if not isinstance(record, dict):
            kind = type(record).__name__
            raise BenchReportError(
                f"{report_path}: results[{index}] is a {kind}, expected an object"
            )

    parsed.setdefault("source_path", str(report_path))
    return parsed


def discover_reports(
    directory: str | Path | None = None,
    model: str | None = None,
    limit: int | None = None,
) -> list[Path]:
    """List saved runs newest-first, optionally filtered by model.

    *model* is matched as a case-insensitive substring of the file name,
    which carries the served model. A missing directory yields an empty
    list rather than an error — nothing has been benchmarked yet.
    """
    root = Path(directory) if directory is not None else default_bench_dir()
    if not root.is_dir():
        return []

    needle = model.lower() if model else None
    candidates: list[tuple[float, str, Path]] = []
    for entry in root.glob("*.json"):
        if not entry.is_file():
            continue
        if needle is not None and needle not in entry.name.lower():
            continue
        candidates.append((_saved_at(entry), entry.name, entry))

    # Newest first; the name breaks ties so the order is deterministic.
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    paths = [entry for _stamp, _name, entry in candidates]
    if limit is not None and limit >= 0:
        paths = paths[:limit]
    return paths


def _saved_at(path: Path) -> float:
    """Return a sortable save time for *path*, preferring its stamp."""
    match = _STAMP_PATTERN.search(path.stem)
    if match:
        try:
            stamp = _dt.datetime.strptime(
                f"{match.group(1)}{match.group(2)}", "%Y%m%d%H%M%S"
            )
        except ValueError:
            pass
        else:
            return stamp.timestamp()
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


# ── Numeric helpers (every one of them divide-by-zero safe) ─────────


def _as_float(value: Any) -> float | None:
    """Coerce *value* to a float, or return ``None`` if it is not one."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _positive(values: Iterable[Any]) -> list[float]:
    """Keep only the strictly positive numbers in *values*."""
    kept: list[float] = []
    for value in values:
        number = _as_float(value)
        if number is not None and number > 0:
            kept.append(number)
    return kept


def _median(values: Sequence[float]) -> float | None:
    """Median of *values*, or ``None`` when there is nothing to take."""
    if not values:
        return None
    return float(statistics.median(values))


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    """Nearest-rank percentile, matching the saved report's convention."""
    if not values:
        return None
    ordered = sorted(values)
    n = len(ordered)
    return float(ordered[min(int(n * quantile), n - 1)])


def _pct_change(baseline: float | None, value: float | None) -> float | None:
    """Percentage change from *baseline* to *value*.

    Returns ``None`` — rendered as :data:`NOT_COMPARABLE` — when either
    figure is absent or the baseline is zero, so a comparison never
    produces an infinity or a NaN.
    """
    if baseline is None or value is None or baseline == 0:
        return None
    return (value - baseline) / baseline * 100.0


def _normalise_acceptance(value: Any) -> float | None:
    """Return a draft-token acceptance rate as a 0-1 fraction.

    The block may record either a fraction or a percentage; anything
    above one is read as a percentage.
    """
    rate = _as_float(value)
    if rate is None or rate < 0:
        return None
    if rate > 1.0:
        return rate / 100.0 if rate <= 100.0 else None
    return rate


def _numeric_rank(name: str) -> tuple[float, str]:
    """Rank a test name by the token size it carries, then by name.

    Keeps ``prefill_4k`` before ``prefill_16k`` and, in the concurrency
    table, ``concurrent_8`` before ``concurrent_32`` — a lexical sort
    gets both pairs the wrong way round.
    """
    match = _SIZE_PATTERN.search(name)
    if not match:
        return (float("inf"), name)
    size = float(match.group(1))
    suffix = match.group(2).lower()
    if suffix == "k":
        size *= 1_000.0
    elif suffix == "m":
        size *= 1_000_000.0
    return (size, name)


def _category_rank(category: str) -> tuple[int, str]:
    """Rank a category by the benchmark's own running order."""
    if category in _CATEGORY_ORDER:
        return (_CATEGORY_ORDER.index(category), "")
    return (len(_CATEGORY_ORDER), category)


# ── Flattening one run ──────────────────────────────────────────────


def _records(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the per-request records of *report*, skipping junk."""
    results = report.get("results")
    if not isinstance(results, list):
        return []
    return [record for record in results if isinstance(record, dict)]


def _server_info(report: dict[str, Any]) -> dict[str, Any]:
    """Return the ``server_info`` block, tolerating its absence."""
    info = report.get("server_info")
    return info if isinstance(info, dict) else {}


def _served_model(report: dict[str, Any]) -> str:
    """Name the model the run exercised, from the best source available."""
    info = _server_info(report)
    provenance = info.get("provenance")
    if isinstance(provenance, dict):
        for key in ("served_name", "model_name"):
            value = provenance.get(key)
            if value:
                return str(value)
    data = info.get("data")
    if isinstance(data, list) and data and isinstance(data[0], dict):
        identifier = data[0].get("id")
        if identifier:
            return str(identifier)
    for record in _records(report):
        model = record.get("model")
        if model:
            return str(model)
    return UNKNOWN


def _provenance(report: dict[str, Any]) -> dict[str, Any]:
    """Return every provenance field, with unrecorded ones marked."""
    block = _server_info(report).get("provenance")
    recorded = block if isinstance(block, dict) else {}
    out: dict[str, Any] = {}
    for field in PROVENANCE_FIELDS:
        value = recorded.get(field)
        out[field] = UNKNOWN if value in (None, "") else value
    return out


def _spec_decode(report: dict[str, Any]) -> dict[str, Any]:
    """Return the draft-token block, with unrecorded fields marked."""
    block = _server_info(report).get("spec_decode")
    recorded = block if isinstance(block, dict) else {}
    out: dict[str, Any] = {}
    for field in _SPEC_DECODE_FIELDS:
        value = recorded.get(field)
        out[field] = UNKNOWN if value in (None, "") else value
    return out


def _acceptance_rate(report: dict[str, Any]) -> float | None:
    """Draft-token acceptance as a fraction, derived if not recorded."""
    block = _server_info(report).get("spec_decode")
    if not isinstance(block, dict):
        return None
    rate = _normalise_acceptance(block.get("acceptance_rate"))
    if rate is not None:
        return rate
    drafted = _as_float(block.get("draft_tokens_total"))
    accepted = _as_float(block.get("accepted_tokens_total"))
    if drafted is None or accepted is None or drafted <= 0:
        return None
    return accepted / drafted


def _status_counts(records: Sequence[dict[str, Any]]) -> dict[str, int]:
    """Count records by status, recomputed rather than trusted."""
    counts = {"passed": 0, "failed": 0, "skipped": 0, "error": 0}
    for record in records:
        status = str(record.get("status", ""))
        if status in counts:
            counts[status] += 1
    counts["total"] = len(records)
    return counts


def _index_tests(report: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    """Collapse repeats into one comparable entry per test.

    Keyed by ``(category, test_name)``; latency and rate figures are
    medians over the passing repeats, so one slow outlier does not move
    the comparison.
    """
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in _records(report):
        key = (str(record.get("category", "")), str(record.get("test_name", "")))
        buckets.setdefault(key, []).append(record)

    index: dict[tuple[str, str], dict[str, Any]] = {}
    for key, group in buckets.items():
        passing = [r for r in group if str(r.get("status", "")) in _OK_STATUSES]
        metadata = group[0].get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        workers = _as_float(metadata.get("n_workers"))
        if workers is None:
            size, _name = _numeric_rank(key[1])
            workers = size if size != float("inf") else None
        index[key] = {
            "category": key[0],
            "test_name": key[1],
            "decode_tps": _median(_positive(r.get("decode_tps") for r in passing)),
            "ttft_s": _median(
                _positive(r.get("time_to_first_token_s") for r in passing)
            ),
            "prefill_tps": _median(_positive(r.get("prefill_tps") for r in passing)),
            "total_time_s": _median(_positive(r.get("total_time_s") for r in passing)),
            "aggregate_tps": _median(_positive([metadata.get("aggregate_tps")])),
            "wall_time_s": _median(_positive([metadata.get("wall_time")])),
            "n_workers": int(workers) if workers is not None else None,
            "records": len(group),
            "passed": sum(1 for r in group if r.get("status") == "passed"),
        }
    return index


def _categories(report: dict[str, Any], index: dict[tuple[str, str], Any]) -> list[str]:
    """Categories the run actually produced records for."""
    seen = {category for category, _test in index}
    declared = report.get("categories_run")
    if isinstance(declared, list):
        seen.update(str(item) for item in declared if item)
    return sorted(seen, key=_category_rank)


def _run_label(report: dict[str, Any], provenance: dict[str, Any], served: str) -> str:
    """Build a short column heading identifying the run."""
    slug = provenance.get("profile_slug")
    name = str(slug) if slug and slug != UNKNOWN else served
    stamp = _format_stamp(report)
    return f"{name} @ {stamp}" if stamp else name


def _format_stamp(report: dict[str, Any]) -> str:
    """Format the run timestamp to the minute, tolerating any shape."""
    raw = report.get("timestamp")
    if isinstance(raw, str) and raw:
        try:
            moment = _dt.datetime.fromisoformat(raw)
        except ValueError:
            return raw[:16]
        return moment.strftime("%Y-%m-%d %H:%M")
    source = report.get("source_path")
    if isinstance(source, str):
        match = _STAMP_PATTERN.search(Path(source).stem)
        if match:
            return f"{match.group(1)}_{match.group(2)}"
    return ""


def describe_run(report: dict[str, Any]) -> dict[str, Any]:
    """Flatten one saved run into comparable headline figures.

    Covers the served model, the provenance fields that identify the
    serving configuration, the single-stream decode rate, median and
    p95 time to first token, the prefill rate at each input size, the
    peak aggregate rate with the concurrency that produced it, the
    draft-token acceptance rate, and the pass/fail counts. Anything the
    run does not record comes back as ``None`` or :data:`UNKNOWN`.

    *report* is read, never modified.
    """
    records = _records(report)
    index = _index_tests(report)
    provenance = _provenance(report)
    served = _served_model(report)

    passing = [r for r in records if str(r.get("status", "")) in _OK_STATUSES]
    # Concurrency records share the machine, so they say nothing about
    # the single-stream rate or about latency under no load.
    solo = [r for r in passing if r.get("category") != "concurrency"]
    ttft_values = _positive(r.get("time_to_first_token_s") for r in solo)
    # The throughput category exists to measure the single-stream decode
    # rate; fall back to the rest only when it was not run.
    decode_values = _positive(
        r.get("decode_tps") for r in solo if r.get("category") == "throughput"
    ) or _positive(r.get("decode_tps") for r in solo)

    prefill: dict[str, float] = {}
    for (category, test_name), entry in sorted(
        index.items(), key=lambda item: _numeric_rank(item[0][1])
    ):
        if category == "prefill" and entry["prefill_tps"] is not None:
            prefill[test_name] = entry["prefill_tps"]

    concurrency: list[dict[str, Any]] = []
    for (category, _test_name), entry in index.items():
        if category != "concurrency":
            continue
        concurrency.append(
            {
                "workers": entry["n_workers"],
                "aggregate_tps": entry["aggregate_tps"],
                "per_stream_tps": entry["decode_tps"],
                "wall_time_s": entry["wall_time_s"],
            }
        )
    # Numeric order: a lexical sort would put 32 workers before 8.
    concurrency.sort(key=lambda item: (item["workers"] is None, item["workers"] or 0))

    peak_workers: int | None = None
    peak_aggregate: float | None = None
    for level in concurrency:
        rate = level["aggregate_tps"]
        if rate is not None and (peak_aggregate is None or rate > peak_aggregate):
            peak_aggregate = rate
            peak_workers = level["workers"]

    counts = _status_counts(records)
    return {
        "label": _run_label(report, provenance, served),
        "source_path": report.get("source_path"),
        "timestamp": report.get("timestamp", ""),
        "served_model": served,
        "categories": _categories(report, index),
        "provenance": provenance,
        "has_provenance": any(value != UNKNOWN for value in provenance.values()),
        "spec_decode": _spec_decode(report),
        "acceptance_rate": _acceptance_rate(report),
        "decode_tps": _median(decode_values),
        "ttft_median_s": _median(ttft_values),
        "ttft_p95_s": _percentile(ttft_values, 0.95),
        "prefill_tps": prefill,
        "concurrency": concurrency,
        "peak_aggregate_tps": peak_aggregate,
        "peak_aggregate_workers": peak_workers,
        "status_counts": counts,
        "passed_count": counts["passed"],
        "failed_count": counts["failed"],
        "tests": index,
    }


# ── Comparing runs ──────────────────────────────────────────────────


def _unique_labels(runs: Sequence[dict[str, Any]]) -> list[str]:
    """Make the run labels distinct so table columns stay readable."""
    labels: list[str] = []
    seen: dict[str, int] = {}
    for run in runs:
        base = str(run["label"])
        seen[base] = seen.get(base, 0) + 1
        labels.append(base if seen[base] == 1 else f"{base} #{seen[base]}")
    return labels


def _metric_row(
    metric: str,
    fmt: str,
    higher_is_better: bool | None,
    values: Sequence[float | None],
) -> dict[str, Any]:
    """Build one comparison row: absolute values, deltas, percentages."""
    baseline = values[0]
    deltas: list[float | None] = [None]
    pcts: list[float | None] = [None]
    for value in values[1:]:
        if baseline is None or value is None:
            deltas.append(None)
            pcts.append(None)
            continue
        deltas.append(value - baseline)
        pcts.append(_pct_change(baseline, value))
    return {
        "metric": metric,
        "format": fmt,
        "higher_is_better": higher_is_better,
        "values": list(values),
        "deltas": deltas,
        "pct": pcts,
    }


def _scaled(value: Any, scale: float) -> float | None:
    """Convert a stored figure into its display unit."""
    number = _as_float(value)
    return None if number is None else number * scale


def compare_runs(reports: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Align two or more runs and compute per-metric deltas.

    The first run is the baseline every delta and percentage is taken
    against. Runs whose category sets differ are compared on the
    intersection only, and everything left out is reported in
    ``excluded_categories`` / ``excluded_tests`` rather than silently
    dropped.

    The reports are read, never modified.
    """
    if len(reports) < 2:
        raise BenchReportError(
            f"compare_runs needs at least two runs, got {len(reports)}"
        )

    runs = [describe_run(report) for report in reports]
    labels = _unique_labels(runs)

    category_sets = [set(run["categories"]) for run in runs]
    common_categories = set.intersection(*category_sets)
    all_categories = set.union(*category_sets)
    excluded_categories = [
        {
            "category": category,
            "present_in": [
                label
                for label, owned in zip(labels, category_sets, strict=True)
                if category in owned
            ],
        }
        for category in sorted(all_categories - common_categories, key=_category_rank)
    ]

    test_sets = [set(run["tests"]) for run in runs]
    common_tests = set.intersection(*test_sets)
    common_tests = {key for key in common_tests if key[0] in common_categories}
    all_tests = set.union(*test_sets)
    excluded_tests = [
        {
            "category": category,
            "test_name": test_name,
            "present_in": [
                label
                for label, owned in zip(labels, test_sets, strict=True)
                if (category, test_name) in owned
            ],
        }
        for category, test_name in sorted(
            all_tests - common_tests,
            key=lambda key: (_category_rank(key[0]), _numeric_rank(key[1])),
        )
    ]

    # Identity first, then the provenance block; a run that predates
    # provenance still shows its model and when it ran.
    config_columns: list[tuple[str, list[Any]]] = [
        ("run timestamp", [_format_stamp_from_run(run) for run in runs]),
        ("served model", [run["served_model"] for run in runs]),
    ]
    config_columns.extend(
        (field, [run["provenance"][field] for run in runs])
        for field in PROVENANCE_FIELDS
    )
    config_rows: list[dict[str, Any]] = [
        {
            "field": field,
            "values": values,
            "differs": len({str(value) for value in values}) > 1,
        }
        for field, values in config_columns
    ]

    metric_rows: list[dict[str, Any]] = [
        _metric_row(
            metric,
            fmt,
            higher_is_better,
            [_scaled(run.get(key), scale) for run in runs],
        )
        for metric, key, fmt, higher_is_better, scale in _HEADLINE_METRICS
    ]

    for category, test_name in sorted(
        common_tests,
        key=lambda key: (_category_rank(key[0]), _numeric_rank(key[1])),
    ):
        specs = _TEST_METRICS.get(category, _DEFAULT_TEST_METRICS)
        for key, label, fmt, higher_is_better, scale in specs:
            values = [
                _scaled(run["tests"][(category, test_name)].get(key), scale)
                for run in runs
            ]
            if all(value is None for value in values):
                continue
            metric_rows.append(
                _metric_row(
                    f"{category}/{test_name} {label}", fmt, higher_is_better, values
                )
            )

    return {
        "runs": runs,
        "labels": labels,
        "baseline_label": labels[0],
        "config_rows": config_rows,
        "metric_rows": metric_rows,
        "common_categories": sorted(common_categories, key=_category_rank),
        "excluded_categories": excluded_categories,
        "common_tests": [
            {"category": category, "test_name": test_name}
            for category, test_name in sorted(
                common_tests,
                key=lambda key: (_category_rank(key[0]), _numeric_rank(key[1])),
            )
        ],
        "excluded_tests": excluded_tests,
    }


def _format_stamp_from_run(run: dict[str, Any]) -> str:
    """Format a described run's timestamp for the configuration table."""
    stamp = _format_stamp(
        {"timestamp": run.get("timestamp"), "source_path": run.get("source_path")}
    )
    return stamp or UNKNOWN


# ── Rendering ───────────────────────────────────────────────────────


def _cell(value: Any) -> str:
    """Render a configuration value for a table cell."""
    if value is None:
        return UNKNOWN
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, list | tuple):
        return ", ".join(str(item) for item in value) if value else UNKNOWN
    return str(value)


def _value_cell(value: float | None, fmt: str) -> str:
    """Render a metric value, marking the ones the run never recorded."""
    return UNKNOWN if value is None else fmt.format(value)


def _delta_cell(delta: float | None, fmt: str, higher_is_better: bool | None) -> str:
    """Render a signed delta, coloured by whether it is an improvement."""
    if delta is None:
        return NOT_COMPARABLE
    text = f"{delta:+{fmt[2:-1]}}" if fmt.startswith("{:") else f"{delta:+.2f}"
    if higher_is_better is None or delta == 0:
        return text
    good = delta > 0 if higher_is_better else delta < 0
    return f"[green]{text}[/]" if good else f"[red]{text}[/]"


def _pct_cell(pct: float | None, higher_is_better: bool | None) -> str:
    """Render a percentage change, or say it has no usable baseline."""
    if pct is None:
        return NOT_COMPARABLE
    text = f"{pct:+.1f}%"
    if higher_is_better is None or pct == 0:
        return text
    good = pct > 0 if higher_is_better else pct < 0
    return f"[green]{text}[/]" if good else f"[red]{text}[/]"


def render_comparison(
    comparison: dict[str, Any], console: Console | None = None
) -> None:
    """Render a :func:`compare_runs` result as rich tables.

    Prints a configuration table with one column per run — rows that
    differ between runs are marked — then a metrics table carrying the
    absolute values, the delta against the baseline, and the percentage
    change, then whatever the intersection left out.
    """
    from rich.console import Console as RichConsole
    from rich.table import Table as RichTable

    out = console or RichConsole()
    labels: list[str] = comparison["labels"]
    baseline = comparison["baseline_label"]

    config = RichTable(title="Configuration")
    config.add_column("Field", style="cyan")
    for label in labels:
        config.add_column(label, overflow="fold")
    for row in comparison["config_rows"]:
        marker = "•" if row["differs"] else " "
        cells = [_cell(value) for value in row["values"]]
        config.add_row(
            f"{marker} {row['field']}",
            *cells,
            style="yellow" if row["differs"] else None,
        )
    out.print(config)
    out.print()

    metrics = RichTable(title=f"Metrics — baseline {baseline}")
    metrics.add_column("Metric", style="cyan", overflow="fold")
    for label in labels:
        metrics.add_column(label, justify="right")
    for label in labels[1:]:
        metrics.add_column(f"Δ {label}", justify="right")
        metrics.add_column(f"Δ% {label}", justify="right")

    for row in comparison["metric_rows"]:
        fmt = row["format"]
        higher = row["higher_is_better"]
        cells = [_value_cell(value, fmt) for value in row["values"]]
        for delta, pct in zip(row["deltas"][1:], row["pct"][1:], strict=True):
            cells.append(_delta_cell(delta, fmt, higher))
            cells.append(_pct_cell(pct, higher))
        metrics.add_row(row["metric"], *cells)
    out.print(metrics)

    _render_exclusions(comparison, out)


def _render_exclusions(comparison: dict[str, Any], out: Console) -> None:
    """Say what the intersection left out, instead of hiding it."""
    excluded_categories = comparison["excluded_categories"]
    excluded_tests = comparison["excluded_tests"]
    if not excluded_categories and not excluded_tests:
        return

    out.print()
    out.print("[yellow]Not compared[/] — absent from at least one run:")
    for item in excluded_categories:
        present = ", ".join(item["present_in"])
        out.print(f"  category {item['category']} — only in {present}")

    grouped: dict[tuple[str, tuple[str, ...]], list[str]] = {}
    for item in excluded_tests:
        key = (item["category"], tuple(item["present_in"]))
        grouped.setdefault(key, []).append(item["test_name"])
    for (category, present_in), names in grouped.items():
        present = ", ".join(present_in)
        out.print(f"  {category}: {', '.join(names)} — only in {present}")


def render_run(report: dict[str, Any], console: Console | None = None) -> None:
    """Render a single saved run as rich tables.

    Shows the serving configuration, the headline figures, the prefill
    rate per input size, the concurrency scaling, and the status counts.
    """
    from rich.console import Console as RichConsole
    from rich.table import Table as RichTable

    out = console or RichConsole()
    run = describe_run(report)

    out.print(f"\n[bold]{run['label']}[/]")
    if run["source_path"]:
        out.print(f"[dim]{run['source_path']}[/]")
    if not run["has_provenance"]:
        out.print("[yellow]No serving provenance recorded[/] — this run predates it.")
    out.print()

    config = RichTable(title="Configuration")
    config.add_column("Field", style="cyan")
    config.add_column("Value", overflow="fold")
    config.add_row("run timestamp", _format_stamp_from_run(run))
    config.add_row("served model", _cell(run["served_model"]))
    config.add_row("categories", _cell(run["categories"]))
    for field in PROVENANCE_FIELDS:
        config.add_row(field, _cell(run["provenance"][field]))
    out.print(config)
    out.print()

    headline = RichTable(title="Headline")
    headline.add_column("Metric", style="cyan")
    headline.add_column("Value", justify="right", style="bold green")
    for metric, key, fmt, _higher, scale in _HEADLINE_METRICS:
        headline.add_row(metric, _value_cell(_scaled(run.get(key), scale), fmt))
    out.print(headline)
    out.print()

    if run["prefill_tps"]:
        prefill = RichTable(title="Prefill")
        prefill.add_column("Input size", style="cyan")
        prefill.add_column("Prefill tok/s", justify="right")
        for test_name, rate in run["prefill_tps"].items():
            prefill.add_row(test_name, f"{rate:.0f}")
        out.print(prefill)
        out.print()

    if run["concurrency"]:
        table = RichTable(title="Concurrency")
        table.add_column("Workers", style="cyan", justify="right")
        table.add_column("Aggregate tok/s", justify="right", style="bold magenta")
        table.add_column("Per-stream tok/s (median)", justify="right")
        table.add_column("Wall time (s)", justify="right")
        for level in run["concurrency"]:
            table.add_row(
                _cell(level["workers"]),
                _value_cell(level["aggregate_tps"], "{:.1f}"),
                _value_cell(level["per_stream_tps"], "{:.1f}"),
                _value_cell(level["wall_time_s"], "{:.2f}"),
            )
        out.print(table)
        out.print()

    counts = run["status_counts"]
    status = RichTable(title="Status")
    status.add_column("Passed", justify="right")
    status.add_column("Failed", justify="right")
    status.add_column("Skipped", justify="right")
    status.add_column("Error", justify="right")
    status.add_column("Total", justify="right")
    status.add_row(
        str(counts["passed"]),
        str(counts["failed"]),
        str(counts["skipped"]),
        str(counts["error"]),
        str(counts["total"]),
    )
    out.print(status)
