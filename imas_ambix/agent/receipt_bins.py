"""Summarise interval serving receipts by recorded concurrency width."""

from __future__ import annotations

import dataclasses
import json
import statistics
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


DEFAULT_WIDTH_BINS: tuple[tuple[int, int], ...] = ((10, 14), (15, 19))


@dataclasses.dataclass(frozen=True)
class PrefixHitDistribution:
    """Distribution of interval prefix-cache hit rates in one width bin."""

    median: float
    deciles: dict[str, float]


@dataclasses.dataclass(frozen=True)
class WidthBinSummary:
    """Receipt-rate summary for one inclusive running-request range."""

    lower_width: int
    upper_width: int
    intervals: int
    decode_toks_per_s_per_worker_median: float | None
    prefill_toks_per_s_median: float | None
    prefix_hit_rate: PrefixHitDistribution | None

    @property
    def label(self) -> str:
        """Human-readable inclusive width range."""
        return f"{self.lower_width}-{self.upper_width}"


@dataclasses.dataclass(frozen=True)
class ReceiptBinReport:
    """All requested width bins plus visible interval exclusions."""

    rows_read: int
    excluded_intervals: int
    unbinned_intervals: int
    bins: tuple[WidthBinSummary, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready representation for evidence records."""
        return {
            "rows_read": self.rows_read,
            "excluded_intervals": self.excluded_intervals,
            "unbinned_intervals": self.unbinned_intervals,
            "bins": {
                summary.label: {
                    "intervals": summary.intervals,
                    "decode_toks_per_s_per_worker_median": (
                        summary.decode_toks_per_s_per_worker_median
                    ),
                    "prefill_toks_per_s_median": summary.prefill_toks_per_s_median,
                    "prefix_hit_rate": (
                        None
                        if summary.prefix_hit_rate is None
                        else dataclasses.asdict(summary.prefix_hit_rate)
                    ),
                }
                for summary in self.bins
            },
        }


_INTERVAL_FIELDS = (
    "generation_throughput_toks_per_s",
    "prompt_throughput_toks_per_s",
    "prefix_cache_hit_rate_interval",
)


def read_receipt_rows(receipts_path: str | Path) -> list[dict[str, Any]]:
    """Read the non-empty JSONL receipt rows, naming malformed input lines."""
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        Path(receipts_path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid receipt JSON at line {line_number}") from error
        if not isinstance(row, dict):
            raise ValueError(f"receipt line {line_number} is not an object")
        rows.append(row)
    return rows


def collect_receipt_bins(
    receipts_path: str | Path,
    *,
    width_bins: tuple[tuple[int, int], ...] = DEFAULT_WIDTH_BINS,
) -> ReceiptBinReport:
    """Read receipt JSONL from a receipt file and bin complete intervals by width.

    **Unwired by the record index on purpose.** The interval statistics have one
    owner, :func:`summarise_receipt_rows`, and both entry points reach it; they
    differ only in where the rows come from. This one reads a path, while the
    record index aggregates the rows it has already ingested and does so without
    touching the file again — re-reading a file it holds rows from is exactly
    the work the index exists to avoid. So the index calls
    :func:`summarise_receipt_rows` directly rather than routing through here, and
    this entry point waits for a caller that holds a receipt path and no index.
    """
    return summarise_receipt_rows(
        read_receipt_rows(receipts_path), width_bins=width_bins
    )


def summarise_receipt_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    width_bins: tuple[tuple[int, int], ...] = DEFAULT_WIDTH_BINS,
) -> ReceiptBinReport:
    """Summarise receipt rows without replacing interval distributions by means."""
    _validate_width_bins(width_bins)
    grouped: dict[tuple[int, int], list[tuple[float, float, float]]] = {
        width_bin: [] for width_bin in width_bins
    }
    rows_read = 0
    excluded_intervals = 0
    unbinned_intervals = 0

    for row in rows:
        rows_read += 1
        if any(row.get(field) is None for field in _INTERVAL_FIELDS):
            excluded_intervals += 1
            continue

        running = row.get("num_requests_running")
        if not isinstance(running, int) or isinstance(running, bool) or running <= 0:
            unbinned_intervals += 1
            continue

        width_bin = _matching_width_bin(running, width_bins)
        if width_bin is None:
            unbinned_intervals += 1
            continue

        generation = _as_float(row["generation_throughput_toks_per_s"])
        prefill = _as_float(row["prompt_throughput_toks_per_s"])
        prefix_hit_rate = _as_float(row["prefix_cache_hit_rate_interval"])
        grouped[width_bin].append((generation / running, prefill, prefix_hit_rate))

    return ReceiptBinReport(
        rows_read=rows_read,
        excluded_intervals=excluded_intervals,
        unbinned_intervals=unbinned_intervals,
        bins=tuple(
            _summarise_bin(lower, upper, grouped[(lower, upper)])
            for lower, upper in width_bins
        ),
    )


def _validate_width_bins(width_bins: tuple[tuple[int, int], ...]) -> None:
    for lower, upper in width_bins:
        if lower < 1 or upper < lower:
            raise ValueError(f"invalid width bin {lower}-{upper}")


def _matching_width_bin(
    running: int, width_bins: tuple[tuple[int, int], ...]
) -> tuple[int, int] | None:
    for lower, upper in width_bins:
        if lower <= running <= upper:
            return lower, upper
    return None


def _as_float(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"receipt interval value must be numeric, got {value!r}")
    return float(value)


def _summarise_bin(
    lower_width: int, upper_width: int, values: list[tuple[float, float, float]]
) -> WidthBinSummary:
    if not values:
        return WidthBinSummary(
            lower_width=lower_width,
            upper_width=upper_width,
            intervals=0,
            decode_toks_per_s_per_worker_median=None,
            prefill_toks_per_s_median=None,
            prefix_hit_rate=None,
        )

    decode_rates, prefill_rates, prefix_hit_rates = zip(*values, strict=True)
    return WidthBinSummary(
        lower_width=lower_width,
        upper_width=upper_width,
        intervals=len(values),
        decode_toks_per_s_per_worker_median=statistics.median(decode_rates),
        prefill_toks_per_s_median=statistics.median(prefill_rates),
        prefix_hit_rate=PrefixHitDistribution(
            median=statistics.median(prefix_hit_rates),
            deciles={
                f"p{decile * 10}": _inclusive_quantile(prefix_hit_rates, decile / 10)
                for decile in range(1, 10)
            },
        ),
    )


def _inclusive_quantile(values: tuple[float, ...], probability: float) -> float:
    """Quantile interpolation that also has a defined one-interval result."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = probability * (len(ordered) - 1)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction
