"""Harvest engine Decode batch log intervals into per-width generation medians.

The vLLM serve logs one ``Decode batch`` line per scheduling step, carrying the
running request width, the speculative accept length and rate, and that step's
generation throughput. Fitting a cost model across widths needs those pairs
grouped by exact width, with a median rather than a mean because the per-step
figures are heavy-tailed. This module is that grouping: the record shape, a
per-width and per-limb summary, and a text rendering for evidence capture.

Each record carries a *limb* -- whether the width was rising, falling or flat at
the moment the interval was logged -- derived from the width trajectory across
the log, not from anything printed on the line. The trajectory is the first
difference between consecutive intervals, so the earliest interval has no
predecessor and its limb stays ``None``; the report renders that group as
``unknown`` rather than calling it flat, because a no-change label would assert
something no interval observed.

A ``Decode batch`` line that cannot be parsed is collected with its line number
and the field that could not be read, never dropped: a log read that silently
discards its unreadable lines reports a smaller sample as if it were the whole
one.
"""

from __future__ import annotations

import dataclasses
import re
import statistics
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

ASCENDING = "ascending"
DESCENDING = "descending"
FLAT = "flat"

_LIMB_ORDER: tuple[str | None, ...] = (ASCENDING, DESCENDING, FLAT, None)
_LIMB_LABELS: dict[str | None, str] = {
    ASCENDING: "ascending",
    DESCENDING: "descending",
    FLAT: "flat",
    None: "unknown",
}

_DECODE_MARKER = "Decode batch"

_TIMESTAMP = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
_RUNNING_WIDTH = re.compile(r"#running-req:\s*(\d+)")
_ACCEPT_LENGTH = re.compile(r"accept len:\s*(\d+(?:\.\d+)?)")
_ACCEPT_RATE = re.compile(r"accept rate:\s*(\d+(?:\.\d+)?)")
_GENERATION_RATE = re.compile(r"gen throughput \(token/s\):\s*(\d+(?:\.\d+)?)")

_FIELDS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("timestamp", _TIMESTAMP),
    ("running width", _RUNNING_WIDTH),
    ("accept length", _ACCEPT_LENGTH),
    ("accept rate", _ACCEPT_RATE),
    ("generation rate", _GENERATION_RATE),
)


@dataclasses.dataclass(frozen=True)
class DecodeInterval:
    """One engine decode step, as logged."""

    timestamp: str
    width: int
    accept_length: float
    accept_rate: float
    generation_rate: float
    limb: str | None = None


@dataclasses.dataclass(frozen=True)
class MalformedDecodeLine:
    """A line announcing a decode interval whose fields could not be read."""

    line_number: int
    reason: str
    text: str


@dataclasses.dataclass(frozen=True)
class LimbSummary:
    """The decode intervals sharing one width and one trajectory limb."""

    limb: str | None
    intervals: int
    generation_rate_median: float
    accept_length_median: float

    @property
    def label(self) -> str:
        """Name the limb, including the no-predecessor group."""
        return _LIMB_LABELS[self.limb]


@dataclasses.dataclass(frozen=True)
class WidthSummary:
    """Decode intervals logged at one exact running width."""

    width: int
    intervals: int
    generation_rate_median: float
    accept_length_median: float
    limbs: tuple[LimbSummary, ...]

    @property
    def label(self) -> str:
        """The width as a report key."""
        return str(self.width)


@dataclasses.dataclass(frozen=True)
class DecodeHarvest:
    """Every decodable interval in one serve log, grouped by exact width."""

    source: str
    lines_read: int
    intervals: tuple[DecodeInterval, ...]
    malformed: tuple[MalformedDecodeLine, ...]
    widths: tuple[WidthSummary, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready representation for evidence records."""
        return {
            "source": self.source,
            "lines_read": self.lines_read,
            "intervals": len(self.intervals),
            "malformed": [
                {
                    "line_number": line.line_number,
                    "reason": line.reason,
                    "text": line.text,
                }
                for line in self.malformed
            ],
            "widths": {
                summary.label: {
                    "intervals": summary.intervals,
                    "generation_rate_median": summary.generation_rate_median,
                    "accept_length_median": summary.accept_length_median,
                    "limbs": {
                        limb.label: {
                            "intervals": limb.intervals,
                            "generation_rate_median": limb.generation_rate_median,
                            "accept_length_median": limb.accept_length_median,
                        }
                        for limb in summary.limbs
                    },
                }
                for summary in self.widths
            },
        }


@dataclasses.dataclass(frozen=True)
class DecodeParse:
    """Decode intervals read from a log, beside the decode lines that failed."""

    intervals: tuple[DecodeInterval, ...]
    malformed: tuple[MalformedDecodeLine, ...]


def parse_decode_lines(lines: Iterable[str]) -> DecodeParse:
    """Read every readable ``Decode batch`` interval from log lines.

    A line that announces a decode interval but cannot be parsed lands in
    ``malformed`` with its line number and the field that could not be read,
    rather than being dropped; lines announcing some other batch type are not
    decode intervals and are ignored.
    """
    intervals: list[DecodeInterval] = []
    malformed: list[MalformedDecodeLine] = []
    for line_number, line in enumerate(lines, start=1):
        if _DECODE_MARKER not in line:
            continue
        parsed = _parse_decode_line(line, line_number)
        if isinstance(parsed, MalformedDecodeLine):
            malformed.append(parsed)
            continue
        intervals.append(parsed)
    return DecodeParse(intervals=tuple(intervals), malformed=tuple(malformed))


def harvest_decode_log(log_path: str | Path) -> DecodeHarvest:
    """Read a serve log and summarise its decode intervals by exact width."""
    text = Path(log_path).read_bytes().decode("utf-8", errors="replace")
    lines = physical_lines(text)
    parsed = parse_decode_lines(lines)
    return summarise_decode_intervals(
        parsed.intervals,
        source=str(log_path),
        lines_read=len(lines),
        malformed=parsed.malformed,
    )


def physical_lines(text: str) -> list[str]:
    """Split a log the way a line counter does: on newlines alone.

    Two things split this engine's log where a line counter does not, and both
    are live on the serve logs here. The loader rewrites its shard-loading
    progress bar in place, emitting a bare carriage return between redraws, and
    ``str.splitlines`` breaks on one, turning a single step into a dozen lines.
    Reaching the splitter at all requires the caller to have kept those
    carriage returns: text-mode reading translates a bare ``\r`` to ``\n``
    before any of this runs, so a reader that opens in text mode reports the
    inflated count even though this function is correct. Hence the byte read in
    ``harvest_decode_log``. A trailing carriage return is removed rather than
    treated as a separator, so a CRLF log counts one line per record, and the
    empty tail left by a final newline is dropped so the count matches the
    file's own line count.
    """
    lines = [line.removesuffix("\r") for line in text.split("\n")]
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def assign_limbs(
    intervals: Sequence[DecodeInterval],
) -> tuple[DecodeInterval, ...]:
    """Label each interval from the width trajectory it sits in.

    The limb is the direction of the first difference against the preceding
    interval, so the earliest interval has no predecessor and keeps ``None``.
    A width that repeats is ``flat``: the trajectory carries that direction and
    the width alone does not.
    """
    labelled: list[DecodeInterval] = []
    previous_width: int | None = None
    for interval in intervals:
        if previous_width is None:
            limb: str | None = None
        elif interval.width > previous_width:
            limb = ASCENDING
        elif interval.width < previous_width:
            limb = DESCENDING
        else:
            limb = FLAT
        labelled.append(dataclasses.replace(interval, limb=limb))
        previous_width = interval.width
    return tuple(labelled)


def summarise_decode_intervals(
    intervals: Iterable[DecodeInterval],
    *,
    source: str = "",
    lines_read: int = 0,
    malformed: Sequence[MalformedDecodeLine] = (),
) -> DecodeHarvest:
    """Group decode intervals by exact width and take medians, overall and by limb.

    Limb labels are derived here from the interval order, so callers pass the
    parsed sequence and do not label it themselves.
    """
    ordered = assign_limbs(intervals)
    by_width: dict[int, list[DecodeInterval]] = {}
    for interval in ordered:
        by_width.setdefault(interval.width, []).append(interval)

    return DecodeHarvest(
        source=source,
        lines_read=lines_read,
        intervals=tuple(ordered),
        malformed=tuple(malformed),
        widths=tuple(
            _summarise_width(width, by_width[width]) for width in sorted(by_width)
        ),
    )


def format_harvest(harvest: DecodeHarvest) -> str:
    """Render a harvest as a fixed-width table for evidence capture."""
    lines = [
        f"source: {harvest.source}",
        (
            f"lines read: {harvest.lines_read}  "
            f"intervals: {len(harvest.intervals)}  "
            f"malformed decode lines: {len(harvest.malformed)}"
        ),
        "",
        (
            f"{'width':>6} {'limb':>10} {'n':>5} "
            f"{'gen tok/s med':>14} {'accept len med':>15}"
        ),
    ]
    for summary in harvest.widths:
        lines.append(
            f"{summary.width:>6} {'all':>10} {summary.intervals:>5} "
            f"{summary.generation_rate_median:>14.3f} "
            f"{summary.accept_length_median:>15.3f}"
        )
        for limb in summary.limbs:
            lines.append(
                f"{'':>6} {limb.label:>10} {limb.intervals:>5} "
                f"{limb.generation_rate_median:>14.3f} "
                f"{limb.accept_length_median:>15.3f}"
            )
    for line in harvest.malformed:
        lines.append(f"malformed line {line.line_number}: {line.reason}")
    return "\n".join(lines)


def _parse_decode_line(
    line: str, line_number: int
) -> DecodeInterval | MalformedDecodeLine:
    read: dict[str, str] = {}
    missing: list[str] = []
    for name, pattern in _FIELDS:
        match = pattern.search(line)
        if match is None:
            missing.append(name)
        else:
            read[name] = match.group(1)
    if missing:
        return MalformedDecodeLine(
            line_number=line_number,
            reason="unreadable " + ", ".join(missing),
            text=line,
        )
    return DecodeInterval(
        timestamp=read["timestamp"],
        width=int(read["running width"]),
        accept_length=float(read["accept length"]),
        accept_rate=float(read["accept rate"]),
        generation_rate=float(read["generation rate"]),
    )


def _summarise_width(width: int, intervals: list[DecodeInterval]) -> WidthSummary:
    by_limb: dict[str | None, list[DecodeInterval]] = {}
    for interval in intervals:
        by_limb.setdefault(interval.limb, []).append(interval)

    return WidthSummary(
        width=width,
        intervals=len(intervals),
        generation_rate_median=statistics.median(
            interval.generation_rate for interval in intervals
        ),
        accept_length_median=statistics.median(
            interval.accept_length for interval in intervals
        ),
        limbs=tuple(
            _summarise_limb(limb, by_limb[limb])
            for limb in _LIMB_ORDER
            if limb in by_limb
        ),
    )


def _summarise_limb(limb: str | None, intervals: list[DecodeInterval]) -> LimbSummary:
    return LimbSummary(
        limb=limb,
        intervals=len(intervals),
        generation_rate_median=statistics.median(
            interval.generation_rate for interval in intervals
        ),
        accept_length_median=statistics.median(
            interval.accept_length for interval in intervals
        ),
    )


if __name__ == "__main__":
    import sys

    print(format_harvest(harvest_decode_log(sys.argv[1])))
