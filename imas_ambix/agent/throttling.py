"""Read cgroup v2 CPU throttling counters for one control group.

Both inputs are the kernel's own text, passed in rather than read from a fixed
path, so the caller decides which control group is measured: ``cpu.max`` states
the ceiling as a quota and a period, and ``cpu.stat`` carries the cumulative
counters the kernel updates as it enforces that ceiling.

Throttling is reported as a share of the CPU thread-time the ceiling permitted
over the periods accounted, which is what makes the figure comparable between
one control group holding four cores and another holding twenty-eight. A
control group with no ceiling has no such share: that is undefined, not zero,
because a share of an unbounded quantity does not exist while a measured share
of zero asserts that nothing was ever refused.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

_CPU_MAX_FILE = "cpu.max"
_CPU_STAT_FILE = "cpu.stat"

# The cpu.max quota field when no ceiling is set.
_UNLIMITED = "max"

_COUNTER_FIELDS = ("usage_usec", "nr_periods", "nr_throttled", "throttled_usec")


@dataclass(frozen=True, slots=True)
class CpuMax:
    """The control group's CPU ceiling, as ``cpu.max`` states it."""

    quota_usec: int | None
    period_usec: int

    @property
    def can_throttle(self) -> bool:
        """Whether a ceiling exists that the group could be stopped by."""
        return self.quota_usec is not None


@dataclass(frozen=True, slots=True)
class CpuStat:
    """Cumulative CPU accounting counters from ``cpu.stat``."""

    usage_usec: int
    nr_periods: int
    nr_throttled: int
    throttled_usec: int


@dataclass(frozen=True, slots=True)
class Sample:
    """One control group's ceiling and counters, read together."""

    cpu_max: CpuMax
    stat: CpuStat


def _int_field(field: str, label: str) -> int:
    try:
        return int(field)
    except ValueError as exc:
        raise ValueError(f"{label} is not an integer: {field!r}") from exc


def parse_cpu_max(text: str) -> CpuMax:
    """Parse ``cpu.max`` text into a quota and a period.

    The quota field is either a microsecond budget per period or the literal
    ``max``, which removes the ceiling instead of setting a large one.
    """
    fields = text.split()
    if len(fields) != 2:
        raise ValueError(f"cpu.max expects a quota and a period, got {text!r}")
    quota_field, period_field = fields
    period_usec = _int_field(period_field, "cpu.max period")
    if period_usec <= 0:
        raise ValueError(f"cpu.max period must be positive, got {period_usec}")
    if quota_field == _UNLIMITED:
        return CpuMax(quota_usec=None, period_usec=period_usec)
    quota_usec = _int_field(quota_field, "cpu.max quota")
    if quota_usec < 0:
        raise ValueError(f"cpu.max quota must not be negative, got {quota_usec}")
    return CpuMax(quota_usec=quota_usec, period_usec=period_usec)


def parse_cpu_stat(text: str) -> CpuStat:
    """Parse ``cpu.stat`` text, taking the counters this module reports on.

    Fields the kernel adds later are ignored rather than refused, so a host on
    a newer kernel does not report an empty reading.
    """
    counters: dict[str, int] = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) != 2:
            continue
        key, value = fields
        if key in _COUNTER_FIELDS:
            counters[key] = _int_field(value, f"cpu.stat {key}")
    missing = [field for field in _COUNTER_FIELDS if field not in counters]
    if missing:
        raise ValueError(f"cpu.stat is missing {', '.join(missing)}")
    return CpuStat(**counters)


def read_sample(directory: str | Path) -> Sample:
    """Read one control group's ceiling and counters from its directory."""
    base = Path(directory)
    return Sample(
        cpu_max=parse_cpu_max((base / _CPU_MAX_FILE).read_text()),
        stat=parse_cpu_stat((base / _CPU_STAT_FILE).read_text()),
    )


def permitted_thread_usec(stat: CpuStat, cpu_max: CpuMax) -> int | None:
    """Thread-time the ceiling permitted over the periods accounted.

    A group allowed one quota per period may run that quota once each period,
    so the permitted total is the quota times the period count; the period
    itself cancels. An unlimited group permits an unbounded total, which has no
    integer value and is reported as ``None``.
    """
    if cpu_max.quota_usec is None:
        return None
    return stat.nr_periods * cpu_max.quota_usec


def throttled_share(stat: CpuStat, cpu_max: CpuMax) -> float | None:
    """Fraction of permitted thread-time spent stalled at the ceiling.

    ``None`` means there is no fraction to state -- either the group has no
    ceiling, or no period has yet been accounted to divide by.
    """
    permitted_usec = permitted_thread_usec(stat, cpu_max)
    if not permitted_usec:
        return None
    return stat.throttled_usec / permitted_usec


def interval_delta(first: Sample, second: Sample) -> CpuStat:
    """Counters accumulated between two samples of the same control group."""
    earlier, later = first.stat, second.stat
    for field in _COUNTER_FIELDS:
        before, after = getattr(earlier, field), getattr(later, field)
        if after < before:
            raise ValueError(
                f"{field} went backwards between samples: {before} -> {after}"
            )
    return CpuStat(
        usage_usec=later.usage_usec - earlier.usage_usec,
        nr_periods=later.nr_periods - earlier.nr_periods,
        nr_throttled=later.nr_throttled - earlier.nr_throttled,
        throttled_usec=later.throttled_usec - earlier.throttled_usec,
    )


def interval_throttled_share(first: Sample, second: Sample) -> float | None:
    """Throttled share over the span between two samples of one group.

    The ceiling is taken from the earlier sample, so the delta share describes
    the quota that was in force while those counters accumulated. A later
    reading of the same group reports its own ceiling, which is how a quota
    changed mid-span is visible rather than silently mixed into the figure.
    """
    return throttled_share(interval_delta(first, second), first.cpu_max)


def format_cpu_max(cpu_max: CpuMax) -> str:
    """Render a ceiling back into the two-field form ``cpu.max`` uses."""
    quota = _UNLIMITED if cpu_max.quota_usec is None else str(cpu_max.quota_usec)
    return f"{quota} {cpu_max.period_usec}"


def _counters_payload(stat: CpuStat, cpu_max: CpuMax) -> dict[str, object]:
    return {
        "usage_usec": stat.usage_usec,
        "nr_periods": stat.nr_periods,
        "nr_throttled": stat.nr_throttled,
        "throttled_usec": stat.throttled_usec,
        "permitted_usec": permitted_thread_usec(stat, cpu_max),
        "throttled_share": throttled_share(stat, cpu_max),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read cgroup v2 CPU throttling counters"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    sample = subparsers.add_parser("sample")
    sample.add_argument("--directory", required=True)
    interval = subparsers.add_parser("interval")
    interval.add_argument("--directory", required=True)
    interval.add_argument("--seconds", required=True, type=float)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Read one control group and print its throttled share as JSON."""
    args = _parser().parse_args(argv)
    first = read_sample(args.directory)
    payload: dict[str, object] = {
        "directory": args.directory,
        "cpu_max": format_cpu_max(first.cpu_max),
        "cumulative": _counters_payload(first.stat, first.cpu_max),
    }
    if args.command == "interval":
        time.sleep(args.seconds)
        second = read_sample(args.directory)
        interval = _counters_payload(interval_delta(first, second), first.cpu_max)
        interval["seconds"] = args.seconds
        interval["cpu_max_at_end"] = format_cpu_max(second.cpu_max)
        payload["interval"] = interval
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
