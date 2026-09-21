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

A share can be absent for two different reasons, and a caller acts differently
on each, so they are kept apart rather than collapsed into one missing value.
An unbounded group can never yield a share however long it is watched, while a
group with no accounted period yet has simply not run long enough and will
state a share at a later reading. ``Unmeasured`` names both.

A group with no ``cpu.max`` file states no ceiling in that file, which is one
side of the comparison this sampler exists to take, so such a group is reported
rather than raised on. The missing file does not decide what the group accounts,
so its counters are taken from ``cpu.stat`` as that text states them. A group
the cpu controller never attached a ceiling to accounts no period, and a compute
node's user slice states none. A group with no ceiling file of its own can still
carry accounted periods — the root group the machine's whole run is accounted
under is exactly that — and those counters are genuine values, zeros included.
Reporting the first as zeros would invent periods that never elapsed; reporting
the second as absent would discard a count the kernel did keep.

A share is comparable only between readings of the same machine, and nothing in
the cgroup text says which machine it came from: the counter paths resolve on
every host in the cluster with different contents. The payload therefore names
the host and the boot it was read on. The boot matters because the counters are
cumulative since boot, so a reboot restarts them and the drop reads as an
improvement unless the two readings can be attributed to different boots.
"""

from __future__ import annotations

import argparse
import json
import re
import socket
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

_CPU_MAX_FILE = "cpu.max"
_CPU_STAT_FILE = "cpu.stat"

# The kernel's own record of which boot this is, and how long it has run.
_BOOT_ID_FILE = "/proc/sys/kernel/random/boot_id"
_UPTIME_FILE = "/proc/uptime"

# The cpu.max quota field when no ceiling is set.
_UNLIMITED = "max"

# How a control group with no cpu.max file at all is stated back for one.
_ABSENT = "absent"

# The canonical boot identifier the kernel draws: five groups of lowercase
# hexadecimal digits, eight then three of four. Nothing else is a boot
# identifier, and a value that merely looks like one would be reported and
# compared as though it named a boot.
_BOOT_ID_SHAPE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)

# The counters a control group accounts for throttling: the cumulative usage is
# accounted everywhere, the three period counters wherever the kernel accounts
# periods — a group with no ceiling of its own may carry them, and a group with
# no accounted period states none.
_USAGE_FIELD = "usage_usec"
_PERIOD_FIELDS = ("nr_periods", "nr_throttled", "throttled_usec")
_COUNTER_FIELDS = (_USAGE_FIELD, *_PERIOD_FIELDS)


@dataclass(frozen=True, slots=True)
class CpuMax:
    """The control group's CPU ceiling, as ``cpu.max`` states it."""

    quota_usec: int | None
    period_usec: int | None

    @property
    def can_throttle(self) -> bool:
        """Whether a ceiling exists that the group could be stopped by."""
        return self.quota_usec is not None


@dataclass(frozen=True, slots=True)
class CpuStat:
    """Cumulative CPU accounting counters from ``cpu.stat``.

    The cumulative usage is accounted for every group the cpu controller sees.
    The three period counters are accounted for a group the controller enforces
    a ceiling on, and are read where the text states them for a group whose
    ceiling is not declared in a ``cpu.max`` file of its own: the root group
    accounts the machine's whole run. They are absent only where the text states
    none, which is not the same as zero — a measured zero asserts that periods
    elapsed and none was exceeded.
    """

    usage_usec: int
    nr_periods: int | None
    nr_throttled: int | None
    throttled_usec: int | None

    @property
    def period_counters(self) -> tuple[int, int, int] | None:
        """The three period counters together, or ``None`` if unaccounted.

        They are stated by one mechanism, so a group either accounts all three
        or none; they are returned as one value so a caller narrows them once.
        """
        if (
            self.nr_periods is None
            or self.nr_throttled is None
            or self.throttled_usec is None
        ):
            return None
        return (self.nr_periods, self.nr_throttled, self.throttled_usec)


@dataclass(frozen=True, slots=True)
class Sample:
    """One control group's ceiling and counters, read together."""

    cpu_max: CpuMax
    stat: CpuStat


@dataclass(frozen=True, slots=True)
class HostIdentity:
    """Which machine a reading came from, and which boot of that machine.

    ``boot_id`` is the kernel's own per-boot identifier, so two readings
    carrying the same value describe one uninterrupted run of counters and two
    different values mark a reboot between them. ``uptime_seconds`` says how
    long that run has been going, which makes the difference readable rather
    than merely detectable.
    """

    hostname: str
    boot_id: str
    uptime_seconds: float


class Unmeasured(StrEnum):
    """Why a throttled share has no numeric value.

    The two causes call for different responses and so are reported apart.
    ``UNBOUNDED`` is permanent: the group has no ceiling, so no ceiling can ever
    be exceeded, and waiting for a figure is waiting for one that cannot exist.
    ``NO_PERIODS`` is provisional: no period has been accounted yet, so the
    group has simply not been measured over a window and the next reading may
    state a share.
    """

    UNBOUNDED = "unbounded"
    NO_PERIODS = "no_periods"


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


def _stat_fields(text: str) -> dict[str, int]:
    """The counters this module reports on, taken out of ``cpu.stat`` text.

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
    return counters


def _required(field: str, counters: dict[str, int]) -> int:
    value = counters.get(field)
    if value is None:
        raise ValueError(f"cpu.stat is missing {field}")
    return value


def parse_cpu_stat(text: str) -> CpuStat:
    """Parse ``cpu.stat`` for a group whose ceiling is enforced.

    All four counters are required together: a group the kernel keeps a ceiling
    on accounts for that ceiling, so a missing counter is malformed text rather
    than a group that throttles nothing, and reading it as zero would report an
    idle group where nothing was measured.
    """
    counters = _stat_fields(text)
    return CpuStat(
        usage_usec=_required(_USAGE_FIELD, counters),
        nr_periods=_required("nr_periods", counters),
        nr_throttled=_required("nr_throttled", counters),
        throttled_usec=_required("throttled_usec", counters),
    )


def parse_cpu_stat_without_ceiling(text: str) -> CpuStat:
    """Parse ``cpu.stat`` for a group whose ceiling is not stated by a file.

    The usage is required, as it is everywhere. The three period counters are
    taken as the text states them rather than assumed either way: a group the
    cpu controller never attached a ceiling to states none of them, while a
    group that accounts periods states all three together with genuine values,
    and a value of zero there means periods elapsed and none was exceeded. The
    three are read as one set, because one kernel account writes them together
    and a text stating some but not others is malformed rather than partly
    accounted.
    """
    counters = _stat_fields(text)
    usage_usec = _required(_USAGE_FIELD, counters)
    stated = [field for field in _PERIOD_FIELDS if field in counters]
    if not stated:
        return CpuStat(
            usage_usec=usage_usec,
            nr_periods=None,
            nr_throttled=None,
            throttled_usec=None,
        )
    unstated = [field for field in _PERIOD_FIELDS if field not in counters]
    if unstated:
        raise ValueError(
            "cpu.stat states some period counters but not "
            f"{', '.join(unstated)}: the kernel accounts them together"
        )
    return CpuStat(
        usage_usec=usage_usec,
        nr_periods=counters["nr_periods"],
        nr_throttled=counters["nr_throttled"],
        throttled_usec=counters["throttled_usec"],
    )


def parse_uptime(text: str) -> float:
    """Seconds since boot, read from the first field of ``/proc/uptime``.

    The second field is the sum of per-CPU idle time, which is not the
    machine's age and is ignored. A machine that has just rebooted reports a
    small value, which is what makes this useful beside a cumulative counter.
    """
    fields = text.split()
    if not fields:
        raise ValueError("uptime is empty")
    try:
        return float(fields[0])
    except ValueError as exc:
        raise ValueError(f"uptime is not a number: {fields[0]!r}") from exc


def parse_boot_id(text: str) -> str:
    """The kernel's identifier for the current boot, from ``boot_id``.

    A fresh value is drawn at each boot, so two readings carrying the same
    identifier share one run of cumulative counters and two different ones do
    not, however similar their shares look.
    """
    boot_id = text.strip()
    if not _BOOT_ID_SHAPE.fullmatch(boot_id):
        raise ValueError(f"boot_id is not a boot identifier: {boot_id!r}")
    return boot_id


# A group the kernel keeps no ceiling on states no cpu.max at all: the cpu
# controller is not enabled for it. That is reported as a ceiling with neither
# field, which is how a reader tells it apart from one stated as ``max``.
_NO_CEILING = CpuMax(quota_usec=None, period_usec=None)


def read_sample(directory: str | Path) -> Sample:
    """Read one control group's ceiling and counters from its directory.

    A group with no ``cpu.max`` file is reported rather than refused: it states
    no ceiling there, which is the state a compute node's user slice is always
    in and one side of the comparison this sampler exists to take. Its counters
    are read from ``cpu.stat`` as that text states them, because two groups with
    no ceiling file do not account the same periods — the root group states
    accounted zeros while a compute node's user slice states no period counter
    at all — and reporting either as the other would state a count the kernel
    did not keep.
    """
    base = Path(directory)
    stat_text = (base / _CPU_STAT_FILE).read_text()
    try:
        cpu_max_text = (base / _CPU_MAX_FILE).read_text()
    except FileNotFoundError:
        return Sample(
            cpu_max=_NO_CEILING,
            stat=parse_cpu_stat_without_ceiling(stat_text),
        )
    return Sample(
        cpu_max=parse_cpu_max(cpu_max_text),
        stat=parse_cpu_stat(stat_text),
    )


def read_host() -> HostIdentity:
    """The running machine's name, boot identifier and time since boot."""
    return HostIdentity(
        hostname=socket.gethostname(),
        boot_id=parse_boot_id(Path(_BOOT_ID_FILE).read_text()),
        uptime_seconds=parse_uptime(Path(_UPTIME_FILE).read_text()),
    )


def permitted_thread_usec(stat: CpuStat, cpu_max: CpuMax) -> int | None:
    """Thread-time the ceiling permitted over the periods accounted.

    A group allowed one quota per period may run that quota once each period,
    so the permitted total is the quota times the period count; the period
    itself cancels. A group with no ceiling permits an unbounded total, and one
    whose counters account no period permits nothing that can be totalled, so
    both report ``None`` rather than a total they cannot state.
    """
    if cpu_max.quota_usec is None:
        return None
    counters = stat.period_counters
    if counters is None:
        return None
    nr_periods, _, _ = counters
    return nr_periods * cpu_max.quota_usec


def throttled_share(stat: CpuStat, cpu_max: CpuMax) -> float | Unmeasured:
    """Fraction of permitted thread-time spent stalled at the ceiling.

    This can exceed one, and that is a property of the quantity rather than a
    fault: the stall counter sums over every task that was stopped, while the
    budget is one quota per period. Once more tasks are runnable than the quota
    admits, several of them stall together for the remainder of each period, so
    the thread-time lost can pass the thread-time the ceiling allowed.

    An :class:`Unmeasured` member means there is no fraction to state, and says
    which of the two absences applies. A group that is both unbounded and
    unaccounted reports ``UNBOUNDED``: no length of observation removes a
    ceiling's absence, so that is the cause a caller must act on.
    """
    permitted_usec = permitted_thread_usec(stat, cpu_max)
    if permitted_usec is None:
        return Unmeasured.UNBOUNDED
    if not permitted_usec:
        return Unmeasured.NO_PERIODS
    counters = stat.period_counters
    # A non-zero permitted total exists only where a quota and a period count
    # were both read, so the stall counter is accounted for alongside them.
    assert counters is not None
    return counters[2] / permitted_usec


def interval_delta(first: Sample, second: Sample) -> CpuStat:
    """Counters accumulated between two samples of the same control group.

    A group that accounts no period reports the usage it did accumulate with
    its period counters absent, which is what its two readings do state; the
    three are not defaulted to zero, which would read as a span over which the
    group was offered a ceiling and never reached it.
    """
    earlier, later = first.stat, second.stat
    before, after = earlier.period_counters, later.period_counters
    if before is None or after is None:
        if later.usage_usec < earlier.usage_usec:
            raise ValueError(
                "usage_usec went backwards between samples: "
                f"{earlier.usage_usec} -> {later.usage_usec}"
            )
        return CpuStat(
            usage_usec=later.usage_usec - earlier.usage_usec,
            nr_periods=None,
            nr_throttled=None,
            throttled_usec=None,
        )
    for field in _COUNTER_FIELDS:
        before_value: int = getattr(earlier, field)
        after_value: int = getattr(later, field)
        if after_value < before_value:
            raise ValueError(
                f"{field} went backwards between samples: "
                f"{before_value} -> {after_value}"
            )
    return CpuStat(
        usage_usec=later.usage_usec - earlier.usage_usec,
        nr_periods=after[0] - before[0],
        nr_throttled=after[1] - before[1],
        throttled_usec=after[2] - before[2],
    )


def interval_throttled_share(first: Sample, second: Sample) -> float | Unmeasured:
    """Throttled share over the span between two samples of one group.

    The ceiling is taken from the earlier sample, so the delta share describes
    the quota that was in force while those counters accumulated. A later
    reading of the same group reports its own ceiling, which is how a quota
    changed mid-span is visible rather than silently mixed into the figure.

    ``NO_PERIODS`` here means no period elapsed between the two samples, so the
    span holds no window to divide by; ``UNBOUNDED`` means the group the span
    began in had no ceiling.
    """
    return throttled_share(interval_delta(first, second), first.cpu_max)


def format_cpu_max(cpu_max: CpuMax) -> str:
    """Render a ceiling back into the form ``cpu.max`` uses.

    A group with no ceiling file states neither field and is rendered as
    ``absent``: ``max`` is a ceiling the group was given and this one has none,
    so a reader comparing two readings must be able to tell them apart.
    """
    if cpu_max.period_usec is None:
        return _ABSENT
    quota = _UNLIMITED if cpu_max.quota_usec is None else str(cpu_max.quota_usec)
    return f"{quota} {cpu_max.period_usec}"


def _counters_payload(stat: CpuStat, cpu_max: CpuMax) -> dict[str, object]:
    """Counters and share for one reading, shaped for JSON.

    An undefined share keeps its cause in the payload rather than becoming a
    ``null``: ``throttled_share`` carries the :class:`Unmeasured` reason as a
    string, so a reader of the JSON separates a group that cannot be throttled
    from one that has not been measured over a period yet.
    """
    share = throttled_share(stat, cpu_max)
    return {
        "usage_usec": stat.usage_usec,
        "nr_periods": stat.nr_periods,
        "nr_throttled": stat.nr_throttled,
        "throttled_usec": stat.throttled_usec,
        "permitted_usec": permitted_thread_usec(stat, cpu_max),
        "throttled_share": share.value if isinstance(share, Unmeasured) else share,
    }


def _host_payload(host: HostIdentity) -> dict[str, object]:
    """The machine a reading came from, shaped for JSON.

    The hostname is what lets a reader compare two shares at all: the counter
    directory resolves identically on every machine in the cluster, so without
    it two readings taken on different computers are indistinguishable and
    compare as though they described one. The boot identity and the uptime
    separate two readings of cumulative counters across a reboot, whose
    restart would otherwise read as a sudden improvement.
    """
    return {
        "hostname": host.hostname,
        "boot_id": host.boot_id,
        "uptime_seconds": host.uptime_seconds,
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


def main(argv: Sequence[str] | None = None, host: HostIdentity | None = None) -> int:
    """Read one control group and print its throttled share as JSON.

    The emitted payload names the machine the reading came from, and which boot
    of it, so two shares can be compared only when they describe one computer
    over one uninterrupted run of counters. The host identity is passed in
    rather than read here, as the control group is, so a caller decides which
    machine is named; it defaults to the machine the command runs on.
    """
    args = _parser().parse_args(argv)
    reading_host = read_host() if host is None else host
    first = read_sample(args.directory)
    payload: dict[str, object] = {
        "directory": args.directory,
        "host": _host_payload(reading_host),
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
