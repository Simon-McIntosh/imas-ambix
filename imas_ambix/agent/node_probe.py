"""The serving node's own readings: GPU cards, host, and its job table.

The continuous recorder samples the engine's ``/metrics`` every five seconds.
That record says how the engine behaved and nothing about the machine it
behaved on, so a slow hour cannot be explained from it: the cards may have been
busy with someone else's job, the host may have been short of memory, and the
node's job table is the only place that shows what else was resident. This
module produces those three readings, as ordinary local calls at the GPU node
inside the serve's own allocation -- ``nvidia-smi``, ``/proc`` and ``squeue``
are all present there, and none of them needs a ``srun`` step.

**A probe that did not run contributes no section.** Each reading is either a
populated mapping or nothing at all: a failed command (non-zero exit, absent
binary, timeout) yields ``None``, and so does a successful command whose output
carries no reading. That is the same rule the engine section follows, for the
same reason -- a field that is present means it was measured, so an unavailable
source must not appear as an empty list or a zero. A reading taken *partially*
keeps only the fields it observed, which is why every field below is omitted
rather than nulled when its own sample was unparseable (``nvidia-smi`` reports
``[N/A]`` for a quantity a card does not expose).

**Cards are labelled by the number the node knows them by.** SLURM restricts
each step to its allocated devices through the device cgroup, so ``nvidia-smi``
inside the step numbers the visible cards from zero while ``SLURM_STEP_GPUS``
carries the physical indices they actually are. A two-card step on a node's
cards 6 and 7 therefore reads ``0, 1`` from ``nvidia-smi`` and ``6, 7`` from the
environment, and recording the first would make one job's card 0 and another's
card 0 different silicon. :func:`parse_cards` maps the position it read to the
physical index, and the section states which numbering it used.

**This module probes; it does not decide how often.** :class:`NodeProbe` holds
the one piece of scheduling the readings need -- the job table is a SLURM RPC
and changes on the scale of minutes, so it is read on its own much lower
cadence and omitted from the ticks in between, where its reading was not taken.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

#: ``nvidia-smi`` query field for each card quantity, and the name it is
#: recorded under. The names carry their unit because the query returns bare
#: numbers under ``nounits`` and a later reader has no other way to know one.
CARD_FIELDS: tuple[tuple[str, str], ...] = (
    ("utilisation_percent", "utilization.gpu"),
    ("temperature_c", "temperature.gpu"),
    ("power_draw_w", "power.draw"),
    ("power_cap_w", "power.limit"),
    ("sm_clock_mhz", "clocks.sm"),
    ("mem_clock_mhz", "clocks.mem"),
    ("memory_used_mib", "memory.used"),
    ("memory_total_mib", "memory.total"),
)

#: The ``index`` column is read for the card's own numbering and is not a
#: quantity; it is queried first so the mapping below can pair by position.
CARD_INDEX_FIELD = "index"

CARD_QUERY_FIELDS: tuple[str, ...] = (
    CARD_INDEX_FIELD,
    *(field_ for _name, field_ in CARD_FIELDS),
)

#: Values ``nvidia-smi`` prints where a card does not expose a quantity. Both
#: spellings arrive inside brackets; anything unparseable as a float is treated
#: the same way, since an unreadable field is not a reading.
_UNAVAILABLE = frozenset({"N/A", "NA", "not supported", "unknown", "[n/a]"})

#: A runner takes an argument vector and returns the command's stdout, or
#: ``None`` when the command could not be run or did not succeed. Nothing in
#: this module raises for a failed probe.
RunFn = Callable[[Sequence[str]], "str | None"]

#: Everything a failed local call raises: a binary that is not on the node, a
#: call that could not be started, and one that outlived its timeout. Held as a
#: name because the serving interpreter requires the parenthesised form of a
#: multi-exception clause and the repository's formatter strips those
#: parentheses back off against its newer syntax target -- a name is stable
#: under both.
_CALL_FAILURES = (OSError, subprocess.SubprocessError)


def run_capture(argv: Sequence[str], timeout_s: float = 10.0) -> str | None:
    """Run *argv* locally and return its stdout, or ``None`` if it failed.

    A non-zero exit is a failure even when the command printed something: the
    probe reports what a command measured, and a command that did not succeed
    did not measure. A missing binary and a hang are the same kind of absence.
    """
    try:
        completed = subprocess.run(
            list(argv),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except _CALL_FAILURES:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


# ── GPU cards ────────────────────────────────────────────────────────


def parse_gpu_index_list(value: str | None) -> list[int] | None:
    """Physical GPU indices from a SLURM index list, or ``None`` if unusable.

    SLURM writes these as a comma-separated list which may carry ``low-high``
    ranges (``0-3`` and ``0,2-3`` both occur). ``None`` means the environment
    did not state the allocation, which is a different fact from an empty one.
    """
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    indices: list[int] = []
    for part in text.split(","):
        token = part.strip()
        if not token:
            continue
        low, sep, high = token.partition("-")
        try:
            if not sep:
                indices.append(int(low))
                continue
            first, last = int(low), int(high)
        except ValueError:
            return None
        if last < first:
            return None
        indices.extend(range(first, last + 1))
    return indices or None


def step_gpu_indices(env: Mapping[str, str] | None = None) -> list[int] | None:
    """Physical indices of this step's cards, from ``SLURM_STEP_GPUS``."""
    source = os.environ if env is None else env
    return parse_gpu_index_list(source.get("SLURM_STEP_GPUS"))


def _card_value(token: str) -> float | None:
    """One ``nvidia-smi`` column as a number, or ``None`` when unavailable."""
    text = token.strip()
    if not text or text.lower().strip("[]") in _UNAVAILABLE:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def mapped_indices(
    count: int, physical_indices: Sequence[int] | None
) -> list[int] | None:
    """The physical index of each read position, or ``None`` if unmappable.

    The allocation describes the whole step, so it pairs with the cards read
    only when the two agree in length. A partial pairing would label a card
    with another job's silicon, which is worse than not labelling it at all --
    so a mismatch yields ``None`` and the caller keeps the read numbering.
    """
    if physical_indices is None:
        return None
    indices = list(physical_indices)
    return indices if len(indices) == count else None


def parse_cards(
    text: str, *, physical_indices: Sequence[int] | None = None
) -> list[dict[str, Any]]:
    """Per-card readings from one ``nvidia-smi --query-gpu`` body.

    *physical_indices* is the allocation the node's scheduler stated, in the
    order ``nvidia-smi`` reports the step's cards. A card is labelled with the
    physical index it maps to; when no allocation was stated, or when its
    length does not describe the cards actually read, the cards keep the
    numbering they were read under and no index is invented.

    Fields a card does not expose are omitted from that card rather than
    recorded as null, so a present key is a measurement.
    """
    cards: list[dict[str, Any]] = []
    for line in text.splitlines():
        row = [cell.strip() for cell in line.strip().split(",")]
        if len(row) < len(CARD_QUERY_FIELDS):
            continue
        try:
            step_index = int(row[0])
        except ValueError:
            continue
        card: dict[str, Any] = {"step_index": step_index}
        for (name, _field), token in zip(CARD_FIELDS, row[1:], strict=False):
            value = _card_value(token)
            if value is not None:
                card[name] = value
        cards.append(card)
    indices = mapped_indices(len(cards), physical_indices)
    for position, card in enumerate(cards):
        card["index"] = indices[position] if indices is not None else card["step_index"]
    return cards


def read_cards(
    run: RunFn = run_capture,
    *,
    env: Mapping[str, str] | None = None,
    timeout_s: float = 10.0,
) -> dict[str, Any] | None:
    """One ``nvidia-smi`` query as the row's card reading, or ``None``.

    All fields come from a single query, so a card's readings describe one
    instant rather than several a query apart.
    """
    text = run(
        (
            "nvidia-smi",
            f"--query-gpu={','.join(CARD_QUERY_FIELDS)}",
            "--format=csv,noheader,nounits",
        )
    )
    if not text:
        return None
    allocation = step_gpu_indices(env)
    cards = parse_cards(text, physical_indices=allocation)
    if not cards:
        return None
    mapped = mapped_indices(len(cards), allocation) is not None
    section: dict[str, Any] = {
        "index_source": "step_gpus" if mapped else "nvidia-smi",
        "count": len(cards),
        "cards": cards,
    }
    raw = (os.environ if env is None else env).get("SLURM_STEP_GPUS")
    if raw:
        # Recorded because it is what makes an unmapped section explicable: a
        # reader sees the allocation beside the numbering that was used.
        section["step_gpus"] = raw
    return section


# ── Host ─────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class CpuTimes:
    """Aggregate host CPU jiffies from ``/proc/stat``'s ``cpu`` line."""

    total: float
    idle: float

    @property
    def busy(self) -> float:
        return self.total - self.idle


def parse_proc_stat(text: str) -> CpuTimes | None:
    """The aggregate ``cpu`` line of ``/proc/stat``, or ``None`` if absent.

    The line's columns are cumulative jiffies: user, nice, system, idle, iowait
    and the interrupt/steal family. Only the line's presence matters, not its
    width -- the trailing softirq columns are optional and are summed when
    present. ``iowait`` counts as not busy, so the fraction is the conventional
    CPU-busy reading rather than one that blames the CPU for waiting on I/O.
    """
    for line in text.splitlines():
        fields = line.split()
        if not fields or fields[0] != "cpu":
            continue
        try:
            values = [float(field) for field in fields[1:]]
        except ValueError:
            return None
        if len(values) < 5:
            return None
        idle = values[3] + values[4]
        return CpuTimes(total=sum(values), idle=idle)
    return None


def cpu_busy_fraction(
    previous: CpuTimes | None, current: CpuTimes | None
) -> float | None:
    """Busy fraction over the interval between two samples, or ``None``.

    The counters are cumulative, so a fraction needs two samples and the first
    tick of a run has none. Reporting a lifetime average as an instantaneous
    reading is the error this refuses; a zero-length interval (two reads of one
    unchanged file) is refused for the same reason a rate needs an interval.
    """
    if previous is None or current is None:
        return None
    elapsed = current.total - previous.total
    if elapsed <= 0:
        return None
    busy = current.busy - previous.busy
    if busy < 0:
        return None
    return round(min(busy / elapsed, 1.0), 6)


def parse_meminfo(text: str) -> dict[str, float]:
    """Host memory in MiB from ``/proc/meminfo``.

    Values are recorded in kibibytes there; they are converted once here so
    every field of a row is in the unit its name states. A key the file does
    not carry is absent from the result, as everywhere else.
    """
    wanted = {
        "MemTotal": "memory_total_mib",
        "MemAvailable": "memory_available_mib",
        "MemFree": "memory_free_mib",
        "SwapTotal": "swap_total_mib",
        "SwapFree": "swap_free_mib",
    }
    observed: dict[str, float] = {}
    for line in text.splitlines():
        key, sep, rest = line.partition(":")
        if not sep:
            continue
        name = wanted.get(key.strip())
        if name is None:
            continue
        token = rest.strip().split()
        if not token:
            continue
        try:
            observed[name] = round(float(token[0]) / 1024.0, 3)
        except ValueError:
            continue
    total = observed.get("memory_total_mib")
    available = observed.get("memory_available_mib")
    if total is not None and available is not None and total > 0:
        observed["memory_used_mib"] = round(total - available, 3)
        observed["memory_used_percent"] = round((total - available) / total * 100.0, 4)
    return observed


def build_host_section(
    times: CpuTimes | None,
    memory: dict[str, float],
    previous_cpu: CpuTimes | None = None,
) -> dict[str, Any] | None:
    """The host reading from two parsed sources, or ``None`` when neither ran.

    The sources are independent, so a host whose ``meminfo`` was unreadable
    still contributes its CPU counters, and a tick where neither answered
    contributes no section. The busy fraction needs a preceding sample, so the
    first tick of a run reports the cumulative jiffies that make one derivable
    later without reporting a fraction it does not have.
    """
    if times is None and not memory:
        return None
    section: dict[str, Any] = dict(memory)
    if times is not None:
        # Cumulative jiffies are carried beside the fraction so a reader can
        # difference any two rows itself -- a window a missed tick landed in is
        # otherwise unrecoverable from the per-row fractions alone.
        section["cpu_busy_jiffies"] = times.busy
        section["cpu_total_jiffies"] = times.total
        fraction = cpu_busy_fraction(previous_cpu, times)
        if fraction is not None:
            section["cpu_busy_fraction"] = fraction
    return section


def read_cpu_times(
    run: RunFn = run_capture, *, timeout_s: float = 10.0
) -> CpuTimes | None:
    """One read of ``/proc/stat``'s aggregate CPU counters."""
    text = run(("cat", "/proc/stat"))
    return parse_proc_stat(text) if text else None


def read_meminfo(
    run: RunFn = run_capture, *, timeout_s: float = 10.0
) -> dict[str, float]:
    """One read of ``/proc/meminfo`` in MiB, empty when it could not be read."""
    text = run(("cat", "/proc/meminfo"))
    return parse_meminfo(text) if text else {}


def read_host(
    run: RunFn = run_capture,
    *,
    previous_cpu: CpuTimes | None = None,
    timeout_s: float = 10.0,
) -> dict[str, Any] | None:
    """Host CPU and memory as the row's host reading, or ``None``."""
    return build_host_section(
        read_cpu_times(run, timeout_s=timeout_s),
        read_meminfo(run, timeout_s=timeout_s),
        previous_cpu,
    )


# ── Node job table ───────────────────────────────────────────────────

#: One pipe-separated column per recorded job field. The widths are not fixed
#: so a long job name cannot shift a neighbour.
SQUEUE_FORMAT = "%i|%u|%j|%T|%P|%C|%m|%b|%M"
SQUEUE_FIELDS: tuple[str, ...] = (
    "job_id",
    "user",
    "name",
    "state",
    "partition",
    "cpus",
    "memory",
    "gres",
    "elapsed",
)


def parse_squeue(text: str) -> list[dict[str, Any]]:
    """One row per job of a ``squeue`` body, empty when the node holds none.

    ``cpus`` is recorded as the integer the scheduler reported; memory, ``gres``
    and elapsed time stay as the strings SLURM printed (``640G``, ``gpu:2``,
    ``[DD-]HH:MM:SS``), because re-encoding them here would add a parsing
    failure mode to the record with nothing gained.
    """
    jobs: list[dict[str, Any]] = []
    for line in text.splitlines():
        columns = line.rstrip("\n").split("|")
        if len(columns) < len(SQUEUE_FIELDS):
            continue
        job = dict(zip(SQUEUE_FIELDS, (cell.strip() for cell in columns), strict=False))
        try:
            job["cpus"] = int(job["cpus"])
        except ValueError:
            # Every field above is present, so this is SLURM printing a count
            # this parser cannot read; the row keeps the rest of its reading.
            job.pop("cpus")
        jobs.append(job)
    return jobs


def read_jobs(
    hostname: str,
    run: RunFn = run_capture,
    *,
    timeout_s: float = 15.0,
) -> dict[str, Any] | None:
    """The node's job table as the row's jobs reading, or ``None``.

    Scoped to *hostname* rather than to a user or an account, because the
    question the record answers is what else was resident on this node. A node
    with no jobs is a reading and is kept; a ``squeue`` that failed is not.
    """
    text = run(("squeue", "-h", "-w", hostname, "-o", SQUEUE_FORMAT))
    if text is None:
        return None
    jobs = parse_squeue(text)
    return {"hostname": hostname, "count": len(jobs), "jobs": jobs}


# ── The tick's readings ──────────────────────────────────────────────


@dataclass
class NodeProbe:
    """The three node readings, at the cadences their costs allow.

    The cards and the host are cheap local reads on every tick. The job table
    is a SLURM RPC, so it is read at *job_interval_s* and omitted from the
    ticks between -- omitted rather than repeated, because a tick that repeats
    an earlier reading records an old measurement as a current one.
    """

    job_interval_s: float = 60.0
    timeout_s: float = 10.0
    env: Mapping[str, str] | None = None
    run: RunFn = run_capture
    hostname: str = field(default_factory=lambda: os.uname().nodename)
    _previous_cpu: CpuTimes | None = field(default=None, init=False, repr=False)
    _jobs_at: float | None = field(default=None, init=False, repr=False)

    def sample(self, now: float) -> dict[str, Any]:
        """Every section due at *now*, as the sparse mapping of the row.

        *now* is a monotonic stamp the caller owns, so the cadence is testable
        without waiting for a clock.
        """
        sections: dict[str, Any] = {}

        cards = read_cards(self.run, env=self.env, timeout_s=self.timeout_s)
        if cards is not None:
            sections["cards"] = cards

        times = read_cpu_times(self.run, timeout_s=self.timeout_s)
        memory = read_meminfo(self.run, timeout_s=self.timeout_s)
        host = build_host_section(times, memory, self._previous_cpu)
        if host is not None:
            sections["host"] = host
        if times is not None:
            self._previous_cpu = times

        if self._jobs_at is None or now - self._jobs_at >= self.job_interval_s:
            jobs = read_jobs(self.hostname, self.run, timeout_s=self.timeout_s)
            # A failed job-table read does not retry on the next tick: it is a
            # SLURM RPC, and the following one is due on the same clock either
            # way, so a node refusing squeue is not asked twice a second.
            self._jobs_at = now
            if jobs is not None:
                sections["jobs"] = jobs

        return sections
