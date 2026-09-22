"""The serving node's own readings: GPU cards, host, and its job table.

The continuous recorder samples the engine's ``/metrics`` every five seconds.
That record says how the engine behaved and nothing about the machine it
behaved on, so a slow hour cannot be explained from it: the cards may have been
busy with someone else's job, the host may have been short of memory, and the
node's job table is the only place that shows what else was resident. This
module produces those three readings, as ordinary local calls at the GPU node
inside the serve's own allocation -- ``nvidia-smi``, ``/proc`` and ``squeue``
are all present there, and none of them needs a ``srun`` step.

**A probe that did not run contributes no section -- with one exception, the
job table, where an unread table is itself a reading about the node.** Each
other reading is either a populated mapping or nothing at all: a failed command
(non-zero exit, timeout) yields ``None``, and so does a successful command whose
output carries no reading. That is the same rule the engine section follows, for
the same reason -- a field that is present means it was measured, so an
unavailable source must not appear as an empty list or a zero. A reading taken
*partially* keeps only the fields it observed, which is why every field below is
omitted rather than nulled when its own sample was unparseable (``nvidia-smi``
reports ``[N/A]`` for a quantity a card does not expose).

**Why the job table is the exception.** The record exists to explain an hour a
serve spent slow, and "the scheduler refused this node's query" is a different
fact from "this node held no jobs": the first says there were neighbours whose
names this record does not carry, the second says there were none. An omitted
section conflates both with a tick where the table was not yet due, so
:func:`read_jobs` records how its read resolved -- a :data:`UNREAD_KEY` field
naming :data:`COMMAND_FAILED` or :data:`COMMAND_ABSENT` -- and only the table's
own *fields* follow the omit-when-unmeasured rule above. Both resolutions are
stored, following the shape a refused boot identity already uses in this spine:
never a refusal to record, and the marker says which degraded case the reading
is.

**Cards are labelled by the number the node knows them by.** SLURM restricts
each process to its allocated devices through the device cgroup, so
``nvidia-smi`` inside it numbers the visible cards from zero while the
allocation carries the physical indices they actually are. A two-card serve on
a node's cards 2 and 3 therefore reads ``0, 1`` from ``nvidia-smi`` and ``2, 3``
from the environment, and recording the first would make one job's card 0 and
another's card 0 different silicon. :func:`parse_cards` maps the position it
read to the physical index, and the section states which numbering it used.

The allocation is read from the variable that exists on the path the serve takes.
A serve is submitted with ``sbatch`` and runs the engine inline in the batch
step, so ``SLURM_STEP_GPUS`` is **not set for it** -- that variable belongs to an
``srun`` step -- while ``SLURM_JOB_GPUS`` carries the batch step's allocation.
:data:`GPU_ALLOCATION_ENV` is the precedence ladder: the step variable first,
because it is the narrower allocation when a serve is launched under ``srun``
inside a job, and the job variable otherwise. ``CUDA_VISIBLE_DEVICES`` is
deliberately not on that ladder: SLURM remaps it to ``0..N-1`` for the process,
so it states the position in the visible set rather than the silicon, which is
the confusion the map exists to remove.

**This module probes; it does not decide how often.** :class:`NodeProbe` holds
the one piece of scheduling the readings need -- the job table is a SLURM RPC
and changes on the scale of minutes, so it is read on its own much lower
cadence and omitted from the ticks in between, where its reading was not taken.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

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


class RunFn(Protocol):
    """A runner takes an argument vector and a timeout, and returns stdout.

    ``None`` is the answer for a command that did not succeed or that outlived
    *timeout_s*. A program the node does not carry at all is raised as
    ``FileNotFoundError`` instead -- the one failure whose reason only the spawn
    knows, and the one that :func:`resolve_command` turns into a marker rather
    than propagating. No read in this module raises for a failed probe.

    The timeout is part of the call rather than the runner's own construction
    because it is the caller that knows how long this reading may take: a
    probe holding a recorder tick open must be bounded, and a runner that
    cannot be told is a knob that does nothing.
    """

    def __call__(self, argv: Sequence[str], *, timeout_s: float) -> str | None: ...


#: Everything a failed local call raises on its own: a call that could not be
#: started and one that outlived its timeout. Held as a name because the
#: serving interpreter requires the parenthesised form of a multi-exception
#: clause and the repository's formatter strips those parentheses back off
#: against its newer syntax target -- a name is stable under both.
#:
#: ``FileNotFoundError`` is deliberately not in this tuple even though it is an
#: ``OSError``: it is the one failure that says *why* the call did not happen,
#: and swallowing it here is what made a node with no ``squeue`` on it
#: indistinguishable from one whose ``squeue`` refused the query.
_CALL_FAILURES = (OSError, subprocess.SubprocessError)


def run_capture(argv: Sequence[str], *, timeout_s: float = 10.0) -> str | None:
    """Run *argv* locally and return its stdout, or ``None`` if it failed.

    A non-zero exit is a failure even when the command printed something: the
    probe reports what a command measured, and a command that did not succeed
    did not measure. A hang is the same kind of absence.

    A program this node does not carry is raised rather than answered: the
    spawn's own exception is the only place that fact exists, and
    :func:`resolve_command` is where it becomes a marker.
    """
    try:
        completed = subprocess.run(
            list(argv),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except FileNotFoundError:
        raise
    except _CALL_FAILURES:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


#: The field a job section names its resolution under when the table could not
#: be read, and the two resolutions that field can carry. Present only when
#: there is no table: a section holding rows states its resolution by holding
#: them, on the rule every other field here follows.
UNREAD_KEY = "unread"

#: ``squeue`` ran and refused -- a non-zero exit, or a call that outlived its
#: bound. The scheduler was asked and declined, so the node's job table exists
#: and this record does not carry it.
COMMAND_FAILED = "command-failed"

#: No ``squeue`` on this node at all. The question could not be put, which is a
#: different fact from an answer that was declined and is worth telling apart:
#: a refused query is worth retrying on a node whose scheduler was momentarily
#: busy, and a missing command never is.
COMMAND_ABSENT = "command-absent"


def resolve_command(
    argv: Sequence[str], run: RunFn, *, timeout_s: float
) -> tuple[str | None, str | None]:
    """``(stdout, unread marker)`` for one local call.

    ``(text, None)`` when the command answered with output, and
    ``(None, marker)`` when it did not, the marker naming which kind of
    not-answering it was -- :data:`COMMAND_FAILED` for a command that ran and
    refused, :data:`COMMAND_ABSENT` for a program this node does not carry.

    The two are separable only at this point. A runner reports both as no
    output, and only the spawn knows why: a missing program raises
    ``FileNotFoundError`` while a command that ran and refused returns non-zero.
    A caller that records a reading *of the node* needs the distinction (see
    :func:`read_jobs`); a caller whose field is simply absent either way may
    ignore the marker, which is what the card and host reads do.
    """
    try:
        text = run(argv, timeout_s=timeout_s)
    except FileNotFoundError:
        return None, COMMAND_ABSENT
    if text is None:
        return None, COMMAND_FAILED
    return text, None


# ── GPU cards ────────────────────────────────────────────────────────

#: Environment variables that state the physical devices this process holds, in
#: precedence order. Both are SLURM's own statement of the allocation rather
#: than a view derived from it, which is what makes either usable as the
#: physical numbering; see the module docstring for why one of them is absent on
#: the serve's launch path and why ``CUDA_VISIBLE_DEVICES`` cannot substitute.
GPU_ALLOCATION_ENV: tuple[str, ...] = ("SLURM_STEP_GPUS", "SLURM_JOB_GPUS")


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


def gpu_allocation(
    env: Mapping[str, str] | None = None,
) -> tuple[str, list[int]] | None:
    """The variable stating this process's cards, and the indices in it.

    Returns ``(variable, indices)`` so the section can name the source that
    labelled its cards, or ``None`` when no variable stated an allocation. The
    indices come back ascending, because ``nvidia-smi`` enumerates the cards it
    sees in ascending device order and the pairing below is positional: an
    allocation printed in another order would otherwise label the wrong
    silicon.
    """
    source = os.environ if env is None else env
    for name in GPU_ALLOCATION_ENV:
        indices = parse_gpu_index_list(source.get(name))
        if indices is not None:
            return name, sorted(indices)
    return None


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

    The allocation describes the whole process, so it pairs with the cards read
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

    *physical_indices* is the allocation the scheduler stated, ascending, which
    is the order ``nvidia-smi`` reports the cards it can see. A card is labelled
    with the physical index it maps to; when no allocation was stated, or when
    its length does not describe the cards actually read, the cards keep the
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
            read_index = int(row[0])
        except ValueError:
            continue
        card: dict[str, Any] = {"read_index": read_index}
        for (name, _field), token in zip(CARD_FIELDS, row[1:], strict=False):
            value = _card_value(token)
            if value is not None:
                card[name] = value
        cards.append(card)
    indices = mapped_indices(len(cards), physical_indices)
    for position, card in enumerate(cards):
        card["index"] = indices[position] if indices is not None else card["read_index"]
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
    text, _unread = resolve_command(
        (
            "nvidia-smi",
            f"--query-gpu={','.join(CARD_QUERY_FIELDS)}",
            "--format=csv,noheader,nounits",
        ),
        run,
        timeout_s=timeout_s,
    )
    if not text:
        return None
    source = os.environ if env is None else env
    stated = gpu_allocation(source)
    allocation = stated[1] if stated is not None else None
    cards = parse_cards(text, physical_indices=allocation)
    if not cards:
        return None
    mapped = mapped_indices(len(cards), allocation) is not None
    section: dict[str, Any] = {
        "index_source": stated[0] if mapped else "nvidia-smi",
        "count": len(cards),
        "cards": cards,
    }
    if stated is not None:
        # Recorded because it is what makes an unmapped section explicable: a
        # reader sees the allocation beside the numbering that was used. The
        # variable that carried it is named because the ladder has two rungs,
        # and which one answered decides how wide an allocation this is.
        section["allocation"] = {
            "variable": stated[0],
            "value": source.get(stated[0]),
        }
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
    text, _unread = resolve_command(("cat", "/proc/stat"), run, timeout_s=timeout_s)
    return parse_proc_stat(text) if text else None


def read_meminfo(
    run: RunFn = run_capture, *, timeout_s: float = 10.0
) -> dict[str, float]:
    """One read of ``/proc/meminfo`` in MiB, empty when it could not be read."""
    text, _unread = resolve_command(("cat", "/proc/meminfo"), run, timeout_s=timeout_s)
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


def scheduler_node_name(nodename: str) -> str:
    """The name the scheduler knows a node by, from the node's own nodename.

    A node's own nodename is its fully qualified domain name -- ``hostname``
    and ``os.uname().nodename`` both report ``98dci4-gpu-0003.iter.org`` on the
    serving node -- while the scheduler's node names are the short host names,
    which is the same string cut at the first dot. ``squeue -w`` does not accept
    the qualified spelling quietly: it refuses it with ``squeue: error: Invalid
    node name 98dci4-gpu-0003.iter.org`` and exit 1, so the job table of any
    node whose nodename carries a domain has never been read, and its absence
    from the record was indistinguishable from a node holding nothing.

    The cut is the whole transformation because it is the relation the two
    spellings have rather than a guess at one: a nodename carrying no dot is
    already the short name and comes back unchanged, so a node reporting either
    spelling is queried under the name the scheduler has for it.
    """
    return nodename.split(".", 1)[0]


def read_jobs(
    hostname: str,
    run: RunFn = run_capture,
    *,
    timeout_s: float = 15.0,
) -> dict[str, Any]:
    """The node's job table as the row's jobs reading.

    Scoped to *hostname* rather than to a user or an account, because the
    question the record answers is what else was resident on this node. A node
    with no jobs is a reading and is kept.

    *hostname* is the recorded identity and is kept verbatim, because a reader
    joins rows on the node's fully qualified name. The query is put to the
    scheduler under :func:`scheduler_node_name`, which is a different string:
    the two spellings are not interchangeable at the wire, and passing the
    recorded one to ``squeue`` is what left this section unread on the serving
    node.

    A ``squeue`` that did not answer still contributes a section, carrying a
    :data:`UNREAD_KEY` marker that says whether the scheduler refused the query
    or the node has no ``squeue`` at all -- see the module docstring for why the
    job table is the one reading that records its own non-answer. Such a section
    holds no ``count`` and no ``jobs``, so it cannot be read as an empty table.

    *timeout_s* bounds a command that crosses the scheduler, so the default is
    wider than the local reads' -- see ``NodeProbe.job_timeout_s``, which is
    what the probe passes here.
    """
    argv = ("squeue", "-h", "-w", scheduler_node_name(hostname), "-o", SQUEUE_FORMAT)
    text, unread = resolve_command(argv, run, timeout_s=timeout_s)
    if text is None:
        return {"hostname": hostname, UNREAD_KEY: unread}
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

    *timeout_s* bounds each local read. *job_timeout_s* bounds the job-table
    RPC, which reaches the scheduler rather than a file or the driver, so it
    is allowed longer -- its own bound, and the one the job read is invoked
    with, rather than a figure nothing could reach.
    """

    job_interval_s: float = 60.0
    timeout_s: float = 10.0
    job_timeout_s: float = 15.0
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
            # read_jobs always answers a section, so an unread table reaches the
            # row here rather than being dropped -- see the module docstring.
            sections["jobs"] = read_jobs(
                self.hostname, self.run, timeout_s=self.job_timeout_s
            )
            # A failed job-table read does not retry on the next tick: it is a
            # SLURM RPC, and the following one is due on the same clock either
            # way, so a node refusing squeue is not asked twice a second.
            self._jobs_at = now

        return sections
