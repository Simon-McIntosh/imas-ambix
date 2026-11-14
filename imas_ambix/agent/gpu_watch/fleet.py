"""The GPU servers: which exist, what the scheduler says, what their cards read.

A server is one scheduler node with cards. The set is discovered, not listed:
every node of the site's GPU partitions that reports a ``gpu`` resource is a
server, with its card family and count read from that resource, so a node added
to a partition appears with no change here. A server the site declares
(``AMBIX_AGENT_SERVERS``) is added to that set and its declared family and
count win.

Card readings come from a sampler running as a step inside this account's own
jobs on the node. SLURM cgroups hide another job's devices, so only the cards
of this account's jobs are read; the rest are known from the scheduler as
allocated or free, never as a utilisation.

Three cadences keep the cost flat however many servers there are. The
scheduler is asked every ``POLL_SECONDS`` on one thread: one ``sinfo`` and two
``squeue`` calls for the whole fleet, and one ``scontrol`` per job when it is
first seen and again when it starts. The queue line alone does not say how
many cards a job holds: its resource field is empty for a job that asked for
cards per job and not per node, so the count comes from the job's detail. Each
sampler streams one line per card per second. The caller takes a
:meth:`Fleet.tick` once a second, which only copies what is already known and
extends the rolling means.
"""

from __future__ import annotations

import contextlib
import getpass
import os
import re
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field

SAMPLER = os.path.join(os.path.dirname(os.path.realpath(__file__)), "sampler.py")
USER = getpass.getuser()

#: Seconds between scheduler reads. The queue and the node state move on the
#: scale of jobs starting and ending, not of a screen refresh.
POLL_SECONDS = 5.0

#: Seconds before a sampler that exited is started again for the same job. A
#: step is a bounded resource of the job it joins, so a sampler that keeps
#: failing is retried at this pace and not once a tick.
RELAUNCH_SECONDS = 20.0

#: Comma-separated partitions to watch besides the site's own. Set, it
#: replaces :data:`OTHER_PARTITIONS`; set empty, only the site's are watched.
PARTITIONS_ENV = "AMBIX_WATCH_PARTITIONS"

#: The partitions watched besides the site's own when the environment names
#: none: the other GPU servers of the cluster.
OTHER_PARTITIONS = ("titan",)

_GRES = re.compile(r"gpu:(?:([^:(,]+):)?(\d+)")
_GROUP = re.compile(r"grp([a-z])$", re.IGNORECASE)
_ASKED = re.compile(r"ReqTRES=\S*gres/gpu=(\d+)")
_JOB_FORMAT = "%i|%u|%a|%j|%P|%T|%M|%b|%v|%N"
_NODE_FORMAT = "%N|%P|%G|%t|%E"


def short_name(node: str) -> str:
    """A node as the pages print it: no domain and no site prefix."""
    from imas_ambix.agent.registry import host_name

    return host_name(node)


@dataclass(frozen=True)
class Server:
    """One scheduler node with cards."""

    node: str
    partition: str
    family: str | None = None
    cards: int | None = None

    @property
    def name(self) -> str:
        return short_name(self.node)

    @property
    def hardware(self) -> str:
        """The cards as ``8×H200``, or as much of that as is known."""
        if self.cards and self.family:
            return f"{self.cards}×{self.family}"
        return self.family or (f"{self.cards} cards" if self.cards else "—")


@dataclass(frozen=True)
class Job:
    """One scheduler job, as the queue lists it."""

    id: str
    user: str
    account: str
    name: str
    partition: str
    state: str
    elapsed: str
    gres: str
    reservation: str
    node: str

    @property
    def running(self) -> bool:
        return self.state.upper().startswith("R")

    @property
    def pending(self) -> bool:
        return self.state.upper() in ("PENDING", "PD")

    @property
    def gpus(self) -> int:
        """Cards the queue line names, which is none for a per-job request."""
        found = _GRES.search(self.gres)
        return int(found.group(2)) if found else 0

    @property
    def group(self) -> str | None:
        """The reservation group the job belongs to, as a single letter."""
        for text in (self.account, self.reservation):
            found = _GROUP.search(text or "")
            if found:
                return found.group(1).upper()
        return None


@dataclass
class Card:
    """One card's reading. A sensor that did not report is None, never zero."""

    index: int
    util: float
    util_mean: float
    mem: float
    temp: float | None = None
    watts: float | None = None
    cap_watts: float | None = None


@dataclass
class Reading:
    """Everything known about one server at one tick."""

    server: Server
    state: str = "unknown"
    reason: str = ""
    jobs: list[Job] = field(default_factory=list)
    allocated: set[int] = field(default_factory=set)
    #: Card indices each running job holds on this server, by job id.
    job_cards: dict[str, set[int]] = field(default_factory=dict)
    cards: list[Card] = field(default_factory=list)
    #: Cards each job asked for, by job id, from the job's own detail.
    asked: dict[str, int] = field(default_factory=dict)
    #: ``(cpu % now, cpu % mean, ram %)`` of the host, when a sampler reports.
    host: tuple[float, float, float] | None = None
    group: str | None = None

    @property
    def running(self) -> list[Job]:
        return [job for job in self.jobs if job.running]

    @property
    def pending(self) -> list[Job]:
        return [job for job in self.jobs if job.pending]

    def gpus(self, job: Job) -> int:
        """Cards a job holds here, or asked for while it waits."""
        return (
            len(self.job_cards.get(job.id) or ()) or self.asked.get(job.id) or job.gpus
        )

    @property
    def allocated_count(self) -> int:
        """Cards running jobs hold, from the scheduler, which needs no cgroup."""
        return sum(self.gpus(job) for job in self.running)

    @property
    def unreadable(self) -> list[int]:
        """Allocated cards no sampler of this account can see."""
        seen = {card.index for card in self.cards}
        return sorted(self.allocated - seen)


def parse_nodes(text: str) -> list[tuple[Server, str, str]]:
    """``sinfo`` lines as ``(server, state, reason)``, one per node with cards."""
    found: dict[str, tuple[Server, str, str]] = {}
    for line in text.splitlines():
        parts = [part.strip() for part in line.split("|")]
        if len(parts) < 4 or not parts[0]:
            continue
        node, partition, gres, state = parts[:4]
        match = _GRES.search(gres)
        if match is None or node in found:
            continue
        family = match.group(1).upper() if match.group(1) else None
        reason = parts[4] if len(parts) > 4 and parts[4].lower() != "none" else ""
        found[node] = (
            Server(node, partition.rstrip("*"), family, int(match.group(2))),
            state,
            reason,
        )
    return list(found.values())


def parse_jobs(text: str) -> list[Job]:
    """``squeue`` lines in the fleet's own format as jobs."""
    jobs = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split("|")]
        if len(parts) < 10 or not parts[0]:
            continue
        jobs.append(Job(*parts[:10]))
    return jobs


def parse_card_indices(text: str) -> dict[str, set[int]]:
    """Card indices a job holds on each node, from ``scontrol show job -d``.

    The detail lines read ``Nodes=<node> CPU_IDs=… GRES=gpu:h200:4(IDX:4-7)``.
    """
    held: dict[str, set[int]] = {}
    for line in text.splitlines():
        if "IDX:" not in line or "Nodes=" not in line:
            continue
        node = line.split("Nodes=", 1)[1].split()[0]
        spec = line.split("IDX:", 1)[1].split(")", 1)[0]
        indices = held.setdefault(node, set())
        for part in spec.split(","):
            low, _, high = part.strip().partition("-")
            if low.isdigit() and (not high or high.isdigit()):
                indices.update(range(int(low), int(high or low) + 1))
    return held


def parse_asked_cards(text: str) -> int:
    """Cards a job asked for, from ``scontrol show job``; zero for none."""
    found = _ASKED.search(text)
    return int(found.group(1)) if found else 0


def sampled_jobs(
    jobs: list[Job], held: dict[str, dict[str, set[int]]], nodes: object
) -> dict[str, str]:
    """This account's jobs to sample through, as ``{job id: node}``.

    A sampler sees only the cards of the job it joins, so every job of this
    account that holds cards gets one. A server where this account runs only
    jobs without cards gets one sampler, for the host figures.
    """
    mine = [
        job for job in jobs if job.running and job.user == USER and job.node in nodes
    ]
    chosen = {job.id: job.node for job in mine if held.get(job.id, {}).get(job.node)}
    covered = set(chosen.values())
    for job in mine:
        if job.node not in covered:
            chosen[job.id] = job.node
            covered.add(job.node)
    return chosen


def _number(text: str) -> float | None:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _run(command: list[str], timeout: float = 20.0) -> str:
    try:
        return subprocess.run(
            command, capture_output=True, text=True, timeout=timeout
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def watched_partitions() -> tuple[tuple[str, ...], list[Server]]:
    """The partitions to ask the scheduler about, and the servers the site names."""
    from imas_ambix.agent.profile import SiteConfig

    site = SiteConfig.from_env()
    declared = [
        Server(
            entry.host.split(".", 1)[0],
            entry.partition,
            entry.accelerator_family,
            entry.accelerator_count,
        )
        for entry in site.servers
    ]
    named = os.environ.get(PARTITIONS_ENV)
    extra = (
        list(OTHER_PARTITIONS)
        if named is None
        else [p.strip() for p in named.split(",")]
    )
    names = [site.partition, *(server.partition for server in declared), *extra]
    return tuple(dict.fromkeys(name for name in names if name)), declared


class Fleet:
    """The live state of every server, read on background threads.

    ``start`` begins the scheduler poll, ``tick`` returns one reading per
    server and ``stop`` ends every sampler this object started. A sampler is a
    step of someone's job; one left running outlives the watcher and holds
    that step until the job ends, so ``stop`` is not optional.
    """

    def __init__(self, window: int = 30) -> None:
        self.window = max(1, window)
        self.partitions, self._declared = watched_partitions()
        self._lock = threading.Lock()
        self._servers: list[Server] = list(self._declared)
        self._state: dict[str, tuple[str, str]] = {}
        self._jobs: list[Job] = []
        self._held: dict[str, dict[str, set[int]]] = {}
        self._asked: dict[str, int] = {}
        #: Whether each job was running when its detail was read.
        self._detailed: dict[str, bool] = {}
        self._latest: dict[tuple[str, int], tuple] = {}
        self._host: dict[str, tuple[float, float, float]] = {}
        self._samplers: dict[str, subprocess.Popen] = {}
        self._launched: dict[str, float] = {}
        self._util: dict[tuple[str, int], deque] = {}
        self._cpu: dict[str, deque] = {}
        self._stop = threading.Event()
        self.polled = threading.Event()

    # ── scheduler ────────────────────────────────────────────────────

    def start(self) -> None:
        threading.Thread(target=self._poll_forever, daemon=True).start()

    def _poll_forever(self) -> None:
        while not self._stop.is_set():
            with contextlib.suppress(Exception):
                self.poll()
            self.polled.set()
            self._stop.wait(POLL_SECONDS)

    def poll(self) -> None:
        """One scheduler read: nodes, jobs, card indices, then samplers."""
        partitions = ",".join(self.partitions)
        nodes = parse_nodes(
            _run(["sinfo", "-h", "-N", "-p", partitions, "-o", _NODE_FORMAT])
        )
        servers = {server.node: server for server, _, _ in nodes}
        for declared in self._declared:
            seen = servers.get(declared.node)
            servers[declared.node] = Server(
                declared.node,
                declared.partition,
                declared.family or (seen.family if seen else None),
                declared.cards or (seen.cards if seen else None),
            )
        state = {server.node: (word, reason) for server, word, reason in nodes}
        jobs: list[Job] = []
        if servers:
            jobs = parse_jobs(
                _run(["squeue", "-h", "-w", ",".join(servers), "-o", _JOB_FORMAT])
            )
        known = {job.id for job in jobs}
        waiting = parse_jobs(
            _run(["squeue", "-h", "-p", partitions, "-t", "PD", "-o", _JOB_FORMAT])
        )
        jobs += [job for job in waiting if job.id not in known]
        # What a job asked for never changes and the cards it holds cannot
        # change while it runs, so a job's detail is read when it is first
        # seen and once more when it starts.
        held = {job.id: self._held[job.id] for job in jobs if job.id in self._held}
        asked = {job.id: self._asked[job.id] for job in jobs if job.id in self._asked}
        detailed = {
            job.id: self._detailed[job.id] for job in jobs if job.id in self._detailed
        }
        for job in jobs:
            if detailed.get(job.id, False) or (job.id in detailed and not job.running):
                continue
            detail = _run(["scontrol", "show", "job", job.id, "-d"])
            if not detail:
                continue
            held[job.id] = parse_card_indices(detail)
            asked[job.id] = parse_asked_cards(detail)
            detailed[job.id] = job.running
        self._detailed = detailed
        with self._lock:
            self._servers = sorted(servers.values(), key=lambda s: s.node)
            self._state, self._jobs = state, jobs
            self._held, self._asked = held, asked
        mine = sampled_jobs(jobs, held, servers)
        for job_id in list(self._launched):
            if job_id not in mine:
                self._launched.pop(job_id, None)
        for job_id, node in mine.items():
            self._ensure_sampler(job_id, node)

    # ── samplers ─────────────────────────────────────────────────────

    def _ensure_sampler(self, job_id: str, node: str) -> None:
        with self._lock:
            if job_id in self._samplers:
                return
        if time.monotonic() - self._launched.get(job_id, -1e9) < RELAUNCH_SECONDS:
            return
        self._launched[job_id] = time.monotonic()
        try:
            process = subprocess.Popen(
                [
                    "srun",
                    "--jobid",
                    job_id,
                    "--overlap",
                    "python3",
                    "-u",
                    SAMPLER,
                    "stream",
                    "1",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except OSError:
            return
        with self._lock:
            # A poll can still be running when the watcher stops; a step
            # started after that would have nothing left to end it.
            stopped = self._stop.is_set()
            if not stopped:
                self._samplers[job_id] = process
        if stopped:
            with contextlib.suppress(Exception):
                process.terminate()
            return
        threading.Thread(
            target=self._read_sampler, args=(job_id, node, process), daemon=True
        ).start()

    def _read_sampler(self, job_id: str, node: str, process: subprocess.Popen) -> None:
        """Keep the newest line per card; forget the cards when the stream ends."""
        owned: set[tuple[str, int]] = set()
        for line in process.stdout or ():
            self._take_line(node, line, owned)
        with self._lock:
            for key in owned:
                self._latest.pop(key, None)
            self._samplers.pop(job_id, None)

    def _take_line(self, node: str, line: str, owned: set[tuple[str, int]]) -> None:
        """One sampler line: ``GPU`` index, uuid, util, mem, then the sensors."""
        parts = line.rstrip("\n").split("\t")
        with self._lock:
            if parts[0] == "GPU" and len(parts) >= 5 and parts[1].isdigit():
                key = (node, int(parts[1]))
                sensors = (parts[5:8] + [""] * 3)[:3]
                self._latest[key] = (
                    float(parts[3]),
                    float(parts[4]),
                    *(_number(value) for value in sensors),
                )
                owned.add(key)
            elif parts[0] == "NODE" and len(parts) >= 4:
                self._host[node] = (float(parts[1]), float(parts[2]), float(parts[3]))

    def stop(self) -> None:
        """End the poll and every sampler this object started."""
        with self._lock:
            self._stop.set()
            doomed = list(self._samplers.values())
            self._samplers.clear()
        for process in doomed:
            with contextlib.suppress(Exception):
                process.terminate()
        for process in doomed:
            try:
                process.wait(timeout=3)
            except Exception:
                with contextlib.suppress(Exception):
                    process.kill()

    # ── readings ─────────────────────────────────────────────────────

    def tick(self) -> list[Reading]:
        """One reading per server, extending the rolling means by one sample."""
        with self._lock:
            servers = list(self._servers)
            state, jobs, held = dict(self._state), list(self._jobs), dict(self._held)
            asked = dict(self._asked)
            latest, host = dict(self._latest), dict(self._host)
            sampled_nodes = {job.node for job in jobs if job.id in self._samplers}
        for key in [key for key in self._util if key not in latest]:
            del self._util[key]
        readings = []
        for server in servers:
            node = server.node
            mine = [job for job in jobs if job.node == node]
            mine += [
                job
                for job in jobs
                if job.pending and not job.node and job.partition == server.partition
            ]
            job_cards = {
                job.id: held.get(job.id, {}).get(node, set())
                for job in mine
                if job.running
            }
            allocated: set[int] = set().union(*job_cards.values())
            cards = []
            for (card_node, index), values in sorted(latest.items()):
                if card_node != node:
                    continue
                util, mem, temp, watts, cap = values
                samples = self._util.setdefault(
                    (node, index), deque(maxlen=self.window)
                )
                samples.append(util)
                cards.append(
                    Card(
                        index, util, sum(samples) / len(samples), mem, temp, watts, cap
                    )
                )
            host_reading = None
            if node in host and node in sampled_nodes:
                cpu, used, total = host[node]
                samples = self._cpu.setdefault(node, deque(maxlen=self.window))
                samples.append(cpu)
                host_reading = (
                    cpu,
                    sum(samples) / len(samples),
                    100.0 * used / max(total, 1.0),
                )
            word, reason = state.get(node, ("unknown", ""))
            group = next(
                (
                    job.group
                    for job in mine
                    if job.running and job.user == USER and job.group
                ),
                None,
            )
            readings.append(
                Reading(
                    server,
                    word,
                    reason,
                    mine,
                    allocated,
                    job_cards,
                    cards,
                    {job.id: asked[job.id] for job in mine if job.id in asked},
                    host_reading,
                    group,
                )
            )
        return readings
