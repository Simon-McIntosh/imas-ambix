"""Attaching a watcher to a server through that server's own running job.

A watcher on a login or fleet node cannot read another machine's cards: the
devices belong to an allocation, and the process holding that allocation is the
only one whose ``nvidia-smi`` sees them. So the readings are taken *on* the
server, by one SLURM step opened inside a running job that already holds the
allocation::

    srun --jobid <job> --overlap -n1 -c1 --cpu-bind=none \\
         <python> -u -m imas_ambix.agent.gpu_watch.sampler

The step runs :mod:`imas_ambix.agent.gpu_watch.sampler`, which frames
:meth:`imas_ambix.agent.node_probe.NodeProbe.stream` as JSON lines; nothing is
copied onto the server, so the probe and its card numbering have one owner.

**One step, not one per sample.** A SLURM job carries a lifetime step quota, so
a watcher that opened a step per tick would spend it in hours. :class:`SamplerPool`
holds one long-lived step per server and redraws from its stream, and the
one-shot form opens one bounded step (`--count`) and collects it.

**A dropped step does not blank the view, and does not multiply either.** A
transient ``srun`` failure is restarted once per server; past that the pool
stops relaunching so a job refusing steps is not asked every second. Every step
the watcher opened is terminated and reaped on :meth:`SamplerPool.close`, so
none outlives the process that made it -- an orphan step holds a step against
the server's job long after the watcher that made it is gone.

**A step that answers nothing says so.** It is a reading about the server, not
an empty card table: :class:`StepReading` carries the reason, which the view
prints for that server.
"""

from __future__ import annotations

import contextlib
import getpass
import json
import subprocess
import sys
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from imas_ambix.agent import node_probe

#: The module the step runs. Named once so the launcher and its tests agree on
#: what a step is, and so no sampler script travels to the server.
SAMPLER_MODULE = "imas_ambix.agent.gpu_watch.sampler"

#: The step's executable; resolved through ``PATH`` so a test can shadow it.
SRUN = "srun"

#: How long a terminating step is given to exit before it is killed, in seconds.
#: Bounded because a step that ignores ``SIGTERM`` must not hold the watcher's
#: own exit, and a step that exits on it is the ordinary case.
STEP_GRACE_S = 5.0

#: The reason a step that produced no sample leaves behind. The view names it,
#: so a running server whose step failed is never drawn as an empty table.
NO_SAMPLES_REASON = "the step produced no samples"

# Everything a failed local call raises, named so the clause parses under the
# serving interpreter's older syntax as well as the repository's.
_CALL_FAILURES = (OSError, subprocess.SubprocessError)


@dataclass(frozen=True)
class StepReading:
    """What one bounded step answered for one server.

    ``job_id`` is the running job the step attached to, or ``None`` when the
    server holds no job we may overlap into -- which the view renders as
    ``no job to sample`` rather than as a failed sample. ``reason`` is set only
    when a step ran and answered nothing, so a failure is told apart from a
    server that is simply not serving.
    """

    samples: tuple[dict[str, Any], ...] = ()
    job_id: str | None = None
    reason: str | None = None


def scheduler_host(host: str) -> str:
    """The node name the scheduler accepts from a declared server's host."""
    return node_probe.scheduler_node_name(host)


def running_job(
    host: str,
    *,
    run: node_probe.RunFn | None = None,
    user: str | None = None,
    timeout_s: float = 15.0,
) -> str | None:
    """One running job of ours on *host*, preferring one that holds cards.

    A step can only be opened into a job of the same user, so the query is
    scoped to this user rather than to the whole node. A job holding a GPU is
    preferred because the card reads want the allocation that owns the devices;
    a running job without one is still accepted, since its step can read the
    host and the job table even where it sees no card.
    """
    run = node_probe.run_capture if run is None else run
    user = getpass.getuser() if user is None else user
    argv = [
        "squeue",
        "-h",
        "-w",
        scheduler_host(host),
        "--states=R",
        "-u",
        user,
        "-o",
        "%i|%b",
    ]
    try:
        text = run(argv, timeout_s=timeout_s)
    except FileNotFoundError:
        # No scheduler on this node: the question could not be put, which is a
        # different fact from a server holding no job, but both leave nothing
        # to attach to.
        return None
    if not text:
        return None
    fallback: str | None = None
    for line in text.splitlines():
        job_id, _, tres = line.partition("|")
        job_id = job_id.strip()
        if not job_id:
            continue
        if fallback is None:
            fallback = job_id
        if "gpu" in tres.lower():
            return job_id
    return fallback


def step_argv(
    job_id: str,
    *,
    interval_s: float,
    count: int,
    python: str | None = None,
    srun: str | None = None,
) -> list[str]:
    """The argv of one sampler step inside *job_id*.

    ``--overlap`` makes the step share the job's allocation instead of asking
    for resources the job has already taken; ``-n1 -c1 --cpu-bind=none`` is what
    a step inside a full allocation needs to start at all -- without it SLURM
    refuses with *More processors requested than permitted* or *CPU binding
    outside of job step allocation*, and the step never runs.
    """
    return [
        srun or SRUN,
        "--jobid",
        str(job_id),
        "--overlap",
        "-n1",
        "-c1",
        "--cpu-bind=none",
        python or sys.executable,
        "-u",
        "-m",
        SAMPLER_MODULE,
        "--interval",
        f"{interval_s:g}",
        "--count",
        str(count),
    ]


def _last_line(text: str | None) -> str | None:
    """The last non-empty line of a step's stderr, or ``None`` when it wrote none."""
    for line in reversed((text or "").splitlines()):
        if line.strip():
            return line.strip()
    return None


def _decode(line: str) -> dict[str, Any] | None:
    """One streamed line as a sample mapping, or ``None`` if it is not one."""
    text = line.strip()
    if not text:
        return None
    try:
        sample = json.loads(text)
    except ValueError:
        return None
    return sample if isinstance(sample, dict) else None


def reap(proc: subprocess.Popen | None, *, grace_s: float = STEP_GRACE_S) -> None:
    """Stop *proc* if it is still running, then collect it.

    A step left unreaped holds a step slot against the server's job long after
    the watcher is gone, which is the leak this function exists to close.
    """
    if proc is None:
        return
    with contextlib.suppress(*_CALL_FAILURES):
        if proc.poll() is not None:
            proc.wait(timeout=grace_s)
            return
        proc.terminate()
        proc.wait(timeout=grace_s)
        return
    with contextlib.suppress(*_CALL_FAILURES):
        proc.kill()
        proc.wait(timeout=grace_s)


def _open_step(
    argv: list[str],
) -> subprocess.Popen:
    """Start one step with a line-buffered pipe. ``OSError`` propagates."""
    return subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )


class RemoteSampler:
    """Bounded samples of one server through one step into its running job."""

    def __init__(
        self,
        *,
        run: node_probe.RunFn | None = None,
        user: str | None = None,
        python: str | None = None,
        srun: str | None = None,
        slack_s: float = 120.0,
    ) -> None:
        self._run = run
        self._user = user
        self._python = python
        self._srun = srun
        self._slack_s = slack_s

    def read(
        self, host: str, *, window_s: float, interval_s: float
    ) -> StepReading:
        """One bounded window of *host*, or the reason nothing was taken.

        The step is asked for the same number of samples a local in-process
        window takes, so a one-shot view means the same thing whether the
        server is this node or another.
        """
        job_id = running_job(host, run=self._run, user=self._user)
        if job_id is None:
            return StepReading()
        ticks = (
            max(1, int(-(-window_s // interval_s)) + 1) if window_s > 0 else 1
        )
        argv = step_argv(
            job_id,
            interval_s=interval_s,
            count=ticks,
            python=self._python,
            srun=self._srun,
        )
        try:
            proc = _open_step(argv)
        except OSError as failure:
            return StepReading(
                job_id=job_id, reason=f"could not start a step: {failure}"
            )
        samples: list[dict[str, Any]] = []
        err = ""
        try:
            try:
                out, err = proc.communicate(timeout=window_s + self._slack_s)
            except subprocess.TimeoutExpired:
                # A step that never finishes its count is stopped rather than
                # waited on, and whatever it already wrote is still read.
                reap(proc)
                try:
                    out, err = proc.communicate(timeout=STEP_GRACE_S)
                except _CALL_FAILURES:
                    out = ""
            for line in (out or "").splitlines():
                sample = _decode(line)
                if sample is not None:
                    samples.append(sample)
        finally:
            reap(proc)
        if not samples:
            # A step that wrote to stderr and streamed nothing explains itself;
            # the generic reason would hide the one line that says why.
            return StepReading(
                job_id=job_id, reason=_last_line(err) or NO_SAMPLES_REASON
            )
        return StepReading(samples=tuple(samples), job_id=job_id)


@dataclass
class _Target:
    """One server's live step and the samples it has streamed since last read."""

    host: str
    job_id: str | None = None
    proc: subprocess.Popen | None = None
    reader: threading.Thread | None = None
    stderr_reader: threading.Thread | None = None
    buffer: list[dict[str, Any]] = field(default_factory=list)
    #: The last few stderr lines, so a step that stops streaming can say why
    #: without the pipe ever filling.
    stderr_tail: deque[str] = field(default_factory=lambda: deque(maxlen=20))
    lock: threading.Lock = field(default_factory=threading.Lock)
    launches: int = 0
    restarts: int = 0
    reason: str | None = None


class SamplerPool:
    """One long-lived step per server, redrawn from its stream.

    *max_restarts* bounds how many times a server's step is reopened after it
    exits, so a step that keeps failing stops being retried rather than burning
    the job's step quota. The default of one is the restart the live view makes
    when a step is dropped.
    """

    def __init__(
        self,
        *,
        interval_s: float = 1.0,
        max_restarts: int = 1,
        run: node_probe.RunFn | None = None,
        user: str | None = None,
        python: str | None = None,
        srun: str | None = None,
        opener: Any = None,
    ) -> None:
        self._interval_s = interval_s
        self._max_restarts = max_restarts
        self._run = run
        self._user = user
        self._python = python
        self._srun = srun
        self._opener = _open_step if opener is None else opener
        self._targets: dict[str, _Target] = {}
        self._lock = threading.Lock()

    def ensure(self, host: str) -> None:
        """Attach to *host* if no step is live, restarting one a bounded number."""
        with self._lock:
            target = self._targets.setdefault(host, _Target(host))
            if target.proc is not None and target.proc.poll() is None:
                return
            if target.proc is not None:
                if target.restarts >= self._max_restarts:
                    return
                target.restarts += 1
            job_id = running_job(host, run=self._run, user=self._user)
            target.job_id = job_id
            if job_id is None:
                target.reason = None
                return
            argv = step_argv(
                job_id,
                interval_s=self._interval_s,
                count=0,
                python=self._python,
                srun=self._srun,
            )
            try:
                proc = self._opener(argv)
            except OSError as failure:
                target.reason = f"could not start a step: {failure}"
                return
            target.proc = proc
            target.reason = None
            with target.lock:
                target.stderr_tail.clear()
            target.launches += 1
            target.stderr_reader = threading.Thread(
                target=self._consume_stderr, args=(target, proc), daemon=True
            )
            target.stderr_reader.start()
            target.reader = threading.Thread(
                target=self._consume, args=(target, proc), daemon=True
            )
            target.reader.start()

    def _consume(self, target: _Target, proc: subprocess.Popen) -> None:
        """Buffer a step's stream until it exits, then collect the process."""
        stdout = proc.stdout
        if stdout is not None:
            for line in stdout:
                sample = _decode(line)
                if sample is not None:
                    with target.lock:
                        target.buffer.append(sample)
        with contextlib.suppress(*_CALL_FAILURES):
            proc.wait(timeout=STEP_GRACE_S)

    def _consume_stderr(self, target: _Target, proc: subprocess.Popen) -> None:
        """Drain a step's stderr, keeping a bounded tail.

        A step whose stderr nobody reads blocks once the pipe buffer fills, and
        stops streaming samples while the pool still counts it as live. The
        tail is what a step that stops streaming leaves behind, so it is the
        reason the view prints for that server.
        """
        stream = proc.stderr
        if stream is None:
            return
        with contextlib.suppress(*_CALL_FAILURES, ValueError):
            for line in stream:
                text = line.strip()
                if text:
                    with target.lock:
                        target.stderr_tail.append(text)

    def drain(self, host: str) -> list[dict[str, Any]]:
        """Every sample streamed since the last drain, oldest first."""
        target = self._targets.get(host)
        if target is None:
            return []
        with target.lock:
            out, target.buffer = target.buffer, []
        return out

    def state(self, host: str) -> tuple[str | None, str | None]:
        """``(job_id, reason)`` for *host*, for a block with no samples yet.

        A step that has exited without streaming is asked why from its own
        stderr, because that line distinguishes a job refusing steps from a
        node whose cards are simply not readable.
        """
        target = self._targets.get(host)
        if target is None:
            return None, None
        reason = target.reason
        proc = target.proc
        if reason is None and proc is not None and proc.poll() is not None:
            with target.lock:
                reason = target.stderr_tail[-1] if target.stderr_tail else None
            reason = reason or NO_SAMPLES_REASON
        return target.job_id, reason

    def launches(self, host: str) -> int:
        """How many steps this pool has opened for *host*; a test's measure."""
        target = self._targets.get(host)
        return 0 if target is None else target.launches

    def close(self) -> None:
        """Terminate and reap every step this pool opened."""
        with self._lock:
            targets = list(self._targets.values())
        for target in targets:
            reap(target.proc)
            if target.proc is not None:
                target.proc = None

    def __enter__(self) -> SamplerPool:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
