"""Persistent SLURM allocation for the interactive agent fleet."""

from __future__ import annotations

from textwrap import dedent
from typing import TYPE_CHECKING

from imas_ambix.agent import slurm

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from imas_ambix.agent.profile import SiteConfig

# The scheduler spells an unbounded wall clock with a non-numeric token. The
# shared header emitter takes the limit as a string and always emits the
# directive, so the fleet's "no limit" case is expressed here as that token
# rather than by omitting the line.
UNBOUNDED_TIME = "UNLIMITED"

# Scheduler identity of the allocation, carried as the job comment so a held
# allocation is found from the queue alone rather than by job id.
FLEET_COMMENT = "ambix-fleet"

# Remaining wall clock below which the status surface warns. Moving sessions
# off an allocation takes minutes of operator work, so the notice has to
# arrive while the allocation still has time left to act in.
REMAINING_WARNING_SECONDS = 30 * 60

# Scheduler node-state token that means the node is being taken out of
# service. A draining node still runs its jobs, but the allocation ends when
# the drain completes, so the operator has to be told before that happens.
# The spelling varies (`DRAIN`, `DRAINED`, `DRAINING`) but always starts with
# this stem, so the match is a token prefix rather than an equality.
_DRAINING_STATE_PREFIX = "DRAIN"


def generate_fleet_hold_script(site: SiteConfig) -> str:
    """Generate the whole-node allocation that hosts interactive sessions.

    The allocation takes the site's CPU partition whole and without a wall
    clock, so it stays up until it is cancelled. The job body points
    ``TMPDIR`` at ``/tmp`` because a compute node cannot write the per-user
    runtime directory, and the comment token is the scheduler identity the job
    is found by from ``squeue`` alone.
    """
    headers = slurm._sbatch_headers(
        job_name=FLEET_COMMENT,
        partition=site.fleet_partition,
        account=site.fleet_account,
        reservation=None,
        gpus=0,
        cpus=site.fleet_cpus,
        memory=site.fleet_memory,
        time_limit=UNBOUNDED_TIME,
        output_name="ambix-fleet-%j.log",
    )
    headers.extend(
        [
            "#SBATCH --nodes=1",
            "#SBATCH --exclusive",
            f"#SBATCH --comment={FLEET_COMMENT}",
        ]
    )
    body = dedent(
        """
        set -euo pipefail

        export TMPDIR=/tmp

        echo "[$(date)] Holding $(hostname) for the interactive agent fleet"
        exec sleep infinity
        """
    ).strip()
    return "\n".join([*headers, "", body, ""])


def submit_fleet_hold(script: str) -> str:
    """Submit a generated fleet allocation through the shared adapter."""
    return slurm.submit_script(script)


def find_fleet_allocation(
    jobs: Iterable[dict[str, str]],
) -> dict[str, str] | None:
    """Return the held fleet allocation among scheduler rows, if any.

    The allocation is identified by the comment token the generator emits, so
    it is found wherever the queue reports it rather than by a remembered job
    id.
    """
    for job in jobs:
        if job.get("comment", "").strip() == FLEET_COMMENT:
            return job
    return None


def placement_argv(job: dict[str, str], command: Sequence[str]) -> list[str]:
    """The scheduler invocation that runs ``command`` inside a held allocation.

    The command becomes an overlapping step of the allocation rather than an
    allocation of its own, so it lands on the allocation's node and shares its
    control group. The job id comes from the row the allocation was found by,
    so no identifier for it is written into any configuration: a resubmit, a
    cancel-and-rehold or a move to another node changes the row and the
    placement follows it.
    """
    return ["srun", "--overlap", f"--jobid={job.get('jobid', '').strip()}", *command]


def remaining_seconds(time_left: str) -> int | None:
    """Remaining wall clock in seconds, or ``None`` when there is no duration.

    ``squeue %L`` reports an unbounded limit as the literal ``UNLIMITED``.
    That token is not a duration: it maps to ``None``, so it can never be
    compared against a threshold or read as an allocation that is ending. A
    value the scheduler does not report as ``[DD-]HH:MM:SS`` is ``None`` as
    well, because an unknown lifetime must not present as an expiring one.
    """
    raw = time_left.strip()
    if raw == UNBOUNDED_TIME:
        return None
    days, _, hms = raw.partition("-")
    if not hms:
        hms, days = days, ""
    try:
        day_count = int(days) if days else 0
        parts = [int(part) for part in hms.split(":")]
    except ValueError:
        return None
    if len(parts) == 3:
        hours, minutes, secs = parts
    elif len(parts) == 2:
        hours, minutes, secs = 0, parts[0], parts[1]
    else:
        return None
    return day_count * 86400 + hours * 3600 + minutes * 60 + secs


def parse_node_state(node_info: str) -> str | None:
    """Upper-cased state value from ``scontrol show node`` output, or ``None``.

    The row is whitespace-separated ``key=value`` fields and the state is the
    ``State=`` one. Its value is a ``+``-separated set of tokens —
    ``ALLOCATED``, ``MIXED``, ``DRAIN``, ``REBOOT_REQUESTED`` and so on — which
    is returned whole and upper-cased, because every token is a fact about the
    node and the first one is not privileged.
    """
    for token in node_info.split():
        if not token.startswith("State="):
            continue
        value = token.split("=", 1)[1].strip().upper()
        return value or None
    return None


def node_is_draining(state: str | None) -> bool:
    """Whether a scheduler node state means the node is going out of service.

    The state is a ``+``-separated set of tokens and the drain token is not
    necessarily the first of them, so every token is inspected rather than the
    leading one. A leading-token match misses a node the scheduler reports as
    ``MIXED+DRAIN+REBOOT_REQUESTED`` — the spelling this cluster actually uses.
    """
    if not state:
        return False
    return any(
        token.startswith(_DRAINING_STATE_PREFIX) for token in state.upper().split("+")
    )


def _format_duration(seconds: int) -> str:
    """Render a second count compactly: ``2d03h``, ``1h05m``, ``25m``, ``40s``."""
    days, rest = divmod(seconds, 86400)
    hours, rest = divmod(rest, 3600)
    minutes, secs = divmod(rest, 60)
    if days:
        return f"{days}d{hours:02d}h"
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m"
    return f"{secs}s"


def describe_fleet_allocation(
    job: dict[str, str], *, node_state: str | None = None
) -> list[str]:
    """Operator-readable lifetime for one scheduler row, as plain-text lines.

    An allocation with no wall clock reads as unbounded and never warns on
    time; a finite one below :data:`REMAINING_WARNING_SECONDS` adds a warning
    line. The time warning is the only place a remaining time is compared to a
    threshold, and an unbounded or unreadable value never reaches it.

    ``node_state`` is the scheduler state of the node the allocation runs on.
    A node that is draining or drained ends the allocation when the drain
    completes regardless of any wall clock, so it warns on its own — which is
    the only way an unbounded allocation is warned about at all.
    """
    job_id = job.get("jobid", "") or "unknown"
    state = job.get("state", "") or "unknown"
    node = job.get("node", "") or "unallocated"
    elapsed = job.get("time", "").strip() or "unknown"
    raw_left = job.get("timeleft", "").strip()
    seconds = remaining_seconds(raw_left)
    if raw_left == UNBOUNDED_TIME:
        remaining = "unbounded (no wall clock)"
    elif seconds is None:
        remaining = "unknown"
    else:
        remaining = _format_duration(seconds)

    lines = [
        f"Fleet allocation {job_id} · {state} · node {node}",
        f"  elapsed    {elapsed}",
        f"  remaining  {remaining}",
    ]
    if node_is_draining(node_state):
        lines.append(
            f"WARNING: fleet allocation {job_id} runs on {node}, which the "
            f"scheduler reports {node_state}; the allocation ends when the "
            f"drain completes"
        )
    if seconds is not None and seconds < REMAINING_WARNING_SECONDS:
        lines.append(
            f"WARNING: fleet allocation {job_id} has {_format_duration(seconds)} "
            f"remaining on {node}"
        )
    return lines
