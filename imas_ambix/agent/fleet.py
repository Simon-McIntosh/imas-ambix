"""Persistent SLURM allocation for the interactive agent fleet."""

from __future__ import annotations

from textwrap import dedent
from typing import TYPE_CHECKING

from imas_ambix.agent import slurm

if TYPE_CHECKING:
    from imas_ambix.agent.profile import SiteConfig

# The scheduler spells an unbounded wall clock with a non-numeric token. The
# shared header emitter takes the limit as a string and always emits the
# directive, so the fleet's "no limit" case is expressed here as that token
# rather than by omitting the line.
UNBOUNDED_TIME = "UNLIMITED"


def generate_fleet_hold_script(site: SiteConfig) -> str:
    """Generate the whole-node allocation that hosts interactive sessions.

    The allocation takes the site's CPU partition whole and without a wall
    clock, so it stays up until it is cancelled. The job body points
    ``TMPDIR`` at ``/tmp`` because a compute node cannot write the per-user
    runtime directory, and the comment token is the scheduler identity the job
    is found by from ``squeue`` alone.
    """
    headers = slurm._sbatch_headers(
        job_name="ambix-fleet",
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
            "#SBATCH --comment=ambix-fleet",
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
