"""Persistent SLURM allocation for the interactive agent fleet."""

from __future__ import annotations

from textwrap import dedent

from imas_ambix.agent import slurm


def generate_fleet_hold_script() -> str:
    """Generate the whole-node allocation that hosts interactive sessions."""
    headers = slurm._sbatch_headers(
        job_name="ambix-fleet",
        partition="rigel",
        account="iter",
        reservation=None,
        gpus=0,
        cpus=28,
        memory="120G",
        time_limit="UNLIMITED",
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
