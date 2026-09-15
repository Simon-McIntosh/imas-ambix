"""Default launch behavior for the load-proven evaluation token pool."""

from __future__ import annotations

from click.testing import CliRunner

from imas_ambix.agent.profile import load_profile
from imas_ambix.cli import main


def test_default_profile_resolves_the_load_proven_pool() -> None:
    """The ordinary serve uses the pool demonstrated under load."""
    profile = load_profile("deepseek-v4-1-flash")

    assert profile.engine.max_total_tokens == 14999808


def test_agent_serve_uses_the_load_proven_pool_by_default() -> None:
    """The plain profile carries the proven pool into the serve command."""
    result = CliRunner().invoke(
        main,
        ["agent", "serve", "deepseek-v4-1-flash", "--dry-run"],
    )

    assert result.exit_code == 0, result.output
    job_name = next(
        line
        for line in result.output.splitlines()
        if line.startswith("#SBATCH --job-name")
    )
    assert job_name == "#SBATCH --job-name=deepseek-v4-1-flash"
    assert "@" not in job_name
    assert "--max-total-tokens 14999808" in result.output
