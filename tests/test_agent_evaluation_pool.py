"""Default launch behavior for the bounded token pool."""

from __future__ import annotations

from click.testing import CliRunner

from imas_ambix.agent.profile import load_profile
from imas_ambix.cli import main


def test_default_profile_resolves_the_bounded_pool() -> None:
    """The ordinary serve reserves per-request context headroom."""
    profile = load_profile("deepseek-v4-1-flash")

    assert profile.engine.max_total_tokens == 8_000_000


def test_agent_serve_uses_the_bounded_pool_by_default() -> None:
    """The plain profile carries its bounded pool into the serve command."""
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
    assert "--max-total-tokens 8000000" in result.output
