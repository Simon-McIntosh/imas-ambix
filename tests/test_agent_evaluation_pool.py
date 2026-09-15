"""Named launch setting for the load-proven evaluation token pool."""

from __future__ import annotations

from click.testing import CliRunner

from imas_ambix.agent.profile import load_profile
from imas_ambix.cli import main


def test_default_and_evaluation_pool_resolve_independently() -> None:
    """Selecting the experiment must not change the ordinary serve default."""
    default = load_profile("deepseek-v4-1-flash")
    evaluation = load_profile("deepseek-v4-1-flash@evaluation")

    assert default.engine.max_total_tokens == 3485952
    assert evaluation.engine.max_total_tokens == 14999808
    assert evaluation.slug == "deepseek-v4-1-flash@evaluation"
    assert evaluation.weights_directory_slug == "deepseek-v4-1-flash"


def test_agent_serve_selects_the_evaluation_pool() -> None:
    """The named profile selection reaches the generated serve command."""
    result = CliRunner().invoke(
        main,
        ["agent", "serve", "deepseek-v4-1-flash@evaluation", "--dry-run"],
    )

    assert result.exit_code == 0, result.output
    assert "#SBATCH --job-name=deepseek-v4-1-flash@evaluation" in result.output
    assert "--max-total-tokens 14999808" in result.output
