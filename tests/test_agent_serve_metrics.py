"""Metrics exposure in generated model serve commands."""

from __future__ import annotations

from click.testing import CliRunner

from imas_ambix.cli import main


def test_sglang_dry_run_enables_engine_metrics() -> None:
    """The DeepSeek-V4.1 launch publishes the metrics the lane reader needs."""
    result = CliRunner().invoke(
        main,
        ["agent", "serve", "deepseek-v4-1-flash", "--dry-run"],
    )

    assert result.exit_code == 0, result.output
    launch = next(
        line for line in result.output.splitlines() if "-m sglang.launch_server" in line
    )
    assert "--enable-metrics" in launch.split()


def test_vllm_dry_run_keeps_its_metrics_default() -> None:
    """The SGLang-only setting does not alter the vLLM launch command."""
    result = CliRunner().invoke(
        main,
        ["agent", "serve", "deepseek-v4-flash", "--dry-run"],
    )

    assert result.exit_code == 0, result.output
    launch = next(
        line
        for line in result.output.splitlines()
        if "-m vllm.entrypoints.openai.api_server" in line
    )
    assert "--enable-metrics" not in launch.split()
