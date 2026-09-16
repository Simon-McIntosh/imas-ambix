"""Context-prefill headroom contract for the four-card SGLang launch."""

from __future__ import annotations

from click.testing import CliRunner

from imas_ambix.cli import main


def _launch_line(output: str, marker: str) -> str:
    """Return the generated engine command that contains ``marker``."""
    return next(line for line in output.splitlines() if marker in line)


def test_dsv41_dry_run_carries_all_context_headroom_levers() -> None:
    """The allocator, prefill chunk, and queue bound reach one SGLang command."""
    result = CliRunner().invoke(
        main,
        ["agent", "serve", "deepseek-v4-1-flash", "--dry-run"],
    )

    assert result.exit_code == 0, result.output
    launch = _launch_line(result.output, "-m sglang.launch_server")
    assert "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" in result.output
    assert "--chunked-prefill-size 16384" in launch
    assert "--max-running-requests 64" in launch


def test_vllm_dry_run_omits_sglang_context_headroom_levers() -> None:
    """SGLang settings do not change the V4-Flash vLLM launch."""
    result = CliRunner().invoke(
        main,
        ["agent", "serve", "deepseek-v4-flash", "--dry-run"],
    )

    assert result.exit_code == 0, result.output
    launch = _launch_line(result.output, "-m vllm.entrypoints.openai.api_server")
    assert "--max-running-requests" not in launch
    assert "--chunked-prefill-size" not in launch
    assert "PYTORCH_CUDA_ALLOC_CONF" not in result.output
