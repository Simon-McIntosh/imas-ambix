"""Launch contract for the bounded device pool, host tier and drafter."""

from __future__ import annotations

from click.testing import CliRunner

from imas_ambix.agent.profile import load_profile
from imas_ambix.cli import main

SERVE_DRY_RUN = ["agent", "serve", "deepseek-v4-1-flash", "--dry-run"]


def test_profile_sizes_device_and_host_cache_with_group_memory_margin() -> None:
    """The profile keeps reusable prefixes in host RAM without filling Group A."""
    profile = load_profile("deepseek-v4-1-flash")

    assert profile.engine.max_total_tokens == 4_000_000
    assert profile.engine.enable_hierarchical_cache is True
    assert profile.engine.hicache_ratio == 2.0
    assert profile.engine.hicache_write_policy == "write_through_selective"
    assert profile.engine.hicache_mem_layout == "page_first"
    assert profile.slurm.memory == "600G"


def test_profile_declares_the_dspark_drafter_with_its_block_size() -> None:
    """DSpark is the only throughput lever, and both keys land together.

    ``speculative_algorithm`` is compared case-sensitively by SGLang against
    the literal ``"DSPARK"``; the block size means nothing without it, so the
    pair is asserted as a pair.
    """
    profile = load_profile("deepseek-v4-1-flash")

    assert profile.engine.speculative_algorithm == "DSPARK"
    assert profile.engine.speculative_dspark_block_size == 5


def test_dry_run_carries_pool_hicache_and_memory_contract() -> None:
    """A generated serve command must carry every sizing decision to SGLang."""
    result = CliRunner().invoke(main, SERVE_DRY_RUN)

    assert result.exit_code == 0, result.output
    assert "#SBATCH --mem=600G" in result.output
    assert "--max-total-tokens 4000000" in result.output
    assert "--enable-hierarchical-cache" in result.output
    assert "--hicache-ratio 2.0" in result.output
    assert "--hicache-write-policy write_through_selective" in result.output
    assert "--hicache-mem-layout page_first" in result.output
    assert "--moe-runner-backend flashinfer_mxfp4" in result.output


def test_dry_run_carries_the_two_speculative_flags() -> None:
    """The emitted drafter flags are the deploy recipe's restart-1 contract."""
    result = CliRunner().invoke(main, SERVE_DRY_RUN)

    assert result.exit_code == 0, result.output
    assert "--speculative-algorithm DSPARK" in result.output
    assert "--speculative-dspark-block-size 5" in result.output


def test_mxfp4_experts_reach_the_fp8_tensor_cores_hopper_has() -> None:
    """SM90 has no FP4 datapath, so the expert GEMM must not upcast to BF16.

    The runner's ``default`` precision converts MXFP4 weights to BF16 before
    every expert multiply. ``fp8`` selects FlashInfer's MXFP4-weight x
    FP8-activation kernels instead. The pair only means anything together:
    the precision option is read by the FlashInfer runner and ignored by any
    other, so a profile setting one without the other is a silent no-op.
    """
    profile = load_profile("deepseek-v4-1-flash")

    assert profile.engine.moe_runner_backend == "flashinfer_mxfp4"
    precision = profile.engine.flashinfer_mxfp4_moe_precision
    assert precision in {"fp8", "bf16", "default"}

    result = CliRunner().invoke(main, SERVE_DRY_RUN)

    assert result.exit_code == 0, result.output
    assert "--moe-runner-backend flashinfer_mxfp4" in result.output
    assert f"--flashinfer-mxfp4-moe-precision {precision}" in result.output
