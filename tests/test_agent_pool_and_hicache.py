"""Launch contract for the bounded device pool and host prefix tier."""

from __future__ import annotations

from click.testing import CliRunner

from imas_ambix.agent.profile import load_profile
from imas_ambix.cli import main


def test_profile_sizes_device_and_host_cache_with_group_memory_margin() -> None:
    """The profile keeps reusable prefixes in host RAM without filling Group A."""
    profile = load_profile("deepseek-v4-1-flash")

    assert profile.engine.max_total_tokens == 4_000_000
    assert profile.engine.enable_hierarchical_cache is True
    assert profile.engine.hicache_ratio == 2.0
    assert profile.engine.hicache_write_policy == "write_through_selective"
    assert profile.engine.hicache_mem_layout == "page_first"
    assert profile.slurm.memory == "480G"


def test_dry_run_carries_pool_hicache_and_memory_contract() -> None:
    """A generated serve command must carry every sizing decision to SGLang."""
    result = CliRunner().invoke(
        main,
        ["agent", "serve", "deepseek-v4-1-flash", "--dry-run"],
    )

    assert result.exit_code == 0, result.output
    assert "#SBATCH --mem=480G" in result.output
    assert "--max-total-tokens 4000000" in result.output
    assert "--enable-hierarchical-cache" in result.output
    assert "--hicache-ratio 2.0" in result.output
    assert "--hicache-write-policy write_through_selective" in result.output
    assert "--hicache-mem-layout page_first" in result.output
    assert "--moe-runner-backend flashinfer_mxfp4" in result.output
