"""Contracts for the DeepSeek-V4 container kernel source overlay."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from imas_ambix.agent.profile import SiteConfig, load_profile
from imas_ambix.agent.slurm import generate_serve_script
from imas_ambix.cli import main

_PATCH_SOURCE = Path(
    "imas_ambix/agent/kernel_patches/sglang-dev-dsv41/python/sglang/kernels/"
    "jit/csrc/deepseek_v4/main_norm_rope.cuh"
)
_IMAGE_TARGET = (
    "/sgl-workspace/sglang/python/sglang/kernels/jit/csrc/deepseek_v4/"
    "main_norm_rope.cuh"
)


def test_kernel_patch_waits_before_reading_cache_location() -> None:
    """The dependency barrier precedes the one K-cache location load."""
    source = (Path(__file__).parents[1] / _PATCH_SOURCE).read_text(encoding="utf-8")
    start = source.index("K_KERNEL void fused_k_norm_rope_flashmla")
    end = source.index("struct FusedQIndexerRopeHadamardQuantParams", start)
    kernel = source[start:end]

    assert kernel.count("const auto out_loc = params.out_loc[work_id];") == 1
    assert kernel.index("PDLWaitPrimary<kUsePDL>();") < kernel.index(
        "const auto out_loc = params.out_loc[work_id];"
    )


def test_profile_binds_the_kernel_patch_over_the_image_source() -> None:
    """The four-card DeepSeek serve makes the patched header its JIT input."""
    profile = load_profile("deepseek-v4-1-flash")
    script = generate_serve_script(profile, SiteConfig())
    expected_source = Path(__file__).parents[1] / _PATCH_SOURCE

    assert f"--bind {expected_source}:{_IMAGE_TARGET}:ro" in script


def test_dry_run_reports_the_kernel_patch_bind() -> None:
    """The public dry-run receipt contains the source-overlay declaration."""
    result = CliRunner().invoke(
        main,
        ["agent", "serve", "deepseek-v4-1-flash", "--dry-run"],
    )

    assert result.exit_code == 0, result.output
    assert f"{_PATCH_SOURCE}:{_IMAGE_TARGET}:ro" in result.output
