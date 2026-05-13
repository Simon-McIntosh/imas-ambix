"""Tests for the ambix agent CLI and profile system."""

from click.testing import CliRunner

from imas_ambix.agent.profile import SiteConfig, list_profiles, load_profile
from imas_ambix.cli import main

# -- Profile loader ----------------------------------------------------------


def test_list_profiles():
    slugs = list_profiles()
    assert isinstance(slugs, list)
    assert "kimi-k2-6" in slugs


def test_load_kimi_profile():
    profile = load_profile("kimi-k2-6")
    assert profile.slug == "kimi-k2-6"
    assert profile.model.hf_repo == "moonshotai/Kimi-K2.6"
    assert profile.engine.type == "ktransformers"
    assert profile.engine.tensor_parallel == 4
    assert profile.engine.ktransformers is not None
    assert profile.engine.ktransformers.gpu_experts == 350
    assert profile.engine.moe_runner_backend == "triton"
    assert profile.model.max_context == 262144


def test_load_deepseek_v4_flash_profile():
    profile = load_profile("deepseek-v4-flash")
    assert profile.slug == "deepseek-v4-flash"
    assert profile.model.hf_repo == "deepseek-ai/DeepSeek-V4-Flash"
    assert profile.engine.type == "vllm"
    assert profile.engine.tensor_parallel == 4
    assert profile.engine.ktransformers is None
    assert profile.engine.enable_auto_tool_choice is True
    assert profile.model.max_context == 1048576
    assert profile.model.size_gb == 164


def test_load_minimax_m2_7_profile():
    profile = load_profile("minimax-m2-7")
    assert profile.slug == "minimax-m2-7"
    assert profile.model.hf_repo == "MiniMaxAI/MiniMax-M2.7"
    assert profile.engine.type == "sglang"
    assert profile.engine.tensor_parallel == 4
    assert profile.engine.ktransformers is None
    assert profile.model.max_context == 204800
    assert profile.engine.parsers.tool_call == "minimax-m2"
    assert profile.engine.parsers.reasoning == "minimax-append-think"


def test_load_profile_not_found():
    import pytest

    with pytest.raises(FileNotFoundError, match="No profile 'nonexistent'"):
        load_profile("nonexistent")


# -- SiteConfig --------------------------------------------------------------


def test_site_config_defaults():
    site = SiteConfig()
    assert site.base_dir == "/work/projects/imas_gpu"
    assert site.partition == "betelgeuse"
    assert site.download_partition == "sirius"
    assert site.account == "grpa"


def test_site_config_model_dir():
    site = SiteConfig()
    profile = load_profile("kimi-k2-6")
    model_dir = site.model_dir(profile)
    assert str(model_dir).endswith("agents/kimi-k2-6/model")


def test_site_config_venv_paths():
    site = SiteConfig()
    assert site.python_path("vllm").name == "python"
    assert site.hf_path("vllm").name == "hf"
    assert site.python_path("sglang").name == "python"
    # ktransformers shares sglang env
    assert site.env_dir("ktransformers") == site.env_dir("sglang")


def test_site_config_engine_isolation():
    """vllm and sglang environments live in separate directories."""
    site = SiteConfig()
    vllm_dir = site.env_dir("vllm")
    sglang_dir = site.env_dir("sglang")
    assert vllm_dir != sglang_dir
    assert "vllm" in str(vllm_dir)
    assert "sglang" in str(sglang_dir)


def test_site_config_from_env(monkeypatch):
    monkeypatch.setenv("AMBIX_AGENT_BASE_DIR", "/tmp/test")
    monkeypatch.setenv("AMBIX_AGENT_PARTITION", "test-partition")
    monkeypatch.setenv("AMBIX_AGENT_DOWNLOAD_PARTITION", "test-dl")
    site = SiteConfig.from_env()
    assert site.base_dir == "/tmp/test"
    assert site.partition == "test-partition"
    assert site.download_partition == "test-dl"


# -- SLURM script generation -------------------------------------------------


def test_generate_serve_script():
    from imas_ambix.agent.slurm import generate_serve_script

    profile = load_profile("kimi-k2-6")
    site = SiteConfig()
    script = generate_serve_script(profile, site, port=8000)

    assert "#!/bin/bash" in script
    assert "#SBATCH --partition=betelgeuse" in script
    assert "#SBATCH --reservation=gpu_0003_grpA" in script
    assert "#SBATCH --gres=gpu:4" in script
    assert "sglang.launch_server" in script
    assert "--kt-method RAWINT4" in script
    assert "--max-total-tokens 131072" in script
    assert "--moe-runner-backend triton" in script
    assert "--port" in script


def test_generate_download_script():
    from imas_ambix.agent.slurm import generate_download_script

    profile = load_profile("kimi-k2-6")
    site = SiteConfig()
    script = generate_download_script(profile, site)

    assert "#!/bin/bash" in script
    assert "#SBATCH --partition=sirius" in script
    # Download should NOT request GPUs
    assert "--gres=gpu" not in script
    # Should use hf CLI, not deprecated huggingface-cli
    assert "hf download" in script
    assert "huggingface-cli" not in script
    assert "moonshotai/Kimi-K2.6" in script


def test_serve_script_uses_venv_python():
    from imas_ambix.agent.slurm import generate_serve_script

    profile = load_profile("kimi-k2-6")
    site = SiteConfig()
    script = generate_serve_script(profile, site)

    # Should use the venv Python, not bare 'python'
    assert str(site.python_path(profile.engine.type)) in script


# -- CLI commands ------------------------------------------------------------


def test_agent_list():
    runner = CliRunner()
    result = runner.invoke(main, ["agent", "list"])
    assert result.exit_code == 0
    assert "kimi-k2-6" in result.output


def test_agent_info():
    runner = CliRunner()
    result = runner.invoke(main, ["agent", "info", "kimi-k2-6"])
    assert result.exit_code == 0
    assert "Kimi-K2.6" in result.output
    assert "ktransformers" in result.output


def test_agent_serve_dry_run():
    runner = CliRunner()
    result = runner.invoke(main, ["agent", "serve", "kimi-k2-6", "--dry-run"])
    assert result.exit_code == 0
    assert "sglang.launch_server" in result.output
    assert "--partition=betelgeuse" in result.output


def test_agent_download_dry_run():
    runner = CliRunner()
    result = runner.invoke(main, ["agent", "download", "kimi-k2-6", "--dry-run"])
    assert result.exit_code == 0
    assert "hf download" in result.output
    assert "--partition=sirius" in result.output


def test_agent_info_not_found():
    runner = CliRunner()
    result = runner.invoke(main, ["agent", "info", "nonexistent"])
    assert result.exit_code != 0
    assert "No profile" in result.output


# -- SGLang engine (non-KTransformers) tests ---------------------------------


def test_generate_sglang_serve_script():
    """SGLang-engine profiles should not include KT or vLLM flags."""
    from imas_ambix.agent.slurm import generate_serve_script

    profile = load_profile("minimax-m2-7")
    site = SiteConfig()
    script = generate_serve_script(profile, site, port=8000)

    assert "#!/bin/bash" in script
    assert "#SBATCH --partition=betelgeuse" in script
    assert "#SBATCH --gres=gpu:4" in script
    assert "sglang.launch_server" in script
    assert "--kt-method" not in script
    assert "--kt-weight-path" not in script
    assert "--kt-cpuinfer" not in script
    assert "engine: sglang" in script
    assert "MALLOC_TRIM_THRESHOLD_" not in script


def test_generate_vllm_serve_script():
    """vLLM-engine profiles should use vllm.entrypoints."""
    from imas_ambix.agent.slurm import generate_serve_script

    profile = load_profile("deepseek-v4-flash")
    site = SiteConfig()
    script = generate_serve_script(profile, site, port=8000)

    assert "#!/bin/bash" in script
    assert "#SBATCH --gres=gpu:4" in script
    assert "vllm.entrypoints.openai.api_server" in script
    assert "--enable-auto-tool-choice" in script
    assert "--tool-call-parser deepseek_v4" in script
    assert "--gpu-memory-utilization" in script
    assert "sglang.launch_server" not in script
    assert "--kt-method" not in script
    assert "engine: vllm" in script


def test_generate_minimax_serve_script():
    from imas_ambix.agent.slurm import generate_serve_script

    profile = load_profile("minimax-m2-7")
    site = SiteConfig()
    script = generate_serve_script(profile, site, port=8000)

    assert "sglang.launch_server" in script
    assert "--tool-call-parser minimax-m2" in script
    assert "--reasoning-parser minimax-append-think" in script
    assert "--kt-method" not in script


def test_generate_download_script_vllm_engine():
    from imas_ambix.agent.slurm import generate_download_script

    profile = load_profile("deepseek-v4-flash")
    site = SiteConfig()
    script = generate_download_script(profile, site)

    assert "hf download" in script
    assert "deepseek-ai/DeepSeek-V4-Flash" in script
    assert "--gres=gpu" not in script


def test_agent_list_includes_new_profiles():
    runner = CliRunner()
    result = runner.invoke(main, ["agent", "list"])
    assert result.exit_code == 0
    assert "kimi-k2-6" in result.output
    assert "deepseek-v4-flash" in result.output
    assert "minimax-m2-7" in result.output


def test_agent_serve_dry_run_deepseek():
    runner = CliRunner()
    result = runner.invoke(
        main, ["agent", "serve", "deepseek-v4-flash", "--dry-run"]
    )
    assert result.exit_code == 0
    assert "vllm.entrypoints.openai.api_server" in result.output
    assert "--kt-method" not in result.output


def test_agent_serve_dry_run_minimax():
    runner = CliRunner()
    result = runner.invoke(
        main, ["agent", "serve", "minimax-m2-7", "--dry-run"]
    )
    assert result.exit_code == 0
    assert "--tool-call-parser minimax-m2" in result.output


# -- Bench module tests ------------------------------------------------------


def test_bench_presets_exist():
    from imas_ambix.agent.bench import BENCH_PRESETS

    assert "short" in BENCH_PRESETS
    assert "medium" in BENCH_PRESETS
    assert "long" in BENCH_PRESETS
    assert "code" in BENCH_PRESETS
    assert "thinking" in BENCH_PRESETS
    assert "tool_use" in BENCH_PRESETS


def test_bench_result_dataclass():
    from imas_ambix.agent.bench import BenchResult

    r = BenchResult(completion_tokens=100, total_time_s=2.0, tokens_per_second=50.0)
    assert r.ok
    assert r.tokens_per_second == 50.0

    r_err = BenchResult(error="timeout")
    assert not r_err.ok


def test_bench_suite_summary():
    from imas_ambix.agent.bench import BenchResult, BenchSuite

    suite = BenchSuite(
        model="test",
        results=[
            BenchResult(
                completion_tokens=100,
                total_time_s=2.0,
                tokens_per_second=50.0,
                time_to_first_token_s=0.1,
            ),
            BenchResult(
                completion_tokens=200,
                total_time_s=4.0,
                tokens_per_second=50.0,
                time_to_first_token_s=0.2,
            ),
        ],
    )
    s = suite.summary()
    assert s["passed"] == 2
    assert s["total_tokens"] == 300
    assert s["avg_tps"] == 50.0


def test_bench_cli_no_server():
    """Bench command should fail gracefully when no server is running."""
    runner = CliRunner()
    result = runner.invoke(
        main, ["agent", "bench", "deepseek-v4-flash", "--url", "http://localhost:59999"]
    )
    assert result.exit_code != 0
    assert "Cannot reach server" in result.output


# -- Setup command tests -----------------------------------------------------


def test_setup_dry_run_vllm():
    """Setup --dry-run should print a valid SLURM script for vllm."""
    runner = CliRunner()
    result = runner.invoke(main, ["agent", "setup", "vllm", "--dry-run"])
    assert result.exit_code == 0
    assert "ambix-setup-vllm" in result.output
    assert "uv sync" in result.output
    assert "vllm" in result.output
    assert "import vllm" in result.output or "vllm" in result.output


def test_setup_dry_run_sglang():
    """Setup --dry-run should print a valid SLURM script for sglang."""
    runner = CliRunner()
    result = runner.invoke(main, ["agent", "setup", "sglang", "--dry-run"])
    assert result.exit_code == 0
    assert "ambix-setup-sglang" in result.output
    assert "uv sync" in result.output
    assert "import sglang" in result.output or "sglang" in result.output


def test_setup_invalid_engine():
    """Setup should reject unknown engine types."""
    runner = CliRunner()
    result = runner.invoke(main, ["agent", "setup", "nonexistent"])
    assert result.exit_code != 0


def test_engine_pyproject_bundled():
    """Engine pyproject.toml files should be accessible via importlib.resources."""
    from imas_ambix.agent.cli import _engine_pyproject

    for engine in ("vllm", "sglang"):
        content = _engine_pyproject(engine)
        assert "[project]" in content
        assert f"ambix-agent-{engine}" in content
