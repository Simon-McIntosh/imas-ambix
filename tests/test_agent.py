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
    assert profile.engine.type == "sglang"
    assert profile.engine.tensor_parallel == 4
    assert profile.engine.ktransformers is None
    assert profile.model.max_context == 1048576
    assert profile.model.size_gb == 164


def test_load_minimax_m2_7_profile():
    profile = load_profile("minimax-m2-7")
    assert profile.slug == "minimax-m2-7"
    assert profile.model.hf_repo == "MiniMaxAI/MiniMax-M2.7"
    assert profile.engine.type == "sglang"
    assert profile.engine.tensor_parallel == 4
    assert profile.engine.ktransformers is None
    assert profile.engine.enable_auto_tool_choice is True
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
    assert site.python_path.name == "python"
    assert site.hf_path.name == "hf"


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
    assert str(site.python_path) in script


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
    from imas_ambix.agent.slurm import generate_serve_script

    profile = load_profile("deepseek-v4-flash")
    site = SiteConfig()
    script = generate_serve_script(profile, site, port=8000)

    assert "#!/bin/bash" in script
    assert "#SBATCH --partition=betelgeuse" in script
    assert "#SBATCH --gres=gpu:4" in script
    assert "sglang.launch_server" in script
    # Should NOT contain KTransformers-specific flags
    assert "--kt-method" not in script
    assert "--kt-weight-path" not in script
    assert "--kt-cpuinfer" not in script
    # Should contain the engine type in the log
    assert "engine: sglang" in script
    # Should NOT contain KTransformers env vars
    assert "MALLOC_TRIM_THRESHOLD_" not in script


def test_generate_minimax_serve_script():
    from imas_ambix.agent.slurm import generate_serve_script

    profile = load_profile("minimax-m2-7")
    site = SiteConfig()
    script = generate_serve_script(profile, site, port=8000)

    assert "sglang.launch_server" in script
    assert "--tool-call-parser minimax-m2" in script
    assert "--reasoning-parser minimax-append-think" in script
    assert "--enable-auto-tool-choice" in script
    assert "--kt-method" not in script


def test_generate_download_script_sglang_engine():
    from imas_ambix.agent.slurm import generate_download_script

    profile = load_profile("deepseek-v4-flash")
    site = SiteConfig()
    script = generate_download_script(profile, site)

    assert "hf download" in script
    assert "deepseek-ai/DeepSeek-V4-Flash" in script
    # Download should NOT request GPUs
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
    assert "sglang.launch_server" in result.output
    assert "--kt-method" not in result.output


def test_agent_serve_dry_run_minimax():
    runner = CliRunner()
    result = runner.invoke(
        main, ["agent", "serve", "minimax-m2-7", "--dry-run"]
    )
    assert result.exit_code == 0
    assert "--tool-call-parser minimax-m2" in result.output
    assert "--enable-auto-tool-choice" in result.output
