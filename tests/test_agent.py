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


def test_site_config_gpu_host():
    site = SiteConfig()
    assert site.gpu_host == "98dci4-gpu-0003"
    assert site.default_url == "http://98dci4-gpu-0003:8000"


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


def test_bench_result_status_variants():
    from imas_ambix.agent.bench import BenchResult

    for status in ("passed", "failed", "skipped", "error"):
        r = BenchResult(status=status)
        assert r.status == status


def test_bench_result_ok_property():
    from imas_ambix.agent.bench import BenchResult

    assert BenchResult(status="passed").ok
    assert BenchResult(status="skipped").ok
    assert not BenchResult(status="failed").ok
    assert not BenchResult(status="error").ok
    # Legacy: error field also works
    r = BenchResult(error="timeout", status="error")
    assert not r.ok


def test_bench_result_dataclass():
    from imas_ambix.agent.bench import BenchResult

    r = BenchResult(
        category="throughput",
        test_name="decode_512",
        completion_tokens=100,
        total_time_s=2.0,
        decode_tps=50.0,
    )
    assert r.ok
    assert r.decode_tps == 50.0
    assert r.category == "throughput"


def test_bench_report_to_json():
    import json

    from imas_ambix.agent.bench import BenchReport, BenchResult

    report = BenchReport(
        results=[
            BenchResult(category="throughput", test_name="decode_128", status="passed",
                        decode_tps=50.0, completion_tokens=128, total_time_s=2.5),
        ],
        server_info={"object": "list"},
        timestamp="2025-01-01T00:00:00Z",
        categories_run=["throughput"],
    )
    j = report.to_json()
    data = json.loads(j)
    assert data["timestamp"] == "2025-01-01T00:00:00Z"
    assert len(data["results"]) == 1
    assert data["results"][0]["test_name"] == "decode_128"
    assert "summary" in data


def test_bench_report_summary():
    from imas_ambix.agent.bench import BenchReport, BenchResult

    report = BenchReport(
        results=[
            BenchResult(category="throughput", test_name="decode_128", status="passed",
                        decode_tps=50.0, time_to_first_token_s=0.1),
            BenchResult(category="throughput", test_name="decode_512", status="passed",
                        decode_tps=40.0, time_to_first_token_s=0.2),
            BenchResult(category="throughput", test_name="decode_1024", status="failed",
                        error="timeout"),
            BenchResult(category="tools", test_name="tool_single", status="skipped"),
        ],
        timestamp="2025-01-01T00:00:00Z",
        categories_run=["throughput", "tools"],
    )
    s = report.summary()
    assert "throughput" in s
    assert s["throughput"]["passed"] == 2
    assert s["throughput"]["failed"] == 1
    assert s["throughput"]["avg_decode_tps"] == 45.0
    assert "tools" in s
    assert s["tools"]["skipped"] == 1


def test_bench_report_percentiles():
    from imas_ambix.agent.bench import BenchReport, BenchResult

    results = [
        BenchResult(category="throughput", test_name=f"t{i}", status="passed",
                    decode_tps=float(i + 1))
        for i in range(100)
    ]
    report = BenchReport(results=results, timestamp="2025-01-01T00:00:00Z",
                         categories_run=["throughput"])
    p = report.percentiles("throughput", "decode_tps")
    assert p["p50"] > 0
    assert p["p95"] > p["p50"]
    assert p["p99"] >= p["p95"]

    # Empty category returns zeros
    p_empty = report.percentiles("nonexistent", "decode_tps")
    assert p_empty == {"p50": 0.0, "p95": 0.0, "p99": 0.0}


# -- Filler text generation --------------------------------------------------


def test_build_filler_prompt_approximate_length():
    from imas_ambix.agent.bench import build_filler_prompt

    for target in (1000, 4000, 16000):
        text = build_filler_prompt(target)
        # With 0.75 ratio, char count should be roughly target / 0.75
        assert len(text) > target  # chars > tokens always


def test_build_filler_prompt_unique_content():
    from imas_ambix.agent.bench import build_filler_prompt

    text = build_filler_prompt(4000)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    # First N paragraphs should all be unique (up to topic count)
    unique = set(paragraphs[:20])
    assert len(unique) >= min(len(paragraphs), 20)


# -- Needle construction -----------------------------------------------------


def test_build_needle_haystack_contains_needle():
    from imas_ambix.agent.bench import build_needle_haystack

    haystack, code = build_needle_haystack(4000, "mid")
    assert code.startswith("AMBIX-")
    assert len(code) == 14  # "AMBIX-" + 8 chars
    assert code in haystack
    assert "Project Stellarator" in haystack


def test_build_needle_positions():
    from imas_ambix.agent.bench import build_needle_haystack

    for pos in ("early", "mid", "late"):
        haystack, code = build_needle_haystack(4000, pos)
        assert code in haystack
        # Verify position is roughly correct
        idx = haystack.index(code)
        total = len(haystack)
        frac = idx / total
        expected = {"early": 0.10, "mid": 0.50, "late": 0.90}[pos]
        # Allow generous tolerance since paragraph boundaries shift things
        assert abs(frac - expected) < 0.35, (
            f"{pos}: frac={frac:.2f}, expected ~{expected}"
        )


# -- Tool test definitions ---------------------------------------------------


def test_tool_test_definitions_valid_json_schema():
    import json

    from imas_ambix.agent.bench import ALL_TOOLS

    for tool in ALL_TOOLS:
        assert tool["type"] == "function"
        func = tool["function"]
        assert "name" in func
        assert "parameters" in func
        params = func["parameters"]
        assert params["type"] == "object"
        assert "properties" in params
        # Should round-trip as JSON
        json.dumps(tool)


# -- Category registry -------------------------------------------------------


def test_all_categories_registered():
    from imas_ambix.agent.bench import CATEGORIES

    expected = {"throughput", "prefill", "context", "tools", "reasoning", "concurrency"}
    assert set(CATEGORIES) == expected


def test_category_filter():
    """run_benchmark should accept a subset of categories."""
    from imas_ambix.agent.bench import run_benchmark

    # Just verify it doesn't crash with a subset — actual execution
    # would need a server, but we can check the report structure.
    # We'll test with an unreachable URL so it errors gracefully.
    report = run_benchmark(
        "http://127.0.0.1:1",  # unreachable
        "test-model",
        categories=["throughput"],
        repeat=1,
        warmup=False,
    )
    assert report.categories_run == ["throughput"]
    # All results should be errors since server is unreachable
    for r in report.results:
        assert r.category == "throughput"
        assert r.status == "error"


# -- CLI argument validation -------------------------------------------------


def test_bench_cli_requires_slug_or_url(monkeypatch, tmp_path):
    """Bench command should require either a slug or --url when no default."""
    # Run from a temp dir with no pyproject.toml and no envvar
    monkeypatch.delenv("AMBIX_AGENT_DEFAULT_PROFILE", raising=False)
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["agent", "bench"])
    assert result.exit_code != 0
    assert "slug" in result.output.lower() or "url" in result.output.lower() or "default_profile" in result.output.lower()


def test_bench_cli_no_server_graceful_error():
    """Bench command should fail gracefully when no server is running."""
    runner = CliRunner()
    result = runner.invoke(
        main, ["agent", "bench", "deepseek-v4-flash", "--url", "http://localhost:59999"]
    )
    assert result.exit_code != 0
    assert "Cannot reach server" in result.output


# -- Default profile resolution ----------------------------------------------


def test_default_profile_from_pyproject():
    """Should resolve default_profile from pyproject.toml."""
    from imas_ambix.agent.cli import _default_profile

    result = _default_profile()
    assert result == "deepseek-v4-flash"


def test_default_profile_envvar_overrides(monkeypatch):
    """Envvar should override pyproject default_profile."""
    from imas_ambix.agent.cli import _default_profile

    monkeypatch.setenv("AMBIX_AGENT_DEFAULT_PROFILE", "kimi-k2-6")
    assert _default_profile() == "kimi-k2-6"


def test_default_url_from_pyproject():
    """Should resolve url from pyproject.toml."""
    from imas_ambix.agent.cli import _default_url

    result = _default_url()
    assert result == "http://98dci4-gpu-0003:8000"


def test_default_url_envvar_overrides(monkeypatch):
    """Envvar should override pyproject url."""
    from imas_ambix.agent.cli import _default_url

    monkeypatch.setenv("AMBIX_AGENT_URL", "http://my-host:9000")
    assert _default_url() == "http://my-host:9000"


def test_agent_config_returns_dict():
    """_agent_config should return tool.ambix.agent section."""
    from imas_ambix.agent.cli import _agent_config

    cfg = _agent_config()
    assert isinstance(cfg, dict)
    assert cfg.get("default_profile") == "deepseek-v4-flash"
    assert cfg.get("url") == "http://98dci4-gpu-0003:8000"


def test_resolve_slug_explicit():
    """Explicit slug takes priority over defaults."""
    from imas_ambix.agent.cli import _resolve_slug

    assert _resolve_slug("kimi-k2-6") == "kimi-k2-6"


def test_resolve_slug_falls_back_to_default():
    """None slug should fall back to default_profile."""
    from imas_ambix.agent.cli import _resolve_slug

    result = _resolve_slug(None)
    assert result == "deepseek-v4-flash"


# -- API key resolution ------------------------------------------------------


def test_resolve_api_key_cli_value():
    """CLI flag takes priority."""
    from imas_ambix.agent.cli import _resolve_api_key

    assert _resolve_api_key("sk-test-123") == "sk-test-123"


def test_resolve_api_key_envvar(monkeypatch):
    """Envvar fallback when no CLI flag."""
    from imas_ambix.agent.cli import _resolve_api_key

    monkeypatch.setenv("AMBIX_AGENT_API_KEY", "sk-env-456")
    assert _resolve_api_key(None) == "sk-env-456"


def test_resolve_api_key_none(monkeypatch):
    """Returns None when no key is set anywhere."""
    from imas_ambix.agent.cli import _resolve_api_key

    monkeypatch.delenv("AMBIX_AGENT_API_KEY", raising=False)
    assert _resolve_api_key(None) is None


def test_load_dotenv_file(monkeypatch, tmp_path):
    """Should load API key from .env file."""
    from imas_ambix.agent.cli import _load_dotenv

    monkeypatch.delenv("AMBIX_AGENT_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    dotenv = tmp_path / ".env"
    dotenv.write_text('AMBIX_AGENT_API_KEY="sk-dotenv-789"\n')
    result = _load_dotenv()
    assert result["AMBIX_AGENT_API_KEY"] == "sk-dotenv-789"


# -- Auth headers ------------------------------------------------------------


def test_auth_headers_with_key():
    """Should include Bearer token when api_key is provided."""
    from imas_ambix.agent.bench import _auth_headers

    headers = _auth_headers("sk-test")
    assert headers["Authorization"] == "Bearer sk-test"
    assert headers["Content-Type"] == "application/json"


def test_auth_headers_without_key():
    """Should not include Authorization when no api_key."""
    from imas_ambix.agent.bench import _auth_headers

    headers = _auth_headers(None)
    assert "Authorization" not in headers
    assert headers["Content-Type"] == "application/json"


# -- Shutdown command --------------------------------------------------------


def test_shutdown_command_exists():
    """Shutdown subcommand should be registered."""
    runner = CliRunner()
    result = runner.invoke(main, ["agent", "shutdown", "--help"])
    assert result.exit_code == 0
    assert "Cancel active" in result.output


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
