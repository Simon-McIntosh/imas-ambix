"""Tests for the imas-ambix agent CLI and profile system."""

from contextlib import contextmanager
from pathlib import Path

import pytest
from click.testing import CliRunner

from imas_ambix.agent.profile import SiteConfig, list_profiles, load_profile
from imas_ambix.cli import main


def _with_checkpoint_precision(profile, precision="fp8"):
    """Return a catalog-ready profile without mutating shipped profile data."""
    model = profile.model.model_copy(update={"checkpoint_precision": precision})
    return profile.model_copy(update={"model": model})


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
    assert profile.engine.ktransformers.gpu_experts == 280
    assert profile.engine.moe_runner_backend == "triton"
    assert profile.model.max_context == 262144


def test_load_deepseek_v4_flash_profile():
    profile = load_profile("deepseek-v4-flash")
    assert profile.slug == "deepseek-v4-flash"
    # The vendor and model are the identity worth pinning; the release suffix
    # is expected to move as new checkpoints ship.
    assert profile.model.hf_repo.startswith("deepseek-ai/DeepSeek-V4-Flash")
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


def test_load_glm_5_1_profile():
    profile = load_profile("glm-5-1")
    assert profile.slug == "glm-5-1"
    assert profile.model.hf_repo == "zai-org/GLM-5.1-FP8"
    assert profile.engine.type == "ktransformers"
    assert profile.engine.tensor_parallel == 4
    assert profile.engine.kv_cache_dtype == "bf16"
    assert profile.engine.ktransformers is not None
    assert profile.engine.ktransformers.method == "FP8"
    assert profile.engine.ktransformers.gpu_experts == 40
    assert profile.engine.ktransformers.gpu_prefill_token_threshold == 1024
    assert profile.engine.ktransformers.enable_dynamic_expert_update is True
    assert profile.engine.ktransformers.expert_placement_strategy == "uniform"
    assert profile.engine.parsers.reasoning == "glm45"
    assert profile.engine.parsers.tool_call == "glm47"
    assert profile.model.max_context == 202752


def test_load_mimo_v2_5_pro_profile():
    profile = load_profile("mimo-v2-5-pro")
    assert profile.slug == "mimo-v2-5-pro"
    assert profile.model.hf_repo == "XiaomiMiMo/MiMo-V2.5-Pro"
    assert profile.engine.type == "ktransformers"
    assert profile.engine.tensor_parallel == 4
    assert profile.engine.ktransformers is not None
    assert profile.engine.ktransformers.method == "RAWINT4"
    assert profile.engine.ktransformers.gpu_experts == 160
    assert profile.engine.ktransformers.gpu_prefill_token_threshold == 1024
    assert profile.engine.ktransformers.enable_dynamic_expert_update is True
    assert profile.engine.ktransformers.expert_placement_strategy == "uniform"
    assert profile.engine.parsers.reasoning == "deepseek_r1"
    assert profile.model.max_context == 131072


def test_load_profile_not_found():
    import pytest

    with pytest.raises(FileNotFoundError, match="No profile 'nonexistent'"):
        load_profile("nonexistent")


# -- Profile inheritance (_base) ---------------------------------------------


def test_load_deepseek_v4_flash_2x_inherits_base():
    """2x variant inherits all base settings and overrides only GPU topology."""
    base = load_profile("deepseek-v4-flash")
    variant = load_profile("deepseek-v4-flash-2x")

    # Inherited identity
    assert variant.model.hf_repo == base.model.hf_repo
    assert variant.model.served_name == base.model.served_name
    assert variant.model.size_gb == base.model.size_gb
    assert variant.model.max_context == base.model.max_context

    # Overridden topology
    assert variant.engine.tensor_parallel == 2
    assert variant.slurm.gpus == 2
    assert variant.slurm.cpus != base.slurm.cpus
    assert variant.slurm.memory == "200G"

    # Inherited engine settings
    assert variant.engine.type == base.engine.type
    assert variant.engine.kv_cache_dtype == base.engine.kv_cache_dtype
    assert variant.engine.enable_auto_tool_choice == base.engine.enable_auto_tool_choice
    # -2x overrides context to the full 1M (KV is cheap on 2 cards); base is 512K.
    assert variant.engine.max_total_tokens == 1048576
    assert base.engine.max_total_tokens == 524288
    assert variant.engine.parsers.tool_call == base.engine.parsers.tool_call
    assert variant.engine.parsers.reasoning == base.engine.parsers.reasoning

    # Overridden concurrency cap
    assert variant.engine.max_num_seqs == 512

    # Slug is the variant, not the base
    assert variant.slug == "deepseek-v4-flash-2x"


def test_variant_weights_slug_redirects_to_base():
    """A topology variant loads the same weights directory as its base."""
    base = load_profile("deepseek-v4-flash")
    variant = load_profile("deepseek-v4-flash-2x")
    assert variant.model.weights_slug == base.weights_directory_slug


def test_profile_without_weights_slug_uses_its_own_slug():
    """With no override declared, a profile's weights live under its own slug.

    A profile MAY declare ``weights_slug`` to keep one release's shards out of
    another's directory, so the absence of an override -- not the absence of a
    value -- is what this pins.
    """
    profile = load_profile("glm-5-2")
    assert profile.model.weights_slug is None
    assert profile.weights_directory_slug == profile.slug


def test_variant_model_dir_uses_base_slug():
    """SiteConfig.model_dir for a variant resolves to the base's directory."""
    site = SiteConfig()
    base = load_profile("deepseek-v4-flash")
    variant = load_profile("deepseek-v4-flash-2x")

    assert site.model_dir(base) == site.model_dir(variant)
    expected = f"agents/{base.weights_directory_slug}/model"
    assert str(site.model_dir(variant)).endswith(expected)


def test_variant_cache_dir_uses_base_slug():
    site = SiteConfig()
    base = load_profile("deepseek-v4-flash")
    variant = load_profile("deepseek-v4-flash-2x")
    expected = f"agents/{base.weights_directory_slug}/.cache"
    assert str(site.cache_dir(variant)).endswith(expected)


def test_inheritance_cycle_detection():
    """A circular _base chain raises a clear ValueError."""
    import pytest

    from imas_ambix.agent import profile as prof_mod

    with pytest.raises(ValueError, match="Circular profile inheritance"):
        prof_mod._load_raw("x", _seen=frozenset({"x"}))


def test_list_profiles_includes_variant():
    """The 2x variant appears in list_profiles()."""
    slugs = list_profiles()
    assert "deepseek-v4-flash" in slugs
    assert "deepseek-v4-flash-2x" in slugs


def test_load_glm_5_2_profile():
    profile = load_profile("glm-5-2")
    assert profile.slug == "glm-5-2"
    assert profile.model.hf_repo == "zai-org/GLM-5.2-FP8"
    assert profile.engine.type == "vllm"
    assert profile.engine.tensor_parallel == 8
    assert profile.engine.ktransformers is None
    assert profile.engine.enable_auto_tool_choice is True
    assert profile.engine.kv_cache_dtype == "bfloat16"
    assert profile.engine.speculative_method == "mtp"
    assert profile.engine.speculative_num_tokens == 5
    assert profile.engine.parsers.tool_call == "glm47"
    assert profile.engine.parsers.reasoning == "glm45"
    assert profile.model.max_context == 1048576
    assert profile.slurm.gpus == 8


def test_generate_glm_5_2_serve_script():
    """GLM-5.2 launch arguments reflect its loaded deployment profile."""
    from imas_ambix.agent.slurm import generate_serve_script

    profile = _with_checkpoint_precision(load_profile("glm-5-2"))
    site = SiteConfig()
    script = generate_serve_script(profile, site, port=8000)

    assert "#SBATCH --gres=gpu:8" in script
    assert "--tensor-parallel-size 8" in script
    assert "vllm.entrypoints.openai.api_server" in script
    assert "--kv-cache-dtype bfloat16" in script
    assert "--speculative-config.method mtp" in script
    assert "--speculative-config.num_speculative_tokens 5" in script
    assert "--tool-call-parser glm47" in script
    assert "--reasoning-parser glm45" in script
    assert "agents/glm-5-2/model" in script
    expected_context = profile.engine.max_total_tokens
    assert expected_context is not None
    assert script.count(f"--max-model-len {expected_context}") == 1


def _serve_cli(monkeypatch, resolved_key):
    """Patch profile loading and key resolution, then return a CLI runner."""
    from imas_ambix.agent import cli as cli_mod

    monkeypatch.setattr(cli_mod, "_resolve_api_key", lambda v: v or resolved_key)
    real_load_profile = cli_mod.load_profile
    monkeypatch.setattr(
        cli_mod,
        "load_profile",
        lambda slug: _with_checkpoint_precision(real_load_profile(slug)),
    )
    return CliRunner()


@pytest.mark.parametrize(
    ("gres", "expected"),
    [
        # squeue %b on this scheduler: TRES form, with and without a type.
        ("gres/gpu:2", 2),
        ("gres/gpu:h200:1", 1),
        ("gres/gpu:h200:8(S:0-1)", 8),
        # A TRES list, where the resource name is neither first nor last.
        ("cpu=30,mem=600G,node=1,billing=30,gres/gpu=8", 8),
        # Unprefixed spellings.
        ("gpu:2", 2),
        ("gpu:h200:1(IDX:0-1)", 1),
        # No GPU allocation at all.
        ("N/A", None),
        ("", None),
        ("cpu=4,mem=16G", None),
    ],
)
def test_allocated_gpus_reads_every_scheduler_spelling(gres, expected):
    """A prefixed resource name must not read as an absent allocation."""
    from imas_ambix.agent.cli import _allocated_gpus

    assert _allocated_gpus(gres) == expected


def test_serve_is_keyless_unless_auth_is_requested(monkeypatch):
    """A resolvable key is not enough: the default endpoint stays open."""
    runner = _serve_cli(monkeypatch, "should-not-be-used")
    result = runner.invoke(main, ["agent", "serve", "glm-5-2", "--dry-run"])
    assert result.exit_code == 0
    # No VLLM_API_KEY export → open endpoint.
    assert "VLLM_API_KEY" not in result.output
    assert "should-not-be-used" not in result.output


def test_serve_auth_flag_enforces_the_resolved_key(monkeypatch):
    """`--auth` arms the engine's key middleware without exposing the value."""
    runner = _serve_cli(monkeypatch, "resolved-from-shared-file")
    result = runner.invoke(main, ["agent", "serve", "glm-5-2", "--auth", "--dry-run"])
    assert result.exit_code == 0
    assert "VLLM_API_KEY" in result.output
    # --dry-run masks the secret rather than printing it.
    assert "resolved-from-shared-file" not in result.output


def test_serve_api_key_arms_auth_without_the_flag(monkeypatch):
    """Naming a key means wanting it enforced, not silently discarded."""
    runner = _serve_cli(monkeypatch, None)
    result = runner.invoke(
        main,
        ["agent", "serve", "glm-5-2", "--api-key", "explicit-key", "--dry-run"],
    )
    assert result.exit_code == 0
    assert "VLLM_API_KEY" in result.output
    assert "explicit-key" not in result.output


def test_serve_auth_without_a_resolvable_key_fails(monkeypatch):
    """Requesting protection and getting an open port instead is a launch error."""
    runner = _serve_cli(monkeypatch, None)
    result = runner.invoke(main, ["agent", "serve", "glm-5-2", "--auth", "--dry-run"])
    assert result.exit_code != 0
    assert "no API key resolved" in result.output


def test_serve_rejects_the_retired_negative_flag(monkeypatch):
    """The old opt-out spelling must fail loudly rather than serve keyed."""
    runner = _serve_cli(monkeypatch, "should-not-be-used")
    result = runner.invoke(
        main, ["agent", "serve", "glm-5-2", "--no-auth", "--dry-run"]
    )
    assert result.exit_code != 0


def test_serve_uses_profile_port_without_an_override(monkeypatch):
    runner = _serve_cli(monkeypatch, None)

    result = runner.invoke(main, ["agent", "serve", "glm-5-3", "--dry-run"])

    assert result.exit_code == 0
    assert "#SBATCH --comment=ambix-serve;port=18801" in result.output


def test_explicit_serve_port_overrides_profile(monkeypatch):
    runner = _serve_cli(monkeypatch, None)

    result = runner.invoke(
        main,
        ["agent", "serve", "glm-5-3", "--port", "19999", "--dry-run"],
    )

    assert result.exit_code == 0
    assert "#SBATCH --comment=ambix-serve;port=19999" in result.output


def test_every_profile_declares_a_distinct_port():
    profiles = [load_profile(slug) for slug in list_profiles()]
    declared = {profile.slug: profile.slurm.port for profile in profiles}

    assert all(port is not None for port in declared.values())
    assert len(set(declared.values())) == len(declared), declared


def test_serve_refuses_a_port_held_by_running_serve(monkeypatch):
    from imas_ambix.agent import cli as cli_mod
    from imas_ambix.agent import slurm as slurm_mod

    runner = _serve_cli(monkeypatch, None)
    monkeypatch.setattr(
        cli_mod,
        "_running_jobs",
        lambda site: [
            _live_job(
                job_id="314",
                name="existing-model",
                port=18801,
                gpus=2,
            )
        ],
    )

    submitted = False

    def unexpected_submit(script):
        nonlocal submitted
        submitted = True
        return "315"

    monkeypatch.setattr(slurm_mod, "submit_script", unexpected_submit)

    result = runner.invoke(main, ["agent", "serve", "glm-5-3"])

    assert result.exit_code != 0
    assert "existing-model" in result.output
    assert "314" in result.output
    assert "18801" in result.output
    assert not submitted


def test_serve_refuses_a_port_held_by_running_router(monkeypatch):
    from imas_ambix.agent import cli as cli_mod
    from imas_ambix.agent import slurm as slurm_mod

    runner = _serve_cli(monkeypatch, None)
    monkeypatch.setattr(
        cli_mod,
        "_running_jobs",
        lambda site: [
            {
                "jobid": "271",
                "name": "ambix-router",
                "state": "RUNNING",
                "time": "4:20",
                "node": "router-node",
                "gres": "N/A",
                "comment": "ambix-router;port=18801",
            }
        ],
    )

    submitted = False

    def unexpected_submit(script):
        nonlocal submitted
        submitted = True
        return "272"

    monkeypatch.setattr(slurm_mod, "submit_script", unexpected_submit)

    result = runner.invoke(main, ["agent", "serve", "glm-5-3"])

    assert result.exit_code != 0
    assert "ambix-router" in result.output
    assert "271" in result.output
    assert "18801" in result.output
    assert not submitted


def test_generate_vllm_2x_serve_script():
    """2x serve script requests 2 GPUs and TP=2."""
    from imas_ambix.agent.slurm import generate_serve_script

    profile = _with_checkpoint_precision(load_profile("deepseek-v4-flash-2x"))
    site = SiteConfig()
    script = generate_serve_script(profile, site, port=8000)

    assert "#SBATCH --gres=gpu:2" in script
    assert "--tensor-parallel-size 2" in script
    assert "vllm.entrypoints.openai.api_server" in script
    assert f"agents/{profile.weights_directory_slug}/model" in script


# -- SiteConfig --------------------------------------------------------------


def test_site_config_defaults():
    site = SiteConfig()
    assert site.base_dir == "/work/projects/imas_gpu"
    assert site.engine_env_root == str(
        Path.home() / ".local" / "share" / "ambix" / "engine-envs"
    )
    assert site.engine_env_min_free_gb == 32
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
    assert site.env_dir("vllm") == Path(site.engine_env_root) / "vllm"
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
    assert site.default_port == 18800
    assert site.global_origin == "http://98dci4-gpu-0003:18800"


def test_site_config_from_env(monkeypatch):
    monkeypatch.setenv("AMBIX_AGENT_BASE_DIR", "/tmp/test")
    monkeypatch.setenv("AMBIX_AGENT_ENGINE_ENV_ROOT", "/tmp/engine-envs")
    monkeypatch.setenv("AMBIX_AGENT_ENGINE_ENV_MIN_FREE_GB", "48")
    monkeypatch.setenv("AMBIX_AGENT_PARTITION", "test-partition")
    monkeypatch.setenv("AMBIX_AGENT_DOWNLOAD_PARTITION", "test-dl")
    site = SiteConfig.from_env()
    assert site.base_dir == "/tmp/test"
    assert site.engine_env_root == "/tmp/engine-envs"
    assert site.engine_env_min_free_gb == 48
    assert site.partition == "test-partition"
    assert site.download_partition == "test-dl"


def test_site_global_origin_is_independent_of_per_serve_overrides(monkeypatch):
    monkeypatch.setenv("AMBIX_AGENT_GLOBAL_URL", "https://catalog.example:19444/")
    monkeypatch.setenv("AMBIX_AGENT_GPU_HOST", "transient-backend.example")
    monkeypatch.setenv("AMBIX_AGENT_PORT", "29999")

    site = SiteConfig.from_env()

    assert site.global_origin == "https://catalog.example:19444"
    assert site.gpu_host == "transient-backend.example"
    assert site.default_port == 29999


@pytest.mark.parametrize(
    "origin",
    [
        "",
        "catalog.example:18800",
        "ftp://catalog.example:18800",
        "http://",
        "http://user@catalog.example:18800",
        "http://catalog.example:bad",
        "http://catalog.example:18800/v1",
        "http://catalog.example:18800?route=other",
        "http://catalog.example:18800#fragment",
    ],
)
def test_site_global_origin_rejects_malformed_values(origin):
    with pytest.raises(ValueError, match="global origin"):
        SiteConfig(global_origin=origin)


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
    assert "--max-total-tokens 49152" in script
    assert "--moe-runner-backend triton" in script
    assert "--port" in script
    assert "#SBATCH --comment=ambix-serve;port=8000" in script
    assert "ssh -N -L" not in script and "http://$(hostname):$PORT" in script


def test_agent_serve_gpu_help_matches_core_scaling():
    result = CliRunner().invoke(main, ["agent", "serve", "--help"])
    assert (
        result.exit_code == 0
        and "Host cores do not scale with cards" in result.output
    )


def test_serve_script_launches_drain_sidecar():
    """Every serving job must launch the drain-forensics sidecar so a
    teardown-time node drain is diagnosable (the serving job is the longest-
    lived, most teardown-prone job on the node)."""
    from imas_ambix.agent.slurm import generate_serve_script

    profile = load_profile("kimi-k2-6")
    site = SiteConfig()
    script = generate_serve_script(profile, site, port=8000)

    assert "drain_sidecar.sh" in script
    # Backgrounded so it samples through the kill window, guarded so a missing
    # script is a no-op rather than a `set -e` serve failure.
    assert 'bash "$_AMBIX_SIDECAR" &' in script
    # Launched before the server so the whole job life is sampled.
    assert script.index("_AMBIX_SIDECAR") < script.index("server")


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


def test_agent_list(monkeypatch):
    from imas_ambix.agent import cli as cli_mod

    monkeypatch.setattr(cli_mod, "_serving_slugs", lambda site: set())
    runner = CliRunner()
    result = runner.invoke(main, ["agent", "list"])
    assert result.exit_code == 0
    assert "kimi-k2-6" in result.output


def test_agent_info():
    runner = CliRunner()
    result = runner.invoke(main, ["agent", "info", "kimi-k2-6"])
    assert result.exit_code == 0
    assert "Kimi-K2.6" in result.output
    assert "ktransformers" in result.output.lower()


def _live_job(
    *,
    job_id="42",
    name="glm-5-2",
    node="98dci4-gpu-0003",
    port=19123,
    gpus=8,
):
    return {
        "jobid": job_id,
        "name": name,
        "state": "RUNNING",
        "time": "5:00",
        "node": node,
        "gres": f"gpu:h200:{gpus}",
        "comment": f"ambix-serve;port={port}",
    }


def _ready_probe(model="glm-5.2", context=131072):
    from imas_ambix.agent.cli import ModelMetadata, ProbeResult

    return ProbeResult("ready", (ModelMetadata(model, context),))


def test_running_jobs_parses_scheduler_route_fields(monkeypatch):
    """Scheduler output preserves allocation, port comment, node, and job id."""
    import subprocess

    from imas_ambix.agent import cli as cli_mod

    monkeypatch.setattr(
        cli_mod.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0],
            0,
            "malformed scheduler row\n"
            "42|glm-5-2|RUNNING|5:00|gpu-node|gpu:h200:6|ambix-serve;port=19123\n",
            "",
        ),
    )
    jobs = cli_mod._running_jobs(SiteConfig())

    assert jobs == [
        {
            "jobid": "42",
            "name": "glm-5-2",
            "state": "RUNNING",
            "time": "5:00",
            "node": "gpu-node",
            "gres": "gpu:h200:6",
            "comment": "ambix-serve;port=19123",
        }
    ]


def test_running_jobs_reports_scheduler_failure(monkeypatch):
    """A scheduler query failure is not reported as an empty queue."""
    import subprocess

    from imas_ambix.agent import cli as cli_mod

    monkeypatch.setattr(
        cli_mod.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 1, "", "denied"),
    )
    with pytest.raises(Exception, match="denied"):
        cli_mod._running_jobs(SiteConfig())


@pytest.mark.parametrize("gpus", [2, 4, 8])
def test_card_count_override_never_inflates_host_cores(gpus):
    """Cores must not follow the card count off a profile default.

    An engine's cores run one API server, one engine core and one worker per
    rank, and every worker is mostly blocked on the device, so a count derived
    from cards exhausts a reservation two groups share and strands a co-running
    job on cores beside idle GPUs.
    """
    from imas_ambix.agent.cli import _scale_profile

    base = load_profile("deepseek-v4-flash")
    resolved = _scale_profile(base, gpus)
    assert resolved.slurm.gpus == gpus
    assert resolved.engine.tensor_parallel == gpus

    # Cores come from exactly one of two places: a variant that declares them
    # for this topology, or the base inherited unchanged. Never a ratio.
    declared = (base.gpu_variants.get(gpus) or {}).get("slurm", {}).get("cpus")
    assert resolved.slurm.cpus == (declared or base.slurm.cpus)

    # Host memory does still follow the cards, since it stages the weights.
    assert resolved.slurm.memory != base.slurm.memory or gpus == base.slurm.gpus


def test_every_full_gpu_serve_leaves_reservation_headroom():
    """A full-GPU profile must not claim the whole shared core reservation.

    Host cores are load-bearing only where experts are computed on the host, so
    a device-resident engine asking for the reservation ceiling is what leaves a
    neighbour pending on cores. CPU-offloading engines are exempt by mechanism.
    """
    from imas_ambix.agent.profile import list_profiles

    reservation_cores = 30
    offenders = []
    for slug in list_profiles():
        profile = load_profile(slug)
        if profile.engine.type == "ktransformers":
            continue  # cold experts are computed on the host; cores are the work
        for resolved in [profile, *(profile.for_gpus(n) for n in (2, 4, 6, 8))]:
            if resolved.slurm.cpus >= reservation_cores:
                offenders.append((slug, resolved.slurm.gpus, resolved.slurm.cpus))
    assert not offenders, f"full-GPU serves claiming the reservation: {offenders}"


def test_endpoint_requires_key_reads_the_endpoint_not_the_key_file(monkeypatch):
    """An open port must report open, whatever the shared key file holds."""
    import io
    import urllib.request

    from imas_ambix.agent.cli import _endpoint_requires_key

    class Response(io.BytesIO):
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    def open_unauthenticated(request, timeout):
        # The whole point is that no credential is offered.
        assert request.get_header("Authorization") is None
        return Response(b'{"data":[{"id":"m"}]}')

    monkeypatch.setattr(urllib.request, "urlopen", open_unauthenticated)
    assert _endpoint_requires_key("http://gpu-node:19123") is False


@pytest.mark.parametrize("code", [401, 403])
def test_endpoint_requires_key_detects_enforcement(monkeypatch, code):
    """A rejected unauthenticated probe means the endpoint enforces a key."""
    import urllib.error
    import urllib.request

    from imas_ambix.agent.cli import _endpoint_requires_key

    def refuse(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url, code, "denied", {}, None
        )

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    assert _endpoint_requires_key("http://gpu-node:19123") is True


def test_endpoint_requires_key_is_inconclusive_when_unreachable(monkeypatch):
    """An unreachable endpoint yields no verdict rather than a false one."""
    import urllib.request

    from imas_ambix.agent.cli import _endpoint_requires_key

    def unreachable(request, timeout):
        raise OSError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", unreachable)
    assert _endpoint_requires_key("http://gpu-node:19123") is None


def test_engine_facts_report_the_allocated_card_count(monkeypatch):
    """Engine facts follow the running allocation, not the profile default."""
    from imas_ambix.agent.cli import _engine_facts, _scale_profile

    base = load_profile("deepseek-v4-flash")
    assert f"TP={base.slurm.gpus}" in _engine_facts(base)
    resolved = _scale_profile(base, 2)
    facts = _engine_facts(resolved)
    assert "TP=2" in facts
    assert f"TP={base.engine.tensor_parallel}" not in facts


def test_probe_endpoint_authenticated_metadata(monkeypatch):
    """Authenticated probes retain model and context metadata in memory."""
    import io
    import urllib.request

    from imas_ambix.agent.cli import _probe_endpoint

    class Response(io.BytesIO):
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    def open_request(request, timeout):
        assert request.get_header("Authorization") == "Bearer secret"
        return Response(b'{"data":[{"id":"future-model","max_model_len":262144}]}')

    monkeypatch.setattr(urllib.request, "urlopen", open_request)
    probe = _probe_endpoint("http://gpu-node:19123", "secret")

    assert probe.readiness == "ready"
    assert probe.models[0].model_id == "future-model"
    assert probe.models[0].max_context == 262144


def test_probe_endpoint_keyless_omits_authorization(monkeypatch):
    """A keyless endpoint is probed without an Authorization header."""
    import io
    import urllib.request

    from imas_ambix.agent.cli import _probe_endpoint

    class Response(io.BytesIO):
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    def open_request(request, timeout):
        assert request.get_header("Authorization") is None
        return Response(b'{"data":[{"id":"keyless-model"}]}')

    monkeypatch.setattr(urllib.request, "urlopen", open_request)
    probe = _probe_endpoint("http://gpu-node:19123", None)

    assert probe.readiness == "ready"
    assert probe.models[0].model_id == "keyless-model"


@pytest.mark.parametrize(
    ("payload", "readiness"),
    [(b"not-json", "malformed response"), (b'{"data":[]}', "empty models")],
)
def test_probe_endpoint_rejects_unusable_payloads(monkeypatch, payload, readiness):
    """Malformed or empty models responses never establish readiness."""
    import io
    import urllib.request

    from imas_ambix.agent.cli import _probe_endpoint

    class Response(io.BytesIO):
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    monkeypatch.setattr(
        urllib.request, "urlopen", lambda request, timeout: Response(payload)
    )
    assert _probe_endpoint("http://gpu-node:19123", None).readiness == readiness


@pytest.mark.parametrize("code", [401, 403])
def test_probe_endpoint_reports_authentication_failure(monkeypatch, code):
    """Authentication failures remain distinct from network unreachability."""
    import urllib.error
    import urllib.request

    from imas_ambix.agent.cli import _probe_endpoint

    def reject(request, timeout):
        raise urllib.error.HTTPError(request.full_url, code, "denied", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", reject)
    assert _probe_endpoint("http://gpu-node:19123", "wrong").readiness == "auth-fail"


def test_probe_endpoint_reports_timeout(monkeypatch):
    """A timed-out endpoint cannot enter the ready route set."""
    import urllib.request

    from imas_ambix.agent.cli import _probe_endpoint

    def time_out(request, timeout):
        raise TimeoutError

    monkeypatch.setattr(urllib.request, "urlopen", time_out)
    assert _probe_endpoint("http://gpu-node:19123", None).readiness == "unreachable"


def test_running_allocation_requires_successful_probe(monkeypatch):
    """A running allocation with an unreachable endpoint is not live."""
    from imas_ambix.agent import cli as cli_mod

    monkeypatch.setattr(
        cli_mod, "_probe_endpoint", lambda url, key: cli_mod.ProbeResult("unreachable")
    )
    routes, rejected = cli_mod._discover_live_routes(
        SiteConfig(), None, jobs=[_live_job()]
    )

    assert routes == []
    assert rejected == ["job 42: unreachable"]


def test_pre_comment_job_recovers_port_without_exposing_batch_script(
    monkeypatch, capsys
):
    """Scheduler batch metadata can qualify a running pre-comment serve."""
    import subprocess

    from imas_ambix.agent import cli as cli_mod

    job = _live_job(port=19444, gpus=4)
    job["comment"] = ""
    script_marker = "batch-script-private-content"
    seen = []

    def batch_script(args, **kwargs):
        seen.append(args)
        return subprocess.CompletedProcess(
            args,
            0,
            f"#!/bin/bash\n{script_marker}\nPORT=${{AMBIX_PORT:-19444}}\n",
            "",
        )

    monkeypatch.setattr(cli_mod.subprocess, "run", batch_script)
    monkeypatch.setattr(
        cli_mod, "_probe_endpoint", lambda url, key: _ready_probe("glm-5.2")
    )

    routes, rejected = cli_mod._discover_live_routes(SiteConfig(), None, jobs=[job])
    captured = capsys.readouterr()

    assert seen == [["scontrol", "write", "batch_script", "42", "-"]]
    assert rejected == []
    assert routes[0].port == 19444
    assert routes[0].base_url == "http://98dci4-gpu-0003:19444"
    assert script_marker not in captured.out + captured.err


def test_future_server_model_is_discovered_without_profile(monkeypatch):
    """Server metadata, not the local profile catalogue, names a live model."""
    from imas_ambix.agent import cli as cli_mod

    monkeypatch.setattr(
        cli_mod, "_probe_endpoint", lambda url, key: _ready_probe("glm-5.3")
    )
    routes, rejected = cli_mod._discover_live_routes(
        SiteConfig(), None, jobs=[_live_job(name="future-serve")]
    )

    assert rejected == []
    assert routes[0].model_id == "glm-5.3"
    assert routes[0].job_name == "future-serve"


@pytest.mark.parametrize("gpus", [2, 4, 6, 8])
def test_live_route_uses_actual_h200_topology(monkeypatch, gpus):
    """Topology labels come from each allocation rather than profile defaults."""
    from imas_ambix.agent import cli as cli_mod

    monkeypatch.setattr(
        cli_mod, "_probe_endpoint", lambda url, key: _ready_probe("deepseek-v4-flash")
    )
    routes, _ = cli_mod._discover_live_routes(
        SiteConfig(), None, jobs=[_live_job(gpus=gpus)]
    )

    assert routes[0].topology == f"{gpus}×H200"
    assert f"@{gpus}xh200#42" in routes[0].selector


def test_duplicate_model_routes_have_unique_selectors(monkeypatch):
    """Matching model ids remain unambiguous through topology and job identity."""
    from imas_ambix.agent import cli as cli_mod

    monkeypatch.setattr(
        cli_mod, "_probe_endpoint", lambda url, key: _ready_probe("glm-5.2")
    )
    routes, _ = cli_mod._discover_live_routes(
        SiteConfig(),
        None,
        jobs=[_live_job(job_id="42", gpus=4), _live_job(job_id="43", gpus=8)],
    )

    assert len({route.selector for route in routes}) == 2


def test_one_live_route_selects_automatically():
    """A sole live route needs no selector."""
    from imas_ambix.agent import cli as cli_mod

    route = cli_mod.LiveRoute(
        "glm-5.2",
        "gpu-node",
        19123,
        8,
        "42",
        "http://gpu-node:19123",
        131072,
        "ready",
        "glm-5-2",
    )
    assert cli_mod._resolve_live_route([route], None, interactive=False) is route


def test_noninteractive_ambiguous_route_requires_selector():
    """Non-interactive selection never guesses between live allocations."""
    from imas_ambix.agent import cli as cli_mod

    routes = [
        cli_mod.LiveRoute("glm-5.2", "n", 1, 4, "42", "http://n:1", None, "ready", "a"),
        cli_mod.LiveRoute("glm-5.2", "n", 2, 8, "43", "http://n:2", None, "ready", "b"),
    ]
    with pytest.raises(Exception, match="Ambiguous live route"):
        cli_mod._resolve_live_route(routes, "glm-5.2", interactive=False)


def test_interactive_ambiguous_route_prompts_for_choice(monkeypatch):
    """Interactive selection presents the live routes and honours the choice."""
    from imas_ambix.agent import cli as cli_mod

    routes = [
        cli_mod.LiveRoute("glm-5.2", "n", 1, 4, "42", "http://n:1", None, "ready", "a"),
        cli_mod.LiveRoute("glm-5.2", "n", 2, 8, "43", "http://n:2", None, "ready", "b"),
    ]
    monkeypatch.setattr(cli_mod.click, "prompt", lambda *args, **kwargs: 2)
    selected = cli_mod._resolve_live_route(routes, "glm-5.2", interactive=True)

    assert selected.job_id == "43"


def test_zero_live_routes_fail_locally():
    """An empty local route set produces an error rather than a fallback."""
    from imas_ambix.agent import cli as cli_mod

    with pytest.raises(Exception, match="No ready local model route"):
        cli_mod._resolve_live_route([], None, interactive=False)


def test_exact_job_and_topology_selectors_resolve():
    """Job ids and topology-qualified model selectors resolve deterministically."""
    from imas_ambix.agent import cli as cli_mod

    routes = [
        cli_mod.LiveRoute("glm-5.2", "n", 1, 4, "42", "http://n:1", None, "ready", "a"),
        cli_mod.LiveRoute("glm-5.2", "n", 2, 8, "43", "http://n:2", None, "ready", "b"),
    ]
    assert cli_mod._resolve_live_route(routes, "42", interactive=False).job_id == "42"
    assert (
        cli_mod._resolve_live_route(routes, "glm-5.2@8xh200", interactive=False).job_id
        == "43"
    )


def test_explicit_url_and_model_are_probe_validated(monkeypatch):
    """Explicit overrides remain candidates and cannot bypass model membership."""
    from imas_ambix.agent import cli as cli_mod

    monkeypatch.setattr(
        cli_mod, "_probe_endpoint", lambda url, key: _ready_probe("reported-model")
    )
    route = cli_mod._explicit_live_route(
        "http://gpu-node:19123/", "reported-model", None
    )
    assert route.base_url == "http://gpu-node:19123"
    with pytest.raises(Exception, match="not reported"):
        cli_mod._explicit_live_route("http://gpu-node:19123", "stale-model", None)


def test_list_shows_serving_marker(monkeypatch):
    """`list` tags the serving profile and leaves others unmarked."""
    from imas_ambix.agent import cli as cli_mod

    monkeypatch.setattr(cli_mod, "_serving_slugs", lambda site: {"glm-5-2"})
    runner = CliRunner()
    result = runner.invoke(main, ["agent", "list"])
    assert result.exit_code == 0
    assert "serving" in result.output


def _status_route(model="glm-5.2-fp8", port=19123, context=229376):
    from imas_ambix.agent.cli import LiveRoute

    return LiveRoute(
        model,
        "98dci4-gpu-0003",
        port,
        8,
        "42",
        f"http://98dci4-gpu-0003:{port}",
        context,
        "ready",
        "glm-5-2",
    )


def test_status_no_jobs(monkeypatch):
    """`status` reports cleanly when there are no jobs."""
    from imas_ambix.agent import cli as cli_mod

    monkeypatch.setattr(cli_mod, "_running_jobs", lambda site: [])
    runner = CliRunner()
    result = runner.invoke(main, ["agent", "status"])
    assert result.exit_code == 0
    assert "No imas-ambix agent jobs" in result.output


def test_status_running_serve_shows_connection(monkeypatch):
    """`status` prints a connection block for a RUNNING serve job."""
    from imas_ambix.agent import cli as cli_mod

    monkeypatch.setattr(
        cli_mod,
        "_running_jobs",
        lambda site: [
            {
                "jobid": "42",
                "name": "glm-5-2",
                "state": "RUNNING",
                "time": "5:00",
                "node": "98dci4-gpu-0003",
            },
        ],
    )
    monkeypatch.setattr(cli_mod, "_read_key_file", lambda path: "supersecretkey1234")
    monkeypatch.setattr(
        cli_mod,
        "_discover_live_routes",
        lambda site, key, jobs: ([_status_route()], []),
    )
    runner = CliRunner()
    result = runner.invoke(main, ["agent", "status"])
    assert result.exit_code == 0
    assert "glm-5.2-fp8" in result.output  # served model name
    assert "19123" in result.output  # scheduler-recorded URL port
    assert "READY" in result.output.upper()  # endpoint probe verdict
    # Key is masked by default (full key absent).
    assert "supersecretkey1234" not in result.output


def test_status_shows_router_as_preferred_consumer_origin(monkeypatch):
    from imas_ambix.agent import cli as cli_mod

    monkeypatch.setattr(
        cli_mod,
        "_running_jobs",
        lambda site: [
            {
                "jobid": "271",
                "name": "ambix-router",
                "state": "RUNNING",
                "time": "4:20",
                "node": "router-node",
                "gres": "N/A",
                "comment": "ambix-router;port=18808",
            }
        ],
    )
    monkeypatch.setattr(cli_mod, "_read_key_file", lambda path: None)
    monkeypatch.setattr(
        cli_mod,
        "_discover_live_routes",
        lambda site, key, jobs: ([], []),
    )
    monkeypatch.setattr(
        cli_mod,
        "_probe_endpoint",
        lambda url, key: cli_mod.ProbeResult(
            "ready",
            (
                cli_mod.ModelMetadata("deepseek-v4-flash", 524288),
                cli_mod.ModelMetadata("glm-5.3", 98304),
            ),
        ),
    )

    result = CliRunner().invoke(main, ["agent", "status"])

    assert result.exit_code == 0
    assert "Preferred consumer origin" in result.output
    assert "http://router-node:18808" in result.output
    assert "deepseek-v4-flash" in result.output
    assert "glm-5.3" in result.output
    assert "RUNNING" in result.output
    assert "other jobs" not in result.output


def test_status_reveal_shows_full_key(monkeypatch):
    """`status --reveal` prints the full key for the owner."""
    from imas_ambix.agent import cli as cli_mod

    monkeypatch.setattr(
        cli_mod,
        "_running_jobs",
        lambda site: [
            {
                "jobid": "42",
                "name": "glm-5-2",
                "state": "RUNNING",
                "time": "5:00",
                "node": "98dci4-gpu-0003",
            },
        ],
    )
    monkeypatch.setattr(cli_mod, "_read_key_file", lambda path: "supersecretkey1234")
    monkeypatch.setattr(
        cli_mod,
        "_discover_live_routes",
        lambda site, key, jobs: ([_status_route()], []),
    )
    runner = CliRunner()
    result = runner.invoke(main, ["agent", "status", "--reveal"])
    assert result.exit_code == 0
    assert "supersecretkey1234" in result.output


def test_status_key_no_access(monkeypatch):
    """`status` shows '(no access)' when the key file can't be read."""
    from imas_ambix.agent import cli as cli_mod

    def _denied(path):
        raise PermissionError

    monkeypatch.setattr(
        cli_mod,
        "_running_jobs",
        lambda site: [
            {
                "jobid": "42",
                "name": "glm-5-2",
                "state": "RUNNING",
                "time": "5:00",
                "node": "98dci4-gpu-0003",
            },
        ],
    )
    monkeypatch.setattr(cli_mod, "_read_key_file", _denied)
    monkeypatch.setattr(
        cli_mod,
        "_discover_live_routes",
        lambda site, key, jobs: ([_status_route()], []),
    )
    runner = CliRunner()
    result = runner.invoke(main, ["agent", "status"])
    assert result.exit_code == 0
    assert "no access" in result.output


def test_status_uses_direct_url(monkeypatch):
    """status probes the GPU node's port directly (no tunnel) and shows the URL."""
    from imas_ambix.agent import cli as cli_mod

    monkeypatch.setattr(
        cli_mod,
        "_running_jobs",
        lambda site: [
            {
                "jobid": "42",
                "name": "glm-5-2",
                "state": "RUNNING",
                "time": "33:10",
                "node": "98dci4-gpu-0003",
            },
        ],
    )
    monkeypatch.setattr(cli_mod, "_read_key_file", lambda path: "k")
    monkeypatch.setattr(
        cli_mod,
        "_discover_live_routes",
        lambda site, key, jobs: ([_status_route(port=19123)], []),
    )
    runner = CliRunner()
    result = runner.invoke(main, ["agent", "status"])
    assert result.exit_code == 0
    assert "98dci4-gpu-0003:19123" in result.output


def test_status_uptime_formatted(monkeypatch):
    """The UP column renders squeue %M compactly (33:10 → 33m)."""
    from imas_ambix.agent import cli as cli_mod

    monkeypatch.setattr(
        cli_mod,
        "_running_jobs",
        lambda site: [
            {
                "jobid": "42",
                "name": "glm-5-2",
                "state": "RUNNING",
                "time": "33:10",
                "node": "98dci4-gpu-0003",
            },
        ],
    )
    monkeypatch.setattr(cli_mod, "_read_key_file", lambda path: "k")
    monkeypatch.setattr(
        cli_mod,
        "_discover_live_routes",
        lambda site, key, jobs: ([_status_route()], []),
    )
    runner = CliRunner()
    result = runner.invoke(main, ["agent", "status"])
    assert result.exit_code == 0
    assert "33m" in result.output


def test_status_engine_facts_use_served_context(monkeypatch):
    """Engine facts show the served context (256K), not the model's 1M max."""
    from imas_ambix.agent import cli as cli_mod

    monkeypatch.setattr(
        cli_mod,
        "_running_jobs",
        lambda site: [
            {
                "jobid": "42",
                "name": "glm-5-2",
                "state": "RUNNING",
                "time": "5:00",
                "node": "98dci4-gpu-0003",
            },
        ],
    )
    monkeypatch.setattr(cli_mod, "_read_key_file", lambda path: "k")
    monkeypatch.setattr(
        cli_mod,
        "_discover_live_routes",
        lambda site, key, jobs: ([_status_route()], []),
    )
    runner = CliRunner()
    result = runner.invoke(main, ["agent", "status"])
    assert result.exit_code == 0
    # glm-5-2 serves 229376 (224K), even though model.max_context is 1M.
    assert "224K" in result.output
    assert "1.0M" not in result.output


def test_fmt_uptime_variants():
    from imas_ambix.agent.cli import _fmt_uptime

    assert _fmt_uptime("33:10") == "33m"
    assert _fmt_uptime("1:02:03") == "1h02m"
    assert _fmt_uptime("2-03:04:05") == "2d03h"
    assert _fmt_uptime("weird") == "weird"


def test_fmt_context_variants():
    from imas_ambix.agent.cli import _fmt_context

    assert _fmt_context(1048576) == "1.0M"
    assert _fmt_context(262144) == "256K"
    assert _fmt_context(131072) == "128K"
    assert _fmt_context(512) == "512"


def test_agent_serve_dry_run():
    runner = CliRunner()
    result = runner.invoke(main, ["agent", "serve", "kimi-k2-6", "--dry-run"])
    assert result.exit_code == 0
    assert "sglang.launch_server" in result.output
    assert "--partition=betelgeuse" in result.output
    assert "#SBATCH --time=0" in result.output


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

    profile = _with_checkpoint_precision(load_profile("deepseek-v4-flash"))
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
    assert "--max-model-len 524288" in script


def test_generate_minimax_serve_script():
    from imas_ambix.agent.slurm import generate_serve_script

    profile = load_profile("minimax-m2-7")
    site = SiteConfig()
    script = generate_serve_script(profile, site, port=8000)

    assert "sglang.launch_server" in script
    assert "--tool-call-parser minimax-m2" in script
    assert "--reasoning-parser minimax-append-think" in script
    assert "--kt-method" not in script


def test_generate_glm_5_1_serve_script():
    from imas_ambix.agent.slurm import generate_serve_script

    profile = load_profile("glm-5-1")
    site = SiteConfig()
    script = generate_serve_script(profile, site, port=8000)

    assert "sglang.launch_server" in script
    assert "--kt-method FP8" in script
    assert "--kt-num-gpu-experts 40" in script
    assert "--kt-gpu-prefill-token-threshold 1024" in script
    assert "--kt-enable-dynamic-expert-update" in script
    assert "--kt-expert-placement-strategy uniform" in script
    assert "--reasoning-parser glm45" in script
    assert "--tool-call-parser glm47" in script
    assert "--kv-cache-dtype bf16" in script
    assert "--fp8-gemm-backend cutlass" in script
    assert "PYTORCH_ALLOC_CONF=expandable_segments:True" in script
    assert "MALLOC_TRIM_THRESHOLD_" in script


def test_generate_mimo_v2_5_pro_serve_script():
    from imas_ambix.agent.slurm import generate_serve_script

    profile = load_profile("mimo-v2-5-pro")
    site = SiteConfig()
    script = generate_serve_script(profile, site, port=8000)

    assert "sglang.launch_server" in script
    assert "--kt-method RAWINT4" in script
    assert "--kt-num-gpu-experts 160" in script
    assert "--kt-gpu-prefill-token-threshold 1024" in script
    assert "--kt-enable-dynamic-expert-update" in script
    assert "--kt-expert-placement-strategy uniform" in script
    assert "--reasoning-parser deepseek_r1" in script
    assert "--kv-cache-dtype" not in script  # FP8-related flags are commented out
    assert "PYTORCH_ALLOC_CONF=expandable_segments:True" in script


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
    assert "glm-5-1" in result.output
    assert "mimo-v2-5-pro" in result.output


def test_agent_serve_dry_run_deepseek(monkeypatch):
    from imas_ambix.agent import cli as cli_mod

    real_load_profile = cli_mod.load_profile
    monkeypatch.setattr(
        cli_mod,
        "load_profile",
        lambda slug: _with_checkpoint_precision(real_load_profile(slug)),
    )
    runner = CliRunner()
    result = runner.invoke(main, ["agent", "serve", "deepseek-v4-flash", "--dry-run"])
    assert result.exit_code == 0
    assert "vllm.entrypoints.openai.api_server" in result.output
    assert "--kt-method" not in result.output


def test_agent_serve_dry_run_minimax():
    runner = CliRunner()
    result = runner.invoke(main, ["agent", "serve", "minimax-m2-7", "--dry-run"])
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
            BenchResult(
                category="throughput",
                test_name="decode_128",
                status="passed",
                decode_tps=50.0,
                completion_tokens=128,
                total_time_s=2.5,
            ),
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
            BenchResult(
                category="throughput",
                test_name="decode_128",
                status="passed",
                decode_tps=50.0,
                time_to_first_token_s=0.1,
            ),
            BenchResult(
                category="throughput",
                test_name="decode_512",
                status="passed",
                decode_tps=40.0,
                time_to_first_token_s=0.2,
            ),
            BenchResult(
                category="throughput",
                test_name="decode_1024",
                status="failed",
                error="timeout",
            ),
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
        BenchResult(
            category="throughput",
            test_name=f"t{i}",
            status="passed",
            decode_tps=float(i + 1),
        )
        for i in range(100)
    ]
    report = BenchReport(
        results=results, timestamp="2025-01-01T00:00:00Z", categories_run=["throughput"]
    )
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
    assert (
        "slug" in result.output.lower()
        or "url" in result.output.lower()
        or "default_profile" in result.output.lower()
    )


def test_bench_cli_no_server_graceful_error():
    """Bench command should fail gracefully when no server is running."""
    runner = CliRunner()
    result = runner.invoke(
        main, ["agent", "bench", "deepseek-v4-flash", "--url", "http://localhost:59999"]
    )
    assert result.exit_code != 0
    assert "Cannot reach server" in result.output


def test_bench_json_save_is_machine_readable(monkeypatch, tmp_path):
    """Saving a JSON report keeps stdout machine-readable."""
    import io
    import json
    import urllib.request

    from imas_ambix.agent import bench as bench_mod
    from imas_ambix.agent import cli as cli_mod
    from imas_ambix.agent.bench import BenchReport

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    models = b'{"data":[{"id":"saved-model"}]}'
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda request, timeout: Response(models)
    )
    monkeypatch.setattr(cli_mod, "_default_profile", lambda: None)
    monkeypatch.setattr(cli_mod, "_resolve_api_key", lambda value: None)
    report = BenchReport(timestamp="2026-09-02T00:00:00Z")
    monkeypatch.setattr(bench_mod, "run_benchmark", lambda *args, **kwargs: report)

    output_path = tmp_path / "benchmark.json"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "agent",
            "bench",
            "--url",
            "http://engine.test",
            "--model",
            "saved-model",
            "--json",
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == json.loads(report.to_json())
    assert json.loads(output_path.read_text()) == json.loads(report.to_json())
    assert "Results saved to" in result.stderr


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
    assert result == "http://98dci4-gpu-0003:18800"


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
    assert cfg.get("url") == "http://98dci4-gpu-0003:18800"


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


def test_resolve_api_key_none(monkeypatch, tmp_path):
    """Returns None when no key is set anywhere."""
    from imas_ambix.agent.cli import _resolve_api_key

    monkeypatch.delenv("AMBIX_AGENT_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    # Isolate the shared-file lookup from any real /work/projects key file.
    monkeypatch.setenv("AMBIX_AGENT_BASE_DIR", str(tmp_path))
    assert _resolve_api_key(None) is None


def test_resolve_api_key_shared_file(monkeypatch, tmp_path):
    """Falls back to shared agents/.env when no other source."""
    from imas_ambix.agent.cli import _resolve_api_key

    monkeypatch.delenv("AMBIX_AGENT_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AMBIX_AGENT_BASE_DIR", str(tmp_path))
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    (agents_dir / ".env").write_text("AMBIX_AGENT_API_KEY=sk-shared-key\n")
    assert _resolve_api_key(None) == "sk-shared-key"


def test_read_key_file_missing(tmp_path):
    """Returns None for nonexistent file."""
    from imas_ambix.agent.cli import _read_key_file

    assert _read_key_file(tmp_path / "nonexistent") is None


def test_read_key_file_empty_value(tmp_path):
    """Returns None when key is present but value is empty."""
    from imas_ambix.agent.cli import _read_key_file

    f = tmp_path / ".env"
    f.write_text("AMBIX_AGENT_API_KEY=\n")
    assert _read_key_file(f) is None


def test_read_key_file_with_value(tmp_path):
    """Returns the key value."""
    from imas_ambix.agent.cli import _read_key_file

    f = tmp_path / ".env"
    f.write_text("# comment\nAMBIX_AGENT_API_KEY=sk-test-value\nOTHER=foo\n")
    assert _read_key_file(f) == "sk-test-value"


def test_update_dotenv_key_new_file(tmp_path):
    """Creates file and writes key."""
    from imas_ambix.agent.cli import _update_dotenv_key

    f = tmp_path / ".env"
    _update_dotenv_key(f, "MY_KEY", "my-value", header="# header")
    content = f.read_text()
    assert "MY_KEY=my-value" in content
    assert "# header" in content
    assert f.stat().st_mode & 0o777 == 0o600


def test_update_dotenv_key_replace_existing(tmp_path):
    """Replaces existing key in-place, preserving other lines."""
    from imas_ambix.agent.cli import _update_dotenv_key

    f = tmp_path / ".env"
    f.write_text("# comment\nFOO=bar\nMY_KEY=old\nBAZ=qux\n")
    _update_dotenv_key(f, "MY_KEY", "new-value")
    lines = f.read_text().splitlines()
    assert "MY_KEY=new-value" in lines
    assert "FOO=bar" in lines
    assert "BAZ=qux" in lines
    assert "MY_KEY=old" not in lines


def test_mask_key():
    from imas_ambix.agent.cli import _mask_key

    assert _mask_key("abcdefghijklmnop") == "abcd...mnop"
    assert _mask_key("short") == "sh***"


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


def test_setup_submits_runtime_check_after_network_install(monkeypatch):
    """A network install is not complete until the serving node verifies it."""
    from imas_ambix.agent import slurm as slurm_mod

    scripts = []

    def fake_submit(script):
        scripts.append(script)
        return str(4100 + len(scripts))

    monkeypatch.setattr(slurm_mod, "submit_script", fake_submit)
    runner = CliRunner()
    result = runner.invoke(main, ["agent", "setup", "vllm"])

    assert result.exit_code == 0
    assert len(scripts) == 2
    install_script, runtime_check_script = scripts
    site = SiteConfig()
    assert f"#SBATCH --partition={site.download_partition}" in install_script
    assert f"#SBATCH --partition={site.partition}" in runtime_check_script
    assert f"#SBATCH --reservation={site.reservation}" in runtime_check_script
    assert "#SBATCH --dependency=afterok:4101" in runtime_check_script
    assert str(site.python_path("vllm")) in runtime_check_script
    identity_line = next(
        line
        for line in install_script.splitlines()
        if line.startswith("SETUP_IDENTITY=")
    )
    identity = identity_line.partition("=")[2]
    assert f"EXPECTED_SETUP_IDENTITY={identity}" in runtime_check_script
    assert "not ready until runtime verification job 4102 passes" in result.output


def test_setup_runtime_check_fails_when_interpreter_is_not_visible(tmp_path):
    """Reproduce the cross-node contract failure as a non-zero runtime check."""
    import subprocess

    from imas_ambix.agent.cli import _engine_runtime_check_script

    site = SiteConfig(base_dir=str(tmp_path), engine_env_root=str(tmp_path / "envs"))
    script = _engine_runtime_check_script(
        "vllm", site, dependency_job_id="4101", expected_identity="abc123"
    )
    result = subprocess.run(
        ["bash"], input=script, capture_output=True, text=True, check=False
    )

    assert result.returncode != 0
    assert "runtime node cannot execute" in result.stderr
    assert str(site.python_path("vllm")) in result.stderr


def test_setup_runtime_check_passes_with_durable_interpreter(tmp_path):
    """The postcondition passes only when the runtime path is executable."""
    import subprocess

    from imas_ambix.agent.cli import _engine_runtime_check_script

    site = SiteConfig(base_dir=str(tmp_path), engine_env_root=str(tmp_path / "envs"))
    python = site.python_path("vllm")
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python.chmod(0o755)
    identity = site.env_dir("vllm") / ".ambix-setup-identity"
    identity.write_text("abc123\n", encoding="utf-8")

    script = _engine_runtime_check_script(
        "vllm", site, dependency_job_id="4101", expected_identity="abc123"
    )
    result = subprocess.run(
        ["bash"], input=script, capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr
    assert "Runtime verification complete" in result.stdout


def test_setup_runtime_check_rejects_stale_environment_identity(tmp_path):
    """An executable from a different setup run is not a durable postcondition."""
    import subprocess

    from imas_ambix.agent.cli import _engine_runtime_check_script

    site = SiteConfig(base_dir=str(tmp_path), engine_env_root=str(tmp_path / "envs"))
    python = site.python_path("vllm")
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python.chmod(0o755)
    identity = site.env_dir("vllm") / ".ambix-setup-identity"
    identity.write_text("stale\n", encoding="utf-8")

    script = _engine_runtime_check_script(
        "vllm", site, dependency_job_id="4101", expected_identity="current"
    )
    result = subprocess.run(
        ["bash"], input=script, capture_output=True, text=True, check=False
    )

    assert result.returncode != 0
    assert "does not match the completed network install" in result.stderr


def test_setup_dry_run_checks_capacity_and_records_size():
    runner = CliRunner()
    result = runner.invoke(main, ["agent", "setup", "vllm", "--dry-run"])

    assert result.exit_code == 0
    assert "AVAILABLE_KIB=$(df -Pk" in result.output
    assert "MIN_FREE_KIB=$((32 * 1024 * 1024))" in result.output
    assert "ENV_SIZE_BYTES=$(du -s --block-size=1" in result.output
    assert ".ambix-setup-identity" in result.output


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


def test_siteconfig_launcher_paths():
    """The clive launcher deploys under agents/."""
    site = SiteConfig()
    assert str(site.clive_path).endswith("agents/clive")


def _catalog_item(
    release_id="reported-model",
    *,
    count=8,
    family="H200",
    precision="fp8",
    max_context=131072,
    **extra,
):
    item = {
        "id": release_id,
        "max_model_len": max_context,
        "ambix": {
            "accelerator_family": family,
            "accelerator_count": count,
            "checkpoint_precision": precision,
        },
    }
    item.update(extra)
    return item


@contextmanager
def _serve_catalog(payload, *, status=200, response_headers=None):
    """Serve one anonymous catalog and record every request header."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append((self.path, dict(self.headers)))
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            for name, value in (response_headers or {}).items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format, *args):
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield SiteConfig(global_origin=f"http://{host}:{port}"), requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _write_executable(path, content):
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def test_generate_clive_script_embeds_standalone_global_contract():
    """The default branch embeds one origin and no operator dependencies."""
    from imas_ambix.agent.clive import generate_clive_script

    site = SiteConfig(global_origin="http://catalog.example:19444/")
    script = generate_clive_script(site)

    assert 'MODE="local"' in script
    assert "--mode local|hybrid" in script
    assert 'if [[ "$MODE" == "local" ]]; then' in script
    assert "CLIVE_OPENROUTER" not in script
    assert "GLOBAL_ORIGIN=http://catalog.example:19444" in script
    assert 'origin + "/v1/models"' in script
    assert "urllib.request" in script
    assert 'ANTHROPIC_BASE_URL="$GLOBAL_ORIGIN"' in script
    assert 'OPENAI_BASE_URL="${GLOBAL_ORIGIN}/v1"' in script
    assert "agent clive --resolve-live" not in script
    assert "AMBIX_AGENT_" not in script
    assert "squeue" not in script
    assert "scontrol" not in script
    assert "sbatch" not in script
    for alias in (
        "ANTHROPIC_DEFAULT_OPUS_MODEL",
        "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        "ANTHROPIC_MODEL",
    ):
        assert f'{alias}="$MODEL_ID"' in script


def test_clive_readable_openrouter_key_does_not_reach_proxy(tmp_path):
    """Personal credentials and route overrides cannot affect default launch."""
    import os
    import subprocess

    from imas_ambix.agent.clive import generate_clive_script

    home = tmp_path / "home"
    key_file = home / ".config" / "openrouter" / "key"
    key_file.parent.mkdir(parents=True)
    key_file.write_text("personal-key\n", encoding="utf-8")
    key_file.chmod(0o600)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    proxy_marker = tmp_path / "proxy-called"
    operator_marker = tmp_path / "operator-called"
    trace = tmp_path / "claude-env"
    _write_executable(
        fake_bin / "systemctl",
        f"#!/bin/sh\nprintf '%s\\n' \"$*\" >> {proxy_marker}\nexit 99\n",
    )
    _write_executable(
        fake_bin / "imas-ambix",
        f"#!/bin/sh\ntouch {operator_marker}\nexit 99\n",
    )
    _write_executable(
        fake_bin / "claude",
        "#!/bin/sh\n"
        f'printf \'%s\\n\' "$ANTHROPIC_BASE_URL" "$ANTHROPIC_MODEL" '
        f'"$ANTHROPIC_DEFAULT_OPUS_MODEL_DESCRIPTION" > {trace}\n',
    )

    with _serve_catalog({"data": [_catalog_item(url="http://attacker.invalid")]}) as (
        site,
        requests,
    ):
        launcher = tmp_path / "clive"
        launcher.write_text(generate_clive_script(site), encoding="utf-8")
        launcher.chmod(0o755)
        env = os.environ.copy()
        env.update(
            {
                "HOME": str(home),
                "PATH": f"{fake_bin}:{env['PATH']}",
                "AMBIX_AGENT_URL": "http://override.invalid",
                "AMBIX_AGENT_MODEL": "override-model",
                "AMBIX_AGENT_SELECTOR": "override-selector",
                "AMBIX_AGENT_CLI": str(fake_bin / "imas-ambix"),
                "AMBIX_AGENT_API_KEY": "private-secret",
                "AMBIX_AGENT_KEY_FILE": str(key_file),
            }
        )
        result = subprocess.run(
            [str(launcher)], capture_output=True, text=True, check=False, env=env
        )

        assert result.returncode == 0, result.stderr
        assert trace.read_text(encoding="utf-8").splitlines() == [
            site.global_origin,
            "reported-model",
            "8×H200 · fp8, 128k ctx",
        ]
        assert requests[0][0] == "/v1/models"
        assert "Authorization" not in requests[0][1]
    assert not proxy_marker.exists()
    assert not operator_marker.exists()
    assert "private-secret" not in result.stdout + result.stderr


def test_clive_clean_directory_uses_no_forbidden_consumer_dependency(tmp_path):
    """Sentinels prove the shared path needs only Bash, Python, and a harness."""
    import json
    import shutil
    import subprocess

    from imas_ambix.agent.clive import generate_clive_script

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "bash").symlink_to(shutil.which("bash"))
    (fake_bin / "python3").symlink_to(shutil.which("python3"))
    forbidden_marker = tmp_path / "forbidden-called"
    for command in ("imas-ambix", "squeue", "scontrol", "sbatch"):
        _write_executable(
            fake_bin / command,
            f"#!/bin/bash\necho {command} >> {forbidden_marker}\nexit 99\n",
        )
    trace = tmp_path / "claude-env"
    _write_executable(
        fake_bin / "claude",
        "#!/bin/bash\n"
        f'printf \'%s\\n\' "$ANTHROPIC_BASE_URL" "$ANTHROPIC_MODEL" > {trace}\n'
        f'printf \'%s\\n\' "$@" >> {trace}\n',
    )
    tempting_key = tmp_path / "consumer.env"
    tempting_key.write_text("AMBIX_AGENT_API_KEY=file-only-private-key\n")
    empty_cwd = tmp_path / "empty"
    empty_cwd.mkdir()

    with _serve_catalog({"data": [_catalog_item("glm-5.3")]}) as (site, requests):
        launcher = tmp_path / "clive"
        launcher.write_text(generate_clive_script(site), encoding="utf-8")
        launcher.chmod(0o755)
        traces = []
        for user in ("first-user", "second-user"):
            home = tmp_path / user
            home.mkdir()
            env = {
                "HOME": str(home),
                "USER": user,
                "PATH": str(fake_bin),
                "AMBIX_AGENT_URL": "http://override.invalid",
                "AMBIX_AGENT_MODEL": "wrong-model",
                "AMBIX_AGENT_CLI": str(fake_bin / "imas-ambix"),
                "AMBIX_AGENT_KEY_FILE": str(tempting_key),
                "AMBIX_AGENT_API_KEY": "private-secret",
            }
            result = subprocess.run(
                [str(launcher), "prompt"],
                cwd=empty_cwd,
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )
            assert result.returncode == 0, result.stderr
            traces.append(trace.read_text(encoding="utf-8"))
            assert "private-secret" not in result.stdout + result.stderr

        assert traces[0] == traces[1]
        trace_lines = traces[0].splitlines()
        assert trace_lines[:2] == [site.global_origin, "glm-5.3"]
        arguments = trace_lines[2:]
        assert arguments[0] == "--settings"
        settings = json.loads(arguments[1])
        assert settings["modelPicker"]["replaceBuiltInOptions"] is True
        assert settings["modelPicker"]["options"][0]["model"] == "glm-5.3"
        assert arguments[2] == "--append-system-prompt"
        assert "sonnet alias is the primary local worker" in arguments[3]
        assert arguments[4:] == ["prompt"]
        assert all("Authorization" not in headers for _, headers in requests)
    assert not forbidden_marker.exists()


def test_clive_list_renders_future_releases_and_all_topologies(tmp_path):
    """Listing is dynamic and displays release, topology, and precision."""
    import os
    import subprocess

    from imas_ambix.agent.clive import generate_clive_script

    items = [
        _catalog_item(f"release-{count}", count=count, precision=f"p{count}")
        for count in (2, 4, 6, 8)
    ]
    items[-1]["id"] = "glm-5.3"
    with _serve_catalog({"data": items}) as (site, _requests):
        launcher = tmp_path / "clive"
        launcher.write_text(generate_clive_script(site), encoding="utf-8")
        launcher.chmod(0o755)
        result = subprocess.run(
            [str(launcher), "--list"],
            capture_output=True,
            text=True,
            check=False,
            env=os.environ.copy(),
        )

    assert result.returncode == 0, result.stderr
    assert "glm-5.3@8xh200\tglm-5.3 · 8×H200 · p8" in result.stdout
    for count in (2, 4, 6, 8):
        assert f"{count}×H200" in result.stdout


@pytest.mark.parametrize("selector", ["glm-5.3", "glm-5.3@8xh200"])
def test_clive_codex_receives_selected_model_and_same_origin(tmp_path, selector):
    """Release and topology selectors both preserve native Codex identity."""
    import os
    import subprocess

    from imas_ambix.agent.clive import generate_clive_script

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    trace = tmp_path / "codex-env"
    _write_executable(
        fake_bin / "codex",
        f'#!/bin/sh\nprintf \'%s\\n\' "$OPENAI_BASE_URL" "$*" > {trace}\n',
    )
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    with _serve_catalog(
        {"data": [_catalog_item("glm-5.3", url="http://ignored.invalid")]}
    ) as (site, _requests):
        launcher = tmp_path / "clive"
        launcher.write_text(generate_clive_script(site), encoding="utf-8")
        launcher.chmod(0o755)
        result = subprocess.run(
            [str(launcher), "--codex", "--selector", selector, "prompt"],
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )

    assert result.returncode == 0, result.stderr
    assert trace.read_text(encoding="utf-8").splitlines() == [
        f"{site.global_origin}/v1",
        "--model glm-5.3 prompt",
    ]


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"data": []}, "contains no models"),
        (b"not-json", "malformed"),
        ({"data": "wrong"}, "model data list"),
        (
            {"data": [_catalog_item("same"), _catalog_item("same")]},
            "repeats release id",
        ),
        ({"data": [{"id": "missing-metadata"}]}, "no ambix metadata"),
        ({"data": [_catalog_item(count=3)]}, "accelerator count"),
        ({"data": [_catalog_item(count=True)]}, "accelerator count"),
        ({"data": [_catalog_item(count=2.0)]}, "accelerator count"),
        ({"data": [_catalog_item(family="")]}, "accelerator family"),
        ({"data": [_catalog_item(precision="bad\nvalue")]}, "checkpoint precision"),
        ({"data": [_catalog_item(max_context=False)]}, "maximum context"),
    ],
)
def test_clive_rejects_empty_malformed_duplicate_or_invalid_catalogs(
    tmp_path, payload, message
):
    import os
    import subprocess

    from imas_ambix.agent.clive import generate_clive_script

    with _serve_catalog(payload) as (site, _requests):
        launcher = tmp_path / "clive"
        launcher.write_text(generate_clive_script(site), encoding="utf-8")
        launcher.chmod(0o755)
        result = subprocess.run(
            [str(launcher), "--list"],
            capture_output=True,
            text=True,
            check=False,
            env=os.environ.copy(),
        )

    assert result.returncode != 0
    assert message in result.stderr


def test_clive_requires_noninteractive_selection_for_multiple_items(tmp_path):
    import os
    import subprocess

    from imas_ambix.agent.clive import generate_clive_script

    with _serve_catalog(
        {"data": [_catalog_item("alpha", count=2), _catalog_item("beta", count=6)]}
    ) as (site, _requests):
        launcher = tmp_path / "clive"
        launcher.write_text(generate_clive_script(site), encoding="utf-8")
        launcher.chmod(0o755)
        result = subprocess.run(
            [str(launcher)],
            capture_output=True,
            text=True,
            check=False,
            env=os.environ.copy(),
        )

    assert result.returncode == 2
    assert "multiple models are available; use --selector" in result.stderr
    assert "alpha@2xh200" in result.stderr
    assert "beta@6xh200" in result.stderr


@pytest.mark.parametrize("choice", ["0", "-1", "3"])
def test_clive_interactive_selection_rejects_undisplayed_integers(tmp_path, choice):
    import os
    import pty
    import subprocess

    from imas_ambix.agent.clive import generate_clive_script

    with _serve_catalog(
        {"data": [_catalog_item("alpha", count=2), _catalog_item("beta", count=6)]}
    ) as (site, _requests):
        launcher = tmp_path / "clive"
        launcher.write_text(generate_clive_script(site), encoding="utf-8")
        launcher.chmod(0o755)
        master, slave = pty.openpty()
        process = subprocess.Popen(
            [str(launcher)],
            stdin=slave,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=os.environ.copy(),
        )
        os.close(slave)
        try:
            os.write(master, f"{choice}\n".encode())
            stdout, stderr = process.communicate(timeout=5)
        finally:
            os.close(master)

    assert process.returncode == 2, stdout + stderr
    assert "model selection must be an integer from 1 through 2" in stderr


def test_clive_unreachable_catalog_never_falls_back_to_openrouter(tmp_path):
    import os
    import socket
    import subprocess

    from imas_ambix.agent.clive import generate_clive_script

    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    host, port = probe.getsockname()
    probe.close()
    launcher = tmp_path / "clive"
    launcher.write_text(
        generate_clive_script(
            SiteConfig(global_origin=f"http://{host}:{port}"),
            mode="hybrid",
            openrouter_native_release="configured-release",
        ),
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    marker = tmp_path / "proxy-called"
    _write_executable(fake_bin / "systemctl", f"#!/bin/sh\ntouch {marker}\n")
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    result = subprocess.run(
        [str(launcher), "--mode", "hybrid"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert result.returncode != 0
    assert "global catalog is unreachable" in result.stderr
    assert not marker.exists()


def test_clive_redirect_catalog_fails_without_leaving_global_origin(tmp_path):
    import os
    import subprocess

    from imas_ambix.agent.clive import generate_clive_script

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    harness_marker = tmp_path / "harness-called"
    proxy_marker = tmp_path / "proxy-called"
    _write_executable(fake_bin / "claude", f"#!/bin/sh\ntouch {harness_marker}\n")
    _write_executable(
        fake_bin / "systemctl",
        f"#!/bin/sh\ntouch {proxy_marker}\nexit 99\n",
    )
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"

    with (
        _serve_catalog({"data": [_catalog_item("redirected-release")]}) as (
            redirect_target,
            target_requests,
        ),
        _serve_catalog(
            b"",
            status=302,
            response_headers={"Location": f"{redirect_target.global_origin}/v1/models"},
        ) as (global_site, global_requests),
    ):
        launcher = tmp_path / "clive"
        launcher.write_text(
            generate_clive_script(
                global_site,
                mode="hybrid",
                openrouter_native_release="redirected-release",
            ),
            encoding="utf-8",
        )
        launcher.chmod(0o755)
        result = subprocess.run(
            [str(launcher), "--mode", "hybrid"],
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )

    assert result.returncode == 2
    assert "HTTP Error 302: global catalog redirect refused" in result.stderr
    assert len(global_requests) == 1
    assert global_requests[0][0] == "/v1/models"
    assert "Authorization" not in global_requests[0][1]
    assert target_requests == []
    assert not harness_marker.exists()
    assert not proxy_marker.exists()


def test_clive_operator_command_removes_hidden_consumer_machine_api():
    result = CliRunner().invoke(main, ["agent", "clive", "--help"])

    assert result.exit_code == 0, result.output
    for removed in ("--live-list", "--resolve-live", "--selector", "--url", "--model"):
        assert removed not in result.output
    for retained in ("--deploy", "--print", "--path", "--destination"):
        assert retained in result.output


@pytest.mark.parametrize(
    "mode_arguments",
    [
        ("--mode", "hybrid", "--claude"),
        ("--claude", "--mode", "hybrid"),
    ],
)
def test_clive_openrouter_opt_in_starts_proxy_and_presents_picker(
    tmp_path, mode_arguments
):
    """Explicit hybrid selection starts the proxy and exports every picker slot."""
    import os
    import socket
    import subprocess
    import threading

    from imas_ambix.agent.clive import generate_clive_script

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    proxy_port = listener.getsockname()[1]
    accepted = []

    def accept_probe():
        connection, _ = listener.accept()
        accepted.append(True)
        connection.close()
        listener.close()

    probe = threading.Thread(target=accept_probe, daemon=True)
    probe.start()

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    proxy_marker = tmp_path / "proxy-called"
    trace = tmp_path / "claude-env"
    _write_executable(
        fake_bin / "systemctl",
        "#!/bin/sh\n"
        'if [ "$2" = "is-active" ]; then exit 1; fi\n'
        f"printf '%s\\n' \"$*\" >> {proxy_marker}\n",
    )
    _write_executable(
        fake_bin / "claude",
        "#!/bin/sh\n"
        f'printf \'%s\\n\' "$ANTHROPIC_BASE_URL" "$ANTHROPIC_MODEL" '
        f'"$ANTHROPIC_DEFAULT_HAIKU_MODEL" "$ANTHROPIC_DEFAULT_OPUS_MODEL" '
        f'"$ANTHROPIC_DEFAULT_SONNET_MODEL" "$ANTHROPIC_CUSTOM_MODEL_OPTION" '
        f'"$ANTHROPIC_DEFAULT_FABLE_MODEL" > {trace}\n',
    )

    env = os.environ.copy()
    env.update(
        {
            "HOME": str(tmp_path / "home"),
            "PATH": f"{fake_bin}:{env['PATH']}",
        }
    )
    with _serve_catalog(
        {
            "data": [
                _catalog_item("served-local-model"),
                _catalog_item("served-secondary-model", count=4),
            ]
        }
    ) as (site, _requests):
        script = generate_clive_script(
            site,
            mode="hybrid",
            openrouter_native_release="served-local-model",
        ).replace('LITELLM_PORT="18399"', f'LITELLM_PORT="{proxy_port}"')
        launcher = tmp_path / "clive"
        launcher.write_text(script, encoding="utf-8")
        launcher.chmod(0o755)
        command = [
            str(launcher),
            *mode_arguments,
            "--model",
            "served-local-model",
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
    probe.join(timeout=2)

    assert result.returncode == 0, result.stderr
    assert accepted == [True]
    assert "start imas-ambix-llm.service" in proxy_marker.read_text(encoding="utf-8")
    assert "or-opus-4.8, or-gpt-5.5, or-glm-5.2" in result.stderr
    assert trace.read_text(encoding="utf-8").splitlines() == [
        f"http://127.0.0.1:{proxy_port}",
        "served-local-model",
        "served-secondary-model",
        "or-opus-4.8",
        "served-local-model",
        "or-gpt-5.5",
        "or-glm-5.2",
    ]


def test_clive_openrouter_rejects_unconfigured_dynamic_release_before_proxy(
    tmp_path,
):
    import os
    import subprocess

    from imas_ambix.agent.clive import generate_clive_script
    from imas_ambix.agent.litellm_config import generate_litellm_config

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    proxy_marker = tmp_path / "proxy-called"
    _write_executable(
        fake_bin / "systemctl",
        f"#!/bin/sh\nprintf '%s\\n' \"$*\" >> {proxy_marker}\nexit 99\n",
    )
    _write_executable(fake_bin / "claude", "#!/bin/sh\nexit 0\n")
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"

    with _serve_catalog({"data": [_catalog_item("dynamic-release")]}) as (
        site,
        _requests,
    ):
        config = generate_litellm_config(site, "profile-alias")
        launcher = tmp_path / "clive"
        launcher.write_text(
            generate_clive_script(
                site,
                mode="hybrid",
                openrouter_native_release="profile-alias",
            ),
            encoding="utf-8",
        )
        launcher.chmod(0o755)
        result = subprocess.run(
            [str(launcher), "--mode", "hybrid"],
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )

    assert "model_name: profile-alias" in config
    assert "model_name: dynamic-release" not in config
    assert result.returncode == 2
    assert (
        "selected native release 'dynamic-release' is not configured in the "
        "OpenRouter proxy (configured: 'profile-alias')"
    ) in result.stderr
    assert not proxy_marker.exists()


def test_clive_deploy_writes_one_launcher(monkeypatch):
    """One deploy invocation emits exactly one generated launcher artifact."""
    from imas_ambix.agent import cli as cli_mod

    deployed = []
    monkeypatch.setattr(
        cli_mod,
        "_deploy_launcher",
        lambda name, path, content: deployed.append((name, path, content)),
    )

    result = CliRunner().invoke(main, ["agent", "clive", "--deploy"])

    assert result.exit_code == 0, result.output
    assert len(deployed) == 1
    name, path, content = deployed[0]
    assert name == "clive"
    assert path == SiteConfig().clive_path
    assert content.startswith("#!/usr/bin/env bash\n")
