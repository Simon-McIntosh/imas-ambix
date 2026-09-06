"""Profile-to-launch contracts for the DSpark four-card DeepSeek-V4-Flash serve."""

from __future__ import annotations

import json
import shlex

from imas_ambix.agent.profile import SiteConfig, load_profile
from imas_ambix.agent.slurm import generate_serve_script


def _speculative_config(script: str) -> dict[str, object]:
    line = next(
        line
        for line in script.splitlines()
        if "vllm.entrypoints.openai.api_server" in line
    )
    tokens = shlex.split(line)
    value = tokens[tokens.index("--speculative-config") + 1]
    return json.loads(value)


def test_four_card_profile_declares_dspark_engine_flags() -> None:
    profile = load_profile("deepseek-v4-flash").for_gpus(4)

    assert profile.engine.type == "vllm"
    assert profile.engine.block_size == 256
    assert profile.engine.tokenizer_mode == "deepseek_v4"
    assert profile.engine.moe_backend == "auto"
    assert profile.engine.enable_expert_parallel is True
    assert profile.engine.speculative_method == "dspark"
    assert profile.engine.speculative_num_tokens == 5
    assert profile.engine.speculative_draft_sample_method == "greedy"


def test_four_card_serve_script_emits_dspark_recipe_flags() -> None:
    profile = load_profile("deepseek-v4-flash").for_gpus(4)

    script = generate_serve_script(profile, SiteConfig(), port=18800)

    assert "--block-size 256" in script
    assert "--enable-expert-parallel" in script
    assert "--tokenizer-mode deepseek_v4" in script
    assert "--moe-backend auto" in script
    assert "--speculative-config" in script
    assert _speculative_config(script) == {
        "method": "dspark",
        "num_speculative_tokens": 5,
        "draft_sample_method": "greedy",
    }
    # DSpark's module is fused into the checkpoint -- no separate draft model.
    assert "model" not in _speculative_config(script)


def test_four_card_gpu_variant_selection_matches_gpus_flag() -> None:
    profile = load_profile("deepseek-v4-flash").for_gpus(4)

    assert profile.slurm.gpus == 4
    assert profile.engine.tensor_parallel == 4


def test_four_card_serve_script_carries_full_native_context_window() -> None:
    """The four-card serve sets the checkpoint's native 1M context at a memory
    fraction that fits its measured KV pool. The single compressed KV head
    does not shard across tensor parallelism, so the four-card pool matches
    the two-card one.
    """
    profile = load_profile("deepseek-v4-flash").for_gpus(4)

    script = generate_serve_script(profile, SiteConfig(), port=18800)

    assert "--max-model-len 1048576" in script
    assert "--gpu-memory-utilization 0.92" in script


def test_speculative_config_dotted_form_survives_when_no_draft_sample_method() -> None:
    """The GLM-5.2 MTP path (no draft model, no draft-sample method) keeps the
    dotted ``--speculative-config.*`` flags rather than the compact JSON form.
    """
    profile = load_profile("glm-5-2")

    script = generate_serve_script(profile, SiteConfig(), port=18800)

    assert "--speculative-config.method mtp" in script
    assert "--speculative-config.num_speculative_tokens 5" in script
    assert "--speculative-config '" not in script


def test_no_profile_declares_the_router_port():
    """No serving profile may claim the port the router runs on.

    A profile declaring the router's port makes a bare ``agent serve <slug>``
    refuse at submit with a port-holder error naming ``ambix-router``, which
    reads as a router fault rather than as a profile defect. The two-card
    DeepSeek variant declared 18802 until 2026-09-06.
    """
    from imas_ambix.agent.profile import list_profiles, load_profile

    router_port = 18802
    for slug in list_profiles():
        profile = load_profile(slug)
        assert profile.slurm.port != router_port, (
            f"profile {slug!r} declares the router port {router_port}"
        )
