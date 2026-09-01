"""Focused profile-to-launch contracts for GLM-5.3 deployments."""

from __future__ import annotations

import json
import shlex

from imas_ambix.agent.profile import SiteConfig, load_profile
from imas_ambix.agent.slurm import generate_serve_script


def _catalog_metadata(script: str) -> dict[str, object]:
    prefix = "export AMBIX_VLLM_CATALOG_METADATA="
    line = next(
        line.strip() for line in script.splitlines() if line.strip().startswith(prefix)
    )
    assignment = shlex.split(line)[1]
    return json.loads(assignment.split("=", 1)[1])


def test_four_card_int4_profile_and_serve_script() -> None:
    profile = load_profile("glm-5-3").for_gpus(4)

    assert profile.slurm.gpus == 4
    assert profile.slurm.cpus == 12
    assert profile.model.checkpoint_precision == "int4"
    assert profile.model.weights_slug == "glm-5-3-int4"
    assert profile.model.served_name == "glm-5.3"
    assert profile.engine.tensor_parallel == 4
    assert profile.engine.kv_cache_dtype == "bfloat16"
    assert profile.engine.max_total_tokens == 262144
    assert profile.engine.parsers.tool_call == "glm47"
    assert profile.engine.parsers.reasoning == "glm45"

    script = generate_serve_script(profile, SiteConfig(), port=18801)

    assert "#SBATCH --gres=gpu:4" in script
    assert "#SBATCH --cpus-per-task=12" in script
    assert "agents/glm-5-3-int4/model" in script
    assert "--served-model-name glm-5.3" in script
    assert "--tensor-parallel-size 4" in script
    assert "--kv-cache-dtype bfloat16" in script
    assert "--max-model-len 262144" in script
    assert "--tool-call-parser glm47" in script
    assert "--reasoning-parser glm45" in script
    assert "--speculative-config" not in script
    assert "#SBATCH --comment=ambix-serve;port=18801" in script
    assert _catalog_metadata(script) == {
        "glm-5.3": {
            "accelerator_family": "H200",
            "accelerator_count": 4,
            "checkpoint_precision": "int4",
        }
    }


def test_eight_card_fp8_profile_and_serve_script() -> None:
    profile = load_profile("glm-5-3").for_gpus(8)

    assert profile.slurm.gpus == 8
    assert profile.slurm.cpus == 16
    assert profile.model.checkpoint_precision == "fp8"
    assert profile.model.weights_slug == "glm-5-3"
    assert profile.model.served_name == "glm-5.3"
    assert profile.engine.tensor_parallel == 8
    assert profile.engine.max_total_tokens == 204800

    script = generate_serve_script(profile, SiteConfig(), port=18801)

    assert "#SBATCH --gres=gpu:8" in script
    assert "#SBATCH --cpus-per-task=16" in script
    assert "agents/glm-5-3/model" in script
    assert "--served-model-name glm-5.3" in script
    assert "--tensor-parallel-size 8" in script
    assert "--max-model-len 204800" in script
    assert "#SBATCH --comment=ambix-serve;port=18801" in script
    assert _catalog_metadata(script) == {
        "glm-5.3": {
            "accelerator_family": "H200",
            "accelerator_count": 8,
            "checkpoint_precision": "fp8",
        }
    }
