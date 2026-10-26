"""Identity and download contracts for the candidate backup profiles.

Two profiles exist so that trying a candidate later costs a launch rather than
a multi-terabyte transfer. They carry identity and weights only: the release,
the repository, the commit the weights are fetched at, and the directory the
checkpoint lands in. No serving topology is declared, because a deployment is
sized against hardware that has not been chosen yet, and a topology written now
would be a guess that later reads as a decision.

The download script is the surface these tests measure, because it is where the
pin either reaches the transfer or does not: a profile holding a revision that
its batch script never passes leaves the fetch free to follow the default
branch, which is exactly the failure a pin exists to remove.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from imas_ambix.agent import profile as profile_module
from imas_ambix.agent.profile import SiteConfig, load_profile
from imas_ambix.agent.slurm import generate_download_script

# The revision and repository each candidate must be fetched at. Written out
# rather than read from the profile, so a profile edited to a different commit
# fails here instead of silently moving the pin it is tested against.
CANDIDATES = {
    "kimi-k3": (
        "moonshotai/Kimi-K3",
        "f831ab66814297da540d832a5235f8e904f29d06",
    ),
    "mimo-v2-6-pro": (
        "XiaomiMiMo/MiMo-V2.6-Pro-MOPD",
        "6a7f202e8dbd0e57b3fe1da03917d08d75abc4b8",
    ),
}

# Keys that describe how a model is served rather than what it is. A candidate
# profile declares no card count, no memory split, no port and no engine
# tuning until a deployment is chosen.
SERVING_TOPOLOGY_KEYS = {
    "slurm",
    "gpus",
    "cpus",
    "memory",
    "port",
    "tensor_parallel",
    "data_parallel",
    "ep_size",
    "mem_fraction_static",
    "max_total_tokens",
    "context_length",
    "chunked_prefill_size",
}


def _profile_text(slug: str) -> str:
    directory = Path(profile_module.__file__).parent / "profiles"
    return (directory / f"{slug}.toml").read_text(encoding="utf-8")


@pytest.mark.parametrize("slug", sorted(CANDIDATES))
def test_candidate_pins_its_release_and_weights_directory(slug: str) -> None:
    """Each candidate names the exact commit and the directory it lands in."""
    repo, revision = CANDIDATES[slug]
    profile = load_profile(slug)

    assert profile.model.hf_repo == repo
    assert profile.model.revision == revision
    assert len(revision) == 40
    assert all(character in "0123456789abcdef" for character in revision)
    assert profile.model.weights_slug
    assert profile.weights_directory_slug == profile.model.weights_slug
    assert profile.model.served_name


@pytest.mark.parametrize("slug", sorted(CANDIDATES))
def test_candidate_declares_no_serving_topology(slug: str) -> None:
    """A candidate carries identity and weights only."""
    document = tomllib.loads(_profile_text(slug))

    assert "slurm" not in document
    declared = set(document.get("engine", {}))
    assert not declared & SERVING_TOPOLOGY_KEYS, (
        f"{slug} declares serving topology: {sorted(declared & SERVING_TOPOLOGY_KEYS)}"
    )


@pytest.mark.parametrize("slug", sorted(CANDIDATES))
def test_download_script_names_the_pinned_revision(slug: str) -> None:
    """The pin reaches the transfer, and the transfer lands in its own dir.

    A revision the batch script never passes is not a pin: ``hf download``
    then follows whatever the default branch holds on the day the job runs.
    """
    repo, revision = CANDIDATES[slug]
    profile = load_profile(slug)
    site = SiteConfig()

    script = generate_download_script(profile, site)

    assert f"REVISION={revision}" in script
    assert '--revision "$REVISION"' in script
    assert repo in script
    assert f"MODEL_DIR={site.model_dir(profile)}" in script
    assert '--local-dir "$MODEL_DIR"' in script
    assert f"{profile.model.weights_slug}/model" in script


def test_mimo_candidate_is_the_agentic_variant() -> None:
    """The MiMo choice is the checkpoint the model card names for agents.

    The family publishes -RL and -MOPD. The -MOPD card states that the stage
    mitigates the tool-call repetition that "emerged as one of the most
    noticeable issues in agentic settings", so that variant is the one the
    profile tracks and the weights directory names it.
    """
    profile = load_profile("mimo-v2-6-pro")

    assert profile.model.hf_repo.endswith("-MOPD")
    assert "mopd" in profile.weights_directory_slug
