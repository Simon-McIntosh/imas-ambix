"""Identity and download contracts for the candidate backup profiles.

Two profiles exist so that trying a candidate later costs a launch rather than
a multi-terabyte transfer. They carry the release, the repository, the commit
the weights are fetched at, and the directory the checkpoint lands in. Their
engine tuning and deployment sizing are deferred, because a deployment is sized
against hardware that has not been chosen yet and a value written now would be
a guess that later reads as a decision.

Two catalogue invariants still apply to this node however little is being
served from it, and are declared in ``[slurm]``: an endpoint no other profile
holds, and a core request that leaves room inside the shared reservation. The
agent suite asserts both across the catalogue; these tests assert them where a
reader of the candidate contract will look, and name the profile that broke one.

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
from imas_ambix.agent.profile import SiteConfig, list_profiles, load_profile
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

# The shared Group A reservation. A device-resident engine computes nothing on
# the host, so a profile asking for the whole ceiling leaves a neighbour
# pending on cores; CPU-offloading engines are the mechanism that legitimately
# needs them and are not candidates here.
RESERVATION_CORES = 30

# Engine keys that describe how a model is served rather than what it is. A
# candidate declares none of them until a deployment is chosen.
SERVING_TOPOLOGY_KEYS = {
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
def test_candidate_defers_its_serving_topology(slug: str) -> None:
    """A candidate declares no card count, memory split or engine tuning.

    What it does declare in ``[slurm]`` are the two catalogue invariants that
    apply even to a profile nothing is served from yet: an endpoint no other
    profile holds, and a core request inside the shared reservation.
    """
    document = tomllib.loads(_profile_text(slug))

    declared = set(document.get("engine", {}))
    assert not declared & SERVING_TOPOLOGY_KEYS, (
        f"{slug} declares engine topology: {sorted(declared & SERVING_TOPOLOGY_KEYS)}"
    )
    slurm = document.get("slurm", {})
    assert set(slurm) <= {"port", "cpus"}, (
        f"{slug} declares deployment sizing: {sorted(set(slurm) - {'port', 'cpus'})}"
    )


@pytest.mark.parametrize("slug", sorted(CANDIDATES))
def test_candidate_holds_the_catalogue_invariants(slug: str) -> None:
    """One endpoint per profile, and cores left for the neighbours.

    The catalogue test in the agent suite asserts both across every profile;
    asserting them here as well makes the candidate contract legible where a
    reader looks for it, and localises a failure to the profile that caused it.
    """
    profile = load_profile(slug)

    assert profile.slurm.port is not None
    assert profile.slurm.cpus < RESERVATION_CORES, (
        f"{slug} claims {profile.slurm.cpus} of {RESERVATION_CORES} reserved cores"
    )


def test_candidate_ports_are_distinct_across_the_catalogue() -> None:
    """No two profiles may resolve to one endpoint."""
    ports = {slug: load_profile(slug).slurm.port for slug in list_profiles()}

    assert all(port is not None for port in ports.values()), ports
    assert len(set(ports.values())) == len(ports), ports


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
