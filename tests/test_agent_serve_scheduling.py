"""Launch-line contracts for the engine's waiting-queue policy and bound.

The engine drains its waiting queue under a schedule policy and, optionally,
refuses arrivals past a queue bound. Both are profile fields, and the contract
under test is the generated ``sglang.launch_server`` line rather than the
profile attribute alone: a field set on a profile that never reaches the engine
is the failure this guards.
"""

from __future__ import annotations

from imas_ambix.agent.profile import EngineConfig, SiteConfig, load_profile
from imas_ambix.agent.slurm import generate_serve_script


def _launch_line(script: str) -> str:
    return next(
        line for line in script.splitlines() if "sglang.launch_server" in line
    )


def test_dsv41_serve_script_emits_schedule_policy_and_queue_bound() -> None:
    profile = load_profile("deepseek-v4-1-flash")

    launch = _launch_line(generate_serve_script(profile, SiteConfig(), port=18810))

    assert "--schedule-policy lpm" in launch
    assert "--max-queued-requests 64" in launch
    # The running bound is untouched by this change and still bounds the batch.
    assert "--max-running-requests 36" in launch


def test_engine_config_defaults_both_fields_to_unset() -> None:
    engine = EngineConfig(type="sglang")

    assert engine.schedule_policy is None
    assert engine.max_queued_requests is None


def test_profile_leaving_both_unset_emits_neither_flag() -> None:
    profile = load_profile("minimax-m2-7")
    assert profile.engine.schedule_policy is None
    assert profile.engine.max_queued_requests is None

    launch = _launch_line(generate_serve_script(profile, SiteConfig(), port=18800))

    assert "--schedule-policy" not in launch
    assert "--max-queued-requests" not in launch
