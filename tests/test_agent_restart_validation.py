"""Restart validation: the weights preflight and the serving dialect probes.

A restart replaces a serving engine only after the replacement answers a tool
call in both dialects, because a completion is not evidence that a caller can
use the engine. Before anything is paused, the profile's weights directory is
checked against its own index, so an incomplete checkpoint refuses while the
lane is still serving.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import click
import pytest

from imas_ambix.agent import bench
from imas_ambix.agent.cli import _serving_validator, weights_preflight
from imas_ambix.agent.profile import SiteConfig, load_profile
from imas_ambix.agent.reload import run_overlap, run_reload

INDEX_NAME = "model.safetensors.index.json"


# ---------------------------------------------------------------------------
# Recorded engine responses
# ---------------------------------------------------------------------------


def _tool_call_payload() -> dict:
    return {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "Cadarache, France"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 40, "completion_tokens": 20},
    }


def _text_only_payload() -> dict:
    """The negative-control fixture: the tool call arrives as text in content."""
    return {
        "choices": [
            {
                "message": {
                    "content": "I'll check the weather in Cadarache, France.",
                    "tool_calls": [],
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 40, "completion_tokens": 12},
    }


def _tool_use_payload() -> dict:
    return {
        "content": [
            {"type": "text", "text": "I'll look that up."},
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "get_weather",
                "input": {"location": "Cadarache, France"},
            },
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 40, "output_tokens": 20},
    }


class Engine:
    """One engine's recorded answers, keyed by base URL."""

    def __init__(self, tool_payload: dict) -> None:
        self.tool_payload = tool_payload
        self.openai_calls: list[str] = []
        self.messages_calls: list[str] = []


def _install_engines(
    monkeypatch: pytest.MonkeyPatch, engines: dict[str, Engine]
) -> None:
    """Serve each engine's recorded responses through bench's transports."""

    def chat(base_url, model, messages, max_tokens=512, tools=None, **kwargs):
        engines[base_url].openai_calls.append(base_url)
        from imas_ambix.agent.bench import BenchResult

        result = BenchResult(model=model)
        payload = engines[base_url].tool_payload
        result.finish_reason = payload["choices"][0]["finish_reason"]
        return result, payload

    def messages(base_url, model, prompt, max_tokens=512, tools=None, **kwargs):
        engines[base_url].messages_calls.append(base_url)
        from imas_ambix.agent.bench import BenchResult

        result = BenchResult(model=model)
        payload = _tool_use_payload()
        result.finish_reason = payload["stop_reason"]
        return result, payload

    monkeypatch.setattr(bench, "_chat", chat)
    monkeypatch.setattr(bench, "_messages", messages)


# ---------------------------------------------------------------------------
# Weights preflight
# ---------------------------------------------------------------------------


def _profile_and_site(tmp_path):
    profile = load_profile("deepseek-v4-flash")
    site = SiteConfig(base_dir=str(tmp_path))
    site.model_dir(profile).mkdir(parents=True)
    return profile, site


def _write_index(model_dir, shards: list[str]) -> None:
    (model_dir / INDEX_NAME).write_text(
        json.dumps(
            {"metadata": {}, "weight_map": {f"t{i}": s for i, s in enumerate(shards)}}
        ),
    )


def test_weights_preflight_names_a_missing_index(tmp_path) -> None:
    profile, site = _profile_and_site(tmp_path)

    missing = weights_preflight(profile, site)

    assert missing == [str(site.model_dir(profile) / INDEX_NAME)]


def test_weights_preflight_names_a_missing_shard(tmp_path) -> None:
    profile, site = _profile_and_site(tmp_path)
    model_dir = site.model_dir(profile)
    _write_index(
        model_dir,
        ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"],
    )
    (model_dir / "model-00001-of-00002.safetensors").write_bytes(b"x")

    missing = weights_preflight(profile, site)

    assert missing == [str(model_dir / "model-00002-of-00002.safetensors")]


def test_weights_preflight_passes_a_complete_checkpoint(tmp_path) -> None:
    """Positive control: the check sees a complete directory before an absence."""
    profile, site = _profile_and_site(tmp_path)
    model_dir = site.model_dir(profile)
    _write_index(model_dir, ["model-00001-of-00001.safetensors"])
    (model_dir / "model-00001-of-00001.safetensors").write_bytes(b"x")

    assert weights_preflight(profile, site) == []


def test_restart_refuses_a_missing_shard_before_pausing(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An incomplete checkpoint stops the restart while the lane still serves."""
    profile, site = _profile_and_site(tmp_path)
    model_dir = site.model_dir(profile)
    _write_index(
        model_dir,
        ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"],
    )
    (model_dir / "model-00001-of-00002.safetensors").write_bytes(b"x")

    monkeypatch.setattr(cli_module(), "SiteConfig", _site_class(site))
    monkeypatch.setattr(
        cli_module(), "_require_engine_environment", lambda *a, **k: None
    )
    paused: list[str] = []
    monkeypatch.setattr(
        cli_module(), "_pause_gate", lambda *a, **k: paused.append("pause")
    )
    inspected: list[str] = []
    monkeypatch.setattr(
        cli_module(), "_profile_jobs", lambda *a, **k: inspected.append("jobs") or []
    )

    with pytest.raises(click.ClickException) as refusal:
        cli_module().restart.callback(
            slug=profile.slug,
            dry_run=False,
            port=None,
            api_key=None,
            auth=False,
            gpus=None,
            dspark_block_size=None,
            cpus=None,
            no_speculative=False,
            time_limit=None,
            reason="validation test",
        )

    assert "model-00002-of-00002.safetensors" in str(refusal.value)
    assert paused == []
    assert inspected == []


def test_restart_dry_run_prints_the_preflight_verdict(
    tmp_path, monkeypatch, capsys
) -> None:
    profile, site = _profile_and_site(tmp_path)
    model_dir = site.model_dir(profile)
    _write_index(model_dir, ["model-00001-of-00001.safetensors"])
    (model_dir / "model-00001-of-00001.safetensors").write_bytes(b"x")

    monkeypatch.setattr(cli_module(), "SiteConfig", _site_class(site))

    cli_module().restart.callback(
        slug=profile.slug,
        dry_run=True,
        port=None,
        api_key=None,
        auth=False,
        gpus=None,
        cpus=None,
        no_speculative=False,
        dspark_block_size=None,
        time_limit=None,
        reason="validation test",
    )

    assert "Weights preflight: ok" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Dialect probes through the restart validation
# ---------------------------------------------------------------------------


def cli_module():
    from imas_ambix.agent import cli

    return cli


def _site_class(site: SiteConfig):
    return type("FixedSite", (SiteConfig,), {"from_env": classmethod(lambda cls: site)})


def _validator(engines: dict[str, Engine], origin_by_job: dict[str, str], monkeypatch):
    """The real restart validator, with completion stubbed and probes recorded."""
    monkeypatch.setattr(cli_module(), "_completion_verdict", lambda *a, **k: True)
    profile = load_profile("deepseek-v4-flash")
    return _serving_validator(
        profile, None, None, lambda job: origin_by_job.get(job, "")
    )


class PausedWorld:
    def __init__(self) -> None:
        self.now = 0.0
        self.events: list[tuple] = []
        self.gate = {"paused": False}
        self.active = {"old"}

    def jobs(self):
        return [{"jobid": item, "name": "sample"} for item in self.active]

    def cancel(self, job):
        self.events.append(("cancel", job))
        self.active.discard(job)

    def submit(self, script):
        job = f"new-{len([e for e in self.events if e[0] == 'submit']) + 1}"
        self.events.append(("submit", job))
        self.active.add(job)
        return job

    def lane(self):
        return {
            "state": "available",
            "observed_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "running": 0,
            "router_generation_gate": {"waiting": 0},
        }

    def pause(self, reason):
        self.events.append(("pause",))
        self.gate["paused"] = True

    def resume(self):
        self.gate["paused"] = False

    def sleep(self, seconds):
        self.now += seconds

    def wall_clock(self):
        return datetime(2026, 10, 23, tzinfo=UTC) + timedelta(seconds=self.now)


def test_pause_restart_rolls_back_when_the_tool_call_arrives_as_text(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engines = {
        "http://node/new-1": Engine(_text_only_payload()),
        "http://node/new-2": Engine(_tool_call_payload()),
    }
    _install_engines(monkeypatch, engines)
    validate = _validator(
        engines,
        {"new-1": "http://node/new-1", "new-2": "http://node/new-2"},
        monkeypatch,
    )
    world = PausedWorld()

    result = run_reload(
        profile="sample",
        script="new script",
        rollback_script="saved batch",
        old_job_id="old",
        dry_run=False,
        read_jobs=world.jobs,
        cancel=world.cancel,
        submit=world.submit,
        read_lane=world.lane,
        read_gate=lambda: world.gate,
        pause=world.pause,
        resume=world.resume,
        job_running=lambda job: job in world.active,
        ready=lambda job: True,
        validate=validate,
        clock=lambda: world.now,
        sleep=world.sleep,
        wall_clock=world.wall_clock,
    )

    assert result.outcome == "rolled-back"
    assert ("cancel", "new-1") in world.events
    # The rollback engine answered both dialects; the replacement did not.
    assert engines["http://node/new-1"].openai_calls
    assert engines["http://node/new-2"].messages_calls


def test_overlap_restart_cancels_the_replacement_when_the_probe_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engines = {"http://node/new-1": Engine(_text_only_payload())}
    _install_engines(monkeypatch, engines)
    validate = _validator(engines, {"new-1": "http://node/new-1"}, monkeypatch)
    cancelled: list[str] = []

    with pytest.raises(RuntimeError, match="validation failed"):
        run_overlap(
            script="held script",
            old_job_id="old",
            submit=lambda script: "new-1",
            cancel=lambda job: cancelled.append(job),
            job_state=lambda job: "RUNNING",
            ready=lambda job: True,
            validate=validate,
            release=lambda job: None,
            confirm=lambda job: True,
            read_old_running=lambda timeout: 0,
            publish=lambda: None,
        )

    assert cancelled == ["new-1"]
    assert "old" not in cancelled
