"""Deterministic checks for the paused serving reload cycle."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from imas_ambix.agent.cli import _pause_gate
from imas_ambix.agent.reload import run_reload


class World:
    def __init__(self, *, gate=None, readings=None, ready=True, valid=True):
        self.now = 0.0
        self.drain_seconds = 0.0
        self.events = []
        self.gate = gate or {"paused": False}
        stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.readings = readings or [
            {
                "state": "available",
                "observed_at": stamp,
                "running": 0,
                "router_generation_gate": {"waiting": 3},
            }
        ]
        self.ready_result = ready
        self.valid_result = valid
        self.completion_tokens = {"new-1": 250, "new-2": 250}
        self.running_result = True
        self.submits = 0
        self.active = {"old"}

    def jobs(self):
        self.events.append("jobs")
        return [{"jobid": item, "name": "sample"} for item in self.active]

    def cancel(self, job):
        self.events.append(("cancel", job))
        self.active.discard(job)

    def submit(self, script):
        self.submits += 1
        job = f"new-{self.submits}"
        self.events.append(("submit", script, job, self.now))
        self.active.add(job)
        return job

    def lane(self):
        self.events.append("lane")
        reading = dict(self.readings[0])
        if self.now < self.drain_seconds:
            reading["running"] = 1
        return reading

    def pause(self, reason):
        self.events.append(("pause", reason))
        self.gate["paused"] = True

    def resume(self):
        self.events.append("resume")
        self.gate["paused"] = False

    def sleep(self, seconds):
        self.now += seconds

    def wall_clock(self):
        return datetime(2026, 10, 20, tzinfo=UTC) + timedelta(seconds=self.now)

    def run(self, **kwargs):
        return run_reload(
            profile="sample",
            script="new script",
            rollback_script="saved batch",
            old_job_id="old",
            dry_run=False,
            read_jobs=self.jobs,
            cancel=self.cancel,
            submit=self.submit,
            read_lane=self.lane,
            read_gate=lambda: self.gate,
            pause=self.pause,
            resume=self.resume,
            job_running=lambda job: (
                self.running_result(job)
                if callable(self.running_result)
                else self.running_result
            ),
            ready=lambda job: (
                self.ready_result(job)
                if callable(self.ready_result)
                else self.ready_result
            ),
            validate=lambda job: (
                (
                    self.valid_result(job)
                    if callable(self.valid_result)
                    else self.valid_result
                )
                and self.completion_tokens.get(job, 250) >= 200
            ),
            clock=lambda: self.now,
            wall_clock=self.wall_clock,
            sleep=self.sleep,
            **kwargs,
        )


def test_restart_pause_with_stale_cut_withdraws_cut(tmp_path):
    gate_path = tmp_path / "router-gate.json"
    gate_path.write_text(
        json.dumps(
            {
                "paused": True,
                "reason": "earlier pause",
                "cut": True,
                "cut_form": "error",
            }
        ),
        encoding="utf-8",
    )

    _pause_gate(gate_path, "agent restart")

    saved = json.loads(gate_path.read_text(encoding="utf-8"))
    assert saved["paused"] is True
    assert saved["reason"] == "agent restart"
    assert "cut" not in saved
    assert "cut_form" not in saved


def test_success_order_and_cycle_record(tmp_path):
    world = World()
    record = tmp_path / "lane-cycles.jsonl"
    result = world.run(record_path=record, settings_changed={"gpus": 4})
    assert result.outcome == "reloaded"
    assert world.events.index(("pause", "agent restart")) < world.events.index("lane")
    assert world.events.index(("cancel", "old")) < next(
        i
        for i, event in enumerate(world.events)
        if isinstance(event, tuple) and event[0] == "submit"
    )
    assert world.events[-1] == "resume"
    saved = json.loads(record.read_text())
    assert saved["outcome"] == "reloaded"
    assert saved["settings_changed"] == {"gpus": 4}
    assert saved["pause_start"] == "2026-10-20T00:00:00Z"
    assert saved["pause_end"].endswith("Z")


def test_readiness_miss_resubmits_saved_script_and_resumes():
    world = World(ready=lambda job: job != "new-1")
    result = world.run()
    assert result.outcome == "rolled-back"
    assert any(
        event[:2] == ("submit", "saved batch")
        for event in world.events
        if isinstance(event, tuple)
    )
    assert world.events[-1] == "resume"


def test_short_completion_rolls_back_and_resumes():
    world = World()
    world.completion_tokens["new-1"] = 199
    result = world.run()
    assert result.outcome == "rolled-back"
    assert any(
        event[:2] == ("submit", "saved batch")
        for event in world.events
        if isinstance(event, tuple)
    )
    assert world.events[-1] == "resume"


def test_submit_failure_after_old_job_stops_submits_saved_script():
    world = World()
    submit = world.submit

    def fail_replacement(script):
        if script == "new script":
            raise RuntimeError("replacement submission failed")
        return submit(script)

    world.submit = fail_replacement
    result = world.run()
    assert result.outcome == "rolled-back"
    assert any(
        event[0:2] == ("submit", "saved batch")
        for event in world.events
        if isinstance(event, tuple)
    )
    assert world.events[-1] == "resume"


@pytest.mark.parametrize(
    "reading",
    [
        {"state": "unavailable", "running": 0},
        {"state": "available", "observed_at": "2000-01-01T00:00:00Z", "running": 0},
    ],
)
def test_unavailable_or_stale_drain_resumes_without_cancelling(reading):
    world = World(readings=[reading])
    # Keep yielding an unavailable reading until the ten-minute bound expires.
    result = world.run()
    assert result.outcome == "aborted-before-cancel"
    assert not any(
        isinstance(event, tuple) and event[0] == "cancel" for event in world.events
    )
    assert world.events[-1] == "resume"


@pytest.mark.parametrize(
    "failure", ["lane", "cancel", "submit", "job", "ready", "validate"]
)
def test_exception_after_pause_resumes(failure):
    world = World()
    if failure == "lane":
        world.lane = lambda: (_ for _ in ()).throw(RuntimeError("lane"))
    elif failure == "cancel":
        world.cancel = lambda _job: (_ for _ in ()).throw(RuntimeError("cancel"))
    elif failure == "submit":
        world.submit = lambda _script: (_ for _ in ()).throw(RuntimeError("submit"))
    elif failure == "job":
        world.running_result = lambda _job: (_ for _ in ()).throw(RuntimeError("job"))
    elif failure == "ready":
        world.ready_result = lambda _job: (_ for _ in ()).throw(RuntimeError("ready"))
    elif failure == "validate":
        world.valid_result = lambda _job: (_ for _ in ()).throw(
            RuntimeError("validate")
        )
    with pytest.raises(RuntimeError):
        world.run()
    assert world.events[-1] == "resume"


def test_operator_pause_is_preserved(tmp_path):
    world = World(gate={"paused": True})
    record = tmp_path / "operator-pause-record.jsonl"
    world.run(record_path=record)
    saved = json.loads(record.read_text().splitlines()[-1])
    assert not any(
        isinstance(event, tuple) and event[0] == "pause" for event in world.events
    )
    assert "resume" not in world.events
    assert world.gate["paused"] is True
    assert saved["pause_start"] is None
    assert saved["pause_end"].endswith("Z")


def test_dry_run_has_no_mutations():
    world = World()
    result = run_reload(
        profile="sample",
        script="new",
        rollback_script="old",
        old_job_id="old",
        dry_run=True,
        read_jobs=world.jobs,
        cancel=world.cancel,
        submit=world.submit,
        read_lane=world.lane,
        read_gate=lambda: world.gate,
        pause=world.pause,
        resume=world.resume,
        job_running=lambda _job: True,
        ready=lambda _job: True,
        validate=lambda _job: True,
        clock=lambda: world.now,
        sleep=world.sleep,
    )
    assert result.outcome == "dry-run"
    assert not any(
        isinstance(event, tuple) and event[0] in {"cancel", "submit", "pause"}
        for event in world.events
    )


def test_unvalidated_settings_roll_back_by_twenty_two_minutes():
    world = World()
    world.running_result = lambda job: job == "new-2"
    result = world.run()
    assert result.outcome == "rolled-back"
    assert world.now <= 22 * 60 + 2
    assert world.events[-1] == "resume"


def test_nine_minute_drain_keeps_replacement_deadline_anchored_to_pause():
    world = World(ready=lambda job: job != "new-1")
    world.drain_seconds = 9 * 60
    result = world.run()
    rollback_submits = [
        event
        for event in world.events
        if isinstance(event, tuple) and event[:2] == ("submit", "saved batch")
    ]
    assert result.outcome == "rolled-back"
    assert rollback_submits
    assert rollback_submits[0][3] <= 22 * 60


def test_unvalidated_rollback_resumes_at_forty_minute_ceiling():
    world = World()
    world.running_result = False
    with pytest.raises(RuntimeError, match="40-minute pause bound"):
        world.run()
    assert 40 * 60 <= world.now <= 40 * 60 + 2
    assert world.events[-1] == "resume"


def test_failed_rollback_is_recorded_with_non_zero_result(tmp_path):
    world = World(ready=lambda job: False)
    world.running_result = lambda _job: True
    record = tmp_path / "lane-cycles.jsonl"
    with pytest.raises(RuntimeError, match="40-minute pause bound"):
        world.run(record_path=record)
    saved = json.loads(record.read_text().splitlines()[-1])
    assert saved["outcome"] == "rollback-failed"
    assert saved["pause_start"] == "2026-10-20T00:00:00Z"
    assert saved["pause_end"].endswith("Z")


def test_peak_waiting_is_sampled_after_drain(tmp_path):
    world = World()
    lane = world.lane

    def growing_queue():
        reading = lane()
        reading["router_generation_gate"] = {"waiting": 19 if world.submits else 3}
        return reading

    world.lane = growing_queue
    record = tmp_path / "lane-cycles.jsonl"
    result = world.run(record_path=record)
    assert result.peak_held == 19
    assert json.loads(record.read_text())["peak_held"] == 19


@pytest.mark.parametrize("tokens, outcome", [(199, "rolled-back"), (200, "reloaded")])
def test_real_validation_closure_enforces_completion_threshold(
    tmp_path, monkeypatch, tokens, outcome
):
    import urllib.request
    from types import SimpleNamespace

    from imas_ambix.agent import cli, slurm
    from imas_ambix.agent.profile import SiteConfig, load_profile

    monkeypatch.setenv("TMPDIR", str(tmp_path))
    profile = load_profile("glm-5-3").for_gpus(4)
    site = SiteConfig(
        base_dir=str(tmp_path), endpoint_document_path=str(tmp_path / "endpoints.json")
    )
    world = World()
    (tmp_path / "lane.json").write_text(json.dumps(world.lane()))
    requests = []

    def jobs(_site, **kwargs):
        return [
            dict(
                jobid=job,
                name=profile.slug,
                state="RUNNING",
                node="engine",
                comment="ambix-serve;port=18800",
            )
            for job in world.active
            if job in kwargs.get("job_ids", world.active)
        ]

    def command(args, **kwargs):
        if args[:3] == ["scontrol", "write", "batch_script"]:
            from pathlib import Path

            Path(args[-1]).write_text("saved batch")
        elif args[0] == "scancel":
            world.cancel(args[1])
        else:
            pytest.fail(str(args))
        return SimpleNamespace(returncode=0, stderr="")

    class Answer:
        """A urlopen response the engine's own readers can consume.

        The validator reads ``status`` and ``read()`` through the bench
        probes and ``headers`` through the routed completion check, so a
        bare file object is not a substitute for the engine's answer.
        """

        def __init__(self, payload, headers=None):
            self._body = json.dumps(payload).encode()
            self.status = 200
            self.headers = headers or {}

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    def completion(request, **kwargs):
        body = json.loads(request.data)
        if request.full_url.endswith("/v1/messages"):
            return Answer(
                {
                    "stop_reason": "tool_use",
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "get_weather",
                            "input": {"location": "Cadarache, France"},
                        }
                    ],
                }
            )
        if body.get("tools"):
            return Answer(
                {
                    "choices": [
                        {
                            "finish_reason": "tool_calls",
                            "message": {
                                "tool_calls": [
                                    {
                                        "function": {
                                            "name": "get_weather",
                                            "arguments": (
                                                '{"location": "Cadarache, France"}'
                                            ),
                                        }
                                    }
                                ]
                            },
                        }
                    ],
                }
            )
        requests.append(body)
        count = tokens if world.submits == 1 else 200
        return Answer(
            {
                "usage": {"completion_tokens": count},
                "choices": [{"finish_reason": "length"}],
            }
        )

    monkeypatch.setattr(cli.subprocess, "run", command)
    monkeypatch.setattr(cli, "_running_jobs", jobs)
    monkeypatch.setattr(cli, "_probe_endpoint", lambda *_: cli.ProbeResult("ready"))
    monkeypatch.setattr(slurm, "submit_script", world.submit)
    monkeypatch.setattr(urllib.request, "urlopen", completion)
    cli._execute_restart_cycle(
        profile, site, {"jobid": "old"}, "new script", None, 18800, "test", {}
    )
    record = json.loads((tmp_path / "lane-cycles.jsonl").read_text())
    assert record["outcome"] == outcome
    assert world.submits == (2 if tokens == 199 else 1)
    assert all(request["max_tokens"] == 400 for request in requests)
    assert (
        ("cancel", "new-1") in world.events
        if tokens == 199
        else ("cancel", "new-1") not in world.events
    )


def test_reload_accepts_jobs_already_selected_by_profile():
    world = World()
    jobs = world.jobs
    world.jobs = lambda: [{"jobid": job["jobid"]} for job in jobs()]
    assert world.run().outcome == "reloaded"


def test_peak_sampler_observes_queue_during_blocking_validation(tmp_path):
    import threading

    world = World()
    entered = threading.Event()
    sampled = threading.Event()
    lane = world.lane

    def reading():
        result = lane()
        if entered.is_set():
            result["router_generation_gate"] = {"waiting": 27}
            sampled.set()
        return result

    def validate(_job):
        entered.set()
        assert sampled.wait(3), "queue was never sampled during the completion"
        entered.clear()
        return True

    world.lane = reading
    world.valid_result = validate
    record = tmp_path / "lane-cycles.jsonl"
    assert world.run(record_path=record).peak_held == 27
    assert json.loads(record.read_text())["peak_held"] == 27
