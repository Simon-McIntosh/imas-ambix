"""Deterministic checks for the paused serving reload cycle."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from imas_ambix.agent.reload import run_reload


class World:
    def __init__(self, *, gate=None, readings=None, ready=True, valid=True):
        self.now = 0.0
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
        self.events.append(("submit", script, job))
        self.active.add(job)
        return job

    def lane(self):
        self.events.append("lane")
        return self.readings[0]

    def pause(self, reason):
        self.events.append(("pause", reason))
        self.gate["paused"] = True

    def resume(self):
        self.events.append("resume")
        self.gate["paused"] = False

    def sleep(self, seconds):
        self.now += seconds

    def run(self, **kwargs):
        return run_reload(
            profile="sample",
            script="new script",
            rollback_script="saved batch",
            old_job_id="old",
            dry_run=False,
            gate_path=Path("gate"),
            lane_path=Path("lane"),
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
            sleep=self.sleep,
            **kwargs,
        )


def test_success_order_and_cycle_record(tmp_path):
    world = World()
    record = tmp_path / "lane-cycles.jsonl"
    result = world.run(record_path=record)
    assert result.outcome == "reloaded"
    assert world.events.index(("pause", "agent restart")) < world.events.index("lane")
    assert world.events.index(("cancel", "old")) < next(
        i
        for i, event in enumerate(world.events)
        if isinstance(event, tuple) and event[0] == "submit"
    )
    assert world.events[-1] == "resume"
    assert json.loads(record.read_text())["outcome"] == "reloaded"


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
    world = World(valid=lambda job: job != "new-1")
    world.completion_tokens["new-1"] = 199
    result = world.run()
    assert result.outcome == "rolled-back"
    assert any(
        event[:2] == ("submit", "saved batch")
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


def test_operator_pause_is_preserved():
    world = World(gate={"paused": True})
    world.run()
    assert not any(
        isinstance(event, tuple) and event[0] == "pause" for event in world.events
    )
    assert "resume" not in world.events
    assert world.gate["paused"] is True


def test_dry_run_has_no_mutations():
    world = World()
    result = run_reload(
        profile="sample",
        script="new",
        rollback_script="old",
        old_job_id="old",
        dry_run=True,
        gate_path=Path("gate"),
        lane_path=Path("lane"),
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


def test_unvalidated_rollback_resumes_at_forty_minute_ceiling():
    world = World()
    world.running_result = False
    with pytest.raises(RuntimeError, match="40-minute pause bound"):
        world.run()
    assert 40 * 60 <= world.now <= 40 * 60 + 2
    assert world.events[-1] == "resume"
