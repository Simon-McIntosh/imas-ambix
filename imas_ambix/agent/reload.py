"""Orchestrate a bounded, paused serving-engine replacement."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


@dataclass(frozen=True)
class ReloadResult:
    outcome: str
    old_job_id: str | None
    new_job_id: str | None
    pause_started: float | None
    pause_ended: float | None
    peak_held: int


def run_reload(
    *,
    profile: str,
    script: str,
    rollback_script: str,
    old_job_id: str | None,
    dry_run: bool,
    gate_path: Path,
    lane_path: Path,
    read_jobs: Callable[[], list[dict[str, str]]],
    cancel: Callable[[str], None],
    submit: Callable[[str], str],
    read_lane: Callable[[], dict[str, object]],
    read_gate: Callable[[], dict[str, object]],
    pause: Callable[[str], None],
    resume: Callable[[], None],
    job_running: Callable[[str], bool],
    ready: Callable[[str], bool],
    validate: Callable[[str], bool],
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    record_path: Path | None = None,
    reason: str = "agent restart",
) -> ReloadResult:
    """Run the paused cycle using small external interfaces for deterministic tests."""
    del gate_path, lane_path
    jobs = [job for job in read_jobs() if job.get("name") == profile]
    if not dry_run and len(jobs) != 1:
        raise RuntimeError(f"expected exactly one running serve job; found {len(jobs)}")
    original = read_gate()
    was_paused = original.get("paused") is True
    if dry_run:
        return ReloadResult("dry-run", old_job_id, None, None, None, 0)

    started: float | None = None
    ended: float | None = None
    peak = 0
    new_job: str | None = None
    outcome = "aborted-before-cancel"
    changed_pause = False
    cycle_started = clock()
    cycle_deadline = cycle_started + 40 * 60
    try:
        if not was_paused:
            pause(reason)
            changed_pause = True
            started = cycle_started
        drain_deadline = min(cycle_started + 600, cycle_deadline)
        while True:
            reading = read_lane()
            try:
                from imas_ambix.agent.lane import classify_reading

                measured = classify_reading(reading) == "measured"
            except (ImportError, TypeError, ValueError):
                measured = False
            gate = reading.get("router_generation_gate")
            if isinstance(gate, dict) and isinstance(gate.get("waiting"), int):
                peak = max(peak, gate["waiting"])
            if measured and reading.get("running") == 0:
                break
            if clock() >= drain_deadline:
                outcome = "aborted-before-cancel"
                return ReloadResult(outcome, old_job_id, None, started, clock(), peak)
            sleep(2)

        if old_job_id:
            cancel(old_job_id)
            end = clock() + 60
            while any(job.get("jobid") == old_job_id for job in read_jobs()):
                if clock() >= end:
                    raise RuntimeError("timed out waiting for old serve to stop")
                sleep(2)
        new_job = submit(script)
        submitted_at = clock()
        running_since: float | None = None
        while clock() - submitted_at < 22 * 60:
            if job_running(new_job):
                if running_since is None:
                    running_since = clock()
                if ready(new_job):
                    break
                if clock() - running_since >= 15 * 60:
                    raise RuntimeError(
                        "replacement did not become ready within 15 minutes of RUNNING"
                    )
            sleep(2)
        else:
            raise RuntimeError("replacement was not validated within 22 minutes")
        if not validate(new_job):
            raise RuntimeError("replacement completion validation failed")
        outcome = "reloaded"
        return ReloadResult(outcome, old_job_id, new_job, started, clock(), peak)
    except Exception:
        if old_job_id and new_job is not None:
            try:
                cancel(new_job)
                rollback = submit(rollback_script)
                new_job = rollback
                while clock() < cycle_deadline:
                    if job_running(rollback) and ready(rollback) and validate(rollback):
                        outcome = "rolled-back"
                        return ReloadResult(
                            outcome, old_job_id, rollback, started, clock(), peak
                        )
                    sleep(2)
                raise RuntimeError(
                    "rollback did not validate within the 40-minute pause bound"
                )
            except Exception:
                raise
        raise
    finally:
        if changed_pause:
            ended = clock()
            resume()
        if record_path is not None and started is not None:
            record = {
                "pause_start": started,
                "pause_end": ended,
                "peak_held": peak,
                "replaced_job_id": old_job_id,
                "new_job_id": new_job,
                "outcome": outcome,
            }
            record_path.parent.mkdir(parents=True, exist_ok=True)
            with record_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
