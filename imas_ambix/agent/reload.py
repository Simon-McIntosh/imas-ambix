"""Orchestrate bounded serving-engine replacements behind the router."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path


@dataclass(frozen=True)
class ReloadResult:
    """Summary of one engine replacement cycle."""

    outcome: str
    old_job_id: str | None
    new_job_id: str | None
    pause_started: float | None
    pause_ended: float | None
    peak_held: int


def _utc_stamp(clock: Callable[[], datetime]) -> str:
    """Return a wall-clock timestamp suitable for a durable cycle record."""
    value = clock()
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def run_reload(
    *,
    profile: str,
    script: str,
    rollback_script: str,
    old_job_id: str | None,
    dry_run: bool,
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
    wall_clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    sleep: Callable[[float], None] = time.sleep,
    record_path: Path | None = None,
    reason: str = "agent restart",
    settings_changed: Mapping[str, object] | None = None,
) -> ReloadResult:
    """Run the paused cycle through small, deterministic external interfaces."""
    jobs = read_jobs()
    if not dry_run and len(jobs) != 1:
        raise RuntimeError(f"expected exactly one running serve job; found {len(jobs)}")
    if dry_run:
        return ReloadResult("dry-run", old_job_id, None, None, None, 0)

    was_paused = read_gate().get("paused") is True
    started: float | None = None
    pause_start_record: str | None = None
    pause_end_record: str | None = None
    peak = 0
    new_job: str | None = None
    old_stopped = False
    outcome = "aborted-before-cancel"
    changed_pause = False
    cycle_observed_at = clock()
    pause_deadline_origin = cycle_observed_at
    drain_deadline = pause_deadline_origin + 10 * 60
    replacement_deadline = pause_deadline_origin + 22 * 60
    cycle_deadline = pause_deadline_origin + 40 * 60
    sampling_done = threading.Event()
    peak_lock = threading.Lock()

    def sample_peak() -> None:
        nonlocal peak
        try:
            gate = read_lane().get("router_generation_gate")
            if isinstance(gate, dict) and isinstance(gate.get("waiting"), int):
                with peak_lock:
                    peak = max(peak, gate["waiting"])
        except (OSError, TypeError, ValueError):
            # Missing telemetry cannot supply a peak. Drain still requires a
            # measured reading through its independent synchronous check.
            pass

    def sample_until_resumed() -> None:
        while not sampling_done.wait(1):
            sample_peak()

    sampler = threading.Thread(target=sample_until_resumed, daemon=True)

    def wait(seconds: float) -> None:
        sleep(seconds)
        sample_peak()

    try:
        if not was_paused:
            pause(reason)
            changed_pause = True
            started = clock()
            pause_deadline_origin = started
            drain_deadline = started + 10 * 60
            replacement_deadline = started + 22 * 60
            cycle_deadline = started + 40 * 60
            pause_start_record = _utc_stamp(wall_clock)

        sampler.start()
        while True:
            reading = read_lane()
            try:
                from imas_ambix.agent.lane import classify_reading

                measured = classify_reading(reading) == "measured"
            except (ImportError, TypeError, ValueError):
                measured = False
            gate = reading.get("router_generation_gate")
            if isinstance(gate, dict) and isinstance(gate.get("waiting"), int):
                with peak_lock:
                    peak = max(peak, gate["waiting"])
            if measured and reading.get("running") == 0:
                break
            if clock() >= drain_deadline:
                return ReloadResult(outcome, old_job_id, None, started, clock(), peak)
            wait(min(2, max(0, drain_deadline - clock())))

        if old_job_id:
            cancel(old_job_id)
            cancel_deadline = min(clock() + 60, cycle_deadline)
            while any(job.get("jobid") == old_job_id for job in read_jobs()):
                if clock() >= cancel_deadline:
                    raise RuntimeError("timed out waiting for old serve to stop")
                wait(min(2, max(0, cancel_deadline - clock())))
            old_stopped = True

        new_job = submit(script)
        sample_peak()
        running_since: float | None = None
        validated = False
        while clock() < replacement_deadline:
            if job_running(new_job):
                if running_since is None:
                    running_since = clock()
                if ready(new_job):
                    if validate(new_job):
                        sample_peak()
                        if clock() <= replacement_deadline:
                            validated = True
                            break
                        raise RuntimeError(
                            "replacement validation finished after 22 minutes"
                        )
                    raise RuntimeError("replacement completion validation failed")
                if clock() - running_since >= 15 * 60:
                    raise RuntimeError(
                        "replacement did not become ready within 15 minutes of RUNNING"
                    )
            wait(min(2, max(0, replacement_deadline - clock())))
        if not validated:
            raise RuntimeError("replacement was not ready and validated by 22 minutes")

        outcome = "reloaded"
        return ReloadResult(outcome, old_job_id, new_job, started, clock(), peak)
    except Exception as cycle_error:
        if old_stopped:
            outcome = "rollback-failed"
            try:
                if new_job is not None:
                    cancel(new_job)
                rollback = submit(rollback_script)
                new_job = rollback
                while clock() < cycle_deadline:
                    if (
                        job_running(rollback)
                        and ready(rollback)
                        and validate(rollback)
                        and clock() <= cycle_deadline
                    ):
                        outcome = "rolled-back"
                        sample_peak()
                        return ReloadResult(
                            outcome, old_job_id, rollback, started, clock(), peak
                        )
                    wait(min(2, max(0, cycle_deadline - clock())))
            except Exception as rollback_error:
                outcome = "rollback-failed"
                raise RuntimeError(
                    f"replacement failed and rollback failed: {rollback_error}"
                ) from cycle_error
            outcome = "rollback-failed"
            raise RuntimeError(
                "rollback did not validate within the 40-minute pause bound"
            ) from cycle_error
        raise
    finally:
        try:
            sampling_done.set()
            if sampler.ident is not None:
                sampler.join()
            if changed_pause:
                resume()
        finally:
            pause_end_record = _utc_stamp(wall_clock)
            if record_path is not None:
                record = {
                    "pause_start": pause_start_record,
                    "pause_end": pause_end_record,
                    "peak_held": peak,
                    "replaced_job_id": old_job_id,
                    "new_job_id": new_job,
                    "settings_changed": dict(settings_changed or {}),
                    "outcome": outcome,
                }
                record_path.parent.mkdir(parents=True, exist_ok=True)
                with record_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
