"""The receipt row's stated section contract against what it records.

The recorder's docstring says a sparse section is omitted rather than nulled,
so a present key is a measurement, and it names the job table as the one
exception: a tick whose ``squeue`` refused carries the ``jobs`` key with the
failed-read marker instead of a measurement. These tests hold that exception at
the record boundary, where a reader meets it.
"""

from __future__ import annotations

import datetime as dt
import json

from imas_ambix.agent import node_probe, serving_receipts


def _probe_with_refused_jobs() -> node_probe.NodeProbe:
    """A probe whose every local read refused, so only the job table is present."""

    def run(argv, *, timeout_s):
        return None

    return node_probe.NodeProbe(run=run, env={}, hostname="n", job_interval_s=60.0)


def _probe_with_absent_jobs() -> node_probe.NodeProbe:
    """A probe on a node carrying no ``squeue`` at all.

    A missing program raises ``FileNotFoundError`` from the spawn, which is the
    only place the two kinds of non-answer are separable -- a refused query
    reports no output just as a missing command does.
    """

    def run(argv, *, timeout_s):
        if argv[0] == "squeue":
            raise FileNotFoundError(argv[0])
        return None

    return node_probe.NodeProbe(run=run, env={}, hostname="n", job_interval_s=60.0)


def _snapshot() -> dict:
    return {"gauges": {}, "counters": {}, "engine": None}


def _row_json(sections: dict) -> dict:
    row = serving_receipts.build_receipt_row(
        None,
        None,
        _snapshot(),
        dt.datetime(2026, 9, 29, 12, 0, 0, tzinfo=dt.UTC),
        job_id="1277272",
        profile_slug="deepseek-v4-1-flash",
        served_name="deepseek-v4-flash",
        gpus=4,
        hostname="n",
        node_sections=sections,
    )
    return json.loads(row.to_json())


def test_a_refused_job_read_and_an_absent_source_are_different_records():
    """The distinction a reader depends on, held at the record boundary.

    A tick whose ``squeue`` read fails and a tick where the job-table read was
    not due must not serialise to the same row: if they did, the record could
    not say whether the node's table was unreadable or simply not sampled.
    """
    probe = _probe_with_refused_jobs()

    refused = _row_json(probe.sample(0.0))  # read due this tick, and it refused
    not_due = _row_json(probe.sample(1.0))  # within the cadence, so no job key

    assert refused != not_due
    assert "jobs" in refused
    assert "jobs" not in not_due


def test_the_refused_job_read_carries_the_marker_not_a_measurement():
    """A present ``jobs`` key here marks a non-answer, not a measured table.

    The marker must arrive alone: no ``jobs`` rows and no ``count``, because a
    count would read as a table that answered with zero -- the reading the
    failed-read marker was added to make impossible.
    """
    probe = _probe_with_refused_jobs()

    jobs = _row_json(probe.sample(0.0))["jobs"]

    assert jobs[node_probe.UNREAD_KEY] == node_probe.COMMAND_FAILED
    assert "count" not in jobs
    assert "jobs" not in jobs
    assert set(jobs) == {"hostname", node_probe.UNREAD_KEY}


def test_an_answered_job_read_records_a_measurement_the_marker_case_lacks():
    """The positive control: the marker case and the answered case differ.

    Without this, the marker assertion could pass against a section that carries
    neither rows nor a marker, which would tell a reader nothing.
    """

    row = "1277272|mcintos|deepseek-v4-1-flash|RUNNING|n|12|600G|gres/gpu:4|1-00:00:00"

    def run(argv, *, timeout_s):
        return f"{row}\n" if argv[0] == "squeue" else None

    probe = node_probe.NodeProbe(run=run, env={}, hostname="n", job_interval_s=60.0)

    jobs = _row_json(probe.sample(0.0))["jobs"]

    assert node_probe.UNREAD_KEY not in jobs
    assert jobs["count"] == 1
    assert len(jobs["jobs"]) == 1


def test_an_absent_squeue_is_absent_from_neither_refused_nor_not_due():
    """A node with no ``squeue`` is a third record, distinct from the other two.

    A refused query is worth retrying and a missing command never is, so the
    row must keep the three cases apart at the record boundary: refused, absent,
    and not-yet-due. An absent command that serialised to the refused read's row
    would erase that distinction, and one that omitted the key would read as a
    tick where the read was simply not due.

    The absent marker must arrive alone, exactly as the refused one does: no
    ``count`` and no ``jobs`` rows, because either would read as a table that
    answered. Naming the marker at section level is not enough -- a reader meets
    the key set, so the absence of a measurement has to hold there.
    """
    refused = _row_json(_probe_with_refused_jobs().sample(0.0))

    probe = _probe_with_absent_jobs()
    absent = _row_json(probe.sample(0.0))
    not_due = _row_json(probe.sample(1.0))

    assert absent != refused
    assert absent != not_due

    jobs = absent["jobs"]
    assert jobs[node_probe.UNREAD_KEY] == node_probe.COMMAND_ABSENT
    assert "count" not in jobs
    assert "jobs" not in jobs
    assert set(jobs) == {"hostname", node_probe.UNREAD_KEY}
