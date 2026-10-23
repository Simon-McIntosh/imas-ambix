"""``imas-ambix agent ingest`` runs the standing ingest the fleet supervisor keeps.

The ingest is a standing loop, not a scheduled or finite job: it keeps the
telemetry index current from the recorded receipts for as long as it runs, so
the tests here pin that it offers no way to submit itself as its own job and no
way to print such a job's script. Each test drives the CLI and observes the seam
the command crosses -- the ingest loop it calls -- rather than what the index
happens to hold.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import types
from datetime import UTC, datetime

from click.testing import CliRunner

from imas_ambix.agent.telemetry_index import IngestReport
from imas_ambix.agent.telemetry_ingest import TickReport
from imas_ambix.cli import main


def _old_schema_index_with_wal(path) -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import os, sqlite3, sys; "
            "db = sqlite3.connect(sys.argv[1]); "
            "db.execute('PRAGMA journal_mode=WAL'); "
            "db.execute('CREATE TABLE source ("
            "mtime_ns INTEGER, head_sha TEXT, tail_sha TEXT)'); "
            "db.execute('CREATE TABLE measurement (name TEXT)'); "
            "db.execute(\"INSERT INTO measurement VALUES ('stale')\"); "
            "db.commit(); os._exit(0)",
            str(path),
        ],
        check=True,
    )
    assert path.with_name(path.name + "-wal").stat().st_size > 0
    assert path.with_name(path.name + "-shm").exists()


def test_old_schema_index_rebuilds_before_first_tick(tmp_path, monkeypatch) -> None:
    from imas_ambix.agent import cli, watch_publish

    records = tmp_path / "receipts"
    records.mkdir()
    (records / "serve.jsonl").write_text(
        '{"timestamp":"2026-10-23T00:00:00Z","job_id":"123","engine":{"generation_tokens":7}}\n'
    )
    index_path = tmp_path / "watch-index.sqlite3"
    _old_schema_index_with_wal(index_path)
    replace = os.replace
    swap_checked = []

    def check_swap(source, destination):
        if destination == index_path:
            assert source.exists()
            swap_checked.append(
                not index_path.with_name(index_path.name + "-wal").exists()
                and not index_path.with_name(index_path.name + "-shm").exists()
            )
        replace(source, destination)

    monkeypatch.setattr(cli.os, "replace", check_swap)
    monkeypatch.setattr(
        watch_publish, "one_iteration", lambda *_args, **_kwargs: "published"
    )
    result = CliRunner().invoke(
        main, ["agent", "ingest", "--once", "--no-retention", *_ingest_args(tmp_path)]
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert swap_checked == [True]
    assert result.output.count("index rebuild") == 1
    assert "duration=" in result.output
    assert "ingest tick" in result.output
    assert not index_path.with_name(index_path.name + "-wal").exists()
    assert not index_path.with_name(index_path.name + "-shm").exists()
    with sqlite3.connect(index_path) as connection:
        assert connection.execute("SELECT count(*) FROM sample").fetchone()[0] == 1


def test_failed_rebuild_keeps_old_index(tmp_path, monkeypatch) -> None:
    from imas_ambix.agent import telemetry_index

    index_path = tmp_path / "watch-index.sqlite3"
    _old_schema_index_with_wal(index_path)
    before = index_path.read_bytes()
    wal_path = index_path.with_name(index_path.name + "-wal")
    before_wal = wal_path.read_bytes()
    shm_path = index_path.with_name(index_path.name + "-shm")

    def fail_rebuild(self, sources):
        self._conn.execute("CREATE TABLE partial_rebuild (value INTEGER)")
        self._conn.commit()
        raise RuntimeError("rebuild interrupted")

    monkeypatch.setattr(telemetry_index.TelemetryIndex, "rebuild", fail_rebuild)
    result = CliRunner().invoke(
        main, ["agent", "ingest", "--once", "--no-retention", *_ingest_args(tmp_path)]
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, RuntimeError)
    assert index_path.read_bytes() == before
    assert wal_path.read_bytes() == before_wal
    assert shm_path.exists()
    assert not list(tmp_path.glob("*rebuild*"))


def _ingest_args(tmp_path) -> list[str]:
    """A record and index inside the temporary test tree, so a real run is safe."""
    return [
        "--record",
        str(tmp_path / "receipts"),
        "--index",
        str(tmp_path / "watch-index.sqlite3"),
    ]


def _help() -> str:
    """The rendered ``ingest --help`` output, whitespace-flattened.

    Flattened so a wrap that lands between the tokens these tests look for
    cannot make an absent string read as present, or a present one as absent.
    """
    result = CliRunner().invoke(main, ["agent", "ingest", "--help"])
    assert result.exit_code == 0, result.output
    return " ".join(result.output.split())


def test_help_offers_neither_submit_nor_dry_run() -> None:
    """A second, unused placement must not be offered, so neither option appears."""
    text = _help()
    assert "--submit" not in text
    assert "--dry-run" not in text


def test_help_describes_the_standing_loop_the_supervisor_keeps() -> None:
    """The help names the placement that actually runs the loop."""
    text = _help()
    assert "loop" in text
    assert "supervisor" in text


def test_help_shows_the_sixty_second_default_cadence() -> None:
    """The default cadence is the locked sixty seconds, and the help says so."""
    text = _help()
    assert "--cadence" in text
    assert "[default: 60.0]" in text


def test_once_invokes_a_single_tick_through_a_stub(tmp_path, monkeypatch):
    """``--once`` runs the loop for exactly one iteration and no more."""
    from imas_ambix.agent import telemetry_ingest

    calls: list[tuple] = []

    def fake_run(index, directory, **kwargs):
        calls.append((index, directory, kwargs))
        return []

    monkeypatch.setattr(telemetry_ingest, "run", fake_run)

    result = CliRunner().invoke(
        main, ["agent", "ingest", "--once", *_ingest_args(tmp_path)]
    )

    assert result.exit_code == 0
    assert len(calls) == 1
    assert calls[0][2]["iterations"] == 1
    # The default cadence is the locked sixty seconds; --once pays nothing for it
    # because the loop never sleeps past the last tick.
    assert calls[0][2]["cadence"] == 60.0


def test_cadence_overrides_the_default(tmp_path, monkeypatch):
    """``--cadence`` reaches the loop instead of the sixty-second default."""
    from imas_ambix.agent import telemetry_ingest

    calls: list[tuple] = []

    def fake_run(index, directory, **kwargs):
        calls.append((index, directory, kwargs))
        return []

    monkeypatch.setattr(telemetry_ingest, "run", fake_run)

    result = CliRunner().invoke(
        main,
        ["agent", "ingest", "--once", "--cadence", "0.5", *_ingest_args(tmp_path)],
    )

    assert result.exit_code == 0
    assert calls[0][2]["cadence"] == 0.5


FROZEN_TICK_TIME = datetime(2026, 10, 2, 8, 30, 4, tzinfo=UTC)
FROZEN_TICK_STAMP = "2026-10-02T08:30:04Z"

#: Two ticks with distinct counts, so a line built from the wrong report's
#: numbers cannot equal the expected literal by coincidence.
FIRST_TICK = (118, 3, 115, 3, 240, 0, 0, 51234)
SECOND_TICK = (120, 2, 118, 1, 7, 5, 1, 2048)


def _tick_report(counts: tuple[int, ...]) -> TickReport:
    discovered, read, skipped, scanned, inserted, duplicate, malformed, size = counts
    return TickReport(
        files_discovered=discovered,
        files_read=read,
        files_skipped=skipped,
        ingest=IngestReport(
            files_scanned=scanned,
            rows_inserted=inserted,
            rows_duplicate=duplicate,
            malformed=malformed,
            bytes_read=size,
        ),
        sources={},
    )


def _stub_two_ticks(monkeypatch, reports: list[TickReport]) -> None:
    """Replace the ingest loop with one that hands *reports* to *on_tick*."""
    from imas_ambix.agent import telemetry_ingest

    def fake_run(index, directory, **kwargs):
        for report in reports:
            kwargs["on_tick"](report)
        return []

    monkeypatch.setattr(telemetry_ingest, "run", fake_run)


def _stub_publication(monkeypatch) -> None:
    from imas_ambix.agent import watch_publish

    def publish(_output, **_kwargs):
        return "watch published"

    monkeypatch.setattr(watch_publish, "one_iteration", publish)


def test_help_names_the_per_tick_line() -> None:
    """The help states the loop prints a named per-tick line."""
    text = _help()
    assert "ingest tick" in text
    for field in (
        "discovered=",
        "read=",
        "skipped=",
        "scanned=",
        "inserted=",
        "duplicate=",
        "malformed=",
        "bytes=",
    ):
        assert field in text


def test_each_tick_prints_one_line_from_its_own_counts(tmp_path, monkeypatch) -> None:
    """Two ticks produce two lines, each stamped and carrying its own counts."""
    from imas_ambix.agent import cli

    _stub_two_ticks(monkeypatch, [_tick_report(FIRST_TICK), _tick_report(SECOND_TICK)])
    _stub_publication(monkeypatch)
    monkeypatch.setattr(cli, "_utc_now", lambda: FROZEN_TICK_TIME)

    result = CliRunner().invoke(
        main, ["agent", "ingest", "--once", *_ingest_args(tmp_path)]
    )

    assert result.exit_code == 0, result.output
    assert result.output.splitlines() == [
        f"{FROZEN_TICK_STAMP} ingest tick discovered=118 read=3 skipped=115 "
        "scanned=3 inserted=240 duplicate=0 malformed=0 bytes=51234",
        "watch published",
        f"{FROZEN_TICK_STAMP} ingest tick discovered=120 read=2 skipped=118 "
        "scanned=1 inserted=7 duplicate=5 malformed=1 bytes=2048",
    ]


class _RecordingStream:
    """A stdout stand-in that records every write and flush, in order."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def write(self, text: str) -> None:
        self.calls.append(("write", text))

    def flush(self) -> None:
        self.calls.append(("flush", ""))


def test_a_flush_follows_each_line(tmp_path, monkeypatch) -> None:
    """Each tick's line is flushed before the next tick's line is written."""
    from imas_ambix.agent import cli

    reports = [_tick_report(FIRST_TICK), _tick_report(SECOND_TICK)]
    _stub_two_ticks(monkeypatch, reports)
    _stub_publication(monkeypatch)
    monkeypatch.setattr(cli, "_utc_now", lambda: FROZEN_TICK_TIME)

    recorded = _RecordingStream()
    monkeypatch.setattr(cli, "sys", types.SimpleNamespace(stdout=recorded))

    cli.ingest.callback(
        record_dir=str(tmp_path / "receipts"),
        index_path=str(tmp_path / "watch-index.sqlite3"),
        cadence=60.0,
        once=True,
    )

    assert recorded.calls == [
        (
            "write",
            f"{FROZEN_TICK_STAMP} ingest tick discovered=118 read=3 skipped=115 "
            "scanned=3 inserted=240 duplicate=0 malformed=0 bytes=51234\n",
        ),
        ("flush", ""),
        ("write", "watch published\n"),
        ("flush", ""),
        (
            "write",
            f"{FROZEN_TICK_STAMP} ingest tick discovered=120 read=2 skipped=118 "
            "scanned=1 inserted=7 duplicate=5 malformed=1 bytes=2048\n",
        ),
        ("flush", ""),
    ]


def test_publication_uses_attempt_cadence_and_resolved_index(
    tmp_path, monkeypatch
) -> None:
    """A quick ingest tick does not force a second publication before its due time."""
    from imas_ambix.agent import cli, telemetry_ingest, watch_publish

    reports = [_tick_report(FIRST_TICK)] * 4
    clock = [0.0]
    attempts: list[tuple[float, object, object]] = []
    monkeypatch.setattr(cli, "_publication_clock", lambda: clock[0])

    def publish(_output, *, record_dir, index_path):
        attempts.append((clock[0], record_dir, index_path))
        return "watch published"

    def fake_run(index, directory, **kwargs):
        for when, report in zip((0.0, 60.0, 299.0, 300.0), reports, strict=True):
            clock[0] = when
            kwargs["on_tick"](report)
        return []

    monkeypatch.setattr(watch_publish, "one_iteration", publish)
    monkeypatch.setattr(telemetry_ingest, "run", fake_run)
    result = CliRunner().invoke(
        main, ["agent", "ingest", "--once", *_ingest_args(tmp_path)]
    )

    assert result.exit_code == 0, result.output
    assert [when for when, _, _ in attempts] == [0.0, 300.0]
    assert all(
        path == (tmp_path / "watch-index.sqlite3").resolve() for _, _, path in attempts
    )
    assert all(path == (tmp_path / "receipts").resolve() for _, path, _ in attempts)
    assert result.output.count("ingest tick") == 4


def test_publish_every_option_changes_due_time(tmp_path, monkeypatch) -> None:
    """The ingest option, not the tick cadence, decides when publication is due."""
    from imas_ambix.agent import cli, telemetry_ingest, watch_publish

    clock = [0.0]
    attempts: list[float] = []
    monkeypatch.setattr(cli, "_publication_clock", lambda: clock[0])

    def publish(_output, **_kwargs):
        attempts.append(clock[0])
        return "watch published"

    def fake_run(index, directory, **kwargs):
        for when in (0.0, 60.0, 120.0):
            clock[0] = when
            kwargs["on_tick"](_tick_report(FIRST_TICK))
        return []

    monkeypatch.setattr(watch_publish, "one_iteration", publish)
    monkeypatch.setattr(telemetry_ingest, "run", fake_run)
    result = CliRunner().invoke(
        main,
        [
            "agent",
            "ingest",
            "--once",
            "--publish-every",
            "120",
            *_ingest_args(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert attempts == [0.0, 120.0]


def test_failed_publication_preserves_file_and_next_tick_ingests(
    tmp_path, monkeypatch
) -> None:
    """A failed derivation leaves the old bytes and does not stop the ingest loop."""
    from imas_ambix.agent import (
        cli,
        telemetry_ingest,
    )
    from imas_ambix.agent import (
        watch as watch_mod,
    )

    target = tmp_path / "watch.json"
    previous = b'{"published_at": "earlier", "keep": true}\n'
    target.write_bytes(previous)
    monkeypatch.setattr(watch_mod, "PUBLISHED_WATCH_PATH", target)
    clock = [0.0]
    monkeypatch.setattr(cli, "_publication_clock", lambda: clock[0])
    seen: list[bytes] = []

    def document(**_kwargs):
        if clock[0] == 0.0:
            raise RuntimeError("database is locked")
        return {"record": {"tick": 2}}

    def fake_run(index, directory, **kwargs):
        for when, counts in ((0.0, FIRST_TICK), (60.0, SECOND_TICK)):
            clock[0] = when
            kwargs["on_tick"](_tick_report(counts))
            seen.append(target.read_bytes())
        return []

    monkeypatch.setattr(watch_mod, "watch_document", document)
    monkeypatch.setattr(telemetry_ingest, "run", fake_run)
    result = CliRunner().invoke(
        main, ["agent", "ingest", "--once", *_ingest_args(tmp_path)]
    )

    assert result.exit_code == 0, result.output
    assert seen[0] == previous
    assert seen[1] == previous
    assert result.output.count("ingest tick") == 2
    assert "publish-watch failed: database is locked" in result.output
    assert "publish-watch wrote" not in result.output


def test_failed_publication_waits_until_the_next_due_time(
    tmp_path, monkeypatch
) -> None:
    """A raised derivation is retried at the publication cadence, not each tick."""
    from imas_ambix.agent import (
        cli,
        telemetry_ingest,
    )
    from imas_ambix.agent import (
        watch as watch_mod,
    )

    target = tmp_path / "watch.json"
    previous = b'{"published_at": "earlier", "keep": true}\n'
    target.write_bytes(previous)
    monkeypatch.setattr(watch_mod, "PUBLISHED_WATCH_PATH", target)
    clock = [0.0]
    attempts: list[float] = []
    seen: list[bytes] = []
    monkeypatch.setattr(cli, "_publication_clock", lambda: clock[0])

    def document(**_kwargs):
        attempts.append(clock[0])
        if len(attempts) == 1:
            raise RuntimeError("database is locked")
        return {"record": {"tick": len(attempts)}}

    def fake_run(index, directory, **kwargs):
        for when in (0.0, 60.0, 299.0, 300.0):
            clock[0] = when
            kwargs["on_tick"](_tick_report(FIRST_TICK))
            seen.append(target.read_bytes())
        return []

    monkeypatch.setattr(watch_mod, "watch_document", document)
    monkeypatch.setattr(telemetry_ingest, "run", fake_run)
    result = CliRunner().invoke(
        main, ["agent", "ingest", "--once", *_ingest_args(tmp_path)]
    )

    assert result.exit_code == 0, result.output
    assert attempts == [0.0, 300.0]
    assert seen[:3] == [previous] * 3
    assert seen[3] != previous
    assert result.output.count("ingest tick") == 4
