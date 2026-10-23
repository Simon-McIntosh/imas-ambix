"""The watch index remains local while verified snapshots survive on shared storage."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from compression import zstd
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from click.testing import CliRunner

from imas_ambix.agent import index_placement, telemetry_ingest, watch, watch_publish
from imas_ambix.agent.telemetry_index import IngestReport, TelemetryIndex
from imas_ambix.agent.telemetry_ingest import TickReport
from imas_ambix.cli import main


def _receipt(path: Path, second: int) -> None:
    row = {
        "timestamp": (
            datetime(2026, 10, 23, tzinfo=UTC) + timedelta(seconds=second)
        ).isoformat(),
        "job_id": "1234567",
        "engine": {"family": "sglang", "generation_tokens": 1000 + second},
    }
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row) + "\n")


def _digest(path: Path) -> str:
    with sqlite3.connect(path) as connection:
        dump = "\n".join(connection.iterdump()).encode()
    return hashlib.sha256(dump).hexdigest()


def _counts(path: Path) -> tuple[int, int, int]:
    with sqlite3.connect(path) as connection:
        return tuple(
            connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            for table in ("source", "sample", "measurement")
        )


def _logical_rows(path: Path) -> tuple[list[tuple], list[tuple], list[tuple]]:
    with sqlite3.connect(path) as connection:
        source_columns = [
            row[1]
            for row in connection.execute("PRAGMA table_info(source)")
            if row[1] != "mtime_ns"
        ]
        return (
            connection.execute(
                f"SELECT {', '.join(source_columns)} FROM source"
            ).fetchall(),
            connection.execute("SELECT * FROM sample").fetchall(),
            connection.execute("SELECT * FROM measurement").fetchall(),
        )


def test_resolver_uses_override_then_local_runtime_then_none(tmp_path, monkeypatch):
    monkeypatch.setenv("FLEET_RUNTIME_DIR", str(tmp_path))
    monkeypatch.delenv("AMBIX_WATCH_INDEX", raising=False)
    assert (
        index_placement.resolve_index_path()
        == tmp_path / "ambix" / "watch-index.sqlite3"
    )
    override = tmp_path / "override.sqlite3"
    monkeypatch.setenv("AMBIX_WATCH_INDEX", str(override))
    assert index_placement.resolve_index_path() == override
    assert (
        index_placement.resolve_index_path(tmp_path / "explicit.sqlite3")
        == tmp_path / "explicit.sqlite3"
    )
    monkeypatch.delenv("AMBIX_WATCH_INDEX")
    monkeypatch.delenv("FLEET_RUNTIME_DIR")
    assert index_placement.resolve_index_path() is None


def test_runtime_candidate_on_network_filesystem_is_skipped(tmp_path, monkeypatch):
    monkeypatch.setenv("FLEET_RUNTIME_DIR", str(tmp_path))
    monkeypatch.delenv("AMBIX_WATCH_INDEX", raising=False)
    monkeypatch.setattr(
        index_placement,
        "filesystem_type",
        lambda path: 0x47504653 if Path(path).name == "watch-index.sqlite3" else 0,
    )
    assert index_placement.resolve_index_path() is None


def test_network_filesystem_refuses_wal(tmp_path, monkeypatch):
    with TelemetryIndex(tmp_path / "local.sqlite3") as index:
        index_placement.enable_wal(index)
        assert index._conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    monkeypatch.setattr(index_placement, "filesystem_type", lambda path: 0x47504653)
    with pytest.raises(ValueError, match="requires local disk for WAL"):
        index_placement.resolve_index_path(tmp_path / "network.sqlite3")
    with (
        TelemetryIndex(tmp_path / "network.sqlite3") as index,
        pytest.raises(ValueError, match="requires local disk for WAL"),
    ):
        index_placement.enable_wal(index)


def test_snapshot_round_trip_preserves_rows_and_dump(tmp_path):
    source = tmp_path / "receipts.jsonl"
    _receipt(source, 0)
    _receipt(source, 5)
    origin = tmp_path / "original.sqlite3"
    snapshots = tmp_path / "snapshots"
    with TelemetryIndex(origin) as index:
        index_placement.enable_wal(index)
        assert index.ingest([source]).rows_inserted == 2
        stored = index_placement.snapshot_index(index, directory=snapshots)
    restored = tmp_path / "restored.sqlite3"
    assert stored.suffix == ".zst"
    assert index_placement.snapshot_exists(directory=snapshots)
    assert index_placement.restore_index(restored, directory=snapshots) == stored
    assert _counts(origin) == _counts(restored)
    assert _digest(origin) == _digest(restored)


def test_restore_clears_stale_wal_before_replacing_index(tmp_path, monkeypatch):
    source = tmp_path / "receipts.jsonl"
    _receipt(source, 0)
    snapshots = tmp_path / "snapshots"
    with TelemetryIndex(tmp_path / "source.sqlite3") as index:
        index.ingest([source])
        stored = index_placement.snapshot_index(index, directory=snapshots)
    target = tmp_path / "restored.sqlite3"
    original = index_placement._decompress_and_check
    stale_connections = []

    def create_stale_wal(snapshot, candidate):
        stale = sqlite3.connect(target)
        stale_connections.append(stale)
        stale.execute("PRAGMA journal_mode=WAL")
        stale.execute("PRAGMA wal_autocheckpoint=0")
        stale.execute("CREATE TABLE stale_marker (value INTEGER)")
        stale.commit()
        assert target.with_name(target.name + "-wal").exists()
        original(snapshot, candidate)

    monkeypatch.setattr(index_placement, "_decompress_and_check", create_stale_wal)
    try:
        assert index_placement.restore_index(target, directory=snapshots) == stored
        with sqlite3.connect(target) as restored:
            assert restored.execute("SELECT COUNT(*) FROM measurement").fetchone()[0]
            assert not restored.execute(
                "SELECT 1 FROM sqlite_master WHERE name = 'stale_marker'"
            ).fetchone()
    finally:
        for stale in stale_connections:
            stale.close()


def test_old_schema_snapshot_is_skipped_and_kept(tmp_path, capsys, monkeypatch):
    snapshots = index_placement.SNAPSHOT_DIR
    snapshots.mkdir()
    old = snapshots / "watch-index-20261023T000000000000Z.sqlite3.zst"
    source = tmp_path / "old.sqlite3"
    with sqlite3.connect(source) as connection:
        connection.execute(
            "CREATE TABLE source (mtime_ns INTEGER, head_sha TEXT, tail_sha TEXT)"
        )
        connection.execute("CREATE TABLE measurement (name TEXT)")
    with source.open("rb") as plain, zstd.open(old, "wb") as compressed:
        compressed.write(plain.read())
    assert TelemetryIndex.requires_rebuild(source)
    target = tmp_path / "restored.sqlite3"
    assert index_placement.restore_index(target, directory=snapshots) is None
    assert old.exists()
    assert not target.exists()
    assert "schema" in capsys.readouterr().err
    records = tmp_path / "records"
    records.mkdir()
    _receipt(records / "serve.jsonl", 0)
    monkeypatch.setattr(
        watch_publish, "one_iteration", lambda *_args, **_kwargs: "published"
    )
    result = CliRunner().invoke(
        main,
        [
            "agent",
            "ingest",
            "--once",
            "--no-retention",
            "--record",
            str(records),
            "--index",
            str(target),
        ],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert "index rebuild" in result.output
    assert "ingest tick" in result.output
    assert old.exists()
    assert _counts(target)[1] == 1


def test_legacy_counter_column_remains_compatible(tmp_path):
    target = tmp_path / "index.sqlite3"
    with TelemetryIndex(target):
        pass
    with sqlite3.connect(target) as connection:
        assert "counter_diff" not in {
            row[1] for row in connection.execute("PRAGMA table_info(hourly_rollup)")
        }
    with sqlite3.connect(target) as connection:
        connection.execute("ALTER TABLE hourly_rollup ADD COLUMN counter_diff REAL")
    assert not TelemetryIndex.requires_rebuild(target)
    source = tmp_path / "receipts.jsonl"
    _receipt(source, 0)
    _receipt(source, 5)
    with TelemetryIndex(target) as index:
        assert index.ingest([source]).rows_inserted == 2
        assert index._conn.execute("SELECT COUNT(*) FROM hourly_rollup").fetchone()[0]


def test_old_schema_snapshot_falls_back_to_compatible_copy(tmp_path, capsys):
    snapshots = tmp_path / "snapshots"
    record = tmp_path / "serve.jsonl"
    _receipt(record, 0)
    with TelemetryIndex(tmp_path / "source.sqlite3") as index:
        index.ingest([record])
        older = index_placement.snapshot_index(
            index, directory=snapshots, now=datetime(2026, 10, 22, tzinfo=UTC)
        )
    source = tmp_path / "old.sqlite3"
    with sqlite3.connect(source) as connection:
        connection.execute("CREATE TABLE source (mtime_ns INTEGER)")
        connection.execute("CREATE TABLE measurement (name TEXT)")
    newer = snapshots / "watch-index-20261023T000000000000Z.sqlite3.zst"
    with source.open("rb") as plain, zstd.open(newer, "wb") as compressed:
        compressed.write(plain.read())
    target = tmp_path / "restored.sqlite3"
    assert index_placement.restore_index(target, directory=snapshots) == older
    assert newer.exists()
    assert "snapshot skipped" in capsys.readouterr().err
    assert _counts(target)[1] == 1


def test_ingest_command_restores_and_snapshots_only_in_test_directory(
    tmp_path, monkeypatch
):
    production = Path.home() / ".local/state/ambix/index-snapshots"
    snapshots = tmp_path / "index-snapshots"

    def reject_production(path):
        if path == production or production in path.parents:
            raise AssertionError(f"ingest accessed production snapshots: {path}")

    for method in ("is_dir", "mkdir", "glob", "open"):
        original = getattr(Path, method)

        def guarded(path, *args, _original=original, **kwargs):
            reject_production(path)
            return _original(path, *args, **kwargs)

        monkeypatch.setattr(Path, method, guarded)

    records = tmp_path / "records"
    records.mkdir()
    receipt = records / "receipt.jsonl"
    _receipt(receipt, 0)
    index_path = tmp_path / "watch-index.sqlite3"
    command = [
        "agent",
        "ingest",
        "--once",
        "--no-retention",
        "--record",
        str(records),
        "--index",
        str(index_path),
    ]
    first = CliRunner().invoke(main, command)
    assert first.exit_code == 0, (first.output, first.exception)
    stored = list(snapshots.glob("*.zst"))
    assert len(stored) == 1
    assert _counts(index_path)[1] == 1

    receipt.unlink()
    index_path.unlink()
    second = CliRunner().invoke(main, command)
    assert second.exit_code == 0, (second.output, second.exception)
    assert _counts(index_path)[1] == 1
    assert list(snapshots.glob("*.zst")) == stored


def test_slow_snapshot_does_not_delay_ticks_or_start_another(tmp_path, monkeypatch):
    source = tmp_path / "receipts.jsonl"
    _receipt(source, 0)
    started = threading.Event()
    release = threading.Event()
    calls = []
    scheduled = []
    tick_times = []

    def slow_snapshot(index, *, directory):
        calls.append(threading.get_ident())
        started.set()
        assert release.wait(2)
        return directory / "snapshot.zst"

    monkeypatch.setattr(index_placement, "snapshot_index", slow_snapshot)
    with TelemetryIndex(tmp_path / "index.sqlite3") as index:
        index_placement.enable_wal(index)
        index.ingest([source])
        scheduler = index_placement.SnapshotScheduler(
            index.path, directory=tmp_path / "snapshots"
        )
        try:
            start = time.monotonic()
            reports = telemetry_ingest.run(
                index,
                tmp_path,
                pattern="*.jsonl",
                cadence=0.02,
                iterations=6,
                on_tick=lambda report: (
                    tick_times.append(time.monotonic()),
                    scheduled.append(scheduler.schedule()),
                ),
            )
            elapsed = time.monotonic() - start
            assert started.is_set()
            assert len(reports) == 6
            assert scheduled == [True, False, False, False, False, False]
            assert elapsed < 0.5
            assert (
                max(b - a for a, b in zip(tick_times, tick_times[1:], strict=False))
                < 0.15
            )
            assert len(calls) == 1
            assert calls[0] != threading.get_ident()
            assert index._conn.execute("SELECT count(*) FROM sample").fetchone()[0] == 1
        finally:
            release.set()
            scheduler.join()


def test_ingest_cli_ticks_continue_during_slow_snapshot(tmp_path, monkeypatch):
    release = threading.Event()
    calls = []
    tick_times = []
    report = TickReport(0, 0, 0, IngestReport(0, 0, 0, 0, 0), {})

    def slow_snapshot(index, *, directory=index_placement.SNAPSHOT_DIR):
        calls.append(threading.get_ident())
        release.wait(0.4)

    def fake_run(index, directory, *, on_tick, **kwargs):
        try:
            for _ in range(6):
                on_tick(report)
                tick_times.append(time.monotonic())
                time.sleep(0.02)
        finally:
            release.set()

    monkeypatch.setattr(index_placement, "restore_index", lambda path: None)
    monkeypatch.setattr(index_placement, "snapshot_if_due", slow_snapshot)
    monkeypatch.setattr(telemetry_ingest, "run", fake_run)
    result = CliRunner().invoke(
        main,
        [
            "agent",
            "ingest",
            "--once",
            "--no-retention",
            "--index",
            str(tmp_path / "index.sqlite3"),
            "--record",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(tick_times) == 6
    assert max(b - a for a, b in zip(tick_times, tick_times[1:], strict=False)) < 0.15
    assert calls == [calls[0]]
    assert calls[0] != threading.get_ident()


def test_scheduler_uses_its_own_connection_for_backup(tmp_path):
    source = tmp_path / "receipts.jsonl"
    _receipt(source, 0)
    snapshots = tmp_path / "snapshots"
    with TelemetryIndex(tmp_path / "index.sqlite3") as index:
        index_placement.enable_wal(index)
        index.ingest([source])
        scheduler = index_placement.SnapshotScheduler(index.path, directory=snapshots)
        assert scheduler.schedule()
        scheduler.join()
        assert index._conn.execute("SELECT count(*) FROM sample").fetchone()[0] == 1
    stored = list(snapshots.glob("*.zst"))
    assert len(stored) == 1
    restored = tmp_path / "restored.sqlite3"
    assert index_placement.restore_index(restored, directory=snapshots) == stored[0]
    assert _counts(restored) == _counts(tmp_path / "index.sqlite3")


def test_existing_snapshot_verifies_by_stream_after_restart(tmp_path, monkeypatch):
    source = tmp_path / "receipts.jsonl"
    _receipt(source, 0)
    with TelemetryIndex(tmp_path / "index.sqlite3") as index:
        index.ingest([source])
        stored = index_placement.snapshot_index(index, directory=tmp_path / "snapshots")
        receipt = index_placement._digest_path(stored)
        assert receipt.exists()
        index_placement._VERIFIED_SIGNATURES.clear()
        monkeypatch.setattr(
            index_placement,
            "_decompress_and_check",
            lambda source, destination: pytest.fail("unexpected uncompressed copy"),
        )
        assert (
            index_placement.snapshot_if_due(
                index.path, directory=tmp_path / "snapshots"
            )
            is None
        )


def test_changed_snapshot_digest_receipt_is_refused(tmp_path, capsys):
    source = tmp_path / "receipts.jsonl"
    _receipt(source, 0)
    with TelemetryIndex(tmp_path / "index.sqlite3") as index:
        index.ingest([source])
        stored = index_placement.snapshot_index(index, directory=tmp_path / "snapshots")
    index_placement._VERIFIED_SIGNATURES.clear()
    index_placement._digest_path(stored).write_text("0" * 64 + "\n")
    assert not index_placement._snapshot_is_verified(stored, tmp_path)
    assert "snapshot digest mismatch" in capsys.readouterr().err


def test_snapshot_checks_scratch_and_verifies_stream_without_second_copy(
    tmp_path, monkeypatch
):
    source = tmp_path / "receipts.jsonl"
    _receipt(source, 0)
    checks = []
    original_connect = sqlite3.connect

    class TracedConnection(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):
            checks.append(sql)
            return super().execute(sql, *args, **kwargs)

    def traced_connect(database, *args, **kwargs):
        if "backup.sqlite3" in str(database):
            kwargs["factory"] = TracedConnection
        return original_connect(database, *args, **kwargs)

    monkeypatch.setattr(index_placement.sqlite3, "connect", traced_connect)
    with TelemetryIndex(tmp_path / "index.sqlite3") as index:
        index.ingest([source])
        stored = index_placement.snapshot_index(index, directory=tmp_path / "snapshots")
    assert stored.exists()
    assert "PRAGMA journal_mode=OFF" in checks
    assert "PRAGMA synchronous=OFF" in checks
    assert "PRAGMA quick_check" in checks
    assert not list(tmp_path.glob("tmp*/backup.sqlite3"))
    assert not list(tmp_path.glob("tmp*/check.sqlite3"))


def test_stream_verification_refuses_corrupt_written_snapshot(tmp_path, monkeypatch):
    source = tmp_path / "receipts.jsonl"
    _receipt(source, 0)
    with TelemetryIndex(tmp_path / "index.sqlite3") as index:
        index.ingest([source])
        original_replace = index_placement.os.replace

        def corrupt_written_snapshot(staged, final):
            original_replace(staged, final)
            data = bytearray(final.read_bytes())
            data[-1] ^= 1
            final.write_bytes(data)

        monkeypatch.setattr(index_placement.os, "replace", corrupt_written_snapshot)
        with pytest.raises((ValueError, index_placement.zstd.ZstdError)):
            index_placement.snapshot_index(index, directory=tmp_path / "snapshots")

    assert not list((tmp_path / "snapshots").glob("*.zst"))
    assert not list(tmp_path.glob("tmp*/backup.sqlite3"))


def test_retention_keeps_three_verified_snapshots(tmp_path):
    source = tmp_path / "receipts.jsonl"
    _receipt(source, 0)
    snapshots = tmp_path / "snapshots"
    with TelemetryIndex(tmp_path / "index.sqlite3") as index:
        index.ingest([source])
        created = [
            index_placement.snapshot_index(
                index, directory=snapshots, now=datetime(2026, 10, day, tzinfo=UTC)
            )
            for day in range(1, 5)
        ]
    assert not created[0].exists()
    assert not index_placement._digest_path(created[0]).exists()
    assert all(path.exists() for path in created[1:])
    assert len(list(snapshots.glob("*.zst"))) == 3


def test_corrupt_newest_falls_back_and_keeps_older(tmp_path, capsys):
    source = tmp_path / "receipts.jsonl"
    _receipt(source, 0)
    snapshots = tmp_path / "snapshots"
    with TelemetryIndex(tmp_path / "index.sqlite3") as index:
        index.ingest([source])
        older = index_placement.snapshot_index(
            index, directory=snapshots, now=datetime(2026, 10, 1, tzinfo=UTC)
        )
        newest = index_placement.snapshot_index(
            index, directory=snapshots, now=datetime(2026, 10, 2, tzinfo=UTC)
        )
    newest.write_bytes(b"damaged compressed data")
    restored = tmp_path / "restored.sqlite3"
    assert index_placement.restore_index(restored, directory=snapshots) == older
    assert "snapshot restore failed" in capsys.readouterr().err
    assert older.exists()
    assert _counts(restored) == _counts(tmp_path / "index.sqlite3")


def test_corrupt_same_day_snapshot_does_not_suppress_replacement(tmp_path, capsys):
    source = tmp_path / "receipts.jsonl"
    _receipt(source, 0)
    snapshots = tmp_path / "snapshots"
    with TelemetryIndex(tmp_path / "index.sqlite3") as index:
        index.ingest([source])
        older = index_placement.snapshot_index(
            index, directory=snapshots, now=datetime.now(UTC) - timedelta(days=1)
        )
        damaged = index_placement.snapshot_index(index, directory=snapshots)
        assert index_placement.snapshot_if_due(index, directory=snapshots) is None
        damaged.write_bytes(b"damaged compressed data")
        replacement = index_placement.snapshot_if_due(index, directory=snapshots)

    assert replacement is not None and replacement != damaged
    assert replacement.exists() and older.exists()
    assert "snapshot verification failed" in capsys.readouterr().err
    restored = tmp_path / "restored.sqlite3"
    assert index_placement.restore_index(restored, directory=snapshots) == replacement
    assert _counts(restored) == _counts(tmp_path / "index.sqlite3")


def test_restored_index_catches_up_without_duplicate_rows(tmp_path):
    source = tmp_path / "receipts.jsonl"
    _receipt(source, 0)
    _receipt(source, 5)
    snapshots = tmp_path / "snapshots"
    continuous = tmp_path / "continuous.sqlite3"
    with TelemetryIndex(continuous) as index:
        index.ingest([source])
        index_placement.snapshot_index(index, directory=snapshots)
        _receipt(source, 10)
        assert index.ingest([source]).rows_inserted == 1
    restored = tmp_path / "restored.sqlite3"
    index_placement.restore_index(restored, directory=snapshots)
    with TelemetryIndex(restored) as index:
        index_placement.enable_wal(index)
        assert index.ingest([source]).rows_inserted == 1
        assert index.ingest([source]).rows_inserted == 0
    assert _counts(restored) == _counts(continuous)
    # A fresh process can verify source mtime later; the indexed rows agree.
    assert _logical_rows(restored) == _logical_rows(continuous)


def test_watch_cli_without_local_index_reads_published_stamp(tmp_path, monkeypatch):
    published = tmp_path / "watch.json"
    published.write_text('{"published_at":"2026-10-23T06:00:00Z","record":{}}')
    monkeypatch.setattr(watch, "PUBLISHED_WATCH_PATH", published)
    monkeypatch.delenv("AMBIX_WATCH_INDEX", raising=False)
    monkeypatch.delenv("FLEET_RUNTIME_DIR", raising=False)
    result = CliRunner().invoke(main, ["agent", "watch", "--json"])
    assert result.exit_code == 0, result.output
    assert result.output == published.read_text()
    result = CliRunner().invoke(main, ["agent", "watch"])
    assert result.exit_code == 0, result.output
    assert result.output == published.read_text()


def test_publisher_refuses_to_refresh_old_published_document(tmp_path, monkeypatch):
    published = tmp_path / "watch.json"
    original = b'{"published_at":"2026-10-23T06:00:00Z","record":{}}'
    published.write_bytes(original)
    monkeypatch.setattr(watch, "PUBLISHED_WATCH_PATH", published)
    monkeypatch.delenv("AMBIX_WATCH_INDEX", raising=False)
    monkeypatch.delenv("FLEET_RUNTIME_DIR", raising=False)

    line = watch_publish.one_iteration(published)

    assert "publish-watch failed: no local watch index" in line
    assert published.read_bytes() == original
