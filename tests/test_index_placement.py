"""The watch index remains local while verified snapshots survive on shared storage."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from click.testing import CliRunner

from imas_ambix.agent import index_placement, watch
from imas_ambix.agent.telemetry_index import TelemetryIndex
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
    assert index_placement.restore_index(restored, directory=snapshots) == stored
    assert _counts(origin) == _counts(restored)
    assert _digest(origin) == _digest(restored)


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
    assert _digest(restored) == _digest(continuous)


def test_watch_without_local_index_reads_published_stamp(tmp_path, monkeypatch):
    published = tmp_path / "watch.json"
    published.write_text('{"published_at":"2026-10-23T06:00:00Z","record":{}}')
    monkeypatch.setattr(watch, "PUBLISHED_WATCH_PATH", published)
    monkeypatch.delenv("AMBIX_WATCH_INDEX", raising=False)
    monkeypatch.delenv("FLEET_RUNTIME_DIR", raising=False)
    assert watch.watch_document()["published_at"] == "2026-10-23T06:00:00Z"
    result = CliRunner().invoke(main, ["agent", "watch", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["published_at"] == "2026-10-23T06:00:00Z"
