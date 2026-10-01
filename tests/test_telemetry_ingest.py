"""The standing ingest appends the record once and re-reads only what changed.

Every assertion here is about what a tick *did* -- the rows it added and the
filesystem calls it made -- rather than what the index happens to hold. A second
tick over an unchanged record is the case that matters, because getting it wrong
costs a network filesystem read of every source on every cadence.
"""

from __future__ import annotations

import datetime as _dt
import json
import sqlite3
import weakref
from pathlib import Path

import pytest

from imas_ambix.agent.telemetry_index import TelemetryIndex, discover
from imas_ambix.agent.telemetry_ingest import (
    DEFAULT_CADENCE_SECONDS,
    SourceFingerprint,
    main,
    pending_sources,
    run,
    tick,
)

_BASE = _dt.datetime(2026, 9, 20, 6, 0, 0, tzinfo=_dt.UTC)

#: Mode characters that make an open a writing open. A read mode carries none of
#: them, so the check runs over the whole mode string, not its first letter.
_WRITING_MODE_CHARS = ("w", "a", "x", "+")

#: Record-file suffixes the ingest reads. A write open of one of these is the
#: ingest writing the record the index is meant only to derive from.
_RECORD_SUFFIXES = (".jsonl", ".gz")


def _at(seconds: float) -> float:
    return _BASE.timestamp() + seconds


def _row(seconds: float, **overrides: object) -> dict:
    """One recorder-shaped sample with a gauge and a cumulative counter."""
    row = {
        "timestamp": (_BASE + _dt.timedelta(seconds=seconds)).isoformat(),
        "job_id": "1277001",
        "profile_slug": "deepseek-v4-1-flash",
        "gpus": 4,
        "num_requests_running": 12,
        "engine": {"family": "sglang", "generation_tokens": 1000.0 + seconds},
    }
    row.update(overrides)
    return row


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _record_opens(
    monkeypatch: pytest.MonkeyPatch, sink: list[tuple[Path, str]]
) -> None:
    real_open = Path.open

    def counting(self: Path, mode: str = "r", *args: object, **kwargs: object):
        sink.append((Path(self), mode))
        return real_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counting)


def _record_writes(opened: list[tuple[Path, str]]) -> list[tuple[Path, str]]:
    return [
        (path, mode)
        for path, mode in opened
        if path.suffix in _RECORD_SUFFIXES
        and any(char in mode for char in _WRITING_MODE_CHARS)
    ]


def test_tick_appends_every_row_and_an_immediate_second_tick_appends_none(tmp_path):
    _write(tmp_path / "serve-1001.jsonl", [_row(0), _row(5)])
    _write(tmp_path / "serve-1002.jsonl", [_row(1)])

    with TelemetryIndex(tmp_path / "index.db") as index:
        opening = tick(index, tmp_path)

        assert opening.files_discovered == 2
        assert opening.files_read == 2
        assert opening.ingest.rows_inserted == 3
        assert index.sample_count() == 3

        again = tick(index, tmp_path, previous=opening.sources)

        assert again.files_read == 0
        assert again.files_skipped == 2
        assert again.ingest.rows_inserted == 0
        assert index.sample_count() == 3


def test_a_tick_opens_no_source_unchanged_since_the_last_tick(tmp_path, monkeypatch):
    source = tmp_path / "serve-1001.jsonl"
    _write(source, [_row(0), _row(5)])

    with TelemetryIndex(tmp_path / "index.db") as index:
        opening = tick(index, tmp_path)
        assert opening.ingest.rows_inserted == 2

        opened: list[tuple[Path, str]] = []
        _record_opens(monkeypatch, opened)
        again = tick(index, tmp_path, previous=opening.sources)

        assert again.files_read == 0
        assert [path for path, _ in opened if path == source] == []


def test_the_next_tick_reads_only_the_source_that_grew(tmp_path):
    grown = tmp_path / "serve-1001.jsonl"
    settled = tmp_path / "serve-1002.jsonl"
    _write(grown, [_row(0)])
    _write(settled, [_row(1)])

    with TelemetryIndex(tmp_path / "index.db") as index:
        opening = tick(index, tmp_path)
        _write(grown, [_row(5)])
        later = tick(index, tmp_path, previous=opening.sources)

        assert later.files_read == 1
        assert later.files_skipped == 1
        assert later.ingest.rows_inserted == 1
        assert index.sample_count() == 3


def test_pending_sources_tracks_metadata_rather_than_contents(tmp_path):
    source = tmp_path / "serve-1001.jsonl"
    _write(source, [_row(0)])

    first, state = pending_sources(tmp_path)
    assert [path.name for path in first] == ["serve-1001.jsonl"]

    second, refreshed = pending_sources(tmp_path, state)
    assert second == []
    assert refreshed == state

    _write(source, [_row(5)])
    third, _ = pending_sources(tmp_path, refreshed)
    assert [path.name for path in third] == ["serve-1001.jsonl"]


def test_a_fingerprint_is_the_inode_size_and_modification_time(tmp_path):
    source = tmp_path / "serve-1001.jsonl"
    _write(source, [_row(0)])
    stat = source.stat()

    assert SourceFingerprint.of(source) == SourceFingerprint(
        inode=stat.st_ino, size=stat.st_size, mtime_ns=stat.st_mtime_ns
    )


def test_the_index_rebuilt_from_the_receipts_equals_the_appended_index(tmp_path):
    _write(tmp_path / "serve-1001.jsonl", [_row(0), _row(5)])
    _write(tmp_path / "serve-1002.jsonl", [_row(1)])

    with (
        TelemetryIndex(tmp_path / "appended.db") as appended,
        TelemetryIndex(tmp_path / "rebuilt.db") as rebuilt,
    ):
        tick(appended, tmp_path)
        rebuilt.rebuild(discover(tmp_path))

        window = (_at(0), _at(60))
        appended_rows = appended.rows(*window)
        assert appended_rows
        assert appended_rows == rebuilt.rows(*window)
        assert appended.sample_count() == rebuilt.sample_count()


def test_the_ingest_opens_no_receipt_file_for_writing(tmp_path, monkeypatch):
    source = tmp_path / "serve-1001.jsonl"
    _write(source, [_row(0), _row(5)])

    opened: list[tuple[Path, str]] = []
    _record_opens(monkeypatch, opened)
    with TelemetryIndex(tmp_path / "index.db") as index:
        reports = run(index, tmp_path, cadence=0.01, iterations=2, sleep=lambda _: None)

    assert [report.ingest.rows_inserted for report in reports] == [2, 0]
    # The record was read, so the write check below is not vacuous.
    assert any(path == source and mode.startswith("r") for path, mode in opened)
    assert _record_writes(opened) == []


def test_the_ingest_never_opens_the_reader_cache_index(tmp_path, monkeypatch):
    _write(tmp_path / "serve-1001.jsonl", [_row(0)])
    reader_index = Path.home() / ".cache" / "ambix" / "watch-index.sqlite3"

    opened: list[tuple[Path, str]] = []
    _record_opens(monkeypatch, opened)
    with TelemetryIndex(tmp_path / "index.db") as index:
        tick(index, tmp_path)

    assert reader_index not in {path for path, _ in opened}


def test_run_ticks_on_its_cadence_and_carries_the_state_between_ticks(tmp_path):
    _write(tmp_path / "serve-1001.jsonl", [_row(0)])
    slept: list[float] = []

    with TelemetryIndex(tmp_path / "index.db") as index:
        reports = run(index, tmp_path, cadence=2.5, iterations=3, sleep=slept.append)

    assert [report.ingest.rows_inserted for report in reports] == [1, 0, 0]
    assert slept == [2.5, 2.5]


def test_a_non_positive_cadence_is_refused(tmp_path):
    with TelemetryIndex(tmp_path / "index.db") as index, pytest.raises(ValueError):
        run(index, tmp_path, cadence=0.0, iterations=1, sleep=lambda _: None)


def test_the_default_cadence_is_sixty_seconds(tmp_path):
    """The standing loop ticks on the locked sixty-second default.

    The default is a module constant rather than a literal at the call site, so
    the value is asserted where the loop reads it and the run below proves the
    constant is what a caller that names no cadence actually gets.
    """
    assert DEFAULT_CADENCE_SECONDS == 60.0

    _write(tmp_path / "serve-1001.jsonl", [_row(0)])
    slept: list[float] = []
    with TelemetryIndex(tmp_path / "index.db") as index:
        run(index, tmp_path, iterations=3, sleep=slept.append)

    assert slept == [60.0, 60.0]


def test_main_takes_a_single_tick_when_asked(tmp_path):
    _write(tmp_path / "serve-1001.jsonl", [_row(0)])
    index_file = tmp_path / "index.db"

    assert main(["--record", str(tmp_path), "--index", str(index_file), "--once"]) == 0
    with TelemetryIndex(index_file) as index:
        assert index.sample_count() == 1


def test_a_tick_offers_the_whole_directory_but_rereads_only_what_changed(
    tmp_path, monkeypatch
):
    """The index is offered every file so a job's tier choice sees them all.

    Offering only what moved would leave a job's tier selection blind to a
    sibling that did not change -- a compaction appearing beside the raw file,
    or the raw file's deletion leaving the minute tier as the finest that
    remains -- so the whole set is handed over every tick. What the metadata
    decision buys is that an unchanged source is never re-read, which is the
    cost the schedule is measured in.
    """
    grown = tmp_path / "serve-1001.jsonl"
    settled = tmp_path / "serve-1002.jsonl"
    _write(grown, [_row(0)])
    _write(settled, [_row(1)])

    handed: list[list[Path]] = []
    real_ingest = TelemetryIndex.ingest

    def spy(self, sources):
        handed.append([Path(source) for source in sources])
        return real_ingest(self, sources)

    monkeypatch.setattr(TelemetryIndex, "ingest", spy)

    with TelemetryIndex(tmp_path / "index.db") as index:
        opening = tick(index, tmp_path)
        assert set(handed[0]) == {grown, settled}

        _write(grown, [_row(5)])
        later = tick(index, tmp_path, previous=opening.sources)

        # The whole directory is offered on the second tick too, so the pass
        # that reads one source is not a pass that saw only that source.
        assert set(handed[1]) == {grown, settled}
        # Only the source whose metadata moved is re-read; the settled one is
        # handed over and left unopened, which the index's own count confirms.
        assert later.files_read == 1
        assert later.ingest.files_scanned == 1
        assert later.ingest.rows_inserted == 1


def test_the_ingest_never_connects_the_reader_cache_index(tmp_path, monkeypatch):
    _write(tmp_path / "serve-1001.jsonl", [_row(0)])
    reader_index = Path.home() / ".cache" / "ambix" / "watch-index.sqlite3"

    connected: list[str] = []
    real_connect = sqlite3.connect

    def spy(address, *args: object, **kwargs: object):
        connected.append(str(address))
        return real_connect(address, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", spy)

    with TelemetryIndex(tmp_path / "index.db") as index:
        tick(index, tmp_path)

    # The spy has to see the index this ingest legitimately opens, or the
    # absence below is a spy that never ran rather than a path never touched.
    assert connected
    assert str(reader_index) not in connected


def test_a_loop_without_iterations_keeps_at_most_the_latest_report(tmp_path):
    _write(tmp_path / "serve-1001.jsonl", [_row(0)])
    target_ticks = 1000
    references: list[weakref.ReferenceType] = []
    live: list[int] = []

    class _EnoughError(Exception):
        pass

    def note(report):
        references.append(weakref.ref(report))

    def stop_sleep(_seconds):
        live.append(sum(1 for reference in references if reference() is not None))
        if len(references) >= target_ticks:
            raise _EnoughError

    with TelemetryIndex(tmp_path / "index.db") as index, pytest.raises(_EnoughError):
        run(index, tmp_path, sleep=stop_sleep, on_tick=note)

    assert len(live) == target_ticks
    # Only the latest report stays strongly reachable. A list that grew with the
    # loop would keep every report alive and push this figure to the tick count.
    assert max(live) <= 1


def test_zero_iterations_takes_no_tick(tmp_path):
    _write(tmp_path / "serve-1001.jsonl", [_row(0)])
    slept: list[float] = []

    with TelemetryIndex(tmp_path / "index.db") as index:
        reports = run(index, tmp_path, iterations=0, sleep=slept.append)

        assert reports == []
        assert index.sample_count() == 0

    assert slept == []
