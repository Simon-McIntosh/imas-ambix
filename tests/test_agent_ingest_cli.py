"""``imas-ambix agent ingest`` runs the standing ingest the fleet supervisor keeps.

The ingest is a standing loop, not a scheduled or finite job: it keeps the
telemetry index current from the recorded receipts for as long as it runs, so
the tests here pin that it offers no way to submit itself as its own job and no
way to print such a job's script. Each test drives the CLI and observes the seam
the command crosses -- the ingest loop it calls -- rather than what the index
happens to hold.
"""

from __future__ import annotations

import types
from datetime import UTC, datetime

from click.testing import CliRunner

from imas_ambix.agent.telemetry_index import IngestReport
from imas_ambix.agent.telemetry_ingest import TickReport
from imas_ambix.cli import main


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
    monkeypatch.setattr(cli, "_utc_now", lambda: FROZEN_TICK_TIME)

    result = CliRunner().invoke(
        main, ["agent", "ingest", "--once", *_ingest_args(tmp_path)]
    )

    assert result.exit_code == 0, result.output
    assert result.output.splitlines() == [
        f"{FROZEN_TICK_STAMP} ingest tick discovered=118 read=3 skipped=115 "
        "scanned=3 inserted=240 duplicate=0 malformed=0 bytes=51234",
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
        (
            "write",
            f"{FROZEN_TICK_STAMP} ingest tick discovered=120 read=2 skipped=118 "
            "scanned=1 inserted=7 duplicate=5 malformed=1 bytes=2048\n",
        ),
        ("flush", ""),
    ]
