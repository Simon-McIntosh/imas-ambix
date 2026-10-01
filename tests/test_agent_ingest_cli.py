"""``imas-ambix agent ingest`` runs the standing ingest the fleet supervisor keeps.

The ingest is a standing loop, not a scheduled or finite job: it keeps the
telemetry index current from the recorded receipts for as long as it runs, so
the tests here pin that it offers no way to submit itself as its own job and no
way to print such a job's script. Each test drives the CLI and observes the seam
the command crosses -- the ingest loop it calls -- rather than what the index
happens to hold.
"""

from __future__ import annotations

from click.testing import CliRunner

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
