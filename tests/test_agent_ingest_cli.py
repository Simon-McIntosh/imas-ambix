"""``imas-ambix agent ingest`` runs the scheduled ingest and submits its service.

Each test drives the CLI and observes the seam the command crosses -- the
ingest loop it calls, or the SLURM script it hands to ``sbatch`` -- rather than
what the index happens to hold. Nothing here submits a job: the one test that
exercises ``--submit`` replaces the submitter with a stub, so a real ``sbatch``
can never run under a test.
"""

from __future__ import annotations

from click.testing import CliRunner

from imas_ambix.agent import slurm as slurm_mod
from imas_ambix.cli import main


def _ingest_args(tmp_path) -> list[str]:
    """A record and index inside the temporary test tree, so a real run is safe."""
    return [
        "--record",
        str(tmp_path / "receipts"),
        "--index",
        str(tmp_path / "watch-index.sqlite3"),
    ]


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
    # The default cadence is the recorder's own sampling interval; --once pays
    # nothing for it because the loop never sleeps past the last tick.
    assert calls[0][2]["cadence"] == telemetry_ingest.DEFAULT_CADENCE_SECONDS == 5.0


def test_cadence_overrides_the_default(tmp_path, monkeypatch):
    """``--cadence`` reaches the loop instead of the five-second default."""
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


def test_dry_run_prints_the_service_script_without_submitting(tmp_path, monkeypatch):
    """The dry-run script carries the finite placement and never submits."""
    submitted: list[str] = []

    def refuse(script: str) -> str:
        submitted.append(script)
        raise AssertionError("dry-run must not submit")

    monkeypatch.setattr(slurm_mod, "submit_script", refuse)

    result = CliRunner().invoke(
        main, ["agent", "ingest", "--dry-run", *_ingest_args(tmp_path)]
    )

    assert result.exit_code == 0
    assert submitted == []
    assert "#SBATCH --cpus-per-task=1" in result.output
    assert "#SBATCH --mem=4G" in result.output
    assert "#SBATCH --comment=ambix-ingest" in result.output
    assert "#SBATCH --gres=gpu" not in result.output
    # A finite walltime, not the standing services' unlimited --time=0.
    assert "#SBATCH --time=01:00:00" in result.output
    assert "#SBATCH --time=0\n" not in result.output


def test_submit_hands_the_script_to_sbatch_through_a_stub(tmp_path, monkeypatch):
    """``--submit`` submits the generated script, with no real sbatch under test."""
    seen: list[str] = []

    def fake_submit(script: str) -> str:
        seen.append(script)
        return "1277001"

    monkeypatch.setattr(slurm_mod, "submit_script", fake_submit)

    result = CliRunner().invoke(
        main, ["agent", "ingest", "--submit", *_ingest_args(tmp_path)]
    )

    assert result.exit_code == 0
    assert len(seen) == 1
    assert "#SBATCH --comment=ambix-ingest" in seen[0]
    assert "1277001" in result.output
