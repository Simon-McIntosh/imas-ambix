"""``imas-ambix agent ingest`` runs the standing ingest service and submits it.

The ingest is a standing service rather than a scheduled or finite job: it keeps
the telemetry index current from the recorded receipts for as long as it runs, so
the tests here pin the unlimited walltime it is placed with rather than a finite
limit. Each test drives the CLI and observes the seam the command crosses -- the
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
    """The dry-run script carries the standing-service placement, never submits."""
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
    # The standing supporting services' shared walltime, not a finite one-hour
    # limit that would end the service on a schedule nobody is watching.
    assert (
        f"#SBATCH --time={slurm_mod._SUPPORTING_SERVICE_TIME_LIMIT}" in result.output
    )
    # The unlimited walltime is pinned by its literal, not only against the
    # shared constant, so a change that moved the constant would move the
    # service's placement with it unnoticed.
    assert "#SBATCH --time=0" in result.output
    assert "#SBATCH --time=01:00:00" not in result.output
    # The exec line names the absolute record and index the submitter chose, so
    # a reader auditing the job reads the target off the script.
    record_abs = str((tmp_path / "receipts").resolve())
    index_abs = str((tmp_path / "watch-index.sqlite3").resolve())
    exec_line = next(
        line for line in result.output.splitlines() if line.startswith("exec ")
    )
    assert f"--record {record_abs}" in exec_line
    assert f"--index {index_abs}" in exec_line


def test_relative_paths_resolve_against_the_scratch_directory(tmp_path, monkeypatch):
    """A relative --record/--index is emitted as the absolute path it resolves to."""
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        main,
        [
            "agent",
            "ingest",
            "--dry-run",
            "--record",
            "receipts",
            "--index",
            "watch-index.sqlite3",
        ],
    )

    assert result.exit_code == 0
    exec_line = next(
        line for line in result.output.splitlines() if line.startswith("exec ")
    )
    assert f"--record {(tmp_path / 'receipts').resolve()}" in exec_line
    assert f"--index {(tmp_path / 'watch-index.sqlite3').resolve()}" in exec_line
    # The bare relative path must not reach the script unresolved.
    assert "--record receipts" not in exec_line


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
