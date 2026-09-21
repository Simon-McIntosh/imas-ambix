"""Persistent interactive fleet allocation tests."""

from __future__ import annotations

from click.testing import CliRunner

from imas_ambix.agent import slurm as slurm_mod
from imas_ambix.agent.fleet import generate_fleet_hold_script
from imas_ambix.cli import main


def test_fleet_hold_script_requests_bounded_whole_rigel_node() -> None:
    script = generate_fleet_hold_script()
    directives = [line for line in script.splitlines() if line.startswith("#SBATCH")]

    assert "#SBATCH --partition=rigel" in directives
    assert "#SBATCH --account=iter" in directives
    assert "#SBATCH --nodes=1" in directives
    assert "#SBATCH --exclusive" in directives
    assert "#SBATCH --cpus-per-task=28" in directives
    assert "#SBATCH --mem=120G" in directives
    assert "#SBATCH --time=UNLIMITED" in directives
    assert "#SBATCH --comment=ambix-fleet" in directives
    assert "export TMPDIR=/tmp" in script
    assert "exec sleep infinity" in script

    assert not any(line.endswith("_debug") for line in directives)
    assert not any(
        line.startswith("#SBATCH --time=") and line != "#SBATCH --time=UNLIMITED"
        for line in directives
    )
    assert "#SBATCH --mem=0" not in directives


def test_fleet_hold_prints_without_submitting(monkeypatch) -> None:
    def unexpected_submit(_script: str) -> str:
        raise AssertionError("printing the script must not submit a job")

    monkeypatch.setattr(slurm_mod, "submit_script", unexpected_submit)
    result = CliRunner().invoke(main, ["agent", "fleet", "hold"])

    assert result.exit_code == 0, result.output
    assert "#SBATCH --comment=ambix-fleet" in result.output
    assert "#SBATCH --time=UNLIMITED" in result.output


def test_fleet_hold_submits_only_with_explicit_flag(monkeypatch) -> None:
    submitted: list[str] = []

    def submit(script: str) -> str:
        submitted.append(script)
        return "1275000"

    monkeypatch.setattr(slurm_mod, "submit_script", submit)
    result = CliRunner().invoke(main, ["agent", "fleet", "hold", "--submit"])

    assert result.exit_code == 0, result.output
    assert result.output == "Submitted fleet allocation job 1275000.\n"
    assert len(submitted) == 1
    assert submitted[0] == generate_fleet_hold_script()
