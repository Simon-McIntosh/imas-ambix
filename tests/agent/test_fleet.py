"""Persistent interactive fleet allocation tests."""

from __future__ import annotations

import re

from click.testing import CliRunner

from imas_ambix.agent import slurm as slurm_mod
from imas_ambix.agent.fleet import generate_fleet_hold_script
from imas_ambix.agent.profile import SiteConfig
from imas_ambix.cli import main

# The scheduler writes a finite wall clock as digits, colons and an optional
# day separator. Anything else is an unbounded marker, so this matches the
# property "the limit expires" instead of one accepted spelling of "it does
# not" -- a spelling-equality check would reject an equally unbounded token and
# pass a spelling the scheduler refuses.
_FINITE_WALL_CLOCK = re.compile(r"[0-9][0-9:\-]*")


def _directives(script: str) -> list[str]:
    return [line for line in script.splitlines() if line.startswith("#SBATCH")]


def _directive_value(directives: list[str], name: str) -> str | None:
    prefix = f"#SBATCH --{name}="
    for line in directives:
        if line.startswith(prefix):
            return line[len(prefix) :]
    return None


def test_fleet_hold_script_requests_unbounded_whole_node() -> None:
    site = SiteConfig()
    script = generate_fleet_hold_script(site)
    directives = _directives(script)

    assert _directive_value(directives, "partition") == site.fleet_partition
    assert _directive_value(directives, "account") == site.fleet_account
    assert _directive_value(directives, "cpus-per-task") == str(site.fleet_cpus)
    assert _directive_value(directives, "mem") == site.fleet_memory
    assert "#SBATCH --nodes=1" in directives
    assert "#SBATCH --exclusive" in directives
    assert "#SBATCH --comment=ambix-fleet" in directives
    assert "export TMPDIR=/tmp" in script
    assert "exec sleep infinity" in script

    # A debug partition is finite by construction, so the fleet must not land
    # on one.
    partition = _directive_value(directives, "partition")
    assert partition is not None
    assert not partition.endswith("_debug")


def test_fleet_hold_script_requests_no_finite_wall_clock() -> None:
    directives = _directives(generate_fleet_hold_script(SiteConfig()))
    limits = [line for line in directives if line.startswith("#SBATCH --time=")]

    assert len(limits) <= 1
    for line in limits:
        value = line.split("=", 1)[1]
        assert not _FINITE_WALL_CLOCK.fullmatch(value), (
            f"fleet allocation requests a finite wall clock: {value!r}"
        )


def test_fleet_hold_script_requests_no_zero_memory() -> None:
    directives = _directives(generate_fleet_hold_script(SiteConfig()))

    # A zero memory request reserves the whole node and cannot start beside
    # anything else, so it is never the intended spelling of "no ceiling".
    assert _directive_value(directives, "mem") != "0"


def test_fleet_hold_prints_without_submitting(monkeypatch) -> None:
    def unexpected_submit(_script: str) -> str:
        raise AssertionError("printing the script must not submit a job")

    monkeypatch.setattr(slurm_mod, "submit_script", unexpected_submit)
    result = CliRunner().invoke(main, ["agent", "fleet", "hold"])

    assert result.exit_code == 0, result.output
    assert "#SBATCH --comment=ambix-fleet" in result.output
    assert "#SBATCH --nodes=1" in result.output


def test_fleet_hold_submits_the_generated_script_once(monkeypatch) -> None:
    submitted: list[str] = []

    def submit(script: str) -> str:
        submitted.append(script)
        return "1275000"

    monkeypatch.setattr(slurm_mod, "submit_script", submit)
    result = CliRunner().invoke(main, ["agent", "fleet", "hold", "--submit"])

    assert result.exit_code == 0, result.output
    assert submitted == [generate_fleet_hold_script(SiteConfig.from_env())]
    assert "1275000" in result.output
