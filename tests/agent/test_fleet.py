"""Persistent interactive fleet allocation tests."""

from __future__ import annotations

import re
import subprocess

import pytest
from click.testing import CliRunner

from imas_ambix.agent import cli as cli_mod
from imas_ambix.agent import slurm as slurm_mod
from imas_ambix.agent.fleet import (
    FLEET_COMMENT,
    REMAINING_WARNING_SECONDS,
    find_fleet_allocation,
    generate_fleet_hold_script,
    node_is_draining,
    parse_node_state,
    placement_argv,
    remaining_seconds,
)
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


# squeue rows in the field order the CLI asks for:
# %i|%j|%T|%M|%R|%b|%k|%L
_SERVE_ROW = (
    "1275001|deepseek-v4-flash|RUNNING|3:00:00|gpu-node|gpu:h200:4|null|1-00:00:00"
)


def _squeue(
    rows: str,
    monkeypatch,
    *,
    node_state: str = "IDLE",
    node_returncode: int = 0,
) -> list[list[str]]:
    """Answer scheduler queries with fixed rows instead of a live queue.

    Returns every argv the CLI handed to the runner, so a test can assert the
    operands a query was written from rather than only the text it printed.
    """

    commands: list[list[str]] = []

    def fake_run(command, *args, **kwargs):
        commands.append(list(command))
        if command and command[0] == "scontrol":
            node = command[-1]
            stdout = (
                "" if node_returncode else f"NodeName={node}\n   State={node_state}\n"
            )
            return subprocess.CompletedProcess(command, node_returncode, stdout, "")
        return subprocess.CompletedProcess(command, 0, rows, "")

    monkeypatch.setattr(cli_mod.subprocess, "run", fake_run)
    return commands


# Two held allocations as the scheduler reports them on two different days.
# Everything except the identifier is identical, so an answer assembled from a
# remembered identifier is indistinguishable from a resolved one on a single
# row and only the pair of them separates the two.
HELD_IDENTIFIERS = ("1275000", "1289017")


def _fleet_row(time_left: str, jobid: str = "1275000") -> str:
    return (
        f"{jobid}|ambix-fleet|RUNNING|1:05:00|rigel-03|"
        f"billing=28,cpu=28,mem=120G,node=1|ambix-fleet|{time_left}"
    )


def test_fleet_status_reports_a_healthy_finite_allocation(monkeypatch) -> None:
    _squeue(f"{_fleet_row('2:30:00')}\n{_SERVE_ROW}\n", monkeypatch)
    result = CliRunner().invoke(main, ["agent", "fleet", "status"])

    assert result.exit_code == 0, result.output
    assert "1275000" in result.output
    assert "rigel-03" in result.output
    assert "RUNNING" in result.output
    assert "1:05:00" in result.output  # elapsed
    assert "2h30m" in result.output  # remaining, rendered
    assert "WARNING" not in result.output


def test_fleet_status_warns_inside_the_threshold(monkeypatch) -> None:
    _squeue(f"{_fleet_row('0:10:00')}\n", monkeypatch)
    result = CliRunner().invoke(main, ["agent", "fleet", "status"])

    assert result.exit_code == 0, result.output
    assert "WARNING" in result.output
    assert "10m" in result.output


def _wall_clock(seconds: int) -> str:
    """Render a second count the way ``squeue %L`` reports a finite limit."""
    days, rest = divmod(seconds, 86400)
    hours, rest = divmod(rest, 3600)
    minutes, secs = divmod(rest, 60)
    if days:
        return f"{days}-{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{hours}:{minutes:02d}:{secs:02d}"


def test_fleet_status_warning_tracks_the_configured_threshold(monkeypatch) -> None:
    """The boundary is derived from the constant, not asserted equal to it.

    Comparing the constant against the number it is defined as cannot fail, so
    it stops discriminating the moment the threshold moves. Both fixtures are
    derived from the constant instead, so a warning firing too late and one
    firing too early are each a failure.
    """
    inside = _wall_clock(max(1, REMAINING_WARNING_SECONDS - 60))
    outside = _wall_clock(REMAINING_WARNING_SECONDS + 60)

    _squeue(f"{_fleet_row(inside)}\n", monkeypatch)
    warned = CliRunner().invoke(main, ["agent", "fleet", "status"])
    assert warned.exit_code == 0, warned.output
    assert "WARNING" in warned.output

    _squeue(f"{_fleet_row(outside)}\n", monkeypatch)
    quiet = CliRunner().invoke(main, ["agent", "fleet", "status"])
    assert quiet.exit_code == 0, quiet.output
    assert "WARNING" not in quiet.output


def test_fleet_status_queries_the_account_the_allocation_is_charged_to(
    monkeypatch,
) -> None:
    """The queue is queried by the account the hold is charged to.

    The hold is billed to the fleet account, so a query filtered on the site's
    serving account cannot see it: the command then reports no allocation while
    the allocation is running.
    """
    site = SiteConfig.from_env()
    commands = _squeue(f"{_fleet_row('2:30:00')}\n", monkeypatch)
    result = CliRunner().invoke(main, ["agent", "fleet", "status"])

    assert result.exit_code == 0, result.output
    squeue = next(command for command in commands if command[0] == "squeue")
    queried = squeue[squeue.index("-A") + 1]
    assert queried == site.fleet_account
    assert queried != site.account
    assert "1275000" in result.output


def test_fleet_status_unbounded_allocation_warns_nothing(monkeypatch) -> None:
    """An allocation with no wall clock never warns on time, on a healthy node."""
    _squeue(f"{_fleet_row('UNLIMITED')}\n", monkeypatch, node_state="ALLOCATED")
    result = CliRunner().invoke(main, ["agent", "fleet", "status"])

    assert result.exit_code == 0, result.output
    assert "unbounded" in result.output
    assert "WARNING" not in result.output


def test_fleet_status_warns_when_the_allocation_node_is_draining(monkeypatch) -> None:
    """The unbounded hold is warned about by its node going out of service.

    This allocation carries no wall clock, so no time threshold can fire for
    it; the node's own scheduler state is the only advance notice that the
    allocation's control group is about to be torn down.
    """
    _squeue(f"{_fleet_row('UNLIMITED')}\n", monkeypatch, node_state="DRAINING")
    result = CliRunner().invoke(main, ["agent", "fleet", "status"])

    assert result.exit_code == 0, result.output
    assert "WARNING" in result.output
    assert "DRAINING" in result.output
    assert "rigel-03" in result.output


def test_fleet_status_warns_on_the_drain_spelling_this_cluster_reports(
    monkeypatch,
) -> None:
    """The drain token is not the leading one, and the warning still fires.

    A live read of ``scontrol show node -o`` on this cluster returned the two
    spellings below and no other draining form; a match restricted to the
    leading token warns on none of the 82 draining nodes.
    """
    spellings = (
        "MIXED+DRAIN+REBOOT_REQUESTED",
        "ALLOCATED+DRAIN+REBOOT_REQUESTED",
    )
    for state in spellings:
        _squeue(f"{_fleet_row('UNLIMITED')}\n", monkeypatch, node_state=state)
        result = CliRunner().invoke(main, ["agent", "fleet", "status"])

        assert result.exit_code == 0, result.output
        assert "WARNING" in result.output, state


def test_fleet_status_does_not_warn_on_a_healthy_node(monkeypatch) -> None:
    """A healthy node carrying another flag is not read as draining.

    The false-positive direction matters as much as the false-negative one: a
    node running work with a reservation is not going out of service.
    """
    for state in ("ALLOCATED", "IDLE", "MIXED", "MIXED+RESERVED"):
        _squeue(
            f"{_fleet_row('UNLIMITED')}\n", monkeypatch, node_state=state
        )
        result = CliRunner().invoke(main, ["agent", "fleet", "status"])

        assert result.exit_code == 0, result.output
        assert "WARNING" not in result.output, state


def test_fleet_status_reads_a_non_node_value_as_unknown(monkeypatch) -> None:
    """A pending job reports its reason where the node name goes.

    That value is not a node name, so it is not queried and reads as unknown —
    which is not warned about, and is not read as healthy either.
    """
    row = (
        "1275000|ambix-fleet|PENDING|0:00|(Priority)|"
        "billing=28,cpu=28,mem=120G,node=1|ambix-fleet|N/A"
    )
    commands = _squeue(f"{row}\n", monkeypatch)
    result = CliRunner().invoke(main, ["agent", "fleet", "status"])

    assert result.exit_code == 0, result.output
    assert "WARNING" not in result.output
    assert not any(
        command[0] == "scontrol" and len(command) > 3 for command in commands
    )


def test_fleet_status_reads_a_failed_node_query_as_unknown(monkeypatch) -> None:
    """A node query that fails reads as unknown, not as a healthy node."""
    commands = _squeue(
        f"{_fleet_row('UNLIMITED')}\n", monkeypatch, node_returncode=1)
    result = CliRunner().invoke(main, ["agent", "fleet", "status"])

    assert result.exit_code == 0, result.output
    assert "WARNING" not in result.output
    assert any(command[0] == "scontrol" and len(command) > 3 for command in commands)


def test_fleet_status_reads_the_allocation_node(monkeypatch) -> None:
    commands = _squeue(f"{_fleet_row('2:30:00')}\n", monkeypatch)
    result = CliRunner().invoke(main, ["agent", "fleet", "status"])

    assert result.exit_code == 0, result.output
    scontrol = next(command for command in commands if command[0] == "scontrol")
    assert scontrol[-1] == "rigel-03"


def test_parse_node_state_reads_the_state_field_and_keeps_the_set() -> None:
    """The parser keeps the whole token set rather than one of its members.

    The drain token sits among the others, so a parser that keeps only the
    leading token discards the very token the operator needs.
    """
    assert parse_node_state("NodeName=rigel-03\n   State=IDLE\n") == "IDLE"
    assert (
        parse_node_state("   State=MIXED+DRAIN+REBOOT_REQUESTED\n")
        == "MIXED+DRAIN+REBOOT_REQUESTED"
    )
    assert parse_node_state("NodeName=rigel-03\n") is None


def test_node_is_draining_matches_a_drain_token_anywhere_in_the_set() -> None:
    """Real scheduler spellings, as data, in both directions.

    The two spellings the live cluster reports carry the drain token second or
    third, so a leading-token match fires on none of the 82 draining nodes.
    """
    warned = (
        "DRAIN",
        "DRAINED",
        "DRAINING",
        "DRAINING+NOT_RESPONDING",
        "MIXED+DRAIN+REBOOT_REQUESTED",
        "ALLOCATED+DRAIN+REBOOT_REQUESTED",
    )
    quiet = (
        None,
        "",
        "ALLOCATED",
        "IDLE",
        "MIXED",
        "MIXED+RESERVED",
        "ALLOCATED+MAINTENANCE+RESERVED",
        "ALLOCATED+REBOOT_REQUESTED",
        "RESUME",
        "DOWN",
        "DOWN+NOT_RESPONDING",
    )
    for state in warned:
        assert node_is_draining(state), state
    for state in quiet:
        assert not node_is_draining(state), state


def test_remaining_seconds_removes_the_unbounded_token_from_arithmetic() -> None:
    # The unbounded token is not a duration, so it must not become a number
    # that a threshold comparison could act on.
    assert remaining_seconds("UNLIMITED") is None
    assert remaining_seconds("2:30:00") == 9000
    assert remaining_seconds("1-00:00:00") == 86400
    assert remaining_seconds("45:00") == 2700
    assert remaining_seconds("N/A") is None


def test_fleet_status_reports_no_allocation_in_words(monkeypatch) -> None:
    _squeue(f"{_SERVE_ROW}\n", monkeypatch)
    result = CliRunner().invoke(main, ["agent", "fleet", "status"])

    assert result.exit_code == 0, result.output
    assert "No fleet allocation is held" in result.output
    assert "1275001" not in result.output


def _place_runner(rows: str, monkeypatch, *, step_returncode: int = 0):
    """Answer the queue query with fixed rows and the placed step with a status.

    Returns every argv the CLI handed to the runner, so a test can assert the
    invocation the placement was written from rather than only what it printed
    — the step's own output goes to the inherited descriptors, not to the
    runner's capture buffer.
    """
    commands: list[list[str]] = []

    def fake_run(command, *args, **kwargs):
        commands.append(list(command))
        if command and command[0] == "srun":
            return subprocess.CompletedProcess(command, step_returncode, "", "")
        return subprocess.CompletedProcess(command, 0, rows, "")

    monkeypatch.setattr(cli_mod.subprocess, "run", fake_run)
    return commands


@pytest.mark.parametrize("jobid", HELD_IDENTIFIERS)
def test_placement_argv_uses_the_identifier_of_the_row_it_matched(jobid) -> None:
    """The invocation is built from the matched row, not from a memory of one.

    The rows carry different identifiers, and the identifier in the emitted
    invocation changes with the row that was handed in, so no identifier
    written into the source satisfies both cases.
    """
    serving = {"jobid": "1275001", "name": "deepseek-v4-flash", "comment": "null"}
    held = {"jobid": jobid, "name": "ambix-fleet", "comment": FLEET_COMMENT}

    matched = find_fleet_allocation([serving, held])

    assert matched is held
    assert placement_argv(matched, ("hostname",)) == [
        "srun",
        "--overlap",
        f"--jobid={jobid}",
        "hostname",
    ]


@pytest.mark.parametrize("jobid", HELD_IDENTIFIERS)
def test_fleet_place_names_the_allocation_it_found(jobid, monkeypatch) -> None:
    """The wrapped command is carried through unchanged behind the step.

    The job id in the step is the one from the row the allocation was found by,
    so the placement follows a resubmit or a cancel-and-rebind as well as it
    follows the allocation it was first written against.
    """
    commands = _place_runner(f"{_fleet_row('UNLIMITED', jobid=jobid)}\n", monkeypatch)
    result = CliRunner().invoke(
        main, ["agent", "fleet", "place", "--", "hostname"]
    )

    assert result.exit_code == 0, result.output
    step = next(command for command in commands if command[0] == "srun")
    assert step == ["srun", "--overlap", f"--jobid={jobid}", "hostname"]


def test_fleet_place_queries_the_account_the_allocation_is_charged_to(
    monkeypatch,
) -> None:
    """A held allocation is only findable under the account it is billed to."""
    site = SiteConfig.from_env()
    commands = _place_runner(f"{_fleet_row('UNLIMITED')}\n", monkeypatch)
    CliRunner().invoke(main, ["agent", "fleet", "place", "--", "hostname"])

    squeue = next(command for command in commands if command[0] == "squeue")
    assert squeue[squeue.index("-A") + 1] == site.fleet_account


def test_fleet_place_refuses_when_no_allocation_is_held(monkeypatch) -> None:
    """No allocation is a refusal, never a launch on the login node.

    A worker that reaches the login node instead of the held node runs under
    the very ceiling the placement exists to escape, and reports as placed.
    """
    commands = _place_runner(f"{_SERVE_ROW}\n", monkeypatch)
    result = CliRunner().invoke(
        main, ["agent", "fleet", "place", "--", "hostname"]
    )

    assert result.exit_code != 0, result.output
    assert "No fleet allocation is held" in result.output
    assert not any(command[0] == "srun" for command in commands)


def test_fleet_place_propagates_the_step_exit_status(monkeypatch) -> None:
    """A step that failed is not reported as a placement that succeeded."""
    _place_runner(
        f"{_fleet_row('UNLIMITED')}\n", monkeypatch, step_returncode=3
    )
    result = CliRunner().invoke(
        main, ["agent", "fleet", "place", "--", "hostname"]
    )

    assert result.exit_code == 3, result.output
