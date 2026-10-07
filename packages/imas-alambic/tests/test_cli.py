"""The engine's command line exposes one command, ``write``.

The command writes a shot's description IDSs, so its verb is ``write``.  This
test reads the group's own listing rather than spelling the retired verb: the
group must register exactly one command, named ``write``, and that command must
carry ``--machine``, ``--shot`` and ``--out``.
"""

from __future__ import annotations

from click.testing import CliRunner

from imas_alambic.cli import main


def test_help_lists_the_write_command_and_its_options():
    group = CliRunner().invoke(main, ["--help"])
    assert group.exit_code == 0, group.output
    assert "write" in group.output
    assert sorted(main.commands) == ["write"]

    command = CliRunner().invoke(main, ["write", "--help"])
    assert command.exit_code == 0, command.output
    for option in ("--machine", "--shot", "--out"):
        assert option in command.output
