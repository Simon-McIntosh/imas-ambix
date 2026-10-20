"""Overlap restart controls without scheduler or live-lane mutations."""

from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from imas_ambix.agent import cli
from imas_ambix.cli import main


@pytest.mark.parametrize("cancel_all", [False, True])
def test_shutdown_refuses_overlap_unless_all(monkeypatch, cancel_all):
    jobs = [{"jobid": "101", "name": "sample"}, {"jobid": "102", "name": "sample"}]
    calls = []

    def command(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(
            returncode=0, stdout="101|sample\n102|sample\n", stderr=""
        )

    monkeypatch.setattr(cli, "_resolve_slug", lambda slug: slug)
    monkeypatch.setattr(cli, "_profile_jobs", lambda *_: jobs)
    monkeypatch.setattr(cli.subprocess, "run", command)
    monkeypatch.setattr(cli, "_remove_cancelled_registrations", lambda *_: None)
    monkeypatch.setattr(cli, "_republish_endpoint_document", lambda *_: None)
    result = CliRunner().invoke(
        main,
        ["agent", "shutdown", "sample", "--yes"] + (["--all"] if cancel_all else []),
    )
    if cancel_all:
        assert result.exit_code == 0, result.output
        assert ["scancel", "101", "102"] in calls
    else:
        assert result.exit_code != 0
        assert "101" in result.output and "102" in result.output
        assert "overlap restart" in result.output
        assert not any(args[0] == "scancel" for args in calls)
