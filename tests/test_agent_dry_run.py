"""A dry run must reach neither the scheduler nor anything it names.

A dry run is what an operator reaches for when unsure, so a ``--dry-run`` that
still cancels a live serving lane, submits a replacement job, or rewrites a
published document turns the safe option into the dangerous one. Each test here
runs a dry run against a live-looking site -- a matching scheduler job, a
published endpoint document and a serve registration -- and asserts that no
cancelling or submitting verb was reached and that no file on disk changed.

The scheduler is replaced wholesale rather than stubbed per call, so a dry run
that reaches ``scancel`` or ``sbatch`` through any path aborts loudly instead of
being mistaken for a pass.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import click
import pytest
from click.testing import CliRunner

from imas_ambix.cli import main

_PROFILE = "deepseek-v4-flash"
_JOB_ID = "424242"

# Scheduler verbs that cancel, submit or place work. A dry run reaches none.
_DESTRUCTIVE_BINARIES = {"scancel", "sbatch", "srun", "scontrol", "systemctl"}


class _SchedulerRecorder:
    """Stand in for the ``subprocess`` module, refusing destructive verbs.

    A dry run is allowed to ask the scheduler what is running -- that is how it
    reports what a real run would stop -- so ``squeue`` answers from a canned
    listing. Every other verb aborts the command.
    """

    def __init__(self, squeue_stdout: str = "") -> None:
        self.argv: list[list[str]] = []
        self._squeue_stdout = squeue_stdout

    def run(self, argv, **kwargs):  # noqa: ANN001, ANN003 - subprocess.run shape
        argv = list(argv)
        self.argv.append(argv)
        assert argv[0] not in _DESTRUCTIVE_BINARIES, (
            f"a dry run issued a destructive command: {argv}"
        )
        if argv[0] == "squeue":
            return subprocess.CompletedProcess(argv, 0, self._squeue_stdout, "")
        raise AssertionError(f"a dry run issued an unexpected command: {argv}")

    @property
    def destructive(self) -> list[list[str]]:
        return [argv for argv in self.argv if argv[0] in _DESTRUCTIVE_BINARIES]


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@pytest.fixture
def scene(tmp_path, monkeypatch):
    """A live-looking site: a running serve job, endpoint document, registration."""
    endpoint = tmp_path / "public" / "endpoints.json"
    endpoint.parent.mkdir(parents=True)
    endpoint.write_text(
        json.dumps({"endpoints": []}, indent=2) + "\n", encoding="utf-8"
    )

    base = tmp_path / "base"
    registry = base / "agents" / "serve-registry"
    registry.mkdir(parents=True)
    (registry / f"{_JOB_ID}.json").write_text(
        json.dumps(
            {
                "job_id": _JOB_ID,
                "slug": _PROFILE,
                "host": "98dci4-gpu-0003",
                "port": 18800,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("AMBIX_AGENT_ENDPOINT_DOCUMENT", str(endpoint))
    monkeypatch.setenv("AMBIX_AGENT_BASE_DIR", str(base))
    monkeypatch.setenv("USER", "operator")
    monkeypatch.delenv("AMBIX_AGENT_API_KEY", raising=False)

    recorder = _SchedulerRecorder(squeue_stdout=f"{_JOB_ID}|{_PROFILE}\n")
    monkeypatch.setattr("imas_ambix.agent.cli.subprocess", recorder)
    monkeypatch.setattr("imas_ambix.agent.slurm.subprocess", recorder)

    return SimpleNamespace(
        root=tmp_path,
        endpoint=endpoint,
        base=base,
        recorder=recorder,
        before=_snapshot(tmp_path),
    )


def _assert_inert(scene) -> None:
    """No cancelling verb was reached and nothing on disk changed."""
    assert scene.recorder.destructive == []
    assert _snapshot(scene.root) == scene.before


def test_download_dry_run_prints_the_script_and_submits_nothing(scene) -> None:
    result = CliRunner().invoke(main, ["agent", "download", _PROFILE, "--dry-run"])

    assert result.exit_code == 0, result.output
    # The script was rendered, not submitted: it carries its own batch header.
    assert "#SBATCH" in result.output
    _assert_inert(scene)


def test_serve_dry_run_prints_the_script_and_submits_nothing(scene) -> None:
    result = CliRunner().invoke(main, ["agent", "serve", _PROFILE, "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "#SBATCH" in result.output
    _assert_inert(scene)


def test_router_submit_dry_run_prints_the_script_and_submits_nothing(scene) -> None:
    result = CliRunner().invoke(
        main, ["agent", "router", "--port", "18802", "--submit", "--dry-run"]
    )

    assert result.exit_code == 0, result.output
    assert "#SBATCH" in result.output
    _assert_inert(scene)


def test_setup_dry_run_prints_the_script_and_submits_nothing(scene) -> None:
    result = CliRunner().invoke(main, ["agent", "setup", "vllm", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "#SBATCH" in result.output
    _assert_inert(scene)


def test_restart_dry_run_names_the_live_job_and_cancels_nothing(scene) -> None:
    """A restart dry run reports the live serve it would stop and stops nothing.

    The cancel-on-dry-run defect this guards against reported nothing and
    cancelled the live engine instead, so the named job and the empty
    destructive log are checked together.
    """
    result = CliRunner().invoke(main, ["agent", "restart", _PROFILE, "--dry-run"])

    assert result.exit_code == 0, result.output
    assert f"would cancel 1 active job(s): {_JOB_ID}" in result.output
    assert any(argv[0] == "squeue" for argv in scene.recorder.argv)
    _assert_inert(scene)


def test_restart_dry_run_leaves_the_endpoint_document_byte_identical(scene) -> None:
    """The published document is not republished by a dry run."""
    before = scene.endpoint.read_bytes()

    result = CliRunner().invoke(main, ["agent", "restart", _PROFILE, "--dry-run"])

    assert result.exit_code == 0, result.output
    assert scene.endpoint.read_bytes() == before


def _walk_commands(group: click.Group, prefix: tuple[str, ...] = ()):
    """Yield ``(path, command)`` for every command reachable from ``group``.

    Descends through nested groups, so a ``--dry-run`` added to a subcommand of
    ``agent fleet`` is enumerated exactly as one added directly under ``agent``.
    """
    for name, command in group.commands.items():
        path = prefix + (name,)
        yield path, command
        if isinstance(command, click.Group):
            yield from _walk_commands(command, path)


def test_every_agent_command_offering_a_dry_run_is_covered() -> None:
    """A newly added ``--dry-run`` command must be added to this file.

    Enumerated from the click group rather than from a hand-kept list, so the
    audit cannot drift away from the surface it claims to cover. The walk
    reaches subcommands of nested groups, so ``agent fleet hold`` is guarded
    even though ``fleet`` sits below the agent group.
    """
    from imas_ambix.agent.cli import agent

    commands = dict(_walk_commands(agent))

    # The walk descends into nested groups; a shallow walk would silently stop
    # guarding them, and every assertion below would still pass.
    assert {"fleet hold", "fleet place", "fleet status"} <= {
        " ".join(path) for path in commands
    }

    offering = {
        " ".join(path)
        for path, command in commands.items()
        if any("--dry-run" in getattr(param, "opts", []) for param in command.params)
    }

    assert offering == {"download", "serve", "router", "restart", "setup"}
