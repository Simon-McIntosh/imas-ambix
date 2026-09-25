from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from click.testing import CliRunner

from imas_ambix.cli import main


def test_importing_the_cli_module_does_not_import_the_router() -> None:
    """Importing the CLI must not drag the router in through a module-level import.

    The router is deferred to the commands that use it, so a CLI invocation that
    never reaches the gate does not pay the router's import cost. Measured in a
    fresh interpreter, because this test process already holds the router for
    its other cases and a same-process check could not see the difference.
    """
    repo_root = Path(__file__).resolve().parents[1]
    probe = (
        "import sys\n"
        "import imas_ambix.agent.cli\n"
        "print(imas_ambix.agent.cli.__file__)\n"
        "raise SystemExit(1 if 'imas_ambix.agent.router' in sys.modules else 0)\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(repo_root), env.get("PYTHONPATH", "")])
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(repo_root),
    )
    assert result.returncode == 0, (
        "importing imas_ambix.agent.cli imported imas_ambix.agent.router\n"
        f"stdout: {result.stdout!r}\nstderr: {result.stderr!r}"
    )
    # The probe names the module it actually imported, so a run that resolved
    # some other checkout rather than this tree would be visible rather than
    # reading as a pass.
    assert str(repo_root) in result.stdout, result.stdout


def test_cut_form_choices_match_the_router() -> None:
    """The CLI's form choices and the router's must not drift apart.

    The ``click.Choice`` tuple is written out as a literal because the decorator
    is evaluated when the module is imported, so no import keeps it equal to the
    router's own. This test is what keeps the pair honest, and it reads the
    choices off the built option rather than a copy so it measures the value the
    command actually presents.
    """
    from imas_ambix.agent.cli import pause
    from imas_ambix.agent.router import GATE_CUT_FORMS

    cut_form = next(param for param in pause.params if param.name == "cut_form")
    assert tuple(cut_form.type.choices) == tuple(GATE_CUT_FORMS)


def _site_paths(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    """Point the CLI at a scratch endpoint document and return gate + lane paths."""
    target = tmp_path / "public" / "endpoints.json"
    monkeypatch.setenv("AMBIX_AGENT_ENDPOINT_DOCUMENT", str(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    return target.with_name("router-gate.json"), target.with_name("lane.json")


def _gate(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_pause_writes_the_flag_and_reason_and_preserves_the_width(
    tmp_path, monkeypatch
) -> None:
    gate_file, lane_file = _site_paths(tmp_path, monkeypatch)
    gate_file.write_text(
        json.dumps({"width": 18, "wait_seconds": 300.0}), encoding="utf-8"
    )
    lane_file.write_text(
        json.dumps(
            {
                "running": 9,
                "waiting": 2,
                "router_generation_gate": {"in_flight": 1, "waiting": 2},
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        main, ["agent", "pause", "--reason", "draining for a relaunch"]
    )

    assert result.exit_code == 0, result.output
    written = _gate(gate_file)
    assert written["paused"] is True
    assert written["reason"] == "draining for a relaunch"
    # A width the launch set is not dropped by writing the pause.
    assert written["width"] == 18
    assert written["wait_seconds"] == 300.0
    # The lane's counts are printed from the published document.
    assert "running=9" in result.output
    assert "waiting=2" in result.output


def test_resume_clears_the_pause_and_the_reason(tmp_path, monkeypatch) -> None:
    gate_file, lane_file = _site_paths(tmp_path, monkeypatch)
    gate_file.write_text(
        json.dumps({"width": 18, "paused": True, "reason": "draining for a relaunch"}),
        encoding="utf-8",
    )
    lane_file.write_text(json.dumps({"running": 0, "waiting": 2}), encoding="utf-8")

    result = CliRunner().invoke(main, ["agent", "resume"])

    assert result.exit_code == 0, result.output
    written = _gate(gate_file)
    assert written["paused"] is False
    assert "reason" not in written
    assert written["width"] == 18
    assert "running=0" in result.output
    assert "waiting=2" in result.output


def test_pause_then_resume_round_trips_the_gate_file(tmp_path, monkeypatch) -> None:
    gate_file, lane_file = _site_paths(tmp_path, monkeypatch)
    lane_file.write_text(json.dumps({"running": 4, "waiting": 1}), encoding="utf-8")

    paused = CliRunner().invoke(main, ["agent", "pause", "--reason", "operator drain"])
    assert paused.exit_code == 0, paused.output
    assert _gate(gate_file)["paused"] is True

    resumed = CliRunner().invoke(main, ["agent", "resume"])
    assert resumed.exit_code == 0, resumed.output
    assert _gate(gate_file)["paused"] is False


def test_pause_creates_the_gate_when_none_exists(tmp_path, monkeypatch) -> None:
    gate_file, lane_file = _site_paths(tmp_path, monkeypatch)
    assert not gate_file.exists()

    result = CliRunner().invoke(
        main, ["agent", "pause", "--reason", "draining for a relaunch"]
    )

    assert result.exit_code == 0, result.output
    written = _gate(gate_file)
    assert written["paused"] is True
    assert written["reason"] == "draining for a relaunch"
    # No lane document was published: the counts read unknown rather than
    # failing the pause, which must land even when the lane cannot be read.
    assert "running=unknown" in result.output


def test_pause_reports_a_malformed_gate_rather_than_overwriting_it(
    tmp_path, monkeypatch
) -> None:
    gate_file, _ = _site_paths(tmp_path, monkeypatch)
    gate_file.write_text("[1, 2, 3]", encoding="utf-8")

    result = CliRunner().invoke(
        main, ["agent", "pause", "--reason", "draining for a relaunch"]
    )

    assert result.exit_code != 0
    assert "does not hold a JSON object" in result.output
    assert json.loads(gate_file.read_text(encoding="utf-8")) == [1, 2, 3]


def test_pause_with_cut_writes_the_cut_beside_the_pause(tmp_path, monkeypatch) -> None:
    gate_file, lane_file = _site_paths(tmp_path, monkeypatch)
    gate_file.write_text(
        json.dumps({"width": 18, "wait_seconds": 300.0}), encoding="utf-8"
    )
    lane_file.write_text(json.dumps({"running": 3, "waiting": 0}), encoding="utf-8")

    result = CliRunner().invoke(
        main, ["agent", "pause", "--reason", "engine relaunch", "--cut"]
    )

    assert result.exit_code == 0, result.output
    written = _gate(gate_file)
    assert written["paused"] is True
    assert written["cut"] is True
    # The default form is written explicitly so a reader of the gate file sees
    # the form in force rather than inferring it from the router's default.
    assert written["cut_form"] == "close"
    assert written["width"] == 18
    assert "cut in flight (form close)" in result.output


def test_pause_cut_form_error_is_written(tmp_path, monkeypatch) -> None:
    gate_file, lane_file = _site_paths(tmp_path, monkeypatch)
    lane_file.write_text(json.dumps({"running": 0, "waiting": 0}), encoding="utf-8")

    result = CliRunner().invoke(
        main,
        [
            "agent",
            "pause",
            "--reason",
            "engine relaunch",
            "--cut",
            "--cut-form",
            "error",
        ],
    )

    assert result.exit_code == 0, result.output
    written = _gate(gate_file)
    assert written["cut"] is True
    assert written["cut_form"] == "error"
    assert "cut in flight (form error)" in result.output


def test_pause_without_cut_leaves_no_cut_declared(tmp_path, monkeypatch) -> None:
    gate_file, lane_file = _site_paths(tmp_path, monkeypatch)
    gate_file.write_text(
        json.dumps({"width": 18, "paused": True, "cut": True, "cut_form": "error"}),
        encoding="utf-8",
    )
    lane_file.write_text(json.dumps({"running": 1, "waiting": 0}), encoding="utf-8")

    result = CliRunner().invoke(main, ["agent", "pause", "--reason", "ordinary drain"])

    assert result.exit_code == 0, result.output
    written = _gate(gate_file)
    assert written["paused"] is True
    # A re-pause that asked for no cut withdraws the cut an earlier pause left,
    # so the gate never carries a cut the operator did not just asked for.
    assert "cut" not in written
    assert "cut_form" not in written
    assert written["width"] == 18


def test_pause_without_cut_adds_no_cut_to_a_plain_gate(tmp_path, monkeypatch) -> None:
    gate_file, lane_file = _site_paths(tmp_path, monkeypatch)
    lane_file.write_text(json.dumps({"running": 0, "waiting": 0}), encoding="utf-8")

    result = CliRunner().invoke(main, ["agent", "pause", "--reason", "ordinary drain"])

    assert result.exit_code == 0, result.output
    written = _gate(gate_file)
    assert written["paused"] is True
    assert "cut" not in written
    assert "cut_form" not in written


def test_resume_clears_the_pause_the_cut_and_the_form(tmp_path, monkeypatch) -> None:
    gate_file, lane_file = _site_paths(tmp_path, monkeypatch)
    gate_file.write_text(
        json.dumps(
            {
                "width": 18,
                "paused": True,
                "reason": "engine relaunch",
                "cut": True,
                "cut_form": "error",
            }
        ),
        encoding="utf-8",
    )
    lane_file.write_text(json.dumps({"running": 0, "waiting": 4}), encoding="utf-8")

    result = CliRunner().invoke(main, ["agent", "resume"])

    assert result.exit_code == 0, result.output
    written = _gate(gate_file)
    assert written["paused"] is False
    assert "reason" not in written
    assert "cut" not in written
    assert "cut_form" not in written
    assert written["width"] == 18


def test_pause_cut_form_requires_cut(tmp_path, monkeypatch) -> None:
    gate_file, _ = _site_paths(tmp_path, monkeypatch)

    result = CliRunner().invoke(
        main, ["agent", "pause", "--reason", "ordinary drain", "--cut-form", "error"]
    )

    assert result.exit_code != 0
    assert "--cut-form requires --cut" in result.output
    assert not gate_file.exists()


def test_an_invalid_cut_form_is_refused_by_click(tmp_path, monkeypatch) -> None:
    gate_file, _ = _site_paths(tmp_path, monkeypatch)

    result = CliRunner().invoke(
        main,
        [
            "agent",
            "pause",
            "--reason",
            "engine relaunch",
            "--cut",
            "--cut-form",
            "abort",
        ],
    )

    assert result.exit_code != 0
    assert "Invalid value" in result.output
    assert not gate_file.exists()


def test_the_gate_write_is_a_write_then_rename(tmp_path, monkeypatch) -> None:
    """The gate is replaced atomically, never written in place.

    The running router reads the gate file live, so a partial write becomes a
    settings parse error for a router that happened to poll mid-write. The
    observable that the write is atomic is that no write lands on the gate path
    itself and the replacement arrives by rename from a sibling scratch file.
    """
    gate_file, lane_file = _site_paths(tmp_path, monkeypatch)
    gate_file.write_text(json.dumps({"width": 18}), encoding="utf-8")
    lane_file.write_text(json.dumps({"running": 0, "waiting": 0}), encoding="utf-8")

    writes: list[Path] = []
    renames: list[tuple[Path, Path]] = []
    real_write_text = Path.write_text
    real_replace = Path.replace

    def spy_write_text(self, data, *args, **kwargs):
        writes.append(Path(self))
        return real_write_text(self, data, *args, **kwargs)

    def spy_replace(self, target):
        renames.append((Path(self), Path(target)))
        return real_replace(self, target)

    monkeypatch.setattr(Path, "write_text", spy_write_text)
    monkeypatch.setattr(Path, "replace", spy_replace)

    result = CliRunner().invoke(
        main, ["agent", "pause", "--reason", "engine relaunch", "--cut"]
    )

    assert result.exit_code == 0, result.output
    assert gate_file not in writes
    scratch = gate_file.with_suffix(".tmp")
    assert (scratch, gate_file) in renames
    assert not scratch.exists()
    # The rename carried the finished payload: what the reader sees is the
    # cut the invocation declared, not a half-written file.
    assert _gate(gate_file)["cut"] is True
