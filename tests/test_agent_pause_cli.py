from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from imas_ambix.cli import main


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
        json.dumps(
            {"width": 18, "paused": True, "reason": "draining for a relaunch"}
        ),
        encoding="utf-8",
    )
    lane_file.write_text(
        json.dumps({"running": 0, "waiting": 2}), encoding="utf-8"
    )

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

    paused = CliRunner().invoke(
        main, ["agent", "pause", "--reason", "operator drain"]
    )
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
