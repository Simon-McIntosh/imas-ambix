"""The site gate file keeps a copy outside the directory it governs."""

from __future__ import annotations

import json

from imas_ambix.agent import cli


def _site(monkeypatch, tmp_path):
    base = tmp_path / "site"
    monkeypatch.setenv("AMBIX_AGENT_BASE_DIR", str(base))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.delenv("AMBIX_ROUTER_GATE_PATH", raising=False)
    gate = base / "agents" / "control" / "router-gate.json"
    copy = tmp_path / "state" / "ambix" / "control" / "router-gate.json"
    return gate, copy


def test_a_site_gate_write_leaves_a_copy(monkeypatch, tmp_path) -> None:
    gate, copy = _site(monkeypatch, tmp_path)

    cli._write_gate_document(gate, {"width": 12, "wait_seconds": 300})

    assert json.loads(copy.read_text())["width"] == 12
    assert copy.read_text() == gate.read_text()


def test_a_lost_site_gate_file_is_read_from_its_copy(monkeypatch, tmp_path) -> None:
    gate, copy = _site(monkeypatch, tmp_path)
    cli._write_gate_document(gate, {"width": 12, "wait_seconds": 240, "paused": False})
    gate.unlink()

    payload = cli._read_gate_document(gate)

    assert payload == {"width": 12, "wait_seconds": 240, "paused": False}


def test_a_gate_file_named_elsewhere_leaves_no_copy(monkeypatch, tmp_path) -> None:
    _, copy = _site(monkeypatch, tmp_path)
    elsewhere = tmp_path / "experiment" / "router-gate.json"

    cli._write_gate_document(elsewhere, {"width": 4})
    elsewhere.unlink()

    assert not copy.exists()
    assert cli._read_gate_document(elsewhere) == {}
