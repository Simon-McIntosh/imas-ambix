"""``agent publish-watch --once`` publishes the watch document atomically.

Each iteration derives the full ``agent watch --json`` document, adds
``compute_seconds`` and ``published_at``, and writes it through a temporary file
renamed over the target. A failure before that rename must leave the previous
publication byte-for-byte unchanged, and an iteration returns one physical log
line. These pin those properties through stubbed derivation, clock and replace.
"""

from __future__ import annotations

import json
import os
import stat
from datetime import UTC, datetime

from click.testing import CliRunner

from imas_ambix.agent import watch_publish
from imas_ambix.cli import main

FROZEN_TIME = datetime(2026, 10, 2, 17, 30, 0, tzinfo=UTC)
FROZEN_STAMP = "2026-10-02T17:30:00Z"


def _stub_document(monkeypatch, document) -> None:
    """Replace ``watch.watch_document`` with a document or a raising callable."""
    from imas_ambix.agent import watch as watch_mod

    monkeypatch.setattr(watch_mod, "watch_document", document)


def _freeze_clock(monkeypatch) -> None:
    monkeypatch.setattr(watch_publish, "_utcnow", lambda: FROZEN_TIME)


def _run_once(tmp_path, *extra: str):
    target = tmp_path / "watch.json"
    result = CliRunner().invoke(
        main,
        ["agent", "publish-watch", "--once", "--output", str(target), *extra],
    )
    return result, target


def test_once_writes_content_and_both_added_keys(tmp_path, monkeypatch) -> None:
    """``--once`` writes the document plus ``compute_seconds`` and ``published_at``."""
    document = {"record": {"served_name": "deepseek-v4-flash"}, "ledger": []}
    _stub_document(monkeypatch, lambda **kwargs: dict(document))
    _freeze_clock(monkeypatch)

    result, target = _run_once(tmp_path)

    assert result.exit_code == 0, result.output
    written = json.loads(target.read_text(encoding="utf-8"))
    assert written["record"] == document["record"]
    assert written["ledger"] == document["ledger"]
    assert written["published_at"] == FROZEN_STAMP
    assert isinstance(written["compute_seconds"], float)


def test_once_publishes_a_world_readable_file(tmp_path, monkeypatch) -> None:
    """The published document is world-readable, not mkstemp's owner-only mode."""
    _stub_document(monkeypatch, lambda **kwargs: {"record": {}})
    _freeze_clock(monkeypatch)

    result, target = _run_once(tmp_path)

    assert result.exit_code == 0, result.output
    assert stat.S_IMODE(target.stat().st_mode) == 0o644


def test_failure_with_newline_prints_one_line_and_keeps_the_target_unchanged(
    tmp_path, monkeypatch
) -> None:
    """A raising derivation prints one line and leaves the previous file intact."""
    target = tmp_path / "watch.json"
    previous = b'{"published_at": "2026-10-02T17:25:00Z", "keep": true}\n'
    target.write_bytes(previous)

    def boom(**_kwargs):
        raise RuntimeError("database is\nlocked")

    _stub_document(monkeypatch, boom)
    _freeze_clock(monkeypatch)

    result = CliRunner().invoke(
        main, ["agent", "publish-watch", "--once", "--output", str(target)]
    )

    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert len(lines) == 1, result.output
    assert lines[0] == f"{FROZEN_STAMP} publish-watch failed: database is locked"
    assert target.read_bytes() == previous


def test_failed_replace_keeps_the_target_unchanged(tmp_path, monkeypatch) -> None:
    """A replace that raises leaves the previous file and prints one failure line."""
    target = tmp_path / "watch.json"
    previous = b"the previous publication\n"
    target.write_bytes(previous)
    _stub_document(monkeypatch, lambda **kwargs: {"record": {}})
    _freeze_clock(monkeypatch)

    def refuse(_src, _dst):
        raise OSError("rename refused")

    monkeypatch.setattr(os, "replace", refuse)

    result = CliRunner().invoke(
        main, ["agent", "publish-watch", "--once", "--output", str(target)]
    )

    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert len(lines) == 1, result.output
    assert "publish-watch failed: rename refused" in lines[0]
    assert target.read_bytes() == previous


def test_one_iteration_marks_only_success(tmp_path, monkeypatch) -> None:
    """The cadence callback is skipped if derivation leaves the old file intact."""
    target = tmp_path / "watch.json"
    _freeze_clock(monkeypatch)
    succeeded: list[str] = []
    _stub_document(monkeypatch, lambda **kwargs: {"record": {}})

    line = watch_publish.one_iteration(
        target, on_success=lambda: succeeded.append("ok")
    )
    previous = target.read_bytes()
    assert "publish-watch wrote" in line
    assert succeeded == ["ok"]

    def fail(**_kwargs):
        raise RuntimeError("derivation refused")

    _stub_document(monkeypatch, fail)
    line = watch_publish.one_iteration(
        target, on_success=lambda: succeeded.append("bad")
    )
    assert "publish-watch failed: derivation refused" in line
    assert target.read_bytes() == previous
    assert succeeded == ["ok"]


def test_help_keeps_manual_once_and_drops_cadence() -> None:
    result = CliRunner().invoke(
        main, ["agent", "publish-watch", "--help"], terminal_width=200
    )
    assert result.exit_code == 0, result.output
    text = " ".join(result.output.split())
    assert "~/public/imas-ambix/watch.json" in text
    assert "--once" in text
    assert "--cadence" not in text


def test_manual_command_requires_once() -> None:
    result = CliRunner().invoke(main, ["agent", "publish-watch"])
    assert result.exit_code != 0
    assert "requires --once" in result.output


def test_published_path_has_one_declaration() -> None:
    from imas_ambix.agent.watch import PUBLISHED_WATCH_PATH

    assert watch_publish.DEFAULT_OUTPUT is PUBLISHED_WATCH_PATH
