"""``agent publish-watch`` publishes the watch document atomically on a cadence.

Each iteration derives the full ``agent watch --json`` document, adds
``compute_seconds`` and ``published_at``, and writes it through a temporary file
renamed over the target. A failure before that rename must leave the previous
publication byte-for-byte unchanged, and every iteration -- success or failure --
prints exactly one flushed physical line. These pin those properties by driving
the command and the loop through their seams (stubbed ``watch_document``, frozen
clock, injected ``os.replace`` and ``sleep``).
"""

from __future__ import annotations

import json
import os
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


class _RecordingStream:
    """A stdout stand-in that records writes and flushes, in order."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def write(self, text: str) -> None:
        self.calls.append(("write", text))

    def flush(self) -> None:
        self.calls.append(("flush", ""))


def test_loop_sleeps_between_iterations_and_not_after_the_last(
    tmp_path, monkeypatch
) -> None:
    """Three iterations print three lines and sleep twice, never past the last."""
    _stub_document(monkeypatch, lambda **kwargs: {"record": {}})
    _freeze_clock(monkeypatch)
    slept: list[float] = []
    stream = _RecordingStream()

    watch_publish.run(
        tmp_path / "watch.json",
        cadence=300.0,
        iterations=3,
        sleep=slept.append,
        stream=stream,
    )

    writes = [text for kind, text in stream.calls if kind == "write"]
    flushes = [kind for kind, _ in stream.calls]
    assert len(writes) == 3
    assert all(_ok_line(text) for text in writes)
    assert flushes == ["write", "flush", "write", "flush", "write", "flush"]
    assert slept == [300.0, 300.0]


def _ok_line(text: str) -> bool:
    return text.startswith(FROZEN_STAMP + " publish-watch wrote ") and text.endswith(
        "\n"
    )


def test_help_names_the_default_output_path_and_cadence() -> None:
    result = CliRunner().invoke(
        main, ["agent", "publish-watch", "--help"], terminal_width=200
    )
    assert result.exit_code == 0, result.output
    text = " ".join(result.output.split())
    assert "~/public/imas-ambix/watch.json" in text
    assert "[default: 300.0]" in text
