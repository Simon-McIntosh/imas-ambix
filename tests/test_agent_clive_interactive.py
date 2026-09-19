"""Terminal-driven contracts for Clive's numbered release prompt.

The release prompt is rendered by the *generated* launcher rather than by the
generator: the script decides whether it is interactive from `[[ -t 0 ]]` and
reads the answer back from a descriptor pointing at that same terminal. So the
prompt is invisible to `--print` and to any run whose stdin is a pipe, and a
launcher test with `capture_output=True` cannot assert anything about what a
human sees. These tests attach the deployed script to a pseudo-terminal, write
an index to the master side the way a person would type one, and assert the
rendered rows and the selection that results.
"""

from __future__ import annotations

import json
import os
import pty
import re
import subprocess
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from imas_ambix.agent.clive import generate_clive_script
from imas_ambix.agent.profile import SiteConfig

# The served id the picker rows map onto so the harness has a description for a
# release its own catalog does not know. It is a name the launcher sends to the
# harness and not one a human should be asked to choose from.
HARNESS_ALIAS = "claude-sonnet-5"

ROW_PATTERN = re.compile(r"^\s+(\d+)\)\s+(.*\S)\s*$")


def _catalog_item(
    model_id: str,
    *,
    accelerator_count: int,
    max_model_len: int,
    precision: str = "fp8",
) -> dict[str, object]:
    return {
        "id": model_id,
        "max_model_len": max_model_len,
        "ambix": {
            "accelerator_family": "H200",
            "accelerator_count": accelerator_count,
            "checkpoint_precision": precision,
        },
    }


def _served_releases() -> list[dict[str, object]]:
    return [
        _catalog_item("release-alpha", accelerator_count=2, max_model_len=524_288),
        _catalog_item(
            "release-beta",
            accelerator_count=4,
            max_model_len=131_072,
            precision="int4",
        ),
    ]


def _expected_label(item: dict[str, object]) -> str:
    ambix = item["ambix"]
    assert isinstance(ambix, dict)
    return (
        f"{item['id']} · {ambix['accelerator_count']}×H200 · "
        f"{ambix['checkpoint_precision']} · {item['max_model_len'] // 1024}k ctx"
    )


def _expected_row(item: dict[str, object], index: int) -> str:
    return f"{index}) {_expected_label(item)}"


@contextmanager
def _serve_catalog(items: list[dict[str, object]]):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            payload = json.dumps({"data": items}).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, _format, *_args) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield SiteConfig(global_origin=f"http://{host}:{port}")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _typed_terminal_run(tmp_path, items, answer: str | None, *extra: str):
    """Run the generated launcher with stdin attached to a pseudo-terminal.

    ``answer`` is written to the master side as a whole line, which is what the
    launcher reads back through the duplicated descriptor. A ``None`` answer
    leaves the terminal untouched, so a launcher that does not prompt still
    exits.
    """
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    arguments_file = tmp_path / "claude-arguments"
    environment_file = tmp_path / "claude-environment"
    launcher = tmp_path / "clive"

    (fake_bin / "claude").write_text(
        "#!/bin/sh\n"
        f'printf \'%s\\n\' "$@" > "{arguments_file}"\n'
        f'env > "{environment_file}"\n',
        encoding="utf-8",
    )
    (fake_bin / "claude").chmod(0o755)

    with _serve_catalog(items) as site:
        launcher.write_text(generate_clive_script(site), encoding="utf-8")
        launcher.chmod(0o755)
        environment = os.environ.copy()
        environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
        master, slave = pty.openpty()
        try:
            process = subprocess.Popen(
                [str(launcher), *extra],
                stdin=slave,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=environment,
            )
        finally:
            os.close(slave)
        try:
            if answer is not None:
                os.write(master, f"{answer}\n".encode())
            stdout, stderr = process.communicate(timeout=120)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            raise
        return _Run(
            returncode=process.returncode,
            stdout=stdout.decode(),
            stderr=stderr.decode(),
            arguments_file=arguments_file,
            environment_file=environment_file,
        )


class _Run:
    def __init__(self, *, returncode, stdout, stderr, arguments_file, environment_file):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.arguments_file = arguments_file
        self.environment_file = environment_file

    @property
    def rows(self) -> list[str]:
        """Rendered prompt rows, index and label together, as a human sees them."""
        return [
            match.group(0).strip()
            for line in self.stderr.splitlines()
            if (match := ROW_PATTERN.match(line))
        ]

    @property
    def labels(self) -> list[str]:
        return [
            match.group(2)
            for line in self.stderr.splitlines()
            if (match := ROW_PATTERN.match(line))
        ]

    @property
    def harness_environment(self) -> dict[str, str]:
        return dict(
            line.split("=", 1)
            for line in self.environment_file.read_text(encoding="utf-8").splitlines()
            if "=" in line
        )


def test_each_release_is_rendered_once_at_its_own_index(tmp_path):
    items = _served_releases()

    run = _typed_terminal_run(tmp_path, items, "1")

    assert run.returncode == 0, run.stderr
    assert run.rows == [
        _expected_row(items[0], 1),
        _expected_row(items[1], 2),
    ]
    assert run.labels == [_expected_label(items[0]), _expected_label(items[1])]


def test_rows_carry_the_release_label_and_no_harness_alias(tmp_path):
    items = _served_releases()

    run = _typed_terminal_run(tmp_path, items, "1")

    assert run.returncode == 0, run.stderr
    for item, row in zip(items, run.labels, strict=True):
        assert row.startswith(f"{item['id']} · ")
    assert HARNESS_ALIAS not in run.stderr
    for alias in ("sonnet", "opus", "haiku", "fable"):
        assert alias not in run.stderr.lower()


@pytest.mark.parametrize("index", [1, 2])
def test_writing_an_index_selects_that_release(tmp_path, index):
    items = _served_releases()

    run = _typed_terminal_run(tmp_path, items, str(index))

    assert run.returncode == 0, run.stderr
    assert run.harness_environment["ANTHROPIC_MODEL"] == items[index - 1]["id"]


def test_an_index_outside_the_catalog_is_refused(tmp_path):
    items = _served_releases()
    arguments_file = tmp_path / "claude-arguments"

    run = _typed_terminal_run(tmp_path, items, str(len(items) + 1))

    assert run.returncode != 0
    assert f"1 through {len(items)}" in run.stderr
    assert not arguments_file.exists()


def test_an_explicit_selection_renders_no_prompt(tmp_path):
    items = _served_releases()

    run = _typed_terminal_run(tmp_path, items, None, "--model", "release-alpha")

    assert run.returncode == 0, run.stderr
    assert "Select model" not in run.stderr
    assert run.rows == []
    assert run.harness_environment["ANTHROPIC_MODEL"] == "release-alpha"
