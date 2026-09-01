"""Contracts for catalog-derived local delegation guidance."""

from __future__ import annotations

import json
import os
import subprocess
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from imas_ambix.agent.clive import generate_clive_script
from imas_ambix.agent.profile import SiteConfig


def _catalog_item(
    model_id: str,
    *,
    accelerator_count: int,
    max_model_len: int,
) -> dict[str, object]:
    return {
        "id": model_id,
        "max_model_len": max_model_len,
        "ambix": {
            "accelerator_family": "H200",
            "accelerator_count": accelerator_count,
            "checkpoint_precision": "int4",
        },
    }


@contextmanager
def _serve_catalog(items: list[dict[str, object]]):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            payload = json.dumps({"data": items}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, _format, *_args):
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


def _run_default_local_launcher(tmp_path, items, selected_model):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    arguments_file = tmp_path / "claude-arguments"
    environment_file = tmp_path / "claude-environment"
    launcher = tmp_path / "clive"

    (fake_bin / "claude").write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' \"$@\" > {arguments_file}\n"
        f"env > {environment_file}\n",
        encoding="utf-8",
    )
    (fake_bin / "claude").chmod(0o755)

    with _serve_catalog(items) as site:
        launcher.write_text(generate_clive_script(site), encoding="utf-8")
        launcher.chmod(0o755)
        environment = os.environ.copy()
        environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
        result = subprocess.run(
            [str(launcher), "--model", selected_model],
            capture_output=True,
            text=True,
            check=False,
            env=environment,
        )

    arguments = arguments_file.read_text(encoding="utf-8").splitlines()
    harness_environment = dict(
        line.split("=", 1)
        for line in environment_file.read_text(encoding="utf-8").splitlines()
        if "=" in line
    )
    return result, arguments, harness_environment


def test_default_local_session_names_the_primary_worker_and_its_catalog_card(tmp_path):
    items = [
        _catalog_item(
            "narrow-release",
            accelerator_count=2,
            max_model_len=524_288,
        ),
        _catalog_item(
            "wide-release",
            accelerator_count=4,
            max_model_len=262_144,
        ),
    ]

    result, arguments, _environment = _run_default_local_launcher(
        tmp_path, items, "wide-release"
    )

    assert result.returncode == 0, result.stderr
    guidance = arguments[arguments.index("--append-system-prompt") + 1]
    assert "sonnet alias is the primary local worker" in guidance
    assert "wide-release (4×H200, 262,144-token engine-reported context)" in guidance
    assert "2×H200" not in guidance
    assert "bulk, parallel, and mechanical work to sonnet" in guidance
    assert "adjudication and physics-critical judgement on a frontier slot" in guidance
    assert "local-only mode does not provide a frontier slot" in guidance


def test_dispatch_statement_does_not_change_alias_mapping(tmp_path):
    items = [
        _catalog_item(
            "primary-release",
            accelerator_count=2,
            max_model_len=524_288,
        ),
        _catalog_item(
            "second-release",
            accelerator_count=4,
            max_model_len=262_144,
        ),
    ]

    result, arguments, environment = _run_default_local_launcher(
        tmp_path, items, "primary-release"
    )

    assert result.returncode == 0, result.stderr
    assert arguments.count("--append-system-prompt") == 1
    for alias in ("OPUS", "SONNET", "HAIKU", "FABLE"):
        assert environment[f"ANTHROPIC_DEFAULT_{alias}_MODEL"] == "primary-release"
