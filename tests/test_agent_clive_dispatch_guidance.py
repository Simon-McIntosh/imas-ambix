"""Contracts for catalog-derived local delegation guidance."""

from __future__ import annotations

import json
import os
import socketserver
import subprocess
import threading
from contextlib import ExitStack, contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

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


@contextmanager
def _serve_proxy_port():
    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            return

    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _run_launcher(tmp_path, items, selected_model, *, mode="local"):
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
    (fake_bin / "systemctl").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    (fake_bin / "systemctl").chmod(0o755)

    with ExitStack() as stack:
        site = stack.enter_context(_serve_catalog(items))
        generator_options = {}
        arguments = [str(launcher), "--model", selected_model]
        if mode == "hybrid":
            proxy_port = stack.enter_context(_serve_proxy_port())
            stack.enter_context(
                patch("imas_ambix.agent.litellm_service.LITELLM_PORT", proxy_port)
            )
            generator_options = {
                "mode": "hybrid",
                "openrouter_native_release": selected_model,
            }
            arguments.extend(("--mode", "hybrid"))
        launcher.write_text(
            generate_clive_script(site, **generator_options), encoding="utf-8"
        )
        launcher.chmod(0o755)
        environment = os.environ.copy()
        environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
        result = subprocess.run(
            arguments,
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

    result, arguments, _environment = _run_launcher(tmp_path, items, "wide-release")

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

    result, arguments, environment = _run_launcher(tmp_path, items, "primary-release")

    assert result.returncode == 0, result.stderr
    assert arguments.count("--append-system-prompt") == 1
    for alias in ("OPUS", "SONNET", "HAIKU", "FABLE"):
        assert environment[f"ANTHROPIC_DEFAULT_{alias}_MODEL"] == "primary-release"


def test_hybrid_uses_both_local_aliases_and_matches_its_statement(tmp_path):
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

    result, arguments, environment = _run_launcher(
        tmp_path,
        items,
        "primary-release",
        mode="hybrid",
    )

    assert result.returncode == 0, result.stderr
    assert environment["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "primary-release"
    assert environment["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "second-release"
    assert environment["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "or-opus-4.8"
    assert environment["ANTHROPIC_DEFAULT_FABLE_MODEL"] == "or-glm-5.2"
    for alias in ("SONNET", "HAIKU"):
        assert not environment[f"ANTHROPIC_DEFAULT_{alias}_MODEL"].startswith("or-")

    guidance = arguments[arguments.index("--append-system-prompt") + 1]
    assert "sonnet alias is the primary local worker" in guidance
    assert "sonnet-4.6" not in guidance
    assert "haiku alias is the secondary local worker" in guidance
    assert "opus or fable frontier slots" in guidance
