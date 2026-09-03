"""Contracts for catalog-derived Clive picker presentation."""

from __future__ import annotations

import json
import os
import subprocess
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from imas_ambix.agent.clive import generate_clive_script
from imas_ambix.agent.profile import SiteConfig


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


@contextmanager
def _serve_catalog(items: list[dict[str, object]]):
    requests: list[dict[str, str]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(dict(self.headers.items()))
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
        yield SiteConfig(global_origin=f"http://{host}:{port}"), requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _run_launcher(
    tmp_path,
    items,
    selected_model=None,
    *,
    preferred_release_id=None,
):
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

    with _serve_catalog(items) as (site, requests):
        site = site.model_copy(update={"preferred_release_id": preferred_release_id})
        launcher.write_text(generate_clive_script(site), encoding="utf-8")
        launcher.chmod(0o755)
        environment = os.environ.copy()
        environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
        command = [str(launcher)]
        if selected_model is not None:
            command.extend(("--model", selected_model))
        result = subprocess.run(
            command,
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
    settings = json.loads(arguments[arguments.index("--settings") + 1])
    return result, settings, harness_environment, requests


def test_preferred_release_selects_without_narrowing_picker(tmp_path):
    items = [
        _catalog_item(
            "first-release",
            accelerator_count=2,
            max_model_len=524_288,
        ),
        _catalog_item(
            "preferred-release",
            accelerator_count=4,
            max_model_len=65_536,
        ),
    ]

    result, settings, environment, _requests = _run_launcher(
        tmp_path,
        items,
        preferred_release_id="preferred-release",
    )

    assert result.returncode == 0, result.stderr
    assert "Select model" not in result.stderr
    assert environment["ANTHROPIC_MODEL"] == "preferred-release"
    assert [row["model"] for row in settings["modelPicker"]["options"]] == [
        "first-release",
        "preferred-release",
    ]


def test_each_release_gets_its_own_topology_and_context(tmp_path):
    items = [
        _catalog_item(
            "narrow-release",
            accelerator_count=2,
            max_model_len=524_288,
            precision="int4",
        ),
        _catalog_item(
            "wide-release",
            accelerator_count=4,
            max_model_len=262_144,
        ),
    ]

    result, settings, environment, requests = _run_launcher(
        tmp_path, items, "narrow-release"
    )

    assert result.returncode == 0, result.stderr
    assert len(requests) == 1
    assert "Authorization" not in requests[0]
    assert settings["modelPicker"]["replaceBuiltInOptions"] is True
    assert settings["modelPicker"]["options"] == [
        {
            "model": "narrow-release",
            "label": "narrow-release",
            "description": "2×H200 · int4 · 512k context",
        },
        {
            "model": "wide-release",
            "label": "wide-release",
            "description": "4×H200 · fp8 · 256k context",
        },
    ]
    # The exported context is the input ceiling, not the served window, so the
    # harness plans against a budget the engine can actually accept.
    assert environment["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "492288"
    assert environment["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "32000"
    assert 492_288 + 32_000 <= 524_288
    assert "492288-token usable input budget" in result.stderr
    assert "32000-token output reservation" in result.stderr


def test_small_context_uses_its_own_safe_output_reservation(tmp_path):
    items = [
        _catalog_item(
            "large-release",
            accelerator_count=2,
            max_model_len=524_288,
        ),
        _catalog_item(
            "small-release",
            accelerator_count=4,
            max_model_len=65_536,
        ),
    ]

    result, _settings, environment, _requests = _run_launcher(
        tmp_path, items, "small-release"
    )

    assert result.returncode == 0, result.stderr
    usable_input = int(environment["CLAUDE_CODE_MAX_CONTEXT_TOKENS"])
    reservation = int(environment["CLAUDE_CODE_MAX_OUTPUT_TOKENS"])
    assert reservation == 16_384
    assert usable_input == 49_152
    assert usable_input > 33_537
    # A prompt filled to the declared ceiling still leaves the reservation
    # inside the served window, which is what keeps the engine from refusing.
    assert usable_input + reservation <= 65_536
    assert "49152-token usable input budget" in result.stderr
    assert "16384-token output reservation" in result.stderr


@pytest.mark.parametrize("max_model_len", [2, 3, 65_536, 524_288])
def test_output_reservation_always_leaves_a_minimal_prompt(tmp_path, max_model_len):
    items = [
        _catalog_item(
            "bounded-release",
            accelerator_count=2,
            max_model_len=max_model_len,
        )
    ]

    result, _settings, environment, _requests = _run_launcher(
        tmp_path, items, "bounded-release"
    )

    assert result.returncode == 0, result.stderr
    usable_input = int(environment["CLAUDE_CODE_MAX_CONTEXT_TOKENS"])
    reservation = int(environment["CLAUDE_CODE_MAX_OUTPUT_TOKENS"])
    assert reservation >= 1
    assert usable_input >= 1
    assert usable_input + reservation <= max_model_len


def test_picker_rows_do_not_create_or_redirect_aliases(tmp_path):
    items = [
        _catalog_item(
            f"catalog-release-{index}",
            accelerator_count=2 if index % 2 else 4,
            max_model_len=131_072 * index,
        )
        for index in range(1, 6)
    ]

    result, settings, environment, _requests = _run_launcher(
        tmp_path, items, "catalog-release-5"
    )

    assert result.returncode == 0, result.stderr
    assert len(settings["modelPicker"]["options"]) == 5
    assert settings["modelPicker"]["options"][-1]["model"] == "catalog-release-5"
    for row in settings["modelPicker"]["options"]:
        assert row["label"] == row["model"]
        assert "×H200" in row["description"]
        assert "fp8" in row["description"]
        assert "context" in row["description"]
    for alias in ("OPUS", "SONNET", "HAIKU", "FABLE"):
        assert environment[f"ANTHROPIC_DEFAULT_{alias}_MODEL"] == "catalog-release-5"


def test_every_declared_alias_has_supported_capabilities(tmp_path):
    items = [
        _catalog_item(
            "future-release",
            accelerator_count=6,
            max_model_len=393_216,
        )
    ]

    result, _settings, environment, _requests = _run_launcher(
        tmp_path, items, "future-release"
    )

    assert result.returncode == 0, result.stderr
    model_variables = {
        name
        for name in environment
        if name.startswith("ANTHROPIC_DEFAULT_")
        and name.endswith("_MODEL")
        and "_NAME" not in name
        and "_DESCRIPTION" not in name
    }
    assert model_variables == {
        "ANTHROPIC_DEFAULT_OPUS_MODEL",
        "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        "ANTHROPIC_DEFAULT_FABLE_MODEL",
    }
    for model_variable in model_variables:
        assert environment[f"{model_variable}_SUPPORTED_CAPABILITIES"] == "thinking"
