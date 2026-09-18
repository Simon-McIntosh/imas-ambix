"""The local alias reaches the local engine, measured where the request arrives.

The launcher's alias mapping is fenced elsewhere by inspecting what it exports to
the harness -- a statement about where a request departs. This module fences the
other end of the boundary: it stands up a recording origin, runs the generated
launcher with the real harness binary, and asserts on the inference request that
actually arrives.

A stub that echoed its own argv would only restate the exported environment the
existing suite already covers, while saying nothing about which model the alias
resolved through. The measure therefore skips with a reason when the harness is
absent rather than substituting such a stub.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from imas_ambix.agent.clive import generate_clive_script
from imas_ambix.agent.profile import SiteConfig

if TYPE_CHECKING:
    from collections.abc import Callable

LOCAL_RELEASE = "fenced-local-release"
HOSTED_SENTINEL = "fenced-hosted-model-sentinel"
ALIAS_REQUEST = "sonnet"
PROMPT = "reply with the single word pong"
PLANTED_CREDENTIALS = {
    "ANTHROPIC_AUTH_TOKEN": "planted-anthropic-auth-token",
    "ANTHROPIC_API_KEY": "planted-anthropic-api-key",
    "OPENROUTER_API_KEY": "planted-openrouter-key",
    "ANTHROPIC_BASE_URL": "http://planted-credential-origin.invalid",
}


@dataclass
class RecordedRequest:
    """One inbound inference request as the local engine receives it."""

    path: str
    headers: dict[str, str]
    raw_body: bytes
    body: dict[str, object]

    @property
    def model(self) -> object:
        return self.body.get("model")

    def mentions(self, needle: str) -> bool:
        decoded = self.raw_body.decode("utf-8", "replace")
        return needle in json.dumps(self.headers) or needle in decoded


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        catalog = {
            "data": [
                {
                    "id": LOCAL_RELEASE,
                    "max_model_len": 524288,
                    "ambix": {
                        "accelerator_family": "H200",
                        "accelerator_count": 2,
                        "checkpoint_precision": "int4",
                    },
                }
            ]
        }
        self._respond(json.dumps(catalog).encode())

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            parsed = json.loads(raw) if raw else {}
        except ValueError:
            parsed = {}
        self.server.requests.append(
            RecordedRequest(
                path=self.path,
                headers={key.lower(): value for key, value in self.headers.items()},
                raw_body=raw,
                body=parsed,
            )
        )
        message = {
            "id": "msg_fenced",
            "type": "message",
            "role": "assistant",
            "model": parsed.get("model", LOCAL_RELEASE),
            "content": [{"type": "text", "text": "pong"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 5, "output_tokens": 1},
        }
        if parsed.get("stream"):
            self._respond_event_stream(message)
        else:
            self._respond(json.dumps(message).encode())

    def _respond(self, payload: bytes, content_type: str = "application/json") -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _respond_event_stream(self, message: dict[str, object]) -> None:
        start = dict(message, content=[], stop_reason=None)
        events = [
            ("message_start", {"type": "message_start", "message": start}),
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "pong"},
                },
            ),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            (
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": 1},
                },
            ),
            ("message_stop", {"type": "message_stop"}),
        ]
        body = "".join(
            f"event: {name}\ndata: {json.dumps(payload)}\n\n" for name, payload in events
        ).encode()
        self._respond(body, content_type="text/event-stream")

    def log_message(self, _format: str, *_args: object) -> None:
        return


class _Origin(ThreadingHTTPServer):
    """A catalog and a recording engine sharing one loopback port."""

    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.requests: list[RecordedRequest] = []


@dataclass
class Arrival:
    returncode: int
    stderr: str
    requests: list[RecordedRequest]
    placeholder: str
    host: str


def _launcher_placeholder(script: str) -> str:
    matches = re.findall(r'^KEY="([^"]*)"$', script, flags=re.MULTILINE)
    assert len(matches) == 1, matches
    return matches[0]


def _observe(
    tmp_path: Path,
    *,
    mutate: Callable[[str], str] | None = None,
) -> Arrival:
    """Generate the launcher against a recording origin, run it, return arrivals."""
    harness = shutil.which("claude")
    if harness is None:
        pytest.skip(
            "the claude harness binary is not on PATH; the arrival measure runs "
            "the real harness, and an argv-echoing stub would only restate the "
            "exported environment the dispatch-guidance suite already fences"
        )
    origin = _Origin()
    threading.Thread(target=origin.serve_forever, daemon=True).start()
    try:
        host, port = origin.server_address
        script = generate_clive_script(
            SiteConfig(global_origin=f"http://{host}:{port}")
        )
        if mutate is not None:
            script = mutate(script)
        launcher = tmp_path / "clive"
        launcher.write_text(script, encoding="utf-8")
        launcher.chmod(0o755)
        environment = os.environ.copy()
        environment["PATH"] = f"{Path(harness).parent}:{environment['PATH']}"
        environment.update(PLANTED_CREDENTIALS)
        result = subprocess.run(
            [
                str(launcher),
                "--model",
                LOCAL_RELEASE,
                "--",
                "--model",
                ALIAS_REQUEST,
                "-p",
                PROMPT,
                "--max-turns",
                "1",
            ],
            capture_output=True,
            text=True,
            env=environment,
            stdin=subprocess.DEVNULL,
            timeout=240,
        )
        return Arrival(
            returncode=result.returncode,
            stderr=result.stderr,
            requests=list(origin.requests),
            placeholder=_launcher_placeholder(script),
            host=f"{host}:{port}",
        )
    finally:
        origin.shutdown()
        origin.server_close()


@pytest.fixture(scope="module")
def arrival(tmp_path_factory: pytest.TempPathFactory) -> Arrival:
    return _observe(tmp_path_factory.mktemp("alias-arrival"))


def test_alias_request_arrives_naming_the_local_release(arrival: Arrival) -> None:
    assert arrival.returncode == 0, arrival.stderr
    assert arrival.requests, "the harness issued no inference request"
    models = {request.model for request in arrival.requests}
    assert models == {LOCAL_RELEASE}, models
    assert ALIAS_REQUEST not in models
    for request in arrival.requests:
        assert request.headers.get("host") == arrival.host
        assert request.path.startswith("/v1/messages")


def test_arrival_carries_no_readable_credential(arrival: Arrival) -> None:
    assert arrival.returncode == 0, arrival.stderr
    assert arrival.requests, "the harness issued no inference request"
    for request in arrival.requests:
        for name, value in PLANTED_CREDENTIALS.items():
            assert not request.mentions(value), (name, request.headers, request.model)
        authorization = request.headers.get("authorization")
        assert authorization in (None, f"Bearer {arrival.placeholder}")
        assert request.headers.get("x-api-key") is None


def test_a_hosted_alias_mapping_is_visible_at_arrival(tmp_path: Path) -> None:
    """The recorder can see a non-local landing, so its local reading means one."""
    needle = 'ANTHROPIC_DEFAULT_SONNET_MODEL="$MODEL_ID"'
    replacement = f'ANTHROPIC_DEFAULT_SONNET_MODEL="{HOSTED_SENTINEL}"'

    def mutate(script: str) -> str:
        assert needle in script
        return script.replace(needle, replacement)

    arrival = _observe(tmp_path, mutate=mutate)

    assert arrival.returncode == 0, arrival.stderr
    assert {request.model for request in arrival.requests} == {HOSTED_SENTINEL}
