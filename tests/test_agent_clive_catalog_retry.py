"""Contracts for the launcher waiting out a catalog origin that is between engines.

A serve relaunch under a router pause leaves the router answering 503 for
``/v1/models`` while no engine is reachable. A launch that reads the catalog
once and exits loses the relaunch; the launcher retries a 502/503 or a refused
or reset connection until the origin answers again or the budget runs out.

The generated script is a bash program, so the retry schedule is exercised by
running it against a stub origin that answers a scripted response per request.
The wait schedule is scaled through the environment: a test that spends the
production bound would spend fifteen minutes, and the bound is only reachable
through the environment because the launcher is a generated script.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from imas_ambix.agent.clive import generate_clive_script
from imas_ambix.agent.profile import SiteConfig

# One scripted answer: the status to send and the JSON body to answer with.
Response = tuple[int, object]


class _Reset:
    """A scripted answer that closes without writing one.

    The client sees the origin drop the connection, which is the reset a serve
    relaunch or a proxy in front of it produces.
    """


RESET = _Reset()


def _catalog_item(model_id: str) -> dict[str, object]:
    return {
        "id": model_id,
        "max_model_len": 512_000,
        "ambix": {
            "accelerator_family": "H200",
            "accelerator_count": 4,
            "checkpoint_precision": "mxfp4",
        },
    }


CATALOG = {"data": [_catalog_item("deepseek-v4.1-flash")]}


@contextmanager
def serve_scripted_catalog(
    responses: Sequence[Response | _Reset],
) -> Iterator[tuple[str, list[str]]]:
    """Answer each catalog fetch with the next scripted response.

    The last response repeats once the script is exhausted, so a stub left
    answering 503 keeps answering 503 until the launcher gives up. A ``RESET``
    entry closes the connection without writing a response.
    """
    hits: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            hits.append(self.path)
            scripted = responses[min(len(hits) - 1, len(responses) - 1)]
            if scripted is RESET:
                self.close_connection = True
                return
            status, payload = scripted
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *_args: object) -> None:
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}", hits
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _run_launcher(
    tmp_path,
    origin: str,
    *,
    bound: str,
    start: str,
    arguments: Sequence[str] = ("--list",),
) -> subprocess.CompletedProcess:
    launcher = tmp_path / "clive"
    launcher.write_text(
        generate_clive_script(SiteConfig(global_origin=origin)), encoding="utf-8"
    )
    launcher.chmod(0o755)
    environment = os.environ.copy()
    environment["CLIVE_CATALOG_RETRY_BOUND"] = bound
    environment["CLIVE_CATALOG_RETRY_START"] = start
    return subprocess.run(
        [str(launcher), *arguments],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )


def test_a_503_before_the_catalog_is_retried_until_it_answers(tmp_path):
    with serve_scripted_catalog(
        [(503, {"error": "no engine"}), (503, {"error": "no engine"}), (200, CATALOG)]
    ) as (origin, hits):
        result = _run_launcher(tmp_path, origin, bound="30", start="0.1")

    assert result.returncode == 0, result.stderr
    assert hits == ["/v1/models"] * 3
    assert "HTTP 503" in result.stderr
    assert result.stderr.count("retrying in") == 2
    assert "budget remaining" in result.stderr
    assert "deepseek-v4.1-flash" in result.stdout


def test_an_origin_that_stays_503_until_the_bound_fails_unreachable(tmp_path):
    with serve_scripted_catalog([(503, {"error": "no engine"})]) as (origin, hits):
        result = _run_launcher(tmp_path, origin, bound="2", start="0.1")

    assert result.returncode == 2, result.stderr
    assert "global catalog is unreachable" in result.stderr
    assert "HTTP 503" in result.stderr
    assert len(hits) > 1


def test_a_404_fails_at_once_without_a_retry(tmp_path):
    responses = [(404, {"error": "no such catalog"})]
    with serve_scripted_catalog(responses) as (origin, hits):
        result = _run_launcher(tmp_path, origin, bound="30", start="0.1")

    assert result.returncode == 2, result.stderr
    assert "global catalog is unreachable" in result.stderr
    assert "HTTP Error 404" in result.stderr
    assert "retrying in" not in result.stderr
    assert hits == ["/v1/models"]


@contextmanager
def serve_stalled_catalog() -> Iterator[tuple[str, list[str]]]:
    """Accept catalog fetches and never answer them.

    The launcher's own request timeout is what ends the wait, so the origin
    stays silent past it rather than failing or answering.
    """
    hits: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            hits.append(self.path)
            time.sleep(30)

        def log_message(self, _format: str, *_args: object) -> None:
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}", hits
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@contextmanager
def serve_refused_catalog() -> Iterator[str]:
    """An origin that refuses every connection.

    The socket is bound but never listens, so the launcher sees a refused
    connection on every attempt rather than a response.
    """
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        host, port = reserved.getsockname()
        yield f"http://{host}:{port}"


def _endpoint_entry(model_id: str, host: str, port: int) -> dict[str, object]:
    return {
        "model_id": model_id,
        "host": host,
        "port": port,
        "accelerator_family": "H200",
        "accelerator_count": 4,
        "checkpoint_precision": "fp8",
        "max_context": 512_000,
    }


def test_a_refused_connection_is_retried_until_the_budget(tmp_path):
    with serve_refused_catalog() as origin:
        result = _run_launcher(tmp_path, origin, bound="2", start="0.1")

    assert result.returncode == 2, result.stderr
    assert "connection refused" in result.stderr
    # A refused connection that were not transient would fail after its first
    # attempt; a retried one prints a status line per attempt.
    assert result.stderr.count("retrying in") >= 2


def test_a_reset_connection_is_retried_until_the_origin_answers(tmp_path):
    with serve_scripted_catalog([RESET, (200, CATALOG)]) as (origin, hits):
        result = _run_launcher(tmp_path, origin, bound="30", start="0.1")

    assert result.returncode == 0, result.stderr
    assert "connection reset" in result.stderr
    assert result.stderr.count("retrying in") == 1
    assert hits == ["/v1/models"] * 2
    assert "deepseek-v4.1-flash" in result.stdout


def test_a_timed_out_catalog_fails_without_waiting(tmp_path):
    with serve_stalled_catalog() as (origin, hits):
        started = time.monotonic()
        result = _run_launcher(tmp_path, origin, bound="30", start="0.1")
        elapsed = time.monotonic() - started

    assert result.returncode == 2, result.stderr
    assert "global catalog is unreachable" in result.stderr
    assert "retrying in" not in result.stderr
    assert hits == ["/v1/models"]
    assert elapsed < 15.0, f"a timeout waited for the budget: {elapsed:.1f}s"


def test_a_document_with_two_absent_origins_spends_one_budget(tmp_path):
    # Both origins stay bound and never listen, so every fetch is refused and
    # the ports cannot be taken by anything else while the test runs.
    with socket.socket() as first, socket.socket() as second:
        first.bind(("127.0.0.1", 0))
        second.bind(("127.0.0.1", 0))
        first_host, first_port = first.getsockname()
        second_host, second_port = second.getsockname()
        document = tmp_path / "endpoint.json"
        document.write_text(
            json.dumps(
                {
                    "endpoints": [
                        _endpoint_entry("alpha", first_host, first_port),
                        _endpoint_entry("beta", second_host, second_port),
                    ]
                }
            ),
            encoding="utf-8",
        )
        launcher = tmp_path / "clive"
        launcher.write_text(
            generate_clive_script(
                SiteConfig(endpoint_document_path=str(document))
            ),
            encoding="utf-8",
        )
        launcher.chmod(0o755)
        environment = os.environ.copy()
        environment["CLIVE_CATALOG_RETRY_BOUND"] = "4"
        environment["CLIVE_CATALOG_RETRY_START"] = "0.1"
        started = time.monotonic()
        result = subprocess.run(
            [str(launcher), "--list"],
            capture_output=True,
            text=True,
            check=False,
            env=environment,
        )
        elapsed = time.monotonic() - started

    assert result.returncode == 2, result.stderr
    assert "connection refused" in result.stderr
    # One deadline shared by every origin ends the launch inside one budget. A
    # deadline per origin spends two, so the elapsed time separates the two.
    assert elapsed < 6.0, (
        f"two absent origins spent more than the 4 s budget: {elapsed:.1f}s"
    )


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("CLIVE_CATALOG_RETRY_BOUND", "inf"),
        ("CLIVE_CATALOG_RETRY_BOUND", "-inf"),
        ("CLIVE_CATALOG_RETRY_BOUND", "nan"),
        ("CLIVE_CATALOG_RETRY_BOUND", "0"),
        ("CLIVE_CATALOG_RETRY_BOUND", "-5"),
        ("CLIVE_CATALOG_RETRY_BOUND", "900.5"),
        ("CLIVE_CATALOG_RETRY_START", "inf"),
        ("CLIVE_CATALOG_RETRY_START", "nan"),
        ("CLIVE_CATALOG_RETRY_START", "0"),
        ("CLIVE_CATALOG_RETRY_START", "-1"),
    ],
)
def test_an_unsafe_retry_override_is_refused(tmp_path, variable, value):
    launcher = tmp_path / "clive"
    launcher.write_text(
        generate_clive_script(SiteConfig(global_origin="http://127.0.0.1:1")),
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    environment = os.environ.copy()
    environment[variable] = value
    result = subprocess.run(
        [str(launcher), "--list"],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )

    assert result.returncode == 2, result.stderr
    # The refusal must name the variable so an operator knows which override
    # was rejected rather than guessing from a bare usage error.
    assert variable in result.stderr
    assert "must be" in result.stderr or "must not" in result.stderr
