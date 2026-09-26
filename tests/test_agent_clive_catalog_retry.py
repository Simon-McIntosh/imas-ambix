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
import subprocess
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from imas_ambix.agent.clive import generate_clive_script
from imas_ambix.agent.profile import SiteConfig

# One scripted answer: the status to send and the JSON body to answer with.
Response = tuple[int, object]


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
    responses: Sequence[Response],
) -> Iterator[tuple[str, list[str]]]:
    """Answer each catalog fetch with the next scripted response.

    The last response repeats once the script is exhausted, so a stub left
    answering 503 keeps answering 503 until the launcher gives up.
    """
    hits: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            hits.append(self.path)
            status, payload = responses[min(len(hits) - 1, len(responses) - 1)]
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
