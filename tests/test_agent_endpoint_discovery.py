"""Published endpoint discovery for the standalone launcher."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from imas_ambix.agent.clive import generate_clive_script
from imas_ambix.agent.profile import SiteConfig
from imas_ambix.agent.registry import (
    PublishedEndpoint,
    ServeRegistration,
    publish_endpoint_document,
    read_endpoint_document,
    write_endpoint_document,
)


def _catalog(model_id, family, count, precision, context):
    return {
        "data": [
            {
                "id": model_id,
                "max_model_len": context,
                "ambix": {
                    "accelerator_family": family,
                    "accelerator_count": count,
                    "checkpoint_precision": precision,
                },
            }
        ]
    }


@contextmanager
def _engine(model_id, family, count, precision, context):
    body = json.dumps(_catalog(model_id, family, count, precision, context)).encode()
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append((self.path, dict(self.headers)))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format, *args):
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield PublishedEndpoint(
            model_id=model_id,
            host=host,
            port=port,
            accelerator_family=family,
            accelerator_count=count,
            checkpoint_precision=precision,
            max_context=context,
        ), requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _launcher(tmp_path, document, *, harness=None, preferred_release_id=None):
    site = SiteConfig(
        endpoint_document_path=str(document),
        preferred_release_id=preferred_release_id,
    )
    path = tmp_path / "clive"
    path.write_text(generate_clive_script(site), encoding="utf-8")
    path.chmod(0o755)
    environment = os.environ.copy()
    if harness is not None:
        environment["PATH"] = f"{harness}:{environment['PATH']}"
    return path, environment


def test_publisher_derives_complete_atomic_document_from_registrations(tmp_path):
    records = (
        ServeRegistration("alpha", "node-a", 19001, "41", 2, "bf16"),
        ServeRegistration("beta", "node-b", 19002, "42", 8, "fp8"),
    )
    catalogs = {
        records[0].origin: _catalog("alpha", "H100", 2, "bf16", 131072),
        records[1].origin: _catalog("beta", "H200", 8, "fp8", 262144),
    }
    target = tmp_path / "public" / "endpoints.json"

    publish_endpoint_document(records, target, fetch_catalog=catalogs.__getitem__)

    endpoints = read_endpoint_document(target)
    assert [(item.model_id, item.host, item.port) for item in endpoints] == [
        ("alpha", "node-a", 19001),
        ("beta", "node-b", 19002),
    ]
    assert [
        (item.accelerator_family, item.accelerator_count) for item in endpoints
    ] == [("H100", 2), ("H200", 8)]
    assert [item.checkpoint_precision for item in endpoints] == ["bf16", "fp8"]
    assert [item.max_context for item in endpoints] == [131072, 262144]
    assert target.stat().st_mode & 0o777 == 0o644
    assert not list(target.parent.glob(".*.tmp"))


def test_default_publication_path_avoids_storage_tree_and_is_traversable():
    path = SiteConfig().endpoint_document.resolve(strict=False)
    forbidden = Path("/work/projects/imas_gpu")

    assert forbidden not in (path, *path.parents)
    assert all(
        ancestor.stat().st_mode & 0o001
        for ancestor in path.parents
        if ancestor.exists()
    )


def test_two_live_releases_list_their_own_topology_and_context(tmp_path):
    with (
        _engine("alpha", "H100", 2, "bf16", 131072) as (alpha, alpha_requests),
        _engine("beta", "H200", 8, "fp8", 262144) as (beta, beta_requests),
    ):
        document = write_endpoint_document((alpha, beta), tmp_path / "endpoints.json")
        launcher, environment = _launcher(tmp_path, document)
        result = subprocess.run(
            [str(launcher), "--list"],
            capture_output=True,
            text=True,
            check=False,
            env=environment,
        )

    assert result.returncode == 0, result.stderr
    assert "alpha@2xh100\talpha · 2×H100 · bf16 · 128k ctx" in result.stdout
    assert "beta@8xh200\tbeta · 8×H200 · fp8 · 256k ctx" in result.stdout
    assert [request[0] for request in alpha_requests] == ["/v1/models"]
    assert [request[0] for request in beta_requests] == ["/v1/models"]
    assert all("Authorization" not in headers for _, headers in alpha_requests)
    assert all("Authorization" not in headers for _, headers in beta_requests)


def test_unreachable_release_is_dropped_and_malformed_document_fails_closed(tmp_path):
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    dead_host, dead_port = probe.getsockname()
    probe.close()
    with _engine("live", "H200", 4, "int4", 131072) as (live, requests):
        dead = PublishedEndpoint(
            "dead", dead_host, dead_port, "H200", 4, "int4", 131072
        )
        document = write_endpoint_document((live, dead), tmp_path / "endpoints.json")
        launcher, environment = _launcher(tmp_path, document)
        result = subprocess.run(
            [str(launcher), "--list"],
            capture_output=True,
            text=True,
            check=False,
            env=environment,
        )
        Path(document).write_text(
            json.dumps({"endpoints": [{"model_id": "incomplete"}]}),
            encoding="utf-8",
        )
        malformed = subprocess.run(
            [str(launcher), "--list"],
            capture_output=True,
            text=True,
            check=False,
            env=environment,
        )

    assert result.returncode == 0, result.stderr
    assert "live@4xh200" in result.stdout
    assert "dead" not in result.stdout
    assert malformed.returncode == 2
    assert "incomplete entry" in malformed.stderr
    assert len(requests) == 1


def test_selected_release_targets_its_own_engine_origin(tmp_path):
    with (
        _engine("alpha", "H100", 2, "bf16", 131072) as (alpha, _),
        _engine("beta", "H200", 8, "fp8", 262144) as (beta, _),
    ):
        document = write_endpoint_document((alpha, beta), tmp_path / "endpoints.json")
        fake_bin = tmp_path / "bin"
        fake_bin.mkdir()
        trace = tmp_path / "trace"
        harness = fake_bin / "codex"
        harness.write_text(
            f'#!/bin/sh\nprintf \'%s\\n\' "$OPENAI_BASE_URL" "$*" > {trace}\n',
            encoding="utf-8",
        )
        harness.chmod(0o755)
        launcher, environment = _launcher(tmp_path, document, harness=fake_bin)
        result = subprocess.run(
            [str(launcher), "--codex", "--selector", "beta", "prompt"],
            capture_output=True,
            text=True,
            check=False,
            env=environment,
        )

    assert result.returncode == 0, result.stderr
    assert trace.read_text(encoding="utf-8").splitlines() == [
        f"{beta.origin}/v1",
        "--model beta prompt",
    ]


def test_preferred_release_is_default_and_explicit_selector_overrides(tmp_path):
    with (
        _engine("alpha", "H100", 2, "bf16", 131072) as (alpha, _),
        _engine("beta", "H200", 8, "fp8", 262144) as (beta, _),
    ):
        document = write_endpoint_document((alpha, beta), tmp_path / "endpoints.json")
        fake_bin = tmp_path / "bin"
        fake_bin.mkdir()
        trace = tmp_path / "trace"
        harness = fake_bin / "codex"
        harness.write_text(
            f'#!/bin/sh\nprintf \'%s\\n\' "$OPENAI_BASE_URL" "$*" > {trace}\n',
            encoding="utf-8",
        )
        harness.chmod(0o755)
        launcher, environment = _launcher(
            tmp_path,
            document,
            harness=fake_bin,
            preferred_release_id="beta",
        )
        unqualified = subprocess.run(
            [str(launcher), "--codex", "prompt"],
            capture_output=True,
            text=True,
            check=False,
            env=environment,
        )
        default_trace = trace.read_text(encoding="utf-8").splitlines()
        explicit = subprocess.run(
            [str(launcher), "--codex", "--selector", "alpha", "prompt"],
            capture_output=True,
            text=True,
            check=False,
            env=environment,
        )
        explicit_trace = trace.read_text(encoding="utf-8").splitlines()

    assert unqualified.returncode == 0, unqualified.stderr
    assert "Select model" not in unqualified.stderr
    assert default_trace == [f"{beta.origin}/v1", "--model beta prompt"]
    assert explicit.returncode == 0, explicit.stderr
    assert explicit_trace == [f"{alpha.origin}/v1", "--model alpha prompt"]


def test_unset_or_absent_preference_keeps_selector_requirement(tmp_path):
    with (
        _engine("alpha", "H100", 2, "bf16", 131072) as (alpha, _),
        _engine("beta", "H200", 8, "fp8", 262144) as (beta, _),
    ):
        document = write_endpoint_document((alpha, beta), tmp_path / "endpoints.json")
        launcher, environment = _launcher(tmp_path, document)
        unset = subprocess.run(
            [str(launcher)],
            capture_output=True,
            text=True,
            check=False,
            env=environment,
        )
        launcher, environment = _launcher(
            tmp_path,
            document,
            preferred_release_id="not-published",
        )
        absent = subprocess.run(
            [str(launcher)],
            capture_output=True,
            text=True,
            check=False,
            env=environment,
        )

    assert unset.returncode == 2
    assert "multiple models are available; use --selector" in unset.stderr
    assert absent.returncode == 2
    assert absent.stderr == unset.stderr


def test_preferred_release_site_setting_reads_environment(monkeypatch):
    monkeypatch.setenv("AMBIX_AGENT_PREFERRED_RELEASE", " preferred-release ")

    site = SiteConfig.from_env()

    assert site.preferred_release_id == "preferred-release"
