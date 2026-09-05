"""Published endpoint discovery for the standalone launcher."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import threading
import urllib.request
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

from click.testing import CliRunner

from imas_ambix.agent import cli as agent_cli
from imas_ambix.agent import registry as registry_mod
from imas_ambix.agent.clive import generate_clive_script
from imas_ambix.agent.profile import SiteConfig
from imas_ambix.agent.registry import (
    PublishedEndpoint,
    PublishedOrigin,
    ServeRegistration,
    publish_endpoint_document,
    read_endpoint_document,
    write_endpoint_document,
    write_registration,
)
from imas_ambix.cli import main


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
        yield (
            PublishedEndpoint(
                model_id=model_id,
                host=host,
                port=port,
                accelerator_family=family,
                accelerator_count=count,
                checkpoint_precision=precision,
                max_context=context,
            ),
            requests,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@contextmanager
def _routing_origin(catalogs):
    body = json.dumps({"data": catalogs}).encode()
    model_ids = {item["id"] for item in catalogs}
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(("GET", self.path, dict(self.headers)))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            request_body = json.loads(self.rfile.read(length))
            requests.append(("POST", self.path, dict(self.headers)))
            response = json.dumps({"model": request_body.get("model")}).encode()
            self.send_response(200 if request_body.get("model") in model_ids else 404)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, _format, *args):
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield PublishedOrigin(host=host, port=port), requests
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


def _publish_with_router_jobs(tmp_path, monkeypatch, jobs, probes):
    records = (
        ServeRegistration("alpha", "node-a", 19001, "41", 2, "bf16"),
        ServeRegistration("beta", "node-b", 19002, "42", 8, "fp8"),
    )
    directory = tmp_path / "records"
    for record in records:
        write_registration(record, directory)
    catalogs = {
        records[0].origin: _catalog("alpha", "H100", 2, "bf16", 131072),
        records[1].origin: _catalog("beta", "H200", 8, "fp8", 262144),
    }
    target = tmp_path / "endpoints.json"
    monkeypatch.setattr(
        registry_mod, "_fetch_anonymous_catalog", catalogs.__getitem__
    )
    monkeypatch.setattr(agent_cli, "_running_jobs", lambda _site: jobs)
    monkeypatch.setattr(
        agent_cli,
        "_probe_endpoint",
        lambda origin, _api_key: probes[origin],
    )

    assert (
        registry_mod.main(
            [
                "publish",
                "--directory",
                str(directory),
                "--output",
                str(target),
            ]
        )
        == 0
    )
    return json.loads(target.read_text(encoding="utf-8"))


def _probe(readiness, *model_ids):
    return SimpleNamespace(
        readiness=readiness,
        models=tuple(SimpleNamespace(model_id=model_id) for model_id in model_ids),
    )


def test_publish_command_records_router_covering_every_release(tmp_path, monkeypatch):
    origin = "http://router-node:19003"
    document = _publish_with_router_jobs(
        tmp_path,
        monkeypatch,
        [
            {
                "state": "RUNNING",
                "node": "router-node",
                "comment": "ambix-router;port=19003",
            }
        ],
        {origin: _probe("ready", "alpha", "beta")},
    )

    assert document["routing_origins"] == [{"host": "router-node", "port": 19003}]


def test_publish_command_omits_router_covering_only_some_releases(
    tmp_path, monkeypatch
):
    origin = "http://router-node:19003"
    document = _publish_with_router_jobs(
        tmp_path,
        monkeypatch,
        [
            {
                "state": "RUNNING",
                "node": "router-node",
                "comment": "ambix-router;port=19003",
            }
        ],
        {origin: _probe("ready", "alpha")},
    )

    assert document["routing_origins"] == []


def test_publish_command_omits_unreachable_router(tmp_path, monkeypatch):
    origin = "http://router-node:19003"
    document = _publish_with_router_jobs(
        tmp_path,
        monkeypatch,
        [
            {
                "state": "RUNNING",
                "node": "router-node",
                "comment": "ambix-router;port=19003",
            }
        ],
        {origin: _probe("unreachable")},
    )

    assert document["routing_origins"] == []


def test_publish_command_without_router_preserves_endpoint_document(
    tmp_path, monkeypatch
):
    document = _publish_with_router_jobs(tmp_path, monkeypatch, [], {})

    assert document == {
        "endpoints": [
            {
                "accelerator_count": 2,
                "accelerator_family": "H100",
                "checkpoint_precision": "bf16",
                "host": "node-a",
                "max_context": 131072,
                "model_id": "alpha",
                "port": 19001,
            },
            {
                "accelerator_count": 8,
                "accelerator_family": "H200",
                "checkpoint_precision": "fp8",
                "host": "node-b",
                "max_context": 262144,
                "model_id": "beta",
                "port": 19002,
            },
        ],
        "routing_origins": [],
    }


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

    routing_origin = PublishedOrigin("router", 19003)
    publish_endpoint_document(
        records,
        target,
        fetch_catalog=catalogs.__getitem__,
        routing_origins=(routing_origin,),
    )

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
    assert json.loads(target.read_text(encoding="utf-8"))["routing_origins"] == [
        {"host": "router", "port": 19003}
    ]
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


def test_complete_routing_origin_serves_every_picker_release(tmp_path):
    alpha_card = _catalog("alpha", "H100", 2, "bf16", 131072)["data"][0]
    beta_card = _catalog("beta", "H200", 8, "fp8", 262144)["data"][0]
    with (
        _engine("alpha", "H100", 2, "bf16", 131072) as (alpha, alpha_requests),
        _engine("beta", "H200", 8, "fp8", 262144) as (beta, beta_requests),
        _routing_origin([alpha_card, beta_card]) as (routing, routing_requests),
    ):
        document = write_endpoint_document(
            (alpha, beta),
            tmp_path / "endpoints.json",
            routing_origins=(routing,),
        )
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
        inference_statuses = []
        for model_id in ("alpha", "beta"):
            request = urllib.request.Request(
                f"{routing.origin}/v1/chat/completions",
                data=json.dumps({"model": model_id}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request) as response:
                inference_statuses.append(response.status)

    assert result.returncode == 0, result.stderr
    assert trace.read_text(encoding="utf-8").splitlines() == [
        f"{routing.origin}/v1",
        "--model beta prompt",
    ]
    assert inference_statuses == [200, 200]
    assert [request[0] for request in alpha_requests] == ["/v1/models"]
    assert [request[0] for request in beta_requests] == ["/v1/models"]
    catalog_requests = [request for request in routing_requests if request[0] == "GET"]
    assert len(catalog_requests) == 1
    assert "Authorization" not in catalog_requests[0][2]


def test_partial_routing_origin_does_not_advertise_unreachable_release(tmp_path):
    alpha_card = _catalog("alpha", "H100", 2, "bf16", 131072)["data"][0]
    with (
        _engine("alpha", "H100", 2, "bf16", 131072) as (alpha, _),
        _engine("beta", "H200", 8, "fp8", 262144) as (beta, _),
        _routing_origin([alpha_card]) as (routing, routing_requests),
    ):
        document = write_endpoint_document(
            (alpha, beta),
            tmp_path / "endpoints.json",
            routing_origins=(routing,),
        )
        fake_bin = tmp_path / "bin"
        fake_bin.mkdir()
        arguments = tmp_path / "arguments"
        environment_file = tmp_path / "environment"
        harness = fake_bin / "claude"
        harness.write_text(
            "#!/bin/sh\n"
            f"printf '%s\\n' \"$@\" > {arguments}\n"
            f"env > {environment_file}\n",
            encoding="utf-8",
        )
        harness.chmod(0o755)
        launcher, environment = _launcher(tmp_path, document, harness=fake_bin)
        result = subprocess.run(
            [str(launcher), "--selector", "beta"],
            capture_output=True,
            text=True,
            check=False,
            env=environment,
        )

    argv = arguments.read_text(encoding="utf-8").splitlines()
    settings = json.loads(argv[argv.index("--settings") + 1])
    harness_environment = dict(
        line.split("=", 1)
        for line in environment_file.read_text(encoding="utf-8").splitlines()
        if "=" in line
    )
    assert result.returncode == 0, result.stderr
    assert harness_environment["ANTHROPIC_BASE_URL"] == beta.origin
    assert harness_environment["ANTHROPIC_MODEL"] == "beta"
    assert [row["model"] for row in settings["modelPicker"]["options"]] == ["beta"]
    assert len([request for request in routing_requests if request[0] == "GET"]) == 1


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


def test_serve_on_changed_port_republishes_document_naming_new_port_and_cards(
    tmp_path, monkeypatch
):
    """A serve on a port the document does not name leaves it naming that port."""
    from imas_ambix.agent import slurm as slurm_mod

    site_dir = tmp_path / "site"
    target = tmp_path / "public" / "endpoints.json"
    monkeypatch.setenv("AMBIX_AGENT_BASE_DIR", str(site_dir))
    monkeypatch.setenv("AMBIX_AGENT_ENDPOINT_DOCUMENT", str(target))
    monkeypatch.setattr(agent_cli, "_running_jobs", lambda _site: [])

    directory = registry_mod.registration_directory(site_dir)
    original = ServeRegistration(
        "deepseek-v4-flash", "node-a", 19001, "41", 2, "fp4+fp8"
    )
    write_registration(original, directory)
    publish_endpoint_document(
        [original],
        target,
        fetch_catalog=lambda _origin: _catalog(
            "deepseek-v4-flash", "H200", 2, "fp4+fp8", 524288
        ),
    )
    assert read_endpoint_document(target)[0].port == 19001

    # The moved-in serve has registered on a different port with a different
    # card count, and the original port has gone silent.
    moved = ServeRegistration(
        "deepseek-v4-flash", "node-a", 19002, "42", 4, "fp4+fp8"
    )
    write_registration(moved, directory)

    def fetch(origin):
        if origin == moved.origin:
            return _catalog("deepseek-v4-flash", "H200", 4, "fp4+fp8", 524288)
        raise OSError(f"unreachable: {origin}")

    monkeypatch.setattr(registry_mod, "_fetch_anonymous_catalog", fetch)
    monkeypatch.setattr(slurm_mod, "submit_script", lambda _script: "42")

    result = CliRunner().invoke(
        main,
        ["agent", "serve", "deepseek-v4-flash", "--port", "19002"],
    )

    assert result.exit_code == 0
    endpoint = read_endpoint_document(target)[0]
    assert (endpoint.model_id, endpoint.host, endpoint.port) == (
        "deepseek-v4-flash",
        "node-a",
        19002,
    )
    assert endpoint.accelerator_count == 4


def test_shutdown_cancels_and_republishes_removing_the_release(
    tmp_path, monkeypatch
):
    """`agent shutdown` drops the cancelled release from the published document."""
    site_dir = tmp_path / "site"
    target = tmp_path / "endpoints.json"
    monkeypatch.setenv("AMBIX_AGENT_BASE_DIR", str(site_dir))
    monkeypatch.setenv("AMBIX_AGENT_ENDPOINT_DOCUMENT", str(target))
    monkeypatch.setattr(agent_cli, "_running_jobs", lambda _site: [])

    directory = registry_mod.registration_directory(site_dir)
    alpha = ServeRegistration("alpha", "node-a", 19001, "41", 2, "bf16")
    beta = ServeRegistration("beta", "node-b", 19002, "42", 8, "fp8")
    write_registration(alpha, directory)
    write_registration(beta, directory)
    catalogs = {
        alpha.origin: _catalog("alpha", "H100", 2, "bf16", 131072),
        beta.origin: _catalog("beta", "H200", 8, "fp8", 262144),
    }
    monkeypatch.setattr(
        registry_mod, "_fetch_anonymous_catalog", catalogs.__getitem__
    )
    publish_endpoint_document(
        [alpha, beta], target, fetch_catalog=catalogs.__getitem__
    )
    assert {endpoint.model_id for endpoint in read_endpoint_document(target)} == {
        "alpha",
        "beta",
    }

    def fake_subprocess(command, *args, **kwargs):
        if command[0] == "squeue":
            return subprocess.CompletedProcess(command, 0, "41|alpha\n", "")
        if command[0] == "scancel":
            return subprocess.CompletedProcess(command, 0, "", "")
        raise AssertionError(f"unexpected subprocess call: {command}")

    monkeypatch.setattr(agent_cli.subprocess, "run", fake_subprocess)

    result = CliRunner().invoke(main, ["agent", "shutdown", "alpha", "--yes"])

    assert result.exit_code == 0
    assert "Cancelled 1 job(s)" in result.output
    endpoints = read_endpoint_document(target)
    assert [endpoint.model_id for endpoint in endpoints] == ["beta"]
    remaining = registry_mod.read_registrations(
        directory, job_is_running=lambda _job_id: True
    ).current
    assert [record.job_id for record in remaining] == ["42"]


def test_status_warns_when_published_document_disagrees_with_live_serves(
    tmp_path, monkeypatch
):
    """`agent status` surfaces document-vs-live drift instead of silence."""
    document = tmp_path / "endpoints.json"
    document.write_text(
        json.dumps(
            {
                "endpoints": [
                    {
                        "model_id": "alpha",
                        "host": "node-a",
                        "port": 19001,
                        "accelerator_family": "H100",
                        "accelerator_count": 2,
                        "checkpoint_precision": "bf16",
                        "max_context": 131072,
                    }
                ],
                "routing_origins": [],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("AMBIX_AGENT_ENDPOINT_DOCUMENT", str(document))
    monkeypatch.setattr(
        agent_cli,
        "_running_jobs",
        lambda _site: [
            {
                "jobid": "42",
                "name": "deepseek-v4-flash",
                "state": "RUNNING",
                "time": "5:00",
                "node": "node-b",
            }
        ],
    )
    monkeypatch.setattr(agent_cli, "_read_key_file", lambda _path: None)
    monkeypatch.setattr(agent_cli, "_endpoint_requires_key", lambda _url: False)
    monkeypatch.setattr(
        agent_cli,
        "_discover_live_routes",
        lambda _site, _key, jobs: (
            [
                agent_cli.LiveRoute(
                    model_id="alpha",
                    node="node-b",
                    port=19002,
                    gpu_count=4,
                    job_id="42",
                    base_url="http://node-b:19002",
                    max_context=131072,
                    readiness="ready",
                    job_name="deepseek-v4-flash",
                )
            ],
            [],
        ),
    )

    result = CliRunner().invoke(main, ["agent", "status"])

    assert result.exit_code == 0
    assert "disagrees with live serves" in result.output
    assert "node-a:19001" in result.output
    assert "node-b:19002" in result.output
