"""Router upstream resolution from shared registrations and scheduler state."""

from __future__ import annotations

import asyncio

from click.testing import CliRunner

from imas_ambix.agent import cli as cli_mod
from imas_ambix.agent import router as router_mod
from imas_ambix.agent.cli import LiveRoute, ModelMetadata, ProbeResult
from imas_ambix.agent.profile import SiteConfig
from imas_ambix.agent.registry import (
    ServeRegistration,
    registration_directory,
    write_registration,
)
from imas_ambix.cli import main


def _record(
    job_id: str = "42", *, port: int = 18801, model_id: str = "glm-5.3"
) -> ServeRegistration:
    return ServeRegistration(
        model_id=model_id,
        host="98dci4-gpu-0003",
        port=port,
        job_id=job_id,
        accelerator_count=4,
        checkpoint_precision="int4",
    )


def _job(job_id: str) -> dict[str, str]:
    return {
        "jobid": job_id,
        "name": "glm-5-3",
        "state": "RUNNING",
        "time": "1:00",
        "node": "98dci4-gpu-0003",
        "gres": "gres/gpu:4",
        "comment": "ambix-serve;port=18801",
    }


def _route(*, port: int = 18801) -> LiveRoute:
    return LiveRoute(
        model_id="glm-5.3",
        node="98dci4-gpu-0003",
        port=port,
        gpu_count=4,
        job_id="42",
        base_url=f"http://98dci4-gpu-0003:{port}",
        max_context=202_752,
        readiness="ready",
        job_name="glm-5-3",
    )


def _ready(model_id: str = "glm-5.3") -> ProbeResult:
    return ProbeResult("ready", (ModelMetadata(model_id, 202_752),))


def test_registration_becomes_probe_qualified_upstream(tmp_path, monkeypatch):
    site = SiteConfig(base_dir=str(tmp_path))
    write_registration(_record(), registration_directory(site.base_dir))
    probe_keys: list[str | None] = []
    monkeypatch.setattr(cli_mod, "_running_jobs", lambda _site, **_: [_job("42")])
    monkeypatch.setattr(
        cli_mod,
        "_probe_endpoint",
        lambda _origin, key: probe_keys.append(key) or _ready(),
    )
    monkeypatch.setattr(cli_mod, "_discover_live_routes", lambda *_: ([], []))

    upstreams = cli_mod._resolve_router_upstreams(site, "secret")

    assert upstreams == [
        router_mod.Upstream(
            "http://98dci4-gpu-0003:18801",
            ("Authorization", "Bearer secret"),
            "glm-5.3",
        )
    ]
    assert probe_keys == [None]


def test_stale_or_unreachable_registration_is_not_offered(tmp_path, monkeypatch):
    site = SiteConfig(base_dir=str(tmp_path))
    directory = registration_directory(site.base_dir)
    write_registration(_record("stale", port=18800), directory)
    write_registration(_record("unreachable", port=18801), directory)
    monkeypatch.setattr(
        cli_mod, "_running_jobs", lambda _site, **_: [_job("unreachable")]
    )
    monkeypatch.setattr(
        cli_mod, "_probe_endpoint", lambda _origin, _key: ProbeResult("unreachable")
    )
    monkeypatch.setattr(cli_mod, "_discover_live_routes", lambda *_: ([], []))

    assert cli_mod._resolve_router_upstreams(site, None) == []


def test_scheduler_discovery_is_the_empty_registry_fallback(tmp_path, monkeypatch):
    site = SiteConfig(base_dir=str(tmp_path))
    monkeypatch.setattr(cli_mod, "_discover_live_routes", lambda *_: ([_route()], []))

    upstreams = cli_mod._resolve_router_upstreams(site, "secret")

    assert upstreams == [
        router_mod.Upstream(
            "http://98dci4-gpu-0003:18801",
            ("Authorization", "Bearer secret"),
            "glm-5.3",
        )
    ]


def test_probe_qualified_record_survives_scheduler_fallback_outage(
    tmp_path, monkeypatch
):
    site = SiteConfig(base_dir=str(tmp_path))
    write_registration(_record(), registration_directory(site.base_dir))
    calls = 0

    def scheduler(_site, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return [_job("42")]
        raise cli_mod.click.ClickException("scheduler unavailable")

    monkeypatch.setattr(cli_mod, "_running_jobs", scheduler)
    monkeypatch.setattr(cli_mod, "_probe_endpoint", lambda *_: _ready())

    upstreams = cli_mod._resolve_router_upstreams(site, None)

    assert upstreams == [
        router_mod.Upstream("http://98dci4-gpu-0003:18801", None, "glm-5.3")
    ]


def test_registration_wins_deduplication_and_restart_uses_new_port(
    tmp_path, monkeypatch
):
    site = SiteConfig(base_dir=str(tmp_path))
    directory = registration_directory(site.base_dir)
    write_registration(_record("old", port=18801), directory)
    write_registration(_record("new", port=19444), directory)
    monkeypatch.setattr(cli_mod, "_running_jobs", lambda _site, **_: [_job("new")])
    monkeypatch.setattr(cli_mod, "_probe_endpoint", lambda *_: _ready())
    monkeypatch.setattr(
        cli_mod, "_discover_live_routes", lambda *_: ([_route(port=19444)], [])
    )

    upstreams = cli_mod._resolve_router_upstreams(site, None)

    assert upstreams == [
        router_mod.Upstream("http://98dci4-gpu-0003:19444", None, "glm-5.3")
    ]


def test_router_command_runs_injected_resolver_on_requested_port(monkeypatch):
    captured: dict[str, object] = {}

    def serve(resolver, *, host, port):
        captured.update(resolver=resolver, host=host, port=port)

    monkeypatch.setattr(router_mod, "serve_router", serve)
    monkeypatch.setattr(cli_mod, "_resolve_router_upstreams", lambda *_: [])
    result = CliRunner().invoke(
        main, ["agent", "router", "--host", "127.0.0.1", "--port", "19000"]
    )

    assert result.exit_code == 0
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 19000
    assert isinstance(captured["resolver"], router_mod.DynamicUpstreamResolver)
    assert asyncio.run(captured["resolver"].resolve()) == []
