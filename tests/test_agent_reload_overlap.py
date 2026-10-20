"""Overlap restart controls without scheduler or live-lane mutations."""

from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from imas_ambix.agent import cli
from imas_ambix.cli import main


@pytest.mark.parametrize("cancel_all", [False, True])
def test_shutdown_refuses_overlap_unless_all(monkeypatch, cancel_all):
    jobs = [{"jobid": "101", "name": "sample"}, {"jobid": "102", "name": "sample"}]
    calls = []

    def command(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(
            returncode=0, stdout="101|sample\n102|sample\n", stderr=""
        )

    monkeypatch.setattr(cli, "_resolve_slug", lambda slug: slug)
    monkeypatch.setattr(cli, "_profile_jobs", lambda *_: jobs)
    monkeypatch.setattr(cli.subprocess, "run", command)
    monkeypatch.setattr(cli, "_remove_cancelled_registrations", lambda *_: None)
    monkeypatch.setattr(cli, "_republish_endpoint_document", lambda *_: None)
    result = CliRunner().invoke(
        main,
        ["agent", "shutdown", "sample", "--yes"] + (["--all"] if cancel_all else []),
    )
    if cancel_all:
        assert result.exit_code == 0, result.output
        assert ["scancel", "101", "102"] in calls
    else:
        assert result.exit_code != 0
        assert "101" in result.output and "102" in result.output
        assert "overlap restart" in result.output
        assert not any(args[0] == "scancel" for args in calls)


class OverlapWorld:
    def __init__(self):
        self.now = 0.0
        self.events = []
        self.active = {"old"}
        self.valid = True
        self.routed = True
        self.pending_seconds = 0
        self.ready_seconds = 0

    def submit(self, script):
        self.events.append(("submit", script))
        self.active.add("replacement")
        return "replacement"

    def cancel(self, job):
        self.events.append(("cancel", job))
        self.active.remove(job)

    def sleep(self, seconds):
        self.now += seconds

    def run(self, **overrides):
        from imas_ambix.agent.reload import run_overlap

        callbacks = dict(
            script="held script",
            old_job_id="old",
            submit=self.submit,
            cancel=self.cancel,
            job_state=lambda job: (
                "PENDING" if self.now < self.pending_seconds else "RUNNING"
            ),
            ready=lambda job: self.now >= self.pending_seconds + self.ready_seconds,
            validate=lambda job: self.events.append(("validate", job)) or self.valid,
            release=lambda job: self.events.append(("release", job)),
            confirm=lambda job: self.events.append(("confirm", job)) or self.routed,
            publish=lambda: self.events.append("publish"),
            clock=lambda: self.now,
            sleep=self.sleep,
        )
        callbacks.update(overrides)
        return run_overlap(**callbacks)


def test_overlap_handover_requires_generation_release_and_routed_confirmation():
    world = OverlapWorld()
    result = world.run()
    assert result.outcome == "reloaded"
    assert result.pause_started is None
    assert world.events == [
        ("submit", "held script"),
        ("validate", "replacement"),
        ("release", "replacement"),
        ("confirm", "replacement"),
        ("cancel", "old"),
        "publish",
    ]
    assert world.active == {"replacement"}


def test_uncertain_old_cancellation_keeps_confirmed_replacement():
    world = OverlapWorld()

    def cancel(job):
        world.cancel(job)
        if job == "old":
            raise OSError("scheduler receipt lost")

    with pytest.raises(OSError, match="receipt lost"):
        world.run(cancel=cancel)
    assert world.active == {"replacement"}
    assert ("cancel", "replacement") not in world.events


@pytest.mark.parametrize(
    "failure", ["validate", "confirm", "release", "ready", "stopped"]
)
def test_overlap_failure_keeps_old_job_and_cancels_only_replacement(failure):
    world = OverlapWorld()
    overrides = {}
    if failure == "validate":
        world.valid = False
    elif failure == "confirm":
        world.routed = False
    elif failure == "release":

        def refuse(_job):
            raise OSError("cannot release registration")

        overrides["release"] = refuse
    elif failure == "ready":
        world.ready_seconds = 901
    else:
        overrides["job_state"] = lambda _job: "FAILED"
    with pytest.raises((RuntimeError, OSError)):
        world.run(**overrides)
    assert world.active == {"old"}
    assert [
        event
        for event in world.events
        if isinstance(event, tuple) and event[0] == "cancel"
    ] == [("cancel", "replacement")]
    if failure == "validate":
        assert ("release", "replacement") not in world.events


def test_overlap_pending_has_separate_bound_from_running_readiness():
    from imas_ambix.agent.reload import OverlapPendingError

    world = OverlapWorld()
    world.pending_seconds = 121
    with pytest.raises(OverlapPendingError, match="120 seconds"):
        world.run()
    assert world.now == 120
    assert world.active == {"old"}
    world = OverlapWorld()
    world.pending_seconds = 118
    world.ready_seconds = 898
    assert world.run().outcome == "reloaded"
    assert world.now == 1016


@pytest.mark.parametrize("mode", ["auto", "overlap"])
@pytest.mark.parametrize(
    "shortage", [None, "feature", "cards", "cores", "memory", "unreadable"]
)
def test_mode_requires_router_support_and_node_wide_room(monkeypatch, mode, shortage):
    import urllib.request

    from imas_ambix.agent import slurm
    from imas_ambix.agent.profile import SiteConfig, load_profile

    site = SiteConfig(global_origin="http://router:18802")
    profile = load_profile("glm-5-3").for_gpus(4)
    profile = profile.model_copy(
        update={"slurm": profile.slurm.model_copy(update={"cpus": 8, "memory": "600G"})}
    )
    commands = []
    own_jobs = [
        {
            "jobid": "101",
            "name": profile.slug,
            "state": "RUNNING",
            "node": site.gpu_host,
            "gres": "gres/gpu:4",
        }
    ]
    monkeypatch.setattr(cli, "_running_jobs", lambda *_: own_jobs)

    class Response:
        headers = {
            "x-ambix-router-features": "held-registrations"
            if shortage != "feature"
            else "other-feature"
        }

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

    monkeypatch.setattr(urllib.request, "urlopen", lambda request, **kwargs: Response())

    def scheduler(args, **kwargs):
        commands.append(args)
        if args[0] == "squeue":
            assert "-u" not in args and "-A" not in args
            used_cores = 18 if shortage == "cores" else 12
            output = (
                f"101|owner|RUNNING|8|{site.reservation}\n"
                f"202|another-user|RUNNING|{used_cores}|{site.reservation}\n"
                "203|group-b|RUNNING|30|other-reservation\n"
                f"204|waiting|PENDING|30|{site.reservation}\n"
            )
        elif "node" in args:
            used_cards = 6 if shortage == "cards" else 4
            used_memory = 1000000 if shortage == "memory" else 614400
            output = (
                f"NodeName={site.gpu_host} CfgTRES=cpu=64,gres/gpu=8 "
                f"AllocTRES=cpu=50,gres/gpu={used_cards} RealMemory=1536000 "
                f"AllocMem={used_memory}"
            )
        else:
            output = "CoreCnt=30"
        return SimpleNamespace(
            returncode=1 if shortage == "unreadable" else 0,
            stdout=output,
            stderr="unreadable" if shortage == "unreadable" else "",
        )

    monkeypatch.setattr(slurm.subprocess, "run", scheduler)
    if mode == "overlap" and shortage:
        with pytest.raises(cli.click.ClickException, match="overlap unavailable"):
            cli._restart_mode(profile, site, mode, None)
    else:
        selected, origin = cli._restart_mode(profile, site, mode, None)
        assert selected == ("pause" if shortage else "overlap")
        assert origin == site.global_origin
    if shortage == "feature":
        assert commands == []
    elif shortage != "unreadable":
        assert any(command[0] == "squeue" for command in commands)


def test_explicit_pause_does_not_probe_overlap_resources(monkeypatch):
    from imas_ambix.agent.profile import SiteConfig, load_profile

    monkeypatch.setattr(
        cli, "_running_jobs", lambda *_: pytest.fail("pause inspected scheduler room")
    )
    assert cli._restart_mode(load_profile("glm-5-3"), SiteConfig(), "pause", None) == (
        "pause",
        None,
    )


def test_free_port_excludes_old_and_every_users_scheduled_ports(monkeypatch):
    import socket

    from imas_ambix.agent.profile import SiteConfig

    def scheduler(args, **kwargs):
        assert args == ["squeue", "-h", "-w", "98dci4-gpu-0003", "-o", "%k"]
        return SimpleNamespace(
            returncode=0,
            stdout="ambix-serve;port=18801\nambix-router;port=18802\n",
            stderr="",
        )

    attempts = []

    def connect(address, **kwargs):
        attempts.append(address)
        raise ConnectionRefusedError()

    monkeypatch.setattr(cli.subprocess, "run", scheduler)
    monkeypatch.setattr(socket, "create_connection", connect)
    port = cli._overlap_port(
        SiteConfig(), {"jobid": "101", "comment": "ambix-serve;port=18800"}, 18800
    )
    assert port == 18803
    assert attempts == [("98dci4-gpu-0003", 18803)]


@pytest.fixture
def bound_restart(tmp_path, monkeypatch):
    import io
    import json
    import urllib.request
    from datetime import UTC, datetime
    from pathlib import Path

    from imas_ambix.agent import registry, slurm
    from imas_ambix.agent.profile import SiteConfig, load_profile

    profile = load_profile("glm-5-3").for_gpus(4)
    site = SiteConfig(
        base_dir=str(tmp_path), endpoint_document_path=str(tmp_path / "endpoints.json")
    )
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    old_job = dict(
        jobid="101",
        name=profile.slug,
        state="RUNNING",
        node="engine",
        comment="ambix-serve;port=18800",
    )
    world = SimpleNamespace(
        events=[],
        active={"101": old_job},
        submits=0,
        tokens=200,
        routed_origin="http://engine:18803",
        pending=False,
    )
    directory = registry.registration_directory(site.base_dir)
    (tmp_path / "lane.json").write_text(
        json.dumps(
            {
                "state": "available",
                "observed_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "running": 0,
                "router_generation_gate": {"waiting": 0},
            }
        )
    )

    def jobs(_site, **kwargs):
        return [
            job
            for job in world.active.values()
            if job["jobid"] in kwargs.get("job_ids", world.active)
        ]

    def command(args, **kwargs):
        if args[:3] == ["scontrol", "write", "batch_script"]:
            Path(args[-1]).write_text("saved batch")
        elif args[0] == "scancel":
            world.events.append(("cancel", args[1]))
            del world.active[args[1]]
        else:
            pytest.fail(str(args))
        return SimpleNamespace(returncode=0, stderr="")

    def submit(batch):
        world.submits += 1
        job_id = str(101 + world.submits)
        held = batch == "held"
        port = 18803 if held else 18800
        world.active[job_id] = dict(
            old_job,
            jobid=job_id,
            state="PENDING" if held and world.pending else "RUNNING",
            comment=f"ambix-serve;port={port}",
        )
        world.events.append(("submit", batch, job_id))
        registry.write_registration(
            registry.ServeRegistration(
                profile.model.served_name,
                "engine",
                port,
                job_id,
                4,
                "int4",
                "H200",
                held=held,
            ),
            directory,
        )
        return job_id

    def completion(request, **kwargs):
        body = json.loads(request.data)
        assert body["max_tokens"] == 400
        routed = request.full_url.startswith("http://router:")
        world.events.append(("completion", request.full_url))
        records = registry.read_registrations(
            directory, job_is_running=lambda _: True
        ).current
        if world.submits == 1 and not world.pending:
            assert records[0].held is not routed
        count = world.tokens if world.submits == 1 else 200
        response = io.BytesIO(
            json.dumps(
                {
                    "usage": {"completion_tokens": count},
                    "choices": [{"finish_reason": "length"}],
                }
            ).encode()
        )
        response.headers = {"x-ambix-upstream": world.routed_origin} if routed else {}
        return response

    monkeypatch.setattr(cli, "_running_jobs", jobs)
    monkeypatch.setattr(cli.subprocess, "run", command)
    monkeypatch.setattr(slurm, "submit_script", submit)
    monkeypatch.setattr(cli, "_probe_endpoint", lambda *_: cli.ProbeResult("ready"))
    monkeypatch.setattr(urllib.request, "urlopen", completion)
    monkeypatch.setattr(
        cli, "_republish_endpoint_document", lambda *_: world.events.append("publish")
    )
    monkeypatch.setattr(cli, "_pause_gate", lambda *_: world.events.append("pause"))
    monkeypatch.setattr(cli, "_resume_gate", lambda *_: world.events.append("resume"))

    def run(*, fallback=False):
        cli._execute_restart_cycle(
            profile,
            site,
            old_job,
            "held",
            None,
            18803,
            "test",
            {},
            mode="overlap",
            router_origin="http://router:18802",
            fallback_script="pause" if fallback else None,
        )

    world.run = run
    return world


@pytest.mark.parametrize(
    "routed_origin", ["http://engine:18803", "http://engine:18800", ""]
)
def test_bound_overlap_checks_real_router_completion_header(
    bound_restart, routed_origin
):
    world = bound_restart
    world.routed_origin = routed_origin
    if routed_origin.endswith(":18803"):
        world.run()
        assert set(world.active) == {"102"}
        assert world.events[-2:] == [("cancel", "101"), "publish"]
    else:
        with pytest.raises(cli.click.ClickException, match="did not confirm"):
            world.run()
        assert set(world.active) == {"101"}
        assert ("cancel", "101") not in world.events
        assert ("cancel", "102") in world.events
    assert "pause" not in world.events and "resume" not in world.events
    assert ("completion", "http://router:18802/v1/chat/completions") in world.events


def test_bound_overlap_short_completion_never_releases_or_cancels_old(bound_restart):
    world = bound_restart
    world.tokens = 199
    with pytest.raises(cli.click.ClickException, match="validation failed"):
        world.run()
    assert set(world.active) == {"101"}
    assert world.events[-1] == ("cancel", "102")
    assert not any(
        isinstance(event, tuple)
        and event == ("completion", "http://router:18802/v1/chat/completions")
        for event in world.events
    )


@pytest.mark.parametrize("fallback", [True, False])
def test_pending_overlap_falls_back_only_in_auto(bound_restart, monkeypatch, fallback):
    from imas_ambix.agent import reload

    world = bound_restart
    world.pending = True
    clock = OverlapWorld()
    run_overlap = reload.run_overlap
    monkeypatch.setattr(
        reload,
        "run_overlap",
        lambda **kwargs: run_overlap(
            **kwargs, clock=lambda: clock.now, sleep=clock.sleep
        ),
    )
    if fallback:
        world.run(fallback=True)
        assert "pause" in world.events and "resume" in world.events
        assert ("submit", "pause", "103") in world.events
        assert world.events.index(("cancel", "102")) < world.events.index("pause")
        assert set(world.active) == {"103"}
    else:
        with pytest.raises(cli.click.ClickException, match="120 seconds"):
            world.run()
        assert set(world.active) == {"101"}
        assert "pause" not in world.events
    assert clock.now == 120


@pytest.mark.parametrize("mode", ["auto", "overlap", "pause"])
def test_restart_cli_wires_mode_and_launch_owned_hold(monkeypatch, mode):
    from imas_ambix.agent.profile import load_profile

    profile = load_profile("glm-5-3").for_gpus(4)
    old_job = {"jobid": "101", "name": profile.slug, "state": "RUNNING"}
    captured = {}
    selected = "pause" if mode == "pause" else "overlap"
    monkeypatch.setattr(cli, "_load_profile", lambda *_: profile)
    monkeypatch.setattr(cli, "_require_engine_environment", lambda *_: None)
    monkeypatch.setattr(cli, "_resolve_serve_auth", lambda *_: None)
    monkeypatch.setattr(cli, "_profile_jobs", lambda *_: [old_job])
    monkeypatch.setattr(
        cli,
        "_restart_mode",
        lambda _profile, _site, requested, _key: (
            (selected, "http://router:18802")
            if requested == mode
            else pytest.fail(requested)
        ),
    )
    monkeypatch.setattr(cli, "_overlap_port", lambda *_: 19444)
    monkeypatch.setattr(
        cli,
        "_execute_restart_cycle",
        lambda *args, **kwargs: captured.update(args=args, kwargs=kwargs),
    )
    options = [] if mode == "auto" else ["--mode", mode]
    result = CliRunner().invoke(
        main, ["agent", "restart", profile.slug, "--reason", "settings"] + options
    )
    assert result.exit_code == 0, result.output
    assert captured["kwargs"]["mode"] == selected
    assert ("registry write --held" in captured["args"][3]) is (selected == "overlap")
    assert (captured["kwargs"]["fallback_script"] is not None) is (mode == "auto")
    if selected == "overlap":
        assert captured["args"][5] == 19444
