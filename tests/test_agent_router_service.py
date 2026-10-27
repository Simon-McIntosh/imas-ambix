"""Standing router service submission tests."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import aiohttp
from aiohttp import web
from click.testing import CliRunner

from imas_ambix.agent import router as router_mod
from imas_ambix.agent import slurm as slurm_mod
from imas_ambix.agent.profile import SiteConfig, load_profile
from imas_ambix.cli import main


def test_router_script_is_cpu_only_and_uses_file_backed_admission(tmp_path):
    site = SiteConfig(
        base_dir=str(tmp_path),
        engine_env_root=str(tmp_path / "engine-envs"),
    )

    script = slurm_mod.generate_router_script(
        site,
        port=18802,
        cpus=3,
        memory="12G",
    )

    assert "#SBATCH --gres=gpu" not in script
    assert "#SBATCH --cpus-per-task=3" in script
    assert "#SBATCH --mem=12G" in script
    assert "#SBATCH --comment=ambix-router;port=18802" in script
    assert "#SBATCH --time=0" in script
    assert str(site.python_path("vllm")) in script
    assert "from imas_ambix.cli import main; main()" in script
    assert "agent router" in script
    assert "--api-key" not in script
    # Admission comes from router-gate.json rather than launch-only flags.
    assert "--max-in-flight" not in script
    assert "--max-queued" not in script


def test_lane_refresher_script_has_no_time_limit(tmp_path):
    site = SiteConfig(
        base_dir=str(tmp_path),
        engine_env_root=str(tmp_path / "engine-envs"),
    )

    script = slurm_mod.generate_lane_refresher_script(
        site,
        origin="http://127.0.0.1:18802",
    )

    assert "#SBATCH --time=0" in script


def test_engine_serve_script_remains_unlimited(tmp_path):
    site = SiteConfig(
        base_dir=str(tmp_path),
        engine_env_root=str(tmp_path / "engine-envs"),
    )

    script = slurm_mod.generate_serve_script(
        load_profile("deepseek-v4-1-flash"),
        site,
    )

    assert "#SBATCH --time=0" in script


def test_router_dry_run_needs_no_admission_flags(monkeypatch):
    """The operator file keeps admission independent of launch composition."""

    def refuse_submission(_script: str) -> str:
        raise AssertionError("dry-run must not submit")

    monkeypatch.setattr(slurm_mod, "submit_script", refuse_submission)
    result = CliRunner().invoke(
        main,
        ["agent", "router", "--port", "18802", "--dry-run"],
    )

    assert result.exit_code == 0
    assert "agent router" in result.output
    for retired in ("--max-in-flight", "--max-queued", "--retry-after-seconds"):
        assert retired not in result.output


def test_router_submit_uses_shared_submission_adapter(monkeypatch):
    captured: dict[str, str] = {}

    def submit(script: str) -> str:
        captured["script"] = script
        return "77"

    monkeypatch.setattr(slurm_mod, "submit_script", submit)
    result = CliRunner().invoke(
        main,
        ["agent", "router", "--port", "18802", "--submit"],
    )

    assert result.exit_code == 0
    assert "Submitted keyless router job 77" in result.output
    assert "#SBATCH --comment=ambix-router;port=18802" in captured["script"]


# --------------------------------------------------------------------------
# Shutdown release
#
# The release lives on the shutdown *signal*, and only a real signal to a real
# server exercises it: a test that called the lifespan handler, or the gate's
# release method, would pass whether or not uvicorn ever reaches it. These
# tests therefore start the router under its own uvicorn server in a
# subprocess and signal that process.
# --------------------------------------------------------------------------

REPO_ROOT = Path(router_mod.__file__).resolve().parents[2]

_ROUTER_LAUNCH = """
import sys
from pathlib import Path

from imas_ambix.agent.router import Upstream, serve_router


class Resolver:
    def __init__(self, url):
        self.url = url

    async def resolve(self):
        return [Upstream(self.url)]


serve_router(
    Resolver(sys.argv[1]),
    host="127.0.0.1",
    port=int(sys.argv[2]),
    gate_file=Path(sys.argv[3]),
    request_receipts_path=Path(sys.argv[4]),
    timeout_graceful_shutdown=float(sys.argv[5]),
)
"""


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _write_gate(path: Path, *, width: int, wait_seconds: float) -> None:
    path.write_text(
        json.dumps({"width": width, "wait_seconds": wait_seconds}), encoding="utf-8"
    )


def _subprocess_env() -> dict[str, str]:
    """The child's environment, with this tree's package on its import path.

    The child runs the same interpreter as the test, but its imports resolve
    through that interpreter's installed packages rather than through the root
    directory pytest inserts, so the tree under test is named explicitly.
    Without it a run inside a worktree would exercise the installed checkout
    instead of the code being measured.
    """
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{REPO_ROOT}{os.pathsep}{existing}" if existing else str(REPO_ROOT)
    )
    return env


class _Upstream:
    """A fake engine whose generation ends only when the test says so."""

    def __init__(self) -> None:
        self.labels: list[str] = []
        self.release = asyncio.Event()

    async def catalog(self, _: web.Request) -> web.Response:
        return web.json_response(
            {
                "object": "list",
                "data": [
                    {
                        "id": "streamer",
                        "object": "model",
                        "created": 123,
                        "owned_by": "engine",
                        "max_model_len": 4096,
                    }
                ],
            }
        )

    async def generate(self, request: web.Request) -> web.Response:
        payload = await request.json()
        label = str(payload.get("label", "request"))
        self.labels.append(label)
        await self.release.wait()
        return web.json_response({"label": label})

    async def wait_for(self, count: int, *, timeout: float = 10.0) -> None:
        deadline = time.monotonic() + timeout
        while len(self.labels) < count:
            if time.monotonic() > deadline:
                raise AssertionError(f"upstream saw {self.labels}, wanted {count}")
            await asyncio.sleep(0.02)


@asynccontextmanager
async def _upstream_server(upstream: _Upstream):
    app = web.Application()
    app.router.add_get("/v1/models", upstream.catalog)
    app.router.add_post("/v1/messages", upstream.generate)
    app.router.add_post("/v1/chat/completions", upstream.generate)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", _free_port())
    await site.start()
    try:
        port = site._server.sockets[0].getsockname()[1]
        yield f"http://127.0.0.1:{port}"
    finally:
        await runner.cleanup()


@asynccontextmanager
async def _router_process(
    upstream: str, gate_file: Path, receipts: Path, grace_seconds: float
):
    port = _free_port()
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            _ROUTER_LAUNCH,
            upstream,
            str(port),
            str(gate_file),
            str(receipts),
            str(grace_seconds),
        ],
        env=_subprocess_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        await _await_ready(base, process)
        yield process, base
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=30)


async def _await_ready(base: str, process: subprocess.Popen) -> None:
    deadline = time.monotonic() + 30.0
    async with aiohttp.ClientSession() as session:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise AssertionError(
                    "router exited before answering: "
                    + (process.stdout.read() if process.stdout else "")
                )
            try:
                async with session.get(f"{base}/v1/models") as response:
                    if response.status == 200:
                        return
            except aiohttp.ClientError:
                pass
            await asyncio.sleep(0.05)
    raise AssertionError("router never answered /v1/models")


def _exited_after_shutdown(process: subprocess.Popen) -> bool:
    """Say whether the server finished its shutdown and then ended.

    uvicorn restores the original signal handler once its own shutdown has
    completed and re-raises the captured signal, so that the parent sees the
    signal that stopped the process rather than a bare zero. Both forms are a
    completed shutdown; neither is a server that died without answering.
    """
    return process.wait(timeout=30) in (0, -signal.SIGTERM)


def _receipt_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_shutdown_signal_answers_every_gate_waiter_with_overload(tmp_path) -> None:
    async def exercise() -> None:
        gate_file = tmp_path / "router-gate.json"
        receipts = tmp_path / "requests.jsonl"
        _write_gate(gate_file, width=1, wait_seconds=60.0)
        upstream = _Upstream()
        async with (
            _upstream_server(upstream) as upstream_url,
            _router_process(upstream_url, gate_file, receipts, 30.0) as (
                process,
                base,
            ),
            aiohttp.ClientSession() as client,
        ):
            first = asyncio.create_task(
                client.post(
                    f"{base}/v1/messages",
                    json={"model": "streamer", "label": "relaying"},
                )
            )
            # The first request takes the only slot; the three behind it sit in
            # the FIFO, which is the state a shutdown must answer.
            await upstream.wait_for(1)
            held = [
                asyncio.create_task(
                    client.post(
                        f"{base}/v1/messages",
                        json={"model": "streamer", "label": f"wait-{index}"},
                    )
                )
                for index in range(3)
            ]
            await asyncio.sleep(1.0)
            assert all(not task.done() for task in held), (
                "a waiter was answered before the signal; the gate was not full"
            )

            process.send_signal(signal.SIGTERM)
            # The relay already in flight finishes inside the grace.
            upstream.release.set()

            responses = await asyncio.gather(first, *held)
            statuses = [response.status for response in responses]
            retry_after = [
                response.headers.get("Retry-After") for response in responses[1:]
            ]
            bodies = [await response.json() for response in responses]
            for response in responses:
                response.close()

            assert statuses[0] == 200
            assert statuses[1:] == [529, 529, 529]
            assert retry_after == ["5", "5", "5"]
            assert {body["error"]["type"] for body in bodies[1:]} == {
                "overloaded_error"
            }

            assert _exited_after_shutdown(process)

        rows = _receipt_rows(receipts)
        expired = [row for row in rows if row["status"] == "failed"]
        assert len(expired) == 3, rows
        assert {row["reason"] for row in expired} == {router_mod.SHUTDOWN_REASON}
        # The readiness probe is itself a completed row answered by the router;
        # the relay that finished inside the grace is the other one.
        finished = [
            row
            for row in rows
            if row["status"] == "completed" and row["model"] == "streamer"
        ]
        assert len(finished) == 1, rows

    asyncio.run(exercise())


def test_relay_outlasting_the_grace_is_recorded_as_cut(tmp_path) -> None:
    async def exercise() -> None:
        gate_file = tmp_path / "router-gate.json"
        receipts = tmp_path / "requests.jsonl"
        _write_gate(gate_file, width=1, wait_seconds=60.0)
        upstream = _Upstream()
        async with (
            _upstream_server(upstream) as upstream_url,
            _router_process(upstream_url, gate_file, receipts, 0.5) as (process, base),
            aiohttp.ClientSession() as client,
        ):
            relaying = asyncio.create_task(
                client.post(
                    f"{base}/v1/messages",
                    json={"model": "streamer", "label": "long"},
                )
            )
            await upstream.wait_for(1)
            process.send_signal(signal.SIGTERM)
            # The upstream stays blocked until the process has ended, so the
            # relay cannot finish inside the half-second grace.
            assert _exited_after_shutdown(process)
            upstream.release.set()
            await asyncio.gather(relaying, return_exceptions=True)

        rows = _receipt_rows(receipts)
        cut = [row for row in rows if row["status"] == "aborted"]
        assert len(cut) == 1, rows
        assert "shutdown" in (cut[0]["reason"] or ""), cut
        assert "cut" in (cut[0]["reason"] or ""), cut

    asyncio.run(exercise())
