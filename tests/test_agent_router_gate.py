from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from aiohttp import web

from imas_ambix.agent import router as router_mod
from imas_ambix.agent.router import RouterApp, Upstream, engine_id_for_origin
from tests.test_request_receipts import _read_rows

AsgiMessage = dict[str, Any]


class Resolver:
    def __init__(self, upstreams: Sequence[Upstream]) -> None:
        self.upstreams = upstreams

    async def resolve(self) -> Sequence[Upstream]:
        return self.upstreams


def _card() -> dict[str, Any]:
    return {
        "id": "streamer",
        "object": "model",
        "created": 123,
        "owned_by": "engine",
        "max_model_len": 4096,
    }


def _write_gate(
    path: Path,
    *,
    width: int,
    wait_seconds: float = 1.0,
    paused: bool = False,
    reason: str | None = None,
) -> None:
    payload: dict[str, Any] = {"width": width, "wait_seconds": wait_seconds}
    if paused:
        payload["paused"] = True
        payload["reason"] = reason
    path.write_text(json.dumps(payload), encoding="utf-8")


async def _await_gate(
    predicate: Callable[[], bool], *, timeout: float = 3.0
) -> None:
    """Poll a gate predicate, because the config cache is refreshed once a second."""
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("gate condition was not observed within the timeout")


def _write_auto_gate(
    path: Path,
    *,
    occupancy_target: float = 0.90,
    width_floor: int = 16,
    width_cap: int = 36,
    wait_seconds: float = 1.0,
) -> None:
    path.write_text(
        json.dumps(
            {
                "width": "auto",
                "occupancy_target": occupancy_target,
                "width_floor": width_floor,
                "width_cap": width_cap,
                "wait_seconds": wait_seconds,
            }
        ),
        encoding="utf-8",
    )


@asynccontextmanager
async def _server(app: web.Application):
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    sockets = site._server.sockets
    try:
        yield f"http://127.0.0.1:{sockets[0].getsockname()[1]}"
    finally:
        await runner.cleanup()


@asynccontextmanager
async def _router(upstream: str, gate_file: Path):
    app = RouterApp(
        Resolver([Upstream(upstream)]),
        lane_document=gate_file.with_name("lane.json"),
        gate_file=gate_file,
    )
    try:
        yield app
    finally:
        if app._session is not None:
            await app._session.close()


def _start_call(
    app: RouterApp,
    method: str,
    path: str,
    body: bytes = b"",
    *,
    on_send: Callable[[AsgiMessage, asyncio.Queue[AsgiMessage]], Awaitable[None]]
    | None = None,
) -> tuple[asyncio.Task[None], asyncio.Queue[AsgiMessage], list[AsgiMessage]]:
    incoming: asyncio.Queue[AsgiMessage] = asyncio.Queue()
    incoming.put_nowait({"type": "http.request", "body": body})
    sent: list[AsgiMessage] = []

    async def receive() -> AsgiMessage:
        return await incoming.get()

    async def send(message: AsgiMessage) -> None:
        sent.append(message)
        if on_send is not None:
            await on_send(message, incoming)

    task = asyncio.create_task(
        app(
            {
                "type": "http",
                "method": method,
                "path": path,
                "raw_path": path.encode(),
                "query_string": b"",
                "headers": [(b"content-type", b"application/json")],
            },
            receive,
            send,
        )
    )
    return task, incoming, sent


async def _call(
    app: RouterApp, method: str, path: str, body: bytes = b""
) -> list[AsgiMessage]:
    task, _, sent = _start_call(app, method, path, body)
    await task
    return sent


def _status(messages: Sequence[AsgiMessage]) -> int:
    return next(
        message["status"]
        for message in messages
        if message["type"] == "http.response.start"
    )


def _body(messages: Sequence[AsgiMessage]) -> bytes:
    return b"".join(
        message.get("body", b"")
        for message in messages
        if message["type"] == "http.response.body"
    )


def _headers(messages: Sequence[AsgiMessage]) -> dict[bytes, bytes]:
    start = next(
        message for message in messages if message["type"] == "http.response.start"
    )
    return {name.lower(): value for name, value in start["headers"]}


class GenerationProbe:
    def __init__(self) -> None:
        self.active = 0
        self.maximum = 0
        self.started: list[str] = []
        self.changed = asyncio.Condition()
        self.release = asyncio.Event()

    async def handler(self, request: web.Request) -> web.Response:
        payload = await request.json()
        label = str(payload.get("label", "request"))
        async with self.changed:
            self.active += 1
            self.maximum = max(self.maximum, self.active)
            self.started.append(label)
            self.changed.notify_all()
        try:
            await self.release.wait()
            return web.json_response({"label": label})
        finally:
            async with self.changed:
                self.active -= 1
                self.changed.notify_all()

    async def wait_for_active(self, count: int, *, timeout: float = 1.0) -> None:
        async with self.changed:
            await asyncio.wait_for(
                self.changed.wait_for(lambda: self.active >= count), timeout=timeout
            )


def _probe_app(probe: GenerationProbe) -> web.Application:
    app = web.Application()

    async def catalog(_: web.Request) -> web.Response:
        return web.json_response({"object": "list", "data": [_card()]})

    async def count_tokens(_: web.Request) -> web.Response:
        return web.json_response({"input_tokens": 7})

    app.router.add_get("/v1/models", catalog)
    app.router.add_post("/v1/messages/count_tokens", count_tokens)
    app.router.add_post("/v1/messages", probe.handler)
    app.router.add_post("/v1/chat/completions", probe.handler)
    return app


def test_global_gate_is_fifo_and_bounds_generation_without_refusal(tmp_path) -> None:
    async def exercise() -> None:
        gate_file = tmp_path / "router-gate.json"
        _write_gate(gate_file, width=2)
        probe = GenerationProbe()
        async with (
            _server(_probe_app(probe)) as upstream,
            _router(upstream, gate_file) as app,
        ):
            calls: list[tuple[asyncio.Task[None], list[AsgiMessage]]] = []
            for index in range(5):
                task, _, sent = _start_call(
                    app,
                    "POST",
                    "/v1/messages",
                    json.dumps({"model": "streamer", "label": str(index)}).encode(),
                )
                calls.append((task, sent))
                await asyncio.sleep(0.01)

            await probe.wait_for_active(2)
            await asyncio.sleep(0.05)
            maximum = probe.maximum
            probe.release.set()
            await asyncio.gather(*(task for task, _ in calls))

        assert maximum == 2
        assert probe.started == [str(index) for index in range(5)]
        assert [_status(sent) for _, sent in calls] == [200] * 5

    asyncio.run(exercise())


def test_wait_bound_returns_anthropic_overload_with_retry_after(tmp_path) -> None:
    async def exercise() -> None:
        gate_file = tmp_path / "router-gate.json"
        _write_gate(gate_file, width=1, wait_seconds=0.05)
        probe = GenerationProbe()
        async with (
            _server(_probe_app(probe)) as upstream,
            _router(upstream, gate_file) as app,
        ):
            first, _, _ = _start_call(
                app,
                "POST",
                "/v1/messages",
                b'{"model":"streamer","label":"first"}',
            )
            await probe.wait_for_active(1)
            refused = await asyncio.wait_for(
                _call(
                    app,
                    "POST",
                    "/v1/messages",
                    b'{"model":"streamer","label":"second"}',
                ),
                timeout=0.5,
            )
            try:
                assert _status(refused) == 529
                assert _headers(refused)[b"retry-after"] == b"5"
                assert json.loads(_body(refused)) == {
                    "type": "error",
                    "error": {
                        "type": "overloaded_error",
                        "message": "router generation queue wait limit exceeded",
                    },
                }
            finally:
                probe.release.set()
                await first

    asyncio.run(exercise())


def test_gate_configuration_is_checked_once_per_second_gate_wide(
    tmp_path, monkeypatch, caplog
) -> None:
    gate_file = tmp_path / "router-gate.json"
    _write_gate(gate_file, width=1)
    clock = [100.0]
    stat_calls = 0
    real_stat = Path.stat

    def count_stat(path, *args, **kwargs):
        nonlocal stat_calls
        if path == gate_file:
            stat_calls += 1
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(router_mod.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(Path, "stat", count_stat)
    app = RouterApp(Resolver([]), gate_file=gate_file)

    with caplog.at_level(logging.INFO, logger=router_mod.__name__):
        assert [app._generation_gate.settings().width for _ in range(30)] == [1] * 30
        clock[0] = 101.001
        assert app._generation_gate.settings().width == 1
        clock[0] = 102.002
        assert app._generation_gate.settings().width == 1

        config_messages = [
            record.getMessage()
            for record in caplog.records
            if record.getMessage().startswith("generation gate config path=")
        ]
        assert len(config_messages) == 1
        assert stat_calls == 3

        _write_gate(gate_file, width=2)
        clock[0] = 103.003
        assert app._generation_gate.settings().width == 2

    config_messages = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("generation gate config path=")
    ]
    assert len(config_messages) == 2
    assert "width=1" in config_messages[0]
    assert "width=2" in config_messages[1]
    assert stat_calls == 4


def test_missing_gate_file_uses_five_minute_wait_default(tmp_path) -> None:
    app = RouterApp(Resolver([]), gate_file=tmp_path / "missing-gate.json")

    settings = app._generation_gate.settings()

    assert settings.width == 22
    assert settings.wait_seconds == 300.0


def test_disconnected_waiter_and_streamer_release_capacity(tmp_path) -> None:
    async def exercise_waiter() -> None:
        gate_file = tmp_path / "waiter-gate.json"
        _write_gate(gate_file, width=1)
        probe = GenerationProbe()
        async with (
            _server(_probe_app(probe)) as upstream,
            _router(upstream, gate_file) as app,
        ):
            first, _, _ = _start_call(
                app,
                "POST",
                "/v1/messages",
                b'{"model":"streamer","label":"first"}',
            )
            await probe.wait_for_active(1)
            waiter, incoming, waiter_sent = _start_call(
                app,
                "POST",
                "/v1/messages",
                b'{"model":"streamer","label":"departed"}',
            )
            await asyncio.sleep(0.05)
            follower, _, follower_sent = _start_call(
                app,
                "POST",
                "/v1/messages",
                b'{"model":"streamer","label":"follower"}',
            )
            incoming.put_nowait({"type": "http.disconnect"})
            try:
                await asyncio.wait_for(waiter, timeout=0.5)
                assert waiter_sent == []
            finally:
                probe.release.set()
            await asyncio.wait_for(
                asyncio.gather(first, follower, return_exceptions=True), timeout=1
            )
            assert _status(follower_sent) == 200
            assert "departed" not in probe.started

    async def exercise_streamer() -> None:
        gate_file = tmp_path / "streamer-gate.json"
        _write_gate(gate_file, width=1)
        stream_open = asyncio.Event()
        keep_streaming = asyncio.Event()

        async def catalog(_: web.Request) -> web.Response:
            return web.json_response({"object": "list", "data": [_card()]})

        async def generate(request: web.Request) -> web.StreamResponse:
            payload = await request.json()
            if payload.get("label") != "stream":
                return web.json_response({"label": payload.get("label")})
            response = web.StreamResponse(headers={"content-type": "text/event-stream"})
            await response.prepare(request)
            await response.write(b"data: first\n\n")
            stream_open.set()
            try:
                await keep_streaming.wait()
                await response.write(b"data: late\n\n")
                await response.write_eof()
            except ConnectionError, RuntimeError:
                pass
            return response

        engine = web.Application()
        engine.router.add_get("/v1/models", catalog)
        engine.router.add_post("/v1/messages", generate)

        disconnected = False

        async def leave_after_first(
            message: AsgiMessage, incoming: asyncio.Queue[AsgiMessage]
        ) -> None:
            nonlocal disconnected
            if (
                message["type"] == "http.response.body"
                and message.get("body")
                and not disconnected
            ):
                disconnected = True
                incoming.put_nowait({"type": "http.disconnect"})

        async with (
            _server(engine) as upstream,
            _router(upstream, gate_file) as app,
        ):
            streamer, _, _ = _start_call(
                app,
                "POST",
                "/v1/messages",
                b'{"model":"streamer","label":"stream"}',
                on_send=leave_after_first,
            )
            await asyncio.wait_for(stream_open.wait(), timeout=1)
            await asyncio.wait_for(streamer, timeout=1)
            following = await asyncio.wait_for(
                _call(
                    app,
                    "POST",
                    "/v1/messages",
                    b'{"model":"streamer","label":"following"}',
                ),
                timeout=1,
            )
            assert _status(following) == 200
            keep_streaming.set()

    asyncio.run(exercise_waiter())
    asyncio.run(exercise_streamer())


def test_gate_file_edit_changes_width_without_restarting_app(tmp_path) -> None:
    async def exercise() -> None:
        gate_file = tmp_path / "router-gate.json"
        _write_gate(gate_file, width=1)
        probe = GenerationProbe()
        async with (
            _server(_probe_app(probe)) as upstream,
            _router(upstream, gate_file) as app,
        ):
            first, _, _ = _start_call(
                app,
                "POST",
                "/v1/messages",
                b'{"model":"streamer","label":"first"}',
            )
            await probe.wait_for_active(1)
            second, _, _ = _start_call(
                app,
                "POST",
                "/v1/messages",
                b'{"model":"streamer","label":"second"}',
            )
            await asyncio.sleep(0.05)
            assert probe.maximum == 1
            _write_gate(gate_file, width=2)
            await probe.wait_for_active(2, timeout=1.5)
            assert probe.maximum == 2
            probe.release.set()
            await asyncio.gather(first, second)

    asyncio.run(exercise())


def test_catalog_and_count_tokens_bypass_a_full_generation_gate(tmp_path) -> None:
    async def exercise() -> None:
        gate_file = tmp_path / "router-gate.json"
        _write_gate(gate_file, width=1)
        probe = GenerationProbe()
        async with (
            _server(_probe_app(probe)) as upstream,
            _router(upstream, gate_file) as app,
        ):
            blocker, _, _ = _start_call(
                app,
                "POST",
                "/v1/messages",
                b'{"model":"streamer","label":"blocker"}',
            )
            await probe.wait_for_active(1)
            catalog, tokens = await asyncio.wait_for(
                asyncio.gather(
                    _call(app, "GET", "/v1/models"),
                    _call(
                        app,
                        "POST",
                        "/v1/messages/count_tokens",
                        b'{"model":"streamer"}',
                    ),
                ),
                timeout=0.5,
            )
            assert _status(catalog) == 200
            assert _status(tokens) == 200
            assert json.loads(_body(tokens)) == {"input_tokens": 7}
            probe.release.set()
            await blocker

    asyncio.run(exercise())


def test_lane_document_publishes_nonzero_gate_counts(tmp_path) -> None:
    async def exercise() -> None:
        gate_file = tmp_path / "router-gate.json"
        lane_document = tmp_path / "lane.json"
        _write_gate(gate_file, width=1)
        lane_document.write_text(
            json.dumps({"running": 9, "waiting": 2}), encoding="utf-8"
        )
        app = RouterApp(Resolver([]), lane_document=lane_document, gate_file=gate_file)
        first_incoming: asyncio.Queue[AsgiMessage] = asyncio.Queue()
        second_incoming: asyncio.Queue[AsgiMessage] = asyncio.Queue()

        async def receive_first() -> AsgiMessage:
            return await first_incoming.get()

        async def receive_second() -> AsgiMessage:
            return await second_incoming.get()

        first = await app._generation_gate.acquire(receive_first)
        second = asyncio.create_task(app._generation_gate.acquire(receive_second))
        for _ in range(20):
            if app._generation_gate.waiting == 1:
                break
            await asyncio.sleep(0.01)
        assert app._generation_gate.waiting == 1

        app._publish_gate_snapshot()
        published = json.loads(lane_document.read_text(encoding="utf-8"))
        assert published["running"] == 9
        assert published["waiting"] == 2
        assert published["router_generation_gate"] == {
            "config_path": str(gate_file),
            "context_estimate": None,
            "effective_width": 1,
            "enabled": True,
            "in_flight": 1,
            "paused": False,
            "reason": None,
            "wait_seconds": 1.0,
            "waiting": 1,
            "width": 1,
        }

        second_incoming.put_nowait({"type": "http.disconnect"})
        departed = await second
        assert departed.outcome == "disconnected"
        if first.disconnect_task is not None:
            first.disconnect_task.cancel()
            await asyncio.gather(first.disconnect_task, return_exceptions=True)
        await app._generation_gate.release()

    asyncio.run(exercise())


def test_auto_width_follows_pool_and_context_within_its_clamp(tmp_path) -> None:
    gate_file = tmp_path / "router-gate.json"
    _write_auto_gate(gate_file)

    # At a 4,000,000-token pool and a 0.90 target, floor(pool * target / ctx)
    # hits the cap for a small context and the floor for a large one.
    cases = [(500_000, 16), (200_000, 18), (20_000, 36)]
    for context, expected in cases:
        gate = router_mod._GenerationGate(gate_file)
        gate.observe_lane(4_000_000, context, now=0.0)
        assert gate.settings().width == expected, (context, expected)


def test_auto_width_estimate_is_slow_enough_to_absorb_an_outlier(tmp_path) -> None:
    assert router_mod.DEFAULT_AUTO_CONTEXT_TIME_CONSTANT_SECONDS >= 600.0
    gate_file = tmp_path / "router-gate.json"
    _write_auto_gate(gate_file)
    gate = router_mod._GenerationGate(gate_file)

    gate.observe_lane(4_000_000, 100_000, now=0.0)
    assert gate._context_estimate == 100_000.0
    # One reading twice the level, one lane interval later, must move the
    # estimate by less than a tenth rather than to the reading itself.
    gate.observe_lane(4_000_000, 200_000, now=30.0)
    assert 100_000.0 < gate._context_estimate < 110_000.0


def test_auto_width_holds_last_width_without_a_reading(tmp_path, monkeypatch) -> None:
    gate_file = tmp_path / "router-gate.json"
    _write_auto_gate(gate_file)
    clock = [1000.0]
    monkeypatch.setattr(router_mod.time, "monotonic", lambda: clock[0])
    gate = router_mod._GenerationGate(gate_file)

    # Before any reading the floor stands, and a missing key is never read as
    # zero: the gate would otherwise admit nothing at all.
    assert gate.settings().width == 16

    gate.observe_lane(4_000_000, 200_000, now=clock[0])
    clock[0] += 2.0
    assert gate.settings().width == 18

    # An idle lane reports no context and a reading without a pool size reports
    # no shape; both are held across rather than folded in.
    gate.observe_lane(4_000_000, None, now=clock[0])
    clock[0] += 2.0
    assert gate.settings().width == 18
    gate.observe_lane(None, 50_000, now=clock[0])
    clock[0] += 2.0
    assert gate.settings().width == 18


def test_integer_width_ignores_lane_readings(tmp_path) -> None:
    gate_file = tmp_path / "router-gate.json"
    _write_gate(gate_file, width=2)
    gate = router_mod._GenerationGate(gate_file)

    # A context that would size the auto mode to the floor leaves a fixed
    # integer width exactly as it was configured.
    gate.observe_lane(4_000_000, 500_000, now=0.0)
    assert gate.settings().width == 2


def test_auto_width_publishes_effective_width_and_estimate(tmp_path) -> None:
    async def exercise() -> None:
        gate_file = tmp_path / "router-gate.json"
        lane_document = tmp_path / "lane.json"
        _write_auto_gate(gate_file)
        lane_document.write_text(
            json.dumps({"running": 9, "waiting": 2}), encoding="utf-8"
        )
        app = RouterApp(Resolver([]), lane_document=lane_document, gate_file=gate_file)
        app._generation_gate.observe_lane(4_000_000, 200_000, now=0.0)

        app._publish_gate_snapshot()
        published = json.loads(lane_document.read_text(encoding="utf-8"))
        gate_snapshot = published["router_generation_gate"]
        assert gate_snapshot["width"] == 18
        assert gate_snapshot["effective_width"] == 18
        assert gate_snapshot["context_estimate"] == 200_000.0

    asyncio.run(exercise())


def test_paused_gate_admits_nothing_new_while_in_flight_completes(tmp_path) -> None:
    async def exercise() -> None:
        gate_file = tmp_path / "router-gate.json"
        _write_gate(gate_file, width=2, wait_seconds=2.0)
        probe = GenerationProbe()
        async with (
            _server(_probe_app(probe)) as upstream,
            _router(upstream, gate_file) as app,
        ):
            try:
                first, _, first_sent = _start_call(
                    app,
                    "POST",
                    "/v1/messages",
                    b'{"model":"streamer","label":"first"}',
                )
                await probe.wait_for_active(1)

                # The pause is declared while one relay is in flight. Width
                # stays 2, so a gate that ignored the pause would admit a
                # second relay at once.
                _write_gate(
                    gate_file,
                    width=2,
                    wait_seconds=2.0,
                    paused=True,
                    reason="draining for a relaunch",
                )
                await _await_gate(lambda: app._generation_gate.settings().paused)

                second, _, second_sent = _start_call(
                    app,
                    "POST",
                    "/v1/messages",
                    b'{"model":"streamer","label":"second"}',
                )
                third, _, third_sent = _start_call(
                    app,
                    "POST",
                    "/v1/messages",
                    b'{"model":"streamer","label":"third"}',
                )
                await _await_gate(lambda: app._generation_gate.waiting == 2)

                # Nothing new joined while paused, even though the width would
                # allow it, and the in-flight relay is untouched by the pause.
                assert probe.maximum == 1
                assert probe.started == ["first"]

                probe.release.set()
                await asyncio.wait_for(first, timeout=1)
                assert _status(first_sent) == 200
                assert probe.maximum == 1
                assert probe.started == ["first"]

                # Clearing the pause admits the held requests, in arrival order.
                _write_gate(gate_file, width=2, wait_seconds=2.0)
                await _await_gate(
                    lambda: not app._generation_gate.settings().paused
                )
                await asyncio.wait_for(asyncio.gather(second, third), timeout=2)

                assert probe.started == ["first", "second", "third"]
                assert _status(second_sent) == 200
                assert _status(third_sent) == 200
            finally:
                # A regression that lets a paused gate admit would leave the
                # held relays parked here and stall the run instead of failing
                # it, so the probe is always released.
                probe.release.set()

    asyncio.run(exercise())


def test_paused_gate_times_out_a_held_request_with_retry_after(tmp_path) -> None:
    async def exercise() -> None:
        gate_file = tmp_path / "router-gate.json"
        _write_gate(
            gate_file,
            width=4,
            wait_seconds=0.05,
            paused=True,
            reason="draining for a relaunch",
        )
        probe = GenerationProbe()
        async with (
            _server(_probe_app(probe)) as upstream,
            _router(upstream, gate_file) as app,
        ):
            try:
                refused = await asyncio.wait_for(
                    _call(
                        app,
                        "POST",
                        "/v1/messages",
                        b'{"model":"streamer","label":"held"}',
                    ),
                    timeout=1,
                )
                assert _status(refused) == 529
                assert _headers(refused)[b"retry-after"] == b"5"
                assert json.loads(_body(refused)) == {
                    "type": "error",
                    "error": {
                        "type": "overloaded_error",
                        "message": "router generation queue wait limit exceeded",
                    },
                }
                assert probe.started == []
            finally:
                # Under a regression the relay is admitted and parks on the
                # probe; release it so the failure is reported rather than hung.
                probe.release.set()

    asyncio.run(exercise())


def test_pause_outranks_a_disabled_width(tmp_path) -> None:
    async def exercise() -> None:
        gate_file = tmp_path / "router-gate.json"
        _write_gate(
            gate_file,
            width=0,
            wait_seconds=0.05,
            paused=True,
            reason="draining for a relaunch",
        )
        probe = GenerationProbe()
        async with (
            _server(_probe_app(probe)) as upstream,
            _router(upstream, gate_file) as app,
        ):
            refused = await asyncio.wait_for(
                _call(
                    app,
                    "POST",
                    "/v1/messages",
                    b'{"model":"streamer","label":"held"}',
                ),
                timeout=1,
            )
            assert _status(refused) == 529
            assert probe.started == []

    asyncio.run(exercise())


def test_lane_snapshot_publishes_the_pause_and_its_reason(tmp_path) -> None:
    gate_file = tmp_path / "router-gate.json"
    lane_document = tmp_path / "lane.json"
    _write_gate(
        gate_file,
        width=1,
        paused=True,
        reason="draining for a relaunch",
    )
    lane_document.write_text(
        json.dumps({"running": 3, "waiting": 0}), encoding="utf-8"
    )
    app = RouterApp(Resolver([]), lane_document=lane_document, gate_file=gate_file)

    app._publish_gate_snapshot()
    published = json.loads(lane_document.read_text(encoding="utf-8"))
    gate_snapshot = published["router_generation_gate"]
    assert gate_snapshot["paused"] is True
    assert gate_snapshot["reason"] == "draining for a relaunch"
    assert published["running"] == 3


def _model_card(model_id: str) -> dict[str, Any]:
    return {
        "id": model_id,
        "object": "model",
        "created": 123,
        "owned_by": "engine",
        "max_model_len": 4096,
    }


class _HeldEngine:
    """An engine that holds every generation until released, recording arrivals."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.arrivals: list[float] = []
        self.running = 0
        self.peak = 0
        self.release = asyncio.Event()

    def application(self) -> web.Application:
        app = web.Application()

        async def catalog(_: web.Request) -> web.Response:
            return web.json_response(
                {"object": "list", "data": [_model_card(self.model_id)]}
            )

        async def generate(_: web.Request) -> web.Response:
            self.arrivals.append(time.monotonic())
            self.running += 1
            self.peak = max(self.peak, self.running)
            try:
                await self.release.wait()
            finally:
                self.running -= 1
            return web.json_response({"choices": [{"message": {"content": "ok"}}]})

        app.router.add_get("/v1/models", catalog)
        app.router.add_post("/v1/messages", generate)
        return app


def _idle_receive() -> Callable[[], Awaitable[AsgiMessage]]:
    """A receive that never delivers, so an acquisition is judged on the gate."""
    queue: asyncio.Queue[AsgiMessage] = asyncio.Queue()

    async def receive() -> AsgiMessage:
        return await queue.get()

    return receive


def test_two_engines_admit_at_their_own_widths(tmp_path: Path) -> None:
    """Each gate admits its own engine's width, and a full gate delays nobody else.

    Two engines of different widths receive six concurrent requests each. The
    narrow gate admits its width and holds the rest; the wide gate fills to its
    own width *while the narrow gate is already full*, which a single shared
    queue at either width could not produce. That is the whole point of a gate per
    engine: capacity is a property of the engine, not of the lane.
    """

    async def exercise() -> None:
        narrow = _HeldEngine("narrow-model")
        wide = _HeldEngine("wide-model")
        async with (
            _server(narrow.application()) as narrow_url,
            _server(wide.application()) as wide_url,
        ):
            narrow_key = engine_id_for_origin(narrow_url)
            wide_key = engine_id_for_origin(wide_url)
            gate_file = tmp_path / "router-gate.json"
            gate_file.write_text(
                json.dumps(
                    {
                        "width": 1,
                        "wait_seconds": 10.0,
                        "engines": {
                            narrow_key: {"width": 2},
                            wide_key: {"width": 4},
                        },
                    }
                ),
                encoding="utf-8",
            )
            app = RouterApp(
                Resolver([Upstream(narrow_url), Upstream(wide_url)]),
                gate_file=gate_file,
            )
            try:
                narrow_calls = [
                    _start_call(
                        app,
                        "POST",
                        "/v1/messages",
                        json.dumps({"model": "narrow-model", "seq": index}).encode(),
                    )[0]
                    for index in range(6)
                ]
                wide_calls = [
                    _start_call(
                        app,
                        "POST",
                        "/v1/messages",
                        json.dumps({"model": "wide-model", "seq": index}).encode(),
                    )[0]
                    for index in range(6)
                ]

                await _await_gate(
                    lambda: narrow.running
                    == 2,
                    timeout=5.0,
                )
                await _await_gate(
                    lambda: wide.running == 4,
                    timeout=5.0,
                )

                narrow_gate = app.gate_for(narrow_key)
                wide_gate = app.gate_for(wide_key)
                assert narrow_gate is not wide_gate
                assert narrow_gate.in_flight == 2
                assert narrow_gate.waiting == 4, narrow_gate.waiting
                assert wide_gate.in_flight == 4
                assert wide_gate.waiting == 2, wide_gate.waiting

                # Reached at its own width while the narrow gate was saturated.
                assert narrow.peak == 2
                assert wide.peak == 4

                narrow.release.set()
                wide.release.set()
                for task in narrow_calls + wide_calls:
                    await asyncio.wait_for(task, timeout=10)
            finally:
                if app._session is not None:
                    await app._session.close()

    asyncio.run(exercise())


def test_a_throughput_reading_from_one_engine_moves_only_its_own_width(
    tmp_path: Path,
) -> None:
    """A reading taken at one engine sizes that engine and no other."""

    async def exercise() -> None:
        gate_file = tmp_path / "router-gate.json"
        gate_file.write_text(
            json.dumps(
                {
                    "width": "auto",
                    "occupancy_target": 0.90,
                    "width_floor": 16,
                    "width_cap": 36,
                    "wait_seconds": 1.0,
                }
            ),
            encoding="utf-8",
        )
        app = RouterApp(Resolver([]), gate_file=gate_file)
        first = Upstream("http://127.0.0.1:21001")
        second = Upstream("http://127.0.0.1:21002")
        first_gate = app.gate_for(engine_id_for_origin(first.base_url))
        second_gate = app.gate_for(engine_id_for_origin(second.base_url))
        assert first_gate.settings().width == 16
        assert second_gate.settings().width == 16

        app.observe_engine(
            first,
            SimpleNamespace(
                pool_tokens=1_000_000,
                mean_context=1000,
                generation_tokens=None,
                running=3,
                prefix_hit_rate=None,
                kv_occupancy=None,
            ),
        )

        await _await_gate(lambda: first_gate.settings().width == 36)
        assert second_gate.settings().width == 16

    asyncio.run(exercise())


def test_gate_expiry_names_the_engine_it_waited_at(tmp_path: Path) -> None:
    """A refusal row keeps the router sentinel and names the gate's engine."""

    async def exercise() -> None:
        engine = _HeldEngine("streamer")
        receipts = tmp_path / "receipts.jsonl"
        async with _server(engine.application()) as origin:
            key = engine_id_for_origin(origin)
            gate_file = tmp_path / "router-gate.json"
            gate_file.write_text(
                json.dumps(
                    {"width": 1, "wait_seconds": 0.3, "engines": {key: {"width": 1}}}
                ),
                encoding="utf-8",
            )
            app = RouterApp(
                Resolver([Upstream(origin)]),
                gate_file=gate_file,
                request_receipts_path=receipts,
            )
            try:
                gate = app.gate_for(key)
                held = await gate.acquire(_idle_receive())
                assert held.outcome == "acquired"
                sent = await _call(
                    app,
                    "POST",
                    "/v1/messages",
                    json.dumps({"model": "streamer"}).encode(),
                )
                assert _status(sent) == 529
                await gate.release(held.busy)
            finally:
                if app._session is not None:
                    await app._session.close()
                if app._receipts is not None:
                    app._receipts.close()

        rows = _read_rows(receipts)
        assert len(rows) == 1
        assert rows[0]["upstream"] == "(router)"
        assert rows[0]["status"] == "failed"
        assert key in rows[0]["reason"]
        assert rows[0]["gate_wait_s"] >= 0.25
        assert engine.arrivals == []

    asyncio.run(exercise())


def test_a_top_level_pause_holds_new_requests_on_every_engine(
    tmp_path: Path,
) -> None:
    """The lane-wide pause the runbook writes still stops every engine's queue."""

    async def exercise() -> None:
        first = Upstream("http://127.0.0.1:21011")
        second = Upstream("http://127.0.0.1:21012")
        gate_file = tmp_path / "router-gate.json"
        gate_file.write_text(
            json.dumps(
                {
                    "width": 4,
                    "wait_seconds": 0.2,
                    "paused": True,
                    "reason": "draining for a relaunch",
                    "engines": {
                        engine_id_for_origin(first.base_url): {"width": 4},
                        engine_id_for_origin(second.base_url): {"width": 4},
                    },
                }
            ),
            encoding="utf-8",
        )
        app = RouterApp(Resolver([]), gate_file=gate_file)
        first_gate = app.gate_for(engine_id_for_origin(first.base_url))
        second_gate = app.gate_for(engine_id_for_origin(second.base_url))

        assert first_gate.settings().paused is True
        assert second_gate.settings().paused is True

        for gate in (first_gate, second_gate):
            admission = await gate.acquire(_idle_receive())
            assert admission.outcome == "timed-out"

    asyncio.run(exercise())


def test_a_pause_in_the_engines_map_holds_only_its_own_engine(
    tmp_path: Path,
) -> None:
    """An engine's own pause stops that engine and leaves its peer admitting.

    This is the discriminating case for one gate per engine: applying the
    engine's entry to every queue would hold the peer too, and this test would
    then see a timeout where it requires an admission.
    """
    async def exercise() -> None:
        first = Upstream("http://127.0.0.1:21021")
        second = Upstream("http://127.0.0.1:21022")
        first_key = engine_id_for_origin(first.base_url)
        second_key = engine_id_for_origin(second.base_url)
        gate_file = tmp_path / "router-gate.json"
        gate_file.write_text(
            json.dumps(
                {
                    "width": 4,
                    "wait_seconds": 0.2,
                    "engines": {
                        first_key: {"paused": True, "reason": "retiring this engine"},
                        second_key: {"width": 4},
                    },
                }
            ),
            encoding="utf-8",
        )
        app = RouterApp(Resolver([]), gate_file=gate_file)
        paused_gate = app.gate_for(first_key)
        open_gate = app.gate_for(second_key)

        assert paused_gate.settings().paused is True
        assert open_gate.settings().paused is False

        held = await paused_gate.acquire(_idle_receive())
        assert held.outcome == "timed-out"

        admission = await open_gate.acquire(_idle_receive())
        assert admission.outcome == "acquired"
        await open_gate.release(admission.busy)

    asyncio.run(exercise())
