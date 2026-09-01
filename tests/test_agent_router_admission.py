from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable, Sequence
from contextlib import asynccontextmanager, suppress
from typing import Any

from aiohttp import web

from imas_ambix.agent.router import AdmissionLimits, RouterApp, Upstream

SendMessage = dict[str, Any]


class Resolver:
    def __init__(self, upstreams: Sequence[Upstream]) -> None:
        self.upstreams = upstreams

    async def resolve(self) -> Sequence[Upstream]:
        return self.upstreams


def _card() -> dict[str, Any]:
    return {
        "id": "capacity-model",
        "object": "model",
        "max_model_len": 4096,
        "ambix": {
            "accelerator_family": "H200 NVL",
            "accelerator_count": 2,
            "checkpoint_precision": "FP8",
        },
    }


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
async def _router(base_url: str, limits: AdmissionLimits):
    app = RouterApp(Resolver([Upstream(base_url)]), admission_limits=limits)
    try:
        yield app
    finally:
        if app._session is not None:
            await app._session.close()


async def _invoke(
    app: RouterApp,
    label: str,
    consumer: str,
    *,
    on_send: Callable[[SendMessage, asyncio.Queue[SendMessage]], Awaitable[None]]
    | None = None,
) -> list[SendMessage]:
    incoming: asyncio.Queue[SendMessage] = asyncio.Queue()
    body = json.dumps({"model": "capacity-model", "label": label}).encode()
    await incoming.put({"type": "http.request", "body": body})
    sent: list[SendMessage] = []

    async def receive() -> SendMessage:
        return await incoming.get()

    async def send(message: SendMessage) -> None:
        sent.append(message)
        if on_send is not None:
            await on_send(message, incoming)

    await app(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/messages",
            "raw_path": b"/v1/messages",
            "query_string": b"",
            "headers": [(b"content-type", b"application/json")],
            "client": (consumer, 12345),
        },
        receive,
        send,
    )
    return sent


def _status(messages: Sequence[SendMessage]) -> int:
    return next(
        message["status"]
        for message in messages
        if message["type"] == "http.response.start"
    )


def _body(messages: Sequence[SendMessage]) -> bytes:
    return b"".join(
        message.get("body", b"")
        for message in messages
        if message["type"] == "http.response.body"
    )


def test_consumer_cap_bounds_queue_and_isolates_other_consumers(caplog) -> None:
    async def exercise() -> None:
        release_first = asyncio.Event()
        started: asyncio.Queue[str] = asyncio.Queue()

        async def catalog(_: web.Request) -> web.Response:
            return web.json_response({"object": "list", "data": [_card()]})

        async def infer(request: web.Request) -> web.Response:
            label = (await request.json())["label"]
            await started.put(label)
            if label == "first":
                await release_first.wait()
            return web.json_response({"label": label})

        engine = web.Application()
        engine.router.add_get("/v1/models", catalog)
        engine.router.add_post("/v1/messages", infer)
        limits = AdmissionLimits(
            max_in_flight=1,
            max_queued=1,
            retry_after_seconds=7,
        )
        async with _server(engine) as engine_url, _router(engine_url, limits) as app:
            first = asyncio.create_task(_invoke(app, "first", "consumer-a"))
            assert await asyncio.wait_for(started.get(), timeout=1) == "first"

            queued = asyncio.create_task(_invoke(app, "queued", "consumer-a"))
            for _ in range(100):
                capacity = app._admission._consumers.get("consumer-a")
                if capacity is not None and capacity.queued == 1:
                    break
                await asyncio.sleep(0.001)
            else:
                raise AssertionError("second request did not enter the bounded queue")
            retry = await asyncio.wait_for(
                _invoke(app, "retry", "consumer-a"), timeout=1
            )
            assert _status(retry) == 429
            retry_start = next(
                message for message in retry if message["type"] == "http.response.start"
            )
            assert (b"retry-after", b"7") in retry_start["headers"]
            assert json.loads(_body(retry))["error"]["retry_after_seconds"] == 7

            other = await asyncio.wait_for(
                _invoke(app, "other", "consumer-b"), timeout=1
            )
            assert _status(other) == 200
            assert await asyncio.wait_for(started.get(), timeout=1) == "other"

            release_first.set()
            assert _status(await asyncio.wait_for(first, timeout=1)) == 200
            assert _status(await asyncio.wait_for(queued, timeout=1)) == 200
            assert await asyncio.wait_for(started.get(), timeout=1) == "queued"

        messages = [record.getMessage() for record in caplog.records]
        assert any("action=queued consumer=consumer-a" in item for item in messages)
        assert any("action=retry consumer=consumer-a" in item for item in messages)
        assert any("action=admitted consumer=consumer-b" in item for item in messages)

    caplog.set_level(logging.INFO, logger="imas_ambix.agent.router")
    asyncio.run(exercise())


def test_disconnect_releases_the_consumer_slot() -> None:
    async def exercise() -> None:
        first_chunk = b"event: message_start\ndata: {}\n\n"

        async def catalog(_: web.Request) -> web.Response:
            return web.json_response({"object": "list", "data": [_card()]})

        async def infer(request: web.Request) -> web.StreamResponse:
            label = (await request.json())["label"]
            if label != "stream":
                return web.json_response({"label": label})
            response = web.StreamResponse(headers={"content-type": "text/event-stream"})
            await response.prepare(request)
            await response.write(first_chunk)
            await asyncio.sleep(0.05)
            with suppress(ConnectionError, RuntimeError):
                await response.write(b"event: message_stop\ndata: {}\n\n")
            return response

        engine = web.Application()
        engine.router.add_get("/v1/models", catalog)
        engine.router.add_post("/v1/messages", infer)
        limits = AdmissionLimits(max_in_flight=1, max_queued=0)
        disconnected = False

        async def disconnect_after_first_chunk(
            message: SendMessage, incoming: asyncio.Queue[SendMessage]
        ) -> None:
            nonlocal disconnected
            if (
                message["type"] == "http.response.body"
                and message.get("body")
                and not disconnected
            ):
                disconnected = True
                await incoming.put({"type": "http.disconnect"})

        async with _server(engine) as engine_url, _router(engine_url, limits) as app:
            await _invoke(
                app,
                "stream",
                "consumer-a",
                on_send=disconnect_after_first_chunk,
            )
            response = await asyncio.wait_for(
                _invoke(app, "after-disconnect", "consumer-a"), timeout=1
            )
            assert _status(response) == 200
            assert json.loads(_body(response)) == {"label": "after-disconnect"}

    asyncio.run(exercise())


def test_admission_limits_reject_invalid_configuration() -> None:
    for values in (
        {"max_in_flight": 0},
        {"max_queued": -1},
        {"retry_after_seconds": 0},
    ):
        try:
            AdmissionLimits(**values)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid limits accepted: {values}")

    configured = AdmissionLimits.from_environment(
        {
            "AMBIX_ROUTER_MAX_IN_FLIGHT": "3",
            "AMBIX_ROUTER_MAX_QUEUED": "6",
            "AMBIX_ROUTER_RETRY_AFTER_SECONDS": "11",
        }
    )
    assert configured == AdmissionLimits(
        max_in_flight=3,
        max_queued=6,
        retry_after_seconds=11,
    )
