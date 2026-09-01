from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from typing import Any

from aiohttp import web

from imas_ambix.agent import router
from imas_ambix.agent.router import RouterApp, Upstream

SendMessage = dict[str, Any]


class Resolver:
    def __init__(self, upstreams: Sequence[Upstream]) -> None:
        self.upstreams = upstreams

    async def resolve(self) -> Sequence[Upstream]:
        return self.upstreams


def _card(model_id: str, *, context: int, count: int) -> dict[str, Any]:
    return {
        "id": model_id,
        "object": "model",
        "created": 123,
        "owned_by": "engine",
        "max_model_len": context,
        "ambix": {
            "accelerator_family": "H200 NVL",
            "accelerator_count": count,
            "checkpoint_precision": "FP8" if count == 8 else "INT4",
            "engine_extension": {"kept": True},
        },
    }


def _catalog_app(cards: Sequence[dict[str, Any]]) -> web.Application:
    app = web.Application()

    async def catalog(_: web.Request) -> web.Response:
        return web.json_response({"object": "list", "data": cards})

    app.router.add_get("/v1/models", catalog)
    return app


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
async def _router(upstreams: Sequence[Upstream]):
    app = RouterApp(Resolver(upstreams))
    try:
        yield app
    finally:
        if app._session is not None:
            await app._session.close()


async def _invoke(
    app: RouterApp,
    method: str,
    path: str,
    body: bytes = b"",
    *,
    headers: list[tuple[bytes, bytes]] | None = None,
    on_send: Callable[[SendMessage, asyncio.Queue[SendMessage]], Awaitable[None]]
    | None = None,
) -> list[SendMessage]:
    incoming: asyncio.Queue[SendMessage] = asyncio.Queue()
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
            "method": method,
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "headers": headers or [(b"content-type", b"application/json")],
        },
        receive,
        send,
    )
    return sent


def _body(messages: Sequence[SendMessage]) -> bytes:
    return b"".join(
        message.get("body", b"")
        for message in messages
        if message["type"] == "http.response.body"
    )


def _status(messages: Sequence[SendMessage]) -> int:
    return next(
        message["status"]
        for message in messages
        if message["type"] == "http.response.start"
    )


def test_union_catalog_preserves_native_cards_and_degrades(monkeypatch) -> None:
    async def exercise() -> None:
        first_card = _card("deepseek-v4-flash", context=131_072, count=2)
        second_card = _card("glm-5.3", context=202_752, count=4)
        validation_calls: list[object] = []
        validator = router.validate_catalog_metadata

        def record_validation(value: object):
            validation_calls.append(value)
            return validator(value)

        monkeypatch.setattr(router, "validate_catalog_metadata", record_validation)
        first = _catalog_app([first_card])
        second = _catalog_app([second_card])
        async with _server(first) as first_url, _server(second) as second_url:
            async with _router([Upstream(first_url), Upstream(second_url)]) as app:
                messages = await _invoke(app, "GET", "/v1/models")
                assert _status(messages) == 200
                cards = json.loads(_body(messages))["data"]
                assert cards == [first_card, second_card]
                assert validation_calls == [
                    {first_card["id"]: first_card["ambix"]},
                    {second_card["id"]: second_card["ambix"]},
                ]

            async with _router(
                [Upstream(first_url), Upstream("http://127.0.0.1:1")]
            ) as app:
                messages = await _invoke(app, "GET", "/v1/models")
                assert _status(messages) == 200
                assert json.loads(_body(messages))["data"] == [first_card]

    asyncio.run(exercise())


def test_duplicate_and_unknown_model_ids_are_refused() -> None:
    async def exercise() -> None:
        duplicate = _card("shared-native-id", context=32_768, count=4)
        engines = [_catalog_app([duplicate]), _catalog_app([duplicate])]
        async with (
            _server(engines[0]) as first_url,
            _server(engines[1]) as second_url,
            _router([Upstream(first_url), Upstream(second_url)]) as app,
        ):
            catalog = await _invoke(app, "GET", "/v1/models")
            assert _status(catalog) == 409
            assert b"duplicate model id: shared-native-id" in _body(catalog)
            request = json.dumps({"model": "shared-native-id"}).encode()
            inference = await _invoke(app, "POST", "/v1/messages", request)
            assert _status(inference) == 409

        only = _catalog_app([_card("known", context=4096, count=2)])
        async with (
            _server(only) as only_url,
            _router([Upstream(only_url)]) as app,
        ):
            request = json.dumps({"model": "not-known"}).encode()
            inference = await _invoke(app, "POST", "/v1/messages", request)
            assert _status(inference) == 404
            assert b"unknown model id: not-known" in _body(inference)

    asyncio.run(exercise())


def test_native_paths_route_with_auth_and_preserve_errors() -> None:
    async def exercise() -> None:
        seen: list[tuple[str, str, bytes]] = []

        async def catalog(_: web.Request) -> web.Response:
            return web.json_response(
                {"object": "list", "data": [_card("glm-5.3", context=8192, count=4)]}
            )

        async def endpoint(request: web.Request) -> web.Response:
            body = await request.read()
            seen.append((request.path, request.headers.get("Authorization", ""), body))
            if request.path == "/v1/chat/completions":
                return web.Response(
                    status=429,
                    body=b'{"upstream":"limited"}',
                    headers={
                        "x-engine-error": "preserved",
                        "content-type": "application/json",
                    },
                )
            return web.Response(
                body=b'{"native":true}', content_type="application/json"
            )

        engine = web.Application()
        engine.router.add_get("/v1/models", catalog)
        engine.router.add_post("/{tail:.*}", endpoint)
        async with _server(engine) as engine_url:
            upstream = Upstream(engine_url, ("Authorization", "Bearer engine-key"))
            async with _router([upstream]) as app:
                for path in ("/v1/messages", "/v1/messages/count_tokens"):
                    request = json.dumps({"model": "glm-5.3", "path": path}).encode()
                    response = await _invoke(app, "POST", path, request)
                    assert _status(response) == 200
                    assert _body(response) == b'{"native":true}'

                request = json.dumps({"model": "glm-5.3"}).encode()
                response = await _invoke(app, "POST", "/v1/chat/completions", request)
                assert _status(response) == 429
                assert _body(response) == b'{"upstream":"limited"}'
                start = next(
                    item for item in response if item["type"] == "http.response.start"
                )
                assert (b"x-engine-error", b"preserved") in start["headers"]

        assert [item[0] for item in seen] == [
            "/v1/messages",
            "/v1/messages/count_tokens",
            "/v1/chat/completions",
        ]
        assert all(item[1] == "Bearer engine-key" for item in seen)

    asyncio.run(exercise())


def test_sse_bytes_are_identical_and_disconnect_closes_upstream() -> None:
    async def exercise() -> None:
        first_event = (
            b"event: content_block_start\n"
            b'data: {"type":"thinking","signature":"sig-raw"}\n\n'
        )
        second_event = (
            b"event: content_block_delta\n"
            b'data: {"type":"tool_use","id":"toolu_exact"}\n\n'
        )
        upstream_closed = asyncio.Event()

        async def catalog(_: web.Request) -> web.Response:
            return web.json_response(
                {"object": "list", "data": [_card("streamer", context=4096, count=2)]}
            )

        async def stream(_: web.Request) -> web.StreamResponse:
            response = web.StreamResponse(headers={"content-type": "text/event-stream"})
            await response.prepare(_)
            try:
                await response.write(first_event)
                await asyncio.sleep(0.02)
                await response.write(second_event)
                await response.write_eof()
            except ConnectionError, RuntimeError:
                upstream_closed.set()
            return response

        engine = web.Application()
        engine.router.add_get("/v1/models", catalog)
        engine.router.add_post("/v1/messages", stream)
        async with _server(engine) as engine_url:
            request = json.dumps({"model": "streamer", "stream": True}).encode()
            async with _router([Upstream(engine_url)]) as app:
                response = await _invoke(app, "POST", "/v1/messages", request)
                assert _status(response) == 200
                assert _body(response) == first_event + second_event

            disconnected = False

            async def disconnect_after_first(
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

            async with _router([Upstream(engine_url)]) as app:
                await _invoke(
                    app,
                    "POST",
                    "/v1/messages",
                    request,
                    on_send=disconnect_after_first,
                )
                await asyncio.wait_for(upstream_closed.wait(), timeout=1)

    asyncio.run(exercise())
