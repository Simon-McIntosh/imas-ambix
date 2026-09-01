"""ASGI pass-through routing for native model-serving protocols."""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import aiohttp

from imas_ambix.agent.vllm_catalog import validate_catalog_metadata

AsgiMessage = dict[str, Any]
Receive = Callable[[], Awaitable[AsgiMessage]]
Send = Callable[[AsgiMessage], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class Upstream:
    """One engine origin, optional auth header, and reported native model id."""

    base_url: str
    auth_header: tuple[str, str] | None = None
    model_id: str | None = None


class UpstreamResolver(Protocol):
    """Return the engine endpoints currently eligible for routing."""

    async def resolve(self) -> Sequence[Upstream]: ...


class DynamicUpstreamResolver:
    """Adapt a synchronous discovery supplier to the router's async interface."""

    def __init__(self, supplier: Callable[[], Sequence[Upstream]]) -> None:
        self._supplier = supplier

    async def resolve(self) -> Sequence[Upstream]:
        return await asyncio.to_thread(self._supplier)


@dataclass(frozen=True, slots=True)
class _Catalog:
    upstream: Upstream
    payload: dict[str, Any]
    cards: tuple[dict[str, Any], ...]


class RouterApp:
    """Present a union catalog and relay native requests to their owning engine."""

    _ROUTED_PATHS = frozenset(
        {
            "/v1/messages",
            "/v1/messages/count_tokens",
            "/v1/chat/completions",
        }
    )

    def __init__(
        self,
        resolver: UpstreamResolver,
        *,
        timeout: aiohttp.ClientTimeout | None = None,
    ) -> None:
        self._resolver = resolver
        self._timeout = timeout or aiohttp.ClientTimeout(total=None, connect=10)
        self._session: aiohttp.ClientSession | None = None

    async def __call__(
        self, scope: dict[str, Any], receive: Receive, send: Send
    ) -> None:
        scope_type = scope.get("type")
        if scope_type == "lifespan":
            await self._lifespan(receive, send)
            return
        if scope_type != "http":
            return

        method = scope.get("method")
        path = scope.get("path")
        if method == "GET" and path == "/v1/models":
            await self._serve_catalog(send)
            return
        if method != "POST" or path not in self._ROUTED_PATHS:
            await self._json_error(send, 404, "unsupported router path")
            return

        body, disconnected = await self._request_body(receive)
        if disconnected:
            return
        try:
            payload = json.loads(body)
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
        ):
            await self._json_error(send, 400, "request body must be valid JSON")
            return
        model_id = payload.get("model") if isinstance(payload, Mapping) else None
        if not isinstance(model_id, str) or not model_id:
            await self._json_error(send, 400, "request body must contain a model id")
            return

        catalogs = await self._reachable_catalogs()
        owners = [
            catalog.upstream
            for catalog in catalogs
            if any(card["id"] == model_id for card in catalog.cards)
        ]
        if not owners:
            await self._json_error(send, 404, f"unknown model id: {model_id}")
            return
        if len(owners) != 1:
            await self._json_error(send, 409, f"duplicate model id: {model_id}")
            return
        await self._relay(scope, receive, send, body, owners[0])

    async def _lifespan(self, receive: Receive, send: Send) -> None:
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                await self._client()
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                if self._session is not None:
                    await self._session.close()
                    self._session = None
                await send({"type": "lifespan.shutdown.complete"})
                return

    async def _client(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=self._timeout,
                auto_decompress=False,
            )
        return self._session

    async def _reachable_catalogs(self) -> list[_Catalog]:
        upstreams = await self._resolver.resolve()
        results = await asyncio.gather(
            *(self._fetch_catalog(upstream) for upstream in upstreams),
            return_exceptions=True,
        )
        return [result for result in results if isinstance(result, _Catalog)]

    async def _fetch_catalog(self, upstream: Upstream) -> _Catalog:
        session = await self._client()
        headers = dict([upstream.auth_header] if upstream.auth_header else [])
        async with session.get(
            f"{upstream.base_url.rstrip('/')}/v1/models", headers=headers
        ) as response:
            if not 200 <= response.status < 300:
                raise RuntimeError(f"catalog returned {response.status}")
            payload = await response.json(content_type=None)
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise ValueError("catalog must contain a data list")

        cards: list[dict[str, Any]] = []
        for card in payload["data"]:
            if not isinstance(card, dict) or not isinstance(card.get("id"), str):
                raise ValueError("catalog cards must carry native ids")
            validate_catalog_metadata({card["id"]: card.get("ambix")})
            cards.append(card)
        return _Catalog(upstream=upstream, payload=payload, cards=tuple(cards))

    async def _serve_catalog(self, send: Send) -> None:
        catalogs = await self._reachable_catalogs()
        if not catalogs:
            await self._json_error(send, 503, "no upstream catalogs are reachable")
            return
        cards = [card for catalog in catalogs for card in catalog.cards]
        duplicates = sorted(
            model_id
            for model_id, count in Counter(card["id"] for card in cards).items()
            if count > 1
        )
        if duplicates:
            await self._json_error(
                send,
                409,
                f"duplicate model id: {', '.join(duplicates)}",
            )
            return
        payload = dict(catalogs[0].payload)
        payload["data"] = cards
        body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode()
        await self._response(
            send,
            200,
            [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
            body,
        )

    async def _relay(
        self,
        scope: dict[str, Any],
        receive: Receive,
        send: Send,
        body: bytes,
        upstream: Upstream,
    ) -> None:
        session = await self._client()
        request_headers = [
            (name.decode("latin-1"), value.decode("latin-1"))
            for name, value in scope.get("headers", [])
            if name.lower() != b"host"
            and (
                upstream.auth_header is None
                or name.decode("latin-1").lower() != upstream.auth_header[0].lower()
            )
        ]
        if upstream.auth_header is not None:
            request_headers.append(upstream.auth_header)
        raw_path = scope.get("raw_path", scope["path"].encode()).decode("ascii")
        query = scope.get("query_string", b"")
        target = f"{upstream.base_url.rstrip('/')}{raw_path}"
        if query:
            target = f"{target}?{query.decode('ascii')}"

        async with session.request(
            scope["method"], target, data=body, headers=request_headers
        ) as response:
            await send(
                {
                    "type": "http.response.start",
                    "status": response.status,
                    "headers": list(response.raw_headers),
                }
            )
            disconnected = asyncio.create_task(self._wait_for_disconnect(receive))
            try:
                while True:
                    next_chunk = asyncio.create_task(response.content.readany())
                    done, _ = await asyncio.wait(
                        {next_chunk, disconnected},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if disconnected in done:
                        next_chunk.cancel()
                        await asyncio.gather(next_chunk, return_exceptions=True)
                        response.close()
                        return
                    chunk = next_chunk.result()
                    if not chunk:
                        break
                    await send(
                        {"type": "http.response.body", "body": chunk, "more_body": True}
                    )
                await send({"type": "http.response.body", "body": b""})
            finally:
                disconnected.cancel()
                await asyncio.gather(disconnected, return_exceptions=True)

    @staticmethod
    async def _request_body(receive: Receive) -> tuple[bytes, bool]:
        parts: list[bytes] = []
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return b"", True
            if message["type"] != "http.request":
                continue
            parts.append(message.get("body", b""))
            if not message.get("more_body", False):
                return b"".join(parts), False

    @staticmethod
    async def _wait_for_disconnect(receive: Receive) -> None:
        while True:
            if (await receive())["type"] == "http.disconnect":
                return

    @staticmethod
    async def _response(
        send: Send, status: int, headers: list[tuple[bytes, bytes]], body: bytes
    ) -> None:
        await send(
            {"type": "http.response.start", "status": status, "headers": headers}
        )
        await send({"type": "http.response.body", "body": body})

    async def _json_error(self, send: Send, status: int, detail: str) -> None:
        body = json.dumps(
            {"error": {"message": detail}}, separators=(",", ":")
        ).encode()
        await self._response(
            send,
            status,
            [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
            body,
        )


def create_router_app(resolver: UpstreamResolver) -> RouterApp:
    """Build the ASGI application around an injected upstream resolver."""
    return RouterApp(resolver)


def serve_router(
    resolver: UpstreamResolver, *, host: str = "0.0.0.0", port: int
) -> None:
    """Run the router ASGI application with the serving runtime."""
    import uvicorn

    uvicorn.run(create_router_app(resolver), host=host, port=port)
