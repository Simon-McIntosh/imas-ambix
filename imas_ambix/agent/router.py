"""ASGI pass-through routing for native model-serving protocols, with per-model
output ceilings so a request can never overflow the owning engine's window."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import aiohttp

from imas_ambix.agent.vllm_catalog import validate_catalog_metadata

AsgiMessage = dict[str, Any]
Receive = Callable[[], Awaitable[AsgiMessage]]
Send = Callable[[AsgiMessage], Awaitable[None]]
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AdmissionLimits:
    """Bound active and waiting requests independently for each consumer."""

    max_in_flight: int = 2
    max_queued: int = 4
    retry_after_seconds: int = 5

    @classmethod
    def from_environment(
        cls, environment: Mapping[str, str] | None = None
    ) -> AdmissionLimits:
        """Read router limits while retaining group-sized defaults."""
        source = os.environ if environment is None else environment
        names = {
            "max_in_flight": "AMBIX_ROUTER_MAX_IN_FLIGHT",
            "max_queued": "AMBIX_ROUTER_MAX_QUEUED",
            "retry_after_seconds": "AMBIX_ROUTER_RETRY_AFTER_SECONDS",
        }
        values: dict[str, int] = {}
        for field, name in names.items():
            raw = source.get(name)
            if raw is None:
                continue
            try:
                values[field] = int(raw)
            except ValueError as error:
                raise ValueError(f"{name} must be an integer") from error
        return cls(**values)

    def __post_init__(self) -> None:
        if self.max_in_flight < 1:
            raise ValueError("max_in_flight must be positive")
        if self.max_queued < 0:
            raise ValueError("max_queued must not be negative")
        if self.retry_after_seconds < 1:
            raise ValueError("retry_after_seconds must be positive")


@dataclass(slots=True)
class _ConsumerCapacity:
    condition: asyncio.Condition
    in_flight: int = 0
    queued: int = 0


class _AdmissionController:
    def __init__(self, limits: AdmissionLimits) -> None:
        self._limits = limits
        self._consumers: dict[str, _ConsumerCapacity] = {}

    async def acquire(self, consumer: str) -> bool:
        capacity = self._consumers.setdefault(
            consumer, _ConsumerCapacity(asyncio.Condition())
        )
        async with capacity.condition:
            if capacity.in_flight < self._limits.max_in_flight:
                capacity.in_flight += 1
                self._log("admitted", consumer, capacity)
                return True
            if capacity.queued >= self._limits.max_queued:
                self._log("retry", consumer, capacity)
                return False

            capacity.queued += 1
            self._log("queued", consumer, capacity)
            try:
                await capacity.condition.wait_for(
                    lambda: capacity.in_flight < self._limits.max_in_flight
                )
            except BaseException:
                capacity.queued -= 1
                self._log("queue-cancelled", consumer, capacity)
                raise
            capacity.queued -= 1
            capacity.in_flight += 1
            self._log("admitted-from-queue", consumer, capacity)
            return True

    async def release(self, consumer: str) -> None:
        capacity = self._consumers[consumer]
        async with capacity.condition:
            capacity.in_flight -= 1
            self._log("released", consumer, capacity)
            capacity.condition.notify(1)

    @staticmethod
    def _log(action: str, consumer: str, capacity: _ConsumerCapacity) -> None:
        logger.info(
            "router admission action=%s consumer=%s in_flight=%d queued=%d",
            action,
            consumer,
            capacity.in_flight,
            capacity.queued,
        )


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


_Owner = tuple[Upstream, dict[str, Any]]


def _routing_rank(card: Mapping[str, Any]) -> tuple[int, int | None]:
    """Score one engine by how many accelerators it holds and when it started.

    Every card the router accepts has passed catalog validation, so the
    accelerator count is always an integer and always comparable. The start
    stamp is the card's ``created`` field, which an engine may omit or report
    in a form that cannot be ordered; ``None`` records that absence rather
    than substituting a value, so a pair that only differs there stays
    unranked instead of being separated by an invented default.
    """
    accelerators = card["ambix"]["accelerator_count"]
    created = card.get("created")
    return accelerators, created if type(created) is int else None


def _preferred_owner(owners: Sequence[_Owner]) -> _Owner | None:
    """Return the engine that outranks every peer, or None when two are level.

    Several reachable engines advertising one model id is the normal state
    while a serve is being replaced by a wider one: the incoming engine joins
    the catalog the moment it answers, and both are healthy for the length of
    the overlap. Routing to the widest engine keeps that window free of a
    capacity regression, and the later start stamp breaks a tie towards the
    engine that is replacing its peer rather than the one being retired. Where
    the two leaders are indistinguishable on both terms there is no ground for
    preferring either, and the caller refuses instead of choosing arbitrarily.
    """
    ranks = [_routing_rank(card) for _, card in owners]
    widest = max(accelerators for accelerators, _ in ranks)
    leaders = [
        (owner, started)
        for owner, (accelerators, started) in zip(owners, ranks, strict=True)
        if accelerators == widest
    ]
    if len(leaders) == 1:
        return leaders[0][0]
    if any(started is None for _, started in leaders):
        return None
    newest = max(started for _, started in leaders)
    latest = [owner for owner, started in leaders if started == newest]
    return latest[0] if len(latest) == 1 else None


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
        admission_limits: AdmissionLimits | None = None,
    ) -> None:
        self._resolver = resolver
        self._timeout = timeout or aiohttp.ClientTimeout(total=None, connect=10)
        self._session: aiohttp.ClientSession | None = None
        self._admission_limits = admission_limits or AdmissionLimits()
        self._admission = _AdmissionController(self._admission_limits)

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
        owners: list[_Owner] = [
            (catalog.upstream, card)
            for catalog in catalogs
            for card in catalog.cards
            if card["id"] == model_id
        ]
        if not owners:
            await self._json_error(send, 404, f"unknown model id: {model_id}")
            return
        selected = _preferred_owner(owners)
        if selected is None:
            await self._json_error(send, 409, f"duplicate model id: {model_id}")
            return
        upstream, card = selected
        if len(owners) > 1:
            logger.info(
                "router preference model=%s candidates=%d origin=%s accelerators=%d",
                model_id,
                len(owners),
                upstream.base_url,
                card["ambix"]["accelerator_count"],
            )

        consumer = self._consumer_id(scope)
        if not await self._admission.acquire(consumer):
            await self._retry_later(send)
            return
        try:
            relay_body = self._clamp_output_tokens(payload, body, card)
            await self._relay(scope, receive, send, relay_body, upstream)
        finally:
            await self._admission.release(consumer)

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
    def _clamp_output_tokens(
        payload: Mapping[str, Any], body: bytes, card: Mapping[str, Any]
    ) -> bytes:
        """Cap requested output so a prompt plus response fits the model window.

        The engine rejects a request whose declared output exceeds what the
        model window leaves after the prompt. Agent harnesses fix a large
        output reservation at launch and do not re-derive it when the model is
        switched mid-session, so a prompt that fits one engine can overflow a
        narrower one merely because the old reservation was carried over.
        Leaving a quarter of the window for the response mirrors the launcher
        convention and bounds every request without tokenizing the prompt here;
        a prompt beyond three quarters of the window is handled by the engine
        as before. The body is re-encoded only when a cap actually applied, so
        otherwise the request passes through byte-for-byte.
        """
        maximum = card.get("max_model_len")
        if not isinstance(maximum, int) or maximum <= 0:
            return body
        ceiling = max(1, maximum // 4)
        clamped: dict[str, Any] | None = None
        for field in ("max_tokens", "max_completion_tokens"):
            requested = payload.get(field)
            if isinstance(requested, int) and requested > ceiling:
                if clamped is None:
                    clamped = dict(payload)
                clamped[field] = ceiling
        if clamped is None:
            return body
        return json.dumps(clamped, ensure_ascii=False, separators=(",", ":")).encode()

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

    async def _retry_later(self, send: Send) -> None:
        wait = self._admission_limits.retry_after_seconds
        body = json.dumps(
            {
                "error": {
                    "message": f"consumer queue full; retry after {wait} seconds",
                    "retry_after_seconds": wait,
                }
            },
            separators=(",", ":"),
        ).encode()
        await self._response(
            send,
            429,
            [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
                (b"retry-after", str(wait).encode()),
            ],
            body,
        )

    @staticmethod
    def _consumer_id(scope: Mapping[str, Any]) -> str:
        client = scope.get("client")
        if isinstance(client, Sequence) and client and isinstance(client[0], str):
            return client[0]
        return "unknown"


def create_router_app(
    resolver: UpstreamResolver,
    *,
    admission_limits: AdmissionLimits | None = None,
) -> RouterApp:
    """Build the ASGI application around an injected upstream resolver."""
    return RouterApp(resolver, admission_limits=admission_limits)


def serve_router(
    resolver: UpstreamResolver,
    *,
    host: str = "0.0.0.0",
    port: int,
    admission_limits: AdmissionLimits | None = None,
) -> None:
    """Run the router ASGI application with the serving runtime."""
    import uvicorn

    limits = admission_limits or AdmissionLimits.from_environment()
    uvicorn.run(
        create_router_app(resolver, admission_limits=limits),
        host=host,
        port=port,
    )
