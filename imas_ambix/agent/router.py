"""ASGI pass-through routing for native model-serving protocols, with per-model
output ceilings so a request can never overflow the owning engine's window."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import sys
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import aiohttp

from imas_ambix.agent.vllm_catalog import validate_catalog_metadata

AsgiMessage = dict[str, Any]
Receive = Callable[[], Awaitable[AsgiMessage]]
Send = Callable[[AsgiMessage], Awaitable[None]]
if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Upstream:
    """One engine origin, optional auth header, and reported native model id."""

    base_url: str
    auth_header: tuple[str, str] | None = None
    model_id: str | None = None
    # Accelerator width, carried from the registration the serving allocation
    # wrote. Ranking reads it from here rather than from the engine's card so
    # an engine with no way to echo our own metadata back is still rankable.
    # ``None`` where the upstream was discovered without a registration.
    accelerator_count: int | None = None


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


def _routing_rank(
    upstream: Upstream, card: Mapping[str, Any]
) -> tuple[int | None, int | None]:
    """Score one engine by how many accelerators it holds and when it started.

    The width comes from the upstream's registration, which the serving
    allocation wrote from its own profile, and falls back to the engine's card
    where an upstream was discovered without one. Either way ``None`` records
    that the width is unknown rather than substituting a value -- an engine
    that cannot echo site metadata must not be ranked as though it were narrow.

    The start stamp is the card's ``created`` field, which an engine may omit
    or report in a form that cannot be ordered; ``None`` records that absence
    for the same reason, so a pair that only differs there stays unranked
    instead of being separated by an invented default.
    """
    accelerators = upstream.accelerator_count
    if accelerators is None:
        metadata = card.get("ambix")
        if isinstance(metadata, Mapping):
            declared = metadata.get("accelerator_count")
            accelerators = declared if type(declared) is int else None
    created = card.get("created")
    return accelerators, created if type(created) is int else None


def _reported_width(upstream: Upstream, card: Mapping[str, Any]) -> int | None:
    """Return the accelerator width for logging, or None when it is unknown.

    Shares _routing_rank's resolution so a log line never claims a width that
    ranking did not use, and never raises on an engine that carries no site
    metadata.
    """
    width, _ = _routing_rank(upstream, card)
    return width


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
    ranks = [_routing_rank(upstream, card) for upstream, card in owners]
    widths = [accelerators for accelerators, _ in ranks]
    if any(width is None for width in widths):
        # One unknown width makes the whole comparison unsound: preferring a
        # known 4 over an unknown would be ranking on the absence of metadata
        # rather than on capacity. Drop the width term for everyone and let the
        # start stamp decide, which still favours the engine replacing its peer
        # and still refuses when there is no ground to choose.
        leaders = [
            (owner, started) for owner, (_, started) in zip(owners, ranks, strict=True)
        ]
    else:
        widest = max(widths)
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


def _by_model_id(owners: Sequence[_Owner]) -> dict[str, list[_Owner]]:
    """Partition reachable engine cards by the native model id they advertise.

    Insertion order preserves first-seen position, so a resolved duplicate
    takes the slot of its first occurrence and the collapsed union keeps a
    deterministic ordering across the source catalogs.
    """
    grouped: dict[str, list[_Owner]] = {}
    for upstream, card in owners:
        grouped.setdefault(card["id"], []).append((upstream, card))
    return grouped


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
        connection_limit: int = 2048,
        lane_document: Path | None = None,
        lane_interval: int = 30,
    ) -> None:
        self._resolver = resolver
        # The shared lane budget is published from here rather than from its own
        # allocation. A separate job spent a core of a 30-core GPU reservation
        # on one HTTP read every thirty seconds, while a peer's GPU work pended
        # on Resources with two cards idle -- the reservation binds on cores
        # allocated, not on cores used. This process is already standing, is
        # already blocked on IO, and already survives serve rotations, so the
        # refresh costs nothing additional. It must not live in a session:
        # a producer that dies with its coordinator stops silently and looks
        # exactly like a quiet lane.
        self._lane_document = lane_document
        self._lane_interval = lane_interval
        self._lane_task: asyncio.Task[None] | None = None
        # Opt-in, because it logs one line per routed request. Hashes only.
        self._prefix_diagnostic = (
            os.environ.get("AMBIX_ROUTER_PREFIX_PROBE", "").strip() == "1"
        )
        self._timeout = timeout or aiohttp.ClientTimeout(total=None, connect=10)
        self._session: aiohttp.ClientSession | None = None
        # The engine schedules its own work -- continuous batching, a waiting
        # queue and preemption under KV pressure -- so the relay must not be a
        # second, blinder scheduler in front of it. This ceiling exists only to
        # keep the process from exhausting file descriptors, and sits far above
        # the engine's own running-sequence ceiling so the engine is always the
        # binding constraint. aiohttp's unset default is 100, which would
        # silently cap concurrency below that.
        self._connection_limit = connection_limit

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
                "router preference model=%s candidates=%d origin=%s accelerators=%s",
                model_id,
                len(owners),
                upstream.base_url,
                _reported_width(upstream, card),
            )

        self._log_prefix_divergence(payload, scope)
        relay_body = self._clamp_output_tokens(payload, body, card)
        await self._relay(scope, receive, send, relay_body, upstream)

    async def _lifespan(self, receive: Receive, send: Send) -> None:
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                await self._client()
                if self._lane_document is not None:
                    self._lane_task = asyncio.create_task(self._publish_lane())
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                if self._lane_task is not None:
                    self._lane_task.cancel()
                    self._lane_task = None
                if self._session is not None:
                    await self._session.close()
                    self._session = None
                await send({"type": "lifespan.shutdown.complete"})
                return

    def _log_prefix_divergence(self, payload: object, scope: Mapping[str, Any]) -> None:
        """Record where one caller's prompt stops matching its previous turn.

        A prefix cache that is consulted and still misses is being asked about a
        prefix that genuinely differs. Measured 2026-09-15: queries ran at 1.12x
        prompt tokens, so every token was looked up, while the hit rate sat near
        37% on an almost empty lane -- which contention cannot explain. The
        remaining candidate is that something varies at the HEAD of the prompt,
        because a single changed byte early invalidates every block after it.

        Hashes only, at increasing depths, so the log carries no prompt content:
        the first depth whose digest changes between two consecutive turns is
        where reuse dies. Divergence at the shallowest depth means the very top
        of the prompt moves per request and no reuse is possible at all.
        """
        if not self._prefix_diagnostic:
            return
        try:
            messages = payload.get("messages") if isinstance(payload, dict) else None
            if not isinstance(messages, list):
                return
            flat = json.dumps(messages, separators=(",", ":"), sort_keys=False)
        except (TypeError, ValueError):
            return
        digests = []
        for depth in (512, 2048, 8192, 32768, 131072):
            chunk = flat[:depth]
            digests.append(f"{depth}:{hashlib.sha256(chunk.encode()).hexdigest()[:8]}")
            if len(flat) <= depth:
                break
        logger.info(
            "prefix-probe consumer=%s chars=%d %s",
            self._caller_hint(scope),
            len(flat),
            " ".join(digests),
        )

    @staticmethod
    def _caller_hint(scope: Mapping[str, Any]) -> str:
        """Identify the calling process well enough to group its own turns."""
        headers = scope.get("headers") or ()
        agent = b""
        for name, value in headers:
            if bytes(name).lower() == b"user-agent":
                agent = bytes(value)
                break
        client = scope.get("client")
        host = client[0] if isinstance(client, Sequence) and client else "unknown"
        port = client[1] if isinstance(client, Sequence) and len(client) > 1 else 0
        return f"{host}:{port}|{agent.decode('utf-8', 'replace')[:32]}"

    async def _publish_lane(self) -> None:
        """Republish the shared lane budget for as long as the router runs.

        Never allowed to disturb routing: every failure publishes an explicit
        unavailable state and the loop continues, because a relay that stopped
        serving requests to keep a metrics file current would have inverted its
        own purpose.
        """
        from collections import deque

        from imas_ambix.agent.lane import (
            LaneWindow,
            detect_settling,
            parse_lane_capacity,
            write_lane_document,
            write_unavailable_document,
        )

        # Five samples, so the published budget rests on ~2.5 minutes at the
        # default cadence rather than on one draw. The window is what makes the
        # figure a level instead of a ratio caught mid-flicker; the length is a
        # trade the reader should know about, since a genuine ramp is reported
        # late by up to that span. `settling` and `volatile` both cover that gap
        # by marking the figure an upper bound, which is the safe direction.
        readings: deque = deque(maxlen=5)
        previous = None
        while True:
            try:
                upstreams = await self._resolver.resolve()
                if not upstreams:
                    raise RuntimeError("no upstream is registered")
                session = await self._client()
                target = f"{upstreams[0].base_url.rstrip('/')}/metrics"
                async with session.get(target) as response:
                    body = await response.text()
                capacity = parse_lane_capacity(body)
            except (aiohttp.ClientError, OSError, RuntimeError, ValueError) as error:
                write_unavailable_document(str(error), self._lane_document)
                # Both histories are dropped, not just the last sample. A window
                # spanning an outage would average across a gap of unknown
                # length and publish it as a continuous measurement.
                readings.clear()
                previous = None
            except asyncio.CancelledError:
                raise
            else:
                readings.append(capacity)
                write_lane_document(
                    capacity,
                    self._lane_document,
                    settling=detect_settling(previous, capacity),
                    refresh_interval=self._lane_interval,
                    window=LaneWindow(readings=tuple(readings)),
                )
                previous = capacity
            await asyncio.sleep(self._lane_interval)

    async def _client(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=self._timeout,
                auto_decompress=False,
                connector=aiohttp.TCPConnector(limit=self._connection_limit),
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
            # Validate the site metadata when the engine carries it, and accept
            # the card when it does not. Requiring it here made routing a
            # vLLM-only surface, because the block is launch-owned data echoed
            # back through a middleware that only vLLM accepts -- so an engine
            # without that channel was unroutable regardless of health. The
            # width used for ranking now comes from the registration instead.
            if card.get("ambix") is not None:
                validate_catalog_metadata({card["id"]: card["ambix"]})
            cards.append(card)
        return _Catalog(upstream=upstream, payload=payload, cards=tuple(cards))

    async def _serve_catalog(self, send: Send) -> None:
        catalogs = await self._reachable_catalogs()
        if not catalogs:
            await self._json_error(send, 503, "no upstream catalogs are reachable")
            return
        owners = [
            (catalog.upstream, card) for catalog in catalogs for card in catalog.cards
        ]
        cards: list[dict[str, Any]] = []
        duplicates: list[str] = []
        for model_id, peers in _by_model_id(owners).items():
            if len(peers) == 1:
                cards.append(peers[0][1])
                continue
            preferred = _preferred_owner(peers)
            if preferred is None:
                duplicates.append(model_id)
                continue
            upstream, card = preferred
            logger.info(
                "router preference model=%s candidates=%d origin=%s accelerators=%s",
                model_id,
                len(peers),
                upstream.base_url,
                _reported_width(upstream, card),
            )
            cards.append(card)
        if duplicates:
            await self._json_error(
                send,
                409,
                f"duplicate model id: {', '.join(sorted(duplicates))}",
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


def create_router_app(
    resolver: UpstreamResolver, *, lane_document: Path | None = None
) -> RouterApp:
    """Build the ASGI application around an injected upstream resolver."""
    return RouterApp(resolver, lane_document=lane_document)


def serve_router(
    resolver: UpstreamResolver,
    *,
    host: str = "0.0.0.0",
    port: int,
    lane_document: Path | None = None,
) -> None:
    """Run the router ASGI application with the serving runtime."""
    import uvicorn

    # uvicorn configures its own loggers and leaves everyone else on the root
    # logger, which defaults to WARNING -- so every INFO this module emits was
    # dropped before reaching the job log. Measured 2026-09-15: 32 requests
    # routed with zero lines from this module, including the upstream-preference
    # diagnostic that has been silently absent since it was written. A
    # diagnostic that cannot be read is not a diagnostic, so attach a handler
    # rather than assume one.
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    uvicorn.run(
        create_router_app(resolver, lane_document=lane_document),
        host=host,
        port=port,
    )
