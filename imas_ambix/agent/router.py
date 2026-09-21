"""ASGI pass-through routing for native model-serving protocols, with per-model
output ceilings so a request can never overflow the owning engine's window."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import sys
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol

import aiohttp

from imas_ambix.agent.request_receipts import (
    RECEIPTS_FILENAME,
    STATUS_ABORTED,
    STATUS_COMPLETED,
    STATUS_FAILED,
    RequestReceiptSink,
    StreamAccounting,
)
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
        request_receipts_path: Path | None = None,
    ) -> None:
        self._resolver = resolver
        # Per-request attribution lands beside the lane document the launching
        # command already names, so it needs no further operator step to be
        # reachable once the router is running. Kept off entirely when there is
        # no document directory to write into, and never able to disturb
        # routing -- the sink logs a failure and discards.
        if request_receipts_path is not None:
            self._receipts_path: Path | None = request_receipts_path
        elif lane_document is not None:
            self._receipts_path = lane_document.with_name(RECEIPTS_FILENAME)
        else:
            self._receipts_path = None
        self._receipts: RequestReceiptSink | None = None
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
        payload, body = self._repair_system_roles(payload, body)
        relay_body = self._clamp_output_tokens(payload, body, card)
        await self._relay(
            scope,
            receive,
            send,
            relay_body,
            upstream,
            model_id=model_id,
            caller_hint=self._caller_hint(scope),
            started_at=datetime.now(UTC),
        )

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
                if self._receipts is not None:
                    self._receipts.close()
                    self._receipts = None
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
            fetch_lane_capacity,
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
            target = "unresolved"
            try:
                upstreams = await self._resolver.resolve()
                if not upstreams:
                    raise RuntimeError("no upstream is registered")
                target = f"{upstreams[0].base_url.rstrip('/')}/metrics"
                capacity = await asyncio.to_thread(
                    fetch_lane_capacity, upstreams[0].base_url
                )
            except (aiohttp.ClientError, OSError, RuntimeError, ValueError) as error:
                logger.warning(
                    "lane publisher unavailable target=%s error=%s: %s",
                    target,
                    type(error).__name__,
                    error,
                )
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
        *,
        model_id: str,
        caller_hint: str,
        started_at: datetime,
    ) -> None:
        session = await self._client()
        # content-length and transfer-encoding describe the body the CLIENT
        # sent, and this relay may forward a different one: both
        # _repair_system_roles and _clamp_output_tokens re-encode the payload,
        # and re-labelling a role shortens the body by two bytes per message.
        # Forwarding the original length makes the upstream wait forever for
        # bytes that will never arrive -- the request hangs rather than failing,
        # so it reads as a slow serve rather than a malformed relay. Measured
        # 2026-09-16: every relabelled request hung for six minutes and then
        # retried, taking the lane down for agent traffic while small probes
        # that needed no rewrite passed in 0.2 s. Dropping both lets the client
        # library set the framing from the body actually being sent.
        drop = {b"host", b"content-length", b"transfer-encoding"}
        request_headers = [
            (name.decode("latin-1"), value.decode("latin-1"))
            for name, value in scope.get("headers", [])
            if name.lower() not in drop
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

        accounting = StreamAccounting()
        began = time.perf_counter()
        # Anything that is not a 2xx relayed whole reads as failed, so a request
        # the engine refused is recorded as such rather than dropped or counted
        # as a success. The outcome is only rewritten by the two paths that can
        # tell better: a 2xx relay, and a caller that left mid-relay.
        status = STATUS_FAILED
        try:
            async with session.request(
                scope["method"], target, data=body, headers=request_headers
            ) as response:
                if 200 <= response.status < 300:
                    status = STATUS_COMPLETED
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
                            status = STATUS_ABORTED
                            next_chunk.cancel()
                            await asyncio.gather(next_chunk, return_exceptions=True)
                            response.close()
                            return
                        chunk = next_chunk.result()
                        if not chunk:
                            break
                        accounting.feed(chunk)
                        await send(
                            {
                                "type": "http.response.body",
                                "body": chunk,
                                "more_body": True,
                            }
                        )
                    accounting.finish()
                    await send({"type": "http.response.body", "body": b""})
                finally:
                    disconnected.cancel()
                    await asyncio.gather(disconnected, return_exceptions=True)
        except asyncio.CancelledError:
            # The router is shutting down under an in-flight request; the
            # caller's answer is incomplete, which is exactly what the row
            # should say.
            status = STATUS_ABORTED
            raise
        finally:
            self._record_receipt(
                accounting=accounting,
                status=status,
                model_id=model_id,
                upstream=upstream,
                caller_hint=caller_hint,
                started_at=started_at,
                began=began,
            )

    def _receipt_sink(self) -> RequestReceiptSink | None:
        """The process's receipt sink, built on first use.

        Built lazily so a router with no receipts path pays nothing, and
        constructed without IO so a bad path cannot fail a request.
        """
        if self._receipts_path is None:
            return None
        if self._receipts is None:
            self._receipts = RequestReceiptSink(self._receipts_path)
        return self._receipts

    def _record_receipt(
        self,
        *,
        accounting: StreamAccounting,
        status: str,
        model_id: str,
        upstream: Upstream,
        caller_hint: str,
        started_at: datetime,
        began: float,
    ) -> None:
        """Append the row for one relayed request, whatever its outcome.

        Never raises into the relay: this runs in a ``finally`` whose exception
        may still be propagating, so a failure here would replace the real
        error with a bookkeeping one.
        """
        sink = self._receipt_sink()
        if sink is None:
            return
        try:
            sink.record(
                model=model_id,
                upstream=upstream.base_url,
                caller_hint=caller_hint,
                status=status,
                duration_s=time.perf_counter() - began,
                accounting=accounting,
                timestamp=started_at,
            )
        except (OSError, TypeError, ValueError) as error:
            logger.warning(
                "request receipt dropped model=%s error=%s: %s",
                model_id,
                type(error).__name__,
                error,
            )

    @staticmethod
    def _repair_system_roles(
        payload: Mapping[str, Any], body: bytes
    ) -> tuple[Mapping[str, Any], bytes]:
        """Re-label mid-conversation system messages so the prefix can be reused.

        The DeepSeek-V4.1 encoder re-emits the ENTIRE tool block after every
        message whose role is ``system``, and agent harnesses inject one such
        message per turn -- a session hook, server instructions, a remaining-token
        notice. With a large tool set that rewrites tens of thousands of tokens at
        a fresh position on every request, so no two consecutive turns share a
        prefix and the cache can never be consulted usefully.

        Measured on this deployment 2026-09-16, one worker, identical task:
        conversational turns reused 1.6-4.2% of their prompt with the roles as
        sent, and 99.3-99.7% with them re-labelled. An ablation isolated the
        cause: removing ``context_management``, ``output_config`` or ``metadata``
        changed nothing, while re-labelling alone moved a 48,159-token turn from
        0.0% to 99.4%. Across a fleet the waste was the dominant cost of serving
        -- a width-1 phase spent 23.0 of 23.7 minutes of card time re-computing a
        prefix the engine would otherwise have had.

        ``user`` is the encoder's own reading rather than an invention: it treats
        a mid-conversation system message as a user turn when deciding where the
        assistant header goes, and says so. Only messages after the first are
        touched, because a leading system message is the conventional way to open
        a conversation and the encoder handles it without re-emitting tools. The
        body is re-encoded only when a message actually changed, so an untouched
        request passes through byte-for-byte.
        """
        messages = payload.get("messages") if isinstance(payload, Mapping) else None
        if not isinstance(messages, list):
            return payload, body
        repaired: list[Any] = []
        changed = False
        for index, message in enumerate(messages):
            if (
                index > 0
                and isinstance(message, Mapping)
                and message.get("role") == "system"
            ):
                amended = dict(message)
                amended["role"] = "user"
                repaired.append(amended)
                changed = True
            else:
                repaired.append(message)
        if not changed:
            return payload, body
        amended_payload = dict(payload)
        amended_payload["messages"] = repaired
        logger.info(
            "router prefix-repair relabelled=%d messages=%d",
            sum(
                1
                for index, message in enumerate(messages)
                if index > 0
                and isinstance(message, Mapping)
                and message.get("role") == "system"
            ),
            len(messages),
        )
        return amended_payload, json.dumps(
            amended_payload, ensure_ascii=False, separators=(",", ":")
        ).encode()

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
    resolver: UpstreamResolver,
    *,
    lane_document: Path | None = None,
    request_receipts_path: Path | None = None,
) -> RouterApp:
    """Build the ASGI application around an injected upstream resolver."""
    return RouterApp(
        resolver,
        lane_document=lane_document,
        request_receipts_path=request_receipts_path,
    )


def serve_router(
    resolver: UpstreamResolver,
    *,
    host: str = "0.0.0.0",
    port: int,
    lane_document: Path | None = None,
    request_receipts_path: Path | None = None,
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
        create_router_app(
            resolver,
            lane_document=lane_document,
            request_receipts_path=request_receipts_path,
        ),
        host=host,
        port=port,
    )
