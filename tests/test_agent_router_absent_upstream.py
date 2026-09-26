"""A generation request for a known model whose engine is absent waits, not 404s.

An engine comes and goes: a serve restarts, a rotation leaves a gap. A model id
the deployment can serve must not read as meaningless during that gap, so a
generation request for it holds its place in the generation gate's FIFO and is
relayed the moment an engine advertising it appears. Only an id nothing can
serve is refused as unknown. These tests drive a stub engine and a stub catalog
and never touch the network.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from aiohttp import web

from imas_ambix.agent.router import RouterApp, Upstream
from tests.test_agent_router import (
    SendMessage,
    _body,
    _card,
    _server,
    _status,
)

# A model id no shipped profile names, so it is known only once this process has
# resolved an owner for it.
_RESOLVED_MODEL = "held-model"
# A model id a shipped profile declares as its client-visible name, so a fresh
# router that has resolved nothing still knows it. ``deepseek-v4-flash.toml``
# sets ``model.served_name = "deepseek-v4-flash"``.
_PROFILE_MODEL = "deepseek-v4-flash"
_UNKNOWN_MODEL = "no-such-model"


class MutableResolver:
    """A resolver whose upstream set a test can change while a call is waiting."""

    def __init__(self, upstreams: Sequence[Upstream] = ()) -> None:
        self.upstreams = list(upstreams)

    async def resolve(self) -> Sequence[Upstream]:
        return list(self.upstreams)


def _engine(model_id: str) -> web.Application:
    """A stub engine advertising one model id and answering generation."""
    app = web.Application()

    async def catalog(_: web.Request) -> web.Response:
        return web.json_response(
            {"object": "list", "data": [_card(model_id, context=4096, count=4)]}
        )

    async def generate(request: web.Request) -> web.Response:
        payload = await request.json()
        return web.json_response({"model": model_id, "echo": payload.get("model")})

    app.router.add_get("/v1/models", catalog)
    app.router.add_post("/v1/messages", generate)
    app.router.add_post("/v1/chat/completions", generate)
    return app


def _write_gate(
    path: Path,
    *,
    width: int = 4,
    wait_seconds: float = 5.0,
    paused: bool = False,
) -> None:
    payload: dict[str, Any] = {"width": width, "wait_seconds": wait_seconds}
    if paused:
        payload["paused"] = True
        payload["reason"] = "test"
    path.write_text(json.dumps(payload), encoding="utf-8")


def _request_body(model_id: str) -> bytes:
    return json.dumps({"model": model_id, "max_tokens": 16}).encode()


@asynccontextmanager
async def _router(
    resolver: MutableResolver,
    gate_file: Path,
    receipts_path: Path | None = None,
):
    app = RouterApp(resolver, gate_file=gate_file, request_receipts_path=receipts_path)
    try:
        yield app
    finally:
        if app._session is not None:
            await app._session.close()
        if app._receipts is not None:
            app._receipts.close()


def _start_call(
    app: RouterApp, path: str, body: bytes
) -> tuple[asyncio.Task[None], list[SendMessage]]:
    """Begin a call without awaiting it, so a held request can be observed."""
    incoming: asyncio.Queue[SendMessage] = asyncio.Queue()
    incoming.put_nowait({"type": "http.request", "body": body})
    sent: list[SendMessage] = []

    async def receive() -> SendMessage:
        return await incoming.get()

    async def send(message: SendMessage) -> None:
        sent.append(message)

    task = asyncio.create_task(
        app(
            {
                "type": "http",
                "method": "POST",
                "path": path,
                "raw_path": path.encode(),
                "query_string": None,
                "headers": [(b"content-type", b"application/json")],
            },
            receive,
            send,
        )
    )
    return task, sent


async def _invoke(app: RouterApp, path: str, body: bytes) -> list[SendMessage]:
    task, sent = _start_call(app, path, body)
    await asyncio.wait_for(task, timeout=5)
    return sent


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def test_a_known_model_with_no_owner_is_held_then_relayed(tmp_path: Path) -> None:
    """The process resolves an id once, and a gap in its engine does not 404 it.

    The upstream is dropped from the resolver after the id is known, so the hold
    is driven by an engine disappearing rather than by an id that was never
    resolvable. If the id's knowledge were not kept the held request would be
    refused at once, and if the gate did not re-resolve on admission the request
    would either relay to no engine or never leave the queue.
    """

    async def exercise() -> None:
        gate = tmp_path / "router-gate.json"
        _write_gate(gate, width=4, wait_seconds=5.0)
        engine = _engine(_RESOLVED_MODEL)
        async with _server(engine) as engine_url:
            resolver = MutableResolver([Upstream(engine_url)])
            async with _router(resolver, gate) as app:
                # Resolve the id once, so it is known to this process.
                first = await _invoke(
                    app, "/v1/messages", _request_body(_RESOLVED_MODEL)
                )
                assert _status(first) == 200

                # The engine goes away: the id is known but has no owner.
                resolver.upstreams = []
                held_task, held_sent = _start_call(
                    app, "/v1/messages", _request_body(_RESOLVED_MODEL)
                )
                await asyncio.sleep(0.4)
                assert not held_task.done()
                assert held_sent == []

                # The engine appears again: the held request relays.
                resolver.upstreams = [Upstream(engine_url)]
                await asyncio.wait_for(held_task, timeout=5)
                assert _status(held_sent) == 200

    asyncio.run(exercise())


def test_a_known_model_past_the_wait_gets_a_retryable_overload_not_a_404(
    tmp_path: Path,
) -> None:
    """A model whose engine never appears times out as a queue wait, not unknown.

    The answer is the gate's own wait expiry -- 529 with a short Retry-After and
    an overloaded_error body -- so a client retries the same turn rather than
    treating the model as unavailable.
    """

    async def exercise() -> None:
        gate = tmp_path / "router-gate.json"
        _write_gate(gate, width=4, wait_seconds=0.2)
        receipts = tmp_path / "requests.jsonl"
        engine = _engine(_RESOLVED_MODEL)
        async with _server(engine) as engine_url:
            resolver = MutableResolver([Upstream(engine_url)])
            async with _router(resolver, gate, receipts) as app:
                assert (
                    _status(
                        await _invoke(
                            app, "/v1/messages", _request_body(_RESOLVED_MODEL)
                        )
                    )
                    == 200
                )
                resolver.upstreams = []
                sent = await _invoke(
                    app, "/v1/messages", _request_body(_RESOLVED_MODEL)
                )

        assert _status(sent) == 529
        start = next(
            message for message in sent if message["type"] == "http.response.start"
        )
        headers = {name.lower(): value for name, value in start["headers"]}
        assert headers[b"retry-after"] == b"5"
        assert json.loads(_body(sent))["error"]["type"] == "overloaded_error"

        rows = _read_rows(receipts)
        assert len(rows) == 2
        timed_out = rows[-1]
        assert timed_out["model"] == _RESOLVED_MODEL
        assert timed_out["gate_wait_s"] is not None
        assert timed_out["gate_wait_s"] >= 0.2

    asyncio.run(exercise())


def test_a_paused_gate_holds_a_known_model_with_no_owner(tmp_path: Path) -> None:
    """The pause holds an absent-upstream request exactly as a resolvable one.

    Nothing is admitted while the pause holds, so the request stays in the FIFO;
    clearing the pause and bringing the engine up relays it, which shows the hold
    was the pause rather than a refusal.
    """

    async def exercise() -> None:
        gate = tmp_path / "router-gate.json"
        _write_gate(gate, width=4, wait_seconds=5.0, paused=True)
        resolver = MutableResolver()
        async with _router(resolver, gate) as app:
            held_task, held_sent = _start_call(
                app, "/v1/messages", _request_body(_PROFILE_MODEL)
            )
            await asyncio.sleep(0.4)
            assert not held_task.done()
            assert held_sent == []

            # The pause is cleared and the engine arrives together.
            engine = _engine(_PROFILE_MODEL)
            async with _server(engine) as engine_url:
                resolver.upstreams = [Upstream(engine_url)]
                _write_gate(gate, width=4, wait_seconds=5.0, paused=False)
                await asyncio.wait_for(held_task, timeout=6)
            assert _status(held_sent) == 200

    asyncio.run(exercise())


def test_an_unknown_model_is_still_refused_at_once(tmp_path: Path) -> None:
    """An id nothing can serve and no profile names keeps today's 404."""

    async def exercise() -> None:
        gate = tmp_path / "router-gate.json"
        _write_gate(gate, width=4, wait_seconds=5.0)
        async with _router(MutableResolver(), gate) as app:
            sent = await _invoke(app, "/v1/messages", _request_body(_UNKNOWN_MODEL))

        assert _status(sent) == 404
        assert _UNKNOWN_MODEL in _body(sent).decode()

    asyncio.run(exercise())


def test_a_profile_named_model_is_known_on_a_fresh_router(tmp_path: Path) -> None:
    """A model a shipped profile serves is known before any engine is seen.

    A fresh router has resolved nothing, so its only knowledge of the id is the
    profile set. It must hold and then time out as a queue wait rather than 404,
    which is what makes a lane that has not yet come up routable by name.
    """

    async def exercise() -> None:
        gate = tmp_path / "router-gate.json"
        _write_gate(gate, width=4, wait_seconds=0.2)
        async with _router(MutableResolver(), gate) as app:
            sent = await _invoke(app, "/v1/messages", _request_body(_PROFILE_MODEL))

        assert _status(sent) == 529
        assert json.loads(_body(sent))["error"]["type"] == "overloaded_error"

    asyncio.run(exercise())
