"""A cut pause ends in-flight generation relays in the form the client retried.

These drive the router over a stub engine that streams slowly, with the gate
file written directly, so a cut is exercised exactly as an operator's
``cut: true`` reaches a router that already has relays in flight. The stub
engine records how each request ended, which is what lets a test tell a router
cut apart from the engine's own stream ending.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import aiohttp
from aiohttp import web

from imas_ambix.agent.router import RouterApp, Upstream
from tests.test_agent_router_gate import (
    Resolver,
    _await_gate,
    _body,
    _card,
    _headers,
    _server,
    _start_call,
    _status,
)
from tests.test_request_receipts import _read_rows, _router_with_receipts

AsgiMessage = dict[str, Any]

# The client connection being closed mid-stream reaches the stub engine as one
# of these, and nothing else in the handler raises: an engine whose write fails
# is the observable that the router cancelled its upstream request.
_UPSTREAM_ABORTS = (
    ConnectionResetError,
    aiohttp.ClientConnectionResetError,
    asyncio.CancelledError,
)


def _write_gate(
    path: Path,
    *,
    width: int,
    wait_seconds: float = 5.0,
    paused: bool = False,
    cut: bool = False,
    cut_form: str | None = None,
    reason: str | None = None,
) -> None:
    payload: dict[str, Any] = {"width": width, "wait_seconds": wait_seconds}
    if paused:
        payload["paused"] = True
    if cut:
        payload["cut"] = True
    if cut_form is not None:
        payload["cut_form"] = cut_form
    if reason is not None:
        payload["reason"] = reason
    path.write_text(json.dumps(payload), encoding="utf-8")


def _body_bytes(label: str) -> bytes:
    return json.dumps({"model": "streamer", "label": label}).encode()


def _body_frames(messages: Sequence[AsgiMessage]) -> list[AsgiMessage]:
    return [message for message in messages if message["type"] == "http.response.body"]


def _terminal_chunks(messages: Sequence[AsgiMessage]) -> list[AsgiMessage]:
    """The zero-length body that ends a response cleanly, absent from a cut."""
    return [
        message
        for message in _body_frames(messages)
        if not message.get("more_body", False)
    ]


def _relayed(messages: Sequence[AsgiMessage]) -> bytes:
    return b"data: ".join(
        message.get("body", b"")
        for message in _body_frames(messages)
        if message.get("more_body", False)
    )


class StreamingEngine:
    """A stub engine whose completion streams slowly and records its arrival.

    ``cancelled`` is the engine observing that the router dropped the request;
    ``completed`` is it reaching the end of its own stream. A test that asserts
    one of them shows which of the two ended a relay, which a status code
    alone cannot.
    """

    def __init__(self, *, chunks: int = 200, chunk_delay: float = 0.02) -> None:
        self.chunks = chunks
        self.chunk_delay = chunk_delay
        self.arrivals: list[str] = []
        self.started = asyncio.Event()
        self.completed = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def completion(self, request: web.Request) -> web.StreamResponse:
        payload = await request.json()
        self.arrivals.append(str(payload.get("label", "request")))
        self.started.set()
        response = web.StreamResponse(headers={"content-type": "text/event-stream"})
        await response.prepare(request)
        try:
            for index in range(self.chunks):
                frame = {"choices": [{"delta": {"content": str(index)}}]}
                await response.write(b"data: " + json.dumps(frame).encode() + b"\n\n")
                await asyncio.sleep(self.chunks and self.chunk_delay)
            await response.write(b"data: [DONE]\n\n")
            await response.write_eof()
        except _UPSTREAM_ABORTS:
            self.cancelled.set()
            raise
        self.completed.set()
        return response


class HeldEngine:
    """A stub engine that withholds its answer until its generation is done.

    A non-streaming request receives neither bytes nor response headers before
    the engine finishes, so its relay waits on the upstream response itself
    rather than inside the body loop where a streaming relay spends its time.
    ``cancelled`` is the engine observing that the router dropped the request
    before it answered -- the observable that tells a cut from an engine that
    simply took its time -- and ``completed`` is the engine reaching its own end.
    """

    def __init__(self, *, delay: float = 30.0) -> None:
        self.delay = delay
        self.arrivals: list[str] = []
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.completed = asyncio.Event()

    async def completion(self, request: web.Request) -> web.Response:
        payload = await request.json()
        self.arrivals.append(str(payload.get("label", "request")))
        self.started.set()
        try:
            await asyncio.sleep(self.delay)
        except _UPSTREAM_ABORTS:
            self.cancelled.set()
            raise
        self.completed.set()
        return web.json_response({"choices": [{"message": {"content": "done"}}]})


def _engine_app(engine: StreamingEngine | HeldEngine) -> web.Application:
    app = web.Application()

    async def catalog(_: web.Request) -> web.Response:
        return web.json_response({"object": "list", "data": [_card()]})

    app.router.add_get("/v1/models", catalog)
    app.router.add_post("/v1/messages", engine.completion)
    app.router.add_post("/v1/chat/completions", engine.completion)
    return app


@asynccontextmanager
async def _cancelling_server(app: web.Application):
    """Serve ``app``, cancelling a handler when its own client goes away.

    aiohttp runs a handler to completion on a client disconnect unless the
    server is asked not to, so a stub engine that withholds its answer cannot
    otherwise observe the router dropping its request -- and without that
    observation a cut is indistinguishable from an engine that took its time.
    """
    runner = web.AppRunner(app, handler_cancellation=True)
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
    app = RouterApp(Resolver([Upstream(upstream)]), gate_file=gate_file)
    try:
        yield app
    finally:
        if app._session is not None:
            await app._session.close()


async def _wait_started(engine: StreamingEngine) -> None:
    await asyncio.wait_for(engine.started.wait(), timeout=3.0)


def test_pause_cut_closes_the_client_stream_without_a_terminal_chunk(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        # cut_form absent and cut_form "close" are the same form: the default is
        # close, and both must leave the client without a terminal chunk.
        for cut_form in (None, "close"):
            gate_file = tmp_path / f"router-gate-{cut_form}.json"
            _write_gate(gate_file, width=4)
            engine = StreamingEngine()
            async with (
                _server(_engine_app(engine)) as upstream,
                _router(upstream, gate_file) as app,
            ):
                task, _, sent = _start_call(
                    app, "POST", "/v1/messages", _body_bytes("first")
                )
                await _wait_started(engine)
                began = asyncio.get_running_loop().time()
                _write_gate(
                    gate_file,
                    width=4,
                    paused=True,
                    cut=True,
                    cut_form=cut_form,
                )
                await asyncio.wait_for(task, timeout=10)
                elapsed = asyncio.get_running_loop().time() - began
                await asyncio.wait_for(engine.cancelled.wait(), timeout=3)

            assert elapsed < 10, f"cut_form={cut_form} took {elapsed:.1f}s"
            assert _status(sent) == 200
            assert _relayed(sent), "the relay forwarded no content before the cut"
            assert _terminal_chunks(sent) == [], (
                f"cut_form={cut_form} ended the stream instead of closing it"
            )
            assert engine.cancelled.is_set()
            assert not engine.completed.is_set()

    asyncio.run(exercise())


def test_pause_cut_in_the_error_form_writes_an_overloaded_event(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        gate_file = tmp_path / "router-gate.json"
        _write_gate(gate_file, width=4)
        engine = StreamingEngine()
        async with (
            _server(_engine_app(engine)) as upstream,
            _router(upstream, gate_file) as app,
        ):
            task, _, sent = _start_call(
                app, "POST", "/v1/messages", _body_bytes("first")
            )
            await _wait_started(engine)
            _write_gate(gate_file, width=4, paused=True, cut=True, cut_form="error")
            await asyncio.wait_for(task, timeout=10)
            await asyncio.wait_for(engine.cancelled.wait(), timeout=3)

        frames = _body_frames(sent)
        terminals = _terminal_chunks(sent)
        assert len(terminals) == 1
        error_index = next(
            index
            for index, frame in enumerate(frames)
            if b"overloaded_error" in frame.get("body", b"")
        )
        assert error_index < frames.index(terminals[0])
        error_body = frames[error_index]["body"]
        assert error_body.startswith(b"event: error\n")
        assert b"overloaded_error" in error_body
        # The client still received the engine's own SSE frames before the cut.
        assert _relayed(sent)
        assert engine.cancelled.is_set()

    asyncio.run(exercise())


def test_cut_pause_holds_new_requests_and_resume_admits_them_in_order(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        gate_file = tmp_path / "router-gate.json"
        _write_gate(gate_file, width=4)
        engine = StreamingEngine()
        async with (
            _server(_engine_app(engine)) as upstream,
            _router(upstream, gate_file) as app,
        ):
            first, _, first_sent = _start_call(
                app, "POST", "/v1/messages", _body_bytes("first")
            )
            await _wait_started(engine)
            _write_gate(gate_file, width=4, paused=True, cut=True, cut_form="close")
            await asyncio.wait_for(first, timeout=10)
            assert _terminal_chunks(first_sent) == []

            held = [
                _start_call(
                    app, "POST", "/v1/messages", _body_bytes(f"held-{index}")
                )[0]
                for index in range(2)
            ]
            await _await_gate(lambda: app._generation_gate.waiting == 2)
            await asyncio.sleep(0.1)
            # Nothing new reached the engine while the cut pause held.
            assert engine.arrivals == ["first"]

            _write_gate(gate_file, width=4)
            await _await_gate(lambda: not app._generation_gate.settings().paused)
            await asyncio.wait_for(asyncio.gather(*held), timeout=15)

        assert engine.arrivals == ["first", "held-0", "held-1"]

    asyncio.run(exercise())


def test_cut_written_before_a_fresh_router_is_honoured(tmp_path: Path) -> None:
    async def exercise() -> None:
        gate_file = tmp_path / "router-gate.json"
        # The gate file already carries the cut when the process starts, so the
        # fresh router has nothing in memory to remember it by.
        _write_gate(
            gate_file,
            width=4,
            paused=True,
            cut=True,
            cut_form="error",
            reason="cut across a router restart",
        )
        engine = StreamingEngine(chunks=5)
        async with (
            _server(_engine_app(engine)) as upstream,
            _router(upstream, gate_file) as app,
        ):
            settings = app._generation_gate.settings()
            assert settings.paused is True
            assert settings.cut is True
            assert settings.cut_form == "error"

            task, _, sent = _start_call(
                app, "POST", "/v1/messages", _body_bytes("held")
            )
            await _await_gate(lambda: app._generation_gate.waiting == 1)
            await asyncio.sleep(0.1)
            assert engine.arrivals == []

            _write_gate(gate_file, width=4)
            await _await_gate(lambda: not app._generation_gate.settings().paused)
            await asyncio.wait_for(task, timeout=5)

        assert _status(sent) == 200
        assert engine.arrivals == ["held"]

    asyncio.run(exercise())


def test_pause_without_cut_lets_the_in_flight_relay_finish(tmp_path: Path) -> None:
    async def exercise() -> None:
        gate_file = tmp_path / "router-gate.json"
        _write_gate(gate_file, width=4)
        engine = StreamingEngine(chunks=8)
        async with (
            _server(_engine_app(engine)) as upstream,
            _router(upstream, gate_file) as app,
        ):
            task, _, sent = _start_call(
                app, "POST", "/v1/messages", _body_bytes("first")
            )
            await _wait_started(engine)
            _write_gate(gate_file, width=4, paused=True)
            await _await_gate(lambda: app._generation_gate.settings().paused)
            await asyncio.wait_for(task, timeout=5)
            await asyncio.sleep(0.05)

        assert _status(sent) == 200
        assert len(_terminal_chunks(sent)) == 1
        assert engine.completed.is_set()
        assert not engine.cancelled.is_set()

    asyncio.run(exercise())


def test_pause_cut_ends_a_non_streaming_relay_with_a_retryable_overload(
    tmp_path: Path,
) -> None:
    """A relay that has not been answered yet is cut out of its upstream wait.

    A non-streaming request receives neither bytes nor response headers before
    the engine finishes, so the relay is waiting on the upstream response itself
    and never reaches the body loop a streaming relay spends its time in. The
    cut form does not apply here because nothing has been handed to the caller:
    every form answers the same retryable overload, which a client re-sends into
    the paused FIFO.
    """

    async def exercise() -> None:
        receipts = tmp_path / "requests.jsonl"
        for cut_form in (None, "close", "error"):
            gate_file = tmp_path / f"router-gate-held-{cut_form}.json"
            _write_gate(gate_file, width=4)
            engine = HeldEngine()
            async with (
                _cancelling_server(_engine_app(engine)) as upstream,
                _router_with_receipts(
                    [Upstream(upstream)],
                    receipts,
                    gate_file=gate_file,
                    receipt_max_rows_per_s=float("inf"),
                ) as app,
            ):
                body = json.dumps(
                    {"model": "streamer", "stream": False, "label": "held"}
                ).encode()
                task, _, sent = _start_call(
                    app, "POST", "/v1/chat/completions", body
                )
                await _wait_started(engine)
                began = asyncio.get_running_loop().time()
                _write_gate(
                    gate_file, width=4, paused=True, cut=True, cut_form=cut_form
                )
                await asyncio.wait_for(task, timeout=10)
                elapsed = asyncio.get_running_loop().time() - began
                await asyncio.wait_for(engine.cancelled.wait(), timeout=3)

            assert elapsed < 10, f"cut_form={cut_form} took {elapsed:.1f}s"
            assert _status(sent) == 529
            assert b"overloaded_error" in _body(sent)
            assert _headers(sent).get(b"retry-after")
            assert not engine.completed.is_set()

        # One row per cut form, each recording the relay the cut ended. The
        # cut is a departure the router performed, so the row says the answer
        # was never handed over rather than that the caller received it.
        rows = _read_rows(receipts)
        assert len(rows) == 3
        assert {row["status"] for row in rows} == {"aborted"}

    asyncio.run(exercise())
