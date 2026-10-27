"""Which engine a session's requests reach, across a pool of several engines.

A pool is two or more engines serving one model id from different servers. A
request is keyed by the session it declares, and the key is held to one engine
so a conversation keeps the cache it has already built there. The first request
from a key places it, later requests follow the pin, and the pin yields only to
a pause, a lost engine, a queue wait past the operator's bound, or its own idle
horizon.

The engines here bind distinct loopback aliases so the two are different
servers, which is what makes them a pool rather than one server's rotation.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from aiohttp import web

from imas_ambix.agent.request_receipts import (
    COORDINATOR_SESSION_HEADER,
    RUN_ID_HEADER,
)
from imas_ambix.agent.router import (
    DEFAULT_FAILOVER_WAIT_SECONDS,
    PIN_IDLE_SECONDS,
    RouterApp,
    Upstream,
    engine_id_for_origin,
)

AsgiMessage = dict[str, Any]

MODEL_ID = "deepseek-v4-flash"

# Two distinct servers. A pool needs them distinct, because two engines on one
# server are a rotation and only one of them takes new keys.
LEFT_HOST = "127.0.0.1"
RIGHT_HOST = "127.0.0.2"


class Resolver:
    def __init__(self, upstreams: Sequence[Upstream]) -> None:
        self.upstreams = upstreams

    async def resolve(self) -> Sequence[Upstream]:
        return self.upstreams


class Engine:
    """An engine that names itself in its answer and can be held open."""

    def __init__(self, label: str, *, accelerators: int) -> None:
        self.label = label
        self.served: list[str] = []
        # Set once a request has been admitted and reached this engine, which is
        # how a test knows the engine's only slot is occupied.
        self.entered = asyncio.Event()
        self._hold = asyncio.Event()
        self._hold.set()
        self._card: dict[str, Any] = {
            "id": MODEL_ID,
            "object": "model",
            "owned_by": "engine",
            "max_model_len": 131_072,
            "created": 1_757_000_000,
            "ambix": {
                "accelerator_family": "H200 NVL",
                "accelerator_count": accelerators,
                "checkpoint_precision": "FP8",
            },
        }

    @property
    def card(self) -> dict[str, Any]:
        return self._card

    def hold(self) -> None:
        """Answer nothing until let_go, so the engine keeps its slot occupied."""
        self.entered.clear()
        self._hold.clear()

    def let_go(self) -> None:
        self._hold.set()

    def app(self) -> web.Application:
        app = web.Application()

        async def catalog(_: web.Request) -> web.Response:
            return web.json_response({"object": "list", "data": [self._card]})

        async def messages(_: web.Request) -> web.Response:
            self.entered.set()
            await self._hold.wait()
            self.served.append(self.label)
            return web.json_response({"served_by": self.label})

        app.router.add_get("/v1/models", catalog)
        app.router.add_post("/v1/messages", messages)
        return app


@asynccontextmanager
async def _bind(engine: Engine, host: str):
    runner = web.AppRunner(engine.app())
    await runner.setup()
    site = web.TCPSite(runner, host, 0)
    await site.start()
    sockets = site._server.sockets
    try:
        yield f"http://{host}:{sockets[0].getsockname()[1]}"
    finally:
        engine.let_go()
        await runner.cleanup()


@asynccontextmanager
async def _router(
    upstreams: Sequence[Upstream],
    *,
    gate_file: Path | None = None,
    receipts_path: Path | None = None,
):
    app = RouterApp(
        Resolver(upstreams),
        gate_file=gate_file,
        request_receipts_path=receipts_path,
        receipt_max_rows_per_s=1e9,
    )
    try:
        yield app
    finally:
        if app._session is not None:
            await app._session.close()
        if app._receipts is not None:
            app._receipts.close()


async def _invoke(
    app: RouterApp,
    *,
    session: str | None = None,
    run_id: str | None = None,
    client: tuple[str, int] | None = None,
    body: bytes | None = None,
    path: str = "/v1/messages",
) -> list[AsgiMessage]:
    headers = [(b"content-type", b"application/json")]
    if session is not None:
        headers.append((COORDINATOR_SESSION_HEADER.encode(), session.encode()))
    if run_id is not None:
        headers.append((RUN_ID_HEADER.encode(), run_id.encode()))
    incoming: asyncio.Queue[AsgiMessage] = asyncio.Queue()
    await incoming.put(
        {
            "type": "http.request",
            "body": body or json.dumps({"model": MODEL_ID}).encode(),
        }
    )
    sent: list[AsgiMessage] = []

    async def receive() -> AsgiMessage:
        return await incoming.get()

    async def send(message: AsgiMessage) -> None:
        sent.append(message)

    scope: dict[str, Any] = {
        "type": "http",
        "method": "POST",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": headers,
    }
    if client is not None:
        scope["client"] = client
    await app(scope, receive, send)
    return sent


def _body(messages: Sequence[AsgiMessage]) -> bytes:
    return b"".join(
        message.get("body", b"")
        for message in messages
        if message["type"] == "http.response.body"
    )


def _status(messages: Sequence[AsgiMessage]) -> int:
    return next(
        message["status"]
        for message in messages
        if message["type"] == "http.response.start"
    )


def _served_by(messages: Sequence[AsgiMessage]) -> str:
    return json.loads(_body(messages))["served_by"]


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _write_gate(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


async def _reconfigure(path: Path, payload: dict[str, Any]) -> None:
    """Rewrite the control file and wait past the router's refresh interval."""
    _write_gate(path, payload)
    await asyncio.sleep(1.1)


def test_fifty_requests_from_one_session_all_land_on_one_engine() -> None:
    """A session keeps the engine its first request was placed on."""

    async def exercise() -> None:
        left = Engine("left", accelerators=4)
        right = Engine("right", accelerators=8)
        async with (
            _bind(left, LEFT_HOST) as left_url,
            _bind(right, RIGHT_HOST) as right_url,
            _router([Upstream(left_url), Upstream(right_url)]) as app,
        ):
            reached = [
                _served_by(await _invoke(app, session="one-session")) for _ in range(50)
            ]
        assert len(set(reached)) == 1, reached
        assert reached[0] == "right", reached

    asyncio.run(exercise())


def test_sixteen_sessions_split_by_the_gate_widths_they_land_on(
    tmp_path: Path,
) -> None:
    """Placement counts pins per unit of width, so 16 sessions split 12:4."""

    async def exercise() -> None:
        # The widths that spread sessions are the gates', not the cards'; the
        # engines differ in card count only so a tie between them has a ground.
        left = Engine("left", accelerators=4)
        right = Engine("right", accelerators=8)
        async with (
            _bind(left, LEFT_HOST) as left_url,
            _bind(right, RIGHT_HOST) as right_url,
        ):
            gate_file = tmp_path / "router-gate.json"
            _write_gate(
                gate_file,
                {
                    "engines": {
                        engine_id_for_origin(left_url): {"width": 4},
                        engine_id_for_origin(right_url): {"width": 12},
                    }
                },
            )
            async with _router(
                [Upstream(left_url), Upstream(right_url)], gate_file=gate_file
            ) as app:
                reached = [
                    _served_by(await _invoke(app, session=f"session-{index}"))
                    for index in range(16)
                ]
        assert reached.count("left") + reached.count("right") == 16
        assert reached.count("right") / reached.count("left") == 3.0, reached
        assert reached.count("right") - reached.count("left") == 8

    asyncio.run(exercise())


def test_pausing_the_pinned_engine_fails_the_session_over(tmp_path: Path) -> None:
    """A pause is one of the three reasons a pin lets go, and the row says so."""

    async def exercise() -> None:
        left = Engine("left", accelerators=4)
        right = Engine("right", accelerators=8)
        async with (
            _bind(left, LEFT_HOST) as left_url,
            _bind(right, RIGHT_HOST) as right_url,
        ):
            gate_file = tmp_path / "router-gate.json"
            receipts = tmp_path / "receipts.jsonl"
            _write_gate(gate_file, {})
            async with _router(
                [Upstream(left_url), Upstream(right_url)],
                gate_file=gate_file,
                receipts_path=receipts,
            ) as app:
                assert _served_by(await _invoke(app, session="moving")) == "right"
                await _reconfigure(
                    gate_file,
                    {"engines": {engine_id_for_origin(right_url): {"paused": True}}},
                )
                moved = await _invoke(app, session="moving")
                assert _status(moved) == 200
                assert _served_by(moved) == "left"
                again = await _invoke(app, session="moving")
                assert _served_by(again) == "left"
        rows = _rows(receipts)
        assert [row["routing"] for row in rows] == ["placed", "failed-over", "pinned"]
        assert rows[1]["upstream"] == left_url

    asyncio.run(exercise())


def test_a_pin_expires_after_its_idle_horizon(tmp_path: Path) -> None:
    """A key idle past the horizon is placed afresh, not failed over.

    The engine it held is paused here so the two outcomes are distinguishable
    by the word the row carries: an expired key is a new placement, while a
    live key whose engine paused is a failover.
    """

    async def exercise() -> None:
        left = Engine("left", accelerators=4)
        right = Engine("right", accelerators=8)
        async with (
            _bind(left, LEFT_HOST) as left_url,
            _bind(right, RIGHT_HOST) as right_url,
        ):
            gate_file = tmp_path / "router-gate.json"
            receipts = tmp_path / "receipts.jsonl"
            _write_gate(gate_file, {})
            async with _router(
                [Upstream(left_url), Upstream(right_url)],
                gate_file=gate_file,
                receipts_path=receipts,
            ) as app:
                assert _served_by(await _invoke(app, session="idle")) == "right"
                await _reconfigure(
                    gate_file,
                    {"engines": {engine_id_for_origin(right_url): {"paused": True}}},
                )
                pin = app._engine_pins["session:idle"]
                pin.touched = time.monotonic() - (PIN_IDLE_SECONDS + 1.0)
                assert _served_by(await _invoke(app, session="idle")) == "left"
        rows = _rows(receipts)
        assert rows[-1]["routing"] == "placed"
        assert PIN_IDLE_SECONDS == 180.0

    asyncio.run(exercise())


def test_the_failover_bound_defaults_to_thirty_seconds_and_the_file_sets_it(
    tmp_path: Path,
) -> None:
    """The bound a pinned request waits is operator configuration."""

    async def exercise() -> None:
        gate_file = tmp_path / "router-gate.json"
        engine_id = engine_id_for_origin(f"http://{LEFT_HOST}:18000")
        other = engine_id_for_origin(f"http://{RIGHT_HOST}:18000")
        _write_gate(gate_file, {"engines": {engine_id: {"failover_wait_seconds": 7.5}}})
        async with _router([], gate_file=gate_file) as app:
            assert app.gate_for(engine_id).settings().failover_wait_seconds == 7.5
            assert (
                app.gate_for(other).settings().failover_wait_seconds
                == DEFAULT_FAILOVER_WAIT_SECONDS
            )
        assert DEFAULT_FAILOVER_WAIT_SECONDS == 30.0

    asyncio.run(exercise())


def test_a_zero_bound_fails_a_pinned_session_over_from_a_full_engine(
    tmp_path: Path,
) -> None:
    """A bound of zero says a pinned request never waits: it moves at once."""

    async def exercise() -> None:
        left = Engine("left", accelerators=4)
        right = Engine("right", accelerators=8)
        async with (
            _bind(left, LEFT_HOST) as left_url,
            _bind(right, RIGHT_HOST) as right_url,
        ):
            gate_file = tmp_path / "router-gate.json"
            receipts = tmp_path / "receipts.jsonl"
            _write_gate(gate_file, {})
            # Both sessions are pinned to the right engine before the pool has
            # a second engine.
            upstreams = [Upstream(right_url)]
            async with _router(
                upstreams, gate_file=gate_file, receipts_path=receipts
            ) as app:
                right_id = engine_id_for_origin(right_url)
                assert _served_by(await _invoke(app, session="moved")) == "right"
                assert _served_by(await _invoke(app, session="blocker")) == "right"
                upstreams.append(Upstream(left_url))
                # One slot at the pinned engine, held by another session, so a
                # request with a bound of zero finds nowhere to wait.
                await _reconfigure(
                    gate_file,
                    {"engines": {right_id: {"width": 1, "failover_wait_seconds": 0}}},
                )
                right.hold()
                blocker = asyncio.create_task(_invoke(app, session="blocker"))
                await asyncio.wait_for(right.entered.wait(), 5)
                moved = asyncio.create_task(_invoke(app, session="moved"))
                await asyncio.sleep(0.2)
                right.let_go()
                assert _served_by(await asyncio.wait_for(moved, 5)) == "left"
                assert _status(await asyncio.wait_for(blocker, 5)) == 200
        rows = {row["coordinator_session"]: row for row in _rows(receipts)}
        assert rows["moved"]["routing"] == "failed-over"
        assert rows["moved"]["upstream"] == left_url

    asyncio.run(exercise())


def test_an_unset_bound_keeps_a_pinned_session_at_its_serving_engine(
    tmp_path: Path,
) -> None:
    """Without a bound the pin holds: a full queue is waited out.

    This is the arm the bound needs beside it, because a router that moved a
    session whenever its engine was busy would pass the zero-bound test above
    while having no affinity at all.
    """

    async def exercise() -> None:
        left = Engine("left", accelerators=4)
        right = Engine("right", accelerators=8)
        async with (
            _bind(left, LEFT_HOST) as left_url,
            _bind(right, RIGHT_HOST) as right_url,
        ):
            gate_file = tmp_path / "router-gate.json"
            receipts = tmp_path / "receipts.jsonl"
            _write_gate(gate_file, {})
            # Both sessions are pinned to the right engine before the pool has
            # a second engine.
            upstreams = [Upstream(right_url)]
            async with _router(
                upstreams, gate_file=gate_file, receipts_path=receipts
            ) as app:
                right_id = engine_id_for_origin(right_url)
                assert _served_by(await _invoke(app, session="waiting")) == "right"
                assert _served_by(await _invoke(app, session="blocker")) == "right"
                upstreams.append(Upstream(left_url))
                await _reconfigure(gate_file, {"engines": {right_id: {"width": 1}}})
                right.hold()
                blocker = asyncio.create_task(_invoke(app, session="blocker"))
                await asyncio.wait_for(right.entered.wait(), 5)
                waiting = asyncio.create_task(_invoke(app, session="waiting"))
                await asyncio.sleep(0.2)
                right.let_go()
                assert _served_by(await asyncio.wait_for(waiting, 5)) == "right"
                assert _status(await asyncio.wait_for(blocker, 5)) == 200
        rows = {row["coordinator_session"]: row for row in _rows(receipts)}
        assert rows["waiting"]["routing"] == "pinned"
        assert rows["waiting"]["upstream"] == right_url
        assert rows["waiting"]["gate_wait_s"] > 0

    asyncio.run(exercise())


def test_a_request_held_for_its_engine_records_why_it_was_placed(
    tmp_path: Path,
) -> None:
    """A request the router held until an engine appeared still says ``placed``.

    The readiness predicate runs inside the gate, so its decision reaches the
    row through the caller's own variable rather than its return value. A row
    that reads null here would say the request reached its engine for no reason
    the router can name, while llm-watch counts by that column.
    """

    async def exercise() -> None:
        engine = Engine("only", accelerators=8)
        async with _bind(engine, LEFT_HOST) as engine_url:
            receipts = tmp_path / "receipts.jsonl"
            resolver = Resolver([])
            app = RouterApp(
                resolver,
                request_receipts_path=receipts,
                receipt_max_rows_per_s=1e9,
            )
            try:
                held = asyncio.create_task(_invoke(app, session="late"))
                await asyncio.sleep(0.2)
                resolver.upstreams = [Upstream(engine_url)]
                assert _served_by(await asyncio.wait_for(held, 10)) == "only"
            finally:
                if app._session is not None:
                    await app._session.close()
                if app._receipts is not None:
                    app._receipts.close()

    asyncio.run(exercise())
    rows = _rows(tmp_path / "receipts.jsonl")
    assert [row["routing"] for row in rows] == ["placed"]


def test_a_placement_that_stays_on_the_pinned_engine_is_recorded_as_pinned(
    tmp_path: Path,
) -> None:
    """Pausing every gate must not make an unmoved key look like a failover.

    A paused gate holds rather than refusing, so the decision is asserted where
    it is made: the placement rule's own fallback pool, when every candidate is
    paused, can return the engine the key is already on, and that is a pin
    holding rather than a key moving.
    """

    async def exercise() -> None:
        engine = Engine("only", accelerators=8)
        async with _bind(engine, LEFT_HOST) as engine_url:
            gate_file = tmp_path / "router-gate.json"
            _write_gate(gate_file, {})
            async with _router([Upstream(engine_url)], gate_file=gate_file) as app:
                assert _served_by(await _invoke(app, session="am")) == "only"
                await _reconfigure(
                    gate_file,
                    {"engines": {engine_id_for_origin(engine_url): {"paused": True}}},
                )
                pin = app._engine_pins["session:am"]
                before = pin.touched
                owners = [(Upstream(engine_url), engine.card)]
                selected, reason = app._select_engine(owners, "session:am")
                assert reason == "pinned"
                assert engine_id_for_origin(selected[0].base_url) == pin.engine_id
                assert app._engine_pins["session:am"].engine_id == pin.engine_id
                assert app._engine_pins["session:am"].touched >= before

    asyncio.run(exercise())


def test_a_failover_records_the_wait_at_both_gates_and_the_admitting_engine(
    tmp_path: Path,
) -> None:
    """The row owes the whole wait, not the last queue's share of it.

    A keyed request can queue at its own engine, give up on the bound, and be
    admitted at a peer. Counting only the second admission understates the wait
    by everything spent at the first, and leaving the engine blank leaves a
    reader unable to attribute the wait to the queue that charged it.
    """

    async def exercise() -> None:
        left = Engine("left", accelerators=4)
        right = Engine("right", accelerators=8)
        async with (
            _bind(left, LEFT_HOST) as left_url,
            _bind(right, RIGHT_HOST) as right_url,
        ):
            gate_file = tmp_path / "router-gate.json"
            receipts = tmp_path / "receipts.jsonl"
            _write_gate(gate_file, {})
            upstreams = [Upstream(right_url)]
            async with _router(
                upstreams, gate_file=gate_file, receipts_path=receipts
            ) as app:
                right_id = engine_id_for_origin(right_url)
                assert _served_by(await _invoke(app, session="moved")) == "right"
                assert _served_by(await _invoke(app, session="blocker")) == "right"
                upstreams.append(Upstream(left_url))
                await _reconfigure(
                    gate_file,
                    {"engines": {right_id: {"width": 1, "failover_wait_seconds": 0.4}}},
                )
                right.hold()
                blocker = asyncio.create_task(_invoke(app, session="blocker"))
                await asyncio.wait_for(right.entered.wait(), 5)
                moved = asyncio.create_task(_invoke(app, session="moved"))
                served = await asyncio.wait_for(moved, 10)
                assert _served_by(served) == "left"
                right.let_go()
                assert _status(await asyncio.wait_for(blocker, 5)) == 200
        rows = _rows(receipts)
        moved_row = next(
            r
            for r in rows
            if r["coordinator_session"] == "moved" and r["upstream"] == left_url
        )
        assert moved_row["routing"] == "failed-over"
        assert moved_row["gate_wait_s"] >= 0.35
        assert moved_row["gate_engine"] == engine_id_for_origin(left_url)

    asyncio.run(exercise())


def test_a_pin_whose_engine_leaves_the_catalog_fails_over(
    tmp_path: Path,
) -> None:
    """An engine that stops answering is the second reason a pin lets go."""

    async def exercise() -> None:
        left = Engine("left", accelerators=4)
        right = Engine("right", accelerators=8)
        async with (
            _bind(left, LEFT_HOST) as left_url,
            _bind(right, RIGHT_HOST) as right_url,
        ):
            receipts = tmp_path / "receipts.jsonl"
            resolver = Resolver([Upstream(left_url), Upstream(right_url)])
            app = RouterApp(
                resolver,
                request_receipts_path=receipts,
                receipt_max_rows_per_s=1e9,
            )
            try:
                assert _served_by(await _invoke(app, session="orphan")) == "right"
                resolver.upstreams = [Upstream(left_url)]
                moved = await _invoke(app, session="orphan")
                assert _served_by(moved) == "left"
                again = await _invoke(app, session="orphan")
                assert _served_by(again) == "left"
            finally:
                if app._session is not None:
                    await app._session.close()
                if app._receipts is not None:
                    app._receipts.close()

    asyncio.run(exercise())
    rows = _rows(tmp_path / "receipts.jsonl")
    assert [row["routing"] for row in rows] == ["placed", "failed-over", "pinned"]


def test_the_key_falls_back_from_the_session_to_the_run_to_the_caller_host(
    tmp_path: Path,
) -> None:
    """A caller that names no session is still held to one engine."""

    async def exercise() -> None:
        left = Engine("left", accelerators=4)
        right = Engine("right", accelerators=8)
        async with (
            _bind(left, LEFT_HOST) as left_url,
            _bind(right, RIGHT_HOST) as right_url,
        ):
            receipts = tmp_path / "receipts.jsonl"
            async with _router(
                [Upstream(left_url), Upstream(right_url)], receipts_path=receipts
            ) as app:
                await _invoke(app, session="s1", run_id="r1")
                assert "session:s1" in app._engine_pins
                assert "run:r1" not in app._engine_pins
                await _invoke(app, run_id="r2")
                assert "run:r2" in app._engine_pins
                assert "host:203.0.113.9" not in app._engine_pins
                first = await _invoke(app, client=("203.0.113.9", 4000))
                second = await _invoke(app, client=("203.0.113.9", 4001))
                assert "host:203.0.113.9" in app._engine_pins
                assert _served_by(first) == _served_by(second)
        rows = _rows(receipts)
        assert rows[-1]["routing"] == "pinned"

    asyncio.run(exercise())
