"""Routing when several reachable engines advertise one model id.

Two engines serving one released name is what a card-count cutover looks like
from the router: the wider engine joins the catalog as soon as it answers, and
both are healthy until the narrower one is retired. These tests pin which
engine a request reaches during that window, and pin that a pair the
preference cannot separate is still refused rather than picked at random.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from contextlib import asynccontextmanager
from typing import Any

from aiohttp import web

from imas_ambix.agent.router import RouterApp, Upstream

AsgiMessage = dict[str, Any]

MODEL_ID = "deepseek-v4-flash"


class Resolver:
    def __init__(self, upstreams: Sequence[Upstream]) -> None:
        self.upstreams = upstreams

    async def resolve(self) -> Sequence[Upstream]:
        return self.upstreams


def _card(*, accelerators: int, created: int | None = 1_757_000_000) -> dict[str, Any]:
    card: dict[str, Any] = {
        "id": MODEL_ID,
        "object": "model",
        "owned_by": "engine",
        "max_model_len": 131_072,
        "ambix": {
            "accelerator_family": "H200 NVL",
            "accelerator_count": accelerators,
            "checkpoint_precision": "FP8",
        },
    }
    if created is not None:
        card["created"] = created
    return card


def _engine(label: str, card: dict[str, Any]) -> web.Application:
    """Serve one card and answer inference with the label that identifies it."""
    app = web.Application()

    async def catalog(_: web.Request) -> web.Response:
        return web.json_response({"object": "list", "data": [card]})

    async def messages(_: web.Request) -> web.Response:
        return web.json_response({"served_by": label})

    app.router.add_get("/v1/models", catalog)
    app.router.add_post("/v1/messages", messages)
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


async def _invoke(app: RouterApp, body: bytes) -> list[AsgiMessage]:
    incoming: asyncio.Queue[AsgiMessage] = asyncio.Queue()
    await incoming.put({"type": "http.request", "body": body})
    sent: list[AsgiMessage] = []

    async def receive() -> AsgiMessage:
        return await incoming.get()

    async def send(message: AsgiMessage) -> None:
        sent.append(message)

    await app(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/messages",
            "raw_path": b"/v1/messages",
            "query_string": b"",
            "headers": [(b"content-type", b"application/json")],
        },
        receive,
        send,
    )
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


def _request() -> bytes:
    return json.dumps({"model": MODEL_ID}).encode()


def test_wider_engine_answers_while_both_card_counts_are_reachable() -> None:
    """The four-card engine takes the traffic, in either resolution order."""

    async def exercise() -> None:
        narrow = _engine("two-card", _card(accelerators=2))
        wide = _engine("four-card", _card(accelerators=4))
        async with _server(narrow) as narrow_url, _server(wide) as wide_url:
            for upstreams in (
                [Upstream(narrow_url), Upstream(wide_url)],
                [Upstream(wide_url), Upstream(narrow_url)],
            ):
                async with _router(upstreams) as app:
                    relayed = await _invoke(app, _request())
                    assert _status(relayed) == 200
                    assert json.loads(_body(relayed)) == {"served_by": "four-card"}

    asyncio.run(exercise())


def test_card_count_outranks_a_later_start() -> None:
    """A newer narrow engine does not pull traffic off a wider one."""

    async def exercise() -> None:
        narrow = _engine("two-card", _card(accelerators=2, created=1_757_009_999))
        wide = _engine("four-card", _card(accelerators=4, created=1_757_000_000))
        async with (
            _server(narrow) as narrow_url,
            _server(wide) as wide_url,
            _router([Upstream(narrow_url), Upstream(wide_url)]) as app,
        ):
            relayed = await _invoke(app, _request())
            assert _status(relayed) == 200
            assert json.loads(_body(relayed)) == {"served_by": "four-card"}

    asyncio.run(exercise())


def test_equal_card_counts_break_towards_the_later_start() -> None:
    """Two four-card engines resolve to the one that started more recently."""

    async def exercise() -> None:
        older = _engine("older", _card(accelerators=4, created=1_757_000_000))
        newer = _engine("newer", _card(accelerators=4, created=1_757_003_600))
        async with _server(older) as older_url, _server(newer) as newer_url:
            for upstreams in (
                [Upstream(older_url), Upstream(newer_url)],
                [Upstream(newer_url), Upstream(older_url)],
            ):
                async with _router(upstreams) as app:
                    relayed = await _invoke(app, _request())
                    assert _status(relayed) == 200
                    assert json.loads(_body(relayed)) == {"served_by": "newer"}

    asyncio.run(exercise())


def test_indistinguishable_engines_are_still_refused() -> None:
    """Equal cards and equal start stamps leave no ground for a choice."""

    async def exercise() -> None:
        card = _card(accelerators=4, created=1_757_000_000)
        first = _engine("first", card)
        second = _engine("second", card)
        async with (
            _server(first) as first_url,
            _server(second) as second_url,
            _router([Upstream(first_url), Upstream(second_url)]) as app,
        ):
            refused = await _invoke(app, _request())
            assert _status(refused) == 409
            assert f"duplicate model id: {MODEL_ID}".encode() in _body(refused)

    asyncio.run(exercise())


def test_a_missing_start_stamp_is_not_treated_as_oldest() -> None:
    """An unstamped engine cannot be ordered, so the pair is refused."""

    async def exercise() -> None:
        stamped = _engine("stamped", _card(accelerators=4, created=1_757_000_000))
        unstamped = _engine("unstamped", _card(accelerators=4, created=None))
        async with (
            _server(stamped) as stamped_url,
            _server(unstamped) as unstamped_url,
            _router([Upstream(stamped_url), Upstream(unstamped_url)]) as app,
        ):
            refused = await _invoke(app, _request())
            assert _status(refused) == 409

    asyncio.run(exercise())


def test_a_single_reachable_engine_still_answers() -> None:
    """The preference must not disturb the ordinary one-upstream path."""

    async def exercise() -> None:
        only = _engine("only", _card(accelerators=2))
        async with _server(only) as only_url, _router([Upstream(only_url)]) as app:
            relayed = await _invoke(app, _request())
            assert _status(relayed) == 200
            assert json.loads(_body(relayed)) == {"served_by": "only"}

    asyncio.run(exercise())
