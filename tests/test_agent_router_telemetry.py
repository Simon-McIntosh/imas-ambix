"""The router's per-request receipt stream, as the production launch reaches it.

Driven through the harness in ``tests.test_agent_router`` and the receipt
fixtures beside it, so these tests measure the same router object the router's
own tests do rather than building a second way to drive it.

What this file holds is the wiring rather than the row contract: the launch
command names exactly one path -- the lane document -- so the stream is
reachable only if the router derives it from there, and nothing below asserts
the derived location or that the outcomes this section exists to record arrive
through it.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from imas_ambix.agent.request_receipts import SELF_ANSWERED_UPSTREAM
from imas_ambix.agent.router import RouterApp, Upstream
from tests.test_agent_router import (
    Resolver,
    _body,
    _invoke,
    _server,
    _status,
)
from tests.test_request_receipts import (
    _invoke_over,
    _read_rows,
    _request_body,
    _sse_engine,
)

SendMessage = dict[str, Any]


@asynccontextmanager
async def _router_on(lane_document: Path, upstreams: Sequence[Upstream]):
    """Build the router the way the launch command builds it: one path, no more.

    ``cli.py``'s router command passes the site's endpoint document with
    ``lane.json`` for a name and nothing else, so a test that hands over an
    explicit receipts path measures a construction production never uses.
    """
    app = RouterApp(Resolver(upstreams), lane_document=lane_document)
    try:
        yield app
    finally:
        if app._session is not None:
            await app._session.close()
        if app._receipts is not None:
            app._receipts.close()


def test_a_relayed_request_lands_a_row_beside_the_lane_document(
    tmp_path: Path,
) -> None:
    """The launch names the lane document, and the stream must follow from it.

    A relayed request is driven through the router built from that one path
    only, and the row is read back from where the launch would find it. If the
    derivation is dropped the relay still serves and no file appears, so a
    request record that exists only for callers passing an explicit path is
    exactly the state this assertion separates from a reachable one.
    """

    async def exercise() -> None:
        lane = tmp_path / "lane.json"
        engine = _sse_engine()
        async with (
            _server(engine) as engine_url,
            _router_on(lane, [Upstream(engine_url)]) as app,
        ):
            response = await _invoke(
                app, "POST", "/v1/chat/completions", _request_body()
            )
            assert _status(response) == 200
            assert _body(response)

        rows = _read_rows(tmp_path / "requests.jsonl")
        assert len(rows) == 1
        assert rows[0]["model"] == "streamer"
        assert rows[0]["upstream"] == engine_url
        assert rows[0]["status"] == "completed"
        assert rows[0]["prompt_tokens"] == 1234

    asyncio.run(exercise())


def test_a_caller_that_departs_before_the_body_is_read_leaves_one_aborted_row(
    tmp_path: Path,
) -> None:
    """The outcome this section exists to record rather than drop, through the wire.

    A caller that leaves while still uploading is answered nothing at all, so
    its only trace is this row. Driven through the launch-shaped router, so the
    departure is recorded at the location the launch derives and not only at an
    explicit one.
    """

    async def exercise() -> None:
        lane = tmp_path / "lane.json"
        engine = _sse_engine()
        head = json.dumps({"model": "streamer", "messages": []}).encode()
        async with (
            _server(engine) as engine_url,
            _router_on(lane, [Upstream(engine_url)]) as app,
        ):
            sent = await _invoke_over(
                app,
                "POST",
                "/v1/chat/completions",
                [
                    {"type": "http.request", "body": head[:10], "more_body": True},
                    {"type": "http.disconnect"},
                ],
            )
            assert sent == [], sent

        rows = _read_rows(tmp_path / "requests.jsonl")
        assert len(rows) == 1
        assert rows[0]["status"] == "aborted"
        assert rows[0]["upstream"] == SELF_ANSWERED_UPSTREAM
        assert rows[0]["model"] == ""

    asyncio.run(exercise())


def test_an_explicit_receipts_path_is_not_displaced_by_the_lane_document(
    tmp_path: Path,
) -> None:
    """An operator's chosen location wins, and the derived one is not also written.

    Both locations are accepted, so the precedence between them is behaviour
    rather than an implementation detail: a launch that redirects the stream
    must not leave a second one growing beside the lane document.
    """

    async def exercise() -> None:
        lane = tmp_path / "lane.json"
        chosen = tmp_path / "elsewhere" / "chosen.jsonl"
        chosen.parent.mkdir()
        engine = _sse_engine()

        class _EngineDiscoverableLater:
            """Resolve from a list filled once the engine's port is known."""

            async def resolve(self) -> Sequence[Upstream]:
                return list(upstreams)

        upstreams: list[Upstream] = []
        async with _server(engine) as engine_url:
            upstreams.append(Upstream(engine_url))
            app = RouterApp(
                _EngineDiscoverableLater(),
                lane_document=lane,
                request_receipts_path=chosen,
            )
            try:
                response = await _invoke(
                    app, "POST", "/v1/chat/completions", _request_body()
                )
                assert _status(response) == 200
            finally:
                if app._session is not None:
                    await app._session.close()
                if app._receipts is not None:
                    app._receipts.close()

        assert len(_read_rows(chosen)) == 1
        assert not (tmp_path / "requests.jsonl").exists()

    asyncio.run(exercise())
