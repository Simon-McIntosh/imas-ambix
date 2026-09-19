"""The lane publisher must leave a readable failure receipt in the router log."""

from __future__ import annotations

import asyncio
import contextlib
import gzip
import json
import logging

from aiohttp import web

from imas_ambix.agent import lane, router
from imas_ambix.agent.router import RouterApp, Upstream


class _RefusedMetricsResolver:
    async def resolve(self):
        return (Upstream("http://127.0.0.1:1"),)


class _MetricsResolver:
    def __init__(self, base_url: str) -> None:
        self._base_url = base_url

    async def resolve(self):
        return (Upstream(self._base_url),)


_SGLANG_METRICS = '''\
sglang:max_total_num_tokens{model_name="fixture",tp_rank="0"} 4000000
sglang:num_running_reqs{model_name="fixture",tp_rank="0"} 2
sglang:num_queue_reqs{model_name="fixture",tp_rank="0"} 0
sglang:full_token_usage{model_name="fixture",tp_rank="0"} 0.25
'''


def test_lane_publish_refusal_names_target_and_exception(tmp_path, monkeypatch, caplog):
    """A refused metrics connection must not make an unavailable lane silent."""

    async def exercise() -> None:
        published = asyncio.Event()
        writer = lane.write_unavailable_document

        def record_unavailable(reason, destination):
            result = writer(reason, destination)
            published.set()
            return result

        monkeypatch.setattr(lane, "write_unavailable_document", record_unavailable)
        app = RouterApp(
            _RefusedMetricsResolver(),
            lane_document=tmp_path / "lane.json",
            lane_interval=3600,
        )
        task = asyncio.create_task(app._publish_lane())
        try:
            await asyncio.wait_for(published.wait(), timeout=2)
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            if app._session is not None:
                await app._session.close()

        lane_state = json.loads((tmp_path / "lane.json").read_text(encoding="utf-8"))
        assert lane_state["state"] == "unavailable"

    with caplog.at_level(logging.WARNING, logger=router.__name__):
        asyncio.run(exercise())

    messages = [record.getMessage() for record in caplog.records]
    assert any("http://127.0.0.1:1/metrics" in message for message in messages)
    assert any("URLError" in message for message in messages)
    assert any("Connection refused" in message for message in messages)


def test_lane_publish_reads_gzip_metrics_body(tmp_path, monkeypatch):
    """The standing publisher must parse a compressed metrics response."""

    async def exercise() -> None:
        async def metrics(_: web.Request) -> web.Response:
            return web.Response(
                body=gzip.compress(_SGLANG_METRICS.encode("utf-8")),
                headers={"Content-Encoding": "gzip"},
            )

        application = web.Application()
        application.router.add_get("/metrics", metrics)
        runner = web.AppRunner(application)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        sockets = site._server.sockets
        port = sockets[0].getsockname()[1]

        published = asyncio.Event()
        writer = lane.write_lane_document

        def record_document(capacity, destination, **kwargs):
            result = writer(capacity, destination, **kwargs)
            published.set()
            return result

        monkeypatch.setattr(lane, "write_lane_document", record_document)
        app = RouterApp(
            _MetricsResolver(f"http://127.0.0.1:{port}"),
            lane_document=tmp_path / "lane.json",
            lane_interval=3600,
        )
        task = asyncio.create_task(app._publish_lane())
        try:
            await asyncio.wait_for(published.wait(), timeout=2)
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            if app._session is not None:
                await app._session.close()
            await runner.cleanup()

        lane_state = json.loads((tmp_path / "lane.json").read_text(encoding="utf-8"))
        assert lane_state["state"] == "measured"
        assert lane_state["pool_tokens"] == 4_000_000

    asyncio.run(exercise())
