"""The lane publisher must leave a readable failure receipt in the router log."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging

from imas_ambix.agent import lane, router
from imas_ambix.agent.router import RouterApp, Upstream


class _RefusedMetricsResolver:
    async def resolve(self):
        return (Upstream("http://127.0.0.1:1"),)


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
    assert any("ClientConnectorError" in message for message in messages)
    assert any("Cannot connect to host" in message for message in messages)
