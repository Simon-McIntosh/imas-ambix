"""The lane publisher outlives an upstream resolution that fails in any way.

Resolving the router's upstreams asks the scheduler for its running jobs, and a
scheduler error arrives as a ``click.ClickException``. The publisher loop caught
only transport and parse failures, so that exception ended its task while the
router went on serving: on 24 November ``lane.json`` stopped at 04:33 UTC and
every reader of the lane saw a two-hour-old reading. A failed resolution now
publishes the unavailable state, and the publisher keeps running to try again.
"""

from __future__ import annotations

import asyncio
import contextlib
import json

import click

from imas_ambix.agent import lane
from imas_ambix.agent.router import RouterApp


class _SchedulerErrorResolver:
    async def resolve(self):
        raise click.ClickException("slurm_load_jobs error: Unexpected message received")


def test_a_scheduler_error_publishes_unavailable_and_keeps_the_publisher(
    tmp_path, monkeypatch
) -> None:
    async def exercise() -> bool:
        published = asyncio.Event()
        writer = lane.write_unavailable_document

        def record_unavailable(reason, destination, **kwargs):
            result = writer(reason, destination, **kwargs)
            published.set()
            return result

        monkeypatch.setattr(lane, "write_unavailable_document", record_unavailable)
        app = RouterApp(
            _SchedulerErrorResolver(),
            lane_document=tmp_path / "lane.json",
            lane_interval=3600,
        )
        task = asyncio.create_task(app._publish_lane())
        try:
            await asyncio.wait_for(published.wait(), timeout=5)
            await asyncio.sleep(0.05)
            alive = not task.done()
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            if app._session is not None:
                await app._session.close()
        return alive

    alive = asyncio.run(exercise())
    lane_state = json.loads((tmp_path / "lane.json").read_text(encoding="utf-8"))
    assert lane_state["state"] == "unavailable"
    assert "slurm_load_jobs" in json.dumps(lane_state)
    assert alive, "the publisher task ended on a scheduler error"
