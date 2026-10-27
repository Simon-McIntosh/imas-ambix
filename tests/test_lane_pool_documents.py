"""One lane document per pool, composed from each engine's own reading.

A pool is one model; an engine is one served process. The composer writes the
default pool's document at ``lane.json`` with a ``lanes`` index, and every other
pool's document at ``lanes/<model id>.json``, so a reader of today's shape is
unaffected and a second model's backend can name its own file.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
from pathlib import Path

import pytest
from aiohttp import web

from imas_ambix.agent import lane, router
from imas_ambix.agent.router import RouterApp, Upstream

_REPO = Path(__file__).resolve().parents[1]


def _sglang_metrics(
    model: str, pool_tokens: int, occupancy: float, running: int, *, generated: int = 0
) -> str:
    labels = f'{{model_name="{model}",tp_rank="0"}}'
    return "".join(
        [
            f"sglang:max_total_num_tokens{labels} {pool_tokens}\n",
            f"sglang:num_running_reqs{labels} {running}\n",
            f"sglang:num_queue_reqs{labels} 0\n",
            f"sglang:full_token_usage{labels} {occupancy}\n",
            f"sglang:generation_tokens_total{labels} {generated}\n",
        ]
    )


class _Resolver:
    def __init__(self, upstreams) -> None:
        self._upstreams = tuple(upstreams)

    async def resolve(self):
        return self._upstreams


class _Engine:
    """One fixture engine: a metrics endpoint serving one reading."""

    def __init__(self, body: str) -> None:
        self.body = body
        self.runner = None
        self.base_url = ""

    async def start(self) -> None:
        async def metrics(_: web.Request) -> web.Response:
            return web.Response(text=self.body, content_type="text/plain")

        application = web.Application()
        application.router.add_get("/metrics", metrics)
        self.runner = web.AppRunner(application)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        self.base_url = f"http://127.0.0.1:{port}"

    async def stop(self) -> None:
        if self.runner is not None:
            await self.runner.cleanup()


async def _publish_once(tmp_path: Path, resolver: _Resolver, *, until) -> None:
    """Drive one publishing tick and wait until *until* holds."""
    app = RouterApp(resolver, lane_document=tmp_path / "lane.json", lane_interval=3600)
    task = asyncio.create_task(app._publish_lane())
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 5
    try:
        while loop.time() < deadline:
            if until(tmp_path):
                break
            await asyncio.sleep(0.02)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        if app._session is not None:
            await app._session.close()


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_one_engine_golden_document_is_the_base_shape_plus_the_lanes_index(tmp_path):
    """From one engine's reading, lane.json is today's document plus the index.

    The golden is the one-engine writer's own output for the same reading: the
    composer must add the ``lanes`` index and nothing else, so a reader of
    today's shape sees the keys and values it already sees.
    """
    body = _sglang_metrics("fixture-a", 4_000_000, 0.30, 10, generated=500_000)

    async def exercise() -> tuple[dict, dict]:
        engine = _Engine(body)
        await engine.start()
        try:
            # Read the engine directly before the loop is held by the router's
            # own request, so the golden is the same reading the router saw.
            capacity = await asyncio.to_thread(
                lane.fetch_lane_capacity, engine.base_url
            )
            await _publish_once(
                tmp_path,
                _Resolver([Upstream(engine.base_url, model_id="fixture-a")]),
                until=lambda base: (base / "lane.json").exists(),
            )
        finally:
            await engine.stop()
        published = _read(tmp_path / "lane.json")
        base = _read(
            lane.write_lane_document(capacity, tmp_path / "base.json", window=None)
        )
        return published, base

    published, base = asyncio.run(exercise())
    assert published["lanes"] == {"fixture-a": "lane.json"}
    assert published["model_id"] == "fixture-a"
    assert published["pool_tokens"] == 4_000_000
    assert published["running"] == 10
    assert published["state"] == "measured"
    # Every key the one-engine writer publishes is present, and the only keys
    # the composer adds are the pool index and the router's own gate block --
    # which the base revision also merged into lane.json after the reading.
    assert set(published) - set(base) == {
        "lanes",
        "router_generation_gate",
        "admission",
        "engine_headroom",
        "availability_percent",
    }
    assert not set(base) - set(published)
    for key in ("pool_tokens", "running", "waiting", "kv_occupancy", "mean_context"):
        assert published[key] == base[key], key
    assert (tmp_path / "lanes").exists() is False


def test_two_pools_write_two_documents_and_lane_json_keeps_the_default(
    tmp_path, monkeypatch
):
    """A second pool is its own file; lane.json stays the default pool's."""
    monkeypatch.delenv("AMBIX_AGENT_PREFERRED_RELEASE", raising=False)
    first = _Engine(_sglang_metrics("model-a", 4_000_000, 0.30, 10))
    second = _Engine(_sglang_metrics("model-b", 2_000_000, 0.20, 4))

    async def exercise() -> tuple[dict, dict]:
        await first.start()
        await second.start()
        try:
            await _publish_once(
                tmp_path,
                _Resolver(
                    [
                        Upstream(first.base_url, model_id="model-a"),
                        Upstream(second.base_url, model_id="model-b"),
                    ]
                ),
                until=lambda base: (base / "lanes" / "model-b.json").exists(),
            )
        finally:
            await first.stop()
            await second.stop()
        return _read(tmp_path / "lane.json"), _read(tmp_path / "lanes" / "model-b.json")

    default, other = asyncio.run(exercise())
    assert default["model_id"] == "model-a"
    assert default["lanes"] == {
        "model-a": "lane.json",
        "model-b": "lanes/model-b.json",
    }
    # lane.json carries no second copy of model-b's fields beyond the index.
    assert default["pool_tokens"] == 4_000_000
    assert other["model_id"] == "model-b"
    assert other["pool_tokens"] == 2_000_000
    assert other["state"] == "measured"
    assert "lanes" not in other


def test_preferred_release_names_the_default_pool(tmp_path, monkeypatch):
    """The deployment's preferred model owns lane.json whatever resolved first."""
    monkeypatch.setenv("AMBIX_AGENT_PREFERRED_RELEASE", "model-b")
    first = _Engine(_sglang_metrics("model-a", 4_000_000, 0.30, 10))
    second = _Engine(_sglang_metrics("model-b", 2_000_000, 0.20, 4))

    async def exercise() -> dict:
        await first.start()
        await second.start()
        try:
            await _publish_once(
                tmp_path,
                _Resolver(
                    [
                        Upstream(first.base_url, model_id="model-a"),
                        Upstream(second.base_url, model_id="model-b"),
                    ]
                ),
                until=lambda base: (base / "lanes" / "model-a.json").exists(),
            )
        finally:
            await first.stop()
            await second.stop()
        return _read(tmp_path / "lane.json")

    default = asyncio.run(exercise())
    assert default["model_id"] == "model-b"
    assert default["lanes"] == {
        "model-a": "lanes/model-a.json",
        "model-b": "lane.json",
    }


def test_a_pool_spanning_two_engines_sums_counts_and_carries_each_engine(
    tmp_path, monkeypatch
):
    """Two engines on one model compose into one measured pool document.

    Counts and capacities sum, the token rate sums, the mean context is the
    running-weighted mean, and ``engines`` carries each engine's own reading.
    """
    monkeypatch.delenv("AMBIX_AGENT_PREFERRED_RELEASE", raising=False)
    # Engine one: pool 4,000,000, occupancy 0.4, running 20 -> mean 80,000,
    # 400,000 tokens generated over the tick.
    first = _Engine(_sglang_metrics("model-a", 4_000_000, 0.4, 20))
    # Engine two: pool 2,000,000, occupancy 0.2, running 4 -> mean 100,000.
    second = _Engine(_sglang_metrics("model-a", 2_000_000, 0.2, 4))

    async def exercise() -> dict:
        await first.start()
        await second.start()
        try:
            await _publish_once(
                tmp_path,
                _Resolver(
                    [
                        Upstream(first.base_url, model_id="model-a"),
                        Upstream(second.base_url, model_id="model-a"),
                    ]
                ),
                until=lambda base: (
                    (base / "lane.json").exists()
                    and len(_read(base / "lane.json").get("engines", ())) == 2
                ),
            )
        finally:
            await first.stop()
            await second.stop()
        return _read(tmp_path / "lane.json")

    document = asyncio.run(exercise())
    assert document["model_id"] == "model-a"
    assert document["pool_tokens"] == 6_000_000
    assert document["running"] == 24
    assert document["waiting"] == 0
    assert len(document["engines"]) == 2
    # The mean context is each engine's mean weighted by its running count:
    # (80,000 * 20 + 100,000 * 4) / 24 = 83,333.
    assert document["mean_context"] == pytest.approx(83_333.3, rel=1e-3)
    engines = {entry["engine_id"]: entry for entry in document["engines"]}
    assert set(engines) == {
        router.engine_id_for_origin(first.base_url),
        router.engine_id_for_origin(second.base_url),
    }
    # The pool's gate width, in-flight count and queue depth are the two
    # engines' own summed.
    gate = document["router_generation_gate"]
    engine_gates = [entry["router_generation_gate"] for entry in document["engines"]]
    assert gate["effective_width"] == sum(g["effective_width"] for g in engine_gates)
    assert gate["in_flight"] == sum(g["in_flight"] for g in engine_gates)
    assert gate["waiting"] == sum(g["waiting"] for g in engine_gates)


def test_one_engine_stopped_leaves_its_pool_unavailable_and_the_other_measured(
    tmp_path, monkeypatch
):
    """A stopped engine's pool reads unavailable; the other pool stays measured.

    Two single-engine pools, and the second engine's endpoint is closed before
    the tick runs. Its document must say so explicitly rather than ageing into
    a plausible reading, while the pool that answered stays a measurement.
    """
    monkeypatch.delenv("AMBIX_AGENT_PREFERRED_RELEASE", raising=False)
    live = _Engine(_sglang_metrics("model-a", 4_000_000, 0.30, 10))
    stopped = _Engine(_sglang_metrics("model-b", 2_000_000, 0.20, 4))

    async def exercise() -> tuple[dict, dict]:
        await live.start()
        await stopped.start()
        stopped_url = stopped.base_url
        await stopped.stop()
        try:
            await _publish_once(
                tmp_path,
                _Resolver(
                    [
                        Upstream(live.base_url, model_id="model-a"),
                        Upstream(stopped_url, model_id="model-b"),
                    ]
                ),
                until=lambda base: (base / "lanes" / "model-b.json").exists(),
            )
        finally:
            await live.stop()
        return _read(tmp_path / "lane.json"), _read(tmp_path / "lanes" / "model-b.json")

    default, other = asyncio.run(exercise())
    assert default["state"] == "measured"
    assert default["model_id"] == "model-a"
    assert other["state"] == "unavailable"
    assert other["model_id"] == "model-b"
    assert other["reason"]
    assert "headroom" not in other


def test_a_pool_document_reads_through_reckon_the_same_as_lane_json(tmp_path):
    """reckon's reader resolves a pool document exactly as it reads lane.json.

    The contract is that a second model's document is the same shape, so a
    reckon backend pointing its lane document at that file needs no reckon
    change. The reader resolves a path through ``read_lane_document_file``;
    ``read_lane_document`` takes the parsed object or its raw JSON text and
    reports a path as malformed, so passing it a path tests nothing.
    Skipped, with its reason printed, where reckon is not importable.
    """
    reason = "reckon is not importable here, so the contract cannot be read"
    lane_document = pytest.importorskip("reckon.crew.lane_document", reason=reason)
    document = lane.build_lane_document(
        lane.LaneCapacity(
            model_id="fixture-a",
            pool_tokens=4_000_000,
            running=10,
            waiting=0,
            kv_occupancy=0.3,
            preemptions=0,
            prefix_hit_rate=0.4,
        ),
        refresh_interval=30,
    )
    document["lanes"] = {"fixture-a": "lane.json"}
    destination = tmp_path / "lanes" / "fixture-a.json"
    lane.write_document(document, destination)

    read = lane_document.read_lane_document_file(destination)

    assert read["malformed"] is False, read.get("detail")
    assert read["unknown_fields"] == []
    assert read["state"] == "measured"
    assert read["running"] == 10
    assert read["kv_occupancy"] == pytest.approx(0.3)
    assert read["headroom"] == document["headroom"]


def test_no_module_writes_a_lane_document_outside_the_composer():
    """The composer is the only writer of any lane document.

    The lane command's own publisher retired with this section: it wrote
    ``lane.json`` from outside the router, so a guard covering only one of its
    two write calls would have passed while the other kept overwriting the
    default pool's document.
    """
    writers = ("write_lane_document", "write_unavailable_document", "write_document")
    allowed_module = "router.py"
    offenders = []
    for path in sorted((_REPO / "imas_ambix").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for writer in writers:
            if re.search(rf"\b{writer}\(", text):
                if path.name == "lane.py":
                    continue
                if path.name != allowed_module:
                    offenders.append(f"{path.relative_to(_REPO)}:{writer}")
    assert offenders == [], offenders
    # And inside the router, every call sits in the composer itself.
    text = (_REPO / "imas_ambix/agent/router.py").read_text(encoding="utf-8")
    composer = text[
        text.index("async def _publish_lane") : text.index("def _pool_engines")
    ]
    for writer in writers:
        assert text.count(f"{writer}(") == composer.count(f"{writer}("), writer


def test_the_lane_command_and_its_refresher_are_retired():
    """Neither the lane command nor its standing refresher generator survives."""
    from imas_ambix.agent import cli, slurm

    assert not hasattr(slurm, "generate_lane_refresher_script")
    commands = getattr(cli.agent, "commands", {})
    assert "lane" not in commands
