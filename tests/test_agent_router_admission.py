"""Published router admission state and request gate wait accounting."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from imas_ambix.agent.router import RouterApp, Upstream
from tests.test_agent_router import Resolver, _invoke, _server, _status
from tests.test_request_receipts import (
    _read_rows,
    _request_body,
    _router_with_receipts,
    _sse_engine,
)


async def _receive_from(queue: asyncio.Queue[dict[str, Any]]) -> dict[str, Any]:
    return await queue.get()


async def _acquire_slots(
    app: RouterApp, count: int
) -> tuple[list[Any], list[asyncio.Queue[dict[str, Any]]]]:
    admissions = []
    queues = []
    for _ in range(count):
        incoming: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        queues.append(incoming)
        admissions.append(
            await app._generation_gate.acquire(
                lambda incoming=incoming: _receive_from(incoming)
            )
        )
    return admissions, queues


async def _release_slots(app: RouterApp, admissions: list[Any]) -> None:
    for admission in admissions:
        if admission.disconnect_task is not None:
            admission.disconnect_task.cancel()
            await asyncio.gather(admission.disconnect_task, return_exceptions=True)
        await app._generation_gate.release()


def test_lane_document_publishes_gate_headroom_and_queue_age(tmp_path: Path) -> None:
    async def exercise() -> None:
        gate_file = tmp_path / "router-gate.json"
        lane_document = tmp_path / "lane.json"
        gate_file.write_text(
            json.dumps({"width": 14, "wait_seconds": 1.0}), encoding="utf-8"
        )
        lane_document.write_text(
            json.dumps({"headroom": 4, "running": 10, "waiting": 0}),
            encoding="utf-8",
        )
        app = RouterApp(Resolver([]), lane_document=lane_document, gate_file=gate_file)
        admitted, _ = await _acquire_slots(app, 14)
        waiting_queues: list[asyncio.Queue[dict[str, Any]]] = []
        waiting_tasks = []
        for _ in range(9):
            incoming: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
            waiting_queues.append(incoming)
            waiting_tasks.append(
                asyncio.create_task(
                    app._generation_gate.acquire(
                        lambda incoming=incoming: _receive_from(incoming)
                    )
                )
            )
        for _ in range(20):
            if app._generation_gate.waiting == 9:
                break
            await asyncio.sleep(0.01)
        assert app._generation_gate.waiting == 9
        await asyncio.sleep(0.05)

        app._publish_gate_snapshot()
        published = json.loads(lane_document.read_text(encoding="utf-8"))
        assert published["engine_headroom"] == 4
        assert published["headroom"] == -9
        assert published["admission"]["headroom"] == -9
        assert published["admission"]["verdict"] == "congested"
        assert published["admission"]["waiting"] == 9
        assert published["admission"]["oldest_wait_seconds"] >= 0.04

        for incoming in waiting_queues:
            incoming.put_nowait({"type": "http.disconnect"})
        for task in waiting_tasks:
            result = await task
            assert result.outcome == "disconnected"
        await _release_slots(app, admitted)

    asyncio.run(exercise())


def test_admission_headroom_reports_open_and_full_states(tmp_path: Path) -> None:
    async def exercise() -> None:
        gate_file = tmp_path / "router-gate.json"
        gate_file.write_text(
            json.dumps({"width": 14, "wait_seconds": 1.0}), encoding="utf-8"
        )
        app = RouterApp(Resolver([]), gate_file=gate_file)

        admitted, _ = await _acquire_slots(app, 10)
        open_state = app._generation_gate.admission_snapshot()
        assert open_state == {
            "headroom": 4,
            "oldest_wait_seconds": None,
            "verdict": "open",
            "waiting": 0,
        }
        await _release_slots(app, admitted)

        admitted, _ = await _acquire_slots(app, 14)
        full_state = app._generation_gate.admission_snapshot()
        assert full_state == {
            "headroom": 0,
            "oldest_wait_seconds": None,
            "verdict": "full",
            "waiting": 0,
        }

        gate_file.write_text(
            json.dumps({"width": 12, "wait_seconds": 1.0}), encoding="utf-8"
        )
        app._generation_gate._next_config_refresh_at = 0.0
        narrowed_state = app._generation_gate.admission_snapshot()
        assert narrowed_state == {
            "headroom": -2,
            "oldest_wait_seconds": None,
            "verdict": "full",
            "waiting": 0,
        }
        await _release_slots(app, admitted)

    asyncio.run(exercise())


def test_paused_admission_reports_zero_headroom_and_paused_verdict(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        gate_file = tmp_path / "router-gate.json"
        gate_file.write_text(
            json.dumps({"width": 14, "wait_seconds": 1.0}), encoding="utf-8"
        )
        app = RouterApp(Resolver([]), gate_file=gate_file)
        admitted, _ = await _acquire_slots(app, 3)

        gate_file.write_text(
            json.dumps(
                {"width": 14, "wait_seconds": 1.0, "paused": True, "reason": "test"}
            ),
            encoding="utf-8",
        )
        app._generation_gate._next_config_refresh_at = 0.0
        paused_state = app._generation_gate.admission_snapshot()
        assert paused_state == {
            "headroom": 0,
            "oldest_wait_seconds": None,
            "verdict": "paused",
            "waiting": 0,
        }
        await _release_slots(app, admitted)

    asyncio.run(exercise())


def test_completed_receipt_records_controlled_gate_wait(tmp_path: Path) -> None:
    async def exercise() -> None:
        receipts = tmp_path / "requests.jsonl"
        gate_file = tmp_path / "router-gate.json"
        gate_file.write_text(
            json.dumps({"width": 1, "wait_seconds": 1.0}), encoding="utf-8"
        )
        async with (
            _server(_sse_engine()) as engine_url,
            _router_with_receipts(
                [Upstream(engine_url)], receipts, gate_file=gate_file
            ) as app,
        ):
            held_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
            held = await app._generation_gate.acquire(lambda: _receive_from(held_queue))
            request = asyncio.create_task(
                _invoke(app, "POST", "/v1/chat/completions", _request_body())
            )
            for _ in range(20):
                if app._generation_gate.waiting == 1:
                    break
                await asyncio.sleep(0.01)
            assert app._generation_gate.waiting == 1
            await asyncio.sleep(0.05)
            await app._generation_gate.release()
            if held.disconnect_task is not None:
                held.disconnect_task.cancel()
                await asyncio.gather(held.disconnect_task, return_exceptions=True)
            response = await request
            assert _status(response) == 200

        rows = _read_rows(receipts)
        assert len(rows) == 1
        assert rows[0]["gate_wait_s"] >= 0.04
        assert rows[0]["gate_wait_s"] < 0.5

    asyncio.run(exercise())


def test_gate_timeout_receipt_records_wait(tmp_path: Path) -> None:
    async def exercise() -> None:
        receipts = tmp_path / "requests.jsonl"
        gate_file = tmp_path / "router-gate.json"
        gate_file.write_text(
            json.dumps({"width": 1, "wait_seconds": 0.05}), encoding="utf-8"
        )
        async with (
            _server(_sse_engine()) as engine_url,
            _router_with_receipts(
                [Upstream(engine_url)], receipts, gate_file=gate_file
            ) as app,
        ):
            held_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
            held = await app._generation_gate.acquire(lambda: _receive_from(held_queue))
            response = await _invoke(
                app, "POST", "/v1/chat/completions", _request_body()
            )
            assert _status(response) == 529
            await app._generation_gate.release()
            if held.disconnect_task is not None:
                held.disconnect_task.cancel()
                await asyncio.gather(held.disconnect_task, return_exceptions=True)

        rows = _read_rows(receipts)
        assert len(rows) == 1
        assert rows[0]["status"] == "failed"
        assert rows[0]["gate_wait_s"] >= 0.04

    asyncio.run(exercise())
