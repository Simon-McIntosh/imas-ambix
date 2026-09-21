"""Per-request receipts from the router.

Driven through the router harness in ``tests.test_agent_router`` rather than
through a second way of exercising the router, so these tests measure the same
object the router's own tests do.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from collections.abc import Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from aiohttp import web

from imas_ambix.agent.request_receipts import (
    STATUS_ABORTED,
    STATUS_COMPLETED,
    STATUS_FAILED,
    RequestReceiptSink,
    StreamAccounting,
)
from imas_ambix.agent.router import RouterApp, Upstream
from tests.test_agent_router import (
    Resolver,
    _card,
    _invoke,
    _server,
    _status,
)

SendMessage = dict[str, Any]

_USAGE_EVENT = {
    "choices": [],
    "usage": {
        "prompt_tokens": 1234,
        "completion_tokens": 7,
        "prompt_tokens_details": {"cached_tokens": 1200},
    },
}


def _sse_events() -> list[dict[str, Any]]:
    return [
        {"choices": [{"delta": {"content": "he"}}]},
        {"choices": [{"delta": {"content": "llo"}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        _USAGE_EVENT,
    ]


def _sse_engine(*, status: int = 200, delay: float = 0.0) -> web.Application:
    """An engine whose completion streams SSE, optionally faulting or stalling."""

    async def catalog(_: web.Request) -> web.Response:
        return web.json_response(
            {"object": "list", "data": [_card("streamer", context=4096, count=2)]}
        )

    async def completion(_: web.Request) -> web.StreamResponse:
        if status != 200:
            return web.json_response({"error": "engine refused"}, status=status)
        response = web.StreamResponse(headers={"content-type": "text/event-stream"})
        await response.prepare(_)
        for event in _sse_events():
            await response.write(b"data: " + json.dumps(event).encode() + b"\n\n")
            if delay:
                await asyncio.sleep(delay)
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_get("/v1/models", catalog)
    app.router.add_post("/v1/chat/completions", completion)
    return app


@asynccontextmanager
async def _router_with_receipts(
    upstreams: Sequence[Upstream], receipts_path: Path | None
):
    app = RouterApp(Resolver(upstreams), request_receipts_path=receipts_path)
    try:
        yield app
    finally:
        if app._session is not None:
            await app._session.close()
        if app._receipts is not None:
            app._receipts.close()


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _request_body() -> bytes:
    return json.dumps({"model": "streamer", "messages": [], "stream": True}).encode()


def test_every_relayed_request_writes_one_fully_populated_row(tmp_path: Path) -> None:
    """N requests yield exactly N rows, each carrying the whole row contract."""

    async def exercise() -> None:
        receipts = tmp_path / "requests.jsonl"
        engine = _sse_engine()
        async with (
            _server(engine) as engine_url,
            _router_with_receipts([Upstream(engine_url)], receipts) as app,
        ):
            for _ in range(5):
                response = await _invoke(
                    app, "POST", "/v1/chat/completions", _request_body()
                )
                assert _status(response) == 200

        rows = _read_rows(receipts)
        assert len(rows) == 5
        for row in rows:
            assert row["model"] == "streamer"
            assert row["upstream"] == engine_url
            assert row["prompt_tokens"] == 1234
            assert row["cached_prompt_tokens"] == 1200
            assert row["completion_tokens"] == 7
            assert row["time_to_first_token_s"] is not None
            assert row["duration_s"] >= 0.0
            assert row["caller_hint"]
            assert row["status"] == STATUS_COMPLETED
            assert row["sample_fraction"] == 1.0
            assert row["timestamp"].endswith("Z")

    asyncio.run(exercise())


def test_aborted_request_is_recorded_as_aborted_rather_than_dropped(
    tmp_path: Path,
) -> None:
    """A caller that leaves mid-relay still produces exactly one row."""

    async def exercise() -> None:
        receipts = tmp_path / "requests.jsonl"
        engine = _sse_engine(delay=0.02)
        disconnected = False

        async def disconnect_after_first(
            message: SendMessage, incoming: asyncio.Queue[SendMessage]
        ) -> None:
            nonlocal disconnected
            if (
                message["type"] == "http.response.body"
                and message.get("body")
                and not disconnected
            ):
                disconnected = True
                await incoming.put({"type": "http.disconnect"})

        async with (
            _server(engine) as engine_url,
            _router_with_receipts([Upstream(engine_url)], receipts) as app,
        ):
            await _invoke(
                app,
                "POST",
                "/v1/chat/completions",
                _request_body(),
                on_send=disconnect_after_first,
            )

        rows = _read_rows(receipts)
        assert len(rows) == 1
        assert rows[0]["status"] == STATUS_ABORTED

    asyncio.run(exercise())


def test_upstream_error_is_recorded_as_failed(tmp_path: Path) -> None:
    """An engine refusal yields a row too, marked failed, with no usage at all."""

    async def exercise() -> None:
        receipts = tmp_path / "requests.jsonl"
        engine = _sse_engine(status=500)
        async with (
            _server(engine) as engine_url,
            _router_with_receipts([Upstream(engine_url)], receipts) as app,
        ):
            response = await _invoke(
                app, "POST", "/v1/chat/completions", _request_body()
            )
            assert _status(response) == 500

        rows = _read_rows(receipts)
        assert len(rows) == 1
        assert rows[0]["status"] == STATUS_FAILED
        assert rows[0]["prompt_tokens"] is None
        assert rows[0]["cached_prompt_tokens"] is None
        assert rows[0]["completion_tokens"] is None

    asyncio.run(exercise())


def test_no_receipts_path_configured_writes_nothing(tmp_path: Path) -> None:
    """Without a destination the router relays normally and records nothing."""

    async def exercise() -> None:
        engine = _sse_engine()
        async with (
            _server(engine) as engine_url,
            _router_with_receipts([Upstream(engine_url)], None) as app,
        ):
            response = await _invoke(
                app, "POST", "/v1/chat/completions", _request_body()
            )
            assert _status(response) == 200
            assert app._receipts is None

        assert list(tmp_path.iterdir()) == []

    asyncio.run(exercise())


class _Clock:
    """A monotonic clock the test advances by hand."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _idle_accounting() -> StreamAccounting:
    return StreamAccounting(clock=lambda: 0.0)


def _record_onto(
    sink: RequestReceiptSink, accounting: StreamAccounting, *, tick: float
) -> None:
    sink.record(
        model="streamer",
        upstream="http://engine",
        caller_hint="local",
        status=STATUS_COMPLETED,
        duration_s=tick,
        accounting=accounting,
    )


def test_sampling_records_the_fraction_it_kept_against_the_offered_rate(
    tmp_path: Path,
) -> None:
    """Above the configured rate the fraction stamped is kept over offered."""

    clock = _Clock()
    sink = RequestReceiptSink(
        tmp_path / "requests.jsonl",
        max_rows_per_s=2.0,
        window_s=1.0,
        random_float=lambda: 0.0,
        monotonic=clock,
    )
    for index in range(5):
        clock.now = index * 0.1
        _record_onto(sink, _idle_accounting(), tick=0.1)
    rows = _read_rows(tmp_path / "requests.jsonl")

    # Every row is kept because the injected draw is always below the fraction,
    # so the fractions themselves are what this test reads, and the first two
    # requests arrive within the configured rate and so keep the whole record.
    assert sink.rows_written == 5
    assert sink.rows_sampled_out == 0
    assert [row["sample_fraction"] for row in rows[:2]] == [1.0, 1.0]
    assert rows[2]["sample_fraction"] == round(2.0 / 3.0, 6)
    assert rows[4]["sample_fraction"] == round(2.0 / 5.0, 6)
    assert rows[4]["sampling_rate_per_s"] == 5.0
    sink.close()


def test_sampling_drops_rows_and_keeps_the_written_ones_self_describing(
    tmp_path: Path,
) -> None:
    """A draw above the fraction drops the row, and the kept rows still scale."""

    clock = _Clock()
    sink = RequestReceiptSink(
        tmp_path / "requests.jsonl",
        max_rows_per_s=2.0,
        window_s=1.0,
        random_float=lambda: 0.999,
        monotonic=clock,
    )
    for index in range(5):
        clock.now = index * 0.1
        _record_onto(sink, _idle_accounting(), tick=0.1)
    rows = _read_rows(tmp_path / "requests.jsonl")

    assert sink.rows_written == 2
    assert sink.rows_sampled_out == 3
    assert [row["sample_fraction"] for row in rows] == [1.0, 1.0]
    assert all(row["prompt_tokens"] is None for row in rows)
    sink.close()


def test_added_receipt_work_at_the_p99_over_two_hundred_requests(
    tmp_path: Path,
) -> None:
    """The tee parse plus the appended row stays under a millisecond at the p99.

    Measured as the work the receipts add to one request -- the parse of the
    relayed bytes and the flushed append -- and not end to end, because the row
    is written after the final response byte and an end-to-end figure would be
    dominated by engine and socket jitter that no change here moves.
    """

    sink = RequestReceiptSink(tmp_path / "requests.jsonl", max_rows_per_s=1e9)
    samples: list[float] = []
    for _ in range(200):
        began = time.perf_counter()
        accounting = StreamAccounting()
        for event in _sse_events():
            accounting.feed(b"data: " + json.dumps(event).encode() + b"\n\n")
        accounting.feed(b"data: [DONE]\n\n")
        accounting.finish()
        sink.record(
            model="streamer",
            upstream="http://engine",
            caller_hint="local",
            status=STATUS_COMPLETED,
            duration_s=0.0,
            accounting=accounting,
        )
        samples.append(time.perf_counter() - began)
    sink.close()

    samples.sort()
    p99 = samples[math.ceil(0.99 * len(samples)) - 1]
    median = samples[len(samples) // 2]
    print(
        f"receipt overhead over 200 requests: median {median * 1e6:.1f} us, p99 "
        f"{p99 * 1e6:.1f} us"
    )
    assert sink.rows_written == 200
    assert p99 < 0.001


def test_stream_accounting_reads_the_same_figures_as_the_benchmark_client() -> None:
    """The tee and the benchmark client agree on one stream's usage and prompt.

    The benchmark reads server-reported token counts and takes the first token
    at the first content delta or reasoning delta, whichever arrives first. The
    tee is asserted against those same two figures on the same bytes, so a
    relay figure and a benchmark figure remain comparable rather than merely
    similar.
    """

    from imas_ambix.agent.bench import _stream_chat

    async def exercise() -> None:
        engine = _sse_engine()
        async with _server(engine) as engine_url:
            accounting = StreamAccounting()
            for event in _sse_events():
                accounting.feed(b"data: " + json.dumps(event).encode() + b"\n\n")
            accounting.feed(b"data: [DONE]\n\n")
            accounting.finish()

            benched = await asyncio.to_thread(
                _stream_chat,
                engine_url,
                "streamer",
                [{"role": "user", "content": "hello"}],
                16,
            )

        assert benched.prompt_tokens == accounting.prompt_tokens == 1234
        assert benched.completion_tokens == accounting.completion_tokens == 7
        assert benched.time_to_first_token_s > 0.0
        assert accounting.time_to_first_token_s is not None

    asyncio.run(exercise())


def test_a_single_json_body_is_read_as_the_non_streaming_equivalent() -> None:
    """A body that streams nothing still yields its usage.

    The benchmark's non-streaming path reads usage and leaves the first-token
    time unset, and the tee matches that: a body that arrives whole has no
    first token to time from the relayed bytes.
    """

    accounting = StreamAccounting()
    body = json.dumps(
        {
            "choices": [{"message": {"role": "assistant", "content": "hello"}}],
            "usage": {
                "prompt_tokens": 40,
                "completion_tokens": 3,
                "prompt_tokens_details": {"cached_tokens": 32},
            },
        }
    ).encode()
    accounting.feed(body)
    accounting.finish()

    assert accounting.prompt_tokens == 40
    assert accounting.completion_tokens == 3
    assert accounting.cached_prompt_tokens == 32
    assert accounting.time_to_first_token_s is None


def test_a_truncated_body_is_not_read_as_a_confident_wrong_answer() -> None:
    """Past the head bound the fallback refuses rather than parsing a fragment."""

    import imas_ambix.agent.request_receipts as receipts

    accounting = StreamAccounting()
    accounting.feed(b"x" * (receipts.HEAD_BYTES + 1))
    accounting.finish()

    assert accounting.prompt_tokens is None
    assert accounting.completion_tokens is None


def test_an_unwritable_destination_cannot_disturb_the_relay(
    tmp_path: Path, caplog
) -> None:
    """A sink that cannot open its file logs once and drops, and the relay serves.

    The guarded case is made to happen rather than assumed: the destination's
    parent is a regular file, so opening the receipt stream raises and the
    router must still answer the caller.
    """

    async def exercise() -> None:
        blocker = tmp_path / "not-a-directory"
        blocker.write_text("", encoding="utf-8")
        engine = _sse_engine()
        async with (
            _server(engine) as engine_url,
            _router_with_receipts(
                [Upstream(engine_url)], blocker / "requests.jsonl"
            ) as app,
        ):
            response = await _invoke(
                app, "POST", "/v1/chat/completions", _request_body()
            )
            assert _status(response) == 200

        assert blocker.read_text(encoding="utf-8") == ""
        assert "request receipts disabled" in caplog.text

    asyncio.run(exercise())
