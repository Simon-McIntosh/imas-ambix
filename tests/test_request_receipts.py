"""Per-request receipts from the router.

Driven through the router harness in ``tests.test_agent_router`` rather than
through a second way of exercising the router, so these tests measure the same
object the router's own tests do.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from collections.abc import Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import aiohttp
import pytest
from aiohttp import web

import imas_ambix.agent.request_receipts as request_receipts
from imas_ambix.agent.request_receipts import (
    MAX_IDENTITY_CHARS,
    SELF_ANSWERED_UPSTREAM,
    STATUS_ABORTED,
    STATUS_COMPLETED,
    STATUS_FAILED,
    RequestReceiptSink,
    StreamAccounting,
    identity_from_headers,
)
from imas_ambix.agent.router import (
    RECEIPT_MAX_ROWS_PER_S_ENV,
    RECEIPT_WINDOW_S_ENV,
    RouterApp,
    Upstream,
)
from tests.test_agent_router import (
    Resolver,
    _body,
    _card,
    _invoke,
    _server,
    _status,
)

SendMessage = dict[str, Any]

_FIRST_DELTA = {"choices": [{"delta": {"content": "he"}}]}

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


def _truncating_engine() -> web.Application:
    """An engine that accepts the request, answers 200, then aborts its transport.

    It declares a content-length, writes one SSE event of it, and drops the
    connection, so the relay's read of the body raises part-way through a
    response it has already begun forwarding as a success.
    """

    async def catalog(_: web.Request) -> web.Response:
        return web.json_response(
            {"object": "list", "data": [_card("streamer", context=4096, count=2)]}
        )

    async def completion(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(
            headers={"content-type": "text/event-stream", "content-length": "4096"}
        )
        await response.prepare(request)
        await response.write(b"data: " + json.dumps(_FIRST_DELTA).encode() + b"\n\n")
        request.transport.abort()
        return response

    app = web.Application()
    app.router.add_get("/v1/models", catalog)
    app.router.add_post("/v1/chat/completions", completion)
    return app


@asynccontextmanager
async def _router_with_receipts(
    upstreams: Sequence[Upstream],
    receipts_path: Path | None,
    **router_kwargs: Any,
):
    app = RouterApp(
        Resolver(upstreams), request_receipts_path=receipts_path, **router_kwargs
    )
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
            assert row["status"] == "completed"
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
        assert rows[0]["status"] == "aborted"

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
        assert rows[0]["status"] == "failed"
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


def test_an_upstream_that_dies_mid_body_is_recorded_as_failed(tmp_path: Path) -> None:
    """A 200 whose body aborts mid-relay leaves a failed row, not a completed one.

    The engine here answers 200, promises a content-length it never fulfils, and
    drops the transport part-way through the body. The caller has already been
    handed the 200 header, so the status alone attests to nothing the caller saw,
    and the row must carry the outcome the relay actually observed.
    """

    async def exercise() -> None:
        receipts = tmp_path / "requests.jsonl"
        engine = _truncating_engine()
        sent: list[SendMessage] = []

        async def capture(message: SendMessage, _: asyncio.Queue[SendMessage]) -> None:
            sent.append(message)

        async with (
            _server(engine) as engine_url,
            _router_with_receipts([Upstream(engine_url)], receipts) as app,
        ):
            with pytest.raises(aiohttp.ClientPayloadError):
                await _invoke(
                    app,
                    "POST",
                    "/v1/chat/completions",
                    _request_body(),
                    on_send=capture,
                )

        rows = _read_rows(receipts)
        # The caller did receive the engine's 200, which is what makes this the
        # case the row must not read as a success.
        assert _status(sent) == 200
        assert len(rows) == 1
        assert rows[0]["status"] == "failed"
        assert rows[0]["upstream"] == engine_url

    asyncio.run(exercise())


def test_a_request_the_router_answers_itself_is_recorded(tmp_path: Path) -> None:
    """A refusal and a catalog listing the router served leave rows of their own.

    Both are answered before any engine is chosen, so neither reaches the relay
    and neither is a request any engine saw. The record of what the router served
    is incomplete without them, and the refused request in particular is the one a
    caller reports as broken.
    """

    async def exercise() -> None:
        receipts = tmp_path / "requests.jsonl"
        engine = _sse_engine()
        async with (
            _server(engine) as engine_url,
            _router_with_receipts([Upstream(engine_url)], receipts) as app,
        ):
            refused = await _invoke(
                app,
                "POST",
                "/v1/chat/completions",
                json.dumps({"model": "nope", "messages": []}).encode(),
            )
            assert _status(refused) == 404
            listed = await _invoke(app, "GET", "/v1/models", b"")
            assert _status(listed) == 200
            malformed = await _invoke(app, "POST", "/v1/chat/completions", b"{not json")
            assert _status(malformed) == 400

        rows = _read_rows(receipts)
        assert len(rows) == 3

        assert rows[0]["status"] == "failed"
        assert rows[0]["model"] == "nope"
        # Written out rather than read from the module under test: a sentinel
        # assertion held by the constant it checks follows that constant
        # wherever it goes, including to an engine origin, which is the one
        # value it exists to be distinguishable from.
        assert rows[0]["upstream"] == "(router)"

        assert rows[1]["status"] == "completed"
        assert rows[1]["model"] == ""
        assert rows[1]["upstream"] == "(router)"

        assert rows[2]["status"] == "failed"
        assert rows[2]["model"] == ""

    asyncio.run(exercise())


def test_the_sampling_ceiling_is_read_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The ceiling is a setting, and the record shows which one was in force.

    An unhonoured ceiling and an honoured one differ only in the rows kept, so the
    ceiling in force is read from the rows themselves: at no ceiling every offered
    request is kept whole, and at one row per second the kept rows are few and each
    carries the fraction that lets a reader scale them back up.
    """
    caplog.set_level(logging.INFO, logger="imas_ambix.agent.router")

    offered = 60
    body = _request_body()

    async def exercise() -> None:
        engine = _sse_engine()
        async with _server(engine) as engine_url:
            unlimited = tmp_path / "unlimited.jsonl"
            monkeypatch.setenv(RECEIPT_MAX_ROWS_PER_S_ENV, "inf")
            async with _router_with_receipts([Upstream(engine_url)], unlimited) as app:
                for _ in range(offered):
                    await _invoke(app, "POST", "/v1/chat/completions", body)
            rows = _read_rows(unlimited)
            assert len(rows) == offered
            assert {row["sample_fraction"] for row in rows} == {1.0}
            assert "max_rows_per_s=inf" in caplog.text

            monkeypatch.setenv(RECEIPT_MAX_ROWS_PER_S_ENV, "1")
            throttled = tmp_path / "throttled.jsonl"
            async with _router_with_receipts(
                [Upstream(engine_url)], throttled
            ) as throttled_app:
                for _ in range(offered):
                    await _invoke(throttled_app, "POST", "/v1/chat/completions", body)
            sink = throttled_app._receipts
            assert sink is not None
            kept = _read_rows(throttled)
            assert sink.rows_written + sink.rows_sampled_out == offered
            assert sink.rows_sampled_out > 0
            assert len(kept) < offered
            # Each kept row states the rate it was weighed against and the share
            # it represents, so the rows reconstruct the offered traffic.
            for row in kept:
                assert row["sample_fraction"] * row["sampling_rate_per_s"] <= 1.05

    asyncio.run(exercise())


def test_a_receipt_setting_that_is_not_positive_stops_the_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed ceiling is refused at construction rather than silently defaulted.

    Both a value that is not a number and a ceiling nothing can stay under are
    caught before the router serves: the first would otherwise fall back to a
    default nobody chose, and the second would drop every row while reading as an
    unlimited ceiling.
    """
    engine = Upstream("http://engine")

    for value in ("nope", "0", "-1", "nan"):
        monkeypatch.setenv(RECEIPT_MAX_ROWS_PER_S_ENV, value)
        with pytest.raises(ValueError):
            RouterApp(Resolver([engine]), request_receipts_path=Path("requests.jsonl"))
        monkeypatch.delenv(RECEIPT_MAX_ROWS_PER_S_ENV)

    monkeypatch.setenv(RECEIPT_WINDOW_S_ENV, "0")
    with pytest.raises(ValueError):
        RouterApp(Resolver([engine]), request_receipts_path=Path("requests.jsonl"))


def test_the_self_answered_sentinel_is_the_value_a_reader_sums_around() -> None:
    """The sentinel's value is stated here, once, and nowhere else by reference.

    Every row assertion in this module names the literal, so this is the single
    place that fails loudly if the constant moves. A reader attributing rows per
    upstream splits traffic on this value: a row carrying an engine origin for a
    request no engine served folds the router's own answers into that engine's
    share, and a constant asserted against itself cannot notice.
    """
    assert SELF_ANSWERED_UPSTREAM == "(router)"


def test_the_outcome_labels_are_the_values_a_reader_sums_by() -> None:
    """The outcome words are stated here, once, and nowhere else by reference.

    Every row assertion in this module names the literal, which is what leaves
    this as the one place a moved label fails loudly. An assertion against the
    constant that writes the row follows that constant wherever it goes --
    including to a word no consumer of the record greps for -- and two sides
    that move together cannot fail at all, so a reader counting outcomes would
    have no test telling them the words they sum by had changed.
    """
    assert STATUS_COMPLETED == "completed"
    assert STATUS_ABORTED == "aborted"
    assert STATUS_FAILED == "failed"


async def _invoke_over(
    app: RouterApp, method: str, path: str, incoming: Sequence[SendMessage]
) -> list[SendMessage]:
    """Drive the app with a receive channel already holding the given messages.

    ``_invoke`` always hands over a complete ``http.request`` first, so it
    cannot express a caller that stops part-way or one that is already gone.
    """

    messages = list(incoming)
    sent: list[SendMessage] = []

    async def receive() -> SendMessage:
        return messages.pop(0)

    async def send(message: SendMessage) -> None:
        sent.append(message)

    await app(
        {
            "type": "http",
            "method": method,
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "headers": [(b"content-type", b"application/json")],
        },
        receive,
        send,
    )
    return sent


async def _invoke_through_a_reporting_channel(
    app: RouterApp, method: str, path: str
) -> list[SendMessage]:
    """Drive the app through a channel that reports its response complete.

    ``_invoke`` hands over a caller that never speaks again, so its channel never
    reports anything and a departure read after the hand-over looks exactly like
    one read before it. A real server arms the channel when the response is
    complete: ``uvicorn`` returns ``http.disconnect`` from ``receive()`` once
    ``disconnected or response_complete``, and sets the second inside the send
    that writes the body, so a caller that received the whole answer is then
    indistinguishable from one that had gone.
    """

    incoming: asyncio.Queue[SendMessage] = asyncio.Queue()
    await incoming.put({"type": "http.request", "body": b""})
    sent: list[SendMessage] = []

    async def receive() -> SendMessage:
        return await incoming.get()

    async def send(message: SendMessage) -> None:
        sent.append(message)
        if message["type"] == "http.response.body" and not message.get("more_body"):
            await incoming.put({"type": "http.disconnect"})
            await asyncio.sleep(0)

    await app(
        {
            "type": "http",
            "method": method,
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "headers": [(b"content-type", b"application/json")],
        },
        receive,
        send,
    )
    return sent


def test_the_departure_is_read_before_the_answer_changes_hands(tmp_path: Path) -> None:
    """A caller that stays records ``completed``, and that is what pins the read.

    The channel here reports the server's own completion, so a departure read
    after the hand-over records ``aborted`` for a caller that received
    everything. This assertion therefore holds the read on the near side of the
    send: move it after the body and this test goes red while the rest of the
    suite stays green, which is how the position was held by nothing before.
    """

    async def exercise() -> None:
        receipts = tmp_path / "requests.jsonl"
        engine = _sse_engine()
        async with (
            _server(engine) as engine_url,
            _router_with_receipts([Upstream(engine_url)], receipts) as app,
        ):
            sent = await _invoke_through_a_reporting_channel(app, "GET", "/v1/models")
            assert _status(sent) == 200
            assert _body(sent)

        rows = _read_rows(receipts)
        assert len(rows) == 1
        assert rows[0]["status"] == "completed"
        assert rows[0]["upstream"] == SELF_ANSWERED_UPSTREAM

    asyncio.run(exercise())


def test_a_caller_that_leaves_while_uploading_its_request_is_recorded(
    tmp_path: Path,
) -> None:
    """A request that dies on the way in leaves a row saying so.

    Nothing is answered, so nothing is received -- but the request did reach the
    router and stop there, and an outcome dropped from the record is an outcome
    a caller reports as an unexplained hang.
    """

    async def exercise() -> None:
        receipts = tmp_path / "requests.jsonl"
        engine = _sse_engine()
        async with (
            _server(engine) as engine_url,
            _router_with_receipts([Upstream(engine_url)], receipts) as app,
        ):
            head = json.dumps({"model": "streamer", "messages": []}).encode()
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

        rows = _read_rows(receipts)
        assert len(rows) == 1
        assert rows[0]["status"] == "aborted"
        assert rows[0]["upstream"] == "(router)"
        assert rows[0]["model"] == ""

    asyncio.run(exercise())


def test_a_self_answered_request_whose_caller_had_gone_is_recorded_as_aborted(
    tmp_path: Path,
) -> None:
    """The row says what the caller received, not what the router composed.

    The catalog listing is still built and handed to the server, because the
    router composes it whether or not anyone is listening. Recording that as a
    completed answer puts rows in the record for requests no caller received,
    which is the same confusion as recording a truncated relay as complete.
    """

    async def exercise() -> None:
        receipts = tmp_path / "requests.jsonl"
        engine = _sse_engine()
        async with (
            _server(engine) as engine_url,
            _router_with_receipts([Upstream(engine_url)], receipts) as app,
        ):
            sent = await _invoke_over(
                app, "GET", "/v1/models", [{"type": "http.disconnect"}]
            )
            assert _status(sent) == 200
            assert _body(sent)

        rows = _read_rows(receipts)
        assert len(rows) == 1
        assert rows[0]["status"] == "aborted"
        assert rows[0]["upstream"] == "(router)"
        assert rows[0]["model"] == ""

    asyncio.run(exercise())


# Captured verbatim from the deployed serve on 2026-09-23 by streaming a real
# request through the relay, rather than transcribed from a specification. The
# defect these cover was invisible to every earlier test in this file because
# each one was written in the vocabulary the reader already knew.
_NATIVE_STREAM = (
    '{"type":"message_start","message":{"id":"msg_afeaa41c","type":"message",'
    '"role":"assistant","content":[],"model":"deepseek-v4.1-flash",'
    '"usage":{"input_tokens":7,"output_tokens":0}}}',
    '{"type":"content_block_start","index":0,'
    '"content_block":{"type":"text","text":""}}',
    '{"type":"content_block_delta","index":0,'
    '"delta":{"type":"text_delta","text":"one"}}',
    '{"type":"content_block_delta","index":0,'
    '"delta":{"type":"text_delta","text":", two, three."}}',
    '{"type":"content_block_stop","index":0}',
    '{"type":"message_delta","delta":{"stop_reason":"end_turn"},'
    '"usage":{"output_tokens":7}}',
)


def _feed_events(lines: tuple[str, ...]) -> request_receipts.StreamAccounting:
    """Drive one accounting over server-sent events, as the relay would."""
    ticks = iter(range(1, 500))
    accounting = request_receipts.StreamAccounting(clock=lambda: float(next(ticks)))
    for line in lines:
        accounting.feed(f"data: {line}\n".encode())
    accounting.finish()
    return accounting


def test_counts_are_read_from_a_stream_in_the_engines_own_vocabulary() -> None:
    """The measured shape the deployed serve answers agent traffic with.

    Its opening counts sit inside the message it starts rather than beside it,
    and it spells them differently from the shape this reader was written for.
    """
    accounting = _feed_events(_NATIVE_STREAM)

    assert accounting.dialect == "anthropic"
    assert accounting.prompt_tokens == 7
    assert accounting.completion_tokens == 7
    assert accounting.time_to_first_token_s is not None


def test_a_closing_count_supersedes_the_count_the_stream_opened_with() -> None:
    """The opening event states nothing produced yet; it must not be the answer."""
    accounting = _feed_events(_NATIVE_STREAM)

    assert accounting.completion_tokens == 7, "opened at 0 and must close at 7"


def test_the_first_token_is_the_first_content_delta_not_the_stream_opening() -> None:
    """Time to first token measures content arriving, not the response starting."""
    opening_only = _feed_events(_NATIVE_STREAM[:2])
    with_content = _feed_events(_NATIVE_STREAM[:3])

    assert opening_only.time_to_first_token_s is None
    assert with_content.time_to_first_token_s is not None


def test_a_single_body_in_the_engines_own_vocabulary_is_read() -> None:
    """The non-streaming answer carries the same counts under the same names."""
    accounting = request_receipts.StreamAccounting()
    accounting.feed(
        json.dumps(
            {
                "id": "msg_c61f15c3",
                "type": "message",
                "model": "deepseek-v4.1-flash",
                "content": [{"type": "text", "text": "serving"}],
                "usage": {"input_tokens": 11, "output_tokens": 3},
            }
        ).encode()
    )
    accounting.finish()

    assert accounting.dialect == "anthropic"
    assert accounting.prompt_tokens == 11
    assert accounting.completion_tokens == 3


def test_the_other_vocabulary_still_reads_and_names_itself() -> None:
    """The shape this reader was originally written for must keep working."""
    accounting = _feed_events(
        (
            json.dumps({"choices": [{"delta": {"content": "he"}}]}),
            json.dumps(
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 1234,
                        "completion_tokens": 56,
                        "reasoning_tokens": 12,
                        "prompt_tokens_details": {"cached_tokens": 1200},
                    },
                }
            ),
        )
    )

    assert accounting.dialect == "openai"
    assert accounting.prompt_tokens == 1234
    assert accounting.completion_tokens == 56
    assert accounting.reasoning_tokens == 12
    assert accounting.cached_prompt_tokens == 1200


def test_an_unreadable_vocabulary_says_so_rather_than_reading_as_empty() -> None:
    """A row nobody could read must be distinguishable from one with no counts."""
    accounting = _feed_events(
        (json.dumps({"tokens_consumed": {"in": 40, "out": 9}}),),
    )

    assert accounting.dialect == request_receipts.DIALECT_UNKNOWN
    assert accounting.prompt_tokens is None
    assert accounting.completion_tokens is None


def test_an_engine_reporting_no_cache_figures_records_none_not_zero() -> None:
    """The deployed serve states no cache use in this vocabulary at all."""
    accounting = _feed_events(_NATIVE_STREAM)

    assert accounting.cached_prompt_tokens is None


def test_a_new_shape_is_added_as_a_dialect_entry_not_as_parsing_code() -> None:
    """The registry is the extension point: declare names, gain a vocabulary."""
    invented = request_receipts.ResponseDialect(
        name="invented",
        usage_containers=("", "envelope"),
        prompt_tokens=("tokens_in",),
        completion_tokens=("tokens_out",),
        reasoning_tokens=(),
        cached_tokens=("tokens_reused",),
        cached_containers=(),
        event_key="kind",
        delta_events=("chunk",),
        delta_list="",
        delta_key="piece",
        delta_fields=("body",),
    )
    record = {"envelope": {"usage": {"tokens_in": 3, "tokens_out": 4}}}

    usages = invented.usage_objects(record)

    assert [dict(u) for u in usages] == [{"tokens_in": 3, "tokens_out": 4}]
    assert invented.speaks(usages[0])
    assert invented.carries_first_token({"kind": "chunk", "piece": {"body": "hi"}})


def _identity_headers(
    run_id: bytes | None, session: bytes | None
) -> list[tuple[bytes, bytes]]:
    headers: list[tuple[bytes, bytes]] = [(b"content-type", b"application/json")]
    if run_id is not None:
        headers.append((b"x-reckon-run-id", run_id))
    if session is not None:
        headers.append((b"x-reckon-session", session))
    return headers


def test_a_relayed_request_records_the_session_identity_it_carried(
    tmp_path: Path,
) -> None:
    """The two headers a keyed caller sends land on the row it produced."""

    async def exercise() -> None:
        receipts = tmp_path / "requests.jsonl"
        engine = _sse_engine()
        async with (
            _server(engine) as engine_url,
            _router_with_receipts([Upstream(engine_url)], receipts) as app,
        ):
            response = await _invoke(
                app,
                "POST",
                "/v1/chat/completions",
                _request_body(),
                headers=_identity_headers(b"r-abc", b"s-1"),
            )
            assert _status(response) == 200

        rows = _read_rows(receipts)
        assert len(rows) == 1
        assert rows[0]["run_id"] == "r-abc"
        assert rows[0]["coordinator_session"] == "s-1"

    asyncio.run(exercise())


def test_a_request_reported_while_unkeyed_records_both_identities_null(
    tmp_path: Path,
) -> None:
    """No header means null, not a value the router inferred from the socket."""

    async def exercise() -> None:
        receipts = tmp_path / "requests.jsonl"
        engine = _sse_engine()
        async with (
            _server(engine) as engine_url,
            _router_with_receipts([Upstream(engine_url)], receipts) as app,
        ):
            response = await _invoke(
                app, "POST", "/v1/chat/completions", _request_body()
            )
            assert _status(response) == 200

        rows = _read_rows(receipts)
        assert len(rows) == 1
        assert rows[0]["run_id"] is None
        assert rows[0]["coordinator_session"] is None

    asyncio.run(exercise())


def test_an_over_long_or_control_bearing_identity_is_recorded_as_absent(
    tmp_path: Path,
) -> None:
    """Two malformed values on one request do not contaminate each other."""

    async def exercise() -> None:
        receipts = tmp_path / "requests.jsonl"
        engine = _sse_engine()
        async with (
            _server(engine) as engine_url,
            _router_with_receipts([Upstream(engine_url)], receipts) as app,
        ):
            await _invoke(
                app,
                "POST",
                "/v1/chat/completions",
                _request_body(),
                headers=_identity_headers(b"a" * (MAX_IDENTITY_CHARS + 1), b"s-ok"),
            )
            await _invoke(
                app,
                "POST",
                "/v1/chat/completions",
                _request_body(),
                headers=_identity_headers(b"r-ok", b"s\x07bad"),
            )

        rows = _read_rows(receipts)
        assert len(rows) == 2
        assert rows[0]["run_id"] is None
        assert rows[0]["coordinator_session"] == "s-ok"
        assert rows[1]["run_id"] == "r-ok"
        assert rows[1]["coordinator_session"] is None

    asyncio.run(exercise())


def test_an_identity_that_is_not_a_well_formed_token_is_rejected_at_the_boundary() -> (
    None
):
    """The token rule is stated once and read without a request in flight."""
    at_limit = b"a" * MAX_IDENTITY_CHARS

    assert identity_from_headers([(b"x-reckon-run-id", at_limit)]) == (
        at_limit.decode(),
        None,
    )
    assert identity_from_headers(
        [(b"x-reckon-run-id", b"a" * (MAX_IDENTITY_CHARS + 1))]
    ) == (
        None,
        None,
    )
    assert identity_from_headers([(b"X-Reckon-Session", b"s-1")]) == (None, "s-1")
    assert identity_from_headers([(b"x-reckon-run-id", b"")]) == (None, None)
    assert identity_from_headers([(b"x-reckon-run-id", b"r\x07x")]) == (None, None)
    assert identity_from_headers([]) == (None, None)


def test_the_gate_timeout_receipt_also_carries_the_session_identity(
    tmp_path: Path,
) -> None:
    """A 529 composed by the gate is keyed from the request that waited."""

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
            held_queue: asyncio.Queue[SendMessage] = asyncio.Queue()

            async def _receive() -> SendMessage:
                return await held_queue.get()

            held = await app._generation_gate.acquire(_receive)
            response = await _invoke(
                app,
                "POST",
                "/v1/chat/completions",
                _request_body(),
                headers=_identity_headers(b"r-timeout", b"s-timeout"),
            )
            assert _status(response) == 529
            await app._generation_gate.release()
            if held.disconnect_task is not None:
                held.disconnect_task.cancel()
                await asyncio.gather(held.disconnect_task, return_exceptions=True)

        rows = _read_rows(receipts)
        assert len(rows) == 1
        assert rows[0]["status"] == "failed"
        assert rows[0]["run_id"] == "r-timeout"
        assert rows[0]["coordinator_session"] == "s-timeout"

    asyncio.run(exercise())
