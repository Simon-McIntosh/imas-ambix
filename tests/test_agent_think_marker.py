"""Response-shaping tests for orphaned closing think markers."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable
from typing import Any

from imas_ambix.agent.profile import SiteConfig, load_profile
from imas_ambix.agent.slurm import generate_serve_script
from imas_ambix.agent.vllm_think_marker import OrphanThinkMarkerMiddleware

_CLOSING_MARKER = "</think>"


def _run_app(
    messages: Iterable[dict[str, Any]],
    *,
    path: str = "/v1/chat/completions",
) -> list[dict[str, Any]]:
    source = [dict(message) for message in messages]

    async def app(_scope: Any, _receive: Any, send: Any) -> None:
        for message in source:
            await send(dict(message))

    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b""}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    asyncio.run(
        OrphanThinkMarkerMiddleware(app)(
            {"type": "http", "method": "POST", "path": path},
            receive,
            send,
        )
    )
    return sent


def _buffered_messages(payload: object) -> list[dict[str, Any]]:
    body = json.dumps(payload, separators=(",", ":")).encode()
    return [
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        },
        {"type": "http.response.body", "body": body},
    ]


def test_buffered_orphan_marker_is_removed_from_assistant_content() -> None:
    payload = {
        "id": "response-id",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": f"{_CLOSING_MARKER}Calling the tool now.",
                    "tool_calls": [{"id": "call-id", "type": "function"}],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"completion_tokens": 12},
    }

    sent = _run_app(_buffered_messages(payload))
    rewritten = json.loads(sent[1]["body"])

    assert rewritten["choices"][0]["message"]["content"] == "Calling the tool now."
    expected = dict(payload)
    expected["choices"] = [dict(payload["choices"][0])]
    expected["choices"][0]["message"] = dict(payload["choices"][0]["message"])
    expected["choices"][0]["message"]["content"] = "Calling the tool now."
    assert rewritten == expected
    assert (b"content-length", str(len(sent[1]["body"])).encode()) in sent[0]["headers"]


def test_streamed_orphan_marker_changes_only_its_text_event() -> None:
    events = [
        (
            b"event: message_start\ndata: "
            b'{"type":"message_start","message":{"id":"m"}}\n\n'
        ),
        (
            b"event: content_block_delta\r\ndata: "
            b'{"type":"content_block_delta","index":0,"delta":{"type":"text_delta",'
            b'"text":"</think>Calling the tool now."}}\r\n\r\n'
        ),
        (
            b"event: content_block_start\ndata: "
            b'{"type":"content_block_start","index":1,'
            b'"content_block":{"type":"tool_use","id":"c"}}\n\n'
        ),
        b"data: [DONE]\n\n",
    ]
    messages = [
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"text/event-stream")],
        },
        *[
            {
                "type": "http.response.body",
                "body": event,
                "more_body": index < len(events) - 1,
            }
            for index, event in enumerate(events)
        ],
    ]

    sent = _run_app(messages, path="/v1/messages")
    bodies = [message["body"] for message in sent[1:]]

    assert bodies[0] == events[0]
    assert (
        json.loads(bodies[1].split(b"data: ", 1)[1])["delta"]["text"]
        == "Calling the tool now."
    )
    assert bodies[2:] == events[2:]


def test_genuine_thinking_block_keeps_both_markers_byte_for_byte() -> None:
    messages = _buffered_messages(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "<think>Reasoning.</think>Final answer.",
                    }
                }
            ]
        }
    )

    assert _run_app(messages) == messages


def test_response_without_marker_passes_through_byte_for_byte() -> None:
    messages = _buffered_messages(
        {"choices": [{"message": {"role": "assistant", "content": "Answer."}}]}
    )

    assert _run_app(messages) == messages


def test_marker_inside_assistant_prose_is_not_removed() -> None:
    messages = _buffered_messages(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "The literal marker is </think> in this example.",
                    }
                }
            ]
        }
    )

    assert _run_app(messages) == messages


def test_generated_serve_registers_marker_middleware_alongside_catalog() -> None:
    script = generate_serve_script(load_profile("glm-5-3").for_gpus(4), SiteConfig())

    assert "imas_ambix.agent.vllm_catalog.GlobalModelCatalogMiddleware" in script
    assert "imas_ambix.agent.vllm_think_marker.OrphanThinkMarkerMiddleware" in script
    assert script.count("--middleware") == 2
