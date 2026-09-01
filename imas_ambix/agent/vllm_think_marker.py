"""Remove orphaned closing think markers from assistant response content."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

AsgiMessage = dict[str, Any]
AsgiCallable = Callable[
    [
        dict[str, Any],
        Callable[[], Awaitable[AsgiMessage]],
        Callable[[AsgiMessage], Awaitable[None]],
    ],
    Awaitable[None],
]

_CLOSING_MARKER = "</think>"
_CHAT_PATHS = frozenset({"/v1/chat/completions", "/v1/messages"})


def _strip_prefix(value: object) -> tuple[object, bool]:
    if isinstance(value, str) and value.startswith(_CLOSING_MARKER):
        return value[len(_CLOSING_MARKER) :], True
    return value, False


def _rewrite_buffered_payload(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False

    choices = payload.get("choices")
    if isinstance(choices, list):
        changed = False
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            content, stripped = _strip_prefix(message.get("content"))
            if stripped:
                message["content"] = content
                changed = True
        return changed

    if payload.get("role") != "assistant":
        return False
    content = payload.get("content")
    if not isinstance(content, list) or any(
        isinstance(block, dict) and block.get("type") == "thinking" for block in content
    ):
        return False
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "text":
            continue
        text, stripped = _strip_prefix(block.get("text"))
        if stripped:
            block["text"] = text
        return stripped
    return False


class _StreamState:
    def __init__(self) -> None:
        self.content_started = False
        self.thinking_opened = False


def _rewrite_stream_payload(payload: object, state: _StreamState) -> bool:
    if not isinstance(payload, dict):
        return False

    content_block = payload.get("content_block")
    if isinstance(content_block, dict) and content_block.get("type") == "thinking":
        state.thinking_opened = True

    delta = payload.get("delta")
    if isinstance(delta, dict):
        if delta.get("type") == "thinking_delta" or delta.get("thinking"):
            state.thinking_opened = True
        text = delta.get("text")
        if isinstance(text, str) and text:
            changed = False
            if not state.content_started and not state.thinking_opened:
                rewritten, changed = _strip_prefix(text)
                if changed:
                    delta["text"] = rewritten
            state.content_started = True
            return changed

    choices = payload.get("choices")
    if not isinstance(choices, list):
        return False
    changed = False
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        choice_delta = choice.get("delta")
        if not isinstance(choice_delta, dict):
            continue
        if choice_delta.get("reasoning_content"):
            state.thinking_opened = True
        content = choice_delta.get("content")
        if not isinstance(content, str) or not content:
            continue
        if not state.content_started and not state.thinking_opened:
            rewritten, stripped = _strip_prefix(content)
            if stripped:
                choice_delta["content"] = rewritten
                changed = True
        state.content_started = True
    return changed


def _rewrite_sse_event(event: bytes, state: _StreamState) -> bytes:
    lines = event.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if not line.startswith(b"data: "):
            continue
        raw = line[len(b"data: ") :].rstrip(b"\r\n")
        if raw == b"[DONE]":
            return event
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return event
        if not _rewrite_stream_payload(payload, state):
            return event
        ending = line[len(line.rstrip(b"\r\n")) :]
        lines[index] = (
            b"data: "
            + json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode()
            + ending
        )
        return b"".join(lines)
    return event


def _content_type(headers: list[tuple[bytes, bytes]]) -> bytes:
    for name, value in headers:
        if name.lower() == b"content-type":
            return value.lower()
    return b""


def _pop_sse_event(buffer: bytes) -> tuple[bytes, bytes] | None:
    boundaries = [
        (index, separator)
        for separator in (b"\n\n", b"\r\n\r\n")
        if (index := buffer.find(separator)) >= 0
    ]
    if not boundaries:
        return None
    index, separator = min(boundaries, key=lambda boundary: boundary[0])
    end = index + len(separator)
    return buffer[:end], buffer[end:]


class OrphanThinkMarkerMiddleware:
    """Strip only a leading orphan marker from successful chat responses."""

    def __init__(self, app: AsgiCallable) -> None:
        self.app = app

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Callable[[], Awaitable[AsgiMessage]],
        send: Callable[[AsgiMessage], Awaitable[None]],
    ) -> None:
        if (
            scope.get("type") != "http"
            or scope.get("method") != "POST"
            or scope.get("path") not in _CHAT_PATHS
        ):
            await self.app(scope, receive, send)
            return

        start: AsgiMessage | None = None
        buffered_messages: list[AsgiMessage] = []
        stream_buffer = b""
        stream_state = _StreamState()
        streaming = False

        async def capture(message: AsgiMessage) -> None:
            nonlocal start, stream_buffer, streaming
            if message["type"] == "http.response.start":
                start = dict(message)
                headers = start.get("headers", [])
                streaming = b"text/event-stream" in _content_type(headers)
                if start.get("status", 500) // 100 != 2 or streaming:
                    await send(start)
                return
            if start is None or start.get("status", 500) // 100 != 2:
                await send(message)
                return
            if streaming:
                stream_buffer += message.get("body", b"")
                more_body = message.get("more_body", False)
                events: list[bytes] = []
                while (split := _pop_sse_event(stream_buffer)) is not None:
                    event, stream_buffer = split
                    events.append(event)
                if not more_body and stream_buffer:
                    events.append(stream_buffer)
                    stream_buffer = b""
                for index, event in enumerate(events):
                    await send(
                        {
                            "type": "http.response.body",
                            "body": _rewrite_sse_event(event, stream_state),
                            "more_body": more_body or index < len(events) - 1,
                        }
                    )
                if not more_body and not events:
                    await send({"type": "http.response.body", "body": b""})
                return

            buffered_messages.append(dict(message))
            if message.get("more_body", False):
                return
            raw = b"".join(item.get("body", b"") for item in buffered_messages)
            try:
                payload = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError):
                await send(start)
                for item in buffered_messages:
                    await send(item)
                return
            if not _rewrite_buffered_payload(payload):
                await send(start)
                for item in buffered_messages:
                    await send(item)
                return
            rewritten = json.dumps(
                payload, separators=(",", ":"), ensure_ascii=False
            ).encode()
            headers = [
                (name, value)
                for name, value in start.get("headers", [])
                if name.lower() != b"content-length"
            ]
            headers.append((b"content-length", str(len(rewritten)).encode()))
            rewritten_start = dict(start)
            rewritten_start["headers"] = headers
            await send(rewritten_start)
            await send({"type": "http.response.body", "body": rewritten})

        await self.app(scope, receive, capture)
