"""Mid-conversation system messages must not defeat the engine's prefix cache."""

from __future__ import annotations

import json

from imas_ambix.agent.router import RouterApp


def _relabel(payload: dict) -> tuple[dict, bytes]:
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    return RouterApp._repair_system_roles(payload, body)


def test_mid_conversation_system_messages_are_relabelled() -> None:
    """The encoder re-emits the whole tool block after each system message.

    Agent harnesses inject one such message per turn, so the rendered prompt
    gains tens of thousands of tokens at a fresh position every request and no
    two turns share a prefix. Re-labelling them matches the encoder's own
    reading, which already treats them as user turns when placing the assistant
    header.
    """
    payload = {
        "model": "deepseek-v4.1-flash",
        "messages": [
            {"role": "user", "content": "do the work"},
            {"role": "assistant", "content": "starting"},
            {"role": "system", "content": "<total_tokens>900 left</total_tokens>"},
            {"role": "user", "content": "continue"},
            {"role": "system", "content": "session hook context"},
        ],
    }

    amended, body = _relabel(payload)

    roles = [message["role"] for message in amended["messages"]]
    assert roles == ["user", "assistant", "user", "user", "user"]
    assert json.loads(body)["messages"] == amended["messages"]
    # Content is carried through untouched; only the label changes.
    assert amended["messages"][2]["content"] == "<total_tokens>900 left</total_tokens>"


def test_a_leading_system_message_is_left_alone() -> None:
    """Opening a conversation with a system message is conventional.

    The encoder handles that one without re-emitting the tool block, so
    re-labelling it would change the prompt for no benefit.
    """
    payload = {
        "messages": [
            {"role": "system", "content": "you are an assistant"},
            {"role": "user", "content": "hello"},
        ]
    }

    amended, body = _relabel(payload)

    assert amended["messages"][0]["role"] == "system"
    assert amended is payload


def test_an_untouched_request_passes_through_byte_for_byte() -> None:
    """Re-encoding a request that needed no change would alter it needlessly."""
    payload = {"messages": [{"role": "user", "content": "hello"}]}
    original = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()

    amended, body = RouterApp._repair_system_roles(payload, original)

    assert body is original
    assert amended is payload


def test_a_payload_without_messages_is_returned_unchanged() -> None:
    """Catalog and token-count requests reach the same relay path."""
    payload = {"model": "deepseek-v4.1-flash"}
    original = b'{"model":"deepseek-v4.1-flash"}'

    amended, body = RouterApp._repair_system_roles(payload, original)

    assert body is original
    assert amended is payload
