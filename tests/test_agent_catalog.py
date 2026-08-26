"""Focused contract tests for the server-owned vLLM model catalog."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable
from typing import Any

import pytest

from imas_ambix.agent.profile import (
    EngineConfig,
    ModelConfig,
    ModelProfile,
    SiteConfig,
    SlurmDefaults,
)
from imas_ambix.agent.slurm import generate_serve_script
from imas_ambix.agent.vllm_catalog import (
    CATALOG_METADATA_ENV,
    GlobalModelCatalogMiddleware,
    validate_catalog_metadata,
)


def _metadata(count: int = 8, precision: str = "fp8") -> dict[str, Any]:
    return {
        "future-release": {
            "accelerator_family": "H200",
            "accelerator_count": count,
            "checkpoint_precision": precision,
        }
    }


def _profile(*, gpus: int = 8, precision: str | None = "fp8") -> ModelProfile:
    return ModelProfile(
        slug="catalog-test",
        model=ModelConfig(
            name="Catalog Test",
            hf_repo="example/catalog-test",
            served_name="future-release",
            size_gb=100,
            max_context=131072,
            checkpoint_precision=precision,
        ),
        engine=EngineConfig(type="vllm", tensor_parallel=gpus),
        slurm=SlurmDefaults(gpus=gpus),
    )


def _run_app(
    monkeypatch: pytest.MonkeyPatch,
    messages: Iterable[dict[str, Any]],
    *,
    method: str = "GET",
    path: str = "/v1/models",
    metadata: object | None = None,
) -> list[dict[str, Any]]:
    monkeypatch.setenv(
        CATALOG_METADATA_ENV,
        json.dumps(_metadata() if metadata is None else metadata),
    )
    source = [dict(message) for message in messages]

    async def app(_scope: Any, _receive: Any, send: Any) -> None:
        for message in source:
            await send(dict(message))

    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b""}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    middleware = GlobalModelCatalogMiddleware(app)
    asyncio.run(
        middleware(
            {"type": "http", "method": method, "path": path},
            receive,
            send,
        )
    )
    return sent


def _catalog_messages(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    body = json.dumps({"object": "list", "data": cards}).encode()
    return [
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", b"1"),
            ],
        },
        {"type": "http.response.body", "body": body},
    ]


def test_native_card_is_enriched_without_changing_identity_or_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages = _catalog_messages(
        [{"id": "future-release", "max_model_len": 131072, "owned_by": "vllm"}]
    )
    sent = _run_app(monkeypatch, messages)
    payload = json.loads(sent[1]["body"])
    assert payload["data"] == [
        {
            "id": "future-release",
            "max_model_len": 131072,
            "owned_by": "vllm",
            "ambix": _metadata()["future-release"],
        }
    ]
    assert (b"content-length", str(len(sent[1]["body"])).encode()) in sent[0]["headers"]


def test_absent_native_model_is_never_synthesized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent = _run_app(monkeypatch, _catalog_messages([]))
    assert json.loads(sent[1]["body"])["data"] == []


def test_unmapped_native_model_is_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    sent = _run_app(monkeypatch, _catalog_messages([{"id": "unmapped"}]))
    assert json.loads(sent[1]["body"])["data"] == []


def test_duplicate_release_ids_are_all_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cards = [
        {"id": "future-release", "root": "a"},
        {"id": "future-release", "root": "b"},
    ]
    sent = _run_app(monkeypatch, _catalog_messages(cards))
    assert json.loads(sent[1]["body"])["data"] == []


def test_multiframe_catalog_body_is_rewritten(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    complete = _catalog_messages([{"id": "future-release"}])
    body = complete[1]["body"]
    complete[1:] = [
        {"type": "http.response.body", "body": body[:13], "more_body": True},
        {"type": "http.response.body", "body": body[13:]},
    ]
    sent = _run_app(monkeypatch, complete)
    assert len(sent) == 2
    assert json.loads(sent[1]["body"])["data"][0]["id"] == "future-release"


@pytest.mark.parametrize(
    ("method", "path"),
    [("POST", "/v1/models"), ("GET", "/v1/chat/completions")],
)
def test_non_catalog_responses_pass_through_message_for_message(
    monkeypatch: pytest.MonkeyPatch, method: str, path: str
) -> None:
    messages = [
        {"type": "http.response.start", "status": 200, "headers": [(b"x", b"y")]},
        {"type": "http.response.body", "body": b"first", "more_body": True},
        {"type": "http.response.body", "body": b"second"},
    ]
    assert _run_app(monkeypatch, messages, method=method, path=path) == messages


def test_catalog_error_response_passes_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages = [
        {"type": "http.response.start", "status": 503, "headers": [(b"x", b"y")]},
        {"type": "http.response.body", "body": b"unavailable"},
    ]
    assert _run_app(monkeypatch, messages) == messages


@pytest.mark.parametrize("count", [2, 4, 6, 8])
def test_generated_script_uses_actual_gpu_count(count: int) -> None:
    script = generate_serve_script(_profile(gpus=count), SiteConfig())
    assert f"#SBATCH --gres=gpu:{count}" in script
    assert f'"accelerator_count":{count}' in script
    assert "--middleware" in script
    assert "imas_ambix.agent.vllm_catalog.GlobalModelCatalogMiddleware" in script


def test_generated_script_uses_explicit_checkpoint_precision() -> None:
    script = generate_serve_script(_profile(precision="int4-awq"), SiteConfig())
    assert '"checkpoint_precision":"int4-awq"' in script
    assert "AMBIX_VLLM_CATALOG_METADATA" in script
    assert "PYTHONPATH=" in script


def test_catalog_serve_fails_closed_without_checkpoint_precision() -> None:
    with pytest.raises(ValueError, match="checkpoint_precision"):
        generate_serve_script(_profile(precision=None), SiteConfig())


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        _metadata(count=3),
        _metadata(count=True),
        _metadata(count=2.0),
        _metadata(precision=""),
        {
            "future-release": {
                "accelerator_family": "H200\n",
                "accelerator_count": 8,
                "checkpoint_precision": "fp8",
            }
        },
    ],
)
def test_invalid_launch_metadata_is_rejected(metadata: object) -> None:
    with pytest.raises(ValueError):
        validate_catalog_metadata(metadata)


def test_middleware_fails_closed_without_launch_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(CATALOG_METADATA_ENV, raising=False)

    async def app(_scope: Any, _receive: Any, _send: Any) -> None:
        return None

    with pytest.raises(RuntimeError, match=CATALOG_METADATA_ENV):
        GlobalModelCatalogMiddleware(app)
