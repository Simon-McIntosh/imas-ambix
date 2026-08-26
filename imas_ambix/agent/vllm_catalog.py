"""Enrich vLLM's native model catalog with launch-owned site metadata."""

from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

CatalogMetadata = dict[str, dict[str, str | int]]
AsgiMessage = dict[str, Any]
AsgiCallable = Callable[
    [
        dict[str, Any],
        Callable[[], Awaitable[AsgiMessage]],
        Callable[[AsgiMessage], Awaitable[None]],
    ],
    Awaitable[None],
]

CATALOG_METADATA_ENV = "AMBIX_VLLM_CATALOG_METADATA"
_VALID_ACCELERATOR_COUNTS = frozenset({2, 4, 6, 8})


def _valid_label(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value.strip())
        and not any(ord(character) < 32 or ord(character) == 127 for character in value)
    )


def validate_catalog_metadata(value: object) -> CatalogMetadata:
    """Return validated release metadata or raise for unsafe launch state."""
    if not isinstance(value, Mapping) or not value:
        raise ValueError("catalog metadata must be a non-empty release mapping")

    validated: CatalogMetadata = {}
    for release_id, metadata in value.items():
        if not _valid_label(release_id) or not isinstance(metadata, Mapping):
            raise ValueError("catalog release ids and metadata must be explicit")
        accelerator_family = metadata.get("accelerator_family")
        accelerator_count = metadata.get("accelerator_count")
        checkpoint_precision = metadata.get("checkpoint_precision")
        if not _valid_label(accelerator_family):
            raise ValueError("accelerator_family must be a non-empty label")
        if (
            type(accelerator_count) is not int
            or accelerator_count not in _VALID_ACCELERATOR_COUNTS
        ):
            raise ValueError("accelerator_count must be one of 2, 4, 6, or 8")
        if not _valid_label(checkpoint_precision):
            raise ValueError("checkpoint_precision must be a non-empty label")
        validated[release_id] = {
            "accelerator_family": accelerator_family,
            "accelerator_count": accelerator_count,
            "checkpoint_precision": checkpoint_precision,
        }
    return validated


def catalog_metadata_from_environment() -> CatalogMetadata:
    """Load the required launch-owned metadata map from the environment."""
    raw = os.environ.get(CATALOG_METADATA_ENV)
    if raw is None:
        raise RuntimeError(f"{CATALOG_METADATA_ENV} is required")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"{CATALOG_METADATA_ENV} is not valid JSON") from error
    try:
        return validate_catalog_metadata(value)
    except ValueError as error:
        raise RuntimeError(f"{CATALOG_METADATA_ENV}: {error}") from error


def _rewrite_catalog(body: bytes, metadata: CatalogMetadata) -> bytes:
    try:
        payload = json.loads(body)
    except UnicodeDecodeError, json.JSONDecodeError:
        payload = {"object": "list", "data": []}

    if not isinstance(payload, dict):
        payload = {"object": "list", "data": []}
    native_items = payload.get("data")
    if not isinstance(native_items, list):
        native_items = []

    release_counts = Counter(
        item.get("id")
        for item in native_items
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    )
    retained: list[dict[str, Any]] = []
    for item in native_items:
        if not isinstance(item, dict):
            continue
        release_id = item.get("id")
        if release_counts[release_id] != 1 or release_id not in metadata:
            continue
        enriched = dict(item)
        enriched["ambix"] = dict(metadata[release_id])
        retained.append(enriched)
    payload["data"] = retained
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode()


class GlobalModelCatalogMiddleware:
    """Rewrite only successful native ``GET /v1/models`` responses."""

    def __init__(self, app: AsgiCallable) -> None:
        self.app = app
        self.metadata = catalog_metadata_from_environment()

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Callable[[], Awaitable[AsgiMessage]],
        send: Callable[[AsgiMessage], Awaitable[None]],
    ) -> None:
        if (
            scope.get("type") != "http"
            or scope.get("method") != "GET"
            or scope.get("path") != "/v1/models"
        ):
            await self.app(scope, receive, send)
            return

        start: AsgiMessage | None = None
        body_parts: list[bytes] = []

        async def capture(message: AsgiMessage) -> None:
            nonlocal start
            if message["type"] == "http.response.start":
                start = message
                if not 200 <= message["status"] < 300:
                    await send(message)
                return
            if start is None or not 200 <= start["status"] < 300:
                await send(message)
                return
            body_parts.append(message.get("body", b""))
            if message.get("more_body", False):
                return

            rewritten = _rewrite_catalog(b"".join(body_parts), self.metadata)
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
