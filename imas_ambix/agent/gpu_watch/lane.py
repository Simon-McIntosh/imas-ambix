"""The served lanes, one entry per engine, from the live serving document.

The document is :func:`imas_ambix.agent.llm_watch.build_document`'s: the
router's lane document, the recorder's receipt tail, the endpoint document and
the published ledger. Nothing here reads a source of its own, so a lane on
these pages and the same lane in ``llm-watch`` are one set of figures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from imas_ambix.agent import llm_watch
from imas_ambix.agent.gpu_watch.fleet import short_name


@dataclass
class Lane:
    """One engine: what it serves, where, and its two blocks of figures."""

    model: str
    server: str | None
    job_id: str | None
    hardware: str | None
    context: object
    checkpoint: str | None
    lane: dict[str, Any] = field(default_factory=dict)
    live: dict[str, Any] = field(default_factory=dict)


def _lane(document: dict[str, Any], host: object) -> Lane:
    lane = document.get("lane") or {}
    live = document.get("live") or {}
    endpoint = llm_watch.engine_endpoint(
        document.get("engine") or {}, lane.get("model_id")
    )
    count, family = (
        endpoint.get("accelerator_count"),
        endpoint.get("accelerator_family"),
    )
    where = host or live.get("hostname") or endpoint.get("host")
    job = live.get("job_id")
    return Lane(
        model=str(live.get("served_name") or lane.get("model_id") or "?"),
        server=short_name(where) if isinstance(where, str) and where else None,
        job_id=str(job) if job else None,
        hardware=f"{count}×{family}" if count and family else family,
        context=llm_watch.context_ceiling(lane, endpoint),
        checkpoint=llm_watch.checkpoint_label(document),
        lane=lane,
        live=live,
    )


def lanes(document: dict[str, Any] | None) -> list[Lane]:
    """Every engine the document describes; one when it describes one."""
    if not document:
        return []
    if llm_watch.multi_engine(document):
        return [
            _lane(
                llm_watch.engine_document(document, entry),
                llm_watch._server_name(entry),
            )
            for entry in llm_watch.document_engines(document)
        ]
    if not (document.get("lane") or document.get("live")):
        return []
    return [_lane(document, None)]
