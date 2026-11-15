"""The work lined up for the local lane, read from the dispatcher's own record.

The router's gate counts model requests: a request is in a slot while it
generates and waits at the gate for seconds when every slot is taken. That
says nothing about how much work is lined up, because a worker that has not
been launched sends no request. The dispatcher keeps one small JSON pointer
per request it has accepted, from the moment it is queued until its run is
recorded, and this module counts those pointers:

``workers``
    Runs launched on the local lane and still going. A worker holds a gate
    slot only while it generates; between turns it runs tools and holds none,
    so there are normally more workers than requests in use.
``ready``
    Requests with nothing left to wait for but a free worker slot.
``blocked``
    Requests whose plan section waits on another section that has not landed.
``paused``
    Requests set aside: held by a decision, or backing off after a failed
    launch.
``oldest_ready_seconds``
    How long the longest-waiting ready request has been queued.

A pointer is parsed again only when its file has changed, so a refresh costs
one ``stat`` per pointer and a parse of the few that moved.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from pathlib import Path
from typing import Any

__all__ = ["default_directory", "read"]

RUNNING = {"working", "starting"}

# path -> ((mtime_ns, size), the fields of the pointer this module reads)
_SEEN: dict[str, tuple[tuple[int, int], dict[str, Any] | None]] = {}


def default_directory() -> Path:
    """Where the dispatcher keeps its pointers for requests still in flight."""
    home = os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    return Path(home) / "reckon" / "crew" / "live"


def _pointer(path: Path) -> dict[str, Any] | None:
    """The fields read from one pointer, or None when it cannot be read."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    local = payload.get("local")
    if local is None:
        # A run that is still starting has no lane flag yet, only a backend.
        local = str(payload.get("backend") or "").endswith("local")
    return {
        "phase": payload.get("phase"),
        "local": bool(local),
        "queued_at": payload.get("queued_at"),
        "set_aside": bool(payload.get("queue_hold")),
    }


def _age(stamp: object, now: _dt.datetime) -> float | None:
    if not isinstance(stamp, str):
        return None
    try:
        then = _dt.datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return None
    if then.tzinfo is None:
        then = then.replace(tzinfo=_dt.UTC)
    return max(0.0, (now - then).total_seconds())


def read(
    directory: str | Path | None = None, now: _dt.datetime | None = None
) -> dict[str, Any] | None:
    """The local lane's work, counted; None when there is no record to read."""
    root = Path(directory) if directory is not None else default_directory()
    try:
        entries = [entry for entry in os.scandir(root) if entry.name.endswith(".json")]
    except OSError:
        return None
    now = now or _dt.datetime.now(_dt.UTC)
    counts = {"workers": 0, "ready": 0, "blocked": 0, "paused": 0}
    oldest: float | None = None
    present = set()
    for entry in entries:
        try:
            stat = entry.stat()
        except OSError:
            continue
        present.add(entry.path)
        key = (stat.st_mtime_ns, stat.st_size)
        seen = _SEEN.get(entry.path)
        if seen is None or seen[0] != key:
            seen = (key, _pointer(Path(entry.path)))
            _SEEN[entry.path] = seen
        pointer = seen[1]
        if not pointer or not pointer["local"]:
            continue
        phase = pointer["phase"]
        if phase in RUNNING:
            counts["workers"] += 1
        elif phase == "held" or (phase == "queued" and pointer["set_aside"]):
            counts["paused"] += 1
        elif phase == "awaiting-upstream":
            counts["blocked"] += 1
        elif phase == "queued":
            counts["ready"] += 1
            age = _age(pointer["queued_at"], now)
            if age is not None and (oldest is None or age > oldest):
                oldest = age
    for path in [
        path for path in _SEEN if path.startswith(str(root)) and path not in present
    ]:
        del _SEEN[path]
    return {**counts, "oldest_ready_seconds": oldest}
