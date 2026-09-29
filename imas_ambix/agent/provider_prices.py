"""The provider price table: one owner for fetching, writing and ageing it.

Every cost figure ``agent watch`` renders is priced from a table of
OpenRouter's published rates. That table is fetched over the network, so a
reader that fetched its own would turn a render into a probe of the
provider's availability; and a table written by more than one owner would
leave a reader and a writer disagreeing about which file is in force.

So one module owns the table. It resolves the path through
:class:`~imas_ambix.agent.profile.SiteConfig`, fetches the models endpoint
under a bounded timeout, and writes the result atomically -- a scratch file
renamed onto the path, so a reader never observes a half-written table. It
refreshes only when the table it finds is older than :data:`REFRESH_AGE`.

A fetch that fails or times out leaves the cached table untouched and in
use. The age reported then is the age of the table that was actually read,
never a fresh stamp for bytes that were not fetched: a cost priced from a
stale table is a known approximation, while a stale table wearing a current
timestamp is a wrong answer that looks current.

The table records the instant it was fetched in its own ``at`` field, which
is the time a cost is priced against. The file's modification time is read
only when that field is absent, because a copy, a restore or a touch moves
the mtime without moving the fetch.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from imas_ambix.agent.profile import SiteConfig

#: The provider's model list, carrying each model's published rates.
PRICE_URL = "https://openrouter.ai/api/v1/models"

#: How long a fetch may block. The bound is the whole of the wait a render
#: will accept for a price table: past it the render proceeds on whatever
#: is already cached rather than stall the panel.
FETCH_TIMEOUT = 10.0

#: A table younger than this is not refetched. The provider's published
#: rates change on the order of days, so a daily refresh keeps the figure
#: current without turning each render into a network call.
REFRESH_AGE = 24 * 3600.0

#: The retired workstation cache the table moved from. Consulted once, to
#: seed the resolved path where it is absent, so no price history is lost.
LEGACY_PRICE_PATH = Path.home() / ".cache" / "gpu-watch" / "openrouter-prices.json"


@dataclass(frozen=True)
class Refresh:
    """What one refresh attempt did, and the age of the table now in force."""

    path: Path
    age: float | None
    refreshed: bool
    error: str | None


def table_path(config: SiteConfig | None = None) -> Path:
    """The table in force, resolved through :class:`SiteConfig`."""
    from imas_ambix.agent.profile import SiteConfig as _SiteConfig

    site = config if config is not None else _SiteConfig.from_env()
    return site.price_table


def fetch_models(url: str = PRICE_URL, timeout: float = FETCH_TIMEOUT) -> list[dict]:
    """The provider's model rows, slimmed to the id and pricing this needs.

    Raises on any failure -- an unreachable host, an HTTP error, an expired
    socket, malformed JSON -- so the caller can keep the cached table rather
    than write a partial one. No selected-but-unpriced row is invented: a
    row without an id is dropped, and its model is priced as absent.
    """
    import urllib.request

    with urllib.request.urlopen(url, timeout=timeout) as response:
        body = response.read()
    data = json.loads(body).get("data") or []
    return [
        {"id": row["id"], "pricing": row.get("pricing") or {}}
        for row in data
        if isinstance(row, dict) and row.get("id")
    ]


def read_table(path: Path) -> dict | None:
    """The table document at *path*, or ``None`` when it is absent/unreadable."""
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def table_age(
    doc: dict | None, path: Path, *, now: float | None = None
) -> float | None:
    """Seconds since the table was fetched.

    The instant is the document's own ``at`` field, which is what a cost is
    priced against. The file's modification time is consulted only when that
    field is absent, because a copy, a restore or a touch moves the mtime
    without moving the fetch. ``None`` means the age is unknown -- there is
    no table and no timestamp to read -- which is not the same as fresh.
    """
    moment = time.time() if now is None else now
    stamp: float | None = None
    if doc is not None:
        at = doc.get("at")
        if isinstance(at, (int, float)) and not isinstance(at, bool):
            stamp = float(at)
    if stamp is None:
        try:
            stamp = path.stat().st_mtime
        except OSError:
            return None
    return max(0.0, moment - stamp)


def write_atomic(path: Path, payload: bytes) -> None:
    """Write *payload* to *path* through a scratch file renamed onto it.

    A reader racing the writer sees either the previous table or the new one,
    never a truncated document. The scratch file is a sibling so the rename
    stays within one filesystem, and the result is left world-readable so a
    reader in another session can open it.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    scratch = target.with_suffix(target.suffix + ".tmp")
    scratch.write_bytes(payload)
    os.replace(scratch, target)
    target.chmod(0o644)


def migrate_legacy(path: Path) -> bool:
    """Seed *path* from the retired cache, once, while *path* is absent.

    This is a migration rather than a fallback: it runs only when the table
    is not yet at the resolved path, and never overwrites a table that is
    already there, so a refresh that has moved the table on is not undone by
    a later render reaching for the old file.
    """
    target = Path(path)
    if target.exists() or not LEGACY_PRICE_PATH.exists():
        return False
    try:
        payload = LEGACY_PRICE_PATH.read_bytes()
    except OSError:
        return False
    try:
        write_atomic(target, payload)
    except OSError:
        return False
    return True


def refresh(
    *,
    path: Path | None = None,
    config: SiteConfig | None = None,
    fetch: Callable[[str, float], list[dict]] | None = None,
    now: float | None = None,
) -> Refresh:
    """Ensure the table in force is no older than :data:`REFRESH_AGE`.

    Returns the path, the age of the table now in force, whether a fetch was
    written, and any fetch error. A failed or timed-out fetch leaves the
    cached table in use and reports its age; a fetch that returns nothing is
    treated as a failure rather than overwriting a good table with an empty
    one.
    """
    target = Path(path) if path is not None else table_path(config)
    migrate_legacy(target)
    doc = read_table(target)
    age = table_age(doc, target, now=now)
    if age is not None and age <= REFRESH_AGE:
        return Refresh(target, age, False, None)

    fetcher = fetch if fetch is not None else fetch_models
    try:
        models = fetcher(PRICE_URL, FETCH_TIMEOUT)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return Refresh(target, age, False, f"{type(exc).__name__}: {exc}")
    if not models:
        return Refresh(target, age, False, "empty model list")

    stamp = time.time() if now is None else now
    payload = json.dumps({"at": stamp, "models": models}).encode("utf-8")
    try:
        write_atomic(target, payload)
    except OSError as exc:
        return Refresh(target, age, False, f"{type(exc).__name__}: {exc}")
    return Refresh(target, 0.0, True, None)


__all__ = [
    "FETCH_TIMEOUT",
    "LEGACY_PRICE_PATH",
    "PRICE_URL",
    "REFRESH_AGE",
    "Refresh",
    "fetch_models",
    "migrate_legacy",
    "read_table",
    "refresh",
    "table_age",
    "table_path",
    "write_atomic",
]
