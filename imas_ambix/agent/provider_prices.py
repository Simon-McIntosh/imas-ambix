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
is the time a cost is priced against. The history beside it is the record of
any price the table has ever held. The table states what the rates are now;
the history states what they were. A cost over a period that spans a
repricing can only be totalled from the second, because the first has
forgotten the earlier figure by the time anyone asks.

The history is append-only and records changes rather than fetches, so it
grows with the number of repricings rather than with the number of renders.
Two questions are asked of it and they read different rows: the headline a
client would have been billed at (:func:`rate_in_force`), and the median of
the endpoints that declare eight bits or more, which is what a client paying
market rates for a full-precision route would have been billed
(:func:`provider_median_in_force`). A headline row and an endpoints row are
different products at different prices and neither answers the other's
question.

All prices here are in the provider's own unit, dollars per token, the unit
the models endpoint publishes and the one the ledger multiplies tokens by.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from imas_ambix.agent.profile import SiteConfig

#: The provider's model list, carrying each model's published rates.
PRICE_URL = "https://openrouter.ai/api/v1/models"

#: The listing of every provider that serves one model, each with its own
#: rates and its declared quantisation. It is a different document from the
#: model list, and for one slug it carries prices the model list's blended
#: headline does not match.
ENDPOINTS_URL = "https://openrouter.ai/api/v1/models/{slug}/endpoints"

#: The history that sits beside the table, one JSON object per line.
HISTORY_NAME = "openrouter-price-history.jsonl"

#: How long a fetch may block. The bound is the whole of the wait a render
#: will accept for a price table: past it the render proceeds on whatever
#: is already cached rather than stall the panel.
FETCH_TIMEOUT = 10.0

#: A table younger than this is not refetched. The provider's published
#: rates move within hours, so the window is an hour. A shorter one would
#: sharpen when a change is noticed at the cost of a fetch per render; the
#: history carries the movement between fetches either way.
REFRESH_AGE = 3600.0

#: The retired workstation cache the table moved from. Consulted once, to
#: seed the resolved path where it is absent, so no price history is lost.
LEGACY_PRICE_PATH = Path.home() / ".cache" / "gpu-watch" / "openrouter-prices.json"

#: The slug the recovered observations belong to.
RECOVERED_SLUG = "deepseek/deepseek-v4.1-flash"

#: The first digit run in a declared quantisation is its width in bits.
_QUANT_BITS = re.compile(r"(\d+)")

#: Rates this deployment was actually billed at before the history existed,
#: oldest first. A history created from nothing begins with these, so an
#: estimate over a past period is priced against something observed rather
#: than against today's table's figure applied backwards. Prices are per
#: token, matching the models rows; the table lists them per million for
#: reading. A row carries where its rate came from, because a recovered rate
#: is weaker evidence than a fetched one and a reader must be able to tell
#: which they are looking at. The 2026-10-04 row has no cache-read tier
#: recorded: it was not observed, and pricing it at the prompt rate
#: overstates rather than inventing a discount.
RECOVERED_OBSERVATIONS: tuple[dict, ...] = (
    {
        "at": 1790640000.0,
        "pricing": {
            "prompt": 3e-07,
            "input_cache_read": 6e-09,
            "completion": 1.2e-06,
        },
        "citation": (
            "billed Together responses in the third-party hit-rate measurement"
        ),
    },
    {
        "at": 1790935380.0,
        "pricing": {
            "prompt": 1.37e-08,
            "input_cache_read": 3.6e-09,
            "completion": 8.91e-07,
        },
        "citation": (
            "back-solved: the only rate reproducing a month read's cost of "
            "$243.40 from 22.0 G input and 166 M output tokens, to within 1%; "
            "it equals the cheapest provider's listing taken later"
        ),
    },
    {
        "at": 1791104400.0,
        "pricing": {"prompt": 3e-09, "completion": 2.4e-06},
        "citation": "measured from the price table fetched then",
    },
    {
        "at": 1791190440.0,
        "pricing": {
            "prompt": 3e-07,
            "input_cache_read": 6e-09,
            "completion": 1.2e-06,
        },
        "citation": "measured from the price table fetched then",
    },
    {
        "at": 1791970440.0,
        "pricing": {
            "prompt": 2.088e-08,
            "input_cache_read": 5.4e-09,
            "completion": 8.91e-07,
        },
        "citation": "the site price table's own fetch stamp",
    },
    {
        "at": 1791992220.0,
        "pricing": {
            "prompt": 2.016e-08,
            "input_cache_read": 5.4e-09,
            "completion": 8.91e-07,
        },
        "citation": "the retired workstation cache copy's own fetch stamp",
    },
)


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


def endpoints_url(slug: str) -> str:
    """The endpoints listing for one model slug."""
    return ENDPOINTS_URL.format(slug=slug)


def fetch_endpoints(url: str, timeout: float = FETCH_TIMEOUT) -> list[dict]:
    """The providers serving one model, slimmed to what the median needs.

    Each row keeps the provider, its tag, the quantisation it declares and its
    own pricing block. Raises on any failure, exactly as :func:`fetch_models`
    does, so the caller records no listing."""
    import urllib.request

    with urllib.request.urlopen(url, timeout=timeout) as response:
        body = response.read()
    data = json.loads(body).get("data")
    rows = data.get("endpoints") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise ValueError("endpoints listing carried no rows")
    return [
        {
            "provider": row.get("provider_name") or row.get("provider") or "",
            "tag": row.get("tag") or "",
            "quantization": row.get("quantization") or "",
            "pricing": row["pricing"],
        }
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("pricing"), dict)
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


def history_path(table: Path | None = None, config: SiteConfig | None = None) -> Path:
    """The history beside the table, resolved the way the table is."""
    target = Path(table) if table is not None else table_path(config)
    return target.parent / HISTORY_NAME


def read_history(path: Path) -> list[dict]:
    """Every recorded observation, oldest first.

    A line that does not parse is skipped rather than raised on. The file is
    appended to and never rewritten, so a writer killed mid-line leaves one
    torn final line and every observation before it is intact; discarding the
    whole history for that would lose the part that is sound.
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return []
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def append_history(path: Path, rows: Sequence[dict]) -> None:
    """Append one JSON object per line, in the order given, and nothing else."""
    if not rows:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def seed_history(path: Path) -> bool:
    """Write the recovered observations, once, while the history is absent.

    This is a migration rather than a fallback: it runs only when the history
    does not yet exist, so a refresh that has recorded real observations is
    never rewritten from a constant, and a second refresh adds nothing.
    """
    target = Path(path)
    if target.exists():
        return False
    rows = [
        {
            "at": observation["at"],
            "id": RECOVERED_SLUG,
            "source": "recovered",
            "pricing": dict(observation["pricing"]),
            "citation": observation["citation"],
        }
        for observation in RECOVERED_OBSERVATIONS
    ]
    try:
        append_history(target, rows)
    except OSError:
        return False
    return True


def _rates(pricing: object) -> dict | None:
    """The three rates a cost is priced against, out of one pricing block.

    A block with no prompt or completion rate is not a price this module can
    use and is dropped rather than defaulted, because a zero invented for an
    unstated rate would price traffic as free. A missing cache-read tier is
    the one exception: cached input is priced at the prompt rate, which
    overstates the bill rather than inventing a discount that may not exist.
    """
    if not isinstance(pricing, dict):
        return None
    try:
        prompt = float(pricing["prompt"])
        completion = float(pricing["completion"])
    except (KeyError, TypeError, ValueError):
        return None
    try:
        cache = float(pricing.get("input_cache_read", pricing.get("cache_read")))
    except (TypeError, ValueError):
        cache = prompt
    return {"prompt": prompt, "cache_read": cache, "completion": completion}


def _declares_fewer_than_eight_bits(quantization: object) -> bool:
    """Whether an endpoint declares a quantisation narrower than eight bits.

    The width is the first digit run in the declared quantisation, so ``fp4``,
    ``int4``, ``mxfp4`` and ``fp6_e2m3`` all read as below the floor while
    ``fp8_e4m3`` and ``bf16`` do not. An endpoint that declares nothing is
    kept: a listing silent about its precision has not claimed to be narrow,
    and dropping it would answer a question the listing never raised.
    """
    match = _QUANT_BITS.search(str(quantization or ""))
    return match is not None and int(match.group(1)) < 8


def _median(values: list[float]) -> float:
    """The ordinary median, averaging the two middles of an even count."""
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _observations(history: object) -> list[dict]:
    """The rows to read, from a path or an already-read history."""
    if isinstance(history, (str, Path)):
        return read_history(Path(history))
    if history is None:
        return []
    return [row for row in history if isinstance(row, dict)]


def _at_or_before(row: dict, when: float) -> bool:
    """Whether a row carries a usable fetch stamp at or before *when*."""
    at = row.get("at")
    if not isinstance(at, (int, float)) or isinstance(at, bool):
        return False
    return float(at) <= when


def rate_in_force(history: object, slug: str, when: float) -> dict | None:
    """The headline rate for *slug* at the instant *when*.

    Headline observations are the model rows and the recovered ones; an
    endpoints row records a listing and never answers here. ``None`` means no
    headline had been recorded by *when* -- the rate is unobserved, which is
    not the same as zero and not the same as today's.
    """
    candidates = [
        row
        for row in _observations(history)
        if row.get("id") == slug
        and row.get("source") in ("models", "recovered")
        and _at_or_before(row, when)
    ]
    if not candidates:
        return None
    latest = max(candidates, key=lambda row: row["at"])
    return _rates(latest.get("pricing"))


def provider_median_in_force(history: object, slug: str, when: float) -> dict | None:
    """The median prompt, cache-read and completion rate at the instant *when*.

    Taken over the providers in the last endpoints listing at or before
    *when*, each rate taken separately, and only over those declaring eight
    bits or more. The headline is deliberately not used: it is the cheapest
    blended route, often a narrow-quantisation endpoint, and it is not a
    price a request can actually be billed at. ``None`` means no listing had
    been recorded by *when*, or that no endpoint in it survived the
    precision floor.
    """
    candidates = [
        row
        for row in _observations(history)
        if row.get("id") == slug
        and row.get("source") == "endpoints"
        and _at_or_before(row, when)
    ]
    if not candidates:
        return None
    latest = max(candidates, key=lambda row: row["at"])
    rates = []
    for endpoint in latest.get("endpoints") or []:
        if not isinstance(endpoint, dict):
            continue
        if _declares_fewer_than_eight_bits(endpoint.get("quantization")):
            continue
        rate = _rates(endpoint.get("pricing"))
        if rate is not None:
            rates.append(rate)
    if not rates:
        return None
    return {
        key: _median([rate[key] for rate in rates])
        for key in ("prompt", "cache_read", "completion")
    }


def _recorded_rates(rows: list[dict], slug: str) -> dict | None:
    """The last rates recorded for *slug*, from any source."""
    candidates = [
        row
        for row in rows
        if row.get("id") == slug and _rates(row.get("pricing")) is not None
    ]
    if not candidates:
        return None
    latest = max(candidates, key=lambda row: float(row.get("at") or 0.0))
    return _rates(latest.get("pricing"))


def _model_history_rows(
    recorded: list[dict], models: list[dict], stamp: float
) -> list[dict]:
    """One row per model whose three rates moved since its last recorded row.

    A model is recorded on the refresh that changes it and not on the ones
    that repeat it, so the history grows with the number of repricings rather
    than with the number of fetches. The first refresh records every model
    that carries a usable rate, because nothing has been recorded for it yet.
    A model whose row carries no usable rate is not a price and is left out.
    """
    rows = []
    for model in models:
        if not isinstance(model, dict):
            continue
        slug = model.get("id")
        pricing = model.get("pricing")
        if not slug or _rates(pricing) is None:
            continue
        if _recorded_rates(recorded, slug) == _rates(pricing):
            continue
        rows.append({"at": stamp, "id": slug, "source": "models", "pricing": pricing})
    return rows


def _endpoint_history_rows(
    recorded: list[dict],
    slugs: Sequence[str],
    stamp: float,
    fetch: Callable[[str, float], list[dict]] | None,
) -> list[dict]:
    """One row per named slug whose endpoints listing moved.

    Only the slugs the caller names are fetched: a headline model id is not a
    provider slug, and fetching a listing for every model in the table would
    turn one refresh into hundreds of round trips for rows nothing reads. A
    listing that cannot be fetched is skipped rather than failing the refresh
    -- the model rows are already recorded and are the ones a render needs.
    """
    fetcher = fetch if fetch is not None else fetch_endpoints
    rows = []
    for slug in slugs:
        try:
            endpoints = fetcher(endpoints_url(slug), FETCH_TIMEOUT)
        except Exception:
            continue
        if not endpoints:
            continue
        previous = [
            row
            for row in recorded
            if row.get("id") == slug and row.get("source") == "endpoints"
        ]
        if previous:
            latest = max(previous, key=lambda row: float(row.get("at") or 0.0))
            if json.dumps(latest.get("endpoints"), sort_keys=True) == json.dumps(
                endpoints, sort_keys=True
            ):
                continue
        rows.append(
            {"at": stamp, "id": slug, "source": "endpoints", "endpoints": endpoints}
        )
    return rows


def refresh(
    *,
    path: Path | None = None,
    config: SiteConfig | None = None,
    fetch: Callable[[str, float], list[dict]] | None = None,
    endpoints_fetch: Callable[[str, float], list[dict]] | None = None,
    now: float | None = None,
    slugs: Sequence[str] | None = None,
) -> Refresh:
    """Ensure the table in force is no older than :data:`REFRESH_AGE`.

    Returns the path, the age of the table now in force, whether a fetch was
    written, and any fetch error. A failed or timed-out fetch leaves the
    cached table in use and reports its age; a fetch that returns nothing is
    treated as a failure rather than overwriting a good table with an empty
    one.

    Every exception raised while fetching or parsing the models table is
    contained here -- an unreachable host, an expired socket, a body cut off
    mid-response, a body that parses to something other than an object -- so a
    provider fault degrades the render to a stale table rather than ending it.

    A fetch that succeeds also records what moved. Each model whose three
    rates differ from its last recorded row is appended to the history, and
    for each slug in *slugs* the endpoints listing is fetched and appended
    when its prices changed. A refresh that names no slug fetches no listing.
    A fetch that fails records nothing, exactly as it writes no table: the
    history is evidence of what the provider published, and a failed fetch
    published nothing.
    """
    target = Path(path) if path is not None else table_path(config)
    history = target.parent / HISTORY_NAME
    migrate_legacy(target)
    doc = read_table(target)
    age = table_age(doc, target, now=now)
    if age is not None and age <= REFRESH_AGE:
        return Refresh(target, age, False, None)

    fetcher = fetch if fetch is not None else fetch_models
    try:
        models = fetcher(PRICE_URL, FETCH_TIMEOUT)
    except Exception as exc:
        return Refresh(target, age, False, f"{type(exc).__name__}: {exc}")
    if not models:
        return Refresh(target, age, False, "empty model list")

    stamp = time.time() if now is None else now
    payload = json.dumps({"at": stamp, "models": models}).encode("utf-8")
    try:
        write_atomic(target, payload)
        seed_history(history)
        recorded = read_history(history)
        rows = _model_history_rows(recorded, models, stamp)
        rows.extend(
            _endpoint_history_rows(recorded, slugs or (), stamp, endpoints_fetch)
        )
        append_history(history, rows)
    except OSError as exc:
        return Refresh(target, age, False, f"{type(exc).__name__}: {exc}")
    return Refresh(target, 0.0, True, None)


__all__ = [
    "ENDPOINTS_URL",
    "FETCH_TIMEOUT",
    "HISTORY_NAME",
    "LEGACY_PRICE_PATH",
    "PRICE_URL",
    "RECOVERED_OBSERVATIONS",
    "RECOVERED_SLUG",
    "REFRESH_AGE",
    "Refresh",
    "append_history",
    "endpoints_url",
    "fetch_endpoints",
    "fetch_models",
    "history_path",
    "migrate_legacy",
    "provider_median_in_force",
    "rate_in_force",
    "read_history",
    "read_table",
    "refresh",
    "seed_history",
    "table_age",
    "table_path",
    "write_atomic",
]
