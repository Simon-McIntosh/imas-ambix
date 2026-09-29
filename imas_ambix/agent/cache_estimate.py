"""Third-party cache hit rate estimated from request-receipt gaps.

A hosted provider expires its cached prefixes after a published lifetime, so
whether a request could have been served from that provider's cache is decided
by how long before it the same caller last sent the same model. This module
turns the router's request receipts into that figure: for each request it
compares the arrival time against the same caller's previous request to the
same model, and counts the request as a hit when the gap is strictly inside the
lifetime.

Arrival is the row's own timestamp. The router stamps a receipt at admission,
before the request is relayed, so the timestamp already marks the moment the
provider saw the request and no duration is subtracted from it. A gap exactly
equal to the lifetime is a miss: the entry has expired at the moment the next
request arrives, and the comparison is strict for that reason.

Only rows whose status is ``completed`` score or chain. A row carrying any
other status -- ``aborted``, ``failed``, or a status the record does not name --
is never a hit and never becomes the previous arrival another request chains
from, because its prompt may never have reached the provider cache and the
record cannot say whether it did. It still contributes its prompt tokens to the
denominator, so the figure stays over all the traffic rather than only the part
of it that could be attributed to a completed exchange.

The caller is the receipt's ``run_id`` when that is a non-empty string, else
its ``coordinator_session`` when that is a non-empty string. A request carrying
neither has no caller, never chains to anything, and is never a hit -- it still
contributes its prompt tokens to the denominator, as a non-completed row does.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from imas_ambix.agent.request_receipts import STATUS_COMPLETED


@dataclass(frozen=True, slots=True)
class CacheEstimate:
    """The hit rate one lifetime produces over one set of requests.

    ``hit_rate is the hit requests' prompt tokens over all prompt tokens``,
    which is the quantity a cost counterfactual multiplies: everything cached
    is billed at the cache-read rate and the rest at the prompt rate.
    """

    lifetime_s: float
    requests: int
    hits: int
    prompt_tokens: int
    hit_prompt_tokens: int

    @property
    def hit_rate(self) -> float:
        """Return hit prompt tokens over all prompt tokens, or 0 with none."""
        if self.prompt_tokens <= 0:
            return 0.0
        return self.hit_prompt_tokens / self.prompt_tokens


def _field(row: Any, name: str) -> Any:
    """Read one field from a receipt row, as a mapping or as an object."""
    if isinstance(row, Mapping):
        return row.get(name)
    return getattr(row, name, None)


def caller_of(row: Any) -> str | None:
    """Return the request's caller, or None when it declared no caller.

    The precedence is fixed: a non-empty ``run_id`` wins, then a non-empty
    ``coordinator_session``. Anything else -- absent, null, empty, or a
    non-string -- means no caller, which is a real state of the record rather
    than a value to be inferred from the connection.
    """
    for name in ("run_id", "coordinator_session"):
        value = _field(row, name)
        if isinstance(value, str) and value:
            return value
    return None


def arrival_of(row: Any) -> datetime | None:
    """Return the request's arrival as its timestamp.

    The router stamps the receipt at admission, so the timestamp already marks
    the moment the provider saw the request; the duration says how long the
    exchange then took and does not move the arrival. None when the row states
    no parsable timestamp, which is the only way it can be timed; a row without
    one cannot be placed on the timeline and is dropped rather than placed at an
    arbitrary instant.
    """
    timestamp = _field(row, "timestamp")
    if not isinstance(timestamp, str) or not timestamp:
        return None
    try:
        return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return None


def _completed(row: Any) -> bool:
    """Whether the row records an answer that was handed over whole.

    Only ``completed`` scores or chains; every other status, and a status the
    record does not state, means the exchange's prompt may not have reached the
    provider cache.
    """
    return _field(row, "status") == STATUS_COMPLETED


def _prompt_tokens(row: Any) -> int:
    """Read a row's prompt tokens as a count, or 0 when it states none."""
    value = _field(row, "prompt_tokens")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return int(value)


def estimate_hit_rate(rows: Iterable[Any], lifetime_s: float) -> CacheEstimate:
    """Estimate the third-party cache hit rate over *rows* at *lifetime_s*.

    A request is a hit when the same caller's previous completed request to the
    same model arrived strictly less than *lifetime_s* before it. Requests are
    ordered by arrival across the whole set, so a caller and model chain spans
    every row offered, not only adjacent ones, and are kept in arrival order
    rather than in file order. A row whose status is not ``completed`` neither
    scores nor chains: it is never a hit and never becomes the previous arrival
    another request measures against.
    """
    timed: list[tuple[Any, str | None, Any, int, bool]] = []
    for row in rows:
        arrival = arrival_of(row)
        if arrival is None:
            continue
        timed.append(
            (
                arrival,
                caller_of(row),
                _field(row, "model"),
                _prompt_tokens(row),
                _completed(row),
            )
        )
    timed.sort(key=lambda item: item[0])

    previous_arrival: dict[tuple[str, Any], datetime] = {}
    hits = 0
    hit_prompt_tokens = 0
    prompt_tokens = 0
    for arrival, caller, model, tokens, completed in timed:
        prompt_tokens += tokens
        if caller is None or not completed:
            continue
        key = (caller, model)
        previous = previous_arrival.get(key)
        if previous is not None and (arrival - previous).total_seconds() < lifetime_s:
            hits += 1
            hit_prompt_tokens += tokens
        previous_arrival[key] = arrival
    return CacheEstimate(
        lifetime_s=float(lifetime_s),
        requests=len(timed),
        hits=hits,
        prompt_tokens=prompt_tokens,
        hit_prompt_tokens=hit_prompt_tokens,
    )


def caller_share(rows: Iterable[Any]) -> float:
    """Return the share of rows that declared a caller, over all rows.

    A row with no caller cannot chain and so can never be a hit, which makes
    the share of rows that can chain part of how the estimate should be found:
    it is the fraction of traffic this rule is able to attribute at all.
    """
    seen = 0
    with_caller = 0
    for row in rows:
        seen += 1
        if caller_of(row) is not None:
            with_caller += 1
    if seen == 0:
        return 0.0
    return with_caller / seen
