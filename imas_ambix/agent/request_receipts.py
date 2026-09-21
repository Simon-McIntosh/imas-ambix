"""Per-request receipts written by the routing relay.

The router is the only component that sees every call, so attribution belongs
here rather than at the serve: one append-only row per completed request, so a
lane-wide cost estimate becomes an attributable figure -- which workloads spent
the tokens, what a campaign cost, and which callers carry the prompts whose
cache misses show up as prefill on the cards.

The response-side accounting mirrors the shape of
:func:`imas_ambix.agent.bench._stream_chat` -- token counts come from the
server-reported ``usage`` object, and time to first token is taken at the first
non-empty content or reasoning delta. The shape is mirrored rather than
re-derived because a second, differently-shaped reader of the same stream would
drift from the first and the two sets of figures would stop being comparable.
What is new here is the tee: the router relays bytes it does not consume, so
the parse runs on the relayed copy and the body is never buffered in full.

Volume is bounded by design rather than by hope: a pathological client can
raise the request rate past what a full-fidelity row per call is worth, so the
sink samples above a configured rate and stamps the fraction it kept on every
row it writes, so a reader scaling totals back up does so honestly.
"""

from __future__ import annotations

import contextlib
import json
import logging
import random
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Receipts for one routing process, beside the endpoint and lane documents the
# router's launching command already names.
RECEIPTS_FILENAME = "requests.jsonl"

# The upstream field on a row for a request the router answered itself -- one it
# refused before any engine was chosen, or a catalog listing it served from the
# merged catalogs. Distinct from every engine origin, so a reader summing
# traffic per upstream never adds these to a sink's share of it.
SELF_ANSWERED_UPSTREAM = "(router)"

# Bound the parse only where it protects the relay's memory: a single SSE line
# longer than this is dropped rather than accumulated, and the non-streaming
# fallback keeps at most this much of the body to parse at the end. Neither
# bound limits what the router relays -- the client receives every byte either
# way.
MAX_LINE_BYTES = 8 << 20
HEAD_BYTES = 4 << 20

# Rows kept per second once the offered request rate is at or below this. Above
# it, the kept fraction is this rate over the offered one.
DEFAULT_MAX_ROWS_PER_S = 20.0
DEFAULT_WINDOW_S = 1.0

# Outcome labels. ``completed`` is an answer the router sent whole -- relayed
# from the owning engine, or composed by the router itself where it answered
# without relaying. It says the answer was handed over with no departure
# reported first; whether the caller read it is not observable from the serving
# side of the exchange, so a reader summing completed rows is counting answers
# sent rather than answers received. ``aborted`` is a caller that went away
# before its answer was handed over -- mid-relay, before the answer was sent,
# or while still uploading its request; and ``failed`` is anything else -- an
# upstream error status, a refusal handed over to a caller that had not gone,
# or a relay that raised.
STATUS_COMPLETED = "completed"
STATUS_ABORTED = "aborted"
STATUS_FAILED = "failed"


@dataclass(frozen=True, slots=True)
class RequestReceipt:
    """One completed request, as it appears in the append-only record."""

    timestamp: str
    model: str
    upstream: str
    prompt_tokens: int | None
    cached_prompt_tokens: int | None
    completion_tokens: int | None
    time_to_first_token_s: float | None
    duration_s: float
    caller_hint: str
    status: str
    sample_fraction: float
    sampling_rate_per_s: float

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=False)


class StreamAccounting:
    """Tee a relayed response body and read the request accounting out of it.

    Fed each relayed chunk as it is forwarded, so the parse adds no buffering of
    the body and the bytes the client receives are untouched. Both shapes the
    relay may carry are accepted: an SSE stream, where usage and the first delta
    arrive as separate events, and a single JSON body, which is the
    non-streaming equivalent of the same response.
    """

    def __init__(self, *, clock: Callable[[], float] = time.perf_counter) -> None:
        self._clock = clock
        self.started_at = clock()
        self._first_token_at: float | None = None
        self._pending = b""
        self._discarding = False
        self._head = bytearray()
        self._head_truncated = False
        self._saw_event = False
        self.prompt_tokens: int | None = None
        self.cached_prompt_tokens: int | None = None
        self.completion_tokens: int | None = None

    @property
    def time_to_first_token_s(self) -> float | None:
        if self._first_token_at is None:
            return None
        return self._first_token_at - self.started_at

    def feed(self, chunk: bytes) -> None:
        """Account for one relayed chunk without holding on to it."""
        if not self._saw_event and len(self._head) < HEAD_BYTES:
            room = HEAD_BYTES - len(self._head)
            self._head.extend(chunk[:room])
            if len(chunk) > room:
                self._head_truncated = True
        data = self._pending + chunk
        if self._discarding:
            marker = data.find(b"\n")
            if marker < 0:
                return
            data = data[marker + 1 :]
            self._discarding = False
        lines = data.split(b"\n")
        self._pending = lines.pop()
        if len(self._pending) > MAX_LINE_BYTES:
            self._pending = b""
            self._discarding = True
        for line in lines:
            self._consume_line(line)

    def finish(self) -> None:
        """Read the body as one JSON object when nothing streamed.

        Skipped when the body was truncated, because a partial object either
        fails to parse or parses to a confident wrong answer.
        """
        if self._saw_event or self._head_truncated or not self._head:
            return
        try:
            record = json.loads(bytes(self._head))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return
        if isinstance(record, Mapping):
            self._apply(record)

    def _consume_line(self, line: bytes) -> None:
        stripped = line.strip()
        if not stripped.startswith(b"data:"):
            return
        payload = stripped[5:].strip()
        if not payload or payload == b"[DONE]":
            return
        try:
            record = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return
        if not isinstance(record, Mapping):
            return
        self._saw_event = True
        self._apply(record)

    def _apply(self, record: Mapping[str, Any]) -> None:
        """Read one event's usage and first-token arrival.

        Field names and the first-token definition follow
        ``bench._stream_chat`` so a figure from the relay and a figure from the
        benchmark mean the same thing and can be compared.
        """
        usage = record.get("usage")
        if isinstance(usage, Mapping):
            prompt = _as_int(usage.get("prompt_tokens"))
            if prompt is not None:
                self.prompt_tokens = prompt
            completion = _as_int(usage.get("completion_tokens"))
            if completion is not None:
                self.completion_tokens = completion
            cached = _cached_tokens(usage)
            if cached is not None:
                self.cached_prompt_tokens = cached
        if self._first_token_at is not None:
            return
        choices = record.get("choices")
        if not isinstance(choices, list) or not choices:
            return
        first = choices[0]
        if not isinstance(first, Mapping):
            return
        delta = first.get("delta")
        if not isinstance(delta, Mapping):
            return
        if (
            delta.get("content")
            or delta.get("reasoning_content")
            or delta.get("reasoning")
        ):
            self._first_token_at = self._clock()


def _as_int(value: Any) -> int | None:
    """Return *value* as an int, or None when it is not a count."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def _cached_tokens(usage: Mapping[str, Any]) -> int | None:
    """Read the cached prompt tokens, which no other parser in this repository does.

    The nested ``prompt_tokens_details`` object is how the OpenAI-compatible
    engines report it; a top-level ``cached_tokens`` is accepted too because a
    serve that reports the count directly should not read as zero cache use.
    """
    details = usage.get("prompt_tokens_details")
    if isinstance(details, Mapping):
        nested = _as_int(details.get("cached_tokens"))
        if nested is not None:
            return nested
    return _as_int(usage.get("cached_tokens"))


def _isoformat(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class RequestReceiptSink:
    """Append receipts to one stream, sampling above a configured rate.

    One file handle for the process rather than one open per request, and each
    row flushed as it is written so the record is durable if the router is
    killed mid-service. Writing never disturbs routing: a sink that cannot write
    logs once and discards, because a relay that stopped serving requests to
    keep a metrics file current would have inverted its own purpose.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        max_rows_per_s: float = DEFAULT_MAX_ROWS_PER_S,
        window_s: float = DEFAULT_WINDOW_S,
        random_float: Callable[[], float] = random.random,
        monotonic: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if max_rows_per_s <= 0:
            raise ValueError("max_rows_per_s must be positive")
        if window_s <= 0:
            raise ValueError("window_s must be positive")
        self.path = Path(path)
        self.max_rows_per_s = max_rows_per_s
        self.window_s = window_s
        self._random = random_float
        self._monotonic = monotonic
        self._now = now
        self._arrivals: deque[float] = deque()
        self._handle: Any = None
        self._write_failed = False
        self.rows_written = 0
        self.rows_sampled_out = 0

    def sample_decision(self) -> tuple[float, float]:
        """Record an arrival and return ``(keep_fraction, offered_rate_per_s)``.

        The rate is measured over the trailing window, so the fraction follows
        a burst without needing to know one was coming. At or below the
        configured rate every request is kept, which is the state the record is
        normally in; above it the kept fraction is the configured rate over the
        offered one, so the rows scaled by their own fraction reconstruct the
        whole. Its inverse is also the expected share of arrivals kept, which is
        what makes the sampling reproducible from the record alone.
        """
        moment = self._monotonic()
        horizon = moment - self.window_s
        arrivals = self._arrivals
        while arrivals and arrivals[0] < horizon:
            arrivals.popleft()
        arrivals.append(moment)
        rate = len(arrivals) / self.window_s
        if rate <= self.max_rows_per_s:
            return 1.0, rate
        return self.max_rows_per_s / rate, rate

    def record(
        self,
        *,
        model: str,
        upstream: str,
        caller_hint: str,
        status: str,
        duration_s: float,
        accounting: StreamAccounting,
        timestamp: datetime | None = None,
    ) -> RequestReceipt | None:
        """Append one row, or drop it. Returns the row written, or None."""
        fraction, rate = self.sample_decision()
        if fraction < 1.0 and self._random() >= fraction:
            self.rows_sampled_out += 1
            return None
        receipt = RequestReceipt(
            timestamp=_isoformat(timestamp if timestamp is not None else self._now()),
            model=model,
            upstream=upstream,
            prompt_tokens=accounting.prompt_tokens,
            cached_prompt_tokens=accounting.cached_prompt_tokens,
            completion_tokens=accounting.completion_tokens,
            time_to_first_token_s=_round_or_none(accounting.time_to_first_token_s),
            duration_s=round(duration_s, 6),
            caller_hint=caller_hint,
            status=status,
            sample_fraction=round(fraction, 6),
            sampling_rate_per_s=round(rate, 3),
        )
        if self._write(receipt):
            self.rows_written += 1
            return receipt
        return None

    def _write(self, receipt: RequestReceipt) -> bool:
        if self._write_failed:
            return False
        try:
            if self._handle is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._handle = self.path.open("a", encoding="utf-8")
            self._handle.write(receipt.to_json() + "\n")
            self._handle.flush()
        except OSError as error:
            self._write_failed = True
            logger.warning(
                "request receipts disabled path=%s error=%s: %s",
                self.path,
                type(error).__name__,
                error,
            )
            return False
        return True

    def close(self) -> None:
        if self._handle is not None:
            with contextlib.suppress(OSError):
                self._handle.close()
            self._handle = None


def _round_or_none(value: float | None) -> float | None:
    return None if value is None else round(value, 6)
