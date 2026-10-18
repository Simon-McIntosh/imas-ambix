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
import gzip
import json
import logging
import os
import random
import re
import shutil
import time
import unicodedata
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Receipts for one routing process, beside the endpoint and lane documents the
# router's launching command already names. This is the legacy single-file name,
# which the roll below no longer appends to: it is kept because a reader still
# yields it first and because a deployed record predates the roll.
RECEIPTS_FILENAME = "requests.jsonl"

# A day's rows land in their own file beside the legacy name, dated by the UTC
# day of the row's own timestamp: ``requests-2026-10-01.jsonl``. A day file is
# closed when its UTC day ends, and is compressed to ``.jsonl.gz`` once closed
# for this many whole UTC days. Nothing deletes a receipt.
DAY_FILE_SUFFIX = ".jsonl"
CLOSED_DAYS_BEFORE_COMPRESSION = 2

_DAY_FILE_PATTERN = re.compile(
    r"^(?P<stem>.+)-(?P<day>\d{4}-\d{2}-\d{2})"
    + re.escape(DAY_FILE_SUFFIX)
    + r"(\.gz)?$"
)

# The upstream field on a row for a request the router answered itself -- one it
# refused before any engine was chosen, or a catalog listing it served from the
# merged catalogs. Distinct from every engine origin, so a reader summing
# traffic per upstream never adds these to a sink's share of it.
SELF_ANSWERED_UPSTREAM = "(router)"

# The two request headers carrying a session identity into the record. They are
# absent on most consumers, and are never inferred when absent: a missing value
# recorded as a deducible one would read exactly like a reported one, and the
# share of unkeyed requests is itself a published figure. A caller sets them for
# its workers, and Claude Code forwards every custom header on every request.
RUN_ID_HEADER = "x-reckon-run-id"
COORDINATOR_SESSION_HEADER = "x-reckon-session"

# A well-formed identity is a short token. A value longer than this, empty, or
# carrying a control character is not one this router will publish, and is
# recorded as absent rather than truncated -- a truncated id would collide with
# a different session's, which is the whole property the field exists to give.
MAX_IDENTITY_CHARS = 128

# Bound the parse only where it protects the relay's memory: a single SSE line
# longer than this is dropped rather than accumulated, and the non-streaming
# fallback keeps at most this much of the body to parse at the end. Neither
# bound limits what the router relays -- the client receives every byte either
# way.
MAX_LINE_BYTES = 8 << 20
HEAD_BYTES = 4 << 20

# A trailing-window read starts this far from a file's end and grows fourfold
# until it reaches a row that ended before the window opened. A busy minute is a
# few hundred kilobytes of rows, so the first read nearly always suffices; the
# ceiling only stops a pathological file from being read whole on every poll.
RECENT_TAIL_FIRST_BYTES = 1 << 20
RECENT_TAIL_MAX_BYTES = 32 << 20

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


def _identity_token(raw: bytes) -> str | None:
    """Read one header value as a session token, or None when it is not one.

    None is the answer for absent, empty, over-long and control-bearing values
    alike, because all four mean the same thing to a reader: this request
    carried no usable identity. A caller that sends a malformed value therefore
    lands in the unkeyed share rather than inventing a key.
    """
    if not raw:
        return None
    value = raw.decode("utf-8", "replace")
    if len(value) > MAX_IDENTITY_CHARS:
        return None
    if any(unicodedata.category(character) == "Cc" for character in value):
        return None
    return value


def identity_from_headers(
    headers: Iterable[tuple[bytes, bytes]],
) -> tuple[str | None, str | None]:
    """Return ``(run_id, coordinator_session)`` from a request's raw headers.

    Both are None unless the caller reported them, and a header repeated within
    one request takes its last value -- the same rule a client library applies
    when it assembles them.
    """
    found: dict[str, str | None] = {
        RUN_ID_HEADER: None,
        COORDINATOR_SESSION_HEADER: None,
    }
    for name, value in headers or ():
        key = bytes(name).decode("latin-1").lower()
        if key in found:
            found[key] = _identity_token(bytes(value))
    return found[RUN_ID_HEADER], found[COORDINATOR_SESSION_HEADER]


@dataclass(frozen=True, slots=True)
class RequestReceipt:
    """One completed request, as it appears in the record.

    Two identities ride beside the connection-level ``caller_hint``: ``run_id``
    is the crew run that issued the request and ``coordinator_session`` the
    coordinator session behind it, both None when the caller reported neither.
    They are copied from request headers and never inferred, so a row's null
    says the caller was unkeyed rather than that the router failed to read it.
    """

    timestamp: str
    model: str
    upstream: str
    dialect: str
    prompt_tokens: int | None
    cached_prompt_tokens: int | None
    completion_tokens: int | None
    reasoning_tokens: int | None
    time_to_first_token_s: float | None
    duration_s: float
    gate_wait_s: float
    caller_hint: str
    run_id: str | None
    coordinator_session: str | None
    status: str
    sample_fraction: float
    sampling_rate_per_s: float

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=False)


# The dialect a response was read in, when no event identified one. A row
# carrying it is a row whose null counts are unexplained, which is the state
# worth seeing: every dialect below names the counts it can fill.
DIALECT_UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ResponseDialect:
    """One response vocabulary the relay can read accounting out of.

    Engines answer the same request in different shapes, and a reader that
    knows one of them records nothing from the others while still writing a
    well-formed row -- so the vocabulary is declared as data here rather than
    spelled into the parse, and a new shape is an entry in ``DIALECTS`` rather
    than a change to the reader.

    ``usage_containers`` names the record keys a usage object may be nested
    under, because a streamed first event may carry its counts inside the
    message it opens rather than beside it. The empty string means the record
    itself.

    The token fields are ordered candidates: the first key present in a usage
    object wins, so a spelling an engine renames can be accepted alongside the
    one it replaced without a second dialect.
    """

    name: str
    usage_containers: tuple[str, ...]
    prompt_tokens: tuple[str, ...]
    completion_tokens: tuple[str, ...]
    reasoning_tokens: tuple[str, ...]
    cached_tokens: tuple[str, ...]
    cached_containers: tuple[str, ...]
    event_key: str
    delta_events: tuple[str, ...]
    delta_list: str
    delta_key: str
    delta_fields: tuple[str, ...]

    def usage_objects(self, record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        """Return every usage object this dialect can reach in one record."""
        found: list[Mapping[str, Any]] = []
        for container in self.usage_containers:
            holder: Any = record
            if container:
                holder = record.get(container)
            if not isinstance(holder, Mapping):
                continue
            usage = holder.get("usage")
            if isinstance(usage, Mapping):
                found.append(usage)
        return found

    def speaks(self, usage: Mapping[str, Any]) -> bool:
        """Say whether a usage object is spelled in this dialect."""
        for names in (self.prompt_tokens, self.completion_tokens):
            for name in names:
                if name in usage:
                    return True
        return False

    def carries_first_token(self, record: Mapping[str, Any]) -> bool:
        """Say whether this record is the arrival of a first content token."""
        deltas: list[Any] = []
        if self.delta_list:
            listed = record.get(self.delta_list)
            if isinstance(listed, list) and listed:
                first = listed[0]
                if isinstance(first, Mapping):
                    deltas.append(first.get(self.delta_key))
        elif self.event_key:
            if str(record.get(self.event_key) or "") not in self.delta_events:
                return False
            deltas.append(record.get(self.delta_key))
        for delta in deltas:
            if not isinstance(delta, Mapping):
                continue
            for field in self.delta_fields:
                if delta.get(field):
                    return True
        return False


# Measured against the deployed serve on 2026-09-23 rather than transcribed
# from a specification: the OpenAI shape reports a null ``prompt_tokens_details``
# and a flat ``reasoning_tokens``, and the Anthropic shape nests the opening
# counts inside the message it starts and emits no cache figures at all.
DIALECTS: tuple[ResponseDialect, ...] = (
    ResponseDialect(
        name="openai",
        usage_containers=("",),
        prompt_tokens=("prompt_tokens",),
        completion_tokens=("completion_tokens",),
        reasoning_tokens=("reasoning_tokens",),
        cached_tokens=("cached_tokens",),
        cached_containers=("prompt_tokens_details",),
        event_key="",
        delta_events=(),
        delta_list="choices",
        delta_key="delta",
        delta_fields=("content", "reasoning_content", "reasoning"),
    ),
    ResponseDialect(
        name="anthropic",
        usage_containers=("", "message"),
        prompt_tokens=("input_tokens",),
        completion_tokens=("output_tokens",),
        reasoning_tokens=("reasoning_output_tokens",),
        cached_tokens=("cache_read_input_tokens", "cache_creation_input_tokens"),
        cached_containers=(),
        event_key="type",
        delta_events=("content_block_delta",),
        delta_list="",
        delta_key="delta",
        delta_fields=("text", "thinking"),
    ),
)


def _first_present(usage: Mapping[str, Any], names: tuple[str, ...]) -> int | None:
    """Return the first of *names* the usage object states as a count."""
    for name in names:
        if name in usage:
            value = _as_int(usage.get(name))
            if value is not None:
                return value
    return None


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
        self.reasoning_tokens: int | None = None
        self.dialect: str = DIALECT_UNKNOWN

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
        # The parentheses are load-bearing and the directive keeps them: this
        # module is imported by the serving interpreter, which predates
        # unparenthesised except expressions, and the formatter removes them.
        except (UnicodeDecodeError, json.JSONDecodeError):  # fmt: skip
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
        # The parentheses are load-bearing and the directive keeps them: this
        # module is imported by the serving interpreter, which predates
        # unparenthesised except expressions, and the formatter removes them.
        except (UnicodeDecodeError, json.JSONDecodeError):  # fmt: skip
            return
        if not isinstance(record, Mapping):
            return
        self._saw_event = True
        self._apply(record)

    def _apply(self, record: Mapping[str, Any]) -> None:
        """Read one event's usage and first-token arrival, in any dialect.

        Every declared dialect is offered the record, because one response may
        be read by only one of them and the relay is not told in advance which.
        A dialect that recognises the spelling of a usage object claims the
        stream, so the row states which vocabulary its counts came from and a
        row with no counts is distinguishable from a row nobody could read.

        The counts are last-write-wins within a stream: a streamed response
        opens with the prompt it was given and closes with the completion it
        produced, so the closing figure is the authoritative one.
        """
        for dialect in DIALECTS:
            for usage in dialect.usage_objects(record):
                if not dialect.speaks(usage):
                    continue
                self.dialect = dialect.name
                prompt = _first_present(usage, dialect.prompt_tokens)
                if prompt is not None:
                    self.prompt_tokens = prompt
                completion = _first_present(usage, dialect.completion_tokens)
                if completion is not None:
                    self.completion_tokens = completion
                reasoning = _first_present(usage, dialect.reasoning_tokens)
                if reasoning is not None:
                    self.reasoning_tokens = reasoning
                cached = _cached_tokens(usage, dialect)
                if cached is not None:
                    self.cached_prompt_tokens = cached
        if self._first_token_at is not None:
            return
        for dialect in DIALECTS:
            if dialect.carries_first_token(record):
                if self.dialect == DIALECT_UNKNOWN:
                    self.dialect = dialect.name
                self._first_token_at = self._clock()
                return


def _as_int(value: Any) -> int | None:
    """Return *value* as an int, or None when it is not a count."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def _cached_tokens(usage: Mapping[str, Any], dialect: ResponseDialect) -> int | None:
    """Read the cached prompt tokens the way *dialect* reports them.

    A nested object is tried before a flat key because an engine that carries
    both states the detail in the nested one. A dialect declaring no cache
    spelling returns nothing rather than zero, which keeps "this engine does
    not report cache use" distinct from "this engine reported none".
    """
    for container in dialect.cached_containers:
        details = usage.get(container)
        if isinstance(details, Mapping):
            nested = _first_present(details, dialect.cached_tokens)
            if nested is not None:
                return nested
    return _first_present(usage, dialect.cached_tokens)


def _isoformat(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _utc_date(moment: datetime | date | None) -> date:
    """The UTC day *moment* falls in, or today when none is given."""
    if moment is None:
        return datetime.now(UTC).date()
    if isinstance(moment, datetime):
        return moment.astimezone(UTC).date()
    return moment


def day_file_path(receipts_path: str | Path, day: date) -> Path:
    """The day file a row dated *day* belongs in, beside the legacy name.

    ``requests.jsonl`` beside the lane document becomes
    ``requests-2026-10-01.jsonl``: the same stem, the day it covers, and the
    same suffix, so a reader that knows the legacy name can find the rolled set
    without being told a second name.
    """
    base = Path(receipts_path)
    return base.with_name(f"{base.stem}-{day.isoformat()}{base.suffix}")


def _day_from_name(name: str, *, stem: str | None = None) -> date | None:
    """The UTC day a day-file name covers, or ``None`` when it is not one."""
    match = _DAY_FILE_PATTERN.match(name)
    if match is None:
        return None
    if stem is not None and match["stem"] != stem:
        return None
    return date.fromisoformat(match["day"])


def _rolled_files(directory: Path) -> list[Path]:
    """The legacy file, then every day file under *directory* in date order.

    Legacy rows were written before the roll took effect and the legacy file is
    never rewritten, so yielding it first and the day files in ascending date
    order reconstructs the record in the order the rows were written.
    """
    stem = Path(RECEIPTS_FILENAME).stem
    files: list[Path] = []
    legacy = directory / RECEIPTS_FILENAME
    if legacy.is_file():
        files.append(legacy)
    dated: list[tuple[date, str, Path]] = []
    for candidate in directory.iterdir():
        if not candidate.is_file() or candidate.name == RECEIPTS_FILENAME:
            continue
        day = _day_from_name(candidate.name, stem=stem)
        if day is not None:
            dated.append((day, candidate.name, candidate))
    dated.sort(key=lambda entry: (entry[0], entry[1]))
    files.extend(entry[2] for entry in dated)
    return files


def _read_rows_from(path: Path) -> list[dict[str, Any]]:
    """Every object row in one receipt file, plain or gzip-compressed.

    A blank line is skipped, as the sink never writes one. A line that does not
    parse as an object names its file and line rather than being dropped: a
    malformed receipt is a defect in the writer, and a reader that silently
    omitted it would understate the record.
    """
    rows: list[dict[str, Any]] = []
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid receipt JSON at {path}:{number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"receipt line {number} of {path} is not an object")
            rows.append(row)
    return rows


def read_receipt_rows(location: str | Path) -> list[dict[str, Any]]:
    """Every receipt row under *location*, in the order the rows were written.

    Given one file, that file's rows whether it is plain or compressed. Given a
    directory, the whole rolled set: the legacy ``requests.jsonl`` first, then
    each day file, plain or compressed, in date order.
    """
    path = Path(location)
    files = _rolled_files(path) if path.is_dir() else [path]
    rows: list[dict[str, Any]] = []
    for file in files:
        rows.extend(_read_rows_from(file))
    return rows


def receipt_interval(row: Mapping[str, Any]) -> tuple[float, float, float] | None:
    """When one request arrived, was admitted and ended, as epoch seconds.

    A relayed row's ``timestamp`` is the moment the relay began, after the gate
    admitted it, so the request arrived ``gate_wait_s`` earlier and ended
    ``duration_s`` later. ``None`` when the row carries no readable timestamp or
    duration, because an interval guessed for it would be a figure the record
    never stated.
    """
    stamp = row.get("timestamp")
    duration = row.get("duration_s")
    if not isinstance(stamp, str) or isinstance(duration, bool):
        return None
    if not isinstance(duration, (int, float)):
        return None
    try:
        admitted = datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None
    wait = row.get("gate_wait_s")
    if isinstance(wait, bool) or not isinstance(wait, (int, float)):
        wait = 0.0
    return admitted - max(0.0, wait), admitted, admitted + max(0.0, duration)


def _tail_rows_since(path: Path, since: float) -> tuple[list[dict[str, Any]], bool]:
    """Rows of *path* that ended at or after *since*, read back from its end.

    A row is appended when its request ends, so the file is in end order and a
    read that reaches a row ending before *since* has covered the window. The
    read starts at :data:`RECENT_TAIL_FIRST_BYTES` and grows until it does, or
    until :data:`RECENT_TAIL_MAX_BYTES`; the flag says which. The line the read
    lands inside is dropped, and so is a final line the writer has not finished;
    any other line that does not parse is a writer defect and raises, as the
    full reader does.
    """
    size = path.stat().st_size
    span = RECENT_TAIL_FIRST_BYTES
    while True:
        start = max(0, size - span)
        with path.open("rb") as handle:
            handle.seek(start)
            data = handle.read(size - start)
        lines = data.split(b"\n")
        if start > 0:
            lines = lines[1:]
        rows: list[dict[str, Any]] = []
        last = len(lines) - 1
        for number, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                if number == last:
                    break
                raise ValueError(
                    f"invalid receipt JSON in the tail of {path}"
                ) from error
            if isinstance(row, dict):
                rows.append(row)
        ends = [
            interval[2]
            for interval in (receipt_interval(row) for row in rows)
            if interval is not None
        ]
        covered = start == 0 or (bool(ends) and min(ends) < since)
        if covered or span >= RECENT_TAIL_MAX_BYTES:
            kept = []
            for row in rows:
                interval = receipt_interval(row)
                if interval is not None and interval[2] >= since:
                    kept.append(row)
            return kept, covered
        span *= 4


def recent_receipt_rows(
    location: str | Path, *, since: datetime
) -> tuple[list[dict[str, Any]], bool]:
    """Every row that ended at or after *since*, and whether the read covered it.

    The trailing-window counterpart of :func:`read_receipt_rows`, for a reader
    that polls: each file of the rolled set is read back from its end rather
    than whole, a compressed file is skipped because it holds a closed day, and
    a file not written since *since* is skipped unread. The flag is false when
    a file was cut at :data:`RECENT_TAIL_MAX_BYTES` before reaching the window's
    start, so a figure built from these rows can say it covers less than asked.
    A location holding no record at all raises :class:`FileNotFoundError`,
    because an empty answer there would read as a lane nobody used.
    """
    path = Path(location)
    files = _rolled_files(path) if path.is_dir() else [path]
    if not any(file.is_file() for file in files):
        raise FileNotFoundError(f"no request record at {path}")
    cutoff = since.timestamp()
    rows: list[dict[str, Any]] = []
    covered = True
    for file in files:
        if file.name.endswith(".gz") or not file.is_file():
            continue
        if file.stat().st_mtime < cutoff:
            continue
        found, reached = _tail_rows_since(file, cutoff)
        rows.extend(found)
        covered = covered and reached
    return rows, covered


def _compress(source: Path) -> Path:
    """Compress *source* in place and return the ``.gz`` file it became."""
    target = source.with_name(source.name + ".gz")
    temporary = target.with_name(target.name + ".partial")
    with source.open("rb") as reader, gzip.open(temporary, "wb") as writer:
        shutil.copyfileobj(reader, writer)
    os.replace(temporary, target)
    source.unlink()
    return target


def compress_closed_day_files(
    directory: str | Path,
    *,
    base_name: str = RECEIPTS_FILENAME,
    now: datetime | date | None = None,
    closed_days: int = CLOSED_DAYS_BEFORE_COMPRESSION,
) -> list[Path]:
    """Compress every day file closed for *closed_days* whole UTC days.

    A day file covers one UTC day and is closed when that day ends, so it has
    been closed for *closed_days* whole days once the current UTC day is
    ``closed_days + 1`` days after the day it covers. An already-compressed file
    and the still-open current day are both left alone, and nothing is deleted.
    """
    today = _utc_date(now)
    stem = Path(base_name).stem
    compressed: list[Path] = []
    for candidate in sorted(Path(directory).iterdir()):
        if not candidate.is_file() or candidate.name.endswith(".gz"):
            continue
        day = _day_from_name(candidate.name, stem=stem)
        if day is None:
            continue
        if (today - day).days >= closed_days + 1:
            compressed.append(_compress(candidate))
    return compressed


class RequestReceiptSink:
    """Append receipts to one stream, sampling above a configured rate.

    One file handle for the process rather than one open per request, and each
    row flushed as it is written so the record is durable if the router is
    killed mid-service. Writing never disturbs routing: a sink that cannot write
    logs once and discards, because a relay that stopped serving requests to
    keep a metrics file current would have inverted its own purpose.

    A rolling sink writes each row to a day file beside the name it was given,
    so the record is a rolled set rather than one file that grows without
    bound. A launch that names its own receipts path gets one file at that
    path instead: the operator named a location to read back, and
    :func:`read_receipt_rows` reads a file it is handed, so a redirect that
    scattered that path's rows into siblings would answer the operator's own
    reader with nothing.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        roll_daily: bool = True,
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
        self.roll_daily = roll_daily
        self.max_rows_per_s = max_rows_per_s
        self.window_s = window_s
        self._random = random_float
        self._monotonic = monotonic
        self._now = now
        self._arrivals: deque[float] = deque()
        self._handle: Any = None
        self._handle_day: date | None = None
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
        gate_wait_s: float = 0.0,
        run_id: str | None = None,
        coordinator_session: str | None = None,
        timestamp: datetime | None = None,
    ) -> RequestReceipt | None:
        """Append one row, or drop it. Returns the row written, or None."""
        fraction, rate = self.sample_decision()
        if fraction < 1.0 and self._random() >= fraction:
            self.rows_sampled_out += 1
            return None
        moment = timestamp if timestamp is not None else self._now()
        receipt = RequestReceipt(
            timestamp=_isoformat(moment),
            model=model,
            upstream=upstream,
            dialect=accounting.dialect,
            prompt_tokens=accounting.prompt_tokens,
            cached_prompt_tokens=accounting.cached_prompt_tokens,
            completion_tokens=accounting.completion_tokens,
            reasoning_tokens=accounting.reasoning_tokens,
            time_to_first_token_s=_round_or_none(accounting.time_to_first_token_s),
            duration_s=round(duration_s, 6),
            gate_wait_s=round(max(0.0, gate_wait_s), 6),
            caller_hint=caller_hint,
            run_id=run_id,
            coordinator_session=coordinator_session,
            status=status,
            sample_fraction=round(fraction, 6),
            sampling_rate_per_s=round(rate, 3),
        )
        if self._write(receipt, moment):
            self.rows_written += 1
            return receipt
        return None

    def _write(self, receipt: RequestReceipt, moment: datetime) -> bool:
        if self._write_failed:
            return False
        try:
            self._ensure_handle(moment)
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

    def _ensure_handle(self, moment: datetime) -> None:
        """Open the day file *moment* falls in, rolling and ageing on a new day.

        The handle is held open across a day's rows and re-opened when the UTC
        day moves, which is the roll. The sweep is idempotent, so a process that
        started after the roll began still compresses whatever has come of age.
        A sink that keeps one file opens that file once and never closes it.
        """
        day = _utc_date(moment)
        if self._handle is not None and self._handle_day == day:
            return
        self.close()
        target = day_file_path(self.path, day) if self.roll_daily else self.path
        target.parent.mkdir(parents=True, exist_ok=True)
        self._handle = target.open("a", encoding="utf-8")
        self._handle_day = day
        if self.roll_daily:
            compress_closed_day_files(
                target.parent, base_name=self.path.name, now=moment
            )

    def close(self) -> None:
        if self._handle is not None:
            with contextlib.suppress(OSError):
                self._handle.close()
            self._handle = None
            self._handle_day = None


def _round_or_none(value: float | None) -> float | None:
    return None if value is None else round(value, 6)
