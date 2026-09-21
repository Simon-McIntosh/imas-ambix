"""A disposable SQLite index over the append-only serving record.

The durable record is line-delimited JSON appended by the on-node recorder
(:mod:`imas_ambix.agent.serving_receipts`) and by the router
(:mod:`imas_ambix.agent.request_receipts`). It is written on GPFS, where
SQLite's WAL mode does not exist and rollback-journal locking beside a
continuous writer carries a documented risk, so the durable side is a short
append of one line and the query side is a local index derived from it. The
index can therefore never itself be the only copy of anything: it is built by
reading the record, deleted and rebuilt on corruption, on a schema change, or
on demand, and a rebuild is expected to reproduce the same query results.

**The tail is keyed on the inode, and a line's identity is its inode and byte
offset.** The two are different jobs, and each covers what the other cannot.
The offset records how far this index has consumed a byte stream, so an
ordinary append adds only the new lines to the index, and it is looked up by
inode rather than by path because a rolled file keeps its bytes and loses its
name —
resuming per path would re-read everything the rolled file holds. The
``(inode, offset)`` uniqueness on the sample table then makes a re-read
harmless anyway: the bytes come back under a new path, and every line collides
with the sample already stored for that inode and offset. Offset alone would
not do it either, because a path replaced by a new file is a new inode at
offset zero, and its lines must be read.

**A quantity the record does not carry stays absent.** An aggregate over a
measurement no sample in the window carried returns ``None`` rather than
``0.0``, which is the distinction the flat receipt row's recorded nulls
destroyed: a missing reading and a measured zero must not read the same.

**A rewrite is detected by the content of the whole consumed region, not by
length.** A file shorter than the recorded offset was plainly rebuilt in place,
but a rewrite that is not shorter is invisible to a length comparison, and one
that happens to leave the final line intact is invisible to a digest of that
line alone. Either way the index would answer with the values the content it
discarded produced, and the answer would carry nothing that is true of the file.
So the digest of every byte this index has consumed from a file is stored with
its source and re-checked on resume, which catches a change anywhere in the
region already read -- a same-length rewrite, and a rewrite reusing the line
that was last read, included. The check re-reads the consumed region, which is
what a content identity costs; what a resume keeps incremental is the database
work of consuming a line, not the bytes the digest is computed over. A source
with no digest recorded carries no identity to compare against, so its region is
re-read rather than trusted.

**Every key carries the host that recorded the row, because an inode does not
name a machine.** An inode number is unique within one filesystem and nowhere
else, so two hosts recording a file of the same name at the same path produce
the same ``(path, inode)`` and the same ``(inode, offset)`` while holding
different readings -- and a key without a host merges the two irrecoverably,
which is worse than either being wrong. The host is what the row says it was
recorded on (:func:`row_host`), and a row that carries none falls back to the
host this index is reading for -- its own nodename unless one is named --
because a file whose rows do not name their machine is, by construction, being
consumed where it was written.

**Discovery reaches the roll suffixes it intends and nothing else.** A record and
its numbered roll are selected by default; a name that merely contains
``.jsonl`` -- a summary written beside the record, or an archive of it -- is not,
because a file selected by accident is either a parse failure that takes the
whole pass with it or, worse, a roll whose bytes are counted as nothing. A
compressed roll is refused by name with its reason rather than read as text, so a
pass that meets one fails where the operator can see which file caused it.

Aggregation of receipt intervals is not reimplemented here.
:mod:`imas_ambix.agent.receipt_bins` already owns it, and
:meth:`TelemetryIndex.receipt_bins` calls
:func:`~imas_ambix.agent.receipt_bins.summarise_receipt_rows` directly, handing
over rows it already holds. The module's path-reading entry point is a different
caller and is not wired into this index.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import json
import os
import re
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from imas_ambix.agent.receipt_bins import (
    DEFAULT_WIDTH_BINS,
    ReceiptBinReport,
    summarise_receipt_rows,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

#: Record keys a row can name its recording host under, in precedence order.
#: ``host`` is the row-level spelling and ``hostname`` the one the job section
#: uses for the node it queried, so a producer that promotes that reading to the
#: row needs no change here.
_HOST_KEYS = ("host", "hostname")

#: Quantity names whose value is a cumulative total since the engine started,
#: so their period figure is the difference of two endpoints and never a sum.
#: The flat row spells them one way and the canonical engine section another;
#: both spellings are listed, and a name ending in ``_total`` is treated as
#: cumulative as well so a later producer needs no edit here.
CUMULATIVE_MEASUREMENTS: frozenset[str] = frozenset(
    {
        "prefix_cache_queries_total",
        "prefix_cache_hits_total",
        "engine.prompt_tokens",
        "engine.generation_tokens",
        "engine.prefix_cache_queries",
        "engine.prefix_cache_hits",
    }
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS source (
    host     TEXT    NOT NULL,
    path     TEXT    NOT NULL,
    inode    INTEGER NOT NULL,
    size     INTEGER NOT NULL,
    offset   INTEGER NOT NULL,
    rows       INTEGER NOT NULL DEFAULT 0,
    -- digest of the first `offset` bytes of the file; NULL means no identity is
    -- recorded for the consumed region, which makes that region unverifiable
    prefix_sha TEXT,
    PRIMARY KEY (host, path, inode)
);

CREATE TABLE IF NOT EXISTS sample (
    id           INTEGER PRIMARY KEY,
    host         TEXT    NOT NULL,
    inode        INTEGER NOT NULL,
    offset       INTEGER NOT NULL,
    path         TEXT    NOT NULL,
    ts           TEXT,
    ts_epoch     REAL,
    job_id       TEXT,
    profile_slug TEXT,
    served_name  TEXT,
    gpus         INTEGER,
    payload      TEXT    NOT NULL,
    UNIQUE (host, inode, offset)
);

CREATE TABLE IF NOT EXISTS measurement (
    sample_id INTEGER NOT NULL REFERENCES sample(id) ON DELETE CASCADE,
    name      TEXT    NOT NULL,
    kind      TEXT    NOT NULL,
    value     REAL    NOT NULL,
    PRIMARY KEY (sample_id, name)
);

CREATE INDEX IF NOT EXISTS measurement_lookup
    ON measurement (name, kind, sample_id);

CREATE INDEX IF NOT EXISTS sample_time
    ON sample (ts_epoch);
"""


@dataclasses.dataclass(frozen=True)
class IngestReport:
    """What one ingest pass consumed, per source and in total."""

    files_scanned: int
    rows_inserted: int
    rows_duplicate: int
    malformed: int
    bytes_read: int

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def row_host(row: Mapping[str, Any]) -> str | None:
    """The host a record row says it was recorded on, or ``None``.

    Only the row's own top level is read. A host named inside a section is not
    this row's recording host but the scope of the reading that section holds,
    and a record whose job table is refreshed on a slower cadence than its
    other readings would then label its ticks with two different hosts.
    """
    for key in _HOST_KEYS:
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _kind(name: str) -> str:
    """``counter`` for a cumulative total, ``interval`` for everything else."""
    if name in CUMULATIVE_MEASUREMENTS or name.endswith("_total"):
        return "counter"
    return "interval"


def measure_row(row: Mapping[str, Any]) -> dict[str, float]:
    """Every numeric leaf of one record, keyed by its dotted path.

    Nested sections keep their names (``engine.generation_tokens``,
    ``cards.0.utilisation``), so a quantity is addressable without the index
    knowing any producer's schema. Strings, booleans and nulls are not
    measurements and are left out rather than coerced -- a null recorded as a
    zero is the defect this whole spine exists to repair.
    """
    measurements: dict[str, float] = {}
    _walk("", row, measurements)
    return measurements


def _walk(prefix: str, value: Any, out: dict[str, float]) -> None:
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, int | float):
        out[prefix] = float(value)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _walk(f"{prefix}.{key}" if prefix else str(key), item, out)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _walk(f"{prefix}.{index}", item, out)


def _parse_timestamp(value: Any) -> float | None:
    """Epoch seconds from a record's ISO timestamp, or ``None`` if unusable."""
    if not isinstance(value, str) or not value:
        return None
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = _dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_dt.UTC)
    return parsed.timestamp()


_ROLL_SUFFIX = re.compile(r"\.jsonl(?:\.\d+(?:\.gz)?)?\Z")

_COMPRESSED_ROLL = ".gz"


def discover(directory: str | Path, pattern: str | None = None) -> list[Path]:
    """Record files under *directory*, in a stable order.

    Sorted so that two ingests of the same directory agree, which is what lets
    a rebuild be compared against the index it replaces.

    The default rule reaches a rolled file as well as the live one. A roll
    renames the live file to a name carrying the roll suffix -- ``serve.jsonl``
    becomes ``serve.jsonl.1`` -- and keeps writing to the renamed bytes until
    its writer notices, so a rule that matches only a trailing ``.jsonl`` drops
    the tail that landed after the roll. It reaches the roll suffixes and
    nothing else: a name that merely contains ``.jsonl`` is not a record, and
    selecting one either ends the pass on a parse failure or counts a file's
    bytes as nothing.

    A compressed roll is selected so that it is refused by name at ingest
    rather than skipped in silence, and a caller that has deliberately discarded
    the plain bytes of a roll learns at the ingest that it has to be handled.

    *pattern* is an explicit glob for a caller whose records sit under names
    this rule would not select. Prefer the default, because a glob is widened by
    whoever writes it and a widened one selects files that are not records.
    """
    base = Path(directory)
    if pattern is not None:
        return sorted(base.glob(pattern))
    try:
        entries = list(base.iterdir())
    except OSError:
        return []
    return sorted(
        entry
        for entry in entries
        if entry.is_file() and _ROLL_SUFFIX.search(entry.name)
    )


class TelemetryIndex:
    """A local, rebuildable query layer over one or more record files.

    *host* is the host rows that carry none are attributed to, because a record
    whose rows do not name their machine was written where it is being read.
    Every sample and source row records its host, so two machines' readings
    cannot take each other's key.
    """

    def __init__(self, path: str | Path, *, host: str | None = None) -> None:
        self.path = Path(path)
        self.host = host or os.uname().nodename
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> TelemetryIndex:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ── ingest ───────────────────────────────────────────────────────

    def ingest(self, sources: Iterable[str | Path]) -> IngestReport:
        """Consume every byte appended to *sources* since the last pass.

        A source is read from its recorded offset; a file whose consumed region
        no longer carries the bytes this index read there was rebuilt in place,
        so its samples are dropped before it is re-read from zero. Consumption
        stops at the last complete line, so a line still being written is left
        for the next pass rather than parsed half-founded.

        A compressed roll is refused by name rather than read as text: handed to
        the parser it is one long malformed line at best, and it is reported
        with the bytes it holds only where the caller handles it deliberately.
        """
        scanned = inserted = duplicate = malformed = read = 0
        for source in sources:
            path = Path(source)
            try:
                stat = path.stat()
            except OSError:
                continue
            if not path.is_file():
                continue
            if path.name.endswith(_COMPRESSED_ROLL):
                raise ValueError(
                    f"{path} is a compressed roll and this index reads record "
                    "text: decompress it, or give the ingest only the files it "
                    "can read"
                )
            scanned += 1
            lines, tail, prefix_sha, host = self._resume(path, stat)
            added = 0
            carried: str | None = None
            for offset, raw in lines:
                read += len(raw)
                parsed = self._parse(raw, path, offset)
                if parsed is None:
                    malformed += 1
                    continue
                if carried is None:
                    # One file is written by one recorder on one machine, so the
                    # first row that names its host names it for every row here.
                    carried = row_host(parsed)
                if self._insert(path, stat.st_ino, carried or host, offset, parsed):
                    inserted += 1
                    added += 1
                else:
                    duplicate += 1
            file_host = carried or host
            with self._conn:
                self._conn.execute(
                    "INSERT INTO source "
                    "(host, path, inode, size, offset, rows, prefix_sha) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT (host, path, inode) DO UPDATE SET "
                    "  size = excluded.size, offset = excluded.offset, "
                    "  rows = source.rows + excluded.rows, "
                    "  prefix_sha = COALESCE(excluded.prefix_sha, source.prefix_sha)",
                    (
                        file_host,
                        str(path),
                        stat.st_ino,
                        stat.st_size,
                        tail,
                        added,
                        prefix_sha,
                    ),
                )
        with self._conn:
            self._conn.commit()
        return IngestReport(
            files_scanned=scanned,
            rows_inserted=inserted,
            rows_duplicate=duplicate,
            malformed=malformed,
            bytes_read=read,
        )

    def _resumed_row(self, path: Path, inode: int) -> sqlite3.Row | None:
        """The source row this file resumes from, or ``None`` if it has none.

        The file's own path is tried first and the inode alone second, because
        two filesystems number their inodes independently: an inode alone can
        name one file here and a different file on another host, so matching it
        without the path would resume this file from a stranger's offset. A roll
        has no row under its new name, which is what the second lookup is for: it
        finds the row the old name holds, and the bytes keep their offset while
        the name is gone.

        The two lookups are not interchangeable and the caller does not treat
        them so: only a match on the path says *this file* has been consumed
        before, and only that match can conclude the file was rewritten in place.
        A row found by the inode alone that does not describe these bytes belongs
        to another file sharing the number, and reading it as a rewrite would
        discard that other file's samples for content they never held.
        """
        row = self._conn.execute(
            "SELECT host, path, offset, prefix_sha FROM source "
            "WHERE inode = ? AND path = ? LIMIT 1",
            (inode, str(path)),
        ).fetchone()
        if row is not None:
            return row
        return self._conn.execute(
            "SELECT host, path, offset, prefix_sha FROM source WHERE inode = ? "
            "ORDER BY offset DESC LIMIT 1",
            (inode,),
        ).fetchone()

    def _resume(
        self, path: Path, stat: os.stat_result
    ) -> tuple[Iterator[tuple[int, bytes]], int, str | None, str]:
        """Lines to consume, the offset they carry to, the region's digest, host.

        The host returned is the one the file's already-consumed region was
        attributed to, which is the host its samples must be dropped under if
        the region turns out to have been rewritten -- the previous pass's rows
        carry exactly that host, so attributing the drop to any other would
        leave them behind or take another machine's.
        """
        row = self._resumed_row(path, stat.st_ino)
        host = self.host if row is None else row["host"]
        start = 0
        if row is not None and row["offset"]:
            if self._was_rewritten(path, stat, row["offset"], row["prefix_sha"]):
                if row["path"] == str(path):
                    # Rewritten in place: the bytes this index already consumed
                    # no longer describe anything, so every sample from this
                    # inode goes with them before the new content is read.
                    with self._conn:
                        self._conn.execute(
                            "DELETE FROM sample WHERE host = ? AND inode = ?",
                            (host, stat.st_ino),
                        )
                # Otherwise this row was found by the inode alone and does not
                # describe these bytes: it belongs to another file that happens
                # to share the number, so this file is new here. It is read from
                # zero, and the stranger's samples are left where they are --
                # discarding them would destroy content this file never held.
            else:
                start = row["offset"]
        if start >= stat.st_size:
            # Nothing was appended, so the consumed region is the whole file and
            # the digest stored beside it already describes it.
            return iter(()), start, None, host
        with path.open("rb") as handle:
            head = handle.read(start)
            data = handle.read(stat.st_size - start)
        complete = data.rfind(b"\n")
        if complete < 0:
            # Nothing but a partial line so far: leave the offset alone.
            return iter(()), start, None, host
        consumed = head + data[: complete + 1]
        return (
            iter(_lines(consumed[start:], start)),
            start + complete + 1,
            hashlib.sha256(consumed).hexdigest(),
            host,
        )

    def _was_rewritten(
        self, path: Path, stat: os.stat_result, offset: int, prefix_sha: str | None
    ) -> bool:
        """Whether *path* still carries the bytes at *offset* this index read.

        Size is not identity: a file rewritten in place at a length no shorter
        than the recorded offset passes a size comparison while its content is
        entirely different, and the index then answers with the values the
        previous content produced. Nor is the last line identity: a rewrite that
        leaves the line this index read last byte-identical passes a digest of
        that line alone, and every earlier value the index answers with is gone.
        So the digest of the whole consumed region is compared, which means
        re-reading that region. That read is what a content identity over the
        region costs, and the alternative is an answer no byte of the file
        supports.

        A source with no digest recorded has no identity to compare, so its
        region is re-read rather than assumed: unknown is not unchanged.
        """
        if stat.st_size < offset:
            return True
        if prefix_sha is None:
            return True
        return _digest_of(path, offset) != prefix_sha

    def _parse(
        self, raw: bytes, path: Path, offset: int
    ) -> Mapping[str, Any] | None:
        text = raw.decode("utf-8", errors="replace").strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid record JSON at {path}:{offset}"
            ) from error
        if not isinstance(parsed, Mapping):
            raise ValueError(f"record at {path}:{offset} is not an object")
        return parsed

    def _insert(
        self,
        path: Path,
        inode: int,
        host: str,
        offset: int,
        row: Mapping[str, Any],
    ) -> bool:
        """Store one sample; ``False`` when this host, inode and offset are known.

        A sample is identified by where it was recorded as well as by which byte
        of which inode it came from: inode numbers are per-filesystem, so two
        machines recording at the same path produce the same ``(inode, offset)``
        while holding different readings.
        """
        with self._conn:
            cursor = self._conn.execute(
                "INSERT OR IGNORE INTO sample "
                "(host, inode, offset, path, ts, ts_epoch, job_id, profile_slug, "
                " served_name, gpus, payload) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    host,
                    inode,
                    offset,
                    str(path),
                    row.get("timestamp"),
                    _parse_timestamp(row.get("timestamp")),
                    _as_text(row.get("job_id")),
                    _as_text(row.get("profile_slug")),
                    _as_text(row.get("served_name")),
                    _as_int(row.get("gpus")),
                    json.dumps(row, sort_keys=True),
                ),
            )
            if cursor.rowcount == 0:
                return False
            sample_id = cursor.lastrowid
            self._conn.executemany(
                "INSERT OR REPLACE INTO measurement "
                "(sample_id, name, kind, value) VALUES (?, ?, ?, ?)",
                [
                    (sample_id, name, _kind(name), value)
                    for name, value in measure_row(row).items()
                ],
            )
        return True

    def rebuild(self, sources: Iterable[str | Path]) -> IngestReport:
        """Drop every table and ingest from the record again.

        The index holds nothing the record does not, so a rebuild is the whole
        recovery procedure for a corrupted, stale or schema-changed index.
        """
        with self._conn:
            self._conn.executescript(
                "DROP TABLE IF EXISTS measurement;"
                "DROP TABLE IF EXISTS sample;"
                "DROP TABLE IF EXISTS source;"
            )
            self._conn.executescript(_SCHEMA)
        return self.ingest(sources)

    # ── queries ──────────────────────────────────────────────────────

    def sample_count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM sample").fetchone()[0])

    def measurement_count(self, name: str) -> int:
        return int(
            self._conn.execute(
                "SELECT COUNT(*) FROM measurement WHERE name = ?", (name,)
            ).fetchone()[0]
        )

    def sum_measurements(
        self, name: str, start: float, end: float
    ) -> float | None:
        """Total of an interval quantity over ``[start, end)``.

        ``None`` when no sample in the window carried *name* -- which is not
        the same answer as ``0.0``, the total of samples that carried it as
        zero.
        """
        row = self._conn.execute(
            "SELECT SUM(m.value) AS total, COUNT(*) AS n FROM measurement m "
            "JOIN sample s ON s.id = m.sample_id "
            "WHERE m.name = ? AND m.kind = 'interval' "
            "  AND s.ts_epoch >= ? AND s.ts_epoch < ?",
            (name, start, end),
        ).fetchone()
        return None if row["n"] == 0 else float(row["total"])

    def time_weighted_mean(
        self, name: str, start: float, end: float
    ) -> float | None:
        """Mean of a gauge over ``[start, end)``, weighted by time held.

        Each sample stands for the interval running to the next sample of its
        own source, clipped to the window, so a gap in coverage contributes no
        weight rather than pulling the mean toward a stale reading. The final
        sample of a source is extended by the interval that preceded it -- the
        record ends there, so inventing coverage past its last line would be
        reading a value into time that was never observed.
        """
        row = self._conn.execute(
            """
            WITH weighted AS (
                SELECT s.id AS sample_id,
                       MAX(0.0, MIN(
                           COALESCE(
                               LEAD(s.ts_epoch) OVER (
                                   PARTITION BY s.host, s.inode
                                   ORDER BY s.ts_epoch, s.offset
                               ),
                               s.ts_epoch + COALESCE(
                                   s.ts_epoch - LAG(s.ts_epoch) OVER (
                                       PARTITION BY s.host, s.inode
                                       ORDER BY s.ts_epoch, s.offset
                                   ),
                                   0.0
                               )
                           ),
                           ?
                       ) - MAX(s.ts_epoch, ?)) AS weight
                FROM sample s
                WHERE s.ts_epoch IS NOT NULL
            )
            SELECT SUM(m.value * w.weight) AS num, SUM(w.weight) AS den
            FROM measurement m JOIN weighted w ON w.sample_id = m.sample_id
            WHERE m.name = ? AND m.kind = 'interval' AND w.weight > 0
            """,
            (end, start, name),
        ).fetchone()
        total_seconds = row["den"] or 0.0
        if not total_seconds:
            return None
        return float(row["num"]) / float(total_seconds)

    def counter_span(self, name: str, start: float, end: float) -> float | None:
        """Advance of a cumulative quantity across ``[start, end)``.

        The endpoints are the last reading at or before each bound, so a
        window whose own first sample already carries the counter needs no
        earlier row to be meaningful, and a counter that rose and fell in
        between is not turned into a sum of its snapshots.
        """
        opening = self._last_at_or_before(name, start)
        if opening is None:
            # The record begins inside the window: open on its first sample
            # rather than declining the span for want of an earlier reading.
            opening = self._first_in_window(name, start, end)
        closing = self._last_at_or_before(name, end, strict=True)
        if opening is None or closing is None:
            return None
        return closing - opening

    def _last_at_or_before(
        self, name: str, bound: float, *, strict: bool = False
    ) -> float | None:
        comparison = "<" if strict else "<="
        row = self._conn.execute(
            "SELECT m.value AS value FROM measurement m "
            "JOIN sample s ON s.id = m.sample_id "
            f"WHERE m.name = ? AND s.ts_epoch {comparison} ? "
            "ORDER BY s.ts_epoch DESC, s.offset DESC LIMIT 1",
            (name, bound),
        ).fetchone()
        return None if row is None else float(row["value"])

    def _first_in_window(self, name: str, start: float, end: float) -> float | None:
        row = self._conn.execute(
            "SELECT m.value AS value FROM measurement m "
            "JOIN sample s ON s.id = m.sample_id "
            "WHERE m.name = ? AND s.ts_epoch >= ? AND s.ts_epoch < ? "
            "ORDER BY s.ts_epoch, s.offset LIMIT 1",
            (name, start, end),
        ).fetchone()
        return None if row is None else float(row["value"])

    def rows(self, start: float, end: float) -> list[dict[str, Any]]:
        """Records whose timestamp falls in ``[start, end)``, in time order."""
        return [
            json.loads(row["payload"])
            for row in self._conn.execute(
                "SELECT payload FROM sample WHERE ts_epoch >= ? AND ts_epoch < ? "
                "ORDER BY ts_epoch, host, inode, offset",
                (start, end),
            )
        ]

    def receipt_bins(
        self,
        start: float,
        end: float,
        *,
        width_bins: tuple[tuple[int, int], ...] = DEFAULT_WIDTH_BINS,
    ) -> ReceiptBinReport:
        """Width-binned interval summary over the window, via ``receipt_bins``.

        The aggregation belongs to :mod:`imas_ambix.agent.receipt_bins`; this
        only supplies the rows, so the interval statistics have one owner
        whether they are asked of a file or of the index.
        """
        return summarise_receipt_rows(self.rows(start, end), width_bins=width_bins)


def _lines(data: bytes, base: int) -> Iterator[tuple[int, bytes]]:
    """``(absolute offset, line)`` for each newline-terminated line in *data*."""
    offset = base
    for line in data.splitlines(keepends=True):
        yield offset, line
        offset += len(line)


def _digest_of(path: Path, end: int, chunk: int = 1 << 20) -> str | None:
    """Digest of the first *end* bytes of *path*, or None if it cannot be read.

    Read in bounded chunks so the digest of a large consumed region does not
    have to be held in memory to be computed.
    """
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            remaining = end
            while remaining > 0:
                block = handle.read(min(chunk, remaining))
                if not block:
                    break
                digest.update(block)
                remaining -= len(block)
    except OSError:
        return None
    return digest.hexdigest()


def _as_text(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return int(value)


__all__ = [
    "CUMULATIVE_MEASUREMENTS",
    "IngestReport",
    "TelemetryIndex",
    "discover",
    "measure_row",
    "row_host",
]
