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

**Tail state is kept per path, and content identity is per inode and byte
offset.** The two are different jobs. The path-keyed row records how far this
index has consumed each file, so ordinary appends cost only the new bytes. The
``(inode, offset)`` uniqueness on the sample table is what makes a re-read
harmless: a rolled file is rediscovered under a new path, its lines are read
from the start, and every one of them collides with the sample already stored
under that inode and offset, so nothing is counted twice and nothing that was
appended after the roll is missed. Offset alone would not do it — a path that
is replaced by a new file is a new inode at offset zero — and path alone would
not do it either, because the rolled file keeps its bytes and loses its name.

**A quantity the record does not carry stays absent.** An aggregate over a
measurement no sample in the window carried returns ``None`` rather than
``0.0``, which is the distinction the flat receipt row's recorded nulls
destroyed: a missing reading and a measured zero must not read the same.

Aggregation of receipt intervals is not reimplemented here.
:mod:`imas_ambix.agent.receipt_bins` already owns it, and
:meth:`TelemetryIndex.receipt_bins` is the production caller it was written
for, reconstructing rows from the stored payload and handing them over
unchanged.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
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
    import os
    from collections.abc import Iterable, Iterator

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
    path   TEXT    NOT NULL,
    inode  INTEGER NOT NULL,
    size   INTEGER NOT NULL,
    offset INTEGER NOT NULL,
    rows   INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (path, inode)
);

CREATE TABLE IF NOT EXISTS sample (
    id           INTEGER PRIMARY KEY,
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
    UNIQUE (inode, offset)
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


def discover(directory: str | Path, pattern: str = "*.jsonl") -> list[Path]:
    """Record files under *directory*, in a stable order.

    Sorted so that two ingests of the same directory agree, which is what lets
    a rebuild be compared against the index it replaces.
    """
    return sorted(Path(directory).glob(pattern))


class TelemetryIndex:
    """A local, rebuildable query layer over one or more record files."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
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

        A source is read from its recorded offset; a file shorter than that
        offset was rebuilt in place, so its samples from the new end onward are
        dropped before it is re-read from zero. Consumption stops at the last
        complete line, so a line still being written is left for the next pass
        rather than parsed half-founded.
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
            scanned += 1
            lines, tail = self._resume(path, stat)
            added = 0
            for offset, raw in lines:
                read += len(raw)
                parsed = self._parse(raw, path, offset)
                if parsed is None:
                    malformed += 1
                    continue
                if self._insert(path, stat.st_ino, offset, parsed):
                    inserted += 1
                    added += 1
                else:
                    duplicate += 1
            with self._conn:
                self._conn.execute(
                    "INSERT INTO source (path, inode, size, offset, rows) "
                    "VALUES (?, ?, ?, ?, ?) "
                    "ON CONFLICT (path, inode) DO UPDATE SET "
                    "  size = excluded.size, offset = excluded.offset, "
                    "  rows = source.rows + excluded.rows",
                    (str(path), stat.st_ino, stat.st_size, tail, added),
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

    def _resume(
        self, path: Path, stat: os.stat_result
    ) -> tuple[Iterator[tuple[int, bytes]], int]:
        """Complete lines to consume and the offset they carry the file to."""
        row = self._conn.execute(
            "SELECT offset FROM source WHERE path = ? AND inode = ?",
            (str(path), stat.st_ino),
        ).fetchone()
        start = 0
        if row is not None:
            if stat.st_size < row["offset"]:
                # Rewritten in place: the bytes past the new end no longer
                # describe anything, so their samples go with them.
                with self._conn:
                    self._conn.execute(
                        "DELETE FROM sample WHERE inode = ? AND offset >= ?",
                        (stat.st_ino, stat.st_size),
                    )
            else:
                start = row["offset"]
        if start >= stat.st_size:
            return iter(()), start
        with path.open("rb") as handle:
            handle.seek(start)
            data = handle.read(stat.st_size - start)
        complete = data.rfind(b"\n")
        if complete < 0:
            # Nothing but a partial line so far: leave the offset alone.
            return iter(()), start
        end = start + complete + 1
        return iter(_lines(data[: complete + 1], start)), end

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
        self, path: Path, inode: int, offset: int, row: Mapping[str, Any]
    ) -> bool:
        """Store one sample; ``False`` when this inode and offset are known."""
        with self._conn:
            cursor = self._conn.execute(
                "INSERT OR IGNORE INTO sample "
                "(inode, offset, path, ts, ts_epoch, job_id, profile_slug, "
                " served_name, gpus, payload) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
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
                                   PARTITION BY s.inode
                                   ORDER BY s.ts_epoch, s.offset
                               ),
                               s.ts_epoch + COALESCE(
                                   s.ts_epoch - LAG(s.ts_epoch) OVER (
                                       PARTITION BY s.inode
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
                "ORDER BY ts_epoch, inode, offset",
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
]
