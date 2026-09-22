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
recorded on (:func:`row_host`). A row that names none is stored under an
explicit unknown-host marker (:data:`UNKNOWN_HOST`) and its ``key_kind`` says so
(:data:`UNKNOWN_HOST_SCOPE`), because a record shared across nodes is written on
one machine and read on another: the reading machine's nodename is not evidence
of where a row was written, and a row that adopted it would be keyed under an
identity nothing recorded. A caller that knows the machine passes *host*, and
:func:`receipts_host` resolves it from a receipts file's own job id for the
caller that does not.

**Every key carries the boot as well, because a hostname does not survive a
reboot.** The quantities this record holds are largely cumulative counters, and
every one of them restarts from zero when the machine reboots, so two readings
taken from one hostname on either side of a reboot difference as though the
counter had jumped -- a real measurement, silently wrong, with nothing on either
row to say so. The boot identity (:func:`row_boot_id`) is what separates them:
the same value on two rows means one uninterrupted run of counters, and a
different value means the counters between them are not differenceable at all.
It is resolved per row rather than per file, because one file may hold rows from
both sides of a reboot while its hostname never moves. A boot identity is
validated by :func:`~imas_ambix.agent.throttling.parse_boot_id`, which accepts
only the kernel's canonical lowercase spelling and raises on anything else, and
the refusal does not drop the row: it is keyed by its host alone, and its
``key_kind`` records which of the two keys it got. A row that announces its key
is a different object from one that merely lacks a field -- the reboot guarantee
holds on one side of a comparison and not the other, and a reader can see which.

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
import re
import sqlite3
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from imas_ambix.agent.receipt_bins import (
    DEFAULT_WIDTH_BINS,
    ReceiptBinReport,
    summarise_receipt_rows,
)
from imas_ambix.agent.throttling import parse_boot_id, read_host

if TYPE_CHECKING:
    import os
    from collections.abc import Iterable, Iterator

#: Record keys a row can name its recording host under, in precedence order.
#: ``host`` is the row-level spelling and ``hostname`` the one the job section
#: uses for the node it queried, so a producer that promotes that reading to the
#: row needs no change here.
_HOST_KEYS = ("host", "hostname")

#: The record section a producer states its machine under when it does not state
#: it at the row's top level, and the key it names the boot under.
_HOST_SECTION = "host"
_BOOT_ID_KEY = "boot_id"

#: The column naming which of the two keys a stored row got. It is carried back
#: out to every read surface that returns rows, so a degraded key announces
#: itself to a reader instead of looking like the keyed case.
_KEY_KIND_KEY = "key_kind"

#: Key-scope markers, stored beside every key so a row says which identity it
#: was keyed by. ``BOOT_SCOPE`` means the key carried a boot identity and two
#: such rows may be differenced; ``HOST_SCOPE`` means the boot was unknown, so
#: the row is stored and readable but no reboot guarantee attaches to it;
#: ``UNKNOWN_HOST_SCOPE`` means the record named no machine at all, so not even
#: the host is a recorded identity.
BOOT_SCOPE = "host+boot"
HOST_SCOPE = "host"
UNKNOWN_HOST_SCOPE = "unknown-host"

#: The stored host of a row whose machine is unknown. It is the empty string
#: rather than ``NULL`` for the same reason as the unknown boot below: a
#: ``UNIQUE`` treats NULLs as distinct from each other, so a null host would
#: make the key accept every duplicate instead of refusing the second.
UNKNOWN_HOST = ""

#: The stored boot identity of a row whose boot is unknown. It is the empty
#: string rather than ``NULL`` because ``UNIQUE`` treats NULLs as distinct from
#: each other, which would make the uniqueness guard insert every unkeyed
#: duplicate instead of refusing it.
UNKNOWN_BOOT_ID = ""

#: The job id a receipts filename carries, in the suffix the recorder writes:
#: ``<slug>-<job id>.jsonl``. A roll suffix or a compression suffix trails the
#: ``.jsonl`` and does not move the digits.
_JOB_ID_IN_NAME = re.compile(r"-(\d+)\.jsonl(?:\.\d+)?(?:\.gz)?\Z")

#: What ``sacct`` prints in place of a node name when the field carries none.
#: These are words rather than machines, and ``scontrol`` echoes each one back
#: as though it had resolved it -- one line, exit zero -- so a placeholder must
#: be refused by name and not by shape.
_UNRESOLVED_NODE_NAMES = frozenset({"none", "unknown", "n/a", "null"})

#: The shape of one node name: a single unpunctuated token carrying no hostlist
#: syntax and no whitespace. An unexpanded hostlist token carries ``[``, ``]``,
#: ``,`` or ``+``, and a report naming several nodes carries spaces, so a value
#: matching this is one machine's name rather than a list or its truncation.
_NODE_NAME = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]*\Z")

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
    boot_id  TEXT    NOT NULL,
    key_kind TEXT    NOT NULL,
    path     TEXT    NOT NULL,
    inode    INTEGER NOT NULL,
    size     INTEGER NOT NULL,
    offset   INTEGER NOT NULL,
    rows       INTEGER NOT NULL DEFAULT 0,
    -- digest of the first `offset` bytes of the file; NULL means no identity is
    -- recorded for the consumed region, which makes that region unverifiable
    prefix_sha TEXT,
    PRIMARY KEY (host, boot_id, path, inode)
);

CREATE TABLE IF NOT EXISTS sample (
    id           INTEGER PRIMARY KEY,
    host         TEXT    NOT NULL,
    boot_id      TEXT    NOT NULL,
    key_kind     TEXT    NOT NULL,
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
    UNIQUE (host, boot_id, inode, offset)
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


def _section_text(row: Mapping[str, Any], name: str) -> str | None:
    """*name* from the row's own ``host`` section, or ``None``.

    A producer that states its machine in a section of its own keeps the row's
    top-level keys for the readings, and the section ``host`` is exactly the
    machine the reading came from. Any other section is not read: a host named
    inside one describes the scope of the reading that section holds, not the
    machine the row was recorded on.
    """
    section = row.get(_HOST_SECTION)
    if not isinstance(section, Mapping):
        return None
    value = section.get(name)
    return value if isinstance(value, str) and value else None


def row_host(row: Mapping[str, Any]) -> str | None:
    """The host a record row says it was recorded on, or ``None``.

    The row's own top level is read first, then a ``host`` section of its own.
    A host named inside any other section is not this row's recording host but
    the scope of the reading that section holds, and a record whose job table is
    refreshed on a slower cadence than its other readings would then label its
    ticks with two different hosts.
    """
    for key in _HOST_KEYS:
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    return _section_text(row, "hostname")


def row_boot_id(row: Mapping[str, Any]) -> str | None:
    """The boot identity a record row says it was recorded on, or ``None``.

    Read from the row's own top level, then from its own ``host`` section, on
    the same rule as :func:`row_host`. The value is not judged here: a producer's
    spelling is validated where it is used as a key, by the same parser the
    producer validated it with.
    """
    value = row.get(_BOOT_ID_KEY)
    if isinstance(value, str) and value:
        return value
    return _section_text(row, _BOOT_ID_KEY)


def resolve_boot_id(candidate: str | None) -> tuple[str, str]:
    """``(stored boot identity, key scope)`` for a candidate boot identity.

    A canonical boot identifier is stored as itself and the scope says the key
    carried it. Anything the producer's parser refuses -- a malformed string, an
    empty one, an uppercase spelling of a canonical identifier -- is not used as
    a key and the scope says so; the row is still stored, under the same empty
    sentinel for every unknown boot so that its own uniqueness still holds.
    Never a refusal to store: a row whose boot cannot be established is a row to
    read with a caveat, not a row to drop.
    """
    if candidate is not None:
        try:
            return parse_boot_id(candidate), BOOT_SCOPE
        except ValueError:
            pass
    return UNKNOWN_BOOT_ID, HOST_SCOPE


def resolve_host(candidate: str | None) -> str:
    """The host a row is keyed by: the stated one, or the unknown-host marker.

    Never the reading machine's own name. A record whose rows state no host has
    not said where it was written -- a receipts directory on shared storage
    collects files written on one node and read on another -- so keying such a
    row under the reader's nodename asserts a machine nothing recorded. The
    empty-string marker is stored rather than ``NULL`` so the key's uniqueness
    still refuses a second unkeyed row, on the same rule as the unknown boot.
    """
    return candidate if candidate else UNKNOWN_HOST


def key_scope(host: str, boot_scope: str) -> str:
    """The ``key_kind`` a row keyed on *host* and *boot_scope* is stored with.

    The host is the outer dimension: a row with no host of its own is announced
    as unknown-host whatever its boot resolved to, because a boot identity says
    which uninterrupted run of counters a machine accumulated, and a record that
    never named the machine has no such run to point at.
    """
    return UNKNOWN_HOST_SCOPE if host == UNKNOWN_HOST else boot_scope


def receipts_job_id(path: str | Path) -> str | None:
    """The SLURM job id a receipts filename names, or ``None``.

    The on-node recorder writes ``<slug>-<job id>.jsonl``, so the job that
    produced a file -- and therefore the node that ran the serve -- is
    recoverable from the name alone, before any row states it.
    """
    match = _JOB_ID_IN_NAME.search(Path(path).name)
    return None if match is None else match.group(1)


def receipts_host(path: str | Path) -> str:
    """The node a receipts file was written on, from the job id in its name.

    The recorder runs inside the allocation, so the job's node list is the
    machine its rows came from. Two reads are needed because the scheduler
    reports an allocation's nodes as a single compressed hostlist token -- a
    fourteen-node job is ``98dci4-clu-[5073-5086]``, one whitespace-delimited
    field that names fourteen machines -- and because ``sacct`` truncates that
    field at its default column width, so the same job also reads
    ``98dci4-clu-[50+``. Counting fields therefore cannot separate one node from
    many, and a truncated token is not a hostname at all. ``sacct`` is asked for
    parsable output so the value is never cut short, and ``scontrol show
    hostnames`` expands the list to one node per line, which is the count that
    means what it says.

    The expansion must also be a plausible node name rather than merely a single
    line, because ``scontrol`` echoes a placeholder it did not resolve instead
    of refusing it: handed the word ``None`` it prints that word back, one line,
    exit zero. A name carrying hostlist syntax or whitespace is a token that was
    echoed unexpanded rather than parsed, and the words ``sacct`` prints for an
    absent field name no machine at all.

    The lookup is deliberately allowed to fail: a job still running, purged from
    ``sacct``, or spanning more than one node resolves to nothing usable, and a
    guess here is a key silently asserting a machine no row recorded. Refusing
    is the only safe answer, because the caller can name *host* explicitly or
    leave the row under the unknown-host marker. An unavailable ``sacct`` or
    ``scontrol`` refuses the same way, so a caller catching ``ValueError`` for
    an unresolvable host never meets an ``OSError`` instead.
    """
    job_id = receipts_job_id(path)
    if job_id is None:
        raise ValueError(
            f"{path} does not name a job id, so the host it was written on "
            "cannot be resolved from its name"
        )
    try:
        completed = subprocess.run(
            ["sacct", "-j", job_id, "-X", "-n", "-P", "-o", "NodeList"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as error:
        raise ValueError(
            f"could not resolve the host of {path}: sacct is unavailable"
        ) from error
    hostlists = [
        line for line in completed.stdout.splitlines() if line.strip()
    ]
    if completed.returncode != 0 or len(hostlists) != 1:
        raise ValueError(
            f"job {job_id} ({path}) did not resolve to one node list: "
            f"{', '.join(hostlists) or completed.stderr.strip() or 'no node reported'}"
        )
    try:
        expanded = subprocess.run(
            ["scontrol", "show", "hostnames", hostlists[0].strip()],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as error:
        raise ValueError(
            f"could not resolve the host of {path}: scontrol is unavailable"
        ) from error
    nodes = [line.strip() for line in expanded.stdout.splitlines() if line.strip()]
    if (
        expanded.returncode != 0
        or len(nodes) != 1
        or nodes[0].lower() in _UNRESOLVED_NODE_NAMES
        or _NODE_NAME.match(nodes[0]) is None
    ):
        raise ValueError(
            f"job {job_id} ({path}) did not resolve to one node: "
            f"{', '.join(nodes) or expanded.stderr.strip() or 'no node reported'}"
        )
    return nodes[0]


def local_boot_id() -> str | None:
    """The boot identity of the machine this process is on, or ``None``.

    Read through the producer's own reader, so a boot identifier means one thing
    in the record and in whatever consumes it. ``None`` covers a machine without
    one to read, which is a state a caller keys around rather than raises on.
    """
    try:
        return read_host().boot_id
    except (OSError, ValueError):
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

    *host* is the host rows that carry none are attributed to, and *boot_id*
    the boot. Neither is inferred from the machine doing the reading: a record
    whose rows name no machine is keyed under the unknown-host marker and read
    with that caveat, because a receipts file on shared storage is routinely
    written on the node that ran the serve and read somewhere else. A caller
    that knows where the rows were written passes both, and
    :func:`receipts_host` resolves the host from a receipts file's own job id
    for the caller that does not. Every sample and source row records its host,
    its boot identity where one could be established, and which of the keys it
    got, so two machines -- or two boots of one machine -- cannot take each
    other's key, and a row keyed on neither announces itself as such.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        host: str | None = None,
        boot_id: str | None = None,
    ) -> None:
        self.path = Path(path)
        self.host = resolve_host(host)
        self.boot_id = UNKNOWN_BOOT_ID if boot_id is None else boot_id
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
            lines, tail, prefix_sha, source_host, source_boot = self._resume(
                path, stat
            )
            added = 0
            carried: str | None = None
            # The boot the file's consumed region was keyed by, if the previous
            # pass established one: a row that names no boot of its own belongs
            # to the boot that was writing this file, not to this reader's.
            carried_boot: str | None = source_boot
            file_host = resolve_host(source_host)
            file_boot, file_scope = resolve_boot_id(source_boot or self.boot_id)
            file_kind = key_scope(file_host, file_scope)
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
                own_boot = row_boot_id(parsed)
                if own_boot is not None:
                    # Resolved per row, unlike the host: a reboot inside one file
                    # leaves the hostname alone and changes only this.
                    carried_boot = own_boot
                file_host = resolve_host(carried or source_host)
                file_boot, file_scope = resolve_boot_id(
                    carried_boot if carried_boot is not None else self.boot_id
                )
                file_kind = key_scope(file_host, file_scope)
                if self._insert(
                    path,
                    stat.st_ino,
                    file_host,
                    file_boot,
                    file_kind,
                    offset,
                    parsed,
                ):
                    inserted += 1
                    added += 1
                else:
                    duplicate += 1
            with self._conn:
                self._conn.execute(
                    "INSERT INTO source "
                    "(host, boot_id, key_kind, path, inode, size, offset, rows, "
                    " prefix_sha) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT (host, boot_id, path, inode) DO UPDATE SET "
                    "  size = excluded.size, offset = excluded.offset, "
                    "  rows = source.rows + excluded.rows, "
                    "  prefix_sha = COALESCE(excluded.prefix_sha, source.prefix_sha)",
                    (
                        file_host,
                        file_boot,
                        file_kind,
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

        The boot is not part of these lookups even though it is part of the
        source key. This is the one place the boot cannot be known before the
        file is read, and a lookup that required it would miss a record written
        under a different boot -- or read on another machine -- and re-read the
        whole file as new. The highest offset for the file is taken instead,
        which is the resume point whichever boot wrote last.
        """
        row = self._conn.execute(
            "SELECT host, boot_id, path, offset, prefix_sha FROM source "
            "WHERE inode = ? AND path = ? ORDER BY offset DESC LIMIT 1",
            (inode, str(path)),
        ).fetchone()
        if row is not None:
            return row
        return self._conn.execute(
            "SELECT host, boot_id, path, offset, prefix_sha FROM source "
            "WHERE inode = ? ORDER BY offset DESC LIMIT 1",
            (inode,),
        ).fetchone()

    def _resume(
        self, path: Path, stat: os.stat_result
    ) -> tuple[Iterator[tuple[int, bytes]], int, str | None, str, str | None]:
        """Lines to consume, the offset they carry to, digest, host, boot.

        The host returned is the one the file's already-consumed region was
        attributed to, which is the host its samples must be dropped under if
        the region turns out to have been rewritten -- the previous pass's rows
        carry exactly that host, so attributing the drop to any other would
        leave them behind or take another machine's. The boot returned is the
        one that region was keyed by, which is where a row naming no boot of its
        own is attributed; ``None`` means the file is new here, so this index's
        own *boot_id* applies -- the caller's, or the unknown-boot marker when
        none was given.
        """
        row = self._resumed_row(path, stat.st_ino)
        host = self.host if row is None else row["host"]
        boot = None if row is None else row["boot_id"]
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
            return iter(()), start, None, host, boot
        with path.open("rb") as handle:
            head = handle.read(start)
            data = handle.read(stat.st_size - start)
        complete = data.rfind(b"\n")
        if complete < 0:
            # Nothing but a partial line so far: leave the offset alone.
            return iter(()), start, None, host, boot
        consumed = head + data[: complete + 1]
        return (
            iter(_lines(consumed[start:], start)),
            start + complete + 1,
            hashlib.sha256(consumed).hexdigest(),
            host,
            boot,
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
        boot_id: str,
        key_kind: str,
        offset: int,
        row: Mapping[str, Any],
    ) -> bool:
        """Store one sample; ``False`` when this key is already known.

        A sample is identified by where it was recorded as well as by which byte
        of which inode it came from, and *where* takes two forms: inode numbers
        are per-filesystem, so two machines recording at the same path produce
        the same ``(inode, offset)`` while holding different readings, and the
        counters are cumulative since a boot, so two readings of one machine on
        either side of a reboot. The boot is empty when none could be
        established, and it is empty rather than null so that the uniqueness
        below still refuses a duplicate unkeyed row.
        """
        with self._conn:
            cursor = self._conn.execute(
                "INSERT OR IGNORE INTO sample "
                "(host, boot_id, key_kind, inode, offset, path, ts, ts_epoch, "
                " job_id, profile_slug, served_name, gpus, payload) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    host,
                    boot_id,
                    key_kind,
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
        reading a value into time that was never observed. A source is one
        machine's file over one boot, so the gap a reboot leaves is a gap in
        coverage and not an interval one sample was held across.
        """
        row = self._conn.execute(
            """
            WITH weighted AS (
                SELECT s.id AS sample_id,
                       MAX(0.0, MIN(
                           COALESCE(
                               LEAD(s.ts_epoch) OVER (
                                   PARTITION BY s.host, s.boot_id, s.inode
                                   ORDER BY s.ts_epoch, s.offset
                               ),
                               s.ts_epoch + COALESCE(
                                   s.ts_epoch - LAG(s.ts_epoch) OVER (
                                       PARTITION BY s.host, s.boot_id, s.inode
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

        A cumulative counter is one machine's over one boot: a reboot resets it
        and no other machine ever advanced it. So the two endpoints must belong
        to one ``(host, boot)``; where they do not, the span is declined rather
        than differenced, because their difference is a number no machine ever
        advanced and it reads exactly like a measurement.
        """
        opening = self._last_at_or_before(name, start)
        if opening is None:
            # The record begins inside the window: open on its first sample
            # rather than declining the span for want of an earlier reading.
            opening = self._first_in_window(name, start, end)
        closing = self._last_at_or_before(name, end, strict=True)
        if opening is None or closing is None:
            return None
        if opening[:2] != closing[:2]:
            return None
        return closing[2] - opening[2]

    def _last_at_or_before(
        self, name: str, bound: float, *, strict: bool = False
    ) -> tuple[str, str, float] | None:
        """``(host, boot_id, value)`` of the latest reading at or before *bound*."""
        comparison = "<" if strict else "<="
        row = self._conn.execute(
            "SELECT s.host AS host, s.boot_id AS boot_id, m.value AS value "
            "FROM measurement m JOIN sample s ON s.id = m.sample_id "
            f"WHERE m.name = ? AND s.ts_epoch {comparison} ? "
            "ORDER BY s.ts_epoch DESC, s.offset DESC, s.host, s.boot_id LIMIT 1",
            (name, bound),
        ).fetchone()
        return None if row is None else self._keyed_value(row)

    def _first_in_window(
        self, name: str, start: float, end: float
    ) -> tuple[str, str, float] | None:
        """``(host, boot_id, value)`` of the earliest reading inside the window."""
        row = self._conn.execute(
            "SELECT s.host AS host, s.boot_id AS boot_id, m.value AS value "
            "FROM measurement m JOIN sample s ON s.id = m.sample_id "
            "WHERE m.name = ? AND s.ts_epoch >= ? AND s.ts_epoch < ? "
            "ORDER BY s.ts_epoch, s.offset, s.host, s.boot_id LIMIT 1",
            (name, start, end),
        ).fetchone()
        return None if row is None else self._keyed_value(row)

    @staticmethod
    def _keyed_value(row: sqlite3.Row) -> tuple[str, str, float]:
        return (row["host"], row["boot_id"], float(row["value"]))

    def rows(self, start: float, end: float) -> list[dict[str, Any]]:
        """Records whose timestamp falls in ``[start, end)``, in time order.

        Each record carries the key it was stored under -- ``host`` and
        ``boot_id`` -- together with ``key_kind``, so a reader grouping rows by
        ``(host, boot)`` groups them exactly as the store did and a row whose
        boot was refused is visible as the degraded case it is.

        The record's own spelling of those fields is replaced rather than
        passed through: a boot the producer's parser refused is not the
        identity any lookup matched on, and presented as one it makes the
        reboot guarantee look like it holds on a row where it does not. The
        spelling is not lost, because it is in the record itself -- the file
        this index is derived from and can be rebuilt from at any time.
        """
        return [
            self._keyed_row(row)
            for row in self._conn.execute(
                "SELECT host, boot_id, key_kind, payload FROM sample "
                "WHERE ts_epoch >= ? AND ts_epoch < ? "
                "ORDER BY ts_epoch, host, boot_id, inode, offset",
                (start, end),
            )
        ]

    @staticmethod
    def _keyed_row(row: sqlite3.Row) -> dict[str, Any]:
        """The stored record, stamped with the key it was stored under."""
        record = json.loads(row["payload"])
        record[_HOST_SECTION] = row["host"]
        record[_BOOT_ID_KEY] = row["boot_id"]
        record[_KEY_KIND_KEY] = row["key_kind"]
        return record

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
    "BOOT_SCOPE",
    "CUMULATIVE_MEASUREMENTS",
    "HOST_SCOPE",
    "UNKNOWN_BOOT_ID",
    "UNKNOWN_HOST",
    "UNKNOWN_HOST_SCOPE",
    "IngestReport",
    "TelemetryIndex",
    "discover",
    "key_scope",
    "local_boot_id",
    "measure_row",
    "receipts_host",
    "receipts_job_id",
    "resolve_boot_id",
    "resolve_host",
    "row_boot_id",
    "row_host",
]
