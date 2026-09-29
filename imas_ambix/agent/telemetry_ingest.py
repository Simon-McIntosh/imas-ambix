"""A scheduled ingest that owns the index, so a reader only reads.

The durable record is an append-only JSONL file per serve, written on shared
storage by the on-node recorder. The query side is a disposable SQLite index
derived from it (:mod:`imas_ambix.agent.telemetry_index`). This module is the
process that keeps that index current on its own schedule -- one writer, many
readers -- so a reader such as ``agent watch`` is not the process that maintains
the store it queries. A reader that ingested would make the store current only
as often as somebody looked, and would have two readers contend on one file.

**A tick consults the sources' own metadata before it opens anything.** The
record is written continuously, and on a network filesystem re-reading a file
to discover it has not changed costs the whole file. The ingest inside the
index already declines to re-read a source it has consumed to its end whose
length and modification time have not moved, but that decision is only reachable
once the file has been handed to it and its identity looked up. A tick
remembers each source's ``(inode, size, mtime)`` from the previous tick and
hands the index only the sources whose metadata has moved, so an unchanged
source is never opened at all -- which is the cost the schedule is measured in.
The fingerprint is the same metadata the index itself trusts for the same
decision, so a rewrite that leaves both length and modification time untouched
escapes this check exactly as it escapes the index's own; neither the recorder,
a roll nor a compaction can produce one, because each of them writes.

**The tick is a pure step over one pass.** It holds no clock and no loop of its
own, so a test can drive it directly and a caller can wrap it however the
process is scheduled. :func:`run` is that loop: it ticks on a cadence in
seconds, carrying the source fingerprints from one tick to the next, and stops
after *iterations* when a caller asks it to.

**One writer is this module's own property, and it is not yet the system's.**
A tick opens record files for reading only; the index file is the sole thing
written. That is what makes a process started through this module the only
writer of the index it holds, and it is a property of the code rather than a
convention: nothing here opens a record file in a writing mode. It says nothing
about the rest of the system. A reader still ingests the record itself -- the
reader-side change that would stop it is separate work and has not landed -- so a
reader running today writes the same index this loop is meant to own, and the
single-writer guarantee holds only while no reader runs.
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from typing import TYPE_CHECKING, Any

from imas_ambix.agent.telemetry_index import IngestReport, TelemetryIndex, discover

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

#: Default cadence, in seconds, between ticks of the scheduled loop. It matches
#: the recorder's own sampling interval, so the index is at most one recorded
#: row behind the record without the ingest spinning faster than the record
#: grows.
DEFAULT_CADENCE_SECONDS = 5.0


@dataclasses.dataclass(frozen=True)
class SourceFingerprint:
    """The metadata that decides whether a record file needs to be read again.

    ``inode`` names the file behind the path, so a name repointed at a fresh
    file -- a roll that renames the live file and starts a new one -- is read
    rather than mistaken for the bytes the old name held. ``size`` catches an
    append, and ``mtime_ns`` catches a rewrite that does not change the length.
    ``None`` from :meth:`of` means the path is not a readable regular file and
    is not a source this tick should track.
    """

    inode: int
    size: int
    mtime_ns: int

    @classmethod
    def of(cls, path: Path) -> SourceFingerprint | None:
        try:
            stat = path.stat()
        except OSError:
            return None
        if not path.is_file():
            return None
        return cls(inode=stat.st_ino, size=stat.st_size, mtime_ns=stat.st_mtime_ns)


#: A source path's fingerprint, keyed by path. The map is what one tick hands
#: the next, so a caller never has to know how the decision is made.
SourceStates = dict[str, SourceFingerprint]


@dataclasses.dataclass(frozen=True)
class TickReport:
    """What one tick discovered, what it read, and the state it carries forward.

    ``files_read`` counts the sources handed to the index because their metadata
    moved; ``files_skipped`` counts the sources whose fingerprint was unchanged
    and which were therefore never opened. ``sources`` is the fingerprint map a
    caller passes as *previous* to the next tick.
    """

    files_discovered: int
    files_read: int
    files_skipped: int
    ingest: IngestReport
    sources: Mapping[str, SourceFingerprint]

    def to_dict(self) -> dict[str, Any]:
        return {
            "files_discovered": self.files_discovered,
            "files_read": self.files_read,
            "files_skipped": self.files_skipped,
            "ingest": self.ingest.to_dict(),
            "sources": len(self.sources),
        }


def pending_sources(
    directory: str | Path,
    previous: Mapping[str, SourceFingerprint] | None = None,
    *,
    pattern: str | None = None,
) -> tuple[list[Path], SourceStates]:
    """The sources whose metadata moved since *previous*, and the new state.

    A source is pending when its fingerprint differs from the one recorded for
    it, and when it has no recorded fingerprint -- a file this ingest has not
    seen. The returned state carries a fingerprint for every source discovered
    this pass, pending or not, so an unchanged source is carried forward rather
    than dropped and re-read on the next tick.

    *pattern* is passed through to :func:`~imas_ambix.agent.telemetry_index.discover`;
    the default roll-aware rule is the right one for a receipts directory.
    """
    known = previous or {}
    states: SourceStates = {}
    pending: list[Path] = []
    for path in discover(directory, pattern):
        fingerprint = SourceFingerprint.of(path)
        if fingerprint is None:
            continue
        key = str(path)
        states[key] = fingerprint
        if known.get(key) != fingerprint:
            pending.append(path)
    return pending, states


def tick(
    index: TelemetryIndex,
    directory: str | Path,
    *,
    previous: Mapping[str, SourceFingerprint] | None = None,
    pattern: str | None = None,
) -> TickReport:
    """Append the rows new to every changed source under *directory*.

    Only the sources whose fingerprint moved are handed to the index, so a
    second tick over an unchanged record opens nothing. The report's ``sources``
    is what a caller threads into the next tick's *previous*.
    """
    pending, states = pending_sources(directory, previous, pattern=pattern)
    report = index.ingest(pending)
    return TickReport(
        files_discovered=len(states),
        files_read=len(pending),
        files_skipped=len(states) - len(pending),
        ingest=report,
        sources=states,
    )


def run(
    index: TelemetryIndex,
    directory: str | Path,
    *,
    cadence: float = DEFAULT_CADENCE_SECONDS,
    iterations: int | None = None,
    sleep: Callable[[float], None] = time.sleep,
    on_tick: Callable[[TickReport], None] | None = None,
    previous: Mapping[str, SourceFingerprint] | None = None,
    pattern: str | None = None,
) -> list[TickReport]:
    """Tick *index* over *directory* every *cadence* seconds.

    The source fingerprints are carried across ticks here, so the loop is the
    only place the previous-tick state lives; a caller that drives
    :func:`tick` itself threads the same state. A finite count returns the
    reports it collected and never sleeps past the last tick, so a caller asking
    for one tick pays nothing for a cadence it did not use; a caller asking for
    zero ticks gets an empty list and touches nothing. With *iterations*
    ``None`` the loop runs until the process ends, or the injected *sleep*
    raises, which is the scheduled service; it keeps only the latest report, so
    a service that runs for months holds no growing history.

    *sleep* and *on_tick* are injected so a caller -- a test, or a service that
    wants to log each pass -- can supply its own without the loop knowing.
    """
    if cadence <= 0:
        raise ValueError(f"cadence must be positive seconds, not {cadence!r}")
    if iterations is not None and iterations < 0:
        raise ValueError(f"iterations must not be negative, not {iterations!r}")
    if iterations == 0:
        return []
    state = previous
    reports: list[TickReport] = []
    while True:
        report = tick(index, directory, previous=state, pattern=pattern)
        state = report.sources
        if iterations is None:
            # The scheduled service never returns, so a list that grew by one
            # per tick would grow for the life of the process. Rebind so the
            # previous list is dropped for collection; the latest report is all
            # a caller can observe of an endless loop, so it is all that is kept.
            reports = [report]
        else:
            reports.append(report)
        if on_tick is not None:
            on_tick(report)
        if iterations is not None and len(reports) >= iterations:
            break
        sleep(cadence)
    return reports


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Keep the telemetry index current from the recorded receipts"
    )
    parser.add_argument(
        "--record",
        "--record-dir",
        dest="record_dir",
        required=True,
        help="receipts directory to ingest",
    )
    parser.add_argument(
        "--index",
        dest="index_path",
        required=True,
        help="index file to append to",
    )
    parser.add_argument(
        "--cadence",
        type=float,
        default=DEFAULT_CADENCE_SECONDS,
        help="seconds between ticks (default: %(default)s)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="take one tick and exit rather than loop on the cadence",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the scheduled ingest: one tick, or a loop on the cadence.

    This is the entry point the scheduled job runs. It names the index it
    writes explicitly rather than defaulting to a reader's cache path, so the
    service that owns the store is never pointed at the one ``agent watch``
    queries by accident.
    """
    args = _parser().parse_args(argv)
    index = TelemetryIndex(args.index_path)
    try:
        run(
            index,
            args.record_dir,
            cadence=args.cadence,
            iterations=1 if args.once else None,
        )
    finally:
        index.close()
    return 0


__all__ = [
    "DEFAULT_CADENCE_SECONDS",
    "SourceFingerprint",
    "SourceStates",
    "TickReport",
    "main",
    "pending_sources",
    "run",
    "tick",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
