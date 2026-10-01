"""Retire raw serving receipts once their compacted siblings are proven equal.

A raw receipt file grows about 8.7 MB a day and never rolls, so the record
outlives any single serve by a long way. Once its minute and hour compactions
reproduce the raw file's own totals window by window, the raw bytes are the only
copy of nothing the coarser tiers do not already carry, and the file may be
retired.

Deletion cannot be undone, so the retirement is split in two: a listing pass
that writes a manifest of the files that qualify, and an apply pass that deletes
exactly the manifest's files and nothing else. The manifest carries each file's
digest and its siblings' digests, so the apply pass deletes a file only while it
still holds the bytes the listing saw. A file that has grown, been rewritten or
whose compaction no longer agrees is skipped and reported rather than removed.

A raw file is named ``<profile slug>-<job id>.jsonl``; its compacted siblings
are the same stem with a ``.minute`` and an ``.hour`` tier inserted before the
suffix. A file qualifies only where every one of these holds:

* its first parseable row's ``job_id`` equals the job id its name claims, so a
  name alone never makes a file a raw receipt -- a router request rotation such
  as ``requests-2026-10-01.jsonl`` has the raw name shape and carries no such
  row;
* its job appears in no ``squeue`` state for the account, so a running serve's
  growing file is never a candidate;
* it was not modified in the last day, because a serve the scheduler cannot see
  may still be flushing its tail;
* its last parseable row is more than ``--older-than-days`` days old, measured
  from the row's own timestamp rather than the file's modification time;
* both compacted siblings exist and, for every UTC hour the job spans and for
  each cumulative token counter, the partitioned totals over raw, over minute
  and over hour are equal exactly. The partition uses each tier's own opening
  block, so a serve restart inside a window is totalled run by run in every
  tier, and one counter in one window that disagrees keeps the raw file.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from imas_ambix.agent.serving_receipts import tier_paths
from imas_ambix.agent.telemetry_index import TelemetryIndex, receipts_job_id

#: Where a listing pass writes the manifest an apply pass consumes. Kept under
#: the user's state directory; the manifest is the frozen list and a durable
#: record of what a retirement considered.
RETENTION_MANIFEST_DIR = Path.home() / ".local" / "state" / "ambix" / "retention"

#: A raw file's last row must be older than this many days by default.
DEFAULT_OLDER_THAN_DAYS = 14.0

#: A file written by a job the scheduler cannot see is still ineligible until it
#: has been quiet this long, because a serve whose allocation has ended may
#: still be flushing its tail.
QUIET_SECONDS = 24 * 60 * 60

#: The engine-section counters that must agree across the three tiers. Their
#: canonical spelling is what the receipts row records them under, so the three
#: partition through one name each in every tier.
COUNTER_NAMES: tuple[str, ...] = (
    "engine.generation_tokens",
    "engine.prompt_tokens",
    "engine.uncached_prompt_tokens",
)

#: The markers a compacted tier file carries in its name. A raw file never
#: carries either, so a name carrying one is never a retirement candidate
#: however it is spelled, and neither is a name that resolves to no job id.
_TIER_MARKERS = (".minute.", ".hour.")

RunningProvider = Callable[[], frozenset[str]]


def default_running_job_ids() -> frozenset[str]:
    """Every job id the scheduler currently reports for this account.

    One ``squeue`` for the account answers the liveness question for every
    candidate at once, so a listing does not ask per file. Only a scheduler that
    ran and reported nothing answers with an empty set; a failure to run raises,
    because an absent reading is not an empty one and reading it as empty would
    make every running serve look ended.
    """
    try:
        completed = subprocess.run(
            ["squeue", "-h", "-o", "%i", "-u", os.environ.get("USER", "")],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:  # pragma: no cover - off-cluster only
        raise RuntimeError(
            "squeue is unavailable, so a live job cannot be told from an ended one"
        ) from exc
    if completed.returncode != 0:
        raise RuntimeError(
            f"squeue failed with status {completed.returncode}: "
            f"{completed.stderr.strip()}"
        )
    ids: set[str] = set()
    for line in completed.stdout.splitlines():
        token = line.strip().split()[0] if line.strip() else ""
        if token.isdigit():
            ids.add(token)
    return frozenset(ids)


@dataclass(frozen=True)
class CounterComparison:
    """One counter's totals per tier and whether the three agree.

    ``raw``, ``minute`` and ``hour`` are each the sum, over every window the job
    spans, of that tier's partitioned total for the counter. ``agrees`` is
    decided window by window, before the sums are taken, so a counter that
    disagrees in one window cannot be masked by another window compensating it.
    """

    name: str
    raw: float | None
    minute: float | None
    hour: float | None
    agrees: bool


@dataclass(frozen=True)
class FileVerdict:
    """What a listing pass concluded about one raw receipt file."""

    path: Path
    job_id: str | None
    size: int
    last_row_time: float | None
    eligible: bool
    reason: str
    counters: tuple[CounterComparison, ...] = ()


@dataclass(frozen=True)
class ScanResult:
    """The whole listing: every candidate file and its verdict."""

    record_dir: Path
    older_than_days: float
    now: float
    files: tuple[FileVerdict, ...]

    @property
    def eligible(self) -> tuple[FileVerdict, ...]:
        return tuple(verdict for verdict in self.files if verdict.eligible)


@dataclass(frozen=True)
class ApplyReport:
    """What an apply pass removed and what it left, with the reason."""

    manifest_path: Path
    removed: tuple[Path, ...] = ()
    skipped: tuple[tuple[Path, str], ...] = ()


def _sha256(path: Path) -> str | None:
    """The hex digest of a file's bytes, or ``None`` if it cannot be read."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _timestamp_of(row: object) -> float | None:
    """Epoch seconds from a row's ISO timestamp, or ``None`` if unusable."""
    if not isinstance(row, dict):
        return None
    value = row.get("timestamp")
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


def row_span(path: Path) -> tuple[float | None, float | None]:
    """The earliest and latest timestamps among a file's parseable rows.

    Read tolerantly line by line: a trailing line still being written is not a
    row, and a file whose last complete write landed mid-line must still yield
    the rows that completed before it rather than refuse the whole file. A file
    with no parseable row answers ``(None, None)``.
    """
    first: float | None = None
    last: float | None = None
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                stamp = _timestamp_of(row)
                if stamp is None:
                    continue
                if first is None or stamp < first:
                    first = stamp
                if last is None or stamp > last:
                    last = stamp
    except OSError:
        return (None, None)
    return (first, last)


def _iso(stamp: float | None) -> str | None:
    """An epoch time as a retirement manifest spells it, or ``None``."""
    if stamp is None:
        return None
    return _dt.datetime.fromtimestamp(stamp, tz=_dt.UTC).isoformat()


def _first_row(path: Path) -> dict | None:
    """The first parseable JSON object in *path*, or ``None``.

    Read tolerantly line by line, as :func:`row_span` is: a trailing line still
    being written is not a row, and a blank or malformed line before the data
    must not refuse the whole file. The first line that parses to an object is
    the file's opening row, which carries the identity a raw receipt is judged
    on.
    """
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    return row
    except OSError:
        return None
    return None


def is_raw_receipt(path: str | Path) -> bool:
    """Whether *path* is a raw receipt file rather than a tier or a request log.

    A raw file is the ``<slug>-<job id>.jsonl`` the recorder appends to, and in a
    raw file that job id is also the one its first parseable row carries. A
    compacted tier carries a ``.minute`` or ``.hour`` marker, and a router
    request file carries no such row however its name reads: a rotation such as
    ``requests-2026-10-01.jsonl`` has the raw name shape and its ``01`` parses
    as a job id, so a name alone would admit it. Neither is ever a retirement
    candidate, so the guard reads the opening row and refuses whatever does not
    carry the id its name claims, rather than trusting the manifest.
    """
    name = Path(path).name
    if any(marker in name for marker in _TIER_MARKERS):
        return False
    job = receipts_job_id(path)
    if job is None:
        return False
    row = _first_row(Path(path))
    if row is None:
        return False
    return str(row.get("job_id")) == job


def raw_receipt_files(record_dir: str | Path) -> list[Path]:
    """Every raw receipt file under *record_dir*, in a stable order."""
    directory = Path(record_dir)
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.glob("*.jsonl") if is_raw_receipt(path))


def _hour_windows(first: float, last: float) -> list[tuple[float, float]]:
    """Every UTC hour the span ``[first, last]`` touches, in order.

    A window is ``[start, start + 3600)`` aligned to the epoch, so the same hour
    boundary names the same window for every tier and every run.
    """
    start = int(first // 3600) * 3600
    windows: list[tuple[float, float]] = []
    while start <= last:
        windows.append((float(start), float(start) + 3600.0))
        start += 3600
    return windows


def _sum_or_none(values: list[float | None]) -> float | None:
    """The sum of a tier's per-window totals, or ``None`` if any is missing.

    A window a tier cannot answer is not a zero of that tier: it is a window the
    tier does not cover, and the absence must survive to the comparison rather
    than be summed away.
    """
    if any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


def compare_tiers(
    raw_path: Path,
    minute_path: Path,
    hour_path: Path,
    windows: list[tuple[float, float]],
) -> tuple[CounterComparison, ...]:
    """Per-counter agreement between the raw file and its two compactions.

    Each tier is read through the same partitioned-total query the record
    warehouse answers with, so the triple is compared on the one figure whose
    equality licenses deletion. The comparison is per window and per counter:
    every window must agree for every counter, or the tuple carries a
    ``CounterComparison`` whose ``agrees`` is false.
    """
    with tempfile.TemporaryDirectory(prefix="ambix-retention-") as scratch:
        scratch_path = Path(scratch)
        raw_index = TelemetryIndex(scratch_path / "raw.sqlite3")
        minute_index = TelemetryIndex(scratch_path / "minute.sqlite3")
        hour_index = TelemetryIndex(scratch_path / "hour.sqlite3")
        try:
            raw_index.ingest([raw_path])
            minute_index.ingest([minute_path])
            hour_index.ingest([hour_path])
            comparisons: list[CounterComparison] = []
            for name in COUNTER_NAMES:
                raw_by_window: list[float | None] = []
                minute_index_by_window: list[float | None] = []
                hour_by_window: list[float | None] = []
                agrees = True
                for begin, end in windows:
                    raw_total = raw_index.partitioned_total(name, begin, end).total
                    minute_total = minute_index.partitioned_total(
                        name, begin, end
                    ).total
                    hour_total = hour_index.partitioned_total(name, begin, end).total
                    raw_by_window.append(raw_total)
                    minute_index_by_window.append(minute_total)
                    hour_by_window.append(hour_total)
                    if not (
                        raw_total is not None
                        and raw_total == minute_total
                        and raw_total == hour_total
                    ):
                        agrees = False
                comparisons.append(
                    CounterComparison(
                        name=name,
                        raw=_sum_or_none(raw_by_window),
                        minute=_sum_or_none(minute_index_by_window),
                        hour=_sum_or_none(hour_by_window),
                        agrees=agrees,
                    )
                )
            return tuple(comparisons)
        finally:
            raw_index.close()
            minute_index.close()
            hour_index.close()


def _modification_time(path: Path) -> float | None:
    """A file's modification time, or ``None`` when it cannot be read."""
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _verdict(
    path: Path, *, now: float, older_than_days: float, live: frozenset[str]
) -> FileVerdict:
    """Everything the listing concluded about one raw receipt file."""
    job = receipts_job_id(path)
    mtime = _modification_time(path)
    size = path.stat().st_size if mtime is not None else 0
    first, last = row_span(path)
    minute_path, hour_path = tier_paths(path)

    counters: tuple[CounterComparison, ...] = ()
    siblings_present = (
        minute_path.is_file()
        and hour_path.is_file()
        and first is not None
        and last is not None
    )
    if siblings_present:
        assert first is not None and last is not None
        counters = compare_tiers(
            path, minute_path, hour_path, _hour_windows(first, last)
        )

    reasons: list[str] = []
    if job is None:
        reasons.append("name carries no job id")
    elif job in live:
        reasons.append(f"job {job} is running")
    if mtime is None:
        reasons.append("file cannot be read")
    elif now - mtime <= QUIET_SECONDS:
        reasons.append("modified within the last day")
    if last is None:
        reasons.append("no parseable row")
    elif now - last <= older_than_days * 86400:
        reasons.append(f"last row is not older than {older_than_days:g} days")
    if job is not None:
        missing = [
            sibling.name
            for sibling in (minute_path, hour_path)
            if not sibling.is_file()
        ]
        if missing:
            reasons.append("missing compacted sibling: " + ", ".join(missing))
        elif not counters:
            reasons.append("no agreement computed for the three tiers")
        elif not all(counter.agrees for counter in counters):
            reasons.append("compaction totals differ from the raw record")

    return FileVerdict(
        path=path,
        job_id=job,
        size=size,
        last_row_time=last,
        eligible=not reasons,
        reason="; ".join(reasons),
        counters=counters,
    )


def _fmt(value: float | None) -> str:
    """A total as the listing spells it, with an absent one shown as absent."""
    if value is None:
        return "n/a"
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.6g}"


def _short(name: str) -> str:
    """A measurement name without its section prefix, for a listing line."""
    return name.split(".", 1)[-1]


def scan(
    record_dir: str | Path,
    *,
    older_than_days: float = DEFAULT_OLDER_THAN_DAYS,
    now: float | None = None,
    running: RunningProvider | None = None,
) -> ScanResult:
    """List every raw receipt file under *record_dir* and judge each one.

    Liveness is read once for the whole listing, so every candidate is judged
    against the same scheduler reading rather than against one taken per file.
    """
    resolved = Path(record_dir)
    moment = time.time() if now is None else float(now)
    provider = default_running_job_ids if running is None else running
    live = frozenset(provider())
    verdicts = tuple(
        _verdict(path, now=moment, older_than_days=float(older_than_days), live=live)
        for path in raw_receipt_files(resolved)
    )
    return ScanResult(
        record_dir=resolved,
        older_than_days=float(older_than_days),
        now=moment,
        files=verdicts,
    )


def render_listing(result: ScanResult) -> str:
    """The listing as text: per file its identity and verdict, per counter its
    three tier totals and whether they agree."""
    lines: list[str] = []
    for verdict in result.files:
        status = "eligible" if verdict.eligible else f"ineligible ({verdict.reason})"
        lines.append(
            f"{verdict.path.name}  job={verdict.job_id}  "
            f"last_row={_iso(verdict.last_row_time)}  size={verdict.size}  {status}"
        )
        for counter in verdict.counters:
            outcome = "PASS" if counter.agrees else "FAIL"
            lines.append(
                f"    {_short(counter.name)}  raw={_fmt(counter.raw)}  "
                f"minute={_fmt(counter.minute)}  "
                f"hour={_fmt(counter.hour)}  {outcome}"
            )
    lines.append(
        f"{len(result.eligible)} of {len(result.files)} raw files "
        "eligible for retirement"
    )
    return "\n".join(lines) + "\n"


def write_manifest(
    result: ScanResult,
    *,
    manifest_dir: str | Path = RETENTION_MANIFEST_DIR,
    now: float | None = None,
) -> Path:
    """Write the eligible files as a manifest and return its path.

    Only eligible files are recorded. Each file's own digest and its two
    siblings' digests are frozen beside it, so the apply pass can prove the
    bytes it deletes are the bytes the listing judged.
    """
    moment = result.now if now is not None else time.time()
    directory = Path(manifest_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.fromtimestamp(moment, tz=_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    manifest_path = directory / f"retire-raw-{stamp}.json"
    files: list[dict[str, object]] = []
    for verdict in result.eligible:
        minute_path, hour_path = tier_paths(verdict.path)
        files.append(
            {
                "path": str(verdict.path),
                "job_id": verdict.job_id,
                "size": verdict.size,
                "sha256": _sha256(verdict.path),
                "last_row_time": _iso(verdict.last_row_time),
                "minute_sha256": _sha256(minute_path),
                "hour_sha256": _sha256(hour_path),
            }
        )
    document = {
        "created_at": _iso(moment),
        "record_dir": str(result.record_dir),
        "older_than_days": result.older_than_days,
        "files": files,
    }
    manifest_path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path


def _recheck(
    path: Path,
    entry: dict[str, object],
    *,
    now: float,
    older_than_days: float,
    live: frozenset[str],
) -> str | None:
    """Why a manifest file must not be deleted now, or ``None`` to delete it.

    The re-checks are the eligibility conditions that can have moved since the
    listing: liveness, quietness, age, and the bytes of the file and of the two
    compactions it depends on. A raw file is deleted only while all of them
    still hold.
    """
    if not is_raw_receipt(path):
        return "not a raw receipt file"
    job = receipts_job_id(path)
    if job is not None and job in live:
        return f"job {job} is running"
    if not path.is_file():
        return "file is gone"
    mtime = _modification_time(path)
    if mtime is None:
        return "file cannot be read"
    if now - mtime <= QUIET_SECONDS:
        return "modified within the last day"
    _first, last = row_span(path)
    if last is None:
        return "no parseable row"
    if now - last <= older_than_days * 86400:
        return "last row is no longer old enough"
    if _sha256(path) != entry.get("sha256"):
        return "file bytes changed since the listing"
    minute_path, hour_path = tier_paths(path)
    if not minute_path.is_file() or _sha256(minute_path) != entry.get("minute_sha256"):
        return "minute sibling changed"
    if not hour_path.is_file() or _sha256(hour_path) != entry.get("hour_sha256"):
        return "hour sibling changed"
    return None


def apply_manifest(
    manifest_path: str | Path,
    *,
    now: float | None = None,
    running: RunningProvider | None = None,
) -> ApplyReport:
    """Delete exactly the manifest's files that still pass every re-check.

    Only a raw receipt file is ever removed: the guard refuses a compacted tier
    or a request log by name, and a file that fails any re-check is skipped and
    reported rather than deleted. Nothing outside the manifest is touched.
    """
    resolved = Path(manifest_path)
    document = json.loads(resolved.read_text(encoding="utf-8"))
    older_than_days = float(document.get("older_than_days", DEFAULT_OLDER_THAN_DAYS))
    moment = time.time() if now is None else float(now)
    provider = default_running_job_ids if running is None else running
    live = frozenset(provider())

    removed: list[Path] = []
    skipped: list[tuple[Path, str]] = []
    for entry in document.get("files", []):
        path = Path(str(entry.get("path", "")))
        reason = _recheck(
            path, entry, now=moment, older_than_days=older_than_days, live=live
        )
        if reason is not None:
            skipped.append((path, reason))
            continue
        os.remove(path)
        removed.append(path)
    return ApplyReport(
        manifest_path=resolved, removed=tuple(removed), skipped=tuple(skipped)
    )
