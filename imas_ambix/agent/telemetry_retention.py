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
* both compacted siblings exist and the (hour, counter) pairs they carry agree:
  a pair is one UTC hour and one cumulative token counter, and a tier is
  present for a pair when one of its rows in that hour carries the counter. A
  pair absent in every tier is skipped -- there is nothing to verify -- so a
  file whose early rows predate the counter, or whose every row does, is not
  refused for the counter it never recorded. A pair present in some tiers and
  absent in others is a disagreement, and a pair present in all three agrees
  when the partitioned totals over raw, over minute and over hour are equal
  exactly. The partition uses each tier's own opening block, so a serve restart
  inside a window is totalled run by run in every tier. A file is eligible only
  once at least one pair was compared: with no compared pair there is no
  evidence a retirement would rest on, so the file is kept. A file whose only
  pairs are absent everywhere is kept with the reason ``no hour carries the
  counters``.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from imas_ambix.agent.serving_receipts import tier_paths
from imas_ambix.agent.telemetry_index import (
    counter_run_continues,
    receipts_job_id,
    row_boot_id,
    row_host,
    row_job_id,
)

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
    """One counter's totals per tier and whether its pairs agree.

    ``raw``, ``minute`` and ``hour`` are each the sum, over the windows that
    carry the counter in any tier, of that tier's partitioned total for it.
    ``agrees`` is decided pair by pair -- one UTC hour and this counter -- before
    the sums are taken, so a pair that disagrees in one hour cannot be masked by
    another hour compensating it. A window in which no tier carries the counter
    contributes nothing and is neither compared nor a disagreement.
    """

    name: str
    raw: float | None
    minute: float | None
    hour: float | None
    agrees: bool


@dataclass(frozen=True)
class TierComparison:
    """The three-tier comparison for one raw file.

    ``counters`` carries one :class:`CounterComparison` per engine counter.
    ``compared_pairs`` counts the (hour, counter) pairs present in all three
    tiers -- the pairs actually compared. ``compared_hours`` counts the distinct
    UTC hours any compared pair falls in. A file with ``compared_pairs`` zero was
    compared on no evidence at all.
    """

    counters: tuple[CounterComparison, ...]
    compared_pairs: int
    compared_hours: int


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
    compared_pairs: int = 0
    compared_hours: int = 0


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
    than be summed away. No window at all is an absence too, so an empty list
    answers ``None`` rather than the zero an empty ``sum`` would give.
    """
    if not values or any(value is None for value in values):
        return None
    return sum(values)


def _read_tier_rows(path: Path) -> list[dict]:
    """Every row a tier file holds, in time order.

    Read tolerantly, as :func:`row_span` is: a trailing line still being written
    is not a row, and a line that does not parse as an object is skipped rather
    than refusing the whole file. Rows are ordered by their own timestamp, ties
    keeping the file's own order, so the run partition is stable across passes.
    A file that cannot be opened holds no rows, which is the same thing an
    absent tier offers.
    """
    rows: list[dict] = []
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
                    rows.append(row)
    except OSError:
        return []
    rows.sort(key=lambda row: _timestamp_of(row) or 0.0)
    return rows


def _counter_value(row: dict, name: str) -> float | None:
    """The value of the cumulative counter *name* on *row*, or ``None``.

    *name* is the dotted path the record spells the counter under
    (``engine.generation_tokens``). A leaf the row does not carry, a null and a
    non-numeric leaf each answer ``None``. That is deliberate: a quantity the
    record does not carry is absent, and an absent reading must not be summed as
    a measured zero.
    """
    node: object = row
    for part in name.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    if isinstance(node, bool) or not isinstance(node, (int, float)):
        return None
    return float(node)


def _counter_opening(row: object, name: str) -> float | None:
    """The opening a compacted row declares for its own window, or ``None``.

    A compacted row carries an ``open`` block naming, per cumulative counter, the
    value the run held at the start of the row's own window. Reading it back is
    what lets a window be totalled from any tier: differencing a compacted row
    against its carried opening gives the same figure the raw rows would. A raw
    row has no such block and answers ``None``, so the run's own reading is the
    window's opening.
    """
    if not isinstance(row, dict):
        return None
    block = row.get("open")
    if not isinstance(block, dict):
        return None
    value = block.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _run_opening(
    run: list[tuple[float, float, float | None]], start: float, end: float
) -> tuple[float, float, float | None] | None:
    """A run's endpoint at or before the window, or its first inside it.

    The opening reading is the last at or before *start* so a difference counts
    the traffic served from that reading onward, and a window whose own first
    sample already carries the counter needs no earlier row. The entry carries
    the opening a compacted row declares as well as its closing, so the caller
    can tell a first-in-window compacted row -- which spans its own window and so
    has two endpoints -- from a compacted row whose opening lies earlier.
    """
    opening: tuple[float, float, float | None] | None = None
    for entry in run:
        if entry[0] <= start:
            opening = entry
        else:
            break
    if opening is not None:
        return opening
    for entry in run:
        if start <= entry[0] < end:
            return entry
    return None


def _run_closing(
    run: list[tuple[float, float, float | None]], end: float
) -> tuple[float, float, float | None] | None:
    """A run's latest reading strictly before the window's end."""
    closing: tuple[float, float, float | None] | None = None
    for entry in run:
        if entry[0] < end:
            closing = entry
        else:
            break
    return closing


def _counter_runs(
    rows: list[dict], name: str
) -> list[list[tuple[float, float, float | None]]]:
    """Partition a tier's rows into contiguous runs for one counter.

    Each entry is ``(ts_epoch, closing, opening)``: *opening* is the value the
    row declares its own window opened at (``None`` for a raw row). A run is one
    serving process on one host over one boot whose counter never fell -- the
    predicate the compactor also splits its strides on -- so a window is totalled
    per run and a restart inside a window is not differenced across.
    """
    runs: list[list[tuple[float, float, float | None]]] = []
    previous: tuple[tuple[object, ...], dict[str, float]] | None = None
    for row in rows:
        value = _counter_value(row, name)
        if value is None:
            continue
        stamp = _timestamp_of(row)
        if stamp is None:
            continue
        key = (row_host(row), row_boot_id(row), row_job_id(row))
        current = (key, {"value": value})
        if previous is None or not counter_run_continues(previous, current):
            runs.append([])
        runs[-1].append((stamp, value, _counter_opening(row, name)))
        previous = current
    return runs


def _run_window_total(
    runs: list[list[tuple[float, float, float | None]]], start: float, end: float
) -> float | None:
    """One counter's total over ``[start, end)``, summed over its runs.

    Each run contributes its closing minus its opening, in the same shape the
    record warehouse's partitioned total uses. ``None`` when no run contributed
    a second endpoint, which is not the same answer as ``0.0`` for a run that was
    observed and did not advance.
    """
    total: float | None = None
    for run in runs:
        opening = _run_opening(run, start, end)
        if opening is None:
            continue
        closing = _run_closing(run, end)
        if closing is None or closing[0] < start:
            continue
        carried = opening[2] if opening[0] > start else None
        if closing is opening and carried is None:
            continue
        base = carried if carried is not None else opening[1]
        total = (0.0 if total is None else total) + (closing[1] - base)
    return total


def _window_holds_counter(
    runs: list[list[tuple[float, float, float | None]]], begin: float, end: float
) -> bool:
    """Whether any row of a tier carries the counter inside ``[begin, end)``.

    The runs hold only rows that carry the counter, so a tier is present for a
    pair exactly when one of its rows falls in the hour. A row of the tier that
    omits the counter is not in the runs, so a tier whose hour holds rows but
    none of the counter answers absent rather than present.
    """
    for run in runs:
        for entry in run:
            if begin <= entry[0] < end:
                return True
    return False


def compare_tiers(
    raw_path: Path,
    minute_path: Path,
    hour_path: Path,
    windows: list[tuple[float, float]],
) -> TierComparison:
    """Per-pair agreement between the raw file and its two compactions.

    The comparison works on (hour, counter) pairs. Each tier's rows are read
    directly and totalled per UTC hour with the partitioned-total arithmetic the
    record warehouse answers with -- one run per serving process, a compacted
    row's carried opening used as the base where the run begins inside the
    window -- so the same comparison is made without building an index for
    either tier. For each pair:

    * absent in every tier -- no tier's rows in that hour carry the counter --
      the pair is skipped, neither compared nor a disagreement;
    * present in some tiers and absent in others, the pair is a disagreement;
    * present in all three, the pair is compared, and agrees when the raw,
      minute and hour totals are equal exactly.

    A ``CounterComparison``'s ``agrees`` is false if any of its pairs is a
    disagreement or any compared pair's totals differ. ``compared_pairs`` counts
    the pairs present in all three tiers, and ``compared_hours`` the distinct
    hours they fall in.
    """
    tiers = {
        "raw": _read_tier_rows(raw_path),
        "minute": _read_tier_rows(minute_path),
        "hour": _read_tier_rows(hour_path),
    }
    comparisons: list[CounterComparison] = []
    compared_pairs = 0
    compared_hours: set[int] = set()
    for name in COUNTER_NAMES:
        runs = {tier: _counter_runs(rows, name) for tier, rows in tiers.items()}
        totals: dict[str, list[float | None]] = {tier: [] for tier in tiers}
        agrees = True
        for index, (begin, end) in enumerate(windows):
            present = {
                tier: _window_holds_counter(runs[tier], begin, end) for tier in tiers
            }
            if not any(present.values()):
                continue
            raw_total = _run_window_total(runs["raw"], begin, end)
            minute_total = _run_window_total(runs["minute"], begin, end)
            hour_total = _run_window_total(runs["hour"], begin, end)
            totals["raw"].append(raw_total)
            totals["minute"].append(minute_total)
            totals["hour"].append(hour_total)
            if not all(present.values()):
                agrees = False
                continue
            compared_pairs += 1
            compared_hours.add(index)
            if not (raw_total == minute_total == hour_total):
                agrees = False
        comparisons.append(
            CounterComparison(
                name=name,
                raw=_sum_or_none(totals["raw"]),
                minute=_sum_or_none(totals["minute"]),
                hour=_sum_or_none(totals["hour"]),
                agrees=agrees,
            )
        )
    return TierComparison(
        counters=tuple(comparisons),
        compared_pairs=compared_pairs,
        compared_hours=len(compared_hours),
    )


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

    comparison: TierComparison | None = None
    siblings_present = (
        minute_path.is_file()
        and hour_path.is_file()
        and first is not None
        and last is not None
    )
    if siblings_present:
        assert first is not None and last is not None
        comparison = compare_tiers(
            path, minute_path, hour_path, _hour_windows(first, last)
        )
    counters = comparison.counters if comparison is not None else ()

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
        elif comparison is None:
            reasons.append("no agreement computed for the three tiers")
        else:
            if not all(counter.agrees for counter in comparison.counters):
                reasons.append("compaction totals differ from the raw record")
            if comparison.compared_pairs == 0:
                reasons.append("no hour carries the counters")

    return FileVerdict(
        path=path,
        job_id=job,
        size=size,
        last_row_time=last,
        eligible=not reasons,
        reason="; ".join(reasons),
        counters=counters,
        compared_pairs=comparison.compared_pairs if comparison is not None else 0,
        compared_hours=comparison.compared_hours if comparison is not None else 0,
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
    """The listing as text: per file its identity and verdict, its compared-pair
    and compared-hour counts, and per counter its three tier totals and whether
    they agree."""
    lines: list[str] = []
    for verdict in result.files:
        status = "eligible" if verdict.eligible else f"ineligible ({verdict.reason})"
        lines.append(
            f"{verdict.path.name}  job={verdict.job_id}  "
            f"last_row={_iso(verdict.last_row_time)}  size={verdict.size}  {status}  "
            f"compared_pairs={verdict.compared_pairs}  "
            f"compared_hours={verdict.compared_hours}"
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
                "compared_pairs": verdict.compared_pairs,
                "compared_hours": verdict.compared_hours,
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
