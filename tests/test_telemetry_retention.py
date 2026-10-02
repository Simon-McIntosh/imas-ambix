"""Retiring a raw receipt file only where its compactions prove it redundant.

The listing is judged on conditions that can be seen and re-seen, so every test
here drives the real compaction to build the minute and hour tiers and then
perturbs exactly the one condition it is about. The apply pass is judged on what
it left behind as much as on what it removed, so the tests assert the compacted
siblings survive and that a file whose bytes moved is skipped rather than
deleted.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import time
from pathlib import Path

from imas_ambix.agent import telemetry_retention as retention
from imas_ambix.agent.telemetry_store import run_compaction

_NOW = _dt.datetime(2026, 10, 1, 12, 0, 0, tzinfo=_dt.UTC).timestamp()

#: A counter sequence rising 100 a row, one row every twenty minutes, so both
#: the two hour windows the job spans hold several raw rows and each tier has
#: two endpoints to difference.
_OFFSETS = (600, 1800, 3000, 4200, 5400, 6600)


def _row(epoch: float, generation: float, job_id: str = "1278105") -> dict:
    """One recorder sample with the canonical engine counters populated."""
    return {
        "timestamp": _dt.datetime.fromtimestamp(epoch, tz=_dt.UTC).isoformat(),
        "hostname": "98dci4-gpu-0003",
        "job_id": job_id,
        "engine": {
            "generation_tokens": generation,
            "prompt_tokens": generation * 2,
            "uncached_prompt_tokens": generation * 2.5,
        },
    }


def _write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _row_omitting(epoch: float, generation: float, omit: tuple[str, ...]) -> dict:
    """A recorder sample with the named engine counters left out entirely."""
    row = _row(epoch, generation)
    for name in omit:
        del row["engine"][name]
    return row


def _build_job(
    directory: Path,
    *,
    job_id: str = "1278105",
    slug: str = "deepseek-v4-1-flash",
    age_days: float = 20.0,
) -> tuple[Path, Path, Path]:
    """A raw receipt file with its two compactions, as the recorder would leave.

    The rows sit inside two full UTC hour windows, so no window holds a lone
    sample with nothing to difference against -- the shape the recorder's own
    cadence ensures in production.
    """
    hour = int((_NOW - age_days * 86400) // 3600) * 3600
    rows = [
        _row(hour + offset, 100.0 * index, job_id)
        for index, offset in enumerate(_OFFSETS)
    ]
    raw = directory / f"{slug}-{job_id}.jsonl"
    _write_rows(raw, rows)
    minute, hour_path = retention.tier_paths(raw)
    run_compaction(raw, minute, hour_path)
    # The file is quiet: last modified well before the one-day threshold is over.
    quiet = _NOW - 21 * 86400
    for path in (raw, minute, hour_path):
        os.utime(path, (quiet, quiet))
    return raw, minute, hour_path


def _none_running() -> frozenset[str]:
    return frozenset()


def _listing(directory: Path, *, age_days: float = 14.0) -> retention.ScanResult:
    return retention.scan(
        directory, older_than_days=age_days, now=_NOW, running=_none_running
    )


def _verdict_for(result: retention.ScanResult, raw: Path) -> retention.FileVerdict:
    by_path = {verdict.path: verdict for verdict in result.files}
    return by_path[raw]


def test_a_raw_file_whose_opening_row_carries_its_job_id_is_a_candidate(tmp_path):
    """The row check admits a genuine raw file, so its refusals carry weight."""
    raw, _minute, _hour = _build_job(tmp_path)

    assert retention.is_raw_receipt(raw) is True
    assert raw.name in {verdict.path.name for verdict in _listing(tmp_path).files}


def test_a_request_rotation_named_like_a_raw_file_is_not_a_candidate(tmp_path):
    """A router request file whose name has the raw shape is not a raw receipt.

    ``requests-2026-10-01.jsonl`` matches the raw name shape and its ``01``
    parses as a job id, so a name-only test would list it. Its rows are router
    requests carrying no ``job_id``, so reading the opening row refuses it.
    """
    requests_path = tmp_path / "requests-2026-10-01.jsonl"
    _write_rows(
        requests_path,
        [
            {
                "timestamp": _dt.datetime.fromtimestamp(_NOW, tz=_dt.UTC).isoformat(),
                "caller": "clive",
                "model": "deepseek-v4-flash",
                "outcome": "ok",
            }
        ],
    )

    result = _listing(tmp_path)

    assert retention.is_raw_receipt(requests_path) is False
    assert requests_path.name not in {verdict.path.name for verdict in result.files}


def test_a_raw_name_whose_opening_row_job_id_differs_is_not_a_candidate(tmp_path):
    """The name and the opening row must agree on the job id.

    A file named for one job whose first row carries another is not the raw
    receipt its name claims, so it is never listed and never deleted.
    """
    raw = tmp_path / "deepseek-v4-1-flash-1278105.jsonl"
    _write_rows(raw, [_row(_NOW, 100.0, job_id="1278999")])

    result = _listing(tmp_path)

    assert retention.is_raw_receipt(raw) is False
    assert raw.name not in {verdict.path.name for verdict in result.files}


def test_a_running_jobs_file_is_listed_ineligible(tmp_path):
    """A job the scheduler still reports is never a retirement candidate."""
    raw, _minute, _hour = _build_job(tmp_path)

    def running() -> frozenset[str]:
        return frozenset({"1278105"})

    result = retention.scan(tmp_path, now=_NOW, running=running)

    verdict = _verdict_for(result, raw)
    assert verdict.eligible is False
    assert "running" in verdict.reason


def test_an_ended_old_job_whose_check_passes_is_eligible(tmp_path):
    """An ended job older than the threshold, with agreeing tiers, is eligible."""
    raw, _minute, _hour = _build_job(tmp_path)

    verdict = _verdict_for(_listing(tmp_path), raw)

    assert verdict.eligible is True, verdict.reason
    assert verdict.reason == ""
    assert verdict.counters, "the three-tier comparison was not run"
    assert all(counter.agrees for counter in verdict.counters)


def test_an_ended_old_job_whose_minute_window_differs_is_ineligible(tmp_path):
    """One window of one counter that disagrees with raw keeps the file.

    The minute tier's row closing the first hour window is raised, and its
    opening is left alone, so the tier's whole-job span is unchanged while that
    one window's total is not. The check must see the window.
    """
    raw, minute, _hour = _build_job(tmp_path)
    hour_start = int((_NOW - 20 * 86400) // 3600) * 3600
    boundary = _dt.datetime.fromtimestamp(hour_start + 3000, tz=_dt.UTC).isoformat()
    rows = [
        json.loads(line) for line in minute.read_text().splitlines() if line.strip()
    ]
    raised = False
    for row in rows:
        if row["timestamp"] == boundary:
            row["engine"]["generation_tokens"] += 50.0
            raised = True
    assert raised, "the minute row to alter was not found"
    minute.write_text("".join(json.dumps(row) + "\n" for row in rows))

    verdict = _verdict_for(_listing(tmp_path), raw)

    assert verdict.eligible is False
    assert "differ" in verdict.reason
    by_name = {counter.name: counter for counter in verdict.counters}
    assert by_name["engine.generation_tokens"].agrees is False
    # The disagreement is a window's, not the whole job's: the totals still match.
    assert by_name["engine.generation_tokens"].raw == (
        by_name["engine.generation_tokens"].minute
    )


def test_an_absent_counter_is_absent_not_zero(tmp_path):
    """A counter no row of a tier carries in a window is absent, not zero.

    The raw file records generation and prompt tokens but never the uncached
    counter, while the minute and hour tiers carry it. The absent reading must
    survive as absent rather than be summed as a measured zero, and a tier that
    holds a value where the raw file holds none is a disagreement -- even though
    the other counters agree over the same window.
    """
    hour = int((_NOW - 20 * 86400) // 3600) * 3600
    windows = [(float(hour), float(hour + 3600))]
    raw = tmp_path / "deepseek-v4-1-flash-1278105.jsonl"
    minute, hour_path = retention.tier_paths(raw)
    _write_rows(
        raw,
        [
            _row_omitting(hour + 600, 100.0, ("uncached_prompt_tokens",)),
            _row_omitting(hour + 1800, 200.0, ("uncached_prompt_tokens",)),
        ],
    )
    _write_rows(minute, [_row(hour + 600, 100.0), _row(hour + 1800, 200.0)])
    _write_rows(hour_path, [_row(hour + 600, 100.0), _row(hour + 1800, 200.0)])

    comparisons = retention.compare_tiers(raw, minute, hour_path, windows)
    by_name = {counter.name: counter for counter in comparisons}

    uncached = by_name["engine.uncached_prompt_tokens"]
    assert uncached.raw is None, "an absent counter was reported as a value"
    assert uncached.raw != 0.0
    assert uncached.minute == 250.0
    assert uncached.hour == 250.0
    assert uncached.agrees is False
    # The counters both tiers carry still agree over the same window.
    assert by_name["engine.generation_tokens"].agrees is True


def test_the_tier_check_over_a_large_record_completes_quickly(tmp_path):
    """The check scales to the real record: 20,000 raw rows and their tiers.

    The retirement listing runs over every candidate file, and a raw receipt
    grows about 8.7 MB a day, so the comparison is read directly from the rows
    rather than through a rebuilt index. Twenty thousand rows -- several days at
    the recorder's five-second cadence -- with their minute and hour compactions
    must be checked well inside ten seconds.
    """
    start = int((_NOW - 20 * 86400) // 3600) * 3600
    raw = tmp_path / "deepseek-v4-1-flash-1278105.jsonl"
    # A one-second phase offset keeps no reading exactly on a UTC hour boundary,
    # so the raw file and its compactions differencing the same endpoints agree
    # window by window -- the shape a free-running recorder cadence produces.
    _write_rows(
        raw,
        [_row(start + 1 + index * 5, 100.0 * index) for index in range(20000)],
    )
    minute, hour_path = retention.tier_paths(raw)
    run_compaction(raw, minute, hour_path)
    first, last = retention.row_span(raw)
    assert first is not None and last is not None
    windows = retention._hour_windows(first, last)

    began = time.monotonic()
    comparisons = retention.compare_tiers(raw, minute, hour_path, windows)
    elapsed = time.monotonic() - began

    assert comparisons and all(counter.agrees for counter in comparisons)
    assert elapsed < 10.0, f"the tier comparison took {elapsed:.1f}s"


def test_a_five_day_old_ended_job_is_ineligible(tmp_path):
    """A job younger than the threshold is ineligible however healthy its tiers."""
    raw, _minute, _hour = _build_job(tmp_path, age_days=5.0)

    verdict = _verdict_for(_listing(tmp_path), raw)

    assert verdict.eligible is False
    assert "older than" in verdict.reason


def test_apply_deletes_only_the_eligible_raw_file(tmp_path):
    """The apply pass removes the manifest's file and leaves every other file.

    A second, too-young job sits beside the eligible one; it is never in the
    manifest, so the apply pass never sees it, and its raw file must remain. The
    eligible job's compacted siblings must survive the deletion of the raw file.
    """
    eligible, minute, hour = _build_job(tmp_path, slug="deepseek-v4-1-flash")
    young, _young_minute, _young_hour = _build_job(
        tmp_path, job_id="1279999", age_days=5.0
    )

    result = _listing(tmp_path)
    manifest = retention.write_manifest(result, manifest_dir=tmp_path / "manifests")
    report = retention.apply_manifest(manifest, now=_NOW, running=_none_running)

    assert eligible in report.removed
    assert not eligible.exists()
    assert young.exists(), "an ineligible job's raw file was removed"
    assert minute.exists(), "a compacted sibling was removed"
    assert hour.exists(), "a compacted sibling was removed"


def test_apply_skips_a_file_whose_bytes_changed_after_listing(tmp_path):
    """A raw file rewritten after the listing is skipped, not deleted.

    The bytes are changed and the modification time reset so the only check that
    can catch it is the digest -- which is exactly the one the manifest froze.
    """
    raw, minute, hour = _build_job(tmp_path)
    manifest = retention.write_manifest(
        _listing(tmp_path), manifest_dir=tmp_path / "manifests"
    )
    with raw.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_row(_NOW, 1.0)) + "\n")
    quiet = _NOW - 21 * 86400
    os.utime(raw, (quiet, quiet))

    report = retention.apply_manifest(manifest, now=_NOW, running=_none_running)

    assert raw not in report.removed
    assert raw.exists()
    assert any(str(raw) == str(path) for path, _ in report.skipped)
    assert minute.exists() and hour.exists()


def test_no_compacted_file_is_ever_removed(tmp_path):
    """A manifest naming a compacted tier must not let the apply pass delete it."""
    raw, minute, hour = _build_job(tmp_path)
    manifest = retention.write_manifest(
        _listing(tmp_path), manifest_dir=tmp_path / "manifeasts"
    )
    # Forge a manifest that names the compacted tiers, to prove the guard refuses
    # them by name even when handed their path.
    document = json.loads(manifest.read_text())
    forged = dict(document)
    forged["files"] = [
        {"path": str(minute), "sha256": retention._sha256(minute)},
        {"path": str(hour), "sha256": retention._sha256(hour)},
    ]
    forged_path = tmp_path / "forged.json"
    forged_path.write_text(json.dumps(forged))

    report = retention.apply_manifest(forged_path, now=_NOW, running=_none_running)

    assert report.removed == ()
    assert minute.exists() and hour.exists() and raw.exists()
