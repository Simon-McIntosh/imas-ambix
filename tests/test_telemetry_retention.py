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

import pytest

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


def _old_row(epoch: float, job_id: str = "1278105") -> dict:
    """A recorder sample from before the engine block existed.

    The counters the check totals are absent, as they are in every pre-engine
    row; only a throughput rate is carried, which is a gauge and not one of the
    counters the retirement check compares.
    """
    return {
        "timestamp": _dt.datetime.fromtimestamp(epoch, tz=_dt.UTC).isoformat(),
        "hostname": "98dci4-gpu-0003",
        "job_id": job_id,
        "engine": {"generation_throughput_toks_per_s": 100.0},
    }


def _build_hybrid_job(
    directory: Path,
    *,
    old_hours: int = 2,
    covered_hours: int = 2,
    job_id: str = "1278105",
    slug: str = "deepseek-v4-1-flash",
    age_days: float = 20.0,
) -> tuple[Path, Path, Path]:
    """A raw receipt whose early hours predate the engine block.

    The first ``old_hours`` UTC hours carry no counter in any tier, the way the
    recorder's rows read before it gained the engine section; the following
    ``covered_hours`` carry all three counters in every tier and agree. The
    shape is job 1273253's: a hybrid file the check must retire, its counters
    verified only where they exist.
    """
    base = int((_NOW - age_days * 86400) // 3600) * 3600
    rows: list[dict] = []
    for hour_index in range(old_hours):
        hour = base + hour_index * 3600
        rows.extend(_old_row(hour + offset, job_id) for offset in (600, 1800, 3000))
    generation = 100.0
    for hour_index in range(old_hours, old_hours + covered_hours):
        hour = base + hour_index * 3600
        for offset in (600, 1800, 3000):
            rows.append(_row(hour + offset, generation, job_id))
            generation += 1.0
    raw = directory / f"{slug}-{job_id}.jsonl"
    _write_rows(raw, rows)
    minute, hour_path = retention.tier_paths(raw)
    run_compaction(raw, minute, hour_path)
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


def test_a_rebuilt_window_disagreement_keeps_the_file(tmp_path):
    """A file whose rebuilt tiers still disagree over its own window is kept.

    A job that closes a day segment and reopens its record inside one UTC hour
    leaves two raw files sharing a window. The rebuild reads both into one row
    stream, so the installed tiers carry the reopened file's traffic while this
    file carries only its own, and the window totals differ. The rebuild cannot
    make this file redundant, so it is reported and kept rather than retired on
    an ambiguous comparison.
    """
    hour = int((_NOW - 20 * 86400) // 3600) * 3600
    segment = tmp_path / "deepseek-v4-1-flash-1278105.20260911.jsonl"
    reopened = tmp_path / "deepseek-v4-1-flash-1278105.jsonl"
    _write_rows(segment, [_row(hour + 600, 0.0), _row(hour + 1800, 100.0)])
    _write_rows(reopened, [_row(hour + 3000, 1000.0), _row(hour + 4200, 1100.0)])
    quiet = _NOW - 21 * 86400
    for path in (segment, reopened):
        os.utime(path, (quiet, quiet))

    verdict = _verdict_for(_listing(tmp_path), segment)

    assert verdict.eligible is False
    assert "differ" in verdict.reason
    by_name = {counter.name: counter for counter in verdict.counters}
    assert by_name["engine.generation_tokens"].agrees is False
    assert by_name["engine.generation_tokens"].raw == 100.0
    assert by_name["engine.generation_tokens"].minute == 1000.0


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
    by_name = {counter.name: counter for counter in comparisons.counters}

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
    comparison = retention.compare_tiers(raw, minute, hour_path, windows)
    elapsed = time.monotonic() - began

    assert comparison.counters and all(
        counter.agrees for counter in comparison.counters
    )
    assert comparison.compared_pairs > 0
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


def test_a_hybrid_file_whose_old_format_hours_carry_no_counter_is_eligible(tmp_path):
    """Hours carrying no counter in any tier are skipped, not disagreements.

    The early hours predate the engine block, so no tier records the three
    counters the check totals are; the later hours carry all three and agree.
    The file is eligible once its covered hours are compared and agree, which is
    what lets a hybrid file such as job 1273253 be retired while a purely
    old-format one is not.
    """
    raw, _minute, _hour = _build_hybrid_job(tmp_path, old_hours=2, covered_hours=2)

    verdict = _verdict_for(_listing(tmp_path), raw)

    assert verdict.eligible is True, verdict.reason
    assert verdict.compared_pairs == 6, verdict.counters
    assert verdict.compared_hours == 2


def test_a_pure_old_format_file_with_no_compared_pair_is_kept(tmp_path):
    """A file whose every hour carries no counter is kept, not deleted on nothing.

    With no counter recorded anywhere there is no comparison to rest a
    retirement on, so the file is kept with the reason that says so rather than
    being retired on an empty check.
    """
    raw, _minute, _hour = _build_hybrid_job(tmp_path, old_hours=2, covered_hours=0)

    verdict = _verdict_for(_listing(tmp_path), raw)

    assert verdict.eligible is False
    assert "no hour carries the counters" in verdict.reason
    assert verdict.compared_pairs == 0
    assert verdict.compared_hours == 0


def test_a_pair_present_in_raw_and_absent_in_the_minute_tier_is_refused(tmp_path):
    """A counter one tier records and another does not is a disagreement.

    The raw and hour tiers carry generation tokens while the minute tier's rows
    for the same hour omit them. The pair is present in some tiers and absent in
    others, so it is refused rather than skipped.
    """
    hour = int((_NOW - 20 * 86400) // 3600) * 3600
    windows = [(float(hour), float(hour + 3600))]
    raw = tmp_path / "deepseek-v4-1-flash-1278105.jsonl"
    minute, hour_path = retention.tier_paths(raw)
    _write_rows(raw, [_row(hour + 600, 100.0), _row(hour + 1800, 200.0)])
    _write_rows(
        minute,
        [
            _row_omitting(hour + 600, 100.0, ("generation_tokens",)),
            _row_omitting(hour + 1800, 200.0, ("generation_tokens",)),
        ],
    )
    _write_rows(hour_path, [_row(hour + 600, 100.0), _row(hour + 1800, 200.0)])

    comparison = retention.compare_tiers(raw, minute, hour_path, windows)
    by_name = {counter.name: counter for counter in comparison.counters}

    generation = by_name["engine.generation_tokens"]
    assert generation.agrees is False
    assert generation.raw is not None and generation.hour is not None
    assert generation.minute is None
    # The two counters every tier records are present in all three and counted.
    assert comparison.compared_pairs == 2


def test_a_pair_present_in_all_tiers_with_unequal_totals_is_refused(tmp_path):
    """All three tiers carry the counter, but the minute window differs.

    The minute tier's closing reading for the hour is raised, so raw, minute and
    hour are all present and the compared totals are not equal. The pair is
    present in all three tiers, so the refusal is the totals' and not a presence
    gap.
    """
    hour = int((_NOW - 20 * 86400) // 3600) * 3600
    windows = [(float(hour), float(hour + 3600))]
    raw = tmp_path / "deepseek-v4-1-flash-1278105.jsonl"
    minute, hour_path = retention.tier_paths(raw)
    _write_rows(raw, [_row(hour + 600, 100.0), _row(hour + 1800, 200.0)])
    _write_rows(minute, [_row(hour + 600, 100.0), _row(hour + 1800, 260.0)])
    _write_rows(hour_path, [_row(hour + 600, 100.0), _row(hour + 1800, 200.0)])

    comparison = retention.compare_tiers(raw, minute, hour_path, windows)
    by_name = {counter.name: counter for counter in comparison.counters}

    generation = by_name["engine.generation_tokens"]
    assert generation.agrees is False
    assert generation.raw == 100.0
    assert generation.minute == 160.0
    assert generation.hour == 100.0
    # Present in all three tiers: the pair was compared, not skipped.
    assert comparison.compared_pairs == 3


def test_a_file_whose_only_present_pairs_yield_no_total_is_kept(tmp_path):
    """A pair present in every tier but yielding no total is skipped, not agreed.

    Each tier holds one row in the hour, so a tier is genuinely present for the
    pair, but a lone row offers one endpoint and no earlier opening to
    difference against, so ``_run_window_total`` answers None for all three.
    ``None == None == None`` is not the agreement a retirement may rest on, so
    the pair is neither compared nor a disagreement. With no compared pair the
    file is kept rather than retired on an empty comparison.
    """
    hour = int((_NOW - 20 * 86400) // 3600) * 3600
    raw = tmp_path / "deepseek-v4-1-flash-1278105.jsonl"
    minute, hour_path = retention.tier_paths(raw)
    _write_rows(raw, [_row(hour + 600, 100.0)])
    _write_rows(minute, [_row(hour + 600, 100.0)])
    _write_rows(hour_path, [_row(hour + 600, 100.0)])
    quiet = _NOW - 21 * 86400
    for path in (raw, minute, hour_path):
        os.utime(path, (quiet, quiet))

    verdict = _verdict_for(_listing(tmp_path), raw)

    assert verdict.eligible is False
    assert "no hour carries the counters" in verdict.reason
    assert verdict.compared_pairs == 0
    assert verdict.compared_hours == 0
    assert all(counter.raw is None for counter in verdict.counters), (
        "a present tier with no endpoint was totalled as something other than absent"
    )


def test_a_pair_with_a_total_in_raw_and_none_in_the_minute_tier_is_refused(tmp_path):
    """A present pair whose total is a number in one tier and absent in another.

    All three tiers carry the counter in the hour, so the pair is present in
    every tier, but the minute tier holds a single row -- one endpoint and no
    opening -- so its total is absent while raw and hour yield numbers. A
    present pair whose totals are a number in some tiers and absent in others is
    a disagreement rather than a skip, so the file cannot be retired on it.
    """
    hour = int((_NOW - 20 * 86400) // 3600) * 3600
    windows = [(float(hour), float(hour + 3600))]
    raw = tmp_path / "deepseek-v4-1-flash-1278105.jsonl"
    minute, hour_path = retention.tier_paths(raw)
    _write_rows(raw, [_row(hour + 600, 100.0), _row(hour + 1800, 200.0)])
    _write_rows(minute, [_row(hour + 1800, 200.0)])
    _write_rows(hour_path, [_row(hour + 600, 100.0), _row(hour + 1000, 200.0)])

    comparison = retention.compare_tiers(raw, minute, hour_path, windows)
    by_name = {counter.name: counter for counter in comparison.counters}

    generation = by_name["engine.generation_tokens"]
    assert generation.agrees is False
    assert generation.raw == 100.0
    assert generation.minute is None
    # The pair was not compared: a total absent in one tier with a number in the
    # others is a disagreement, not an agreement to skip.
    assert comparison.compared_pairs == 0


def test_the_listing_shows_compared_pair_and_hours_counts(tmp_path):
    """The listing prints, per file, the compared pairs and their hours.

    A healthy two-hour job compares three counters over each of two hours: six
    pairs falling in two distinct hours.
    """
    raw, _minute, _hour = _build_job(tmp_path)

    text = retention.render_listing(_listing(tmp_path))

    assert "compared_pairs=6" in text
    assert "compared_hours=2" in text
    assert raw.name in text


def test_a_running_jobs_closed_day_segment_is_not_held_by_its_run(tmp_path):
    """A live job's closed days may retire; only its open record may not.

    The open record is still being appended to, so retiring it would lose the
    rows that arrive after the listing is taken. A day segment is closed by
    construction -- the recorder stops appending to it when the day it covers
    ends -- so the job being live says nothing about whether that day may go.
    A scan that held every file of a running job would keep a long serve's whole
    history for as long as it ran, which is the accumulation the roll exists to
    end.
    """
    raw, _minute, _hour = _build_job(tmp_path)
    segment = tmp_path / "deepseek-v4-1-flash-1278105.20260911.jsonl"
    os.rename(raw, segment)
    quiet = _NOW - 21 * 86400
    os.utime(segment, (quiet, quiet))

    # The record one day on, under the name the roll reopened: same job, still
    # being written, so its rows are recent by construction.
    open_path = tmp_path / "deepseek-v4-1-flash-1278105.jsonl"
    _write_rows(
        open_path,
        [
            _row(_NOW - 3600 + offset, 900.0 + index * 100)
            for index, offset in enumerate((600, 1800, 3000))
        ],
    )

    def running() -> frozenset[str]:
        return frozenset({"1278105"})

    result = retention.scan(tmp_path, now=_NOW, running=running)

    segment_verdict = _verdict_for(result, segment)
    assert "running" not in segment_verdict.reason
    assert segment_verdict.eligible is True, segment_verdict.reason

    open_verdict = _verdict_for(result, open_path)
    assert open_verdict.eligible is False
    assert "job 1278105 is running" in open_verdict.reason


def test_a_record_whose_tiers_lack_its_tail_is_rebuilt_and_then_proves_equal(
    tmp_path,
):
    """An ended job's tiers are rebuilt from its whole record before judging.

    The recorder stopping between passes leaves the tiers holding the head of
    the record and none of its tail, and a tier compacted before rows carried an
    opening block loses its first window's opening advance. Both are the shape
    job 1278105 had, and both make the installed tiers disagree with the record
    however healthy the record is. The listing must re-derive them from the
    record rather than refuse the file.
    """
    raw, minute, hour_path = _build_job(tmp_path)
    # Only the head of the record was ever compacted, and without an opening
    # block, so every tier's totals fall short of the record's own span.
    for tier in (minute, hour_path):
        rows = [
            json.loads(line) for line in tier.read_text().splitlines() if line.strip()
        ]
        rows[0].pop("open", None)
        _write_rows(tier, rows[:1])

    before = _verdict_for(
        retention.scan(tmp_path, now=_NOW, running=_none_running, rebuild=False), raw
    )
    assert before.eligible is False, "the tiers were not actually short of the tail"

    after = _verdict_for(_listing(tmp_path), raw)

    assert after.eligible is True, after.reason
    assert after.counters, "the rebuilt tiers were not compared"
    assert all(counter.agrees for counter in after.counters)


def test_a_rebuilt_tier_is_staged_beside_the_tier_and_renamed_into_place(
    tmp_path, monkeypatch
):
    """A rebuilt tier is written to a scratch path and renamed onto the tier.

    A comparison must never read a half-written tier, so the rebuild stages each
    one beside the path it replaces and renames it into place in the same
    directory.
    """
    raw, minute, hour_path = _build_job(tmp_path)
    staged: list[tuple[Path, Path]] = []
    real_replace = os.replace

    def spy(source, destination):
        source, destination = Path(source), Path(destination)
        assert source.exists(), "the staged tier was renamed before it was written"
        assert source.name.endswith(retention.REBUILD_SCRATCH_SUFFIX)
        staged.append((source, destination))
        real_replace(source, destination)

    monkeypatch.setattr(retention.os, "replace", spy)

    assert retention.rebuild_job_tiers([raw], "1278105") is True

    assert {destination for _s, destination in staged} == {minute, hour_path}
    assert all(source.parent == destination.parent for source, destination in staged)


@pytest.mark.parametrize("present_days", [(2,), (0, 2)])
def test_rebuild_keeps_rows_from_retired_days(tmp_path, present_days):
    """Only windows belonging to present raw segments may be replaced."""
    base = int((_NOW - 24 * 86400) // 86400) * 86400
    raw = tmp_path / "deepseek-v4-1-flash-1278105.jsonl"
    all_rows = [
        _row(base + day * 86400 + offset, day * 1000 + index * 100)
        for day in range(3)
        for index, offset in enumerate((600, 1800, 3000))
    ]
    minute, hour_path = retention.tier_paths(raw)
    from imas_ambix.agent.telemetry_store import (
        TIER_HOUR,
        TIER_MINUTE,
        compact_rows,
        write_rows,
    )

    minute_rows = compact_rows(all_rows, tier=TIER_MINUTE)
    hour_rows = compact_rows(minute_rows, tier=TIER_HOUR)
    write_rows(minute, minute_rows)
    write_rows(hour_path, hour_rows)
    retired_days = set(range(3)) - set(present_days)
    for path in (minute, hour_path):
        lines = path.read_bytes().splitlines(keepends=True)
        for index, line in enumerate(lines):
            day = int((retention._timestamp_of(json.loads(line)) - base) // 86400)
            if day in retired_days:
                lines[index] = line.replace(b'":', b'": ', 1)
        path.write_bytes(b"".join(lines))
    before = {
        path: [
            line
            for line in path.read_bytes().splitlines(keepends=True)
            if int((retention._timestamp_of(json.loads(line)) - base) // 86400)
            in retired_days
        ]
        for path in (minute, hour_path)
    }
    assert all(before.values()), "the test built no retired-day tier rows"

    present = []
    for day in present_days:
        segment = tmp_path / f"deepseek-v4-1-flash-1278105.202609{day + 7:02d}.jsonl"
        _write_rows(segment, all_rows[day * 3 : day * 3 + 3])
        present.append(segment)

    assert retention.rebuild_job_tiers(present, "1278105") is True
    for path in (minute, hour_path):
        after = [
            line
            for line in path.read_bytes().splitlines(keepends=True)
            if int((retention._timestamp_of(json.loads(line)) - base) // 86400)
            in retired_days
        ]
        assert after == before[path], f"{path.name} lost retired-day bytes"


def test_the_liveness_query_is_account_wide_bounded_and_takes_no_subprocess(
    monkeypatch,
):
    """A live job may be on any node, so liveness is asked account-wide and bounded.

    The pass asks one question for every candidate at once, through the node
    probe's bounded reader rather than a subprocess of its own, and it does not
    narrow the query to the node the ingest happens to run on.
    """
    asked: list[tuple[tuple[str, ...], float]] = []

    def fake_run_capture(argv, timeout_s=10.0):
        asked.append((tuple(argv), timeout_s))
        return "1281970\n1281971\n"

    monkeypatch.setattr(retention.node_probe, "run_capture", fake_run_capture)

    assert retention.default_running_job_ids() == frozenset({"1281970", "1281971"})

    argv, timeout_s = asked[0]
    assert argv[0] == "squeue"
    assert "-u" in argv
    assert "-w" not in argv, "the liveness question was narrowed to one node"
    assert timeout_s == retention.SCHEDULER_TIMEOUT_SECONDS
    source = Path(retention.__file__).read_text(encoding="utf-8")
    assert "subprocess" not in source
    assert "run_capture" in source


def test_a_scheduler_that_times_out_or_refuses_is_raised_not_read_as_empty(
    monkeypatch,
):
    """An absent scheduler answer is not an empty job table.

    Reading a refused read as "no jobs are running" would make every live serve
    look ended, so both a timeout and a refusal raise out of the liveness read
    and the pass retires nothing.
    """
    monkeypatch.setattr(
        retention.node_probe, "run_capture", lambda argv, timeout_s=10.0: None
    )

    with pytest.raises(RuntimeError):
        retention.default_running_job_ids()


def test_a_rewrite_that_keeps_the_time_and_length_is_caught_by_the_digest(tmp_path):
    """The manifest digest refuses a rewrite the other re-checks cannot see.

    A file can be rewritten in place with its length and modification time
    unchanged, so quietness, age and liveness all still hold. The digest of the
    bytes the listing read is the only re-check that can refuse it, and the pass
    must re-read rather than trust the listing.
    """
    raw, _minute, _hour = _build_job(tmp_path)
    result = retention.scan(tmp_path, now=_NOW, running=_none_running)
    manifest = retention.write_manifest(
        result, manifest_dir=tmp_path / "manifests", now=_NOW
    )

    before = raw.stat()
    rows = [json.loads(line) for line in raw.read_text().splitlines() if line.strip()]
    last = rows[-1]["engine"]["generation_tokens"]
    rows[-1]["engine"]["generation_tokens"] = last + 1.0
    _write_rows(raw, rows)
    os.utime(raw, (before.st_atime, before.st_mtime))
    after = raw.stat()
    assert after.st_size == before.st_size, "the rewrite changed the length"
    assert after.st_mtime == before.st_mtime, "the rewrite moved the time"
    assert after.st_ino == before.st_ino, "the rewrite replaced the file"

    report = retention.apply_manifest(manifest, now=_NOW, running=_none_running)

    assert report.removed == ()
    assert any("bytes changed" in reason for _path, reason in report.skipped)
