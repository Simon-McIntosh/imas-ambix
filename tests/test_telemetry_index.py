"""The derived index reproduces what the record says, and only once."""

from __future__ import annotations

import datetime as _dt
import gzip
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import imas_ambix.agent.telemetry_index as telemetry_index
from imas_ambix.agent.receipt_bins import summarise_receipt_rows
from imas_ambix.agent.telemetry_index import (
    BOOT_SCOPE,
    HOST_SCOPE,
    UNKNOWN_BOOT_ID,
    UNKNOWN_HOST,
    UNKNOWN_HOST_SCOPE,
    TelemetryIndex,
    discover,
    key_scope,
    local_boot_id,
    measure_row,
    receipts_host,
    receipts_job_id,
    resolve_boot_id,
    resolve_host,
    row_boot_id,
)

_BASE = _dt.datetime(2026, 9, 20, 6, 0, 0, tzinfo=_dt.UTC)

# Two canonical boot identifiers in the lowercase form the producer accepts.
_BOOT_BEFORE = "3f8a1c2e-9b4d-4e6f-8a1b-2c3d4e5f6071"
_BOOT_AFTER = "7d2e4f60-1a3b-4c5d-9e8f-0a1b2c3d4e5f"


def _at(seconds: float) -> float:
    return _BASE.timestamp() + seconds


def _row(seconds: float, **overrides: object) -> dict:
    """One recorder-shaped sample, with an interval and a cumulative counter."""
    row = {
        "timestamp": (_BASE + _dt.timedelta(seconds=seconds)).isoformat(),
        "job_id": "1273253",
        "profile_slug": "deepseek-v4-1-flash",
        "served_name": "deepseek-v4.1-flash",
        "gpus": 4,
        "prefix_cache_query_delta": 10 + int(seconds),
        "spec_draft_tokens": 100 + int(seconds),
        "num_requests_running": 12,
        "kv_cache_usage_perc": 0.42,
        "engine": {
            "family": "sglang",
            "generation_tokens": 1000.0 + seconds * 7,
            "prompt_tokens": 500.0 + seconds * 3,
        },
    }
    row.update(overrides)
    return row


def _engine_tokens(value: float) -> dict:
    """A row's engine section carrying one cumulative counter reading."""
    return {"family": "sglang", "generation_tokens": value}


def _write(path: Path, rows: list[dict], **kwargs: object) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _direct_rows(path: Path) -> list[dict]:
    """Read the record the way any other reader would."""
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_reingesting_one_record_inserts_no_second_copy(tmp_path):
    """One record ingested twice is one record: asserted on every path in."""
    source = tmp_path / "serve.jsonl"
    _write(source, [_row(0), _row(5), _row(10)])

    with TelemetryIndex(tmp_path / "index.db") as index:
        first = index.ingest([source])
        after_first = index.sample_count()
        second = index.ingest([source])
        third = index.ingest([source])

        assert first.rows_inserted == 3
        assert after_first == 3
        assert second.rows_inserted == 0
        assert third.rows_inserted == 0
        assert index.sample_count() == after_first
        assert index.sum_measurements("prefix_cache_query_delta", _at(0), _at(11)) == (
            10 + 15 + 20
        )


def test_reingest_after_lost_tail_state_is_still_one_copy(tmp_path):
    """Content identity is the inode and offset, not the tail bookkeeping.

    A state file that is lost -- a fresh index over files already ingested, a
    rebuild that keeps its samples -- must not double-count, which is what the
    offset alone could not guarantee.
    """
    source = tmp_path / "serve.jsonl"
    _write(source, [_row(0), _row(5), _row(10)])

    with TelemetryIndex(tmp_path / "index.db") as index:
        index.ingest([source])
        # Forget how far the file was consumed; the bytes are unchanged.
        with index._conn:
            index._conn.execute("DELETE FROM source")
        report = index.ingest([source])

        assert report.rows_inserted == 0
        assert report.rows_duplicate == 3
        assert index.sample_count() == 3


def test_ingest_resumes_across_a_roll_without_gap_or_double_count(tmp_path):
    """A file rolled mid-stream is neither re-read nor skipped past its roll."""
    records = tmp_path / "serve.jsonl"
    _write(records, [_row(0), _row(5), _row(10)])

    with TelemetryIndex(tmp_path / "index.db") as index:
        # Discovery is the caller's glob; the roll renames the file, so the
        # pattern must reach the name the old bytes now wear.
        index.ingest(discover(tmp_path, "serve.jsonl*"))
        assert index.sample_count() == 3

        # The recorder rolls: the live file keeps its bytes and gains a name,
        # a fresh file takes over the path, and the rolled one keeps growing
        # until its writer notices.
        rolled = tmp_path / "serve.jsonl.1"
        records.rename(rolled)
        _write(records, [_row(15), _row(20)])
        _write(rolled, [_row(12)])
        index.ingest(discover(tmp_path, "serve.jsonl*"))

        # 6 distinct lines written, 6 rows: the three renamed bytes were
        # recognised by inode and offset rather than counted again, and the
        # line appended after the roll was not lost with the old name.
        assert index.sample_count() == 6
        assert index.sum_measurements("prefix_cache_query_delta", _at(0), _at(60)) == (
            10 + 15 + 20 + 22 + 25 + 30
        )


def test_a_path_replaced_by_a_new_file_reads_only_the_new_bytes(tmp_path):
    """A new job's file at the same path is read, and the old one is not."""
    records = tmp_path / "serve.jsonl"
    _write(records, [_row(0), _row(5)])

    with TelemetryIndex(tmp_path / "index.db") as index:
        index.ingest([records])

        # A new job takes over the path: new inode, offset zero, new bytes.
        records.unlink()
        _write(records, [_row(10)])
        index.ingest([records])

        assert index.sample_count() == 3
        assert index.sum_measurements("prefix_cache_query_delta", _at(0), _at(60)) == (
            10 + 15 + 20
        )


def test_restart_resumes_from_inode_and_offset(tmp_path):
    """A new process continues the tail rather than restarting it."""
    records = tmp_path / "serve.jsonl"
    _write(records, [_row(0), _row(5)])
    index_path = tmp_path / "index.db"

    with TelemetryIndex(index_path) as index:
        index.ingest([records])
    with TelemetryIndex(index_path) as reopened:
        assert reopened.ingest([records]).rows_inserted == 0
        _write(records, [_row(10)])
        report = reopened.ingest([records])
        assert report.rows_inserted == 1
        assert report.rows_duplicate == 0
        assert reopened.sample_count() == 3


def test_a_partial_trailing_line_waits_for_its_remainder(tmp_path):
    """A line still being written is left alone, then read exactly once."""
    source = tmp_path / "serve.jsonl"
    _write(source, [_row(0), _row(5)])
    complete = source.stat().st_size
    with source.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_row(10))[:40])

    with TelemetryIndex(tmp_path / "index.db") as index:
        assert index.ingest([source]).rows_inserted == 2
        assert index.sample_count() == 2

        with source.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_row(10))[40:] + "\n")
        assert index.ingest([source]).rows_inserted == 1

        assert index.sample_count() == 3
        assert index.measurement_count("prefix_cache_query_delta") == 3
        span = index.counter_span("engine.generation_tokens", _at(-1), _at(11))
        assert span == pytest.approx(70.0)
    assert source.stat().st_size > complete


def test_a_malformed_complete_line_is_named_not_swallowed(tmp_path):
    """The index refuses a complete line it cannot read, naming where."""
    source = tmp_path / "serve.jsonl"
    _write(source, [_row(0)])
    with source.open("a", encoding="utf-8") as handle:
        handle.write('{"timestamp": "2026-09-20T06:00:05+00:00" \n')

    with (
        TelemetryIndex(tmp_path / "index.db") as index,
        pytest.raises(ValueError, match="invalid record JSON"),
    ):
        index.ingest([source])


def test_a_source_rebuilt_in_place_drops_its_stale_samples(tmp_path):
    """A rewritten file replaces its samples rather than adding to them."""
    source = tmp_path / "serve.jsonl"
    _write(source, [_row(0), _row(5), _row(10)])

    with TelemetryIndex(tmp_path / "index.db") as index:
        index.ingest([source])
        assert index.sample_count() == 3

        # Same inode, smaller file: the bytes past the new end are gone, so
        # the samples they produced are stale.
        os.truncate(source, 0)
        _write(source, [_row(0)])
        index.ingest([source])

        assert index.sample_count() == 1
        kept = index.rows(_at(-1), _at(60))
        assert len(kept) == 1
        # The record's own fields survive the key the read surface adds.
        assert {key: kept[0][key] for key in _row(0)} == _row(0)


def test_period_query_agrees_with_the_jsonl_computed_directly(tmp_path):
    """Token totals are checked against the record, not against the index."""
    records = tmp_path / "serve.jsonl"
    _write(records, [_row(second) for second in range(0, 120, 5)])
    rows = _direct_rows(records)
    start, end = _at(20), _at(70)
    in_window = [row for row in rows if start <= _epoch(row) < end]

    with TelemetryIndex(tmp_path / "index.db") as index:
        index.ingest([records])

        assert index.sum_measurements("spec_draft_tokens", start, end) == sum(
            row["spec_draft_tokens"] for row in in_window
        )
        assert index.sum_measurements("prefix_cache_query_delta", start, end) == sum(
            row["prefix_cache_query_delta"] for row in in_window
        )

        # A cumulative counter's period figure is its endpoint advance, which
        # the direct computation takes from the same record independently.
        opening = max((row for row in rows if _epoch(row) <= start), key=_epoch)
        closing = max((row for row in rows if _epoch(row) < end), key=_epoch)
        assert index.counter_span(
            "engine.generation_tokens", start, end
        ) == pytest.approx(
            closing["engine"]["generation_tokens"]
            - opening["engine"]["generation_tokens"]
        )


def test_an_absent_measurement_is_not_a_zero(tmp_path):
    """Absence and a measured zero are different answers."""
    records = tmp_path / "serve.jsonl"
    _write(
        records,
        [
            _row(0, prefix_cache_query_delta=0),
            _row(5, prefix_cache_query_delta=0),
        ],
    )
    with TelemetryIndex(tmp_path / "index.db") as index:
        index.ingest([records])
        assert (
            index.sum_measurements("prefix_cache_query_delta", _at(0), _at(10)) == 0.0
        )
        assert index.sum_measurements("prefix_cache_hit_delta", _at(0), _at(10)) is None
        assert index.sum_measurements("spec_accepted_tokens", _at(0), _at(10)) is None


def test_measure_row_keeps_only_observed_numbers():
    """Nulls, strings and booleans are not measurements."""
    measured = measure_row(
        {
            "timestamp": "2026-09-20T06:00:00+00:00",
            "job_id": "1273253",
            "gpus": 4,
            "kv_cache_usage_perc": None,
            "prefix_cache_query_delta": 0,
            "engine": {"family": "sglang", "generation_tokens": 12.5},
            "spec_num_accepted_per_pos": [3, 4],
        }
    )
    assert measured == {
        "gpus": 4.0,
        "prefix_cache_query_delta": 0.0,
        "engine.generation_tokens": 12.5,
        "spec_num_accepted_per_pos.0": 3.0,
        "spec_num_accepted_per_pos.1": 4.0,
    }


def test_time_weighted_mean_weights_each_gauge_by_time_held(tmp_path):
    """A gauge mean is weighted by how long each reading stood."""
    records = tmp_path / "serve.jsonl"
    _write(
        records,
        [
            _row(0, kv_cache_usage_perc=0.10),
            _row(10, kv_cache_usage_perc=0.50),
            _row(20, kv_cache_usage_perc=0.90),
        ],
    )
    with TelemetryIndex(tmp_path / "index.db") as index:
        index.ingest([records])
        # 10 s at 0.10 then 10 s at 0.50 over the window; the third sample
        # stands for no time inside it.
        assert index.time_weighted_mean("kv_cache_usage_perc", _at(0), _at(20)) == (
            pytest.approx(0.30)
        )
        assert index.time_weighted_mean("kv_cache_usage_perc", _at(0), _at(10)) == (
            pytest.approx(0.10)
        )
        assert index.time_weighted_mean("kv_other", _at(0), _at(20)) is None


def test_deleting_and_rebuilding_reproduces_every_query(tmp_path):
    """The index is disposable: a rebuild answers identically."""
    records = tmp_path / "serve.jsonl"
    _write(records, [_row(second) for second in range(0, 60, 5)])
    index_path = tmp_path / "index.db"

    def probe(index: TelemetryIndex) -> dict:
        return {
            "count": index.sample_count(),
            "draft": index.sum_measurements("spec_draft_tokens", _at(0), _at(60)),
            "queries": index.sum_measurements(
                "prefix_cache_query_delta", _at(15), _at(45)
            ),
            "generation": index.counter_span(
                "engine.generation_tokens", _at(10), _at(50)
            ),
            "kv_mean": index.time_weighted_mean("kv_cache_usage_perc", _at(0), _at(55)),
            "absent": index.sum_measurements("prefix_cache_hit_delta", _at(0), _at(60)),
        }

    with TelemetryIndex(index_path) as index:
        index.ingest([records])
        before = probe(index)

    index_path.unlink()
    with TelemetryIndex(index_path) as rebuilt:
        report = rebuilt.ingest([records])
        after = probe(rebuilt)

    assert report.rows_inserted == 12
    assert before == after


def test_rebuild_in_place_matches_a_fresh_index(tmp_path):
    """A rebuild after a schema change reproduces the same answers."""
    records = tmp_path / "serve.jsonl"
    _write(records, [_row(second) for second in range(0, 30, 5)])

    with TelemetryIndex(tmp_path / "index.db") as index:
        index.ingest([records])
        index.rebuild([records])
        assert index.sample_count() == 6
        assert index.sum_measurements("spec_draft_tokens", _at(0), _at(30)) == sum(
            row["spec_draft_tokens"] for row in _direct_rows(records)
        )


def test_receipt_bins_reads_its_rows_from_the_index(tmp_path):
    """The interval aggregation has one owner, whichever store asks."""
    records = tmp_path / "serve.jsonl"
    rows = [
        _row(
            second,
            num_requests_running=12,
            generation_throughput_toks_per_s=480.0,
            prompt_throughput_toks_per_s=120.0,
            engine={
                "family": "sglang",
                "generation_tokens": 1000.0 + second * 7,
                "prompt_tokens": 500.0 + second * 3,
                "prefix_cache_hit_rate": 0.25,
            },
        )
        for second in (0, 5)
    ]
    _write(records, rows)

    with TelemetryIndex(tmp_path / "index.db") as index:
        index.ingest([records])
        report = index.receipt_bins(_at(-1), _at(60))

    assert report == summarise_receipt_rows(rows)
    assert report.rows_read == 2
    assert report.bins[0].intervals == 2


def test_dropping_a_sample_cascades_to_its_measurements(tmp_path):
    """The declared foreign key is enforced: a dropped sample leaves no orphan."""
    source = tmp_path / "serve.jsonl"
    _write(source, [_row(0)])
    with TelemetryIndex(tmp_path / "index.db") as index:
        index.ingest([source])
        assert index.ingest([source]).rows_inserted == 0
        assert index.measurement_count("spec_draft_tokens") == 1

        with index._conn:
            index._conn.execute("DELETE FROM sample")

        assert index.measurement_count("spec_draft_tokens") == 0
        assert (
            index._conn.execute(
                "SELECT COUNT(*) FROM measurement WHERE sample_id NOT IN "
                "(SELECT id FROM sample)"
            ).fetchone()[0]
            == 0
        )


def test_a_rewrite_that_is_not_shorter_still_replaces_its_samples(tmp_path):
    """A file rewritten in place at the same length is caught by content.

    Length alone cannot separate an append from a replacement: the rewritten
    file here is byte-for-byte the same size as the one already ingested, so a
    size comparison passes and the index would answer 60.0 from the values the
    discarded content produced rather than the 150.0 the file now carries.
    """
    source = tmp_path / "serve.jsonl"
    _write(source, [_row(0), _row(10), _row(20)])
    consumed_size = source.stat().st_size

    with TelemetryIndex(tmp_path / "index.db") as index:
        index.ingest([source])
        assert (
            index.sum_measurements("prefix_cache_query_delta", _at(-1), _at(60))
            == 10 + 20 + 30
        )

        source.write_text(
            "".join(
                json.dumps(row) + "\n"
                for row in (
                    _row(0, prefix_cache_query_delta=40),
                    _row(10, prefix_cache_query_delta=50),
                    _row(20, prefix_cache_query_delta=60),
                )
            ),
            encoding="utf-8",
        )
        assert source.stat().st_size == consumed_size

        index.ingest([source])
        assert index.sample_count() == 3
        assert (
            index.sum_measurements("prefix_cache_query_delta", _at(-1), _at(60))
            == 40 + 50 + 60
        )


def test_running_totals_are_differenced_and_intervals_are_summed(tmp_path):
    """The counter-versus-interval split, read through the two query shapes."""
    source = tmp_path / "serve.jsonl"
    _write(
        source,
        [
            _row(0, requests_seen_total=1000.0),
            _row(10, requests_seen_total=1100.0),
            _row(20, requests_seen_total=1250.0),
        ],
    )
    with TelemetryIndex(tmp_path / "index.db") as index:
        index.ingest([source])

        # Both cumulative routes in one assertion, so a red log shows WHICH
        # clause gave way: the suffix route alone, the listed route alone, or
        # both. Split across two assertions, either mutation would report the
        # same first failure, and the log could not say which clause was held.
        assert (
            index.sum_measurements("requests_seen_total", _at(0), _at(30)),
            index.sum_measurements("engine.generation_tokens", _at(0), _at(30)),
        ) == (None, None)
        assert index.counter_span("requests_seen_total", _at(-1), _at(30)) == 250.0

        assert index.measurement_count("engine.generation_tokens") == 3

        # A per-interval quantity is still a sum over the window.
        assert index.sum_measurements("spec_draft_tokens", _at(0), _at(30)) == (
            100 + 110 + 120
        )


def test_a_window_spanning_two_serves_totals_the_sum_of_its_runs(tmp_path):
    """A cumulative total is summed run by run, not read off two endpoints.

    A serve restart resets the counter, so the first reading of the window and
    the last belong to different runs. Differencing those two endpoints yields a
    figure no counter ever advanced -- negative here, and positive when the
    newer serve happens to be the further ahead -- and either reads exactly like
    a measured total. The window is partitioned at the restart instead, each run
    is differenced between its own endpoints, and the differences are summed.
    """
    early = tmp_path / "serve-a.jsonl"
    later = tmp_path / "serve-b.jsonl"
    _write(
        early,
        [
            _row(0, engine=_engine_tokens(1000.0)),
            _row(5, engine=_engine_tokens(1100.0)),
        ],
    )
    _write(
        later,
        [
            _row(10, job_id="1273254", engine=_engine_tokens(50.0)),
            _row(15, job_id="1273254", engine=_engine_tokens(150.0)),
        ],
    )

    with TelemetryIndex(tmp_path / "index.db") as index:
        index.ingest([early, later])

        # Two endpoints straddling the restart difference negative; that is the
        # reading the partition exists to replace.
        assert index.counter_span("engine.generation_tokens", _at(0), _at(20)) < 0

        partition = index.partitioned_total("engine.generation_tokens", _at(0), _at(20))
        assert partition.runs == 2
        # 1100-1000 from the first serve, 150-50 from the second.
        assert partition.total == pytest.approx(200.0)


def test_the_coverage_beside_a_total_is_the_union_of_its_runs(tmp_path):
    """Coverage is what the contributing runs account for, not the window.

    The two figures describe the same thing by construction, so a window far
    longer than the record that fills it must report the runs' own span. Laying the
    two serves end to end over a short stretch of a long window is the shape
    that would otherwise print the window's nominal length beside a much smaller
    total.
    """
    early = tmp_path / "serve-c.jsonl"
    later = tmp_path / "serve-d.jsonl"
    _write(
        early,
        [
            _row(0, engine=_engine_tokens(1000.0)),
            _row(5, engine=_engine_tokens(1100.0)),
        ],
    )
    _write(
        later,
        [
            _row(10, job_id="1273255", engine=_engine_tokens(50.0)),
            _row(15, job_id="1273255", engine=_engine_tokens(150.0)),
        ],
    )

    with TelemetryIndex(tmp_path / "index.db") as index:
        index.ingest([early, later])
        partition = index.partitioned_total(
            "engine.generation_tokens", _at(0), _at(100)
        )
        # Each run spans 5 s of the 100 s window, and the union is their sum
        # because the runs are disjoint.
        assert partition.coverage == pytest.approx(10.0)
        assert partition.coverage != _at(100) - _at(0)


def test_the_coverage_counts_a_shared_stretch_of_time_once(tmp_path, monkeypatch):
    """The covered figure is a union: an overlap is counted once, not twice.

    Two recorders writing over one stretch of wall clock cover that stretch once
    between them, and summing their spans instead would report 40 s over a
    stretch of 30 s, reading every rate whose denominator this figure is low by
    the overlap. Spans that merely touch cover one continuous stretch, since
    nothing is uncovered across the join.

    The arithmetic is asserted directly because the overlap it rules out cannot be
    built through the store: a run is a contiguous slice of a time-ordered scan,
    so its clipped span can touch a neighbour's but never cross it. The caller is
    therefore asserted separately -- a total whose coverage came from summing the
    spans rather than from the union would leave the spy unentered, which is the
    regression the arithmetic alone cannot see.
    """
    assert telemetry_index._union_length([(0.0, 20.0), (10.0, 30.0)]) == pytest.approx(
        30.0
    )
    assert telemetry_index._union_length([(0.0, 20.0), (20.0, 40.0)]) == pytest.approx(
        40.0
    )
    assert telemetry_index._union_length([]) == 0.0

    seen: list[list[tuple[float, float]]] = []
    union = telemetry_index._union_length

    def spy(spans):
        seen.append(list(spans))
        return union(spans)

    monkeypatch.setattr(telemetry_index, "_union_length", spy)

    early = tmp_path / "serve-e.jsonl"
    later = tmp_path / "serve-f.jsonl"
    _write(
        early,
        [
            _row(0, engine=_engine_tokens(1000.0)),
            _row(60, engine=_engine_tokens(1300.0)),
        ],
    )
    _write(
        later,
        [
            _row(120, job_id="1273299", engine=_engine_tokens(5.0)),
            _row(180, job_id="1273299", engine=_engine_tokens(25.0)),
        ],
    )

    with TelemetryIndex(tmp_path / "index.db") as index:
        index.ingest([early, later])
        index.partitioned_total("engine.generation_tokens", _at(-100), _at(600))

    assert seen, "coverage must be the union of the runs' spans"
    assert len(seen[0]) == 2
    assert [span[0] for span in seen[0]] == pytest.approx([_at(0), _at(120)])
    assert [span[1] for span in seen[0]] == pytest.approx([_at(60), _at(180)])


def test_a_lone_reading_in_the_window_is_effective_only_as_one_endpoint(tmp_path):
    """One reading cannot be differenced, so the total is absent and not zero.

    A reading inside the window with no earlier one to difference it against is
    one endpoint rather than two, so no run of the window advanced by a
    measurable amount and the total is ``None``. Reporting ``0.0`` there would
    be a claim the record does not support -- the figure a run observed and not
    advancing earns -- and a caller reading it as "no data" would understate the
    period instead of declaring it unknown.
    """
    source = tmp_path / "serve.jsonl"
    _write(source, [_row(10, engine=_engine_tokens(1000.0))])

    with TelemetryIndex(tmp_path / "index.db") as index:
        index.ingest([source])
        partition = index.partitioned_total(
            "engine.generation_tokens", _at(0), _at(100)
        )
        assert partition.total is None
        assert partition.runs == 0
        assert partition.coverage == 0.0


def test_default_discovery_reaches_a_rolled_file(tmp_path):
    """The default pattern follows the naming a roll produces."""
    records = tmp_path / "serve.jsonl"
    _write(records, [_row(0), _row(5)])
    records.rename(tmp_path / "serve.jsonl.1")
    _write(records, [_row(10)])

    assert [path.name for path in discover(tmp_path)] == [
        "serve.jsonl",
        "serve.jsonl.1",
    ]

    with TelemetryIndex(tmp_path / "index.db") as index:
        index.ingest(discover(tmp_path))
        assert index.sample_count() == 3
        assert index.sum_measurements("prefix_cache_query_delta", _at(0), _at(20)) == (
            10 + 15 + 20
        )


def test_a_rewrite_reusing_the_last_line_still_replaces_its_samples(tmp_path):
    """The final line surviving a rewrite is not evidence the region did.

    Asserted on the file rather than on the code: the rewriting text is the
    same length as what it replaces and its last line is byte-identical, so a
    check against the file's size, or against a digest of that last line,
    reports no change while the values the index holds belong to content the
    file no longer carries.
    """
    source = tmp_path / "serve.jsonl"
    _write(source, [_row(0), _row(10), _row(20)])
    consumed_size = source.stat().st_size
    last_line = source.read_text(encoding="utf-8").splitlines()[-1]

    with TelemetryIndex(tmp_path / "index.db") as index:
        index.ingest([source])
        assert index.sum_measurements("prefix_cache_query_delta", _at(0), _at(60)) == (
            10 + 20 + 30
        )

        rewritten = [
            _row(0, prefix_cache_query_delta=40),
            _row(10, prefix_cache_query_delta=50),
            _row(20, prefix_cache_query_delta=30),
        ]
        source.write_text(
            "".join(json.dumps(row) + "\n" for row in rewritten), encoding="utf-8"
        )
        assert source.stat().st_size == consumed_size
        assert source.read_text(encoding="utf-8").splitlines()[-1] == last_line

        index.ingest([source])
        assert index.sample_count() == 3
        assert index.sum_measurements("prefix_cache_query_delta", _at(0), _at(60)) == (
            40 + 50 + 30
        )


def test_a_source_with_no_recorded_identity_is_read_again(tmp_path):
    """An unknown consumed region is re-read, not treated as unchanged.

    The recorded digest is cleared the way a source row written before the
    column existed would carry it, and the file is then rewritten at its own
    length. A missing identity compares equal to nothing, so treating NULL as
    agreement answers with the old content's values: 60 where the file sums to
    120.
    """
    source = tmp_path / "serve.jsonl"
    _write(source, [_row(0), _row(10), _row(20)])

    with TelemetryIndex(tmp_path / "index.db") as index:
        index.ingest([source])
        assert index.sum_measurements("prefix_cache_query_delta", _at(-1), _at(60)) == (
            10 + 20 + 30
        )
        with index._conn:
            index._conn.execute("UPDATE source SET prefix_sha = NULL")

        rewritten = [_row(0), _row(10), _row(20, prefix_cache_query_delta=90)]
        assert len("".join(json.dumps(row) + "\n" for row in rewritten)) == (
            source.stat().st_size
        )
        source.write_text(
            "".join(json.dumps(row) + "\n" for row in rewritten), encoding="utf-8"
        )

        index.ingest([source])
        assert index.sample_count() == 3
        assert index.sum_measurements("prefix_cache_query_delta", _at(-1), _at(60)) == (
            10 + 20 + 90
        )


def test_discovery_passes_over_a_name_that_merely_contains_jsonl(tmp_path):
    """Only the roll suffixes are selected; a longer name is not a record.

    A file whose name carries the suffix inside it is not one of the names a
    roll produces, so it is not offered to the parser. Selecting it costs the
    whole pass: the records beside it are never consumed.
    """
    records = tmp_path / "serve.jsonl"
    _write(records, [_row(0), _row(5)])
    records.rename(tmp_path / "serve.jsonl.1")
    _write(records, [_row(10)])
    (tmp_path / "my-notes.jsonl.summary").write_text("not a record\n", encoding="utf-8")

    assert [path.name for path in discover(tmp_path)] == [
        "serve.jsonl",
        "serve.jsonl.1",
    ]

    with TelemetryIndex(tmp_path / "index.db") as index:
        index.ingest(discover(tmp_path))
        assert index.sample_count() == 3
        assert index.sum_measurements("prefix_cache_query_delta", _at(0), _at(20)) == (
            10 + 15 + 20
        )


def test_a_compressed_roll_is_refused_by_name(tmp_path):
    """A name the index cannot read is named and refused, never counted as zero.

    Selected but unreadable would report a source with no rows and no bytes,
    which is what a healthy empty file reports; refused by name, the caller
    learns which file it has to decompress.
    """
    records = tmp_path / "serve.jsonl"
    _write(records, [_row(0)])
    rolled = tmp_path / "serve.jsonl.2.gz"
    with gzip.open(rolled, "wb") as handle:
        for seconds in (5, 10):
            handle.write(json.dumps(_row(seconds)).encode("utf-8") + b"\n")

    assert [path.name for path in discover(tmp_path)] == [
        "serve.jsonl",
        "serve.jsonl.2.gz",
    ]

    with TelemetryIndex(tmp_path / "index.db") as index:
        with pytest.raises(ValueError) as raised:
            index.ingest(discover(tmp_path))
        assert "compressed roll" in str(raised.value)
        assert rolled.name in str(raised.value)
        assert index.sample_count() == 1


def _epoch(row: dict) -> float:
    return _dt.datetime.fromisoformat(row["timestamp"]).timestamp()


def _pin_inode(monkeypatch, inode: int) -> None:
    """Give every path the same inode, as two filesystems' numbering can."""
    real_stat = Path.stat

    def stat(path: Path, *args: object, **kwargs: object) -> os.stat_result:
        fields = list(real_stat(path, *args, **kwargs))
        fields[1] = inode
        return os.stat_result(fields)

    monkeypatch.setattr(Path, "stat", stat)


def test_a_row_naming_its_host_is_keyed_on_that_host(tmp_path):
    """A record that says where it was written is not attributed to its reader.

    A host the row records is the identity it is keyed by, whatever the index
    was told for the hostless case, or two hosts' readings would land under one
    key and the second writer's row would be dropped as a duplicate.
    """
    source = tmp_path / "serve.jsonl"
    _write(source, [_row(0, host="node-a"), _row(5, host="node-a")])

    with TelemetryIndex(tmp_path / "index.db", host="node-z") as index:
        index.ingest([source])
        kept = index._conn.execute("SELECT host FROM sample ORDER BY id").fetchall()
        assert [row["host"] for row in kept] == ["node-a", "node-a"]
        sources = index._conn.execute("SELECT host, path FROM source").fetchall()
        assert [(row["host"], row["path"]) for row in sources] == [
            ("node-a", str(source))
        ]


def test_a_row_naming_no_host_is_keyed_under_the_unknown_host_marker(tmp_path):
    """A record that names no machine has not said where it was written.

    A receipts directory on shared storage collects files written on the node
    that ran the serve and read from a login node, so the reading machine's
    nodename is not evidence of the writer. Adopting it keys the row under an
    identity nothing recorded, and -- because the same fallback also supplied
    the boot -- presents the reader's boot as the row's, so the row reads as
    keyed on a machine and a boot that are both inferred rather than recorded.
    """
    here = tmp_path / "plain.jsonl"
    named = tmp_path / "named.jsonl"
    _write(here, [_row(0), _row(5)])
    _write(named, [_row(5)])

    with TelemetryIndex(tmp_path / "own.db") as index:
        index.ingest([here])
        stored = index._conn.execute(
            "SELECT host, boot_id, key_kind FROM sample ORDER BY id"
        ).fetchall()
        assert [(row["host"], row["boot_id"], row["key_kind"]) for row in stored] == [
            (UNKNOWN_HOST, UNKNOWN_BOOT_ID, UNKNOWN_HOST_SCOPE),
            (UNKNOWN_HOST, UNKNOWN_BOOT_ID, UNKNOWN_HOST_SCOPE),
        ]
        # The reader's own identity appears nowhere in the row a caller holds.
        assert os.uname().nodename not in json.dumps(index.rows(_at(-1), _at(60)))
        assert "host+boot" not in json.dumps(index.rows(_at(-1), _at(60)))

    with TelemetryIndex(tmp_path / "other.db", host="node-b") as index:
        index.ingest([named])
        row = index._conn.execute("SELECT host FROM sample").fetchone()
        assert row["host"] == "node-b"


def test_the_host_key_is_the_recorded_one_or_the_unknown_marker():
    """A host enters the key only as recorded; the reader's name never does.

    The unknown-host marker is the outer dimension of the key kind: a row with
    no host of its own is announced as unknown-host whatever its boot resolved
    to, because a boot identity names one uninterrupted run of counters on one
    machine and a record that never named the machine has no such run.
    """
    assert resolve_host("node-a") == "node-a"
    assert resolve_host(None) == UNKNOWN_HOST
    assert resolve_host("") == UNKNOWN_HOST
    assert key_scope("node-a", BOOT_SCOPE) == BOOT_SCOPE
    assert key_scope("node-a", HOST_SCOPE) == HOST_SCOPE
    assert key_scope(UNKNOWN_HOST, BOOT_SCOPE) == UNKNOWN_HOST_SCOPE
    assert key_scope(UNKNOWN_HOST, HOST_SCOPE) == UNKNOWN_HOST_SCOPE


def test_a_receipts_name_yields_the_job_id_it_carries():
    """The digits before the ``.jsonl`` suffix, and nothing that merely looks
    like one."""
    assert receipts_job_id("deepseek-v4-1-flash-1271709.jsonl") == "1271709"
    assert receipts_job_id("/shared/receipts/serve-9.jsonl.1") == "9"
    assert receipts_job_id("serve-9.jsonl.1.gz") == "9"
    assert receipts_job_id("serve.jsonl") is None
    # A tier file is a derived sibling, not a receipts file's own name.
    assert receipts_job_id("summary-1271709.minute.jsonl") is None


def test_the_receipts_host_is_resolved_from_the_job_id(monkeypatch):
    """The job's node list is read parsably and expanded to one node per line.

    Two reads, because the scheduler compresses an allocation's nodes into a
    single hostlist token: ``-P`` keeps that token from being cut short at a
    column width, and ``scontrol show hostnames`` turns it into one line per
    machine. A one-node job passes through the expansion unchanged.
    """
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        if argv[0] == "sacct":
            return SimpleNamespace(returncode=0, stdout="98dci4-gpu-0003\n", stderr="")
        assert argv == ["scontrol", "show", "hostnames", "98dci4-gpu-0003"]
        return SimpleNamespace(returncode=0, stdout="98dci4-gpu-0003\n", stderr="")

    monkeypatch.setattr(telemetry_index.subprocess, "run", fake_run)
    assert receipts_host("/shared/receipts/deepseek-v4-1-flash-1271709.jsonl") == (
        "98dci4-gpu-0003"
    )
    assert calls == [
        ["sacct", "-j", "1271709", "-X", "-n", "-P", "-o", "NodeList"],
        ["scontrol", "show", "hostnames", "98dci4-gpu-0003"],
    ]


def test_a_job_that_does_not_resolve_is_refused_not_guessed(monkeypatch):
    """A name with no job id, a purged job, a placeholder, and a many-node job
    all refuse.

    A guess here is a key silently asserting a machine no row recorded, so
    every unresolved shape raises rather than falling back to the reader. The
    many-node case is the one the scheduler's own output makes easy to miss: a
    fourteen-node allocation is reported as a single compressed token, so
    counting the fields of that report sees one host, and the same token at the
    default column width reads ``98dci4-clu-[50+`` -- still one field, and not
    a hostname at all. Counting the lines of the expanded list is what separates
    one machine from many, and both shapes are refused here. The single line is
    not sufficient on its own either: a field ``sacct`` leaves empty prints the
    word ``None``, and ``scontrol`` echoes that word back as one line rather
    than rejecting it, so a placeholder is refused by name and an echoed token
    by its shape.
    """
    with pytest.raises(ValueError):
        receipts_host("serve.jsonl")

    def scheduler(
        sacct_stdout, hosts_stdout="", sacct_rc=0, hosts_rc=0, hosts_stderr=""
    ):
        def run(argv, **kwargs):
            if argv[0] == "sacct":
                return SimpleNamespace(
                    returncode=sacct_rc, stdout=sacct_stdout, stderr=""
                )
            return SimpleNamespace(
                returncode=hosts_rc, stdout=hosts_stdout, stderr=hosts_stderr
            )

        return run

    fourteen_nodes = "".join(f"98dci4-clu-{n}\n" for n in range(5073, 5087))

    for fake in (
        # Purged from sacct: a job that does not exist reports nothing and
        # exits zero, and the empty report names no node list.
        scheduler("", sacct_rc=0),
        # A field sacct left empty prints a placeholder, which scontrol echoes
        # back as one line rather than refusing.
        scheduler("None\n", "None\n"),
        # Fourteen nodes compressed into one token -- one field, many machines.
        scheduler("98dci4-clu-[5073-5086]\n", fourteen_nodes),
        # The same token truncated at the default column width: scontrol rejects
        # it on stderr, exits 0, and prints no node at all.
        scheduler(
            "98dci4-clu-[50+\n",
            "",
            hosts_stderr="Invalid hostlist: 98dci4-clu-[50+\n",
        ),
    ):
        monkeypatch.setattr(telemetry_index.subprocess, "run", fake)
        with pytest.raises(ValueError):
            receipts_host("deepseek-v4-1-flash-1271709.jsonl")


def test_an_unavailable_scheduler_refuses_the_promised_exception(monkeypatch):
    """A missing ``sacct`` or ``scontrol`` refuses with ``ValueError``.

    The tools may simply not be on the path, and the docstring promises a
    caller the same exception for that as for an unresolvable host: a caller
    catching ``ValueError`` to fall back to the unknown-host marker must not
    meet an ``OSError`` instead, which would abort the ingest rather than
    degrade it.
    """

    def missing(tool):
        def run(argv, **kwargs):
            if argv[0] == tool:
                raise FileNotFoundError(2, "No such file or directory", tool)
            return SimpleNamespace(returncode=0, stdout="98dci4-gpu-0003\n", stderr="")

        return run

    monkeypatch.setattr(telemetry_index.subprocess, "run", missing("sacct"))
    with pytest.raises(ValueError):
        receipts_host("deepseek-v4-1-flash-1271709.jsonl")

    monkeypatch.setattr(telemetry_index.subprocess, "run", missing("scontrol"))
    with pytest.raises(ValueError):
        receipts_host("deepseek-v4-1-flash-1271709.jsonl")


def test_two_hosts_at_one_inode_and_offset_are_both_kept(tmp_path, monkeypatch):
    """An inode names a file within one filesystem and nowhere else.

    Two machines recording a file of the same name at the same path produce the
    same ``(inode, offset)`` for readings that are not the same, so an index
    keyed on that pair alone stores the second machine's row as a duplicate of
    the first's and drops it -- no error, no count of what was lost, and nothing
    afterwards that tells the two apart. The inode is pinned here to reproduce
    what two filesystems' independent numbering makes collide in the record this
    index reads.
    """
    _pin_inode(monkeypatch, 424_242)
    here = tmp_path / "here" / "serve.jsonl"
    there = tmp_path / "there" / "serve.jsonl"
    here.parent.mkdir()
    there.parent.mkdir()
    _write(here, [_row(0, host="node-a"), _row(5, host="node-a")])
    _write(there, [_row(0, host="node-b"), _row(5, host="node-b")])

    with TelemetryIndex(tmp_path / "index.db", host="node-a") as index:
        report = index.ingest([here, there])

        assert report.rows_inserted == 4
        assert report.rows_duplicate == 0
        kept = index._conn.execute(
            "SELECT host, COUNT(*) AS n FROM sample GROUP BY host ORDER BY host"
        ).fetchall()
        assert [(row["host"], row["n"]) for row in kept] == [
            ("node-a", 2),
            ("node-b", 2),
        ]
        assert index._conn.execute("SELECT COUNT(*) FROM source").fetchone()[0] == 2
        assert index.sample_count() == 4


def test_two_boots_of_one_host_at_one_inode_and_offset_are_both_kept(
    tmp_path, monkeypatch
):
    """A hostname does not survive a reboot, and the counters prove it.

    The counters this record carries are cumulative, so they reset when the
    machine reboots. Two readings from one genuinely correct hostname on either
    side of a reboot therefore difference as a large drop, and nothing on either
    row says a reboot happened -- a real measurement of the wrong quantity.

    The inode is pinned here because two files on one filesystem cannot collide:
    the record this index reads is two machines', or one machine's across two
    boots, where the same path and the same inode occur in two filesystems that
    hand out their own numbering. Measured on this node, a search of 4,000
    candidates in /tmp (xfs, inodes from about 5,000) against the pytest temp
    filesystem (inodes near 39,761,983) found no match: the ranges do not
    overlap, so a collision is contrived here rather than waited for.
    """
    _pin_inode(monkeypatch, 424_242)
    before = tmp_path / "before" / "serve.jsonl"
    after = tmp_path / "after" / "serve.jsonl"
    before.parent.mkdir()
    after.parent.mkdir()
    _write(
        before,
        [
            _row(0, host="node-a", boot_id=_BOOT_BEFORE),
            _row(5, host="node-a", boot_id=_BOOT_BEFORE),
        ],
    )
    _write(
        after,
        [
            _row(0, host="node-a", boot_id=_BOOT_AFTER),
            _row(5, host="node-a", boot_id=_BOOT_AFTER),
        ],
    )

    with TelemetryIndex(tmp_path / "index.db", host="node-a") as index:
        report = index.ingest([before, after])

        assert report.rows_inserted == 4
        assert report.rows_duplicate == 0
        kept = index._conn.execute(
            "SELECT boot_id, COUNT(*) AS n FROM sample "
            "GROUP BY boot_id ORDER BY boot_id"
        ).fetchall()
        assert [(row["boot_id"], row["n"]) for row in kept] == [
            (_BOOT_BEFORE, 2),
            (_BOOT_AFTER, 2),
        ]
        # The boot is part of the source key, so one file spanning a reboot is
        # two sources rather than one whose rows contradict each other.
        assert (
            index._conn.execute(
                "SELECT COUNT(*) FROM source WHERE key_kind = ?", (BOOT_SCOPE,)
            ).fetchone()[0]
            == 2
        )
        assert index.sample_count() == 4


def test_a_row_with_an_unusable_boot_is_keyed_by_its_host_and_says_so(tmp_path):
    """A boot the producer refuses is a degraded key, announced and not dropped.

    The producer's parser raises on anything but a canonical lowercase
    identifier, and that strictness is not the index's to loosen. What the index
    must not do is store such a row as though it were keyed: an unkeyed row that
    looks keyed compares against keyed rows with nothing to say the guarantee
    holds on one side only.
    """
    source = tmp_path / "serve.jsonl"
    _write(
        source,
        [_row(0, host="node-a", boot_id="not-a-uuid"), _row(5, host="node-a")],
    )

    with TelemetryIndex(tmp_path / "index.db", host="node-z") as index:
        report = index.ingest([source])

        assert report.rows_inserted == 2
        rows = index._conn.execute(
            "SELECT host, boot_id, key_kind FROM sample ORDER BY id"
        ).fetchall()
        assert [(row["host"], row["boot_id"], row["key_kind"]) for row in rows] == [
            ("node-a", UNKNOWN_BOOT_ID, HOST_SCOPE),
            ("node-a", UNKNOWN_BOOT_ID, HOST_SCOPE),
        ]


def test_a_boot_identity_is_used_as_a_key_only_in_the_form_it_was_validated():
    """Every spelling the producer refuses degrades the key; the canonical one keys it.

    The empty string and an uppercase canonical identifier are both refused by
    the same shape check, so they degrade rather than half-key: a row stored
    under an uppercase spelling would never match its own lowercase spelling.
    """
    assert resolve_boot_id(_BOOT_BEFORE) == (_BOOT_BEFORE, BOOT_SCOPE)
    assert resolve_boot_id(None) == (UNKNOWN_BOOT_ID, HOST_SCOPE)
    for refused in ("not-a-uuid", UNKNOWN_BOOT_ID, _BOOT_BEFORE.upper()):
        assert resolve_boot_id(refused) == (UNKNOWN_BOOT_ID, HOST_SCOPE)


def test_a_row_carrying_its_identity_in_a_host_section_is_keyed_on_it(tmp_path):
    """The producer states the machine and the boot in a section of its own.

    Reading only the row's top level would leave the key inert against the one
    producer here that emits a boot identity at all, so the section is read for
    both -- and a host or boot named inside any other section is not this row's.
    """
    source = tmp_path / "serve.jsonl"
    _write(
        source,
        [
            _row(
                0,
                host={
                    "hostname": "node-a",
                    "boot_id": _BOOT_BEFORE,
                    "uptime_seconds": 12.5,
                },
            )
        ],
    )

    with TelemetryIndex(tmp_path / "index.db", host="node-z") as index:
        index.ingest([source])
        row = index._conn.execute(
            "SELECT host, boot_id, key_kind FROM sample"
        ).fetchone()
        assert (row["host"], row["boot_id"], row["key_kind"]) == (
            "node-a",
            _BOOT_BEFORE,
            BOOT_SCOPE,
        )


def test_the_recording_boot_is_the_one_this_process_is_on():
    """A boot identity read here is the producer's, or absent rather than raised.

    The value is read through the producer's own reader so a boot identifier
    means one thing in the record and in whatever consumes it; a machine with
    none to read is a state to key around, not one to fail on.
    """
    here = local_boot_id()
    assert here is None or resolve_boot_id(here) == (here, BOOT_SCOPE)
    assert row_boot_id({"host": {"boot_id": _BOOT_BEFORE}}) == _BOOT_BEFORE
    assert row_boot_id({"boot_id": _BOOT_AFTER}) == _BOOT_AFTER
    assert row_boot_id({"host": {"hostname": "node-a"}}) is None


def test_a_counter_span_between_two_hosts_is_declined_not_differenced(tmp_path):
    """A counter's advance is one machine's, so two hosts are not differenced.

    A window opening on one machine's reading and closing on another's yields
    the difference of two numbers that never described one counter: negative
    when the later reading is the further behind, positive when it is the
    further ahead, and both read exactly like a measured advance. The host is
    carried per file -- one recorder writes one file -- so two machines
    recording one path arrive as two files, which is the shape built here.
    """
    early = tmp_path / "early.jsonl"
    later = tmp_path / "later.jsonl"
    _write(
        early,
        [
            _row(0, host="node-a", engine=_engine_tokens(1000.0)),
            _row(5, host="node-a", engine=_engine_tokens(1100.0)),
        ],
    )
    _write(later, [_row(10, host="node-b", engine=_engine_tokens(50.0))])
    late_first = tmp_path / "late-first.jsonl"
    first = tmp_path / "first.jsonl"
    _write(late_first, [_row(0, host="node-b", engine=_engine_tokens(50.0))])
    _write(first, [_row(10, host="node-a", engine=_engine_tokens(1000.0))])

    with TelemetryIndex(tmp_path / "index.db") as index:
        index.ingest([early, later])
        # Both machines are in the index under their own keys, so the windows
        # below are read against a genuinely mixed record and not one host's.
        hosts = index._conn.execute(
            "SELECT DISTINCT host FROM sample ORDER BY host"
        ).fetchall()
        assert [row["host"] for row in hosts] == ["node-a", "node-b"]
        # One machine's readings alone still difference to its own advance.
        assert index.counter_span("engine.generation_tokens", _at(-1), _at(6)) == (
            pytest.approx(100.0)
        )
        # An endpoint on each host is declined, at both window ends.
        assert index.counter_span("engine.generation_tokens", _at(-1), _at(11)) is None
        assert index.counter_span("engine.generation_tokens", _at(-1), _at(15)) is None

    with TelemetryIndex(tmp_path / "reversed.db") as index:
        index.ingest([late_first, first])
        # The same pair the other way round, which would otherwise return the
        # larger figure of the two.
        assert index.counter_span("engine.generation_tokens", _at(-1), _at(11)) is None


def test_a_row_a_reader_gets_carries_the_key_it_was_stored_under(tmp_path):
    """A refused boot spelling is not read back as a boot identity.

    The stored columns hold the distinction between a row keyed on its boot and
    one keyed on its host alone, and a reader that takes the row's own spelling
    sees the second as the first: the refused text reads as the identity the
    store keyed on, when the key it holds is the unknown-boot sentinel. So the
    row a reader gets carries the resolved key and its kind, and the reboot
    guarantee is visible as absent exactly where it is.
    """
    source = tmp_path / "serve.jsonl"
    _write(
        source,
        [
            _row(0, host="node-a", boot_id=_BOOT_BEFORE),
            _row(5, host="node-a", boot_id="not-a-uuid"),
        ],
    )

    with TelemetryIndex(tmp_path / "index.db", host="node-z") as index:
        index.ingest([source])
        keyed, degraded = index.rows(_at(-1), _at(60))

        # The spelling the parser refused is not presented as a key anywhere in
        # the row a reader holds...
        assert "not-a-uuid" not in json.dumps(degraded)
        # ...while the record this index was read from still carries it.
        assert _direct_rows(source)[1]["boot_id"] == "not-a-uuid"

        # Each row carries the key it was stored under, and the kind of key.
        assert (keyed["host"], keyed["boot_id"], keyed["key_kind"]) == (
            "node-a",
            _BOOT_BEFORE,
            BOOT_SCOPE,
        )
        assert (degraded["host"], degraded["boot_id"], degraded["key_kind"]) == (
            "node-a",
            UNKNOWN_BOOT_ID,
            HOST_SCOPE,
        )


def test_one_file_spanning_a_reboot_keys_each_of_its_rows_apart(tmp_path):
    """The boot is resolved per row, because one file can span a reboot.

    A reboot leaves the hostname alone, so a file written across one carries
    two boots and only the rows themselves say where the change falls. Resolved
    once per file from its first named boot, every later row is keyed on a boot
    it was not written in -- silently, because the rows stay storable and the
    counters they carry then difference across a reset as though the machine
    had merely been quiet.
    """
    source = tmp_path / "serve.jsonl"
    _write(
        source,
        [
            _row(0, host="node-a", boot_id=_BOOT_BEFORE),
            _row(5, host="node-a", boot_id=_BOOT_AFTER),
            _row(10, host="node-a"),
        ],
    )

    with TelemetryIndex(tmp_path / "index.db", host="node-z") as index:
        index.ingest([source])
        stored = index._conn.execute(
            "SELECT boot_id FROM sample ORDER BY id"
        ).fetchall()
        assert [row["boot_id"] for row in stored] == [
            _BOOT_BEFORE,
            _BOOT_AFTER,
            _BOOT_AFTER,
        ]
        # The row that names no boot belongs to the boot that was writing the
        # file when it appeared, not to the reader's.
        assert [row["boot_id"] for row in index.rows(_at(-1), _at(60))] == [
            _BOOT_BEFORE,
            _BOOT_AFTER,
            _BOOT_AFTER,
        ]


def test_a_source_consumed_whole_and_untouched_is_not_read_again(tmp_path):
    """Re-proving a finished serve unchanged costs the record and learns nothing.

    Measured 2026-09-23 over 49 sources and 220 MB: a pass with nothing to add
    spent 8.6 s re-digesting serves that had already exited.
    """
    source = tmp_path / "serve.jsonl"
    _write(source, [_row(0), _row(5)])
    index_path = tmp_path / "index.db"

    with TelemetryIndex(index_path) as index:
        assert index.ingest([source]).files_scanned == 1
        assert index.ingest([source]).files_scanned == 0, "re-read an unchanged source"


def test_a_source_that_grew_is_read_again_despite_the_skip(tmp_path):
    """The skip must never swallow rows appended after the consuming pass."""
    source = tmp_path / "serve.jsonl"
    _write(source, [_row(0)])
    index_path = tmp_path / "index.db"

    with TelemetryIndex(index_path) as index:
        index.ingest([source])
        _write(source, [_row(5)])
        report = index.ingest([source])

    assert report.files_scanned == 1
    assert report.rows_inserted == 1


def test_a_rewrite_that_moves_the_modification_time_is_still_caught(tmp_path):
    """The skip holds only while both length and modification time stand still.

    A recorder, a roll and a compaction all write, so each moves the time; the
    digest comparison is what answers once anything has.
    """
    source = tmp_path / "serve.jsonl"
    _write(source, [_row(0), _row(5)])
    index_path = tmp_path / "index.db"

    with TelemetryIndex(index_path) as index:
        index.ingest([source])
        before = source.stat()
        # Equal-length content, so length alone cannot force the re-read and the
        # modification time is the only thing left to notice the rewrite. If the
        # lengths ever diverge this test stops measuring what it claims, so the
        # premise is asserted rather than assumed.
        source.write_text(
            "".join(
                json.dumps(row) + "\n"
                for row in (
                    _row(0, prefix_cache_query_delta=40),
                    _row(5, prefix_cache_query_delta=50),
                )
            ),
            encoding="utf-8",
        )
        assert source.stat().st_size == before.st_size, "rewrite changed the length"
        os.utime(
            source, ns=(before.st_mtime_ns + 1_000_000, before.st_mtime_ns + 1_000_000)
        )
        assert index.ingest([source]).files_scanned == 1, "skipped a rewritten source"


def test_a_row_written_before_the_time_was_recorded_is_not_assumed_unchanged(tmp_path):
    """A null modification time means unknown, and unknown is not unchanged."""
    source = tmp_path / "serve.jsonl"
    _write(source, [_row(0)])
    index_path = tmp_path / "index.db"

    with TelemetryIndex(index_path) as index:
        index.ingest([source])
        index._conn.execute("UPDATE source SET mtime_ns = NULL")
        index._conn.commit()
        assert index.ingest([source]).files_scanned == 1, "trusted an unrecorded time"
