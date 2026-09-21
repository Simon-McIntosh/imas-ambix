"""A compacted telemetry record preserves what it must not average.

The store exists to make a multi-day record affordable without changing the
quantities a reader derives from it. Three properties matter and they are not
the same operation: cumulative counters must survive as endpoints so a
difference across a tier boundary equals the raw difference; gauges must compact
as a time-weighted mean that carries the weight behind it; and a null leaf is an
absent observation, so it must neither erase what its window accumulated nor
read downstream as zero. These tests drive a synthetic multi-day record through
both tiers and assert all three, plus the two durability properties a tiered
store must have -- a compaction is idempotent, and it never retires the tier it
read.
"""

from __future__ import annotations

import html
import json
import math
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import imas_ambix.agent.telemetry_store as telemetry_store
from imas_ambix.agent.telemetry_store import (
    TIER_HOUR,
    TIER_MINUTE,
    TIER_WINDOW_SECONDS,
    TelemetryStoreError,
    compact_file,
    compact_rows,
    main,
    read_rows,
    run_compaction,
)

_CADENCE_S = 20
_DAYS = 3
_START = datetime(2026, 9, 1, tzinfo=UTC)
_SAMPLES = _DAYS * 24 * 60 * 60 // _CADENCE_S
_ROWS_PER_MINUTE = 60 // _CADENCE_S

# The leaves a live serve's receipt record carries in every row and never
# populates, measured over all 3,943 rows of
# /work/projects/imas_gpu/agents/receipts/deepseek-v4-1-flash-1271903.jsonl.
_UNOBSERVED_LEAVES = (
    "num_requests_running",
    "num_requests_waiting",
    "kv_cache_usage_perc",
    "prefix_cache_queries_total",
    "prefix_cache_query_delta",
    "prefix_cache_hits_total",
    "prefix_cache_hit_delta",
    "prefix_cache_hit_rate",
    "prefix_cache_hit_rate_interval",
    "spec_draft_tokens",
    "spec_accepted_tokens",
    "spec_acceptance_rate",
    "spec_num_accepted_per_pos",
)

# A counter and a gauge the store must compact that the record does not carry.
# Every one of the record's cumulative leaves is null in every row it holds, so
# an endpoint assertion needs a leaf of its own rather than one borrowed from the
# record's always-null set.
_SYNTHETIC_COUNTER = "requests_served_total"
_SYNTHETIC_GAUGE = "queue_utilisation_perc"

# The published record of what this store does. Its figures are a claim about the
# code, so the suite reads it here and checks them against the cadence the fixture
# actually runs at.
_LANDING_RECORD = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "evidence"
    / "archive"
    / "serve-telemetry-spine-landed.html"
)

# Tier geometry the suite expects, written out rather than read from the store.
# An expectation computed from the thing it checks is green under any change to
# it: a case that took its row count, window length and declared weight from
# TIER_WINDOW_SECONDS stayed green while TIER_HOUR moved from 3600 to 1800. The
# store's own table is read in exactly one place,
# test_the_store_publishes_the_tier_windows_these_numbers_are_written_against,
# which holds it against the numbers below.
_EXPECTED_TIER_GEOMETRY = {
    # window seconds, rows over the fixture's span, samples behind each row
    TIER_MINUTE: (60, 4_320, 3),
    TIER_HOUR: (3_600, 72, 180),
}


def _raw_rows() -> list[dict]:
    """A three-day record carrying the live record's leaves and its null pattern.

    The key set, the null pattern and the identity leaves are measured from
    ``/work/projects/imas_gpu/agents/receipts/deepseek-v4-1-flash-1271903.jsonl``
    (3,943 rows, 2026-09-16): 13 of its 20 top-level leaves are null in every
    row, its two throughput gauges are null in one row and numeric in the other
    3,942, and its five identity leaves never move. A fixture that populated
    every leaf in every row cannot fail on a null, so the shape is reproduced.

    Three features are deliberately *not* the record's, and each is here for a
    reason the record does not supply:

    * the throughput gauges are null *inside* every window rather than only on
      the record's opening tick, because an interior null is the position that
      discards the observations taken before it while a leading null discards
      nothing;
    * a rising cumulative counter and an oscillating gauge are carried under
      names of their own, because the record's cumulative leaves are all null in
      every row and its flat payload has no nested section -- without them the
      endpoint rule, the weighted mean and the recursion would go unasserted;
    * the cadence is 20 s where the record ticks at about 5 s, so a three-day
      span stays affordable to compact inside a test.
    """
    rows = []
    for index in range(_SAMPLES):
        stamp = _START + timedelta(seconds=index * _CADENCE_S)
        unobserved_tick = index % _ROWS_PER_MINUTE == 1
        row = {
            "timestamp": stamp.isoformat(),
            "job_id": "1271903",
            "profile_slug": "deepseek-v4-1-flash",
            "served_name": "deepseek-v4.1-flash",
            "gpus": 4,
            "generation_throughput_toks_per_s": (
                None if unobserved_tick else 1.3 + 0.4 * math.sin(index / 9.0)
            ),
            "prompt_throughput_toks_per_s": (
                None if unobserved_tick else 32.0 + 6.0 * math.sin(index / 25.0)
            ),
            _SYNTHETIC_COUNTER: 1000 + 7 * index,
            _SYNTHETIC_GAUGE: 30.0 + 5.0 * math.sin(index / 40.0),
            "engine": {
                "family": "sglang",
                "prompt_tokens": 5000 + 13 * index,
                "generation_tokens": 2000 + 11 * index,
                "cached_prompt_tokens": {"device": 100 + index},
            },
        }
        row.update(dict.fromkeys(_UNOBSERVED_LEAVES))
        rows.append(row)
    return rows


def _write(path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _window_start(stamp: datetime, window_s: int) -> int:
    return math.floor(stamp.timestamp() / window_s) * window_s


def _raw_endpoints(rows: list[dict], window_s: int, key: str) -> dict[int, float]:
    """The last raw counter value in each window -- the endpoint a compacted row
    must carry. Rows are ascending, so the last write per window wins."""
    endpoints: dict[int, float] = {}
    for row in rows:
        stamp = datetime.fromisoformat(row["timestamp"])
        endpoints[_window_start(stamp, window_s)] = row[key]
    return endpoints


def _bucket_of(row: dict) -> int:
    return int(datetime.fromisoformat(row["window_start"]).timestamp())


@pytest.fixture(scope="session")
def record(tmp_path_factory):
    """The raw record, built once -- no test mutates it, only reads it."""
    raw_path = tmp_path_factory.mktemp("record") / "raw.jsonl"
    rows = _raw_rows()
    _write(raw_path, rows)
    return raw_path, rows


def test_multi_day_record_compacts_through_both_tiers(tmp_path, record):
    raw_path, rows = record
    counts = run_compaction(
        raw_path, tmp_path / "minute.jsonl", tmp_path / "hour.jsonl"
    )

    minute = read_rows(tmp_path / "minute.jsonl")
    hour = read_rows(tmp_path / "hour.jsonl")

    assert counts == {TIER_MINUTE: len(minute), TIER_HOUR: len(hour)}
    assert len(minute) == _DAYS * 24 * 60  # one row per minute
    assert len(hour) == _DAYS * 24  # one row per hour
    for row in (*minute, *hour):
        assert row["obs"]["samples"] > 0
        assert row["obs"]["seconds"] > 0
        assert row["window_start"] <= row["timestamp"]


def test_counter_difference_across_a_tier_boundary_equals_the_raw_difference(
    tmp_path, record
):
    """The load-bearing property: endpoints are kept, never averaged.

    A mean of a rising counter is a number no reader can difference, so the
    compacted value must equal the raw endpoint exactly, and a difference taken
    between two compacted rows must equal the same difference taken on the raw
    rows those endpoints came from.
    """
    raw_path, rows = record
    run_compaction(raw_path, tmp_path / "minute.jsonl", tmp_path / "hour.jsonl")
    minute = read_rows(tmp_path / "minute.jsonl")
    hour = read_rows(tmp_path / "hour.jsonl")

    for window_s, tier_rows in ((60, minute), (3600, hour)):
        endpoints = _raw_endpoints(rows, window_s, _SYNTHETIC_COUNTER)
        for row in tier_rows:
            assert row[_SYNTHETIC_COUNTER] == endpoints[_bucket_of(row)]

        # A difference between two compacted rows must equal the raw difference.
        first, last = tier_rows[5], tier_rows[len(tier_rows) // 2]
        compacted_delta = last[_SYNTHETIC_COUNTER] - first[_SYNTHETIC_COUNTER]
        raw_delta = endpoints[_bucket_of(last)] - endpoints[_bucket_of(first)]
        assert compacted_delta == raw_delta > 0

        # A mean would have produced a different number; show it is not that.
        window = [r for r in rows if _window_start(
            datetime.fromisoformat(r["timestamp"]), window_s
        ) == _bucket_of(first)]
        mean = sum(r[_SYNTHETIC_COUNTER] for r in window) / len(window)
        assert first[_SYNTHETIC_COUNTER] > mean

    # Nested engine counters compact by endpoint too.
    raw_gen = {
        _window_start(datetime.fromisoformat(r["timestamp"]), 60): r["engine"][
            "generation_tokens"
        ]
        for r in rows
    }
    for row in minute:
        assert row["engine"]["generation_tokens"] == raw_gen[_bucket_of(row)]


def test_compacted_gauge_rows_carry_the_observation_weight_behind_them(
    tmp_path, record
):
    raw_path, rows = record
    run_compaction(raw_path, tmp_path / "minute.jsonl", tmp_path / "hour.jsonl")
    minute = read_rows(tmp_path / "minute.jsonl")
    hour = read_rows(tmp_path / "hour.jsonl")

    # Every minute of a three-day record is complete, so each carries its window.
    for row in minute:
        assert row["obs"]["samples"] == 60 // _CADENCE_S
        assert row["obs"]["seconds"] == pytest.approx(60, abs=1e-6)

    # The hour tier integrates the minute weights it is built from.
    assert sum(row["obs"]["samples"] for row in minute) == len(rows)
    for row in hour:
        assert row["obs"]["samples"] == (60 // _CADENCE_S) * 60
        assert row["obs"]["seconds"] == pytest.approx(3600, abs=1e-6)

    # A gauge that carried no weight could not be integrated at the next tier;
    # the hour mean must equal the sample-count-weighted mean of the minute rows.
    minute_by_hour: dict[int, list[dict]] = {}
    for row in minute:
        minute_by_hour.setdefault(_bucket_of(row) // 3600 * 3600, []).append(row)
    for row in hour:
        group = minute_by_hour[_bucket_of(row)]
        weighted = sum(
            m[_SYNTHETIC_GAUGE] * m["obs"]["samples"] for m in group
        ) / sum(m["obs"]["samples"] for m in group)
        assert row[_SYNTHETIC_GAUGE] == pytest.approx(weighted, abs=1e-6)


def test_gauge_means_are_time_weighted_not_single_samples(tmp_path, record):
    raw_path, rows = record
    run_compaction(raw_path, tmp_path / "minute.jsonl", tmp_path / "hour.jsonl")
    minute = read_rows(tmp_path / "minute.jsonl")

    target = minute[len(minute) // 3]
    window = [
        r
        for r in rows
        if _window_start(datetime.fromisoformat(r["timestamp"]), 60)
        == _bucket_of(target)
    ]
    expected_running = sum(r[_SYNTHETIC_GAUGE] for r in window) / len(window)
    assert target[_SYNTHETIC_GAUGE] == pytest.approx(expected_running, abs=1e-9)

    # An identity leaf is carried from the last row, never averaged.
    assert target["served_name"] == "deepseek-v4.1-flash"
    assert target["job_id"] == "1271903"


def test_a_null_leaf_does_not_erase_what_its_window_accumulated():
    """A null mid-window must not discard the samples already folded in.

    The weights are equal, so the window's mean is the mean of the four rows
    that observed the gauge. A null that overwrote the accumulator would leave
    only the rows after it, which is a different number declared under the same
    window weight.
    """
    values = [10.0, 20.0, None, 40.0, 50.0]
    rows = [
        {
            "timestamp": (_START + timedelta(seconds=12 * index)).isoformat(),
            "gauge": value,
        }
        for index, value in enumerate(values)
    ]
    compacted = compact_rows(rows, tier=TIER_MINUTE)
    assert len(compacted) == 1

    row = compacted[0]
    observed = [value for value in values if value is not None]
    assert row["gauge"] == pytest.approx(sum(observed) / len(observed))
    assert row["obs"]["samples"] == len(values)
    assert row["obs"]["seconds"] == pytest.approx(60.0)


def test_a_counter_keeps_its_endpoint_across_a_null():
    """A null carries no value, so the window's endpoint is its last observed one.

    The null in the middle and the null at the end of the window are separate
    positions: a null that overwrote the accumulator would leave the first
    without the 110 it accumulated, and the second with nothing at all -- and a
    window whose endpoint is null reads downstream as a counter that reset.
    """
    cases = (
        ([100.0, 110.0, None, 130.0], 130.0),
        ([100.0, 110.0, None], 110.0),
    )
    for values, expected in cases:
        rows = [
            {
                "timestamp": (_START + timedelta(seconds=12 * index)).isoformat(),
                "hits_total": value,
            }
            for index, value in enumerate(values)
        ]
        assert compact_rows(rows, tier=TIER_MINUTE)[0]["hits_total"] == expected


def test_a_gauge_null_in_part_of_a_window_means_only_its_observations(
    tmp_path, record
):
    """The fixture's throughput gauge is null once per window, as in the live record.

    Its mean must be taken over the rows that carried it, and the window must
    still declare every row it covered.
    """
    raw_path, rows = record
    compact_file(raw_path, tmp_path / "minute.jsonl", tier=TIER_MINUTE)
    minute = read_rows(tmp_path / "minute.jsonl")

    observed: dict[int, list[float]] = {}
    covered: dict[int, int] = {}
    for r in rows:
        bucket = _window_start(datetime.fromisoformat(r["timestamp"]), 60)
        covered[bucket] = covered.get(bucket, 0) + 1
        value = r["generation_throughput_toks_per_s"]
        if value is not None:
            observed.setdefault(bucket, []).append(value)

    assert len(observed) == len(minute)
    for row in minute:
        bucket = _bucket_of(row)
        values = observed[bucket]
        assert values, "a window with no observation cannot carry a mean"
        assert len(values) == covered[bucket] - 1  # one unobserved tick per window
        assert row["generation_throughput_toks_per_s"] == pytest.approx(
            sum(values) / len(values)
        )
        assert row["obs"]["samples"] == covered[bucket]

    assert sum(len(v) for v in observed.values()) == len(rows) - len(minute)


def test_a_leaf_null_in_every_row_compacts_to_null(tmp_path, record):
    """*Not observed* is not *zero*: an unpopulated leaf stays null, at both tiers.

    A counter that compacted to 0.0 here would read to a downstream difference as
    a counter that reset.
    """
    raw_path, rows = record
    run_compaction(raw_path, tmp_path / "minute.jsonl", tmp_path / "hour.jsonl")
    for tier_rows in (
        read_rows(tmp_path / "minute.jsonl"),
        read_rows(tmp_path / "hour.jsonl"),
    ):
        for row in tier_rows:
            for name in _UNOBSERVED_LEAVES:
                assert name in row
                assert row[name] is None


def test_compaction_is_idempotent(tmp_path, record):
    raw_path, _rows = record
    compact_file(raw_path, tmp_path / "a.jsonl", tier=TIER_MINUTE)
    compact_file(raw_path, tmp_path / "b.jsonl", tier=TIER_MINUTE)

    first = (tmp_path / "a.jsonl").read_bytes()
    second = (tmp_path / "b.jsonl").read_bytes()
    assert first == second

    run_compaction(raw_path, tmp_path / "m1.jsonl", tmp_path / "h1.jsonl")
    run_compaction(raw_path, tmp_path / "m2.jsonl", tmp_path / "h2.jsonl")
    assert (tmp_path / "m1.jsonl").read_bytes() == (tmp_path / "m2.jsonl").read_bytes()
    assert (tmp_path / "h1.jsonl").read_bytes() == (tmp_path / "h2.jsonl").read_bytes()


def test_a_compaction_that_did_not_land_is_refused(tmp_path, record, monkeypatch):
    """A successor is re-read, so a write that silently did not land is an error.

    A caller handed a row count for a destination holding none of them cannot
    tell that compaction from a complete one, and the source may then be treated
    as superseded by a file that holds nothing. Redirecting the write makes that
    state and requires the store to refuse it.
    """
    raw_path, _rows = record

    def write_nothing(path, rows):
        Path(path).write_text("", encoding="utf-8")

    monkeypatch.setattr(telemetry_store, "write_rows", write_nothing)
    with pytest.raises(TelemetryStoreError, match="does not hold what was written"):
        compact_file(raw_path, tmp_path / "minute.jsonl", tier=TIER_MINUTE)


def test_a_source_tier_survives_its_successor(tmp_path, record):
    raw_path, rows = record
    run_compaction(raw_path, tmp_path / "minute.jsonl", tmp_path / "hour.jsonl")

    # Both sources are still present and readable after their successors landed.
    assert raw_path.exists()
    assert (tmp_path / "minute.jsonl").exists()
    assert len(read_rows(raw_path)) == len(rows)
    assert len(read_rows(tmp_path / "minute.jsonl")) == _DAYS * 60 * 24


def test_the_store_publishes_the_tier_windows_these_numbers_are_written_against():
    """The store's own window table, read in one place.

    Everything else here writes its tier geometry out, from
    _EXPECTED_TIER_GEOMETRY, so an edit to the table lands as one named failure
    rather than silently moving the expectations that exist to hold the store.
    The written row counts are the fixture's own span read at those windows and
    are checked against it, so a change of span cannot leave them stale.
    """
    assert TIER_WINDOW_SECONDS[TIER_MINUTE] == 60
    assert TIER_WINDOW_SECONDS[TIER_HOUR] == 3_600

    assert _EXPECTED_TIER_GEOMETRY[TIER_MINUTE][1] == _DAYS * 24 * 60
    assert _EXPECTED_TIER_GEOMETRY[TIER_HOUR][1] == _DAYS * 24


@pytest.mark.parametrize("tier", (TIER_MINUTE, TIER_HOUR))
def test_the_rebuild_entry_point_compacts_a_tier_from_its_source(
    tmp_path, record, tier
):
    """The subcommand's entry point is driven by an argument vector, and what it
    is judged on is the file it leaves, not the function it delegates to.

    A test asserting the delegate was called passes against an entry point that
    parses its arguments into the wrong names, ignores one of them, or hands back
    a success for a destination holding nothing. So the argument vector is the
    input and the destination on disk is the assertion: the row count, the
    endpoint each row carries, and the weight behind it.

    Both tiers are exercised, because an entry point that ignores --tier and
    always compacts in one direction is invisible to a case that only ever asks
    for that direction: the row count, the window length and the declared weight
    all have to move with the flag for the invocation to be honest. The geometry
    comes from _EXPECTED_TIER_GEOMETRY rather than from TIER_WINDOW_SECONDS, so
    the case holds the entry point to a written figure and cannot be re-labelled
    by an edit to the store.
    """
    raw_path, rows = record
    window_seconds, expected_rows, expected_samples = _EXPECTED_TIER_GEOMETRY[tier]
    destination = tmp_path / "rebuilt" / f"{tier}.jsonl"

    exit_code = main(
        [
            "--source",
            str(raw_path),
            "--destination",
            str(destination),
            "--tier",
            tier,
        ]
    )

    assert exit_code == 0
    rebuilt = read_rows(destination)
    assert len(rebuilt) == expected_rows
    endpoints = _raw_endpoints(rows, window_seconds, _SYNTHETIC_COUNTER)
    for row in rebuilt:
        assert row["tier"] == tier
        assert row[_SYNTHETIC_COUNTER] == endpoints[_bucket_of(row)]
        assert row["obs"]["samples"] == expected_samples
    assert raw_path.exists()  # the entry point does not retire its source


def test_the_rebuild_entry_point_refuses_a_tier_it_cannot_produce(
    tmp_path, record, capsys
):
    """An unusable argument vector ends the run before any rebuild happens.

    Three things have to hold together, because a status alone does not separate
    a refusal from anything else that ends the process: an entry point that exits
    0 for a tier it cannot produce reports success for work it never did, one that
    exits 1 is a crash rather than a refusal, and one that exits 2 while saying
    nothing has told the caller neither the argument nor the value at fault. So
    the status is pinned to the one argparse uses for an unusable choice, and the
    reason is pinned to the stream the subcommand's own caller reads.

    It must leave no successor on disk either, because a destination half-written
    by an accepted-but-wrong parse is a file a later reader would compact from.
    """
    raw_path, _rows = record
    destination = tmp_path / "rebuilt" / "minute.jsonl"

    with pytest.raises(SystemExit) as refusal:
        main(
            [
                "--source",
                str(raw_path),
                "--destination",
                str(destination),
                "--tier",
                "second",
            ]
        )

    assert refusal.value.code == 2  # argparse's status for an unusable choice
    refusal_message = capsys.readouterr().err
    assert "--tier" in refusal_message
    assert "invalid choice" in refusal_message
    assert "second" in refusal_message
    assert not destination.exists()


def _landing_record_text() -> str:
    """The landing record as plain text: tags stripped, entities resolved and
    whitespace collapsed, so a figure wrapped across source lines is one string
    to search for."""
    markup = _LANDING_RECORD.read_text(encoding="utf-8")
    return " ".join(html.unescape(re.sub(r"<[^>]+>", "", markup)).split())


def test_the_landing_record_publishes_the_weights_the_cadence_yields():
    """The record's numbers are derived here from the cadence the fixture uses.

    A record that restates its figures is free to drift from the store, and did:
    a minute row was published as 12 samples and an hour row as 720, which is
    what a five-second cadence yields, and both survived a review round and a
    commit whose stated subject was correcting them, because nothing read the
    file. So every figure the record states about the compaction is rebuilt here
    from the cadence and the tier windows and matched against the file: change
    the cadence without changing the record and this test reddens.
    """
    text = _landing_record_text()
    minute_seconds = TIER_WINDOW_SECONDS[TIER_MINUTE]
    hour_seconds = TIER_WINDOW_SECONDS[TIER_HOUR]
    minutes_per_hour = hour_seconds // minute_seconds
    minute_samples = minute_seconds // _CADENCE_S
    hour_samples = minute_samples * minutes_per_hour

    assert f"{_CADENCE_S} s cadence" in text
    assert f"{_SAMPLES:,} raw rows" in text
    assert f"{_SAMPLES // minute_samples:,} minute rows" in text
    assert f"{_DAYS * 24:,} hour rows" in text
    assert (
        f"{minute_seconds} / {_CADENCE_S} = {minute_samples} samples over "
        f"{minute_seconds} s"
    ) in text
    assert (
        f"{minutes_per_hour} × {minute_samples} = {hour_samples} samples over "
        f"{hour_seconds:,} s"
    ) in text


def test_a_row_without_a_usable_timestamp_is_refused():
    with pytest.raises(TelemetryStoreError):
        compact_rows([{_SYNTHETIC_COUNTER: 1}], tier=TIER_MINUTE)
    with pytest.raises(TelemetryStoreError):
        compact_rows(
            [{"timestamp": "2026-09-01T00:00:00"}], tier=TIER_MINUTE
        )
