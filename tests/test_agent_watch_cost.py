"""The watch command prices a period at the third-party cache rate, not ours.

Our own prefix cache is ours: one large cache we control, shared by the whole
fleet, which a hosted endpoint serving the same model would not have. So the
cost the panel prints is the realistic market estimate -- input split by the
estimated third-party hit rate, cached input at the cache-read tier and the
rest at the prompt tier -- using the estimate record installed beside the price
table. Our own cached count stays in the token columns and never reaches the
bill.

The tests fix the figure by literal so a change to the formula is a change to
the pinned number, and they assert the own-cache figure and the no-cache figure
are absent from the output, because a band or a second figure beside the
estimate invites the reader to pick an endpoint the command was never asked to
name.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
from pathlib import Path

import pytest

from imas_ambix.agent import provider_prices, watch
from imas_ambix.agent.telemetry_index import TelemetryIndex, discover

#: A fixture price row whose rates land the candidate figures at distinct cents,
#: so the one that is printed is separated from the ones that must not be. Per
#: token: $10/M prompt, $1/M cache read, $20/M completion.
PRICE_ROW = {
    "id": "deepseek/deepseek-v4-flash",
    "pricing": {"prompt": 1e-5, "completion": 2e-5, "input_cache_read": 1e-6},
}

#: The fixture period: one million prompt tokens, of which nine hundred
#: thousand were served from our own cache, and a hundred thousand generated.
TOKENS_IN = 1_000_000
TOKENS_CACHED = 900_000
TOKENS_OUT = 100_000

#: The fixture estimate record, in the shape the measurement node writes.
ESTIMATE_RECORD = {
    "h": 0.55,
    "source_kind": "measured",
    "citation": {
        "endpoint": "https://openrouter.ai/api/v1/chat/completions",
        "date": "2026-09-29",
    },
    "observation_time": "2026-09-29T09:05:42+00:00",
}

#: h * in * cache_read + (1 - h) * in * prompt + out * completion, at h = 0.55:
#:   0.55 * 1e6 * 1e-6 = 0.55
#:   0.45 * 1e6 * 1e-5 = 4.50
#:   1e5 * 2e-5       = 2.00  ->  $7.05
ESTIMATED_COST = 7.05


def _iso(epoch: float) -> str:
    return _dt.datetime.fromtimestamp(epoch, _dt.UTC).isoformat()


def _row(
    epoch: float,
    *,
    prompt_tokens: int,
    generation_tokens: int,
    cached_device: int | None = None,
) -> dict:
    """One receipt row carrying the canonical ``engine`` section."""
    engine: dict[str, object] = {
        "family": "vllm",
        "model_id": "deepseek-v4-flash",
        "prompt_tokens": prompt_tokens,
        "generation_tokens": generation_tokens,
        "requests_running": 3,
        "requests_queued": 0,
        "kv_pool_occupancy": 0.25,
    }
    if cached_device is not None:
        engine["cached_prompt_tokens"] = {"device": cached_device}
    return {
        "timestamp": _iso(epoch),
        "job_id": "1234",
        "profile_slug": "deepseek-v4-flash",
        "served_name": "deepseek-v4-flash",
        "gpus": 4,
        "engine": engine,
    }


def _write_receipts(tmp_path: Path, now: float) -> Path:
    """The fixture receipts directory, for a caller that builds its own index."""
    directory = tmp_path / "receipts"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "serve.jsonl"
    rows = [
        _row(now - 1800, prompt_tokens=0, generation_tokens=0, cached_device=0),
        _row(
            now - 100,
            prompt_tokens=TOKENS_IN,
            generation_tokens=TOKENS_OUT,
            cached_device=TOKENS_CACHED,
        ),
    ]
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return directory


def _fixture_index(tmp_path: Path, now: float) -> TelemetryIndex:
    """An index whose one-hour period carries the fixture token totals."""
    directory = _write_receipts(tmp_path, now)
    index = TelemetryIndex(tmp_path / "index.sqlite3")
    index.ingest(discover(directory))
    return index


def _build_index(directory: Path, index_path: Path) -> None:
    """Build *index_path* from *directory*, as the standing ingest does.

    The watch read is read-only, so a test that calls it must lay the index
    down first; this stands in for the fleet's ingest pass.
    """
    index = TelemetryIndex(index_path)
    index.ingest(discover(directory))
    index.close()


def _estimate_record(tmp_path: Path) -> Path:
    """Write the fixture estimate record beside a fixture price table."""
    table = tmp_path / "openrouter-prices.json"
    table.write_text(json.dumps({"models": [PRICE_ROW]}), encoding="utf-8")
    record = table.parent / watch.ESTIMATE_FILENAME
    record.write_text(json.dumps(ESTIMATE_RECORD), encoding="utf-8")
    return table


def _install_fixture_table(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Wire the fixture record into the readers through the table they resolve.

    ``watch_text`` and ``watch_document`` find both the price table and the
    estimate record through :func:`provider_prices.table_path`, so pointing
    that at the fixture table is what makes them read the fixture record
    without an *estimate* argument being passed in.
    """
    table = _estimate_record(tmp_path)
    monkeypatch.setattr(provider_prices, "table_path", lambda *a, **k: table)
    return table


def test_the_record_is_read_from_the_price_tables_own_directory(
    tmp_path: Path,
) -> None:
    """One function joins the table's directory with one filename."""
    table = _estimate_record(tmp_path)
    assert watch.estimate_path(table) == tmp_path / watch.ESTIMATE_FILENAME
    assert watch.estimate_path(table).name == "third-party-cache-rate.json"

    estimate = watch.load_estimate(table)
    assert estimate is not None
    assert estimate.rate == pytest.approx(0.55)
    assert estimate.source_kind == "measured"
    assert estimate.citation["date"] == "2026-09-29"


def test_a_missing_record_is_no_estimate_not_a_rate_of_our_own(
    tmp_path: Path,
) -> None:
    """With no record installed, the loader declines rather than inventing."""
    table = tmp_path / "openrouter-prices.json"
    table.write_text(json.dumps({"models": [PRICE_ROW]}), encoding="utf-8")
    assert watch.load_estimate(table) is None


def test_the_period_is_pinned_to_the_estimate_formula(tmp_path: Path) -> None:
    """The printed figure is h in the cache tier, the rest at prompt."""
    now = _dt.datetime.now(_dt.UTC).timestamp()
    index = _fixture_index(tmp_path, now)
    estimate = watch.CacheRateEstimate(
        rate=0.55,
        source_kind="measured",
        citation=ESTIMATE_RECORD["citation"],
    )
    period = watch.ledger(
        index,
        now=now,
        periods=((3600.0, "1 hour"),),
        prices=[PRICE_ROW],
        model="deepseek-v4-flash",
        estimate=estimate,
    )[0]
    assert period.tokens_in == float(TOKENS_IN)
    assert period.cached == float(TOKENS_CACHED)
    assert period.tokens_out == float(TOKENS_OUT)
    assert period.cost == pytest.approx(ESTIMATED_COST)
    index.close()


def test_the_own_cache_and_no_cache_figures_are_absent(tmp_path: Path) -> None:
    """Neither what our cache really cost nor the h=0 figure is printed.

    Both are computed here from the same period and asserted absent from the
    output, so a change that reverted the figure to our own cached split, or
    priced all input at the prompt tier, reddens a test rather than passing a
    figure that merely looks like a cost.
    """
    now = _dt.datetime.now(_dt.UTC).timestamp()
    index = _fixture_index(tmp_path, now)
    estimate = watch.CacheRateEstimate(
        rate=0.55,
        source_kind="measured",
        citation=ESTIMATE_RECORD["citation"],
    )
    prices = [PRICE_ROW]
    rate = watch.price_at(watch.match_price("deepseek-v4-flash", prices), now)

    # What our own cached split would bill, and what pricing all input at the
    # prompt tier would bill. Both must be absent from the printed output.
    own_cache = watch.ledger_cost(
        {"in": TOKENS_IN, "cached": TOKENS_CACHED, "out": TOKENS_OUT}, rate
    )
    no_cache = watch.estimated_cost({"in": TOKENS_IN, "out": TOKENS_OUT}, rate, 0.0)

    text = watch.render_document(index, now=now, prices=prices, estimate=estimate)
    assert watch.fmt_usd(ESTIMATED_COST) in text
    assert watch.fmt_usd(own_cache) not in text
    assert watch.fmt_usd(no_cache) not in text
    index.close()


def test_the_figure_and_label_reach_the_json_period_rows(tmp_path: Path) -> None:
    """The JSON document carries the same figure and the label beside it."""
    now = _dt.datetime.now(_dt.UTC).timestamp()
    index = _fixture_index(tmp_path, now)
    estimate = watch.CacheRateEstimate(
        rate=0.55,
        source_kind="measured",
        citation=ESTIMATE_RECORD["citation"],
    )
    payload = watch.document(
        index,
        now=now,
        prices=[PRICE_ROW],
        estimate=estimate,
    )
    row = payload["ledger"][0]
    assert row["cost"] == pytest.approx(ESTIMATED_COST)
    assert "55.0%" in row["cost_label"]
    assert "measured" in row["cost_label"]
    assert "2026-09-29" in row["cost_label"]
    index.close()


def test_the_label_gives_the_rate_the_source_and_the_citation(
    tmp_path: Path,
) -> None:
    """The label carries h as a percentage, the source kind and its citation."""
    now = _dt.datetime.now(_dt.UTC).timestamp()
    index = _fixture_index(tmp_path, now)
    estimate = watch.CacheRateEstimate(
        rate=0.55,
        source_kind="measured",
        citation=ESTIMATE_RECORD["citation"],
    )
    text = watch.render_document(index, now=now, prices=[PRICE_ROW], estimate=estimate)
    assert "55.0%" in text
    assert "measured" in text
    assert "https://openrouter.ai/api/v1/chat/completions" in text
    assert "2026-09-29" in text
    index.close()


def test_a_missing_record_prints_a_dash_and_says_so(tmp_path: Path) -> None:
    """No installed estimate dashes the cost and names the reason."""
    now = _dt.datetime.now(_dt.UTC).timestamp()
    index = _fixture_index(tmp_path, now)
    text = watch.render_document(index, now=now, prices=[PRICE_ROW], estimate=None)
    assert "no third-party estimate installed" in text
    # The cost column, and only it, dashes: the token columns remain.
    assert "$" not in text
    index.close()


def test_watch_text_prices_from_the_installed_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The text panel reads the record the table's directory holds.

    No *estimate* is passed, so the figure can only arrive through the reader
    path: the table the panel resolves holds the fixture row and its directory
    holds the fixture record. Replacing the installed-record read with ``None``
    dashes the cost and reddens this test, which is the mutation this wiring
    exists to catch.
    """
    now = _dt.datetime.now(_dt.UTC).timestamp()
    _install_fixture_table(tmp_path, monkeypatch)
    directory = _write_receipts(tmp_path, now)
    index_path = tmp_path / "text.sqlite3"
    _build_index(directory, index_path)
    text = watch.watch_text(directory, index_path, now=now)
    assert watch.fmt_usd(ESTIMATED_COST) in text
    assert "55.0%" in text
    assert "measured" in text
    assert "2026-09-29" in text


def test_watch_document_prices_from_the_installed_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The JSON document reads the same installed record and carries its label.

    *estimate* is not passed here either, so the period row's ``cost`` and
    ``cost_label`` both depend on the installed-record read.
    """
    now = _dt.datetime.now(_dt.UTC).timestamp()
    _install_fixture_table(tmp_path, monkeypatch)
    directory = _write_receipts(tmp_path, now)
    _build_index(directory, tmp_path / "doc.sqlite3")
    payload = watch.watch_document(directory, tmp_path / "doc.sqlite3", now=now)
    row = payload["ledger"][0]
    assert row["cost"] == pytest.approx(ESTIMATED_COST)
    assert "55.0%" in row["cost_label"]
    assert "measured" in row["cost_label"]


@pytest.mark.parametrize(
    "value",
    [-0.01, 1.01, 2.0, "0.5", None, True, False, {"h": 0.5}, [0.5]],
)
def test_load_estimate_declines_a_rate_the_unit_interval_excludes(
    tmp_path: Path, value: object
) -> None:
    """A record whose h is out of range, non-numeric or boolean is refused.

    The reader declines the whole record rather than clamping what it holds, so
    no cost is priced from a rate the record could not support.
    """
    table = _estimate_record(tmp_path)
    record = table.parent / watch.ESTIMATE_FILENAME
    doc = dict(ESTIMATE_RECORD)
    doc["h"] = value
    record.write_text(json.dumps(doc), encoding="utf-8")
    assert watch.load_estimate(table) is None


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_load_estimate_accepts_the_unit_interval_bounds(
    tmp_path: Path, value: float
) -> None:
    """Positive control: the bounds the decline rule names are themselves kept."""
    table = _estimate_record(tmp_path)
    record = table.parent / watch.ESTIMATE_FILENAME
    doc = dict(ESTIMATE_RECORD)
    doc["h"] = value
    record.write_text(json.dumps(doc), encoding="utf-8")
    estimate = watch.load_estimate(table)
    assert estimate is not None
    assert estimate.rate == pytest.approx(value)


@pytest.mark.parametrize("value", [1.2, 2.0, -0.5, "0.5", True])
def test_estimated_cost_refuses_a_rate_outside_the_unit_interval(
    value: object,
) -> None:
    """A rate outside 0..1, non-numeric or boolean is refused, not priced.

    The guard names the rate it refused, so no caller -- direct or through the
    loader -- can obtain a negative or inflated figure by handing this
    function a value the unit interval excludes.
    """
    price = {"prompt": 1e-5, "cache_read": 1e-6, "completion": 2e-5}
    with pytest.raises(ValueError, match=re.escape(repr(value))):
        watch.estimated_cost({"in": TOKENS_IN, "out": TOKENS_OUT}, price, value)


# ── the ledger walks the price history, segment by segment ────────────
#
# The served slug the fixture table's row carries, and the id the fixture
# history records its listings and its headline under.
SLUG = "deepseek/deepseek-v4-flash"

#: Three endpoints of eight bits or more, and the medians they produce. The
#: rates are per token.
WIDE_A = ((1e-5, 1e-6, 2e-5), (2e-5, 2e-6, 4e-5), (3e-5, 3e-6, 6e-5))
MEDIAN_A = (2e-5, 2e-6, 4e-5)
WIDE_B = ((4e-5, 4e-6, 8e-5), (6e-5, 6e-6, 12e-5), (8e-5, 8e-6, 16e-5))
MEDIAN_B = (6e-5, 6e-6, 12e-5)
#: A narrow endpoint priced three orders of magnitude below the wide ones. It
#: must not enter the median, and must not be counted in the label.
FP4 = (1e-9, 1e-10, 1e-9)
#: The headline a listing of nothing but narrow endpoints falls back to.
HEADLINE = (9e-5, 9e-6, 18e-5)
#: The rates the fixture table's own row lists.
PRICE_RATES = (1e-5, 1e-6, 2e-5)

#: The fixture period's own tokens, split at the recorded change: the first
#: segment carries a third of the period's input and a third of its output.
FIRST_IN, FIRST_OUT = 1_000_000, 100_000
TOTAL_IN, TOTAL_OUT = 3_000_000, 300_000
SECOND_IN, SECOND_OUT = TOTAL_IN - FIRST_IN, TOTAL_OUT - FIRST_OUT

HOUR = 3600.0


def _endpoint(prompt: float, cache: float, completion: float, provider: str) -> dict:
    return {
        "provider": provider,
        "tag": provider,
        "quantization": "fp8",
        "pricing": {
            "prompt": prompt,
            "completion": completion,
            "input_cache_read": cache,
        },
    }


def _narrow(provider: str) -> dict:
    endpoint = _endpoint(*FP4, provider=provider)
    endpoint["quantization"] = "fp4"
    return endpoint


def _listing(at: float, prices: tuple, narrow: bool = False) -> dict:
    """One endpoints observation, in the shape §17's refresh appends."""
    endpoints = [
        _endpoint(*triple, provider=f"wide{i}") for i, triple in enumerate(prices)
    ]
    if narrow:
        endpoints.append(_narrow("narrow"))
    return {"at": at, "id": SLUG, "source": "endpoints", "endpoints": endpoints}


def _headline_row(at: float, rates: tuple) -> dict:
    """One headline observation, the row ``rate_in_force`` reads."""
    prompt, cache, completion = rates
    return {
        "at": at,
        "id": SLUG,
        "source": "models",
        "pricing": {
            "prompt": prompt,
            "completion": completion,
            "input_cache_read": cache,
        },
    }


def _seg_cost(tokens_in: float, tokens_out: float, rates: tuple) -> float:
    """h in the cache tier, the rest at prompt, plus output -- asserted, not imported.

    Written out here rather than taken from the module, so a figure that agrees
    with the ledger's is two independent derivations rather than one function
    read twice.
    """
    prompt, cache, completion = rates
    cached = 0.55 * tokens_in * cache
    uncached = 0.45 * tokens_in * prompt
    return cached + uncached + tokens_out * completion


def _estimate() -> watch.CacheRateEstimate:
    return watch.CacheRateEstimate(
        rate=0.55,
        source_kind="measured",
        citation=ESTIMATE_RECORD["citation"],
    )


def _segmented_index(
    tmp_path: Path, rows: list[tuple[float, int, int]]
) -> TelemetryIndex:
    """An index over receipts placed at caller-chosen instants.

    Each row is ``(epoch, prompt_tokens, generation_tokens)``, cumulative since
    the serve started, so a caller can put a reading inside a stretch of the
    period it means to total.
    """
    directory = tmp_path / "segmented"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "serve.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for epoch, prompt, generation in rows:
            fh.write(
                json.dumps(
                    _row(
                        epoch,
                        prompt_tokens=prompt,
                        generation_tokens=generation,
                        cached_device=prompt // 2,
                    )
                )
                + "\n"
            )
    index = TelemetryIndex(tmp_path / "segmented.sqlite3")
    index.ingest(discover(directory))
    return index


def _one_period(index: TelemetryIndex, now: float, **kwargs) -> watch.Period:
    rows = watch.ledger(
        index, now=now, periods=((HOUR, "1 hour"),), model="deepseek-v4-flash", **kwargs
    )
    assert len(rows) == 1
    return rows[0]


def _split_rows(now: float) -> list[tuple[float, int, int]]:
    """Three readings inside the period, with one either side of the change."""
    return [
        (now - 2700, 0, 0),
        (now - 1860, FIRST_IN, FIRST_OUT),
        (now - 1740, FIRST_IN, FIRST_OUT),
        (now - 60, TOTAL_IN, TOTAL_OUT),
    ]


def test_a_period_spanning_a_listing_change_prices_each_segment_alone(
    tmp_path: Path,
) -> None:
    """Each stretch of the period is billed at the median recorded for it.

    The fixture records one change, so the period is two segments. The first
    segment's tokens are priced at the first median and the second segment's at
    the second, which is not the same figure the period costs at either rate
    applied to all of it -- the mutation the declared negative control makes.
    """
    now = _dt.datetime.now(_dt.UTC).timestamp()
    index = _segmented_index(tmp_path, _split_rows(now))
    history = [_listing(now - 7200, WIDE_A), _listing(now - 1800, WIDE_B)]

    row = _one_period(
        index,
        now,
        prices=[PRICE_ROW],
        estimate=_estimate(),
        history=history,
    )
    assert row.price_segments == 2
    assert row.price_policy == watch.PRICE_POLICY_MEDIAN
    assert row.price_history_from == watch._utc_date(now - 7200)
    assert row.cost == pytest.approx(
        _seg_cost(FIRST_IN, FIRST_OUT, MEDIAN_A)
        + _seg_cost(SECOND_IN, SECOND_OUT, MEDIAN_B)
    )
    # Not the whole period at either one of the two rates.
    assert row.cost != pytest.approx(_seg_cost(TOTAL_IN, TOTAL_OUT, MEDIAN_A))
    assert row.cost != pytest.approx(_seg_cost(TOTAL_IN, TOTAL_OUT, MEDIAN_B))
    index.close()


def test_a_period_after_the_last_change_costs_that_median_alone(
    tmp_path: Path,
) -> None:
    """No change lands inside the period, so one segment answers for it."""
    now = _dt.datetime.now(_dt.UTC).timestamp()
    index = _segmented_index(tmp_path, _split_rows(now))
    history = [_listing(now - 7200, WIDE_A), _listing(now - 5400, WIDE_B)]

    row = _one_period(
        index, now, prices=[PRICE_ROW], estimate=_estimate(), history=history
    )
    # One segment, because no change falls inside the period: `_listing` at
    # now-1800 is gone and both listings precede the period's start.
    assert row.price_segments == 1
    assert row.cost == pytest.approx(_seg_cost(TOTAL_IN, TOTAL_OUT, MEDIAN_B))
    index.close()


def test_a_segment_before_the_first_listing_is_carried_back_and_says_so(
    tmp_path: Path,
) -> None:
    """A stretch the history does not reach is priced at the first listing.

    The listing lands inside the period, so the stretch before it has no
    recorded price. It is priced at the first listing's median, and the label
    says the rate was carried back rather than presenting it as observed.
    """
    now = _dt.datetime.now(_dt.UTC).timestamp()
    index = _segmented_index(tmp_path, _split_rows(now))
    history = [_listing(now - 1800, WIDE_A)]

    row = _one_period(
        index, now, prices=[PRICE_ROW], estimate=_estimate(), history=history
    )
    assert row.price_segments == 2
    assert "carried back" in row.cost_label
    # Both segments carry the one median, so the split totals the same as the
    # period priced in one go -- which is what the carried-back claim means.
    assert row.cost == pytest.approx(_seg_cost(TOTAL_IN, TOTAL_OUT, MEDIAN_A))
    index.close()


def test_a_narrow_endpoint_moves_neither_the_median_nor_the_count(
    tmp_path: Path,
) -> None:
    """An fp4 endpoint is not a price the median may be taken over."""
    now = _dt.datetime.now(_dt.UTC).timestamp()
    index = _segmented_index(tmp_path, _split_rows(now))
    wide = _one_period(
        index,
        now,
        prices=[PRICE_ROW],
        estimate=_estimate(),
        history=[_listing(now - 7200, WIDE_A)],
    )
    listed = _one_period(
        index,
        now,
        prices=[PRICE_ROW],
        estimate=_estimate(),
        history=[_listing(now - 7200, WIDE_A, narrow=True)],
    )
    assert listed.cost == pytest.approx(wide.cost)
    assert listed.cost == pytest.approx(_seg_cost(TOTAL_IN, TOTAL_OUT, MEDIAN_A))
    assert "3 endpoints" in listed.cost_label
    assert "4 endpoints" not in listed.cost_label
    index.close()


def test_a_listing_of_narrow_endpoints_alone_falls_back_to_the_headline(
    tmp_path: Path,
) -> None:
    """Every endpoint below the floor leaves no median, so the headline answers.

    The headline is the blended route, and it is used only here. The label
    counts the segments that fell back, so a figure resting on the headline is
    never read as one resting on a listing.
    """
    now = _dt.datetime.now(_dt.UTC).timestamp()
    index = _segmented_index(tmp_path, _split_rows(now))
    history = [
        _headline_row(now - 7200, HEADLINE),
        _listing(now - 7200, (), narrow=True),
    ]

    row = _one_period(
        index, now, prices=[PRICE_ROW], estimate=_estimate(), history=history
    )
    # One segment, so the count of segments that fell back is the whole period.
    assert row.price_segments == 1
    assert row.cost == pytest.approx(_seg_cost(TOTAL_IN, TOTAL_OUT, HEADLINE))
    assert "fell back to the headline" in row.cost_label
    assert "1 segment(s) fell back to the headline" in row.cost_label
    assert "0 endpoints" in row.cost_label
    index.close()


def test_an_unrecorded_segment_prices_the_period_in_one_segment_and_says_so(
    tmp_path: Path,
) -> None:
    """A stretch with no reading of its own is not a zero.

    The record stops before the change, so the second segment has no reading to
    total. The period is priced in one go at the median in force at its end
    rather than dashing or summing the part that was recorded, and the label
    says segmentation was not possible.
    """
    now = _dt.datetime.now(_dt.UTC).timestamp()
    index = _segmented_index(
        tmp_path,
        [(now - 2700, 0, 0), (now - 1860, FIRST_IN, FIRST_OUT)],
    )
    history = [_listing(now - 7200, WIDE_A), _listing(now - 1800, WIDE_B)]

    row = _one_period(
        index, now, prices=[PRICE_ROW], estimate=_estimate(), history=history
    )
    assert row.price_segments == 1
    assert "segmentation not possible" in row.cost_label
    assert row.cost == pytest.approx(_seg_cost(FIRST_IN, FIRST_OUT, MEDIAN_B))
    index.close()


def test_with_no_endpoint_history_the_period_is_one_segment_at_the_headline(
    tmp_path: Path,
) -> None:
    """A history that records no listing leaves today's figure, and says so."""
    now = _dt.datetime.now(_dt.UTC).timestamp()
    index = _segmented_index(tmp_path, _split_rows(now))

    row = _one_period(index, now, prices=[PRICE_ROW], estimate=_estimate())
    # The whole period at the table's own listed rates, which is the figure
    # today's behaviour already produced and the one this path must keep.
    assert row.cost == pytest.approx(_seg_cost(TOTAL_IN, TOTAL_OUT, PRICE_RATES))
    assert row.price_segments == 1
    assert row.price_policy == watch.PRICE_POLICY_HEADLINE
    assert row.price_history_from is None
    assert "no endpoint history" in row.cost_label
    index.close()


def test_a_scheduled_price_row_is_priced_hour_by_hour(tmp_path: Path) -> None:
    """A row priced by time of day is cut at every hour boundary too.

    The rate in force can move between one hour and the next with no listing
    changing, so the period is split hourly. The figure is the same as the one
    -segment figure here, because the listing's median holds across the period
    and the formula is linear in the tokens -- the segment count is what the
    schedule moves, and the unscheduled control proves it does not otherwise.
    """
    base = float(int(_dt.datetime.now(_dt.UTC).timestamp() // HOUR) * HOUR) - 6 * HOUR
    now = base + 6 * HOUR
    rows = [
        (base + HOUR * k + 300, 100_000 * (2 * k), 10_000 * (2 * k)) for k in range(6)
    ]
    rows += [
        (base + HOUR * k + 3300, 100_000 * (2 * k + 1), 10_000 * (2 * k + 1))
        for k in range(6)
    ]
    index = _segmented_index(tmp_path, sorted(rows))
    history = [_listing(base - HOUR, WIDE_A)]
    scheduled = {
        **PRICE_ROW,
        "pricing": {
            "prompt": 1e-5,
            "completion": 2e-5,
            "input_cache_read": 1e-6,
            "overrides": [
                {
                    "prompt": 5e-5,
                    "completion": 10e-5,
                    "input_cache_read": 5e-6,
                    "utc_start": 0,
                    "utc_end": 2359,
                }
            ],
        },
    }
    span = ((6 * HOUR, "6 hours"),)
    split = watch.ledger(
        index,
        now=now,
        periods=span,
        prices=[scheduled],
        model="deepseek-v4-flash",
        estimate=_estimate(),
        history=history,
    )[0]
    whole = watch.ledger(
        index,
        now=now,
        periods=span,
        prices=[PRICE_ROW],
        model="deepseek-v4-flash",
        estimate=_estimate(),
        history=history,
    )[0]
    assert split.price_segments == 6
    assert whole.price_segments == 1
    assert split.cost == pytest.approx(_seg_cost(1_100_000, 110_000, MEDIAN_A))
    assert split.cost == pytest.approx(whole.cost)
    index.close()


def test_the_index_reads_follow_the_listings_not_the_hours(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hours inside one listing are totalled in one read, whatever the span.

    A scheduled row cuts its period at every hour boundary, so walking the
    cuts one by one reads the index once per hour per counter and a longer
    period costs more. The rate in force is fixed between two listings, so the
    hours of one listing's stretch are totalled together: more hours over one
    listing cost the same reads.
    """
    base = float(int(_dt.datetime.now(_dt.UTC).timestamp() // HOUR) * HOUR) - 48 * HOUR
    now = base + 48 * HOUR
    rows = [(base + HOUR * k + 300, 10_000 * k, 1_000 * k) for k in range(1, 49)]
    index = _segmented_index(tmp_path, rows)
    history = [_listing(base - HOUR, WIDE_A)]
    scheduled = {
        **PRICE_ROW,
        "pricing": {
            "prompt": 1e-5,
            "completion": 2e-5,
            "input_cache_read": 1e-6,
            "overrides": [
                {
                    "prompt": 5e-5,
                    "completion": 10e-5,
                    "input_cache_read": 5e-6,
                    "utc_start": 0,
                    "utc_end": 2359,
                }
            ],
        },
    }

    reads: list[tuple[str, float, float]] = []
    original = watch._span

    def counted(inner: TelemetryIndex, name: str, start: float, end: float):
        reads.append((name, start, end))
        return original(inner, name, start, end)

    monkeypatch.setattr(watch, "_span", counted)

    def period_reads(hours: float) -> int:
        reads.clear()
        watch.ledger(
            index,
            now=now,
            periods=((hours * HOUR, "span"),),
            prices=[scheduled],
            model="deepseek-v4-flash",
            estimate=_estimate(),
            history=history,
        )
        return len(reads)

    assert period_reads(48) == period_reads(12)
    assert period_reads(48) < 10
    index.close()


def test_a_day_priced_only_by_a_schedule_costs_its_hours_one_by_one(
    tmp_path: Path,
) -> None:
    """When the only rate on offer is a time-of-day window, it moves hourly.

    No listing here yields a median -- every endpoint in it is below the
    precision floor -- and the history carries no headline to borrow, so each
    hour is priced at the window open then. Those hours sit under one listing,
    so a walk that totalled them together would price the whole day at the
    first hour's rate. The day is asserted against its own 24 hours, each
    rendered on its own: a period is the sum of the periods it is cut into.
    """
    base = _dt.datetime(2026, 10, 14, tzinfo=_dt.UTC).timestamp()
    now = base + 24 * HOUR
    rows = [
        (base + HOUR * k + 300, 100_000 * (2 * k), 10_000 * (2 * k)) for k in range(24)
    ]
    rows += [
        (base + HOUR * k + 3300, 100_000 * (2 * k + 1), 10_000 * (2 * k + 1))
        for k in range(24)
    ]
    index = _segmented_index(tmp_path, sorted(rows))
    history = [_listing(base - HOUR, (), narrow=True)]
    scheduled = {
        **PRICE_ROW,
        "pricing": {
            "prompt": 1e-5,
            "completion": 2e-5,
            "input_cache_read": 1e-6,
            "overrides": [
                {
                    "prompt": 5e-5,
                    "completion": 10e-5,
                    "input_cache_read": 5e-6,
                    "utc_start": 900,
                    "utc_end": 1700,
                }
            ],
        },
    }

    def priced(start: float, end: float) -> float:
        row = watch.period_row(
            index,
            start,
            end,
            label="span",
            period=end - start,
            now=end,
            prices=[scheduled],
            model="deepseek-v4-flash",
            estimate=_estimate(),
            history=history,
        )
        assert row is not None and row.cost is not None
        return row.cost

    day = priced(base, now)
    hours = sum(priced(base + HOUR * k, base + HOUR * (k + 1)) for k in range(24))
    assert day == pytest.approx(hours)
    index.close()


def test_the_json_period_row_carries_the_segmentation_figures(tmp_path: Path) -> None:
    """The document states the policy, the segment count and the history's start."""
    now = _dt.datetime.now(_dt.UTC).timestamp()
    index = _segmented_index(tmp_path, _split_rows(now))
    payload = watch.document(
        index,
        now=now,
        prices=[PRICE_ROW],
        estimate=_estimate(),
        history=[_listing(now - 7200, WIDE_A), _listing(now - 1800, WIDE_B)],
    )
    row = payload["ledger"][0]
    assert row["price_segments"] == 2
    assert row["price_policy"] == watch.PRICE_POLICY_MEDIAN
    assert row["price_history_from"] == watch._utc_date(now - 7200)
    assert "policy" in row["cost_label"]
    index.close()


def test_watch_document_prices_over_the_history_beside_the_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The render path reads the history the table's own directory holds.

    No history is passed in, so the segments can only arrive through the reader
    path: the fixture table the render resolves sits in a directory holding
    the fixture history, and nothing else in the call names either.
    """
    now = _dt.datetime.now(_dt.UTC).timestamp()
    table = _install_fixture_table(tmp_path, monkeypatch)
    history = provider_prices.history_path(table)
    history.write_text(
        "\n".join(
            json.dumps(row)
            for row in [_listing(now - 7200, WIDE_A), _listing(now - 1800, WIDE_B)]
        )
        + "\n",
        encoding="utf-8",
    )
    directory = _write_segmented_receipts(tmp_path, _split_rows(now))
    index_path = tmp_path / "hist.sqlite3"
    _build_index(directory, index_path)
    payload = watch.watch_document(directory, index_path, now=now)
    row = payload["ledger"][0]
    assert row["price_segments"] == 2
    assert row["cost"] == pytest.approx(
        _seg_cost(FIRST_IN, FIRST_OUT, MEDIAN_A)
        + _seg_cost(SECOND_IN, SECOND_OUT, MEDIAN_B)
    )


def test_the_refresh_names_the_served_slug_and_names_none_without_a_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The listing is fetched for the slug the served model already resolves to.

    The refresh is what fetches an endpoints listing and it takes the slug as an
    argument, so the model is matched against the table on disk before the
    refresh runs. With no table there is nothing to match against, and naming
    none is what defers the listing to the next refresh rather than fetching one
    for a model this render could not resolve.
    """
    now = _dt.datetime.now(_dt.UTC).timestamp()
    table = _install_fixture_table(tmp_path, monkeypatch)
    calls: list[dict] = []

    def _capture(**kwargs):
        calls.append(kwargs)
        return provider_prices.Refresh(table, 1.0, False, None)

    monkeypatch.setattr(provider_prices, "refresh", _capture)
    directory = _write_segmented_receipts(tmp_path, _split_rows(now))
    index_path = tmp_path / "slugs.sqlite3"
    _build_index(directory, index_path)

    watch.watch_text(directory, index_path, now=now)
    assert calls[0]["slugs"] == (SLUG,)

    # With no table on disk there is no row to match, so nothing is named and
    # the refresh is not pointed at a listing this render could not resolve.
    missing = tmp_path / "absent" / "openrouter-prices.json"
    monkeypatch.setattr(provider_prices, "table_path", lambda *a, **k: missing)
    watch.watch_text(directory, index_path, now=now)
    assert calls[-1]["slugs"] == ()


def _write_segmented_receipts(
    tmp_path: Path, rows: list[tuple[float, int, int]]
) -> Path:
    """The receipts directory the segmented fixture index is built from."""
    directory = tmp_path / "segmented"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "serve.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(
                _row(
                    epoch,
                    prompt_tokens=prompt,
                    generation_tokens=generation,
                    cached_device=prompt // 2,
                )
            )
            for epoch, prompt, generation in rows
        )
        + "\n",
        encoding="utf-8",
    )
    return directory
