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
    no_cache = watch.estimated_cost(
        {"in": TOKENS_IN, "out": TOKENS_OUT}, rate, 0.0
    )

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
    text = watch.render_document(
        index, now=now, prices=[PRICE_ROW], estimate=estimate
    )
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
    text = watch.watch_text(directory, tmp_path / "text.sqlite3", now=now)
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
