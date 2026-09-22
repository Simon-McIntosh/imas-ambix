"""The serving watcher derives its panels from the record, not from a probe.

Every test here builds a receipt record on disk, points the watcher at it, and
reads the figures back. Nothing binds a socket, and nothing measures a live
serve: if a figure on a panel could only come from an HTTP scrape, no test in
this file would pass.

The record is read through :mod:`imas_ambix.agent.telemetry_index`, so these
tests also fix the contract between the two: the watcher's period figures are
counter endpoints, its card figures are time-weighted means, and a quantity the
record never carried comes back absent rather than as zero.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import pytest

from imas_ambix.agent import watch
from imas_ambix.agent.telemetry_index import TelemetryIndex

HOUR = 3600.0
MONTH = 2592000.0


def _iso(epoch: float) -> str:
    return _dt.datetime.fromtimestamp(epoch, _dt.UTC).isoformat()


def _row(
    epoch: float,
    *,
    prompt_tokens: int,
    generation_tokens: int,
    cached_device: int | None = None,
    cached_external: int | None = None,
    utilisation: float | None = 50.0,
    temperature: float | None = 60.0,
) -> dict:
    """One receipt row carrying the canonical ``engine`` section.

    ``None`` means the scrape did not carry that series, so the key is left out
    entirely — which is what the recorder does, and the difference between an
    absent quantity and a measured zero is precisely what several tests below
    are about.
    """
    engine: dict[str, object] = {
        "family": "vllm",
        "model_id": "deepseek-v4-flash",
        "prompt_tokens": prompt_tokens,
        "generation_tokens": generation_tokens,
        "requests_running": 3,
        "requests_queued": 0,
        "kv_pool_occupancy": 0.25,
    }
    cached: dict[str, int] = {}
    if cached_device is not None:
        cached["device"] = cached_device
    if cached_external is not None:
        cached["external"] = cached_external
    if cached:
        engine["cached_prompt_tokens"] = cached
    row: dict[str, object] = {
        "timestamp": _iso(epoch),
        "job_id": "1234",
        "profile_slug": "deepseek-v4-flash",
        "served_name": "deepseek-v4-flash",
        "gpus": 4,
        "engine": engine,
    }
    if utilisation is not None or temperature is not None:
        row["cards"] = {
            "index_source": "nvidia-smi",
            "count": 1,
            "cards": [
                {
                    "read_index": 0,
                    "index": 0,
                    "utilisation_percent": utilisation,
                    "temperature_c": temperature,
                    "power_draw_w": 300.0,
                    "power_cap_w": 700.0,
                }
            ],
        }
    return row


def _write(tmp_path: Path, rows: list[dict], name: str = "serve.jsonl") -> Path:
    directory = tmp_path / "receipts"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return directory


def _index(tmp_path: Path, rows: list[dict]) -> TelemetryIndex:
    directory = _write(tmp_path, rows)
    index = TelemetryIndex(tmp_path / "index.sqlite3")
    from imas_ambix.agent.telemetry_index import discover

    index.ingest(discover(directory))
    return index


def _dense_rows(now: float, *, steps: int = 40, spacing: float = 60.0) -> list[dict]:
    """A serve whose counters rise monotonically over a short, dense span."""
    rows = []
    for step in range(steps):
        epoch = now - steps * spacing + step * spacing
        rows.append(
            _row(
                epoch,
                prompt_tokens=step * 1000,
                generation_tokens=step * 100,
                cached_device=step * 400,
            )
        )
    return rows


def _spanning_rows(now: float) -> list[dict]:
    """One reading at each period bound, so every period has endpoints.

    A record dense enough to cover a month directly is tens of thousands of
    samples; endpoints one period apart are what a *continuously recorded*
    serve actually presents to each period window.
    """
    bounds = [MONTH, 604800.0, 86400.0, 3600.0, 60.0]
    return [
        _row(now - bound, prompt_tokens=1000 * (k + 1), generation_tokens=100 * (k + 1))
        for k, bound in enumerate(bounds)
    ]


def test_ledger_periods_are_populated_from_the_record(tmp_path: Path) -> None:
    """Every period the record spans gets a row, including the month."""
    now = _dt.datetime.now(_dt.UTC).timestamp()
    index = _index(tmp_path, _spanning_rows(now))
    rows = watch.ledger(index, now=now)
    labels = [row.label for row in rows]
    assert labels == ["1 hour", "1 day", "1 week", "1 month"]
    for row in rows:
        assert row.tokens_in is not None and row.tokens_in > 0
        assert row.tokens_out is not None and row.tokens_out > 0
    index.close()


def test_counters_are_differenced_not_summed(tmp_path: Path) -> None:
    """A period's tokens are the advance of the counter, not its total.

    The record's counter is cumulative, so summing the samples would return
    roughly twice the traffic the endpoints bound. The test asserts the
    difference exactly, which is what makes it fail if the span is ever
    replaced by a sum.
    """
    now = _dt.datetime.now(_dt.UTC).timestamp()
    rows = _dense_rows(now, steps=40, spacing=60.0)
    index = _index(tmp_path, rows)
    period = watch.ledger(index, now=now, periods=((3600.0, "1 hour"),))[0]
    # The window opens at the last sample at or before now-3600 and closes at
    # the last sample before now: 59 advance per step is not used; the counter
    # rises 1000 per step and 100 of the 40 steps fall in the hour.
    in_window = [
        row
        for row in rows
        if now - HOUR
        <= _dt.datetime.fromisoformat(str(row["timestamp"])).timestamp()
        < now
    ]
    first = in_window[0]["engine"]["prompt_tokens"]  # type: ignore[index]
    last = in_window[-1]["engine"]["prompt_tokens"]  # type: ignore[index]
    assert period.tokens_in == pytest.approx(float(last - first))
    summed = sum(row["engine"]["prompt_tokens"] for row in in_window)  # type: ignore[index]
    assert period.tokens_in != pytest.approx(float(summed))
    index.close()


def test_absent_quantity_is_none_and_not_zero(tmp_path: Path) -> None:
    """A series the record never carried reports nothing, not an idle zero."""
    now = _dt.datetime.now(_dt.UTC).timestamp()
    rows = _dense_rows(now, steps=40, spacing=60.0)
    for row in rows:
        row["engine"].pop("cached_prompt_tokens", None)  # type: ignore[union-attr]
    index = _index(tmp_path, rows)
    period = watch.ledger(index, now=now, periods=((3600.0, "1 hour"),))[0]
    assert period.cached is None
    # The uncached span is present, so the row is not simply missing data:
    # the record carried this period and declined to carry this quantity.
    assert period.tokens_in is not None
    index.close()


def test_utilisation_declines_on_thin_coverage(tmp_path: Path) -> None:
    """Two samples in an hour do not speak for the hour.

    The record here holds a handful of readings inside a one-hour period, so
    coverage is a few percent. Reporting their mean would describe the
    watcher's sampling rather than the deployment's load, so the figure is
    declined — and the same record at a short period, which it does cover,
    carries a utilisation.
    """
    now = _dt.datetime.now(_dt.UTC).timestamp()
    rows = [
        _row(now - 180 + step * 60, prompt_tokens=step * 10, generation_tokens=1)
        for step in range(3)
    ]
    index = _index(tmp_path, rows)
    # Two minutes of record inside an hour: refused.
    assert watch.card_utilisation(index, now - HOUR, now) is None
    # The same two minutes read over a five-minute window: reported.
    assert watch.card_utilisation(index, now - 300, now) == pytest.approx(50.0)
    index.close()


def test_bars_are_drawn_for_every_percentage(tmp_path: Path) -> None:
    """Every proportion on the panel is a bar and a figure, or a dash."""
    drawn = watch.bar(50.0)
    assert "█" in drawn and "░" in drawn
    assert "50" in drawn
    assert watch.bar(None) == "—"
    assert "█" not in watch.bar(None)

    now = _dt.datetime.now(_dt.UTC).timestamp()
    index = _index(tmp_path, _dense_rows(now, steps=40, spacing=60.0))
    text = watch.render_document(index, now=now, prices=None)
    assert "█" in text
    assert "ledger" in text
    assert "no cached price table" in text
    index.close()


def test_rendering_never_probes_a_live_endpoint(tmp_path: Path) -> None:
    """The panels come from the record even with nothing serving.

    Reads through the subcommand's own entry point rather than the renderer,
    so the discovery-and-ingest path is exercised too: a watcher that reached
    for a socket would have nothing to answer it here.
    """
    now = _dt.datetime.now(_dt.UTC).timestamp()
    directory = _write(tmp_path, _dense_rows(now, steps=40, spacing=60.0))
    text = watch.watch_text(
        record_dir=directory,
        index_path=tmp_path / "i.sqlite3",
        now=now,
        prices=[],
    )
    assert "deepseek-v4-flash" in text
    assert "█" in text


def test_index_rebuilds_to_the_same_figures(tmp_path: Path) -> None:
    """Deleting the index costs a rebuild, not a different answer."""
    now = _dt.datetime.now(_dt.UTC).timestamp()
    directory = _write(tmp_path, _dense_rows(now, steps=40, spacing=60.0))
    from imas_ambix.agent.telemetry_index import discover

    sources = discover(directory)
    first = TelemetryIndex(tmp_path / "a.sqlite3")
    first.ingest(sources)
    figures = watch.document(first, now=now, prices=[])
    first.close()

    second = TelemetryIndex(tmp_path / "b.sqlite3")
    second.ingest(sources)
    rebuilt = watch.document(second, now=now, prices=[])
    second.close()

    assert figures == rebuilt


def test_price_schedule_resolves_per_weekday(tmp_path: Path) -> None:
    """A rate schedule is honoured, and the base rate is the fallback."""
    base = {"prompt": 1e-6, "completion": 2e-6, "cache_read": 1e-7}
    monday_peak = {
        **base,
        "days": {"monday"},
        "start": 0,
        "end": 1200,
    }
    priced = {**base, "schedule": [monday_peak]}
    # 2026-09-21 is a Monday; 06:00 UTC falls inside the 0000-1200 window.
    monday = _dt.datetime(2026, 9, 21, 6, 0, tzinfo=_dt.UTC).timestamp()
    tuesday = _dt.datetime(2026, 9, 22, 6, 0, tzinfo=_dt.UTC).timestamp()
    assert watch.price_at(priced, monday) is monday_peak
    off_peak = watch.price_at(priced, tuesday)
    assert off_peak["prompt"] == base["prompt"]
    assert off_peak["completion"] == base["completion"]
    assert off_peak["cache_read"] == base["cache_read"]


def test_cost_bills_cached_input_at_the_cache_tier(tmp_path: Path) -> None:
    """Cached input is charged at the cache rate, not the prompt rate."""
    price = {"prompt": 1e-6, "completion": 2e-6, "cache_read": 1e-7}
    total = {"in": 1000.0, "cached": 900.0, "out": 0.0}
    cost = watch.ledger_cost(total, price)
    assert cost == pytest.approx(100 * 1e-6 + 900 * 1e-7)
    assert cost < 1000 * 1e-6


def test_counter_reset_declines_rather_than_going_negative(tmp_path: Path) -> None:
    """A serve restarted inside the period cannot be integrated from two
    endpoints, so the row declines instead of reporting a negative."""
    now = _dt.datetime.now(_dt.UTC).timestamp()
    rows = [
        _row(now - 1800, prompt_tokens=5000, generation_tokens=500),
        _row(now - 1200, prompt_tokens=9000, generation_tokens=900),
        _row(now - 100, prompt_tokens=10, generation_tokens=1),
    ]
    index = _index(tmp_path, rows)
    period = watch.ledger(index, now=now, periods=((3600.0, "1 hour"),))[0]
    assert period.tokens_in is None
    # The raw span really is negative; the refusal is what turns it into None.
    assert index.counter_span("engine.prompt_tokens", now - HOUR, now) < 0
    index.close()


def test_cost_is_declined_when_the_token_totals_are_absent(tmp_path: Path) -> None:
    """An unmeasured period is not a free one.

    A price is a measurement of traffic, so with no traffic figure to price the
    cost must decline alongside the token columns rather than coerce the
    missing totals to zero and publish ``$0.00`` for a period nothing measured.
    """
    price = {"prompt": 1e-6, "completion": 2e-6, "cache_read": 1e-7}
    assert watch.ledger_cost({"in": None, "cached": None, "out": None}, price) is None
    # One absent total is enough: the cached split is what the two input rates
    # divide, so the bill cannot be reconstructed without it either.
    assert watch.ledger_cost({"in": 1000.0, "cached": None, "out": 5.0}, price) is None

    now = _dt.datetime.now(_dt.UTC).timestamp()
    rows = [
        _row(now - 1800, prompt_tokens=5000, generation_tokens=500),
        _row(now - 1200, prompt_tokens=9000, generation_tokens=900),
        _row(now - 100, prompt_tokens=10, generation_tokens=1),
    ]
    index = _index(tmp_path, rows)
    prices = [
        {
            "id": "deepseek/deepseek-v4-flash",
            "pricing": {"prompt": 1e-6, "completion": 2e-6, "input_cache_read": 1e-7},
        }
    ]
    period = watch.ledger(
        index,
        now=now,
        periods=((3600.0, "1 hour"),),
        prices=prices,
        model="deepseek-v4-flash",
    )[0]
    assert period.tokens_in is None
    assert period.cost is None
    # And the panel dashes it rather than printing a figure of zero.
    text = watch.render_document(index, now=now, prices=prices)
    assert "$0.00" not in text
    index.close()


def test_the_subcommand_renders_the_record_it_is_pointed_at(tmp_path: Path) -> None:
    """The command the operator runs reaches the renderer over a real record.

    The watcher's entry point and the terminal panels are covered above; what
    this holds is the registration itself, so a rename on either side of the
    ``imas-ambix agent watch`` boundary fails here rather than at the terminal.
    """
    from click.testing import CliRunner

    from imas_ambix.agent import cli

    now = _dt.datetime.now(_dt.UTC).timestamp()
    directory = _write(tmp_path, _dense_rows(now, steps=40, spacing=60.0))
    index_path = str(tmp_path / "cli.sqlite3")

    runner = CliRunner()
    text = runner.invoke(
        cli.agent, ["watch", "--record", str(directory), "--index", index_path]
    )
    assert text.exit_code == 0, text.output
    assert "ledger" in text.output
    assert "deepseek-v4-flash" in text.output

    figures = runner.invoke(
        cli.agent,
        ["watch", "--record", str(directory), "--index", index_path, "--json"],
    )
    assert figures.exit_code == 0, figures.output
    document = json.loads(figures.output)
    assert {"record", "ledger", "periods"} <= set(document)


def test_latest_reading_assembles_the_hit_rate_from_engine_counters(
    tmp_path: Path,
) -> None:
    """A family stating only the cumulative pair still yields a hit rate."""
    now = 1_800_000_000.0
    row = _row(now - 1, prompt_tokens=1000, generation_tokens=100)
    row["engine"]["prefix_cache_queries"] = 8_000
    row["engine"]["prefix_cache_hits"] = 6_800
    index = _index(tmp_path, [row])

    reading = watch._latest_reading(index, now)

    assert reading["hit_rate"] == pytest.approx(0.85)


def test_latest_reading_prefers_the_engine_rate_gauge(tmp_path: Path) -> None:
    """When the family publishes the rate itself, the counters are not used."""
    now = 1_800_000_000.0
    row = _row(now - 1, prompt_tokens=1000, generation_tokens=100)
    row["engine"]["family"] = "sglang"
    row["engine"]["prefix_cache_hit_rate"] = 0.41
    row["engine"]["prefix_cache_queries"] = 8_000
    row["engine"]["prefix_cache_hits"] = 6_800
    index = _index(tmp_path, [row])

    reading = watch._latest_reading(index, now)

    assert reading["hit_rate"] == pytest.approx(0.41)
