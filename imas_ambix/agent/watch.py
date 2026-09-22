"""``imas-ambix agent watch`` — the serving panels, derived from the record.

The panels moved here from the workstation's ``gpu-watch``, which read a live
serve over HTTP and carried its own private ledger alongside. A reader
derives; it does not probe. So the serving figures, the serve's own card
readings and the whole ledger come from the append-only record the on-node
recorder writes (:mod:`imas_ambix.agent.serving_receipts`), reached through
:class:`~imas_ambix.agent.telemetry_index.TelemetryIndex`. The subcommand
therefore spawns no sampler step for the serve's cards and maintains no ledger
of its own: the durable record is the store, the index is a disposable query
layer over it, and this module only renders.

**Period figures are counter endpoints, not sums.** The engine's token counters
are cumulative since it started, so a period's tokens are the difference of the
last reading at each bound. A counter that fell between the bounds means the
serve restarted inside the period, and two endpoints cannot recover the traffic
on either side of the ledger; the row declines rather than reporting a negative
or a clamped figure that no byte of the record supports.

**A quantity the record does not carry stays absent.** A period whose record
never carried card utilisation reports no utilisation rather than an idle
serve, and a utilisation figure is declined outright when too little of the
period was observed — the watcher's own habits are not the deployment's load.

**Cost is per interval at the rate then in force.** The served model is priced
by time of day, so a period billed at one instant's rate is wrong by the
peak-to-off-peak ratio. Prices are read from the local cache the workstation
watcher already maintains; this reader never fetches, because a reader derives.
With no cached table the cost column is dashed rather than guessed.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from imas_ambix.agent.telemetry_index import TelemetryIndex, discover

if TYPE_CHECKING:
    from collections.abc import Sequence

#: Where the on-node recorder appends its rows. Matches
#: ``imas_ambix.agent.cli``'s receipts subcommand, so the two agree without a
#: shared constant that only one of them reads.
DEFAULT_RECORD_DIR = Path.home() / ".local" / "share" / "ambix" / "receipts"

#: The index is disposable by construction, so it lives under the user's cache
#: rather than beside the record it is derived from.
DEFAULT_INDEX_PATH = Path.home() / ".cache" / "ambix" / "watch-index.sqlite3"

#: Open-market rates, cached on disk by the workstation watcher. Read only.
DEFAULT_PRICE_PATH = Path.home() / ".cache" / "gpu-watch" / "openrouter-prices.json"

#: The canonical engine section of a receipt row, as settled by
#: :mod:`imas_ambix.agent.engine_metrics`. These are the names the ledger
#: integrates; the flat legacy spelling is not consulted, because a reader that
#: reached for it would work only for the one engine family that carries it.
PROMPT_TOKENS = "engine.prompt_tokens"
GENERATION_TOKENS = "engine.generation_tokens"
PREFIX_CACHE_HITS = "engine.prefix_cache_hits"
CACHED_TIERS = (
    "engine.cached_prompt_tokens.device",
    "engine.cached_prompt_tokens.external",
)

KV_OCCUPANCY = "engine.kv_pool_occupancy"
REQUESTS_RUNNING = "engine.requests_running"
REQUESTS_QUEUED = "engine.requests_queued"

#: The index has no name enumeration, so the card panel probes the indices it
#: can hold; a node's card count does not change under a run.
CARD_LIMIT = 16
#: The recorder stores the card section as the structured reading it was, so a
#: card's utilisation is nested twice — the section, then its list. Both that
#: nesting and a flat list are probed, because the section's own schema is the
#: producer's to choose and a reader that assumed one of them would report an
#: idle serve for the other.
CARD_PREFIXES = ("cards.cards", "cards")
CARD_UTILISATION = "{prefix}.{i}.utilisation_percent"

#: Period, label. Only periods the record actually spans are rendered.
LEDGER_PERIODS: tuple[tuple[float, str], ...] = (
    (3600.0, "1 hour"),
    (86400.0, "1 day"),
    (604800.0, "1 week"),
    (2592000.0, "1 month"),
)

#: A sample speaks for the time until the next one, but only so far: across a
#: gap while nothing was recording, the last reading before it says nothing
#: about the hours that followed.
SAMPLE_SPEAKS_FOR = 300.0

#: How much of a period must have been observed before its mean is worth
#: reporting. Below this the figure describes the reader's sampling, not the
#: deployment's load.
OBSERVED_ENOUGH = 0.25

#: A period longer than the shortest one has to be backed by a quarter of
#: itself; otherwise it is the same few minutes relabelled.
MIN_PERIOD_COVERAGE = 0.25

#: Die temperature at which these cards begin to clock down, printed on the
#: column so a temperature coloured against an unstated ceiling is not judged.
THROTTLE_C = 84.0


@dataclasses.dataclass(frozen=True)
class Period:
    """One ledger row: what the record supports for one period."""

    label: str
    period: float
    covered: float
    tokens_in: float | None
    cached: float | None
    tokens_out: float | None
    utilisation: float | None
    cost: float | None

    @property
    def cached_share(self) -> float | None:
        """Percentage of prompt tokens answered from cache, or ``None``."""
        if self.tokens_in is None or self.cached is None or self.tokens_in <= 0:
            return None
        return 100.0 * self.cached / self.tokens_in


# ── formatting ───────────────────────────────────────────────────────


def fmt_tokens(n: float | None) -> str:
    """Token count at three significant figures on the SI ladder, k to T."""
    if n is None:
        return "—"
    n = float(n)
    for scale, suffix in ((1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k")):
        value = n / scale
        if abs(value) >= 0.9995:
            return f"{value:.3g}{suffix}"
    return f"{n:.0f}"


def fmt_usd(value: float | None) -> str:
    """Dollars: cents below $1000, whole dollars above."""
    if value is None:
        return "—"
    value = float(value)
    if abs(value) < 1000:
        return f"${value:.2f}"
    return f"${int(value + 0.5)}"


def fmt_span(seconds: float | None) -> str:
    """Coarse duration — what a row is actually backed by."""
    if seconds is None:
        return "—"
    s = max(0.0, float(seconds))
    for scale, suffix in ((86400.0, "d"), (3600.0, "h"), (60.0, "m")):
        if s >= scale:
            return f"{int(s // scale)}{suffix}"
    return f"{int(s)}s"


def bar(value: float | None, width: int = 5) -> str:
    """A proportion as a filled bar and its figure, the same way everywhere.

    Every proportion on these panels is drawn like this — cache share, pool
    occupancy, card utilisation, power against cap — because a rule applied to
    some percentages and not others has to be learnt, while one applied to all
    of them is simply read. A quantity the record did not carry is a dash, not
    an empty bar: an empty bar is a measured zero.
    """
    if value is None:
        return "—"
    v = max(0.0, min(100.0, float(value)))
    filled = round(v / 100 * width)
    return "█" * filled + "░" * (width - filled) + f" {v:.0f}"


# ── pricing ──────────────────────────────────────────────────────────


_WEEKDAYS = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
)


def price_at(price: dict, when: float) -> dict:
    """The rate in force at a UTC instant, for a model priced by time of day.

    Several providers publish a schedule alongside the headline rate, and the
    headline is whichever window was open when the list was fetched. A period
    billed at that one rate is wrong by the peak-to-off-peak ratio, in a
    direction set by when someone happened to look — the worst property a cost
    estimate can have.

    Windows are ``HHMM`` integers; one whose end is not after its start wraps
    past midnight, and a window with neither bound covers the whole day.
    """
    schedule = price.get("schedule") or []
    if not schedule:
        return price
    moment = time.gmtime(when)
    day = _WEEKDAYS[moment.tm_wday]
    hhmm = moment.tm_hour * 100 + moment.tm_min
    for window in schedule:
        days = window.get("days") or ()
        if days and day not in days:
            continue
        start, end = window.get("start") or 0, window.get("end") or 0
        inside = (
            True
            if start == end
            else (start <= hhmm < end if end > start else hhmm >= start or hhmm < end)
        )
        if inside:
            return window
    return price


def _rates(p: dict, fallback: float | None = None) -> dict | None:
    """The three $/token figures out of one pricing block."""
    try:
        prompt = float(p["prompt"])
        completion = float(p["completion"])
    except KeyError, TypeError, ValueError:
        return None
    try:
        cache = float(p.get("input_cache_read", p.get("cache_read")))
    except TypeError, ValueError:
        # No published cache tier: charge cached input at the prompt rate,
        # which overstates rather than inventing a discount that may not exist.
        cache = prompt if fallback is None else fallback
    return {"prompt": prompt, "completion": completion, "cache_read": cache}


def _price_row(entry: dict) -> dict | None:
    """Normalise one provider row to $/token, with its time-of-day schedule."""
    p = entry.get("pricing", entry)
    base = _rates(p)
    if base is None:
        return None
    schedule = []
    for window in p.get("overrides") or []:
        rates = _rates(window, fallback=base["cache_read"])
        if rates is None:
            continue
        rates["days"] = {str(d).lower() for d in (window.get("utc_days") or [])}
        rates["start"] = int(window.get("utc_start") or 0)
        rates["end"] = int(window.get("utc_end") or 0)
        schedule.append(rates)
    base.update({"slug": entry.get("id", "?"), "schedule": schedule})
    return base


def _norm_model(name: str) -> str:
    return "".join(c for c in (name or "").lower() if c.isalnum())


def match_price(model_id: str, models: Sequence[dict]) -> dict | None:
    """Resolve a served model id to a provider's published price row.

    A serve names its model however it was launched, so the id is matched
    against the full slug, against the slug's family-less tail, and finally
    against a punctuation-stripped form. Variant slugs (``:free``, ``:batch``)
    and aliases never win on their own, because they price a different product
    from the one being served.
    """
    want = (model_id or "").strip().lower()
    if not want:
        return None
    want_norm = _norm_model(want)
    best: tuple[int, dict] | None = None
    for entry in models:
        slug = str(entry.get("id", "")).lower()
        if not slug:
            continue
        tail = slug.split("/", 1)[1] if "/" in slug else slug
        if slug == want or tail == want:
            rank = 0
        elif _norm_model(tail) == want_norm or _norm_model(slug) == want_norm:
            rank = 1
        else:
            continue
        if slug.startswith("~") or ":" in tail:
            rank += 2  # an alias or a variant, only if nothing better exists
        if best is None or rank < best[0]:
            row = _price_row(entry)
            if row is not None:
                best = (rank, row)
    return best[1] if best else None


def load_prices(path: str | Path | None = None) -> list[float | dict] | None:
    """Provider list prices from the local cache, or ``None`` when absent.

    The workstation watcher fetches this table; a reader must not, because a
    network round trip is a probe and the panel would then depend on the
    provider being reachable. An absent or unreadable cache leaves the cost
    column dashed for the whole render rather than stalling it.
    """
    try:
        doc = json.loads(Path(path or DEFAULT_PRICE_PATH).read_text(encoding="utf-8"))
    except OSError, ValueError:
        return None
    models = doc.get("models")
    return models if isinstance(models, list) else None


def ledger_cost(total: dict, price: dict) -> float | None:
    """Open-market cost of one period's traffic at a provider's list rates.

    Cached input is billed at the cache-read tier and only the remainder at
    the prompt rate, which is the point of the split: on this deployment the
    two differ by a factor of fifty, so pricing all input at the prompt rate
    overstates the figure by an order of magnitude.

    A total the record did not carry is not a zero. When a counter reset inside
    the period the traffic cannot be integrated at all, and when a series was
    never recorded its share of the bill is unknown; in either case a price
    derived from the remaining terms publishes a measurement that was never
    taken, so the cost is declined and the row dashes.
    """
    tokens_in = total.get("in")
    cached_in = total.get("cached")
    tokens_out = total.get("out")
    if tokens_in is None or cached_in is None or tokens_out is None:
        return None
    cached = max(0.0, float(cached_in))
    computed = max(0.0, float(tokens_in) - cached)
    return (
        computed * price["prompt"]
        + cached * price["cache_read"]
        + max(0.0, float(tokens_out)) * price["completion"]
    )


# ── the ledger, as queries over the index ────────────────────────────


def _span(index: TelemetryIndex, name: str, start: float, end: float) -> float | None:
    """Advance of a counter across the window, or ``None`` when unusable.

    A negative advance means the counter reset between the bounds — the serve
    restarted inside the period. Two endpoints cannot recover the traffic on
    either side of that reset, so the figure is declined rather than clamped,
    which would report the restarted serve as having carried nothing.
    """
    advance = index.counter_span(name, start, end)
    if advance is None or advance < 0:
        return None
    return advance


def _cached_span(index: TelemetryIndex, start: float, end: float) -> float | None:
    """Cached prompt tokens across the window, summed over the tiers present.

    The tier counters are the canonical carrier; the flat prefix-cache hit
    counter is consulted only when no tier was recorded, so the two spellings
    are never both counted into one figure.
    """
    tiers = [name for name in CACHED_TIERS if index.measurement_count(name)]
    if tiers:
        parts = [_span(index, name, start, end) for name in tiers]
        present = [part for part in parts if part is not None]
        return sum(present) if present else None
    return _span(index, PREFIX_CACHE_HITS, start, end)


def _timestamps(index: TelemetryIndex, start: float, end: float) -> list[float]:
    """Sample epochs in the window, ascending."""
    stamps: list[float] = []
    for row in index.rows(start, end):
        epoch = _parse_timestamp(row.get("timestamp"))
        if epoch is not None:
            stamps.append(epoch)
    return sorted(stamps)


def observed_fraction(index: TelemetryIndex, start: float, end: float) -> float:
    """Fraction of the window the record actually speaks for.

    Each sample stands for the interval running to the next one, capped by
    :data:`SAMPLE_SPEAKS_FOR`, so a gap while nothing was recording contributes
    nothing, and coverage is the total time the record accounts for over the
    window's length.
    """
    if end <= start:
        return 0.0
    stamps = _timestamps(index, start, end)
    if len(stamps) < 2:
        return 0.0
    weight = 0.0
    for previous, current in zip(stamps, stamps[1:], strict=False):
        weight += min(current - previous, SAMPLE_SPEAKS_FOR)
    return weight / (end - start)


def card_utilisation(index: TelemetryIndex, start: float, end: float) -> float | None:
    """Mean busy fraction of the serve's cards across the window.

    This is what ``nvidia-smi`` reported for those cards, recorded beside the
    engine counters and averaged with each reading weighted by the interval it
    actually stood for. It is deliberately not derived from token throughput:
    output tokens a second swing with how many requests are decoding rather
    than prefilling, so a throughput ratio reads low across an hour in which
    the cards never left 100%.

    Declined when too little of the window was observed, because reporting the
    fraction that happened to be recorded as if it were the whole of it is
    reading the watcher's habits as the deployment's load.
    """
    if observed_fraction(index, start, end) < OBSERVED_ENOUGH:
        return None
    readings = []
    for prefix in CARD_PREFIXES:
        names = [
            CARD_UTILISATION.format(prefix=prefix, i=position)
            for position in range(CARD_LIMIT)
        ]
        present = [name for name in names if index.measurement_count(name)]
        if not present:
            continue
        for name in present:
            value = index.time_weighted_mean(name, start, end)
            if value is not None:
                readings.append(value)
        break  # one card section per record; the other spelling is a fallback
    if not readings:
        return None
    return float(sum(readings) / len(readings))


def period_row(
    index: TelemetryIndex,
    start: float,
    end: float,
    *,
    label: str,
    period: float,
    now: float,
    prices: Sequence[dict] | None = None,
    model: str | None = None,
) -> Period | None:
    """One ledger row derived from the record, or ``None`` when unbacked.

    A period the record never reached is left out rather than printed as a row
    of dashes, which would claim a period was observed when it was not. A
    period the record did reach keeps its row even when a counter reset inside
    it made the tokens unintegrable: the dashes then say the period happened
    and could not be measured, which is a different statement from silence.
    """
    stamps = _timestamps(index, start, end)
    if len(stamps) < 2:
        return None

    tokens_in = _span(index, PROMPT_TOKENS, start, end)
    tokens_out = _span(index, GENERATION_TOKENS, start, end)
    cached = _cached_span(index, start, end)

    cost = None
    if prices:
        price = match_price(model or "", prices)
        if price is not None:
            cost = ledger_cost(
                {"in": tokens_in, "cached": cached, "out": tokens_out},
                price_at(price, now),
            )

    covered = min(period, stamps[-1] - stamps[0])
    return Period(
        label=label,
        period=period,
        covered=covered,
        tokens_in=tokens_in,
        cached=cached,
        tokens_out=tokens_out,
        utilisation=card_utilisation(index, start, end),
        cost=cost,
    )


def ledger(
    index: TelemetryIndex,
    *,
    now: float | None = None,
    periods: Sequence[tuple[float, str]] = LEDGER_PERIODS,
    prices: Sequence[dict] | None = None,
    model: str | None = None,
) -> list[Period]:
    """Every period the record spans, longest last.

    A longer period backed by the same few minutes of record as a shorter one
    is the first row relabelled, so it must be backed by
    :data:`MIN_PERIOD_COVERAGE` of itself once a shorter row exists.
    """
    when = time.time() if now is None else now
    rows: list[Period] = []
    for period, label in periods:
        start, end = when - period, when
        row = period_row(
            index,
            start,
            end,
            label=label,
            period=period,
            now=when,
            prices=prices,
            model=model,
        )
        if row is None:
            continue
        if rows and row.covered < MIN_PERIOD_COVERAGE * period:
            continue
        rows.append(row)
    return rows


# ── the panels ───────────────────────────────────────────────────────


def _latest_reading(
    index: TelemetryIndex, now: float, lookback: float = 3600.0
) -> dict:
    """The newest record row, and the gauges the rail reads from it."""
    rows = index.rows(now - lookback, now)
    latest = rows[-1] if rows else {}
    engine = latest.get("engine") if isinstance(latest.get("engine"), dict) else {}

    def gauge(name: str) -> float | None:
        leaf = name.split(".", 1)[1]
        value = engine.get(leaf)
        return float(value) if isinstance(value, int | float) else None

    return {
        "row": latest,
        "family": engine.get("family"),
        "model_id": engine.get("model_id") and str(engine["model_id"]),
        "served_name": latest.get("served_name"),
        "gpus": latest.get("gpus"),
        "running": gauge(REQUESTS_RUNNING),
        "queued": gauge(REQUESTS_QUEUED),
        "kv": gauge(KV_OCCUPANCY),
        "hit_rate": latest.get("prefix_cache_hit_rate"),
    }


def render_rail(reading: dict) -> str:
    """One line: what is served, on what, at what occupancy."""
    model = reading.get("served_name") or reading.get("model_id") or "?"
    bits = [f"[bold]{model}[/]"]
    if reading.get("family"):
        bits.append(f"[dim]{reading['family']}[/]")
    if reading.get("gpus"):
        bits.append(f"[dim]{reading['gpus']}×H200[/]")
    bits.append(f"[dim]kv[/] {bar(_percent(reading.get('kv')))}")
    bits.append(
        f"[dim]running[/] "
        f"{'—' if reading.get('running') is None else int(reading['running'])}"
        f"  [dim]queued[/] "
        f"{'—' if reading.get('queued') is None else int(reading['queued'])}"
    )
    hit = reading.get("hit_rate")
    hit_pct = None if hit is None else hit * 100.0
    bits.append(f"[dim]hit[/] {bar(_percent(hit_pct))}")
    return " [dim]·[/] ".join(bits)


def _percent(fraction: float | None) -> float | None:
    """A 0-1 occupancy as a percentage, or ``None`` when unrecorded."""
    if fraction is None:
        return None
    return float(fraction) * 100.0 if fraction <= 1.0 else float(fraction)


def render_cards(
    index: TelemetryIndex, now: float, lookback: float = 3600.0
) -> list[str]:
    """One line per card, from the cards the record carries.

    A card is drawn only when the record carries a reading for it: a card the
    recorder never saw is absent, not idle, and printing an empty bar for it
    would be the recorded-null defect wearing a glyph, so it is left out.
    """
    latest = _latest_reading(index, now, lookback)["row"]
    section = latest.get("cards")
    # A record that carried the list without a wrapper is read directly.
    cards = section.get("cards") if isinstance(section, dict) else section
    if not isinstance(cards, list):
        return []
    lines: list[str] = []
    for card in cards:
        if not isinstance(card, dict):
            continue
        index_label = card.get("index", card.get("read_index", "?"))
        util = card.get("utilisation_percent")
        temp = card.get("temperature_c")
        power = card.get("power_draw_w")
        cap = card.get("power_cap_w")
        power_text = "—" if power is None else f"{power:.0f}W"
        if cap is not None and power is not None:
            power_text += f"/{cap:.0f}W"
        lines.append(
            f"card {index_label:>2}  {bar(util)}  "
            f"temp {'—' if temp is None else f'{temp:.0f}'}C "
            f"[dim](throttle {THROTTLE_C:.0f}C)[/]  power {power_text}"
        )
    return lines


def render_ledger_rows(rows: Sequence[Period], prices: bool) -> list[str]:
    """The ledger, one line per period, in one column grid."""
    lines = [
        "[bold]ledger[/]  "
        + (
            "[dim]period · covered · in · cached · out · cards · cost[/]"
        )
    ]
    if not rows:
        lines.append("[dim]no record yet[/]")
        return lines
    for row in rows:
        cached_text = fmt_tokens(row.cached)
        share = row.cached_share
        if share is not None:
            cached_text += f" ({share:.0f}%)"
        lines.append(
            f"{row.label:>8}  {fmt_span(row.covered):>5}  "
            f"{fmt_tokens(row.tokens_in):>7}  {cached_text:>14}  "
            f"{fmt_tokens(row.tokens_out):>7}  "
            f"{bar(row.utilisation):>9}  {fmt_usd(row.cost)}"
        )
    if not prices:
        lines.append("[dim]cost: no cached price table[/]")
    return lines


def render_document(
    index: TelemetryIndex,
    *,
    now: float | None = None,
    prices: Sequence[dict] | None = None,
) -> str:
    """The whole reading as one text document, in rich markup."""
    when = time.time() if now is None else now
    reading = _latest_reading(index, when)
    rows = ledger(index, now=when, prices=prices, model=reading.get("model_id"))
    blocks = [render_rail(reading)]
    cards = render_cards(index, when)
    if cards:
        blocks.append("\n".join(cards))
    blocks.extend(render_ledger_rows(rows, prices is not None))
    return "\n".join(blocks)


def watch_text(
    record_dir: str | Path | None = None,
    index_path: str | Path | None = None,
    *,
    now: float | None = None,
    prices: Sequence[dict] | None = None,
    width: int = 110,
) -> str:
    """Ingest the record, query the index, and return the rendered panels.

    This is the whole subcommand minus argument parsing: it discovers the
    record files under *record_dir*, consumes what has been appended since the
    last pass, and asks the index for every figure the panels show. Nothing
    here probes the serve, and nothing here writes a ledger. *prices* is passed
    in for a caller that already holds a table; ``None`` reads the local cache.
    """
    directory = Path(record_dir or DEFAULT_RECORD_DIR)
    target = Path(index_path or DEFAULT_INDEX_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    if prices is None:
        prices = load_prices()
    sources = discover(directory)
    with TelemetryIndex(target) as index:
        index.ingest(sources)
        document = render_document(index, now=now, prices=prices)
    return _plain(document, width=width)


def document(
    index: TelemetryIndex,
    *,
    now: float | None = None,
    prices: Sequence[dict] | None = None,
) -> dict:
    """The same figures :func:`render_document` prints, as plain data.

    The text renderer and this share every query, so a figure a caller reads
    out of the JSON is the figure the panel showed rather than a second
    derivation that could drift from it.
    """
    when = time.time() if now is None else now
    reading = _latest_reading(index, when)
    reading.pop("row", None)
    rows = ledger(index, now=when, prices=prices, model=reading.get("model_id"))
    return {
        "record": reading,
        "ledger": [dataclasses.asdict(row) for row in rows],
        "periods": [row.label for row in rows],
    }


def watch_document(
    record_dir: str | Path | None = None,
    index_path: str | Path | None = None,
    *,
    now: float | None = None,
    prices: Sequence[dict] | None = None,
) -> dict:
    """Consume the record, then return the panels' figures as plain data."""
    directory = Path(record_dir or DEFAULT_RECORD_DIR)
    target = Path(index_path or DEFAULT_INDEX_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    if prices is None:
        prices = load_prices()
    sources = discover(directory)
    with TelemetryIndex(target) as index:
        index.ingest(sources)
        return document(index, now=now, prices=prices)


def _plain(markup: str, *, width: int = 110) -> str:
    """Render rich markup to plain text, so the output is comparable."""
    import io

    from rich.console import Console

    buffer = io.StringIO()
    Console(file=buffer, no_color=True, width=width, highlight=False).print(
        markup, soft_wrap=False
    )
    return buffer.getvalue()


def _parse_timestamp(value: Any) -> float | None:
    """Epoch seconds from a record's ISO timestamp, or ``None`` if unusable."""
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


__all__ = [
    "DEFAULT_INDEX_PATH",
    "DEFAULT_PRICE_PATH",
    "DEFAULT_RECORD_DIR",
    "LEDGER_PERIODS",
    "Period",
    "bar",
    "card_utilisation",
    "fmt_span",
    "fmt_tokens",
    "fmt_usd",
    "ledger",
    "ledger_cost",
    "load_prices",
    "match_price",
    "observed_fraction",
    "period_row",
    "price_at",
    "render_document",
    "watch_text",
]
