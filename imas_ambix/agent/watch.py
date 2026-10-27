"""``imas-ambix agent watch`` — the serving figures, derived from the record.

The derivation moved here from the workstation's ``gpu-watch``, which read a
live serve over HTTP and carried its own private ledger alongside. A reader
derives; it does not probe. So the serving figures, the serve's own card
readings and the whole ledger come from the append-only record the on-node
recorder writes (:mod:`imas_ambix.agent.serving_receipts`), reached through
:class:`~imas_ambix.agent.telemetry_index.TelemetryIndex`. The subcommand
therefore spawns no sampler step for the serve's cards and maintains no ledger
of its own: the durable record is the store, the index is a disposable query
layer over it, and this module returns figures as data for the renderer.

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

**A period is priced segment by segment, over the price history.** The rate a
provider charges for a model is not one number over a month: the endpoints
serving it change, and each listing carries its own prices. So a period is
split at each instant the history records a new endpoint listing for the
served slug, and each segment's own tokens are totalled and priced at the
median of the listing then in force. A period priced at one instant's median
is wrong by however far the median moved inside it -- measured on the published
document, a month reading that changed by a factor of three while nothing about
our traffic did. A segment before the first recorded listing is priced at that
first listing's median and says the rate was carried back, so a borrowed rate
is never presented as an observed one. Where a segment's own total is
unrecorded the period is priced as a single segment instead of dashing, and
says so: a split must not turn a period that prices today into one that does
not. The headline rate answers only where the listing's endpoints all declare
fewer than eight bits, and the label counts the segments that fell back.

**The cost printed is the realistic third-party estimate.** Our own cache is
ours: one large prefix cache we control, which a hosted endpoint serving the
same model would not have. So the figure prices input at the estimated
third-party hit rate -- cached input at the cache-read tier, the remainder at
the prompt tier -- using the estimate record installed beside the price table.
Our own cached count enters the token columns but never the bill, and no band
is printed beside the estimate: a band invites the reader to pick an endpoint.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import itertools
import json
import sqlite3
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from imas_ambix.agent import provider_prices
from imas_ambix.agent.index_placement import resolve_index_path
from imas_ambix.agent.telemetry_index import TelemetryIndex

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence


#: Where the on-node recorder appends its rows. Matches
#: ``imas_ambix.agent.cli``'s receipts subcommand, so the two agree without a
#: shared constant that only one of them reads.
def default_record_dir() -> Path:
    """The directory the recorder writes to, read from the same config.

    A function rather than a constant because the site base is taken from
    the environment, and a constant computed at import freezes whatever
    was set then -- the same staleness this resolution exists to remove.
    """
    from imas_ambix.agent.profile import SiteConfig

    return SiteConfig.from_env().receipts_dir


PUBLISHED_WATCH_PATH = Path.home() / "public" / "imas-ambix" / "watch.json"

#: The canonical engine section of a receipt row, as settled by
#: :mod:`imas_ambix.agent.engine_metrics`. These are the names the ledger
#: integrates; the flat legacy spelling is not consulted, because a reader that
#: reached for it would work only for the one engine family that carries it.
PROMPT_TOKENS = "engine.prompt_tokens"
GENERATION_TOKENS = "engine.generation_tokens"
#: The prefix-cache rate and the two cumulative counters it may be assembled
#: from. The two engine families report it in different shapes rather than
#: under different names: one publishes the rate itself as a gauge, the other
#: only the counter pair, from which the ratio is taken. Both spellings are
#: canonical engine-section names; no flat row column is consulted.
PREFIX_CACHE_HIT_RATE = "engine.prefix_cache_hit_rate"
PREFIX_CACHE_QUERIES = "engine.prefix_cache_queries"
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

#: The estimate record the coordinator installs beside the price table. It is a
#: small record of its own -- the third-party hit rate, which of the three
#: sources produced it, and that source's citation -- kept beside the table and
#: never inside it, because the table is refetched and overwritten on its
#: refresh interval while the estimate must survive it.
ESTIMATE_FILENAME = "third-party-cache-rate.json"

#: The pricing policies a period's cost can be derived under. The median of a
#: slug's endpoint listing is the price a request is actually billed at; the
#: headline is the blended cheapest route, often a narrow-quantisation
#: endpoint, and it answers only where no listing survives the precision floor.
PRICE_POLICY_MEDIAN = "provider median of >=8-bit endpoints"
PRICE_POLICY_HEADLINE = "headline rate"


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
    #: The label printed beside the cost figure: the estimated third-party hit
    #: rate as a percentage, its source kind and its citation, or why no
    #: estimate is installed. Carried on the row so the JSON document and the
    #: text panel label the same figure the same way.
    cost_label: str | None = None
    #: How many stretches of the period the cost was summed over. 1 is a period
    #: priced in one go, whether because no listing changed inside it or
    #: because its segment totals could not be recovered.
    price_segments: int | None = None
    #: The policy the figure was priced under: the endpoint listing's median, or
    #: the headline rate where no endpoint survived the precision floor.
    price_policy: str | None = None
    #: The date the priced history begins, when the figure came from it.
    price_history_from: str | None = None
    #: The model this row is priced for. ``None`` on a row that totals several
    #: models, and on a single-model row that was derived without naming one,
    #: because a row prices one model's tokens or sums rows that each did.
    model: str | None = None

    @property
    def cached_share(self) -> float | None:
        """Percentage of prompt tokens answered from cache, or ``None``."""
        if self.tokens_in is None or self.cached is None or self.tokens_in <= 0:
            return None
        return 100.0 * self.cached / self.tokens_in


@dataclasses.dataclass(frozen=True)
class CacheRateEstimate:
    """The installed third-party prompt-cache hit rate and its provenance."""

    rate: float
    source_kind: str
    citation: dict[str, Any]
    observed_at: str | None = None


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


def _price_row(entry: dict) -> dict | None:
    """Normalise one provider row to $/token, with its time-of-day schedule."""
    p = entry.get("pricing", entry)
    base = provider_prices.rates(p)
    if base is None:
        return None
    schedule = []
    for window in p.get("overrides") or []:
        rates = provider_prices.rates(window, fallback=base["cache_read"])
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
    """Provider list prices from the owned table, or ``None`` when absent.

    :mod:`imas_ambix.agent.provider_prices` owns the table: it fetches and
    refreshes it, and this module reads back whatever it last wrote. A reader
    must not fetch, because a network round trip is a probe and the panel
    would then depend on the provider being reachable. An absent or
    unreadable table leaves the cost column dashed for the whole render
    rather than stalling it.
    """
    if path is None:
        from imas_ambix.agent import provider_prices

        path = provider_prices.table_path()
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    models = doc.get("models")
    return models if isinstance(models, list) else None


def ledger_cost(total: dict, price: dict) -> float | None:
    """Open-market cost of one period's traffic at a provider's list rates.

    Cached input is billed at the cache-read tier and only the remainder at
    the prompt rate, which is the point of the split: on this deployment the
    two differ by a factor of fifty, so pricing all input at the prompt rate
    overstates the figure by an order of magnitude.

    A total the record did not carry is not a zero. When a series was never
    recorded its share of the bill is unknown, and a price derived from the
    remaining terms publishes a measurement that was never taken, so the cost
    is declined and the row dashes. A serve that restarted inside the period
    does not make the traffic unintegrable: the window is totalled run by run,
    so each run contributes what it carried and the reset costs nothing.
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


def estimated_cost(total: dict, price: dict, rate: float) -> float | None:
    """The best-effort market cost of one period at a third-party cache rate.

    *rate* is the installed third-party estimate's hit rate: the share of the
    input a hosted endpoint would serve from its own prefix cache. The estimate
    prices that share at the cache-read tier and the remainder at the prompt
    tier. Our own cached count never enters the bill: the input split is the
    estimated third-party rate, not what we measured, because our own cache --
    one large store shared by the fleet -- reaches a hit rate no moving
    endpoint does.

    The gap is what makes the split matter. At the 2026-09-29 prices for
    ``deepseek/deepseek-v4.1-flash`` -- $0.30 per million prompt tokens, $0.006
    cache read, $1.20 completion -- the input term at *h* 0.93 is about twice
    the input term at our own 0.975, because the prompt rate is fifty times the
    cache-read rate. The total ratio depends on the period's output share: from
    the 278,174 input tokens the small-turn third-party hit-rate record carries,
    plus an assumed 40,000 output tokens, output dominates the bill and the
    total rises only about 1.07 times. So the
    total can look close to unmoved while the input is priced quite
    differently, and assuming our own cache understates what a provider would
    charge.

    *rate* is required to be a number in 0..1. A boolean, a non-numeric value,
    or a rate outside the unit interval raises :class:`ValueError` naming the
    rate. Clamping would silently reshape a value the caller's own guard
    refused, and pricing outside the interval would hand back a negative or
    inflated figure no cache rate supports.

    A total the record did not carry is not a zero, so a period whose input or
    output was never recorded is declined and the row dashes rather than
    publishing a cost for traffic nothing measured.
    """
    if isinstance(rate, bool) or not isinstance(rate, (int, float)):
        raise ValueError(f"cache hit rate must be a number in 0..1, got {rate!r}")
    hit_rate = float(rate)
    if not 0.0 <= hit_rate <= 1.0:
        raise ValueError(f"cache hit rate must lie in 0..1, got {rate!r}")

    tokens_in = total.get("in")
    tokens_out = total.get("out")
    if tokens_in is None or tokens_out is None:
        return None
    prompt_in = max(0.0, float(tokens_in))
    return (
        hit_rate * prompt_in * price["cache_read"]
        + (1.0 - hit_rate) * prompt_in * price["prompt"]
        + max(0.0, float(tokens_out)) * price["completion"]
    )


def citation_text(citation: dict[str, Any]) -> str:
    """One citation's values as a single line, in the record's own order."""
    parts: list[str] = []
    for value in citation.values():
        if isinstance(value, (list, tuple)):
            parts.append("..".join(str(item) for item in value))
        elif value not in (None, ""):
            parts.append(str(value))
    return " ".join(parts)


def cost_label(estimate: CacheRateEstimate | None) -> str:
    """The label beside the single estimate, or why there is no estimate.

    Installing the estimate record is a deploy step, so a render
    reports its absence plainly rather than falling back to a rate of its own:
    a rate invented to fill the gap is a measurement nobody took.
    """
    if estimate is None:
        return "no third-party estimate installed"
    percent = f"{100.0 * estimate.rate:.1f}%"
    cited = citation_text(estimate.citation)
    label = f"third-party cache hit h={percent} ({estimate.source_kind})"
    return f"{label} {cited}".strip()


def _cost_label(estimate: CacheRateEstimate | None, walk: PriceWalk | None) -> str:
    """The label beside the figure: the estimate, then how the price was read.

    The two halves answer different questions and are kept apart. The estimate
    clause says which cache hit rate was assumed and where that rate came from;
    the walk clause says which price the tokens were multiplied by, over how
    much history, and whether the rate rested on anything the reader would
    otherwise have to infer.
    """
    base = cost_label(estimate)
    note = walk.note if walk is not None else ""
    return f"{base} {note}".strip() if note else base


def estimate_path(price_path: str | Path | None = None) -> Path:
    """The estimate record's path: the price table's directory, one filename.

    The record is kept beside the table rather than inside it, so a refresh
    that overwrites the table never destroys the estimate, and no path is
    resolved separately for the record.
    """
    if price_path is None:
        from imas_ambix.agent import provider_prices

        price_path = provider_prices.table_path()
    return Path(price_path).parent / ESTIMATE_FILENAME


def load_estimate(
    price_path: str | Path | None = None,
) -> CacheRateEstimate | None:
    """Read the installed third-party cache-rate record, or ``None``.

    One function reads the record and one record answers for every surface, so
    the rate is never a constant repeated inside two tools. An absent or
    unreadable record, or one carrying no usable rate, reports ``None`` and the
    panel prints no estimate rather than a guess.
    """
    try:
        doc = json.loads(estimate_path(price_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict):
        return None
    rate = doc.get("h")
    if not isinstance(rate, (int, float)) or isinstance(rate, bool):
        return None
    if not 0.0 <= float(rate) <= 1.0:
        return None
    citation = doc.get("citation")
    return CacheRateEstimate(
        rate=float(rate),
        source_kind=str(doc.get("source_kind") or "unstated"),
        citation=citation if isinstance(citation, dict) else {},
        observed_at=doc.get("observation_time"),
    )


# ── the ledger, as queries over the index ────────────────────────────


def _span(index: TelemetryIndex, name: str, start: float, end: float) -> float | None:
    """Advance of a counter across the window, or ``None`` when unusable.

    The window is totalled run by run, so a serve that restarted inside the
    period contributes what each of its runs carried rather than declining the
    whole period: the counter reset at the process boundary, not at the
    window's, and the record carries the readings on both sides of it.
    """
    return index.partitioned_total(name, start, end).total


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


def window_models(index: TelemetryIndex, start: float, end: float) -> list[str]:
    """The models whose samples fall in the window, from the stored column.

    The served name is written beside every sample when it is ingested, so the
    set of models a period covers is one grouped read of that column. Reading
    it from the payloads instead would decode every record in the window, which
    is the cost a period is computed from columns in order to avoid.
    """
    return sorted(
        str(row[0])
        for row in index._conn.execute(
            "SELECT DISTINCT served_name FROM sample "
            "WHERE ts_epoch >= ? AND ts_epoch < ? "
            "AND served_name IS NOT NULL AND served_name != ''",
            (start, end),
        )
    )


def _names_by_job(
    index: TelemetryIndex, job_ids: Iterable[object]
) -> dict[str, str]:
    """The model each of these jobs served, at the name it recorded most.

    A job is one serving process and a process serves one model, so the name
    its samples recorded is that job's model for every reading it wrote. Where
    a job carries more than one name the most-recorded one wins, so a single
    mislabelled sample cannot rename a serve in the ledger.
    """
    ids = sorted({str(job) for job in job_ids if job is not None})
    if not ids:
        return {}
    marks = ", ".join("?" for _ in ids)
    names: dict[str, str] = {}
    for job_id, served_name, _ in index._conn.execute(
        "SELECT job_id, served_name, COUNT(*) AS n FROM sample "
        f"WHERE job_id IN ({marks}) AND served_name IS NOT NULL "
        "AND served_name != '' GROUP BY job_id, served_name ORDER BY n DESC",
        ids,
    ):
        names.setdefault(str(job_id), str(served_name))
    return names


def model_series(
    index: TelemetryIndex, names: Sequence[str], start: float, end: float
) -> dict[str, dict[str, list[Mapping[str, Any]]]]:
    """Each counter's window readings, grouped by the model that wrote them.

    The readings are the index's own, so a compacted hour contributes its
    endpoints exactly as it does to an unrouted total; the model is resolved
    from the job each reading belongs to, which is a column on the reading and
    on the sample it came from. A reading whose job recorded no served name is
    left out rather than collected under a placeholder: there is no rate to
    price it at, and a bucket named "unknown" would appear in the ledger as a
    model nobody serves.
    """
    readings = {name: index._counter_window_rows(name, start, end) for name in names}
    by_job = _names_by_job(
        index, [row["job_id"] for rows in readings.values() for row in rows]
    )
    grouped: dict[str, dict[str, list[Mapping[str, Any]]]] = {}
    for name, rows in readings.items():
        for row in rows:
            model = by_job.get(str(row["job_id"]))
            if model is None:
                continue
            grouped.setdefault(model, {}).setdefault(name, []).append(row)
    return grouped


def _series_total(
    series: Mapping[str, list[Mapping[str, Any]]],
    name: str,
    start: float,
    end: float,
) -> float | None:
    """One model's advance of one counter over the window.

    The index's own run partition, applied to the readings that model wrote: a
    run still ends at a host, boot or job change and at a counter that falls,
    so a model's figure is the same one the unrouted query would give for a
    window holding that model alone.
    """
    return TelemetryIndex._partitioned_rows(
        name, start, end, series.get(name, ())
    ).total


def _series_stamps(series: Mapping[str, list[Mapping[str, Any]]]) -> list[float]:
    """Every epoch this model's readings carry, ascending and distinct."""
    return sorted(
        {float(row["ts"]) for rows in series.values() for row in rows}
    )


def _series_cached(
    series: Mapping[str, list[Mapping[str, Any]]], start: float, end: float
) -> float | None:
    """One model's cached prompt tokens, summed over the tiers it recorded.

    The tier counters are the canonical carrier and the flat hit counter is
    consulted only when this model recorded no tier, so the two spellings are
    never both counted into one figure.
    """
    tiers = [name for name in CACHED_TIERS if series.get(name)]
    if tiers:
        parts = [_series_total(series, name, start, end) for name in tiers]
        present = [part for part in parts if part is not None]
        return sum(present) if present else None
    return _series_total(series, PREFIX_CACHE_HITS, start, end)


def _timestamps(index: TelemetryIndex, start: float, end: float) -> list[float]:
    """Sample epochs in the window, ascending.

    Read from the store's own ``ts_epoch`` column rather than decoded one by
    one from the samples' payloads: the column is what the window filters on
    and the index orders, and decoding a payload to re-derive a value the store
    already carries is what made each period's stamps cost seconds.
    """
    return index.timestamps(start, end)


def observed_fraction(
    index: TelemetryIndex,
    start: float,
    end: float,
    *,
    stamps: Sequence[float] | None = None,
) -> float:
    """Fraction of the window the record actually speaks for.

    Each sample stands for the interval running to the next one, capped by
    :data:`SAMPLE_SPEAKS_FOR`, so a gap while nothing was recording contributes
    nothing, and coverage is the total time the record accounts for over the
    window's length.

    *stamps* lets a caller that already read the window's epochs hand them in,
    so a period's stamps are fetched once and shared with its utilisation
    rather than read a second time for the same window.
    """
    if end <= start:
        return 0.0
    if stamps is None:
        stamps = _timestamps(index, start, end)
    if len(stamps) < 2:
        return 0.0
    weight = 0.0
    for previous, current in zip(stamps, stamps[1:], strict=False):
        weight += min(current - previous, SAMPLE_SPEAKS_FOR)
    return weight / (end - start)


def card_utilisation(
    index: TelemetryIndex,
    start: float,
    end: float,
    *,
    stamps: Sequence[float] | None = None,
) -> float | None:
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
    if observed_fraction(index, start, end, stamps=stamps) < OBSERVED_ENOUGH:
        return None
    names_by_prefix = [
        [
            CARD_UTILISATION.format(prefix=prefix, i=position)
            for position in range(CARD_LIMIT)
        ]
        for prefix in CARD_PREFIXES
    ]
    means = index.time_weighted_means(
        [name for group in names_by_prefix for name in group], start, end
    )
    readings = []
    for names in names_by_prefix:
        if not any(means[name][0] for name in names):
            continue
        readings = [means[name][1] for name in names if means[name][1] is not None]
        break  # one card section per record; the other spelling is a fallback
    if not readings:
        return None
    return float(sum(readings) / len(readings))


# ── the price history, walked segment by segment ─────────────────────


def _utc_date(epoch: float) -> str:
    """An epoch as the UTC calendar date the ledger labels a history with."""
    return _dt.datetime.fromtimestamp(epoch, _dt.UTC).strftime("%Y-%m-%d")


def _endpoint_listings(history: Sequence[dict] | None, slug: str) -> list[dict]:
    """The endpoint listings recorded for *slug*, oldest first.

    The history records a listing only when the prices in it moved, so each row
    it holds is itself a change instant: the moments a period is split at are
    the rows' own stamps, and no diff between adjacent rows is needed. The
    headline rows the same history carries are not listings -- they record what
    the model's blended route cost, not what the endpoints serving it charge --
    and an unreadable stamp is left out rather than sorted as zero.
    """
    rows = [
        row
        for row in (history or ())
        if isinstance(row, dict)
        and row.get("id") == slug
        and row.get("source") == "endpoints"
        and not isinstance(row.get("at"), bool)
        and isinstance(row.get("at"), (int, float))
    ]
    return sorted(rows, key=lambda row: float(row["at"]))


def _endpoints_in_median(listing: dict) -> int:
    """How many of a listing's endpoints the median was taken over.

    The precision floor is the price history's own policy, read from the module
    that owns it rather than restated: a count derived from a second copy of
    the rule could disagree with the median printed beside it about which
    endpoints were priced.
    """
    count = 0
    for endpoint in listing.get("endpoints") or ():
        if not isinstance(endpoint, dict):
            continue
        if provider_prices._declares_fewer_than_eight_bits(  # noqa: SLF001
            endpoint.get("quantization")
        ):
            continue
        if provider_prices.rates(endpoint.get("pricing")) is None:
            continue
        count += 1
    return count


def _segment_bounds(
    start: float, end: float, changes: Sequence[float], *, hourly: bool
) -> list[float]:
    """The instants a period is cut at, ascending, ends included.

    A change recorded inside the period cuts it; one recorded outside does not,
    because the segment it would make is empty. Hour boundaries are cut only
    for a price row that carries a schedule, where the rate in force can move
    between one hour and the next without any listing changing.
    """
    marks = {start, end}
    marks.update(at for at in changes if start < at < end)
    if hourly:
        boundary = (int(start // 3600) + 1) * 3600
        while boundary < end:
            marks.add(float(boundary))
            boundary += 3600
    return sorted(marks)


@dataclasses.dataclass(frozen=True)
class PriceWalk:
    """One period's cost, and how the walk over the history arrived at it."""

    cost: float | None
    policy: str | None
    segments: int | None
    history_from: str | None
    #: The clause appended to the cost label: what the policy was, how much
    #: history it was read over, and anything the reader must not have to infer
    #: -- a borrowed rate, a fallback, or a segmentation that did not happen.
    note: str


def _price_walk(
    index: TelemetryIndex,
    start: float,
    end: float,
    *,
    history: Sequence[dict] | None,
    slug: str,
    price: dict,
    estimate: CacheRateEstimate,
    now: float,
    tokens_in: float | None,
    tokens_out: float | None,
) -> PriceWalk:
    """Price one period over the recorded listings, one stretch at a time.

    A segment's tokens are its own: the counters are cumulative, so a stretch
    is totalled between its own bounds exactly as the whole period is, and a
    serve that restarted inside a stretch contributes what that run carried.
    The median a stretch is priced at is the one in force at its start, so a
    change takes effect from the instant it was recorded rather than from the
    instant the render happened to run. Cuts that select the same rate are
    totalled together, whatever that rate was selected from, and a period no
    listing cuts is priced on the totals its caller already holds, so the reads
    this walk costs follow the rate changes rather than the number of cuts.
    """
    listings = _endpoint_listings(history, slug)
    if not listings:
        return PriceWalk(
            cost=estimated_cost(
                {"in": tokens_in, "out": tokens_out},
                price_at(price, now),
                estimate.rate,
            ),
            policy=PRICE_POLICY_HEADLINE,
            segments=1,
            history_from=None,
            note=("no endpoint history: priced as one segment at the current headline"),
        )

    first_at = float(listings[0]["at"])
    history_from = _utc_date(first_at)
    closing = max(listings, key=lambda row: float(row["at"]))
    listed = _endpoints_in_median(closing)
    # A closing listing that precedes the period still answers for it: the
    # rate carried back into the period is the first observation's own, and
    # naming the count beside the policy is what tells the reader the figure
    # rests on an older listing rather than on none.
    if start < first_at:
        listed = _endpoints_in_median(listings[0])
    opening = (
        f"policy {PRICE_POLICY_MEDIAN}, {listed} endpoints, history from {history_from}"
    )

    marks = _segment_bounds(
        start,
        end,
        [float(row["at"]) for row in listings],
        hourly=bool(price.get("schedule")),
    )
    # A cut is totalled together with the neighbouring cuts only when they
    # select the same rate, because that is what makes one read equal the walk.
    # The rate is not always the listing's own: where no listing yields a
    # median the walk borrows the headline, and a row priced by time of day
    # moves its headline between one hour and the next. So the grouping follows
    # the selected rate rather than the listing a cut sits under, and a
    # scheduled row's hours -- each selecting its own rate -- stay one read
    # each. Grouping never prices an hour at another hour's rate.
    stretches: list[tuple[float, float, dict, bool, int]] = []
    carried_back = False
    for mark, nxt in itertools.pairwise(marks):
        when = max(mark, first_at)
        carried_back = carried_back or mark < first_at
        rate = provider_prices.provider_median_in_force(history, slug, when)
        borrowed = rate is None
        if borrowed:
            # Every endpoint in the listing declared fewer than eight bits, so
            # the median was taken over nothing; the headline is the only price
            # the history carries for this stretch.
            rate = provider_prices.rate_in_force(history, slug, when)
        if rate is None:
            rate = price_at(price, when)
        if stretches and stretches[-1][2] == rate and stretches[-1][3] == borrowed:
            head = stretches[-1]
            stretches[-1] = (head[0], nxt, head[2], head[3], head[4] + 1)
        else:
            stretches.append((mark, nxt, rate, borrowed, 1))
    # A period that no listing cuts is one stretch over the window the caller
    # has already totalled, so it is priced on those totals rather than on a
    # second read of the same window.
    uncut = len(marks) == 2
    segments = [
        (
            tokens_in if uncut else _span(index, PROMPT_TOKENS, lo, hi),
            tokens_out if uncut else _span(index, GENERATION_TOKENS, lo, hi),
            rate,
            borrowed,
            cut,
        )
        for lo, hi, rate, borrowed, cut in stretches
    ]
    unrecorded = any(
        in_tokens is None or out_tokens is None
        for _, in_tokens, out_tokens, *_ in segments
    )
    if unrecorded:
        # A stretch whose own total is unrecorded is not a zero, and summing the
        # segments that survive would publish a bill for traffic nothing
        # measured. The period is priced in one go instead, at the median in
        # force at its end, so the split declines rather than the row.
        when = max(end, first_at)
        rate = (
            provider_prices.provider_median_in_force(history, slug, when)
            or provider_prices.rate_in_force(history, slug, when)
            or price_at(price, when)
        )
        return PriceWalk(
            cost=estimated_cost(
                {"in": tokens_in, "out": tokens_out}, rate, estimate.rate
            ),
            policy=PRICE_POLICY_MEDIAN,
            segments=1,
            history_from=history_from,
            note=(
                f"{opening}; segmentation not possible: a segment total is unrecorded"
            ),
        )

    total = 0.0
    fell_back = 0
    for in_tokens, out_tokens, rate, borrowed, cut in segments:
        if borrowed:
            fell_back += cut
        total += (
            estimated_cost({"in": in_tokens, "out": out_tokens}, rate, estimate.rate)
            or 0.0
        )

    notes = [opening]
    if carried_back:
        notes.append("first segment's rate carried back to the first listing")
    if fell_back:
        notes.append(f"{fell_back} segment(s) fell back to the headline")
    return PriceWalk(
        cost=total,
        policy=PRICE_POLICY_MEDIAN,
        segments=len(marks) - 1,
        history_from=history_from,
        note="; ".join(notes),
    )


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
    estimate: CacheRateEstimate | None = None,
    history: Sequence[dict] | None = None,
    series: Mapping[str, list[Mapping[str, Any]]] | None = None,
) -> Period | None:
    """One ledger row derived from the record, or ``None`` when unbacked.

    A period the record never reached is left out rather than printed as a row
    of dashes, which would claim a period was observed when it was not. A
    period the record did reach keeps its row even when a serve restarted
    inside it left part of the window unmeasured: the dashes then say the
    period happened and could not be measured, which is a different statement
    from silence.

    *series* restricts the row to one model's counter readings, which is what
    prices each model's traffic at its own rate rather than applying one
    model's rate to the whole machine's tokens. With no *series* the period is
    totalled over the whole record: the same figure when the window holds one
    model, and the only figure available to a caller that has not grouped its
    readings.
    """
    if series is None:
        stamps = _timestamps(index, start, end)
        if len(stamps) < 2:
            return None
        tokens_in = _span(index, PROMPT_TOKENS, start, end)
        tokens_out = _span(index, GENERATION_TOKENS, start, end)
        cached = _cached_span(index, start, end)
        coverage = index.partitioned_total(PROMPT_TOKENS, start, end).coverage
    else:
        stamps = _series_stamps(series)
        if len(stamps) < 2:
            return None
        tokens_in = _series_total(series, PROMPT_TOKENS, start, end)
        tokens_out = _series_total(series, GENERATION_TOKENS, start, end)
        cached = _series_cached(series, start, end)
        coverage = TelemetryIndex._partitioned_rows(
            PROMPT_TOKENS, start, end, series.get(PROMPT_TOKENS, ())
        ).coverage

    cost, walk = None, None
    if prices and estimate is not None:
        price = match_price(model or "", prices)
        if price is not None:
            walk = _price_walk(
                index,
                start,
                end,
                history=history,
                slug=str(price.get("slug") or ""),
                price=price,
                estimate=estimate,
                now=now,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
            )
            cost = walk.cost

    covered = min(period, coverage)
    return Period(
        label=label,
        period=period,
        covered=covered,
        tokens_in=tokens_in,
        cached=cached,
        tokens_out=tokens_out,
        utilisation=card_utilisation(index, start, end, stamps=stamps),
        cost=cost,
        cost_label=_priced_for(model, _cost_label(estimate, walk)),
        price_segments=None if walk is None else walk.segments,
        price_policy=None if walk is None else walk.policy,
        price_history_from=None if walk is None else walk.history_from,
        model=model,
    )


def _priced_for(model: str | None, label: str | None) -> str | None:
    """The cost label, naming the model whose rate the figure was taken at.

    A row priced at its own model's listing says which one, because a reader
    comparing two rows needs the rate's owner on each; a row derived over the
    whole record has no single owner and keeps the label it had.
    """
    if label is None or model is None:
        return label
    return f"{model}: {label}"


def ledger(
    index: TelemetryIndex,
    *,
    now: float | None = None,
    periods: Sequence[tuple[float, str]] = LEDGER_PERIODS,
    prices: Sequence[dict] | None = None,
    model: str | None = None,
    estimate: CacheRateEstimate | None = None,
    history: Sequence[dict] | None = None,
) -> list[Period]:
    """Every period the record spans, longest last.

    A longer period backed by the same few minutes of record as a shorter one
    is the first row relabelled, so it must be backed by
    :data:`MIN_PERIOD_COVERAGE` of itself once a shorter row exists.

    *history* is the price history the periods are priced over, already read.
    Every period walks the same rows, so it is read once by the caller rather
    than per row.
    """
    when = time.time() if now is None else now
    rows: list[Period] = []
    counters = (PROMPT_TOKENS, GENERATION_TOKENS, PREFIX_CACHE_HITS, *CACHED_TIERS)
    for period, label in periods:
        start, end = when - period, when
        present = (
            [model]
            if model is not None
            else window_models(index, start, end)
        )
        if len(present) > 1:
            grouped = model_series(index, counters, start, end)
            period_rows = [
                row
                for name in present
                if (
                    row := period_row(
                        index,
                        start,
                        end,
                        label=label,
                        period=period,
                        now=when,
                        prices=prices,
                        model=name,
                        estimate=estimate,
                        history=history,
                        series=grouped.get(name, {}),
                    )
                )
                is not None
            ]
            if not period_rows:
                continue
            if rows and max(row.covered for row in period_rows) < (
                MIN_PERIOD_COVERAGE * period
            ):
                continue
            rows.extend(period_rows)
            rows.append(_total_row(index, period_rows, period, label, start, end))
            continue
        row = period_row(
            index,
            start,
            end,
            label=label,
            period=period,
            now=when,
            prices=prices,
            model=model,
            estimate=estimate,
            history=history,
        )
        if row is None:
            continue
        if rows and row.covered < MIN_PERIOD_COVERAGE * period:
            continue
        rows.append(row)
    return rows


def _total_row(
    index: TelemetryIndex,
    period_rows: Sequence[Period],
    period: float,
    label: str,
    start: float,
    end: float,
) -> Period:
    """The period across every model, summed from each model's own figures.

    The cost is the sum of the rows' costs and not a price applied to the
    combined tokens: the models' rates differ, so one rate over the total would
    price one model's traffic at another's listing. A row whose own cost is
    unknown leaves the total unknown for the same reason -- adding the known
    part would publish a figure smaller than the period's and present it as the
    period's.
    """
    costs = [row.cost for row in period_rows]
    complete = all(value is not None for value in costs)
    cost = sum(value for value in costs if value is not None) if complete else None
    return Period(
        label=f"{label} · all models",
        period=period,
        covered=max(row.covered for row in period_rows),
        tokens_in=_summed(period_rows, "tokens_in"),
        cached=_summed(period_rows, "cached"),
        tokens_out=_summed(period_rows, "tokens_out"),
        utilisation=card_utilisation(index, start, end),
        cost=cost,
        cost_label=(
            f"sum of {len(period_rows)} models, each at its own rate"
            if cost is not None
            else None
        ),
        price_segments=_summed(period_rows, "price_segments"),
        price_policy=_uniform(period_rows, "price_policy"),
        price_history_from=_uniform(period_rows, "price_history_from"),
    )


def _summed(rows: Sequence[Period], name: str) -> Any:
    """The sum of one numeric field across *rows*, or ``None`` if any is absent."""
    values = [getattr(row, name) for row in rows]
    if any(value is None for value in values):
        return None
    return sum(values)


def _uniform(rows: Sequence[Period], name: str) -> Any:
    """The value every row shares, or ``None`` when the rows disagree."""
    values = {getattr(row, name) for row in rows}
    return values.pop() if len(values) == 1 else None


# ── the figures ──────────────────────────────────────────────────────


def _latest_reading(
    index: TelemetryIndex, now: float, lookback: float = 3600.0
) -> dict:
    """The newest record row, and the gauges the serving figures read from it."""
    rows = index.rows(now - lookback, now)
    latest = rows[-1] if rows else {}
    engine = latest.get("engine") if isinstance(latest.get("engine"), dict) else {}

    def gauge(name: str) -> float | None:
        leaf = name.split(".", 1)[1]
        value = engine.get(leaf)
        return float(value) if isinstance(value, int | float) else None

    # The rate gauge where the family publishes one; otherwise the ratio of
    # the cumulative counters beside it, which is the same quantity assembled
    # from the pair the engine does carry.
    hit_rate = gauge(PREFIX_CACHE_HIT_RATE)
    if hit_rate is None:
        hits = gauge(PREFIX_CACHE_HITS)
        queries = gauge(PREFIX_CACHE_QUERIES)
        if hits is not None and queries:
            hit_rate = round(hits / queries, 4)

    return {
        "row": latest,
        "family": engine.get("family"),
        "model_id": engine.get("model_id") and str(engine["model_id"]),
        "served_name": latest.get("served_name"),
        "gpus": latest.get("gpus"),
        "running": gauge(REQUESTS_RUNNING),
        "queued": gauge(REQUESTS_QUEUED),
        "kv": gauge(KV_OCCUPANCY),
        "hit_rate": hit_rate,
    }


def index_status(path: str | Path, now: float) -> dict:
    """The index file's presence and last-ingest time, read from its metadata.

    No index file is no index. The reader opens the index read-only, so a read
    never builds one: if the path is absent the caller reports that, rather
    than materialising an empty store and showing an empty record for a fleet
    whose ingest has not run yet. With a file present, the last-ingest time is
    the file's own modification time in UTC to the second -- only an ingest
    pass writes the index, so its modification time is when that pass last
    wrote -- and its age is how long ago that was, so a stale index reads as
    stale rather than as current.
    """
    target = Path(path)
    try:
        info = target.stat()
    except OSError:
        return {"path": str(target), "exists": False}
    written = _dt.datetime.fromtimestamp(info.st_mtime, _dt.UTC).replace(microsecond=0)
    return {
        "path": str(target),
        "exists": True,
        "last_ingest_at": written.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "age_seconds": round(now - info.st_mtime),
    }


def _served_slugs(model: str | None) -> tuple[str, ...]:
    """The provider slugs a refresh should fetch an endpoints listing for.

    The refresh takes the slugs to fetch as an argument, so the served model
    has to be resolved to a provider slug before it runs. The only table that
    can answer that is the one already written: the rows the fetch is about to
    return do not exist yet. A first run, with no table on disk, therefore
    names no slug, and no listing is fetched for a model this render could not
    have matched; the listing arrives on the next refresh, once there is a
    table to match against.
    """
    table = load_prices()
    if not model or not table:
        return ()
    price = match_price(model, table)
    slug = str(price.get("slug") or "") if price else ""
    return (slug,) if slug and slug != "?" else ()


def _owned_prices(
    model: str | None = None,
) -> tuple[list[float | dict] | None, float | None]:
    """Refresh the owned price table, then read it back.

    The full render path calls this once, naming the model the serve reports,
    so the refresh records the endpoints listing this render prices against.
    The owning module decides whether a fetch is due, so a render waits at most
    the module's fetch bound and the reader never fetches on its own account.
    The age returned is the age of the table that was read, so a kept cache is
    reported at its true age rather than as fresh.
    """
    outcome = provider_prices.refresh(slugs=_served_slugs(model))
    return load_prices(outcome.path), outcome.age


def _owned_history() -> list[dict]:
    """The price history beside the owned table, read against that table.

    Resolved through the table's own path, so a reader pointed at a table
    elsewhere -- a fixture, a second deployment -- reads the history that
    belongs to it rather than whatever sits at the site's default location.
    """
    return provider_prices.read_history(provider_prices.history_path())


def _installed_estimate() -> CacheRateEstimate | None:
    """The estimate record installed beside the price table in force."""
    return load_estimate()


def document(
    index: TelemetryIndex,
    *,
    now: float | None = None,
    prices: Sequence[dict] | None = None,
    price_age: float | None = None,
    estimate: CacheRateEstimate | None = None,
    history: Sequence[dict] | None = None,
) -> dict:
    """The same figures the ledger carries, as plain data.

    Every figure a caller reads out of the JSON comes from one derivation, so
    a consumer renders a number that was measured rather than re-deriving it.
    *price_age* is the age of the price
    table the figures were priced from, in seconds; it is reported here so a
    consumer renders cost from the document rather than reaching for the
    table itself. Each period row carries its ``cost`` figure and the
    ``cost_label`` beside it, so a JSON consumer labels the figure the way the
    panel does, and carries ``price_segments``, ``price_policy`` and
    ``price_history_from`` so it can tell a period priced over the recorded
    listings from one priced at a single instant.
    """
    when = time.time() if now is None else now
    reading = _latest_reading(index, when)
    reading.pop("row", None)
    rows = ledger(
        index,
        now=when,
        prices=prices,
        model=reading.get("model_id"),
        estimate=estimate,
        history=history,
    )
    return {
        "record": reading,
        "ledger": [dataclasses.asdict(row) for row in rows],
        "periods": [row.label for row in rows],
        "price_age": price_age,
    }


def live_document(
    record_dir: str | Path | None = None,
    lane_path: str | Path | None = None,
    *,
    now: _dt.datetime | None = None,
) -> dict:
    """The ``lane``, ``live`` and ``callers`` blocks alone, without opening the index.

    This is the ``--live`` path: published sources already carry every live
    serving figure, and none needs the telemetry index. It reads the published
    lane document, the tail of one raw receipt file and the tail of the router's
    request record, so a consumer can call it on a short cadence while index
    work elsewhere is unfinished.

    Nothing here constructs a :class:`TelemetryIndex`, and nothing here reads or
    refreshes the price table -- a live panel shows what the serve is doing now,
    not what a token cost, and the price fetch lives only on the full document
    path. The blocks are the ones :func:`watch_document` appends, so a consumer
    rendering the live panel and one rendering the whole document read one
    implementation rather than two that could disagree.
    """
    from imas_ambix.agent.live_panel import live_panel

    return live_panel(
        receipts_dir=Path(record_dir) if record_dir else None,
        lane_path=lane_path,
        now=now,
    )


def watch_document(
    record_dir: str | Path | None = None,
    index_path: str | Path | None = None,
    *,
    now: float | None = None,
    prices: Sequence[dict] | None = None,
    price_age: float | None = None,
    estimate: CacheRateEstimate | None = None,
    timeout: float = 5.0,
) -> dict:
    """Query the full document read-only, then return the panels' figures as data.

    The full document carries the index-derived ``record``, ``ledger`` and
    ``price_age`` fields alongside the ``lane`` and ``live`` blocks, so a
    consumer that wants every figure reads one document rather than reaching
    into the index on its own. It never ingests on a read and never creates an
    index: with no index file the ``index`` block reports it absent and the
    period rows are empty, because a read is not what keeps the store current.
    The ``lane`` and ``live`` blocks come from :func:`live_document`, which
    reads the published sources and never touches the index.
    A caller deriving a new document needs a resolved local index. The CLI
    reads an existing publication separately when it has no local index;
    derivation refuses that case so a publisher cannot give old figures a new
    publication timestamp.

    An index file that exists but cannot be read -- a zero-byte file the writer
    has opened and not yet written, or one whose lock is held past *timeout* --
    is reported in the ``index`` block as unreadable with the error it raised,
    and ``record``, ``ledger`` and ``periods`` are left empty rather than a
    traceback reaching the caller. *timeout* is how long the read waits for the
    writer's lock before giving up.
    """
    when = time.time() if now is None else now
    directory = Path(record_dir or default_record_dir())
    target = resolve_index_path(index_path)
    if target is None:
        raise FileNotFoundError(
            "no local watch index resolves; set AMBIX_WATCH_INDEX or FLEET_RUNTIME_DIR"
        )
    status = index_status(target, when)
    if status["exists"]:
        try:
            with TelemetryIndex.readonly(target, timeout=timeout) as index:
                if prices is None:
                    model = _latest_reading(index, when).get("model_id")
                    prices, age = _owned_prices(model)
                    if price_age is None:
                        price_age = age
                if estimate is None:
                    estimate = _installed_estimate()
                payload = document(
                    index,
                    now=when,
                    prices=prices,
                    price_age=price_age,
                    estimate=estimate,
                    history=_owned_history(),
                )
            status["readable"] = True
        except sqlite3.DatabaseError as error:
            status["readable"] = False
            status["error"] = str(error)
            payload = {
                "record": {},
                "ledger": [],
                "periods": [],
                "price_age": price_age,
            }
    else:
        payload = {
            "record": {},
            "ledger": [],
            "periods": [],
            "price_age": price_age,
        }
    payload["index"] = status
    payload.update(
        live_document(
            record_dir=directory,
            now=_dt.datetime.fromtimestamp(when, _dt.UTC),
        )
    )
    return payload


__all__ = [
    "ESTIMATE_FILENAME",
    "CacheRateEstimate",
    "default_record_dir",
    "LEDGER_PERIODS",
    "Period",
    "bar",
    "card_utilisation",
    "citation_text",
    "cost_label",
    "estimate_path",
    "estimated_cost",
    "fmt_span",
    "fmt_tokens",
    "fmt_usd",
    "ledger",
    "ledger_cost",
    "live_document",
    "load_estimate",
    "load_prices",
    "match_price",
    "observed_fraction",
    "period_row",
    "price_at",
    "watch_document",
]
