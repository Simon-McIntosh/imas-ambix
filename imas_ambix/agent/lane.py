"""Derive the shared lane's concurrency budget from the engine's own counters.

The lane is one engine shared by every session on the workstation, so the only
quantity that composes across them is tokens resident in the shared KV pool. A
per-session seat allowance cannot be converted into that without knowing every
other session's working context, and five sessions each holding to the same
number carry either that number or five times it with nothing recording which.

Everything here is read from the engine at the moment of asking -- the pool size
included, which the engine publishes in ``cache_config_info`` -- so there is no
configured figure that a launch can forget and no stored value that can go stale
against the running process.

**This is advisory and must stay advisory.** It reports a budget; it never
refuses a request. Backpressure belongs to the engine, which queues, preempts
and recomputes rather than failing, and a second scheduler in front of it can
only refuse work the engine would have taken.
"""

from __future__ import annotations

import gzip
import json
import os
import urllib.request
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from imas_ambix.agent import engine_metrics

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

#: Fraction of the pool the advertised ceiling is allowed to plan against.
DEFAULT_OCCUPANCY_TARGET = 0.5

OCCUPANCY_TARGET_ENV = "IMAS_AMBIX_LANE_OCCUPANCY_TARGET"


def read_occupancy_target(environ: Mapping[str, str] | None = None) -> float:
    """Resolve the occupancy target, refusing a value that cannot be one.

    Refuses rather than clamps. A clamp turns an operator's mistake into a
    silently different capacity figure, and the figure is advice a dispatcher
    acts on -- so a typo would be indistinguishable from a deliberate setting
    for as long as the lane ran.
    """
    raw = (os.environ if environ is None else environ).get(OCCUPANCY_TARGET_ENV)
    if raw is None or not raw.strip():
        return DEFAULT_OCCUPANCY_TARGET
    try:
        target = float(raw)
    except ValueError:
        raise ValueError(
            f"{OCCUPANCY_TARGET_ENV}={raw!r} is not a number; "
            f"it is a fraction of the pool in (0, 1]"
        ) from None
    if not 0.0 < target <= 1.0:
        raise ValueError(
            f"{OCCUPANCY_TARGET_ENV}={raw!r} is outside (0, 1]; "
            f"0 advertises no capacity at all and above 1 plans for more pool "
            f"than exists"
        )
    return target


@dataclass(frozen=True, slots=True)
class LaneCapacity:
    """One reading of the shared lane, with every figure's provenance explicit."""

    model_id: str
    pool_tokens: int
    running: int
    waiting: int
    kv_occupancy: float
    preemptions: int | None
    prefix_hit_rate: float | None
    # Offload-tier health, None when the engine serves without the connector.
    # A store that is written and never read costs transfer bandwidth, host
    # memory and GPU staging space while returning nothing, and the write
    # counter climbing reads as healthy activity -- so the READ side is what a
    # consumer needs, and it is the half nobody was looking at.
    external_hit_rate: float | None = None
    offload_written_bytes: int | None = None
    offload_restored_bytes: int | None = None
    offload_resident_fraction: float | None = None
    hicache_host_total_tokens: int | None = None
    hicache_host_used_tokens: int | None = None
    # Safety ceiling, independent of workload. The pool arithmetic below is a
    # capacity estimate that rises without bound as the working context shrinks
    # -- on a cold lane with short prompts it read 131, which would invite a
    # dispatch far past anything this serve has survived. The crash edge is a
    # property of the memory configuration rather than the traffic: measured at
    # mem_fraction 0.92, clean at 128 concurrent and fatal at 256. This caps
    # what is advertised so the two cannot be confused.
    max_concurrent: int = 96

    @property
    def resident_tokens(self) -> int:
        """Tokens the pool is currently holding."""
        return round(self.pool_tokens * self.kv_occupancy)

    @property
    def mean_context(self) -> int | None:
        """Mean working context per running request, or None when idle.

        Undefined rather than zero when nothing is running: a lane with no
        traffic says nothing about how large its traffic is, and returning a
        figure there would invite a ceiling computed from noise.
        """
        if self.running <= 0:
            return None
        return round(self.resident_tokens / self.running)

    # The pool must hold active contexts AND the prefixes they will reuse next
    # turn. Sizing to the whole pool leaves nothing to retain: measured at 63%
    # active occupancy the hit rate fell to 23%, and the recomputation that
    # follows is itself what evicts the next session's prefix. That 63% is a
    # physical limit rather than a preference -- a consumer of this figure must
    # stay under it whatever the target is set to.
    #
    # The default lands the advertised ceiling at 23-24 concurrent for the
    # 81.7k-86.5k mean context observed on a 4M pool, which is very nearly the
    # measured throughput knee of ~22 workers. Past that knee the engine
    # absorbs pressure as slower generation rather than as a queue, so waiting
    # stays 0 and the hit rate stays high while aggregate throughput falls --
    # no cheap signal reddens. Raising the target needs a fresh throughput
    # curve, not an absence of refusals.
    #
    # Read once at import: the figure must not change under a lane that is
    # already serving from it.
    OCCUPANCY_TARGET = read_occupancy_target()

    def budget_for(self, mean_context: int | None) -> int:
        """The advertised figure for a given working context, in one place.

        Separated from the reading so the instantaneous mean and a windowed one
        go through identical arithmetic. Two call sites computing "the same"
        ceiling from different code is how a published figure and the advice
        derived from it silently diverge, which happened here once already.
        """
        if mean_context is None or mean_context <= 0:
            return self.max_concurrent
        usable = int(self.pool_tokens * self.OCCUPANCY_TARGET)
        return max(1, min(usable // mean_context, self.max_concurrent))

    @property
    def concurrent_requests(self) -> int | None:
        """Advertised capacity: the lesser of pool arithmetic and the ceiling.

        Two different quantities are combined deliberately. The pool term is a
        workload extrapolation that moves with the working context -- measured
        across one day it implied anywhere from 5 to 131. The ceiling is a
        property of the memory configuration and does not move with traffic.
        Publishing the pool term alone advertised 131 on a cold lane, which is
        beyond anything this serve has been shown to survive.

        On an idle lane the pool term is undefined, and the honest answer is the
        ceiling rather than ``None``: with nothing resident, what bounds a
        dispatch is the configuration, not the traffic. Returning ``None`` there
        forced every reader to guess, and a reader treating it as a hold would
        stall a ramp permanently while one treating it as a green light is right
        only by luck.
        """
        return self.budget_for(self.mean_context)

    @property
    def headroom(self) -> int | None:
        """Additional requests of the present shape before the pool is full."""
        ceiling = self.concurrent_requests
        if ceiling is None:
            return None
        return max(0, ceiling - self.running)

    @property
    def binding_observed(self) -> bool | None:
        """Whether SATURATION has actually been seen, on durable evidence only.

        Preemption is cumulative and monotonic: once the engine has recomputed
        a request it cannot un-recompute it, so a non-zero count cannot be a
        sampling artefact. A queue depth cannot make the same claim -- it is an
        instantaneous reading that is routinely non-zero for a single poll under
        normal scheduling, and every such event measured on this lane cleared
        within one or two polls with no preemption at all, including one at four
        waiting against two running.

        So waiting is deliberately NOT part of this. Measured 2026-09-14: the
        first version included it and went true at `waiting: 2` on a lane that
        was not saturated, with the next sample reading 1. A consumer gating on
        that would refuse work the engine would have taken, which is the defect
        the deleted admission filter embodied. Sustained waiting is reported
        separately, by a producer that can count consecutive samples.

        False means every ceiling here is extrapolated and none of it has been
        tested; treating "not binding at the loads we could produce" as "not
        binding" is the error this flag exists to keep visible. SGLang does not
        expose an equivalent cumulative preemption counter, so its readings
        return None rather than substituting a queue depth or a zero.
        """
        if self.preemptions is None:
            return None
        return self.preemptions > 0

    def summary(self) -> str:
        """Render one line per fact, naming what is measured and what is not."""
        mean = self.mean_context
        ceiling = self.concurrent_requests
        lines = [
            f"lane            {self.model_id}",
            f"pool            {self.pool_tokens:,} tokens",
            f"running         {self.running}",
            f"waiting         {self.waiting}",
            f"kv occupancy    {self.kv_occupancy * 100:.1f}%",
            f"resident        {self.resident_tokens:,} tokens",
            (
                f"preemptions     {self.preemptions}"
                if self.preemptions is not None
                else "preemptions     unavailable from this engine"
            ),
        ]
        if self.prefix_hit_rate is not None:
            hit = self.prefix_hit_rate * 100
            lines.append(f"prefix hits     {hit:.1f}% cumulative")
        if mean is None:
            lines.append("mean context    undefined (lane idle)")
            lines.append("fleet budget    undefined (no traffic to measure)")
        else:
            lines.append(f"mean context    {mean:,} tokens/request")
            lines.append(f"fleet budget    {ceiling} concurrent requests of this shape")
            lines.append(f"headroom        {self.headroom} more")
        if self.binding_observed:
            lines.append(
                "STATUS          SATURATED — the engine is preempting and recomputing"
            )
        elif self.binding_observed is None:
            lines.append(
                "STATUS          cumulative preemption pressure unavailable "
                "from this engine"
            )
        elif self.waiting > 0:
            lines.append(
                f"STATUS          {self.waiting} waiting this sample; a single "
                "poll is scheduler granularity, not pressure"
            )
        else:
            lines.append(
                "STATUS          nothing observed to bind; the budget is an "
                "extrapolation, not a measured limit"
            )
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class LaneWindow:
    """Several consecutive readings, because one ratio is not a level.

    ``mean_context`` divides a pool-resident token count by an INSTANTANEOUS
    request count, so it carries that count's sampling noise multiplied by
    whatever the numerator happens to be, and the noise is worst exactly where
    the denominator is smallest. Measured 2026-09-15 by two sessions
    independently: four readings inside one 120 s shelf life gave mean contexts
    of 19k, 247k, 14k and 51k, so the derived budget read 125, 1, 82 and 7.
    Every one of them was current -- this is not staleness, and no shelf life
    can express it. A wave sized from any single sample is sized from a figure
    that will be wrong within a minute in EITHER direction, which is what makes
    it different from the settling error: too low stalls a fleet, too high is
    the failure this module exists to prevent.

    The median is the estimator rather than the arithmetic mean because the
    excursions are spikes rather than a symmetric spread -- one 247k sample
    moves an average far more than it moves a median, and it is precisely the
    sample least likely to describe the next minute.

    The window does NOT replace ``settling``. They answer different questions
    and can disagree: settling asks whether the quantity was still moving in one
    direction when it was read, volatility asks whether repeated reads of a
    steady fleet land in the same place. A ramp is settling and not volatile; a
    lane whose request count flickers between polls is volatile and not
    settling. Both make the published figure an upper bound, for different
    reasons.
    """

    readings: tuple[LaneCapacity, ...]

    @property
    def latest(self) -> LaneCapacity:
        """The most recent reading; the window is never empty by construction."""
        return self.readings[-1]

    @property
    def defined_means(self) -> tuple[int, ...]:
        """Working contexts from readings that had traffic to measure.

        An idle reading contributes nothing rather than a zero. Folding "no
        traffic" in as a small context would drag the median toward a large
        budget, which is the generous direction every measurement error on this
        lane has already run in.
        """
        return tuple(
            sorted(mean for r in self.readings if (mean := r.mean_context) is not None)
        )

    @property
    def mean_context(self) -> int | None:
        """Median working context across the window, or None when all idle."""
        means = self.defined_means
        if not means:
            return None
        return means[len(means) // 2]

    @property
    def spread(self) -> tuple[int, int] | None:
        """Smallest and largest working context in the window.

        Published rather than reduced away: a reader that can see 14k to 247k
        knows to distrust any single figure derived from it, and a reader given
        only the median cannot tell a quiet lane from a thrashing one.
        """
        means = self.defined_means
        if not means:
            return None
        return means[0], means[-1]

    # A window whose extremes differ by more than this is reporting noise, not a
    # level. Set at 2.0 because the measured excursions were seventeenfold and a
    # doubling is already far beyond what a steady fleet produces between polls;
    # it is a threshold for flagging, never for refusing.
    VOLATILITY_RATIO = 2.0

    @property
    def is_volatile(self) -> bool:
        """Whether repeated reads of this lane disagree enough to distrust one."""
        bounds = self.spread
        if bounds is None or len(self.defined_means) < 2 or bounds[0] <= 0:
            return False
        return bounds[1] / bounds[0] > self.VOLATILITY_RATIO

    @property
    def concurrent_requests(self) -> int:
        """Advertised capacity, from the windowed context rather than the last."""
        return self.latest.budget_for(self.mean_context)

    @property
    def headroom(self) -> int:
        """Additional requests before the windowed budget is reached."""
        return max(0, self.concurrent_requests - self.latest.running)


def _lane_reader(
    samples: list[tuple[str, dict[str, str], float]], family: str
) -> Callable[[str], float | None]:
    """A role-to-value reader over one family's eligible samples.

    Every spelling comes from :mod:`imas_ambix.agent.engine_metrics`, so this
    reader and the receipts recorder cannot disagree about what a family
    publishes; a role the family does not publish resolves to ``None``.
    """
    eligible = engine_metrics.eligible_samples(samples, family)

    def read(role: str) -> float | None:
        return engine_metrics.series_first(
            eligible, family, engine_metrics.lane_series(role, family)
        )

    return read


def _lane_pool_tokens(
    samples: list[tuple[str, dict[str, str], float]],
    family: str,
    read: Callable[[str], float | None],
) -> int:
    """The KV pool size, in whichever form the family publishes it.

    SGLang publishes it as a gauge. vLLM publishes it as a label on its cache
    configuration info, so there it is read by label rather than as a series.
    """
    if family == engine_metrics.FAMILY_SGLANG:
        pool = read("pool_tokens")
        return int(pool) if pool is not None else 0
    labels = engine_metrics.series_labels(
        samples, family, (engine_metrics.VLLM_CACHE_CONFIG_INFO,)
    )
    if labels is None:
        return 0
    return int(labels.get(engine_metrics.VLLM_KV_POOL_LABEL, "0") or 0)


def _lane_capacity_from_roles(
    family: str,
    model_id: str,
    pool_tokens: int,
    read: Callable[[str], float | None],
) -> LaneCapacity:
    """Assemble one reading from canonical roles.

    The families differ in two places, and both are about what the engine
    reports rather than about what the quantity means: SGLang publishes a
    prefix hit rate directly and a preemption-free engine, while vLLM publishes
    two cumulative prefix-cache counters and a cumulative preemption counter,
    so its hit rate is their ratio. A role the family does not publish stays
    absent from the reading rather than contributing a zero.
    """
    occupancy = read("kv_pool_occupancy")
    preemptions = read("preemptions")
    external_queries = read("external_queries")
    external_hits = read("external_hits")
    written = read("offload_written_bytes")
    restored = read("offload_restored_bytes")
    host_total = read("hicache_host_total_tokens")
    host_used = read("hicache_host_used_tokens")
    queries = read("prefix_cache_queries") or 0.0
    hits = read("prefix_cache_hits") or 0.0
    prefix_hit_rate = read("prefix_hit_rate")
    if prefix_hit_rate is None and queries > 0:
        prefix_hit_rate = hits / queries
    if family == engine_metrics.FAMILY_VLLM:
        preemptions = int(preemptions) if preemptions is not None else 0
    return LaneCapacity(
        model_id=model_id,
        pool_tokens=pool_tokens,
        running=int(read("requests_running") or 0),
        waiting=int(read("requests_queued") or 0),
        kv_occupancy=occupancy if occupancy is not None else 0.0,
        preemptions=int(preemptions) if preemptions is not None else None,
        prefix_hit_rate=prefix_hit_rate,
        external_hit_rate=(
            (external_hits / external_queries)
            if external_queries and external_hits is not None
            else None
        ),
        offload_written_bytes=int(written) if written is not None else None,
        offload_restored_bytes=int(restored) if restored is not None else None,
        offload_resident_fraction=read("offload_resident_fraction"),
        hicache_host_total_tokens=int(host_total) if host_total is not None else None,
        hicache_host_used_tokens=int(host_used) if host_used is not None else None,
    )


def parse_lane_capacity(metrics: str) -> LaneCapacity:
    """Build a reading from a Prometheus exposition body.

    Parsing text rather than taking numbers on trust keeps the pool size tied to
    the engine that is actually running, so a profile edit cannot make this
    disagree with the process it describes.
    """
    samples = engine_metrics.parse_metrics(metrics)
    family = engine_metrics.detect_family(samples)
    if family is None:
        raise ValueError("engine metrics carry no KV pool size")
    read = _lane_reader(samples, family)
    model_id = engine_metrics.model_id_of(samples, family) or ""
    pool_tokens = _lane_pool_tokens(samples, family, read)
    if pool_tokens <= 0:
        raise ValueError("engine metrics carry no KV pool size")
    return _lane_capacity_from_roles(family, model_id, pool_tokens, read)


def fetch_lane_capacity(origin: str, *, timeout: float = 10.0) -> LaneCapacity:
    """Read the live lane from a serve origin."""
    target = f"{origin.rstrip('/')}/metrics"
    request = urllib.request.Request(target, method="GET")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        body = response.read()
        encodings = [
            encoding.strip().lower()
            for encoding in response.headers.get("Content-Encoding", "").split(",")
            if encoding.strip()
        ]
    for encoding in reversed(encodings):
        if encoding in {"gzip", "x-gzip"}:
            body = gzip.decompress(body)
        elif encoding == "deflate":
            body = zlib.decompress(body)
        else:
            raise ValueError(f"unsupported metrics content encoding: {encoding}")
    body = body.decode("utf-8", "replace")
    return parse_lane_capacity(body)


def detect_settling(
    previous: LaneCapacity | None,
    current: LaneCapacity,
    *,
    tolerance: float = 0.25,
) -> bool | None:
    """Is the measured quantity itself still moving, independent of its age?

    Age and settling are orthogonal qualifiers and no age rule can express
    this: a reading can be one second old and invalid, or ninety seconds old
    and perfectly good. Age asks whether the world moved since we looked;
    settling asks whether the thing we looked at was in a steady state when we
    looked.

    It is detectable only by a producer, because it needs consecutive samples.
    A reader holding one sample cannot distinguish a fleet mid-settle from a
    genuine steady state at the same value. Measured 2026-09-14 at unchanged
    ``running`` of 19: mean context 15,586 then ~62,865 then 73,121 tokens, so
    the budget read 141, 35, 30 within a minute -- a reader seeing only the
    first could not have known.

    Returns None when there is no previous sample, or when the running count
    moved enough that a change in mean context is explained by the mix rather
    than by settling. Unknown is reported as unknown rather than as settled.
    """
    if previous is None:
        return None
    if abs(current.running - previous.running) > 1:
        return None
    before, after = previous.mean_context, current.mean_context
    if not before or not after:
        return None
    return abs(after - before) / before > tolerance


def write_unavailable_document(
    reason: str, path: str | Path, *, model_id: str = ""
) -> Path:
    """Publish the fact that the lane could NOT be read, and why.

    The headroom key is omitted rather than set to null: a present-but-null
    field invites exactly one wrong reading, while a missing key raises on a
    reader that assumed it, failing loudly at the layer that knows. ``reason``
    is required because "unavailable" alone rebuilds the not-yet-published
    versus gone ambiguity one field down.
    """
    from datetime import UTC, datetime

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "observed_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "state": "unavailable",
        "reason": reason,
        "model_id": model_id,
    }
    scratch = target.with_suffix(".tmp")
    scratch.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")
    scratch.replace(target)
    target.chmod(0o644)
    return target


def classify_reading(
    document: dict[str, object], *, now: object = None, shelf_life_seconds: int = 120
) -> str:
    """Return "measured", "stale" or "unavailable" for a published reading.

    Staleness is the READER's classification, not the writer's -- a document is
    always fresh at the moment it is written, and a producer that baked in a
    constant would be right at one fleet age only. Provided here so every
    reader does not reinvent it differently.

    A stale reading KEEPS its figure and reports its age. "We measured 15 four
    minutes ago" and "we could not measure" are different facts, and only the
    first helps somebody debugging; a gate may treat stale as unknown while the
    record retains the discriminating value.
    """
    from datetime import UTC, datetime

    if document.get("state") == "unavailable":
        return "unavailable"
    stamp = document.get("observed_at")
    if not isinstance(stamp, str):
        return "unavailable"
    try:
        observed = datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError:
        return "unavailable"
    current = now if isinstance(now, datetime) else datetime.now(UTC)
    age = (current - observed).total_seconds()
    return "stale" if age > shelf_life_seconds else "measured"


def write_lane_document(
    capacity: LaneCapacity,
    path: str | Path,
    *,
    settling: bool | None = None,
    refresh_interval: int = 30,
    window: LaneWindow | None = None,
) -> Path:
    """Publish the reading so a session need not probe the engine to size work.

    Published rather than configured, and stamped, because a record carrying no
    observation time cannot be distinguished from a current one -- which is how
    a stale verdict gets read forward as a live fact.

    ``concurrent_requests`` and ``headroom`` are published from ``window`` when
    one is supplied, because a single sample of this particular ratio is not a
    level -- see ``LaneWindow``. The instantaneous figures are published beside
    them rather than dropped: a reader comparing the two can see how much the
    smoothing moved, and a reader debugging a specific moment still has the
    number that moment produced.
    """
    from datetime import UTC, datetime

    if window is None:
        window = LaneWindow(readings=(capacity,))
    spread = window.spread

    # An IDLE lane is the dangerous case, not the harmless one. Publishing the
    # ceiling while the pool term is undefined turns an absent workload
    # measurement into the largest available dispatch suggestion.
    #
    # The decisive constraint is WHEN the field is read. A coordinator consults it to
    # decide how large a wave to resume, which is precisely when the lane is
    # quiet -- so the undefined case does not merely return a vague number, it
    # returns the MAXIMUM one at exactly the moment of the largest dispatch
    # decision. Measured 2026-09-15: the highest figure any session recorded,
    # 96, was taken at kv_occupancy 0.0 four minutes after a restart, against a
    # real budget of 35 minutes later. The refusal below is what a reader needs
    # there; the verdict field is what makes it readable rather than a guess.
    idle = window.mean_context is None
    sizing_usable = not idle and not window.is_volatile
    if idle:
        sizing_reason = (
            "nothing is resident, so there is no working context to divide by; "
            "a quiet lane says nothing about how large its traffic will be, and "
            "this is the moment a resuming coordinator reads it"
        )
    elif window.is_volatile and spread is not None:
        sizing_reason = (
            "the working-context denominator is oscillating across this window "
            f"({spread[0]:,}-{spread[1]:,} tokens); size from your dependency "
            "graph and from waiting/preemptions/kv_occupancy, which stayed "
            "stable across the same window"
        )
    else:
        sizing_reason = "the working context is steady across this window"

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "observed_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_id": capacity.model_id,
        "pool_tokens": capacity.pool_tokens,
        "running": capacity.running,
        "waiting": capacity.waiting,
        "kv_occupancy": round(capacity.kv_occupancy, 4),
        "preemptions": capacity.preemptions,
        "mean_context": window.mean_context,
        # `concurrent_requests` and `headroom` appear HERE ONLY WHEN THE FIELD
        # CAN ANSWER. When it cannot they move under `withheld` below, so the
        # document leads with a refusal rather than with a number. A number
        # that is present and wrong gets consumed; a refusal gets read.
        #
        # This repository has landed the same shape twice for the same reason:
        # the budget fence reports unknown rather than a low utilisation, and
        # the run reader refuses rather than half-decoding a status. In both,
        # a partial answer was judged worse than none.
        **(
            {
                "concurrent_requests": window.concurrent_requests,
                "headroom": window.headroom,
            }
            if sizing_usable
            else {
                "withheld": {
                    "concurrent_requests": window.concurrent_requests,
                    "headroom": window.headroom,
                    "why": sizing_reason,
                }
            }
        ),
        "binding_observed": capacity.binding_observed,
        # Cumulative since this engine started, which is the only form the
        # engine offers -- so it spans whatever mix of load has run since, and
        # two such figures from different eras are not comparable. Published
        # because it was the single most load-bearing signal on this lane and
        # no consumer could see it: prefix-cache eviction is the FIRST symptom
        # of KV pressure and shows up long before preemption, so a reader
        # watching only `preemptions` learns nothing until it is far too late.
        "prefix_hit_rate": (
            round(capacity.prefix_hit_rate, 4)
            if capacity.prefix_hit_rate is not None
            else None
        ),
        # Absent entirely when the engine runs without an offload connector,
        # rather than present and zero -- a missing key fails loudly on a
        # reader that assumed it, where a zero reads as a measured verdict.
        **(
            {
                "offload": {
                    "external_hit_rate": round(capacity.external_hit_rate, 6)
                    if capacity.external_hit_rate is not None
                    else None,
                    "written_bytes": capacity.offload_written_bytes,
                    "restored_bytes": capacity.offload_restored_bytes,
                    "resident_fraction": capacity.offload_resident_fraction,
                    # The read side is the whole question. A store whose
                    # written figure climbs while restored stays flat is
                    # spending bandwidth and host memory for nothing, and it
                    # looks busy the entire time.
                    "note": (
                        "restored_bytes flat against a climbing written_bytes "
                        "means the store is not serving reads"
                    ),
                }
            }
            if capacity.offload_written_bytes is not None
            else {}
        ),
        # SGLang's host cache is a separate observable tier. Keep absence as
        # absence: a serve without hierarchical caching must not resemble an
        # empty eight-million-token host pool.
        **(
            {
                "hicache_host": {
                    "total_tokens": capacity.hicache_host_total_tokens,
                    "used_tokens": capacity.hicache_host_used_tokens,
                    "used_fraction": round(
                        capacity.hicache_host_used_tokens
                        / capacity.hicache_host_total_tokens,
                        6,
                    )
                    if capacity.hicache_host_total_tokens
                    and capacity.hicache_host_used_tokens is not None
                    else None,
                }
            }
            if capacity.hicache_host_total_tokens is not None
            else {}
        ),
        # What this sample alone said, kept beside the smoothed figure. Sizing
        # from these is the defect the window exists to fix; they are here so
        # the smoothing is auditable rather than invisible.
        "mean_context_instant": capacity.mean_context,
        "concurrent_requests_instant": capacity.concurrent_requests,
        "headroom_instant": capacity.headroom,
        # The evidence for distrusting any one figure, published rather than
        # reduced away. A reader seeing 14,000 to 247,000 here knows what the
        # median is standing in for; a reader given only the median cannot tell
        # a settled lane from a flickering one.
        "mean_context_spread": list(spread) if spread is not None else None,
        "window_samples": len(window.readings),
        "window_seconds": len(window.readings) * refresh_interval,
        "volatile": window.is_volatile,
        # What to DO, not a flag to interpret. A spread of 1 to 96 tells a
        # coordinator to stop sizing from this field -- but only if it reads
        # the spread, converts it to a judgement, and acts. A smoothed figure
        # published alone moves the same wrong confidence onto a rounder
        # number, and a `volatile` boolean still leaves the decision to be
        # re-derived by every reader, differently.
        #
        # Measured across four sessions on 2026-09-15, this field read 125, 1,
        # 82, 7, 2, 62, 1, 53 and 0 -- oscillating faster than the interval
        # between two coordinators consulting it, so any two of them formed
        # contradictory plans and both were right. When that is the state, the
        # honest output is "do not size from me" stated once, here, rather than
        # nine readers inferring it nine ways.
        "sizing_verdict": ("usable" if sizing_usable else "do-not-size"),
        "sizing_reason": sizing_reason,
        # Validity lives in its own field, never in the figure. `0` and `null`
        # are both falsy, so a reader writing `if not headroom` collapses "the
        # lane is full" into "we could not measure" -- opposite facts. Checking
        # `state` first is the contract; zero headroom is then unmistakably
        # {"state": "measured", "headroom": 0}.
        "state": "measured",
        # A number and its vintage is better than a number; a number, its
        # vintage and its DENOMINATOR is the thing that cannot quietly become
        # false. A utilisation figure computed against the wrong context window
        # is arithmetically perfect and undetectable by inspection, so every
        # derived figure here names what it was divided by.
        "derived_from": {
            "pool_tokens": capacity.pool_tokens,
            "mean_context": window.mean_context,
            "occupancy_target": LaneCapacity.OCCUPANCY_TARGET,
            "running_at_observation": capacity.running,
            "formula": (
                "concurrent_requests = "
                "(pool_tokens * occupancy_target) // mean_context, "
                f"capped at {capacity.max_concurrent}; "
                "mean_context = MEDIAN over the window of "
                "(pool_tokens * kv_occupancy / running); "
                "headroom = concurrent_requests - running_at_observation"
            ),
        },
        # The budget collapses within the first minute of a wave dispatching.
        # Measured 2026-09-14 across three consecutive 30-second samples with
        # `running` unchanged at 19: mean context 15,586 -> ~62,865 -> 73,121
        # tokens, so the budget read 141, then 35, then 30. Nothing joined or
        # left; the same requests went from their first tokens to their real
        # working context. So the WORST moment to read this document is
        # immediately after dispatching, which is exactly when a coordinator
        # would naturally read it. Prefer a reading taken before a dispatch to
        # one taken after, and treat any reading whose age is under about a
        # minute on a just-widened fleet as provisional.
        # Orthogonal to age. True means the denominator was still moving when
        # this was taken, so headroom is an UPPER BOUND rather than a figure --
        # the measured error runs one way, roughly eightfold and always
        # generous. None means no previous sample to compare, reported as
        # unknown rather than as settled.
        "settling": settling,
        # Unknown counts as an upper bound, not as a firm figure. Measured
        # 2026-09-14: a reading of `headroom 134` published unflagged because
        # the running count had moved by 3 between polls, which makes settling
        # UNKNOWN rather than true -- and unknown was being rendered as
        # settled. The very next sample read 17. So the single most dangerous
        # reading, taken moments after a dispatch when the mix is also moving,
        # was the one case the flag did not cover.
        #
        # Every measurement error found on this lane runs toward apparent
        # headroom, so the unknown case resolves that way too: when it cannot
        # be established that the quantity has stopped moving, say the figure
        # is a bound rather than a value.
        #
        # Volatility resolves the same way and for the same reason. A window
        # spanning 14k to 247k tokens cannot support a firm figure whichever
        # estimator is used, and the median of a noisy sample is still a
        # summary of noise.
        "headroom_is_upper_bound": settling is not False or window.is_volatile,
        "settling_caveat": (
            "a reading taken within ~60s of a dispatch reports the fleet at its "
            "lightest; prefer a pre-dispatch reading"
        ),
        # Derived from the publishing cadence rather than written down, so it
        # follows if the cadence changes. One and a half intervals: a reading is
        # at most one interval old when a fresh one is due, and the half is
        # margin for a late poll.
        #
        # It was a flat 120 s until 2026-09-15, when a peer sampling every
        # twenty seconds watched lane_headroom read 21, then 11, then 14 across
        # forty seconds while this field claimed two minutes of freshness. A
        # shelf life several times the measured volatility does not make a
        # figure safe to use -- it clears a stale one for action, which is worse
        # than publishing no shelf life at all. The reader still owns the
        # decision; this is the default it should apply absent its own policy.
        "suggested_shelf_life_seconds": int(refresh_interval * 1.5),
    }
    scratch = target.with_suffix(".tmp")
    scratch.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")
    scratch.replace(target)
    target.chmod(0o644)
    return target
