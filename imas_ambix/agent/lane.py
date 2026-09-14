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

import json
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

_SAMPLE = re.compile(
    r"^(?P<name>vllm:[a-z_]+)\{(?P<labels>[^}]*)\}\s+(?P<value>[-+0-9.eE]+)\s*$",
    re.MULTILINE,
)
_LABEL = re.compile(r'(?P<key>[a-z_0-9]+)="(?P<value>[^"]*)"')


@dataclass(frozen=True, slots=True)
class LaneCapacity:
    """One reading of the shared lane, with every figure's provenance explicit."""

    model_id: str
    pool_tokens: int
    running: int
    waiting: int
    kv_occupancy: float
    preemptions: int
    prefix_hit_rate: float | None

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

    @property
    def concurrent_requests(self) -> int | None:
        """How many requests of the CURRENTLY OBSERVED shape the pool holds.

        This is an extrapolation from the present mix, not a measured limit --
        see ``binding_observed``. It moves whenever the working context moves,
        which is why it is derived on every read instead of being written down.
        """
        mean = self.mean_context
        if mean is None or mean <= 0:
            return None
        return max(1, self.pool_tokens // mean)

    @property
    def headroom(self) -> int | None:
        """Additional requests of the present shape before the pool is full."""
        ceiling = self.concurrent_requests
        if ceiling is None:
            return None
        return max(0, ceiling - self.running)

    @property
    def binding_observed(self) -> bool:
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
        binding" is the error this flag exists to keep visible.
        """
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
            f"preemptions     {self.preemptions}",
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
                "STATUS          SATURATED — the engine is preempting and "
                "recomputing"
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


def parse_lane_capacity(metrics: str) -> LaneCapacity:
    """Build a reading from a Prometheus exposition body.

    Parsing text rather than taking numbers on trust keeps the pool size tied to
    the engine that is actually running, so a profile edit cannot make this
    disagree with the process it describes.
    """
    values: dict[str, float] = {}
    model_id = ""
    pool_tokens = 0
    for sample in _SAMPLE.finditer(metrics):
        name = sample.group("name")
        labels = dict(_LABEL.findall(sample.group("labels")))
        if name == "vllm:cache_config_info":
            pool_tokens = int(labels.get("kv_cache_size_tokens", "0") or 0)
        if "reason" in labels:
            continue
        model_id = model_id or labels.get("model_name", "")
        values[name] = float(sample.group("value"))

    if pool_tokens <= 0:
        raise ValueError("engine metrics carry no KV pool size")

    queries = values.get("vllm:prefix_cache_queries_total", 0.0)
    hits = values.get("vllm:prefix_cache_hits_total", 0.0)
    return LaneCapacity(
        model_id=model_id,
        pool_tokens=pool_tokens,
        running=int(values.get("vllm:num_requests_running", 0.0)),
        waiting=int(values.get("vllm:num_requests_waiting", 0.0)),
        kv_occupancy=values.get("vllm:kv_cache_usage_perc", 0.0),
        preemptions=int(values.get("vllm:num_preemptions_total", 0.0)),
        prefix_hit_rate=(hits / queries) if queries > 0 else None,
    )


def fetch_lane_capacity(origin: str, *, timeout: float = 10.0) -> LaneCapacity:
    """Read the live lane from a serve origin."""
    target = f"{origin.rstrip('/')}/metrics"
    request = urllib.request.Request(target, method="GET")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        body = response.read().decode("utf-8", "replace")
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
) -> Path:
    """Publish the reading so a session need not probe the engine to size work.

    Published rather than configured, and stamped, because a record carrying no
    observation time cannot be distinguished from a current one -- which is how
    a stale verdict gets read forward as a live fact.
    """
    from datetime import UTC, datetime

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
        "mean_context": capacity.mean_context,
        "concurrent_requests": capacity.concurrent_requests,
        "headroom": capacity.headroom,
        "binding_observed": capacity.binding_observed,
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
            "mean_context": capacity.mean_context,
            "running_at_observation": capacity.running,
            "formula": (
                "concurrent_requests = pool_tokens // mean_context; "
                "mean_context = pool_tokens * kv_occupancy / running; "
                "headroom = concurrent_requests - running"
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
        "headroom_is_upper_bound": settling is not False,
        "settling_caveat": (
            "a reading taken within ~60s of a dispatch reports the fleet at its "
            "lightest; prefer a pre-dispatch reading"
        ),
        # Declared, not enforced. The producer must not bake in a constant that
        # is right at one fleet age: measured 2026-09-14 the budget moved from
        # 143 to 38 in roughly twenty minutes as a fresh wave accumulated
        # context, so a bound generous at the start of a wave is tight in the
        # middle of one. The reader owns the decision; this is the default it
        # should apply absent its own policy.
        "suggested_shelf_life_seconds": 120,
    }
    scratch = target.with_suffix(".tmp")
    scratch.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")
    scratch.replace(target)
    target.chmod(0o644)
    return target
