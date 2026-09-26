"""One owner for the metric-name knowledge of every engine family we serve.

A ``/metrics`` scrape names the same physical quantity differently per engine
family, and a reader that knows one family reports a healthy serve as absent:
measured on a live SGLang serve, fourteen of the recorder's twenty fields were
``null`` in every one of 13,791 rows because the reader resolved only the
``vllm:`` namespace. Anything that resolves a serving quantity -- the
continuous recorder, the benchmark, the lane's capacity reader -- resolves it
here, so the mapping exists once.

**Absence from a family is a value, and it is expressed by omission.** A
quantity a family does not publish is left out of a reading rather than
recorded as a zero or a null, because a null that reads as a reading is exactly
the defect being repaired. :meth:`EngineMetrics.row_section` therefore never
emits a key it did not observe, and a quantity genuinely published as zero is
kept with its zero.

**Selection discipline.** SGLang repeats device-pool measurements on every
tensor-parallel rank, so only the rank-zero sample is one shared pool rather
than N pools to sum. vLLM publishes partial series labelled by ``reason``
(preemption breakdowns, for instance), which are sub-series that must not be
folded into their family's total. :func:`eligible_samples` applies both rules,
so every caller shares one definition of which samples count.

**This module reports; it never probes and never decides.** It takes exposition
text and returns quantities.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

FAMILY_SGLANG = "sglang"
FAMILY_VLLM = "vllm"
FAMILIES: tuple[str, ...] = (FAMILY_SGLANG, FAMILY_VLLM)

# ── Exposition parsing ───────────────────────────────────────────────
# ``name{label="value",…} value`` — the Prometheus text exposition format.
# One parser for every caller: a second parser is how two readers come to
# disagree about what a scrape contains.
_SAMPLE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{(?P<labels>[^}]*)\})?"
    r"\s+(?P<value>\S+)"
)
_LABEL_RE = re.compile(
    r'(?P<key>[a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*"(?P<val>(?:[^"\\]|\\.)*)"'
)

#: One parsed sample: fully qualified name, labels, value.
MetricSample = tuple[str, dict[str, str], float]

# ── Canonical quantities, and the series that carry them ─────────────
# Each table maps a canonical name to the bare series names (the name with its
# family prefix stripped) a family publishes it under. An empty tuple means the
# family does not publish it, so the reading omits the quantity entirely.
GAUGE_SERIES: dict[str, dict[str, tuple[str, ...]]] = {
    "requests_running": {
        FAMILY_SGLANG: ("num_running_reqs",),
        FAMILY_VLLM: ("num_requests_running",),
    },
    "requests_queued": {
        FAMILY_SGLANG: ("num_queue_reqs",),
        FAMILY_VLLM: ("num_requests_waiting",),
    },
    "kv_pool_occupancy": {
        FAMILY_SGLANG: ("full_token_usage", "token_usage"),
        FAMILY_VLLM: ("kv_cache_usage_perc", "gpu_cache_usage_perc"),
    },
    # The prefix-cache hit rate. The two families report it in different shapes
    # rather than under different spellings, and the column records which: SGLang
    # publishes the rate itself, as one gauge. vLLM publishes no rate gauge, only
    # the cumulative hits and queries counters in ``COUNTER_SERIES``, from which
    # the record layer derives the ratio. Nothing is invented here for vLLM,
    # because a rate assembled from one family's counters would be a second
    # implementation of a derivation that already exists one layer up.
    "prefix_cache_hit_rate": {
        FAMILY_SGLANG: ("cache_hit_rate",),
        FAMILY_VLLM: (),
    },
}

COUNTER_SERIES: dict[str, dict[str, tuple[str, ...]]] = {
    "prompt_tokens": {
        FAMILY_SGLANG: ("prompt_tokens_total",),
        FAMILY_VLLM: ("prompt_tokens_total",),
    },
    "generation_tokens": {
        FAMILY_SGLANG: ("generation_tokens_total",),
        FAMILY_VLLM: ("generation_tokens_total",),
    },
    # The prefix-cache counters are the cumulative decomposition the legacy
    # row's hit-rate fields read. vLLM publishes them under both a plain and a
    # ``gpu_``-prefixed spelling. SGLang publishes the cached-token counter
    # (labelled by the cache tier that answered) but no queries counter at all:
    # its denominator is a sum of prefill modes rather than a series, so this
    # quantity is absent from an SGLang reading rather than zero, and the rate
    # it does publish is read from ``prefix_cache_hit_rate`` above.
    "prefix_cache_queries": {
        FAMILY_SGLANG: (),
        FAMILY_VLLM: ("prefix_cache_queries_total", "gpu_prefix_cache_queries_total"),
    },
    "prefix_cache_hits": {
        FAMILY_SGLANG: ("cached_tokens_total",),
        FAMILY_VLLM: ("prefix_cache_hits_total", "gpu_prefix_cache_hits_total"),
    },
}

HISTOGRAM_SERIES: dict[str, dict[str, tuple[str, ...]]] = {
    "time_to_first_token": {
        FAMILY_SGLANG: ("time_to_first_token_seconds",),
        FAMILY_VLLM: ("time_to_first_token_seconds",),
    },
    "inter_token_latency": {
        FAMILY_SGLANG: ("inter_token_latency_seconds",),
        FAMILY_VLLM: ("inter_token_latency_seconds",),
    },
}

SPEC_DECODE_SERIES: dict[str, dict[str, tuple[str, ...]]] = {
    "draft_tokens_total": {
        FAMILY_SGLANG: (),
        FAMILY_VLLM: ("spec_decode_num_draft_tokens_total",),
    },
    "accepted_tokens_total": {
        FAMILY_SGLANG: (),
        FAMILY_VLLM: ("spec_decode_num_accepted_tokens_total",),
    },
}

#: Speculative-decode quantities reported as an instantaneous gauge rather than
#: as the cumulative pair above. The two families' vocabularies do not
#: correspond one to one, so each column states what it measures for each
#: family instead of mapping one family's spelling onto the other's meaning:
#:
#: * ``accept_rate`` -- ``accepted drafts / proposed drafts`` in batch. SGLang
#:   publishes it as a gauge. vLLM publishes no gauge and no rate; a rate is
#:   derivable from the two counters above, and it is not derived here.
#: * ``accept_length`` -- mean acceptance length per forward (accepted drafts
#:   plus the bonus token). SGLang publishes it as a gauge. vLLM publishes no
#:   gauge and nothing from which it is recoverable.
#: * ``active_draft_tokens`` -- the currently active
#:   ``speculative_num_draft_tokens``: a configuration value, not a count of
#:   tokens drafted. vLLM's similarly-named ``spec_decode_num_draft_tokens_total``
#:   is the cumulative counter carried by ``draft_tokens_total`` above; the two
#:   are different quantities under similar names, which is why they are
#:   separate columns rather than one.
SPEC_GAUGE_SERIES: dict[str, dict[str, tuple[str, ...]]] = {
    "accept_rate": {
        FAMILY_SGLANG: ("spec_accept_rate",),
        FAMILY_VLLM: (),
    },
    "accept_length": {
        FAMILY_SGLANG: ("spec_accept_length",),
        FAMILY_VLLM: (),
    },
    "active_draft_tokens": {
        FAMILY_SGLANG: ("spec_num_draft_tokens",),
        FAMILY_VLLM: (),
    },
}

#: Per-draft-position acceptance. Two spellings exist across engine versions;
#: both are listed, and each one's ``_created`` sibling is excluded by the
#: exact-name match rather than by a substring test.
SPEC_DECODE_PER_POSITION: dict[str, tuple[str, ...]] = {
    FAMILY_VLLM: (
        "spec_decode_num_accepted_tokens_per_pos_total",
        "spec_decode_num_accepted_tokens_per_pos",
    ),
}

# SGLang publishes cached prompt tokens by the cache tier that answered them,
# as one counter labelled ``mode``. ``input`` is the uncached remainder.
SGLANG_CACHE_SERIES = "prefill_effective_tokens_total"
SGLANG_CACHE_TIERS: dict[str, str] = {
    "device_hit": "device",
    "host_hit": "host",
    "storage_hit": "storage",
}
SGLANG_UNCACHED_MODE = "input"

# vLLM's tier split for the same decomposition is by cache location: the
# in-instance prefix cache, and the offload connector when it is enabled.
VLLM_CACHE_TIERS: dict[str, str] = {
    "prefix_cache_hits_total": "device",
    # ``_active``/``_created`` companions are excluded by the exact-name match.
    "external_prefix_cache_hits_total": "external",
}

#: vLLM publishes its KV pool size and serving configuration as labelled
#: gauges rather than as plain series, so they are read by label.
VLLM_CACHE_CONFIG_INFO = "cache_config_info"
VLLM_KV_POOL_LABEL = "kv_cache_size_tokens"

# ── Lane-only series ─────────────────────────────────────────────────
# Quantities the lane reader needs that have no canonical role. They share the
# name knowledge here so that no family spelling lives outside this module; a
# role mapping to an empty tuple is one the family does not publish.
LANE_SERIES: dict[str, dict[str, tuple[str, ...]]] = {
    "pool_tokens": {
        FAMILY_SGLANG: ("max_total_num_tokens",),
        # vLLM's pool size is a label on ``cache_config_info``, not a series.
        FAMILY_VLLM: (),
    },
    "prefix_hit_rate": {
        FAMILY_SGLANG: ("cache_hit_rate",),
        # vLLM publishes no such gauge: its rate is the hits/queries ratio the
        # caller computes from the two cumulative counters.
        FAMILY_VLLM: (),
    },
    "preemptions": {
        FAMILY_SGLANG: (),
        FAMILY_VLLM: ("num_preemptions_total",),
    },
    "external_queries": {
        FAMILY_SGLANG: (),
        FAMILY_VLLM: ("external_prefix_cache_queries_total",),
    },
    "external_hits": {
        FAMILY_SGLANG: (),
        FAMILY_VLLM: ("external_prefix_cache_hits_total",),
    },
    "offload_written_bytes": {
        FAMILY_SGLANG: (),
        FAMILY_VLLM: ("kv_offload_store_bytes_total",),
    },
    "offload_restored_bytes": {
        FAMILY_SGLANG: (),
        FAMILY_VLLM: ("kv_offload_load_bytes_total",),
    },
    "offload_resident_fraction": {
        FAMILY_SGLANG: (),
        FAMILY_VLLM: ("kv_offload_cpu_cache_usage_perc",),
    },
    "hicache_host_total_tokens": {
        FAMILY_SGLANG: ("hicache_host_total_tokens",),
        FAMILY_VLLM: (),
    },
    "hicache_host_used_tokens": {
        FAMILY_SGLANG: ("hicache_host_used_tokens",),
        FAMILY_VLLM: (),
    },
}

#: Lane roles whose series are cumulative counters rather than gauges, and so
#: share the label-set arithmetic of ``COUNTER_SERIES`` above. Their engine
#: spellings differ from the canonical role names and they are not in
#: ``COUNTER_SERIES``, which is why the two are not the same set.
#:
#: A counter carries one running total whose label sets — ``is_streaming``,
#: ``engine`` rank, a device/``reason`` breakdown that is not filtered — are
#: contributions to that total, so every one of them sums. A gauge instead
#: repeats one instantaneous reading on every rank, so its label sets must not
#: be added. Reading a counter as a gauge takes one series and reports it as the
#: whole, which is the defect this set exists to prevent.
LANE_COUNTER_ROLES: frozenset[str] = frozenset(
    {
        "preemptions",
        "external_queries",
        "external_hits",
        "offload_written_bytes",
        "offload_restored_bytes",
    }
)


def family_of(name: str) -> str | None:
    """The engine family a fully qualified sample name belongs to."""
    prefix = name.rpartition(":")[0]
    return prefix if prefix in FAMILIES else None


def parse_metrics(text: str) -> list[MetricSample]:
    """Parse an exposition body into ``(name, labels, value)`` samples.

    Comments, blank lines and samples with an unparseable value are skipped: a
    scrape is diagnostic, so one malformed line must not discard the counters
    around it.
    """
    samples: list[MetricSample] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE_RE.match(line)
        if match is None:
            continue
        try:
            value = float(match.group("value"))
        except ValueError:
            continue
        labels = {
            m.group("key"): m.group("val")
            for m in _LABEL_RE.finditer(match.group("labels") or "")
        }
        samples.append((match.group("name"), labels, value))
    return samples


def detect_family(samples: list[MetricSample]) -> str | None:
    """The family that prefixes *samples*, or ``None`` when neither does.

    Detection is by family prefix rather than by a particular metric name, so a
    document carrying only some of a family's series is still attributed
    correctly.
    """
    for family in FAMILIES:
        prefix = f"{family}:"
        if any(name.startswith(prefix) for name, _, _ in samples):
            return family
    return None


def eligible_samples(samples: list[MetricSample], family: str) -> list[MetricSample]:
    """*samples* reduced to those describing one shared reading.

    SGLang repeats every device-pool measurement on each tensor-parallel rank,
    so the rank-zero sample is one shared pool and the others are the same
    reading repeated. vLLM's ``reason``-labelled series are breakdowns of a
    total rather than the total, and summing them into it double-counts.
    """
    prefix = f"{family}:"
    kept: list[MetricSample] = []
    for name, labels, value in samples:
        if not name.startswith(prefix):
            continue
        if family == FAMILY_SGLANG and labels.get("tp_rank", "0") != "0":
            continue
        if family == FAMILY_VLLM and "reason" in labels:
            continue
        kept.append((name, labels, value))
    return kept


def _bare(name: str) -> str:
    """A sample's name with its family prefix stripped."""
    return name.rpartition(":")[2]


def _of_family(samples: list[MetricSample], family: str) -> list[MetricSample]:
    """The subset of *samples* that *family*'s namespace publishes.

    A bare series name is shared vocabulary rather than an identity — SGLang
    and vLLM both spell a prompt-token counter ``prompt_tokens_total`` — so a
    reader told to answer for one family must not be answered by the other's
    samples. Absence is the honest reading there, and it is what distinguishes
    this from a lookup that ignores the family it was given.
    """
    prefix = f"{family}:"
    return [sample for sample in samples if sample[0].startswith(prefix)]


def series_total(
    samples: list[MetricSample], family: str, names: tuple[str, ...]
) -> float | None:
    """Sum of one series' values, or ``None`` when the family publishes none.

    Summed across label sets so a multi-label scrape totals correctly, and
    scoped to the family that was asked for so a series name shared by both
    families cannot be answered by the other family's samples.
    """
    total: float | None = None
    for name, _labels, value in _of_family(samples, family):
        if _bare(name) in names:
            total = value if total is None else total + value
    return total


def series_first(
    samples: list[MetricSample], family: str, names: tuple[str, ...]
) -> float | None:
    """First value of one series, for gauges repeated across label sets.

    A gauge repeated per rank or per engine is one reading, so the first is the
    reading rather than a term to be summed.
    """
    for name, _labels, value in _of_family(samples, family):
        if _bare(name) in names:
            return value
    return None


def series_labels(
    samples: list[MetricSample], family: str, names: tuple[str, ...]
) -> dict[str, str] | None:
    """Labels of the first sample of one series, or ``None`` when absent."""
    for name, labels, _value in _of_family(samples, family):
        if _bare(name) in names:
            return labels
    return None


def model_id_of(samples: list[MetricSample], family: str) -> str | None:
    """The served name a family's samples label themselves with."""
    for name, labels, _value in samples:
        if not name.startswith(f"{family}:"):
            continue
        model_name = labels.get("model_name")
        if model_name:
            return model_name
    return None


def tier_split(
    samples: list[MetricSample], family: str
) -> tuple[dict[str, float], float | None]:
    """Cached prompt tokens by the tier that answered, and the uncached rest.

    Each family decomposes the same quantity differently and this is the
    reconciliation: SGLang labels one counter by mode, vLLM publishes one
    series per cache location. Either way the caller reads canonical tier
    names and never has to know which engine it is talking to.
    """
    tiers: dict[str, float] = {}
    uncached: float | None = None
    if family == FAMILY_SGLANG:
        for name, labels, value in samples:
            if _bare(name) != SGLANG_CACHE_SERIES:
                continue
            mode = labels.get("mode", "")
            if mode == SGLANG_UNCACHED_MODE:
                uncached = value if uncached is None else uncached + value
                continue
            tier = SGLANG_CACHE_TIERS.get(mode)
            if tier is not None:
                tiers[tier] = tiers.get(tier, 0.0) + value
        return tiers, uncached

    if family == FAMILY_VLLM:
        cached: float | None = None
        prompt_tokens = series_total(
            samples, family, COUNTER_SERIES["prompt_tokens"][family]
        )
        for series, tier in VLLM_CACHE_TIERS.items():
            value = series_total(samples, family, (series,))
            if value is None:
                continue
            tiers[tier] = tiers.get(tier, 0.0) + value
            cached = value if cached is None else cached + value
        if prompt_tokens is not None and cached is not None:
            remainder = prompt_tokens - cached
            # A negative remainder means the two counters do not describe the
            # same population, which the scrape cannot settle; refusing the
            # figure is honest where reporting a negative token count is not.
            uncached = remainder if remainder >= 0 else None
        return tiers, uncached

    return tiers, uncached


@dataclass(frozen=True, slots=True)
class Histogram:
    """One Prometheus histogram: cumulative buckets, count and sum."""

    buckets: dict[float, float]
    count: float
    total: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "buckets": {
                str(bound): value for bound, value in sorted(self.buckets.items())
            },
            "count": self.count,
            "sum": self.total,
        }


@dataclass(frozen=True, slots=True)
class EngineMetrics:
    """One engine's canonical serving quantities, with absence by omission."""

    family: str | None
    model_id: str | None
    gauges: dict[str, float] = field(default_factory=dict)
    counters: dict[str, float] = field(default_factory=dict)
    cached_prompt_tokens: dict[str, float] = field(default_factory=dict)
    uncached_prompt_tokens: float | None = None
    histograms: dict[str, Histogram] = field(default_factory=dict)
    spec_decode: dict[str, Any] = field(default_factory=dict)

    def row_section(self) -> dict[str, Any]:
        """The sparse engine section of a receipt row.

        Every key here was measured. A quantity the family does not publish,
        or that this scrape did not carry, is absent -- never ``null`` and
        never ``0.0`` -- so a reader cannot mistake a non-observation for a
        reading. That is the property the recorded nulls violated.
        """
        section: dict[str, Any] = {}
        if self.family is not None:
            section["family"] = self.family
        if self.model_id is not None:
            section["model_id"] = self.model_id
        for table, values in (
            (GAUGE_SERIES, self.gauges),
            (COUNTER_SERIES, self.counters),
        ):
            for name in table:
                if name in values:
                    section[name] = values[name]
        if self.cached_prompt_tokens:
            section["cached_prompt_tokens"] = {
                tier: self.cached_prompt_tokens[tier]
                for tier in sorted(self.cached_prompt_tokens)
            }
        if self.uncached_prompt_tokens is not None:
            section["uncached_prompt_tokens"] = self.uncached_prompt_tokens
        if self.histograms:
            section["histograms"] = {
                name: histogram.as_dict() for name, histogram in self.histograms.items()
            }
        observed_spec = {
            key: value for key, value in self.spec_decode.items() if value is not None
        }
        if observed_spec:
            section["spec_decode"] = observed_spec
        return section


def _histogram(
    samples: list[MetricSample], family: str, names: tuple[str, ...]
) -> Histogram | None:
    """Read one histogram family, or ``None`` when the family publishes none."""
    buckets: dict[float, float] = {}
    count: float | None = None
    total: float | None = None
    for name, labels, value in samples:
        if not name.startswith(f"{family}:"):
            continue
        bare = _bare(name)
        if bare == f"{names[0]}_bucket":
            bound = labels.get("le")
            if bound is None:
                continue
            if bound == "+Inf":
                buckets[float("inf")] = value
            else:
                buckets[float(bound)] = value
        elif bare == f"{names[0]}_count":
            count = value
        elif bare == f"{names[0]}_sum":
            total = value
    if not buckets and count is None and total is None:
        return None
    return Histogram(
        buckets=buckets,
        count=count if count is not None else 0.0,
        total=total if total is not None else 0.0,
    )


def read_metrics(text: str) -> EngineMetrics:
    """Resolve a ``/metrics`` body into canonical quantities.

    A body from neither family yields a reading with ``family=None`` and no
    quantities, which is a finding about the document rather than an error:
    the caller decides whether that is fatal.
    """
    samples = parse_metrics(text)
    family = detect_family(samples)
    if family is None:
        return EngineMetrics(family=None, model_id=None)

    eligible = eligible_samples(samples, family)

    gauges: dict[str, float] = {}
    for name, table in GAUGE_SERIES.items():
        value = series_first(eligible, family, table[family])
        if value is None:
            value = series_total(eligible, family, table[family])
        if value is not None:
            gauges[name] = value

    counters: dict[str, float] = {}
    for name, table in COUNTER_SERIES.items():
        value = series_total(eligible, family, table[family])
        if value is not None:
            counters[name] = value

    cached, uncached = tier_split(eligible, family)

    histograms: dict[str, Histogram] = {}
    for name, table in HISTOGRAM_SERIES.items():
        names = table[family]
        if not names:
            continue
        histogram = _histogram(eligible, family, names)
        if histogram is not None:
            histograms[name] = histogram

    draft = series_total(
        eligible, family, SPEC_DECODE_SERIES["draft_tokens_total"][family]
    )
    accepted = series_total(
        eligible, family, SPEC_DECODE_SERIES["accepted_tokens_total"][family]
    )
    per_position: list[float] | None = None
    per_position_names = SPEC_DECODE_PER_POSITION.get(family, ())
    if per_position_names:
        positions: dict[int, float] = {}
        for name, labels, value in eligible:
            if _bare(name) not in per_position_names:
                continue
            position = labels.get("position")
            if position is None:
                continue
            try:
                positions[int(position)] = positions.get(int(position), 0.0) + value
            except ValueError:
                continue
        if positions:
            per_position = [positions[pos] for pos in sorted(positions)]

    spec_gauges: dict[str, float] = {}
    for name, table in SPEC_GAUGE_SERIES.items():
        value = series_first(eligible, family, table[family])
        if value is not None:
            spec_gauges[name] = value

    spec_decode: dict[str, Any] = dict(spec_gauges)
    if draft is not None:
        spec_decode["draft_tokens_total"] = draft
    if accepted is not None:
        spec_decode["accepted_tokens_total"] = accepted
    if per_position is not None:
        spec_decode["num_accepted_per_pos"] = per_position

    return EngineMetrics(
        family=family,
        model_id=model_id_of(eligible, family),
        gauges=gauges,
        counters=counters,
        cached_prompt_tokens=cached,
        uncached_prompt_tokens=uncached,
        histograms=histograms,
        spec_decode=spec_decode,
    )


def lane_series(role: str, family: str) -> tuple[str, ...]:
    """Bare series names carrying one lane role for *family*.

    The lane reader resolves its names through here so that no family spelling
    lives outside this module; an empty tuple is a quantity the family does not
    publish, which the caller reports as an absence rather than a zero.
    """
    if role in LANE_SERIES:
        return LANE_SERIES[role][family]
    if role in COUNTER_SERIES:
        return COUNTER_SERIES[role][family]
    if role in GAUGE_SERIES:
        return GAUGE_SERIES[role][family]
    raise KeyError(role)


def role_is_counter(role: str) -> bool:
    """Whether one lane role's series is cumulative, so its label sets sum.

    True for the canonical counters in ``COUNTER_SERIES`` and for the lane-only
    cumulative roles in ``LANE_COUNTER_ROLES``; a role absent from both is a
    gauge, to be read as one value.
    """
    return role in COUNTER_SERIES or role in LANE_COUNTER_ROLES
