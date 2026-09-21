"""One reader resolves every engine family's serving quantities the same way.

Coverage
--------
1.  A recorded SGLang scrape yields the canonical quantities under their
    family-agnostic names, with the same reader the vLLM path uses.
2.  A recorded-shape vLLM scrape yields the same canonical field set.
3.  A quantity a family does not publish is ABSENT from the reading rather than
    ``null`` or ``0.0`` -- the property the recorded rows violated, where
    fourteen of twenty fields were null on a healthy serve.
4.  A quantity genuinely published as zero is kept, so omission means
    "not observed" and never "zero".
5.  SGLang's per-rank repeats are one shared pool, and vLLM's ``reason``
    sub-series are not folded into their family's total.
6.  The continuous recorder's row carries the canonical section unchanged,
    which is what makes a recorded row comparable across families.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

from imas_ambix.agent import engine_metrics, serving_receipts

_SGLANG_FIXTURE = Path(__file__).parent / "data" / "sglang_metrics_sample.txt"

# Keys ``row_section`` may carry besides the canonical quantities themselves.
_ANNOTATION_KEYS = frozenset({"family", "model_id"})
_CANONICAL_KEYS = (
    frozenset(engine_metrics.GAUGE_SERIES)
    | frozenset(engine_metrics.COUNTER_SERIES)
    | frozenset(
        {
            "cached_prompt_tokens",
            "uncached_prompt_tokens",
            "histograms",
            "spec_decode",
        }
    )
)


def _per_pos_head(labels: str) -> str:
    """The series and label prefix a per-draft-position sample carries."""
    return "vllm:spec_decode_num_accepted_tokens_per_pos{" + labels + ",position="


def _vllm_metrics(
    *,
    prompt_tokens: float = 10_000.0,
    device_hits: float = 2_500.0,
    external_hits: float | None = 1_500.0,
    with_spec_decode: bool = True,
) -> str:
    """A Prometheus scrape shaped like a vLLM deployment's ``/metrics``.

    Deliberately carries the shapes a naive reader gets wrong: a ``_created``
    companion gauge beside each counter, a ``reason``-labelled partial series
    beside its family total, and a second engine rank whose counters must sum.
    """
    labels = 'engine="0",model_name="vllm-model"'
    lines = [
        f"vllm:num_requests_running{{{labels}}} 3.0",
        f"vllm:num_requests_waiting{{{labels}}} 2.0",
        # A breakdown of the total above, not a second queue.
        f'vllm:num_requests_waiting{{{labels},reason="capacity"}} 4.0',
        f"vllm:kv_cache_usage_perc{{{labels}}} 0.4",
        f"vllm:prompt_tokens_total{{{labels}}} {prompt_tokens}",
        f"vllm:generation_tokens_total{{{labels}}} 64.0",
        # The ``_created`` sibling is a creation timestamp, not a data point.
        f"vllm:prompt_tokens_created{{{labels}}} 1.7e9",
        f"vllm:prefix_cache_queries_total{{{labels}}} 40.0",
        f"vllm:prefix_cache_hits_total{{{labels}}} {device_hits}",
        'vllm:prefix_cache_queries_total{engine="1",model_name="vllm-model"} 10.0',
        'vllm:prefix_cache_hits_total{engine="1",model_name="vllm-model"} 500.0',
        'vllm:time_to_first_token_seconds_bucket{le="0.5"} 1.0',
        'vllm:time_to_first_token_seconds_bucket{le="+Inf"} 4.0',
        "vllm:time_to_first_token_seconds_count 4.0",
        "vllm:time_to_first_token_seconds_sum 2.5",
        'vllm:inter_token_latency_seconds_bucket{le="0.05"} 9.0',
        'vllm:inter_token_latency_seconds_bucket{le="+Inf"} 12.0',
        "vllm:inter_token_latency_seconds_count 12.0",
        "vllm:inter_token_latency_seconds_sum 0.75",
    ]
    if external_hits is not None:
        lines.append(
            f"vllm:external_prefix_cache_hits_total{{{labels}}} {external_hits}"
        )
    if with_spec_decode:
        lines += [
            f"vllm:spec_decode_num_draft_tokens_total{{{labels}}} 500.0",
            f"vllm:spec_decode_num_accepted_tokens_total{{{labels}}} 400.0",
            f"vllm:spec_decode_num_draft_tokens_created{{{labels}}} 1.7e9",
            _per_pos_head(labels) + '"0"} 180.0',
            _per_pos_head(labels) + '"1"} 130.0',
        ]
    return "\n".join(lines) + "\n"


def _section(text: str) -> dict[str, object]:
    return engine_metrics.read_metrics(text).row_section()


def _sglang_section() -> dict[str, object]:
    return _section(_SGLANG_FIXTURE.read_text(encoding="utf-8"))


def _published(family: str) -> set[str]:
    """Canonical keys *family* publishes, read from its own series tables."""
    keys = {"histograms"}
    for table in (engine_metrics.GAUGE_SERIES, engine_metrics.COUNTER_SERIES):
        keys |= {name for name, series in table.items() if series[family]}
    return keys


# ---------------------------------------------------------------------------
# One code path, both families
# ---------------------------------------------------------------------------


def test_sglang_scrape_resolves_through_the_shared_reader() -> None:
    """The recorded SGLang scrape yields canonical names, not a vLLM namespace."""
    metrics = engine_metrics.read_metrics(_SGLANG_FIXTURE.read_text(encoding="utf-8"))

    assert metrics.family == engine_metrics.FAMILY_SGLANG
    assert metrics.model_id == "deepseek-v4.1-flash"
    # Only rank zero counts, so this is the one shared pool rather than 4x it.
    assert metrics.gauges["kv_pool_occupancy"] == 0.0
    assert metrics.counters["prompt_tokens"] == 223_435.0
    assert metrics.counters["generation_tokens"] == 104.0
    assert metrics.uncached_prompt_tokens == 223_744.0
    assert set(metrics.histograms) == {
        "time_to_first_token",
        "inter_token_latency",
    }
    assert metrics.histograms["time_to_first_token"].count == 3.0


def test_vllm_scrape_resolves_through_the_same_reader() -> None:
    """The vLLM fixture carries the same canonical quantities, same reader."""
    metrics = engine_metrics.read_metrics(_vllm_metrics())

    assert metrics.family == engine_metrics.FAMILY_VLLM
    assert metrics.model_id == "vllm-model"
    assert metrics.gauges["requests_running"] == 3.0
    assert metrics.gauges["requests_queued"] == 2.0
    assert metrics.gauges["kv_pool_occupancy"] == 0.4
    assert metrics.counters["prompt_tokens"] == 10_000.0
    assert metrics.counters["generation_tokens"] == 64.0
    assert set(metrics.histograms) == {
        "time_to_first_token",
        "inter_token_latency",
    }


def test_both_families_report_every_quantity_they_publish() -> None:
    """Presence follows the family tables, not a branch in the reader.

    Both fixtures carry every series their family publishes, so each family's section
    must contain exactly the canonical keys those tables name plus the two
    annotations -- a family-specific omission would show up as a missing key here.
    """
    delivered = {
        engine_metrics.FAMILY_SGLANG: set(_sglang_section()),
        engine_metrics.FAMILY_VLLM: set(_section(_vllm_metrics())),
    }

    for family in engine_metrics.FAMILIES:
        assert _published(family) <= delivered[family], family
        assert delivered[family] <= _ANNOTATION_KEYS | _CANONICAL_KEYS, family

    # The vLLM fixture carries the full canonical set, including the terms that
    # only exist where a tier answered the prefill.
    expected_vllm = (
        _ANNOTATION_KEYS
        | set(engine_metrics.GAUGE_SERIES)
        | set(engine_metrics.COUNTER_SERIES)
        | {
            "cached_prompt_tokens",
            "uncached_prompt_tokens",
            "histograms",
            "spec_decode",
        }
    )
    assert delivered[engine_metrics.FAMILY_VLLM] == expected_vllm


# ---------------------------------------------------------------------------
# Absence is omission, and a published zero is a reading
# ---------------------------------------------------------------------------


def test_unpublished_quantities_are_absent_rather_than_null_or_zero() -> None:
    """SGLang publishes no prefix-cache counter and no draft counter.

    The recorded rows carried a null for every one of these, which a reader
    cannot distinguish from a measured zero -- the whole defect being repaired.
    """
    section = _sglang_section()

    for absent in ("prefix_cache_queries", "prefix_cache_hits"):
        assert absent not in section, absent
    # SGLang publishes no speculative-decode counters at all, so the whole
    # subsection is gone rather than a triple of nulls.
    assert "spec_decode" not in section
    assert all(value is not None for value in section.values())


def test_a_quantity_published_as_zero_is_kept() -> None:
    """Omission means "not observed"; a measured zero is a reading.

    The SGLang fixture's pool is idle, so its gauges are genuinely zero and the
    host cache tier genuinely answered nothing. Dropping those would make the
    absence rule indistinguishable from a missing measurement.
    """
    section = _sglang_section()

    assert section["requests_running"] == 0.0
    assert section["requests_queued"] == 0.0
    assert section["cached_prompt_tokens"] == {
        "device": 0.0,
        "host": 0.0,
        "storage": 0.0,
    }


def test_a_cache_tier_with_no_series_is_absent_from_the_split() -> None:
    """Without the offload connector vLLM publishes no external tier.

    The split is by tier, so a missing tier is a missing key -- not an
    ``external: 0`` that reads as a measured miss.
    """
    metrics = engine_metrics.read_metrics(_vllm_metrics(external_hits=None))

    # 2,500 on engine 0 and 500 on engine 1: one counter, two label sets.
    assert metrics.cached_prompt_tokens == {"device": 3_000.0}
    assert metrics.uncached_prompt_tokens == 7_000.0
    section = metrics.row_section()
    assert "external" not in section["cached_prompt_tokens"]


def test_an_unpublished_speculative_counter_leaves_no_null_behind() -> None:
    """A vLLM scrape with no speculator carries no spec_decode subsection."""
    section = _section(_vllm_metrics(with_spec_decode=False))
    assert "spec_decode" not in section


def test_a_document_from_neither_family_reads_as_nothing() -> None:
    """A scrape with no engine prefix is a finding, not an invented reading."""
    text = 'process_cpu_seconds_total{component="r"} 1.0'
    metrics = engine_metrics.read_metrics(text)

    assert metrics.family is None
    assert metrics.row_section() == {}


# ---------------------------------------------------------------------------
# Selection discipline
# ---------------------------------------------------------------------------


def test_sglang_rank_zero_is_the_one_shared_pool() -> None:
    """Per-rank repeats are one reading, so a rank's value is not a term to sum."""
    text = "\n".join(
        (
            'sglang:num_running_reqs{tp_rank="0"} 7.0',
            'sglang:num_running_reqs{tp_rank="1"} 7.0',
            'sglang:num_running_reqs{tp_rank="2"} 7.0',
            'sglang:num_running_reqs{tp_rank="3"} 7.0',
        )
    )
    metrics = engine_metrics.read_metrics(text)

    assert metrics.gauges["requests_running"] == 7.0


def test_vllm_reason_series_are_not_folded_into_the_total() -> None:
    """A ``reason``-labelled breakdown double-counts its own family total."""
    metrics = engine_metrics.read_metrics(_vllm_metrics())

    # The fixture's waiting gauge is 2.0 with a reason="capacity" 4.0 beside it.
    assert metrics.gauges["requests_queued"] == 2.0


def test_created_companions_are_not_counted_as_data() -> None:
    """vLLM pairs every counter with a same-named ``_created`` gauge.

    A substring match folds that creation timestamp into the token total, which
    is how a counter comes to read as billions of extra tokens.
    """
    metrics = engine_metrics.read_metrics(_vllm_metrics())

    assert metrics.counters["prompt_tokens"] == 10_000.0
    assert metrics.spec_decode["draft_tokens_total"] == 500.0


# ---------------------------------------------------------------------------
# The recorder's rows
# ---------------------------------------------------------------------------


def _receipt_engine_section(text: str) -> dict[str, object]:
    """The engine section of one row, as the recorder would write it."""
    snapshot = serving_receipts._serving_snapshot(text)
    row = serving_receipts.build_receipt_row(
        None,
        None,
        snapshot,
        _dt.datetime.now(_dt.UTC),
        job_id=None,
        profile_slug=None,
        served_name=None,
        gpus=None,
    )
    return json.loads(row.to_json())["engine"]


def test_a_recorded_row_carries_the_canonical_section_per_family() -> None:
    """Whatever the family, the row's canonical section is the same vocabulary.

    This is the reader the receipts file is written through, so the assertion is
    about the artifact a later analysis actually reads.
    """
    for family, text in (
        (engine_metrics.FAMILY_SGLANG, _SGLANG_FIXTURE.read_text(encoding="utf-8")),
        (engine_metrics.FAMILY_VLLM, _vllm_metrics()),
    ):
        section = _receipt_engine_section(text)
        assert section["family"] == family
        gauges = {"requests_running", "requests_queued", "kv_pool_occupancy"}
        assert gauges <= set(section)
        assert {"prompt_tokens", "generation_tokens"} <= set(section)
        latencies = {"time_to_first_token", "inter_token_latency"}
        assert latencies <= set(section["histograms"])
        assert section["cached_prompt_tokens"]["device"] is not None
        assert all(value is not None for value in section.values())


def test_the_row_omits_the_quantity_sglang_does_not_publish() -> None:
    """The legacy flat fields stay nullable; the canonical section is the record.

    A reader asking whether the engine published a query counter gets no key
    from an SGLang row, where the flat vLLM-shaped fields beside it are null
    because that vocabulary has no meaning here.
    """
    section = _receipt_engine_section(_SGLANG_FIXTURE.read_text(encoding="utf-8"))

    assert "prefix_cache_queries" not in section
    assert "prefix_cache_hits" not in section
