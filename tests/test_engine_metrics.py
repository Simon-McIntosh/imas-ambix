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

# The keys a reading of each family carries, spelled out: the canonical
# quantities that family publishes plus the two annotations. Written out rather
# than read from ``engine_metrics``' own tables for the same reason as
# ``_CANONICAL_FIELD_SOURCES`` below -- an expectation derived from those tables
# moves with them, so emptying a family's series mapping takes the delivered key
# away in the same step that takes the expected one, and the omission leaves no
# failure behind. An SGLang reading carries neither the prefix-cache counters nor
# the speculative-decode terms, since that family publishes no series for them.
_SECTION_KEYS: dict[str, frozenset[str]] = {
    engine_metrics.FAMILY_SGLANG: _ANNOTATION_KEYS
    | frozenset(
        {
            "requests_running",
            "requests_queued",
            "kv_pool_occupancy",
            "prompt_tokens",
            "generation_tokens",
            "cached_prompt_tokens",
            "uncached_prompt_tokens",
            "histograms",
        }
    ),
    engine_metrics.FAMILY_VLLM: _ANNOTATION_KEYS
    | frozenset(
        {
            "requests_running",
            "requests_queued",
            "kv_pool_occupancy",
            "prompt_tokens",
            "generation_tokens",
            "prefix_cache_queries",
            "prefix_cache_hits",
            "cached_prompt_tokens",
            "uncached_prompt_tokens",
            "histograms",
            "spec_decode",
        }
    ),
}


# The canonical quantities an SGLang reading must expose, spelled out here with
# their value path in a reading's canonical section and the vetted scrape series
# each is sourced from. Written out rather than read from ``engine_metrics``' own
# tables because an expectation derived from those tables shrinks with them:
# dropping a family's series mapping empties both the expectation and the
# delivered key together, so a canonical quantity can stop resolving without a
# test noticing. This list is what makes the recorded scrape re-assert the served
# field set offline.
#
# One entry is (quantity, value path, series name, required label fragment).
_CANONICAL_FIELD_SOURCES: tuple[tuple[str, tuple[str, ...], str, str], ...] = (
    ("requests running", ("requests_running",), "num_running_reqs", ""),
    ("requests queued", ("requests_queued",), "num_queue_reqs", ""),
    ("KV-pool occupancy", ("kv_pool_occupancy",), "full_token_usage", ""),
    ("prompt tokens", ("prompt_tokens",), "prompt_tokens_total", ""),
    ("generation tokens", ("generation_tokens",), "generation_tokens_total", ""),
    (
        "cached tokens, split by the tier that answered",
        ("cached_prompt_tokens", "device"),
        "prefill_effective_tokens_total",
        'mode="device_hit"',
    ),
    (
        "cached tokens, split by the tier that answered",
        ("cached_prompt_tokens", "host"),
        "prefill_effective_tokens_total",
        'mode="host_hit"',
    ),
    (
        "cached tokens, split by the tier that answered",
        ("cached_prompt_tokens", "storage"),
        "prefill_effective_tokens_total",
        'mode="storage_hit"',
    ),
    (
        "uncached prompt tokens",
        ("uncached_prompt_tokens",),
        "prefill_effective_tokens_total",
        'mode="input"',
    ),
    (
        "time to first token",
        ("histograms", "time_to_first_token"),
        "time_to_first_token_seconds_count",
        "",
    ),
    (
        "inter-token latency",
        ("histograms", "inter_token_latency"),
        "inter_token_latency_seconds_count",
        "",
    ),
)


def _resolve(section: dict[str, object], path: tuple[str, ...]) -> object:
    """The value *path* addresses inside a canonical section, or ``None``."""
    node: object = section
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _has_sample_line(text: str, name: str, label: str) -> bool:
    """Whether the reader sees a *sample* of *name*, not merely its HELP text.

    Every step is the reader's own, so the answer is the reader's answer for the
    same input rather than a second opinion that can disagree with it in either
    direction. The exposition repeats each series name in a ``# HELP`` and a
    ``# TYPE`` line above its samples, so a substring search for the name is
    satisfied by a series whose samples are all gone: what counts as a sample is
    therefore taken from ``parse_metrics``, which strips each line, drops the
    comments and matches the name-then-labels-then-value form. Which samples a
    reading is made of is taken from ``eligible_samples``, so a sample the
    reader discards -- a repeated tensor-parallel rank, or a ``reason``-labelled
    breakdown -- is not counted here either. The name is compared the way the
    reader compares it, on ``_bare``, so a name carrying more than one colon --
    which resolves in production, since only the last segment is read -- is not
    reported unsourced. A hand-rolled peel beside either of those is stricter
    where the exposition is looser than the reader is, and looser where the
    reader is selective.
    """
    samples = engine_metrics.parse_metrics(text)
    family = engine_metrics.detect_family(samples)
    if family is None:
        return False
    wanted = _label_fragment(label)
    for sample_name, labels, _value in engine_metrics.eligible_samples(samples, family):
        if engine_metrics._bare(sample_name) != name:
            continue
        if wanted is not None and labels.get(wanted[0]) != wanted[1]:
            continue
        return True
    return False


def _label_fragment(label: str) -> tuple[str, str] | None:
    """The key and value a required label fragment names, or ``None`` if empty.

    Read with the reader's own label syntax, so a fragment is parsed the same
    way the line it is looked for in is.
    """
    if not label:
        return None
    match = engine_metrics._LABEL_RE.search(label)
    if match is None:
        raise ValueError(f"not a label fragment: {label!r}")
    return match.group("key"), match.group("val")


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


def test_the_recorded_scrape_sources_every_canonical_field() -> None:
    """Every canonical field a reading must expose is sourced by the scrape's bytes.

    Two things are asserted against the file rather than against the module.
    Every canonical quantity resolves from the recorded scrape, with the
    expectation written out here so a series mapping that is dropped or renamed
    fails rather than silently shrinking the expectation with it. And each
    quantity's source series is read out of the file's own text, so a fixture
    that was hand-written, truncated or left behind by an older engine revision
    cannot satisfy this test by carrying canned values.
    """
    assert _SGLANG_FIXTURE.is_file(), _SGLANG_FIXTURE
    text = _SGLANG_FIXTURE.read_text(encoding="utf-8")
    section = _section(text)

    unresolved: list[str] = []
    unsourced: list[str] = []
    for quantity, path, name, label in _CANONICAL_FIELD_SOURCES:
        if _resolve(section, path) is None:
            unresolved.append(f"{quantity} ({'.'.join(path)})")
        if not _has_sample_line(text, name, label):
            unsourced.append(f"{quantity} ({'.'.join(path)}) <- {name}{label}")

    assert not unresolved and not unsourced, {
        "unresolved": unresolved,
        "unsourced": unsourced,
    }


def test_a_series_name_without_a_value_is_not_a_source() -> None:
    """A name with nothing to parse is not a source, however the line is spelled.

    The exposition repeats every series name in a ``# HELP`` and a ``# TYPE``
    line above its samples, so looking for the name alone is satisfied by the
    header. Asking for a *sample* line is not enough either: a line carrying the
    name and its full label set but no value has nothing to read, and a trailing
    space is the same defect. Only a line that ends in a value token counts, so a
    scrape that lost its samples cannot pass on its labels. Both spellings of a
    sample are covered, label-bearing and unlabelled: the value is required of
    each independently, and an unlabelled line has nothing but the value to hold
    it to.

    The guard is also held to what the reader ACCEPTS, not only to what it
    refuses: the shapes below differ only in leading whitespace or in whether a
    tab separates the value, and a guard requiring the name at column zero
    REFUSED those while the reader resolved every value in them. What counts as
    a sample is therefore asked of the reader itself, and the indented
    assertions below fail if that delegation is ever replaced by a hand-rolled
    peel again.
    """
    name = "num_running_reqs"
    labels = 'engine_type="unified",tp_rank="0"'
    header = "\n".join(
        [
            f"# HELP sglang:{name} The number of running requests.",
            f"# TYPE sglang:{name} gauge",
        ]
    )
    not_a_source = {
        "the name only in HELP/TYPE text": header,
        "the name and its labels, with no value": f"sglang:{name}{{{labels}}}",
        "the name and its labels, then a trailing space": f"sglang:{name}{{{labels}}} ",
        "the bare name, unlabelled, with no value": f"sglang:{name}",
        "the bare name, unlabelled, then a trailing space": f"sglang:{name} ",
        "a longer series name sharing the prefix": (
            f"sglang:{name}_total{{{labels}}} 1.0"
        ),
    }
    for description, text in not_a_source.items():
        assert not _has_sample_line(text, name, ""), description

    value_present = {
        "the name and its labels": f"sglang:{name}{{{labels}}} 0.0",
        "the bare name, unlabelled": f"sglang:{name} 0.0",
        # Leading whitespace and a tab separator are legal exposition the reader
        # accepts, and a guard refusing them would report a series unsourced
        # that the reader resolves.
        "the label-bearing line, indented": f"    sglang:{name}{{{labels}}} 0.0",
        "the unlabelled line, indented": f"  sglang:{name} 0.0",
        "tab-separated after the labels": f"sglang:{name}{{{labels}}}\t0.0",
    }
    for description, text in value_present.items():
        assert _has_sample_line(text, name, ""), description

    # The whole fixture, indented: the shape a hand-rolled peel refused while
    # the reader resolved every value in it.
    indented_fixture = "\n".join(
        f"    {line}"
        for line in _SGLANG_FIXTURE.read_text(encoding="utf-8").splitlines()
    )
    for _quantity, _path, source, fragment in _CANONICAL_FIELD_SOURCES:
        assert _has_sample_line(indented_fixture, source, fragment), source

    assert not _has_sample_line(
        f"sglang:{name}{{{labels}}} 0.0", name, 'mode="input"'
    ), "the required label fragment must be read from the label set"
    assert not _has_sample_line(f"sglang:{name} 0.0", name, 'mode="input"'), (
        "an unlabelled line cannot carry the required label fragment"
    )


def test_the_sourcing_guard_answers_as_the_reader_does() -> None:
    """The guard's verdict and the reading are asserted to agree per input.

    The guard exists to say whether the scrape's own bytes source a field, and
    the reading says whether the field resolved: those are one question seen
    from two sides, so each input below is put to both and the two answers are
    required to match. A guard stricter than the reader reports a field
    unsourced that the reader resolves; a guard looser than the reader counts a
    sample the reader discards. Both directions are exercised. A name carrying
    more than one colon resolves in production, because the reader reads only
    the last segment. A sample on a tensor-parallel rank other than rank zero is
    discarded there -- SGLang repeats each device-pool measurement on every
    rank, so the rank-zero sample is the one shared pool -- as is a
    ``reason``-labelled breakdown, which is a part of a total rather than the
    total. The rank-zero and total rows are the positive controls, so an input
    that agrees only by returning ``False`` twice cannot satisfy the table.
    """
    cases = (
        (
            "a name carrying more than one colon",
            "sglang:x:num_running_reqs 1.0",
            ("requests_running",),
            "num_running_reqs",
            "",
            True,
        ),
        (
            "a sample on a rank other than rank zero",
            'sglang:num_running_reqs{tp_rank="1"} 1.0',
            ("requests_running",),
            "num_running_reqs",
            "",
            False,
        ),
        (
            "the same series on rank zero",
            'sglang:num_running_reqs{tp_rank="0"} 1.0',
            ("requests_running",),
            "num_running_reqs",
            "",
            True,
        ),
        (
            "an uncached-token sample on a rank other than rank zero",
            'sglang:prefill_effective_tokens_total{tp_rank="1",mode="input"} 5.0',
            ("uncached_prompt_tokens",),
            "prefill_effective_tokens_total",
            'mode="input"',
            False,
        ),
        (
            "a reason-labelled breakdown, which is part of a total",
            'vllm:num_requests_waiting{reason="capacity"} 4.0',
            ("requests_queued",),
            "num_requests_waiting",
            "",
            False,
        ),
        (
            "the total that breakdown belongs to",
            "vllm:num_requests_waiting 4.0",
            ("requests_queued",),
            "num_requests_waiting",
            "",
            True,
        ),
    )
    for description, text, path, name, fragment, resolves in cases:
        reader_resolves = _resolve(_section(text), path) is not None
        guard_sources = _has_sample_line(text, name, fragment)
        assert reader_resolves == resolves, description
        assert guard_sources == resolves, f"{description}: guard and reading disagree"


def test_the_recorded_scrape_reads_as_measurements_not_a_column_of_zeros() -> None:
    """The scrape is a working serve, so its readings are readings.

    An all-zero column would resolve every canonical field and still carry no
    information, so the instrument is shown to see something known present: the
    counter and histogram observations are large, and the gauges are a measured
    zero standing beside them rather than the only value the scrape publishes.
    """
    section = _sglang_section()

    for counter in ("prompt_tokens", "generation_tokens", "uncached_prompt_tokens"):
        assert section[counter] > 0, counter
    for kind in ("time_to_first_token", "inter_token_latency"):
        histogram = section["histograms"][kind]
        assert histogram["count"] > 0, kind
        assert histogram["buckets"], kind

    assert section["requests_running"] == 0.0


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

    Both fixtures carry every series their family publishes, so each family's
    section must carry exactly the keys written down in ``_SECTION_KEYS`` -- the
    canonical quantities that family publishes plus the two annotations. The
    expectation is a literal, so a family whose series mapping loses a quantity
    fails here rather than moving its expected key in step with the delivered one.
    """
    delivered = {
        engine_metrics.FAMILY_SGLANG: set(_sglang_section()),
        engine_metrics.FAMILY_VLLM: set(_section(_vllm_metrics())),
    }

    # The literal must cover every family, so a new family cannot arrive with no
    # written-down expectation and pass by default.
    assert set(delivered) == set(engine_metrics.FAMILIES)
    for family, expected in _SECTION_KEYS.items():
        assert delivered[family] == expected, family


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
