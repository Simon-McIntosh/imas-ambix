"""Tests for imas_ambix.agent.bench_report: saved-run summary and diff.

Coverage
--------
-   load_report round-trips a saved document and stamps its source path.
-   load_report names the file on unparseable JSON, a non-object root,
    a missing 'results' list, and a non-object result record.
-   discover_reports lists newest-first, filters by model substring,
    honours a limit, and returns [] for a missing directory.
-   describe_run flattens a run that carries serving provenance.
-   describe_run on a run predating provenance yields explicit unknowns
    instead of raising or inventing values.
-   describe_run derives the acceptance rate from draft-token totals and
    guards the zero-draft case.
-   describe_run leaves the loaded report unmodified.
-   compare_runs delta and percentage arithmetic on known numbers.
-   compare_runs marks a zero or absent baseline as not comparable.
-   Zero completion tokens and a zero wall time produce no rate claim
    and no division by zero.
-   Disjoint category sets are compared on the intersection, with the
    exclusions reported.
-   Concurrency levels order numerically, so 32 follows 8.
-   render_run and render_comparison execute against a string console.
"""

from __future__ import annotations

import io
import json
from typing import TYPE_CHECKING, Any

import pytest
from rich.console import Console

from imas_ambix.agent.bench_report import (
    NOT_COMPARABLE,
    PROVENANCE_FIELDS,
    UNKNOWN,
    BenchReportError,
    compare_runs,
    describe_run,
    discover_reports,
    load_report,
    render_comparison,
    render_run,
)

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Synthetic report builders
# ---------------------------------------------------------------------------


def _record(
    category: str,
    test_name: str,
    *,
    status: str = "passed",
    prompt_tokens: int = 12,
    completion_tokens: int = 128,
    ttft_s: float = 0.05,
    total_time_s: float = 1.0,
    decode_tps: float = 100.0,
    prefill_tps: float = 250.0,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one per-request record in the saved shape."""
    return {
        "category": category,
        "test_name": test_name,
        "status": status,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": 0,
        "time_to_first_token_s": ttft_s,
        "total_time_s": total_time_s,
        "decode_tps": decode_tps,
        "prefill_tps": prefill_tps,
        "model": "test-model",
        "error": None,
        "finish_reason": "length",
        "http_status": 200,
        "repeat_index": 0,
        "metadata": metadata or {},
    }


def _bare_report(
    results: list[dict[str, Any]],
    *,
    served: str = "test-model",
    timestamp: str = "2026-08-26T09:00:00+00:00",
) -> dict[str, Any]:
    """A saved run in the shape that predates serving provenance."""
    categories = sorted({str(r["category"]) for r in results})
    return {
        "timestamp": timestamp,
        "server_info": {
            "object": "list",
            "data": [{"id": served, "object": "model", "max_model_len": 262144}],
        },
        "categories_run": categories,
        "results": results,
        "summary": {},
    }


def _report_with_provenance(
    results: list[dict[str, Any]],
    *,
    served: str = "test-model",
    profile_slug: str = "test-profile",
    tensor_parallel: int = 8,
    gpus: int = 8,
    acceptance: dict[str, Any] | None = None,
    timestamp: str = "2026-08-26T10:00:00+00:00",
) -> dict[str, Any]:
    """A saved run carrying the provenance and draft-token blocks."""
    report = _bare_report(results, served=served, timestamp=timestamp)
    report["server_info"]["provenance"] = {
        "profile_slug": profile_slug,
        "model_name": "Test-Model",
        "served_name": served,
        "engine_type": "vllm",
        "engine_version": "0.23.0",
        "tensor_parallel": tensor_parallel,
        "gpus": gpus,
        "kv_cache_dtype": "auto",
        "max_model_len": 229376,
        "max_num_seqs": 1024,
        "speculative_method": "mtp",
        "speculative_num_tokens": 5,
        "quantization": "fp8",
        "slurm_job_id": "1222821",
        "gpu_host": "test-gpu-node",
        "captured_at": timestamp,
    }
    report["server_info"]["spec_decode"] = acceptance or {
        "draft_tokens_total": 1000,
        "accepted_tokens_total": 720,
        "acceptance_rate": 0.72,
        "num_accepted_per_pos": [400, 200, 80, 30, 10],
        "source": "metrics",
    }
    return report


def _throughput_results(decode_tps: float, ttft_s: float) -> list[dict[str, Any]]:
    """Two throughput tests at a fixed decode rate and latency."""
    return [
        _record("throughput", "decode_128", decode_tps=decode_tps, ttft_s=ttft_s),
        _record(
            "throughput",
            "decode_512",
            decode_tps=decode_tps,
            ttft_s=ttft_s,
            completion_tokens=512,
        ),
    ]


def _concurrency_results(levels: dict[int, float]) -> list[dict[str, Any]]:
    """One record per worker for each ``{n_workers: aggregate_tps}`` level."""
    out: list[dict[str, Any]] = []
    for n_workers, aggregate in levels.items():
        for index in range(2):  # two of the workers is enough to group on
            out.append(
                _record(
                    "concurrency",
                    f"concurrent_{n_workers}",
                    decode_tps=aggregate / n_workers,
                    metadata={
                        "n_workers": n_workers,
                        "aggregate_tps": aggregate,
                        "wall_time": 4.0,
                        "worker_index": index,
                    },
                )
            )
    return out


def _string_console() -> tuple[Console, io.StringIO]:
    """A rich console writing to a buffer, for renderer smoke tests."""
    buffer = io.StringIO()
    return Console(file=buffer, width=200, force_terminal=False), buffer


# ---------------------------------------------------------------------------
# load_report / discover_reports
# ---------------------------------------------------------------------------


def test_load_report_round_trip_stamps_source(tmp_path: Path) -> None:
    """A saved document loads unchanged, with its path recorded."""
    report = _report_with_provenance(_throughput_results(100.0, 0.05))
    path = tmp_path / "test-model_20260826_100000.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    loaded = load_report(path)

    assert loaded["timestamp"] == report["timestamp"]
    assert loaded["source_path"] == str(path)
    assert len(loaded["results"]) == 2


@pytest.mark.parametrize(
    ("content", "fragment"),
    [
        ("{not json", "not valid JSON"),
        ("[1, 2, 3]", "expected a JSON object"),
        ('{"timestamp": "x"}', "missing the 'results' list"),
        ('{"results": [1]}', "results[0] is a int"),
    ],
)
def test_load_report_names_the_file_on_malformed_input(
    tmp_path: Path, content: str, fragment: str
) -> None:
    """Every malformed shape raises an error naming the file."""
    path = tmp_path / "broken_20260826_100000.json"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(BenchReportError) as excinfo:
        load_report(path)

    assert str(path) in str(excinfo.value)
    assert fragment in str(excinfo.value)


def test_load_report_missing_file_names_it(tmp_path: Path) -> None:
    """A path that does not exist is reported, not swallowed."""
    missing = tmp_path / "absent.json"
    with pytest.raises(BenchReportError, match="no such benchmark report"):
        load_report(missing)


def test_discover_reports_newest_first_filter_and_limit(tmp_path: Path) -> None:
    """Discovery orders by save stamp, filters by model, honours a limit."""
    names = [
        "alpha-model_20260826_090000.json",
        "alpha-model_20260826_120000.json",
        "beta-model_20260825_080000.json",
    ]
    for name in names:
        (tmp_path / name).write_text(
            json.dumps(_bare_report(_throughput_results(10.0, 0.1))),
            encoding="utf-8",
        )
    (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")

    found = discover_reports(tmp_path)
    assert [p.name for p in found] == [
        "alpha-model_20260826_120000.json",
        "alpha-model_20260826_090000.json",
        "beta-model_20260825_080000.json",
    ]

    assert [p.name for p in discover_reports(tmp_path, model="beta")] == [
        "beta-model_20260825_080000.json"
    ]
    assert len(discover_reports(tmp_path, limit=2)) == 2
    assert discover_reports(tmp_path / "nowhere") == []


# ---------------------------------------------------------------------------
# describe_run
# ---------------------------------------------------------------------------


def test_describe_run_with_provenance() -> None:
    """A provenance-carrying run reports its configuration and figures."""
    report = _report_with_provenance(
        _throughput_results(120.0, 0.04)
        + [
            _record("prefill", "prefill_1k", prefill_tps=900.0, ttft_s=1.1),
            _record("prefill", "prefill_16k", prefill_tps=4200.0, ttft_s=3.8),
        ]
        + _concurrency_results({8: 210.0, 32: 680.0})
    )

    run = describe_run(report)

    assert run["has_provenance"] is True
    assert run["provenance"]["engine_type"] == "vllm"
    assert run["provenance"]["tensor_parallel"] == 8
    assert run["served_model"] == "test-model"
    assert run["label"].startswith("test-profile @ ")
    # Concurrency is excluded from the single-stream figure.
    assert run["decode_tps"] == pytest.approx(120.0)
    assert run["ttft_median_s"] == pytest.approx(0.57)
    assert run["ttft_p95_s"] == pytest.approx(3.8)
    assert list(run["prefill_tps"]) == ["prefill_1k", "prefill_16k"]
    assert run["prefill_tps"]["prefill_16k"] == pytest.approx(4200.0)
    assert run["peak_aggregate_tps"] == pytest.approx(680.0)
    assert run["peak_aggregate_workers"] == 32
    assert run["acceptance_rate"] == pytest.approx(0.72)
    assert run["status_counts"] == {
        "passed": 8,
        "failed": 0,
        "skipped": 0,
        "error": 0,
        "total": 8,
    }


def test_describe_run_without_provenance_yields_unknowns() -> None:
    """A run predating provenance describes cleanly, marked unknown."""
    report = _bare_report(_throughput_results(100.0, 0.05))

    run = describe_run(report)

    assert run["has_provenance"] is False
    assert set(run["provenance"]) == set(PROVENANCE_FIELDS)
    assert all(value == UNKNOWN for value in run["provenance"].values())
    assert all(value == UNKNOWN for value in run["spec_decode"].values())
    assert run["acceptance_rate"] is None
    # Real figures still come through — only the configuration is absent.
    assert run["served_model"] == "test-model"
    assert run["decode_tps"] == pytest.approx(100.0)
    assert run["peak_aggregate_tps"] is None
    assert run["peak_aggregate_workers"] is None


def test_describe_run_derives_acceptance_from_totals() -> None:
    """An unrecorded acceptance rate is derived from the token totals."""
    report = _report_with_provenance(
        _throughput_results(100.0, 0.05),
        acceptance={"draft_tokens_total": 400, "accepted_tokens_total": 300},
    )
    assert describe_run(report)["acceptance_rate"] == pytest.approx(0.75)


def test_describe_run_zero_draft_tokens_is_not_a_rate() -> None:
    """Zero drafted tokens yields no rate rather than a division error."""
    report = _report_with_provenance(
        _throughput_results(100.0, 0.05),
        acceptance={"draft_tokens_total": 0, "accepted_tokens_total": 0},
    )
    assert describe_run(report)["acceptance_rate"] is None


def test_describe_run_zero_completion_tokens_claims_no_rate() -> None:
    """A run that decoded nothing reports no rate and no wall-time rate."""
    report = _bare_report(
        [
            _record(
                "throughput",
                "decode_128",
                status="error",
                completion_tokens=0,
                decode_tps=0.0,
                prefill_tps=0.0,
                ttft_s=0.0,
                total_time_s=0.0,
            ),
            _record(
                "concurrency",
                "concurrent_4",
                completion_tokens=0,
                decode_tps=0.0,
                metadata={
                    "n_workers": 4,
                    "aggregate_tps": 0.0,
                    "wall_time": 0.0,
                },
            ),
        ]
    )

    run = describe_run(report)

    assert run["decode_tps"] is None
    assert run["ttft_median_s"] is None
    assert run["ttft_p95_s"] is None
    assert run["peak_aggregate_tps"] is None
    assert run["concurrency"][0]["aggregate_tps"] is None
    assert run["status_counts"]["error"] == 1


def test_describe_run_does_not_mutate_the_report() -> None:
    """Describing a run leaves the loaded document untouched."""
    report = _report_with_provenance(_throughput_results(100.0, 0.05))
    before = json.dumps(report, sort_keys=True)

    describe_run(report)

    assert json.dumps(report, sort_keys=True) == before


def test_concurrency_levels_order_numerically() -> None:
    """Worker counts sort as numbers, so 32 follows 8, not precedes it."""
    report = _bare_report(
        _concurrency_results({32: 680.0, 4: 120.0, 16: 457.0, 8: 210.0})
    )

    run = describe_run(report)

    assert [level["workers"] for level in run["concurrency"]] == [4, 8, 16, 32]
    assert run["peak_aggregate_workers"] == 32


def test_concurrency_workers_fall_back_to_the_test_name() -> None:
    """A record without n_workers still lands at the right level."""
    report = _bare_report(
        [
            _record("concurrency", "concurrent_8", metadata={"aggregate_tps": 200.0}),
            _record("concurrency", "concurrent_32", metadata={"aggregate_tps": 600.0}),
        ]
    )
    run = describe_run(report)
    assert [level["workers"] for level in run["concurrency"]] == [8, 32]


# ---------------------------------------------------------------------------
# compare_runs
# ---------------------------------------------------------------------------


def test_compare_runs_delta_and_percentage_arithmetic() -> None:
    """Deltas and percentages match hand-computed values."""
    baseline = _report_with_provenance(
        _throughput_results(100.0, 0.100),
        profile_slug="four-card",
        tensor_parallel=4,
        gpus=4,
        timestamp="2026-08-26T09:00:00+00:00",
    )
    candidate = _report_with_provenance(
        _throughput_results(125.0, 0.080),
        profile_slug="eight-card",
        tensor_parallel=8,
        gpus=8,
        timestamp="2026-08-26T10:00:00+00:00",
    )

    comparison = compare_runs([baseline, candidate])
    rows = {row["metric"]: row for row in comparison["metric_rows"]}

    decode = rows["decode tok/s (median, single-stream)"]
    assert decode["values"] == [pytest.approx(100.0), pytest.approx(125.0)]
    assert decode["deltas"][1] == pytest.approx(25.0)
    assert decode["pct"][1] == pytest.approx(25.0)

    ttft = rows["ttft ms (median)"]
    assert ttft["values"] == [pytest.approx(100.0), pytest.approx(80.0)]
    assert ttft["deltas"][1] == pytest.approx(-20.0)
    assert ttft["pct"][1] == pytest.approx(-20.0)
    assert ttft["higher_is_better"] is False

    per_test = rows["throughput/decode_512 decode tok/s"]
    assert per_test["deltas"][1] == pytest.approx(25.0)

    # Differing configuration rows are flagged; shared ones are not.
    config = {row["field"]: row for row in comparison["config_rows"]}
    assert config["tensor_parallel"]["differs"] is True
    assert config["tensor_parallel"]["values"] == [4, 8]
    assert config["engine_type"]["differs"] is False
    assert comparison["baseline_label"].startswith("four-card @ ")


def test_compare_runs_zero_baseline_is_not_comparable() -> None:
    """A zero baseline gives no percentage rather than an infinity."""
    baseline = _bare_report(
        [
            _record("tools", "tool_single", decode_tps=0.0, ttft_s=0.0),
        ]
    )
    candidate = _bare_report(
        [
            _record("tools", "tool_single", decode_tps=90.0, ttft_s=0.05),
        ]
    )

    comparison = compare_runs([baseline, candidate])
    rows = {row["metric"]: row for row in comparison["metric_rows"]}

    decode = rows["decode tok/s (median, single-stream)"]
    assert decode["values"][0] is None
    assert decode["deltas"][1] is None
    assert decode["pct"][1] is None

    # A tool test records no decode rate, so its repeat count carries.
    tool_row = rows["tools/tool_single repeats passed"]
    assert tool_row["values"] == [pytest.approx(1.0), pytest.approx(1.0)]
    assert tool_row["pct"][1] == pytest.approx(0.0)

    failed = rows["tests failed"]
    assert failed["values"] == [pytest.approx(0.0), pytest.approx(0.0)]
    # Zero to zero: a real delta of 0 but no percentage to speak of.
    assert failed["deltas"][1] == pytest.approx(0.0)
    assert failed["pct"][1] is None

    console, buffer = _string_console()
    render_comparison(comparison, console)
    assert NOT_COMPARABLE in buffer.getvalue()


def test_compare_runs_absent_metric_on_one_side_is_not_comparable() -> None:
    """A figure only one run recorded produces no delta at all."""
    baseline = _report_with_provenance(_throughput_results(100.0, 0.05))
    candidate = _bare_report(_throughput_results(100.0, 0.05))

    comparison = compare_runs([baseline, candidate])
    rows = {row["metric"]: row for row in comparison["metric_rows"]}
    acceptance = rows["draft-token acceptance %"]

    assert acceptance["values"][0] == pytest.approx(72.0)
    assert acceptance["values"][1] is None
    assert acceptance["deltas"][1] is None
    assert acceptance["pct"][1] is None


def test_compare_runs_intersects_disjoint_categories() -> None:
    """Only shared work is compared, and the rest is reported."""
    baseline = _bare_report(
        _throughput_results(100.0, 0.05)
        + [_record("prefill", "prefill_1k", prefill_tps=900.0, ttft_s=1.0)]
    )
    candidate = _bare_report(
        _throughput_results(120.0, 0.05)
        + [_record("reasoning", "reasoning_math", decode_tps=110.0)]
    )

    comparison = compare_runs([baseline, candidate])

    assert comparison["common_categories"] == ["throughput"]
    excluded = {
        item["category"]: item["present_in"]
        for item in comparison["excluded_categories"]
    }
    assert set(excluded) == {"prefill", "reasoning"}
    assert excluded["prefill"] == [comparison["labels"][0]]
    assert excluded["reasoning"] == [comparison["labels"][1]]

    compared = {
        (item["category"], item["test_name"]) for item in comparison["common_tests"]
    }
    assert compared == {("throughput", "decode_128"), ("throughput", "decode_512")}

    dropped = {
        (item["category"], item["test_name"]) for item in comparison["excluded_tests"]
    }
    assert dropped == {("prefill", "prefill_1k"), ("reasoning", "reasoning_math")}


def test_compare_runs_orders_concurrency_rows_numerically() -> None:
    """Concurrency metric rows follow numeric worker order."""
    levels = {32: 680.0, 8: 210.0}
    baseline = _bare_report(_concurrency_results(levels))
    candidate = _bare_report(_concurrency_results({32: 700.0, 8: 220.0}))

    comparison = compare_runs([baseline, candidate])
    aggregate_rows = [
        row["metric"]
        for row in comparison["metric_rows"]
        if row["metric"].startswith("concurrency/")
        and row["metric"].endswith("aggregate tok/s")
    ]

    assert aggregate_rows == [
        "concurrency/concurrent_8 aggregate tok/s",
        "concurrency/concurrent_32 aggregate tok/s",
    ]


def test_compare_runs_needs_two_runs() -> None:
    """One run is not a comparison."""
    report = _bare_report(_throughput_results(100.0, 0.05))
    with pytest.raises(BenchReportError, match="at least two runs"):
        compare_runs([report])


def test_compare_runs_labels_identical_runs_distinctly() -> None:
    """Two runs of one profile at one minute still get distinct columns."""
    report = _report_with_provenance(_throughput_results(100.0, 0.05))
    comparison = compare_runs([report, report])
    assert comparison["labels"][0] != comparison["labels"][1]


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def test_render_run_with_and_without_provenance() -> None:
    """Both saved shapes render without raising."""
    rich_report = _report_with_provenance(
        _throughput_results(120.0, 0.04)
        + [_record("prefill", "prefill_1k", prefill_tps=900.0, ttft_s=1.0)]
        + _concurrency_results({8: 210.0, 32: 680.0})
    )
    console, buffer = _string_console()
    render_run(rich_report, console)
    text = buffer.getvalue()
    assert "Configuration" in text
    assert "Concurrency" in text
    assert "test-gpu-node" in text

    console, buffer = _string_console()
    render_run(_bare_report(_throughput_results(100.0, 0.05)), console)
    text = buffer.getvalue()
    assert UNKNOWN in text
    assert "predates" in text


def test_render_comparison_prints_tables_and_exclusions() -> None:
    """A full comparison renders configuration, metrics and exclusions."""
    baseline = _report_with_provenance(
        _throughput_results(100.0, 0.10),
        profile_slug="four-card",
        tensor_parallel=4,
        gpus=4,
    )
    candidate = _report_with_provenance(
        _throughput_results(125.0, 0.08)
        + [_record("reasoning", "reasoning_logic", decode_tps=118.0)],
        profile_slug="eight-card",
        tensor_parallel=8,
        gpus=8,
    )

    console, buffer = _string_console()
    render_comparison(compare_runs([baseline, candidate]), console)
    text = buffer.getvalue()

    assert "Configuration" in text
    assert "Metrics" in text
    assert "tensor_parallel" in text
    assert "Not compared" in text
    assert "reasoning" in text
