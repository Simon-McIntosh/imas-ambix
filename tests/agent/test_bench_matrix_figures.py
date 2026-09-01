"""Tests for the saved benchmark matrix SVG renderer."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING, Any

import pytest

from imas_ambix.agent.bench_report import BenchReportError
from scripts.bench_matrix_figures import (
    FIGURE_NAMES,
    _metric_values,
    generate_figures,
    load_matrix,
    main,
)

if TYPE_CHECKING:
    from pathlib import Path


def _record(
    category: str,
    test_name: str,
    *,
    decode_tps: float,
    workers: int | None = None,
    aggregate_tps: float | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if workers is not None:
        metadata = {
            "n_workers": workers,
            "aggregate_tps": aggregate_tps,
            "wall_time": 2.0,
        }
    return {
        "category": category,
        "test_name": test_name,
        "status": "passed",
        "prompt_tokens": 16,
        "completion_tokens": 128,
        "time_to_first_token_s": 0.1,
        "total_time_s": 1.0,
        "decode_tps": decode_tps,
        "prefill_tps": 400.0,
        "model": "test-model",
        "metadata": metadata,
    }


def _write_report(
    path: Path,
    *,
    decode_tps: float,
    aggregate_scale: float,
    cards: int,
) -> None:
    results = [_record("throughput", "decode_128", decode_tps=decode_tps)]
    results.extend(
        _record(
            "concurrency",
            f"concurrent_{workers}",
            decode_tps=decode_tps / workers,
            workers=workers,
            aggregate_tps=aggregate_scale * workers,
        )
        for workers in (1, 2, 4)
    )
    report = {
        "timestamp": "2026-08-26T10:00:00+00:00",
        "server_info": {
            "data": [{"id": "test-model"}],
            "provenance": {
                "profile_slug": "test-profile",
                "model_name": "Test Model",
                "served_name": "test-model",
                "engine_type": "vllm",
                "engine_version": "0.28.0",
                "tensor_parallel": cards,
                "gpus": cards,
            },
        },
        "categories_run": ["throughput", "concurrency"],
        "results": results,
        "summary": {},
    }
    path.write_text(json.dumps(report), encoding="utf-8")


@pytest.fixture
def matrix_dir(tmp_path: Path) -> Path:
    """Two minimal comparable benchmark reports."""
    _write_report(
        tmp_path / "dsv4-2card.json",
        decode_tps=100.0,
        aggregate_scale=80.0,
        cards=2,
    )
    _write_report(
        tmp_path / "glm-4card.json",
        decode_tps=75.0,
        aggregate_scale=65.0,
        cards=4,
    )
    return tmp_path


def test_load_matrix_uses_matrix_row_names(matrix_dir: Path) -> None:
    matrix = load_matrix(matrix_dir)
    assert matrix.labels == ("DeepSeek V4 Flash · 2 H200", "GLM-5.2 · 4 H200")
    assert len(matrix.summaries) == 2
    assert matrix.comparison["common_categories"] == ["throughput", "concurrency"]


def test_load_matrix_excludes_nested_superseded_reports(matrix_dir: Path) -> None:
    superseded = matrix_dir / "superseded"
    superseded.mkdir()
    _write_report(
        superseded / "discarded.json",
        decode_tps=999.0,
        aggregate_scale=999.0,
        cards=8,
    )
    assert len(load_matrix(matrix_dir).paths) == 2


def test_load_matrix_requires_two_reports(tmp_path: Path) -> None:
    _write_report(
        tmp_path / "only.json", decode_tps=100.0, aggregate_scale=80.0, cards=2
    )
    with pytest.raises(BenchReportError, match="at least two"):
        load_matrix(tmp_path)


def test_metric_values_come_from_aligned_comparison(matrix_dir: Path) -> None:
    matrix = load_matrix(matrix_dir)
    assert _metric_values(matrix, "decode tok/s (median, single-stream)") == [
        pytest.approx(100.0),
        pytest.approx(75.0),
    ]


def test_generate_figures_writes_three_parseable_svgs(
    matrix_dir: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "figures"
    paths = generate_figures(matrix_dir, output_dir)
    assert tuple(path.name for path in paths) == FIGURE_NAMES
    expected_labels = load_matrix(matrix_dir).labels
    for path in paths:
        root = ET.parse(path).getroot()
        assert root.tag == "{http://www.w3.org/2000/svg}svg"
        text = "".join(root.itertext())
        assert all(label in text for label in expected_labels)
        assert root.find("{http://www.w3.org/2000/svg}text").get("class") == "title"


def test_main_reports_every_written_figure(
    matrix_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_dir = tmp_path / "cli-figures"
    assert main(["--matrix-dir", str(matrix_dir), "--output-dir", str(output_dir)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines == [str(output_dir / name) for name in FIGURE_NAMES]
