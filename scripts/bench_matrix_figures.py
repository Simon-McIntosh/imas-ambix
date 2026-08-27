#!/usr/bin/env python3
"""Render comparable benchmark reports as a compact SVG figure set.

The saved-report authority lives in :mod:`imas_ambix.agent.bench_report`.
This module only selects reports, gives the matrix rows human-readable labels,
and projects the comparison into plots. It does not recompute benchmark
statistics from request records.
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from imas_ambix.agent.bench_report import (
    BenchReportError,
    compare_runs,
    describe_run,
    load_report,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


DEFAULT_OUTPUT_DIR = Path("docs/figures/glm-52-local-serve-performance")
FIGURE_NAMES = (
    "aggregate-throughput.svg",
    "per-stream-throughput.svg",
    "headline-performance.svg",
)

_SVG_NS = "http://www.w3.org/2000/svg"
_COLORS = ("#2563a6", "#4c90c0", "#83b9d8", "#c75b28", "#e39b42", "#6f5ba7")
_MARKERS = ("circle", "square", "triangle", "diamond", "plus", "cross")
_METHOD_NOTES = {
    "glm-4card": (
        "† GLM-5.2 · 4 H200 used 256 generated tokens per worker; "
        "the other rows used 1024. Remeasurement is pending."
    )
}


@dataclass(frozen=True)
class BenchmarkMatrix:
    """Saved reports and their bench-report-derived comparison data."""

    paths: tuple[Path, ...]
    labels: tuple[str, ...]
    method_notes: tuple[str | None, ...]
    reports: tuple[dict[str, Any], ...]
    summaries: tuple[dict[str, Any], ...]
    comparison: dict[str, Any]


def _display_label(path: Path, summary: dict[str, Any]) -> str:
    """Return a concise matrix label, preferring the matrix row name."""
    tokens = path.stem.split("-")
    if len(tokens) >= 2 and tokens[-1].endswith("card"):
        cards = tokens[-1].removesuffix("card")
        family = "-".join(tokens[:-1])
        family_names = {
            "dsv4": "DeepSeek V4 Flash",
            "glm": "GLM-5.2",
        }
        return f"{family_names.get(family, family)} · {cards} H200"
    return str(summary["label"])


def load_matrix(matrix_dir: str | Path) -> BenchmarkMatrix:
    """Load top-level JSON reports and build the canonical comparison."""
    root = Path(matrix_dir)
    if not root.is_dir():
        raise BenchReportError(f"{root}: benchmark matrix directory does not exist")

    paths = tuple(sorted(path for path in root.glob("*.json") if path.is_file()))
    if len(paths) < 2:
        count = len(paths)
        raise BenchReportError(
            f"{root}: benchmark matrix needs at least two JSON reports, got {count}"
        )

    reports = tuple(load_report(path) for path in paths)
    summaries = tuple(describe_run(report) for report in reports)
    comparison = compare_runs(reports)
    method_notes = tuple(_METHOD_NOTES.get(path.stem) for path in paths)
    labels = tuple(
        f"{_display_label(path, summary)}{' †' if note else ''}"
        for path, summary, note in zip(paths, summaries, method_notes, strict=True)
    )
    return BenchmarkMatrix(paths, labels, method_notes, reports, summaries, comparison)


def _element(parent: ET.Element, tag: str, **attributes: object) -> ET.Element:
    """Append an SVG element with stringified attributes."""
    return ET.SubElement(
        parent,
        f"{{{_SVG_NS}}}{tag}",
        {
            key.removesuffix("_").replace("_", "-"): str(value)
            for key, value in attributes.items()
        },
    )


def _text(
    parent: ET.Element,
    value: str,
    x: float,
    y: float,
    *,
    css_class: str = "",
    anchor: str = "start",
) -> ET.Element:
    """Append one SVG text node."""
    node = _element(
        parent,
        "text",
        x=f"{x:.1f}",
        y=f"{y:.1f}",
        class_=css_class,
        text_anchor=anchor,
    )
    node.text = value
    return node


def _svg(title: str, description: str, *, width: int, height: int) -> ET.Element:
    """Create an accessible SVG root with the shared visual vocabulary."""
    root = ET.Element(
        f"{{{_SVG_NS}}}svg",
        {
            "viewBox": f"0 0 {width} {height}",
            "width": str(width),
            "height": str(height),
            "role": "img",
            "aria-labelledby": "figure-title figure-description",
        },
    )
    title_node = _element(root, "title", id="figure-title")
    title_node.text = title
    description_node = _element(root, "desc", id="figure-description")
    description_node.text = description
    style = _element(root, "style")
    style.text = """
      text { font-family: system-ui, sans-serif; fill: #20262d; }
      .title { font-size: 23px; font-weight: 700; }
      .subtitle { font-size: 13px; fill: #55606d; }
      .axis-label { font-size: 13px; font-weight: 600; }
      .tick { font-size: 12px; fill: #56616d; }
      .legend { font-size: 12px; }
      .value { font-size: 11px; font-weight: 600; }
      .grid { stroke: #d8dde2; stroke-width: 1; }
      .axis { stroke: #68737e; stroke-width: 1.2; }
    """
    _element(root, "rect", x=0, y=0, width=width, height=height, fill="#ffffff")
    return root


def _write_svg(root: ET.Element, path: Path) -> None:
    """Write a stable, indented SVG document."""
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def _marker(parent: ET.Element, kind: str, x: float, y: float, color: str) -> None:
    """Draw one compact series marker."""
    if kind == "circle":
        _element(parent, "circle", cx=x, cy=y, r=4.5, fill=color)
    elif kind == "square":
        _element(parent, "rect", x=x - 4.5, y=y - 4.5, width=9, height=9, fill=color)
    elif kind == "triangle":
        points = f"{x:.1f},{y - 5:.1f} {x - 5:.1f},{y + 4:.1f} {x + 5:.1f},{y + 4:.1f}"
        _element(parent, "polygon", points=points, fill=color)
    elif kind == "diamond":
        points = (
            f"{x:.1f},{y - 5:.1f} {x - 5:.1f},{y:.1f} "
            f"{x:.1f},{y + 5:.1f} {x + 5:.1f},{y:.1f}"
        )
        _element(parent, "polygon", points=points, fill=color)
    else:
        _element(
            parent,
            "circle",
            cx=x,
            cy=y,
            r=4.5,
            fill="#ffffff",
            stroke=color,
            stroke_width=2,
        )
        _element(
            parent,
            "line",
            x1=x - 3,
            y1=y,
            x2=x + 3,
            y2=y,
            stroke=color,
            stroke_width=1.5,
        )
        if kind == "cross":
            _element(
                parent,
                "line",
                x1=x,
                y1=y - 3,
                x2=x,
                y2=y + 3,
                stroke=color,
                stroke_width=1.5,
            )


def _plot_concurrency(
    matrix: BenchmarkMatrix,
    *,
    value_key: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """Plot one bench-report concurrency projection for every run."""
    width, height = 1000, 665
    left, right, top, bottom = 92.0, 35.0, 92.0, 197.0
    plot_width = width - left - right
    plot_height = height - top - bottom
    root = _svg(
        title,
        f"{ylabel} for each saved benchmark run from one to 32 concurrent requests.",
        width=width,
        height=height,
    )
    _text(root, title, 28, 38, css_class="title")
    _text(
        root,
        "Same-stack benchmark matrix · aligned by bench_report.compare_runs",
        28,
        62,
        css_class="subtitle",
    )

    workers = sorted(
        {
            level["workers"]
            for run in matrix.comparison["runs"]
            for level in run["concurrency"]
            if level["workers"] is not None
        }
    )
    all_values = [
        float(level[value_key])
        for run in matrix.comparison["runs"]
        for level in run["concurrency"]
        if level[value_key] is not None
    ]
    maximum = max(all_values, default=1.0) * 1.08

    for step in range(6):
        value = maximum * step / 5
        y = top + plot_height - plot_height * step / 5
        _element(
            root,
            "line",
            x1=left,
            y1=y,
            x2=width - right,
            y2=y,
            class_="grid",
        )
        _text(root, f"{value:,.0f}", left - 10, y + 4, css_class="tick", anchor="end")
    _element(
        root,
        "line",
        x1=left,
        y1=top,
        x2=left,
        y2=top + plot_height,
        class_="axis",
    )
    _element(
        root,
        "line",
        x1=left,
        y1=top + plot_height,
        x2=width - right,
        y2=top + plot_height,
        class_="axis",
    )

    x_positions = {
        worker: left + index * plot_width / max(len(workers) - 1, 1)
        for index, worker in enumerate(workers)
    }
    for worker, x in x_positions.items():
        _text(
            root,
            str(worker),
            x,
            top + plot_height + 24,
            css_class="tick",
            anchor="middle",
        )
    _text(
        root,
        "Concurrent requests",
        left + plot_width / 2,
        top + plot_height + 52,
        css_class="axis-label",
        anchor="middle",
    )
    y_label = _text(
        root,
        ylabel,
        24,
        top + plot_height / 2,
        css_class="axis-label",
        anchor="middle",
    )
    y_label.set("transform", f"rotate(-90 24 {top + plot_height / 2:.1f})")

    for index, (label, note, run) in enumerate(
        zip(
            matrix.labels,
            matrix.method_notes,
            matrix.comparison["runs"],
            strict=True,
        )
    ):
        color = _COLORS[index % len(_COLORS)]
        marker = _MARKERS[index % len(_MARKERS)]
        points = [
            (
                x_positions[level["workers"]],
                top + plot_height * (1 - float(level[value_key]) / maximum),
            )
            for level in run["concurrency"]
            if level["workers"] in x_positions and level[value_key] is not None
        ]
        if not points:
            continue
        path_data = " ".join(
            f"{'M' if position == 0 else 'L'} {x:.1f} {y:.1f}"
            for position, (x, y) in enumerate(points)
        )
        _element(
            root,
            "path",
            d=path_data,
            fill="none",
            stroke=color,
            stroke_width=2.4,
            stroke_dasharray="8 5" if note else "none",
        )
        for x, y in points:
            _marker(root, marker, x, y, color)

        legend_x = 55 + (index % 2) * 470
        legend_y = 555 + (index // 2) * 23
        _element(
            root,
            "line",
            x1=legend_x,
            y1=legend_y,
            x2=legend_x + 28,
            y2=legend_y,
            stroke=color,
            stroke_width=2.4,
            stroke_dasharray="8 5" if note else "none",
        )
        _marker(root, marker, legend_x + 14, legend_y, color)
        _text(root, label, legend_x + 38, legend_y + 4, css_class="legend")

    for note in matrix.method_notes:
        if note:
            _text(root, note, 28, 644, css_class="subtitle")

    _write_svg(root, output_path)


def _metric_values(matrix: BenchmarkMatrix, metric: str) -> list[float | None]:
    """Read one aligned metric row produced by ``compare_runs``."""
    for row in matrix.comparison["metric_rows"]:
        if row["metric"] == metric:
            return list(row["values"])
    raise BenchReportError(f"comparison did not produce required metric {metric!r}")


def _plot_headlines(matrix: BenchmarkMatrix, output_path: Path) -> None:
    """Plot three headline measures with their native units."""
    measures = (
        ("decode tok/s (median, single-stream)", "Single-stream decode", "tok/s"),
        ("ttft ms (median)", "Median first-token latency", "ms"),
        ("peak aggregate tok/s", "Peak aggregate throughput", "tok/s"),
    )
    width, height = 1200, 620
    left, top = 205.0, 105.0
    row_height = 76.0
    panel_gap = 34.0
    panel_width = (width - left - 35 - panel_gap * 2) / 3
    root = _svg(
        "Benchmark matrix headline performance",
        "Three aligned bar panels compare single-stream decode, median "
        "first-token latency, and peak aggregate throughput.",
        width=width,
        height=height,
    )
    _text(root, "Benchmark matrix headline performance", 28, 38, css_class="title")
    _text(
        root,
        "Native units shown separately; lower latency is better, "
        "higher throughput is better · † different output-length method",
        28,
        62,
        css_class="subtitle",
    )

    for row, label in enumerate(matrix.labels):
        y = top + row * row_height + 23
        _text(root, label, left - 14, y, css_class="legend", anchor="end")

    for panel, (metric, subtitle, unit) in enumerate(measures):
        x0 = left + panel * (panel_width + panel_gap)
        raw_values = _metric_values(matrix, metric)
        maximum = max((value for value in raw_values if value is not None), default=1.0)
        _text(root, subtitle, x0, 86, css_class="axis-label")
        _text(root, unit, x0 + panel_width, 86, css_class="tick", anchor="end")
        for row, value in enumerate(raw_values):
            y = top + row * row_height
            _element(
                root,
                "rect",
                x=x0,
                y=y,
                width=panel_width,
                height=31,
                fill="#f0f2f4",
                rx=2,
            )
            if value is None:
                _text(root, "not recorded", x0 + 7, y + 21, css_class="tick")
                continue
            bar_width = panel_width * float(value) / maximum
            color = _COLORS[row % len(_COLORS)]
            _element(
                root,
                "rect",
                x=x0,
                y=y,
                width=bar_width,
                height=31,
                fill=color,
                rx=2,
            )
            label_x = min(x0 + bar_width + 6, x0 + panel_width - 4)
            anchor = "end" if label_x >= x0 + panel_width - 4 else "start"
            _text(
                root,
                f"{value:,.1f}",
                label_x,
                y + 21,
                css_class="value",
                anchor=anchor,
            )

    for note in matrix.method_notes:
        if note:
            _text(root, note, 28, 584, css_class="subtitle")

    _write_svg(root, output_path)


def generate_figures(
    matrix_dir: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> tuple[Path, ...]:
    """Render the benchmark matrix and return the three SVG paths."""
    ET.register_namespace("", _SVG_NS)
    matrix = load_matrix(matrix_dir)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    aggregate, per_stream, headlines = (destination / name for name in FIGURE_NAMES)
    _plot_concurrency(
        matrix,
        value_key="aggregate_tps",
        title="Aggregate throughput under concurrent load",
        ylabel="Aggregate output (tok/s)",
        output_path=aggregate,
    )
    _plot_concurrency(
        matrix,
        value_key="per_stream_tps",
        title="Per-stream throughput under concurrent load",
        ylabel="Median stream decode (tok/s)",
        output_path=per_stream,
    )
    _plot_headlines(matrix, headlines)
    return aggregate, per_stream, headlines


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render saved agent benchmark reports as SVG comparisons."
    )
    parser.add_argument(
        "--matrix-dir",
        type=Path,
        required=True,
        help="Directory containing the comparable top-level JSON reports.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"SVG destination (default: {DEFAULT_OUTPUT_DIR}).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = _parser().parse_args(argv)
    for path in generate_figures(args.matrix_dir, args.output_dir):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
