"""Closed-loop tokenizer benchmark framework.

Public surface
--------------
:class:`BenchConfig`
    Configuration for one benchmark run (tokenizer kind, factory,
    metrics, device, item cap).

:class:`PerShotResult`
    Per-shot encode/decode timings, byte counts, and reconstruction metrics.

:class:`BenchResult`
    Aggregated result: per-shot tuple + aggregate stats + elapsed time.

:func:`benchmark_frame_tokenizer`
    Run a frame-tokenizer benchmark over a list of shot IDs.

:func:`benchmark_signal_tokenizer`
    Run a signal-tokenizer benchmark over a list of shot IDs.

:func:`render_comparison_table`
    Produce a Rich table comparing multiple :class:`BenchResult` objects.

:func:`save_results_json`
    Serialise results to JSON.

:func:`load_results_json`
    Restore results from JSON.
"""

from __future__ import annotations

from imas_ambix.bench.report import (
    load_results_json,
    render_comparison_table,
    save_results_json,
)
from imas_ambix.bench.tokenizer import (
    BenchConfig,
    BenchResult,
    PerShotResult,
    benchmark_frame_tokenizer,
    benchmark_signal_tokenizer,
)
from imas_ambix.eval.metrics import modality_coherence

__all__ = [
    "BenchConfig",
    "BenchResult",
    "PerShotResult",
    "benchmark_frame_tokenizer",
    "benchmark_signal_tokenizer",
    "modality_coherence",
    "render_comparison_table",
    "save_results_json",
    "load_results_json",
]
