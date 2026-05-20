"""Benchmark result rendering and serialisation.

Functions
---------
render_comparison_table
    Produce a :class:`rich.table.Table` comparing multiple
    :class:`~imas_ambix.bench.tokenizer.BenchResult` objects side-by-side.

save_results_json / load_results_json
    Round-trip serialisation to/from JSON so results can be compared
    across sessions without re-running benchmarks.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rich.table import Table as RichTable

    from imas_ambix.bench.tokenizer import BenchConfig, BenchResult, PerShotResult


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _fmt(value: object, decimals: int = 3) -> str:
    """Format a float or non-float as a short string."""
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return f"{value:.{decimals}f}"
    return str(value)


def render_comparison_table(results: list[BenchResult]) -> RichTable:
    """Return a Rich table with one row per :class:`BenchResult`.

    Columns
    -------
    name, n_shots, throughput (items/s), bytes_in, bytes_out,
    compression_ratio, then one column per metric found in any result.
    """
    from rich.table import Table

    # Collect all metric keys across all results
    metric_keys: list[str] = []
    for res in results:
        for k in res.aggregate:
            if k.startswith("mean_") and k not in (
                "mean_encode_s",
                "mean_decode_s",
                "mean_codebook_utilisation",
            ):
                bare = k[len("mean_") :]
                if bare not in metric_keys:
                    metric_keys.append(bare)

    table = Table(title="Tokenizer benchmark comparison")
    table.add_column("name", style="bold cyan")
    table.add_column("n_shots", justify="right")
    table.add_column("throughput (items/s)", justify="right")
    table.add_column("bytes_in", justify="right")
    table.add_column("bytes_out", justify="right")
    table.add_column("compression_ratio", justify="right")
    for m in metric_keys:
        table.add_column(m, justify="right")

    for res in results:
        agg = res.aggregate
        n_shots = int(agg.get("n_shots_ok", 0))
        tput = _fmt(agg.get("throughput_items_per_s", float("nan")), 2)
        b_in = _fmt(agg.get("total_bytes_in", float("nan")), 0)
        b_out = _fmt(agg.get("total_bytes_out", float("nan")), 0)
        cr = _fmt(agg.get("compression_ratio", float("nan")), 3)
        metric_vals = [_fmt(agg.get(f"mean_{m}", float("nan")), 3) for m in metric_keys]
        table.add_row(
            res.config.name,
            str(n_shots),
            tput,
            b_in,
            b_out,
            cr,
            *metric_vals,
        )

    return table


# ---------------------------------------------------------------------------
# JSON serialisation
# ---------------------------------------------------------------------------


def _per_shot_to_dict(ps: PerShotResult) -> dict:
    return {
        "shot_id": ps.shot_id,
        "n_items": ps.n_items,
        "encode_seconds": ps.encode_seconds,
        "decode_seconds": ps.decode_seconds,
        "bytes_in": ps.bytes_in,
        "bytes_out": ps.bytes_out,
        "metrics": ps.metrics,
        "codebook_utilisation": ps.codebook_utilisation,
        "error": ps.error,
    }


def _config_to_dict(cfg: BenchConfig) -> dict:
    factory_name = getattr(
        cfg.tokenizer_factory, "__qualname__", str(cfg.tokenizer_factory)
    )
    return {
        "name": cfg.name,
        "tokenizer_kind": cfg.tokenizer_kind,
        "tokenizer_factory": factory_name,
        "max_items_per_shot": cfg.max_items_per_shot,
        "metrics": list(cfg.metrics),
        "device": cfg.device,
    }


def _result_to_dict(res: BenchResult) -> dict:
    return {
        "config": _config_to_dict(res.config),
        "per_shot": [_per_shot_to_dict(ps) for ps in res.per_shot],
        "aggregate": res.aggregate,
        "elapsed_s": res.elapsed_s,
    }


def save_results_json(results: list[BenchResult], path: Path) -> None:
    """Serialise a list of :class:`BenchResult` to a JSON file at *path*.

    The ``tokenizer_factory`` callable is stored as its ``__qualname__``
    string (not a callable reference). Use :func:`load_results_json` to
    restore the data, or re-instantiate configs manually.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [_result_to_dict(r) for r in results]
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, allow_nan=True)


def load_results_json(path: Path) -> list[BenchResult]:
    """Load a list of :class:`BenchResult` from a JSON file written by
    :func:`save_results_json`.

    The ``tokenizer_factory`` field of the restored :class:`BenchConfig`
    is set to a sentinel lambda that raises :class:`RuntimeError` when
    called, since the factory callable cannot be serialised. All other
    fields are fully restored for display and comparison purposes.
    """
    from imas_ambix.bench.tokenizer import BenchConfig, BenchResult, PerShotResult

    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        raw = json.load(fh)

    results: list[BenchResult] = []
    for entry in raw:
        cfg_raw = entry["config"]

        def _missing_factory(factory_name: str = cfg_raw.get("tokenizer_factory", "?")):
            def _f() -> None:
                raise RuntimeError(
                    f"tokenizer_factory {factory_name!r} cannot be called after "
                    "JSON round-trip — reconstruct BenchConfig manually."
                )

            return _f

        cfg = BenchConfig(
            name=cfg_raw["name"],
            tokenizer_kind=cfg_raw["tokenizer_kind"],
            tokenizer_factory=_missing_factory(),
            max_items_per_shot=cfg_raw.get("max_items_per_shot"),
            metrics=tuple(cfg_raw.get("metrics", [])),
            device=cfg_raw.get("device", "cpu"),
        )

        per_shot = tuple(
            PerShotResult(
                shot_id=ps["shot_id"],
                n_items=ps["n_items"],
                encode_seconds=ps["encode_seconds"],
                decode_seconds=ps["decode_seconds"],
                bytes_in=ps["bytes_in"],
                bytes_out=ps["bytes_out"],
                metrics=ps["metrics"],
                codebook_utilisation=ps["codebook_utilisation"],
                error=ps.get("error"),
            )
            for ps in entry["per_shot"]
        )

        results.append(
            BenchResult(
                config=cfg,
                per_shot=per_shot,
                aggregate=entry["aggregate"],
                elapsed_s=entry["elapsed_s"],
            )
        )

    return results
