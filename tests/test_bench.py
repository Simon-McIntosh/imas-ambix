"""Tests for imas_ambix.bench: closed-loop tokenizer benchmark framework.

Coverage
--------
1.  BenchConfig validates tokenizer_kind values.
2.  benchmark_frame_tokenizer with PlaceholderFrameTokenizer on a synthetic
    (T, H, W) Zarr shot.
3.  Per-shot result fields populated (n_items, timings, bytes).
4.  benchmark_signal_tokenizer with UniformQuantizer on a synthetic Dataset.
5.  Codebook utilisation reported correctly for frame tokenizer.
6.  Codebook utilisation reported correctly for signal tokenizer.
7.  render_comparison_table produces a table with the expected columns.
8.  save_results_json / load_results_json JSON round-trip preserves aggregate.
9.  load_results_json factory sentinel raises RuntimeError on call.
10. Error handling: missing shot path sets error field on PerShotResult.
11. CLI smoke test: imas-ambix tokenize bench --tokenizer placeholder --kind frame.
12. CLI bench --kind signal smoke test with monkeypatched paths.
13. benchmark_frame_tokenizer multi-shot aggregation is consistent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import xarray as xr

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_frame_zarr(path: Path, *, t: int = 8, h: int = 32, w: int = 32) -> Path:
    """Create a minimal level-1 Zarr shot with a 'rbb' camera group."""
    import zarr

    path.mkdir(parents=True, exist_ok=True)
    g = zarr.open_group(str(path), mode="w")
    sub = g.create_group("rbb")
    sub.create_array(
        "data",
        data=np.random.default_rng(42).integers(0, 256, (t, h, w), dtype=np.uint16),
        dimension_names=["time", "y", "x"],
    )
    sub.create_array(
        "time",
        data=np.arange(t, dtype=np.float64) * 0.01,
        dimension_names=["time"],
    )
    return path


def _make_signal_zarr(path: Path, *, t: int = 64, group: str = "magnetics") -> Path:
    """Create a minimal level-2 Zarr shot with a signal group."""
    group_dir = path / group
    group_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    ds = xr.Dataset(
        {
            "ip": ("time", rng.standard_normal(t)),
            "beta": ("time", rng.standard_normal(t)),
        },
        coords={"time": np.linspace(0.0, 1.0, t)},
    )
    ds.to_zarr(str(group_dir), mode="w")
    return path


# ---------------------------------------------------------------------------
# 1. BenchConfig validation
# ---------------------------------------------------------------------------


def test_benchconfig_valid_frame_kind() -> None:
    from imas_ambix.bench import BenchConfig
    from imas_ambix.tokenizer.frames import PlaceholderFrameTokenizer

    cfg = BenchConfig(
        name="test",
        tokenizer_kind="frame",
        tokenizer_factory=PlaceholderFrameTokenizer,
        metrics=("psnr",),
    )
    assert cfg.tokenizer_kind == "frame"
    assert cfg.metrics == ("psnr",)
    assert cfg.device == "cpu"


def test_benchconfig_valid_signal_kind() -> None:
    from imas_ambix.bench import BenchConfig
    from imas_ambix.tokenizer.signals import UniformQuantizer

    cfg = BenchConfig(
        name="sig-test",
        tokenizer_kind="signal",
        tokenizer_factory=UniformQuantizer,
        metrics=("mae", "nrmse"),
    )
    assert cfg.tokenizer_kind == "signal"


# ---------------------------------------------------------------------------
# 2. benchmark_frame_tokenizer: basic run
# ---------------------------------------------------------------------------


def test_benchmark_frame_tokenizer_basic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from imas_ambix.bench import BenchConfig, benchmark_frame_tokenizer
    from imas_ambix.tokenizer.frames import PlaceholderFrameTokenizer

    level1_shots = tmp_path / "level1" / "shots"
    shot_zarr = level1_shots / "99001.zarr"
    _make_frame_zarr(shot_zarr, t=8, h=32, w=32)

    import imas_ambix.data.paths as paths_mod

    monkeypatch.setattr(paths_mod, "LEVEL1_DIR", level1_shots)

    cfg = BenchConfig(
        name="placeholder-cpu",
        tokenizer_kind="frame",
        tokenizer_factory=PlaceholderFrameTokenizer,
        max_items_per_shot=8,
        metrics=("psnr",),
    )
    result = benchmark_frame_tokenizer(cfg, [99001], camera="rbb", tier="level1")

    assert len(result.per_shot) == 1
    ps = result.per_shot[0]
    assert ps.shot_id == 99001
    assert ps.error is None
    assert ps.n_items == 8
    assert ps.encode_seconds >= 0.0
    assert ps.decode_seconds >= 0.0


# ---------------------------------------------------------------------------
# 3. Per-shot result fields
# ---------------------------------------------------------------------------


def test_per_shot_result_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from imas_ambix.bench import BenchConfig, benchmark_frame_tokenizer
    from imas_ambix.tokenizer.frames import PlaceholderFrameTokenizer

    level1_shots = tmp_path / "level1" / "shots"
    shot_zarr = level1_shots / "99002.zarr"
    _make_frame_zarr(shot_zarr, t=8, h=32, w=32)

    import imas_ambix.data.paths as paths_mod

    monkeypatch.setattr(paths_mod, "LEVEL1_DIR", level1_shots)

    cfg = BenchConfig(
        name="placeholder-cpu",
        tokenizer_kind="frame",
        tokenizer_factory=PlaceholderFrameTokenizer,
        max_items_per_shot=8,
        metrics=("psnr",),
    )
    result = benchmark_frame_tokenizer(cfg, [99002], camera="rbb", tier="level1")
    ps = result.per_shot[0]

    assert ps.bytes_in > 0
    assert ps.bytes_out > 0
    assert isinstance(ps.metrics, dict)


# ---------------------------------------------------------------------------
# 4. benchmark_signal_tokenizer: basic run
# ---------------------------------------------------------------------------


def test_benchmark_signal_tokenizer_basic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from imas_ambix.bench import BenchConfig, benchmark_signal_tokenizer
    from imas_ambix.tokenizer.signals import UniformQuantizer

    level2_shots = tmp_path / "level2" / "shots"
    shot_zarr = level2_shots / "99003.zarr"
    _make_signal_zarr(shot_zarr, t=64)

    import imas_ambix.data.paths as paths_mod

    monkeypatch.setattr(paths_mod, "LEVEL2_DIR", level2_shots)

    cfg = BenchConfig(
        name="uniform",
        tokenizer_kind="signal",
        tokenizer_factory=UniformQuantizer,
        max_items_per_shot=64,
        metrics=("mae", "nrmse", "correlation"),
    )
    result = benchmark_signal_tokenizer(cfg, [99003], group="magnetics", tier="level2")

    assert len(result.per_shot) == 1
    ps = result.per_shot[0]
    assert ps.error is None
    assert ps.n_items > 0
    assert "mae" in ps.metrics
    assert "nrmse" in ps.metrics
    assert "correlation" in ps.metrics


# ---------------------------------------------------------------------------
# 5. Codebook utilisation — frame tokenizer
# ---------------------------------------------------------------------------


def test_codebook_utilisation_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from imas_ambix.bench import BenchConfig, benchmark_frame_tokenizer
    from imas_ambix.tokenizer.frames import PlaceholderFrameTokenizer

    level1_shots = tmp_path / "level1" / "shots"
    shot_zarr = level1_shots / "99004.zarr"
    _make_frame_zarr(shot_zarr, t=8, h=32, w=32)

    import imas_ambix.data.paths as paths_mod

    monkeypatch.setattr(paths_mod, "LEVEL1_DIR", level1_shots)

    cfg = BenchConfig(
        name="placeholder-cpu",
        tokenizer_kind="frame",
        tokenizer_factory=PlaceholderFrameTokenizer,
        max_items_per_shot=8,
        metrics=("psnr",),
    )
    result = benchmark_frame_tokenizer(cfg, [99004], camera="rbb", tier="level1")
    ps = result.per_shot[0]

    assert ps.codebook_utilisation is not None
    assert 0.0 <= ps.codebook_utilisation <= 1.0


# ---------------------------------------------------------------------------
# 6. Codebook utilisation — signal tokenizer
# ---------------------------------------------------------------------------


def test_codebook_utilisation_signal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from imas_ambix.bench import BenchConfig, benchmark_signal_tokenizer
    from imas_ambix.tokenizer.signals import UniformQuantizer

    level2_shots = tmp_path / "level2" / "shots"
    shot_zarr = level2_shots / "99005.zarr"
    _make_signal_zarr(shot_zarr, t=64)

    import imas_ambix.data.paths as paths_mod

    monkeypatch.setattr(paths_mod, "LEVEL2_DIR", level2_shots)

    cfg = BenchConfig(
        name="uniform",
        tokenizer_kind="signal",
        tokenizer_factory=UniformQuantizer,
        metrics=("mae",),
    )
    result = benchmark_signal_tokenizer(cfg, [99005], group="magnetics", tier="level2")
    ps = result.per_shot[0]

    assert ps.codebook_utilisation is not None
    assert 0.0 <= ps.codebook_utilisation <= 1.0


# ---------------------------------------------------------------------------
# 7. render_comparison_table column check
# ---------------------------------------------------------------------------


def test_render_comparison_table_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from imas_ambix.bench import BenchConfig, benchmark_frame_tokenizer
    from imas_ambix.bench.report import render_comparison_table
    from imas_ambix.tokenizer.frames import PlaceholderFrameTokenizer

    level1_shots = tmp_path / "level1" / "shots"
    shot_zarr = level1_shots / "99006.zarr"
    _make_frame_zarr(shot_zarr, t=8, h=32, w=32)

    import imas_ambix.data.paths as paths_mod

    monkeypatch.setattr(paths_mod, "LEVEL1_DIR", level1_shots)

    cfg = BenchConfig(
        name="placeholder-cpu",
        tokenizer_kind="frame",
        tokenizer_factory=PlaceholderFrameTokenizer,
        max_items_per_shot=8,
        metrics=("psnr",),
    )
    result = benchmark_frame_tokenizer(cfg, [99006], camera="rbb", tier="level1")
    table = render_comparison_table([result])

    column_names = [col.header for col in table.columns]
    assert "name" in column_names
    assert any("throughput" in c for c in column_names)
    assert any("bytes_in" in c for c in column_names)
    assert any("bytes_out" in c for c in column_names)
    assert any("compression" in c for c in column_names)
    assert len(table.rows) == 1


# ---------------------------------------------------------------------------
# 8. save_results_json / load_results_json round-trip
# ---------------------------------------------------------------------------


def test_json_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from imas_ambix.bench import BenchConfig, benchmark_frame_tokenizer
    from imas_ambix.bench.report import load_results_json, save_results_json
    from imas_ambix.tokenizer.frames import PlaceholderFrameTokenizer

    level1_shots = tmp_path / "level1" / "shots"
    shot_zarr = level1_shots / "99007.zarr"
    _make_frame_zarr(shot_zarr, t=8, h=32, w=32)

    import imas_ambix.data.paths as paths_mod

    monkeypatch.setattr(paths_mod, "LEVEL1_DIR", level1_shots)

    cfg = BenchConfig(
        name="placeholder-cpu",
        tokenizer_kind="frame",
        tokenizer_factory=PlaceholderFrameTokenizer,
        max_items_per_shot=8,
        metrics=("psnr",),
    )
    original = benchmark_frame_tokenizer(cfg, [99007], camera="rbb", tier="level1")

    out_path = tmp_path / "bench.json"
    save_results_json([original], out_path)
    assert out_path.exists()

    loaded = load_results_json(out_path)
    assert len(loaded) == 1
    assert loaded[0].config.name == original.config.name
    assert loaded[0].aggregate == original.aggregate
    assert len(loaded[0].per_shot) == len(original.per_shot)
    assert loaded[0].per_shot[0].shot_id == original.per_shot[0].shot_id


# ---------------------------------------------------------------------------
# 9. load_results_json factory sentinel raises RuntimeError
# ---------------------------------------------------------------------------


def test_json_roundtrip_factory_sentinel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from imas_ambix.bench import BenchConfig, benchmark_frame_tokenizer
    from imas_ambix.bench.report import load_results_json, save_results_json
    from imas_ambix.tokenizer.frames import PlaceholderFrameTokenizer

    level1_shots = tmp_path / "level1" / "shots"
    shot_zarr = level1_shots / "99008.zarr"
    _make_frame_zarr(shot_zarr, t=8, h=32, w=32)

    import imas_ambix.data.paths as paths_mod

    monkeypatch.setattr(paths_mod, "LEVEL1_DIR", level1_shots)

    cfg = BenchConfig(
        name="placeholder-cpu",
        tokenizer_kind="frame",
        tokenizer_factory=PlaceholderFrameTokenizer,
        max_items_per_shot=8,
        metrics=("psnr",),
    )
    original = benchmark_frame_tokenizer(cfg, [99008], camera="rbb", tier="level1")

    out_path = tmp_path / "bench2.json"
    save_results_json([original], out_path)
    loaded = load_results_json(out_path)

    with pytest.raises(RuntimeError, match="cannot be called after JSON round-trip"):
        loaded[0].config.tokenizer_factory()


# ---------------------------------------------------------------------------
# 10. Error handling: missing shot path
# ---------------------------------------------------------------------------


def test_benchmark_frame_missing_shot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from imas_ambix.bench import BenchConfig, benchmark_frame_tokenizer
    from imas_ambix.tokenizer.frames import PlaceholderFrameTokenizer

    level1_shots = tmp_path / "level1" / "shots"
    # Do NOT create the shot zarr

    import imas_ambix.data.paths as paths_mod

    monkeypatch.setattr(paths_mod, "LEVEL1_DIR", level1_shots)

    cfg = BenchConfig(
        name="placeholder-cpu",
        tokenizer_kind="frame",
        tokenizer_factory=PlaceholderFrameTokenizer,
        metrics=("psnr",),
    )
    result = benchmark_frame_tokenizer(cfg, [99999], camera="rbb", tier="level1")

    assert len(result.per_shot) == 1
    ps = result.per_shot[0]
    assert ps.error is not None
    assert ps.n_items == 0


# ---------------------------------------------------------------------------
# 11. CLI smoke test: bench --kind frame
# ---------------------------------------------------------------------------


def test_cli_bench_frame_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from click.testing import CliRunner

    import imas_ambix.data.paths as paths_mod
    import imas_ambix.tokenizer.cli as tok_cli
    from imas_ambix.tokenizer.cli import tokenize

    level1_shots = tmp_path / "level1" / "shots"
    shot_zarr = level1_shots / "99010.zarr"
    _make_frame_zarr(shot_zarr, t=8, h=32, w=32)

    monkeypatch.setattr(paths_mod, "LEVEL1_DIR", level1_shots)
    monkeypatch.setattr(tok_cli, "LEVEL1_DIR", level1_shots)

    runner = CliRunner()
    result = runner.invoke(
        tokenize,
        [
            "bench",
            "--tokenizer",
            "placeholder",
            "--kind",
            "frame",
            "--shot-ids",
            "99010",
            "--max-items-per-shot",
            "4",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Tokenizer benchmark comparison" in result.output or "name" in result.output


# ---------------------------------------------------------------------------
# 12. CLI smoke test: bench --kind signal
# ---------------------------------------------------------------------------


def test_cli_bench_signal_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from click.testing import CliRunner

    import imas_ambix.data.paths as paths_mod
    import imas_ambix.tokenizer.cli as tok_cli
    from imas_ambix.tokenizer.cli import tokenize

    level2_shots = tmp_path / "level2" / "shots"
    shot_zarr = level2_shots / "99011.zarr"
    _make_signal_zarr(shot_zarr, t=64)

    monkeypatch.setattr(paths_mod, "LEVEL2_DIR", level2_shots)
    monkeypatch.setattr(tok_cli, "LEVEL2_DIR", level2_shots)

    runner = CliRunner()
    result = runner.invoke(
        tokenize,
        [
            "bench",
            "--tokenizer",
            "uniform",
            "--kind",
            "signal",
            "--shot-ids",
            "99011",
            "--max-items-per-shot",
            "32",
        ],
    )
    assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# 13. Multi-shot aggregation consistency
# ---------------------------------------------------------------------------


def test_benchmark_frame_multi_shot_aggregate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from imas_ambix.bench import BenchConfig, benchmark_frame_tokenizer
    from imas_ambix.tokenizer.frames import PlaceholderFrameTokenizer

    level1_shots = tmp_path / "level1" / "shots"
    for sid in [99020, 99021]:
        shot_zarr = level1_shots / f"{sid}.zarr"
        _make_frame_zarr(shot_zarr, t=8, h=32, w=32)

    import imas_ambix.data.paths as paths_mod

    monkeypatch.setattr(paths_mod, "LEVEL1_DIR", level1_shots)

    cfg = BenchConfig(
        name="placeholder-cpu",
        tokenizer_kind="frame",
        tokenizer_factory=PlaceholderFrameTokenizer,
        max_items_per_shot=8,
        metrics=("psnr",),
    )
    result = benchmark_frame_tokenizer(cfg, [99020, 99021], camera="rbb", tier="level1")

    assert len(result.per_shot) == 2
    assert result.aggregate["n_shots_ok"] == 2.0
    assert result.aggregate["n_shots_err"] == 0.0
    assert result.aggregate["throughput_items_per_s"] > 0.0
    assert "mean_psnr" in result.aggregate


# ---------------------------------------------------------------------------
# 14. benchmark_frame_tokenizer: equilibrium_loader sets modality_coherence
# ---------------------------------------------------------------------------


def test_benchmark_frame_tokenizer_with_equilibrium_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When equilibrium_loader returns a (T,) array, mean_modality_coherence is set."""
    from imas_ambix.bench import BenchConfig, benchmark_frame_tokenizer
    from imas_ambix.tokenizer.frames import PlaceholderFrameTokenizer

    level1_shots = tmp_path / "level1" / "shots"
    shot_zarr = level1_shots / "99030.zarr"
    t_frames = 8
    _make_frame_zarr(shot_zarr, t=t_frames, h=32, w=32)

    import imas_ambix.data.paths as paths_mod

    monkeypatch.setattr(paths_mod, "LEVEL1_DIR", level1_shots)

    cfg = BenchConfig(
        name="placeholder-cpu",
        tokenizer_kind="frame",
        tokenizer_factory=PlaceholderFrameTokenizer,
        max_items_per_shot=t_frames,
        metrics=("psnr",),
    )

    # Provide a fake equilibrium loader returning a linearly varying R series
    def fake_eq_loader(shot_id: int) -> np.ndarray:
        return np.linspace(1.5, 2.0, t_frames)

    result = benchmark_frame_tokenizer(
        cfg, [99030], camera="rbb", tier="level1", equilibrium_loader=fake_eq_loader
    )

    assert len(result.per_shot) == 1
    ps = result.per_shot[0]
    assert ps.error is None
    # modality_coherence field must be set (may be nan if centroid is constant,
    # but must not be None)
    assert ps.modality_coherence is not None
    # Aggregate must contain the key
    assert "mean_modality_coherence" in result.aggregate


def test_benchmark_frame_tokenizer_loader_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Leave modality coherence unset when equilibrium loading returns nothing."""
    from imas_ambix.bench import BenchConfig, benchmark_frame_tokenizer
    from imas_ambix.tokenizer.frames import PlaceholderFrameTokenizer

    level1_shots = tmp_path / "level1" / "shots"
    shot_zarr = level1_shots / "99031.zarr"
    _make_frame_zarr(shot_zarr, t=8, h=32, w=32)

    import imas_ambix.data.paths as paths_mod

    monkeypatch.setattr(paths_mod, "LEVEL1_DIR", level1_shots)

    cfg = BenchConfig(
        name="placeholder-cpu",
        tokenizer_kind="frame",
        tokenizer_factory=PlaceholderFrameTokenizer,
        max_items_per_shot=8,
        metrics=("psnr",),
    )

    def none_eq_loader(shot_id: int) -> None:
        return None

    result = benchmark_frame_tokenizer(
        cfg, [99031], camera="rbb", tier="level1", equilibrium_loader=none_eq_loader
    )

    ps = result.per_shot[0]
    assert ps.error is None
    assert ps.modality_coherence is None
    # Aggregate key present but nan (all-None → no finite values)
    assert "mean_modality_coherence" in result.aggregate
    import math

    assert math.isnan(result.aggregate["mean_modality_coherence"])


# ---------------------------------------------------------------------------
# Serving-bench speculative-decode snapshot
# ---------------------------------------------------------------------------


def test_spec_snapshot_excludes_created_timestamp_siblings() -> None:
    """An absolute snapshot reads the counters, not their same-named gauges.

    vLLM's OpenMetrics exposition pairs every speculative-decode counter
    with a ``_created`` gauge holding the engine's creation timestamp
    (~1.79e9 in 2026), labelled with the counter. A matcher that read the
    family name by substring summed that timestamp into the counter's
    absolute value — a draft-token total of 1.7886e9 instead of a few
    thousand. The snapshot is pinned to the counter values alone, across the
    draft, accepted and per-position counters and their ``_created``
    siblings, with live values read from the four-card DSpark engine.
    """
    from imas_ambix.agent.bench import _spec_decode_snapshot

    text = "\n".join(
        [
            "# TYPE vllm:spec_decode_num_drafts_total counter",
            'vllm:spec_decode_num_drafts_total{engine="0"} 1912.0',
            "vllm:spec_decode_num_drafts_created 1.7886160770693e+09",
            "# TYPE vllm:spec_decode_num_draft_tokens_total counter",
            'vllm:spec_decode_num_draft_tokens_total{engine="0"} 9560.0',
            "vllm:spec_decode_num_draft_tokens_created 1.788616077069351e+09",
            "# TYPE vllm:spec_decode_num_accepted_tokens_total counter",
            'vllm:spec_decode_num_accepted_tokens_total{engine="0"} 4418.0',
            "vllm:spec_decode_num_accepted_tokens_created 1.7886160770694067e+09",
            "# TYPE vllm:spec_decode_num_accepted_tokens_per_pos_total counter",
            "vllm:spec_decode_num_accepted_tokens_per_pos_total"
            '{engine="0",position="0"} 1423.0',
            "vllm:spec_decode_num_accepted_tokens_per_pos_total"
            '{engine="0",position="1"} 1108.0',
            "vllm:spec_decode_num_accepted_tokens_per_pos_total"
            '{engine="0",position="2"} 848.0',
            "vllm:spec_decode_num_accepted_tokens_per_pos_total"
            '{engine="0",position="3"} 612.0',
            "vllm:spec_decode_num_accepted_tokens_per_pos_total"
            '{engine="0",position="4"} 427.0',
            "vllm:spec_decode_num_accepted_tokens_per_pos_created"
            '{engine="0",position="0"} 1.788616077069438e+09',
            "",
        ]
    )

    snapshot = _spec_decode_snapshot(text)

    # The counters alone — a folded ``_created`` timestamp would read ~1.79e9
    # and a folded drafts-per-request counter would read 1912, not 9560.
    assert snapshot["draft_tokens_total"] == 9560.0
    assert snapshot["accepted_tokens_total"] == 4418.0
    assert snapshot["num_accepted_per_pos"] == [
        1423.0,
        1108.0,
        848.0,
        612.0,
        427.0,
    ]
    for value in (snapshot["draft_tokens_total"], snapshot["accepted_tokens_total"]):
        assert value < 1.0e6
