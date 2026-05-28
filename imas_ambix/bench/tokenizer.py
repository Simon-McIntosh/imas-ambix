"""Closed-loop tokenizer benchmarking framework.

Encodes then decodes frame or signal data, times each step, computes
reconstruction metrics, and aggregates results across a shot corpus.

Usage
-----
::

    from imas_ambix.bench import BenchConfig, benchmark_frame_tokenizer
    from imas_ambix.tokenizer.frames import PlaceholderFrameTokenizer

    cfg = BenchConfig(
        name="placeholder-cpu",
        tokenizer_kind="frame",
        tokenizer_factory=PlaceholderFrameTokenizer,
        max_items_per_shot=8,
        metrics=("psnr",),
        device="cpu",
    )
    result = benchmark_frame_tokenizer(cfg, shot_ids=[15085], camera="rbb")
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from imas_ambix.data.paths import Tier
    from imas_ambix.tokenizer.base import Tokenizer

# ---------------------------------------------------------------------------
# Configuration and result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchConfig:
    """Configuration for a single benchmark run."""

    name: str
    """Logical name of this run, e.g. ``"open-magvit2-cpu"``."""

    tokenizer_kind: str
    """``"frame"`` or ``"signal"``."""

    tokenizer_factory: Callable[[], Tokenizer]
    """Zero-arg factory that constructs the tokenizer instance."""

    max_items_per_shot: int | None = None
    """Cap on frames (frame mode) or timesteps (signal mode) per shot.

    Useful for CPU benchmarks where full shots are slow.
    """

    metrics: tuple[str, ...] = ("psnr",)
    """Metric names to compute. Frame metrics dispatched to
    :mod:`imas_ambix.eval.metrics`; signal metrics are ``mae``,
    ``nrmse``, ``correlation``."""

    device: str = "cpu"
    """Device hint passed to the tokenizer factory when relevant."""


@dataclass(frozen=True)
class PerShotResult:
    """Benchmark measurements for a single shot."""

    shot_id: int
    n_items: int
    """Frames or timesteps actually encoded."""

    encode_seconds: float
    decode_seconds: float
    bytes_in: int
    """Approximate uncompressed input bytes."""

    bytes_out: int
    """Token stream bytes (int32 × token count)."""

    metrics: dict[str, float]
    codebook_utilisation: float | None
    """Fraction of vocab seen across this shot, or ``None`` if not applicable."""

    modality_coherence: float | None = None
    """Pearson r between centroid R and equilibrium magnetic axis R, or ``None``
    when no equilibrium loader was supplied or the loader returned ``None``."""

    error: str | None = None
    """Non-``None`` when the shot failed; other fields may be zero/empty."""


@dataclass(frozen=True)
class BenchResult:
    """Aggregated result for one :class:`BenchConfig` across all shots."""

    config: BenchConfig
    per_shot: tuple[PerShotResult, ...]
    aggregate: dict[str, float]
    """Mean per metric across successful shots, plus throughput and byte totals."""

    elapsed_s: float
    """Wall-clock time from first to last shot."""


# ---------------------------------------------------------------------------
# Signal metric helpers (inline, no external dep)
# ---------------------------------------------------------------------------


def _mae(reference: np.ndarray, prediction: np.ndarray) -> float:
    """Mean absolute error between two equal-shape float arrays."""
    diff = reference.astype(np.float64) - prediction.astype(np.float64)
    return float(np.mean(np.abs(diff)))


def _nrmse(reference: np.ndarray, prediction: np.ndarray) -> float:
    """Per-channel normalised RMSE averaged across channels.

    Normalisation uses the std of the reference channel; if std == 0 the
    channel contributes 0.0 to the average.
    """
    ref = reference.astype(np.float64)
    pred = prediction.astype(np.float64)
    if ref.ndim == 1:
        ref = ref[:, np.newaxis]
        pred = pred[:, np.newaxis]
    n_ch = ref.shape[1]
    vals: list[float] = []
    for i in range(n_ch):
        rmse = float(np.sqrt(np.mean((ref[:, i] - pred[:, i]) ** 2)))
        std = float(np.std(ref[:, i]))
        vals.append(rmse / std if std > 1e-12 else 0.0)
    return float(np.mean(vals)) if vals else 0.0


def _correlation(reference: np.ndarray, prediction: np.ndarray) -> float:
    """Per-channel Pearson r averaged across channels."""
    ref = reference.astype(np.float64)
    pred = prediction.astype(np.float64)
    if ref.ndim == 1:
        ref = ref[:, np.newaxis]
        pred = pred[:, np.newaxis]
    n_ch = ref.shape[1]
    vals: list[float] = []
    for i in range(n_ch):
        r_std = float(np.std(ref[:, i]))
        p_std = float(np.std(pred[:, i]))
        if r_std < 1e-12 or p_std < 1e-12:
            vals.append(0.0)
            continue
        r = float(np.corrcoef(ref[:, i], pred[:, i])[0, 1])
        vals.append(r if np.isfinite(r) else 0.0)
    return float(np.mean(vals)) if vals else 0.0


def _codebook_utilisation(token_ids: np.ndarray, vocab_size: int) -> float | None:
    """Fraction of local token ids seen across a shot."""
    if vocab_size <= 0:
        return None
    unique = len(np.unique(token_ids.ravel()))
    return float(unique) / float(vocab_size)


# ---------------------------------------------------------------------------
# Frame benchmark
# ---------------------------------------------------------------------------


def benchmark_frame_tokenizer(
    config: BenchConfig,
    shot_ids: list[int],
    camera: str = "rbb",
    tier: Tier = "level1",
    equilibrium_loader: Callable[[int], np.ndarray | None] | None = None,
) -> BenchResult:
    """Benchmark a frame tokenizer over a list of shots.

    For each shot, loads ``(T, H, W)`` frames from the level-1 Zarr
    mirror, encodes with the tokenizer, decodes, and measures:

    - encode/decode latency
    - bytes_in / bytes_out
    - requested metrics from :mod:`imas_ambix.eval.metrics`
    - codebook utilisation
    - optional cross-modality coherence (Pearson r)

    Parameters
    ----------
    config:
        Benchmark configuration.
    shot_ids:
        Shots to benchmark.
    camera:
        Camera source name at level-1 (e.g. ``"rbb"``).
    tier:
        Data tier (always ``"level1"`` for camera data).
    equilibrium_loader:
        Optional callable ``(shot_id) -> np.ndarray | None``. When provided
        and returns a non-``None`` ``(T,)`` array of magnetic axis R values,
        the cross-modality coherence score is computed for that shot.
    """
    import xarray as xr

    from imas_ambix.data.paths import LEVEL1_DIR, LEVEL2_DIR
    from imas_ambix.eval.metrics import (
        centroid_mse,
        chord_nrmse,
        lpips,
        modality_coherence as _modality_coherence,
        psnr,
        rfid,
    )

    data_root = LEVEL1_DIR if tier == "level1" else LEVEL2_DIR

    tok = config.tokenizer_factory()
    per_shot: list[PerShotResult] = []
    t0_wall = time.perf_counter()

    for shot_id in shot_ids:
        shot_path = data_root / f"{shot_id}.zarr"
        try:
            ds = xr.open_zarr(str(shot_path), group=camera, consolidated=False)
            frames = np.asarray(ds["data"].values)

            if config.max_items_per_shot is not None:
                frames = frames[: config.max_items_per_shot]

            n_items = frames.shape[0]
            bytes_in = int(frames.nbytes)

            # Encode
            t_enc0 = time.perf_counter()
            encoded = tok.encode(frames)
            encode_seconds = time.perf_counter() - t_enc0

            bytes_out = int(encoded.token_ids.nbytes)

            # Decode
            t_dec0 = time.perf_counter()
            decoded = tok.decode(encoded)
            decode_seconds = time.perf_counter() - t_dec0

            # Align shapes for metric computation
            n_compare = min(decoded.shape[0], frames.shape[0])
            src = frames[:n_compare]
            dec = decoded[:n_compare]
            # If decoded is (T, H, W, C) but src is (T, H, W), upcast
            if src.ndim == 3 and dec.ndim == 4:
                src = np.repeat(src[..., np.newaxis], dec.shape[-1], axis=-1)
            # Both to uint8
            src_u8 = np.clip(src, 0, 255).astype(np.uint8)
            dec_u8 = np.clip(dec, 0, 255).astype(np.uint8)

            # Compute requested metrics
            metrics: dict[str, float] = {}
            metric_fns = {
                "psnr": psnr,
                "centroid_mse": centroid_mse,
                "chord_nrmse": chord_nrmse,
                "rfid": rfid,
                "lpips": lpips,
            }
            for m in config.metrics:
                if m in metric_fns:
                    try:
                        metrics[m] = metric_fns[m](src_u8, dec_u8)  # type: ignore[call-arg,operator]
                    except Exception:
                        metrics[m] = float("nan")

            util = _codebook_utilisation(
                encoded.token_ids, getattr(tok, "vocab_size", 0)
            )

            # Cross-modality coherence (optional)
            coh: float | None = None
            if equilibrium_loader is not None:
                try:
                    mag_axis_r = equilibrium_loader(shot_id)
                    if mag_axis_r is not None:
                        coh = _modality_coherence(dec_u8, np.asarray(mag_axis_r))
                except Exception:
                    coh = None

            per_shot.append(
                PerShotResult(
                    shot_id=shot_id,
                    n_items=n_items,
                    encode_seconds=encode_seconds,
                    decode_seconds=decode_seconds,
                    bytes_in=bytes_in,
                    bytes_out=bytes_out,
                    metrics=metrics,
                    codebook_utilisation=util,
                    modality_coherence=coh,
                )
            )
        except Exception:
            per_shot.append(
                PerShotResult(
                    shot_id=shot_id,
                    n_items=0,
                    encode_seconds=0.0,
                    decode_seconds=0.0,
                    bytes_in=0,
                    bytes_out=0,
                    metrics={},
                    codebook_utilisation=None,
                    error=traceback.format_exc(limit=3),
                )
            )

    elapsed_s = time.perf_counter() - t0_wall
    aggregate = _aggregate(per_shot, config.metrics, elapsed_s)
    # Add mean_modality_coherence to aggregate when loader was supplied
    if equilibrium_loader is not None:
        coh_vals = [
            s.modality_coherence
            for s in per_shot
            if s.modality_coherence is not None and s.error is None
        ]
        import math as _math
        finite_coh = [v for v in coh_vals if _math.isfinite(v)]
        aggregate["mean_modality_coherence"] = (
            float(np.mean(finite_coh)) if finite_coh else float("nan")
        )
    return BenchResult(
        config=config,
        per_shot=tuple(per_shot),
        aggregate=aggregate,
        elapsed_s=elapsed_s,
    )


# ---------------------------------------------------------------------------
# Signal benchmark
# ---------------------------------------------------------------------------


def benchmark_signal_tokenizer(
    config: BenchConfig,
    shot_ids: list[int],
    group: str = "magnetics",
    tier: Tier = "level2",
) -> BenchResult:
    """Benchmark a signal tokenizer over a list of shots.

    For each shot, loads the ``group`` xarray Dataset from the level-2
    Zarr mirror, fits the tokenizer on that shot (single-shot fit), then
    encodes and decodes.

    Signal metrics computed: ``mae``, ``nrmse``, ``correlation``.

    Parameters
    ----------
    config:
        Benchmark configuration.
    shot_ids:
        Shots to benchmark.
    group:
        Level-2 group to load (e.g. ``"magnetics"``).
    tier:
        Data tier (always ``"level2"`` for signal data).
    """
    import xarray as xr

    from imas_ambix.data.paths import LEVEL1_DIR, LEVEL2_DIR

    data_root = LEVEL1_DIR if tier == "level1" else LEVEL2_DIR

    tok = config.tokenizer_factory()
    per_shot: list[PerShotResult] = []
    t0_wall = time.perf_counter()

    for shot_id in shot_ids:
        shot_path = data_root / f"{shot_id}.zarr"
        try:
            ds = xr.open_zarr(str(shot_path), group=group, consolidated=False)

            # Fit on this single shot so tokenizer has normalisation stats
            if hasattr(tok, "fit"):
                tok.fit([ds])

            # Cap timesteps if requested
            if config.max_items_per_shot is not None:
                ds = ds.isel(time=slice(0, config.max_items_per_shot))

            n_items = int(ds.sizes.get("time", 0))
            bytes_in = sum(int(np.asarray(ds[v].values).nbytes) for v in ds.data_vars)

            # Encode
            t_enc0 = time.perf_counter()
            encoded = tok.encode(ds)
            encode_seconds = time.perf_counter() - t_enc0

            bytes_out = int(encoded.token_ids.nbytes)

            # Decode
            t_dec0 = time.perf_counter()
            decoded_ds = tok.decode(encoded)
            decode_seconds = time.perf_counter() - t_dec0

            # Build aligned reference and prediction arrays (T, n_channels)
            channel_names = encoded.channel_names
            ref_cols = []
            pred_cols = []
            for ch in channel_names:
                if ch in ds and ch in decoded_ds:
                    ref_arr = np.asarray(ds[ch].values, dtype=np.float64)
                    pred_arr = np.asarray(decoded_ds[ch].values, dtype=np.float64)
                    n = min(len(ref_arr), len(pred_arr))
                    ref_cols.append(ref_arr[:n])
                    pred_cols.append(pred_arr[:n])

            metrics: dict[str, float] = {}
            if ref_cols and pred_cols:
                n_t = min(len(c) for c in ref_cols)
                ref_mat = np.column_stack([c[:n_t] for c in ref_cols])
                pred_mat = np.column_stack([c[:n_t] for c in pred_cols])

                sig_metric_fns: dict[str, Callable[..., float]] = {
                    "mae": _mae,
                    "nrmse": _nrmse,
                    "correlation": _correlation,
                }
                for m in config.metrics:
                    if m in sig_metric_fns:
                        try:
                            metrics[m] = sig_metric_fns[m](ref_mat, pred_mat)
                        except Exception:
                            metrics[m] = float("nan")

            util = _codebook_utilisation(
                encoded.token_ids, getattr(tok, "vocab_size", 0)
            )

            per_shot.append(
                PerShotResult(
                    shot_id=shot_id,
                    n_items=n_items,
                    encode_seconds=encode_seconds,
                    decode_seconds=decode_seconds,
                    bytes_in=bytes_in,
                    bytes_out=bytes_out,
                    metrics=metrics,
                    codebook_utilisation=util,
                )
            )
        except Exception:
            per_shot.append(
                PerShotResult(
                    shot_id=shot_id,
                    n_items=0,
                    encode_seconds=0.0,
                    decode_seconds=0.0,
                    bytes_in=0,
                    bytes_out=0,
                    metrics={},
                    codebook_utilisation=None,
                    error=traceback.format_exc(limit=3),
                )
            )

    elapsed_s = time.perf_counter() - t0_wall
    aggregate = _aggregate(per_shot, config.metrics, elapsed_s)
    return BenchResult(
        config=config,
        per_shot=tuple(per_shot),
        aggregate=aggregate,
        elapsed_s=elapsed_s,
    )


# ---------------------------------------------------------------------------
# Aggregation helper
# ---------------------------------------------------------------------------


def _aggregate(
    per_shot: list[PerShotResult],
    metric_names: tuple[str, ...],
    elapsed_s: float,
) -> dict[str, float]:
    """Compute aggregate statistics across successful shots."""
    successful = [s for s in per_shot if s.error is None]
    if not successful:
        return {
            "n_shots_ok": 0.0,
            "n_shots_err": float(len(per_shot)),
            "throughput_items_per_s": 0.0,
            "total_bytes_in": 0.0,
            "total_bytes_out": 0.0,
            "compression_ratio": float("nan"),
            "mean_encode_s": 0.0,
            "mean_decode_s": 0.0,
        }

    total_items = sum(s.n_items for s in successful)
    total_bytes_in = sum(s.bytes_in for s in successful)
    total_bytes_out = sum(s.bytes_out for s in successful)

    tput = float(total_items) / elapsed_s if elapsed_s > 0 else 0.0
    cr = (
        float(total_bytes_in) / float(total_bytes_out)
        if total_bytes_out > 0
        else float("nan")
    )
    enc_mean = float(np.mean([s.encode_seconds for s in successful]))
    dec_mean = float(np.mean([s.decode_seconds for s in successful]))
    agg: dict[str, float] = {
        "n_shots_ok": float(len(successful)),
        "n_shots_err": float(len(per_shot) - len(successful)),
        "throughput_items_per_s": tput,
        "total_bytes_in": float(total_bytes_in),
        "total_bytes_out": float(total_bytes_out),
        "compression_ratio": cr,
        "mean_encode_s": enc_mean,
        "mean_decode_s": dec_mean,
    }

    for m in metric_names:
        vals = [
            s.metrics[m]
            for s in successful
            if m in s.metrics and np.isfinite(s.metrics[m])
        ]
        agg[f"mean_{m}"] = float(np.mean(vals)) if vals else float("nan")

    utils = [
        s.codebook_utilisation for s in successful if s.codebook_utilisation is not None
    ]
    if utils:
        agg["mean_codebook_utilisation"] = float(np.mean(utils))

    return agg
