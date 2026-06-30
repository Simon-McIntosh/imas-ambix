"""Signal calibration — corpus-wide per-channel statistics.

Accumulates mean, std, min, max, and approximate quantiles across a
collection of shot Zarr stores.  Uses Welford's online algorithm for
mean/std to avoid materialising the entire corpus in memory.

Typical usage::

    from pathlib import Path
    from imas_ambix.calibration.signals import compute_signal_calibration

    shots = [Path(f"/work/.../shots/{s}.zarr") for s in shot_ids]
    cal = compute_signal_calibration(shots, "summary")
    # cal is a dict[str, ChannelCalibration]
"""

from __future__ import annotations

import contextlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChannelCalibration:
    """Corpus-wide statistics for a single 1-D time-indexed channel.

    Attributes
    ----------
    name:
        Channel / variable name within the group.
    mean:
        Population mean over all non-NaN samples across all shots.
    std:
        Population std (Bessel-uncorrected) over all non-NaN samples.
    min_value:
        Global minimum observed value (NaN excluded).
    max_value:
        Global maximum observed value (NaN excluded).
    q01:
        Approximate 1st percentile (weighted mean of per-shot 1st percentiles).
    q50:
        Approximate 50th percentile (median).
    q99:
        Approximate 99th percentile.
    n_samples:
        Total number of non-NaN samples counted.
    n_shots:
        Number of shots that contained this channel.
    """

    name: str
    mean: float
    std: float
    min_value: float
    max_value: float
    q01: float
    q50: float
    q99: float
    n_samples: int
    n_shots: int

    def is_physical(self) -> bool:
        """True if this channel's statistics are usable for absolute mode.

        A real instrument channel has finite statistics whose spread is
        consistent with its quantiles.  Some MAST raw streams instead carry a
        dead/saturated detector: a stuck OVERFLOW SENTINEL — the same enormous
        value (~1e17–1e29) reported on (nearly) every sample, so its 1st and
        99th percentiles collapse to a single constant (``q01 == q99``) while
        the corpus ``std`` is simultaneously HUGE (rare overflow excursions
        across shots blow up the variance).  That ``q01 == q99`` AND large-std
        signature is the unambiguous fingerprint of a dead detector and never
        occurs for a real signal:

        - a real *varying* channel (Dα ~1e22, line density ~1e19, gas counts
          ~1e21, neutron rate ~1e13 — all legitimately LARGE SI values) has
          ``q01 != q99``, so magnitude alone must NOT condemn it;
        - a real *constant* channel (e.g. an all-zero / flat channel) has
          ``q01 == q99`` but ``std ≈ 0`` — consistent, hence physical (the
          downstream ``std<=0 -> 1`` guard handles it).

        Standardising against a dead sentinel propagates garbage through the
        absolute tokeniser, so the consumer treats a non-physical channel as if
        it had no calibration (per-window/per-shot fallback) rather than
        encoding the sentinel.

        Non-physical when ANY of:

        - ``mean``/``std``/``min_value``/``max_value`` is non-finite (NaN/inf);
        - the distribution is a degenerate constant (``q01 == q99``) yet its
          ``std`` is not negligible relative to that constant — the stuck-value
          overflow fingerprint (``std`` orders of magnitude above |q50|).
        """
        import math

        vals = (self.mean, self.std, self.min_value, self.max_value)
        if any(not math.isfinite(float(v)) for v in vals):
            return False
        q01, q99, q50, std = (
            float(self.q01),
            float(self.q99),
            float(self.q50),
            float(self.std),
        )
        if not (math.isfinite(q01) and math.isfinite(q99)):
            # No usable quantiles — fall back on a non-finite std/mean check
            # only (already passed above), so treat as physical.
            return True
        # Stuck-overflow fingerprint: collapsed quantiles (constant value) but a
        # std that is NOT consistent with a constant — i.e. std dwarfs |q50|.
        # A genuine constant (all-zero / flat) has q01==q99 with std≈0 and is
        # physical; a dead sentinel has q01==q99 with std >> |q50|.
        # A genuine constant channel (all-zero / flat) has std≈0 (<= |q50|);
        # a dead sentinel has std MANY times |q50| (measured ~44-49x on the
        # MAST dead SXR chords).  Requiring std > |q50| cleanly separates the
        # two with ~40x of margin, and never flags a real (varying or constant)
        # channel.
        degenerate = q01 == q99
        std_inconsistent = std > (abs(q50) + 1e-12)
        return not (degenerate and std_inconsistent)


# ---------------------------------------------------------------------------
# Welford accumulator (streaming mean / variance)
# ---------------------------------------------------------------------------


class _WelfordAccumulator:
    """Online mean + M2 aggregator (Chan's parallel form for merging chunks).

    References
    ----------
    - Welford (1962), "Note on a Method for Calculating Corrected Sums of
      Squares and Products".
    - Chan et al. (1979), parallel algorithm for combining two accumulators.
    """

    __slots__ = ("_n", "_mean", "_m2")

    def __init__(self) -> None:
        self._n: int = 0
        self._mean: float = 0.0
        self._m2: float = 0.0  # sum of squared deviations from mean

    def update(self, values: np.ndarray) -> None:
        """Incorporate a 1-D array of finite values (NaNs must be stripped first)."""
        import numpy as np

        arr = np.asarray(values, dtype=np.float64).ravel()
        arr = arr[np.isfinite(arr)]
        n_b = int(arr.size)
        if n_b == 0:
            return

        # Chunk mean and M2
        mean_b = float(arr.mean())
        m2_b = float(np.sum((arr - mean_b) ** 2))

        # Parallel merge (Chan 1979)
        n_a = self._n
        n_ab = n_a + n_b
        delta = mean_b - self._mean
        self._mean = (n_a * self._mean + n_b * mean_b) / n_ab
        self._m2 += m2_b + delta**2 * n_a * n_b / n_ab
        self._n = n_ab

    @property
    def n(self) -> int:
        return self._n

    @property
    def mean(self) -> float:
        return self._mean if self._n > 0 else float("nan")

    @property
    def std(self) -> float:
        if self._n < 1:
            return float("nan")
        return float(self._m2 / self._n) ** 0.5


# ---------------------------------------------------------------------------
# Per-shot loader
# ---------------------------------------------------------------------------


def _load_shot_channel_data(
    shot_path: Path,
    group: str,
    channels: tuple[str, ...] | None,
    time_dim: str,
) -> dict[str, np.ndarray]:
    """Open one shot Zarr and return finite 1-D arrays per channel.

    Returns an empty dict on any IO error (bad shot silently skipped).
    """
    import numpy as np
    import xarray as xr

    result: dict[str, np.ndarray] = {}
    try:
        ds: xr.Dataset = xr.open_zarr(str(shot_path), group=group, consolidated=False)
    except Exception:
        return result

    for name, var in ds.data_vars.items():
        if time_dim not in var.dims or var.ndim != 1:
            continue
        if channels is not None and name not in channels:
            continue
        arr = np.asarray(var.values, dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        if finite.size:
            result[str(name)] = finite
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_signal_calibration(
    shot_paths: list[Path],
    group: str,
    *,
    channels: tuple[str, ...] | None = None,
    time_dim: str = "time",
    max_workers: int = 4,
) -> dict[str, ChannelCalibration]:
    """Accumulate per-channel corpus statistics for a Zarr group.

    Parameters
    ----------
    shot_paths:
        List of paths to individual shot ``.zarr`` stores.
    group:
        Zarr group name to open within each shot (e.g. ``"summary"``).
    channels:
        Restrict to these channel names. ``None`` means all 1-D
        time-indexed variables found in the first reachable shot.
    time_dim:
        Name of the time dimension. Defaults to ``"time"``.
    max_workers:
        Thread-pool concurrency for parallel shot loading.

    Returns
    -------
    dict[str, ChannelCalibration]
        Keyed by channel name.  Channels present in *zero* shots are
        excluded from the output.  A channel present in some shots but
        all-NaN in those shots gets ``n_samples=0`` with NaN statistics.
    """
    import numpy as np

    # --- Accumulator state ------------------------------------------------
    # Keyed by channel name.
    welford: dict[str, _WelfordAccumulator] = {}
    global_min: dict[str, float] = {}
    global_max: dict[str, float] = {}
    # Per-shot quantile lists for weighted approximation
    q01_acc: dict[str, list[tuple[float, int]]] = {}  # (quantile, n_samples) pairs
    q50_acc: dict[str, list[tuple[float, int]]] = {}
    q99_acc: dict[str, list[tuple[float, int]]] = {}
    n_shots_seen: dict[str, int] = {}

    def _accumulate(shot_data: dict[str, np.ndarray]) -> None:
        for ch, arr in shot_data.items():
            if ch not in welford:
                welford[ch] = _WelfordAccumulator()
                global_min[ch] = float("inf")
                global_max[ch] = float("-inf")
                q01_acc[ch] = []
                q50_acc[ch] = []
                q99_acc[ch] = []
                n_shots_seen[ch] = 0

            n = int(arr.size)
            if n == 0:
                continue

            welford[ch].update(arr)
            global_min[ch] = min(global_min[ch], float(arr.min()))
            global_max[ch] = max(global_max[ch], float(arr.max()))
            q01_acc[ch].append((float(np.percentile(arr, 1)), n))
            q50_acc[ch].append((float(np.percentile(arr, 50)), n))
            q99_acc[ch].append((float(np.percentile(arr, 99)), n))
            n_shots_seen[ch] += 1

    # --- Parallel loading -------------------------------------------------
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_load_shot_channel_data, p, group, channels, time_dim): p
            for p in shot_paths
        }
        for fut in as_completed(futures):
            with contextlib.suppress(Exception):
                _accumulate(fut.result())

    # --- Assemble results -------------------------------------------------

    def _weighted_mean_quantile(pairs: list[tuple[float, int]]) -> float:
        """Weighted mean of per-shot quantiles (weight = n_samples)."""
        if not pairs:
            return float("nan")
        total_n = sum(n for _, n in pairs)
        if total_n == 0:
            return float("nan")
        return sum(q * n for q, n in pairs) / total_n

    out: dict[str, ChannelCalibration] = {}
    for ch in welford:
        acc = welford[ch]
        out[ch] = ChannelCalibration(
            name=ch,
            mean=acc.mean,
            std=acc.std,
            min_value=global_min.get(ch, float("nan")),
            max_value=global_max.get(ch, float("nan")),
            q01=_weighted_mean_quantile(q01_acc.get(ch, [])),
            q50=_weighted_mean_quantile(q50_acc.get(ch, [])),
            q99=_weighted_mean_quantile(q99_acc.get(ch, [])),
            n_samples=acc.n,
            n_shots=n_shots_seen.get(ch, 0),
        )
    return out
