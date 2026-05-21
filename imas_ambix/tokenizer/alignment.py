"""Time-grid alignment helpers.

The world model trains on tokens at a **fixed model time grid** (the
v0 default is 100 Hz — every 10 ms). Raw FAIR-MAST data carries
heterogeneous time bases:

- level-2 magnetics: 4 kHz interpolated grid (2,065 samples in shot 11766)
- level-2 equilibrium: 5 ms cadence (83 samples)
- level-2 summary, pf_active, pulse_schedule: 1 ms cadence (~1,652 samples)
- level-1 rbb/rba camera: native cadence, typically 100-400 Hz
- level-1 raw magnetics (xmo/OMAHA): up to MHz cadence

The aligner re-samples all of these onto a single uniform grid so the
downstream tokenizer can operate on time-aligned blocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import xarray as xr


MODEL_HZ_DEFAULT = 100.0
"""Default model time grid frequency (Hz) — every 10 ms."""


@dataclass(frozen=True)
class TimeGrid:
    """A uniform model time grid for a shot."""

    t_start: float
    t_end: float
    hz: float

    def as_array(self) -> np.ndarray:
        import numpy as np

        n = int(round((self.t_end - self.t_start) * self.hz)) + 1
        return self.t_start + (np.arange(n) / self.hz).astype(float)

    def __len__(self) -> int:
        return int(round((self.t_end - self.t_start) * self.hz)) + 1


def shot_time_window(*time_arrays: np.ndarray) -> tuple[float, float]:
    """Return the largest common ``(t_start, t_end)`` across input axes.

    Used to define the model grid bounds — the intersection of every
    diagnostic's time axis, so resampling never extrapolates past the
    data.
    """
    import numpy as np

    starts = []
    ends = []
    for arr in time_arrays:
        a = np.asarray(arr)
        if a.size == 0:
            continue
        starts.append(float(a.min()))
        ends.append(float(a.max()))
    if not starts:
        raise ValueError("no non-empty time arrays")
    return (max(starts), min(ends))


def resample_to_grid(
    ds: xr.Dataset, grid: TimeGrid, time_dim: str = "time"
) -> xr.Dataset:
    """Linearly interpolate every 1-D `time`-axis variable onto ``grid``.

    Variables that lack ``time_dim`` are passed through unchanged.
    Variables on a different time dimension (e.g. ``time_saddle`` for
    high-rate magnetics) are dropped — callers should select the right
    time axis before calling.
    """

    t_target = grid.as_array()
    out_vars = {}
    for name, var in ds.data_vars.items():
        if time_dim not in var.dims:
            out_vars[name] = var
            continue
        if time_dim not in ds.coords:
            continue
        # Interp along the time axis only — preserve other dims.
        out_vars[name] = var.interp({time_dim: t_target}, method="linear")
    import xarray as xr

    return xr.Dataset(out_vars, coords={time_dim: t_target})


def align_frames_signals(
    frames: np.ndarray | None,
    signals: xr.Dataset,
    model_hz: float,
    time_dim: str = "time",
) -> tuple[np.ndarray | None, xr.Dataset]:
    """Resample frames and signals onto the shared model time grid.

    This function factors out the resampling / index-selection logic used
    by :class:`~imas_ambix.tokenizer.multimodal.ShotTokenizer` when
    ``enforce_alignment=True``, and is exposed here for reuse by callers
    that want to pre-process data before passing it in.

    Time grid derivation
    --------------------
    The authoritative time window comes from the signals ``time_dim``
    coordinate when present.  If the coordinate is absent (unusual but
    possible for synthetic datasets), a synthetic ``0 … N/hz`` grid is
    derived from the first time-axis dimension of ``signals`` (falling
    back to ``frames.shape[0]`` when signals has no time dimension either).

    Signal resampling
    -----------------
    ``signals`` is linearly interpolated onto the model grid via
    :func:`resample_to_grid`.  Variables without a ``time_dim`` dimension
    pass through unchanged.

    Frame sub-sampling
    ------------------
    If ``frames`` is not ``None`` and has **more** time steps than the
    model grid expects, evenly-spaced indices are selected to match the
    grid length.  If ``frames`` has **fewer** steps the array is returned
    as-is — the encoder is responsible for padding (e.g. repeat-last).

    Edge cases
    ----------
    - ``frames is None`` → returned as ``None``; only signals are aligned.
    - ``signals`` is empty (no data-vars) → resampling is a no-op but the
      time coordinate is still updated.
    - Single-step signals (degenerate shot) → the grid degenerates to a
      one-element array; interpolation is still numerically valid.
    - ``t_start >= t_end`` (zero-duration window) → ``TimeGrid`` falls back
      to a single grid point at ``t_start``.

    Args:
        frames: ``(T, H, W)`` or ``(T, H, W, C)`` frame array, or ``None``.
        signals: ``xr.Dataset`` containing 1-D signal variables.
        model_hz: Model time-grid frequency in Hz (e.g. ``100.0``).
        time_dim: Name of the time coordinate / dimension in ``signals``.

    Returns:
        ``(frames_aligned, signals_aligned)`` — frames may be a sub-sampled
        view (not a copy when index selection is used) and signals are a new
        :class:`xr.Dataset` on the model grid.
    """
    import numpy as np
    import xarray as xr

    # --- Determine time window from signals coords ---------------------
    if time_dim in (signals.coords or {}):
        t_arr = np.asarray(signals.coords[time_dim])
        if t_arr.size >= 2:
            t_start = float(t_arr.min())
            t_end = float(t_arr.max())
        elif t_arr.size == 1:
            t_start = t_end = float(t_arr[0])
        else:
            # Empty coord — fall through to synthetic grid
            t_arr = None
            t_start = t_end = 0.0
    else:
        t_arr = None
        t_start = t_end = 0.0

    if t_arr is None:
        # Synthetic grid: 0 … N/hz using whichever axis has a time dimension.
        n_sig = signals.sizes.get(time_dim, 0)
        n_frames = frames.shape[0] if frames is not None else 0
        n_steps = n_sig if n_sig > 0 else n_frames
        t_end = max(n_steps - 1, 0) / model_hz if n_steps > 0 else 0.0
        t_start = 0.0

    grid = TimeGrid(t_start=t_start, t_end=t_end, hz=model_hz)
    n_grid = len(grid)

    # --- Resample signals ----------------------------------------------
    signals_out = resample_to_grid(signals, grid, time_dim=time_dim)

    # --- Sub-sample frames ---------------------------------------------
    frames_out: np.ndarray | None
    if frames is None:
        frames_out = None
    else:
        n_frame_steps = frames.shape[0]
        if n_frame_steps > n_grid:
            # Evenly-spaced index selection to match the grid length.
            indices = np.round(
                np.linspace(0, n_frame_steps - 1, n_grid)
            ).astype(int)
            frames_out = frames[indices]
        else:
            # Fewer or equal steps — pass through; encoder handles padding.
            frames_out = frames

    return frames_out, signals_out
