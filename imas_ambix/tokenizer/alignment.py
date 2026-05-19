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
