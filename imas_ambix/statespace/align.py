"""Time-alignment helpers for multi-family statespace data.

Thin wrapper around :mod:`imas_ambix.tokenizer.alignment` for use in the
statespace pipeline.  Adds per-family native-rate documentation and
family-specific alignment choices.

Native rates (measured from MAST level-1 data)
----------------------------------------------
Family          | Primary group | Native rate        | Notes
----------------|---------------|--------------------|----------------------------------
magnetics       | ama/amb/amc   | ~4 kHz             | Interpolated 250 µs grid
magnetics_raw   | xmo (OMAHA)   | 1–4 MHz            | Sub-sample only, NO interp
xim (Da raw)    | xim           | ~4-10 kHz          | High-rate; alias risk
ada (Da proc)   | ada           | ~4 kHz             | Lower-rate after processing
ane (density)   | ane           | ~10 Hz – 1 kHz     | Interferometer, varies per shot
thomson         | atm/ayc/aye   | ~1–50 Hz           | YAG-pulsed, sparse in time
bolometer       | abm           | ~4 kHz             | 24-chord Abel
soft_xray       | xsx           | ~100 Hz – 4 kHz    | Array, rate varies
camera rbb      | rbb           | 100–400 Hz         | Visible, frame-by-frame
nbi             | anb           | ~1 kHz             | NBI power signals
pulse_schedule  | xdc           | ~1 kHz             | Programmed control waveforms

Alignment choice (v0 default: 100 Hz model grid)
-------------------------------------------------
- HIGH-RATE diagnostics (magnetics, xim, magnetics_raw): DECIMATE by taking
  every Nth sample (nearest neighbour) or linear interpolation to model grid.
  WARNING: xim/Dα at 4–10 kHz downsampled to 100 Hz WILL ALIAS ELM spikes.
  Document the aliasing risk and flag to orchestrator for target-grid choice.
  Recommendation: for Dα as prediction target, raise the model grid to 1 kHz
  or use a per-window max-pool to preserve ELM peak amplitude.
- LOW-RATE diagnostics (thomson, ane): LINEAR INTERPOLATION to model grid
  (sparse to dense). Extrapolation is clipped to the data range.
- CAMERA (rbb): nearest-frame assignment at model grid points.

Control / conditioning stream (xdc) is kept separate from input diagnostics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import xarray as xr

from imas_ambix.tokenizer.alignment import (  # re-export
    MODEL_HZ_DEFAULT,
    TimeGrid,
    align_frames_signals,
    resample_to_grid,
    shot_time_window,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-family native rate documentation
# ---------------------------------------------------------------------------

# Documented native sample rates (Hz).  These are empirical estimates from
# MAST level-1 data; actual rates vary by shot and era.
FAMILY_NATIVE_RATE_HZ: dict[str, tuple[float, float, str]] = {
    # (min_hz, max_hz, alignment_method)
    "magnetics": (4000.0, 4000.0, "linear_interp"),
    "magnetics_raw": (1_000_000.0, 4_000_000.0, "nearest_neighbour"),
    "dalpha": (4000.0, 10_000.0, "linear_interp"),  # alias risk! see module docstring
    "dalpha_analysis": (100.0, 4000.0, "linear_interp"),
    "interferometer": (10.0, 1000.0, "linear_interp"),
    "thomson_scattering": (1.0, 50.0, "linear_interp"),
    "bolometer": (4000.0, 4000.0, "linear_interp"),
    "soft_xray": (100.0, 4000.0, "linear_interp"),
    "hard_xray": (10.0, 100.0, "linear_interp"),
    "mse": (10.0, 100.0, "linear_interp"),
    "charge_exchange": (100.0, 1000.0, "linear_interp"),
    "langmuir": (1000.0, 10000.0, "linear_interp"),
    "neutron": (100.0, 1000.0, "linear_interp"),
    "camera_visible": (100.0, 400.0, "nearest_neighbour"),
    "camera_ir": (50.0, 200.0, "nearest_neighbour"),
    "nbi": (1000.0, 1000.0, "linear_interp"),
    "gas": (100.0, 1000.0, "linear_interp"),
    "pulse_schedule": (1000.0, 1000.0, "linear_interp"),
    "ir_analysis": (50.0, 200.0, "linear_interp"),
}

# Families where downsampling to a 100 Hz model grid will cause aliasing
ALIAS_RISK_FAMILIES: frozenset[str] = frozenset(
    {"magnetics", "magnetics_raw", "dalpha", "bolometer"}
)

ALIAS_RISK_NOTE = (
    "WARNING: Downsampling {family} (native {min_hz}–{max_hz} Hz) to "
    "{model_hz} Hz model grid aliases high-frequency structure "
    "(ELM spikes for dalpha, MHD modes for magnetics). "
    "For Dα as prediction target, consider raising model_hz to ≥1000 Hz "
    "or using per-window max-pool to preserve ELM peak amplitude."
)


@dataclass(frozen=True)
class FamilyAlignmentPlan:
    """Alignment plan for one diagnostic family.

    Attributes
    ----------
    family:
        Family name.
    native_rate_min, native_rate_max:
        Empirical native rate range (Hz).
    method:
        Alignment method: ``"linear_interp"`` or ``"nearest_neighbour"``.
    model_hz:
        Target model grid frequency.
    alias_risk:
        True when native rate >> model_hz and aliasing is likely.
    alias_note:
        Human-readable warning when alias_risk is True.
    """

    family: str
    native_rate_min: float
    native_rate_max: float
    method: str
    model_hz: float
    alias_risk: bool
    alias_note: str = ""

    def to_dict(self) -> dict:
        return {
            "family": self.family,
            "native_rate_min_hz": self.native_rate_min,
            "native_rate_max_hz": self.native_rate_max,
            "alignment_method": self.method,
            "model_hz": self.model_hz,
            "alias_risk": self.alias_risk,
            "alias_note": self.alias_note,
        }


def alignment_plan(
    families: list[str],
    model_hz: float = MODEL_HZ_DEFAULT,
) -> dict[str, FamilyAlignmentPlan]:
    """Return :class:`FamilyAlignmentPlan` for each family.

    Parameters
    ----------
    families:
        List of family names (from FAMILY_GROUPS keys in ``families.py``).
    model_hz:
        Target model grid frequency.

    Returns
    -------
    dict mapping family name → :class:`FamilyAlignmentPlan`.
    """
    plans: dict[str, FamilyAlignmentPlan] = {}
    for fam in families:
        if fam not in FAMILY_NATIVE_RATE_HZ:
            logger.debug("No native rate documented for family '%s' — skipping", fam)
            continue
        min_hz, max_hz, method = FAMILY_NATIVE_RATE_HZ[fam]
        alias = fam in ALIAS_RISK_FAMILIES and max_hz > 5 * model_hz
        note = ""
        if alias:
            note = ALIAS_RISK_NOTE.format(
                family=fam, min_hz=min_hz, max_hz=max_hz, model_hz=model_hz
            )
        plans[fam] = FamilyAlignmentPlan(
            family=fam,
            native_rate_min=min_hz,
            native_rate_max=max_hz,
            method=method,
            model_hz=model_hz,
            alias_risk=alias,
            alias_note=note,
        )
    return plans


def align_family_dataset(
    ds: xr.Dataset,
    family: str,
    model_hz: float = MODEL_HZ_DEFAULT,
    time_dim: str = "time",
) -> xr.Dataset:
    """Align one family's xr.Dataset to the model grid.

    Wraps :func:`~imas_ambix.tokenizer.alignment.resample_to_grid`.

    Parameters
    ----------
    ds:
        xarray Dataset with a ``time`` coordinate.
    family:
        Family name — used to look up the native rate and determine
        whether alias warnings should be emitted.
    model_hz:
        Target model grid frequency.
    time_dim:
        Name of the time dimension in *ds*.

    Returns
    -------
    xr.Dataset resampled to the model grid.
    """

    if time_dim not in ds.coords:
        logger.warning(
            "Family '%s' dataset has no '%s' coordinate — returning as-is",
            family,
            time_dim,
        )
        return ds

    t_arr = ds.coords[time_dim].values
    if t_arr.size < 2:
        return ds

    t_start = float(t_arr.min())
    t_end = float(t_arr.max())
    grid = TimeGrid(t_start=t_start, t_end=t_end, hz=model_hz)

    if family in ALIAS_RISK_FAMILIES:
        native_rate = FAMILY_NATIVE_RATE_HZ.get(family, (0, 0, ""))[1]
        if native_rate > 5 * model_hz:
            logger.warning(
                "Aliasing risk for '%s': native %g Hz → model %g Hz",
                family,
                native_rate,
                model_hz,
            )

    return resample_to_grid(ds, grid, time_dim=time_dim)


__all__ = [
    "MODEL_HZ_DEFAULT",
    "TimeGrid",
    "align_frames_signals",
    "resample_to_grid",
    "shot_time_window",
    "FAMILY_NATIVE_RATE_HZ",
    "ALIAS_RISK_FAMILIES",
    "FamilyAlignmentPlan",
    "alignment_plan",
    "align_family_dataset",
]
