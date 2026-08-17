"""Typed access to the one-row, nested-array challenge Parquet schema."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow.parquet as pq

if TYPE_CHECKING:
    from pathlib import Path

_COIL_FIELDS = (
    "coil_name",
    "coil_input_column",
    "coil_R",
    "coil_Z",
    "coil_width",
    "coil_height",
    "coil_angle1",
    "coil_angle2",
)
_CHORD_FIELDS = ("thomson_chord_name", "thomson_chord_R", "thomson_chord_Z")
_EFIT_SCALARS = (
    "efit_beta_n",
    "efit_li",
    "efit_q95",
    "efit_r_axis",
    "efit_z_axis",
    "efit_lcfs_n",
)


@dataclass(frozen=True)
class SignalSeries:
    time_ms: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class ThomsonProfile:
    time_ms: np.ndarray
    temperature_ev: np.ndarray
    density_m3: np.ndarray
    spatial_m: np.ndarray


@dataclass(frozen=True)
class EfitLabels:
    time_ms: np.ndarray
    psirz: np.ndarray
    grid_r_m: np.ndarray
    grid_z_m: np.ndarray
    scalars: dict[str, np.ndarray]


@dataclass(frozen=True)
class ChallengeShot:
    source: str
    actuators: dict[str, SignalSeries]
    thomson: dict[str, ThomsonProfile]
    coil_geometry: dict[str, np.ndarray]
    chord_geometry: dict[str, np.ndarray]
    labels: EfitLabels


def _array(table: Any, name: str, *, string: bool = False) -> np.ndarray:
    dtype: Any = str if string else float
    return np.asarray(table[name][0].as_py(), dtype=dtype)


def _series(table: Any, name: str, time_name: str) -> SignalSeries:
    return SignalSeries(
        time_ms=_array(table, time_name), values=_array(table, name)
    )


def load_shot(path: str | Path, *, validate: bool = True) -> ChallengeShot:
    """Load one shot, preserving each signal's native time base and geometry."""

    table = pq.read_table(path)
    if table.num_rows != 1:
        raise ValueError(f"expected one row per shot, found {table.num_rows} in {path}")
    magnetics_time = "magnetics_time"
    actuator_names = sorted(
        name
        for name in table.column_names
        if name.startswith("magnetics_")
        and not name.endswith(("_time", "_times"))
        and name != "magnetics_dsep"
    )
    actuators: dict[str, SignalSeries] = {}
    for name in actuator_names:
        time_name = (
            "magnetics_plasma_current_times"
            if name == "magnetics_plasma_current"
            else magnetics_time
        )
        actuators[name] = _series(table, name, time_name)

    thomson = {
        "core": ThomsonProfile(
            time_ms=_array(table, "thomson_core_times"),
            temperature_ev=_array(table, "thomson_core_Te"),
            density_m3=_array(table, "thomson_core_ne"),
            spatial_m=_array(table, "thomson_core_R"),
        ),
        "edge": ThomsonProfile(
            time_ms=_array(table, "thomson_edge_times"),
            temperature_ev=_array(table, "thomson_edge_Te"),
            density_m3=_array(table, "thomson_edge_ne"),
            spatial_m=_array(table, "thomson_edge_spatial"),
        ),
    }
    labels = EfitLabels(
        time_ms=_array(table, "efit_times"),
        psirz=_array(table, "efit_psirz"),
        grid_r_m=_array(table, "efit_grid_R"),
        grid_z_m=_array(table, "efit_grid_Z"),
        scalars={name: _array(table, name) for name in _EFIT_SCALARS},
    )
    shot = ChallengeShot(
        source=str(table["source"][0].as_py()),
        actuators=actuators,
        thomson=thomson,
        coil_geometry={
            name: _array(
                table,
                name,
                string=name in {"coil_name", "coil_input_column"},
            )
            for name in _COIL_FIELDS
        },
        chord_geometry={
            name: _array(table, name, string=name == "thomson_chord_name")
            for name in _CHORD_FIELDS
        },
        labels=labels,
    )
    if validate:
        validate_loaded_shot(shot)
    return shot


def validate_loaded_shot(shot: ChallengeShot) -> None:
    """Enforce cross-field lengths and the released 65-by-65 label contract."""

    frame_count = len(shot.labels.time_ms)
    if frame_count == 0:
        raise ValueError("shot has no labeled frames")
    if shot.labels.psirz.shape != (frame_count, 65, 65):
        raise ValueError(f"invalid efit_psirz shape {shot.labels.psirz.shape}")
    if shot.labels.grid_r_m.shape != (65,) or shot.labels.grid_z_m.shape != (65,):
        raise ValueError("EFIT grids must each contain 65 coordinates")
    for name, values in shot.labels.scalars.items():
        if values.shape != (frame_count,):
            raise ValueError(f"{name} shape {values.shape} does not match efit_times")
    for name, series in shot.actuators.items():
        if series.time_ms.ndim != 1 or series.values.shape != series.time_ms.shape:
            raise ValueError(f"{name} does not match its native time base")
    for name, profile in shot.thomson.items():
        expected = (len(profile.time_ms), len(profile.spatial_m))
        if (
            profile.temperature_ev.shape != expected
            or profile.density_m3.shape != expected
        ):
            message = f"Thomson {name} profile shape does not match time and space"
            raise ValueError(message)
    coil_lengths = {len(values) for values in shot.coil_geometry.values()}
    chord_lengths = {len(values) for values in shot.chord_geometry.values()}
    if len(coil_lengths) != 1 or len(chord_lengths) != 1:
        raise ValueError("geometry field lengths disagree")
    if shot.source == "DIII-D" and coil_lengths != {19}:
        message = f"DIII-D must contain 19 coil geometry rows, found {coil_lengths}"
        raise ValueError(message)


def validate_shot_schema(path: str | Path) -> ChallengeShot:
    """Load and validate one corpus object, returning the typed shot."""

    return load_shot(path, validate=True)
