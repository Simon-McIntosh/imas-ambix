"""Typed access to the one-row, nested-array challenge Parquet schema."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow.parquet as pq

from imas_alambic.cocos import CANONICAL_COCOS

from .convention import DIIID_CONVENTION

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
    lcfs_r_m: np.ndarray
    lcfs_z_m: np.ndarray
    scalars: dict[str, np.ndarray]
    cocos: int


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
    return SignalSeries(time_ms=_array(table, time_name), values=_array(table, name))


_LABEL_COLUMNS = (
    "source",
    "efit_times",
    "efit_psirz",
    "efit_grid_R",
    "efit_grid_Z",
    "efit_lcfs_r",
    "efit_lcfs_z",
    *_EFIT_SCALARS,
    "magnetics_plasma_current",
    "magnetics_plasma_current_times",
    "magnetics_bcoil",
    "magnetics_time",
)


def _build_labels(table: Any) -> EfitLabels:
    """Assemble the canonical EfitLabels record from a shot's columns.

    The magnetics channels the convention audit reads are resampled onto the
    EFIT time base, so the audit sees one record per equilibrium rather than
    two native-rate series.
    """

    source = str(table["source"][0].as_py())
    efit_times = _array(table, "efit_times")
    source_scalars = {name: _array(table, name) for name in _EFIT_SCALARS}
    plasma_current = _series(
        table, "magnetics_plasma_current", "magnetics_plasma_current_times"
    )
    bcoil = _series(table, "magnetics_bcoil", "magnetics_time")
    if source == "DIII-D":
        psirz = DIIID_CONVENTION.canonical_flux(_array(table, "efit_psirz"))
        source_scalars["efit_q95"] = DIIID_CONVENTION.canonical_q(
            source_scalars["efit_q95"]
        )
        plasma_current = SignalSeries(
            time_ms=plasma_current.time_ms,
            values=DIIID_CONVENTION.canonical_plasma_current(plasma_current.values),
        )
        bcoil = SignalSeries(
            time_ms=bcoil.time_ms,
            values=DIIID_CONVENTION.canonical_toroidal_field(bcoil.values),
        )
    else:
        psirz = _array(table, "efit_psirz")
    source_scalars["magnetics_plasma_current"] = np.interp(
        efit_times, plasma_current.time_ms, plasma_current.values
    )
    source_scalars["magnetics_bcoil"] = np.interp(
        efit_times, bcoil.time_ms, bcoil.values
    )
    return EfitLabels(
        time_ms=efit_times,
        psirz=psirz,
        grid_r_m=_array(table, "efit_grid_R"),
        grid_z_m=_array(table, "efit_grid_Z"),
        lcfs_r_m=_array(table, "efit_lcfs_r"),
        lcfs_z_m=_array(table, "efit_lcfs_z"),
        scalars=source_scalars,
        cocos=CANONICAL_COCOS,
    )


def load_labels(path: str | Path) -> EfitLabels:
    """Read only the equilibrium label columns into one canonical record.

    The convention audit reads :class:`EfitLabels` alone, so this avoids
    materialising the full :class:`ChallengeShot` for every audited shot.
    """

    table = pq.read_table(path, columns=list(_LABEL_COLUMNS))
    if table.num_rows != 1:
        raise ValueError(f"expected one row per shot, found {table.num_rows} in {path}")
    return _build_labels(table)


def load_geqdsk(path: str | Path, *, time_ms: float | None = None) -> EfitLabels:
    """Read one G-EQDSK file into a canonical :class:`EfitLabels` record.

    The file is read through the ``eqdsk`` package's ``EQDSKInterface``, which
    owns the format and its COCOS handling, so this loader carries no G-EQDSK
    reader of its own.  EFIT writes its G-EQDSK with psi per radian
    (``e_Bp = 0``), so the file is read as its declared COCOS 1 and the record
    keeps the stored psi unchanged rather than converting it; the poloidal-flux
    relation is then scored from the file's own values.  A G-EQDSK carries one
    equilibrium snapshot and no time base, so the record holds a single frame:
    at ``time_ms`` when it is given, and otherwise at the snapshot time the
    file's own header declares.
    """

    from eqdsk import EQDSKInterface  # noqa: PLC0415

    instance = EQDSKInterface.from_file(
        path, clockwise_phi=False, volt_seconds_per_radian=True, to_cocos=None
    )
    radial = np.asarray(instance.x, dtype=np.float64)
    vertical = np.asarray(instance.z, dtype=np.float64)
    flux = np.asarray(instance.psi, dtype=np.float64)
    if time_ms is None:
        time_ms = geqdsk_declared_time_ms(path)
        if time_ms is None:
            raise ValueError(
                f"{path} declares no snapshot time in its header; pass time_ms"
            )
    if flux.shape != (radial.size, vertical.size):
        raise ValueError(
            f"G-EQDSK flux grid {flux.shape} does not match its "
            f"{radial.size}x{vertical.size} coordinate vectors"
        )
    safety_factor = _geqdsk_safety_factor(instance)
    return EfitLabels(
        time_ms=np.asarray([float(time_ms)], dtype=np.float64),
        psirz=np.transpose(flux, (1, 0))[np.newaxis, :, :],
        grid_r_m=radial,
        grid_z_m=vertical,
        lcfs_r_m=np.asarray(instance.xbdry, dtype=np.float64)[np.newaxis, :],
        lcfs_z_m=np.asarray(instance.zbdry, dtype=np.float64)[np.newaxis, :],
        scalars={
            "efit_q95": np.asarray([safety_factor], dtype=np.float64),
            "efit_r_axis": np.asarray([instance.xmag], dtype=np.float64),
            "efit_z_axis": np.asarray([instance.zmag], dtype=np.float64),
            "magnetics_bcoil": np.asarray([instance.bcentre], dtype=np.float64),
        },
        cocos=int(instance.cocos.index),
    )


def geqdsk_declared_time_ms(path: str | Path) -> float | None:
    """Return the snapshot time the file's comment header declares, if any.

    A G-EQDSK has no time field in the format itself, so a writer that knows
    when its snapshot belongs records it in the free-form header line —
    ``4000ms`` for the four-second snapshot.  Absent that token there is no
    declared time and the caller must supply one.
    """

    with open(path, encoding="latin-1") as handle:
        header = handle.readline()
    match = re.search(r"(\d+(?:\.\d+)?)\s*ms(?![A-Za-z])", header)
    return None if match is None else float(match.group(1))


def _geqdsk_safety_factor(instance: object) -> float:
    """Return q at 95 percent of the normalised flux from a G-EQDSK record."""

    qpsi = getattr(instance, "qpsi", None)
    psinorm = getattr(instance, "psinorm", None)
    if qpsi is None or psinorm is None:
        raise ValueError("G-EQDSK carries no q profile to read q95 from")
    return float(np.interp(0.95, np.asarray(psinorm), np.asarray(qpsi)))


def load_shot(path: str | Path, *, validate: bool = True) -> ChallengeShot:
    """Load one shot in the canonical convention with native time bases."""

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
    source = str(table["source"][0].as_py())
    if source == "DIII-D":
        for name, series in tuple(actuators.items()):
            if name == "magnetics_plasma_current":
                values = DIIID_CONVENTION.canonical_plasma_current(series.values)
            elif name == "magnetics_bcoil":
                values = DIIID_CONVENTION.canonical_toroidal_field(series.values)
            else:
                continue
            actuators[name] = SignalSeries(time_ms=series.time_ms, values=values)
    labels = _build_labels(table)
    shot = ChallengeShot(
        source=source,
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
    if shot.labels.cocos != CANONICAL_COCOS:
        raise ValueError(f"labels must be canonical COCOS {CANONICAL_COCOS}")
    if shot.labels.grid_r_m.shape != (65,) or shot.labels.grid_z_m.shape != (65,):
        raise ValueError("EFIT grids must each contain 65 coordinates")
    if (
        shot.labels.lcfs_r_m.shape != shot.labels.lcfs_z_m.shape
        or shot.labels.lcfs_r_m.shape[0] != frame_count
    ):
        raise ValueError("last-closed-surface point arrays do not match efit_times")
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
