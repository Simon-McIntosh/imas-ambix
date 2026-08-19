"""Measured coordinate convention for the labelled DIII-D challenge corpus.

The source convention is identified from one standard-field frame in each of
twenty distinct train shots.  The audit discriminates flux per radian from
total flux by integrating the Grad-Shafranov current, measures the flux sense
against recorded plasma current, reads toroidal-field polarity from the bcoil
channel for the q95 handedness test, and checks the Delta-star current
orientation.  No response coefficient is fitted.

All challenge readers use :data:`DIIID_CONVENTION`; the factors themselves are
derived by the shared COCOS algebra in :mod:`imas_ambix.cocos`.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import tau
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow.parquet as pq
from nova.io.cocos import convention
from scipy.constants import mu_0
from scipy.interpolate import RegularGridInterpolator

from imas_ambix.cocos import CANONICAL_COCOS, canonical_factor

if TYPE_CHECKING:
    from collections.abc import Iterable

DIIID_SOURCE_COCOS = 5
"""Empirically identified convention of labelled DIII-D train maps."""

MINIMUM_AUDIT_SHOTS = 20
MINIMUM_PLASMA_CURRENT_KA = 500.0
STANDARD_FIELD_BCOIL_MAXIMUM = 0.0

_TOTAL_FLUX_CANDIDATE_COCOS = DIIID_SOURCE_COCOS + 10
_AUDIT_COLUMNS = (
    "efit_times",
    "efit_psirz",
    "efit_grid_R",
    "efit_grid_Z",
    "efit_r_axis",
    "efit_z_axis",
    "efit_lcfs_n",
    "efit_lcfs_r",
    "efit_lcfs_z",
    "efit_q95",
    "magnetics_plasma_current_times",
    "magnetics_plasma_current",
    "magnetics_time",
    "magnetics_bcoil",
)


@dataclass(frozen=True)
class CorpusConvention:
    """One measured source convention and its transform to Ambix canonical data."""

    source_cocos: int
    target_cocos: int
    source_digits: tuple[int, int, int, int]
    psi_to_canonical: float
    total_flux_to_canonical: float
    ip_to_canonical: float
    toroidal_field_to_canonical: float
    q_to_canonical: float
    derivative_to_canonical: float

    def canonical_flux(self, values: Any) -> np.ndarray:
        """Convert source flux per radian to canonical total flux in webers."""

        return self.psi_to_canonical * np.asarray(values, dtype=float)

    def source_flux(self, values: Any) -> np.ndarray:
        """Convert canonical total flux back to source flux per radian."""

        return np.asarray(values, dtype=float) / self.psi_to_canonical

    def canonical_total_flux(self, values: Any) -> np.ndarray:
        """Convert source-sense total-flux receipts to canonical total flux."""

        return self.total_flux_to_canonical * np.asarray(values, dtype=float)

    def canonical_plasma_current(self, values: Any) -> np.ndarray:
        """Convert plasma-current direction while retaining the input unit."""

        return self.ip_to_canonical * np.asarray(values, dtype=float)

    def canonical_toroidal_field(self, values: Any) -> np.ndarray:
        """Convert the bcoil/toroidal-field direction to canonical handedness."""

        return self.toroidal_field_to_canonical * np.asarray(values, dtype=float)

    def canonical_q(self, values: Any) -> np.ndarray:
        """Convert safety factor to canonical flux-surface handedness."""

        return self.q_to_canonical * np.asarray(values, dtype=float)

    def canonical_derivative(self, values: Any) -> np.ndarray:
        """Convert a conventional derivative with respect to source flux."""

        return self.derivative_to_canonical * np.asarray(values, dtype=float)


_SOURCE = convention(DIIID_SOURCE_COCOS)
DIIID_CONVENTION = CorpusConvention(
    source_cocos=DIIID_SOURCE_COCOS,
    target_cocos=CANONICAL_COCOS,
    source_digits=_SOURCE.digits,
    psi_to_canonical=canonical_factor("psi_like", source_cocos=DIIID_SOURCE_COCOS),
    total_flux_to_canonical=canonical_factor(
        "psi_like", source_cocos=_TOTAL_FLUX_CANDIDATE_COCOS
    ),
    ip_to_canonical=canonical_factor("ip_like", source_cocos=DIIID_SOURCE_COCOS),
    toroidal_field_to_canonical=canonical_factor(
        "b0_like", source_cocos=DIIID_SOURCE_COCOS
    ),
    q_to_canonical=canonical_factor("q_like", source_cocos=DIIID_SOURCE_COCOS),
    derivative_to_canonical=canonical_factor(
        "dodpsi_like", source_cocos=DIIID_SOURCE_COCOS
    ),
)


@dataclass(frozen=True)
class ConventionFrameReceipt:
    """Independent convention discriminators from one train shot."""

    shot: str
    plasma_current_ka: float
    bcoil: float
    q95: float
    axis_to_boundary_flux: float
    per_radian_current_ratio: float
    total_flux_current_ratio: float

    @property
    def psi_ip_sign(self) -> int:
        flux_sign = np.sign(self.axis_to_boundary_flux)
        return int(flux_sign * np.sign(self.plasma_current_ka))

    @property
    def q_ip_bcoil_sign(self) -> int:
        return int(
            np.sign(self.q95) * np.sign(self.plasma_current_ka) * np.sign(self.bcoil)
        )

    @property
    def delta_star_ip_sign(self) -> int:
        return int(np.sign(self.per_radian_current_ratio))


@dataclass(frozen=True)
class ConventionReceipt:
    """Aggregated empirical evidence identifying the source convention."""

    frames: tuple[ConventionFrameReceipt, ...]

    @property
    def shots(self) -> int:
        return len(self.frames)

    @property
    def per_radian_wins(self) -> int:
        return sum(
            abs(frame.per_radian_current_ratio - 1.0)
            < abs(frame.total_flux_current_ratio - 1.0)
            for frame in self.frames
        )

    @property
    def psi_ip_positive(self) -> int:
        return sum(frame.psi_ip_sign == 1 for frame in self.frames)

    @property
    def q_ip_bcoil_negative(self) -> int:
        return sum(frame.q_ip_bcoil_sign == -1 for frame in self.frames)

    @property
    def delta_star_ip_positive(self) -> int:
        return sum(frame.delta_star_ip_sign == 1 for frame in self.frames)

    @property
    def per_radian_ratio_median(self) -> float:
        ratios = [frame.per_radian_current_ratio for frame in self.frames]
        return float(np.median(ratios))

    @property
    def total_flux_ratio_median(self) -> float:
        ratios = [frame.total_flux_current_ratio for frame in self.frames]
        return float(np.median(ratios))


def _read(path: Path) -> dict[str, Any]:
    table = pq.read_table(path, columns=list(_AUDIT_COLUMNS))
    return {name: table[name][0].as_py() for name in table.column_names}


def _candidate_frame(row: dict[str, Any]) -> int | None:
    times = np.asarray(row["efit_times"], dtype=float)
    plasma_current = np.interp(
        times,
        np.asarray(row["magnetics_plasma_current_times"], dtype=float),
        np.asarray(row["magnetics_plasma_current"], dtype=float),
    )
    bcoil = np.interp(
        times,
        np.asarray(row["magnetics_time"], dtype=float),
        np.asarray(row["magnetics_bcoil"], dtype=float),
    )
    q95 = np.asarray(row["efit_q95"], dtype=float)
    eligible = np.flatnonzero(
        np.isfinite(plasma_current + bcoil + q95)
        & (plasma_current >= MINIMUM_PLASMA_CURRENT_KA)
        & (bcoil < STANDARD_FIELD_BCOIL_MAXIMUM)
        & (q95 != 0.0)
    )
    if eligible.size == 0:
        return None
    return int(eligible[np.argmax(plasma_current[eligible])])


def _axis_and_boundary(row: dict[str, Any], frame: int) -> tuple[float, float]:
    radius = np.asarray(row["efit_grid_R"], dtype=float)
    height = np.asarray(row["efit_grid_Z"], dtype=float)
    flux = np.asarray(row["efit_psirz"][frame], dtype=float)
    sampler = RegularGridInterpolator(
        (height, radius), flux, bounds_error=False, fill_value=np.nan
    )
    axis = float(sampler([[row["efit_z_axis"][frame], row["efit_r_axis"][frame]]])[0])
    count = int(row["efit_lcfs_n"][frame])
    boundary_points = np.column_stack(
        (
            np.asarray(row["efit_lcfs_z"][frame][:count], dtype=float),
            np.asarray(row["efit_lcfs_r"][frame][:count], dtype=float),
        )
    )
    boundary = float(np.nanmedian(sampler(boundary_points)))
    return axis, boundary


def _integrated_current(row: dict[str, Any], frame: int, flux_factor: float) -> float:
    radius = np.asarray(row["efit_grid_R"], dtype=float)
    height = np.asarray(row["efit_grid_Z"], dtype=float)
    source_flux = np.asarray(row["efit_psirz"][frame], dtype=float)
    total_flux = flux_factor * source_flux
    derivative_z, derivative_r = np.gradient(total_flux, height, radius, edge_order=2)
    second_z = np.gradient(derivative_z, height, axis=0, edge_order=2)
    second_r = np.gradient(derivative_r, radius, axis=1, edge_order=2)
    delta_star = second_r - derivative_r / radius[np.newaxis, :] + second_z
    density = -delta_star / (tau * mu_0 * radius[np.newaxis, :])

    axis, boundary = _axis_and_boundary(row, frame)
    normalised = (source_flux - axis) / (boundary - axis)
    selected = np.isfinite(density) & np.isfinite(normalised) & (normalised <= 1.0)
    interior = np.zeros_like(selected)
    interior[2:-2, 2:-2] = True
    selected &= interior
    cell_area = float(np.diff(radius).mean() * np.diff(height).mean())
    return float(np.sum(density[selected]) * cell_area)


def measure_diiid_convention(
    paths: Iterable[str | Path], *, shots: int = MINIMUM_AUDIT_SHOTS
) -> ConventionReceipt:
    """Compute convention receipts over distinct eligible DIII-D train shots."""

    if shots < MINIMUM_AUDIT_SHOTS:
        raise ValueError(
            f"convention evidence requires at least {MINIMUM_AUDIT_SHOTS} shots"
        )
    selected: list[ConventionFrameReceipt] = []
    total_candidate_factor = canonical_factor(
        "psi_like", source_cocos=_TOTAL_FLUX_CANDIDATE_COCOS
    )
    for item in paths:
        path = Path(item)
        row = _read(path)
        frame = _candidate_frame(row)
        if frame is None:
            continue
        time_ms = float(row["efit_times"][frame])
        plasma_current_ka = float(
            np.interp(
                time_ms,
                row["magnetics_plasma_current_times"],
                row["magnetics_plasma_current"],
            )
        )
        bcoil = float(np.interp(time_ms, row["magnetics_time"], row["magnetics_bcoil"]))
        axis, boundary = _axis_and_boundary(row, frame)
        recorded_current_a = 1000.0 * plasma_current_ka
        per_radian_current = _integrated_current(
            row, frame, DIIID_CONVENTION.psi_to_canonical
        )
        total_flux_current = _integrated_current(row, frame, total_candidate_factor)
        selected.append(
            ConventionFrameReceipt(
                shot=path.name,
                plasma_current_ka=plasma_current_ka,
                bcoil=bcoil,
                q95=float(row["efit_q95"][frame]),
                axis_to_boundary_flux=boundary - axis,
                per_radian_current_ratio=per_radian_current / recorded_current_a,
                total_flux_current_ratio=total_flux_current / recorded_current_a,
            )
        )
        if len(selected) == shots:
            break
    if len(selected) < shots:
        raise RuntimeError(
            f"only {len(selected)} shots contain the declared convention frame"
        )
    return ConventionReceipt(frames=tuple(selected))


__all__ = [
    "DIIID_CONVENTION",
    "DIIID_SOURCE_COCOS",
    "MINIMUM_AUDIT_SHOTS",
    "ConventionFrameReceipt",
    "ConventionReceipt",
    "CorpusConvention",
    "measure_diiid_convention",
]
