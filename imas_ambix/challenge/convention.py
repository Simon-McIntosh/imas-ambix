"""Measured coordinate convention for a labelled equilibrium corpus.

The source convention is identified from one standard-field frame in each of
twenty distinct train shots.  The audit discriminates flux per radian from
total flux by integrating the Grad-Shafranov current, measures the flux sense
against recorded plasma current, reads toroidal-field polarity from the bcoil
channel for the q95 handedness test, and checks the Delta-star current
orientation.  No response coefficient is fitted.

The audit reads the loader's :class:`~imas_ambix.challenge.loader.EfitLabels`
record, so it is corpus-agnostic: the DIII-D challenge corpus supplies the
first loader and each further machine supplies an eligibility declaration
beside its loader.  :func:`measure_diiid_convention` is the DIII-D entry.

The records the kernel reads are in the canonical convention (COCOS 17), which
the loader applies before the audit sees them.  All challenge readers use
:data:`DIIID_CONVENTION`; the factors themselves are derived by the shared
COCOS algebra in :mod:`imas_alambic.cocos`.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import tau
from typing import TYPE_CHECKING, Any

import numpy as np
from nova_cocos import convention
from scipy.constants import mu_0
from scipy.interpolate import RegularGridInterpolator

from imas_alambic.cocos import CANONICAL_COCOS, canonical_factor

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from .loader import EfitLabels

DIIID_SOURCE_COCOS = 5
"""Empirically identified convention of labelled DIII-D train maps."""

MINIMUM_AUDIT_SHOTS = 20
"""Minimum number of audit shots the DIII-D corpus declaration requires."""

_TOTAL_FLUX_CANDIDATE_COCOS = DIIID_SOURCE_COCOS + 10

_PLASMA_CURRENT_KEY = "magnetics_plasma_current"
_BCOIL_KEY = "magnetics_bcoil"
_Q95_KEY = "efit_q95"
_R_AXIS_KEY = "efit_r_axis"
_Z_AXIS_KEY = "efit_z_axis"
_LCFS_N_KEY = "efit_lcfs_n"


@dataclass(frozen=True)
class ConventionEligibility:
    """Per-corpus thresholds declaring which equilibrium frames are auditable.

    A loader publishes one of these beside its :class:`EfitLabels` records so
    the shared kernel holds no corpus-specific constants.
    """

    minimum_plasma_current_ka: float
    standard_field_bcoil_maximum: float
    minimum_audit_shots: int


DIIID_ELIGIBILITY = ConventionEligibility(
    minimum_plasma_current_ka=500.0,
    standard_field_bcoil_maximum=0.0,
    minimum_audit_shots=MINIMUM_AUDIT_SHOTS,
)
"""The DIII-D challenge corpus's own eligibility declaration."""


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
    def psi_ip_negative(self) -> int:
        return sum(frame.psi_ip_sign == -1 for frame in self.frames)

    @property
    def q_ip_bcoil_negative(self) -> int:
        return sum(frame.q_ip_bcoil_sign == -1 for frame in self.frames)

    @property
    def q_ip_bcoil_positive(self) -> int:
        return sum(frame.q_ip_bcoil_sign == 1 for frame in self.frames)

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


def _candidate_frame(
    labels: EfitLabels, eligibility: ConventionEligibility
) -> int | None:
    plasma_current = np.asarray(labels.scalars[_PLASMA_CURRENT_KEY], dtype=float)
    bcoil = np.asarray(labels.scalars[_BCOIL_KEY], dtype=float)
    q95 = np.asarray(labels.scalars[_Q95_KEY], dtype=float)
    eligible = np.flatnonzero(
        np.isfinite(plasma_current + bcoil + q95)
        & (plasma_current >= eligibility.minimum_plasma_current_ka)
        & (bcoil < eligibility.standard_field_bcoil_maximum)
        & (q95 != 0.0)
    )
    if eligible.size == 0:
        return None
    return int(eligible[np.argmax(plasma_current[eligible])])


def _axis_and_boundary(labels: EfitLabels, frame: int) -> tuple[float, float]:
    radius = np.asarray(labels.grid_r_m, dtype=float)
    height = np.asarray(labels.grid_z_m, dtype=float)
    flux = np.asarray(labels.psirz[frame], dtype=float)
    sampler = RegularGridInterpolator(
        (height, radius), flux, bounds_error=False, fill_value=np.nan
    )
    r_axis = float(labels.scalars[_R_AXIS_KEY][frame])
    z_axis = float(labels.scalars[_Z_AXIS_KEY][frame])
    axis = float(sampler([[z_axis, r_axis]])[0])
    count = int(labels.scalars[_LCFS_N_KEY][frame])
    boundary_points = np.column_stack(
        (
            np.asarray(labels.lcfs_z_m[frame][:count], dtype=float),
            np.asarray(labels.lcfs_r_m[frame][:count], dtype=float),
        )
    )
    boundary = float(np.nanmedian(sampler(boundary_points)))
    return axis, boundary


def _integrated_current(labels: EfitLabels, frame: int, flux_factor: float) -> float:
    radius = np.asarray(labels.grid_r_m, dtype=float)
    height = np.asarray(labels.grid_z_m, dtype=float)
    source_flux = np.asarray(labels.psirz[frame], dtype=float)
    total_flux = flux_factor * source_flux
    derivative_z, derivative_r = np.gradient(total_flux, height, radius, edge_order=2)
    second_z = np.gradient(derivative_z, height, axis=0, edge_order=2)
    second_r = np.gradient(derivative_r, radius, axis=1, edge_order=2)
    delta_star = second_r - derivative_r / radius[np.newaxis, :] + second_z
    density = -delta_star / (tau * mu_0 * radius[np.newaxis, :])

    axis, boundary = _axis_and_boundary(labels, frame)
    normalised = (source_flux - axis) / (boundary - axis)
    selected = np.isfinite(density) & np.isfinite(normalised) & (normalised <= 1.0)
    interior = np.zeros_like(selected)
    interior[2:-2, 2:-2] = True
    selected &= interior
    cell_area = float(np.diff(radius).mean() * np.diff(height).mean())
    return float(np.sum(density[selected]) * cell_area)


def measure_convention(
    records: Iterable[EfitLabels],
    eligibility: ConventionEligibility,
    *,
    shots: int,
) -> ConventionReceipt:
    """Compute convention receipts over distinct eligible loader records.

    Each record is a canonical-convention
    :class:`~imas_ambix.challenge.loader.EfitLabels`; ``eligibility`` is the
    loader's declared corpus thresholds.  Records whose standard-field frame is
    absent are skipped.
    """

    if shots < eligibility.minimum_audit_shots:
        raise ValueError(
            "convention evidence requires at least "
            f"{eligibility.minimum_audit_shots} shots"
        )
    selected: list[ConventionFrameReceipt] = []
    for labels in records:
        frame = _candidate_frame(labels, eligibility)
        if frame is None:
            continue
        plasma_current_ka = float(labels.scalars[_PLASMA_CURRENT_KEY][frame])
        bcoil = float(labels.scalars[_BCOIL_KEY][frame])
        axis, boundary = _axis_and_boundary(labels, frame)
        recorded_current_a = 1000.0 * plasma_current_ka
        per_radian_current = _integrated_current(labels, frame, 1.0)
        total_flux_current = _integrated_current(labels, frame, 1.0 / tau)
        selected.append(
            ConventionFrameReceipt(
                shot=getattr(labels, "shot", f"record-{len(selected)}"),
                plasma_current_ka=plasma_current_ka,
                bcoil=bcoil,
                q95=float(labels.scalars[_Q95_KEY][frame]),
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


def measure_diiid_convention(
    paths: Iterable[str | Path], *, shots: int = MINIMUM_AUDIT_SHOTS
) -> ConventionReceipt:
    """Compute convention receipts over distinct eligible DIII-D train shots."""

    from .loader import load_labels

    records = (load_labels(path) for path in paths)
    return measure_convention(records, DIIID_ELIGIBILITY, shots=shots)


__all__ = [
    "DIIID_CONVENTION",
    "DIIID_ELIGIBILITY",
    "DIIID_SOURCE_COCOS",
    "MINIMUM_AUDIT_SHOTS",
    "ConventionEligibility",
    "ConventionFrameReceipt",
    "ConventionReceipt",
    "CorpusConvention",
    "measure_convention",
    "measure_diiid_convention",
]
