"""Training-free Thomson geometry and validity observation operators."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np


class TopologyClass(StrEnum):
    """Geometry class controlling Thomson channel uncertainty."""

    CORE_VERTICAL = "core_vertical"
    TANGENTIAL_EDGE = "tangential_edge"
    DIVERTOR = "divertor"


class ElmPhase(StrEnum):
    """Typed ELM phase used by the uncertainty policy."""

    QUIESCENT = "quiescent"
    RISING = "rising"
    CRASH = "crash"
    RECOVERY = "recovery"


class ValidityReason(StrEnum):
    """Reason for widening, retained as observation provenance."""

    VALID = "valid"
    NONFINITE = "nonfinite"
    NONPOSITIVE = "nonpositive"
    SATURATED = "saturated"
    ELM_AFFECTED = "elm_affected"


@dataclass(frozen=True)
class ChannelAssessment:
    topology: TopologyClass
    elm_phase: ElmPhase
    reason: ValidityReason
    sigma_multiplier: float
    retained: bool = True


class ChannelValidityPolicy:
    """Map topology, ELM phase and channel health to a sigma multiplier.

    Every channel is retained.  Invalid or ELM-affected observations therefore
    remain visible to downstream estimators but cannot acquire excessive weight.
    """

    _TOPOLOGY_MULTIPLIER = {
        TopologyClass.CORE_VERTICAL: 1.0,
        TopologyClass.TANGENTIAL_EDGE: 1.25,
        TopologyClass.DIVERTOR: 1.8,
    }
    _ELM_MULTIPLIER = {
        ElmPhase.QUIESCENT: 1.0,
        ElmPhase.RISING: 1.8,
        ElmPhase.CRASH: 4.0,
        ElmPhase.RECOVERY: 2.5,
    }

    def assess(
        self,
        temperature_ev: float,
        density_m3: float,
        *,
        topology: TopologyClass,
        elm_phase: ElmPhase = ElmPhase.QUIESCENT,
        saturation_temperature_ev: float = 5.0e4,
    ) -> ChannelAssessment:
        values = np.asarray((temperature_ev, density_m3), dtype=float)
        if not np.all(np.isfinite(values)):
            reason = ValidityReason.NONFINITE
            health_multiplier = 20.0
        elif temperature_ev <= 0.0 or density_m3 <= 0.0:
            reason = ValidityReason.NONPOSITIVE
            health_multiplier = 12.0
        elif temperature_ev >= saturation_temperature_ev:
            reason = ValidityReason.SATURATED
            health_multiplier = 8.0
        elif elm_phase is not ElmPhase.QUIESCENT:
            reason = ValidityReason.ELM_AFFECTED
            health_multiplier = 1.0
        else:
            reason = ValidityReason.VALID
            health_multiplier = 1.0
        multiplier = (
            self._TOPOLOGY_MULTIPLIER[topology]
            * self._ELM_MULTIPLIER[elm_phase]
            * health_multiplier
        )
        return ChannelAssessment(
            topology=topology,
            elm_phase=elm_phase,
            reason=reason,
            sigma_multiplier=float(multiplier),
        )


@dataclass(frozen=True)
class SeparatrixEstimate:
    coordinate_m: float
    coordinate_sigma_m: float
    psi_n: float
    psi_n_sigma: float
    bracketed: bool


@dataclass(frozen=True)
class IsofluxPair:
    first_index: int
    second_index: int
    log_temperature_mismatch: float
    psi_n: float
    psi_n_sigma: float
    geometry_determinant: float


class IsofluxPairer:
    """Pair matched-temperature samples from non-collinear chord families."""

    def __init__(
        self,
        *,
        maximum_log_temperature_mismatch: float = 0.18,
        minimum_geometry_determinant: float = 0.08,
    ) -> None:
        self.maximum_log_temperature_mismatch = maximum_log_temperature_mismatch
        self.minimum_geometry_determinant = minimum_geometry_determinant

    def pair(
        self,
        first_temperature_ev: np.ndarray,
        second_temperature_ev: np.ndarray,
        first_psi_n: np.ndarray,
        second_psi_n: np.ndarray,
        *,
        first_direction_rz: tuple[float, float],
        second_direction_rz: tuple[float, float],
        first_sigma: np.ndarray | None = None,
        second_sigma: np.ndarray | None = None,
    ) -> tuple[IsofluxPair, ...]:
        first_temperature_ev = np.asarray(first_temperature_ev, dtype=float)
        second_temperature_ev = np.asarray(second_temperature_ev, dtype=float)
        first_psi_n = np.asarray(first_psi_n, dtype=float)
        second_psi_n = np.asarray(second_psi_n, dtype=float)
        if first_temperature_ev.shape != first_psi_n.shape:
            raise ValueError("first temperature and psi_N shapes differ")
        if second_temperature_ev.shape != second_psi_n.shape:
            raise ValueError("second temperature and psi_N shapes differ")

        first_direction = _unit_vector(first_direction_rz)
        second_direction = _unit_vector(second_direction_rz)
        determinant = abs(
            float(np.linalg.det(np.stack((first_direction, second_direction))))
        )
        if determinant < self.minimum_geometry_determinant:
            raise ValueError("isoflux chords must be non-collinear")

        first_sigma = _sigma_array(first_sigma, first_psi_n.shape)
        second_sigma = _sigma_array(second_sigma, second_psi_n.shape)
        first_log = np.log(np.maximum(first_temperature_ev, 1.0e-12))
        second_log = np.log(np.maximum(second_temperature_ev, 1.0e-12))
        candidates: list[tuple[float, int, int]] = []
        for first_index in range(first_log.size):
            for second_index in range(second_log.size):
                mismatch = abs(first_log[first_index] - second_log[second_index])
                if (
                    np.isfinite(mismatch)
                    and mismatch <= self.maximum_log_temperature_mismatch
                ):
                    candidates.append((float(mismatch), first_index, second_index))
        candidates.sort()

        used_first: set[int] = set()
        used_second: set[int] = set()
        pairs: list[IsofluxPair] = []
        for mismatch, first_index, second_index in candidates:
            if first_index in used_first or second_index in used_second:
                continue
            variance_first = first_sigma[first_index] ** 2
            variance_second = second_sigma[second_index] ** 2
            weight_first = 1.0 / variance_first
            weight_second = 1.0 / variance_second
            psi_n = (
                weight_first * first_psi_n[first_index]
                + weight_second * second_psi_n[second_index]
            ) / (weight_first + weight_second)
            psi_sigma = np.sqrt(1.0 / (weight_first + weight_second))
            pairs.append(
                IsofluxPair(
                    first_index=first_index,
                    second_index=second_index,
                    log_temperature_mismatch=mismatch,
                    psi_n=float(psi_n),
                    psi_n_sigma=float(psi_sigma),
                    geometry_determinant=determinant,
                )
            )
            used_first.add(first_index)
            used_second.add(second_index)
        return tuple(pairs)


class IsothermAsymmetryOperator:
    r"""Measure :math:`\beta_p + \ell_i/2` from an isotherm centre shift.

    In the large-aspect-ratio observation model,
    ``delta_R = a**2 / R_ref * (beta_p + li/2)``.  Equal-temperature points
    on distinct chord families locate the inboard and outboard isotherm edges;
    their midpoint supplies ``delta_R``.  The relation is analytic and has no
    fit method, learned coefficient or machine-specific parameter.
    """

    def measure(
        self,
        inboard_radius_m: float,
        outboard_radius_m: float,
        *,
        reference_major_radius_m: float,
        minor_radius_m: float,
    ) -> float:
        if reference_major_radius_m <= 0.0 or minor_radius_m <= 0.0:
            raise ValueError("reference and minor radii must be positive")
        isotherm_centre = 0.5 * (inboard_radius_m + outboard_radius_m)
        shift = isotherm_centre - reference_major_radius_m
        return float(shift * reference_major_radius_m / minor_radius_m**2)

    def synthesize_radii(
        self,
        beta_p_plus_li_half: float,
        *,
        reference_major_radius_m: float,
        minor_radius_m: float,
        isotherm_half_width_m: float,
    ) -> tuple[float, float]:
        """Generate the analytic equal-temperature radii for a round trip."""

        shift = beta_p_plus_li_half * minor_radius_m**2 / reference_major_radius_m
        centre = reference_major_radius_m + shift
        return centre - isotherm_half_width_m, centre + isotherm_half_width_m


def _unit_vector(direction: tuple[float, float]) -> np.ndarray:
    vector = np.asarray(direction, dtype=float)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("chord direction must be finite and nonzero")
    return vector / norm


def _sigma_array(sigma: np.ndarray | None, shape: tuple[int, ...]) -> np.ndarray:
    if sigma is None:
        return np.full(shape, 0.08)
    result = np.asarray(sigma, dtype=float)
    if result.shape != shape or np.any(result <= 0.0):
        raise ValueError("psi_N sigma must be positive and match its chord")
    return result
