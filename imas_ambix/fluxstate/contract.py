"""Immutable flux-function state shared with deterministic physics consumers.

The contract keeps host-owned metadata separate from an array-only tree.  A
consumer may therefore convert the numerical leaves to NumPy, JAX, or another
array namespace without importing that runtime here or weakening the physical
validation performed at construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum, StrEnum
from typing import Any, Protocol

import numpy as np


class FluxStateError(ValueError):
    """Base class for invalid or internally inconsistent state payloads."""


class DerivativeConsistencyError(FluxStateError):
    """Raised when a primitive and its declared physical derivative disagree."""


class HandoffEligibilityError(FluxStateError):
    """Raised when a smoothing product is offered as an online handoff."""


class ArrayNamespace(Protocol):
    """Small NumPy-compatible surface needed by :meth:`array_tree`."""

    def asarray(self, value: Any) -> Any:
        """Convert one numerical leaf to the namespace's array type."""


def _immutable_array(value: Any, label: str, *, ndim: int = 1) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.ndim != ndim:
        raise FluxStateError(f"{label} must have {ndim} dimensions")
    if not np.all(np.isfinite(array)):
        raise FluxStateError(f"{label} contains a non-finite value")
    array.setflags(write=False)
    return array


def _nonempty(value: str, label: str) -> None:
    if not value.strip():
        raise FluxStateError(f"{label} must be non-empty")


def _aware(value: datetime, label: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise FluxStateError(f"{label} must carry an explicit timezone")


class CoordinateKind(StrEnum):
    """Radial coordinate carried by this producer contract."""

    NORMALIZED_POLOIDAL_FLUX = "psi_N"


class FluxDirection(IntEnum):
    """Sign of total poloidal flux while moving outward in normalized flux."""

    DECREASES_OUTWARD = -1
    INCREASES_OUTWARD = 1


class PlasmaDomain(StrEnum):
    """Stable names matching the topology partition consumed by Nova."""

    CORE = "core"
    COMMON_SOL = "common_sol"
    PRIVATE_FLUX = "private_flux"


@dataclass(frozen=True, slots=True)
class RadialCoordinate:
    """Normalized poloidal flux and its physical total-flux Jacobian."""

    values: np.ndarray
    axis_total_flux_wb: float
    separatrix_total_flux_wb: float
    dtotal_flux_dpsi_n_wb: float
    kind: CoordinateKind = CoordinateKind.NORMALIZED_POLOIDAL_FLUX
    coordinate_units: str = "1"
    total_flux_units: str = "Wb"

    def __post_init__(self) -> None:
        values = _immutable_array(self.values, "radial coordinate")
        object.__setattr__(self, "values", values)
        if values.size < 3:
            raise FluxStateError("radial coordinate needs at least three points")
        if not np.all(np.diff(values) > 0.0):
            raise FluxStateError("radial coordinate must be strictly increasing")
        if self.kind is not CoordinateKind.NORMALIZED_POLOIDAL_FLUX:
            raise FluxStateError("only normalized poloidal flux is supported")
        if self.coordinate_units != "1" or self.total_flux_units != "Wb":
            raise FluxStateError(
                "normalized flux must use unit 1 and total flux must use Wb"
            )
        references = np.asarray(
            [
                self.axis_total_flux_wb,
                self.separatrix_total_flux_wb,
                self.dtotal_flux_dpsi_n_wb,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(references)):
            raise FluxStateError("coordinate references and Jacobian must be finite")
        expected = self.separatrix_total_flux_wb - self.axis_total_flux_wb
        if expected == 0.0:
            raise FluxStateError("axis and separatrix total flux must differ")
        if not np.isclose(
            self.dtotal_flux_dpsi_n_wb,
            expected,
            rtol=1e-12,
            atol=1e-15,
        ):
            raise FluxStateError(
                "total-flux Jacobian must equal separatrix minus axis flux"
            )

    @property
    def total_flux_wb(self) -> np.ndarray:
        """Return physical total poloidal flux at every radial point."""

        values = self.axis_total_flux_wb + (self.values * self.dtotal_flux_dpsi_n_wb)
        values.setflags(write=False)
        return values


@dataclass(frozen=True, slots=True)
class DerivativeTolerance:
    """Shared relative and quantity-specific absolute derivative tolerances."""

    relative: float = 1e-8
    pressure_absolute_pa_per_wb: float = 1e-6
    f_df_absolute_t2m2_per_wb: float = 1e-10

    def __post_init__(self) -> None:
        values = (
            self.relative,
            self.pressure_absolute_pa_per_wb,
            self.f_df_absolute_t2m2_per_wb,
        )
        if not all(np.isfinite(value) and value >= 0.0 for value in values):
            raise FluxStateError(
                "derivative tolerances must be finite and non-negative"
            )


@dataclass(frozen=True, slots=True)
class DomainProfilePolicy:
    """Physical continuation policy for one topology-qualified domain."""

    functional_form: str
    continuity_order: int
    support_limit_psi_n: float
    outer_condition: str

    def __post_init__(self) -> None:
        _nonempty(self.functional_form, "domain functional form")
        _nonempty(self.outer_condition, "domain outer condition")
        if self.continuity_order < 0:
            raise FluxStateError("domain continuity order cannot be negative")
        if not np.isfinite(self.support_limit_psi_n):
            raise FluxStateError("domain support limit must be finite")


@dataclass(frozen=True, slots=True)
class DomainProfile:
    """Sources declared independently on a Nova topology class."""

    domain: PlasmaDomain
    coordinate: RadialCoordinate
    pressure: np.ndarray
    dpressure_dpsi: np.ndarray
    f: np.ndarray
    f_df_dpsi: np.ndarray
    policy: DomainProfilePolicy

    def __post_init__(self) -> None:
        if self.domain is PlasmaDomain.CORE:
            raise FluxStateError("the core profile belongs on FluxFunctionState")
        size = self.coordinate.values.size
        for label in ("pressure", "dpressure_dpsi", "f", "f_df_dpsi"):
            array = _immutable_array(
                getattr(self, label), f"{self.domain.value} {label}"
            )
            if array.shape != (size,):
                raise FluxStateError(
                    f"{self.domain.value} {label} must match its coordinate"
                )
            object.__setattr__(self, label, array)
        if (
            self.domain is PlasmaDomain.COMMON_SOL
            and np.min(self.coordinate.values) < 1.0
        ):
            raise FluxStateError("common SOL coordinates cannot lie below psi_N=1")


@dataclass(frozen=True, slots=True)
class SpeciesAssumption:
    """One ion species required by an isothermal-surface flow closure."""

    name: str
    mass_amu: float
    charge_e: float
    fraction: float

    def __post_init__(self) -> None:
        _nonempty(self.name, "species name")
        if self.mass_amu <= 0.0 or not np.isfinite(self.mass_amu):
            raise FluxStateError("species mass must be finite and positive")
        if not np.isfinite(self.charge_e) or self.charge_e == 0.0:
            raise FluxStateError("species charge must be finite and non-zero")
        if not np.isfinite(self.fraction) or self.fraction <= 0.0:
            raise FluxStateError("species fraction must be finite and positive")


@dataclass(frozen=True, slots=True)
class IsothermalToroidalFlow:
    """Toroidal rotation closure with temperature constant on each surface."""

    omega_rad_per_s: np.ndarray
    temperature_ev: np.ndarray
    species: tuple[SpeciesAssumption, ...]
    thermodynamic_model: str = "isothermal_flux_surface"

    def __post_init__(self) -> None:
        omega = _immutable_array(self.omega_rad_per_s, "toroidal angular velocity")
        temperature = _immutable_array(self.temperature_ev, "temperature")
        object.__setattr__(self, "omega_rad_per_s", omega)
        object.__setattr__(self, "temperature_ev", temperature)
        object.__setattr__(self, "species", tuple(self.species))
        if omega.shape != temperature.shape:
            raise FluxStateError(
                "rotation and temperature profiles must have one shape"
            )
        if np.any(temperature <= 0.0):
            raise FluxStateError("isothermal temperature must be positive")
        if not self.species:
            raise FluxStateError("toroidal flow needs at least one species assumption")
        if not np.isclose(sum(item.fraction for item in self.species), 1.0):
            raise FluxStateError("species fractions must sum to one")
        if self.thermodynamic_model != "isothermal_flux_surface":
            raise FluxStateError("unsupported toroidal-flow thermodynamic model")


@dataclass(frozen=True, slots=True)
class ConventionProvenance:
    """COCOS identity and externally established sign declarations."""

    cocos: int
    flux_direction: FluxDirection
    toroidal_field_sign: int
    plasma_current_sign: int
    evidence: str

    def __post_init__(self) -> None:
        valid = {*range(1, 9), *range(11, 19)}
        if isinstance(self.cocos, bool) or self.cocos not in valid:
            raise FluxStateError(f"unrecognised COCOS convention {self.cocos!r}")
        if self.toroidal_field_sign not in {-1, 1}:
            raise FluxStateError("toroidal-field sign must be -1 or +1")
        if self.plasma_current_sign not in {-1, 1}:
            raise FluxStateError("plasma-current sign must be -1 or +1")
        _nonempty(self.evidence, "COCOS and sign evidence")


@dataclass(frozen=True, slots=True)
class WeightedEnsembleIdentity:
    """Identity of one member within a normalized weighted ensemble."""

    ensemble_id: str
    member_ids: tuple[str, ...]
    weights: tuple[float, ...]
    active_member: int
    ensemble_digest: str

    def __post_init__(self) -> None:
        _nonempty(self.ensemble_id, "ensemble id")
        _nonempty(self.ensemble_digest, "ensemble digest")
        member_ids = tuple(self.member_ids)
        weights = tuple(float(value) for value in self.weights)
        object.__setattr__(self, "member_ids", member_ids)
        object.__setattr__(self, "weights", weights)
        if not member_ids or len(member_ids) != len(weights):
            raise FluxStateError(
                "ensemble members and weights must have one non-empty shape"
            )
        if len(set(member_ids)) != len(member_ids) or any(
            not item for item in member_ids
        ):
            raise FluxStateError("ensemble member ids must be non-empty and unique")
        if not 0 <= self.active_member < len(member_ids):
            raise FluxStateError("active ensemble member is out of range")
        if any(not np.isfinite(value) or value < 0.0 for value in weights):
            raise FluxStateError("ensemble weights must be finite and non-negative")
        if not np.isclose(sum(weights), 1.0, rtol=1e-12, atol=1e-12):
            raise FluxStateError("ensemble weights must sum to one")

    @property
    def member_id(self) -> str:
        """Return the active member's stable identity."""

        return self.member_ids[self.active_member]


@dataclass(frozen=True, slots=True)
class ValidityFlag:
    """One named validity result retained at the state boundary."""

    name: str
    valid: bool
    reason: str

    def __post_init__(self) -> None:
        _nonempty(self.name, "validity flag name")
        _nonempty(self.reason, "validity flag reason")


@dataclass(frozen=True, slots=True)
class StateProvenance:
    """Producer and diagnostic identity needed to reproduce one state."""

    model_digest: str
    checkpoint_digest: str
    source_diagnostics: tuple[str, ...]
    validity_flags: tuple[ValidityFlag, ...]

    def __post_init__(self) -> None:
        _nonempty(self.model_digest, "model digest")
        _nonempty(self.checkpoint_digest, "checkpoint digest")
        diagnostics = tuple(self.source_diagnostics)
        flags = tuple(self.validity_flags)
        object.__setattr__(self, "source_diagnostics", diagnostics)
        object.__setattr__(self, "validity_flags", flags)
        if not diagnostics or any(not item.strip() for item in diagnostics):
            raise FluxStateError("source diagnostics must be non-empty names")
        names = [item.name for item in flags]
        if len(names) != len(set(names)):
            raise FluxStateError("validity flag names must be unique")


@dataclass(frozen=True, slots=True)
class ForecastHandoff:
    """Causal state propagated beyond the latest assimilated observation."""

    valid_at: datetime
    information_cutoff: datetime
    produced_at: datetime

    def __post_init__(self) -> None:
        for label in ("valid_at", "information_cutoff", "produced_at"):
            _aware(getattr(self, label), label)
        if self.information_cutoff >= self.valid_at:
            raise FluxStateError(
                "a forecast must be valid after its information cutoff"
            )
        if self.produced_at < self.information_cutoff:
            raise FluxStateError(
                "forecast production cannot predate its information cutoff"
            )


@dataclass(frozen=True, slots=True)
class AnalysisHandoff:
    """Causal state incorporating observations through its validity time."""

    valid_at: datetime
    information_cutoff: datetime
    produced_at: datetime

    def __post_init__(self) -> None:
        for label in ("valid_at", "information_cutoff", "produced_at"):
            _aware(getattr(self, label), label)
        if self.information_cutoff != self.valid_at:
            raise FluxStateError("an analysis cutoff must equal its validity time")
        if self.produced_at < self.information_cutoff:
            raise FluxStateError(
                "analysis production cannot predate its information cutoff"
            )


@dataclass(frozen=True, slots=True)
class FixedLagSmoothing:
    """Non-causal state using a bounded interval of future information."""

    valid_at: datetime
    information_cutoff: datetime
    produced_at: datetime
    lag_seconds: float

    def __post_init__(self) -> None:
        for label in ("valid_at", "information_cutoff", "produced_at"):
            _aware(getattr(self, label), label)
        lookahead = (self.information_cutoff - self.valid_at).total_seconds()
        if not np.isfinite(self.lag_seconds) or self.lag_seconds <= 0.0:
            raise FluxStateError("fixed smoothing lag must be finite and positive")
        if lookahead <= 0.0 or lookahead > self.lag_seconds:
            raise FluxStateError(
                "fixed-lag smoothing must use future data within its lag"
            )
        if self.produced_at < self.information_cutoff:
            raise FluxStateError(
                "smoothing production cannot predate its information cutoff"
            )


@dataclass(frozen=True, slots=True)
class FullSequenceSmoothing:
    """Non-causal state conditioned on an explicitly bounded full sequence."""

    valid_at: datetime
    information_cutoff: datetime
    produced_at: datetime
    sequence_start: datetime
    sequence_end: datetime

    def __post_init__(self) -> None:
        for label in (
            "valid_at",
            "information_cutoff",
            "produced_at",
            "sequence_start",
            "sequence_end",
        ):
            _aware(getattr(self, label), label)
        if not self.sequence_start <= self.valid_at < self.information_cutoff:
            raise FluxStateError("full smoothing must use information after validity")
        if self.information_cutoff > self.sequence_end:
            raise FluxStateError("smoothing cutoff lies outside the declared sequence")
        if self.produced_at < self.information_cutoff:
            raise FluxStateError(
                "smoothing production cannot predate its information cutoff"
            )


type OnlineHandoff = ForecastHandoff | AnalysisHandoff
type TemporalProvenance = (
    ForecastHandoff | AnalysisHandoff | FixedLagSmoothing | FullSequenceSmoothing
)


def require_online_handoff(value: TemporalProvenance) -> OnlineHandoff:
    """Return a causal online handoff and reject both smoothing products."""

    if isinstance(value, (ForecastHandoff, AnalysisHandoff)):
        return value
    raise HandoffEligibilityError(
        f"{type(value).__name__} is smoothing and cannot be an online handoff"
    )


class IntegralMoment(StrEnum):
    """Integral observations retained in the result ledger."""

    PLASMA_CURRENT = "Ip"
    POLOIDAL_BETA = "beta_p"
    INTERNAL_INDUCTANCE = "li"


_MOMENT_UNITS = {
    IntegralMoment.PLASMA_CURRENT: "A",
    IntegralMoment.POLOIDAL_BETA: "1",
    IntegralMoment.INTERNAL_INDUCTANCE: "1",
}


@dataclass(frozen=True, slots=True)
class IntegralTarget:
    """One explicit moment target and its acceptance tolerance."""

    moment: IntegralMoment
    value: float
    tolerance: float
    units: str

    def __post_init__(self) -> None:
        if self.units != _MOMENT_UNITS[self.moment]:
            raise FluxStateError(
                f"{self.moment.value} target units must be {_MOMENT_UNITS[self.moment]}"
            )
        if not np.isfinite(self.value):
            raise FluxStateError("integral target must be finite")
        if not np.isfinite(self.tolerance) or self.tolerance <= 0.0:
            raise FluxStateError("integral target tolerance must be positive")


@dataclass(frozen=True, slots=True)
class FreeAmplitude:
    """Physically named profile degree of freedom used by conditioning."""

    name: str
    units: str
    prior_mean: float

    def __post_init__(self) -> None:
        _nonempty(self.name, "free amplitude name")
        _nonempty(self.units, "free amplitude units")
        if not np.isfinite(self.prior_mean):
            raise FluxStateError("free amplitude prior mean must be finite")


class IntegralPolicyKind(StrEnum):
    """Whether supplied sources are absolute or explicitly conditioned."""

    ABSOLUTE_SOURCE = "absolute_source"
    MOMENT_CLOSURE = "moment_closure"


@dataclass(frozen=True, slots=True)
class IntegralConstraintPolicy:
    """Absolute-source policy or a fully declared moment-conditioning family."""

    kind: IntegralPolicyKind
    free_amplitudes: tuple[FreeAmplitude, ...] = ()
    targets: tuple[IntegralTarget, ...] = ()
    prior_covariance: np.ndarray | None = None

    def __post_init__(self) -> None:
        amplitudes = tuple(self.free_amplitudes)
        targets = tuple(self.targets)
        object.__setattr__(self, "free_amplitudes", amplitudes)
        object.__setattr__(self, "targets", targets)
        names = [item.name for item in amplitudes]
        moments = [item.moment for item in targets]
        if len(names) != len(set(names)):
            raise FluxStateError("free amplitude names must be unique")
        if len(moments) != len(set(moments)):
            raise FluxStateError("integral target moments must be unique")
        if self.kind is IntegralPolicyKind.ABSOLUTE_SOURCE:
            if amplitudes or targets or self.prior_covariance is not None:
                raise FluxStateError("absolute sources cannot carry fitting controls")
            return
        if not amplitudes or not targets:
            raise FluxStateError("moment closure needs free amplitudes and targets")
        if len(targets) == 1:
            if (
                len(amplitudes) != 1
                or targets[0].moment is not IntegralMoment.PLASMA_CURRENT
            ):
                raise FluxStateError("a scalar closure is one amplitude targeting Ip")
        elif len(amplitudes) != len(targets):
            raise FluxStateError(
                "multi-moment closure needs one physical amplitude per target"
            )
        if self.prior_covariance is None:
            raise FluxStateError("moment closure needs an Ambix prior covariance")
        covariance = _immutable_array(self.prior_covariance, "prior covariance", ndim=2)
        object.__setattr__(self, "prior_covariance", covariance)
        expected = (len(amplitudes), len(amplitudes))
        if covariance.shape != expected:
            raise FluxStateError(f"prior covariance must have shape {expected}")
        if not np.allclose(covariance, covariance.T, rtol=1e-12, atol=1e-15):
            raise FluxStateError("prior covariance must be symmetric")
        if np.min(np.linalg.eigvalsh(covariance)) <= 0.0:
            raise FluxStateError("prior covariance must be positive definite")

    @classmethod
    def absolute_sources(cls) -> IntegralConstraintPolicy:
        """Construct the no-renormalization policy."""

        return cls(kind=IntegralPolicyKind.ABSOLUTE_SOURCE)


@dataclass(frozen=True, slots=True)
class IntegralResult:
    """Target and achieved value for one immutable result-ledger row."""

    moment: IntegralMoment
    achieved: float
    target: float | None
    tolerance: float | None
    units: str

    def __post_init__(self) -> None:
        if self.units != _MOMENT_UNITS[self.moment]:
            raise FluxStateError(
                f"{self.moment.value} result units must be {_MOMENT_UNITS[self.moment]}"
            )
        if not np.isfinite(self.achieved):
            raise FluxStateError("achieved integral result must be finite")
        if (self.target is None) != (self.tolerance is None):
            raise FluxStateError("target and tolerance must be present together")
        if self.target is not None and not np.isfinite(self.target):
            raise FluxStateError("integral result target must be finite")
        if self.tolerance is not None and (
            not np.isfinite(self.tolerance) or self.tolerance <= 0.0
        ):
            raise FluxStateError("integral result tolerance must be positive")

    @property
    def residual(self) -> float | None:
        """Return achieved minus target without changing the supplied profile."""

        return None if self.target is None else self.achieved - self.target


@dataclass(frozen=True, slots=True)
class IntegralResultLedger:
    """Complete plasma-current, poloidal-beta and inductance result ledger."""

    plasma_current: IntegralResult
    poloidal_beta: IntegralResult
    internal_inductance: IntegralResult

    def __post_init__(self) -> None:
        expected = (
            IntegralMoment.PLASMA_CURRENT,
            IntegralMoment.POLOIDAL_BETA,
            IntegralMoment.INTERNAL_INDUCTANCE,
        )
        actual = (
            self.plasma_current.moment,
            self.poloidal_beta.moment,
            self.internal_inductance.moment,
        )
        if actual != expected:
            raise FluxStateError("integral ledger rows do not match their fields")

    def rows(self) -> tuple[IntegralResult, ...]:
        """Return all ledger rows in stable physical order."""

        return (self.plasma_current, self.poloidal_beta, self.internal_inductance)


def _check_derivative(
    primitive: np.ndarray,
    declared: np.ndarray,
    total_flux: np.ndarray,
    *,
    transform: str,
    relative: float,
    absolute: float,
) -> None:
    if transform.endswith("pressure"):
        numerical = np.gradient(primitive, total_flux, edge_order=2)
    else:
        numerical = np.gradient(0.5 * primitive**2, total_flux, edge_order=2)
    error = np.abs(numerical - declared)
    allowed = absolute + relative * np.maximum(np.abs(numerical), np.abs(declared))
    if np.any(error > allowed):
        index = int(np.argmax(error - allowed))
        raise DerivativeConsistencyError(
            f"{transform} derivative disagrees at index {index}: "
            f"declared={declared[index]:.12g}, "
            f"finite_difference={numerical[index]:.12g}, "
            f"error={error[index]:.12g}, allowed={allowed[index]:.12g}"
        )


@dataclass(frozen=True, slots=True)
class FluxFunctionState:
    """Validated immutable state sufficient for one force-balance solve."""

    coordinate: RadialCoordinate
    pressure: np.ndarray
    dpressure_dpsi: np.ndarray
    f: np.ndarray
    f_df_dpsi: np.ndarray
    convention: ConventionProvenance
    ensemble: WeightedEnsembleIdentity
    handoff: TemporalProvenance
    provenance: StateProvenance
    integral_policy: IntegralConstraintPolicy
    integral_ledger: IntegralResultLedger
    domain_profiles: tuple[DomainProfile, ...] = ()
    flow: IsothermalToroidalFlow | None = None
    derivative_tolerance: DerivativeTolerance = DerivativeTolerance()
    pressure_units: str = "Pa"
    dpressure_dpsi_units: str = "Pa/Wb"
    f_units: str = "T m"
    f_df_dpsi_units: str = "T^2 m^2/Wb"

    def __post_init__(self) -> None:
        size = self.coordinate.values.size
        for label in ("pressure", "dpressure_dpsi", "f", "f_df_dpsi"):
            array = _immutable_array(getattr(self, label), label)
            if array.shape != (size,):
                raise FluxStateError(f"{label} must match the core radial coordinate")
            object.__setattr__(self, label, array)
        expected_units = ("Pa", "Pa/Wb", "T m", "T^2 m^2/Wb")
        actual_units = (
            self.pressure_units,
            self.dpressure_dpsi_units,
            self.f_units,
            self.f_df_dpsi_units,
        )
        if actual_units != expected_units:
            raise FluxStateError(f"source units must be exactly {expected_units}")
        direction = int(np.sign(self.coordinate.dtotal_flux_dpsi_n_wb))
        if direction != int(self.convention.flux_direction):
            raise FluxStateError(
                "coordinate Jacobian sign disagrees with COCOS sign provenance"
            )
        profiles = tuple(self.domain_profiles)
        object.__setattr__(self, "domain_profiles", profiles)
        domains = [item.domain for item in profiles]
        if len(domains) != len(set(domains)):
            raise FluxStateError("each topology class may carry at most one profile")
        if self.flow is not None and self.flow.omega_rad_per_s.shape != (size,):
            raise FluxStateError("flow profiles must match the core radial coordinate")
        self.validate_derivatives()
        self._validate_integral_ledger()

    def _validate_integral_ledger(self) -> None:
        targets = {item.moment: item for item in self.integral_policy.targets}
        for row in self.integral_ledger.rows():
            target = targets.get(row.moment)
            if target is None:
                if row.target is not None or row.tolerance is not None:
                    raise FluxStateError(
                        f"{row.moment.value} ledger target is absent from the policy"
                    )
            elif row.target != target.value or row.tolerance != target.tolerance:
                raise FluxStateError(
                    f"{row.moment.value} ledger target differs from the policy"
                )

    def validate_derivatives(self) -> None:
        """Check core and topology-qualified derivatives against primitives."""

        tolerance = self.derivative_tolerance
        _check_derivative(
            self.pressure,
            self.dpressure_dpsi,
            self.coordinate.total_flux_wb,
            transform="pressure",
            relative=tolerance.relative,
            absolute=tolerance.pressure_absolute_pa_per_wb,
        )
        _check_derivative(
            self.f,
            self.f_df_dpsi,
            self.coordinate.total_flux_wb,
            transform="F F'",
            relative=tolerance.relative,
            absolute=tolerance.f_df_absolute_t2m2_per_wb,
        )
        for profile in self.domain_profiles:
            _check_derivative(
                profile.pressure,
                profile.dpressure_dpsi,
                profile.coordinate.total_flux_wb,
                transform=f"{profile.domain.value} pressure",
                relative=tolerance.relative,
                absolute=tolerance.pressure_absolute_pa_per_wb,
            )
            _check_derivative(
                profile.f,
                profile.f_df_dpsi,
                profile.coordinate.total_flux_wb,
                transform=f"{profile.domain.value} F F'",
                relative=tolerance.relative,
                absolute=tolerance.f_df_absolute_t2m2_per_wb,
            )

    def profile_for(self, domain: PlasmaDomain) -> DomainProfile | None:
        """Return the profile keyed to one open topology class."""

        if domain is PlasmaDomain.CORE:
            raise FluxStateError("core arrays are direct FluxFunctionState fields")
        return next(
            (item for item in self.domain_profiles if item.domain is domain), None
        )

    def array_tree(self, array_namespace: ArrayNamespace = np) -> dict[str, Any]:
        """Return an array-only pytree using NumPy or a JAX-compatible namespace."""

        convert = array_namespace.asarray
        tree: dict[str, Any] = {
            "core": {
                "psi_n": convert(self.coordinate.values),
                "pressure": convert(self.pressure),
                "dpressure_dpsi": convert(self.dpressure_dpsi),
                "f": convert(self.f),
                "f_df_dpsi": convert(self.f_df_dpsi),
            },
            "domains": tuple(
                {
                    "psi_n": convert(item.coordinate.values),
                    "pressure": convert(item.pressure),
                    "dpressure_dpsi": convert(item.dpressure_dpsi),
                    "f": convert(item.f),
                    "f_df_dpsi": convert(item.f_df_dpsi),
                }
                for item in self.domain_profiles
            ),
        }
        if self.flow is not None:
            tree["flow"] = {
                "omega_rad_per_s": convert(self.flow.omega_rad_per_s),
                "temperature_ev": convert(self.flow.temperature_ev),
            }
        if self.integral_policy.prior_covariance is not None:
            tree["prior_covariance"] = convert(self.integral_policy.prior_covariance)
        return tree

    def to_dict(self) -> dict[str, Any]:
        """Return a lossless JSON-compatible representation."""

        return {
            "coordinate": _coordinate_to_dict(self.coordinate),
            "pressure": self.pressure.tolist(),
            "dpressure_dpsi": self.dpressure_dpsi.tolist(),
            "f": self.f.tolist(),
            "f_df_dpsi": self.f_df_dpsi.tolist(),
            "units": {
                "pressure": self.pressure_units,
                "dpressure_dpsi": self.dpressure_dpsi_units,
                "f": self.f_units,
                "f_df_dpsi": self.f_df_dpsi_units,
            },
            "convention": {
                "cocos": self.convention.cocos,
                "flux_direction": self.convention.flux_direction.name,
                "toroidal_field_sign": self.convention.toroidal_field_sign,
                "plasma_current_sign": self.convention.plasma_current_sign,
                "evidence": self.convention.evidence,
            },
            "ensemble": {
                "ensemble_id": self.ensemble.ensemble_id,
                "member_ids": list(self.ensemble.member_ids),
                "weights": list(self.ensemble.weights),
                "active_member": self.ensemble.active_member,
                "ensemble_digest": self.ensemble.ensemble_digest,
            },
            "handoff": _handoff_to_dict(self.handoff),
            "provenance": {
                "model_digest": self.provenance.model_digest,
                "checkpoint_digest": self.provenance.checkpoint_digest,
                "source_diagnostics": list(self.provenance.source_diagnostics),
                "validity_flags": [
                    {"name": item.name, "valid": item.valid, "reason": item.reason}
                    for item in self.provenance.validity_flags
                ],
            },
            "integral_policy": _policy_to_dict(self.integral_policy),
            "integral_ledger": [
                {
                    "moment": row.moment.value,
                    "achieved": row.achieved,
                    "target": row.target,
                    "tolerance": row.tolerance,
                    "units": row.units,
                }
                for row in self.integral_ledger.rows()
            ],
            "domain_profiles": [_domain_to_dict(item) for item in self.domain_profiles],
            "flow": _flow_to_dict(self.flow),
            "derivative_tolerance": {
                "relative": self.derivative_tolerance.relative,
                "pressure_absolute_pa_per_wb": (
                    self.derivative_tolerance.pressure_absolute_pa_per_wb
                ),
                "f_df_absolute_t2m2_per_wb": (
                    self.derivative_tolerance.f_df_absolute_t2m2_per_wb
                ),
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FluxFunctionState:
        """Restore and revalidate a state from :meth:`to_dict` output."""

        convention = payload["convention"]
        ensemble = payload["ensemble"]
        provenance = payload["provenance"]
        units = payload["units"]
        tolerance = payload["derivative_tolerance"]
        rows = {
            IntegralMoment(item["moment"]): IntegralResult(
                moment=IntegralMoment(item["moment"]),
                achieved=item["achieved"],
                target=item["target"],
                tolerance=item["tolerance"],
                units=item["units"],
            )
            for item in payload["integral_ledger"]
        }
        return cls(
            coordinate=_coordinate_from_dict(payload["coordinate"]),
            pressure=payload["pressure"],
            dpressure_dpsi=payload["dpressure_dpsi"],
            f=payload["f"],
            f_df_dpsi=payload["f_df_dpsi"],
            pressure_units=units["pressure"],
            dpressure_dpsi_units=units["dpressure_dpsi"],
            f_units=units["f"],
            f_df_dpsi_units=units["f_df_dpsi"],
            convention=ConventionProvenance(
                cocos=convention["cocos"],
                flux_direction=FluxDirection[convention["flux_direction"]],
                toroidal_field_sign=convention["toroidal_field_sign"],
                plasma_current_sign=convention["plasma_current_sign"],
                evidence=convention["evidence"],
            ),
            ensemble=WeightedEnsembleIdentity(
                ensemble_id=ensemble["ensemble_id"],
                member_ids=tuple(ensemble["member_ids"]),
                weights=tuple(ensemble["weights"]),
                active_member=ensemble["active_member"],
                ensemble_digest=ensemble["ensemble_digest"],
            ),
            handoff=_handoff_from_dict(payload["handoff"]),
            provenance=StateProvenance(
                model_digest=provenance["model_digest"],
                checkpoint_digest=provenance["checkpoint_digest"],
                source_diagnostics=tuple(provenance["source_diagnostics"]),
                validity_flags=tuple(
                    ValidityFlag(**item) for item in provenance["validity_flags"]
                ),
            ),
            integral_policy=_policy_from_dict(payload["integral_policy"]),
            integral_ledger=IntegralResultLedger(
                plasma_current=rows[IntegralMoment.PLASMA_CURRENT],
                poloidal_beta=rows[IntegralMoment.POLOIDAL_BETA],
                internal_inductance=rows[IntegralMoment.INTERNAL_INDUCTANCE],
            ),
            domain_profiles=tuple(
                _domain_from_dict(item) for item in payload["domain_profiles"]
            ),
            flow=_flow_from_dict(payload["flow"]),
            derivative_tolerance=DerivativeTolerance(**tolerance),
        )


def _coordinate_to_dict(value: RadialCoordinate) -> dict[str, Any]:
    return {
        "values": value.values.tolist(),
        "kind": value.kind.value,
        "coordinate_units": value.coordinate_units,
        "axis_total_flux_wb": value.axis_total_flux_wb,
        "separatrix_total_flux_wb": value.separatrix_total_flux_wb,
        "dtotal_flux_dpsi_n_wb": value.dtotal_flux_dpsi_n_wb,
        "total_flux_units": value.total_flux_units,
    }


def _coordinate_from_dict(payload: dict[str, Any]) -> RadialCoordinate:
    return RadialCoordinate(
        values=payload["values"],
        kind=CoordinateKind(payload["kind"]),
        coordinate_units=payload["coordinate_units"],
        axis_total_flux_wb=payload["axis_total_flux_wb"],
        separatrix_total_flux_wb=payload["separatrix_total_flux_wb"],
        dtotal_flux_dpsi_n_wb=payload["dtotal_flux_dpsi_n_wb"],
        total_flux_units=payload["total_flux_units"],
    )


def _domain_to_dict(value: DomainProfile) -> dict[str, Any]:
    return {
        "domain": value.domain.value,
        "coordinate": _coordinate_to_dict(value.coordinate),
        "pressure": value.pressure.tolist(),
        "dpressure_dpsi": value.dpressure_dpsi.tolist(),
        "f": value.f.tolist(),
        "f_df_dpsi": value.f_df_dpsi.tolist(),
        "policy": {
            "functional_form": value.policy.functional_form,
            "continuity_order": value.policy.continuity_order,
            "support_limit_psi_n": value.policy.support_limit_psi_n,
            "outer_condition": value.policy.outer_condition,
        },
    }


def _domain_from_dict(payload: dict[str, Any]) -> DomainProfile:
    return DomainProfile(
        domain=PlasmaDomain(payload["domain"]),
        coordinate=_coordinate_from_dict(payload["coordinate"]),
        pressure=payload["pressure"],
        dpressure_dpsi=payload["dpressure_dpsi"],
        f=payload["f"],
        f_df_dpsi=payload["f_df_dpsi"],
        policy=DomainProfilePolicy(**payload["policy"]),
    )


def _flow_to_dict(value: IsothermalToroidalFlow | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "thermodynamic_model": value.thermodynamic_model,
        "omega_rad_per_s": value.omega_rad_per_s.tolist(),
        "temperature_ev": value.temperature_ev.tolist(),
        "species": [
            {
                "name": item.name,
                "mass_amu": item.mass_amu,
                "charge_e": item.charge_e,
                "fraction": item.fraction,
            }
            for item in value.species
        ],
    }


def _flow_from_dict(payload: dict[str, Any] | None) -> IsothermalToroidalFlow | None:
    if payload is None:
        return None
    return IsothermalToroidalFlow(
        thermodynamic_model=payload["thermodynamic_model"],
        omega_rad_per_s=payload["omega_rad_per_s"],
        temperature_ev=payload["temperature_ev"],
        species=tuple(SpeciesAssumption(**item) for item in payload["species"]),
    )


def _handoff_to_dict(value: TemporalProvenance) -> dict[str, Any]:
    payload = {
        "kind": type(value).__name__,
        "valid_at": value.valid_at.isoformat(),
        "information_cutoff": value.information_cutoff.isoformat(),
        "produced_at": value.produced_at.isoformat(),
    }
    if isinstance(value, FixedLagSmoothing):
        payload["lag_seconds"] = value.lag_seconds
    elif isinstance(value, FullSequenceSmoothing):
        payload["sequence_start"] = value.sequence_start.isoformat()
        payload["sequence_end"] = value.sequence_end.isoformat()
    return payload


def _handoff_from_dict(payload: dict[str, Any]) -> TemporalProvenance:
    common = {
        "valid_at": datetime.fromisoformat(payload["valid_at"]),
        "information_cutoff": datetime.fromisoformat(payload["information_cutoff"]),
        "produced_at": datetime.fromisoformat(payload["produced_at"]),
    }
    classes = {
        "ForecastHandoff": ForecastHandoff,
        "AnalysisHandoff": AnalysisHandoff,
        "FixedLagSmoothing": FixedLagSmoothing,
        "FullSequenceSmoothing": FullSequenceSmoothing,
    }
    kind = payload["kind"]
    if kind not in classes:
        raise FluxStateError(f"unknown temporal provenance kind {kind!r}")
    if kind == "FixedLagSmoothing":
        return FixedLagSmoothing(**common, lag_seconds=payload["lag_seconds"])
    if kind == "FullSequenceSmoothing":
        return FullSequenceSmoothing(
            **common,
            sequence_start=datetime.fromisoformat(payload["sequence_start"]),
            sequence_end=datetime.fromisoformat(payload["sequence_end"]),
        )
    return classes[kind](**common)


def _policy_to_dict(value: IntegralConstraintPolicy) -> dict[str, Any]:
    return {
        "kind": value.kind.value,
        "free_amplitudes": [
            {"name": item.name, "units": item.units, "prior_mean": item.prior_mean}
            for item in value.free_amplitudes
        ],
        "targets": [
            {
                "moment": item.moment.value,
                "value": item.value,
                "tolerance": item.tolerance,
                "units": item.units,
            }
            for item in value.targets
        ],
        "prior_covariance": (
            None if value.prior_covariance is None else value.prior_covariance.tolist()
        ),
    }


def _policy_from_dict(payload: dict[str, Any]) -> IntegralConstraintPolicy:
    return IntegralConstraintPolicy(
        kind=IntegralPolicyKind(payload["kind"]),
        free_amplitudes=tuple(
            FreeAmplitude(**item) for item in payload["free_amplitudes"]
        ),
        targets=tuple(
            IntegralTarget(
                moment=IntegralMoment(item["moment"]),
                value=item["value"],
                tolerance=item["tolerance"],
                units=item["units"],
            )
            for item in payload["targets"]
        ),
        prior_covariance=payload["prior_covariance"],
    )
