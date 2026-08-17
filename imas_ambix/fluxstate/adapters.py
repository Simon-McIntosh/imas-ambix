"""Producer adapters that stop at the deterministic Nova call boundary.

The transport adapter depends only on the public shape of Nova's reviewed
``CurrentDiffusion`` object.  No equilibrium solver, response kernel, or
measurement fit is imported into Ambix.  Sequential products retain their
member and temporal identities, while integral conditioning consumes a moment
linearization supplied by Nova without choosing a physics model there.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Protocol

import numpy as np

from imas_ambix.fluxstate.contract import (
    AnalysisHandoff,
    FixedLagSmoothing,
    FluxFunctionState,
    FluxStateError,
    ForecastHandoff,
    FreeAmplitude,
    FullSequenceSmoothing,
    IntegralConstraintPolicy,
    IntegralMoment,
    IntegralPolicyKind,
    IntegralResultLedger,
    IntegralTarget,
    OnlineHandoff,
    RadialCoordinate,
    StateProvenance,
    require_online_handoff,
)

REVIEWED_CURRENT_DIFFUSION_REVISION = "de3277a3238513b81be04dbc0980030b200ce420"


class AdapterContractError(FluxStateError):
    """Raised when a producer payload cannot satisfy the state boundary."""


class IntegralConditioningError(AdapterContractError):
    """Raised when a requested profile conditioning problem is ill-posed."""


class CurrentDiffusionGeometry(Protocol):
    """Reviewed public geometry fields consumed by the transport adapter."""

    f_face: Any
    flux_sign: float


class CurrentDiffusionLike(Protocol):
    """Structural boundary matching Nova's reviewed transport solver."""

    geometry: CurrentDiffusionGeometry

    def evolve(
        self,
        t_grid: np.ndarray,
        ip_of_t: np.ndarray,
        psi0_face: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Return the reviewed current-diffusion step mapping."""


def _readonly(value: Any, label: str, *, ndim: int = 1) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.ndim != ndim or not np.all(np.isfinite(array)):
        raise AdapterContractError(f"{label} must be a finite {ndim}-dimensional array")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class LearnedClosureCorrection:
    """Additive primitive corrections inferred upstream of the Nova handoff."""

    pressure_delta_pa: np.ndarray
    f_delta_tm: np.ndarray
    model_digest: str

    def __post_init__(self) -> None:
        pressure = _readonly(self.pressure_delta_pa, "pressure correction")
        field = _readonly(self.f_delta_tm, "field-function correction")
        object.__setattr__(self, "pressure_delta_pa", pressure)
        object.__setattr__(self, "f_delta_tm", field)
        if pressure.shape != field.shape:
            raise AdapterContractError("closure correction arrays must have one shape")
        if not self.model_digest.strip():
            raise AdapterContractError("closure correction needs a model digest")


@dataclass(frozen=True, slots=True)
class TransportForwardReceipt:
    """Immutable evidence that identifies one deterministic transport step."""

    nova_revision: str
    time_grid_s: np.ndarray
    requested_current_a: np.ndarray
    final_total_flux_wb: np.ndarray
    correction_digest: str | None

    def __post_init__(self) -> None:
        time_grid = _readonly(self.time_grid_s, "transport time grid")
        current = _readonly(self.requested_current_a, "requested plasma current")
        flux = _readonly(self.final_total_flux_wb, "final total flux")
        object.__setattr__(self, "time_grid_s", time_grid)
        object.__setattr__(self, "requested_current_a", current)
        object.__setattr__(self, "final_total_flux_wb", flux)
        if time_grid.size < 2 or not np.all(np.diff(time_grid) > 0.0):
            raise AdapterContractError("transport time grid must increase")
        if current.shape != time_grid.shape:
            raise AdapterContractError("current waveform must match the time grid")
        if flux.size < 3:
            raise AdapterContractError("transport state needs at least three faces")
        if self.nova_revision != REVIEWED_CURRENT_DIFFUSION_REVISION:
            raise AdapterContractError("transport receipt does not use reviewed Nova")
        if self.correction_digest is not None and not self.correction_digest.strip():
            raise AdapterContractError("correction digest cannot be blank")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible transport receipt."""

        return {
            "nova_revision": self.nova_revision,
            "time_grid_s": self.time_grid_s.tolist(),
            "requested_current_a": self.requested_current_a.tolist(),
            "final_total_flux_wb": self.final_total_flux_wb.tolist(),
            "correction_digest": self.correction_digest,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TransportForwardReceipt:
        """Restore a transport receipt and rerun its validation."""

        return cls(
            nova_revision=str(payload["nova_revision"]),
            time_grid_s=payload["time_grid_s"],
            requested_current_a=payload["requested_current_a"],
            final_total_flux_wb=payload["final_total_flux_wb"],
            correction_digest=payload["correction_digest"],
        )


@dataclass(frozen=True, slots=True)
class TransportForwardResult:
    """Adapted causal state and its transport-only evidence receipt."""

    state: FluxFunctionState
    receipt: TransportForwardReceipt

    def to_dict(self) -> dict[str, Any]:
        """Return a lossless transport-adapter payload."""

        return {"state": self.state.to_dict(), "receipt": self.receipt.to_dict()}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TransportForwardResult:
        """Restore both the state and receipt without invoking transport."""

        return cls(
            state=FluxFunctionState.from_dict(dict(payload["state"])),
            receipt=TransportForwardReceipt.from_dict(payload["receipt"]),
        )


@dataclass(frozen=True, slots=True)
class TransportForwardAdapter:
    """Convert a reviewed ``CurrentDiffusion`` step into a causal state."""

    nova_revision: str = REVIEWED_CURRENT_DIFFUSION_REVISION

    def __post_init__(self) -> None:
        if self.nova_revision != REVIEWED_CURRENT_DIFFUSION_REVISION:
            raise AdapterContractError(
                "CurrentDiffusion adapter requires the reviewed Nova revision"
            )

    def evolve(
        self,
        current_diffusion: CurrentDiffusionLike,
        *,
        time_grid_s: Any,
        requested_current_a: Any,
        source_state: FluxFunctionState,
        handoff: OnlineHandoff,
        correction: LearnedClosureCorrection | None = None,
        psi0_face: Any | None = None,
    ) -> TransportForwardResult:
        """Run transport and deterministically adapt its final face state."""

        online = require_online_handoff(handoff)
        time_grid = _readonly(time_grid_s, "transport time grid")
        current = _readonly(requested_current_a, "requested plasma current")
        if time_grid.shape != current.shape or time_grid.size < 2:
            raise AdapterContractError(
                "transport time and current waveforms need one non-trivial shape"
            )
        if not np.all(np.diff(time_grid) > 0.0):
            raise AdapterContractError("transport time grid must increase")
        initial = (
            None if psi0_face is None else _readonly(psi0_face, "initial total flux")
        )
        step = current_diffusion.evolve(time_grid, current, psi0_face=initial)
        required = {"psi_face", "psidot_face", "v_axis", "v_bdry"}
        missing = required.difference(step)
        if missing:
            raise AdapterContractError(
                f"CurrentDiffusion output is missing {sorted(missing)}"
            )
        history = _readonly(step["psi_face"], "transport flux history", ndim=2)
        if history.shape != (time_grid.size, source_state.coordinate.values.size):
            raise AdapterContractError(
                "transport flux history must match time and source radial grids"
            )
        final_flux = history[-1]
        span = float(final_flux[-1] - final_flux[0])
        if span == 0.0:
            raise AdapterContractError("transport axis and boundary flux must differ")
        psi_n = (final_flux - final_flux[0]) / span
        coordinate = RadialCoordinate(
            values=psi_n,
            axis_total_flux_wb=float(final_flux[0]),
            separatrix_total_flux_wb=float(final_flux[-1]),
            dtotal_flux_dpsi_n_wb=span,
        )
        geometry_sign = int(np.sign(float(current_diffusion.geometry.flux_sign)))
        if geometry_sign != int(source_state.convention.flux_direction):
            raise AdapterContractError(
                "CurrentDiffusion flux sign disagrees with source-state provenance"
            )
        field_function = _readonly(
            current_diffusion.geometry.f_face, "transport field function"
        )
        pressure = np.asarray(source_state.pressure)
        correction_digest = None
        if correction is not None:
            if correction.pressure_delta_pa.shape != pressure.shape:
                raise AdapterContractError(
                    "closure correction must match the source radial grid"
                )
            pressure = pressure + correction.pressure_delta_pa
            field_function = field_function + correction.f_delta_tm
            correction_digest = correction.model_digest
        if field_function.shape != pressure.shape:
            raise AdapterContractError(
                "CurrentDiffusion field function must match the source radial grid"
            )
        dpressure = np.gradient(pressure, final_flux, edge_order=2)
        field_drive = np.gradient(0.5 * field_function**2, final_flux, edge_order=2)
        signed_current = source_state.convention.plasma_current_sign * abs(
            float(current[-1])
        )
        current_row = replace(
            source_state.integral_ledger.plasma_current, achieved=signed_current
        )
        ledger = IntegralResultLedger(
            plasma_current=current_row,
            poloidal_beta=source_state.integral_ledger.poloidal_beta,
            internal_inductance=source_state.integral_ledger.internal_inductance,
        )
        diagnostics = source_state.provenance.source_diagnostics + (
            f"Nova CurrentDiffusion@{self.nova_revision}",
        )
        if correction_digest is not None:
            diagnostics += (f"learned closure@{correction_digest}",)
        provenance = StateProvenance(
            model_digest=source_state.provenance.model_digest,
            checkpoint_digest=source_state.provenance.checkpoint_digest,
            source_diagnostics=diagnostics,
            validity_flags=source_state.provenance.validity_flags,
        )
        state = replace(
            source_state,
            coordinate=coordinate,
            pressure=pressure,
            dpressure_dpsi=dpressure,
            f=field_function,
            f_df_dpsi=field_drive,
            handoff=online,
            provenance=provenance,
            integral_ledger=ledger,
        )
        receipt = TransportForwardReceipt(
            nova_revision=self.nova_revision,
            time_grid_s=time_grid,
            requested_current_a=current,
            final_total_flux_wb=final_flux,
            correction_digest=correction_digest,
        )
        return TransportForwardResult(state=state, receipt=receipt)


class SequentialProductKind(StrEnum):
    """Stable label derived from the temporal provenance type."""

    FORECAST = "forecast"
    ANALYSIS = "analysis"
    FIXED_LAG_SMOOTHING = "fixed_lag_smoothing"
    FULL_SEQUENCE_SMOOTHING = "full_sequence_smoothing"


_PRODUCT_KIND = {
    ForecastHandoff: SequentialProductKind.FORECAST,
    AnalysisHandoff: SequentialProductKind.ANALYSIS,
    FixedLagSmoothing: SequentialProductKind.FIXED_LAG_SMOOTHING,
    FullSequenceSmoothing: SequentialProductKind.FULL_SEQUENCE_SMOOTHING,
}


@dataclass(frozen=True, slots=True)
class SequentialEstimatorBatch:
    """Complete weighted ensemble retaining one estimator-product label."""

    members: tuple[FluxFunctionState, ...]

    def __post_init__(self) -> None:
        members = tuple(self.members)
        object.__setattr__(self, "members", members)
        if not members:
            raise AdapterContractError("sequential batch cannot be empty")
        reference = members[0]
        identity = reference.ensemble
        temporal_type = type(reference.handoff)
        active = []
        for member in members:
            other = member.ensemble
            if (
                other.ensemble_id != identity.ensemble_id
                or other.member_ids != identity.member_ids
                or other.weights != identity.weights
                or other.ensemble_digest != identity.ensemble_digest
            ):
                raise AdapterContractError(
                    "sequential members must share one weighted-ensemble identity"
                )
            if type(member.handoff) is not temporal_type:
                raise AdapterContractError(
                    "sequential batch cannot collapse unlike temporal labels"
                )
            if member.handoff != reference.handoff:
                raise AdapterContractError(
                    "sequential members must share causal timestamp identity"
                )
            active.append(other.active_member)
        if len(members) != len(identity.member_ids) or set(active) != set(
            range(len(identity.member_ids))
        ):
            raise AdapterContractError(
                "sequential batch must preserve every ensemble member exactly once"
            )

    @property
    def kind(self) -> SequentialProductKind:
        """Return the non-collapsed temporal product label."""

        return _PRODUCT_KIND[type(self.members[0].handoff)]

    def online_handoff(self) -> SequentialEstimatorBatch:
        """Admit forecast or analysis members and reject smoothing products."""

        for member in self.members:
            require_online_handoff(member.handoff)
        return self

    def to_dict(self) -> dict[str, Any]:
        """Return a lossless JSON-compatible member batch."""

        return {
            "kind": self.kind.value,
            "members": [member.to_dict() for member in self.members],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SequentialEstimatorBatch:
        """Restore all members and verify the stored label against their types."""

        batch = cls(
            members=tuple(
                FluxFunctionState.from_dict(dict(item)) for item in payload["members"]
            )
        )
        if batch.kind is not SequentialProductKind(payload["kind"]):
            raise AdapterContractError("stored sequential label disagrees with members")
        return batch


@dataclass(frozen=True, slots=True)
class SequentialEstimatorAdapter:
    """Create a validated member batch without taking an ensemble mean."""

    def batch(self, members: tuple[FluxFunctionState, ...]) -> SequentialEstimatorBatch:
        """Preserve every member and its temporal provenance."""

        return SequentialEstimatorBatch(members=members)


@dataclass(frozen=True, slots=True)
class IntegralConditioningRequest:
    """Named profile family, observations, and Ambix prior for conditioning."""

    free_amplitudes: tuple[FreeAmplitude, ...]
    targets: tuple[IntegralTarget, ...]
    prior_covariance: np.ndarray | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "free_amplitudes", tuple(self.free_amplitudes))
        object.__setattr__(self, "targets", tuple(self.targets))
        if self.prior_covariance is not None:
            object.__setattr__(
                self,
                "prior_covariance",
                _readonly(self.prior_covariance, "conditioning prior", ndim=2),
            )


@dataclass(frozen=True, slots=True)
class IntegralConditioningResult:
    """Profile-amplitude posterior and exact linearized moment residuals."""

    policy: IntegralConstraintPolicy
    posterior_amplitudes: np.ndarray
    amplitude_delta: np.ndarray
    posterior_covariance: np.ndarray
    predicted_moments: np.ndarray
    residuals: np.ndarray
    jacobian_condition: float
    jacobian_rank: int

    def __post_init__(self) -> None:
        for label in (
            "posterior_amplitudes",
            "amplitude_delta",
            "predicted_moments",
            "residuals",
        ):
            object.__setattr__(self, label, _readonly(getattr(self, label), label))
        object.__setattr__(
            self,
            "posterior_covariance",
            _readonly(self.posterior_covariance, "posterior covariance", ndim=2),
        )
        if not np.isfinite(self.jacobian_condition):
            raise IntegralConditioningError("Jacobian condition must be finite")


@dataclass(frozen=True, slots=True)
class IntegralConstraintConditioner:
    """Condition named profile amplitudes from Nova moment residuals/Jacobians."""

    maximum_condition: float = 1e10

    def condition(
        self,
        request: IntegralConditioningRequest,
        *,
        achieved_moments: dict[IntegralMoment, float],
        jacobian: Any,
    ) -> IntegralConditioningResult:
        """Solve a square, locally unique linearized conditioning problem."""

        problems = []
        amplitudes = request.free_amplitudes
        targets = request.targets
        if not amplitudes or not targets:
            problems.append("conditioning needs named amplitudes and targets")
        if len(amplitudes) != len(targets):
            problems.append(
                "one named profile degree of freedom is required per moment"
            )
        if request.prior_covariance is None:
            problems.append("an Ambix prior covariance is required")
        if problems:
            raise IntegralConditioningError("; ".join(problems))
        policy = IntegralConstraintPolicy(
            kind=IntegralPolicyKind.MOMENT_CLOSURE,
            free_amplitudes=amplitudes,
            targets=targets,
            prior_covariance=request.prior_covariance,
        )
        matrix = _readonly(jacobian, "moment Jacobian", ndim=2)
        expected = (len(targets), len(amplitudes))
        if matrix.shape != expected:
            raise IntegralConditioningError(
                f"moment Jacobian must have shape {expected}"
            )
        rank = int(np.linalg.matrix_rank(matrix))
        condition = float(np.linalg.cond(matrix))
        if rank != len(amplitudes) or condition > self.maximum_condition:
            raise IntegralConditioningError(
                "moment Jacobian does not establish a unique conditioned state"
            )
        try:
            achieved = np.asarray(
                [float(achieved_moments[item.moment]) for item in targets],
                dtype=np.float64,
            )
        except KeyError as error:
            raise IntegralConditioningError(
                f"missing achieved moment {error.args[0].value}"
            ) from error
        if not np.all(np.isfinite(achieved)):
            raise IntegralConditioningError("achieved moments must be finite")
        desired = np.asarray([item.value for item in targets], dtype=np.float64)
        prior_mean = np.asarray(
            [item.prior_mean for item in amplitudes], dtype=np.float64
        )
        delta = np.linalg.solve(matrix, desired - achieved)
        posterior = prior_mean + delta
        predicted = achieved + matrix @ delta
        residuals = predicted - desired
        observation_covariance = np.diag(
            np.asarray([item.tolerance**2 for item in targets], dtype=np.float64)
        )
        prior = np.asarray(policy.prior_covariance)
        posterior_covariance = np.linalg.inv(
            np.linalg.inv(prior)
            + matrix.T @ np.linalg.inv(observation_covariance) @ matrix
        )
        return IntegralConditioningResult(
            policy=policy,
            posterior_amplitudes=posterior,
            amplitude_delta=delta,
            posterior_covariance=posterior_covariance,
            predicted_moments=predicted,
            residuals=residuals,
            jacobian_condition=condition,
            jacobian_rank=rank,
        )
