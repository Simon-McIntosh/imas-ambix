"""Array-only handoff from Ambix flux states to deterministic consumers.

This module deliberately has no Nova API dependency.  It freezes the quantities
that a forward-equilibrium consumer needs, identifies the supplied source by
content, and turns returned integral observations into residuals.  Selecting
an equilibrium algorithm and evaluating magnetic-field kernels remain the
consumer's responsibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import TYPE_CHECKING

import numpy as np

from imas_ambix.fluxstate.adapters import REVIEWED_CURRENT_DIFFUSION_REVISION
from imas_ambix.fluxstate.contract import (
    FluxFunctionState,
    IntegralMoment,
    IntegralPolicyKind,
    PlasmaDomain,
    require_online_handoff,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

NOVA_FORWARD_REVISION = REVIEWED_CURRENT_DIFFUSION_REVISION


def _frozen(value: object) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class ConsumerDomainProfile:
    """One topology-qualified source profile at the consumer boundary."""

    domain: PlasmaDomain
    psi_n: np.ndarray
    pressure_pa: np.ndarray
    dpressure_dtotal_flux_pa_per_wb: np.ndarray
    f_tm: np.ndarray
    f_df_dtotal_flux_t2m2_per_wb: np.ndarray
    functional_form: str
    continuity_order: int
    support_limit_psi_n: float
    outer_condition: str


@dataclass(frozen=True, slots=True)
class NovaForwardPayload:
    """Immutable numerical and provenance payload consumed by Nova."""

    psi_n: np.ndarray
    total_flux_wb: np.ndarray
    pressure_pa: np.ndarray
    dpressure_dtotal_flux_pa_per_wb: np.ndarray
    f_tm: np.ndarray
    f_df_dtotal_flux_t2m2_per_wb: np.ndarray
    domains: tuple[ConsumerDomainProfile, ...]
    omega_rad_per_s: np.ndarray | None
    temperature_ev: np.ndarray | None
    cocos: int
    plasma_current_sign: int
    toroidal_field_sign: int
    source_policy: str
    requested_moments: tuple[IntegralMoment, ...]
    ensemble_id: str
    member_id: str
    source_digest: str


@dataclass(frozen=True, slots=True)
class IntegralResidual:
    """One returned integral observation without any source adjustment."""

    moment: IntegralMoment
    achieved: float
    target: float | None
    tolerance: float | None
    residual: float | None
    units: str


@dataclass(frozen=True, slots=True)
class NovaForwardReceipt:
    """Consumer result identity and the unmodified-source moment ledger."""

    nova_revision: str
    source_digest: str
    residuals: tuple[IntegralResidual, ...]


def _source_digest(state: FluxFunctionState) -> str:
    digest = sha256()
    for array in (
        state.coordinate.values,
        state.coordinate.total_flux_wb,
        state.pressure,
        state.dpressure_dpsi,
        state.f,
        state.f_df_dpsi,
    ):
        digest.update(np.asarray(array, dtype="<f8").tobytes())
    for profile in state.domain_profiles:
        digest.update(profile.domain.value.encode())
        for array in (
            profile.coordinate.values,
            profile.pressure,
            profile.dpressure_dpsi,
            profile.f,
            profile.f_df_dpsi,
        ):
            digest.update(np.asarray(array, dtype="<f8").tobytes())
    if state.flow is not None:
        digest.update(np.asarray(state.flow.omega_rad_per_s, dtype="<f8").tobytes())
        digest.update(np.asarray(state.flow.temperature_ev, dtype="<f8").tobytes())
    return digest.hexdigest()


def to_nova_forward_payload(state: FluxFunctionState) -> NovaForwardPayload:
    """Freeze one causal state at the Nova call boundary.

    Integral targets are carried as requests only.  Both absolute-source and
    upstream-conditioned states reach the deterministic solve with a unit
    source amplitude; this boundary never renormalises a supplied profile.
    """

    require_online_handoff(state.handoff)
    domains = tuple(
        ConsumerDomainProfile(
            domain=profile.domain,
            psi_n=_frozen(profile.coordinate.values),
            pressure_pa=_frozen(profile.pressure),
            dpressure_dtotal_flux_pa_per_wb=_frozen(profile.dpressure_dpsi),
            f_tm=_frozen(profile.f),
            f_df_dtotal_flux_t2m2_per_wb=_frozen(profile.f_df_dpsi),
            functional_form=profile.policy.functional_form,
            continuity_order=profile.policy.continuity_order,
            support_limit_psi_n=profile.policy.support_limit_psi_n,
            outer_condition=profile.policy.outer_condition,
        )
        for profile in state.domain_profiles
    )
    flow = state.flow
    policy = (
        "absolute"
        if state.integral_policy.kind is IntegralPolicyKind.ABSOLUTE_SOURCE
        else "conditioned_upstream"
    )
    return NovaForwardPayload(
        psi_n=_frozen(state.coordinate.values),
        total_flux_wb=_frozen(state.coordinate.total_flux_wb),
        pressure_pa=_frozen(state.pressure),
        dpressure_dtotal_flux_pa_per_wb=_frozen(state.dpressure_dpsi),
        f_tm=_frozen(state.f),
        f_df_dtotal_flux_t2m2_per_wb=_frozen(state.f_df_dpsi),
        domains=domains,
        omega_rad_per_s=None if flow is None else _frozen(flow.omega_rad_per_s),
        temperature_ev=None if flow is None else _frozen(flow.temperature_ev),
        cocos=state.convention.cocos,
        plasma_current_sign=state.convention.plasma_current_sign,
        toroidal_field_sign=state.convention.toroidal_field_sign,
        source_policy=policy,
        requested_moments=tuple(item.moment for item in state.integral_policy.targets),
        ensemble_id=state.ensemble.ensemble_id,
        member_id=state.ensemble.member_id,
        source_digest=_source_digest(state),
    )


def record_nova_integrals(
    state: FluxFunctionState,
    achieved: Mapping[IntegralMoment, float],
    *,
    nova_revision: str = NOVA_FORWARD_REVISION,
) -> NovaForwardReceipt:
    """Record achieved moments and residuals without modifying profiles."""

    if nova_revision != NOVA_FORWARD_REVISION:
        raise ValueError("Nova forward result does not use the installed revision")
    targets = {item.moment: item for item in state.integral_policy.targets}
    residuals = []
    for row in state.integral_ledger.rows():
        value = float(achieved[row.moment])
        if not np.isfinite(value):
            raise ValueError(f"Nova returned a non-finite {row.moment.value}")
        target = targets.get(row.moment)
        residuals.append(
            IntegralResidual(
                moment=row.moment,
                achieved=value,
                target=None if target is None else target.value,
                tolerance=None if target is None else target.tolerance,
                residual=None if target is None else value - target.value,
                units=row.units,
            )
        )
    return NovaForwardReceipt(
        nova_revision=nova_revision,
        source_digest=_source_digest(state),
        residuals=tuple(residuals),
    )
