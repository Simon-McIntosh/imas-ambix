"""Reusable physically consistent flux-function state fixtures."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from imas_ambix.fluxstate import (
    ConventionProvenance,
    DomainProfile,
    DomainProfilePolicy,
    FluxDirection,
    FluxFunctionState,
    ForecastHandoff,
    IntegralConstraintPolicy,
    IntegralMoment,
    IntegralResult,
    IntegralResultLedger,
    IsothermalToroidalFlow,
    PlasmaDomain,
    RadialCoordinate,
    SpeciesAssumption,
    StateProvenance,
    ValidityFlag,
    WeightedEnsembleIdentity,
)


def _coordinate(values: np.ndarray) -> RadialCoordinate:
    return RadialCoordinate(
        values=values,
        axis_total_flux_wb=0.2,
        separatrix_total_flux_wb=1.2,
        dtotal_flux_dpsi_n_wb=1.0,
    )


def _arrays(coordinate: RadialCoordinate) -> tuple[np.ndarray, ...]:
    psi = coordinate.total_flux_wb
    pressure = 20_000.0 - 10_000.0 * psi + 500.0 * psi**2
    dpressure = -10_000.0 + 1_000.0 * psi
    field_function = 2.0 + 0.1 * psi
    field_drive = 0.1 * field_function
    return pressure, dpressure, field_function, field_drive


def _domain(
    domain: PlasmaDomain,
    values: np.ndarray,
    form: str,
    outer_condition: str,
) -> DomainProfile:
    coordinate = _coordinate(values)
    pressure, dpressure, field_function, field_drive = _arrays(coordinate)
    return DomainProfile(
        domain=domain,
        coordinate=coordinate,
        pressure=pressure,
        dpressure_dpsi=dpressure,
        f=field_function,
        f_df_dpsi=field_drive,
        policy=DomainProfilePolicy(
            functional_form=form,
            continuity_order=1,
            support_limit_psi_n=float(max(values)),
            outer_condition=outer_condition,
        ),
    )


@pytest.fixture
def flux_state() -> FluxFunctionState:
    coordinate = _coordinate(np.linspace(0.0, 1.0, 9))
    pressure, dpressure, field_function, field_drive = _arrays(coordinate)
    cutoff = datetime(2026, 8, 17, 12, 0, tzinfo=UTC)
    return FluxFunctionState(
        coordinate=coordinate,
        pressure=pressure,
        dpressure_dpsi=dpressure,
        f=field_function,
        f_df_dpsi=field_drive,
        convention=ConventionProvenance(
            cocos=17,
            flux_direction=FluxDirection.INCREASES_OUTWARD,
            toroidal_field_sign=1,
            plasma_current_sign=-1,
            evidence="facility declaration sha256:convention-receipt",
        ),
        ensemble=WeightedEnsembleIdentity(
            ensemble_id="pulse-30420-window-12",
            member_ids=("member-a", "member-b", "member-c"),
            weights=(0.2, 0.3, 0.5),
            active_member=1,
            ensemble_digest="sha256:ensemble-receipt",
        ),
        handoff=ForecastHandoff(
            valid_at=cutoff + timedelta(milliseconds=10),
            information_cutoff=cutoff,
            produced_at=cutoff + timedelta(milliseconds=2),
        ),
        provenance=StateProvenance(
            model_digest="sha256:model",
            checkpoint_digest="sha256:checkpoint",
            source_diagnostics=("magnetics", "interferometry"),
            validity_flags=(
                ValidityFlag(
                    name="diagnostic-window-complete",
                    valid=True,
                    reason="all required channels cover the information cutoff",
                ),
            ),
        ),
        integral_policy=IntegralConstraintPolicy.absolute_sources(),
        integral_ledger=IntegralResultLedger(
            plasma_current=IntegralResult(
                moment=IntegralMoment.PLASMA_CURRENT,
                achieved=-780_000.0,
                target=None,
                tolerance=None,
                units="A",
            ),
            poloidal_beta=IntegralResult(
                moment=IntegralMoment.POLOIDAL_BETA,
                achieved=0.72,
                target=None,
                tolerance=None,
                units="1",
            ),
            internal_inductance=IntegralResult(
                moment=IntegralMoment.INTERNAL_INDUCTANCE,
                achieved=0.81,
                target=None,
                tolerance=None,
                units="1",
            ),
        ),
        domain_profiles=(
            _domain(
                PlasmaDomain.COMMON_SOL,
                np.linspace(1.0, 1.3, 5),
                "compact_cubic_decay",
                "zero_gradient_at_support",
            ),
            _domain(
                PlasmaDomain.PRIVATE_FLUX,
                np.linspace(0.7, 1.0, 5),
                "independent_low_pressure_cubic",
                "low_pressure_material_connection",
            ),
        ),
        flow=IsothermalToroidalFlow(
            omega_rad_per_s=np.linspace(8_000.0, 2_000.0, 9),
            temperature_ev=np.linspace(1_500.0, 200.0, 9),
            species=(
                SpeciesAssumption(
                    name="deuterium", mass_amu=2.014, charge_e=1.0, fraction=1.0
                ),
            ),
        ),
    )
