"""Cross-repository fixtures for the Ambix-to-Nova forward boundary."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from importlib.metadata import distribution
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from urllib.parse import unquote, urlsplit

import numpy as np
import pytest
from scipy.constants import atomic_mass, electron_volt

from imas_ambix.fluxstate import (
    DerivativeTolerance,
    DomainProfile,
    DomainProfilePolicy,
    FreeAmplitude,
    IntegralConstraintPolicy,
    IntegralMoment,
    IntegralPolicyKind,
    IntegralResult,
    IntegralResultLedger,
    IntegralTarget,
    IsothermalToroidalFlow,
    PlasmaDomain,
    RadialCoordinate,
    SpeciesAssumption,
)
from imas_ambix.fluxstate.consumer_contract import (
    NOVA_FORWARD_REVISION,
    record_nova_integrals,
    to_nova_forward_payload,
)

NOVA_DISTRIBUTION = distribution("nova-stella")
NOVA_DIRECT_URL = json.loads(NOVA_DISTRIBUTION.read_text("direct_url.json"))
NOVA_SOURCE = Path(unquote(urlsplit(NOVA_DIRECT_URL["url"]).path))
CURRENT_LEDGER_RTOL = 1.0e-12
FLUX_PARITY_RTOL = 1.0e-6
FORCE_BALANCE_TOLERANCE = 0.1
CORE_PSI_N = np.linspace(0.0, 1.0, 33)


def _load_path(name: str, path: Path):
    spec = spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_reference(name: str, filename: str):
    return _load_path(name, NOVA_SOURCE / "tests" / filename)


@pytest.fixture(scope="module")
def nova_references():
    sys.path.insert(0, str(NOVA_SOURCE))
    static = _load_reference(
        "nova_reference_static", "test_equilibrium_forward_solve.py"
    )
    _load_path(
        "tests.rotating_equilibrium_references",
        NOVA_SOURCE / "tests" / "rotating_equilibrium_references.py",
    )
    rotation = _load_reference(
        "nova_reference_rotation", "test_equilibrium_rotation.py"
    )
    sol = _load_reference("nova_reference_sol", "test_equilibrium_sol.py")
    return static, rotation, sol


def test_nova_reference_source_is_the_installed_editable_checkout():
    assert NOVA_DIRECT_URL["dir_info"]["editable"] is True
    assert NOVA_SOURCE.is_dir()
    assert len(NOVA_FORWARD_REVISION) == 40
    int(NOVA_FORWARD_REVISION, 16)


def _integrate_linear_gradient(psi_n, gradient, *, offset):
    coefficients = np.polynomial.polynomial.polyfit(psi_n, gradient, 1)
    primitive = np.polynomial.polynomial.polyint(coefficients)
    return (
        offset
        + np.polynomial.polynomial.polyval(psi_n, primitive)
        - np.polynomial.polynomial.polyval(psi_n[0], primitive)
    )


def _integration_tolerance(pressure, p_prime, field_function, ff_prime, domains):
    pressure_errors = [
        np.max(np.abs(np.gradient(pressure, CORE_PSI_N, edge_order=2) - p_prime))
    ]
    field_errors = [
        np.max(
            np.abs(
                np.gradient(0.5 * field_function**2, CORE_PSI_N, edge_order=2)
                - ff_prime
            )
        )
    ]
    for domain in domains:
        flux = domain.coordinate.total_flux_wb
        pressure_errors.append(
            np.max(
                np.abs(
                    np.gradient(domain.pressure, flux, edge_order=2)
                    - domain.dpressure_dpsi
                )
            )
        )
        field_errors.append(
            np.max(
                np.abs(
                    np.gradient(0.5 * domain.f**2, flux, edge_order=2)
                    - domain.f_df_dpsi
                )
            )
        )
    return DerivativeTolerance(
        relative=1.0e-10,
        pressure_absolute_pa_per_wb=1.01 * max(pressure_errors) + 1.0e-8,
        f_df_absolute_t2m2_per_wb=1.01 * max(field_errors) + 1.0e-12,
    )


def _state_from_source(base, source, *, flow, domain_profiles=()):
    psi_n = CORE_PSI_N
    coordinate = RadialCoordinate(
        values=psi_n,
        axis_total_flux_wb=0.0,
        separatrix_total_flux_wb=1.0,
        dtotal_flux_dpsi_n_wb=1.0,
    )
    p_prime = np.asarray(source.core.p_prime(psi_n), dtype=float)
    ff_prime = np.asarray(source.core.ff_prime(psi_n), dtype=float)
    pressure = _integrate_linear_gradient(psi_n, p_prime, offset=2.0e6)
    half_f_squared = _integrate_linear_gradient(psi_n, ff_prime, offset=12.5)
    field_function = np.sqrt(2.0 * half_f_squared)
    profiles = tuple(domain_profiles)
    return replace(
        base,
        coordinate=coordinate,
        pressure=pressure,
        dpressure_dpsi=p_prime,
        f=field_function,
        f_df_dpsi=ff_prime,
        convention=replace(base.convention, plasma_current_sign=1),
        domain_profiles=profiles,
        flow=flow,
        derivative_tolerance=_integration_tolerance(
            pressure, p_prime, field_function, ff_prime, profiles
        ),
    )


def _flow_from_analytic_case(reference, case):
    closure = reference.rotation_closure(case)
    temperature_j = np.asarray(closure.temperature(CORE_PSI_N), dtype=float)
    return IsothermalToroidalFlow(
        omega_rad_per_s=np.asarray(closure.angular_frequency(CORE_PSI_N), dtype=float),
        temperature_ev=temperature_j / electron_volt,
        species=(
            SpeciesAssumption(
                name="deuterium",
                mass_amu=case.mean_particle_mass / atomic_mass,
                charge_e=1.0,
                fraction=1.0,
            ),
        ),
    )


def _domain_from_source(source, support):
    psi_n = np.linspace(1.0, 1.0 + support, 33)
    coordinate = RadialCoordinate(
        values=psi_n,
        axis_total_flux_wb=0.0,
        separatrix_total_flux_wb=1.0,
        dtotal_flux_dpsi_n_wb=1.0,
    )
    p_prime = np.asarray(source.p_prime(psi_n), dtype=float)
    ff_prime = np.asarray(source.ff_prime(psi_n), dtype=float)
    pressure_coefficients = np.polynomial.polynomial.polyfit(psi_n, p_prime, 3)
    pressure_primitive = np.polynomial.polynomial.polyint(pressure_coefficients)
    pressure = (
        2.0e6
        + np.polynomial.polynomial.polyval(psi_n, pressure_primitive)
        - np.polynomial.polynomial.polyval(psi_n[0], pressure_primitive)
    )
    field_coefficients = np.polynomial.polynomial.polyfit(psi_n, ff_prime, 3)
    field_primitive = np.polynomial.polynomial.polyint(field_coefficients)
    half_f_squared = (
        12.5
        + np.polynomial.polynomial.polyval(psi_n, field_primitive)
        - np.polynomial.polynomial.polyval(psi_n[0], field_primitive)
    )
    return DomainProfile(
        domain=PlasmaDomain.COMMON_SOL,
        coordinate=coordinate,
        pressure=pressure,
        dpressure_dpsi=p_prime,
        f=np.sqrt(2.0 * half_f_squared),
        f_df_dpsi=ff_prime,
        policy=DomainProfilePolicy(
            functional_form="hermite_polynomial",
            continuity_order=1,
            support_limit_psi_n=1.0 + support,
            outer_condition="compact_zero_source",
        ),
    )


def _assert_equilibrium(result, *, residual_tolerance, force_tolerance):
    assert bool(result.finite.passed)
    assert float(result.fixed_point.residual) < residual_tolerance
    conservation = result.conservation
    assert int(conservation.checked_cells) > 20
    assert float(conservation.relative_force) < force_tolerance
    assert float(conservation.relative_grad_shafranov) < force_tolerance
    ledger = result.ledger
    domain_sum = (
        ledger.core + ledger.common_sol + ledger.private_flux + ledger.excluded_material
    )
    np.testing.assert_allclose(ledger.total, domain_sum, rtol=CURRENT_LEDGER_RTOL)
    np.testing.assert_allclose(
        result.moments.plasma_current, ledger.core, rtol=CURRENT_LEDGER_RTOL
    )


@pytest.fixture(scope="module")
def solved_contracts(nova_references, flux_state):
    static_ref, rotation_ref, sol_ref = nova_references

    static_machine = static_ref.machine.__wrapped__()
    static_profile, static_seed, static_vacuum = static_machine
    static_result = static_profile.solve(
        static_seed, route="anderson", evaluations=static_ref.EVALUATIONS
    )
    static_state = _state_from_source(
        flux_state, static_profile.source, flow=None, domain_profiles=()
    )

    _lattice, rungs = rotation_ref.ladder.__wrapped__()
    rotating_profile, rotating_seed, case = rungs[0.35]
    rotating_result = rotating_profile.solve(
        rotating_seed, route="anderson", evaluations=rotation_ref.EVALUATIONS
    )
    rotating_state = _state_from_source(
        flux_state,
        rotating_profile.source,
        flow=_flow_from_analytic_case(rotation_ref, case),
        domain_profiles=(),
    )

    build_sol, sol_seed = sol_ref.machine.__wrapped__()
    sol_core = sol_ref._core()
    sol_open = sol_ref._policy().extend(sol_core, sol_ref.PlasmaDomain.COMMON_SOL)
    sol_profile = build_sol(sol_core, common_sol=sol_open)
    sol_result = sol_profile.solve(
        sol_seed, route="anderson", evaluations=sol_ref.EVALUATIONS
    )
    sol_domain = _domain_from_source(sol_open, sol_ref.SUPPORT)
    sol_state = _state_from_source(
        flux_state, sol_profile.source, flow=None, domain_profiles=(sol_domain,)
    )

    return {
        "static": (
            static_state,
            static_profile,
            static_seed,
            static_vacuum,
            static_result,
        ),
        "rotating": (
            rotating_state,
            rotating_profile,
            rotating_seed,
            None,
            rotating_result,
        ),
        "sol": (sol_state, sol_profile, sol_seed, None, sol_result),
    }


@pytest.mark.parametrize("kind", ("static", "rotating", "sol"))
def test_manufactured_states_close_the_nova_force_and_current_ledgers(
    solved_contracts, kind
):
    state, profile, _seed, _vacuum, result = solved_contracts[kind]
    payload = to_nova_forward_payload(state)
    np.testing.assert_allclose(
        payload.dpressure_dtotal_flux_pa_per_wb,
        np.asarray(profile.source.core.p_prime(payload.psi_n)),
        rtol=1.0e-12,
    )
    np.testing.assert_allclose(
        payload.f_df_dtotal_flux_t2m2_per_wb,
        np.asarray(profile.source.core.ff_prime(payload.psi_n)),
        rtol=1.0e-12,
    )
    if kind == "rotating":
        assert payload.omega_rad_per_s is not None
        assert result.rotation.closure_name == "isothermal_surface"
        closure = profile.source.core.rotation
        np.testing.assert_allclose(
            payload.omega_rad_per_s,
            np.asarray(closure.angular_frequency(payload.psi_n)),
            rtol=1.0e-12,
        )
        np.testing.assert_allclose(
            payload.temperature_ev * electron_volt,
            np.asarray(closure.temperature(payload.psi_n)),
            rtol=1.0e-12,
        )
    if kind == "sol":
        assert tuple(item.domain for item in payload.domains) == (
            PlasmaDomain.COMMON_SOL,
        )
        domain = payload.domains[0]
        np.testing.assert_allclose(
            domain.dpressure_dtotal_flux_pa_per_wb,
            np.asarray(profile.source.common_sol.p_prime(domain.psi_n)),
            rtol=1.0e-12,
        )
        np.testing.assert_allclose(
            domain.f_df_dtotal_flux_t2m2_per_wb,
            np.asarray(profile.source.common_sol.ff_prime(domain.psi_n)),
            rtol=1.0e-12,
        )
        assert float(result.ledger.common_sol) != 0.0
    _assert_equilibrium(
        result,
        residual_tolerance=1.0e-5 if kind == "sol" else 1.0e-6,
        force_tolerance=0.05 if kind == "sol" else FORCE_BALANCE_TOLERANCE,
    )


def test_absolute_sources_and_inconsistent_targets_never_mutate_profiles(
    solved_contracts,
):
    state, _profile, _seed, _vacuum, result = solved_contracts["static"]
    payload = to_nova_forward_payload(state)
    assert payload.source_policy == "absolute"
    assert result.normalisation.policy_name == "absolute"
    assert float(result.normalisation.amplitude) == 1.0
    assert not bool(result.normalisation.rescaled)

    achieved = {
        IntegralMoment.PLASMA_CURRENT: float(result.moments.plasma_current),
        IntegralMoment.POLOIDAL_BETA: float(result.moments.poloidal_beta),
        IntegralMoment.INTERNAL_INDUCTANCE: float(result.moments.internal_inductance),
    }
    targets = tuple(
        IntegralTarget(
            moment=moment,
            value=value + (1.0e4 if moment is IntegralMoment.PLASMA_CURRENT else 0.1),
            tolerance=1.0,
            units="A" if moment is IntegralMoment.PLASMA_CURRENT else "1",
        )
        for moment, value in achieved.items()
    )
    policy = IntegralConstraintPolicy(
        kind=IntegralPolicyKind.MOMENT_CLOSURE,
        free_amplitudes=(
            FreeAmplitude("pressure_scale", "1", 1.0),
            FreeAmplitude("diamagnetic_scale", "1", 1.0),
            FreeAmplitude("edge_shape", "1", 0.0),
        ),
        targets=targets,
        prior_covariance=np.eye(3),
    )
    rows = tuple(
        IntegralResult(
            moment=target.moment,
            achieved=achieved[target.moment],
            target=target.value,
            tolerance=target.tolerance,
            units=target.units,
        )
        for target in targets
    )
    conditioned = replace(
        state,
        integral_policy=policy,
        integral_ledger=IntegralResultLedger(*rows),
    )
    before = to_nova_forward_payload(conditioned)
    receipt = record_nova_integrals(conditioned, achieved)
    after = to_nova_forward_payload(conditioned)
    assert all(item.residual not in (None, 0.0) for item in receipt.residuals)
    assert receipt.source_digest == before.source_digest == after.source_digest
    np.testing.assert_array_equal(before.pressure_pa, after.pressure_pa)
    np.testing.assert_array_equal(before.f_tm, after.f_tm)


def test_eager_jitted_and_batched_members_agree(solved_contracts, nova_references):
    static_ref, _rotation_ref, _sol_ref = nova_references
    _state, profile, seed, vacuum, converged = solved_contracts["static"]
    eager = profile.solve(
        converged.flux,
        route="host",
        evaluations=static_ref.TRANSIENT,
        tolerance=0.0,
    )
    traced = static_ref.jax.jit(
        lambda values: profile.solve(
            values, route="picard", evaluations=static_ref.TRANSIENT
        )
    )(converged.flux)
    factors = (0.85, 1.0, 1.15)
    seeds = static_ref.jnp.stack(
        [vacuum + factor * (seed - vacuum) for factor in factors]
    )
    batched = static_ref.jax.jit(
        lambda values: profile.solve_batch(
            values, route="anderson", evaluations=static_ref.EVALUATIONS
        )
    )(seeds)
    scale = float(np.max(np.abs(np.asarray(converged.flux))))
    for candidate in (eager.flux, traced.flux):
        relative = float(np.max(np.abs(np.asarray(candidate - converged.flux)))) / scale
        assert relative < FLUX_PARITY_RTOL
    implementation_parity = (
        float(np.max(np.abs(np.asarray(eager.flux - traced.flux)))) / scale
    )
    assert implementation_parity < static_ref.IMPLEMENTATION_PARITY
    for index in range(len(factors)):
        single = profile.solve(
            seeds[index], route="anderson", evaluations=static_ref.EVALUATIONS
        )
        relative = (
            float(np.max(np.abs(np.asarray(batched.flux[index] - single.flux)))) / scale
        )
        assert relative < FLUX_PARITY_RTOL
    assert batched.flux.shape == (3, seed.size)
    assert bool(static_ref.jnp.all(batched.finite.passed))
