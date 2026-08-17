"""Round-trip, immutability, and array-runtime contract checks."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from imas_ambix.fluxstate import (
    CoordinateKind,
    FluxFunctionState,
    IntegralPolicyKind,
    PlasmaDomain,
)


def test_json_compatible_round_trip_preserves_physical_identity(
    flux_state: FluxFunctionState,
):
    payload = flux_state.to_dict()
    restored = FluxFunctionState.from_dict(payload)

    assert restored.coordinate.kind is CoordinateKind.NORMALIZED_POLOIDAL_FLUX
    assert restored.coordinate.coordinate_units == "1"
    assert restored.coordinate.total_flux_units == "Wb"
    assert restored.pressure_units == "Pa"
    assert restored.dpressure_dpsi_units == "Pa/Wb"
    assert restored.f_units == "T m"
    assert restored.f_df_dpsi_units == "T^2 m^2/Wb"
    assert restored.convention == flux_state.convention
    assert restored.ensemble == flux_state.ensemble
    assert restored.ensemble.member_id == "member-b"
    assert restored.integral_policy.kind is IntegralPolicyKind.ABSOLUTE_SOURCE
    assert restored.integral_ledger == flux_state.integral_ledger
    assert [item.domain for item in restored.domain_profiles] == [
        PlasmaDomain.COMMON_SOL,
        PlasmaDomain.PRIVATE_FLUX,
    ]
    assert [item.policy for item in restored.domain_profiles] == [
        item.policy for item in flux_state.domain_profiles
    ]
    assert restored.flow is not None
    assert flux_state.flow is not None
    assert restored.flow.species == flux_state.flow.species
    np.testing.assert_array_equal(
        restored.flow.omega_rad_per_s, flux_state.flow.omega_rad_per_s
    )
    np.testing.assert_array_equal(
        restored.flow.temperature_ev, flux_state.flow.temperature_ev
    )
    np.testing.assert_array_equal(restored.pressure, flux_state.pressure)
    np.testing.assert_array_equal(restored.f_df_dpsi, flux_state.f_df_dpsi)


def test_arrays_and_dataclass_fields_are_immutable(flux_state: FluxFunctionState):
    with pytest.raises(ValueError, match="read-only"):
        flux_state.pressure[0] = 0.0
    with pytest.raises(FrozenInstanceError):
        flux_state.pressure = np.zeros_like(flux_state.pressure)


def test_array_tree_has_identical_numpy_and_jax_leaves(flux_state: FluxFunctionState):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    jax.config.update("jax_enable_x64", True)

    numpy_tree = flux_state.array_tree(np)
    jax_tree = flux_state.array_tree(jnp)
    transformed = jax.jit(lambda tree: jax.tree.map(lambda leaf: leaf + 0.0, tree))(
        jax_tree
    )

    numpy_leaves = jax.tree.leaves(numpy_tree)
    jax_leaves = jax.tree.leaves(transformed)
    assert len(numpy_leaves) == len(jax_leaves) == 17
    for expected, actual in zip(numpy_leaves, jax_leaves, strict=True):
        np.testing.assert_allclose(np.asarray(actual), expected, rtol=0.0, atol=0.0)


def test_total_flux_jacobian_is_explicit_and_consistent(flux_state: FluxFunctionState):
    coordinate = flux_state.coordinate

    assert coordinate.dtotal_flux_dpsi_n_wb == 1.0
    np.testing.assert_allclose(
        coordinate.total_flux_wb,
        coordinate.axis_total_flux_wb
        + coordinate.values * coordinate.dtotal_flux_dpsi_n_wb,
    )
