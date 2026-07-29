"""Focused metadata and array conversion checks for circuits and transport."""

import numpy as np

from imas_ambix.physics import (
    CircuitCoupling,
    CurrentDiffusion,
    circuit_table_from_metadata,
    current_diffusion_from_mapping,
    emit_circuit_coupling,
)


def test_ambix_circuit_metadata_emits_nova_circuit_payload():
    active = [
        {
            "circuit_id": 1,
            "coil_label": "poloidal_coil",
            "l1_coil_channel": "coil_current",
            "l1_feed_channel": None,
        }
    ]
    cases = [
        {
            "circuit_id": 2,
            "geometry_confusable_with": "poloidal_coil",
            "l1_case_channel": "case_current",
            "constrained_zero": False,
        }
    ]
    table = circuit_table_from_metadata(
        active,
        cases,
        {"poloidal_coil": (1.0, 0.0)},
    )
    filaments = [
        {
            "r": 1.0,
            "z": 0.0,
            "width": 0.08,
            "height": 0.1,
            "xmult": 1.0,
            "circuit": 1,
        },
        {
            "r": 1.02,
            "z": 0.0,
            "width": 0.1,
            "height": 0.12,
            "xmult": 1.0,
            "circuit": 2,
        },
        {
            "r": 1.7,
            "z": 0.4,
            "width": 0.02,
            "height": 0.03,
            "xmult": 1.0,
            "circuit": 3,
        },
    ]

    coupling = emit_circuit_coupling(
        filaments,
        ["coil_current", "case_current"],
        table,
    )

    assert isinstance(coupling, CircuitCoupling)
    assert coupling.channel_circuits == {
        "case_current": [2],
        "coil_current": [1],
    }
    assert coupling.passive_circuits == (3,)
    np.testing.assert_array_equal(coupling.conductors.circuit, [1, 2, 3])


def test_transport_mapping_builds_nova_geometry_and_solver():
    face = np.array([0.0, 0.5, 1.0])
    cell = np.array([0.25, 0.75])
    geometry = {
        "rho_face": face,
        "rho_cell": cell,
        "psi_face": np.array([0.0, 0.2, 0.5]),
        "psi_n_face": face,
        "psi_n_cell": cell,
        "vpr_face": np.array([0.0, 1.0, 2.0]),
        "vpr_cell": np.array([0.5, 1.5]),
        "g2_face": np.array([1.0, 1.1, 1.2]),
        "g3_face": np.array([1.0, 1.0, 1.0]),
        "g3_cell": np.array([1.0, 1.0]),
        "f_face": np.array([0.9, 0.9, 0.9]),
        "f_cell": np.array([0.9, 0.9]),
        "b2_cell": np.array([1.1, 1.2]),
        "inv_r_cell": np.array([1.0, 0.9]),
        "phi_b": 0.7,
        "r0": 0.9,
        "ip_amperes": 5.0e5,
        "axis_psi": 0.0,
        "boundary_psi": 0.5,
        "volume": 8.0,
        "q_face": np.array([1.0, 1.5, 2.0]),
        "flux_sign": 1.0,
    }

    solver = current_diffusion_from_mapping(
        geometry,
        {"eta0": 8.0e-8, "contrast": 1.5, "shape": 2.0},
        theta=0.75,
    )

    assert isinstance(solver, CurrentDiffusion)
    np.testing.assert_allclose(solver.geometry.psi_n_cell, cell)
    assert solver.geometry.ip_amperes == 5.0e5
    assert solver.eta.eta0 == 8.0e-8
    assert solver.theta == 0.75
