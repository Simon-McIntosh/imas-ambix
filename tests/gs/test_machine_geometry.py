"""Contract tests for the shot-addressed machine-geometry facade."""

from __future__ import annotations

import numpy as np
import pytest

from imas_ambix.gs import machine_geometry

SHOT = 11766
REPRESENTATION_KEY = "mp78-fl80-fc70-lim37-2f6472393311692b"
REPRESENTATION_DIGEST = "2f6472393311692b"
DERIVATION_ID = "ddv4-directed-probe-angle"


@pytest.fixture(scope="module")
def service():
    return machine_geometry.MachineGeometryService(
        channel_shots=(SHOT,), amc_channel_shot=SHOT
    )


def test_public_boundary_is_only_the_service_and_three_projections():
    public_names = sorted(
        name for name in vars(machine_geometry) if not name.startswith("_")
    )

    assert public_names == [
        "GeometryIdentity",
        "MachineGeometryService",
        "OperatorGeometry",
        "SensorGeometry",
    ]
    assert "GeometryTable" not in machine_geometry.__all__
    assert not {
        "BProbe",
        "CircuitDrive",
        "FluxLoop",
        "PFFilament",
        "PassiveStructure",
        "PolygonSection",
        "SensorMapping",
        "SetupSignature",
    } & set(machine_geometry.__all__)


def test_identity_preserves_the_compatibility_kernel_key(service):
    identity = service.identity(SHOT)
    kernel = service._compatibility_kernel(SHOT)

    assert isinstance(identity, machine_geometry.GeometryIdentity)
    assert identity.representation_key == REPRESENTATION_KEY
    assert identity.representation_key == kernel.signature.key
    assert identity.representation_digest == REPRESENTATION_DIGEST
    assert identity.representation_digest == kernel.signature.digest
    assert identity.derivation_id == DERIVATION_ID
    assert len(identity.physical_digest) == 16
    assert len(identity.registry_digest) == 64


def test_operator_projection_preserves_the_coil_column_matrix(service):
    projected = service.operator(SHOT)
    existing = service._compatibility_operator(SHOT)

    assert isinstance(projected, machine_geometry.OperatorGeometry)
    assert projected.coil_column_matrix.shape == (81, 21)
    assert np.array_equal(projected.coil_column_matrix, existing.g_pf)
    assert float(np.max(np.abs(projected.coil_column_matrix - existing.g_pf))) == 0.0
    assert not projected.coil_column_matrix.flags.writeable


def test_unresolved_p6_turns_cross_as_named_missing_values(service):
    unresolved = service.operator(SHOT).unresolved_turns

    assert dict(unresolved) == {"p6_lower": None, "p6_upper": None}
    assert all(value is None for value in unresolved.values())


def test_sensor_projection_is_channel_aligned_and_immutable(service):
    projected = service.sensors(SHOT, ("ccbv01", "fl_cc01", "ip"))

    assert isinstance(projected, machine_geometry.SensorGeometry)
    assert projected.channels == ("ccbv01", "fl_cc01", "ip")
    assert projected.feature_matrix.shape == (3, len(projected.feature_names))
    assert projected.sensor_kinds == ("bpol_probe", "flux_loop", "coil")
    assert projected.identity is service.identity(SHOT)
    assert not projected.feature_matrix.flags.writeable
