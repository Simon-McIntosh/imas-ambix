"""Focused geometry conversion checks for Nova equilibrium construction."""

from dataclasses import dataclass

import numpy as np

from imas_ambix.physics import (
    ReconstructProfile,
    magnetics_from_records,
    profile_reconstructor,
)


@dataclass(frozen=True)
class FieldProbe:
    r: float
    z: float
    angle_deg: float


def test_magnetics_preserves_positions_orientations_and_row_kind():
    magnetics = magnetics_from_records(
        [FieldProbe(0.8, -0.2, 35.0), {"r": 1.1, "z": 0.3, "angle_deg": -90.0}],
        [{"r": 0.6, "z": 0.0}],
    )

    np.testing.assert_allclose(magnetics.r, [0.8, 1.1, 0.6])
    np.testing.assert_allclose(magnetics.z, [-0.2, 0.3, 0.0])
    np.testing.assert_allclose(magnetics.angle, [35.0, -90.0, 0.0])
    np.testing.assert_array_equal(magnetics.flux_loop, [False, False, True])


def test_profile_factory_builds_the_nova_solver_from_ambix_columns():
    geometry = {
        "grid_r": np.array([0.7, 1.0]),
        "grid_z": np.array([-0.15, 0.15]),
        "inside_limiter": np.ones((2, 2), dtype=bool),
        "cell_width": np.full(4, 0.1),
        "cell_height": np.full(4, 0.1),
        "source_r": np.array([1.35]),
        "source_z": np.array([0.0]),
        "source_width": np.array([0.08]),
        "source_height": np.array([0.12]),
        "source_names": ["poloidal_coil"],
        "axis_seed": (0.85, 0.0),
        "wall_r": np.array([0.55, 1.25, 1.25, 0.55]),
        "wall_z": np.array([-0.35, -0.35, 0.35, 0.35]),
    }
    solver = profile_reconstructor(
        geometry,
        field_probes=[FieldProbe(1.45, 0.1, 90.0)],
        flux_loops=[{"r": 1.4, "z": -0.1}],
        n_pressure=1,
        n_diamagnetic=1,
        options={"iterations": 2},
    )

    assert isinstance(solver, ReconstructProfile)
    assert solver.source_names == ("poloidal_coil",)
    assert solver.degrees.names == ("pressure_0", "diamagnetic_0")
    assert solver.source_to_grid.shape == (4, 1)
    assert solver.source_to_sensor.shape == (2, 1)
