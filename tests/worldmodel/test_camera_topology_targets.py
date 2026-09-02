"""Tests for camera-frame topology labels derived from the L2 flux field."""

from __future__ import annotations

import numpy as np
import zarr

from imas_ambix.latent import topology
from imas_ambix.worldmodel.camera_topology_targets import (
    TOPOLOGY_CLASS_NAMES,
    TOPOLOGY_UNDEFINED,
    build_camera_topology_targets_from_arrays,
    load_camera_topology_targets,
)


def _circle(r0: float, z0: float, radius: float, n: int = 180):
    angle = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return r0 + radius * np.cos(angle), z0 + radius * np.sin(angle)


def _single_slice_inputs(psi, rg, zg, axis, nulls, boundary, wall):
    return {
        "shot_id": 7,
        "frame_times": np.array([0.0]),
        "equilibrium_times": np.array([0.0]),
        "psi": psi[:, :, None],
        "major_radius": rg,
        "z": zg,
        "axis_r": np.array([axis[0]]),
        "axis_z": np.array([axis[1]]),
        "x_point_r": np.asarray([p[0] for p in nulls] + [-9.99] * (2 - len(nulls)))[
            :, None
        ],
        "x_point_z": np.asarray([p[1] for p in nulls] + [-9.99] * (2 - len(nulls)))[
            :, None
        ],
        "lcfs_r": boundary[0][:, None],
        "lcfs_z": boundary[1][:, None],
        "wall_r": wall[0],
        "wall_z": wall[1],
    }


def test_limited_field_has_class_but_masks_divertor_targets():
    rg = np.linspace(0.3, 1.7, 161)
    zg = np.linspace(-0.8, 0.8, 161)
    rr, zz = np.meshgrid(rg, zg)
    axis = (1.0, 0.0)
    psi = (rr - axis[0]) ** 2 + zz**2
    wall = _circle(*axis, 0.45)
    result = build_camera_topology_targets_from_arrays(
        **_single_slice_inputs(psi, rg, zg, axis, [], wall, wall)
    )

    assert result.class_distribution()["limited"] == 1
    assert result.boundary_flux_mask.tolist() == [True]
    assert result.primary_xpoint_mask.tolist() == [False]
    assert not result.strike_point_mask.any()


def test_primary_null_is_ordered_by_boundary_flux_and_strikes_are_radial():
    rg = np.linspace(0.3, 1.7, 241)
    zg = np.linspace(-1.0, 1.0, 241)
    rr, zz = np.meshgrid(rg, zg)
    x = (rr - 1.0) / 0.32
    y = zz / 0.32
    psi = (y**2 - 1.0) ** 2 + x**2 - 0.5 * y
    critical = topology.find_critical_points(psi, rg, zg)
    axis = tuple(critical.o_points[int(np.argmin(critical.o_psi))])
    null_index = int(np.argmin(np.abs(critical.x_psi - critical.o_psi.min())))
    null = critical.x_points[null_index]
    wall = _circle(*axis, 0.45)
    boundary_read = topology.lcfs_contour(
        psi, rg, zg, axis, limiter_r=wall[0], limiter_z=wall[1], clip_legs=True
    )
    assert boundary_read.found
    boundary = (boundary_read.ring[:, 0], boundary_read.ring[:, 1])

    result = build_camera_topology_targets_from_arrays(
        **_single_slice_inputs(psi, rg, zg, axis, [null], boundary, wall)
    )

    np.testing.assert_allclose(result.primary_xpoint[0], null, atol=0.03)
    assert result.primary_xpoint_mask.tolist() == [True]
    assert TOPOLOGY_CLASS_NAMES[result.topology_class[0]].startswith("single-null")
    present = result.strike_points[0, result.strike_point_mask[0]]
    assert present.shape == (2, 2)
    assert np.all(np.diff(present[:, 0]) >= 0.0)


def test_equal_flux_pair_is_connected_double_null():
    import contourpy

    rg = np.linspace(0.2, 1.8, 241)
    zg = np.linspace(-1.0, 1.0, 241)
    rr, zz = np.meshgrid(rg, zg)
    axis = (1.0, 0.0)
    saddle_z = 0.5
    psi = (rr - axis[0]) ** 2 + zz**2 - zz**4 / (2.0 * saddle_z**2)
    nulls = np.array([[1.0, -saddle_z], [1.0, saddle_z]])
    boundary_level = saddle_z**2 / 2.0
    generator = contourpy.contour_generator(rg, zg, psi, line_type="Separate")
    rings = generator.lines(0.999 * boundary_level)
    ring = max(rings, key=lambda points: points.shape[0])
    boundary = (ring[:, 0], ring[:, 1])
    wall = _circle(*axis, 0.75)

    result = build_camera_topology_targets_from_arrays(
        **_single_slice_inputs(psi, rg, zg, axis, nulls, boundary, wall)
    )

    assert TOPOLOGY_CLASS_NAMES[result.topology_class[0]] == "connected-double-null"
    np.testing.assert_allclose(result.primary_xpoint[0], nulls[0], atol=0.03)


def test_nearest_native_sampling_preserves_masks_and_classes():
    rg = np.linspace(0.4, 1.6, 121)
    zg = np.linspace(-0.6, 0.6, 121)
    rr, zz = np.meshgrid(rg, zg)
    psi0 = (rr - 1.0) ** 2 + zz**2
    psi = np.stack([psi0, psi0], axis=2)
    wall = _circle(1.0, 0.0, 0.4)
    boundary_r = np.column_stack([wall[0], np.full_like(wall[0], np.nan)])
    boundary_z = np.column_stack([wall[1], np.full_like(wall[1], np.nan)])

    result = build_camera_topology_targets_from_arrays(
        shot_id=9,
        frame_times=np.array([-0.1, 0.1, 0.9, 1.1]),
        equilibrium_times=np.array([0.0, 1.0]),
        psi=psi,
        major_radius=rg,
        z=zg,
        axis_r=np.array([1.0, 1.0]),
        axis_z=np.array([0.0, 0.0]),
        x_point_r=np.full((2, 2), -9.99),
        x_point_z=np.full((2, 2), -9.99),
        lcfs_r=boundary_r,
        lcfs_z=boundary_z,
        wall_r=wall[0],
        wall_z=wall[1],
    )

    assert result.topology_class.tolist() == [
        TOPOLOGY_UNDEFINED,
        TOPOLOGY_CLASS_NAMES.index("limited"),
        TOPOLOGY_UNDEFINED,
        TOPOLOGY_UNDEFINED,
    ]
    assert result.boundary_flux_mask.tolist() == [False, True, False, False]


def test_loader_reads_equilibrium_and_wall_groups(tmp_path):
    shot = 12
    rg = np.linspace(0.5, 1.5, 81)
    zg = np.linspace(-0.5, 0.5, 81)
    rr, zz = np.meshgrid(rg, zg)
    psi = ((rr - 1.0) ** 2 + zz**2)[:, :, None]
    wall = _circle(1.0, 0.0, 0.35, n=80)
    store = zarr.open_group(str(tmp_path / f"{shot}.zarr"), mode="w")
    equilibrium = store.create_group("equilibrium")
    arrays = {
        "time": np.array([0.0]),
        "psi": psi,
        "major_radius": rg,
        "z": zg,
        "magnetic_axis_r": np.array([1.0]),
        "magnetic_axis_z": np.array([0.0]),
        "x_point_r": np.full((2, 1), -9.99),
        "x_point_z": np.full((2, 1), -9.99),
        "lcfs_r": wall[0][:, None],
        "lcfs_z": wall[1][:, None],
    }
    for name, values in arrays.items():
        equilibrium.create_array(name, data=values)
    wall_group = store.create_group("wall")
    wall_group.create_array("limiter_r", data=wall[0])
    wall_group.create_array("limiter_z", data=wall[1])

    result = load_camera_topology_targets(shot, np.array([0.0]), level2_root=tmp_path)

    assert result.shot_id == shot
    assert result.class_distribution()["limited"] == 1


def test_loader_masks_omitted_null_arrays_but_keeps_defined_class(tmp_path):
    shot = 13
    rg = np.linspace(0.5, 1.5, 81)
    zg = np.linspace(-0.5, 0.5, 81)
    rr, zz = np.meshgrid(rg, zg)
    psi = ((rr - 1.0) ** 2 + zz**2)[:, :, None]
    wall = _circle(1.0, 0.0, 0.35, n=80)
    store = zarr.open_group(str(tmp_path / f"{shot}.zarr"), mode="w")
    equilibrium = store.create_group("equilibrium")
    arrays = {
        "time": np.array([0.0]),
        "psi": psi,
        "major_radius": rg,
        "z": zg,
        "magnetic_axis_r": np.array([1.0]),
        "magnetic_axis_z": np.array([0.0]),
        "lcfs_r": wall[0][:, None],
        "lcfs_z": wall[1][:, None],
    }
    for name, values in arrays.items():
        equilibrium.create_array(name, data=values)
    wall_group = store.create_group("wall")
    wall_group.create_array("limiter_r", data=wall[0])
    wall_group.create_array("limiter_z", data=wall[1])

    result = load_camera_topology_targets(shot, np.array([0.0]), level2_root=tmp_path)

    assert np.isnan(result.primary_xpoint).all()
    assert result.primary_xpoint_mask.tolist() == [False]
    assert np.isnan(result.strike_points).all()
    assert not result.strike_point_mask.any()
    assert result.topology_class.tolist() == [TOPOLOGY_CLASS_NAMES.index("limited")]
    assert result.boundary_flux_mask.tolist() == [True]
