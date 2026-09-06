from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from imas_ambix.worldmodel.flux_conditioning import (
    FluxGrid,
    geometry_vector,
    render_flux_conditioning,
)

SESSION_PATH = Path(
    "/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/21858.nc"
)


def _ellipse_fields(*, secondary_present: bool = True) -> dict[str, object]:
    angles = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    levels = np.linspace(0.0, 1.0, 11)
    centre_r = 0.82
    centre_z = 0.04
    radial_scale = 0.38
    vertical_scale = 0.72
    radii = np.sqrt(levels)
    surface_r = centre_r + radial_scale * radii[:, None] * np.cos(angles)
    surface_z = centre_z + vertical_scale * radii[:, None] * np.sin(angles)
    return {
        "flux_surface_psi_norm": levels,
        "flux_surface_r": surface_r,
        "flux_surface_z": surface_z,
        "magnetic_axis_r": centre_r,
        "magnetic_axis_z": centre_z,
        "x_point_r": np.array([0.72, 0.91]),
        "x_point_z": np.array([-0.62, 0.61]),
        "finite_mask": np.array([True, True, secondary_present, False, False, True]),
        "diverted": True,
        "elongation": 1.9,
        "delta_upper": 0.23,
        "delta_lower": 0.19,
        "R_major": centre_r,
        "a_minor": radial_scale,
    }


def _nearest_cell(grid: FluxGrid, r_value: float, z_value: float) -> tuple[int, int]:
    row = int(np.argmin(np.abs(grid.height - z_value)))
    column = int(np.argmin(np.abs(grid.radius - r_value)))
    return row, column


def test_flux_surfaces_render_piecewise_geometry() -> None:
    fields = _ellipse_fields()
    grid = FluxGrid()
    conditioning = render_flux_conditioning(fields, grid)

    assert conditioning.shape == (6, 64, 64)
    assert conditioning.dtype == np.float32
    assert np.isfinite(conditioning).all()

    inside = conditioning[1].astype(bool)
    assert np.all(conditioning[0, ~inside] == 1.0)
    cell_area = np.diff(grid.radius).mean() * np.diff(grid.height).mean()
    rendered_area = float(inside.sum() * cell_area)
    analytic_area = float(np.pi * 0.38 * 0.72)
    assert rendered_area == pytest.approx(analytic_area, rel=0.05)

    axis_row, axis_column = _nearest_cell(grid, 0.82, 0.04)
    radial_profile = conditioning[0, axis_row, axis_column:]
    assert np.all(np.diff(radial_profile) >= 0.0)
    assert radial_profile[-1] == 1.0

    expected_points = ((0.82, 0.04), (0.72, -0.62), (0.91, 0.61))
    for channel, (point_r, point_z) in enumerate(expected_points, start=2):
        assert np.unravel_index(conditioning[channel].argmax(), grid.shape) == (
            _nearest_cell(grid, point_r, point_z)
        )
        assert conditioning[channel].max() == 1.0
    assert np.all(conditioning[5] == 1.0)


def test_masked_point_produces_zero_channel_and_vector_coordinates() -> None:
    fields = _ellipse_fields(secondary_present=False)
    conditioning = render_flux_conditioning(fields)
    vector = geometry_vector(fields)

    assert np.all(conditioning[4] == 0.0)
    assert vector.shape == (12,)
    assert vector.dtype == np.float32
    assert np.all(vector[4:6] == 0.0)
    assert np.isfinite(vector).all()
    assert vector[-1] == 1.0


def test_grid_receipt_records_mast_wall_bounds() -> None:
    grid = FluxGrid()

    assert grid.receipt() == {
        "coordinate_order": "ZR",
        "shape": [64, 64],
        "r_bounds_m": [0.19524440169334412, 1.899999976158142],
        "z_bounds_m": [-1.8250000476837158, 1.8250000476837158],
        "bounds_source": "MAST era wall polygon",
    }


def test_real_steering_slice_renders_finite_conditioning() -> None:
    if not SESSION_PATH.exists():
        pytest.skip("nova labeller session is unavailable on this host")
    xarray = pytest.importorskip("xarray")
    with xarray.open_dataset(SESSION_PATH, group="steering") as session:
        fields = session.isel(time=50).load()

    conditioning = render_flux_conditioning(fields)
    vector = geometry_vector(fields)

    assert conditioning.shape == (6, 64, 64)
    assert conditioning.dtype == np.float32
    assert np.isfinite(conditioning).all()
    assert vector.shape == (12,)
    assert vector.dtype == np.float32
    assert np.isfinite(vector).all()
    assert vector[6] == pytest.approx(float(fields.elongation.values[-1]))
