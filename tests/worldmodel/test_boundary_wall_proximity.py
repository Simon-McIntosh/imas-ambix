from __future__ import annotations

import numpy as np
import pytest

from imas_ambix.worldmodel.boundary_wall_proximity import (
    FLOATING_DISTANCE_MM,
    BoundaryMeasurement,
    boundary_polygon_metrics,
    select_outermost_finite_surface,
    summarize_measurements,
)


def _square(half_width: float) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray([-half_width, half_width, half_width, -half_width]),
        np.asarray([-half_width, -half_width, half_width, half_width]),
    )


def test_boundary_inside_limiter_recovers_known_distance_and_area() -> None:
    limiter_r, limiter_z = _square(1.0)
    boundary_r, boundary_z = _square(0.75)

    result = boundary_polygon_metrics(
        boundary_r,
        boundary_z,
        limiter_r=limiter_r,
        limiter_z=limiter_z,
    )

    assert result["boundary_to_limiter_distance_m"] == pytest.approx(0.25)
    assert result["boundary_area_m2"] == pytest.approx(2.25)
    assert result["boundary_points_inside_limiter"] is True


def test_boundary_crossing_limiter_has_zero_minimum_distance() -> None:
    limiter_r, limiter_z = _square(1.0)
    boundary_r = np.asarray([-1.5, 0.5, 0.5, -1.5])
    boundary_z = np.asarray([-0.5, -0.5, 0.5, 0.5])

    result = boundary_polygon_metrics(
        boundary_r,
        boundary_z,
        limiter_r=limiter_r,
        limiter_z=limiter_z,
    )

    assert result["boundary_to_limiter_distance_m"] == 0.0
    assert result["boundary_points_inside_limiter"] is False


def test_shrunken_boundary_is_counted_as_floating() -> None:
    limiter_r, limiter_z = _square(1.0)
    boundary_r, boundary_z = _square(0.2)
    metric = boundary_polygon_metrics(
        boundary_r,
        boundary_z,
        limiter_r=limiter_r,
        limiter_z=limiter_z,
    )
    distance_mm = 1000.0 * float(metric["boundary_to_limiter_distance_m"])
    measurement = BoundaryMeasurement(
        shot_id=22086,
        manifest_row=0,
        session_index=0,
        time_s=0.0,
        diverted=False,
        recorded_conditioned=False,
        conditioned=False,
        reclassified_as_free=False,
        conditioned_branch_guard_ok=False,
        surface_category="nominal_outer_surface",
        selected_surface_index=2,
        selected_surface_psi_norm=1.0,
        boundary_to_limiter_distance_mm=distance_mm,
        boundary_area_m2=float(metric["boundary_area_m2"]),
        boundary_points_inside_limiter=bool(metric["boundary_points_inside_limiter"]),
        floating_boundary=distance_mm > FLOATING_DISTANCE_MM,
        measurement_error=None,
    )

    summary = summarize_measurements([measurement])

    assert distance_mm == pytest.approx(800.0)
    assert measurement.floating_boundary is True
    assert summary["all"]["floating_boundary_count"] == 1
    assert summary["all"]["floating_boundary_fraction"] == 1.0
    assert summary["diverted_false"]["admitted_slice_count"] == 1
    assert summary["diverted_true"]["admitted_slice_count"] == 0


def test_missing_outer_surface_uses_highest_finite_fallback() -> None:
    levels = np.asarray([0.5, 0.75, 1.0])
    surface_r = np.full((3, 4), np.nan)
    surface_z = np.full((3, 4), np.nan)
    surface_r[1], surface_z[1] = _square(0.6)
    surface_r[0], surface_z[0] = _square(0.4)

    result = select_outermost_finite_surface(levels, surface_r, surface_z)

    assert result.category == "fallback_outermost_finite_surface"
    assert result.index == 1
    assert result.psi_norm == pytest.approx(0.75)
    assert result.r == pytest.approx(surface_r[1])
    assert result.z == pytest.approx(surface_z[1])


def test_no_finite_surface_is_an_explicit_category() -> None:
    result = select_outermost_finite_surface(
        np.asarray([0.5, 0.75, 1.0]),
        np.full((3, 4), np.nan),
        np.full((3, 4), np.nan),
    )

    assert result.category == "no_qualifying_surface"
    assert result.index is None
    assert result.psi_norm is None
    assert result.r is None
    assert result.z is None


def test_later_duplicate_outer_level_is_recorded_as_fallback() -> None:
    levels = np.asarray([0.75, 1.0, 1.0])
    surface_r = np.full((3, 4), np.nan)
    surface_z = np.full((3, 4), np.nan)
    surface_r[2], surface_z[2] = _square(0.7)

    result = select_outermost_finite_surface(levels, surface_r, surface_z)

    assert result.category == "fallback_outermost_finite_surface"
    assert result.index == 2
    assert result.psi_norm == pytest.approx(1.0)
