"""Render solved flux geometry into spatial camera-model conditioning.

Nova steering frames carry nested flux-surface polylines rather than a raster.
This module draws those polylines deterministically on a fixed MAST vessel
grid.  The point-component mask follows Nova's order: magnetic axis, primary
X-point, secondary X-point, two strike points, then LCFS.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

MAST_WALL_R_BOUNDS = (0.19524440169334412, 1.899999976158142)
MAST_WALL_Z_BOUNDS = (-1.8250000476837158, 1.8250000476837158)
GRID_SHAPE = (64, 64)
SURFACE_LEVELS = np.linspace(0.0, 1.0, 11, dtype=np.float64)
SURFACE_COUNT = 11
SURFACE_ANGLE_COUNT = 64
POINT_COMPONENT_COUNT = 3

FloatArray = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class FluxGrid:
    """Fixed-size rectangular flux grid with MAST wall bounds by default.

    Bounds are inclusive and expressed in metres.  Array rows follow increasing
    Z and columns follow increasing R.
    """

    r_bounds: tuple[float, float] = MAST_WALL_R_BOUNDS
    z_bounds: tuple[float, float] = MAST_WALL_Z_BOUNDS
    shape: ClassVar[tuple[int, int]] = GRID_SHAPE

    def __post_init__(self) -> None:
        _validate_bounds("r_bounds", self.r_bounds)
        _validate_bounds("z_bounds", self.z_bounds)

    @property
    def radius(self) -> NDArray[np.float64]:
        """Return the 64 major-radius sample coordinates."""
        return np.linspace(*self.r_bounds, self.shape[1], dtype=np.float64)

    @property
    def height(self) -> NDArray[np.float64]:
        """Return the 64 vertical sample coordinates."""
        return np.linspace(*self.z_bounds, self.shape[0], dtype=np.float64)

    def receipt(self) -> dict[str, object]:
        """Return the complete grid identity for a rendering receipt."""
        return {
            "coordinate_order": "ZR",
            "shape": list(self.shape),
            "r_bounds_m": [float(value) for value in self.r_bounds],
            "z_bounds_m": [float(value) for value in self.z_bounds],
            "bounds_source": "MAST era wall polygon",
        }


def _validate_bounds(name: str, bounds: tuple[float, float]) -> None:
    values = np.asarray(bounds, dtype=np.float64)
    if values.shape != (2,) or not np.isfinite(values).all():
        raise ValueError(f"{name} must contain two finite values")
    if values[0] >= values[1]:
        raise ValueError(f"{name} must be strictly increasing")


def _field(fields: object, name: str) -> Any:
    if isinstance(fields, Mapping):
        try:
            return fields[name]
        except KeyError as exc:
            raise KeyError(f"missing flux-conditioning field {name!r}") from exc
    try:
        return getattr(fields, name)
    except AttributeError as exc:
        raise AttributeError(f"missing flux-conditioning field {name!r}") from exc


def _array(fields: object, name: str, *, dtype: np.dtype[Any]) -> np.ndarray:
    value = _field(fields, name)
    if hasattr(value, "values"):
        value = value.values
    return np.asarray(value, dtype=dtype)


def _point_coordinates(fields: object) -> NDArray[np.float64]:
    axis_r = _array(fields, "magnetic_axis_r", dtype=np.dtype(np.float64))
    axis_z = _array(fields, "magnetic_axis_z", dtype=np.dtype(np.float64))
    if axis_r.size != 1 or axis_z.size != 1:
        raise ValueError("magnetic-axis coordinates must be scalar for one frame")

    x_r = _array(fields, "x_point_r", dtype=np.dtype(np.float64)).reshape(-1)
    x_z = _array(fields, "x_point_z", dtype=np.dtype(np.float64)).reshape(-1)
    if x_r.shape != (2,) or x_z.shape != (2,):
        raise ValueError("X-point coordinates must contain primary and secondary slots")
    return np.asarray(
        [[axis_r.item(), axis_z.item()], *zip(x_r, x_z, strict=True)],
        dtype=np.float64,
    )


def _point_mask(fields: object) -> NDArray[np.bool_]:
    mask = _array(fields, "finite_mask", dtype=np.dtype(np.bool_)).reshape(-1)
    if mask.size < POINT_COMPONENT_COUNT:
        raise ValueError("finite_mask must cover the axis and two X-point slots")
    return mask[:POINT_COMPONENT_COUNT]


def _polygon_contains(
    polygon_r: NDArray[np.float64],
    polygon_z: NDArray[np.float64],
    sample_r: NDArray[np.float64],
    sample_z: NDArray[np.float64],
) -> NDArray[np.bool_]:
    """Return strict polygon membership for every point on a mesh."""
    vertices = np.column_stack((polygon_r, polygon_z))
    vertices = vertices[np.isfinite(vertices).all(axis=1)]
    if vertices.shape[0] < 3 or np.unique(vertices, axis=0).shape[0] < 3:
        return np.zeros(sample_r.shape, dtype=np.bool_)

    inside = np.zeros(sample_r.shape, dtype=np.bool_)
    on_boundary = np.zeros(sample_r.shape, dtype=np.bool_)
    scale = max(float(np.ptp(vertices[:, 0])), float(np.ptp(vertices[:, 1])), 1.0)
    tolerance = 32.0 * np.finfo(np.float64).eps * scale

    previous = vertices[-1]
    for current in vertices:
        edge_r = current[0] - previous[0]
        edge_z = current[1] - previous[1]
        edge_length_sq = edge_r * edge_r + edge_z * edge_z
        if edge_length_sq <= tolerance * tolerance:
            previous = current
            continue
        offset_r = sample_r - previous[0]
        offset_z = sample_z - previous[1]
        cross = offset_r * edge_z - offset_z * edge_r
        projection = offset_r * edge_r + offset_z * edge_z
        on_boundary |= (
            (np.abs(cross) <= tolerance)
            & (projection >= -tolerance)
            & (projection <= edge_length_sq + tolerance)
        )

        crosses_row = (current[1] > sample_z) != (previous[1] > sample_z)
        with np.errstate(divide="ignore", invalid="ignore"):
            crossing_r = previous[0] + ((sample_z - previous[1]) * edge_r / edge_z)
        inside ^= crosses_row & (sample_r < crossing_r)
        previous = current
    return inside & ~on_boundary


def _gaussian_mark(
    point: NDArray[np.float64],
    grid: FluxGrid,
) -> FloatArray:
    if (
        not np.isfinite(point).all()
        or point[0] < grid.r_bounds[0]
        or point[0] > grid.r_bounds[1]
        or point[1] < grid.z_bounds[0]
        or point[1] > grid.z_bounds[1]
    ):
        return np.zeros(grid.shape, dtype=np.float32)

    column = (
        (point[0] - grid.r_bounds[0])
        / (grid.r_bounds[1] - grid.r_bounds[0])
        * (grid.shape[1] - 1)
    )
    row = (
        (point[1] - grid.z_bounds[0])
        / (grid.z_bounds[1] - grid.z_bounds[0])
        * (grid.shape[0] - 1)
    )
    rows, columns = np.indices(grid.shape, dtype=np.float64)
    mark = np.exp(-0.5 * ((rows - row) ** 2 + (columns - column) ** 2))
    peak = float(mark.max())
    if peak > 0.0:
        mark /= peak
    return mark.astype(np.float32)


def _profile_edge_scalar(fields: object, name: str) -> float:
    values = _array(fields, name, dtype=np.dtype(np.float64)).squeeze()
    if values.ndim == 0:
        result = float(values)
    elif values.ndim == 1 and values.size:
        result = float(values[-1])
    else:
        raise ValueError(f"{name} must be scalar or a one-dimensional frame profile")
    if not np.isfinite(result):
        raise ValueError(f"{name} must have a finite outermost value")
    return result


def geometry_vector(fields: object) -> FloatArray:
    """Return the 12-value geometry token for one steering frame.

    Masked axis or X-point coordinates are zeroed so that an absent point never
    injects a non-finite model input.  Shape profiles use their outermost value.
    """
    points = _point_coordinates(fields)
    mask = _point_mask(fields)
    points = np.where(mask[:, None] & np.isfinite(points), points, 0.0)
    diverted = _array(fields, "diverted", dtype=np.dtype(np.bool_))
    if diverted.size != 1:
        raise ValueError("diverted must be scalar for one frame")

    vector = np.asarray(
        [
            *points.reshape(-1),
            _profile_edge_scalar(fields, "elongation"),
            _profile_edge_scalar(fields, "delta_upper"),
            _profile_edge_scalar(fields, "delta_lower"),
            _profile_edge_scalar(fields, "R_major"),
            _profile_edge_scalar(fields, "a_minor"),
            float(diverted.item()),
        ],
        dtype=np.float32,
    )
    if vector.shape != (12,):
        raise RuntimeError("geometry-vector layout must contain exactly 12 values")
    return vector


def render_flux_conditioning(
    fields: object,
    grid: FluxGrid | None = None,
) -> FloatArray:
    """Render one steering frame as a finite ``(6, 64, 64)`` tensor.

    The channels are piecewise normalised flux, strict inside-LCFS membership,
    Gaussian magnetic-axis, primary-X and secondary-X marks, and the diverted
    flag.  Call :meth:`FluxGrid.receipt` on the same grid to record its bounds.
    """
    target_grid = grid or FluxGrid()
    levels = _array(fields, "flux_surface_psi_norm", dtype=np.dtype(np.float64))
    surface_r = _array(fields, "flux_surface_r", dtype=np.dtype(np.float64))
    surface_z = _array(fields, "flux_surface_z", dtype=np.dtype(np.float64))
    if levels.shape != (SURFACE_COUNT,) or not np.allclose(
        levels, SURFACE_LEVELS, rtol=0.0, atol=1.0e-7
    ):
        raise ValueError("flux surfaces must use eleven levels from 0.0 to 1.0")
    expected_shape = (SURFACE_COUNT, SURFACE_ANGLE_COUNT)
    if surface_r.shape != expected_shape or surface_z.shape != expected_shape:
        raise ValueError(f"flux-surface coordinates must have shape {expected_shape}")

    sample_r, sample_z = np.meshgrid(target_grid.radius, target_grid.height)
    membership = np.zeros((SURFACE_COUNT, *target_grid.shape), dtype=np.bool_)
    for index in range(SURFACE_COUNT):
        membership[index] = _polygon_contains(
            surface_r[index], surface_z[index], sample_r, sample_z
        )

    psi_norm = np.ones(target_grid.shape, dtype=np.float32)
    for level, contained in zip(levels, membership, strict=True):
        np.minimum(psi_norm, np.float32(level), out=psi_norm, where=contained)
    inside_lcfs = membership[-1].astype(np.float32)

    points = _point_coordinates(fields)
    mask = _point_mask(fields)
    marks = [
        _gaussian_mark(point, target_grid)
        if present
        else np.zeros(target_grid.shape, dtype=np.float32)
        for point, present in zip(points, mask, strict=True)
    ]

    diverted = _array(fields, "diverted", dtype=np.dtype(np.bool_))
    if diverted.size != 1:
        raise ValueError("diverted must be scalar for one frame")
    diverted_channel = np.full(
        target_grid.shape, float(diverted.item()), dtype=np.float32
    )
    conditioning = np.stack(
        [psi_norm, inside_lcfs, *marks, diverted_channel], axis=0
    ).astype(np.float32, copy=False)
    if not np.isfinite(conditioning).all():
        raise ValueError("rendered flux conditioning must be finite")
    return conditioning


__all__ = ["FluxGrid", "geometry_vector", "render_flux_conditioning"]
