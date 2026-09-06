"""Per-camera-frame topology targets derived from the L2 equilibrium field.

This is an evaluator-only label reader.  It consumes the reconstructed L2
``equilibrium`` and ``wall`` groups and never belongs on a camera-model input
path.  Every non-continuous quantity is sampled from the nearest native
equilibrium slice: a primary X-point, up to two radially ordered strike points,
and one topology class.  Missing equilibrium or topology stays masked.

The field read delegates to :mod:`imas_ambix.latent.topology`.  The L2 LCFS
polygon fixes the reference boundary flux; stored null coordinates are ordered
against that flux; connected-component labels distinguish a disconnected
private pocket; and strike points are intersections of the separatrix level set
with the supplied wall polygon.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from imas_ambix.latent import topology
from imas_ambix.worldmodel.equilibrium_labels import (
    DEFAULT_LEVEL2_ROOT,
    XPOINT_SENTINEL,
    equilibrium_store_path,
)

MAX_STRIKE_POINTS = 2
"""Inner and outer strike-point slots, ordered by major radius."""

TOPOLOGY_CLASS_NAMES: tuple[str, ...] = (
    "limited",
    "single-null-lower",
    "single-null-upper",
    "connected-double-null",
    "disconnected-double-null",
)
TOPOLOGY_UNDEFINED = -1

# Normalised-flux distance at which an L2 null binds the LCFS.  The L2 maps put
# a diverted null at u=1 to roughly 1e-3 interpolation noise, while wall-bound
# slices start beyond about 1e-2.
XPOINT_BIND_TOL = 0.005
MIN_BOUNDARY_POINTS = 8
MIN_FLUX_SPAN_WB = 1.0e-4
MAST_WALL_SOURCE_SHOT = 15276
"""Shot carrying the era-constant MAST wall when a store omits that group."""

REQUIRED_EQUILIBRIUM_ARRAYS: tuple[str, ...] = (
    "time",
    "psi",
    "major_radius",
    "z",
    "magnetic_axis_r",
    "magnetic_axis_z",
    "lcfs_r",
    "lcfs_z",
)
"""Arrays required to derive any flux-map topology for a shot."""

_TOPOLOGY_EXCLUDED_SHOTS: dict[str, set[int]] = {}


@dataclass
class CameraTopologyTargets:
    """Topology labels on one camera frame-time axis.

    ``primary_xpoint`` is ``(F, 2)`` in ``(R, Z)`` metres.  ``strike_points``
    is ``(F, 2, 2)`` with inner then outer strike point.  ``topology_class`` is
    an integer index into :data:`TOPOLOGY_CLASS_NAMES`, or ``-1`` when the L2
    boundary is undefined.  Masks are explicit; absent values remain NaN.
    """

    shot_id: int
    frame_times: np.ndarray
    primary_xpoint: np.ndarray
    primary_xpoint_mask: np.ndarray
    strike_points: np.ndarray
    strike_point_mask: np.ndarray
    topology_class: np.ndarray
    boundary_psi: np.ndarray
    boundary_flux_mask: np.ndarray
    wall_source_shot_id: int | None = None
    wall_digest: str | None = None
    exclusion_reason: str | None = None
    class_names: tuple[str, ...] = TOPOLOGY_CLASS_NAMES
    units: str = "m"

    @property
    def n_frames(self) -> int:
        return int(self.frame_times.size)

    def class_distribution(self) -> dict[str, int]:
        """Return counts for every class plus explicitly undefined frames."""
        counts = {
            name: int(np.count_nonzero(self.topology_class == index))
            for index, name in enumerate(self.class_names)
        }
        counts["undefined"] = int(
            np.count_nonzero(self.topology_class == TOPOLOGY_UNDEFINED)
        )
        return counts


def reset_camera_topology_exclusion_census() -> None:
    """Clear shot exclusions before a complete corpus pass."""
    _TOPOLOGY_EXCLUDED_SHOTS.clear()


def camera_topology_exclusion_census() -> dict[str, dict[str, object]]:
    """Return shot-unique exclusion counts and identifiers by reason."""
    return {
        reason: {"count": len(shots), "shot_ids": sorted(shots)}
        for reason, shots in sorted(_TOPOLOGY_EXCLUDED_SHOTS.items())
    }


def _excluded_camera_topology_targets(
    shot_id: int,
    frame_times: np.ndarray,
    reason: str,
) -> CameraTopologyTargets:
    """Return explicitly absent topology and count the excluded shot once."""
    shot = int(shot_id)
    _TOPOLOGY_EXCLUDED_SHOTS.setdefault(reason, set()).add(shot)
    times = np.asarray(frame_times, dtype=np.float64).ravel()
    return CameraTopologyTargets(
        shot_id=shot,
        frame_times=times,
        primary_xpoint=np.full((times.size, 2), np.nan, dtype=np.float32),
        primary_xpoint_mask=np.zeros(times.size, dtype=bool),
        strike_points=np.full(
            (times.size, MAX_STRIKE_POINTS, 2), np.nan, dtype=np.float32
        ),
        strike_point_mask=np.zeros((times.size, MAX_STRIKE_POINTS), dtype=bool),
        topology_class=np.full(times.size, TOPOLOGY_UNDEFINED, dtype=np.int8),
        boundary_psi=np.full(times.size, np.nan, dtype=np.float32),
        boundary_flux_mask=np.zeros(times.size, dtype=bool),
        exclusion_reason=reason,
    )


def _missing_equilibrium_reason(missing: tuple[str, ...]) -> str:
    if missing == ("psi",):
        return "missing_flux_map"
    label = (
        "missing_equilibrium_array"
        if len(missing) == 1
        else "missing_equilibrium_arrays"
    )
    return f"{label}:{','.join(missing)}"


def _bilinear_points(
    field: np.ndarray,
    r_1d: np.ndarray,
    z_1d: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    """Vectorised bilinear field values at ``(R, Z)`` points."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    out = np.full(pts.shape[0], np.nan, dtype=np.float64)
    if pts.size == 0:
        return out

    r = pts[:, 0]
    z = pts[:, 1]
    valid = (
        np.isfinite(r)
        & np.isfinite(z)
        & (r >= r_1d[0])
        & (r <= r_1d[-1])
        & (z >= z_1d[0])
        & (z <= z_1d[-1])
    )
    if not valid.any():
        return out
    rv = r[valid]
    zv = z[valid]
    jr = np.clip(np.searchsorted(r_1d, rv, side="right") - 1, 0, r_1d.size - 2)
    iz = np.clip(np.searchsorted(z_1d, zv, side="right") - 1, 0, z_1d.size - 2)
    r0, r1 = r_1d[jr], r_1d[jr + 1]
    z0, z1 = z_1d[iz], z_1d[iz + 1]
    tr = (rv - r0) / (r1 - r0)
    tz = (zv - z0) / (z1 - z0)
    out[valid] = (
        field[iz, jr] * (1.0 - tz) * (1.0 - tr)
        + field[iz, jr + 1] * (1.0 - tz) * tr
        + field[iz + 1, jr] * tz * (1.0 - tr)
        + field[iz + 1, jr + 1] * tz * tr
    )
    return out


def _valid_nulls(x_r: np.ndarray, x_z: np.ndarray) -> np.ndarray:
    points = np.column_stack([x_r, x_z]).astype(np.float64, copy=False)
    keep = (
        np.isfinite(points).all(axis=1)
        & (points[:, 0] > XPOINT_SENTINEL)
        & (points[:, 1] > XPOINT_SENTINEL)
    )
    return points[keep]


def _boundary_flux_from_polygon(
    psi: np.ndarray,
    r_1d: np.ndarray,
    z_1d: np.ndarray,
    boundary_r: np.ndarray,
    boundary_z: np.ndarray,
) -> float:
    points = np.column_stack([boundary_r, boundary_z])
    points = points[np.isfinite(points).all(axis=1)]
    if points.shape[0] < MIN_BOUNDARY_POINTS:
        return float("nan")
    values = _bilinear_points(psi, r_1d, z_1d, points)
    if np.count_nonzero(np.isfinite(values)) < MIN_BOUNDARY_POINTS:
        return float("nan")
    return float(np.nanmean(values))


def _segment_intersections(lines: list[np.ndarray], wall: np.ndarray) -> np.ndarray:
    """Intersections between contour polylines and a closed wall polygon."""
    found: list[np.ndarray] = []

    def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    for line in lines:
        if line.shape[0] < 2:
            continue
        start = line[:-1]
        direction = line[1:] - start
        for index in range(wall.shape[0]):
            wall_start = wall[index]
            wall_direction = wall[(index + 1) % wall.shape[0]] - wall_start
            denominator = cross(direction, wall_direction)
            offset = wall_start - start
            with np.errstate(divide="ignore", invalid="ignore"):
                along_line = cross(offset, wall_direction) / denominator
                along_wall = cross(offset, direction) / denominator
            hit = (
                (np.abs(denominator) > 1.0e-12)
                & (along_line >= 0.0)
                & (along_line <= 1.0)
                & (along_wall >= 0.0)
                & (along_wall <= 1.0)
            )
            found.extend(start[hit] + along_line[hit, None] * direction[hit])
    if not found:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(found, dtype=np.float64)


def _strike_points(
    psi: np.ndarray,
    r_1d: np.ndarray,
    z_1d: np.ndarray,
    wall_r: np.ndarray,
    wall_z: np.ndarray,
    boundary_psi: float,
    binding_xpoints: np.ndarray,
) -> np.ndarray:
    """Inner/outer separatrix-wall intersections, NaN padded."""
    out = np.full((MAX_STRIKE_POINTS, 2), np.nan, dtype=np.float64)
    if binding_xpoints.shape[0] == 0:
        return out

    import contourpy  # noqa: PLC0415

    generator = contourpy.contour_generator(
        r_1d, z_1d, psi, line_type="Separate", quad_as_tri=True
    )
    lines = generator.lines(float(boundary_psi))
    grid_tol = 2.0 * float(np.hypot(np.median(np.diff(r_1d)), np.median(np.diff(z_1d))))
    separatrix_lines = [
        line
        for line in lines
        if line.shape[0]
        and np.min(
            np.linalg.norm(line[:, None, :] - binding_xpoints[None, :, :], axis=2)
        )
        <= grid_tol
    ]
    wall = np.column_stack([wall_r, wall_z]).astype(np.float64, copy=False)
    wall = wall[np.isfinite(wall).all(axis=1)]
    if wall.shape[0] < 3:
        return out
    points = _segment_intersections(separatrix_lines, wall)
    if points.shape[0] == 0:
        return out

    # Polygon vertices can report the same crossing on two adjoining segments.
    dedup_tol = 2.0 * grid_tol
    unique: list[np.ndarray] = []
    for point in points[np.lexsort((points[:, 1], points[:, 0]))]:
        if all(np.linalg.norm(point - other) > dedup_tol for other in unique):
            unique.append(point)
    points = np.asarray(unique, dtype=np.float64)
    if points.shape[0] > MAX_STRIKE_POINTS:
        points = points[[int(np.argmin(points[:, 0])), int(np.argmax(points[:, 0]))]]
    points = points[np.lexsort((points[:, 1], points[:, 0]))]
    out[: points.shape[0]] = points
    return out


def _native_target(
    psi: np.ndarray,
    r_1d: np.ndarray,
    z_1d: np.ndarray,
    axis: tuple[float, float],
    x_r: np.ndarray,
    x_z: np.ndarray,
    boundary_r: np.ndarray,
    boundary_z: np.ndarray,
    wall_r: np.ndarray,
    wall_z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    primary = np.full(2, np.nan, dtype=np.float64)
    strikes = np.full((MAX_STRIKE_POINTS, 2), np.nan, dtype=np.float64)
    boundary_psi = _boundary_flux_from_polygon(psi, r_1d, z_1d, boundary_r, boundary_z)
    if not np.isfinite(boundary_psi) or not np.isfinite(axis).all():
        return primary, strikes, TOPOLOGY_UNDEFINED, boundary_psi

    field_read = topology.read_topology(
        psi,
        r_1d,
        z_1d,
        limiter_r=wall_r,
        limiter_z=wall_z,
    )
    read_axis = field_read.axis if field_read.axis is not None else axis
    axis_psi = float(_bilinear_points(psi, r_1d, z_1d, np.asarray([axis]))[0])
    span = boundary_psi - axis_psi
    if not np.isfinite(axis_psi) or abs(span) <= MIN_FLUX_SPAN_WB:
        return primary, strikes, TOPOLOGY_UNDEFINED, boundary_psi

    nulls = _valid_nulls(x_r, x_z)
    null_psi = _bilinear_points(psi, r_1d, z_1d, nulls)
    finite = np.isfinite(null_psi)
    nulls = nulls[finite]
    null_psi = null_psi[finite]

    # Use the topology authority's boundary selector to order the supplied L2
    # nulls, then compare to the independently defined L2 boundary flux.
    supplied = topology.CriticalPoints(
        o_points=np.asarray([read_axis], dtype=np.float64),
        o_psi=np.asarray([axis_psi], dtype=np.float64),
        x_points=nulls,
        x_psi=null_psi,
    )
    selected_flux = topology.boundary_flux(
        supplied,
        read_axis,
        axis_psi,
        limiter_r=wall_r,
        limiter_z=wall_z,
    )
    binding = np.abs((null_psi - boundary_psi) / span) <= XPOINT_BIND_TOL
    binding_nulls = nulls[binding]
    if selected_flux is not None and nulls.shape[0]:
        primary_index = int(np.argmin(np.abs(null_psi - selected_flux)))
        if binding[primary_index]:
            primary = nulls[primary_index]

    connected_boundary = topology.lcfs_contour(
        psi,
        r_1d,
        z_1d,
        read_axis,
        limiter_r=wall_r,
        limiter_z=wall_z,
        clip_legs=True,
    )
    region_boundary_psi = boundary_psi
    if connected_boundary.found:
        boundary_gap = abs((connected_boundary.psi_bnd - boundary_psi) / span)
        if boundary_gap <= 0.05:
            region_boundary_psi = connected_boundary.psi_bnd
    regions = topology.classify_regions(psi, r_1d, z_1d, read_axis, region_boundary_psi)
    has_private = bool(np.any(regions == topology.REGION_PRIVATE))

    n_binding = int(np.count_nonzero(binding))
    if n_binding == 0:
        topology_class = TOPOLOGY_CLASS_NAMES.index("limited")
    elif n_binding >= 2:
        topology_class = TOPOLOGY_CLASS_NAMES.index("connected-double-null")
    elif nulls.shape[0] >= 2 and has_private:
        topology_class = TOPOLOGY_CLASS_NAMES.index("disconnected-double-null")
    elif primary[1] < 0.0:
        topology_class = TOPOLOGY_CLASS_NAMES.index("single-null-lower")
    else:
        topology_class = TOPOLOGY_CLASS_NAMES.index("single-null-upper")

    strikes = _strike_points(
        psi,
        r_1d,
        z_1d,
        wall_r,
        wall_z,
        boundary_psi,
        binding_nulls,
    )
    return primary, strikes, topology_class, boundary_psi


def build_camera_topology_targets_from_arrays(
    *,
    shot_id: int,
    frame_times: np.ndarray,
    equilibrium_times: np.ndarray,
    psi: np.ndarray,
    major_radius: np.ndarray,
    z: np.ndarray,
    axis_r: np.ndarray,
    axis_z: np.ndarray,
    x_point_r: np.ndarray,
    x_point_z: np.ndarray,
    lcfs_r: np.ndarray,
    lcfs_z: np.ndarray,
    wall_r: np.ndarray,
    wall_z: np.ndarray,
    wall_source_shot_id: int | None = None,
    wall_digest: str | None = None,
) -> CameraTopologyTargets:
    """Derive native topology once, then sample it onto camera frame times."""
    ft = np.asarray(frame_times, dtype=np.float64).ravel()
    teq = np.asarray(equilibrium_times, dtype=np.float64).ravel()
    psi = np.asarray(psi, dtype=np.float64)
    if psi.shape != (len(z), len(major_radius), teq.size):
        raise ValueError(
            "psi must have shape (len(z), len(major_radius), len(equilibrium_times))"
        )

    native_primary = np.full((teq.size, 2), np.nan, dtype=np.float64)
    native_strikes = np.full((teq.size, MAX_STRIKE_POINTS, 2), np.nan, dtype=np.float64)
    native_class = np.full(teq.size, TOPOLOGY_UNDEFINED, dtype=np.int8)
    native_boundary = np.full(teq.size, np.nan, dtype=np.float64)
    for index in range(teq.size):
        (
            native_primary[index],
            native_strikes[index],
            native_class[index],
            native_boundary[index],
        ) = _native_target(
            psi[:, :, index],
            np.asarray(major_radius, dtype=np.float64),
            np.asarray(z, dtype=np.float64),
            (float(axis_r[index]), float(axis_z[index])),
            np.asarray(x_point_r, dtype=np.float64)[:, index],
            np.asarray(x_point_z, dtype=np.float64)[:, index],
            np.asarray(lcfs_r, dtype=np.float64)[:, index],
            np.asarray(lcfs_z, dtype=np.float64)[:, index],
            np.asarray(wall_r, dtype=np.float64),
            np.asarray(wall_z, dtype=np.float64),
        )

    nearest = np.zeros(ft.size, dtype=np.int64)
    in_range = np.zeros(ft.size, dtype=bool)
    if teq.size:
        order = np.argsort(teq)
        sorted_times = teq[order]
        in_range = (ft >= sorted_times[0]) & (ft <= sorted_times[-1])
        hi = np.clip(np.searchsorted(sorted_times, ft, side="left"), 0, teq.size - 1)
        lo = np.clip(hi - 1, 0, teq.size - 1)
        use_hi = np.abs(sorted_times[hi] - ft) <= np.abs(ft - sorted_times[lo])
        nearest = order[np.where(use_hi, hi, lo)]

    primary = np.full((ft.size, 2), np.nan, dtype=np.float32)
    strikes = np.full((ft.size, MAX_STRIKE_POINTS, 2), np.nan, dtype=np.float32)
    classes = np.full(ft.size, TOPOLOGY_UNDEFINED, dtype=np.int8)
    boundary = np.full(ft.size, np.nan, dtype=np.float32)
    primary[in_range] = native_primary[nearest[in_range]]
    strikes[in_range] = native_strikes[nearest[in_range]]
    classes[in_range] = native_class[nearest[in_range]]
    boundary[in_range] = native_boundary[nearest[in_range]]
    return CameraTopologyTargets(
        shot_id=int(shot_id),
        frame_times=ft,
        primary_xpoint=primary,
        primary_xpoint_mask=np.isfinite(primary).all(axis=1),
        strike_points=strikes,
        strike_point_mask=np.isfinite(strikes).all(axis=2),
        topology_class=classes,
        boundary_psi=boundary,
        boundary_flux_mask=np.isfinite(boundary),
        wall_source_shot_id=wall_source_shot_id,
        wall_digest=wall_digest,
    )


def _load_wall(
    store: object,
    root: Path,
    shot_id: int,
) -> tuple[np.ndarray, np.ndarray, int, str]:
    """Load the shot wall or the fixed MAST-era wall when it is omitted."""
    source_shot = shot_id
    if "wall" in store:
        wall = store["wall"]
    else:
        import zarr  # noqa: PLC0415

        source_shot = MAST_WALL_SOURCE_SHOT
        source_path = equilibrium_store_path(source_shot, root)
        source_store = zarr.open_group(str(source_path), mode="r")
        if "wall" not in source_store:
            raise KeyError(
                f"wall source shot {source_shot}: no wall group at {source_path}"
            )
        wall = source_store["wall"]
    wall_r = np.asarray(wall["limiter_r"])
    wall_z = np.asarray(wall["limiter_z"])
    digest = hashlib.sha256(wall_r.tobytes() + wall_z.tobytes()).hexdigest()
    return wall_r, wall_z, source_shot, digest


def load_camera_topology_targets(
    shot_id: int,
    frame_times: np.ndarray,
    *,
    level2_root: Path | None = None,
) -> CameraTopologyTargets:
    """Load one L2 shot and derive topology labels at ``frame_times``."""
    import zarr  # noqa: PLC0415

    root = Path(level2_root) if level2_root is not None else DEFAULT_LEVEL2_ROOT
    path = equilibrium_store_path(int(shot_id), root)
    store = zarr.open_group(str(path), mode="r")
    if "equilibrium" not in store:
        return _excluded_camera_topology_targets(
            int(shot_id), frame_times, "missing_equilibrium_group"
        )
    equilibrium = store["equilibrium"]
    missing = tuple(
        name for name in REQUIRED_EQUILIBRIUM_ARRAYS if name not in equilibrium
    )
    if missing:
        return _excluded_camera_topology_targets(
            int(shot_id), frame_times, _missing_equilibrium_reason(missing)
        )
    wall_r, wall_z, wall_source_shot_id, wall_digest = _load_wall(
        store, root, int(shot_id)
    )
    equilibrium_times = np.asarray(equilibrium["time"])
    if "x_point_r" in equilibrium and "x_point_z" in equilibrium:
        x_point_r = np.asarray(equilibrium["x_point_r"])
        x_point_z = np.asarray(equilibrium["x_point_z"])
    else:
        x_point_r = np.empty((0, equilibrium_times.size), dtype=np.float64)
        x_point_z = np.empty((0, equilibrium_times.size), dtype=np.float64)
    return build_camera_topology_targets_from_arrays(
        shot_id=int(shot_id),
        frame_times=frame_times,
        equilibrium_times=equilibrium_times,
        psi=np.asarray(equilibrium["psi"]),
        major_radius=np.asarray(equilibrium["major_radius"]),
        z=np.asarray(equilibrium["z"]),
        axis_r=np.asarray(equilibrium["magnetic_axis_r"]),
        axis_z=np.asarray(equilibrium["magnetic_axis_z"]),
        x_point_r=x_point_r,
        x_point_z=x_point_z,
        lcfs_r=np.asarray(equilibrium["lcfs_r"]),
        lcfs_z=np.asarray(equilibrium["lcfs_z"]),
        wall_r=wall_r,
        wall_z=wall_z,
        wall_source_shot_id=wall_source_shot_id,
        wall_digest=wall_digest,
    )


__all__ = [
    "MAX_STRIKE_POINTS",
    "MAST_WALL_SOURCE_SHOT",
    "REQUIRED_EQUILIBRIUM_ARRAYS",
    "TOPOLOGY_CLASS_NAMES",
    "TOPOLOGY_UNDEFINED",
    "CameraTopologyTargets",
    "build_camera_topology_targets_from_arrays",
    "camera_topology_exclusion_census",
    "load_camera_topology_targets",
    "reset_camera_topology_exclusion_census",
]
