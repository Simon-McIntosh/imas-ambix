"""Machine-agnostic wall as DATA: supercover raster (tiles-as-holes) + nodes.

The connectivity topology read
(:mod:`imas_ambix.latent.connectivity_boundary`) takes the wall ONLY as a
raster boolean mask (``inside_limiter``) plus a string of wall boundary sample
points (the sub-grid tangency).  This module builds both from an arbitrary list
of wall UNITS, so a single closed loop (MAST), a union of discrete
tiles/limiters (AUG), or a per-pulse movable wall (WEST) is *data*, not a new
code path.

The wall model — two rules, one flood:

* **Occupiable = inside vessel AND NOT material.**  A ``"vessel"`` unit
  contributes its interior to the occupiable region (ray-cast point-in-polygon);
  a ``"material"`` unit (tile / limiter / heat shield) is a HOLE excised from it.
  Every material unit — a closed tile or an open line primitive — is
  *supercover-rasterised*: every cell a wall segment crosses is material, and
  (for a closed unit) every cell whose centre lies inside is material.  Tiles as
  holes make limiter contact and vessel poke-through the SAME escape event in the
  flood: the connectivity read only ever grows the axis-connected component, so
  whatever a hole disconnects is never flooded or queried.  ψ_bnd is unknown
  during the push-out, so geometric contact (a material cell) is the only
  in-sweep stopping test available.

* **Supercover guarantees a ≥1-cell obstacle** for arbitrarily thin blades
  (``t < Δ``): the ≤1-cell thickening biases only the coarse flood bracket, while
  the binding flux is re-read sub-grid from the exact reachable-node flux (the
  campaign ``g_wall`` matrix), so ψ_bnd carries no thickening error.

* **Wall nodes** are every unit's surface resampled at ~Δ/2 arc spacing (finer
  buys nothing — the gridded current model bounds sub-grid flux structure).  The
  tangency read takes the minimum node flux over the REACHABLE nodes (reachable
  from the flood), which naturally excludes private-flux plate nodes (ψ_N < 1)
  and shadowed far-side limiters.

* **Loud diagnostics, never errors, at the (cached) campaign build.**  A material
  unit thinner than the grid (empty interior-fill cell set) is reported with a
  ``2·Area/Perimeter`` thickness proxy vs Δ; a pair of units whose polygons are
  disjoint but whose rasters fuse (sub-cell gap) is reported.  Warning not error —
  the binding flux never comes from the raster, and erroring would reject
  AUG-class thin heat shields at standard campaign grids.

Everything here is pure host geometry, built once per campaign; the mask is a
fixed ``(nz, nr)`` boolean that admits a soft/smoothed differentiable form.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from imas_ambix.latent.topology import _inside_polygon

logger = logging.getLogger("imas_ambix.wall_mask")

__all__ = [
    "WallUnit",
    "WallDiagnostic",
    "supercover_raster",
    "build_wall_mask",
    "densify_units",
    "polygon_area",
    "thickness_proxy",
    "vessel_unit",
    "material_unit",
]


@dataclass(frozen=True)
class WallUnit:
    """One wall primitive: a vessel outline or a material tile/limiter.

    ``kind='vessel'`` marks the occupiable interior (points inside are plasma
    domain); ``kind='material'`` is a HOLE (a tile/limiter/heat shield) excised
    from the domain.  ``closed`` distinguishes a closed polygon (fill the
    interior) from an open line primitive (only the crossed cells are material).
    """

    r: np.ndarray  # (n,) polyline / polygon vertices, R [m]
    z: np.ndarray  # (n,) polyline / polygon vertices, Z [m]
    kind: str = "material"  # "vessel" | "material"
    closed: bool = True
    name: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "r", np.asarray(self.r, dtype=np.float64).ravel())
        object.__setattr__(self, "z", np.asarray(self.z, dtype=np.float64).ravel())
        if self.kind not in ("vessel", "material"):
            raise ValueError(
                f"WallUnit.kind must be vessel|material, got {self.kind!r}"
            )
        if self.r.size != self.z.size:
            raise ValueError("WallUnit r and z must have equal length")


def vessel_unit(r, z, name: str = "vessel") -> WallUnit:
    """A closed vessel outline (occupiable interior)."""
    return WallUnit(r=r, z=z, kind="vessel", closed=True, name=name)


def material_unit(r, z, *, closed: bool = True, name: str = "") -> WallUnit:
    """A material tile/limiter (a hole); ``closed=False`` for an open blade."""
    return WallUnit(r=r, z=z, kind="material", closed=closed, name=name)


@dataclass
class WallDiagnostic:
    """One structured build-time diagnostic (WARNING severity)."""

    kind: str  # "thin_unit" | "gap_merge"
    message: str
    units: tuple[int, ...] = field(default_factory=tuple)
    detail: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# supercover rasterisation
# ---------------------------------------------------------------------------


def _mark_segment(mask: np.ndarray, rg, zg, r0, z0, r1, z1) -> None:
    """Mark every grid cell the segment (r0,z0)→(r1,z1) crosses as material.

    A conservative raster: the segment is walked in steps ≤ ¼ cell and each
    sample's containing cell (nearest grid node) is flagged, so the marked cells
    form a contiguous path and any segment — however thin — leaves at least one
    cell obstacle.  ``mask`` is mutated in place.
    """
    nz, nr = mask.shape
    dr = float(rg[1] - rg[0])
    dz = float(zg[1] - zg[0])
    length = float(np.hypot(r1 - r0, z1 - z0))
    step = 0.25 * min(dr, dz)
    n = max(2, int(np.ceil(length / step)) + 1)
    ts = np.linspace(0.0, 1.0, n)
    rs = r0 + ts * (r1 - r0)
    zs = z0 + ts * (z1 - z0)
    j = np.round((rs - rg[0]) / dr).astype(np.int64)
    i = np.round((zs - zg[0]) / dz).astype(np.int64)
    ok = (i >= 0) & (i < nz) & (j >= 0) & (j < nr)
    mask[i[ok], j[ok]] = True


def supercover_raster(rg, zg, unit: WallUnit) -> np.ndarray:
    """``(nz, nr)`` boolean material raster for one wall unit.

    Every segment of the unit (closed → last→first included) is supercover-marked;
    a closed unit additionally fills the cells whose centres lie inside it.
    """
    rg = np.asarray(rg, dtype=np.float64)
    zg = np.asarray(zg, dtype=np.float64)
    nz, nr = zg.size, rg.size
    mask = np.zeros((nz, nr), dtype=bool)
    r, z = unit.r, unit.z
    if r.size < 2:
        return mask
    idx = list(range(r.size - 1))
    for k in idx:
        _mark_segment(mask, rg, zg, r[k], z[k], r[k + 1], z[k + 1])
    if unit.closed:
        _mark_segment(mask, rg, zg, r[-1], z[-1], r[0], z[0])
        mesh_r, mesh_z = np.meshgrid(rg, zg)
        fill = _inside_polygon(mesh_r.ravel(), mesh_z.ravel(), r, z).reshape(nz, nr)
        mask |= fill
    return mask


# ---------------------------------------------------------------------------
# the machine-agnostic mask
# ---------------------------------------------------------------------------


def _dilate4_np(mask: np.ndarray) -> np.ndarray:
    """4-neighbour boolean dilation (host; no scipy dependency)."""
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def polygon_area(r, z) -> float:
    """Absolute shoelace area of the polygon (closed implicitly)."""
    r = np.asarray(r, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    if r.size < 3:
        return 0.0
    return 0.5 * abs(float(np.dot(r, np.roll(z, -1)) - np.dot(z, np.roll(r, -1))))


def _perimeter(r, z, closed: bool) -> float:
    r = np.asarray(r, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    seg = np.hypot(np.diff(r), np.diff(z)).sum()
    if closed and r.size >= 2:
        seg += float(np.hypot(r[0] - r[-1], z[0] - z[-1]))
    return float(seg)


def thickness_proxy(r, z, closed: bool) -> float:
    """``2·Area/Perimeter`` — the mean cross-section of a thin closed unit [m].

    Zero for an open blade (no enclosed area); a slab of thickness ``t`` and
    length ``L ≫ t`` gives ≈ ``t``.  Compared against Δ to flag sub-grid units.
    """
    per = _perimeter(r, z, closed)
    if per <= 0.0:
        return 0.0
    return 2.0 * polygon_area(r, z) / per


def build_wall_mask(
    rg, zg, units: list[WallUnit]
) -> tuple[np.ndarray, list[WallDiagnostic]]:
    """Occupiable ``(nz, nr)`` boolean mask + build diagnostics from wall units.

    ``inside_limiter = (inside ANY vessel unit) AND NOT (ANY material raster)``.
    One vessel unit with no material reproduces the plain point-in-polygon
    limiter mask (MAST parity).  Returns ``(inside_limiter, diagnostics)``; the
    diagnostics are WARNING-severity structured records (never raised) — a thin
    unit (empty interior fill) with its thickness proxy, and any pair of disjoint
    units whose rasters fuse across a sub-cell gap.
    """
    rg = np.asarray(rg, dtype=np.float64)
    zg = np.asarray(zg, dtype=np.float64)
    nz, nr = zg.size, rg.size
    mesh_r, mesh_z = np.meshgrid(rg, zg)
    flat_r, flat_z = mesh_r.ravel(), mesh_z.ravel()
    delta = min(float(rg[1] - rg[0]), float(zg[1] - zg[0]))

    inside_vessel = np.zeros((nz, nr), dtype=bool)
    material = np.zeros((nz, nr), dtype=bool)
    unit_rasters: dict[int, np.ndarray] = {}
    diagnostics: list[WallDiagnostic] = []

    for k, u in enumerate(units):
        if u.kind == "vessel":
            inside_vessel |= _inside_polygon(flat_r, flat_z, u.r, u.z).reshape(nz, nr)
            continue
        raster = supercover_raster(rg, zg, u)
        unit_rasters[k] = raster
        material |= raster
        # thin-unit diagnostic: the unit is thinner than the grid, so it lives
        # only (or almost only) in its supercover boundary.  Triggered by the
        # 2·Area/Perimeter thickness proxy < Δ (grid-alignment independent — a
        # sub-grid tile that happens to fall on a column still warns) OR an empty
        # interior fill (an open blade, or a unit that cleared no cell centre).
        fill = np.zeros((nz, nr), dtype=bool)
        if u.closed:
            fill = _inside_polygon(flat_r, flat_z, u.r, u.z).reshape(nz, nr)
        t = thickness_proxy(u.r, u.z, u.closed)
        if t < delta or int(fill.sum()) == 0:
            diagnostics.append(
                WallDiagnostic(
                    kind="thin_unit",
                    message=(
                        f"wall unit {k} ({u.name or u.kind}) is sub-grid: "
                        f"supercover-only, thickness proxy {t * 100:.2f} cm vs "
                        f"Δ {delta * 100:.2f} cm — ψ_bnd re-read from exact nodes"
                    ),
                    units=(k,),
                    detail={
                        "thickness_proxy_m": float(t),
                        "delta_m": float(delta),
                        "n_supercover_cells": int(raster.sum()),
                    },
                )
            )

    # gap-merge diagnostic: two material units whose polygons are geometrically
    # separate but whose rasters touch (a sub-cell gap fused by the raster).
    keys = list(unit_rasters)
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            ka, kb = keys[a], keys[b]
            ma, mb = unit_rasters[ka], unit_rasters[kb]
            if not (_dilate4_np(ma) & mb).any():
                continue
            gap = _min_polygon_gap(units[ka], units[kb])
            if gap > 0.5 * delta:  # genuinely separate, yet the rasters fused
                diagnostics.append(
                    WallDiagnostic(
                        kind="gap_merge",
                        message=(
                            f"wall units {ka} and {kb} are {gap * 100:.2f} cm apart "
                            f"(> ½Δ {0.5 * delta * 100:.2f} cm) but their rasters fuse "
                            "— sub-cell gap not resolved at this grid"
                        ),
                        units=(ka, kb),
                        detail={"gap_m": float(gap), "delta_m": float(delta)},
                    )
                )

    for d in diagnostics:
        logger.warning("wall build: %s", d.message)
    inside_limiter = inside_vessel & ~material
    return inside_limiter, diagnostics


def _min_polygon_gap(ua: WallUnit, ub: WallUnit) -> float:
    """Approximate minimum distance between two units' vertex sets [m]."""
    d = np.hypot(ua.r[:, None] - ub.r[None, :], ua.z[:, None] - ub.z[None, :])
    return float(d.min())


# ---------------------------------------------------------------------------
# wall nodes (the sub-grid tangency string)
# ---------------------------------------------------------------------------


def densify_units(
    units: list[WallUnit], spacing: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample every unit's surface to ~``spacing`` arc points.

    Returns ``(wall_r, wall_z, unit_id)`` — the wall node string for the sub-grid
    tangency, tagged by unit.  Every unit (vessel and material) contributes nodes,
    so the plasma can lean on any surface; ``u_wall`` then takes the minimum over
    the reachable subset.  Spacing ~Δ/2 (finer buys nothing).
    """
    rs: list[np.ndarray] = []
    zs: list[np.ndarray] = []
    ids: list[np.ndarray] = []
    spacing = max(float(spacing), 1e-6)
    for k, u in enumerate(units):
        r, z = u.r, u.z
        if r.size < 2:
            continue
        if u.closed:
            r = np.append(r, r[0])
            z = np.append(z, z[0])
        seg = np.hypot(np.diff(r), np.diff(z))
        s = np.concatenate([[0.0], np.cumsum(seg)])
        total = float(s[-1])
        if total <= 0.0:
            continue
        m = max(2, int(np.ceil(total / spacing)))
        # closed units sample [0, total) (endpoint duplicates the start); open
        # units include both endpoints so a blade tip is a node.
        q = np.linspace(0.0, total, m, endpoint=not u.closed)
        rs.append(np.interp(q, s, r))
        zs.append(np.interp(q, s, z))
        ids.append(np.full(q.size, k, dtype=np.int64))
    if not rs:
        return (
            np.array([1.0e30]),
            np.array([1.0e30]),
            np.array([-1], dtype=np.int64),
        )
    return np.concatenate(rs), np.concatenate(zs), np.concatenate(ids)
