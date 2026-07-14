"""Deterministic field topology read from the solved poloidal-flux field ψ(R,Z).

Field topology — magnetic axis, X-points, last-closed-flux-surface (LCFS), and
the public/private flux-region split — is a **deterministic read of the one
solved ψ field** (locked decision ``topology-from-psi``), never a supervised
label and never a separate regression head.  The GS observation operator
(:mod:`imas_ambix.latent.gs_observation`) reconstructs ψ from the latent; this
module turns that ψ into geometry the firewalled EFIT referee can *score*.

Method (steered by NOVA ``biot/fieldnull.py`` critical-point finding and
imas-efit ``src/EFIT/contour_tree.f90`` public/private structure):

* **Critical points.**  Grid-bracket where both ∂ψ/∂R and ∂ψ/∂Z change sign,
  Newton-refine with the local gradient + Hessian (bilinear-interpolated), and
  classify by the Hessian: definite → O-point (extremum, magnetic-axis
  candidate), indefinite (det < 0) → X-point (saddle).  Duplicates are merged.
* **Magnetic axis.**  The interior O-point whose ψ is furthest from the domain-
  edge flux — the confinement extremum, current-sign-agnostic.
* **X-point null set.**  The in-vessel saddles, returned as an ORDER-INVARIANT
  set of ≤2 unordered slots (matches the evaluator label design — no sign-of-Z
  split, no ψ-proximity filter; the multimodal-flux caveat §3a).
* **LCFS radii.**  Ray-cast outward from the axis at 8 fixed poloidal angles to
  the bounding flux ψ_bnd (the innermost X-point flux, or the limiter) — the
  same fixed parameterisation the evaluator uses
  (:data:`imas_ambix.worldmodel.equilibrium_labels.LCFS_ANGLES`).
* **Public/private.**  Connected components of the confined-side level set at
  ψ_bnd: the component containing the axis is the CORE; any *other* closed
  pocket at comparable flux is PRIVATE (distinguished by connectivity, NOT by
  ψ height or sign-of-Z — the exact §3a hard case).

Everything here is a numpy read of a solved field: non-differentiable by design
(argmax / connectivity), and it sits *downstream* of the differentiable GS
residual that actually trains the latent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage  # type: ignore[import-untyped]

from imas_ambix.worldmodel.equilibrium_labels import (
    LCFS_ANGLES,
    N_XPOINT_SLOTS,
    TARGET_DIM,
    TARGET_NAMES,
    resample_lcfs_radii,
)

# Region codes returned by :func:`classify_regions`.
REGION_SOL = 0
"""Open scrape-off / outside the LCFS (connected to the domain edge)."""
REGION_CORE = 1
"""The confined region — the closed component containing the magnetic axis."""
REGION_PRIVATE = 2
"""A *closed* pocket at comparable flux NOT containing the axis (private flux)."""


@dataclass
class CriticalPoints:
    """Critical points of ψ: O-points (extrema) and X-points (saddles)."""

    o_points: np.ndarray  # (M, 2) (R, Z) [m]
    o_psi: np.ndarray  # (M,) ψ at each O-point [Wb]
    x_points: np.ndarray  # (K, 2) (R, Z) [m]
    x_psi: np.ndarray  # (K,) ψ at each X-point [Wb]


@dataclass
class TopologyReadout:
    """The oracle-shaped 14-D geometry read from ψ + auxiliary fields."""

    target: np.ndarray  # (14,) axis(2) + X-set(2 slots × 2) + 8 LCFS radii [m]
    names: tuple[str, ...]
    axis: tuple[float, float] | None
    axis_psi: float
    boundary_psi: float
    xpoint_set: np.ndarray  # (N_XPOINT_SLOTS, 2) NaN-padded, order-invariant
    critical_points: CriticalPoints


# --- interpolation helpers -------------------------------------------------


def _bilerp(
    field: np.ndarray, r_1d: np.ndarray, z_1d: np.ndarray, r: float, z: float
) -> float:
    """Bilinear-interpolate a ``(nz, nr)`` field at a physical point ``(r, z)``."""
    nr, nz = r_1d.size, z_1d.size
    fr = np.clip(np.interp(r, r_1d, np.arange(nr)), 0, nr - 1 - 1e-9)
    fz = np.clip(np.interp(z, z_1d, np.arange(nz)), 0, nz - 1 - 1e-9)
    i0, j0 = int(np.floor(fz)), int(np.floor(fr))
    di, dj = fz - i0, fr - j0
    f00 = field[i0, j0]
    f01 = field[i0, j0 + 1]
    f10 = field[i0 + 1, j0]
    f11 = field[i0 + 1, j0 + 1]
    return float(
        f00 * (1 - di) * (1 - dj)
        + f01 * (1 - di) * dj
        + f10 * di * (1 - dj)
        + f11 * di * dj
    )


def _gradient_hessian(
    psi: np.ndarray, r_1d: np.ndarray, z_1d: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Grid gradient (ψ_R, ψ_Z) and Hessian (ψ_RR, ψ_RZ, ψ_ZZ) via np.gradient."""
    gz, gr = np.gradient(psi, z_1d, r_1d)  # axis 0 = Z, axis 1 = R
    grz, grr = np.gradient(gr, z_1d, r_1d)
    gzz, gzr = np.gradient(gz, z_1d, r_1d)
    grz = 0.5 * (grz + gzr)  # symmetrise the mixed partial
    return gr, gz, grr, grz, gzz


# --- critical-point finding ------------------------------------------------


def find_critical_points(
    psi: np.ndarray,
    r_1d: np.ndarray,
    z_1d: np.ndarray,
    *,
    edge_skip: int = 1,
    dedup_tol: float = 1e-3,
) -> CriticalPoints:
    """Find + classify all interior critical points of ψ (O-points, X-points).

    Brackets each 2×2 cell where both gradient components change sign, refines
    with a couple of Newton steps against the bilinear-interpolated gradient,
    and classifies by the Hessian determinant (>0 extremum → O; <0 saddle → X).
    """
    psi = np.asarray(psi, dtype=np.float64)
    r_1d = np.asarray(r_1d, dtype=np.float64)
    z_1d = np.asarray(z_1d, dtype=np.float64)
    gr, gz, grr, grz, gzz = _gradient_hessian(psi, r_1d, z_1d)
    nz, nr = psi.shape

    o_pts: list[tuple[float, float]] = []
    o_val: list[float] = []
    x_pts: list[tuple[float, float]] = []
    x_val: list[float] = []

    # Vectorised bracket detection: a 2×2 cell brackets a critical point iff both
    # gradient components straddle 0 across it (<=/>= so an exactly-on-grid zero
    # is caught) with genuine variation.  Only these few candidate cells get the
    # (Python) Newton refinement — the full-grid scan is pure numpy.
    def _cell_stats(g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a, b = g[:-1, :-1], g[:-1, 1:]
        c, d = g[1:, :-1], g[1:, 1:]
        stack = np.stack([a, b, c, d])
        return stack.min(axis=0), stack.max(axis=0)

    gr_lo, gr_hi = _cell_stats(gr)
    gz_lo, gz_hi = _cell_stats(gz)
    cand = (
        (gr_lo <= 0)
        & (gr_hi >= 0)
        & (gz_lo <= 0)
        & (gz_hi >= 0)
        & (gr_hi > gr_lo)
        & (gz_hi > gz_lo)
    )
    cand[:edge_skip, :] = False
    cand[-edge_skip:, :] = False
    cand[:, :edge_skip] = False
    cand[:, -edge_skip:] = False

    for i, j in np.argwhere(cand):
        i, j = int(i), int(j)
        # refine from the cell centre with Newton on the interpolated grad
        r = 0.5 * (r_1d[j] + r_1d[j + 1])
        z = 0.5 * (z_1d[i] + z_1d[i + 1])
        ok = True
        for _ in range(8):
            g = np.array([_bilerp(gr, r_1d, z_1d, r, z), _bilerp(gz, r_1d, z_1d, r, z)])
            h = np.array(
                [
                    [
                        _bilerp(grr, r_1d, z_1d, r, z),
                        _bilerp(grz, r_1d, z_1d, r, z),
                    ],
                    [
                        _bilerp(grz, r_1d, z_1d, r, z),
                        _bilerp(gzz, r_1d, z_1d, r, z),
                    ],
                ]
            )
            det = h[0, 0] * h[1, 1] - h[0, 1] * h[1, 0]
            if abs(det) < 1e-30:
                ok = False
                break
            dr, dz = np.linalg.solve(h, -g)
            r += float(dr)
            z += float(dz)
            if not (r_1d[0] <= r <= r_1d[-1] and z_1d[0] <= z <= z_1d[-1]):
                ok = False
                break
            if abs(dr) < 1e-6 and abs(dz) < 1e-6:
                break
        if not ok:
            continue
        hrr = _bilerp(grr, r_1d, z_1d, r, z)
        hrz = _bilerp(grz, r_1d, z_1d, r, z)
        hzz = _bilerp(gzz, r_1d, z_1d, r, z)
        det = hrr * hzz - hrz * hrz
        val = _bilerp(psi, r_1d, z_1d, r, z)
        if det > 0:
            o_pts.append((r, z))
            o_val.append(val)
        elif det < 0:
            x_pts.append((r, z))
            x_val.append(val)

    o_points, o_psi = _dedup(o_pts, o_val, dedup_tol)
    x_points, x_psi = _dedup(x_pts, x_val, dedup_tol)
    return CriticalPoints(o_points, o_psi, x_points, x_psi)


def _dedup(
    pts: list[tuple[float, float]], vals: list[float], tol: float
) -> tuple[np.ndarray, np.ndarray]:
    """Merge critical points closer than ``tol`` (metres) into one."""
    if not pts:
        return np.zeros((0, 2)), np.zeros((0,))
    arr = np.array(pts, dtype=np.float64)
    val = np.array(vals, dtype=np.float64)
    keep_p: list[np.ndarray] = []
    keep_v: list[float] = []
    for p, v in zip(arr, val, strict=True):
        if all(np.hypot(*(p - kp)) > tol for kp in keep_p):
            keep_p.append(p)
            keep_v.append(v)
    return np.array(keep_p), np.array(keep_v)


# --- axis / X-point selection ---------------------------------------------


def magnetic_axis(
    psi: np.ndarray,
    r_1d: np.ndarray,
    z_1d: np.ndarray,
    *,
    limiter_r: np.ndarray | None = None,
    limiter_z: np.ndarray | None = None,
    cp: CriticalPoints | None = None,
) -> tuple[float, float] | None:
    """The magnetic axis = interior O-point whose ψ is furthest from the edge.

    Current-sign-agnostic: uses the domain-edge median flux as the reference and
    picks the O-point maximising ``|ψ_O − ψ_edge|`` (the deepest well / highest
    peak).  When a limiter is supplied the search is STRICT: no O-point inside
    the polygon means no axis (None) — an out-of-vessel candidate (a coil
    O-point) is never a magnetic axis, and falling back to it silently was the
    mechanism that put the axis on a PF coil in the first real-data run.
    """
    if cp is None:
        cp = find_critical_points(psi, r_1d, z_1d)
    if cp.o_points.shape[0] == 0:
        return None
    edge = np.concatenate([psi[0, :], psi[-1, :], psi[:, 0], psi[:, -1]])
    psi_edge = float(np.median(edge))
    pts = cp.o_points
    vals = cp.o_psi
    if limiter_r is not None and limiter_z is not None:
        inside = _inside_polygon(
            pts[:, 0], pts[:, 1], np.asarray(limiter_r), np.asarray(limiter_z)
        )
        if not inside.any():
            return None
        pts = pts[inside]
        vals = vals[inside]
    idx = int(np.argmax(np.abs(vals - psi_edge)))
    return float(pts[idx, 0]), float(pts[idx, 1])


def xpoint_set(
    cp: CriticalPoints,
    axis: tuple[float, float] | None,
    *,
    max_slots: int = N_XPOINT_SLOTS,
    limiter_r: np.ndarray | None = None,
    limiter_z: np.ndarray | None = None,
) -> np.ndarray:
    """Order-invariant set of ≤``max_slots`` in-vessel X-points nearest the plasma.

    Returns ``(max_slots, 2)`` NaN-padded (R, Z); the slot index carries NO
    ordering / topology meaning (matches the permutation-invariant evaluator
    label — no sign-of-Z split, no ψ-proximity public/private filter, §3a).
    """
    out = np.full((max_slots, 2), np.nan, dtype=np.float64)
    if cp.x_points.shape[0] == 0:
        return out
    pts = cp.x_points
    if limiter_r is not None and limiter_z is not None:
        inside = _inside_polygon(
            pts[:, 0], pts[:, 1], np.asarray(limiter_r), np.asarray(limiter_z)
        )
        pts = pts[inside]
    if pts.shape[0] == 0:
        return out
    if axis is not None:
        d = np.hypot(pts[:, 0] - axis[0], pts[:, 1] - axis[1])
        pts = pts[np.argsort(d)]
    n = min(max_slots, pts.shape[0])
    out[:n] = pts[:n]
    return out


def boundary_flux(
    cp: CriticalPoints,
    axis: tuple[float, float] | None,
    axis_psi: float,
    *,
    limiter_r: np.ndarray | None = None,
    limiter_z: np.ndarray | None = None,
) -> float | None:
    """The bounding flux ψ_bnd = the innermost in-vessel X-point flux.

    Going outward from the axis, the first saddle reached bounds the LCFS, so
    ψ_bnd is the X-point flux closest to ψ_axis.  Returns None if there is no
    in-vessel X-point (a limited plasma — the caller falls back to the limiter).
    """
    if cp.x_points.shape[0] == 0:
        return None
    xr, xz, xpsi = cp.x_points[:, 0], cp.x_points[:, 1], cp.x_psi
    if limiter_r is not None and limiter_z is not None:
        inside = _inside_polygon(xr, xz, np.asarray(limiter_r), np.asarray(limiter_z))
        xpsi = xpsi[inside]
    if xpsi.size == 0:
        return None
    return float(xpsi[int(np.argmin(np.abs(xpsi - axis_psi)))])


def boundary_flux_robust(
    cp: CriticalPoints,
    axis: tuple[float, float] | None,
    axis_psi: float,
    *,
    limiter_r: np.ndarray | None = None,
    limiter_z: np.ndarray | None = None,
    min_axis_dist: float = 0.0,
) -> float | None:
    """Like :func:`boundary_flux`, with an opt-in minimum-radius saddle guard.

    On a free (non-force-balanced) current the innermost-ψ pick sometimes hits
    a spurious saddle in the current's sensor-null-space concentration — a
    saddle a hair's-breadth from the axis, not the genuine separatrix that
    bounds the whole confined region.  A saddle closer than ``min_axis_dist``
    [m] to the axis is rejected before the innermost-ψ selection runs, so a
    genuine, more distant bounding saddle (or the limiter-contact fallback the
    caller applies when no candidate survives) is picked instead.
    ``min_axis_dist=0.0`` (default) reproduces :func:`boundary_flux` exactly.
    """
    if cp.x_points.shape[0] == 0:
        return None
    xr, xz, xpsi = cp.x_points[:, 0], cp.x_points[:, 1], cp.x_psi
    keep = np.ones(xr.shape, dtype=bool)
    if limiter_r is not None and limiter_z is not None:
        keep &= _inside_polygon(xr, xz, np.asarray(limiter_r), np.asarray(limiter_z))
    if min_axis_dist > 0.0 and axis is not None:
        d = np.hypot(xr - axis[0], xz - axis[1])
        keep &= d >= min_axis_dist
    xpsi = xpsi[keep]
    if xpsi.size == 0:
        return None
    return float(xpsi[int(np.argmin(np.abs(xpsi - axis_psi)))])


# --- LCFS ray-cast ---------------------------------------------------------


def lcfs_radii(
    psi: np.ndarray,
    r_1d: np.ndarray,
    z_1d: np.ndarray,
    axis: tuple[float, float],
    psi_bnd: float,
    *,
    angles: np.ndarray = LCFS_ANGLES,
    n_samples: int = 400,
) -> np.ndarray:
    """Ray-cast radius from the axis to the ψ=ψ_bnd surface at each poloidal angle.

    Matches the evaluator's fixed 8-angle boundary parameterisation
    (:func:`imas_ambix.worldmodel.equilibrium_labels.resample_lcfs_radii`):
    marches outward from the axis along each ray and returns the first radius
    where ψ crosses ψ_bnd (NaN if the ray leaves the grid without crossing).
    """
    ang = np.asarray(angles, dtype=np.float64)
    out = np.full(ang.shape, np.nan, dtype=np.float64)
    ar, az = axis
    psi_axis = _bilerp(psi, r_1d, z_1d, ar, az)
    sign = np.sign(psi_bnd - psi_axis)
    if sign == 0:
        sign = 1.0
    # max ray length: the grid diagonal
    rmax = float(np.hypot(r_1d[-1] - r_1d[0], z_1d[-1] - z_1d[0]))
    ss = np.linspace(0.0, rmax, n_samples)
    for k, th in enumerate(ang):
        cr, sr = np.cos(th), np.sin(th)
        prev_val = 0.0
        prev_s = 0.0
        for m, s in enumerate(ss):
            r = ar + s * cr
            z = az + s * sr
            if not (r_1d[0] <= r <= r_1d[-1] and z_1d[0] <= z <= z_1d[-1]):
                break
            val = (_bilerp(psi, r_1d, z_1d, r, z) - psi_axis) * sign
            target = (psi_bnd - psi_axis) * sign
            if m > 0 and prev_val <= target <= val:
                # linear crossing between prev_s and s
                frac = (
                    0.0 if val == prev_val else (target - prev_val) / (val - prev_val)
                )
                out[k] = prev_s + frac * (s - prev_s)
                break
            prev_val, prev_s = val, s
    return out


# --- LCFS by outermost closed axis-enclosing contour -----------------------


@dataclass
class LcfsContour:
    """The last-closed-flux-surface read as a continuous boundary SHAPE.

    The boundary is the OUTERMOST closed flux contour that encloses the axis and
    still lies entirely inside the limiter — found by one monotone flux-offset
    push, with no limited/diverted branch, no X-point gate, and no distance cap
    (locked design in the ``connectivity-boundary-classification`` plan).  The
    stationary points and the limited/diverted class are read off AFTERWARD, by
    proximity to this ring (:func:`emergent_xpoints`).
    """

    found: bool
    ring: np.ndarray  # (N, 2) (R, Z) closed boundary polygon (empty if not found)
    psi_bnd: float  # flux at the outermost closed in-limiter ring (the separatrix/wall)
    psi_lcfs: float  # the 99.9%-normalised flux the reported ring sits on
    radii: np.ndarray  # (len(angles),) LCFS radii about the axis [m]


_CLOSEPOLY = 79  # contourpy SeparateCode end-of-closed-polygon marker


def _axis_enclosing_ring(
    gen,
    level: float,
    axis: tuple[float, float],
    *,
    min_points: int = 6,
) -> np.ndarray | None:
    """The largest CLOSED contour at ``level`` whose interior contains the axis.

    Uses a persistent contourpy generator (``line_type='SeparateCode'``): a ring
    is closed iff its code array ends with ``CLOSEPOLY`` and it has enough
    vertices to be a real surface (nova's ``Surface.closed`` guard).  Among the
    closed rings that enclose the axis the largest-area one is returned — the
    confined-region boundary — so a nested interior ring never shadows it.
    """
    ar = np.array([axis[0]], dtype=np.float64)
    az = np.array([axis[1]], dtype=np.float64)
    best: np.ndarray | None = None
    best_area = -1.0
    points_list, codes_list = gen.lines(float(level))
    for points, code in zip(points_list, codes_list, strict=True):
        if points.shape[0] < min_points or code[-1] != _CLOSEPOLY:
            continue
        if not _inside_polygon(ar, az, points[:, 0], points[:, 1])[0]:
            continue
        # shoelace area (unsigned) — pick the outermost axis-enclosing ring
        x, y = points[:, 0], points[:, 1]
        area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
        if area > best_area:
            best_area = area
            best = points
    return best


def lcfs_contour(
    psi: np.ndarray,
    r_1d: np.ndarray,
    z_1d: np.ndarray,
    axis: tuple[float, float],
    *,
    limiter_r: np.ndarray | None = None,
    limiter_z: np.ndarray | None = None,
    angles: np.ndarray = LCFS_ANGLES,
    lcfs_norm: float = 0.999,
    n_coarse: int = 6,
    n_bisect: int = 14,
) -> LcfsContour:
    """LCFS = the outermost closed axis-enclosing flux contour inside the limiter.

    One monotone flux-offset push (the ``connectivity-boundary-classification``
    §3 algorithm).  Starting from the axis flux, the level is swept OUTWARD
    (toward the domain-edge extreme); at each level the closed contour that
    encloses the axis is traced (:func:`_axis_enclosing_ring`).  The predicate
    "a closed axis-enclosing ring exists AND lies entirely inside the limiter" is
    monotone — true near the axis, false once the ring pinches at an X-point
    (diverted: level set opens along the legs) or pokes through the wall
    (limited).  A coarse scan brackets the true/false transition, then a
    bisection converges the largest-valid level — the separatrix/wall flux.

    The reported ring sits at the 99.9% normalised flux
    ``ψ_lcfs = ψ_axis + lcfs_norm·(ψ_bnd − ψ_axis)`` (nova's convention) — a hair
    inside the exact separatrix, avoiding the X-point cusp / zero poloidal field.
    The 8 radii are read straight off that polygon with the SAME angle-interp the
    evaluator uses to build its EFIT LCFS labels
    (:func:`imas_ambix.worldmodel.equilibrium_labels.resample_lcfs_radii`) — no
    outward ray-march.

    Handles both topologies with ONE code path and no distance cap: far-field
    garbage is a separate contour ring (fails the enclose-the-axis test) and the
    masked near-pole interior is a deep plateau that never produces a contour in
    the outward sweep, so an interior null can never be selected.
    """
    import contourpy  # noqa: PLC0415 — matplotlib's engine, already a transitive dep

    psi = np.asarray(psi, dtype=np.float64)
    r_1d = np.asarray(r_1d, dtype=np.float64)
    z_1d = np.asarray(z_1d, dtype=np.float64)
    ang = np.asarray(angles, dtype=np.float64)
    nan = LcfsContour(
        found=False,
        ring=np.zeros((0, 2)),
        psi_bnd=float("nan"),
        psi_lcfs=float("nan"),
        radii=np.full(ang.shape, np.nan),
    )

    psi_axis = _bilerp(psi, r_1d, z_1d, float(axis[0]), float(axis[1]))
    if not np.isfinite(psi_axis):
        return nan

    # Outward reference: the domain-edge flux furthest from the axis flux — the
    # boundary side (the masked interior plateau is the deep CONFINED extreme, so
    # the far reference is always on the boundary side, giving a signed span that
    # brackets past the true separatrix and the limiter-contact flux alike).
    edge = np.concatenate([psi[0, :], psi[-1, :], psi[:, 0], psi[:, -1]])
    edge = edge[np.isfinite(edge)]
    if edge.size == 0:
        return nan
    psi_out = float(edge[int(np.argmax(np.abs(edge - psi_axis)))])
    if psi_out == psi_axis:
        return nan

    gen = contourpy.contour_generator(
        r_1d, z_1d, psi, line_type="SeparateCode", quad_as_tri=True
    )
    lim_r = None if limiter_r is None else np.asarray(limiter_r, dtype=np.float64)
    lim_z = None if limiter_z is None else np.asarray(limiter_z, dtype=np.float64)

    def _level(s: float) -> float:
        return psi_axis + s * (psi_out - psi_axis)

    def _valid_ring(s: float) -> np.ndarray | None:
        ring = _axis_enclosing_ring(gen, _level(s), axis)
        if ring is None:
            return None
        if lim_r is not None and lim_z is not None:
            # a poke-out spans many consecutive ring vertices, so a decimated
            # sample (≤~200 points) detects it and keeps the containment test
            # cheap on the dense contourpy rings.
            step = max(1, ring.shape[0] // 200)
            rr, zz = ring[::step, 0], ring[::step, 1]
            if not _inside_polygon(rr, zz, lim_r, lim_z).all():
                return None
        return ring

    # Bracket the monotone valid→invalid transition.  The valid band [0, s*] can
    # be arbitrarily small (a tight limiter, a small plasma), so find a valid
    # lower bracket by geometric backoff from mid-span rather than a fixed grid
    # (a fixed coarse grid misses a narrow valid band and reports no boundary).
    hi = 1.0  # the edge-extreme level: ring spans the grid → outside any limiter
    lo = 0.0
    s = 0.5
    found_lo = False
    for _ in range(n_coarse + n_bisect):
        if _valid_ring(s) is not None:
            lo = s
            found_lo = True
            break
        hi = s
        s *= 0.5
    if not found_lo:
        return nan  # no closed axis-enclosing ring inside the limiter at any level

    # Bisection: converge the largest valid level (the separatrix / wall flux).
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        if _valid_ring(mid) is not None:
            lo = mid
        else:
            hi = mid
    psi_bnd = _level(lo)

    # Step a hair inside (99.9% convention) and read the reported ring + radii.
    psi_lcfs = psi_axis + lcfs_norm * (psi_bnd - psi_axis)
    ring = _axis_enclosing_ring(gen, psi_lcfs, axis)
    if ring is None:
        ring = _valid_ring(lo)  # fall back to the exact largest-valid ring
    if ring is None:
        return nan
    radii = resample_lcfs_radii(
        ring[:, 0], ring[:, 1], float(axis[0]), float(axis[1]), ang
    )
    return LcfsContour(
        found=True,
        ring=ring,
        psi_bnd=float(psi_bnd),
        psi_lcfs=float(psi_lcfs),
        radii=radii,
    )


def emergent_xpoints(
    x_points: np.ndarray,
    ring: np.ndarray,
    *,
    tol: float,
    max_slots: int = N_XPOINT_SLOTS,
) -> tuple[np.ndarray, bool]:
    """The limited/diverted class + X-point slots, read AFTER the boundary.

    Given the already-found LCFS ``ring`` and candidate saddles ``x_points``
    ((K, 2) (R, Z)), an X-point lying within ``tol`` [m] of the ring is ON the
    boundary ⇒ the plasma is DIVERTED and that X-point is reported; otherwise
    the ring sits on the wall ⇒ LIMITED and the X-slots are NaN.  The tolerance
    is a SOFT margin, so near the limited↔diverted transition (where the saddle
    sits a hair off the ring) the class is genuinely undefined rather than
    forced — never a code-path switch.  Returns ``(xset (max_slots, 2) NaN-
    padded, is_diverted)``.
    """
    out = np.full((max_slots, 2), np.nan, dtype=np.float64)
    x_points = np.asarray(x_points, dtype=np.float64).reshape(-1, 2)
    if x_points.shape[0] == 0 or ring.shape[0] == 0:
        return out, False
    # distance of each candidate saddle to the ring polygon (nearest vertex —
    # the ring is densely sampled by contourpy, so vertex distance ≈ edge dist).
    d = np.hypot(
        x_points[:, None, 0] - ring[None, :, 0],
        x_points[:, None, 1] - ring[None, :, 1],
    ).min(axis=1)
    on_ring = d <= tol
    if not on_ring.any():
        return out, False  # no boundary X-point → limited
    near = x_points[on_ring]
    order = np.argsort(d[on_ring])  # closest to the ring first
    near = near[order]
    n = min(max_slots, near.shape[0])
    out[:n] = near[:n]
    return out, True


# --- public/private via connectivity --------------------------------------


def classify_regions(
    psi: np.ndarray,
    r_1d: np.ndarray,
    z_1d: np.ndarray,
    axis: tuple[float, float],
    psi_bnd: float,
) -> np.ndarray:
    """Label each grid cell CORE / PRIVATE / SOL by connectivity at ψ=ψ_bnd.

    The confined side is ``(ψ − ψ_axis)·sign(ψ_bnd − ψ_axis) ≤ (ψ_bnd − ψ_axis)·
    sign(...)`` — i.e. flux between the axis and the bounding surface.  Connected
    components of that set are found; the component containing the axis is CORE,
    any *other* component NOT touching the domain edge is PRIVATE, and the rest
    is SOL.  This distinguishes a private pocket from the core by CONNECTIVITY,
    never by ψ height or sign-of-Z (§3a).
    """
    psi = np.asarray(psi, dtype=np.float64)
    ar, az = axis
    psi_axis = _bilerp(psi, r_1d, z_1d, ar, az)
    sign = np.sign(psi_bnd - psi_axis) or 1.0
    level = (psi - psi_axis) * sign
    thr = (psi_bnd - psi_axis) * sign
    confined = level <= thr  # boolean (nz, nr)

    labels, _ = ndimage.label(confined)
    ia = int(np.argmin(np.abs(z_1d - az)))
    ja = int(np.argmin(np.abs(r_1d - ar)))
    axis_comp = labels[ia, ja]

    edge_comps = (
        set(labels[0, :]) | set(labels[-1, :]) | set(labels[:, 0]) | set(labels[:, -1])
    )
    edge_comps.discard(0)

    out = np.full(psi.shape, REGION_SOL, dtype=np.int64)
    if axis_comp != 0:
        out[labels == axis_comp] = REGION_CORE
    for comp in range(1, labels.max() + 1):
        if comp == axis_comp:
            continue
        if comp in edge_comps:
            continue  # open → SOL
        out[labels == comp] = REGION_PRIVATE  # closed pocket, not the core
    return out


# --- top-level read --------------------------------------------------------


def filter_critical_points(
    cp: CriticalPoints, bbox: tuple[float, float, float, float]
) -> CriticalPoints:
    """Keep only critical points inside ``bbox = (r_lo, r_hi, z_lo, z_hi)``.

    Used to restrict the axis / X-point search to the plasma-current region so
    strong PF-coil flux extrema (O-points AT the coils, outside the plasma) are
    not mistaken for the magnetic axis.
    """
    r_lo, r_hi, z_lo, z_hi = bbox

    def _keep(pts: np.ndarray, val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if pts.shape[0] == 0:
            return pts, val
        m = (
            (pts[:, 0] >= r_lo)
            & (pts[:, 0] <= r_hi)
            & (pts[:, 1] >= z_lo)
            & (pts[:, 1] <= z_hi)
        )
        return pts[m], val[m]

    o_pts, o_psi = _keep(cp.o_points, cp.o_psi)
    x_pts, x_psi = _keep(cp.x_points, cp.x_psi)
    return CriticalPoints(o_pts, o_psi, x_pts, x_psi)


def exclude_near_points(
    cp: CriticalPoints, points: np.ndarray, radius: float
) -> CriticalPoints:
    """Drop critical points within ``radius`` of any of ``points`` (M, 2).

    MAST has IN-VESSEL PF coils: the total-ψ field has strong O-points AT the
    coil filaments that are not the plasma.  Excluding critical points near the
    known filament locations removes those coil artifacts while keeping the
    plasma axis / divertor X-points (which sit in the open plasma volume).
    """
    if points is None or len(points) == 0:
        return cp
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)

    def _keep(cpts: np.ndarray, val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if cpts.shape[0] == 0:
            return cpts, val
        d = np.hypot(
            cpts[:, None, 0] - pts[None, :, 0], cpts[:, None, 1] - pts[None, :, 1]
        )
        keep = d.min(axis=1) > radius
        return cpts[keep], val[keep]

    o_pts, o_psi = _keep(cp.o_points, cp.o_psi)
    x_pts, x_psi = _keep(cp.x_points, cp.x_psi)
    return CriticalPoints(o_pts, o_psi, x_pts, x_psi)


def read_topology(
    psi: np.ndarray,
    r_1d: np.ndarray,
    z_1d: np.ndarray,
    *,
    limiter_r: np.ndarray | None = None,
    limiter_z: np.ndarray | None = None,
    search_bbox: tuple[float, float, float, float] | None = None,
    exclude_rz: np.ndarray | None = None,
    exclude_radius: float = 0.12,
) -> TopologyReadout:
    """Full deterministic read of ψ → the 14-D oracle-shaped geometry target.

    Layout matches :data:`imas_ambix.worldmodel.equilibrium_labels.TARGET_NAMES`:
    ``axis_R, axis_Z`` then the order-invariant X-point set (``xpt0``, ``xpt1``)
    then 8 LCFS control-point radii about the axis.  Absent components are NaN.

    ``search_bbox = (r_lo, r_hi, z_lo, z_hi)`` restricts the axis / X-point
    search to the plasma-current region so PF-coil O-points (outside the plasma)
    are not picked as the magnetic axis.
    """
    psi = np.asarray(psi, dtype=np.float64)
    r_1d = np.asarray(r_1d, dtype=np.float64)
    z_1d = np.asarray(z_1d, dtype=np.float64)
    cp = find_critical_points(psi, r_1d, z_1d)
    if search_bbox is not None:
        cp = filter_critical_points(cp, search_bbox)
    if exclude_rz is not None:
        cp = exclude_near_points(cp, exclude_rz, exclude_radius)
    axis = magnetic_axis(
        psi, r_1d, z_1d, limiter_r=limiter_r, limiter_z=limiter_z, cp=cp
    )

    target = np.full(TARGET_DIM, np.nan, dtype=np.float64)
    axis_psi = np.nan
    psi_bnd = np.nan
    xset = np.full((N_XPOINT_SLOTS, 2), np.nan, dtype=np.float64)

    if axis is not None:
        target[0], target[1] = axis
        axis_psi = _bilerp(psi, r_1d, z_1d, axis[0], axis[1])
        xset = xpoint_set(cp, axis, limiter_r=limiter_r, limiter_z=limiter_z)
        target[2 : 2 + 2 * N_XPOINT_SLOTS] = xset.reshape(-1)

        bnd = boundary_flux(
            cp, axis, axis_psi, limiter_r=limiter_r, limiter_z=limiter_z
        )
        if bnd is None:
            # limited plasma: bound at the limiter flux closest to the axis flux
            if limiter_r is not None and limiter_z is not None:
                lp = np.array(
                    [
                        _bilerp(psi, r_1d, z_1d, float(lr), float(lz))
                        for lr, lz in zip(
                            np.asarray(limiter_r), np.asarray(limiter_z), strict=True
                        )
                    ]
                )
                bnd = (
                    float(lp[int(np.argmin(np.abs(lp - axis_psi)))])
                    if lp.size
                    else np.nan
                )
            else:
                # no limiter: bound at the edge-nearest closed surface (median edge)
                edge = np.concatenate([psi[0, :], psi[-1, :], psi[:, 0], psi[:, -1]])
                bnd = float(np.median(edge))
        psi_bnd = float(bnd)
        if np.isfinite(psi_bnd):
            target[2 + 2 * N_XPOINT_SLOTS :] = lcfs_radii(
                psi, r_1d, z_1d, axis, psi_bnd
            )

    return TopologyReadout(
        target=target,
        names=TARGET_NAMES,
        axis=axis,
        axis_psi=float(axis_psi),
        boundary_psi=float(psi_bnd),
        xpoint_set=xset,
        critical_points=cp,
    )


# --- geometry helper -------------------------------------------------------


def _inside_polygon(
    px: np.ndarray, py: np.ndarray, vx: np.ndarray, vy: np.ndarray
) -> np.ndarray:
    """Ray-casting point-in-polygon (limiter mask); no shapely dependency.

    Fully vectorised over BOTH the query points and the polygon edges (no Python
    per-vertex loop) — the boundary contour push tests hundred-vertex rings
    against the limiter tens of times per slice, so the O(points × edges) numpy
    form is the difference between a sub-ms and a tens-of-ms read.
    """
    px = np.asarray(px, dtype=np.float64)
    py = np.asarray(py, dtype=np.float64)
    shape = px.shape
    pxf = px.ravel()
    pyf = py.ravel()
    vx = np.asarray(vx, dtype=np.float64).ravel()
    vy = np.asarray(vy, dtype=np.float64).ravel()
    vxj = np.roll(vx, 1)  # edge (j, i) with j = i-1 (matches the original loop)
    vyj = np.roll(vy, 1)
    # (P, E): does the ray from each point cross each edge?
    straddle = (vy[None, :] > pyf[:, None]) != (vyj[None, :] > pyf[:, None])
    x_cross = (vxj - vx)[None, :] * (pyf[:, None] - vy[None, :]) / (
        (vyj - vy)[None, :] + 1e-30
    ) + vx[None, :]
    crossings = straddle & (pxf[:, None] < x_cross)
    inside = (crossings.sum(axis=1) & 1).astype(bool)
    return inside.reshape(shape)


__all__ = [
    "REGION_SOL",
    "REGION_CORE",
    "REGION_PRIVATE",
    "CriticalPoints",
    "TopologyReadout",
    "LcfsContour",
    "lcfs_contour",
    "emergent_xpoints",
    "find_critical_points",
    "filter_critical_points",
    "exclude_near_points",
    "magnetic_axis",
    "xpoint_set",
    "boundary_flux",
    "boundary_flux_robust",
    "lcfs_radii",
    "classify_regions",
    "read_topology",
]
