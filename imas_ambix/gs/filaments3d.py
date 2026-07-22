"""Non-axisymmetric filament Green's functions: straight segments + circular arcs.

Companion to :mod:`imas_ambix.gs.cylinder` (which handles *complete*,
axisymmetric toroidal conductors).  This module handles the toroidally-**broken**
conductors — in-vessel RMP / error-field-correction / horseshoe coils — whose
field is not axisymmetric and therefore cannot be captured by the elliptic-ring
kernels.  Everything here is a thin-wire (filament) Biot–Savart evaluation of a
3-D conductor built from straight legs and circular arcs.

Self-contained (numpy + ``scipy.special`` only), re-implemented in-tree so the
equilibrium decoder carries its own kernels rather than importing the ``nova``
prototype (``nova/biot/{line,arc}.py``, ``nova/geometry/{polyline,rdp}.py``,
author Simon McIntosh).  ``nova`` remains only the cross-check oracle
(:func:`tests.test_filaments3d`).

Primitives
----------
The straight-segment field ``B`` and vector potential ``A`` are the standard
closed-form filament integrals (arctan / arcsinh antiderivatives), validated to
machine precision against the analytic square-loop centre field and to ~1e-5
against the Maxwell coaxial-mutual formula (the two checks lifted from the
prototype ``scripts/segment_biot_sensor_coupling.py``).

Arc kernel
----------
A circular arc's field is obtained by **adaptive sub-segmentation** of the arc
into straight chords, doubling the chord count until the field/potential at the
evaluation points stops changing to a relative tolerance.  Because the
straight-segment primitive is *exact*, the polyline limit converges to the true
arc integral; the adaptive loop makes that convergence guaranteed and auditable
(demonstrated in the tests) without importing ``nova``'s elliptic-integral arc
form, which is coupled to its dataclass base classes.  A full circle assembled
this way reproduces the analytic on-axis current-loop field.

Geometry / decimation
----------------------
:func:`rdp` (Ramer–Douglas–Peucker, n-D, iterative) and :func:`decimate` defeature
a densely-sampled conductor centreline into a minimal list of :class:`Line` and
:class:`Arc` elements — enough to turn a picture-frame coil into two arcs plus
two legs.  The arc fit is an SVD plane-alignment followed by a least-squares
circle fit, the algorithm ported (numpy-only) from ``nova.geometry.polyline``.

Sensor coupling
---------------
* :func:`probe_response` — ``B·n̂`` at an oriented pickup point;
* :func:`flux_through_loop` — ``∮ A·dl`` flux linkage of an arbitrary closed 3-D
  loop (midpoint rule over loop segments);
* :func:`mutual_inductance` — circuit-level mutual between two conductor loops via
  the ``∮ A·dl`` reciprocity integral (unit source current).

Sign/unit conventions match the prototype: SI, current in ampere-turns, ``B`` in
tesla, ``A`` in T·m, flux in weber, mutual in henry.  ``MU0 = 4e-7·π``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

MU0 = 4.0e-7 * np.pi
_EPS = np.finfo(float).eps


# ------------------------------------------------------------ segment primitives


def segment_B(
    points: np.ndarray, a: np.ndarray, b: np.ndarray, current: float
) -> np.ndarray:
    """``B`` [T] at ``points`` (N,3) from a finite segment ``a→b`` carrying ``current``.

    Closed-form thin-wire Biot–Savart: ``B = μ0 I/(4π ρ)·(cosθ1 − cosθ2)`` in the
    azimuthal direction about the wire, with ``ρ`` the perpendicular distance and
    ``θ`` measured from the two ends.  Exact for a straight filament; singular on
    the wire axis (returned as 0 there via ``nan_to_num``).
    """
    p = np.atleast_2d(np.asarray(points, dtype=np.float64))
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = b - a
    length = np.linalg.norm(d)
    u = d / length
    ap = p - a
    s = ap @ u
    perp = ap - np.outer(s, u)
    rho = np.linalg.norm(perp, axis=1)
    s2 = s - length
    r1 = np.linalg.norm(ap, axis=1)
    r2 = np.linalg.norm(p - b, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        b_mag = MU0 * current / (4.0 * np.pi * rho) * (s / r1 - s2 / r2)
        e_phi = np.cross(np.broadcast_to(u, perp.shape), perp)
        n = np.linalg.norm(e_phi, axis=1, keepdims=True)
        e_phi = np.where(n > 0, e_phi / n, 0.0)
        out = b_mag[:, None] * e_phi
    return np.nan_to_num(out)


def segment_A(
    points: np.ndarray, a: np.ndarray, b: np.ndarray, current: float
) -> np.ndarray:
    """Vector potential ``A`` [T·m] at ``points`` of a finite segment ``a→b`` (Coulomb gauge).

    ``A = μ0 I/(4π)·û·ln((r1 + s)/(r2 + s − L))`` — the arcsinh antiderivative of
    ``1/r`` along the wire, directed along the wire tangent ``û``.
    """
    p = np.atleast_2d(np.asarray(points, dtype=np.float64))
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = b - a
    length = np.linalg.norm(d)
    u = d / length
    s = (p - a) @ u
    r1 = np.linalg.norm(p - a, axis=1)
    r2 = np.linalg.norm(p - b, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        val = MU0 * current / (4.0 * np.pi) * np.log((r1 + s) / (r2 + (s - length)))
    return np.nan_to_num(np.outer(val, u))


def polyline_B(points: np.ndarray, path: np.ndarray, current: float) -> np.ndarray:
    """``B`` at ``points`` from a polyline ``path`` (M,3) carrying ``current`` (sum of segments)."""
    path = np.asarray(path, dtype=np.float64)
    return sum(
        segment_B(points, path[i], path[i + 1], current)
        for i in range(len(path) - 1)
    )


def polyline_A(points: np.ndarray, path: np.ndarray, current: float) -> np.ndarray:
    """``A`` at ``points`` from a polyline ``path`` (M,3) carrying ``current`` (sum of segments)."""
    path = np.asarray(path, dtype=np.float64)
    return sum(
        segment_A(points, path[i], path[i + 1], current)
        for i in range(len(path) - 1)
    )


# ----------------------------------------------------------------- arc geometry


def _orthonormal_frame(axis: np.ndarray, start_radial: np.ndarray) -> tuple:
    """Return in-plane unit vectors ``(e1, e2)`` with ``e1`` along ``start_radial``.

    ``axis`` is the arc-plane normal; ``e1`` is the unit vector from the arc
    centre to the start point; ``e2 = axis × e1`` closes a right-handed frame so a
    point at angle ``θ`` is ``centre + R·(cosθ·e1 + sinθ·e2)``.
    """
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    e1 = np.asarray(start_radial, dtype=np.float64)
    e1 = e1 - (e1 @ axis) * axis  # project into the plane
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    return e1, e2


@dataclass
class Line:
    """A straight conductor leg between two 3-D endpoints."""

    start: np.ndarray
    end: np.ndarray
    name: str = "line"

    def __post_init__(self) -> None:
        self.start = np.asarray(self.start, dtype=np.float64)
        self.end = np.asarray(self.end, dtype=np.float64)

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.end - self.start))

    def sample(self, n: int = 2) -> np.ndarray:
        """Return ``n`` points along the leg (2 is exact for the field)."""
        t = np.linspace(0.0, 1.0, max(2, n))[:, None]
        return (1.0 - t) * self.start + t * self.end

    def field(self, points: np.ndarray, current: float, **_: object) -> np.ndarray:
        return segment_B(points, self.start, self.end, current)

    def potential(self, points: np.ndarray, current: float, **_: object) -> np.ndarray:
        return segment_A(points, self.start, self.end, current)


@dataclass
class Arc:
    """A circular arc conductor in an arbitrary 3-D plane.

    Defined by its ``centre``, plane ``axis`` (normal), ``radius``, the unit
    ``start_radial`` direction (centre→start), and a **signed** span ``angle``
    [rad].  Field and potential are evaluated by adaptive chord sub-segmentation.
    """

    centre: np.ndarray
    axis: np.ndarray
    radius: float
    start_radial: np.ndarray
    angle: float
    name: str = "arc"

    def __post_init__(self) -> None:
        self.centre = np.asarray(self.centre, dtype=np.float64)
        self._e1, self._e2 = _orthonormal_frame(self.axis, self.start_radial)
        self.axis = np.cross(self._e1, self._e2)

    # -- constructors --

    @classmethod
    def from_center(
        cls,
        centre: np.ndarray,
        axis: np.ndarray,
        start_point: np.ndarray,
        angle: float,
    ) -> "Arc":
        """Build an arc from its centre, plane normal, a start point and signed span."""
        centre = np.asarray(centre, dtype=np.float64)
        start_radial = np.asarray(start_point, dtype=np.float64) - centre
        radius = float(np.linalg.norm(start_radial))
        return cls(centre, np.asarray(axis, float), radius, start_radial, float(angle))

    @classmethod
    def from_points(cls, p0: np.ndarray, pmid: np.ndarray, p1: np.ndarray) -> "Arc":
        """Build the circular arc through three 3-D points ``p0 → pmid → p1``."""
        centre, radius, axis = _circle_through_three(p0, pmid, p1)
        e1 = np.asarray(p0, float) - centre
        e1 = e1 / np.linalg.norm(e1)
        e2 = np.cross(axis, e1)
        # signed angles of pmid and p1 in the (e1, e2) frame
        def _ang(pt: np.ndarray) -> float:
            v = np.asarray(pt, float) - centre
            return float(np.arctan2(v @ e2, v @ e1))

        a_mid, a_end = _ang(pmid), _ang(p1)
        a_mid %= 2.0 * np.pi
        a_end %= 2.0 * np.pi
        # choose the sweep that passes through the midpoint
        if a_end == 0.0:
            a_end = 2.0 * np.pi
        angle = a_end if a_mid <= a_end else a_end - 2.0 * np.pi
        return cls(centre, axis, radius, np.asarray(p0, float) - centre, angle)

    @property
    def length(self) -> float:
        return float(self.radius * abs(self.angle))

    def sample(self, n: int) -> np.ndarray:
        """Return ``n`` points along the arc (endpoints included)."""
        theta = np.linspace(0.0, self.angle, max(2, n))[:, None]
        return self.centre + self.radius * (
            np.cos(theta) * self._e1[None, :] + np.sin(theta) * self._e2[None, :]
        )

    def _default_n(self) -> int:
        # ~2 deg chords as the adaptive seed
        return max(8, int(np.ceil(abs(self.angle) / np.deg2rad(2.0))))

    def _adaptive(
        self,
        points: np.ndarray,
        current: float,
        kernel,
        tol: float,
        max_n: int,
    ) -> np.ndarray:
        n = self._default_n()
        prev = kernel(points, self.sample(n), current)
        while n < max_n:
            n *= 2
            cur = kernel(points, self.sample(n), current)
            scale = np.abs(cur).max() + _EPS
            if np.abs(cur - prev).max() / scale < tol:
                return cur
            prev = cur
        return prev

    def field(
        self, points: np.ndarray, current: float, *, tol: float = 1e-8, max_n: int = 8192
    ) -> np.ndarray:
        return self._adaptive(points, current, polyline_B, tol, max_n)

    def potential(
        self, points: np.ndarray, current: float, *, tol: float = 1e-8, max_n: int = 8192
    ) -> np.ndarray:
        return self._adaptive(points, current, polyline_A, tol, max_n)


def _circle_through_three(p0, pmid, p1):
    """Return ``(centre, radius, axis)`` of the circle through three 3-D points."""
    p0 = np.asarray(p0, dtype=np.float64)
    pmid = np.asarray(pmid, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    v1 = pmid - p0
    v2 = p1 - p0
    axis = np.cross(v1, v2)
    norm = np.linalg.norm(axis)
    if norm < 1e-14:
        raise ValueError("three points are collinear — no unique circle")
    axis = axis / norm
    # solve for centre in the plane: |c-p0|=|c-pmid|=|c-p1|
    # centre = p0 + s·v1 + t·v2 with the two perpendicular-bisector conditions
    a = np.array([[v1 @ v1, v1 @ v2], [v1 @ v2, v2 @ v2]])
    rhs = 0.5 * np.array([v1 @ v1, v2 @ v2])
    s, t = np.linalg.solve(a, rhs)
    centre = p0 + s * v1 + t * v2
    radius = float(np.linalg.norm(centre - p0))
    return centre, radius, axis


# ------------------------------------------------------------------- conductor


@dataclass
class Conductor:
    """A 3-D filament conductor: an ordered sequence of :class:`Line` / :class:`Arc`.

    ``field`` / ``potential`` sum the element contributions (arcs adaptively
    sub-segmented); ``path`` returns the assembled discretised centreline used by
    the flux / mutual-inductance integrals.
    """

    elements: list
    name: str = "conductor"

    def field(self, points: np.ndarray, current: float, **kw: object) -> np.ndarray:
        return sum(e.field(points, current, **kw) for e in self.elements)

    def potential(self, points: np.ndarray, current: float, **kw: object) -> np.ndarray:
        return sum(e.potential(points, current, **kw) for e in self.elements)

    def path(self, arc_points: int = 180) -> np.ndarray:
        """Assemble a discretised centreline (arcs sampled at ~``arc_points``/turn)."""
        chunks: list[np.ndarray] = []
        for e in self.elements:
            if isinstance(e, Arc):
                n = max(2, int(np.ceil(arc_points * abs(e.angle) / (2.0 * np.pi))))
                pts = e.sample(n)
            else:
                pts = e.sample(2)
            chunks.append(pts if not chunks else pts[1:])
        return np.vstack(chunks)


# --------------------------------------------------------------- sensor coupling


def probe_response(
    source, current: float, point: np.ndarray, normal: np.ndarray, **kw: object
) -> float:
    """``B·n̂`` [T] at an oriented pickup ``point`` from ``source`` carrying ``current``.

    ``source`` is a :class:`Conductor`, a :class:`Line`/:class:`Arc`, or a
    polyline ``path`` (M,3).  ``normal`` is the probe pickup direction (need not be
    unit — it is normalised here).
    """
    normal = np.asarray(normal, dtype=np.float64)
    normal = normal / np.linalg.norm(normal)
    b = _field_of(source, np.atleast_2d(point), current, **kw)
    return float(b[0] @ normal)


def flux_through_loop(
    source, current: float, loop_points: np.ndarray, **kw: object
) -> float:
    """Flux ``Φ = ∮ A·dl`` [Wb] linked by a closed 3-D ``loop_points`` polyline.

    Midpoint rule over the loop segments.  ``loop_points`` must be closed (first
    point repeated as the last); ``source`` is any object accepted by
    :func:`_field_of` (its ``A`` is evaluated at the segment midpoints).
    """
    loop_points = np.asarray(loop_points, dtype=np.float64)
    mid = 0.5 * (loop_points[:-1] + loop_points[1:])
    dl = loop_points[1:] - loop_points[:-1]
    a = _potential_of(source, mid, current, **kw)
    return float(np.sum(a * dl))


def mutual_inductance(
    source, loop, *, source_arc_points: int = 360, loop_points: int = 720, **kw: object
) -> float:
    """Circuit-level mutual inductance ``M`` [H] between two conductor loops.

    ``M = ∮_loop A_source·dl`` per unit source current (``∮A·dl`` reciprocity).
    ``source`` and ``loop`` may each be a :class:`Conductor` or a closed polyline;
    a :class:`Conductor` loop is discretised at ``loop_points`` per turn.
    """
    loop_path = _as_path(loop, arc_points=loop_points)
    return flux_through_loop(source, 1.0, loop_path, **kw)


def _field_of(source, points: np.ndarray, current: float, **kw: object) -> np.ndarray:
    if isinstance(source, (Conductor, Line, Arc)):
        return source.field(points, current, **kw)
    return polyline_B(points, np.asarray(source, float), current)


def _potential_of(source, points: np.ndarray, current: float, **kw: object) -> np.ndarray:
    if isinstance(source, (Conductor, Line, Arc)):
        return source.potential(points, current, **kw)
    return polyline_A(points, np.asarray(source, float), current)


def _as_path(obj, *, arc_points: int) -> np.ndarray:
    if isinstance(obj, Conductor):
        return obj.path(arc_points=arc_points)
    if isinstance(obj, (Line, Arc)):
        return obj.sample(arc_points if isinstance(obj, Arc) else 2)
    return np.asarray(obj, dtype=np.float64)


# ------------------------------------------------------- RDP / arc decimation


def _point_line_distance(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Perpendicular distance from each of ``points`` to the line ``start–end`` (n-D)."""
    d = end - start
    dl = np.linalg.norm(d)
    if dl < 1e-14:
        return np.linalg.norm(points - start, axis=1)
    # component of (p-start) perpendicular to the unit line direction
    ap = points - start
    proj = np.outer(ap @ (d / dl), d / dl)
    return np.linalg.norm(ap - proj, axis=1)


def rdp(points: np.ndarray, epsilon: float) -> np.ndarray:
    """Ramer–Douglas–Peucker mask (iterative, n-D).

    Returns a boolean mask over ``points`` selecting the decimated vertices such
    that no dropped point lies more than ``epsilon`` from the retained polyline.
    """
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    keep = np.ones(n, dtype=bool)
    stack = [(0, n - 1)]
    while stack:
        i0, i1 = stack.pop()
        if i1 <= i0 + 1:
            continue
        seg = points[i0 + 1 : i1]
        d = _point_line_distance(seg, points[i0], points[i1])
        # only consider indices still kept
        d = np.where(keep[i0 + 1 : i1], d, -1.0)
        k = int(np.argmax(d))
        if d[k] > epsilon:
            idx = i0 + 1 + k
            stack.append((i0, idx))
            stack.append((idx, i1))
        else:
            keep[i0 + 1 : i1] = False
    return keep


def _arc_residual(points: np.ndarray) -> tuple:
    """Fit a circular arc to ``points`` (>=3, N,3); return ``(Arc, normalised_residual)``.

    SVD plane alignment (smallest singular direction = plane normal), 2-D
    least-squares circle fit in that plane.  The residual is the **maximum**
    3-D point-to-circle deviation — combining the in-plane radial error and the
    out-of-plane distance — normalised by the arc length.  Max (not RMS) is used
    deliberately: a couple of corner points where a straight leg meets an arc
    barely move an RMS over dozens of arc points, so an RMS metric silently
    absorbs the corner and tilts the fit; the max breaks the run at the corner.
    """
    points = np.asarray(points, dtype=np.float64)
    centroid = points.mean(axis=0)
    centred = points - centroid
    # plane normal = right-singular vector of smallest singular value
    _, _, vh = np.linalg.svd(centred)
    axis = vh[2]
    e1 = vh[0]
    e2 = vh[1]
    xy = np.column_stack([centred @ e1, centred @ e2])
    # least-squares circle: x^2+y^2 = 2*cx*x + 2*cy*y + (r^2-cx^2-cy^2)
    a_mat = np.column_stack([xy, np.ones(len(xy))])
    rhs = (xy**2).sum(axis=1)
    sol, *_ = np.linalg.lstsq(a_mat, rhs, rcond=None)
    c2d = sol[:2] / 2.0
    radius = float(np.sqrt(sol[2] + c2d @ c2d))
    centre = centroid + c2d[0] * e1 + c2d[1] * e2
    in_plane = np.hypot(xy[:, 0] - c2d[0], xy[:, 1] - c2d[1]) - radius
    out_plane = centred @ axis
    deviation = float(np.max(np.hypot(in_plane, out_plane)))
    arc = Arc.from_points(points[0], points[len(points) // 2], points[-1])
    arc_len = arc.length + _EPS
    return arc, deviation / arc_len


def decimate(
    points: np.ndarray,
    *,
    arc_eps: float = 1e-3,
    line_eps: float = 5e-2,
    rdp_eps: float = 1e-3,
    minimum_arc_nodes: int = 4,
) -> list:
    """Greedily defeature a dense centreline ``points`` (N,3) into ``Line`` / ``Arc`` elements.

    Ported (numpy-only) from ``nova.geometry.polyline``'s hybrid arc/line RDP: at
    each position grow the longest run of points that a single circular arc fits
    to within ``arc_eps`` (normalised residual); arcs subtending less than
    ``line_eps`` radians collapse to their chord; remaining straight runs are
    merged by n-D :func:`rdp` at ``rdp_eps``.  Enough to reduce a picture-frame
    coil to two arcs plus two legs.
    """
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    segments: list = []
    start = 0

    def _fit_run(pts: np.ndarray) -> int:
        """Return the number of leading points a single arc fits within arc_eps."""
        m = len(pts)
        best = 2
        for i in range(minimum_arc_nodes, m + 1):
            try:
                _, res = _arc_residual(pts[:i])
            except (ValueError, np.linalg.LinAlgError):
                break
            if res <= arc_eps:
                best = i
            else:
                break
        return best

    while start <= n - minimum_arc_nodes and minimum_arc_nodes > 0:
        run = _fit_run(points[start:])
        if run >= minimum_arc_nodes:
            arc = Arc.from_points(
                points[start],
                points[start + run // 2],
                points[start + run - 1],
            )
            if abs(arc.angle) < line_eps:
                segments.append(Line(points[start], points[start + run - 1]))
            else:
                segments.append(arc)
            start += run - 1
        else:
            segments.append(Line(points[start], points[start + 1]))
            start += 1

    # trailing straight run
    if n - start > 1:
        segments.append(Line(points[start], points[-1]))

    return _merge_collinear(segments, rdp_eps)


def _merge_collinear(segments: list, rdp_eps: float) -> list:
    """Merge consecutive ``Line`` runs via RDP; leave ``Arc`` elements untouched."""
    out: list = []
    run: list = []

    def _flush() -> None:
        if not run:
            return
        nodes = [run[0].start] + [seg.end for seg in run]
        nodes = np.asarray(nodes, dtype=np.float64)
        mask = rdp(nodes, rdp_eps)
        kept = nodes[mask]
        for i in range(len(kept) - 1):
            out.append(Line(kept[i], kept[i + 1]))
        run.clear()

    for seg in segments:
        if isinstance(seg, Line):
            run.append(seg)
        else:
            _flush()
            out.append(seg)
    _flush()
    return out


# ----------------------------------------------------------------- convenience


def circle(radius: float, z: float, n: int = 720, *, centre_r: float = 0.0) -> np.ndarray:
    """Return a closed axisymmetric loop of ``radius`` at height ``z`` (n+1 points)."""
    t = np.linspace(0.0, 2.0 * np.pi, n + 1)
    return np.column_stack(
        [centre_r + radius * np.cos(t), radius * np.sin(t), np.full(t.size, z)]
    )


def maxwell_mutual(r1: float, r2: float, dz: float) -> float:
    """Maxwell's coaxial-circles mutual inductance [H] (elliptic-integral form)."""
    import scipy.special  # noqa: PLC0415

    k2 = 4.0 * r1 * r2 / ((r1 + r2) ** 2 + dz**2)
    k = np.sqrt(k2)
    return float(
        MU0
        * np.sqrt(r1 * r2)
        * ((2.0 / k - k) * scipy.special.ellipk(k2) - (2.0 / k) * scipy.special.ellipe(k2))
    )


def picture_frame(
    phi0: float,
    dphi: float,
    radius: float,
    z_lo: float,
    z_hi: float,
    *,
    n_arc: int = 60,
    n_leg: int = 40,
) -> np.ndarray:
    """Densely-sampled picture-frame coil on a cylinder (two arcs + two legs).

    A rectangular saddle spanning ``dphi`` in toroidal angle about ``phi0`` at
    major radius ``radius``, between heights ``z_lo`` and ``z_hi``.  Returns a
    closed polyline with every side sampled (so :func:`decimate` recovers
    2 arcs + 2 lines).
    """
    p_lo = np.linspace(phi0 - dphi / 2, phi0 + dphi / 2, n_arc)
    p_hi = p_lo[::-1]
    lower = np.column_stack(
        [radius * np.cos(p_lo), radius * np.sin(p_lo), np.full(n_arc, z_lo)]
    )
    upper = np.column_stack(
        [radius * np.cos(p_hi), radius * np.sin(p_hi), np.full(n_arc, z_hi)]
    )
    # legs sampled densely (exclude shared endpoints to avoid duplicates)
    leg_up = np.linspace(lower[-1], upper[0], n_leg)[1:-1]
    leg_dn = np.linspace(upper[-1], lower[0], n_leg)[1:-1]
    return np.vstack([lower, leg_up, upper, leg_dn, lower[:1]])


__all__ = [
    "MU0",
    "Arc",
    "Line",
    "Conductor",
    "segment_A",
    "segment_B",
    "polyline_A",
    "polyline_B",
    "probe_response",
    "flux_through_loop",
    "mutual_inductance",
    "rdp",
    "decimate",
    "circle",
    "maxwell_mutual",
    "picture_frame",
]
