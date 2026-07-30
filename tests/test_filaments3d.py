"""Tests for the 3-D filament Biot–Savart kernels (:mod:`imas_ambix.gs.filaments3d`).

Analytic pins (square-loop centre field, coaxial-circles Maxwell mutual, on-axis
current-loop field), arc-kernel convergence, the line/arc RDP round-trip, and a
cross-check against ``nova``.  All tests except the two explicitly ``nova``-guarded
ones are ``nova``-free; the ``nova`` cross-check is carried by baked golden values
generated offline from ``nova`` (see :data:`NOVA_PROBE_B`), with a live variant
guarded by ``pytest.importorskip("nova")``.
"""

from __future__ import annotations

import numpy as np
import pytest

from imas_ambix.gs.filaments3d import (
    MU0,
    Arc,
    Conductor,
    Line,
    circle,
    decimate,
    flux_through_loop,
    maxwell_mutual,
    mutual_inductance,
    picture_frame,
    polyline_B,
    probe_response,
    rdp,
    segment_B,
)

# ------------------------------------------------------------------ primitives


def test_square_loop_centre_field():
    """Field at a square loop's centre = 2√2·μ0·I/(π·a) (exact)."""
    a_side, current = 0.6, 1000.0
    sq = np.array(
        [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0], [-1, -1, 0]], float
    ) * (a_side / 2)
    bc = polyline_B(np.array([[0.0, 0.0, 0.0]]), sq, current)[0]
    analytic = 2.0 * np.sqrt(2.0) * MU0 * current / (np.pi * a_side)
    assert abs(bc[2] - analytic) / analytic < 1e-12
    assert abs(bc[0]) < 1e-18 and abs(bc[1]) < 1e-18


def test_coaxial_circles_mutual_vs_maxwell():
    """Coaxial-circles mutual via ∮A·dl matches Maxwell's elliptic formula."""
    m_num = flux_through_loop(circle(1.0, 0.0), 1.0, circle(0.5, 0.3))
    m_ana = maxwell_mutual(1.0, 0.5, 0.3)
    assert abs(m_num - m_ana) / abs(m_ana) < 1e-4


def test_segment_field_perpendicular_bisector():
    """Check the straight-segment field on its perpendicular bisector."""
    a = np.array([-0.5, 0.0, 0.0])
    b = np.array([0.5, 0.0, 0.0])
    rho, current = 0.2, 500.0
    length = 1.0
    bvec = segment_B(np.array([[0.0, rho, 0.0]]), a, b, current)[0]
    analytic = (
        MU0 * current * length / (2 * np.pi * rho * np.sqrt(length**2 + 4 * rho**2))
    )
    assert abs(np.linalg.norm(bvec) - analytic) / analytic < 1e-12


# ------------------------------------------------------------------ arc kernel


def test_arc_field_converges_to_polyline():
    """The adaptive arc field converges to the fine-polyline limit as tol tightens."""
    arc = Arc.from_center(
        (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.2), np.deg2rad(150.0)
    )
    pts = np.array([[0.3, 0.1, 0.5], [1.4, -0.2, -0.3], [0.0, 0.0, 1.0]])
    fine = polyline_B(pts, arc.sample(8000), 1000.0)
    coarse = arc.field(pts, 1000.0, tol=1e-4)
    tight = arc.field(pts, 1000.0, tol=1e-11)
    err_coarse = np.abs(coarse - fine).max() / np.abs(fine).max()
    err_tight = np.abs(tight - fine).max() / np.abs(fine).max()
    assert err_tight < err_coarse  # tighter tolerance is closer to the truth
    assert err_tight < 1e-6


def test_full_circle_on_axis_field():
    """A four-arc circle reproduces the analytic on-axis loop field."""
    radius, current = 0.8, 1234.0
    quarters = [
        Arc.from_center((0, 0, 0), (0, 0, 1), (radius, 0, 0), np.deg2rad(90.0)),
        Arc.from_center((0, 0, 0), (0, 0, 1), (0, radius, 0), np.deg2rad(90.0)),
        Arc.from_center((0, 0, 0), (0, 0, 1), (-radius, 0, 0), np.deg2rad(90.0)),
        Arc.from_center((0, 0, 0), (0, 0, 1), (0, -radius, 0), np.deg2rad(90.0)),
    ]
    loop = Conductor(quarters)
    for z in (0.0, 0.5, -1.2):
        b = loop.field(np.array([[0.0, 0.0, z]]), current)[0]
        analytic = MU0 * current * radius**2 / (2.0 * (radius**2 + z**2) ** 1.5)
        assert abs(b[2] - analytic) / analytic < 1e-6
        assert np.hypot(b[0], b[1]) < 1e-9 * abs(analytic)  # purely axial on axis


def test_arc_from_points_geometry():
    """``Arc.from_points`` recovers the circle radius/centre through three points."""
    arc = Arc.from_points((1.0, 0.0, 0.5), (0.0, 1.0, 0.5), (-1.0, 0.0, 0.5))
    assert abs(arc.radius - 1.0) < 1e-12
    assert np.allclose(arc.centre, (0.0, 0.0, 0.5), atol=1e-12)
    assert abs(abs(arc.angle) - np.pi) < 1e-9


# --------------------------------------------------------------- RDP decimation


def test_rdp_drops_collinear_keeps_corner():
    """RDP keeps endpoints + a genuine corner, drops near-collinear interior points."""
    pts = np.array(
        [[0, 0, 0], [1, 0, 0], [2, 0.0005, 0], [3, 0, 0], [3, 1, 0], [3, 2, 0]], float
    )
    mask = rdp(pts, 1e-2)
    assert mask[0] and mask[-1]
    assert mask.sum() == 3  # start, corner at (3,0,0), end
    assert mask[3]


def test_picture_frame_decimates_to_two_arcs_two_lines():
    """A densely-sampled picture-frame defeatures to 2 arcs + 2 legs, field-faithful."""
    r, z_lo, z_hi = 1.45, -1.0, -0.6
    dense = picture_frame(0.0, np.deg2rad(40.0), r, z_lo, z_hi, n_arc=60, n_leg=40)
    segments = decimate(dense)
    n_arc = sum(isinstance(s, Arc) for s in segments)
    n_line = sum(isinstance(s, Line) for s in segments)
    assert n_arc == 2 and n_line == 2
    # recovered arcs sit on the true cylinder (R, horizontal planes)
    for s in segments:
        if isinstance(s, Arc):
            assert abs(s.radius - r) < 1e-6
            assert abs(s.centre[0]) < 1e-6 and abs(s.centre[1]) < 1e-6

    probes = np.array(
        [[1.85, 0.1, -0.15], [1.7, 0.2, -0.8], [2.0, 0.4, -0.5], [1.3, -0.3, -0.9]]
    )
    b_decimated = Conductor(segments).field(probes, 1.0)
    b_dense = polyline_B(
        probes,
        picture_frame(0.0, np.deg2rad(40.0), r, z_lo, z_hi, n_arc=400, n_leg=200),
        1.0,
    )
    rel = np.abs(b_decimated - b_dense).max() / np.abs(b_dense).max()
    assert rel < 1e-4


# --------------------------------------------------------------- sensor coupling


def test_probe_response_projects_field():
    """Project the probe response onto the normalised pickup normal."""
    loop = circle(0.5, 0.0)
    point = np.array([0.0, 0.0, 0.3])
    resp = probe_response(
        loop, 1000.0, point, np.array([0.0, 0.0, 2.0])
    )  # +z, unnormalised
    b = polyline_B(point[None, :], loop, 1000.0)[0]
    assert abs(resp - b[2]) < 1e-15


def test_mutual_inductance_matches_maxwell_and_is_symmetric():
    """Coaxial ring mutual matches Maxwell's formula and reciprocates."""
    ring_a = Conductor(
        [Arc.from_center((0, 0, 0.0), (0, 0, 1), (1.0, 0, 0.0), 2 * np.pi)]
    )
    ring_b = Conductor(
        [Arc.from_center((0, 0, 0.3), (0, 0, 1), (0.5, 0, 0.3), 2 * np.pi)]
    )
    m_ab = mutual_inductance(ring_a, ring_b, source_arc_points=1440, loop_points=1440)
    m_ba = mutual_inductance(ring_b, ring_a, source_arc_points=1440, loop_points=1440)
    m_ana = maxwell_mutual(1.0, 0.5, 0.3)
    assert abs(m_ab - m_ana) / abs(m_ana) < 1e-3
    assert abs(m_ab - m_ba) / abs(m_ab) < 1e-3


def test_full_loop_flux_consistent_between_a_and_surface_b():
    """The line integral of A equals the surface integral of B."""
    src = circle(1.0, 0.0)
    # sensor loop: small circle in the x=0.3 plane, normal +x
    t = np.linspace(0, 2 * np.pi, 400 + 1)
    r = 0.15
    c = np.array([0.0, 0.0, 0.4])
    loop = c + r * (
        np.cos(t)[:, None] * np.array([1.0, 0, 0])
        + np.sin(t)[:, None] * np.array([0.0, 1, 0])
    )
    flux_a = flux_through_loop(src, 1000.0, loop)
    # surface integral of B_z over the disk (normal +z)
    nrad, nth = 30, 90
    flux_b = 0.0
    for i in range(nrad):
        rho = (i + 0.5) / nrad * r
        for j in range(nth):
            th = 2 * np.pi * j / nth
            p = c + rho * np.array([np.cos(th), np.sin(th), 0.0])
            bz = polyline_B(p[None, :], src, 1000.0)[0, 2]
            flux_b += bz * (r / nrad) * (rho * 2 * np.pi / nth)
    assert abs(flux_a - flux_b) / abs(flux_b) < 1e-3


# ------------------------------------------------------------------ nova cross-check
#
# One picture-frame coil (phi0=0, dphi=40°, R=1.45 m, z∈[-1.0,-0.6], unit current)
# evaluated at three general-position probe points.  The golden field below was
# generated from ``nova`` by decimating the winding to 2 arcs + 2 legs and
# evaluating its independent analytic line/arc kernels.  Bz at the second probe
# is ~0 because z=-0.8 is the coil's up-down symmetry plane.

_FRAME_KW = dict(phi0=0.0, dphi=np.deg2rad(40.0), radius=1.45, z_lo=-1.0, z_hi=-0.6)
NOVA_PROBES = np.array([[1.85, 0.10, -0.15], [1.55, 0.12, -0.80], [2.00, 0.40, -0.50]])
NOVA_PROBE_B = np.array(
    [
        [-1.4392930892263897e-08, 5.494123692414541e-09, 8.565550722245217e-08],
        [1.6376652817195686e-06, 1.5308939612119896e-07, 1.0587911840678754e-22],
        [8.150845301063235e-08, 5.602936265752137e-08, 7.788008401236233e-08],
    ]
)


def _exact_picture_frame_conductor(phi0, dphi, radius, z_lo, z_hi):
    """The picture frame as an exact 2-arc + 2-leg :class:`Conductor`."""
    half = dphi / 2.0
    corner = lambda phi, z: np.array(  # noqa: E731
        [radius * np.cos(phi0 + phi), radius * np.sin(phi0 + phi), z]
    )
    lo_m, lo_p = corner(-half, z_lo), corner(half, z_lo)
    hi_m, hi_p = corner(-half, z_hi), corner(half, z_hi)
    return Conductor(
        [
            Arc.from_center((0, 0, z_lo), (0, 0, 1), lo_m, dphi),
            Line(lo_p, hi_p),
            Arc.from_center((0, 0, z_hi), (0, 0, 1), hi_p, -dphi),
            Line(hi_m, lo_m),
        ]
    )


def test_nova_cross_check_probe_field_golden():
    """Our picture-frame probe field matches the baked ``nova`` golden to ~1e-8."""
    cond = _exact_picture_frame_conductor(**_FRAME_KW)
    b = cond.field(NOVA_PROBES, 1.0)
    assert np.allclose(b, NOVA_PROBE_B, atol=1e-13, rtol=1e-5)


def test_nova_live_cross_check():
    """Cross-check the picture-frame probe field against live ``nova``."""
    pytest.importorskip("nova")
    from nova.biot.arc import Arc as NovaArc  # noqa: PLC0415
    from nova.biot.biotframe import Source, Target  # noqa: PLC0415
    from nova.biot.line import Line as NovaLine  # noqa: PLC0415
    from nova.geometry.polyline import PolyLine  # noqa: PLC0415

    dense = picture_frame(**_FRAME_KW, n_arc=60, n_leg=40)
    polyline = PolyLine(dense, minimum_arc_nodes=4, rdp_eps=1e-4)
    geometry = polyline.path_geometry
    for axis in "xyz":
        geometry[axis] = geometry[f"{axis}0"]
    source = Source(geometry, nturn=1)
    target = Target(
        dict(zip("xyz", NOVA_PROBES.T, strict=True)),
        available=[],
    )

    b_nova = np.zeros_like(NOVA_PROBES)
    source_segments = np.asarray(source["segment"])
    for segment, kernel in (("arc", NovaArc), ("line", NovaLine)):
        mask = source_segments == segment
        segment_source = Source(
            {column: np.asarray(source[column])[mask] for column in source.columns},
            index=list(np.asarray(source.index)[mask]),
        )
        operator = kernel(
            segment_source,
            target,
            turns=False,
            reduce=False,
        )
        b_nova += np.stack(
            [operator.Bx, operator.By, operator.Bz],
            axis=-1,
        ).sum(axis=1)

    assert np.count_nonzero(source_segments == "arc") == 2
    assert np.count_nonzero(source_segments == "line") == 2
    b_tree = _exact_picture_frame_conductor(**_FRAME_KW).field(NOVA_PROBES, 1.0)
    assert np.allclose(b_tree, b_nova, atol=1e-13, rtol=1e-4)
