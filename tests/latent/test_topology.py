"""Tests for deterministic field topology read from the solved ψ field.

Topology (magnetic axis, X-points, LCFS, public/private) is a *deterministic
read* of the one solved poloidal-flux field ψ(R,Z) — never a supervised label
(locked: topology-from-psi).  These tests pin the read on analytic fields with
KNOWN critical-point structure so correctness is absolute (no EFIT, no data):

* Hessian classification — a well is an O-point (extremum), a saddle is an
  X-point (indefinite Hessian);
* a double-well field has two O-points (minima) bracketing one X-point
  (saddle) — locations + types recovered;
* the LCFS ray-cast radii on a circular boundary equal the boundary radius;
* **the public/private caveat (§3a)**: two nulls at *comparable ψ height* that
  are separated only by connectivity are put in DIFFERENT regions — the axis
  region does not swallow the private pocket.  This is the exact failure a
  sign-of-Z or ψ-proximity heuristic makes and the contour-tree read must not.
"""

from __future__ import annotations

import numpy as np

from imas_ambix.latent import topology as topo


def _grid(nr=161, nz=161, r=(0.3, 1.7), z=(-1.0, 1.0)):
    r_1d = np.linspace(r[0], r[1], nr)
    z_1d = np.linspace(z[0], z[1], nz)
    RR, ZZ = np.meshgrid(r_1d, z_1d)  # (nz, nr)
    return r_1d, z_1d, RR, ZZ


def test_single_well_is_an_o_point():
    """ψ = (R-R0)^2 + Z^2 (a minimum) → exactly one O-point at (R0, 0)."""
    r_1d, z_1d, RR, ZZ = _grid()
    r0 = 1.0
    psi = (RR - r0) ** 2 + ZZ**2
    cp = topo.find_critical_points(psi, r_1d, z_1d)
    assert cp.o_points.shape[0] == 1
    assert cp.x_points.shape[0] == 0
    np.testing.assert_allclose(cp.o_points[0], [r0, 0.0], atol=0.02)


def test_saddle_is_an_x_point():
    """ψ = (R-R0)^2 - Z^2 (a saddle) → exactly one X-point at (R0, 0)."""
    r_1d, z_1d, RR, ZZ = _grid()
    r0 = 1.0
    psi = (RR - r0) ** 2 - ZZ**2
    cp = topo.find_critical_points(psi, r_1d, z_1d)
    assert cp.x_points.shape[0] == 1
    assert cp.o_points.shape[0] == 0
    np.testing.assert_allclose(cp.x_points[0], [r0, 0.0], atol=0.02)


def test_double_well_two_o_points_bracket_one_x_point():
    """ψ = ((R-R0)^2/s^2 - 1)^2 + (Z/s)^2 → O at (R0±s, 0), X at (R0, 0)."""
    r_1d, z_1d, RR, ZZ = _grid(nr=221, nz=181)
    r0, s = 1.0, 0.35
    x = (RR - r0) / s
    y = ZZ / s
    psi = (x**2 - 1.0) ** 2 + y**2
    cp = topo.find_critical_points(psi, r_1d, z_1d)
    assert cp.o_points.shape[0] == 2
    assert cp.x_points.shape[0] == 1
    np.testing.assert_allclose(cp.x_points[0], [r0, 0.0], atol=0.02)
    o_sorted = cp.o_points[np.argsort(cp.o_points[:, 0])]
    np.testing.assert_allclose(o_sorted[0], [r0 - s, 0.0], atol=0.03)
    np.testing.assert_allclose(o_sorted[1], [r0 + s, 0.0], atol=0.03)


def test_magnetic_axis_is_the_interior_extremum():
    """A biased double-well → the deeper minimum is the magnetic axis."""
    r_1d, z_1d, RR, ZZ = _grid(nr=221, nz=181)
    r0, s = 1.0, 0.35
    x = (RR - r0) / s
    y = ZZ / s
    # tilt makes the LEFT well (x<0) deeper → that is the confinement centre
    psi = (x**2 - 1.0) ** 2 + y**2 + 0.4 * x
    axis = topo.magnetic_axis(psi, r_1d, z_1d)
    assert axis is not None
    assert axis[0] < r0  # the deeper (left) well


def test_lcfs_radii_on_circular_boundary_equal_radius():
    """ψ = distance-from-axis^2 → LCFS at ψ=Rb^2 gives all 8 radii ≈ Rb."""
    r_1d, z_1d, RR, ZZ = _grid(nr=241, nz=241, r=(0.2, 1.8), z=(-0.8, 0.8))
    axis = (1.0, 0.0)
    psi = (RR - axis[0]) ** 2 + ZZ**2
    rb = 0.3
    radii = topo.lcfs_radii(psi, r_1d, z_1d, axis, rb**2)
    assert radii.shape == (8,)
    assert np.isfinite(radii).all()
    np.testing.assert_allclose(radii, rb, atol=0.02)


def test_public_private_separated_by_connectivity_not_height():
    """§3a: two wells at EQUAL ψ depth, separated by a saddle, must land in
    DIFFERENT regions — connectivity distinguishes them, not ψ height."""
    r_1d, z_1d, RR, ZZ = _grid(nr=221, nz=181)
    r0, s = 1.0, 0.35
    x = (RR - r0) / s
    y = ZZ / s
    psi = (x**2 - 1.0) ** 2 + y**2  # symmetric: both minima at ψ=0
    # pick the LEFT well as the axis; classify at a level just above the saddle
    axis = (r0 - s, 0.0)
    saddle_psi = 1.0  # ψ at (R0,0): (0-1)^2 = 1
    labels = topo.classify_regions(psi, r_1d, z_1d, axis, saddle_psi * 0.6)
    # axis cell and the RIGHT-well cell must have different region labels
    ir_axis = np.argmin(np.abs(r_1d - (r0 - s)))
    iz_axis = np.argmin(np.abs(z_1d - 0.0))
    ir_other = np.argmin(np.abs(r_1d - (r0 + s)))
    lab_axis = labels[iz_axis, ir_axis]
    lab_other = labels[iz_axis, ir_other]
    assert lab_axis == topo.REGION_CORE
    assert lab_other != topo.REGION_CORE
    assert lab_other != topo.REGION_SOL  # it IS a closed pocket, not open SOL


def test_read_topology_returns_oracle_shaped_target():
    """read_topology yields the 14-D oracle target (axis, ≤2 X-set, 8 LCFS)."""
    r_1d, z_1d, RR, ZZ = _grid(nr=201, nz=201, r=(0.2, 1.8), z=(-0.9, 0.9))
    axis_true = (1.0, 0.0)
    psi = (RR - axis_true[0]) ** 2 + ZZ**2  # limited circular plasma
    out = topo.read_topology(psi, r_1d, z_1d)
    assert out.target.shape == (14,)
    # axis recovered
    np.testing.assert_allclose(out.target[:2], axis_true, atol=0.03)


def test_search_bbox_excludes_coil_like_edge_extremum():
    """A strong edge 'coil' O-point must not be picked as the axis when the
    search is restricted to the plasma-current region (the real-data failure:
    the GS ψ read picked the PF-coil O-point instead of the plasma axis)."""
    r_1d, z_1d, RR, ZZ = _grid(nr=161, nz=161, r=(0.3, 1.7), z=(-1.0, 1.0))
    # weak plasma well near (0.9, 0) + a strong 'coil' well near the edge (1.5, 0.8)
    plasma = -np.exp(-(((RR - 0.9) / 0.15) ** 2 + (ZZ / 0.2) ** 2))
    coil = -5.0 * np.exp(-(((RR - 1.5) / 0.05) ** 2 + ((ZZ - 0.8) / 0.05) ** 2))
    psi = plasma + coil
    # unrestricted: the deep coil well dominates → axis near the coil
    axis_all = topo.read_topology(psi, r_1d, z_1d).axis
    assert axis_all[0] > 1.3  # picked the coil (the failure mode)
    # restricted to the plasma-current bbox → the plasma axis is recovered
    read = topo.read_topology(psi, r_1d, z_1d, search_bbox=(0.4, 1.2, -0.6, 0.6))
    assert read.axis is not None
    np.testing.assert_allclose(read.axis, [0.9, 0.0], atol=0.05)


def test_magnetic_axis_returns_none_when_no_o_point_inside_limiter():
    """With a limiter supplied and NO O-point inside it, the axis must be None
    — never a fall-back to out-of-vessel candidates (a coil O-point is not a
    magnetic axis)."""
    r_1d, z_1d, RR, ZZ = _grid(nr=121, nz=121, r=(0.3, 1.7), z=(-1.0, 1.0))
    # single deep well OUTSIDE the (small, central) limiter polygon
    psi = -np.exp(-(((RR - 1.5) / 0.06) ** 2 + ((ZZ - 0.8) / 0.06) ** 2))
    lim_r = np.array([0.7, 1.1, 1.1, 0.7])
    lim_z = np.array([-0.3, -0.3, 0.3, 0.3])
    axis = topo.magnetic_axis(psi, r_1d, z_1d, limiter_r=lim_r, limiter_z=lim_z)
    assert axis is None


# --- LCFS by outermost closed axis-enclosing contour (connectivity read) ----


def _circle_polygon(r0, z0, rad, n=200):
    """A closed circular polygon (limiter / boundary reference)."""
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return r0 + rad * np.cos(t), z0 + rad * np.sin(t)


def test_lcfs_contour_limited_ring_on_wall_xslots_empty():
    """A circular well with NO saddle: the outermost closed axis-enclosing ring
    stops at the limiter (the wall), the 8 radii equal the limiter radius, and
    with no X-point the emergent read is LIMITED (X-slots NaN)."""
    r_1d, z_1d, RR, ZZ = _grid(nr=201, nz=201, r=(0.3, 1.7), z=(-1.0, 1.0))
    r0, rb = 1.0, 0.4
    psi = (RR - r0) ** 2 + ZZ**2  # minimum (axis) at (r0, 0)
    lim_r, lim_z = _circle_polygon(r0, 0.0, rb)
    out = topo.lcfs_contour(
        psi, r_1d, z_1d, (r0, 0.0), limiter_r=lim_r, limiter_z=lim_z
    )
    assert out.found
    # ring rides the wall: every fixed-angle radius ≈ the limiter radius
    np.testing.assert_allclose(out.radii, rb, atol=0.03)
    # no saddle anywhere → limited, X-slots empty
    cp = topo.find_critical_points(psi, r_1d, z_1d)
    xset, diverted = topo.emergent_xpoints(cp.x_points, out.ring, tol=0.05)
    assert not diverted
    assert np.isnan(xset).all()


def test_lcfs_contour_diverted_ring_pinches_at_xpoint():
    """A biased double-well (one deep O = axis, a saddle just below it): the
    outermost closed axis-enclosing ring pinches at the X-point (the level set
    opens along the 'legs' past it), so the emergent read reports DIVERTED and
    the X-point sits ON the ring."""
    r_1d, z_1d, RR, ZZ = _grid(nr=241, nz=241, r=(0.3, 1.7), z=(-1.0, 1.0))
    r0, s = 1.0, 0.32
    x = (RR - r0) / s
    y = ZZ / s
    # wells at Z=±s, saddle at Z≈0; the -g·y bias deepens the UPPER well (axis)
    psi = (y**2 - 1.0) ** 2 + x**2 - 0.5 * y
    cp = topo.find_critical_points(psi, r_1d, z_1d)
    # the deepest (most negative) O-point is the axis; the true saddle is the X
    axis = tuple(cp.o_points[int(np.argmin(cp.o_psi))])
    assert cp.x_points.shape[0] == 1
    x_true = cp.x_points[0]
    # limiter hugs the confined region: past the separatrix the level set opens
    # toward the second (divertor-side) well, which lies OUTSIDE this wall — so
    # the outermost closed in-limiter ring is the separatrix (the real-plasma
    # mechanism: the divertor legs exit the vessel).
    lim_r, lim_z = _circle_polygon(axis[0], axis[1], 0.45)
    out = topo.lcfs_contour(psi, r_1d, z_1d, axis, limiter_r=lim_r, limiter_z=lim_z)
    assert out.found
    # the X-point lies on the found ring (within a few cm) → diverted
    d_ring = np.hypot(out.ring[:, 0] - x_true[0], out.ring[:, 1] - x_true[1]).min()
    assert d_ring < 0.05, f"X-point not on the ring: min dist {d_ring:.3f} m"
    xset, diverted = topo.emergent_xpoints(cp.x_points, out.ring, tol=0.05)
    assert diverted
    np.testing.assert_allclose(xset[0], x_true, atol=0.05)


def test_lcfs_contour_far_field_feature_excluded_no_distance_cap():
    """A spurious far-field feature at |Z|≈0.85 (its own closed contour, NOT
    enclosing the axis) must not perturb the LCFS — it is excluded by the
    enclose-the-axis test with NO tuned distance cap."""
    r_1d, z_1d, RR, ZZ = _grid(nr=241, nz=241, r=(0.3, 1.7), z=(-1.0, 1.0))
    r0, rb = 1.0, 0.35
    psi = (RR - r0) ** 2 + ZZ**2
    # a deep, tight spurious dimple far from the axis (a separate closed ring)
    psi = psi - 3.0 * np.exp(-(((RR - r0) / 0.05) ** 2 + ((ZZ - 0.85) / 0.05) ** 2))
    lim_r, lim_z = _circle_polygon(r0, 0.0, rb)  # limiter excludes the far feature
    out = topo.lcfs_contour(
        psi, r_1d, z_1d, (r0, 0.0), limiter_r=lim_r, limiter_z=lim_z
    )
    assert out.found
    # LCFS is still the near circular boundary at the wall — the far dimple, a
    # separate non-axis-enclosing ring, changed nothing.
    np.testing.assert_allclose(out.radii, rb, atol=0.04)


def test_lcfs_contour_masked_interior_null_never_selected():
    """An interior null inside the masked near-pole disk must never be selected:
    mask_invalid_interior fills the disk with a deep plateau that produces no
    contour in the outward sweep, so the LCFS is the true annulus boundary."""
    from imas_ambix.latent.boundary_harmonic import mask_invalid_interior

    r_1d, z_1d, RR, ZZ = _grid(nr=241, nz=241, r=(0.3, 1.7), z=(-1.0, 1.0))
    r0, rb = 0.9, 0.42
    axis = (r0, 0.0)
    pole_r = 0.62  # inboard of the axis, comfortably inside the limiter
    psi = (RR - r0) ** 2 + ZZ**2
    # a spurious tight interior dimple AT the pole that, unmasked, manufactures
    # an extra interior null; the mask disk fully contains it.
    psi = psi - 2.0 * np.exp(-(((RR - pole_r) / 0.03) ** 2 + (ZZ / 0.03) ** 2))
    # unmasked, the dimple adds an interior O-point (the artifact the mask kills)
    assert topo.find_critical_points(psi, r_1d, z_1d).o_points.shape[0] >= 2
    field = mask_invalid_interior(psi, r_1d, z_1d, pole_r, 0.0, 0.10, axis_rz=axis)
    lim_r, lim_z = _circle_polygon(r0, 0.0, rb)  # limiter contains the plateau disk
    out = topo.lcfs_contour(field, r_1d, z_1d, axis, limiter_r=lim_r, limiter_z=lim_z)
    assert out.found
    # the boundary rides the wall (encloses the masked pole region), NOT a tiny
    # interior ring around the spurious null.
    np.testing.assert_allclose(out.radii, rb, atol=0.05)
    assert np.nanmin(out.radii) > 0.3  # no collapse onto the interior null
