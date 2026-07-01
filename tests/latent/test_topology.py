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
