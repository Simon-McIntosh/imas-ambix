"""Tests for the polygonal-section finite-area Green's functions.

The kernel computes ψ / B_R / B_Z per ampere of a complete toroidal conductor
with an arbitrary POLYGON cross-section (Urankar Part V sloped-edge contour
reduction), generalising the rectangular kernel in
:mod:`imas_ambix.gs.cylinder`.  Pinned five ways, no nova import needed:

* RECTANGULAR REDUCTION — for an axis-aligned rectangle the polygon kernel
  must reproduce ``cylinder_greens`` (the validated Part III kernel) to 1e-10;
* FILAMENT ORACLE — for slanted sections (parallelogram, trapezoid, triangle)
  the result must converge to a centroid-rule filament tiling built from
  affine maps of the exact section (no bounding-box staircase error);
* ψ↔B CONSISTENCY — B_Z = (1/2πR) ∂ψ/∂R and B_R = −(1/2πR) ∂ψ/∂Z, cross-checked
  by independent central finite differences of ψ (the kernel itself uses the
  exact complex-step curl);
* AMPÈRE — the poloidal-plane circulation of B around the section equals μ0·I;
* INVARIANCES — vertex ordering (CW vs CCW) and starting vertex must not
  matter; a horizontal-edge (Δz = 0) contribution vanishes by construction.
"""

from __future__ import annotations

import numpy as np
import pytest

from imas_ambix.gs.cylinder import MU0, cylinder_greens
from imas_ambix.gs.operator import greens_bz_br, greens_psi
from imas_ambix.gs.polygon import polygon_greens

# ---------------------------------------------------------------- geometry --

RECT = np.array([(0.84, 0.01), (0.96, 0.01), (0.96, 0.19), (0.84, 0.19)])
# 45-degree slanted parallelogram (P2-arm-like)
PARA = np.array([(0.85, 0.00), (0.97, 0.00), (1.05, 0.20), (0.93, 0.20)])
# steep crown-like parallelogram (~65 degrees)
CROWN = np.array([(0.30, 1.20), (0.42, 1.20), (0.50, 1.37), (0.38, 1.37)])
TRAP = np.array([(0.80, -0.10), (1.00, -0.10), (0.95, 0.08), (0.85, 0.08)])
TRI = np.array([(0.90, 0.00), (1.10, 0.05), (0.95, 0.18)])

# far/mid/near field points (outside every section above)
TARGETS = np.array(
    [(1.30, 0.30), (0.70, -0.20), (1.10, 0.35), (1.60, 0.00), (0.50, 0.60)]
)


def _tri_area(v0, v1, v2):
    return 0.5 * abs(
        (v1[0] - v0[0]) * (v2[1] - v0[1]) - (v2[0] - v0[0]) * (v1[1] - v0[1])
    )


def _oracle_fields(r, z, verts, n=120):
    """(ψ, B_R, B_Z) per ampere from an exact affine-tiled filament sum.

    Fan-triangulate the (convex) polygon, split each triangle into n² affine
    sub-triangles (barycentric refinement: "lower" and "upper" families), and
    place a point filament at each sub-triangle centroid with weight ∝ its
    exact area.  The cells tile the section exactly — no bounding-box
    staircase error — so the sum converges O(1/n²).
    """
    v = np.asarray(verts, dtype=float)
    ar, az, wt = [], [], []
    for i in range(1, len(v) - 1):
        v0, v1, v2 = v[0], v[i], v[i + 1]
        area = _tri_area(v0, v1, v2)
        idx_j, idx_k = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        mask_lo = (idx_j + idx_k) < n
        u_lo = (idx_j[mask_lo] + 1 / 3) / n
        w_lo = (idx_k[mask_lo] + 1 / 3) / n
        mask_up = (idx_j + idx_k) < (n - 1)
        u_up = (idx_j[mask_up] + 2 / 3) / n
        w_up = (idx_k[mask_up] + 2 / 3) / n
        uu = np.concatenate([u_lo, u_up])
        ww = np.concatenate([w_lo, w_up])
        ar.append(v0[0] + uu * (v1[0] - v0[0]) + ww * (v2[0] - v0[0]))
        az.append(v0[1] + uu * (v1[1] - v0[1]) + ww * (v2[1] - v0[1]))
        wt.append(np.full(uu.shape, area / n**2))
    ar = np.concatenate(ar)
    az = np.concatenate(az)
    wt = np.concatenate(wt)
    wt = wt / wt.sum()  # unit total current
    rr = np.full_like(ar, float(r))
    zz = np.full_like(az, float(z))
    psi = float(np.sum(wt * greens_psi(rr, zz, ar, az)))
    bz, br = greens_bz_br(rr, zz, ar, az)
    return psi, float(np.sum(wt * br)), float(np.sum(wt * bz))


# ------------------------------------------------------ rectangular reduction


def test_rectangle_reduces_to_cylinder_kernel():
    """b1 = 0 polygon == validated Part III rectangular kernel, 1e-10."""
    a, z0 = 0.90, 0.10
    da, dz = 0.12, 0.18
    pts_r = np.array([1.02, 1.20, 1.50, 0.93, 0.30, 1.90])
    pts_z = np.array([0.10, 0.40, 0.00, 0.13, -1.20, 1.50])
    psi_p, br_p, bz_p = polygon_greens(pts_r, pts_z, RECT)
    psi_c, br_c, bz_c = cylinder_greens(pts_r, pts_z, a, z0, da, dz)
    np.testing.assert_allclose(psi_p, psi_c, rtol=1e-10)
    np.testing.assert_allclose(br_p, br_c, rtol=1e-10, atol=1e-20)
    np.testing.assert_allclose(bz_p, bz_c, rtol=1e-10, atol=1e-20)


def test_rectangle_interior_point_matches_cylinder():
    """The kernel is smooth and exact inside the conductor too."""
    psi_p, br_p, bz_p = polygon_greens(np.array([0.90]), np.array([0.10]), RECT)
    psi_c, br_c, bz_c = cylinder_greens(
        np.array([0.90]), np.array([0.10]), 0.90, 0.10, 0.12, 0.18
    )
    np.testing.assert_allclose(psi_p, psi_c, rtol=1e-9)
    np.testing.assert_allclose(bz_p, bz_c, rtol=1e-9)
    np.testing.assert_allclose(br_p, br_c, atol=1e-16)


# ------------------------------------------------------------ filament oracle


@pytest.mark.parametrize(
    "verts", [PARA, CROWN, TRAP, TRI], ids=["para45", "crown65", "trap", "tri"]
)
def test_matches_filament_oracle(verts):
    """Slanted/general sections agree with the exact-tiling filament sum.

    The oracle converges O(1/n²); at n=120 its residual is ~1e-6 relative,
    so agreement is gated at 5e-5 relative on ψ and on |B| (component-wise
    with an absolute floor scaled to the field magnitude, so near-zero
    crossings of one component don't inflate the relative error).
    """
    for r, z in TARGETS:
        psi, br, bz = polygon_greens(np.array([r]), np.array([z]), verts)
        psi_o, br_o, bz_o = _oracle_fields(r, z, verts, n=120)
        b_scale = max(abs(br_o), abs(bz_o))
        assert abs(psi[0] - psi_o) <= 5e-5 * abs(psi_o)
        assert abs(br[0] - br_o) <= 5e-5 * b_scale
        assert abs(bz[0] - bz_o) <= 5e-5 * b_scale


# ------------------------------------------------------- internal consistency


def test_psi_b_finite_difference_consistency():
    """B_Z = (1/2πR) ∂ψ/∂R, B_R = −(1/2πR) ∂ψ/∂Z (central differences)."""
    h = 1e-5
    for r, z in [(1.25, 0.30), (0.70, -0.15), (1.05, 0.40)]:
        rr = np.array([r + h, r - h, r, r])
        zz = np.array([z, z, z + h, z - h])
        psi, _, _ = polygon_greens(rr, zz, PARA)
        _, br, bz = polygon_greens(np.array([r]), np.array([z]), PARA)
        dpsi_dr = (psi[0] - psi[1]) / (2 * h)
        dpsi_dz = (psi[2] - psi[3]) / (2 * h)
        np.testing.assert_allclose(bz[0], dpsi_dr / (2 * np.pi * r), rtol=2e-6)
        np.testing.assert_allclose(br[0], -dpsi_dz / (2 * np.pi * r), rtol=2e-6)


def test_ampere_circulation():
    """|∮ B·dl| around a loop enclosing the section = μ0 I (I = 1 A).

    In the (r, φ, z) convention of :func:`greens_bz_br` a CCW loop in the
    poloidal plane is left-handed w.r.t. +φ̂, so the circulation of the field
    of a +1 A toroidal current is −μ0 (pinned point-filament check in the
    module tests); the physics is |∮ B·dl| = μ0 I.
    """
    v = PARA
    r0, z0 = v[:, 0].mean(), v[:, 1].mean()
    a = 0.35  # loop radius comfortably enclosing the section
    theta = np.linspace(0.0, 2 * np.pi, 720, endpoint=False)
    rr = r0 + a * np.cos(theta)
    zz = z0 + a * np.sin(theta)
    _, br, bz = polygon_greens(rr, zz, v)
    # tangent d l = a dθ (−sinθ, cosθ)
    circ = (
        np.sum(-br * np.sin(theta) + bz * np.cos(theta)) * a * (2 * np.pi / theta.size)
    )
    np.testing.assert_allclose(circ, -MU0, rtol=1e-6)


def test_ampere_circulation_excluding_section_is_zero():
    """A loop NOT enclosing the section links no current."""
    theta = np.linspace(0.0, 2 * np.pi, 720, endpoint=False)
    rr = 1.60 + 0.08 * np.cos(theta)
    zz = 0.60 + 0.08 * np.sin(theta)
    _, br, bz = polygon_greens(rr, zz, PARA)
    circ = (
        np.sum(-br * np.sin(theta) + bz * np.cos(theta))
        * 0.08
        * (2 * np.pi / theta.size)
    )
    assert abs(circ) < 1e-9 * MU0


# ----------------------------------------------------------------- invariance


def test_vertex_order_invariance():
    """CW vs CCW ordering and starting-vertex rotation give identical fields."""
    r = np.array([1.30, 0.70])
    z = np.array([0.30, -0.20])
    base = polygon_greens(r, z, PARA)
    flipped = polygon_greens(r, z, PARA[::-1])
    rolled = polygon_greens(r, z, np.roll(PARA, 2, axis=0))
    for got, want in zip(flipped + rolled, base + base, strict=True):
        np.testing.assert_allclose(got, want, rtol=1e-13)


def test_smooth_transect_through_conductor():
    """Finite values with no spikes on a transect crossing the section."""
    rr = np.linspace(0.80, 1.10, 61)
    zz = np.full_like(rr, 0.10)
    psi, br, bz = polygon_greens(rr, zz, PARA)
    assert np.all(np.isfinite(psi)) and np.all(np.isfinite(br))
    assert np.all(np.isfinite(bz))
    # ψ of a unit ring current is O(1e-6)·μ0-ish here; no orders-of-magnitude spikes
    assert psi.max() / max(psi.min(), 1e-12) < 1e3


def test_broadcasts_target_shape():
    rr = np.linspace(1.1, 1.4, 6).reshape(2, 3)
    zz = np.full_like(rr, 0.25)
    psi, br, bz = polygon_greens(rr, zz, PARA)
    assert psi.shape == rr.shape and br.shape == rr.shape and bz.shape == rr.shape
