"""Tests for the finite-area (cylinder) Green's functions.

The kernel computes ψ / B_R / B_Z per ampere of a complete toroidal conductor
with RECTANGULAR cross-section — smooth everywhere, including inside the
conductor, unlike a point filament (log-singular at the source).  Pinned four
ways, no EFIT and no nova import needed at test time:

* GOLDEN values generated from the reference implementation (nova.biot
  cylinder formulation, run standalone in nova's own environment) — the
  extraction must reproduce them;
* the FAR-FIELD limit must converge to the point-filament loop formulas
  already validated in :mod:`imas_ambix.gs.operator`;
* SMOOTHNESS: finite values on a transect crossing the conductor, with no
  off-source spikes;
* AMPÈRE's law: the poloidal-plane circulation of B around the cross-section
  equals μ0·I.
"""

from __future__ import annotations

import numpy as np

from imas_ambix.gs import operator as op
from imas_ambix.gs.cylinder import cylinder_greens

# Golden reference values (nova.biot cylinder formulation, scipy mu_0),
# per ampere of total conductor current.
# source A: a=0.9, z0=0.1, da=0.12, dz=0.18 (plasma-cell scale)
_GOLDEN_A = {
    (0.90, 0.10): (3.317070799989e-06, 0.000000000000e00, 4.793561783072e-07),
    (0.93, 0.13): (3.250535756010e-06, 6.089726044006e-07, -6.074095834724e-07),
    (1.02, 0.10): (2.554049416276e-06, 0.000000000000e00, -1.088863299876e-06),
    (1.20, 0.40): (1.383297566724e-06, 2.493152564078e-07, -1.142910221325e-07),
    (1.50, 0.00): (1.237782754508e-06, -3.360553102306e-08, -1.176816553936e-07),
    (0.30, -1.20): (3.535050457921e-08, -2.903978666350e-08, 1.209742696911e-07),
    (1.90, 1.50): (4.188278163282e-07, 2.859924384742e-08, 4.690729320803e-09),
}
# source B: a=0.19, z0=-0.5, da=0.04, dz=0.30 (solenoid/coil scale)
_GOLDEN_B = {
    (0.90, 0.10): (4.573497990728e-08, 1.249856387393e-08, -6.422468221970e-10),
    (0.93, 0.13): (4.360276278947e-08, 1.122252988102e-08, -4.098605684425e-10),
    (1.02, 0.10): (4.488564583814e-08, 9.039618161549e-09, -1.546416329833e-09),
    (1.20, 0.40): (3.049051332441e-08, 4.862056800333e-09, 2.817575558670e-10),
    (1.50, 0.00): (4.073188034412e-08, 2.601217323635e-09, -2.017138962361e-09),
    (0.30, -1.20): (1.432086397463e-08, -2.723599772132e-08, 3.927507588421e-08),
    (1.90, 1.50): (1.229231157267e-08, 8.119794404350e-10, 3.133642506030e-10),
}


def _check_golden(golden, src, rtol):
    pts = np.array(list(golden.keys()))
    want = np.array(list(golden.values()))
    psi, br, bz = cylinder_greens(pts[:, 0], pts[:, 1], *src)
    np.testing.assert_allclose(psi, want[:, 0], rtol=rtol)
    np.testing.assert_allclose(br, want[:, 1], rtol=rtol, atol=1e-14)
    np.testing.assert_allclose(bz, want[:, 2], rtol=rtol)


def test_matches_reference_implementation_source_a():
    _check_golden(_GOLDEN_A, (0.9, 0.1, 0.12, 0.18), rtol=1e-6)


def test_matches_reference_implementation_source_b():
    _check_golden(_GOLDEN_B, (0.19, -0.5, 0.04, 0.30), rtol=1e-6)


def test_small_section_limit_is_the_point_filament_loop():
    """Shrinking the cross-section must converge to the validated loop formulas
    (the point filament is the zero-area limit of the cylinder)."""
    a, z0 = 0.9, 0.1
    tr = np.array([1.2, 0.4, 1.8, 0.7])
    tz = np.array([0.6, -1.2, -1.0, 0.4])
    psi, br, bz = cylinder_greens(tr, tz, a, z0, 1e-3, 1e-3)
    psi_pt = op.greens_psi(tr, tz, a, z0)
    bz_pt, br_pt = op.greens_bz_br(tr, tz, a, z0)
    np.testing.assert_allclose(psi, psi_pt, rtol=1e-6)
    np.testing.assert_allclose(br, br_pt, rtol=1e-5, atol=1e-14)
    np.testing.assert_allclose(bz, bz_pt, rtol=1e-5, atol=1e-14)


def test_equivalent_to_dense_filament_average_over_section():
    """The closed form must equal the brute-force average of point filaments
    distributed over the cross-section (the definition of a finite-area
    source).  Note the far field does NOT converge to the CENTROID loop —
    the second moment shifts the effective radius to √⟨a²⟩ (≈ +da²/12a₀²
    relative in ψ), which this reference reproduces and the closed form must
    match, near field and far."""
    a, z0, da, dz = 0.9, 0.1, 0.12, 0.18
    tr = np.array([1.15, 1.4, 2.0, 3.5, 6.0, 0.4, 0.65])
    tz = np.array([0.35, 0.1, -0.6, 0.1, 0.1, -1.2, 0.1])
    psi, br, bz = cylinder_greens(tr, tz, a, z0, da, dz)
    n = 60
    offs = (np.arange(n) + 0.5) / n - 0.5
    psi_ref = np.zeros_like(tr)
    br_ref = np.zeros_like(tr)
    bz_ref = np.zeros_like(tr)
    for or_ in offs:
        for oz in offs:
            psi_ref += op.greens_psi(tr, tz, a + or_ * da, z0 + oz * dz)
            bz_f, br_f = op.greens_bz_br(tr, tz, a + or_ * da, z0 + oz * dz)
            br_ref += br_f
            bz_ref += bz_f
    psi_ref /= n * n
    br_ref /= n * n
    bz_ref /= n * n
    np.testing.assert_allclose(psi, psi_ref, rtol=1e-5)
    np.testing.assert_allclose(br, br_ref, rtol=1e-4, atol=1e-13)
    np.testing.assert_allclose(bz, bz_ref, rtol=1e-4, atol=1e-13)


def test_smooth_through_the_conductor():
    """A transect crossing the section: finite everywhere, peak AT the source
    (a point filament diverges there — the defect this kernel removes)."""
    a, z0, da, dz = 0.9, 0.1, 0.12, 0.18
    tr = np.linspace(0.6, 1.2, 241)
    tz = np.full_like(tr, z0)
    psi, br, bz = cylinder_greens(tr, tz, a, z0, da, dz)
    assert np.isfinite(psi).all() and np.isfinite(br).all() and np.isfinite(bz).all()
    peak = tr[int(np.argmax(np.abs(psi)))]
    assert abs(peak - a) < 0.05  # |ψ| peaks at the conductor, no off-source spikes


def test_ampere_circulation_equals_mu0():
    """∮ B·dl around the cross-section = μ0·I (per unit current: μ0)."""
    a, z0, da, dz = 0.9, 0.1, 0.12, 0.18
    theta = np.linspace(0, 2 * np.pi, 4001)[:-1]
    rad = 0.35
    tr = a + rad * np.cos(theta)
    tz = z0 + rad * np.sin(theta)
    _psi, br, bz = cylinder_greens(tr, tz, a, z0, da, dz)
    # tangent along the contour: (-sinθ, cosθ)·rad·dθ
    dtheta = theta[1] - theta[0]
    circ = np.sum((-br * np.sin(theta) + bz * np.cos(theta)) * rad * dtheta)
    mu0 = 4e-7 * np.pi
    np.testing.assert_allclose(abs(circ), mu0, rtol=1e-3)


def test_hybrid_greens_matches_cylinder_near_and_filament_far():
    """The hybrid kernel must be the cylinder form inside the switch band and
    the (cheap) point filament beyond it."""
    from imas_ambix.gs.cylinder import hybrid_greens

    a, z0, da, dz = 0.9, 0.1, 0.12, 0.18
    near = np.array([0.92, 0.85, 1.05])
    near_z = np.array([0.12, 0.05, 0.18])
    far = np.array([1.6, 0.3, 1.9])
    far_z = np.array([0.9, -1.2, 1.5])
    tr = np.concatenate([near, far])
    tz = np.concatenate([near_z, far_z])
    psi, br, bz = hybrid_greens(tr, tz, a, z0, da, dz, switch=3.0)
    psi_cyl, br_cyl, bz_cyl = cylinder_greens(tr, tz, a, z0, da, dz)
    psi_pt = op.greens_psi(tr, tz, a, z0)
    bz_pt, br_pt = op.greens_bz_br(tr, tz, a, z0)
    np.testing.assert_allclose(psi[:3], psi_cyl[:3], rtol=1e-12)
    np.testing.assert_allclose(br[:3], br_cyl[:3], rtol=1e-12, atol=1e-16)
    np.testing.assert_allclose(psi[3:], psi_pt[3:], rtol=1e-12)
    np.testing.assert_allclose(bz[3:], bz_pt[3:], rtol=1e-12, atol=1e-16)
    # and the hybrid is everywhere finite at/inside the source
    p0, b0, z0v = hybrid_greens(
        np.array([a]), np.array([z0]), a, z0, da, dz, switch=3.0
    )
    assert np.isfinite(p0).all() and np.isfinite(b0).all() and np.isfinite(z0v).all()
