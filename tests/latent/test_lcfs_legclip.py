"""ψ_N=1 + X-point leg-clip boundary read (topology.lcfs_contour clip_legs).

The 0.999 reader traces the confined lobe a hair inside the separatrix, rounding
the X-point corner.  clip_legs reports the lobe AT the separatrix (ψ_N=1) and
snaps it to the on-separatrix X-point — reaching the true corner while the legs
(open level-set branches) stay excluded by the closed-ring selection.
"""

from __future__ import annotations

import numpy as np

from imas_ambix.latent.topology import find_critical_points, lcfs_contour


def _diverted_field(nr=101, nz=141):
    """Two same-sign Gaussians (upper 'plasma' O-point + lower O-point) with a
    saddle between them — a clean diverted separatrix. Axis = upper O-point."""
    rg = np.linspace(0.2, 1.8, nr)
    zg = np.linspace(-1.2, 1.2, nz)
    rr, zz = np.meshgrid(rg, zg)
    s = 0.28

    def blob(r0, z0):
        return np.exp(-(((rr - r0) ** 2 + (zz - z0) ** 2) / s**2))

    psi = blob(1.0, 0.25) + 0.9 * blob(1.0, -0.75)  # max (axis) is upper blob
    limiter_r = np.array([0.25, 1.75, 1.75, 0.25, 0.25])
    limiter_z = np.array([-1.1, -1.1, 1.1, 1.1, -1.1])
    return psi, rg, zg, (1.0, 0.25), limiter_r, limiter_z


def test_clip_legs_reaches_xpoint_and_excludes_legs():
    psi, rg, zg, axis, lr, lz = _diverted_field()
    cp = find_critical_points(psi, rg, zg)
    assert cp.x_points.shape[0] >= 1  # a genuine saddle exists
    # the saddle nearest the axis, below it
    xs = cp.x_points
    xk = xs[np.argmin(np.hypot(xs[:, 0] - axis[0], xs[:, 1] - axis[1]))]

    plain = lcfs_contour(psi, rg, zg, axis, limiter_r=lr, limiter_z=lz)
    clipped = lcfs_contour(
        psi, rg, zg, axis, limiter_r=lr, limiter_z=lz, clip_legs=True
    )
    assert plain.found and clipped.found
    # clipped ring is closed & encloses the axis (a real lobe)
    assert clipped.ring.shape[0] >= 6
    # ψ_N=1 lobe reaches AT LEAST as close to the X-point corner as the 0.999 ring
    d_clip = np.hypot(clipped.ring[:, 0] - xk[0], clipped.ring[:, 1] - xk[1]).min()
    d_plain = np.hypot(plain.ring[:, 0] - xk[0], plain.ring[:, 1] - xk[1]).min()
    assert d_clip <= d_plain + 1e-9
    # legs excluded: the confined lobe does not dip into the private-flux region
    # well BELOW the X-point (a divertor leg would)
    assert clipped.ring[:, 1].min() >= xk[1] - 0.12


def test_smooth_modes_reduces_ripple():
    """Fourier r(θ) smoothing removes grid-scale contour jaggedness while
    preserving the lobe (the fix for the visible boundary ripple)."""
    psi, rg, zg, axis, lr, lz = _diverted_field()

    def ripple(ring):
        th = np.arctan2(ring[:, 1] - axis[1], ring[:, 0] - axis[0])
        r = np.hypot(ring[:, 0] - axis[0], ring[:, 1] - axis[1])
        r = r[np.argsort(th)]
        return float(np.std(np.diff(r, 2)) / max(np.mean(r), 1e-9))

    raw = lcfs_contour(psi, rg, zg, axis, limiter_r=lr, limiter_z=lz, clip_legs=True)
    sm = lcfs_contour(
        psi, rg, zg, axis, limiter_r=lr, limiter_z=lz, clip_legs=True, smooth_modes=4
    )
    assert sm.found
    assert ripple(sm.ring) < 0.5 * ripple(raw.ring)  # markedly smoother
    # shape preserved: the two boundaries agree on the 8 scored radii
    assert np.nanmax(np.abs(sm.radii - raw.radii)) < 0.06


def test_clip_legs_limited_case_matches_plain():
    """A single O-point (no separatrix X-point) — clip_legs changes nothing
    material (no leg to clip)."""
    rg = np.linspace(0.2, 1.8, 81)
    zg = np.linspace(-1.0, 1.0, 101)
    rr, zz = np.meshgrid(rg, zg)
    psi = np.exp(-(((rr - 1.0) ** 2 + zz**2) / 0.3**2))
    lr = np.array([0.25, 1.75, 1.75, 0.25, 0.25])
    lz = np.array([-0.95, -0.95, 0.95, 0.95, -0.95])
    a = lcfs_contour(psi, rg, zg, (1.0, 0.0), limiter_r=lr, limiter_z=lz)
    b = lcfs_contour(
        psi, rg, zg, (1.0, 0.0), limiter_r=lr, limiter_z=lz, clip_legs=True
    )
    assert a.found and b.found
    # radii agree to a few cm (no X-point snapping happened)
    assert np.nanmax(np.abs(a.radii - b.radii)) < 0.05
