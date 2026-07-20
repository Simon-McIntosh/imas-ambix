"""Sub-grid stencil null locator — synthetic-field unit tests.

Fast, self-contained (no IMAS data): verifies the whole-grid 8-ring classifier
(0 → O, 4 → X), the biquadratic sub-grid refinement against a symmetry-known
X-point, and that the axis position carries a finite ``jax.grad``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from imas_ambix.latent.stencil_nulls import (
    magnetic_axis_subgrid,
    ring_sign_changes,
    xpoint_candidates,
)


def _two_peak_field(nr=81, nz=101, rc=1.007, z1=-0.30, z2=0.30, w=0.15):
    """Two positive Gaussians stacked in Z: two O-points, one X-point between.

    By the up-down symmetry the saddle sits exactly at ``(rc, (z1+z2)/2)``, giving
    a ground-truth sub-grid target the biquadratic refinement must recover.
    """
    rg = np.linspace(0.5, 1.5, nr)
    zg = np.linspace(-0.8, 0.8, nz)
    rr, zz = np.meshgrid(rg, zg)  # (nz, nr)
    psi = np.exp(-(((rr - rc) ** 2 + (zz - z1) ** 2) / w**2)) + np.exp(
        -(((rr - rc) ** 2 + (zz - z2) ** 2) / w**2)
    )
    return jnp.asarray(psi), jnp.asarray(rg), jnp.asarray(zg), rc, 0.5 * (z1 + z2)


def test_classifier_finds_o_and_x():
    psi, rg, zg, _rc, _xz = _two_peak_field()
    counts = np.asarray(ring_sign_changes(psi))
    assert (counts == 0).any(), "no O-point (0 sign changes) classified"
    assert (counts == 4).any(), "no X-point (4 sign changes) classified"
    # the border carries no full ring
    assert (counts[0, :] == -1).all() and (counts[:, 0] == -1).all()


def test_axis_is_a_gaussian_peak():
    psi, rg, zg, rc, _xz = _two_peak_field()
    inside = jnp.ones(psi.shape, dtype=bool)
    ax = magnetic_axis_subgrid(psi, rg, zg, inside)
    assert bool(ax["found"])
    dr = float(rg[1] - rg[0])
    # the axis is one of the two peaks: near rc in R, near ±0.30 in Z
    assert abs(float(ax["r"]) - rc) < 3 * dr
    assert min(abs(float(ax["z"]) - 0.30), abs(float(ax["z"]) + 0.30)) < 3 * float(
        zg[1] - zg[0]
    )
    assert float(ax["ntype"]) > 0  # a maximum (both curvatures negative → +1)


def test_xpoint_subgrid_matches_symmetry():
    psi, rg, zg, rc, xz = _two_peak_field()
    inside = jnp.ones(psi.shape, dtype=bool)
    xc = xpoint_candidates(psi, rg, zg, inside, k_slots=6)
    valid = np.asarray(xc["valid"])
    assert valid.any(), "no valid X-point found"
    rr = np.asarray(xc["r"])[valid]
    zz = np.asarray(xc["z"])[valid]
    # the symmetry saddle nearest the known midpoint
    d = np.hypot(rr - rc, zz - xz)
    k = int(np.argmin(d))
    dr = float(rg[1] - rg[0])
    dz = float(zg[1] - zg[0])
    assert abs(rr[k] - rc) < dr, f"X R off by {abs(rr[k] - rc):.4f} (> {dr:.4f})"
    assert abs(zz[k] - xz) < dz, f"X Z off by {abs(zz[k] - xz):.4f} (> {dz:.4f})"
    types = np.asarray(xc["ntype"])[valid]
    assert abs(types[k]) < 0.5, "midpoint null is not typed as a saddle"


def test_axis_gradient_is_finite():
    psi, rg, zg, _rc, _xz = _two_peak_field()
    inside = jnp.ones(psi.shape, dtype=bool)

    def axis_r(p):
        return magnetic_axis_subgrid(p, rg, zg, inside)["r"]

    g = jax.grad(axis_r)(psi)
    g = np.asarray(g)
    assert np.all(np.isfinite(g)), "axis-R gradient has non-finite entries"
    assert np.any(g != 0.0), "axis-R gradient is identically zero (no signal)"


def test_extra_mask_restricts_candidates():
    psi, rg, zg, rc, xz = _two_peak_field()
    inside = jnp.ones(psi.shape, dtype=bool)
    # a flux-proximity band far from the saddle flux removes the X entirely
    xc_all = xpoint_candidates(psi, rg, zg, inside, k_slots=6)
    assert np.asarray(xc_all["valid"]).any()
    empty_mask = jnp.zeros(psi.shape, dtype=bool)
    xc_none = xpoint_candidates(psi, rg, zg, inside, k_slots=6, extra_mask=empty_mask)
    assert not np.asarray(xc_none["valid"]).any()
