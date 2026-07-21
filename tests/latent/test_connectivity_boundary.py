"""Tests for the accelerator-native connectivity boundary read (JAX).

The device flood-fill boundary read reproduces the shipped host
``topology.lcfs_contour`` (the monotone connectivity push) with fixed-shape,
``jit`` / ``vmap`` / ``grad``-safe primitives.  It is pinned four ways, none of
which touch EFIT or real data:

* **accelerator compliance** — compiles + runs under ``jit`` / ``vmap`` (a batch
  of ψ fields on one grid) / ``grad``, output fixed-shape regardless of core size;
* **contour-free** — imports no contourpy / marching-squares / ``scipy.ndimage``
  and calls no ``argwhere`` (AST-checked);
* **reproduction** — on synthetic LIMITED and DIVERTED fields the device ψ_bnd and
  LCFS radii match the host ``lcfs_contour`` to grid tolerance;
* **continuity** — through a synthetic limited→diverted sweep the connectivity
  ψ_bnd is continuous where the classify-first read (innermost in-vessel X flux,
  else limiter) steps.
"""

from __future__ import annotations

import ast
import inspect

import numpy as np

from imas_ambix.latent import connectivity_boundary as cb
from imas_ambix.latent.topology import _inside_polygon, lcfs_contour
from imas_ambix.worldmodel.equilibrium_labels import LCFS_ANGLES

# --- synthetic ψ (no solve, no data) ---------------------------------------


def _limited_field(nr=81, nz=101):
    """A single O-point Gaussian — a wall-limited plasma (no separatrix X).

    The limiter box is kept close to the plasma so the wall tangency sits at a
    real flux gradient (~2σ), not out on the flat Gaussian tail — otherwise the
    boundary radius is ill-conditioned in ψ (a tiny ψ shift moves it centimetres)
    and would swamp the reproduction test with conditioning, not read error.
    """
    rg = np.linspace(0.2, 1.8, nr)
    zg = np.linspace(-1.0, 1.0, nz)
    rr, zz = np.meshgrid(rg, zg)
    psi = np.exp(-(((rr - 1.0) ** 2 + zz**2) / 0.3**2))
    lr = np.array([0.55, 1.45, 1.45, 0.55, 0.55])
    lz = np.array([-0.55, -0.55, 0.55, 0.55, -0.55])
    inside = _inside_polygon(rr.ravel(), zz.ravel(), lr, lz).reshape(nz, nr)
    return psi, rg, zg, (1.0, 0.0), lr, lz, inside


def _diverted_field(nr=101, nz=141):
    """Two same-sign Gaussians with a saddle between them — a diverted separatrix."""
    rg = np.linspace(0.2, 1.8, nr)
    zg = np.linspace(-1.2, 1.2, nz)
    rr, zz = np.meshgrid(rg, zg)
    s = 0.28

    def blob(r0, z0):
        return np.exp(-(((rr - r0) ** 2 + (zz - z0) ** 2) / s**2))

    psi = blob(1.0, 0.25) + 0.9 * blob(1.0, -0.75)
    lr = np.array([0.25, 1.75, 1.75, 0.25, 0.25])
    lz = np.array([-1.1, -1.1, 1.1, 1.1, -1.1])
    inside = _inside_polygon(rr.ravel(), zz.ravel(), lr, lz).reshape(nz, nr)
    return psi, rg, zg, (1.0, 0.25), lr, lz, inside


class _Grid:
    """Minimal grid duck-type for the host adapters."""

    def __init__(self, rg, zg, inside, lr=None, lz=None):
        self.rg, self.zg, self.inside_limiter = rg, zg, inside
        self.limiter_r, self.limiter_z = lr, lz


# --- accelerator compliance -------------------------------------------------


def test_jit_vmap_grad_safe_and_fixed_shape():
    import jax
    import jax.numpy as jnp

    ang = jnp.asarray(np.asarray(LCFS_ANGLES))
    small = _limited_field(nr=61, nz=61)
    big = _diverted_field(nr=61, nz=61)

    def read(psi, rg, zg, inside, ar, az):
        return cb.boundary_read_jax(
            jnp.asarray(psi),
            jnp.asarray(rg),
            jnp.asarray(zg),
            jnp.asarray(inside),
            jnp.asarray(float(ar)),
            jnp.asarray(float(az)),
            64,
            14,
            256,
            ang,
            jnp.asarray(0.999),
        )

    o_s = read(small[0], small[1], small[2], small[6], *small[3])
    o_b = read(big[0], big[1], big[2], big[6], *big[3])
    # fp64 + fixed radii shape regardless of very different core sizes
    assert o_s["radii"].dtype == jnp.float64
    assert int(o_s["n_core_cells"]) != int(o_b["n_core_cells"])
    assert np.asarray(o_s["radii"]).shape == (len(LCFS_ANGLES),)
    assert bool(o_s["found"]) and bool(o_b["found"])

    # vmap over a batch of ψ fields sharing the grid
    psi, rg, zg, axis, _lr, _lz, inside = _limited_field(nr=61, nz=61)
    batch = jnp.stack(
        [jnp.asarray(psi), jnp.asarray(psi * 1.03), jnp.asarray(psi * 0.97)]
    )
    vfun = jax.vmap(
        lambda p: cb.boundary_read_jax(
            p,
            jnp.asarray(rg),
            jnp.asarray(zg),
            jnp.asarray(inside),
            jnp.asarray(1.0),
            jnp.asarray(0.0),
            64,
            14,
            256,
            ang,
            jnp.asarray(0.999),
        )["psi_bnd"]
    )
    vb = vfun(batch)
    assert vb.shape == (3,)
    assert np.all(np.isfinite(np.asarray(vb)))

    # grad of ψ_bnd w.r.t. the axis position flows and is finite
    def loss(az):
        return cb.boundary_read_jax(
            jnp.asarray(psi),
            jnp.asarray(rg),
            jnp.asarray(zg),
            jnp.asarray(inside),
            jnp.asarray(1.0),
            az,
            64,
            14,
            256,
            ang,
            jnp.asarray(0.999),
        )["psi_bnd"]

    g = jax.grad(loss)(jnp.asarray(0.0))
    assert np.isfinite(float(g))


def test_module_is_contour_free():
    """Imports no contour / ndimage machinery and calls no argwhere (AST-checked)."""
    tree = ast.parse(inspect.getsource(cb))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            imported += [base] + [f"{base}.{a.name}" for a in node.names]
    banned = ("contourpy", "matplotlib", "skimage", "scipy.ndimage", "ndimage")
    for imp in imported:
        assert not any(b in imp for b in banned), f"boundary read imports {imp!r}"
    calls = {
        n.func.attr
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    }
    assert "argwhere" not in calls and "label" not in calls


# --- reproduction of the host lcfs_contour ----------------------------------


def test_reproduces_host_lcfs_limited():
    """The wall-tangency binding, read SUB-GRID on the wall — no inward bias."""
    psi, rg, zg, axis, lr, lz, inside = _limited_field()
    cpu = lcfs_contour(psi, rg, zg, axis, limiter_r=lr, limiter_z=lz)
    gpu = cb.boundary_read(psi, _Grid(rg, zg, inside, lr, lz), axis, lcfs_norm=0.999)
    assert cpu.found and gpu.found
    span = abs(cpu.psi_bnd - gpu.psi_axis)
    # sub-grid wall interpolation: ψ_bnd reproduces the host well inside a cell,
    # with no systematic sign (the single-/two-sided cell tests were biased inward)
    assert abs(gpu.psi_bnd - cpu.psi_bnd) / span < 0.01
    ok = np.isfinite(cpu.radii) & np.isfinite(gpu.radii)
    assert np.median(np.abs(gpu.radii[ok] - cpu.radii[ok])) < 0.02  # < 2 cm


def test_reproduces_host_lcfs_diverted():
    psi, rg, zg, axis, lr, lz, inside = _diverted_field()
    cpu = lcfs_contour(psi, rg, zg, axis, limiter_r=lr, limiter_z=lz)
    gpu = cb.boundary_read(psi, _Grid(rg, zg, inside, lr, lz), axis, lcfs_norm=0.999)
    assert cpu.found and gpu.found
    span = abs(cpu.psi_bnd - gpu.psi_axis)
    assert abs(gpu.psi_bnd - cpu.psi_bnd) / span < 0.01
    ok = np.isfinite(cpu.radii) & np.isfinite(gpu.radii)
    # a hair inside the separatrix the rays stay on the closed lobe (no leg run-out)
    assert np.median(np.abs(gpu.radii[ok] - cpu.radii[ok])) < 0.02


def test_clip_legs_radii_stay_on_lobe():
    """A ray cast AT ψ_bnd would run down an open X-point leg; the read clamps the
    ray a hair inside so the radii match the host closed-lobe read even for the
    clip_legs (ψ_N=1) convention."""
    psi, rg, zg, axis, lr, lz, inside = _diverted_field()
    cpu = lcfs_contour(psi, rg, zg, axis, limiter_r=lr, limiter_z=lz, clip_legs=True)
    gpu = cb.boundary_read(psi, _Grid(rg, zg, inside, lr, lz), axis, lcfs_norm=1.0)
    ok = np.isfinite(cpu.radii) & np.isfinite(gpu.radii)
    # no single ray blows out down a leg (would be tens of cm)
    assert np.max(np.abs(gpu.radii[ok] - cpu.radii[ok])) < 0.05


# --- continuity through the marginal transition -----------------------------


def test_continuous_through_marginal_transition():
    from scripts.connectivity_boundary_gate_eval import _transition_continuity_gate

    r = _transition_continuity_gate()
    assert r["n_xpoint_transitions"] >= 1  # the sweep really crosses limited→diverted
    # connectivity is smooth where classify-first steps
    assert r["conn_max_rel_step"] < 0.05
    assert r["classify_max_rel_step"] > 3.0 * r["conn_max_rel_step"]
    assert r["verdict"] == "PASS"


# --- emergent X-set: distinct nulls, not stencil duplicates ------------------


def _double_null_field(nr=45, nz=61):
    """A near-balanced double-null whose saddles each fire TWO stencil vertices.

    The saddles sit half a cell off the R vertex line and the field is flat in
    R around them (anisotropic blobs), so the 4-sign-change classifier fires on
    two adjacent vertices per PHYSICAL saddle — the coarse-grid degeneracy real
    65×65 EFIT maps show.  The emergent-set trim must not let the two hits on
    one saddle crowd the opposite null out of the two slots.
    """
    rg = np.linspace(0.2, 1.8, nr)
    zg = np.linspace(-1.4, 1.4, nz)
    r0 = 1.0 + 0.5 * float(rg[1] - rg[0])
    rr, zz = np.meshgrid(rg, zg)

    def blob(z0, a):
        return a * np.exp(-((rr - r0) ** 2 / 0.45**2 + (zz - z0) ** 2 / 0.28**2))

    psi = blob(0.0, 1.0) + blob(-0.9, 0.9) + blob(0.9, 0.88)
    lr = np.array([0.25, 1.75, 1.75, 0.25, 0.25])
    lz = np.array([-1.3, -1.3, 1.3, 1.3, -1.3])
    inside = _inside_polygon(rr.ravel(), zz.ravel(), lr, lz).reshape(nz, nr)
    return psi, rg, zg, (r0, 0.0), lr, lz, inside


def test_emergent_xset_holds_both_nulls_of_a_double_null():
    psi, rg, zg, axis, lr, lz, inside = _double_null_field()
    gpu = cb.boundary_read(psi, _Grid(rg, zg, inside, lr, lz), axis, lcfs_norm=1.0)
    assert gpu.found and gpu.is_diverted
    xset = np.asarray(gpu.xset, dtype=np.float64)
    finite = xset[np.isfinite(xset).all(axis=1)]
    # both slots filled, one null per side — never two copies of one saddle
    assert finite.shape[0] == 2
    assert np.sign(finite[0, 1]) != np.sign(finite[1, 1])
    assert np.all(np.abs(np.abs(finite[:, 1]) - 0.46) < 0.15)
