"""Tests for the contour-free, accelerator-native (JAX) flux-surface averaging.

The connectivity FSA is the device-native replacement for the host coarea
binning.  It is pinned three ways, none of which touch EFIT or real data:

* **accelerator compliance** — the kernel compiles and runs under ``jax.jit``,
  ``jax.vmap`` (a batch of ψ fields on one grid), and ``jax.grad``; its output
  is FIXED-SHAPE regardless of how many cells fall in the core (the property
  the host coarea path lacks, and the one a GPU batch needs);
* **contour-free** — the module imports no contour / marching-squares / level-
  set / ``scipy.ndimage`` machinery (gate G2c), and its flood-fill core matches
  ``scipy.ndimage.label`` on a synthetic confined set (the connectivity is
  correct, just computed as a fixed-shape device kernel);
* **physics agreement** — on a solved synthetic equilibrium the connectivity
  geometry closes Ampère's law and reproduces the plasma volume (the same
  invariants the coarea path is held to), and its diffusion coefficient d(ρ̂)
  tracks the coarea read — same surfaces, a smoother estimator.

The default ``fsa_mode`` is verified byte-identical to the explicit coarea path.
"""

from __future__ import annotations

import inspect

import numpy as np

from imas_ambix.latent import flux_surface_connectivity as fsc
from imas_ambix.latent.current_diffusion import flux_surface_geometry

from .test_current_diffusion import _interior_limiter_fixture, _ladder_slice

# --- synthetic ψ (no solve, no data) ---------------------------------------


def _solovev_psi(*, nr=65, nz=97, rax=0.9, a=0.55, elong=1.6):
    """A Solov'ev-like ψ with ψ_axis > ψ_bdry (the MAST sign), + its grid."""
    rg = np.linspace(0.2, 1.6, nr)
    zg = np.linspace(-1.1, 1.1, nz)
    R, Z = np.meshgrid(rg, zg)
    psi_n = ((R - rax) / a) ** 2 + (Z / (a * elong)) ** 2
    psi = -psi_n  # axis at 0 (high), decreasing outward
    inside = np.ones((nz, nr), dtype=bool)
    inside[R < 0.25] = False
    return psi, rg, zg, inside


def _bins(psi, rg, zg, inside, psi_axis=0.0, psi_bnd=-1.0, n_psin=28):
    import jax.numpy as jnp

    return fsc.flux_surface_bins_jax(
        jnp.asarray(psi),
        jnp.asarray(rg),
        jnp.asarray(zg),
        jnp.asarray(inside),
        jnp.asarray(float(psi_axis)),
        jnp.asarray(float(psi_bnd)),
        jnp.asarray(0.04),
        jnp.asarray(0.985),
        int(n_psin),
        jnp.asarray(1.25),
    )


def test_jax_fsa_is_fp64_jit_vmap_grad_safe():
    """The kernel runs fp64 and compiles under jit / vmap / grad — the concrete
    accelerator-compliance contract, not just numpy that 'could' be ported."""
    import jax
    import jax.numpy as jnp

    psi, rg, zg, inside = _solovev_psi()
    out = _bins(psi, rg, zg, inside)  # jit is on the function decorator
    assert out["inv_r2"].dtype == jnp.float64
    assert bool(out["well_posed"])
    assert np.all(np.isfinite(np.asarray(out["inv_r2"])))
    # ⟨1/R²⟩ on the innermost surface ≈ 1/R_ax² (small finite extent → slightly above)
    assert abs(float(out["inv_r2"][0]) - 1.0 / 0.9**2) < 0.1

    # vmap over a batch of ψ fields sharing the grid
    batch = jnp.stack(
        [jnp.asarray(psi), jnp.asarray(psi * 1.01), jnp.asarray(psi * 0.99)]
    )
    vfun = jax.vmap(
        lambda p: fsc.flux_surface_bins_jax(
            p,
            jnp.asarray(rg),
            jnp.asarray(zg),
            jnp.asarray(inside),
            jnp.asarray(0.0),
            jnp.asarray(-1.0),
            jnp.asarray(0.04),
            jnp.asarray(0.985),
            28,
            jnp.asarray(1.25),
        )["inv_r2"]
    )
    vb = vfun(batch)
    assert vb.shape == (3, 28)

    # grad of a scalar of the metrics w.r.t. the boundary flux flows and is finite
    def loss(pb):
        o = fsc.flux_surface_bins_jax(
            jnp.asarray(psi),
            jnp.asarray(rg),
            jnp.asarray(zg),
            jnp.asarray(inside),
            jnp.asarray(0.0),
            pb,
            jnp.asarray(0.04),
            jnp.asarray(0.985),
            28,
            jnp.asarray(1.25),
        )
        return jnp.mean(o["inv_r2"])

    g = jax.grad(loss)(jnp.asarray(-1.0))
    assert np.isfinite(float(g))


def test_output_shape_is_fixed_independent_of_core_size():
    """The metric arrays are (n_psin,) whatever the core size — the fixed-shape
    property a device batch requires (the coarea path's intermediates are not)."""
    n_psin = 28
    small = _solovev_psi(a=0.35)  # small plasma → few core cells
    big = _solovev_psi(a=0.7)  # large plasma → many core cells
    o_s = _bins(*small, n_psin=n_psin)
    o_b = _bins(*big, n_psin=n_psin)
    assert int(o_s["n_core_cells"]) != int(o_b["n_core_cells"])  # genuinely different
    for k in ("pn_s", "dv_dpn", "inv_r2", "inv_r", "grad2_r2", "v_cum"):
        assert np.asarray(o_s[k]).shape == (n_psin,)
        assert np.asarray(o_b[k]).shape == (n_psin,)


def test_module_is_contour_free():
    """G2c: the FSA path IMPORTS no contour / marching-squares / level-set /
    ndimage machinery (AST-checked, so docstring mentions of what it avoids do
    not trip it)."""
    import ast

    tree = ast.parse(inspect.getsource(fsc))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            imported += [base] + [f"{base}.{a.name}" for a in node.names]
    banned = ("contourpy", "matplotlib", "skimage", "scipy.ndimage", "ndimage")
    for imp in imported:
        assert not any(b in imp for b in banned), f"contour-free FSA imports {imp!r}"
    # and no sort over cells anywhere in the path (fixed-shape reductions only)
    calls = {
        n.func.attr
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    }
    assert "argsort" not in calls and "sort" not in calls


def test_flood_fill_core_matches_ndimage_label():
    """The device flood-fill selects exactly the axis-connected component that
    scipy.ndimage.label would — connectivity correct, computed fixed-shape."""
    import jax.numpy as jnp
    from scipy import ndimage

    psi, rg, zg, inside = _solovev_psi()
    R, _ = np.meshgrid(rg, zg)
    psi_n = (psi - 0.0) / (-1.0)
    confined = (psi_n < 1.0) & inside
    # inject a disconnected private-like pocket at comparable flux, off-axis
    confined[5:10, 3:6] = True
    ia = int(np.argmin(np.abs(zg - 0.0)))
    ja = int(np.argmin(np.abs(rg - 0.9)))
    seed = np.zeros_like(confined)
    seed[ia, ja] = True

    core = np.asarray(
        fsc.flood_fill_core(jnp.asarray(confined), jnp.asarray(seed), rg.size + zg.size)
    ).astype(bool)
    labels, _ = ndimage.label(confined)
    ref = labels == labels[ia, ja]
    assert np.array_equal(core, ref)
    assert not core[7, 4]  # the disconnected pocket is correctly excluded


# --- engine-level physics agreement (synthetic solve, no data) --------------


def test_connectivity_geometry_closes_ampere_and_volume():
    """The connectivity FSA on a solved equilibrium closes Ampère (edge enclosed
    current = Ip) and reproduces the core volume — the coarea invariants."""
    grid, table = _interior_limiter_fixture()
    ip = 4.0e5
    i_pf = np.array([-8.0e4, -8.0e4])
    lf, _, _ = _ladder_slice(grid, table, i_pf, ip)
    # at converged radial resolution the edge extrapolation is exact for both
    # reads (n_rho=24 leaves an O(Δρ̂) edge-extrapolation gap shared by coarea).
    geo = flux_surface_geometry(
        lf.result.psi,
        grid,
        coeffs=lf.coeffs,
        ip_amperes=ip,
        n_p=1,
        n_f=1,
        nonneg=True,
        b_phi0=1.0,
        n_rho=48,
        fsa_mode="connectivity",
    )
    assert geo is not None
    i_edge = geo.enclosed_current(geo.psi_face)[-1]
    assert abs(i_edge - ip) / ip < 0.03
    core = lf.result.core_mask.ravel()
    v_cells = float((2.0 * np.pi * grid.flat_r[core]).sum() * grid.dr * grid.dz)
    assert abs(geo.volume - v_cells) / v_cells < 0.05
    assert np.all(np.isfinite(geo.q_face)) and np.all(geo.q_face > 0)
    assert np.all(np.diff(geo.psi_n_face) >= -1e-12)


def test_connectivity_d_coefficient_tracks_coarea():
    """Same surfaces, a smoother estimator: the two FSA reads of d = g2·g3/ρ̂
    must agree in the mid-radius band to a physical tolerance."""
    grid, table = _interior_limiter_fixture()
    ip = 4.0e5
    i_pf = np.array([-8.0e4, -8.0e4])
    lf, _, _ = _ladder_slice(grid, table, i_pf, ip)
    kw = dict(coeffs=lf.coeffs, ip_amperes=ip, n_p=1, n_f=1, nonneg=True, b_phi0=1.0)
    g_co = flux_surface_geometry(lf.result.psi, grid, n_rho=48, fsa_mode="coarea", **kw)
    g_cn = flux_surface_geometry(
        lf.result.psi, grid, n_rho=48, fsa_mode="connectivity", **kw
    )
    assert g_co is not None and g_cn is not None
    band = slice(6, 42)  # skip the axis regular limit and the edge taper
    d_co = g_co.d_face[band]
    d_cn = g_cn.d_face[band]
    denom = float(np.sqrt(np.mean(d_co**2)))
    rel = float(np.sqrt(np.mean((d_cn - d_co) ** 2))) / max(denom, 1e-30)
    assert rel < 0.35, (
        rel
    )  # same physical surfaces; a smoother read, not a different one


def test_default_fsa_mode_is_coarea_byte_identical():
    """The default path is byte-identical to explicit coarea (opt-in guarantee)."""
    grid, table = _interior_limiter_fixture()
    ip = 4.0e5
    i_pf = np.array([-8.0e4, -8.0e4])
    lf, _, _ = _ladder_slice(grid, table, i_pf, ip)
    kw = dict(coeffs=lf.coeffs, ip_amperes=ip, n_p=1, n_f=1, nonneg=True, b_phi0=1.0)
    g_default = flux_surface_geometry(lf.result.psi, grid, **kw)
    g_coarea = flux_surface_geometry(lf.result.psi, grid, fsa_mode="coarea", **kw)
    assert g_default is not None and g_coarea is not None
    assert np.array_equal(g_default.d_face, g_coarea.d_face)
    assert np.array_equal(g_default.g2_face, g_coarea.g2_face)
    assert np.array_equal(g_default.psi_face, g_coarea.psi_face)
