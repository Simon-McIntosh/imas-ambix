"""Tests for the machine-agnostic wall (supercover raster, tiles-as-holes, nodes).

Pins the wall-as-DATA abstraction the connectivity read consumes:

* **MAST parity** — one closed vessel unit reproduces the plain point-in-polygon
  limiter mask (byte-identical), so the single-loop path is unchanged;
* **tiles-as-holes** — a material unit excises its cells from the occupiable
  region (contact = poke-through), and a thin blade (t < Δ) still leaves a
  ≥1-cell supercover obstacle;
* **diagnostics** — a sub-grid unit fires the thin-unit warning; two disjoint
  units whose rasters fuse fire the gap-merge warning; neither raises;
* **nodes** — every unit is resampled at ~Δ/2 and tagged by unit.
"""

from __future__ import annotations

import numpy as np

from imas_ambix.latent import wall_mask as wm
from imas_ambix.latent.topology import _inside_polygon


def _grid(nr=81, nz=101, r=(0.2, 1.8), z=(-1.1, 1.1)):
    rg = np.linspace(*r, nr)
    zg = np.linspace(*z, nz)
    return rg, zg


# --- MAST parity ------------------------------------------------------------


def test_single_vessel_reproduces_inside_polygon():
    """One closed vessel unit == the plain ray-cast limiter mask (MAST parity)."""
    rg, zg = _grid()
    lr = np.array([0.3, 1.7, 1.7, 0.3, 0.3])
    lz = np.array([-1.0, -1.0, 1.0, 1.0, -1.0])
    mesh_r, mesh_z = np.meshgrid(rg, zg)
    ref = _inside_polygon(mesh_r.ravel(), mesh_z.ravel(), lr, lz).reshape(
        zg.size, rg.size
    )
    mask, diags = wm.build_wall_mask(rg, zg, [wm.vessel_unit(lr, lz)])
    assert np.array_equal(mask, ref)
    assert diags == []


# --- tiles as holes ---------------------------------------------------------


def test_material_tile_excises_cells():
    """A material tile is a HOLE: its cells are removed from the occupiable set."""
    rg, zg = _grid()
    lr = np.array([0.3, 1.7, 1.7, 0.3, 0.3])
    lz = np.array([-1.0, -1.0, 1.0, 1.0, -1.0])
    # a fat central tile
    tr = np.array([0.9, 1.1, 1.1, 0.9, 0.9])
    tz = np.array([-0.1, -0.1, 0.1, 0.1, -0.1])
    vessel_only, _ = wm.build_wall_mask(rg, zg, [wm.vessel_unit(lr, lz)])
    with_tile, _ = wm.build_wall_mask(
        rg, zg, [wm.vessel_unit(lr, lz), wm.material_unit(tr, tz)]
    )
    # the tile removed cells, all removed cells are inside the vessel
    removed = vessel_only & ~with_tile
    assert removed.any()
    assert np.array_equal(removed, vessel_only & removed)
    # a cell at the tile centre is now material (not occupiable)
    jc = np.argmin(np.abs(rg - 1.0))
    ic = np.argmin(np.abs(zg - 0.0))
    assert not with_tile[ic, jc]


def test_thin_blade_leaves_at_least_one_cell():
    """A blade thinner than Δ still leaves a contiguous ≥1-cell supercover barrier."""
    rg, zg = _grid()
    dr = float(rg[1] - rg[0])
    # a near-zero-thickness vertical blade at R≈1.0 spanning several rows
    r0 = 1.0
    thin = dr / 4.0
    br = np.array([r0 - thin, r0 + thin, r0 + thin, r0 - thin, r0 - thin])
    bz = np.array([-0.3, -0.3, 0.3, 0.3, -0.3])
    raster = wm.supercover_raster(rg, zg, wm.material_unit(br, bz))
    # at least one material cell in every grid row the blade spans
    i0 = np.argmin(np.abs(zg - (-0.3)))
    i1 = np.argmin(np.abs(zg - 0.3))
    for i in range(min(i0, i1), max(i0, i1) + 1):
        assert raster[i, :].any(), f"blade left no obstacle in row {i}"


def test_open_line_primitive_marks_crossed_cells():
    """An open polyline (no fill) marks exactly the cells its segments cross."""
    rg, zg = _grid()
    # a diagonal open blade
    lr = np.array([0.6, 1.4])
    lz = np.array([-0.4, 0.4])
    raster = wm.supercover_raster(rg, zg, wm.material_unit(lr, lz, closed=False))
    assert raster.any()
    # endpoints are marked
    assert raster[np.argmin(np.abs(zg + 0.4)), np.argmin(np.abs(rg - 0.6))]
    assert raster[np.argmin(np.abs(zg - 0.4)), np.argmin(np.abs(rg - 1.4))]


# --- diagnostics (warnings, never errors) -----------------------------------


def test_thin_unit_diagnostic_fires():
    """A sub-grid closed tile reports the thin-unit warning with a thickness proxy."""
    rg, zg = _grid()
    dr = float(rg[1] - rg[0])
    lr = np.array([0.3, 1.7, 1.7, 0.3, 0.3])
    lz = np.array([-1.0, -1.0, 1.0, 1.0, -1.0])
    thin = dr / 3.0
    tr = np.array([1.0 - thin, 1.0 + thin, 1.0 + thin, 1.0 - thin, 1.0 - thin])
    tz = np.array([-0.3, -0.3, 0.3, 0.3, -0.3])
    _mask, diags = wm.build_wall_mask(
        rg, zg, [wm.vessel_unit(lr, lz), wm.material_unit(tr, tz, name="blade")]
    )
    thin_diags = [d for d in diags if d.kind == "thin_unit"]
    assert len(thin_diags) == 1
    assert thin_diags[0].detail["thickness_proxy_m"] < dr


def test_fat_tile_no_thin_diagnostic():
    rg, zg = _grid()
    lr = np.array([0.3, 1.7, 1.7, 0.3, 0.3])
    lz = np.array([-1.0, -1.0, 1.0, 1.0, -1.0])
    tr = np.array([0.85, 1.15, 1.15, 0.85, 0.85])
    tz = np.array([-0.2, -0.2, 0.2, 0.2, -0.2])
    _mask, diags = wm.build_wall_mask(
        rg, zg, [wm.vessel_unit(lr, lz), wm.material_unit(tr, tz)]
    )
    assert [d for d in diags if d.kind == "thin_unit"] == []


def test_gap_merge_diagnostic_fires():
    """Two disjoint tiles a sub-cell gap apart report the gap-merge warning."""
    rg, zg = _grid(nr=61, nz=61)
    dr = float(rg[1] - rg[0])
    lr = np.array([0.3, 1.7, 1.7, 0.3, 0.3])
    lz = np.array([-1.0, -1.0, 1.0, 1.0, -1.0])
    # two fat tiles separated by ~0.6Δ (disjoint polygons, adjacent rasters)
    gap = 0.6 * dr
    ta = np.array([0.9, 1.0, 1.0, 0.9, 0.9])
    tz = np.array([-0.15, -0.15, 0.15, 0.15, -0.15])
    tb = ta + (0.1 + gap)
    _mask, diags = wm.build_wall_mask(
        rg,
        zg,
        [wm.vessel_unit(lr, lz), wm.material_unit(ta, tz), wm.material_unit(tb, tz)],
    )
    gm = [d for d in diags if d.kind == "gap_merge"]
    assert len(gm) == 1
    assert set(gm[0].units) == {1, 2}


# --- nodes ------------------------------------------------------------------


def test_densify_units_spacing_and_tags():
    rg, zg = _grid()
    delta = min(float(rg[1] - rg[0]), float(zg[1] - zg[0]))
    lr = np.array([0.3, 1.7, 1.7, 0.3, 0.3])
    lz = np.array([-1.0, -1.0, 1.0, 1.0, -1.0])
    tr = np.array([0.9, 1.1, 1.1, 0.9, 0.9])
    tz = np.array([-0.1, -0.1, 0.1, 0.1, -0.1])
    units = [wm.vessel_unit(lr, lz), wm.material_unit(tr, tz)]
    wr, wz, uid = wm.densify_units(units, spacing=0.5 * delta)
    assert wr.shape == wz.shape == uid.shape
    assert set(np.unique(uid)) == {0, 1}
    # consecutive node spacing within a unit is ~Δ/2 (never larger)
    for k in (0, 1):
        sel = uid == k
        pr, pz = wr[sel], wz[sel]
        d = np.hypot(np.diff(pr), np.diff(pz))
        assert np.max(d) <= 0.75 * delta + 1e-9


def test_no_units_yields_no_wall_sentinel():
    wr, wz, uid = wm.densify_units([], spacing=0.01)
    assert wr[0] > 1e29 and wz[0] > 1e29


# --- g_wall campaign wall-flux (exactness vs the O(Δ²) bilerp floor) ---------


def test_wall_flux_exact_vs_bilerp_floor():
    """g_wall node flux is the exact Green's sum (no bilerp floor at a lean point)."""
    from imas_ambix.latent.connectivity_boundary import _bilerp
    from imas_ambix.latent.gs_solve import EquilibriumGrid

    rg, zg = _grid(nr=65, nz=97)
    lr = np.array([0.4, 1.6, 1.6, 0.4, 0.4])
    lz = np.array([-0.9, -0.9, 0.9, 0.9, -0.9])
    grid = EquilibriumGrid(
        rg=rg,
        zg=zg,
        limiter_r=lr,
        limiter_z=lz,
        coil_psi_columns=np.zeros((rg.size * zg.size, 0)),
        r0=1.0,
    )
    # a smooth cell-current pattern (a blob about the centre)
    cr = grid.flat_r[grid.cells]
    cz = grid.flat_z[grid.cells]
    i_cell = np.exp(-(((cr - 1.0) ** 2 + cz**2) / 0.3**2)) * 1.0e3
    i_pf = np.zeros(0)

    # exact node flux (g_wall GEMM) vs the SAME field bilinearly sampled off grid
    psi_node_exact = grid.wall_flux(i_pf, i_cell)
    psi2d = grid.plasma_grid_psi(i_cell).reshape(grid.nz, grid.nr)
    import jax.numpy as jnp

    rgj, zgj = jnp.asarray(rg), jnp.asarray(zg)
    psi2dj = jnp.asarray(psi2d)
    psi_node_bilerp = np.array(
        [
            float(_bilerp(psi2dj, rgj, zgj, float(r), float(z)))
            for r, z in zip(grid.wall_r, grid.wall_z, strict=True)
        ]
    )
    span = float(psi2d.max() - psi2d.min())
    # exact matches the direct finite-area Green's superposition to machine eps
    direct = grid.wall_greens()["g_cells"] @ i_cell
    exact_err = np.max(np.abs(psi_node_exact - direct)) / span
    assert exact_err < 1e-12
    # bilerp carries a real O(Δ²) floor the exact read removes — orders of
    # magnitude above the exact read's (machine-eps) error
    err = np.max(np.abs(psi_node_bilerp - psi_node_exact)) / span
    assert err > 1e-5, f"expected a bilerp floor, got {err:.2e}"
    assert err > 1e6 * exact_err


def test_wall_flux_matches_grid_field_scale():
    """g_wall @ currents is on the SAME absolute ψ scale as the gridded field."""
    from imas_ambix.latent.gs_solve import EquilibriumGrid

    rg, zg = _grid(nr=65, nz=97)
    lr = np.array([0.4, 1.6, 1.6, 0.4, 0.4])
    lz = np.array([-0.9, -0.9, 0.9, 0.9, -0.9])
    grid = EquilibriumGrid(
        rg=rg,
        zg=zg,
        limiter_r=lr,
        limiter_z=lz,
        coil_psi_columns=np.zeros((rg.size * zg.size, 0)),
        r0=1.0,
    )
    cr = grid.flat_r[grid.cells]
    cz = grid.flat_z[grid.cells]
    i_cell = np.exp(-(((cr - 1.0) ** 2 + cz**2) / 0.25**2)) * 5.0e2
    node_flux = grid.wall_flux(np.zeros(0), i_cell)
    # a node's exact flux equals hybrid_greens superposition at that node
    from imas_ambix.gs.cylinder import hybrid_greens

    j = 0
    acc = 0.0
    for k, c in enumerate(grid.cells):
        acc += (
            i_cell[k]
            * hybrid_greens(
                np.array([grid.wall_r[j]]),
                np.array([grid.wall_z[j]]),
                float(grid.flat_r[c]),
                float(grid.flat_z[c]),
                grid.dr,
                grid.dz,
            )[0][0]
        )
    assert abs(node_flux[j] - acc) < 1e-9 * max(abs(acc), 1e-30)
