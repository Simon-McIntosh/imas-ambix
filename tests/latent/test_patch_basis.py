"""Tests for the patch-current forward substrate.

The substrate represents the plasma as piecewise-constant currents on the
in-limiter grid cells and maps them to flux / field / sensors by precomputed
finite-area Green's interaction matrices.  Correctness is pinned analytically on
synthetic geometry — no MAST data, no EFIT:

* the finite-area kernel reduces to the point filament in the far field;
* the patch→grid flux superposition agrees with an independent Dirichlet FD
  solve of Δ*Φ = −2π μ0 R jφ (the 2π total-flux convention — a missing 2π
  shows up at ~20%);
* the FD Δ* of a single patch's Green's flux integrates back to that patch's
  current (the Ampère identity);
* the batched torch forward equals the per-slice loop, and the fp32 buffers
  track the fp64 numpy topology-read path;
* the cache round-trips;
* the module is firewall-clean by construction (static check).
"""

from __future__ import annotations

import numpy as np
import torch

from imas_ambix.latent.gs_solve import MU0, EquilibriumGrid
from imas_ambix.latent.patch_basis import PatchBasis


def _confining_table():
    """Synthetic machine: rectangular limiter + a vertical-field coil pair."""
    from imas_ambix.gs import geometry as gsg

    probes = [
        gsg.BProbe(index=i, r=1.35, z=-0.6 + 0.3 * i, angle_deg=-90.0, length=0.02)
        for i in range(5)
    ]
    sensor_map = [
        gsg.SensorMapping(f"obv{i:02d}", "b_probe", i, p.r, p.z, p.angle_deg, 0.001, "")
        for i, p in enumerate(probes)
    ]
    pf = [
        gsg.PFFilament(
            r=1.1, z=1.0, turns=1.0, width=0.06, height=0.06, circuit=1, xmult=1.0
        ),
        gsg.PFFilament(
            r=1.1, z=-1.0, turns=1.0, width=0.06, height=0.06, circuit=2, xmult=1.0
        ),
    ]
    return gsg.GeometryTable(
        signature=gsg.SetupSignature(
            n_bprobe=5, n_fluxloop=0, n_pf_filament=2, n_limiter=5, digest="feed0000"
        ),
        shots=[1],
        b_probes=probes,
        flux_loops=[],
        pf_filaments=pf,
        limiter_r=[0.35, 1.45, 1.45, 0.35, 0.35],
        limiter_z=[-0.85, -0.85, 0.85, 0.85, -0.85],
        sensor_map=sensor_map,
        passive_structures=[],
        amc_current_channels=[],
        unmatched_amb=[],
    )


def _confining_table_with_interior_coil():
    """``_confining_table`` plus a third coil sitting INSIDE the limiter.

    The two confining coils sit at z=+-1.0, outside the limiter's z-range
    [-0.85, 0.85], so they never exercise the candidate-mask's conductor
    exclusion (a cell can only be excluded if it is both in-limiter and
    inside a winding pack).  This adds an in-vessel-style pack at
    (r=0.9, z=0.4) that genuinely straddles in-limiter grid cells.
    """
    from imas_ambix.gs import geometry as gsg

    table = _confining_table()
    interior_coil = gsg.PFFilament(
        r=0.9, z=0.4, turns=1.0, width=0.12, height=0.12, circuit=3, xmult=1.0
    )
    table.pf_filaments = [*table.pf_filaments, interior_coil]
    table.signature = gsg.SetupSignature(
        n_bprobe=5, n_fluxloop=0, n_pf_filament=3, n_limiter=5, digest="feed0001"
    )
    return table


def _delta_star(psi2d: np.ndarray, rg: np.ndarray, zg: np.ndarray) -> np.ndarray:
    """5-point FD Δ* on the interior (same stencil as the Picard solver)."""
    dr = float(rg[1] - rg[0])
    dz = float(zg[1] - zg[0])
    out = np.full_like(psi2d, np.nan)
    r = rg[None, 1:-1]
    rp = 0.5 * (rg[1:-1] + rg[2:])[None, :]
    rm = 0.5 * (rg[1:-1] + rg[:-2])[None, :]
    ce = r / (rp * dr * dr)
    cw = r / (rm * dr * dr)
    cn = 1.0 / (dz * dz)
    out[1:-1, 1:-1] = (
        ce * psi2d[1:-1, 2:]
        + cw * psi2d[1:-1, :-2]
        + cn * (psi2d[2:, 1:-1] + psi2d[:-2, 1:-1])
        - (ce + cw + 2.0 * cn) * psi2d[1:-1, 1:-1]
    )
    return out


def test_far_field_kernel_matches_point_filament():
    """The finite-area kernel column reduces to the point filament far away."""
    from imas_ambix.gs.cylinder import cylinder_greens
    from imas_ambix.gs.operator import greens_psi

    grid = EquilibriumGrid.from_table(_confining_table(), nr=41, nz=57)
    cr, cz = grid.flat_r[grid.cells], grid.flat_z[grid.cells]
    c_mid = grid.cells[int(np.argmin(np.hypot(cr - grid.r0, cz)))]
    ar, az = float(grid.flat_r[c_mid]), float(grid.flat_z[c_mid])
    dist = np.hypot(grid.flat_r - ar, grid.flat_z - az)
    far = dist > 10.0 * max(grid.dr, grid.dz)
    fa = cylinder_greens(grid.flat_r[far], grid.flat_z[far], ar, az, grid.dr, grid.dz)[
        0
    ]
    point = greens_psi(grid.flat_r[far], grid.flat_z[far], ar, az)
    rel = np.abs(fa - point) / np.maximum(np.abs(point), 1e-16)
    assert rel.max() < 1e-3


def test_smooth_current_matches_dirichlet_fd_solve():
    """Patch→grid superposition of a broad Gaussian equals a Dirichlet FD solve.

    The FD source is Δ*Φ = −2π μ0 R jφ (the total-flux convention); a per-radian
    source would under-weight the plasma well by 2π and this rel-RMS would jump
    to ~0.2, so the check pins the total-flux factor.
    """
    table = _confining_table()
    grid = EquilibriumGrid.from_table(table, nr=49, nz=65)
    basis = PatchBasis.from_table(table, nr=49, nz=65, cache_dir=None)
    r_c = grid.flat_r[grid.cells]
    z_c = grid.flat_z[grid.cells]
    blob = np.exp(-(((r_c - grid.r0) / 0.35) ** 2 + (z_c / 0.5) ** 2))
    i_cell = blob / blob.sum() * 4.0e5  # [A]

    psi_greens = basis.psi_grid_2d_np(i_cell, np.zeros(1))
    rhs = np.zeros(grid.flat_r.size)
    rhs[grid.cells] = -2.0 * np.pi * MU0 * r_c * (i_cell / (grid.dr * grid.dz))
    rhs2d = rhs.reshape(grid.nz, grid.nr)
    psi_b2d = np.zeros((grid.nz, grid.nr))
    psi_b2d.ravel()[grid.edge_idx] = psi_greens.ravel()[grid.edge_idx]
    psi_fd = grid.solve_dirichlet(rhs2d, psi_b2d)

    span = float(psi_greens.max() - psi_greens.min())
    rel_rms = float(np.sqrt(np.mean((psi_fd - psi_greens) ** 2)) / span)
    assert rel_rms <= 1e-3, f"FD-vs-Green's rel-RMS {rel_rms:.2e} (2π regression?)"


def test_single_patch_flux_integrates_to_its_current():
    """FD Δ* of one patch's Green's flux integrates back to the patch current.

    Ampère identity: ∫ jφ dA = ∫ Δ*Φ / (−2π μ0 R) dA = I_patch.  Generous rtol
    for the FD truncation of the sharply-peaked single-cell source.
    """
    table = _confining_table()
    grid = EquilibriumGrid.from_table(table, nr=49, nz=65)
    basis = PatchBasis.from_table(table, nr=49, nz=65, cache_dir=None)
    # a central patch, away from the FD boundary ring
    cr, cz = grid.flat_r[grid.cells], grid.flat_z[grid.cells]
    c = int(np.argmin(np.hypot(cr - grid.r0, cz)))
    i_cell = np.zeros(grid.cells.size)
    i_cell[c] = 1.0  # 1 A in a single patch
    psi2d = basis.psi_grid_2d_np(i_cell, np.zeros(1))
    lhs = _delta_star(psi2d, grid.rg, grid.zg)  # ≈ −2π μ0 R jφ
    interior = np.isfinite(lhs)
    r2d = grid.mesh_r
    jphi = np.where(interior, lhs / (-2.0 * np.pi * MU0 * r2d), 0.0)
    recovered = float(np.nansum(jphi[interior]) * grid.dr * grid.dz)
    assert abs(recovered - 1.0) < 0.05


def test_batched_forward_matches_per_slice_loop():
    """Batched sensors / psi_grid equal the per-slice loop."""
    table = _confining_table()
    basis = PatchBasis.from_table(table, nr=41, nz=57, cache_dir=None)
    n = int(basis.r_cells.shape[0])
    n_coil = int(basis.psi_coil_grid.shape[1])
    rng = np.random.default_rng(0)
    i_cell = torch.as_tensor(rng.standard_normal((4, n)) * 1e4, dtype=torch.float64)
    i_pf = torch.as_tensor(rng.standard_normal((4, n_coil)) * 1e4, dtype=torch.float64)

    sens_b = basis.sensors(i_cell, i_pf)
    psi_b = basis.psi_grid(i_cell, i_pf)
    for k in range(4):
        sens_k = basis.sensors(i_cell[k], i_pf[k])
        psi_k = basis.psi_grid(i_cell[k], i_pf[k])
        torch.testing.assert_close(sens_b[k], sens_k[0], rtol=1e-6, atol=1e-9)
        torch.testing.assert_close(psi_b[k], psi_k[0], rtol=1e-6, atol=1e-9)


def test_fp32_buffers_track_fp64_numpy_path():
    """The fp32 torch forward matches the fp64 numpy topology path."""
    table = _confining_table()
    basis = PatchBasis.from_table(
        table, nr=41, nz=57, cache_dir=None, dtype=torch.float32
    )
    n = int(basis.r_cells.shape[0])
    n_coil = int(basis.psi_coil_grid.shape[1])
    rng = np.random.default_rng(1)
    i_cell = rng.standard_normal(n)
    i_pf = rng.standard_normal(n_coil)

    psi32 = basis.psi_grid_2d(
        torch.as_tensor(i_cell, dtype=torch.float32),
        torch.as_tensor(i_pf, dtype=torch.float32),
    )[0].numpy()
    psi64 = basis.psi_grid_2d_np(i_cell, i_pf)
    span = float(psi64.max() - psi64.min())
    rel = float(np.abs(psi32 - psi64).max() / span)
    assert rel < 1e-4


def test_cache_round_trip(tmp_path):
    """from_table assembles once, then loads the cached g_pg (byte-equal)."""
    table = _confining_table()
    basis1 = PatchBasis.from_table(table, nr=33, nz=45, cache_dir=tmp_path)
    cache = tmp_path / f"g_pg_{table.signature.key}_33x45.npz"
    assert cache.exists(), "cache file was not written"
    basis2 = PatchBasis.from_table(table, nr=33, nz=45, cache_dir=tmp_path)
    np.testing.assert_array_equal(basis1._g_pg_np, basis2._g_pg_np)


def test_candidate_mask_excludes_interior_coil_cells():
    """Cells inside an in-vessel winding pack are dropped from candidate_mask.

    Both coils in ``_confining_table`` sit outside the limiter's z-range, so
    ``clear_of_conductors`` never actually excludes anything there — the
    exclusion logic is exercised in name only.  This places a third coil
    INSIDE the limiter and checks that cells within its (dilated) footprint
    are excluded from ``PatchBasis.candidate_mask``, while other in-limiter
    cells remain candidates.
    """
    table = _confining_table_with_interior_coil()
    grid = EquilibriumGrid.from_table(table, nr=49, nz=65)
    basis = PatchBasis.from_table(table, nr=49, nz=65, cache_dir=None)

    r_cells = grid.flat_r[grid.cells]
    z_cells = grid.flat_z[grid.cells]
    r0, r1, z0, z1 = grid.conductor_rects[-1]  # the interior coil's raw pack
    dr, dz = grid.dr, grid.dz
    in_pack = (
        (r_cells >= r0 - dr)
        & (r_cells <= r1 + dr)
        & (z_cells >= z0 - dz)
        & (z_cells <= z1 + dz)
    )
    assert in_pack.any(), "fixture coil footprint misses the cell grid"

    candidate = basis.candidate_mask.numpy() > 0.5
    assert not candidate[in_pack].any()  # excluded: conductor interior
    assert candidate[~in_pack].any()  # other in-limiter cells stay candidates


def test_firewall_static_no_evaluator_imports():
    """The substrate module must not touch the EFIT/evaluator side."""
    from pathlib import Path

    import imas_ambix.latent.patch_basis as m

    src = Path(m.__file__).read_text()
    for banned in ("efit_referee", "equilibrium_labels", "worldmodel"):
        assert banned not in src, f"patch_basis imports the firewalled {banned}"
