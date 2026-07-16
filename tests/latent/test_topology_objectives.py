"""Topology-aware training penalties — the load-bearing contracts.

* pointwise interpolation rows reproduce a grid field's value and gradient:
  exact for quadratic fields (central-FD + bilinear of the then-linear
  gradient), and consistent with the exact cylinder-kernel field at an
  off-node point;
* every penalty is a delta form about the classical spine solution: it is
  IDENTICALLY zero at zero correction (``di = 0``, ``da = 0``) — never a
  near-zero residual that would leak spine-emulation pressure at dc = 0;
* the terminator gradient penalty sees a gradient change at an X-point
  candidate through the full projector, but only the tangential component
  at a limiter-contact candidate;
* the boundary-flux consistency term is free for a uniform flux shift across
  terminator candidates and positive for a differential one;
* the critical-point integrity penalty activates only when a correction
  ERODES the spine's own gradient margin (creating a spurious null), never
  when it strengthens the field.
"""

from __future__ import annotations

import numpy as np
import torch

from imas_ambix.latent.topology_objectives import (
    MAX_TERMINATOR_CANDIDATES,
    build_slice_anchor,
    integrity_penalty,
    median_gradient_scale,
    point_rows,
    terminator_penalty,
)

RG = np.linspace(0.2, 2.0, 61)
ZG = np.linspace(-1.5, 1.5, 81)


def _quadratic_field():
    """ψ and its exact gradient for a quadratic test field on the grid."""
    rr, zz = np.meshgrid(RG, ZG)
    psi = 0.3 + 0.5 * rr - 0.2 * zz + 0.1 * rr**2 - 0.15 * rr * zz + 0.05 * zz**2

    def grad(r, z):
        return 0.5 + 0.2 * r - 0.15 * z, -0.2 - 0.15 * r + 0.1 * z

    return psi, grad


def test_point_rows_exact_on_quadratic_field():
    """Central FD is exact for quadratics and the FD-gradient field is linear,
    so bilinear interpolation of it is ALSO exact — gradients must match to
    round-off at nodes and off-node alike; values to O(h²) off-node."""
    psi, grad = _quadratic_field()
    flat = psi.ravel()
    pts = np.array(
        [
            [RG[20], ZG[30]],  # exactly on a node
            [0.987, 0.4321],  # generic interior point
            [1.612, -0.7789],
        ]
    )
    rows = point_rows(RG, ZG, pts)
    assert rows.shape == (3, 3, flat.size)
    for q, (r, z) in enumerate(pts):
        val, d_r, d_z = rows[q] @ flat
        gr, gz = grad(r, z)
        assert np.isclose(d_r, gr, rtol=0, atol=1e-10), f"d/dR at point {q}"
        assert np.isclose(d_z, gz, rtol=0, atol=1e-10), f"d/dZ at point {q}"
        exact = 0.3 + 0.5 * r - 0.2 * z + 0.1 * r**2 - 0.15 * r * z + 0.05 * z**2
        assert np.isclose(val, exact, rtol=1e-3)
    # the on-node point is exact for the value too
    assert np.isclose(rows[0, 0] @ flat, psi[30, 20], rtol=0, atol=1e-14)


def test_point_rows_gradient_matches_cylinder_kernel_field():
    """Rows applied to a kernel-generated grid field reproduce the kernel's own
    pointwise gradient: ∂ψ/∂R = 2πR·B_Z, ∂ψ/∂Z = −2πR·B_R."""
    from imas_ambix.gs.cylinder import hybrid_greens

    rr, zz = np.meshgrid(RG, ZG)
    src = (0.9, 0.1, 0.04, 0.05)
    psi_flat = hybrid_greens(rr.ravel(), zz.ravel(), *src)[0]
    pt = np.array([[1.3111, 0.3499]])
    rows = point_rows(RG, ZG, pt)
    val, d_r, d_z = rows[0] @ psi_flat

    r, z = pt[0]
    psi_x, br_x, bz_x = hybrid_greens(np.array([r]), np.array([z]), *src)
    assert np.isclose(val, float(psi_x[0]), rtol=2e-3)
    assert np.isclose(d_r, 2.0 * np.pi * r * float(bz_x[0]), rtol=2e-2)
    assert np.isclose(d_z, -2.0 * np.pi * r * float(br_x[0]), rtol=2e-2)


def _toy_anchor_tensors(n_cells=4, k=2, q=2):
    """Hand-built batched anchor tensors: candidate gradients read the first
    two cell currents directly, candidate fluxes read them summed."""
    n = 3  # steps
    rows_cell = torch.zeros(n, q, 3, n_cells, dtype=torch.float64)
    rows_mode = torch.zeros(n, q, 3, k, dtype=torch.float64)
    # value row: candidate c reads cell c
    for c in range(q):
        rows_cell[:, c, 0, c] = 1.0
    # gradient rows: (d/dR, d/dZ) read cells (0, 1) for every candidate
    rows_cell[:, :, 1, 0] = 1.0
    rows_cell[:, :, 2, 1] = 1.0
    proj = torch.eye(2, dtype=torch.float64).expand(n, q, 2, 2).clone()
    w_flux = torch.full((n, q), 1.0 / q, dtype=torch.float64)
    cand_mask = torch.ones(n, q, dtype=torch.bool)
    gscale = torch.ones(n, dtype=torch.float64)
    fscale = torch.ones(n, dtype=torch.float64)
    return rows_cell, rows_mode, proj, w_flux, cand_mask, gscale, fscale


def test_terminator_penalty_zero_at_zero_correction():
    rows_cell, rows_mode, proj, w, cm, gs, fs = _toy_anchor_tensors()
    di = torch.zeros(3, 4, dtype=torch.float64)
    da = torch.zeros(3, 2, dtype=torch.float64)
    pen = terminator_penalty(di, da, rows_cell, rows_mode, proj, w, cm, gs, fs)
    assert pen.shape == (3,)
    assert torch.all(pen == 0.0)


def test_terminator_penalty_detects_gradient_change():
    rows_cell, rows_mode, proj, w, cm, gs, fs = _toy_anchor_tensors()
    di = torch.zeros(3, 4, dtype=torch.float64)
    di[:, 0] = 0.7  # induces d/dR = 0.7 at both candidates
    da = torch.zeros(3, 2, dtype=torch.float64)
    pen = terminator_penalty(di, da, rows_cell, rows_mode, proj, w, cm, gs, fs)
    assert torch.all(pen > 0.0)


def test_tangency_projector_ignores_normal_component():
    """A limiter-contact candidate penalises only the tangential gradient
    change: t̂ = R̂ here, so a pure d/dZ change is free while an equal d/dR
    change is not."""
    rows_cell, rows_mode, _proj, w, cm, gs, fs = _toy_anchor_tensors(q=1)
    t_hat = torch.tensor([1.0, 0.0], dtype=torch.float64)
    proj = torch.einsum("i,j->ij", t_hat, t_hat).expand(3, 1, 2, 2).clone()
    da = torch.zeros(3, 2, dtype=torch.float64)

    di_normal = torch.zeros(3, 4, dtype=torch.float64)
    di_normal[:, 1] = 0.9  # d/dZ only — normal to the projector; the value
    # row reads cell 0, so the flux term stays zero for both corrections
    pen_n = terminator_penalty(di_normal, da, rows_cell, rows_mode, proj, w, cm, gs, fs)

    di_tan = torch.zeros(3, 4, dtype=torch.float64)
    di_tan[:, 0] = 0.9  # d/dR only — along the tangent
    pen_t = terminator_penalty(di_tan, da, rows_cell, rows_mode, proj, w, cm, gs, fs)

    # a single candidate's flux change is always consistent (w = 1 → zero
    # spread), so the difference is the projected gradient alone
    assert torch.all(pen_n == 0.0)
    assert torch.all(pen_t > 0.0)


def test_flux_consistency_uniform_shift_is_free():
    """Shifting every terminator candidate's flux together never re-orders
    the softmin — only DIFFERENTIAL flux changes are penalised."""
    rows_cell, rows_mode, proj, w, cm, gs, fs = _toy_anchor_tensors()
    rows_cell[:, :, 1:, :] = 0.0  # kill the gradient rows — flux term only
    da = torch.zeros(3, 2, dtype=torch.float64)

    di_uniform = torch.zeros(3, 4, dtype=torch.float64)
    di_uniform[:, 0] = 0.4
    di_uniform[:, 1] = 0.4  # both candidate fluxes shift by 0.4
    pen_u = terminator_penalty(
        di_uniform, da, rows_cell, rows_mode, proj, w, cm, gs, fs
    )
    assert torch.all(pen_u.abs() < 1e-14)

    di_diff = torch.zeros(3, 4, dtype=torch.float64)
    di_diff[:, 0] = 0.4
    di_diff[:, 1] = -0.4  # candidates move apart — terminator can re-order
    pen_d = terminator_penalty(di_diff, da, rows_cell, rows_mode, proj, w, cm, gs, fs)
    assert torch.all(pen_d > 0.0)


def test_terminator_penalty_invalid_candidates_are_inert():
    rows_cell, rows_mode, proj, w, cm, gs, fs = _toy_anchor_tensors()
    cm[:, 1] = False
    w[:, 1] = 0.0
    w[:, 0] = 1.0
    rows_cell[:, 1] = 0.0  # invalid candidates carry zero rows (as built)
    di = torch.zeros(3, 4, dtype=torch.float64)
    di[:, 1] = 0.5  # moves only candidate 1's flux — which is invalid
    da = torch.zeros(3, 2, dtype=torch.float64)
    pen_flux_only = terminator_penalty(
        di, da, rows_cell * 0 + rows_cell, rows_mode, proj * 0, w, cm, gs, fs
    )
    assert torch.all(torch.isfinite(pen_flux_only))
    assert torch.all(pen_flux_only == 0.0)


def _integrity_setup(n=2):
    rr, _zz = np.meshgrid(RG, ZG)
    psi0 = rr.copy()  # uniform d/dR = 1 everywhere
    g = psi0.size
    psi0_t = torch.tensor(np.tile(psi0.ravel(), (n, 1)), dtype=torch.float64)
    region = torch.ones(g, dtype=torch.bool)
    s_med = torch.ones(n, dtype=torch.float64)
    dr = float(RG[1] - RG[0])
    dz = float(ZG[1] - ZG[0])
    return psi0_t, region, s_med, dr, dz


def test_integrity_penalty_zero_at_zero_correction():
    psi0, region, s_med, dr, dz = _integrity_setup()
    dpsi = torch.zeros_like(psi0)
    pen = integrity_penalty(
        dpsi, psi0, region, s_med, nz=len(ZG), nr=len(RG), dr=dr, dz=dz
    )
    assert pen.shape == (2,)
    assert torch.all(pen == 0.0)


def test_integrity_penalises_gradient_erosion_not_strengthening():
    psi0, region, s_med, dr, dz = _integrity_setup()
    kw = dict(nz=len(ZG), nr=len(RG), dr=dr, dz=dz)
    # erode: cancel the field entirely — a flat ψ is one giant degenerate null
    pen_erode = integrity_penalty(-psi0, psi0, region, s_med, **kw)
    assert torch.all(pen_erode > 0.0)
    # strengthen: double the gradient — no new null is representable
    pen_strong = integrity_penalty(psi0.clone(), psi0, region, s_med, **kw)
    assert torch.all(pen_strong == 0.0)


def test_integrity_penalty_zero_on_all_zero_padded_step():
    psi0, region, s_med, dr, dz = _integrity_setup()
    psi0[1] = 0.0  # padded step: spine flux zero, correction zero
    dpsi = torch.zeros_like(psi0)
    pen = integrity_penalty(
        dpsi, psi0, region, s_med, nz=len(ZG), nr=len(RG), dr=dr, dz=dz
    )
    assert torch.all(torch.isfinite(pen))
    assert float(pen[1]) == 0.0


def test_integrity_gradient_flows_to_correction():
    psi0, region, s_med, dr, dz = _integrity_setup(n=1)
    # erode the gradient BELOW the capped margin (0.3·s_med) so the penalty
    # is active and carries gradient
    dpsi = (-0.9 * psi0).clone().requires_grad_(True)
    pen = integrity_penalty(
        dpsi, psi0, region, s_med, nz=len(ZG), nr=len(RG), dr=dr, dz=dz
    )
    pen.sum().backward()
    assert dpsi.grad is not None
    assert float(dpsi.grad.abs().sum()) > 0.0


def _saddle_field():
    """ψ with an axis max at (1.1, 0) and an X-point-like saddle at (1.1, −0.9)."""
    rr, zz = np.meshgrid(RG, ZG)
    psi = np.exp(-((rr - 1.1) ** 2 + zz**2) / 0.18) - 0.4 * np.exp(
        -((rr - 1.1) ** 2 + (zz + 1.2) ** 2) / 0.25
    )
    return psi


def test_build_slice_anchor_full_path_zero_at_zero_correction():
    psi = _saddle_field()
    g = psi.size
    n_cells, k = 12, 3
    rng = np.random.default_rng(2)
    g_pg = rng.normal(size=(g, n_cells))
    g_grid = rng.normal(size=(g, k))
    target = np.full(14, np.nan)
    target[0], target[1] = 1.1, 0.0  # axis
    target[2], target[3] = 1.1, -0.9  # xpt0; xpt1 stays NaN
    theta = np.linspace(0.0, 2.0 * np.pi, 33)
    lim_r = 1.1 + 0.75 * np.cos(theta)
    lim_z = 1.05 * np.sin(theta)
    s_med = median_gradient_scale(psi, RG, ZG)
    anchor = build_slice_anchor(
        psi, RG, ZG, target, lim_r, lim_z, g_pg, g_grid, grad_scale=s_med
    )
    assert anchor is not None
    assert anchor.rows_cell.shape == (MAX_TERMINATOR_CANDIDATES, 3, n_cells)
    assert anchor.rows_mode.shape == (MAX_TERMINATOR_CANDIDATES, 3, k)
    assert int(anchor.cand_mask.sum()) == 2  # xpt0 + limiter contact
    assert np.isclose(anchor.w_flux.sum(), 1.0)
    assert anchor.flux_scale > 0.0 and anchor.grad_scale > 0.0
    # full-path delta form: zero correction → zero penalty, exactly
    di = torch.zeros(1, n_cells, dtype=torch.float64)
    da = torch.zeros(1, k, dtype=torch.float64)
    to = lambda x: torch.tensor(x, dtype=torch.float64).unsqueeze(0)  # noqa: E731
    pen = terminator_penalty(
        di,
        da,
        to(anchor.rows_cell),
        to(anchor.rows_mode),
        to(anchor.proj),
        to(anchor.w_flux),
        torch.tensor(anchor.cand_mask).unsqueeze(0),
        torch.tensor([anchor.grad_scale], dtype=torch.float64),
        torch.tensor([anchor.flux_scale], dtype=torch.float64),
    )
    assert torch.all(pen == 0.0)
    pen1 = terminator_penalty(
        di + 0.3,
        da,
        to(anchor.rows_cell),
        to(anchor.rows_mode),
        to(anchor.proj),
        to(anchor.w_flux),
        torch.tensor(anchor.cand_mask).unsqueeze(0),
        torch.tensor([anchor.grad_scale], dtype=torch.float64),
        torch.tensor([anchor.flux_scale], dtype=torch.float64),
    )
    assert torch.all(pen1 > 0.0)


def test_build_slice_anchor_returns_none_without_axis():
    psi = _saddle_field()
    target = np.full(14, np.nan)
    anchor = build_slice_anchor(
        psi,
        RG,
        ZG,
        target,
        np.array([0.4, 1.8, 1.8, 0.4]),
        np.array([-1.0, -1.0, 1.0, 1.0]),
        np.zeros((psi.size, 4)),
        np.zeros((psi.size, 2)),
        grad_scale=1.0,
    )
    assert anchor is None
