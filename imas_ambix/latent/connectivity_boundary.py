"""Accelerator-native connectivity boundary read (JAX).

The last-closed-flux-surface resolved by CONNECTIVITY — the outermost closed
axis-enclosing flux contour that still lies inside the wall — computed with
fixed-shape, ``jit`` / ``vmap`` / ``grad``-safe device primitives.  This is the
device-native reimplementation of :func:`imas_ambix.latent.topology.lcfs_contour`
(the shipped monotone flux-offset push): SAME algorithm, no contourpy, no
``scipy.ndimage.label``, no ``argwhere`` / bisection-over-traced-contours.  ONE
code path handles limited and diverted plasmas alike, continuous through the
marginal limited↔diverted transition by construction (the boundary is never a
classify-first decision about *which* surface bounds).

Method (the connectivity read, re-expressed on device):

* **Normalised flux.**  ``u = (ψ − ψ_axis)/(ψ_out − ψ_axis)`` maps the axis to
  0 and the domain-edge extreme (the edge cell whose flux is furthest from the
  axis) to 1, so the confined side at a candidate level ``s`` is simply
  ``u ≤ s`` — sign-agnostic (MAST ψ_axis > ψ_bnd or the reverse).

* **Axis-connected region (flood-fill).**  At level ``s`` the confined-and-in-
  wall set ``(u ≤ s) ∧ inside_wall`` is flood-filled from the axis cell by
  iterated 4-neighbour dilation (the shipped ``flood_fill_core`` device kernel)
  — the axis-connected component, never a disconnected pocket.

* **The binding level, split by connectivity signature.**  A level is *valid*
  while the axis region stays clear of the wall; each escape transition is the
  largest valid level (monotone → coarse sweep + bisection).  ψ_bnd is read at
  the MEAN of an inner escape test (region touches the innermost in-wall shell,
  ~one cell inside) and an outer test (region dilation reaches a still-confined
  out-of-wall/edge cell, ~one cell outside) — unbiased for an interior SADDLE
  binding, so a DIVERTED separatrix (ψ_N = 1) is reproduced by the mean.  A
  LIMITED wall tangency is refined SUB-GRID: the minimum interpolated ψ_N over
  the wall boundary points adjacent to the axis region (the cell mean carries the
  sub-cell wall-position error a tangency is sensitive to; the interpolated wall
  crossing removes it).  Limited and diverted are told apart by the region SIZE
  RATIO across the binding — a saddle opening merges the region with the SOL /
  private flux and jumps the size, a tangency grows smoothly — so the sub-grid
  wall read is applied only to limited plasmas (on a diverted plasma the nearest
  wall sits outside the separatrix and a min-over-wall would overshoot).  The
  divertor legs are open branches the closed axis-region never floods, so the
  lobe — and the radii read off it — are unaffected.

* **LCFS radii.**  Read at ψ_lcfs = ψ_axis + lcfs_norm·(ψ_bnd − ψ_axis) by a
  fixed outward ray-march from the axis at the evaluator's 8 poloidal angles —
  a differentiable interpolated crossing, the same fixed parameterisation
  :func:`imas_ambix.latent.topology.lcfs_radii` uses on the host.

Everything is a fixed-shape reduction over the full grid: no data-dependent
shapes, no host round-trip, no contour extraction — so a batch of slices sharing
one campaign grid is a single ``jax.vmap``.  The only machine input is the wall
as a raster boolean mask (``inside_limiter``), so a single loop (MAST), a union
of discrete limiters (AUG), or a per-pulse movable wall (WEST) is data, not a
new code path.

Sub-grid note: the two-sided mean removes the ~one-cell systematic bias in the
scalar ψ_bnd, leaving an unbiased sub-cell residual; the LCFS radii are sub-grid
(interpolated ray crossing).  The sub-grid saddle/axis POSITION (as opposed to
the binding flux) is the nova stencil refinement — a separate rung, not folded
into the boundary sweep.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp

from imas_ambix.latent.flux_surface_connectivity import _dilate4, flood_fill_core
from imas_ambix.latent.stencil_nulls import (
    magnetic_axis_subgrid,
    xpoint_candidates,
)
from imas_ambix.latent.topology import N_XPOINT_SLOTS
from imas_ambix.worldmodel.equilibrium_labels import LCFS_ANGLES

# fp64 is mandatory: the boundary flux is a small difference of grid fluxes and
# the radii are read as sub-grid crossings — enable it before any array is traced.
jax.config.update("jax_enable_x64", True)

#: the evaluator's 8 fixed poloidal query angles as a device array (default
#: `angles` — a module singleton so the jitted signature has no call in defaults)
_DEFAULT_ANGLES = jnp.asarray(LCFS_ANGLES)

#: default wall boundary points — a single far-away point that is never reachable,
#: so a call without a wall polygon falls back to the flood binding level.
_NO_WALL = jnp.asarray([1.0e30])

#: default wall-node flux — a single NaN so, by broadcasting, every node falls
#: back to the bilinear grid read.  A caller with the exact campaign ``g_wall``
#: node flux passes a finite array aligned with ``wall_r``/``wall_z`` and each
#: finite node then reads its EXACT flux instead of the O(Δ²) bilerp.
_NO_WALL_PSI = jnp.asarray([jnp.nan])

#: static count of X-point candidate slots the stencil classifier fills (a
#: double-null plus spares); the emergent set is then trimmed to N_XPOINT_SLOTS.
_K_XCAND = 8

#: half-width (in ψ_N span units) of the flux band around the flood binding level
#: within which an X-point saddle is accepted as the binding candidate.
_X_FLUX_BAND = 0.05

#: ψ_N tolerance for an X-point to be reported as sitting ON the boundary ring.
_X_ON_RING_U = 0.02

__all__ = [
    "ConnectivityBoundary",
    "boundary_read_jax",
    "boundary_read",
    "boundary_read_batch",
]


# ---------------------------------------------------------------------------
# device primitives
# ---------------------------------------------------------------------------


def _bilerp(field: jnp.ndarray, rg: jnp.ndarray, zg: jnp.ndarray, r, z):
    """Bilinear-interpolate a ``(nz, nr)`` field at physical ``(r, z)`` (device)."""
    nr = rg.shape[0]
    nz = zg.shape[0]
    fr = jnp.clip(
        jnp.interp(r, rg, jnp.arange(nr, dtype=jnp.float64)), 0.0, nr - 1 - 1e-9
    )
    fz = jnp.clip(
        jnp.interp(z, zg, jnp.arange(nz, dtype=jnp.float64)), 0.0, nz - 1 - 1e-9
    )
    j0 = jnp.floor(fr).astype(jnp.int32)
    i0 = jnp.floor(fz).astype(jnp.int32)
    dj = fr - j0
    di = fz - i0
    f00 = field[i0, j0]
    f01 = field[i0, j0 + 1]
    f10 = field[i0 + 1, j0]
    f11 = field[i0 + 1, j0 + 1]
    return (
        f00 * (1 - di) * (1 - dj)
        + f01 * (1 - di) * dj
        + f10 * di * (1 - dj)
        + f11 * di * dj
    )


def _ray_radii(psi2d, rg, zg, ar, az, psi_axis, psi_lcfs, angles, n_ray):
    """LCFS radius at each poloidal angle by an outward ray-march from the axis.

    Marches ``n_ray`` fixed samples out to the grid diagonal, bilinear-interps ψ,
    and returns the first interpolated crossing of ψ_lcfs on each ray (NaN if a
    ray leaves the grid without crossing).  Fixed-shape, differentiable.
    """
    rmax = jnp.hypot(rg[-1] - rg[0], zg[-1] - zg[0])
    ss = jnp.linspace(0.0, rmax, n_ray)
    sign = jnp.sign(psi_lcfs - psi_axis)
    sign = jnp.where(sign == 0.0, 1.0, sign)
    target = (psi_lcfs - psi_axis) * sign
    idx = jnp.arange(n_ray)

    def one_angle(th):
        cr = jnp.cos(th)
        sr = jnp.sin(th)
        r = ar + ss * cr
        z = az + ss * sr
        vals = jax.vmap(lambda rr, zz: _bilerp(psi2d, rg, zg, rr, zz))(r, z)
        in_grid = (r >= rg[0]) & (r <= rg[-1]) & (z >= zg[0]) & (z <= zg[-1])
        # (ψ − ψ_axis)·sign grows from ~0 at the axis toward `target` at ψ_lcfs;
        # off-grid samples get −inf so they never register a crossing.
        f = jnp.where(in_grid, (vals - psi_axis) * sign, -jnp.inf)
        prev_f = jnp.concatenate([f[:1], f[:-1]])
        prev_s = jnp.concatenate([ss[:1], ss[:-1]])
        cross = (prev_f <= target) & (f >= target) & (idx > 0)
        has = jnp.any(cross)
        k = jnp.argmax(cross)  # first True
        fm, fp = f[k], prev_f[k]
        sm, sp = ss[k], prev_s[k]
        frac = jnp.where(fm == fp, 0.0, (target - fp) / (fm - fp))
        radius = sp + frac * (sm - sp)
        return jnp.where(has, radius, jnp.nan)

    return jax.vmap(one_angle)(angles)


# ---------------------------------------------------------------------------
# the connectivity boundary read (device kernel)
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnums=(6, 7, 8))
def boundary_read_jax(
    psi2d: jnp.ndarray,
    rg: jnp.ndarray,
    zg: jnp.ndarray,
    inside_limiter: jnp.ndarray,
    axis_r,
    axis_z,
    n_levels: int = 96,
    n_bisect: int = 18,
    n_ray: int = 512,
    angles: jnp.ndarray = _DEFAULT_ANGLES,
    lcfs_norm=0.999,
    wall_r: jnp.ndarray = _NO_WALL,
    wall_z: jnp.ndarray = _NO_WALL,
    wall_psi: jnp.ndarray = _NO_WALL_PSI,
) -> dict:
    """Connectivity LCFS read from ψ — the device-native ``lcfs_contour``.

    ``psi2d`` is ``(nz, nr)`` total poloidal flux; ``rg``/``zg`` the axis-ordered
    grid coordinates; ``inside_limiter`` the ``(nz, nr)`` boolean wall (raster)
    mask; ``(axis_r, axis_z)`` the read's axis (the current centroid, in metres).
    ``wall_r``/``wall_z`` are the wall boundary sample points (all wall units
    densified) — used for the SUB-GRID binding flux (see below); omit them to fall
    back to the cell-level flood binding.  ``wall_psi`` is the EXACT node flux
    (the campaign ``g_wall`` GEMM) aligned with those points; each finite entry
    reads its exact flux instead of the O(Δ²) bilinear grid read (the sentinel
    NaN default falls every node back to bilerp, so the read is unchanged).

    Returns a dict of fixed-shape arrays: ``found`` (bool — a valid closed
    axis-enclosing level exists), ``psi_axis``, ``psi_out``, ``psi_bnd`` (the
    binding / separatrix / wall flux), ``psi_lcfs`` (the reported ring flux),
    ``s_star`` (the binding level in [0, 1]), ``radii`` ``(len(angles),)`` LCFS
    radii about the axis [m], and ``n_core_cells``.  ``jit``/``vmap``/``grad``-safe.
    """
    nz = zg.shape[0]
    nr = rg.shape[0]
    n_iter = nr + nz  # flood-fill saturation count (≥ the region grid diameter)

    psi_axis = _bilerp(psi2d, rg, zg, axis_r, axis_z)
    edge = jnp.concatenate([psi2d[0, :], psi2d[-1, :], psi2d[:, 0], psi2d[:, -1]])
    psi_out = edge[jnp.argmax(jnp.abs(edge - psi_axis))]
    span = psi_out - psi_axis
    span_safe = jnp.where(jnp.abs(span) < 1e-30, 1e-30, span)
    u = (psi2d - psi_axis) / span_safe  # 0 at axis, 1 at the edge extreme

    ja = jnp.argmin(jnp.abs(rg - axis_r))
    ia = jnp.argmin(jnp.abs(zg - axis_z))
    seed = jnp.zeros((nz, nr), dtype=bool).at[ia, ja].set(True)
    seed_flat = ia * nr + ja

    # wall ring = in-wall cells adjacent to an out-of-wall cell (grid border
    # counts as out-of-wall).  The region "reaches the wall" when it touches this.
    border = (
        jnp.zeros((nz, nr), dtype=bool)
        .at[0, :]
        .set(True)
        .at[-1, :]
        .set(True)
        .at[:, 0]
        .set(True)
        .at[:, -1]
        .set(True)
    )
    outside = (~inside_limiter) | border
    wall_ring = _dilate4(outside) & inside_limiter

    # --- cell-level connectivity binding (the flood) --------------------------
    # A level is valid while the axis region stays clear of the wall; it turns
    # invalid the instant the region first reaches the wall.  Two escape tests
    # bracket the transition from opposite sides — an INNER test (region touches
    # the innermost in-wall shell, ~one cell inside) and an OUTER test (dilating
    # the region reaches a still-confined out-of-wall/edge cell, ~one cell outside).
    # Their mean is unbiased for an interior SADDLE binding (a diverted separatrix,
    # ψ_N=1), where the brackets straddle the saddle symmetrically; a LIMITED wall
    # tangency is then refined sub-grid below (the brackets straddle the wall, whose
    # sub-cell position is not the mean's implicit half-cell).
    def _alive_region(s):
        region = flood_fill_core((u <= s) & inside_limiter, seed, n_iter)
        alive = region.reshape(-1)[seed_flat] > 0.5
        return region, alive

    def valid_inner(s):
        region, alive = _alive_region(s)
        touch = jnp.sum(region * wall_ring.astype(region.dtype)) > 0.5
        return alive & (~touch)

    def valid_outer(s):
        region, alive = _alive_region(s)
        reach = _dilate4(region > 0.5) & outside & (u <= s)
        touch = jnp.sum(reach.astype(region.dtype)) > 0.5
        return alive & (~touch)

    s_grid = jnp.linspace(0.0, 1.0, n_levels + 1)[1:]  # (n_levels,) in (0, 1]
    idxs = jnp.arange(n_levels)

    def _bracket(valid_fn):
        vk = jax.vmap(valid_fn)(s_grid)
        last = jnp.max(jnp.where(vk, idxs, -1))
        lo0 = jnp.where(last >= 0, s_grid[jnp.clip(last, 0, n_levels - 1)], 0.0)
        hi0 = jnp.where(
            last < n_levels - 1, s_grid[jnp.clip(last + 1, 0, n_levels - 1)], 1.0
        )
        return last, lo0, hi0

    def _refine(valid_fn, lo0, hi0):
        def body(_i, carry):
            lo, hi = carry
            mid = 0.5 * (lo + hi)
            v = valid_fn(mid)
            return (jnp.where(v, mid, lo), jnp.where(v, hi, mid))

        lo, _hi = jax.lax.fori_loop(0, n_bisect, body, (lo0, hi0))
        return lo

    last_in, lo_in, hi_in = _bracket(valid_inner)
    _last_out, lo_out, hi_out = _bracket(valid_outer)
    found = last_in >= 0
    s_in = _refine(valid_inner, lo_in, hi_in)
    s_out = _refine(valid_outer, lo_out, hi_out)
    s_flood = 0.5 * (s_in + s_out)  # unbiased for an interior saddle (diverted)

    # The connectivity flood localises the binding to s_flood; the two remaining
    # sub-grid refinements (the wall tangency and the X-point saddle) are read at
    # that level and the binding is the CONFINED-MOST (nearest ψ_N to the axis) of
    # the two — one rule, no limited/diverted branch.  A wall tangency binds a
    # limited plasma; an X-point saddle binds a diverted one; whichever is reached
    # first (smaller ψ_N) closes the boundary, exactly as the outermost-closed
    # connectivity read does.

    # region at the flood binding, and a few-cell dilation of it (the "flood
    # rejoin" band) — a candidate null must sit in this band to be ON the axis
    # region's edge, so a same-flux null elsewhere in (or out of) the vessel is
    # rejected.
    region_at = flood_fill_core((u <= s_flood) & inside_limiter, seed, n_iter)
    reach = region_at > 0.5
    for _ in range(2):  # unrolled (static) — reach the wall polygon near the touch
        reach = _dilate4(reach)
    flood_adjacent = region_at > 0.5
    for _ in range(3):  # unrolled — the separatrix saddle sits at the region edge
        flood_adjacent = _dilate4(flood_adjacent)

    # --- sub-grid wall tangency ------------------------------------------------
    # The confined-most flux the closed surface reaches on the wall, read SUB-GRID
    # (the cell mean carries the sub-cell wall-position error a tangency is
    # sensitive to): the minimum interpolated ψ_N over the wall boundary points
    # adjacent to the axis region.  Computed for EVERY slice; on a diverted plasma
    # the reachable wall sits OUTSIDE the separatrix so this overshoots and loses
    # the confined-most min to the saddle below — no class switch needed.
    ar_idx = jnp.arange(nr, dtype=jnp.float64)
    az_idx = jnp.arange(nz, dtype=jnp.float64)
    wj = jnp.clip(jnp.round(jnp.interp(wall_r, rg, ar_idx)), 0, nr - 1)
    wi = jnp.clip(jnp.round(jnp.interp(wall_z, zg, az_idx)), 0, nz - 1)
    reachable = reach[wi.astype(jnp.int32), wj.astype(jnp.int32)]
    u_wall_bilerp = jax.vmap(
        lambda r_, z_: (_bilerp(psi2d, rg, zg, r_, z_) - psi_axis) / span_safe
    )(wall_r, wall_z)
    # exact node flux (campaign g_wall) where provided; bilerp elsewhere.  A
    # length-1 NaN sentinel broadcasts to "all bilerp"; a per-node finite array
    # reads each tangency EXACTLY (no O(Δ²) floor at the lean point).  Sanitise
    # the exact branch to a finite value BEFORE the select so the NaN sentinel
    # never poisons the gradient (the jnp.where NaN-VJP trap).
    wall_psi_finite = jnp.isfinite(wall_psi)
    wall_psi_safe = jnp.where(wall_psi_finite, wall_psi, psi_axis)
    u_wall_exact = (wall_psi_safe - psi_axis) / span_safe
    u_wall_pts = jnp.where(wall_psi_finite, u_wall_exact, u_wall_bilerp)
    u_wall = jnp.min(jnp.where(reachable, u_wall_pts, jnp.inf))
    u_wall_valid = jnp.any(reachable) & jnp.isfinite(u_wall)
    u_wall_c = jnp.where(u_wall_valid, u_wall, jnp.inf)

    # --- sub-grid X-point saddle at the binding (the diverted separatrix) -------
    # Classify saddles on the whole grid, then keep only those that (a) lie inside
    # the wall (xpoint_candidates ANDs inside_limiter), (b) sit in the flood-rejoin
    # band (spatially at the axis region's edge — rejects a same-flux X elsewhere),
    # (c) have flux within a band of the flood binding, and (d) are clear of the
    # axis (rejects a spurious near-axis null-space saddle — the device analogue of
    # the host min_axis_dist guard).  Refine sub-grid; the binding saddle is the
    # surviving candidate nearest the flood level.  Its flux closes the two-sided-
    # mean residual the diverted binding otherwise carries.
    d2_axis = (rg[None, :] - axis_r) ** 2 + (zg[:, None] - axis_z) ** 2
    min_axis_d = jnp.maximum(3.0 * (rg[1] - rg[0]), 0.05)
    x_mask = (
        (jnp.abs(u - s_flood) <= _X_FLUX_BAND)
        & (d2_axis >= min_axis_d**2)
        & flood_adjacent
    )
    xc = xpoint_candidates(psi2d, rg, zg, inside_limiter, _K_XCAND, extra_mask=x_mask)
    # Sanitise the masked-out (NaN) candidate flux to a finite value BEFORE any
    # arithmetic: a NaN numerator would poison the VJP of the /span_safe division
    # (0·NaN into span_safe, which depends on the axis), even though the value is
    # gated out downstream.  The x_valid flag still drops these slots.
    xc_valid = xc["valid"] & jnp.isfinite(xc["psi"])
    psi_x_safe = jnp.where(xc_valid, xc["psi"], psi_axis)
    u_x = (psi_x_safe - psi_axis) / span_safe  # (_K_XCAND,), finite everywhere
    x_valid = xc_valid
    x_key = jnp.where(x_valid, jnp.abs(u_x - s_flood), jnp.inf)
    kbind = jnp.argmin(x_key)
    x_bind_valid = x_valid[kbind] & jnp.isfinite(x_key[kbind])
    u_x_c = jnp.where(x_bind_valid, u_x[kbind], jnp.inf)

    # --- unified binding: confined-most of {wall tangency, X-point saddle} ------
    u_min = jnp.minimum(u_wall_c, u_x_c)
    s_star = jnp.where(jnp.isfinite(u_min), u_min, s_flood)

    # diverted iff the X-point saddle is the confined-most obstruction; the margin
    # is a SOFT continuous quantity (>0 diverted, <0 limited, ~0 marginal) so the
    # class is never a code-path switch.
    is_diverted = x_bind_valid & (u_x_c <= u_wall_c)
    class_margin = u_wall_c - u_x_c

    psi_bnd = psi_axis + s_star * span
    # Radii are read on the surface the ray-cast sits on, ALWAYS a hair inside the
    # separatrix (≤0.999·span): a ray cast at exactly ψ_bnd runs down an open
    # divertor leg through the X-point cusp (the closed-lobe host read never does),
    # so lcfs_norm is clamped for the ray while ψ_bnd itself reports the true
    # separatrix / wall flux the caller (e.g. the disc pushout, clip_legs) wants.
    ray_norm = jnp.minimum(lcfs_norm, 0.999)
    psi_lcfs = psi_axis + ray_norm * (psi_bnd - psi_axis)
    radii = _ray_radii(psi2d, rg, zg, axis_r, axis_z, psi_axis, psi_lcfs, angles, n_ray)

    confined_star = (u <= s_star) & inside_limiter
    region_star = flood_fill_core(confined_star, seed, n_iter)
    n_core = jnp.sum(region_star)

    # --- classify-after: sub-grid axis O-point + emergent X-set ----------------
    # The nulls are read AFTER the boundary, never as a prerequisite for ψ_N.  The
    # axis is the deepest O-point inside the confined region; the emergent X-set is
    # the saddles sitting ON the boundary (order-invariant, NaN-padded), the device
    # analogue of the host emergent_xpoints soft-margin rule.
    ax = magnetic_axis_subgrid(psi2d, rg, zg, inside_limiter, region=region_star)
    on_bound = x_valid & (jnp.abs(u_x - s_star) <= _X_ON_RING_U)
    ob_key = jnp.where(on_bound, jnp.abs(u_x - s_star), jnp.inf)
    order = jnp.argsort(ob_key)
    xr_s = xc["r"][order]
    xz_s = xc["z"][order]
    ob_s = on_bound[order]
    take = ob_s[:N_XPOINT_SLOTS]
    xset = jnp.stack(
        [
            jnp.where(take, xr_s[:N_XPOINT_SLOTS], jnp.nan),
            jnp.where(take, xz_s[:N_XPOINT_SLOTS], jnp.nan),
        ],
        axis=1,
    )  # (N_XPOINT_SLOTS, 2)

    return {
        "found": found,
        "psi_axis": psi_axis,
        "psi_out": psi_out,
        "psi_bnd": jnp.where(found, psi_bnd, jnp.nan),
        "psi_lcfs": jnp.where(found, psi_lcfs, jnp.nan),
        "s_star": jnp.where(found, s_star, jnp.nan),
        "radii": jnp.where(found, radii, jnp.nan),
        "n_core_cells": n_core,
        # classify-after diagnostics (never feed ψ_N)
        "axis_r": jnp.where(ax["found"], ax["r"], jnp.nan),
        "axis_z": jnp.where(ax["found"], ax["z"], jnp.nan),
        "axis_psi_sub": jnp.where(ax["found"], ax["psi"], jnp.nan),
        "xset": xset,
        "is_diverted": is_diverted,
        "class_margin": class_margin,
        "u_wall": u_wall_c,
        "u_xpoint": u_x_c,
    }


# ---------------------------------------------------------------------------
# host adapters
# ---------------------------------------------------------------------------


def _densify_wall(grid, m: int = 720):
    """Wall surface nodes ``(wall_r, wall_z)`` for the SUB-GRID binding flux.

    Prefers the grid's precomputed multi-unit nodes (``wall_r``/``wall_z`` built
    at ~Δ/2 over every wall unit — vessel, tiles, movable limiters), so an
    arbitrary machine wall is data, not a code path.  Falls back to resampling a
    single ``limiter_r``/``limiter_z`` loop to ``m`` points (a bare test grid or
    an older grid), and to the single far-away no-wall point when neither exists
    (the read then uses the cell-level flood binding).
    """
    import numpy as np  # noqa: PLC0415

    gwr = getattr(grid, "wall_r", None)
    gwz = getattr(grid, "wall_z", None)
    if gwr is not None and gwz is not None and len(np.asarray(gwr)) >= 1:
        return np.asarray(gwr, dtype=np.float64), np.asarray(gwz, dtype=np.float64)

    lr = getattr(grid, "limiter_r", None)
    lz = getattr(grid, "limiter_z", None)
    if lr is None or lz is None or len(np.asarray(lr)) < 2:
        return np.array([1.0e30]), np.array([1.0e30])
    lr = np.asarray(lr, dtype=np.float64)
    lz = np.asarray(lz, dtype=np.float64)
    # close the loop, cumulative arc length, resample uniformly to m points
    rr = np.append(lr, lr[0])
    zz = np.append(lz, lz[0])
    seg = np.hypot(np.diff(rr), np.diff(zz))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total <= 0.0:
        return np.array([1.0e30]), np.array([1.0e30])
    q = np.linspace(0.0, total, m, endpoint=False)
    return np.interp(q, s, rr), np.interp(q, s, zz)


@dataclass
class ConnectivityBoundary:
    """Host-side result of :func:`boundary_read` (mirrors ``LcfsContour`` fields)."""

    found: bool
    psi_bnd: float
    psi_lcfs: float
    psi_axis: float
    radii: object  # np.ndarray (len(angles),) [m]
    s_star: float
    n_core_cells: int
    # classify-after diagnostics (read AFTER the boundary; never feed ψ_N)
    axis: tuple[float, float]  # sub-grid magnetic axis (O-point) [m]
    axis_psi: float  # sub-grid axis flux [Wb]
    xset: object  # np.ndarray (N_XPOINT_SLOTS, 2) NaN-padded emergent X-points [m]
    is_diverted: bool
    class_margin: float  # u_wall − u_xpoint (>0 diverted, <0 limited, ~0 marginal)


def boundary_read(
    psi2d,
    grid,
    axis: tuple[float, float],
    *,
    n_levels: int = 96,
    n_bisect: int = 18,
    n_ray: int = 512,
    angles=LCFS_ANGLES,
    lcfs_norm: float = 0.999,
    wall_psi=None,
) -> ConnectivityBoundary:
    """Host adapter: run :func:`boundary_read_jax` on one slice, return numpy.

    ``grid`` is an :class:`~imas_ambix.latent.gs_solve.EquilibriumGrid` (supplies
    ``rg``/``zg``/``inside_limiter`` and the multi-unit wall nodes).  ``wall_psi``
    is the exact node flux (``grid.wall_flux(i_pf, i_cell)``) aligned with the
    grid's wall nodes; pass it to read the tangency exactly (the campaign
    ``g_wall`` GEMM) instead of bilinear off the grid.  ``lcfs_norm=1.0`` reports
    the ring AT the separatrix (the ``lcfs_contour(clip_legs=True)`` convention
    used by the disc pushout); the 0.999 default reads a hair inside.
    """
    import numpy as np  # noqa: PLC0415

    wall_r, wall_z = _densify_wall(grid)
    # ONE hard error per solve: the flood seed (the axis cell) must be occupiable
    # — if it lands in wall material (or outside the vessel) the connectivity read
    # has no plasma to grow.  (The read itself is fully differentiable; this is a
    # host-side precondition, not a branch inside the device kernel.)
    inside = np.asarray(grid.inside_limiter, dtype=bool)
    rg_np = np.asarray(grid.rg, dtype=np.float64)
    zg_np = np.asarray(grid.zg, dtype=np.float64)
    ia = int(np.argmin(np.abs(zg_np - float(axis[1]))))
    ja = int(np.argmin(np.abs(rg_np - float(axis[0]))))
    if not inside[ia, ja]:
        raise ValueError(
            f"flood seed (axis cell R={axis[0]:.4f}, Z={axis[1]:.4f} → grid "
            f"[{ia}, {ja}]) lies in wall material / outside the vessel — no "
            "axis-connected plasma to grow"
        )
    if wall_psi is None:
        wpsi = _NO_WALL_PSI
    else:
        wpsi = jnp.asarray(np.asarray(wall_psi, dtype=np.float64))
    out = boundary_read_jax(
        jnp.asarray(np.asarray(psi2d, dtype=np.float64)),
        jnp.asarray(rg_np),
        jnp.asarray(zg_np),
        jnp.asarray(inside),
        jnp.asarray(float(axis[0])),
        jnp.asarray(float(axis[1])),
        int(n_levels),
        int(n_bisect),
        int(n_ray),
        jnp.asarray(np.asarray(angles, dtype=np.float64)),
        jnp.asarray(float(lcfs_norm)),
        jnp.asarray(wall_r),
        jnp.asarray(wall_z),
        wpsi,
    )
    return ConnectivityBoundary(
        found=bool(out["found"]),
        psi_bnd=float(out["psi_bnd"]),
        psi_lcfs=float(out["psi_lcfs"]),
        psi_axis=float(out["psi_axis"]),
        radii=np.asarray(out["radii"], dtype=np.float64),
        s_star=float(out["s_star"]),
        n_core_cells=int(out["n_core_cells"]),
        axis=(float(out["axis_r"]), float(out["axis_z"])),
        axis_psi=float(out["axis_psi_sub"]),
        xset=np.asarray(out["xset"], dtype=np.float64),
        is_diverted=bool(out["is_diverted"]),
        class_margin=float(out["class_margin"]),
    )


def boundary_read_batch(
    psi_stack,
    grid,
    axes,
    *,
    n_levels: int = 96,
    n_bisect: int = 18,
    n_ray: int = 512,
    angles=LCFS_ANGLES,
    lcfs_norm: float = 0.999,
    wall_psi=None,
) -> dict:
    """Batched read over ``(B, nz, nr)`` ψ fields sharing one grid — a single vmap.

    ``axes`` is ``(B, 2)`` (R, Z).  ``wall_psi`` is an optional ``(B, n_node)``
    exact node flux (per-slice ``grid.wall_flux``) aligned with the grid's wall
    nodes; omit it for the bilinear read.  Proves the fixed-shape / on-device
    batch the corpus labeller needs: one ``jax.vmap``, no host loop, no per-slice
    contour extraction.  Returns a dict of stacked device arrays.
    """
    import numpy as np  # noqa: PLC0415

    rg = jnp.asarray(np.asarray(grid.rg, dtype=np.float64))
    zg = jnp.asarray(np.asarray(grid.zg, dtype=np.float64))
    inside = jnp.asarray(np.asarray(grid.inside_limiter, dtype=bool))
    ang = jnp.asarray(np.asarray(angles, dtype=np.float64))
    ps = jnp.asarray(np.asarray(psi_stack, dtype=np.float64))
    ax = jnp.asarray(np.asarray(axes, dtype=np.float64))
    wall_r, wall_z = _densify_wall(grid)
    wr = jnp.asarray(wall_r)
    wz = jnp.asarray(wall_z)
    if wall_psi is None:
        wpsi = jnp.broadcast_to(_NO_WALL_PSI, (ps.shape[0], 1))
    else:
        wpsi = jnp.asarray(np.asarray(wall_psi, dtype=np.float64))

    def one(psi2d, axis, wp):
        return boundary_read_jax(
            psi2d,
            rg,
            zg,
            inside,
            axis[0],
            axis[1],
            int(n_levels),
            int(n_bisect),
            int(n_ray),
            ang,
            jnp.asarray(float(lcfs_norm)),
            wr,
            wz,
            wp,
        )

    return jax.vmap(one)(ps, ax, wpsi)
