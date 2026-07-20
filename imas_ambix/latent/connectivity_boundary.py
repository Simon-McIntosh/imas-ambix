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

* **The binding level = the connectivity change.**  A level is *valid* while the
  axis region stays clear of the wall boundary; it becomes invalid the instant
  the region first reaches the wall — either by a flux surface going tangent to
  the wall (LIMITED) or by opening through an X-point so the scrape-off connects
  to the divertor/wall (DIVERTED).  This predicate is monotone in ``s`` (the
  confined set only grows), so ψ_bnd is the largest valid level, found by a
  fixed coarse sweep bracketing the transition + a fixed-count bisection.  This
  is exactly the caution the plan pins: the connectivity-change level IS ψ_bnd
  (no hysteresis, no per-iteration contour cache).

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

Sub-grid note: ψ_bnd is resolved to the flux of the binding grid cell (O(Δψ)
at the wall-tangency / saddle, the discretisation floor the host contour read
also carries); the LCFS radii are sub-grid (interpolated ray crossing).  The
sub-grid saddle/axis position that would polish the scalar ψ_bnd is the nova
stencil refinement (a separate rung), deliberately not folded into the boundary
sweep.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp

from imas_ambix.latent.flux_surface_connectivity import _dilate4, flood_fill_core
from imas_ambix.worldmodel.equilibrium_labels import LCFS_ANGLES

# fp64 is mandatory: the boundary flux is a small difference of grid fluxes and
# the radii are read as sub-grid crossings — enable it before any array is traced.
jax.config.update("jax_enable_x64", True)

#: the evaluator's 8 fixed poloidal query angles as a device array (default
#: `angles` — a module singleton so the jitted signature has no call in defaults)
_DEFAULT_ANGLES = jnp.asarray(LCFS_ANGLES)

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
) -> dict:
    """Connectivity LCFS read from ψ — the device-native ``lcfs_contour``.

    ``psi2d`` is ``(nz, nr)`` total poloidal flux; ``rg``/``zg`` the axis-ordered
    grid coordinates; ``inside_limiter`` the ``(nz, nr)`` boolean wall (raster)
    mask; ``(axis_r, axis_z)`` the read's axis (the current centroid, in metres).

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

    def valid(s):
        confined = (u <= s) & inside_limiter
        region = flood_fill_core(confined, seed, n_iter)  # float 0/1
        alive = region.reshape(-1)[seed_flat] > 0.5
        touches = jnp.sum(region * wall_ring.astype(region.dtype)) > 0.5
        return alive & (~touches)

    # coarse sweep: the valid predicate is True on a single contiguous band
    # [u_seed, s*]; the largest valid grid level brackets the transition s*.
    s_grid = jnp.linspace(0.0, 1.0, n_levels + 1)[1:]  # (n_levels,) in (0, 1]
    vk = jax.vmap(valid)(s_grid)
    idxs = jnp.arange(n_levels)
    last = jnp.max(jnp.where(vk, idxs, -1))
    found = last >= 0

    lo0 = jnp.where(found, s_grid[jnp.clip(last, 0, n_levels - 1)], 0.0)
    hi0 = jnp.where(
        last < n_levels - 1, s_grid[jnp.clip(last + 1, 0, n_levels - 1)], 1.0
    )

    def body(_i, carry):
        lo, hi = carry
        mid = 0.5 * (lo + hi)
        v = valid(mid)
        return (jnp.where(v, mid, lo), jnp.where(v, hi, mid))

    lo, hi = jax.lax.fori_loop(0, n_bisect, body, (lo0, hi0))
    s_star = lo
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

    return {
        "found": found,
        "psi_axis": psi_axis,
        "psi_out": psi_out,
        "psi_bnd": jnp.where(found, psi_bnd, jnp.nan),
        "psi_lcfs": jnp.where(found, psi_lcfs, jnp.nan),
        "s_star": jnp.where(found, s_star, jnp.nan),
        "radii": jnp.where(found, radii, jnp.nan),
        "n_core_cells": n_core,
    }


# ---------------------------------------------------------------------------
# host adapters
# ---------------------------------------------------------------------------


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
) -> ConnectivityBoundary:
    """Host adapter: run :func:`boundary_read_jax` on one slice, return numpy.

    ``grid`` is an :class:`~imas_ambix.latent.gs_solve.EquilibriumGrid` (supplies
    ``rg``/``zg``/``inside_limiter``).  ``lcfs_norm=1.0`` reports the ring AT the
    separatrix (the ``lcfs_contour(clip_legs=True)`` convention used by the disc
    pushout); the 0.999 default reads a hair inside (the plain ``lcfs_contour``).
    """
    import numpy as np  # noqa: PLC0415

    out = boundary_read_jax(
        jnp.asarray(np.asarray(psi2d, dtype=np.float64)),
        jnp.asarray(np.asarray(grid.rg, dtype=np.float64)),
        jnp.asarray(np.asarray(grid.zg, dtype=np.float64)),
        jnp.asarray(np.asarray(grid.inside_limiter, dtype=bool)),
        jnp.asarray(float(axis[0])),
        jnp.asarray(float(axis[1])),
        int(n_levels),
        int(n_bisect),
        int(n_ray),
        jnp.asarray(np.asarray(angles, dtype=np.float64)),
        jnp.asarray(float(lcfs_norm)),
    )
    return ConnectivityBoundary(
        found=bool(out["found"]),
        psi_bnd=float(out["psi_bnd"]),
        psi_lcfs=float(out["psi_lcfs"]),
        psi_axis=float(out["psi_axis"]),
        radii=np.asarray(out["radii"], dtype=np.float64),
        s_star=float(out["s_star"]),
        n_core_cells=int(out["n_core_cells"]),
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
) -> dict:
    """Batched read over ``(B, nz, nr)`` ψ fields sharing one grid — a single vmap.

    ``axes`` is ``(B, 2)`` (R, Z).  Proves the fixed-shape / on-device batch the
    corpus labeller needs: one ``jax.vmap``, no host loop, no per-slice contour
    extraction.  Returns a dict of stacked device arrays.
    """
    import numpy as np  # noqa: PLC0415

    rg = jnp.asarray(np.asarray(grid.rg, dtype=np.float64))
    zg = jnp.asarray(np.asarray(grid.zg, dtype=np.float64))
    inside = jnp.asarray(np.asarray(grid.inside_limiter, dtype=bool))
    ang = jnp.asarray(np.asarray(angles, dtype=np.float64))
    ps = jnp.asarray(np.asarray(psi_stack, dtype=np.float64))
    ax = jnp.asarray(np.asarray(axes, dtype=np.float64))

    def one(psi2d, axis):
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
        )

    return jax.vmap(one)(ps, ax)
