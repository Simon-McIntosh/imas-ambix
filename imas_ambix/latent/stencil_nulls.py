"""Sub-grid critical-point locator from local stencils (JAX, device-native).

The *classify-after* half of the connectivity topology read.  §2
(:mod:`imas_ambix.latent.connectivity_boundary`) resolves the boundary by
connectivity; this module reads the sub-grid null POSITIONS the boundary read
leaves out — the magnetic axis (O-point) and the X-point(s) — and the sub-grid
saddle FLUX that lets the boundary binding be unified without a limited/diverted
branch.  The nulls are read AFTER the boundary, never as a prerequisite for ψ_N.

Method (a fixed-shape reduction over the whole grid — no host round-trip, no
``argwhere``, no variable-count nulls):

* **Classify every vertex.**  Around each interior vertex the eight neighbours
  are traversed as a closed ring; the number of sign changes of ψ − ψ_centre
  around that ring is a topological invariant (Kuijper, *On detecting all saddle
  points in 2D images*): **0 → O-point** (extremum), **2 → regular**, **4 →
  X-point** (saddle).  This is the whole-grid stencil classifier — a handful of
  shifted-array comparisons and one sum, so "find all stationary points" is
  cheap and differentiable, and it fixes the array shape once.

* **Select with a fixed-shape proximity mask.**  We already know where the
  boundary is (§2), so the relevant nulls are picked by ANDing the O/X candidate
  masks with a boundary-derived mask — the axis-connected flood region and the
  in-wall raster for the O-point, a flux-proximity band around the binding level
  for the saddle.  The mask is an elementwise ``(nz, nr)`` boolean, NOT a
  variable-length gather, so the read stays ``jit`` / ``vmap`` / ``grad``-safe.
  It is the device-native analogue of the host read's limiter filter,
  ``exclude_near_points`` (drop coil O-points) and ``min_axis_dist`` (drop
  spurious near-axis saddles).

* **Refine sub-grid.**  A selected vertex's 3×3 neighbourhood is fitted with a
  biquadratic surface ``a·R² + b·Z² + c·R + d·Z + e·R·Z + f``; the stationary
  point (∇ = 0), its flux, and its type (Hessian sign) are read off in closed
  form — a proven, differentiable ``jax.lstsq`` path.  The variable number of
  X-candidates is handled by the static-count idiom (``jnp.where(size=K)`` +
  NaN-pad), so shapes never depend on the data.

The biquadratic null refinement follows the field-null algorithm in the ``nova``
library (``nova.jax.select``); the small pure functions are reproduced here so
this read carries no cross-package dependency and stays a self-contained device
kernel.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

# fp64 is mandatory: the saddle flux is a small difference of grid fluxes and the
# positions are sub-grid — enable it before any array is traced.
jax.config.update("jax_enable_x64", True)

__all__ = [
    "ring_sign_changes",
    "subnull",
    "magnetic_axis_subgrid",
    "xpoint_candidates",
]

# ---------------------------------------------------------------------------
# biquadratic null refinement (ported from nova.jax.select — pure, differentiable)
# ---------------------------------------------------------------------------


@jax.jit
def _quadratic_surface(r, z, psi):
    """Least-squares biquadratic coefficients [a, b, c, d, e, f] for ψ(R, Z).

    ψ ≈ a·R² + b·Z² + c·R + d·Z + e·R·Z + f, fitted over a cluster of points.
    """
    amat = jnp.column_stack((r**2, z**2, r, z, r * z, jnp.ones_like(r)))
    return jnp.linalg.lstsq(amat, psi)[0]


@jax.jit
def _null_coordinate(coef):
    """Stationary point (∇ψ = 0) of the biquadratic surface."""
    a, b, c, d, e, _f = coef
    root = 4 * a * b - e**2
    root = jnp.where(jnp.abs(root) < 1e-30, jnp.sign(root) * 1e-30 + 1e-30, root)
    r0 = (e * d - 2 * b * c) / root
    z0 = (e * c - 2 * a * d) / root
    return r0, z0


@jax.jit
def _null_value(coef, r0, z0):
    """Flux of the biquadratic surface at (r0, z0)."""
    return jnp.array([r0**2, z0**2, r0, z0, r0 * z0, 1.0]) @ coef


@jax.jit
def _null_type(coef, atol=1e-12):
    """Null type from the Hessian: 0 saddle (X), ±1 extremum (O), NaN degenerate."""
    a, b, _c, _d, e, _f = coef
    root = 4 * a * b - e**2
    condlist = [
        jnp.abs(root) < atol,
        root < 0,
        (a > 0) & (b > 0),
        (a < 0) & (b < 0),
    ]
    choicelist = [jnp.nan, 0.0, -1.0, 1.0]
    return jnp.select(condlist, choicelist, default=jnp.nan)


@jax.jit
def subnull(r_cluster, z_cluster, psi_cluster):
    """Sub-grid null (R, Z, ψ, type) from a cluster of (R, Z, ψ) samples.

    ``type`` is 0 for a saddle (X-point), ±1 for an extremum (O-point), NaN for a
    degenerate/planar fit.  Fully differentiable in ``psi_cluster``.
    """
    coef = _quadratic_surface(r_cluster, z_cluster, psi_cluster)
    r0, z0 = _null_coordinate(coef)
    psi0 = _null_value(coef, r0, z0)
    ntype = _null_type(coef)
    return jnp.array([r0, z0, psi0, ntype])


# ---------------------------------------------------------------------------
# whole-grid stencil classifier
# ---------------------------------------------------------------------------

# eight neighbours in cyclic angular order (E, NE, N, NW, W, SW, S, SE) as
# (dz, dr) index offsets — consecutive neighbours are spatially adjacent, so the
# traversal is a genuine closed ring.
_RING = (
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
)


def _shift(field, dz, dr):
    """``out[i, j] = field[i + dz, j + dr]`` (roll; border handled by the caller)."""
    return jnp.roll(field, shift=(-dz, -dr), axis=(0, 1))


def _dilate4(mask):
    """4-neighbour boolean dilation, non-wrapping (border does not fold)."""
    up = jnp.zeros_like(mask).at[1:, :].set(mask[:-1, :])
    dn = jnp.zeros_like(mask).at[:-1, :].set(mask[1:, :])
    lt = jnp.zeros_like(mask).at[:, 1:].set(mask[:, :-1])
    rt = jnp.zeros_like(mask).at[:, :-1].set(mask[:, 1:])
    return mask | up | dn | lt | rt


@jax.jit
def ring_sign_changes(psi):
    """Sign-change count of ψ − ψ_centre around each vertex's 8-neighbour ring.

    Returns a ``(nz, nr)`` integer field: 0 → O-point, 2 → regular, 4 → X-point.
    The one-cell border is set to −1 (no full ring), so border vertices are never
    O/X candidates.  A fixed-shape, differentiable-free (pure comparison) reduction.
    """
    nz, nr = psi.shape
    ring = jnp.stack([_shift(psi, dz, dr) > psi for dz, dr in _RING], axis=0)
    changes = jnp.zeros((nz, nr), dtype=jnp.int32)
    for k in range(8):
        changes = changes + (ring[k] != ring[(k + 1) % 8]).astype(jnp.int32)
    interior = jnp.zeros((nz, nr), dtype=bool).at[1:-1, 1:-1].set(True)
    return jnp.where(interior, changes, -1)


# ---------------------------------------------------------------------------
# sub-grid 3×3 refinement at chosen vertices
# ---------------------------------------------------------------------------

# 3×3 neighbourhood offsets (dz, dr) — the cluster for the biquadratic fit.
_BLOCK = tuple((dz, dr) for dz in (-1, 0, 1) for dr in (-1, 0, 1))


def _refine_at(psi, rg, zg, ia, ja):
    """Sub-grid null (R, Z, ψ, type) from the 3×3 block centred on vertex (ia, ja)."""
    nz, nr = psi.shape
    rows = jnp.clip(jnp.array([ia + dz for dz, _ in _BLOCK]), 0, nz - 1)
    cols = jnp.clip(jnp.array([ja + dr for _, dr in _BLOCK]), 0, nr - 1)
    r_c = rg[cols]
    z_c = zg[rows]
    psi_c = psi[rows, cols]
    return subnull(r_c, z_c, psi_c)


# ---------------------------------------------------------------------------
# the axis O-point (sub-grid)
# ---------------------------------------------------------------------------


def magnetic_axis_subgrid(psi, rg, zg, inside_limiter, region=None):
    """Sub-grid magnetic axis = the deepest in-wall O-point (flux extremum).

    Matches the host :func:`imas_ambix.latent.topology.magnetic_axis` selection —
    the interior O-point whose ψ is furthest from the domain-edge median, taken
    strictly inside the wall (an out-of-vessel coil O-point is never the axis).
    An optional ``region`` mask (the §2 axis-connected flood) further restricts
    the search to the plasma, the device analogue of ``exclude_near_points``.

    Returns ``{r, z, psi, ntype, found}`` — ``found`` is False when no in-wall
    O-point exists.  Differentiable in ``psi`` (the vertex pick is a stop-grad
    integer index; the gradient flows through the sub-grid biquadratic fit).
    """
    nz, nr = psi.shape
    counts = ring_sign_changes(psi)
    is_o = counts == 0
    cand = is_o & inside_limiter
    if region is not None:
        cand = cand & (region > 0.5)

    edge = jnp.concatenate([psi[0, :], psi[-1, :], psi[:, 0], psi[:, -1]])
    psi_edge = jnp.median(edge)
    score = jnp.where(cand, jnp.abs(psi - psi_edge), -jnp.inf)
    flat = jnp.argmax(score)
    ia = flat // nr
    ja = flat % nr
    found = jnp.max(score) > -jnp.inf

    sub = _refine_at(psi, rg, zg, ia, ja)
    return {
        "r": jnp.where(found, sub[0], jnp.nan),
        "z": jnp.where(found, sub[1], jnp.nan),
        "psi": jnp.where(found, sub[2], jnp.nan),
        "ntype": jnp.where(found, sub[3], jnp.nan),
        "found": found,
    }


# ---------------------------------------------------------------------------
# X-point candidates (sub-grid, static-count)
# ---------------------------------------------------------------------------


def xpoint_candidates(
    psi, rg, zg, inside_limiter, k_slots=6, extra_mask=None, material_dilate=1
):
    """Up to ``k_slots`` sub-grid X-point candidates (static-count select).

    Vertices classified as saddles (4 sign changes) inside the wall are selected
    by the static-count idiom (``jnp.where(size=k_slots)`` + NaN-pad) and refined
    sub-grid.  ``extra_mask`` (a ``(nz, nr)`` boolean) further restricts the
    candidates — e.g. a flux-proximity band around the binding level, to drop
    spurious near-axis null-space saddles.

    ``material_dilate`` (default 1) dilates the in-wall gate by that many cells
    so a BINDING saddle sitting within a cell of a tile/limiter (whose cell the
    supercover raster marks material, one cell outside ``inside_limiter``) is not
    masked out — the sub-grid saddle still refines to its true position, and the
    downstream flux-band / flood-adjacency filter rejects genuinely off-surface
    nulls.  Set 0 to gate strictly inside the raster.

    Returns a dict of ``(k_slots,)`` arrays: ``r``, ``z``, ``psi``, ``ntype`` and
    ``valid`` (a real saddle occupies the slot).  Padded slots are NaN / False.
    """
    nz, nr = psi.shape
    counts = ring_sign_changes(psi)
    gate = inside_limiter
    for _ in range(int(material_dilate)):
        gate = _dilate4(gate)
    cand = (counts == 4) & gate
    if extra_mask is not None:
        cand = cand & extra_mask
    idx = jnp.where(cand.reshape(-1), size=k_slots, fill_value=-1)[0]
    slot_valid = idx >= 0
    ia = jnp.clip(idx // nr, 0, nz - 1)
    ja = jnp.clip(idx % nr, 0, nr - 1)

    subs = jax.vmap(lambda i, j: _refine_at(psi, rg, zg, i, j))(ia, ja)
    # a slot is a real X-point when its vertex was selected AND the biquadratic
    # confirms a saddle (type 0)
    is_saddle = jnp.abs(subs[:, 3]) < 0.5
    valid = slot_valid & is_saddle
    # Freeze the gradient on padded / degenerate slots.  A near-planar 3×3
    # cluster (which the one-cell material dilation can admit in a flat flux tail)
    # gives an ill-conditioned lstsq whose VJP is huge/∞; even though the value is
    # masked out downstream, a masked NaN/∞ would still poison the gradient (0·∞).
    # stop_gradient on the unselected slots removes the poison while leaving the
    # selected saddles fully differentiable.
    subs = jnp.where(valid[:, None], subs, jax.lax.stop_gradient(subs))
    nan = jnp.nan * jnp.ones_like(subs[:, 0])
    return {
        "r": jnp.where(valid, subs[:, 0], nan),
        "z": jnp.where(valid, subs[:, 1], nan),
        "psi": jnp.where(valid, subs[:, 2], nan),
        "ntype": jnp.where(valid, subs[:, 3], nan),
        "valid": valid,
    }
