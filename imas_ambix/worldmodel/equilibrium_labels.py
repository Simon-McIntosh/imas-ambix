"""Per-camera-frame plasma-geometry labels from the L2 equilibrium store.

EVALUATOR-ONLY (binding firewall)
---------------------------------
The L2 equilibrium reconstruction is a **third-party label** used to *score*
the camera world model — it is NEVER a world-model input or conditioning
signal.  Nothing in this module may be imported into the WM training path:
it only *consumes* the equilibrium store and *produces* evaluator targets.
Keep it physically downstream of the model (the feasibility oracle and the
geometry probe import it; the model does not).

What it produces
----------------
:func:`load_equilibrium_geometry` interpolates the L2 equilibrium geometry
onto a set of camera ``frame_times`` (seconds) and returns, per frame, a
**14-D target** in METRES plus a finite mask::

    index  name        meaning (metres)
    -----  ----------  --------------------------------------------------
      0    axis_R      magnetic-axis major radius   (fixed regression)
      1    axis_Z      magnetic-axis height         (fixed regression)
      2    xpt0_R      X-point NULL-SET slot 0, R   (order-invariant set)
      3    xpt0_Z      X-point null-set slot 0, Z
      4    xpt1_R      X-point null-set slot 1, R
      5    xpt1_Z      X-point null-set slot 1, Z
    6..13  lcfs_r[k]   LCFS control-point RADIUS at 8 fixed poloidal
                       angles θ_k about the magnetic axis (k = 0..7),
                       θ_k = 2π k / 8 measured CCW from the outboard
                       midplane (+R direction) in the (R, Z) poloidal plane.

The 8 LCFS radii are the ray-cast distance (metres) from the magnetic axis
to the last-closed-flux-surface contour along each of 8 equally-spaced
poloidal angles — a fixed-dimension, rotation-stable, topology-ROBUST
parameterisation of the boundary shape that does not depend on the (variable,
NaN-padded) number of raw boundary points the store carries.

X-point as an ORDER-INVARIANT NULL SET (topology-agnostic)
----------------------------------------------------------
The store carries up to two real nulls per slice in ``x_point_r/z`` (``-9.99``
sentinel for an undefined null).  Earlier designs imposed a topology assumption
— a single "primary" null (bimodal across a divertor switch, a ~2.4 m flip that
linear interpolation drew through Z~0) or a sign-of-Z "lower/upper" split (which
mislabels double-null-near-one-divertor, snowflake, X-divertor, super-X).  Both
are dropped.  Instead the X-point label is the **SET of ≤2 real nulls** written
into two UNORDERED slots — ``(xpt0, xpt1)`` — with NO ordering, NO sign-of-Z
assignment, NO ψ public/private filter, and NO cross-frame continuity tracking.
Because the oracle predicts the **window-centre** geometry, the set is taken
from the **nearest-native centre slice** (NO interpolation of set members across
slices — that is what removes the flip/interp problem entirely).  Presence /
count (0 / 1 / 2 nulls) is encoded by the per-slot finite mask: a present slot
has finite ``(R, Z)``; an absent slot is NaN/masked.  The downstream probe's
X-point head is trained with a PERMUTATION-INVARIANT matched loss, so the label
is identical under swapping the two slots — that is the whole point.

Coarse in-vessel sanity (NOT a public/private discriminator)
------------------------------------------------------------
A reconstructed null is kept only if it is finite and inside a COARSE MAST
limiter bounding box (:data:`XPOINT_VESSEL_R_RANGE` / :data:`XPOINT_VESSEL_Z_ABS`).
This rejects a NaN / sentinel / wildly-out-of-vessel reconstruction artefact —
it is explicitly **NOT** a public/private flux-region discriminator (the flux
map is multimodal; a private-region null can sit at ψ ≈ ψ_LCFS, so a ψ-proximity
test cannot separate boundary from private nulls — that classification is not
attempted here).

Masking, not imputation
-----------------------
The L2 equilibrium is only defined while the plasma exists: early ramp-up and
late ramp-down slices carry **NaN** axis/LCFS values.  A camera frame whose
interpolated target touches an undefined equilibrium slice is **masked**
(``finite_mask[f, c] = False``) — never imputed.  Axis + LCFS are continuous and
LINEARLY interpolated; the X-point null slots are NEAREST-native (window-centre,
no member interpolation).  Each component has its own per-frame mask.

Store facts (verified on the L2 mirror, 2026-06-24)
---------------------------------------------------
``/work/projects/imas_gpu/mast/level2/shots/<id>.zarr`` group ``equilibrium``
is **Zarr V3** — open with ``zarr.open_group(path, mode='r')`` (NOT
imas-python; the corpus is Zarr, not IMAS HDF5).  Keys (time is the LAST axis
on >= 2-D fields):

  - ``magnetic_axis_r``/``magnetic_axis_z``  ``(nt,)``  metres, NaN when off
  - ``x_point_r``/``x_point_z``              ``(2, nt)``  metres when present;
    ``-9.99`` is the sentinel for an undefined null.  Some stores omit both
    arrays; that represents an unavailable null set, so both output slots stay
    NaN/masked while the independent axis and LCFS components remain usable.
    MAST runs **double-null most of the time** (e.g. shot 18504: 79/124 slices
    carry two real nulls) and switches topology.  No ordering / sign-of-Z
    meaning is read from the two rows — the valid nulls are taken as an
    ORDER-INVARIANT SET (:func:`xpoint_null_set`); a swap of the two rows is
    identical under the downstream permutation-invariant loss.
  - ``lcfs_r``/``lcfs_z``                    ``(n_bdy, nt)``  metres, NaN-padded
  - ``n_boundary_coords``                    ``(n_bdy,)``  valid LCFS-point
    count per time slice (the contour uses the first ``n_boundary_coords[i]``
    points of column ``i``)
  - ``time``                                 ``(nt,)``  seconds, ~200 Hz

This module is pure / IO-light: one Zarr open + numpy interpolation.  No GPU,
no torch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# --- Store location ---------------------------------------------------------

#: Default L2 mirror root carrying ``<shot>.zarr/equilibrium`` (Zarr V3).
DEFAULT_LEVEL2_ROOT = Path("/work/projects/imas_gpu/mast/level2/shots")

#: Sentinel the L2 store writes for an undefined X-point coordinate (metres).
#: Any coordinate at or below this magnitude is treated as missing.
XPOINT_SENTINEL = -9.0

#: COARSE MAST limiter bounding box for the in-vessel sanity reject (NOT a
#: public/private discriminator).  Measured limiter extent is R∈[0.20, 1.90],
#: |Z|≤1.83; the bbox is widened slightly so a genuine edge null near the
#: limiter is never wrongly rejected — only a NaN / sentinel / wildly-displaced
#: reconstruction artefact is dropped.
XPOINT_VESSEL_R_RANGE = (0.1, 2.0)
XPOINT_VESSEL_Z_ABS = 2.0

#: Number of X-point null-set candidate slots (the store carries up to 2 nulls).
N_XPOINT_SLOTS = 2

#: Number of fixed poloidal angles the LCFS boundary is resampled onto.
N_LCFS_ANGLES = 8

#: Fixed target dimensionality: axis(2) + X-point null-set(2 slots × 2) + 8 LCFS
#: radii = 14.
TARGET_DIM = 2 + 2 * N_XPOINT_SLOTS + N_LCFS_ANGLES

#: Human-readable name per target component (length == TARGET_DIM).  The X-point
#: is an ORDER-INVARIANT null SET of ``N_XPOINT_SLOTS`` unordered slots
#: (``xpt0``, ``xpt1``); the slot index carries NO ordering / topology meaning —
#: the probe matches predictions to targets with a permutation-invariant loss.
TARGET_NAMES: tuple[str, ...] = (
    "axis_R",
    "axis_Z",
    *tuple(f"xpt{s}_{c}" for s in range(N_XPOINT_SLOTS) for c in ("R", "Z")),
    *tuple(f"lcfs_r_{k}" for k in range(N_LCFS_ANGLES)),
)

#: Fixed poloidal angles (radians, CCW from outboard +R midplane) the LCFS is
#: ray-cast onto.  θ_k = 2π k / N_LCFS_ANGLES.
LCFS_ANGLES = (2.0 * np.pi * np.arange(N_LCFS_ANGLES) / N_LCFS_ANGLES).astype(
    np.float64
)


@dataclass
class EquilibriumGeometry:
    """Per-frame 14-D plasma-geometry labels for one shot's camera window.

    Attributes
    ----------
    shot_id:
        Source shot.
    frame_times:
        ``(F,)`` camera frame times (s) the labels were interpolated onto.
    target:
        ``(F, 14)`` float32 geometry labels in METRES (see module docstring
        for the component layout — axis, two unordered null slots, 8 LCFS radii).
        Masked components are NaN.
    finite_mask:
        ``(F, 14)`` bool — True where the component is a real, finite label
        (the equilibrium slice was defined at that frame time).
    names:
        The 14 component names (``TARGET_NAMES``).
    units:
        Units string for every component ("m").
    """

    shot_id: int
    frame_times: np.ndarray
    target: np.ndarray
    finite_mask: np.ndarray
    names: tuple[str, ...] = TARGET_NAMES
    units: str = "m"

    @property
    def n_frames(self) -> int:
        return int(self.target.shape[0])


# ---------------------------------------------------------------------------
# Store IO
# ---------------------------------------------------------------------------


def equilibrium_store_path(shot_id: int, level2_root: Path | None = None) -> Path:
    """Return the ``<shot>.zarr`` L2 store path for one shot."""
    root = level2_root or DEFAULT_LEVEL2_ROOT
    return Path(root) / f"{shot_id}.zarr"


def _read_equilibrium_group(shot_id: int, level2_root: Path | None):
    """Open the ``equilibrium`` group of one shot's L2 store (Zarr V3)."""
    import zarr  # noqa: PLC0415

    path = equilibrium_store_path(shot_id, level2_root)
    store = zarr.open_group(str(path), mode="r")
    if "equilibrium" not in set(store.group_keys()):
        raise KeyError(f"shot {shot_id}: no 'equilibrium' group at {path}")
    return store["equilibrium"]


# ---------------------------------------------------------------------------
# Geometry extraction primitives
# ---------------------------------------------------------------------------


def _interp_1d_masked(
    t_native: np.ndarray, y_native: np.ndarray, t_target: np.ndarray
) -> np.ndarray:
    """Linear-interpolate ``y_native(t_native)`` onto ``t_target``.

    NaN native samples are dropped before interpolation; a target time that
    is not bracketed by two finite native samples (out of range, or both
    bracketing samples non-finite via gaps) yields NaN.  This is a per-frame
    mask source — never an imputation.
    """
    t_n = np.asarray(t_native, dtype=np.float64)
    y_n = np.asarray(y_native, dtype=np.float64)
    t_g = np.asarray(t_target, dtype=np.float64)
    out = np.full(t_g.shape, np.nan, dtype=np.float64)

    finite = np.isfinite(t_n) & np.isfinite(y_n)
    if finite.sum() < 2:
        return out
    tn = t_n[finite]
    yn = y_n[finite]
    order = np.argsort(tn)
    tn = tn[order]
    yn = yn[order]

    in_range = (t_g >= tn[0]) & (t_g <= tn[-1])
    if not in_range.any():
        return out
    out[in_range] = np.interp(t_g[in_range], tn, yn)
    return out


def _null_in_vessel(r: float, z: float) -> bool:
    """Coarse in-vessel sanity (NOT a public/private discriminator).

    True iff ``(r, z)`` is finite, non-sentinel, and inside the coarse MAST
    limiter bounding box — rejecting only a NaN / sentinel / wildly-displaced
    reconstruction artefact, never separating boundary from private-region nulls.
    """
    if not (np.isfinite(r) and np.isfinite(z)):
        return False
    if r <= XPOINT_SENTINEL or z <= XPOINT_SENTINEL:
        return False
    r_lo, r_hi = XPOINT_VESSEL_R_RANGE
    return r_lo <= r <= r_hi and abs(z) <= XPOINT_VESSEL_Z_ABS


def xpoint_null_set(
    x_point_r: np.ndarray, x_point_z: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per native slice, the ORDER-INVARIANT set of ≤2 real in-vessel nulls.

    The store carries up to two nulls per slice in ``(2, nt)`` arrays with a
    ``-9.99`` sentinel for an undefined null.  This returns the real (finite,
    in-vessel — :func:`_null_in_vessel`) nulls packed into ``N_XPOINT_SLOTS``
    UNORDERED slots, in the store's row order (the slot index carries NO
    ordering / topology meaning — the downstream loss is permutation-invariant).
    An absent slot is NaN.  No sign-of-Z split, no ψ filter, no continuity
    tracking — just the set of valid nulls at each slice.

    Returns ``(set_r, set_z)`` each ``(N_XPOINT_SLOTS, nt)`` float64 (NaN where a
    slot is absent at that slice).
    """
    xr = np.asarray(x_point_r, dtype=np.float64)
    xz = np.asarray(x_point_z, dtype=np.float64)
    n_rows, nt = xr.shape
    set_r = np.full((N_XPOINT_SLOTS, nt), np.nan, dtype=np.float64)
    set_z = np.full((N_XPOINT_SLOTS, nt), np.nan, dtype=np.float64)
    for i in range(nt):
        slot = 0
        for row in range(n_rows):
            if slot >= N_XPOINT_SLOTS:
                break
            r, z = float(xr[row, i]), float(xz[row, i])
            if _null_in_vessel(r, z):
                set_r[slot, i] = r
                set_z[slot, i] = z
                slot += 1
    return set_r, set_z


def _nearest_native_set(
    t_eq: np.ndarray, set_native: np.ndarray, t_target: np.ndarray
) -> np.ndarray:
    """Sample a null-set slot at each target time from its NEAREST native slice.

    The set members are NOT interpolated across slices (that removes the
    flip/interp problem): each query time takes the value of the temporally
    nearest native slice.  A target outside the native range, or whose nearest
    native slot is absent (NaN), yields NaN (masked).  ``set_native`` is
    ``(nt,)`` for one slot.
    """
    t_n = np.asarray(t_eq, dtype=np.float64)
    y_n = np.asarray(set_native, dtype=np.float64)
    t_g = np.asarray(t_target, dtype=np.float64)
    out = np.full(t_g.shape, np.nan, dtype=np.float64)
    if t_n.size == 0:
        return out
    order = np.argsort(t_n)
    tn = t_n[order]
    yn = y_n[order]
    in_range = (t_g >= tn[0]) & (t_g <= tn[-1])
    if not in_range.any():
        return out
    tg = t_g[in_range]
    hi = np.clip(np.searchsorted(tn, tg, side="left"), 0, tn.size - 1)
    lo = np.clip(hi - 1, 0, tn.size - 1)
    take_hi = np.abs(tn[hi] - tg) <= np.abs(tg - tn[lo])
    nearest = np.where(take_hi, yn[hi], yn[lo])
    out[in_range] = nearest  # NaN native slots propagate as masked
    return out


def resample_lcfs_radii(
    lcfs_r_col: np.ndarray,
    lcfs_z_col: np.ndarray,
    axis_r: float,
    axis_z: float,
    angles: np.ndarray = LCFS_ANGLES,
) -> np.ndarray:
    """Ray-cast the LCFS contour to a radius at each fixed poloidal angle.

    Given one time slice's boundary contour points ``(lcfs_r_col, lcfs_z_col)``
    (metres, NaN-padded) and the magnetic axis ``(axis_r, axis_z)``, return the
    distance (metres) from the axis to the contour along each of the ``angles``
    poloidal rays (θ measured CCW from the outboard +R midplane).

    Method: convert each finite boundary point to its (angle, radius) about the
    axis, sort by angle, and linearly interpolate radius vs. angle onto the
    fixed query angles, wrapping the contour periodically in [0, 2π).  A slice
    with fewer than 3 finite boundary points (no usable contour) yields all NaN.

    Parameters
    ----------
    lcfs_r_col, lcfs_z_col:
        ``(n_bdy,)`` boundary R, Z for one time slice (metres; NaN padding).
    axis_r, axis_z:
        Magnetic axis (metres) for the same slice.
    angles:
        Query poloidal angles (radians).

    Returns
    -------
    np.ndarray ``(len(angles),)`` float64 radii (metres); NaN if no contour or
    the axis is undefined.
    """
    ang = np.asarray(angles, dtype=np.float64)
    out = np.full(ang.shape, np.nan, dtype=np.float64)
    if not (np.isfinite(axis_r) and np.isfinite(axis_z)):
        return out

    r = np.asarray(lcfs_r_col, dtype=np.float64)
    z = np.asarray(lcfs_z_col, dtype=np.float64)
    finite = np.isfinite(r) & np.isfinite(z)
    if finite.sum() < 3:
        return out
    rr = r[finite]
    zz = z[finite]

    dr = rr - axis_r
    dz = zz - axis_z
    theta = np.mod(np.arctan2(dz, dr), 2.0 * np.pi)  # [0, 2π) CCW from +R
    radius = np.hypot(dr, dz)

    order = np.argsort(theta)
    theta = theta[order]
    radius = radius[order]

    # Periodic wrap: prepend the last point shifted by -2π and append the first
    # shifted by +2π so np.interp covers the full [0, 2π) ring without gaps.
    theta_ext = np.concatenate(
        [theta[-1:] - 2.0 * np.pi, theta, theta[:1] + 2.0 * np.pi]
    )
    radius_ext = np.concatenate([radius[-1:], radius, radius[:1]])
    out = np.interp(ang, theta_ext, radius_ext)
    return out


# ---------------------------------------------------------------------------
# Top-level label builder
# ---------------------------------------------------------------------------


def load_equilibrium_geometry(
    shot_id: int,
    frame_times: np.ndarray,
    *,
    level2_root: Path | None = None,
    angles: np.ndarray = LCFS_ANGLES,
) -> EquilibriumGeometry:
    """Build per-camera-frame 14-D geometry labels for one shot.

    Interpolates the L2 equilibrium axis / LCFS-shape (linear) and samples the
    X-point null SET (nearest-native, order-invariant) onto ``frame_times``.
    Returns an :class:`EquilibriumGeometry` whose ``target`` is ``(F, 14)`` in
    METRES and whose ``finite_mask`` is ``(F, 14)`` (False = undefined / absent
    — masked, NOT imputed).

    The 14 components (see module docstring): ``axis_R, axis_Z`` then the
    X-point null set ``xpt0_R, xpt0_Z, xpt1_R, xpt1_Z`` (two unordered slots)
    then 8 ``lcfs_r`` control-point radii at the fixed poloidal angles
    ``angles`` about the magnetic axis.

    Parameters
    ----------
    shot_id:
        Shot whose L2 equilibrium store is read.
    frame_times:
        ``(F,)`` camera frame times (seconds).
    level2_root:
        Override the L2 mirror root (defaults to the project mirror).
    angles:
        Fixed poloidal query angles for the LCFS radii (defaults to the 8
        equally-spaced angles in ``LCFS_ANGLES``).

    Returns
    -------
    EquilibriumGeometry
    """
    ft = np.asarray(frame_times, dtype=np.float64).ravel()
    eq = _read_equilibrium_group(shot_id, level2_root)

    t_eq = np.asarray(eq["time"], dtype=np.float64)
    axis_r = np.asarray(eq["magnetic_axis_r"], dtype=np.float64)
    axis_z = np.asarray(eq["magnetic_axis_z"], dtype=np.float64)
    if "x_point_r" in eq and "x_point_z" in eq:
        xpt_r2 = np.asarray(eq["x_point_r"], dtype=np.float64)  # (2, nt)
        xpt_z2 = np.asarray(eq["x_point_z"], dtype=np.float64)
    else:
        xpt_r2 = np.empty((0, t_eq.size), dtype=np.float64)
        xpt_z2 = np.empty((0, t_eq.size), dtype=np.float64)
    lcfs_r = np.asarray(eq["lcfs_r"], dtype=np.float64)  # (n_bdy, nt)
    lcfs_z = np.asarray(eq["lcfs_z"], dtype=np.float64)

    return build_geometry_from_arrays(
        shot_id=shot_id,
        frame_times=ft,
        t_eq=t_eq,
        axis_r=axis_r,
        axis_z=axis_z,
        x_point_r=xpt_r2,
        x_point_z=xpt_z2,
        lcfs_r=lcfs_r,
        lcfs_z=lcfs_z,
        angles=angles,
    )


def build_geometry_from_arrays(
    *,
    shot_id: int,
    frame_times: np.ndarray,
    t_eq: np.ndarray,
    axis_r: np.ndarray,
    axis_z: np.ndarray,
    x_point_r: np.ndarray,
    x_point_z: np.ndarray,
    lcfs_r: np.ndarray,
    lcfs_z: np.ndarray,
    angles: np.ndarray = LCFS_ANGLES,
) -> EquilibriumGeometry:
    """Assemble the 14-D per-frame labels from raw equilibrium arrays.

    Split out from :func:`load_equilibrium_geometry` so tests can drive it
    directly on synthetic arrays without a Zarr store.  Conventions and shapes
    match the store (time is the LAST axis on ``x_point_*`` / ``lcfs_*``).

    Axis + LCFS are CONTINUOUS -> linearly interpolated onto ``frame_times``.
    The X-point is an ORDER-INVARIANT null SET (:func:`xpoint_null_set`): up to
    two unordered slots, each sampled at its NEAREST-native slice (NO member
    interpolation — that removes the flip/interp problem), with absent slots
    masked.  The slot index carries no ordering / topology meaning; the probe's
    permutation-invariant loss matches predictions to the present targets.
    """
    ft = np.asarray(frame_times, dtype=np.float64).ravel()
    n_frames = ft.size
    n_ang = len(angles)
    dim = 2 + 2 * N_XPOINT_SLOTS + n_ang

    t_eq = np.asarray(t_eq, dtype=np.float64)
    axis_r = np.asarray(axis_r, dtype=np.float64)
    axis_z = np.asarray(axis_z, dtype=np.float64)

    # 1) X-point null SET on the native time base (≤2 unordered in-vessel nulls).
    set_r_n, set_z_n = xpoint_null_set(x_point_r, x_point_z)

    # 2) LCFS control-point radii on the native time base, slice by slice.
    nt = t_eq.size
    lcfs_radii_native = np.full((nt, n_ang), np.nan, dtype=np.float64)
    for i in range(nt):
        lcfs_radii_native[i] = resample_lcfs_radii(
            lcfs_r[:, i], lcfs_z[:, i], float(axis_r[i]), float(axis_z[i]), angles
        )

    # 3) Project onto the camera frame times.  Axis + LCFS are continuous ->
    #    LINEAR interp; the X-point null slots are NEAREST-native (window-centre,
    #    no member interpolation).  A frame not bracketed (axis/LCFS) or outside
    #    the native range (nulls) is masked, never imputed.
    target = np.full((n_frames, dim), np.nan, dtype=np.float64)
    target[:, 0] = _interp_1d_masked(t_eq, axis_r, ft)
    target[:, 1] = _interp_1d_masked(t_eq, axis_z, ft)
    for s in range(N_XPOINT_SLOTS):
        sr = _nearest_native_set(t_eq, set_r_n[s], ft)
        sz = _nearest_native_set(t_eq, set_z_n[s], ft)
        # couple each slot's R/Z masks (a null slot is valid only where both finite)
        both = np.isfinite(sr) & np.isfinite(sz)
        target[:, 2 + 2 * s] = np.where(both, sr, np.nan)
        target[:, 3 + 2 * s] = np.where(both, sz, np.nan)
    for k in range(n_ang):
        target[:, 2 + 2 * N_XPOINT_SLOTS + k] = _interp_1d_masked(
            t_eq, lcfs_radii_native[:, k], ft
        )

    finite_mask = np.isfinite(target)
    return EquilibriumGeometry(
        shot_id=int(shot_id),
        frame_times=ft,
        target=target.astype(np.float32),
        finite_mask=finite_mask,
        names=TARGET_NAMES,
        units="m",
    )


__all__ = [
    "DEFAULT_LEVEL2_ROOT",
    "XPOINT_SENTINEL",
    "XPOINT_VESSEL_R_RANGE",
    "XPOINT_VESSEL_Z_ABS",
    "N_XPOINT_SLOTS",
    "N_LCFS_ANGLES",
    "TARGET_DIM",
    "TARGET_NAMES",
    "LCFS_ANGLES",
    "EquilibriumGeometry",
    "equilibrium_store_path",
    "xpoint_null_set",
    "resample_lcfs_radii",
    "load_equilibrium_geometry",
    "build_geometry_from_arrays",
]
