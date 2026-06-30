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

    index  name          meaning (metres)
    -----  ------------  --------------------------------------------------
      0    axis_R        magnetic-axis major radius
      1    axis_Z        magnetic-axis height
      2    lower_xpt_R   LOWER (Z<0) divertor X-point major radius
      3    lower_xpt_Z   LOWER X-point height
      4    upper_xpt_R   UPPER (Z>0) divertor X-point major radius
      5    upper_xpt_Z   UPPER X-point height
    6..13  lcfs_r[k]     LCFS control-point RADIUS at 8 fixed poloidal
                         angles θ_k about the magnetic axis (k = 0..7),
                         θ_k = 2π k / 8 measured CCW from the outboard
                         midplane (+R direction) in the (R, Z) poloidal plane.

The 8 LCFS radii are the ray-cast distance (metres) from the magnetic axis
to the last-closed-flux-surface contour along each of 8 equally-spaced
poloidal angles — a fixed-dimension, rotation-stable parameterisation of the
boundary shape that does not depend on the (variable, NaN-padded) number of
raw boundary points the store carries.

Masking, not imputation
-----------------------
The L2 equilibrium is only defined while the plasma exists: early ramp-up and
late ramp-down slices carry **NaN** axis/LCFS values, and the X-point carries
a ``-9.99`` sentinel when no null is reconstructed.  A camera frame whose
interpolated target touches an undefined equilibrium slice is **masked**
(``finite_mask[f, c] = False``) — never imputed.  Each of the 12 components
has its own per-frame mask, so a frame can contribute an axis label while its
X-point label is masked out.

The X-point is SPLIT into two SEPARATE channels — LOWER (Z<0) and UPPER (Z>0)
— each present-when-present and masked-when-absent (single-null carries one,
double-null both, limiter neither).  This subsumes the earlier "primary
X-point" hack: a single primary target was **bimodal** across a topology switch
(the active null jumps lower↔upper, a ~2.4 m Z step), so linearly interpolating
it drew a line through Z~0 matching no physical null.  As two separate
sign-of-Z channels there is **no flip and no discontinuity** — each channel is
continuous within its own presence, so it is interpolated **linearly** (like
axis / LCFS) and only its NaN-absence gaps are masked (continuity-tracking is
dropped; the masking-on-absence is kept).

Boundary-null filter (psi-proximity)
------------------------------------
The flux map can carry a reconstructed null that is **not** on the plasma
boundary (an internal / far field null).  Each store null is kept only if it is
BOUNDARY-ASSOCIATED: its poloidal flux matches the LCFS flux,
``|ψ(null) − ψ_boundary| <= tol · |ψ_boundary − ψ_axis|`` with ``tol`` =
:data:`XPOINT_PSI_TOL`, where ``ψ_boundary`` is the median ψ bilinearly
interpolated at the LCFS contour points, ``ψ_axis`` the ψ at the magnetic axis,
and ``ψ(null)`` the ψ bilinearly interpolated at the null's ``(R, Z)`` on the
store's ``ψ(R, Z, t)`` grid.  A null failing the test is **masked** (not a
boundary null).  No null re-finding and no separatrix tracing — the store's two
nulls + this single ψ check only.

Store facts (verified on the L2 mirror, 2026-06-24)
---------------------------------------------------
``/work/projects/imas_gpu/mast/level2/shots/<id>.zarr`` group ``equilibrium``
is **Zarr V3** — open with ``zarr.open_group(path, mode='r')`` (NOT
imas-python; the corpus is Zarr, not IMAS HDF5).  Keys (time is the LAST axis
on >= 2-D fields):

  - ``magnetic_axis_r``/``magnetic_axis_z``  ``(nt,)``  metres, NaN when off
  - ``x_point_r``/``x_point_z``              ``(2, nt)``  metres; ``-9.99``
    sentinel for an undefined null; the two rows are the lower / upper nulls.
    MAST runs **double-null most of the time** (e.g. shot 18504: 79/124 slices
    carry two real nulls) and switches topology, so a fixed "lower null" rule
    flips lower<->upper whenever the lower null drops to sentinel — a ~2.4 m
    jump in the picked-Z series.  The *primary* null is therefore selected by
    **temporal continuity** (track the null closest to the previous slice's
    primary), which gives a stable, physical trajectory across DN<->SN
    switches.  See :func:`select_primary_xpoint`.
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

#: Boundary-null ψ-proximity tolerance.  A store null is kept only when its
#: poloidal flux matches the LCFS flux to within this fraction of the
#: axis→boundary flux difference: ``|ψ_null − ψ_boundary| <= tol·|ψ_boundary −
#: ψ_axis|``.  Measured (18502/18504): genuine boundary nulls sit at normalised
#: ψ-distance ~0.001–0.01, so 0.05 keeps every real boundary null with margin
#: while rejecting an internal / far null whose ψ differs by a sizeable fraction
#: of the axis-boundary span.
XPOINT_PSI_TOL = 0.05

#: Number of fixed poloidal angles the LCFS boundary is resampled onto.
N_LCFS_ANGLES = 8

#: Fixed target dimensionality: axis(2) + LOWER X-point(2) + UPPER X-point(2)
#: + 8 LCFS radii = 14.
TARGET_DIM = 6 + N_LCFS_ANGLES

#: Human-readable name per target component (length == TARGET_DIM).  The X-point
#: is split into a LOWER (Z<0) and an UPPER (Z>0) divertor channel.
TARGET_NAMES: tuple[str, ...] = (
    "axis_R",
    "axis_Z",
    "lower_xpt_R",
    "lower_xpt_Z",
    "upper_xpt_R",
    "upper_xpt_Z",
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
        for the component layout — axis, LOWER/UPPER X-point, 8 LCFS radii).
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


def _bilinear_on_grid(
    field_zr: np.ndarray, r_axis: np.ndarray, z_axis: np.ndarray, r: float, z: float
) -> float:
    """Bilinearly interpolate a ``(nz, nr)`` field at ``(r, z)``.

    ``field_zr`` is indexed ``[z, r]`` (the L2 ``psi`` grid layout: the axis ψ
    is the field maximum, verified on the mirror).  ``r_axis`` / ``z_axis`` are
    the monotonically-increasing grid coordinates.  Returns NaN when ``(r, z)``
    is outside the grid or the bracketing cell carries a NaN.
    """
    if not (np.isfinite(r) and np.isfinite(z)):
        return float("nan")
    if not (r_axis[0] <= r <= r_axis[-1] and z_axis[0] <= z <= z_axis[-1]):
        return float("nan")
    ir = min(max(int(np.searchsorted(r_axis, r)) - 1, 0), r_axis.size - 2)
    iz = min(max(int(np.searchsorted(z_axis, z)) - 1, 0), z_axis.size - 2)
    tr = (r - r_axis[ir]) / (r_axis[ir + 1] - r_axis[ir])
    tz = (z - z_axis[iz]) / (z_axis[iz + 1] - z_axis[iz])
    f00 = field_zr[iz, ir]
    f01 = field_zr[iz, ir + 1]
    f10 = field_zr[iz + 1, ir]
    f11 = field_zr[iz + 1, ir + 1]
    return float(
        f00 * (1 - tr) * (1 - tz)
        + f01 * tr * (1 - tz)
        + f10 * (1 - tr) * tz
        + f11 * tr * tz
    )


def _is_boundary_null(
    psi_zr: np.ndarray,
    r_axis: np.ndarray,
    z_axis: np.ndarray,
    null_r: float,
    null_z: float,
    axis_r: float,
    axis_z: float,
    lcfs_r_col: np.ndarray,
    lcfs_z_col: np.ndarray,
    *,
    tol: float = XPOINT_PSI_TOL,
) -> bool:
    """True iff a null's poloidal flux matches the LCFS flux (a boundary null).

    ``ψ_boundary`` is the median ψ bilinearly sampled at the LCFS contour
    points; ``ψ_axis`` the ψ at the magnetic axis; the null is kept iff
    ``|ψ_null − ψ_boundary| <= tol·|ψ_boundary − ψ_axis|``.  When ψ / axis /
    boundary cannot be evaluated the null is conservatively KEPT (the filter
    only ever *rejects* a confidently-non-boundary null — it never invents a
    rejection from missing data).
    """
    psi_axis = _bilinear_on_grid(psi_zr, r_axis, z_axis, axis_r, axis_z)
    bvals = [
        _bilinear_on_grid(psi_zr, r_axis, z_axis, float(rr), float(zz))
        for rr, zz in zip(lcfs_r_col, lcfs_z_col, strict=False)
        if np.isfinite(rr) and np.isfinite(zz)
    ]
    bvals = [v for v in bvals if np.isfinite(v)]
    if not bvals or not np.isfinite(psi_axis):
        return True  # cannot evaluate the filter — keep (do not invent a reject)
    psi_boundary = float(np.median(bvals))
    span = abs(psi_boundary - psi_axis)
    if span <= 0.0:
        return True
    psi_null = _bilinear_on_grid(psi_zr, r_axis, z_axis, null_r, null_z)
    if not np.isfinite(psi_null):
        return True
    return abs(psi_null - psi_boundary) <= tol * span


def select_split_xpoints(
    x_point_r: np.ndarray,
    x_point_z: np.ndarray,
    *,
    psi: np.ndarray | None = None,
    r_axis: np.ndarray | None = None,
    z_axis: np.ndarray | None = None,
    axis_r: np.ndarray | None = None,
    axis_z: np.ndarray | None = None,
    lcfs_r: np.ndarray | None = None,
    lcfs_z: np.ndarray | None = None,
    psi_tol: float = XPOINT_PSI_TOL,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Split the store's two nulls into LOWER (Z<0) and UPPER (Z>0) channels.

    The store carries up to two X-points per slice in ``(2, nt)`` arrays with a
    ``-9.99`` sentinel for an undefined null.  Each real (finite, non-sentinel)
    null is assigned to the LOWER channel when Z < 0 and the UPPER channel when
    Z > 0 — so each channel is its own continuous, present-when-present series
    (no lower↔upper flip, no 2.4 m discontinuity).  When the ψ grid + axis +
    LCFS are supplied, a null is kept only if it is BOUNDARY-ASSOCIATED
    (:func:`_is_boundary_null`); a non-boundary null is dropped (its channel is
    NaN at that slice).  If two real nulls land in the same sign-of-Z channel
    (degenerate; rare), the one closest to the boundary flux is kept.

    Returns
    -------
    (lower_r, lower_z, upper_r, upper_z, n_rejected) : the four ``(nt,)``
    float64 channels (NaN where absent / non-boundary) plus the count of nulls
    the ψ-filter rejected (0 when no ψ grid was supplied).
    """
    xr = np.asarray(x_point_r, dtype=np.float64)
    xz = np.asarray(x_point_z, dtype=np.float64)
    _, nt = xr.shape
    real = (
        np.isfinite(xr)
        & np.isfinite(xz)
        & (xr > XPOINT_SENTINEL)
        & (xz > XPOINT_SENTINEL)
    )
    lower_r = np.full(nt, np.nan, dtype=np.float64)
    lower_z = np.full(nt, np.nan, dtype=np.float64)
    upper_r = np.full(nt, np.nan, dtype=np.float64)
    upper_z = np.full(nt, np.nan, dtype=np.float64)
    n_rejected = 0

    use_psi = (
        psi is not None
        and r_axis is not None
        and z_axis is not None
        and axis_r is not None
        and axis_z is not None
        and lcfs_r is not None
        and lcfs_z is not None
    )
    r_ax = np.asarray(r_axis, dtype=np.float64) if use_psi else None
    z_ax = np.asarray(z_axis, dtype=np.float64) if use_psi else None

    for i in range(nt):
        for row in np.flatnonzero(real[:, i]):
            nr = float(xr[row, i])
            nz = float(xz[row, i])
            if use_psi and not _is_boundary_null(
                psi[:, :, i],
                r_ax,
                z_ax,
                nr,
                nz,
                float(axis_r[i]),
                float(axis_z[i]),
                lcfs_r[:, i],
                lcfs_z[:, i],
                tol=psi_tol,
            ):
                n_rejected += 1
                continue
            if nz < 0.0:
                tgt_r, tgt_z = lower_r, lower_z
            else:
                tgt_r, tgt_z = upper_r, upper_z
            # On the rare same-sign degeneracy, keep the null nearer the
            # midplane-side boundary (smaller |Z| is the conventional divertor
            # null); only overwrite if the slot is empty or this null is lower-|Z|.
            if not np.isfinite(tgt_z[i]) or abs(nz) < abs(tgt_z[i]):
                tgt_r[i] = nr
                tgt_z[i] = nz
    return lower_r, lower_z, upper_r, upper_z, n_rejected


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
    """Build per-camera-frame 12-D geometry labels for one shot.

    Interpolates the L2 equilibrium axis / primary X-point / LCFS-shape onto
    ``frame_times`` (camera frame times, seconds).  Returns an
    :class:`EquilibriumGeometry` whose ``target`` is ``(F, 12)`` in METRES and
    whose ``finite_mask`` is ``(F, 12)`` (False = the equilibrium was undefined
    at that frame, e.g. plasma-off slices — masked, NOT imputed).

    The 12 components (see module docstring): ``axis_R, axis_Z, xpt_R, xpt_Z``
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
    xpt_r2 = np.asarray(eq["x_point_r"], dtype=np.float64)  # (2, nt)
    xpt_z2 = np.asarray(eq["x_point_z"], dtype=np.float64)
    lcfs_r = np.asarray(eq["lcfs_r"], dtype=np.float64)  # (n_bdy, nt)
    lcfs_z = np.asarray(eq["lcfs_z"], dtype=np.float64)

    # ψ grid + axes for the boundary-null filter (ψ[z, r, t]; axis ψ is the
    # field maximum on this store).  Optional — absent on a store without them
    # disables the filter (all nulls kept).
    psi = r_axis = z_axis = None
    keys = set(eq.array_keys())
    if {"psi", "major_radius", "z"} <= keys:
        psi = np.asarray(eq["psi"], dtype=np.float64)  # (nz, nr, nt)
        r_axis = np.asarray(eq["major_radius"], dtype=np.float64)
        z_axis = np.asarray(eq["z"], dtype=np.float64)

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
        psi=psi,
        r_axis=r_axis,
        z_axis=z_axis,
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
    psi: np.ndarray | None = None,
    r_axis: np.ndarray | None = None,
    z_axis: np.ndarray | None = None,
    psi_tol: float = XPOINT_PSI_TOL,
    angles: np.ndarray = LCFS_ANGLES,
    return_rejected: bool = False,
):
    """Assemble the 14-D per-frame labels from raw equilibrium arrays.

    Split out from :func:`load_equilibrium_geometry` so tests can drive it
    directly on synthetic arrays without a Zarr store.  Conventions and shapes
    match the store (time is the LAST axis on ``x_point_*`` / ``lcfs_*`` /
    ``psi``).  The X-point is SPLIT into LOWER (Z<0) and UPPER (Z>0) channels;
    each is continuous within its presence so it is LINEARLY interpolated and
    only its absence gaps are masked.  When ``psi`` + ``r_axis`` + ``z_axis``
    are supplied a null is kept only if it is boundary-associated
    (:func:`_is_boundary_null`).

    Returns an :class:`EquilibriumGeometry`; with ``return_rejected`` a tuple
    ``(geometry, n_rejected_nulls)``.
    """
    ft = np.asarray(frame_times, dtype=np.float64).ravel()
    n_frames = ft.size
    n_ang = len(angles)
    dim = 6 + n_ang

    t_eq = np.asarray(t_eq, dtype=np.float64)
    axis_r = np.asarray(axis_r, dtype=np.float64)
    axis_z = np.asarray(axis_z, dtype=np.float64)

    # 1) split LOWER/UPPER X-points on the native time base (+ ψ boundary filter).
    lower_r_n, lower_z_n, upper_r_n, upper_z_n, n_rejected = select_split_xpoints(
        x_point_r,
        x_point_z,
        psi=psi,
        r_axis=r_axis,
        z_axis=z_axis,
        axis_r=axis_r,
        axis_z=axis_z,
        lcfs_r=lcfs_r,
        lcfs_z=lcfs_z,
        psi_tol=psi_tol,
    )

    # 2) LCFS control-point radii on the native time base, slice by slice.
    nt = t_eq.size
    lcfs_radii_native = np.full((nt, n_ang), np.nan, dtype=np.float64)
    for i in range(nt):
        lcfs_radii_native[i] = resample_lcfs_radii(
            lcfs_r[:, i], lcfs_z[:, i], float(axis_r[i]), float(axis_z[i]), angles
        )

    # 3) Interpolate onto the camera frame times.  EVERY channel is now
    #    CONTINUOUS within its presence (the lower/upper split removes the
    #    lower↔upper flip), so axis, both X-point channels and LCFS all use the
    #    same linear interpolation; a frame not bracketed by two finite native
    #    samples of a channel is masked (never imputed).
    target = np.full((n_frames, dim), np.nan, dtype=np.float64)
    target[:, 0] = _interp_1d_masked(t_eq, axis_r, ft)
    target[:, 1] = _interp_1d_masked(t_eq, axis_z, ft)
    # Couple each null's R/Z masks (a null frame is valid only where both
    # coordinates are physical).
    lr = _interp_1d_masked(t_eq, lower_r_n, ft)
    lz = _interp_1d_masked(t_eq, lower_z_n, ft)
    lboth = np.isfinite(lr) & np.isfinite(lz)
    target[:, 2] = np.where(lboth, lr, np.nan)
    target[:, 3] = np.where(lboth, lz, np.nan)
    ur = _interp_1d_masked(t_eq, upper_r_n, ft)
    uz = _interp_1d_masked(t_eq, upper_z_n, ft)
    uboth = np.isfinite(ur) & np.isfinite(uz)
    target[:, 4] = np.where(uboth, ur, np.nan)
    target[:, 5] = np.where(uboth, uz, np.nan)
    for k in range(n_ang):
        target[:, 6 + k] = _interp_1d_masked(t_eq, lcfs_radii_native[:, k], ft)

    finite_mask = np.isfinite(target)
    names = (
        "axis_R",
        "axis_Z",
        "lower_xpt_R",
        "lower_xpt_Z",
        "upper_xpt_R",
        "upper_xpt_Z",
        *tuple(f"lcfs_r_{k}" for k in range(n_ang)),
    )
    geometry = EquilibriumGeometry(
        shot_id=int(shot_id),
        frame_times=ft,
        target=target.astype(np.float32),
        finite_mask=finite_mask,
        names=names,
        units="m",
    )
    if return_rejected:
        return geometry, n_rejected
    return geometry


__all__ = [
    "DEFAULT_LEVEL2_ROOT",
    "XPOINT_SENTINEL",
    "XPOINT_PSI_TOL",
    "N_LCFS_ANGLES",
    "TARGET_DIM",
    "TARGET_NAMES",
    "LCFS_ANGLES",
    "EquilibriumGeometry",
    "equilibrium_store_path",
    "select_split_xpoints",
    "resample_lcfs_radii",
    "load_equilibrium_geometry",
    "build_geometry_from_arrays",
]
