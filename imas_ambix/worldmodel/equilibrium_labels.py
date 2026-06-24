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
**12-D target** in METRES plus a finite mask::

    index  name        meaning (metres)
    -----  ----------  --------------------------------------------------
      0    axis_R      magnetic-axis major radius
      1    axis_Z      magnetic-axis height
      2    xpt_R       primary (lower-null) X-point major radius
      3    xpt_Z       primary (lower-null) X-point height
    4..11  lcfs_r[k]   LCFS control-point RADIUS at 8 fixed poloidal
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

Store facts (verified on the L2 mirror, 2026-06-24)
---------------------------------------------------
``/work/projects/imas_gpu/mast/level2/shots/<id>.zarr`` group ``equilibrium``
is **Zarr V3** — open with ``zarr.open_group(path, mode='r')`` (NOT
imas-python; the corpus is Zarr, not IMAS HDF5).  Keys (time is the LAST axis
on >= 2-D fields):

  - ``magnetic_axis_r``/``magnetic_axis_z``  ``(nt,)``  metres, NaN when off
  - ``x_point_r``/``x_point_z``              ``(2, nt)``  metres; ``-9.99``
    sentinel for an undefined null; the two rows are the (typically) upper /
    lower nulls in no fixed order — the *primary* null is selected as the
    most-negative-Z real null (lower divertor, the MAST diverted topology).
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

#: Number of fixed poloidal angles the LCFS boundary is resampled onto.
N_LCFS_ANGLES = 8

#: Fixed target dimensionality: axis(2) + primary X-point(2) + 8 LCFS radii.
TARGET_DIM = 4 + N_LCFS_ANGLES

#: Human-readable name per target component (length == TARGET_DIM).
TARGET_NAMES: tuple[str, ...] = (
    "axis_R",
    "axis_Z",
    "xpt_R",
    "xpt_Z",
    *tuple(f"lcfs_r_{k}" for k in range(N_LCFS_ANGLES)),
)

#: Fixed poloidal angles (radians, CCW from outboard +R midplane) the LCFS is
#: ray-cast onto.  θ_k = 2π k / N_LCFS_ANGLES.
LCFS_ANGLES = (2.0 * np.pi * np.arange(N_LCFS_ANGLES) / N_LCFS_ANGLES).astype(
    np.float64
)


@dataclass
class EquilibriumGeometry:
    """Per-frame 12-D plasma-geometry labels for one shot's camera window.

    Attributes
    ----------
    shot_id:
        Source shot.
    frame_times:
        ``(F,)`` camera frame times (s) the labels were interpolated onto.
    target:
        ``(F, 12)`` float32 geometry labels in METRES (see module docstring
        for the component layout).  Masked components are NaN.
    finite_mask:
        ``(F, 12)`` bool — True where the component is a real, finite label
        (the equilibrium slice was defined at that frame time).
    names:
        The 12 component names (``TARGET_NAMES``).
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


def select_primary_xpoint(
    x_point_r: np.ndarray, x_point_z: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Select the primary (lower-null) X-point per time slice.

    The store carries up to two X-points per slice in ``(2, nt)`` arrays with
    a ``-9.99`` sentinel for an undefined null.  The *primary* null is taken as
    the real (finite, non-sentinel) X-point with the **most negative Z** — the
    lower divertor null of the standard MAST diverted topology.  When no real
    null exists at a slice the result is NaN there (later masked).

    Parameters
    ----------
    x_point_r, x_point_z:
        ``(2, nt)`` metre arrays from the equilibrium store.

    Returns
    -------
    (xpt_r, xpt_z) : each ``(nt,)`` float64; NaN where no real lower null.
    """
    xr = np.asarray(x_point_r, dtype=np.float64)
    xz = np.asarray(x_point_z, dtype=np.float64)
    n_xpt, nt = xr.shape
    real = (
        np.isfinite(xr)
        & np.isfinite(xz)
        & (xr > XPOINT_SENTINEL)
        & (xz > XPOINT_SENTINEL)
    )
    out_r = np.full(nt, np.nan, dtype=np.float64)
    out_z = np.full(nt, np.nan, dtype=np.float64)
    # Among real nulls per slice, pick the most-negative Z (lower null).
    z_for_min = np.where(real, xz, np.inf)
    has_real = real.any(axis=0)
    pick = np.argmin(z_for_min, axis=0)  # (nt,)
    cols = np.arange(nt)
    out_r[has_real] = xr[pick[has_real], cols[has_real]]
    out_z[has_real] = xz[pick[has_real], cols[has_real]]
    return out_r, out_z


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
    """Assemble the 12-D per-frame labels from raw equilibrium arrays.

    Split out from :func:`load_equilibrium_geometry` so tests can drive it
    directly on synthetic arrays without a Zarr store.  Conventions and shapes
    match the store (time is the LAST axis on ``x_point_*`` and ``lcfs_*``).
    """
    ft = np.asarray(frame_times, dtype=np.float64).ravel()
    n_frames = ft.size
    n_ang = len(angles)
    dim = 4 + n_ang

    t_eq = np.asarray(t_eq, dtype=np.float64)
    axis_r = np.asarray(axis_r, dtype=np.float64)
    axis_z = np.asarray(axis_z, dtype=np.float64)

    # 1) axis + primary X-point on the native equilibrium time base.
    xpt_r_native, xpt_z_native = select_primary_xpoint(x_point_r, x_point_z)

    # 2) LCFS control-point radii on the native time base, slice by slice.
    nt = t_eq.size
    lcfs_radii_native = np.full((nt, n_ang), np.nan, dtype=np.float64)
    for i in range(nt):
        lcfs_radii_native[i] = resample_lcfs_radii(
            lcfs_r[:, i], lcfs_z[:, i], float(axis_r[i]), float(axis_z[i]), angles
        )

    # 3) Interpolate every native series onto the camera frame times.
    target = np.full((n_frames, dim), np.nan, dtype=np.float64)
    target[:, 0] = _interp_1d_masked(t_eq, axis_r, ft)
    target[:, 1] = _interp_1d_masked(t_eq, axis_z, ft)
    target[:, 2] = _interp_1d_masked(t_eq, xpt_r_native, ft)
    target[:, 3] = _interp_1d_masked(t_eq, xpt_z_native, ft)
    for k in range(n_ang):
        target[:, 4 + k] = _interp_1d_masked(t_eq, lcfs_radii_native[:, k], ft)

    finite_mask = np.isfinite(target)
    names = (
        "axis_R",
        "axis_Z",
        "xpt_R",
        "xpt_Z",
        *tuple(f"lcfs_r_{k}" for k in range(n_ang)),
    )
    return EquilibriumGeometry(
        shot_id=int(shot_id),
        frame_times=ft,
        target=target.astype(np.float32),
        finite_mask=finite_mask,
        names=names,
        units="m",
    )


__all__ = [
    "DEFAULT_LEVEL2_ROOT",
    "XPOINT_SENTINEL",
    "N_LCFS_ANGLES",
    "TARGET_DIM",
    "TARGET_NAMES",
    "LCFS_ANGLES",
    "EquilibriumGeometry",
    "equilibrium_store_path",
    "select_primary_xpoint",
    "resample_lcfs_radii",
    "load_equilibrium_geometry",
    "build_geometry_from_arrays",
]
