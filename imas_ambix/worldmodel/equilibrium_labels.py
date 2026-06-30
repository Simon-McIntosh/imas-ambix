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
      2    xpt_R       primary (continuity-tracked) X-point major radius
      3    xpt_Z       primary (continuity-tracked) X-point height
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

The X-point is special: its primary-null trajectory is **discontinuous** at a
topology switch (the active null jumps from the lower to the upper divertor, or
a null appears / vanishes).  Linearly interpolating across such a jump would
draw a straight line through Z~0 — a label matching *no* physical X-point.  So
``xpt_R``/``xpt_Z`` are interpolated **discontinuity-aware**: each frame takes
the value of its **nearest** native slice, and any frame whose bracketing
native primary-null trajectory jumps by more than
:data:`XPOINT_DISCONTINUITY_M` (or whose nearest native slice is a sentinel /
absent) is **masked**, never imputed.  The continuous components (axis, LCFS)
stay linearly interpolated.

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

#: Largest physical step (metres) the primary-null trajectory may take between
#: adjacent native equilibrium slices before it is treated as a topology
#: switch / discontinuity.  A real X-point drifts smoothly (cm-scale per 5 ms
#: slice on MAST); a lower<->upper flip is ~2.4 m.  Camera frames whose
#: bracketing native slices straddle a jump larger than this are masked, never
#: interpolated across.
XPOINT_DISCONTINUITY_M = 0.3

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


def _interp_nearest_no_jump(
    t_native: np.ndarray,
    y_native: np.ndarray,
    t_target: np.ndarray,
    max_jump: float,
) -> np.ndarray:
    """Nearest-native sampling that refuses to bridge a discontinuity.

    Designed for the primary-null trajectory, which is piecewise-continuous
    with topology-switch jumps (lower<->upper flips, null appearance /
    disappearance) that must NOT be interpolated across.  The series may also
    carry NaN gaps (sentinel slices where no null exists).

    A native *transition* between two consecutive **finite** samples is a
    discontinuity when their values differ by more than ``max_jump`` OR a NaN
    gap separates them in time (the trajectory was undefined in between).  Each
    finite native slice that borders such a transition is a **switch slice**.

    For each target time the value of the **nearest** finite native slice is
    taken (step-like, no blending), and the target is **masked** (NaN) when:

    - the nearest finite native slice is a switch slice (it sits on the edge of
      a topology change — neighbouring frames would land on the other branch),
      OR
    - the target falls strictly between two finite native slices whose
      transition is a discontinuity, OR
    - the target is outside the finite native range.

    This masks a whole neighbourhood around every switch rather than only the
    bracketing segment, so two adjacent camera frames can never straddle a flip
    (the on-sample exemption that would let a flip through is removed).

    Continuous fields must use :func:`_interp_1d_masked` instead — this routine
    is intentionally step-like and only correct for a discontinuous series.

    Parameters
    ----------
    t_native, y_native:
        Native time base and a (possibly NaN-gapped) value series.
    t_target:
        Query times.
    max_jump:
        Native-to-native step above which a transition is a discontinuity.

    Returns
    -------
    ``(len(t_target),)`` float64; NaN where masked.
    """
    t_n = np.asarray(t_native, dtype=np.float64)
    y_n = np.asarray(y_native, dtype=np.float64)
    t_g = np.asarray(t_target, dtype=np.float64)
    out = np.full(t_g.shape, np.nan, dtype=np.float64)

    finite = np.isfinite(t_n) & np.isfinite(y_n)
    if not finite.any():
        return out
    # Keep the ORIGINAL native indices so a NaN gap (a dropped slice between two
    # finite ones) counts as a discontinuous transition, not a smooth step.
    idx_native = np.flatnonzero(finite)
    tn = t_n[idx_native]
    yn = y_n[idx_native]
    order = np.argsort(tn)
    tn = tn[order]
    yn = yn[order]
    idx_native = idx_native[order]

    n = tn.size
    # Per-finite-sample "switch slice" flag: borders a discontinuous transition.
    switch = np.zeros(n, dtype=bool)
    if n >= 2:
        value_jump = np.abs(np.diff(yn)) > max_jump
        gap = np.diff(idx_native) > 1  # a NaN/sentinel slice sat between them
        bad_transition = value_jump | gap  # (n-1,)
        switch[:-1] |= bad_transition  # left side of each bad transition
        switch[1:] |= bad_transition  # right side

    in_range = (t_g >= tn[0]) & (t_g <= tn[-1])
    if not in_range.any():
        return out
    tg = t_g[in_range]

    # Right-bracket index `hi` in [1, n-1]; `lo = hi - 1` is the left.
    hi = np.clip(np.searchsorted(tn, tg, side="right"), 1, n - 1)
    lo = hi - 1
    t_lo, t_hi = tn[lo], tn[hi]
    y_lo, y_hi = yn[lo], yn[hi]

    take_hi = (tg - t_lo) > (t_hi - tg)
    nearest_idx = np.where(take_hi, hi, lo)
    nearest = yn[nearest_idx]

    # Mask if the segment is discontinuous OR the nearest sample is a switch
    # slice (so the whole neighbourhood of a flip is masked, never bridged).
    bridges_jump = (np.abs(y_hi - y_lo) > max_jump) | (
        idx_native[hi] - idx_native[lo] > 1
    )
    nearest_is_switch = switch[nearest_idx]
    keep = ~bridges_jump & ~nearest_is_switch

    out_in = out[in_range]
    out_in[:] = np.where(keep, nearest, np.nan)
    out[in_range] = out_in
    return out


def select_primary_xpoint(
    x_point_r: np.ndarray, x_point_z: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Select a temporally-continuous primary X-point per time slice.

    The store carries up to two X-points per slice in ``(2, nt)`` arrays with
    a ``-9.99`` sentinel for an undefined null.  MAST is double-null most of the
    time and switches topology, so a fixed "most-negative-Z (lower null)" rule
    flips between the lower and upper divertor whenever the lower null drops to
    the sentinel — producing a ~2.4 m jump in the picked-Z series and, after
    linear interpolation, a label that passes through Z~0 matching no physical
    X-point.

    Instead, the primary null is tracked by **temporal continuity**: at each
    slice the chosen null is the real (finite, non-sentinel) X-point closest in
    (R, Z) to the previous slice's primary.  The tracker is seeded at the first
    slice carrying a real null, preferring the lower null there (the
    conventional MAST diverted seed); thereafter it follows whichever null stays
    physically continuous.  A run of all-sentinel slices breaks continuity — the
    tracker re-seeds (lower-null preference) at the next real slice, and the gap
    is flagged so a downstream interpolator can mask across it rather than draw
    a line through it.

    When two genuine nulls coexist (true double-null), a single "primary" is
    ambiguous; the continuity rule resolves it to a stable, physical choice
    (it does **not** average the two).

    Parameters
    ----------
    x_point_r, x_point_z:
        ``(2, nt)`` metre arrays from the equilibrium store.

    Returns
    -------
    (xpt_r, xpt_z) : each ``(nt,)`` float64; NaN where no real null at a slice.
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
    out_r = np.full(nt, np.nan, dtype=np.float64)
    out_z = np.full(nt, np.nan, dtype=np.float64)

    prev_r: float | None = None
    prev_z: float | None = None
    for i in range(nt):
        rows = np.flatnonzero(real[:, i])
        if rows.size == 0:
            # Sentinel slice: continuity is broken (NaN here, re-seed later).
            prev_r = None
            prev_z = None
            continue
        if prev_r is None:
            # (Re-)seed: prefer the lower null (most negative Z) as the
            # conventional MAST diverted starting choice.
            sel = rows[int(np.argmin(xz[rows, i]))]
        else:
            # Follow continuity: nearest real null to the previous primary.
            d2 = (xr[rows, i] - prev_r) ** 2 + (xz[rows, i] - prev_z) ** 2
            sel = rows[int(np.argmin(d2))]
        out_r[i] = xr[sel, i]
        out_z[i] = xz[sel, i]
        prev_r = float(out_r[i])
        prev_z = float(out_z[i])
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

    # 3) Interpolate onto the camera frame times.  Axis + LCFS are continuous
    #    -> linear interp.  The primary X-point is piecewise-continuous with
    #    topology-switch jumps -> nearest-native sampling that masks (never
    #    bridges) any frame straddling a discontinuity in EITHER coordinate, so
    #    R and Z stay masked consistently (a frame is an X-point frame only if
    #    both coords are physical).
    target = np.full((n_frames, dim), np.nan, dtype=np.float64)
    target[:, 0] = _interp_1d_masked(t_eq, axis_r, ft)
    target[:, 1] = _interp_1d_masked(t_eq, axis_z, ft)
    xpt_r_frame = _interp_nearest_no_jump(
        t_eq, xpt_r_native, ft, XPOINT_DISCONTINUITY_M
    )
    xpt_z_frame = _interp_nearest_no_jump(
        t_eq, xpt_z_native, ft, XPOINT_DISCONTINUITY_M
    )
    # Couple the two coordinates' masks: an X-point frame is valid only where
    # both R and Z are physical (a jump in one is a topology switch in both).
    xpt_both = np.isfinite(xpt_r_frame) & np.isfinite(xpt_z_frame)
    target[:, 2] = np.where(xpt_both, xpt_r_frame, np.nan)
    target[:, 3] = np.where(xpt_both, xpt_z_frame, np.nan)
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
    "XPOINT_DISCONTINUITY_M",
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
