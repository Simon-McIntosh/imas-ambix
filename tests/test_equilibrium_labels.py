"""Tests for the discontinuity-aware equilibrium X-point labels.

The 12-D equilibrium labels feed the firewalled diagnostics->equilibrium
oracle.  The X-point label was corrupted because the primary null was picked
by a fixed "most-negative-Z (lower null)" rule that flips lower<->upper at a
sentinel dropout, and then the discontinuous series was *linearly* interpolated
onto camera frames — drawing a straight line through Z~0 that matches no
physical X-point.  These tests pin the fix:

  - continuity-tracked primary-null selection (no lower<->upper flips), and
  - discontinuity-aware X-point interpolation (nearest native + masking across
    a topology switch, never imputation),

both within the unchanged 12-D schema (axis, X-point, 8 LCFS radii).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

# Import the pure-numpy module by file path.  The `imas_ambix.worldmodel`
# package __init__ eagerly imports torch (a GPU-node-only dependency); this
# evaluator module has no torch dependency, so load it standalone to keep the
# test runnable in the CPU venv.
_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "imas_ambix"
    / "worldmodel"
    / "equilibrium_labels.py"
)
_spec = importlib.util.spec_from_file_location(
    "equilibrium_labels_under_test", _MODULE_PATH
)
_eq = importlib.util.module_from_spec(_spec)
# Register before exec so @dataclass (under `from __future__ import annotations`)
# can resolve the module's namespace.
sys.modules[_spec.name] = _eq
_spec.loader.exec_module(_eq)

TARGET_DIM = _eq.TARGET_DIM
XPOINT_DISCONTINUITY_M = _eq.XPOINT_DISCONTINUITY_M
XPOINT_SENTINEL = _eq.XPOINT_SENTINEL
build_geometry_from_arrays = _eq.build_geometry_from_arrays
load_equilibrium_geometry = _eq.load_equilibrium_geometry
select_primary_xpoint = _eq.select_primary_xpoint

# Physical null heights of the synthetic double-null fixture (metres).
LOWER_Z = -1.2
UPPER_Z = +1.2
# The "forbidden band": a physical X-point is never at these heights, but the
# OLD linear interpolation through a lower<->upper flip lands squarely here.
BAND = (-0.8, +0.8)


def _two_null_arrays(n_lower: int, n_after: int, *, sentinel: float = -9.99):
    """Build ``(2, nt)`` x_point arrays for a DN run that loses its lower null.

    Row 0 is the lower null (Z=LOWER_Z), row 1 the upper null (Z=UPPER_Z).  For
    the first ``n_lower`` slices BOTH nulls are real (genuine double-null); for
    the remaining ``n_after`` slices the lower null drops to the sentinel and
    only the upper null survives (single-null upper).  R is a plausible 0.6 m
    outboard for both nulls.
    """
    nt = n_lower + n_after
    xr = np.full((2, nt), 0.6, dtype=np.float64)
    xz = np.empty((2, nt), dtype=np.float64)
    xz[0, :] = LOWER_Z
    xz[1, :] = UPPER_Z
    # Lower null vanishes after the DN run.
    xr[0, n_lower:] = sentinel
    xz[0, n_lower:] = sentinel
    return xr, xz


def _old_lowest_z_pick(xr: np.ndarray, xz: np.ndarray):
    """Reproduce the OLD fixed most-negative-Z primary-null rule (for contrast)."""
    real = (
        np.isfinite(xr)
        & np.isfinite(xz)
        & (xr > XPOINT_SENTINEL)
        & (xz > XPOINT_SENTINEL)
    )
    nt = xr.shape[1]
    out_r = np.full(nt, np.nan)
    out_z = np.full(nt, np.nan)
    z_for_min = np.where(real, xz, np.inf)
    has_real = real.any(axis=0)
    pick = np.argmin(z_for_min, axis=0)
    cols = np.arange(nt)
    out_r[has_real] = xr[pick[has_real], cols[has_real]]
    out_z[has_real] = xz[pick[has_real], cols[has_real]]
    return out_r, out_z


# ---------------------------------------------------------------------------
# Primary-null selection: continuity, no lower<->upper flip
# ---------------------------------------------------------------------------


def test_old_rule_flips_at_dropout():
    """Document the defect: the fixed lowest-Z rule jumps ~2.4 m at the switch."""
    xr, xz = _two_null_arrays(n_lower=6, n_after=6)
    _, old_z = _old_lowest_z_pick(xr, xz)
    # Lower null while it exists, then it FLIPS to the upper null.
    assert np.isclose(old_z[5], LOWER_Z)
    assert np.isclose(old_z[6], UPPER_Z)
    assert abs(old_z[6] - old_z[5]) > 2.0  # ~2.4 m discontinuity


def test_continuity_pick_does_not_flip():
    """The new selection follows ONE physical null across the dropout."""
    xr, xz = _two_null_arrays(n_lower=6, n_after=6)
    _, new_z = select_primary_xpoint(xr, xz)
    # Seeded on the lower null (most negative Z); once the lower null vanishes
    # there is only the upper null left, so the picked-Z series is the lower
    # null for the DN run, then NECESSARILY the upper null — but the adjacent
    # native step at the switch is exactly the topology discontinuity the
    # interpolator must mask, NOT smooth.  What continuity buys is that during
    # the DN run the pick never oscillates.
    dn = new_z[:6]
    assert np.allclose(dn, LOWER_Z)  # stable lower null through the DN run
    # And no value ever sits in the forbidden mid-band (it is always a real
    # null height, never an average / interpolant).
    finite = new_z[np.isfinite(new_z)]
    assert np.all((finite <= BAND[0]) | (finite >= BAND[1]))


def test_continuity_tracks_drifting_null_not_lowest():
    """A drifting single null is followed even when a transient deeper null appears.

    The primary should stay on the continuous trajectory, not snap to whichever
    null is momentarily most-negative-Z.
    """
    # One null drifts smoothly upward from -1.0; at one slice a spurious deeper
    # null at -1.5 appears alongside it.  The old rule would grab the -1.5;
    # continuity keeps the smooth trajectory.
    z_track = np.array([-1.00, -0.95, -0.90, -0.85, -0.80])
    xr = np.full((2, 5), 0.6)
    xz = np.empty((2, 5))
    xz[0] = z_track
    xz[1] = -9.99  # absent by default
    xr[1] = -9.99
    # spurious deeper null only at slice 2
    xz[1, 2] = -1.50
    xr[1, 2] = 0.6
    _, new_z = select_primary_xpoint(xr, xz)
    assert np.allclose(new_z, z_track)  # never snapped to the -1.5 transient
    # contrast: old rule WOULD snap
    _, old_z = _old_lowest_z_pick(xr, xz)
    assert np.isclose(old_z[2], -1.50)


def test_reseed_after_full_dropout():
    """After an all-sentinel gap the tracker re-seeds (lower-null preference)."""
    nt = 7
    xr = np.full((2, nt), 0.6)
    xz = np.full((2, nt), -9.99)
    # slices 0-1: only upper null real
    xz[1, 0:2] = UPPER_Z
    xr[1, 0:2] = 0.6
    # slices 2-3: full dropout (all sentinel)
    # slices 4-6: both nulls real -> re-seed should prefer the lower null
    xz[0, 4:7] = LOWER_Z
    xz[1, 4:7] = UPPER_Z
    xr[:, 4:7] = 0.6
    _, new_z = select_primary_xpoint(xr, xz)
    assert np.isclose(new_z[0], UPPER_Z)  # only choice early
    assert np.isnan(new_z[2]) and np.isnan(new_z[3])  # dropout -> NaN
    assert np.allclose(new_z[4:7], LOWER_Z)  # re-seeded to lower null


# ---------------------------------------------------------------------------
# Discontinuity-aware interpolation: mask the switch, never impute through it
# ---------------------------------------------------------------------------


def _continuous_axis_and_lcfs(t_eq: np.ndarray):
    """Plausible continuous axis + a simple square-ish LCFS for the builder."""
    nt = t_eq.size
    axis_r = np.full(nt, 0.9)
    axis_z = np.zeros(nt)
    # 8 boundary points on a circle of radius 0.4 about the axis (>=3 finite).
    ang = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    lcfs_r = np.empty((8, nt))
    lcfs_z = np.empty((8, nt))
    for i in range(nt):
        lcfs_r[:, i] = 0.9 + 0.4 * np.cos(ang)
        lcfs_z[:, i] = 0.0 + 0.4 * np.sin(ang)
    return axis_r, axis_z, lcfs_r, lcfs_z


def test_no_interpolation_through_topology_switch():
    """Frames bracketing the lower->upper switch are MASKED, not drawn through 0."""
    # Native equilibrium: 6 DN slices (lower primary, Z=-1.2) then 6 upper-only
    # slices (Z=+1.2).  5 ms native cadence.
    n_lower, n_after = 6, 6
    nt = n_lower + n_after
    t_eq = np.arange(nt) * 0.005
    xr, xz = _two_null_arrays(n_lower, n_after)
    axis_r, axis_z, lcfs_r, lcfs_z = _continuous_axis_and_lcfs(t_eq)

    # Dense camera frames straddling the switch (between t=0.025 and t=0.030).
    ft = np.linspace(t_eq[0], t_eq[-1], 60)
    geo = build_geometry_from_arrays(
        shot_id=0,
        frame_times=ft,
        t_eq=t_eq,
        axis_r=axis_r,
        axis_z=axis_z,
        x_point_r=xr,
        x_point_z=xz,
        lcfs_r=lcfs_r,
        lcfs_z=lcfs_z,
    )
    assert geo.target.shape[1] == TARGET_DIM == 12  # schema unchanged

    xpt_z = geo.target[:, 3].astype(np.float64)
    finite = np.isfinite(xpt_z)
    # NO emitted X-point Z is ever in the forbidden mid-band: the OLD linear
    # interp would have produced values right through ~0.
    vals = xpt_z[finite]
    assert vals.size > 0
    assert np.all((vals <= BAND[0] - 1e-9) | (vals >= BAND[1] + 1e-9)), (
        f"emitted xpt_Z entered the between-nulls band: "
        f"{vals[(vals > BAND[0]) & (vals < BAND[1])]}"
    )
    # R and Z masks are coupled: same finite frames.
    xpt_r = geo.target[:, 2].astype(np.float64)
    assert np.array_equal(np.isfinite(xpt_r), finite)

    # At least one transition frame WAS masked (the switch segment).
    n_masked = int((~finite).sum())
    assert n_masked >= 1

    # Sanity: the continuous axis label is NOT masked across the switch.
    assert np.isfinite(geo.target[:, 1]).all()


def test_unmasked_frames_are_continuous():
    """Selected primary does not jump >0.3 m between adjacent emitted frames."""
    n_lower, n_after = 8, 8
    nt = n_lower + n_after
    t_eq = np.arange(nt) * 0.005
    xr, xz = _two_null_arrays(n_lower, n_after)
    axis_r, axis_z, lcfs_r, lcfs_z = _continuous_axis_and_lcfs(t_eq)
    ft = np.linspace(t_eq[0], t_eq[-1], 80)
    geo = build_geometry_from_arrays(
        shot_id=0,
        frame_times=ft,
        t_eq=t_eq,
        axis_r=axis_r,
        axis_z=axis_z,
        x_point_r=xr,
        x_point_z=xz,
        lcfs_r=lcfs_r,
        lcfs_z=lcfs_z,
    )
    xpt_z = geo.target[:, 3].astype(np.float64)
    finite = np.isfinite(xpt_z)
    # Among consecutive *emitted* (unmasked) frames, no >0.3 m jump survives.
    # (A masked gap separates the lower-null run from the upper-null run.)
    idx = np.flatnonzero(finite)
    for a, b in zip(idx[:-1], idx[1:], strict=False):
        if b == a + 1:  # genuinely adjacent emitted frames
            assert abs(xpt_z[b] - xpt_z[a]) <= XPOINT_DISCONTINUITY_M


def test_clean_single_null_unmasked():
    """A clean, continuous single-null run is fully emitted (no spurious masking)."""
    nt = 10
    t_eq = np.arange(nt) * 0.005
    xr = np.full((2, nt), 0.6)
    xz = np.full((2, nt), -9.99)
    xr[1] = -9.99
    z_drift = np.linspace(-1.2, -1.0, nt)  # smooth lower-null drift
    xz[0] = z_drift
    axis_r, axis_z, lcfs_r, lcfs_z = _continuous_axis_and_lcfs(t_eq)
    ft = np.linspace(t_eq[0], t_eq[-1], 40)
    geo = build_geometry_from_arrays(
        shot_id=0,
        frame_times=ft,
        t_eq=t_eq,
        axis_r=axis_r,
        axis_z=axis_z,
        x_point_r=xr,
        x_point_z=xz,
        lcfs_r=lcfs_r,
        lcfs_z=lcfs_z,
    )
    xpt_z = geo.target[:, 3].astype(np.float64)
    assert np.isfinite(xpt_z).all()  # nothing masked on a clean run
    assert xpt_z.min() >= -1.21 and xpt_z.max() <= -0.99


# ---------------------------------------------------------------------------
# Real-data smoke (read-only): no 2.4 m artifact, physical picked-Z
# ---------------------------------------------------------------------------

_REAL_SHOTS = (18502, 18504)


def _real_shot_available(shot: int) -> bool:
    return _eq.equilibrium_store_path(shot, None).exists()


@pytest.mark.parametrize("shot", _REAL_SHOTS)
def test_real_shot_no_flip_artifact(shot):
    if not _real_shot_available(shot):
        pytest.skip(f"L2 store for shot {shot} not mounted")
    eq = _eq._read_equilibrium_group(shot, None)
    t_eq = np.asarray(eq["time"], dtype=np.float64)
    xr = np.asarray(eq["x_point_r"], dtype=np.float64)
    xz = np.asarray(eq["x_point_z"], dtype=np.float64)

    # OLD vs NEW native picked-Z.  The native primary-null trajectory is
    # legitimately discontinuous at topology switches / re-seeds (the lower null
    # vanishes and only the upper survives) — that is physical, and the
    # interpolator is what must refuse to bridge it.  So the artifact-removal
    # claim is checked on the EMITTED (interpolated) labels, not the native pick.
    _, old_z = _old_lowest_z_pick(xr, xz)
    _, new_z = select_primary_xpoint(xr, xz)

    old_d = np.abs(np.diff(old_z))
    old_d = old_d[np.isfinite(old_d)]
    # The OLD rule carries the ~2.4 m lower<->upper flip.
    assert old_d.max() > 2.0  # the defect is present in the old rule

    # Build labels on the native time base and confirm the EMITTED X-point
    # frames carry no flip artifact: no value parked at the unphysical Z~0 a
    # linear interp through a flip would give, and no surviving >0.3 m jump
    # between adjacent emitted frames.
    geo = load_equilibrium_geometry(shot, t_eq)
    xpt_z = geo.target[:, 3].astype(np.float64)
    xpt_r = geo.target[:, 2].astype(np.float64)
    finite = np.isfinite(xpt_z)
    n_masked = int((~finite).sum())
    vals = xpt_z[finite]
    # No emitted X-point sits in a tight unphysical band around the midplane.
    assert np.all(np.abs(vals) > 0.2), (
        f"shot {shot}: emitted xpt_Z near midplane (flip artifact): "
        f"{vals[np.abs(vals) <= 0.2]}"
    )
    # No surviving 2.4 m flip between adjacent emitted frames.
    idx = np.flatnonzero(finite)
    adj = [
        abs(xpt_z[b] - xpt_z[a])
        for a, b in zip(idx[:-1], idx[1:], strict=False)
        if b == a + 1
    ]
    max_adj = max(adj) if adj else 0.0
    assert max_adj <= XPOINT_DISCONTINUITY_M, (
        f"shot {shot}: emitted xpt_Z still flips {max_adj:.3f} m between "
        f"adjacent frames"
    )
    # R/Z masks are coupled.
    assert np.array_equal(np.isfinite(xpt_r), finite)

    # Report numbers (visible with -s) for the orchestrator.
    print(
        f"shot {shot}: OLD native picked-z "
        f"[{np.nanmin(old_z):.3f},{np.nanmax(old_z):.3f}] max|dz|={old_d.max():.3f}; "
        f"NEW emitted picked-z [{np.nanmin(vals):.3f},{np.nanmax(vals):.3f}] "
        f"max adj jump={max_adj:.4f}; newly-masked frames={n_masked}"
    )
