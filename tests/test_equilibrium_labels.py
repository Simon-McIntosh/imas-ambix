"""Tests for the SPLIT upper/lower X-point equilibrium labels + psi-filter.

The 14-D equilibrium labels feed the firewalled diagnostics->equilibrium
oracle.  The X-point is SPLIT into a LOWER (Z<0) and an UPPER (Z>0) divertor
channel, each present-when-present and masked-when-absent.  Because they are
separate sign-of-Z channels there is NO lower<->upper flip and NO ~2.4 m
discontinuity (the old single-"primary" target was bimodal across a topology
switch — ill-posed), so each channel is interpolated linearly within its own
presence.  A psi-proximity filter keeps a null only if it is boundary-associated
(``|psi_null - psi_boundary| <= tol * |psi_boundary - psi_axis|``).  These tests
pin:

  - the 14-D split schema + names;
  - each split channel is sign-pure + continuous within its presence (no flip);
  - the psi-filter masks a synthetic far / internal (non-boundary) null;
  - real-data smoke: split channels clean on the topology-switcher 18502/18504,
    and the psi-filter rejection count is reported.
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
sys.modules[_spec.name] = _eq
_spec.loader.exec_module(_eq)

TARGET_DIM = _eq.TARGET_DIM
TARGET_NAMES = _eq.TARGET_NAMES
XPOINT_SENTINEL = _eq.XPOINT_SENTINEL
XPOINT_PSI_TOL = _eq.XPOINT_PSI_TOL
build_geometry_from_arrays = _eq.build_geometry_from_arrays
load_equilibrium_geometry = _eq.load_equilibrium_geometry
select_split_xpoints = _eq.select_split_xpoints

LOWER_Z = -1.2
UPPER_Z = +1.2
_NAME_IDX = {n: i for i, n in enumerate(TARGET_NAMES)}


def _two_null_arrays(n_lower: int, n_after: int, *, sentinel: float = -9.99):
    """``(2, nt)`` x_point arrays for a DN run that loses its lower null.

    Row 0 is the lower null (Z=LOWER_Z), row 1 the upper null (Z=UPPER_Z).  For
    the first ``n_lower`` slices BOTH nulls are real (double-null); for the
    remaining ``n_after`` the lower null drops to the sentinel (single-null
    upper).  R ~ 0.6 m outboard for both.
    """
    nt = n_lower + n_after
    xr = np.full((2, nt), 0.6, dtype=np.float64)
    xz = np.empty((2, nt), dtype=np.float64)
    xz[0, :] = LOWER_Z
    xz[1, :] = UPPER_Z
    xr[0, n_lower:] = sentinel
    xz[0, n_lower:] = sentinel
    return xr, xz


def _continuous_axis_and_lcfs(t_eq: np.ndarray):
    """Plausible continuous axis + a simple circular LCFS for the builder."""
    nt = t_eq.size
    axis_r = np.full(nt, 0.9)
    axis_z = np.zeros(nt)
    ang = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    lcfs_r = np.empty((8, nt))
    lcfs_z = np.empty((8, nt))
    for i in range(nt):
        lcfs_r[:, i] = 0.9 + 0.4 * np.cos(ang)
        lcfs_z[:, i] = 0.0 + 0.4 * np.sin(ang)
    return axis_r, axis_z, lcfs_r, lcfs_z


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_split_schema_is_14d():
    assert TARGET_DIM == 14
    assert TARGET_NAMES[2:6] == (
        "lower_xpt_R",
        "lower_xpt_Z",
        "upper_xpt_R",
        "upper_xpt_Z",
    )


# ---------------------------------------------------------------------------
# Split: each null is its OWN sign-pure, continuous channel (no flip)
# ---------------------------------------------------------------------------


def test_split_channels_are_sign_pure_no_flip():
    """LOWER channel always Z<0, UPPER always Z>0 — no lower<->upper flip.

    The store loses its lower null mid-run; the OLD single-primary target would
    flip ~2.4 m at that switch.  As two SEPARATE sign-of-Z channels each is its
    own present-when-present series and never crosses the midplane.
    """
    xr, xz = _two_null_arrays(n_lower=6, n_after=6)
    lr, lz, ur, uz, _n = select_split_xpoints(xr, xz)
    # lower present for the DN run only, always negative Z.
    lz_fin = lz[np.isfinite(lz)]
    uz_fin = uz[np.isfinite(uz)]
    assert lz_fin.size == 6 and np.all(lz_fin < 0)
    assert uz_fin.size == 12 and np.all(uz_fin > 0)  # upper present throughout
    # NO single emitted channel ever jumps lower<->upper (each stays on a side).
    assert np.allclose(lz_fin, LOWER_Z)
    assert np.allclose(uz_fin, UPPER_Z)


def test_split_channels_interpolate_linearly_within_presence():
    """Each channel is continuous within presence -> linear interp, masked gaps."""
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
    assert geo.target.shape[1] == TARGET_DIM == 14
    lz = geo.target[:, _NAME_IDX["lower_xpt_Z"]].astype(np.float64)
    uz = geo.target[:, _NAME_IDX["upper_xpt_Z"]].astype(np.float64)
    # lower present only over the DN time span, sign-pure; upper present whole.
    assert np.all(lz[np.isfinite(lz)] < 0)
    assert np.all(uz[np.isfinite(uz)] > 0)
    # the lower channel is masked after its null vanishes (no extrapolation).
    assert np.isnan(lz[-1])
    # upper channel fully present (clean continuous single null) — no spurious mask.
    assert np.isfinite(uz).all()
    # R/Z masks coupled per channel.
    lr = geo.target[:, _NAME_IDX["lower_xpt_R"]].astype(np.float64)
    assert np.array_equal(np.isfinite(lr), np.isfinite(lz))


# ---------------------------------------------------------------------------
# psi-proximity boundary-null filter
# ---------------------------------------------------------------------------


def _psi_fixture():
    """A simple axisymmetric psi(z, r) bowl: max at axis, decreasing outward.

    psi(R,Z) = -((R-R0)^2 + (Z-Z0)^2); axis at (R0, Z0) is the maximum (0).  The
    LCFS at radius 0.4 sits at psi = -0.16; a boundary null also at radius 0.4
    has psi ~ -0.16 (kept); an internal null near the axis has psi ~ 0 (far from
    psi_boundary -> rejected).
    """
    R0, Z0 = 0.9, 0.0
    r_axis = np.linspace(0.2, 1.8, 81)
    z_axis = np.linspace(-1.6, 1.6, 81)
    RR, ZZ = np.meshgrid(r_axis, z_axis)  # (nz, nr)
    psi = -((RR - R0) ** 2 + (ZZ - Z0) ** 2)
    return r_axis, z_axis, psi, R0, Z0


def test_psi_filter_masks_a_non_boundary_null():
    """A far / internal null (psi != psi_boundary) is rejected; a boundary null kept."""
    r_axis, z_axis, psi2d, R0, Z0 = _psi_fixture()
    psi = psi2d[:, :, None]  # (nz, nr, 1)
    axis_r = np.array([R0])
    axis_z = np.array([Z0])
    # LCFS: a circle of radius 0.4 about the axis (8 points) -> psi_boundary=-0.16.
    ang = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    lcfs_r = (R0 + 0.4 * np.cos(ang))[:, None]
    lcfs_z = (Z0 + 0.4 * np.sin(ang))[:, None]
    # two nulls: row0 a genuine boundary null on the circle BELOW the axis (Z<0,
    # radius 0.4 -> psi=-0.16); row1 an INTERNAL null near the axis (radius 0.05,
    # placed at Z>0 so it would land in the upper channel) -> psi~-0.0025, far
    # from psi_boundary -> must be REJECTED.
    xr = np.array([[R0], [R0 + 0.05]])
    xz = np.array([[Z0 - 0.4], [Z0 + 0.05]])
    lr, lz, ur, uz, n_rej = select_split_xpoints(
        xr,
        xz,
        psi=psi,
        r_axis=r_axis,
        z_axis=z_axis,
        axis_r=axis_r,
        axis_z=axis_z,
        lcfs_r=lcfs_r,
        lcfs_z=lcfs_z,
    )
    # the boundary null (lower) is KEPT; the internal null (upper) is REJECTED.
    assert np.isfinite(lz[0]) and lz[0] < 0
    assert np.isnan(uz[0])
    assert n_rej == 1

    # Without the psi grid the internal null would NOT be filtered (kept upper).
    lr2, lz2, ur2, uz2, n_rej2 = select_split_xpoints(xr, xz)
    assert np.isfinite(uz2[0]) and n_rej2 == 0


# ---------------------------------------------------------------------------
# Real-data smoke: split channels clean on the topology-switchers + reject rate
# ---------------------------------------------------------------------------

_REAL_SHOTS = (18502, 18504)


def _real_shot_available(shot: int) -> bool:
    return _eq.equilibrium_store_path(shot, None).exists()


@pytest.mark.parametrize("shot", _REAL_SHOTS)
def test_real_shot_split_channels_clean(shot):
    if not _real_shot_available(shot):
        pytest.skip(f"L2 store for shot {shot} not mounted")
    eq = _eq._read_equilibrium_group(shot, None)
    t_eq = np.asarray(eq["time"], dtype=np.float64)
    geo, n_rej = build_geometry_from_arrays(
        shot_id=shot,
        frame_times=t_eq,
        t_eq=t_eq,
        axis_r=np.asarray(eq["magnetic_axis_r"], dtype=np.float64),
        axis_z=np.asarray(eq["magnetic_axis_z"], dtype=np.float64),
        x_point_r=np.asarray(eq["x_point_r"], dtype=np.float64),
        x_point_z=np.asarray(eq["x_point_z"], dtype=np.float64),
        lcfs_r=np.asarray(eq["lcfs_r"], dtype=np.float64),
        lcfs_z=np.asarray(eq["lcfs_z"], dtype=np.float64),
        psi=np.asarray(eq["psi"], dtype=np.float64),
        r_axis=np.asarray(eq["major_radius"], dtype=np.float64),
        z_axis=np.asarray(eq["z"], dtype=np.float64),
        return_rejected=True,
    )
    lz = geo.target[:, _NAME_IDX["lower_xpt_Z"]].astype(np.float64)
    uz = geo.target[:, _NAME_IDX["upper_xpt_Z"]].astype(np.float64)
    lz_fin = lz[np.isfinite(lz)]
    uz_fin = uz[np.isfinite(uz)]
    # Sign-pure: lower always below the midplane, upper always above. No flip.
    assert lz_fin.size > 0 and np.all(lz_fin < -0.2)
    assert uz_fin.size > 0 and np.all(uz_fin > 0.2)
    # Each channel continuous within presence — max adjacent step is cm-scale,
    # NOT the ~2.4 m lower<->upper flip the single-primary target produced.
    for ch in (lz, uz):
        idx = np.flatnonzero(np.isfinite(ch))
        adj = [
            abs(ch[b] - ch[a])
            for a, b in zip(idx[:-1], idx[1:], strict=False)
            if b == a + 1
        ]
        assert (max(adj) if adj else 0.0) < 0.3

    # count real store nulls for the rejection rate.
    xr = np.asarray(eq["x_point_r"], dtype=np.float64)
    xz = np.asarray(eq["x_point_z"], dtype=np.float64)
    n_real = int(
        ((xr > XPOINT_SENTINEL) & (xz > XPOINT_SENTINEL) & np.isfinite(xr)).sum()
    )
    print(
        f"shot {shot}: lower present {lz_fin.size}, upper present {uz_fin.size}; "
        f"psi-filter rejected {n_rej}/{n_real} real store nulls "
        f"({100 * n_rej / max(n_real, 1):.1f}%)"
    )
