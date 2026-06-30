"""Tests for the ORDER-INVARIANT NULL-SET X-point equilibrium labels.

The 14-D equilibrium labels feed the firewalled diagnostics->equilibrium oracle.
The X-point is an ORDER-INVARIANT SET of ≤2 real in-vessel nulls (DETR-style):
no ordering, no sign-of-Z assignment, no ψ public/private filter, no
single-primary continuity — just the set of valid nulls at the window-centre
native slice (nearest-native, no member interpolation → no flip/interp).  A
coarse finite + in-vessel-bbox sanity reject drops only reconstruction
artefacts (NOT a public/private discriminator).  These tests pin:

  - the 14-D set schema + names (axis, xpt0/xpt1 unordered slots, LCFS);
  - count/presence handling (SN→1, DN→2, limiter/off→0) + masks;
  - NO-FLIP on a synthetic lower→upper topology switch (the set is stable; no
    value forced between the nulls);
  - the coarse in-vessel sanity reject (a synthetic far/out-of-vessel null is
    dropped) — explicitly NOT a public/private claim;
  - real-data smoke (18502/18504): count distribution + nearest-native sampling.

Order-invariance of the matched LOSS is tested in test_spacetime_diag_probe.py
(it needs torch); these label tests are pure-numpy.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

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
N_XPOINT_SLOTS = _eq.N_XPOINT_SLOTS
XPOINT_SENTINEL = _eq.XPOINT_SENTINEL
build_geometry_from_arrays = _eq.build_geometry_from_arrays
load_equilibrium_geometry = _eq.load_equilibrium_geometry
xpoint_null_set = _eq.xpoint_null_set

LOWER_Z = -1.2
UPPER_Z = +1.2
_NAME_IDX = {n: i for i, n in enumerate(TARGET_NAMES)}


def _two_null_arrays(n_lower: int, n_after: int, *, sentinel: float = -9.99):
    """``(2, nt)`` x_point arrays for a DN run that loses its lower null.

    Row 0 is the lower null (Z=LOWER_Z), row 1 the upper null (Z=UPPER_Z).  For
    the first ``n_lower`` slices BOTH are real (double-null); for the remaining
    ``n_after`` the lower null drops to the sentinel (single-null upper).
    """
    nt = n_lower + n_after
    xr = np.full((2, nt), 0.55, dtype=np.float64)
    xz = np.empty((2, nt), dtype=np.float64)
    xz[0, :] = LOWER_Z
    xz[1, :] = UPPER_Z
    xr[0, n_lower:] = sentinel
    xz[0, n_lower:] = sentinel
    return xr, xz


def _continuous_axis_and_lcfs(t_eq: np.ndarray):
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


def test_null_set_schema_is_14d():
    assert TARGET_DIM == 14
    assert N_XPOINT_SLOTS == 2
    assert TARGET_NAMES[2:6] == ("xpt0_R", "xpt0_Z", "xpt1_R", "xpt1_Z")


# ---------------------------------------------------------------------------
# Set extraction: count / presence, NO ordering / sign-of-Z
# ---------------------------------------------------------------------------


def test_xpoint_null_set_count_and_no_ordering():
    """DN → 2 slots, SN → 1 slot; slot index is store order, not sign-of-Z."""
    xr, xz = _two_null_arrays(n_lower=4, n_after=4)
    set_r, set_z = xpoint_null_set(xr, xz)
    assert set_r.shape == (2, 8)
    # DN slices: both slots present.
    dn = np.isfinite(set_z[0, :4]) & np.isfinite(set_z[1, :4])
    assert dn.all()
    # SN slices (lower dropped): exactly one slot present (packed into slot 0).
    sn0 = np.isfinite(set_z[0, 4:])
    sn1 = np.isfinite(set_z[1, 4:])
    assert (sn0 & ~sn1).all()
    # slot 0 in the DN run is the store's row-0 (here the lower null) — the slot
    # index is store order, NOT a Z-sign assignment.  That is the point.
    assert np.allclose(set_z[0, :4], LOWER_Z)


def test_coarse_in_vessel_reject_not_public_private():
    """A finite but OUT-OF-VESSEL null is dropped; an in-vessel one is kept.

    This is a coarse reconstruction-artefact reject, NOT a public/private claim.
    """
    # row 0: a normal in-vessel null; row 1: a far/out-of-vessel null (R=5 m).
    xr = np.array([[0.55], [5.0]])
    xz = np.array([[-1.2], [0.3]])
    set_r, _set_z = xpoint_null_set(xr, xz)
    assert np.isfinite(set_r[0, 0]) and set_r[0, 0] == pytest.approx(0.55)
    assert not np.isfinite(set_r[1, 0])  # out-of-vessel rejected
    # a sentinel null is also dropped.
    xr2 = np.array([[0.55], [-9.99]])
    xz2 = np.array([[-1.2], [-9.99]])
    sr2, _ = xpoint_null_set(xr2, xz2)
    assert np.isfinite(sr2[0, 0]) and not np.isfinite(sr2[1, 0])


# ---------------------------------------------------------------------------
# NO-FLIP across a topology switch (the core correctness property)
# ---------------------------------------------------------------------------


def test_no_flip_on_topology_switch():
    """A lower→upper switch yields a STABLE set; no value forced between nulls."""
    n_lower, n_after = 6, 6
    nt = n_lower + n_after
    t_eq = np.arange(nt) * 0.005
    xr, xz = _two_null_arrays(n_lower, n_after)
    axis_r, axis_z, lcfs_r, lcfs_z = _continuous_axis_and_lcfs(t_eq)
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
    assert geo.target.shape[1] == TARGET_DIM == 14
    for slot in range(N_XPOINT_SLOTS):
        z = geo.target[:, _NAME_IDX[f"xpt{slot}_Z"]].astype(np.float64)
        vals = z[np.isfinite(z)]
        # every emitted null Z is a real null height (±1.2), NEVER the mid-band.
        assert np.all((vals <= -0.8) | (vals >= 0.8)), (
            f"slot {slot} emitted a mid-band Z (flip artifact): "
            f"{vals[(vals > -0.8) & (vals < 0.8)]}"
        )
        r = geo.target[:, _NAME_IDX[f"xpt{slot}_R"]].astype(np.float64)
        assert np.array_equal(np.isfinite(r), np.isfinite(z))  # R/Z masks coupled
    # axis (continuous) is never masked across the switch.
    assert np.isfinite(geo.target[:, _NAME_IDX["axis_Z"]]).all()


# ---------------------------------------------------------------------------
# Real-data smoke
# ---------------------------------------------------------------------------

_REAL_SHOTS = (18502, 18504)


def _real_shot_available(shot: int) -> bool:
    return _eq.equilibrium_store_path(shot, None).exists()


@pytest.mark.parametrize("shot", _REAL_SHOTS)
def test_real_shot_null_set(shot):
    if not _real_shot_available(shot):
        pytest.skip(f"L2 store for shot {shot} not mounted")
    eq = _eq._read_equilibrium_group(shot, None)
    t_eq = np.asarray(eq["time"], dtype=np.float64)
    geo = load_equilibrium_geometry(shot, t_eq)
    s0 = np.isfinite(geo.target[:, _NAME_IDX["xpt0_R"]])
    s1 = np.isfinite(geo.target[:, _NAME_IDX["xpt1_R"]])
    count = s0.astype(int) + s1.astype(int)
    from collections import Counter

    dist = Counter(count.tolist())
    assert set(dist).issubset({0, 1, 2})
    assert dist.get(2, 0) > 0  # MAST is double-null much of the time
    # nearest-native: every emitted null sits at a physical height (|Z|>0.2 m).
    for slot in range(N_XPOINT_SLOTS):
        z = geo.target[:, _NAME_IDX[f"xpt{slot}_Z"]].astype(np.float64)
        zf = z[np.isfinite(z)]
        if zf.size:
            assert np.all(np.abs(zf) > 0.2)
    print(f"shot {shot}: x-point null-set count dist {dict(dist)}")
