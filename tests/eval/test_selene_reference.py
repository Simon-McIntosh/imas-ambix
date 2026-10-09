"""Tests for the SELENE PSRC reference loader on the firewalled referee.

The loader reduces a shot's SELENE reconstruction (EDDB ``PSRC``) into the
seam's ``EquilibriumGeometry``, reached through ``read_efit_geometry``'s
``loader=`` argument and inheriting its firewall.  These tests drive it on a
``tmp_path`` cache seeded read-only from the JT-60SA shot 100599 PSRC arrays,
plus a synthetic ``surfABVxp`` boundary in millimetres whose two slices carry
different vertex counts.  Nothing under ``/work`` is written and no test opens
ssh.

They prove four things: the loader's axis equals PSRC's magnetic axis
(``calRp0``/``calZp0``) interpolated at a frame time, not its geometric centre;
both boundary slices resample to finite LCFS radii; a read outside
``evaluator_context`` raises ``FirewallViolation``; and ``judge_geometry``
scores a geometry equal to the loader's output at zero boundary RMS.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from imas_ambix.eval.efit_referee import (
    FirewallViolation,
    _selene_psrc_loader,
    evaluator_context,
    judge_geometry,
    read_efit_geometry,
)

#: The frozen-shot PSRC cache the tests seed from — read-only, never written.
_SRC_PSRC = Path("/work/projects/imas_gpu/jt60sa/100599.zarr/PSRC")
_SHOT = 100599

#: Two frame times inside the cached shot's PSRC range, where the magnetic axis
#: and the geometric centre disagree (so the axis source is testable).
_FRAME_TIMES = (0.05, 0.10)

#: PSRC channels the record carries, plus the geometric centre the loader must
#: never read (seeded only so the negative control can drive it).
_CHANNELS = ("calIp", "calRX", "calZX", "calRp0", "calZp0", "calCCSRc", "calCCSZc")

#: Synthetic boundary: R and Z blocks in millimetres, two slices of differing
#: vertex count spanning one zero-padded 604-wide packed slot.
_BOUNDARY_SLOT = 604
_R_START, _Z_COUNT, _Z_START = 2, 303, 304
_BOUNDARY_COUNTS = (40, 60)
_BOUNDARY_RADIUS_M = 0.4


def _read_source(dname: str) -> tuple[np.ndarray, np.ndarray]:
    """Read one real PSRC channel and its time base, read-only."""
    data = np.asarray(
        zarr.open_array(str(_SRC_PSRC / dname), mode="r")[...], dtype="<f8"
    )
    time = np.asarray(
        zarr.open_array(str(_SRC_PSRC / f"{dname}_time"), mode="r")[...], dtype="<f8"
    ).reshape(-1)
    return data, time


def _seed_cache(tmp_path: Path) -> Path:
    """Seed a writable ``tmp_path`` cache from the real PSRC arrays and a
    synthetic boundary, then return the cache root."""
    root = tmp_path / f"{_SHOT}.zarr"
    category = zarr.open_group(str(root), mode="a").require_group("PSRC")
    for dname in _CHANNELS:
        data, time = _read_source(dname)
        category.create_array(dname, data=data)
        category.create_array(f"{dname}_time", data=time)

    slot, time = _synthetic_boundary()
    category.create_array("surfABVxp", data=slot)
    category.create_array("surfABVxp_time", data=time)
    return tmp_path


def _synthetic_boundary() -> tuple[np.ndarray, np.ndarray]:
    """A packed ``surfABVxp`` with two slices of different vertex counts.

    Each slice is a circle in millimetres centred on the shot's magnetic axis at
    that time, so ``resample_lcfs_radii`` has a full angle coverage to cut.
    """
    axis_r = _read_source("calRp0")[0].reshape(-1)
    axis_z = _read_source("calZp0")[0].reshape(-1)
    times = np.asarray(_FRAME_TIMES, dtype="<f8")
    time_base = _read_source("calRp0")[1]

    slot = np.zeros((_BOUNDARY_SLOT, len(_FRAME_TIMES)), dtype="<f8")
    for col, (count, t) in enumerate(zip(_BOUNDARY_COUNTS, times, strict=True)):
        r_axis = np.interp(t, time_base, axis_r)
        z_axis = np.interp(t, time_base, axis_z)
        angle = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
        r_m = r_axis + _BOUNDARY_RADIUS_M * np.cos(angle)
        z_m = z_axis + _BOUNDARY_RADIUS_M * np.sin(angle)
        slot[1, col] = count
        slot[_R_START : _R_START + count, col] = r_m * 1000.0
        slot[_Z_COUNT, col] = count
        slot[_Z_START : _Z_START + count, col] = z_m * 1000.0
    return slot, times


@pytest.fixture
def cache(tmp_path: Path) -> Path:
    return _seed_cache(tmp_path)


def _loader_output(cache_root: Path):
    frame_times = np.asarray(_FRAME_TIMES, dtype=np.float64)
    with evaluator_context():
        return read_efit_geometry(
            _SHOT, frame_times, loader=_selene_psrc_loader, cache_root=cache_root
        )


def test_axis_is_psrc_magnetic_axis_not_geometric_centre(cache: Path):
    """The loader's axis is ``calRp0``/``calZp0`` interpolated at each frame."""
    geom = _loader_output(cache)
    axis_r, axis_r_time = _read_source("calRp0")
    axis_z, _ = _read_source("calZp0")
    centre_r, _ = _read_source("calCCSRc")
    for frame, t in enumerate(_FRAME_TIMES):
        r_interp = np.interp(t, axis_r_time, axis_r.reshape(-1))
        z_interp = np.interp(t, axis_r_time, axis_z.reshape(-1))
        assert geom.target[frame, 0] == pytest.approx(r_interp, abs=1e-4)
        assert geom.target[frame, 1] == pytest.approx(z_interp, abs=1e-4)
        # ... and specifically NOT the geometric centre.
        centre_interp = np.interp(t, axis_r_time, centre_r.reshape(-1))
        assert geom.target[frame, 0] != pytest.approx(centre_interp, abs=1e-3)


def test_both_slices_resample_to_finite_lcfs_radii(cache: Path):
    """Both boundary slices, with different vertex counts, yield finite radii."""
    geom = _loader_output(cache)
    assert geom.n_frames == 2
    lcfs = geom.target[:, 6:14]
    assert np.isfinite(lcfs).all()
    # Each slice is a 0.4 m circle about the axis, so the radii recover it.
    assert lcfs == pytest.approx(_BOUNDARY_RADIUS_M, abs=2e-3)


def test_read_outside_evaluator_context_raises(cache: Path):
    """The SELENE loader is gated exactly like every other loader."""
    frame_times = np.asarray([_FRAME_TIMES[0]], dtype=np.float64)
    with pytest.raises(FirewallViolation):
        read_efit_geometry(
            _SHOT, frame_times, loader=_selene_psrc_loader, cache_root=cache
        )


def test_judge_scores_loader_output_against_itself_at_zero_boundary_rms(cache: Path):
    geom = _loader_output(cache)
    verdict = judge_geometry(geom, geom)
    assert verdict.boundary_rms == pytest.approx(0.0)
    assert verdict.n_boundary_points == 8
    assert verdict.axis_error == pytest.approx(0.0, abs=1e-6)
