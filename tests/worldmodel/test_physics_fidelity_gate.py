from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from imas_ambix.worldmodel import equilibrium_labels
from imas_ambix.worldmodel.physics_fidelity_gate import (
    AXIS_LIMIT_CM,
    BOUNDARY_LIMIT_CM,
    DEFAULT_SESSION_ROOT,
    SliceFidelity,
    _aggregate_slices,
    axis_offset_m,
    flat_top_mask_from_current,
    radius_rms_distance_m,
    score_shot,
)


def test_known_radial_offset_is_measured_in_metres():
    angle = np.linspace(0.0, 2.0 * np.pi, 512, endpoint=False)
    nova_r = 0.9 + 0.52 * np.cos(angle)
    nova_z = 0.1 + 0.52 * np.sin(angle)
    nova_radii = equilibrium_labels.resample_lcfs_radii(
        nova_r,
        nova_z,
        0.9,
        0.1,
        equilibrium_labels.LCFS_ANGLES,
    )
    efit_radii = np.full(equilibrium_labels.N_LCFS_ANGLES, 0.50)
    efit_mask = np.ones(equilibrium_labels.N_LCFS_ANGLES, dtype=bool)
    efit_mask[-1] = False

    assert radius_rms_distance_m(nova_radii, efit_radii, efit_mask) == pytest.approx(
        0.02, abs=2.0e-5
    )
    assert axis_offset_m(0.93, 0.14, 0.90, 0.10) == pytest.approx(0.05)


def test_flat_top_is_eighty_percent_of_max_absolute_efit_current():
    current_times = np.arange(5, dtype=np.float64)
    current = np.asarray([0.0, -40.0, -100.0, -79.9, 0.0])
    slice_times = np.asarray([1.0, 2.0, 2.5, 3.0, 5.0])

    mask, receipt = flat_top_mask_from_current(current_times, current, slice_times)

    assert mask.tolist() == [False, True, True, False, False]
    assert receipt == {
        "peak_current_a": 100.0,
        "flat_top_threshold_a": 80.0,
    }


def _slice(
    *,
    conditioned: bool,
    flat_top: bool,
    boundary_cm: float,
    axis_cm: float,
) -> SliceFidelity:
    eligible = flat_top and not conditioned
    boundary_pass = boundary_cm <= BOUNDARY_LIMIT_CM
    axis_pass = axis_cm <= AXIS_LIMIT_CM
    return SliceFidelity(
        manifest_row=0,
        session_index=0,
        time_s=0.2,
        conditioned=conditioned,
        conditioned_branch_guard_ok=True,
        flat_top=flat_top,
        evidence_eligible=eligible,
        boundary_rms_cm=boundary_cm,
        axis_offset_cm=axis_cm,
        boundary_within_limit=boundary_pass,
        axis_within_limit=axis_pass,
        joint_within_limits=boundary_pass and axis_pass,
        nova_solve_wall_seconds=0.25,
        exclusion_reason=None if eligible else "excluded",
    )


def test_aggregate_excludes_conditioned_and_non_flat_top_slices():
    slices = [
        _slice(conditioned=False, flat_top=True, boundary_cm=1.0, axis_cm=1.0),
        _slice(conditioned=False, flat_top=True, boundary_cm=3.0, axis_cm=1.0),
        _slice(conditioned=True, flat_top=True, boundary_cm=1.0, axis_cm=1.0),
        _slice(conditioned=False, flat_top=False, boundary_cm=1.0, axis_cm=1.0),
    ]

    result = _aggregate_slices(slices)

    assert result["converged_slice_count"] == 4
    assert result["conditioned_slice_count"] == 1
    assert result["flat_top_slice_count"] == 3
    assert result["flat_top_time_start_s"] == 0.2
    assert result["flat_top_time_end_s"] == 0.2
    assert result["evidence_slice_count"] == 2
    assert result["excluded_conditioned_flat_top_count"] == 1
    assert result["boundary_pass_fraction"] == 0.5
    assert result["axis_pass_fraction"] == 1.0
    assert result["joint_pass_fraction"] == 0.5
    assert result["passed"] is False


@pytest.mark.skipif(
    not (Path(DEFAULT_SESSION_ROOT) / "22086.nc").is_file(),
    reason="real carrier session is unavailable",
)
def test_real_carrier_shot_has_finite_scored_geometry():
    result = score_shot(22086)

    summary = result["summary"]
    assert summary["converged_slice_count"] == 46
    assert summary["conditioned_slice_count"] == 3
    assert summary["evidence_slice_count"] > 0
    assert summary["flat_top_time_start_s"] is not None
    assert summary["flat_top_time_end_s"] >= summary["flat_top_time_start_s"]
    assert summary["boundary_rms_cm"]["count"] == summary["evidence_slice_count"]
    assert summary["axis_offset_cm"]["count"] == summary["evidence_slice_count"]
    assert np.isfinite(summary["boundary_rms_cm"]["mean"])
    assert np.isfinite(summary["axis_offset_cm"]["mean"])
    assert all("conditioned" in item for item in result["slices"])
    assert all("nova_solve_wall_seconds" in item for item in result["slices"])
