from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from imas_ambix.worldmodel.axis_offset_diagnostic import (
    AxisOffsetSlice,
    aggregate_slices,
    classify_offset,
    conditioned_solve_applied,
    point_distance_cm,
    signed_axis_offset_cm,
    write_diagnostic,
)


def _slice(
    d_r_cm: float,
    d_z_cm: float,
    *,
    nova_centroid_cm: float = 5.0,
    efit_centroid_cm: float = 1.0,
) -> AxisOffsetSlice:
    return AxisOffsetSlice(
        manifest_row=4,
        session_index=3,
        time_s=0.2,
        conditioned_recorded=False,
        conditioned_applied=False,
        conditioned_flag_corrected=False,
        conditioned_branch_guard_ok=False,
        flat_top=True,
        evidence_eligible=True,
        nova_axis_r_m=0.94,
        nova_axis_z_m=0.08,
        efit_axis_r_m=0.90,
        efit_axis_z_m=0.10,
        current_centroid_r_m=0.90,
        current_centroid_z_m=0.10,
        d_r_cm=d_r_cm,
        d_z_cm=d_z_cm,
        axis_offset_cm=(d_r_cm**2 + d_z_cm**2) ** 0.5,
        nova_axis_to_current_centroid_cm=nova_centroid_cm,
        efit_axis_to_current_centroid_cm=efit_centroid_cm,
        exclusion_reason=None,
    )


def test_planted_signed_offset_recovers_components_and_magnitude():
    d_r_cm, d_z_cm = signed_axis_offset_cm(0.94, 0.08, 0.90, 0.10)

    assert d_r_cm == pytest.approx(4.0)
    assert d_z_cm == pytest.approx(-2.0)
    assert point_distance_cm(0.94, 0.08, 0.90, 0.10) == pytest.approx(5.0**0.5 * 2.0)


def test_legacy_conditioned_flag_is_corrected_only_before_zero_trip_solve():
    failed_derivation = {
        "exception": "NoQualifiedAxisError: no qualified candidate",
        "conditioned_trips": 0,
    }
    attempted_solve = {"exception": None, "conditioned_trips": 2}

    assert conditioned_solve_applied(failed_derivation, True) == (False, True)
    assert conditioned_solve_applied(attempted_solve, True) == (True, False)
    assert conditioned_solve_applied(failed_derivation, False) == (False, False)


def test_aggregate_preserves_signs_and_centroid_referents():
    summary = aggregate_slices(
        [_slice(4.0, -1.0), _slice(5.0, -2.0), _slice(6.0, -3.0)]
    )

    assert summary["evidence_slice_count"] == 3
    assert summary["dR_cm"] == {
        "count": 3,
        "mean": 5.0,
        "median": 5.0,
        "std": pytest.approx((2.0 / 3.0) ** 0.5),
        "positive_fraction": 1.0,
    }
    assert summary["dZ_cm"]["mean"] == -2.0
    assert summary["nova_axis_to_current_centroid_cm"]["median"] == 5.0
    assert summary["efit_axis_to_current_centroid_cm"]["median"] == 1.0
    classification, verdict = classify_offset(summary)
    assert classification == "outboard_displacement"
    assert "outboard" in verdict


def test_diagnostic_writer_creates_json_and_figure(tmp_path):
    slices = [_slice(4.0, -1.0), _slice(5.0, -1.5)]
    summary = aggregate_slices(slices)
    classification, verdict = classify_offset(summary)
    diagnostic = {
        "classification": classification,
        "verdict": verdict,
        "aggregate": summary,
        "shots": [
            {
                "shot_id": 12345,
                "summary": summary,
                "slices": [
                    {
                        **asdict(item),
                        "dR_cm": item.d_r_cm,
                        "dZ_cm": item.d_z_cm,
                    }
                    for item in slices
                ],
            }
        ],
    }

    json_path, figure_path = write_diagnostic(diagnostic, tmp_path)

    assert json.loads(json_path.read_text())["classification"] == (
        "outboard_displacement"
    )
    assert figure_path.stat().st_size > 10_000
