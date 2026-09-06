"""Tests for the paired fast-Dalpha conditioning verdict."""

from __future__ import annotations

import json

import pytest

from imas_ambix.camdyn.da_burst_verdict import (
    REGISTERED_FRAME_COUNT,
    REGISTERED_MAX_CANDIDATES,
    RUN_DIRECTORIES,
    build_census,
    build_verdict,
    census_report_line,
    paired_census_records,
    records_from_evaluation,
    shot_clustered_delta,
    verdict_line,
    write_figure,
)


def _records(
    morphology: float,
    nll: float,
    *,
    morphology_step: float = 0.0,
    nll_step: float = 0.0,
):
    rows = []
    for shot_id in (24001, 24002, 24003):
        for frame in (0, 1):
            rows.append(
                {
                    "shot_id": shot_id,
                    "frame_key": f"{shot_id}:frame-{frame}",
                    "actual_horizon_ms": 10.0,
                    "morphology_fidelity": morphology + morphology_step * frame,
                    "token_nll": nll + nll_step * frame,
                }
            )
    return rows


def _six_run_records(*, null_slow_dynamics: bool = False):
    records = {}
    for arm in RUN_DIRECTORIES:
        native_morph = 0.70 if arm == "dynamics" else 0.60
        native_nll = 1.00 if arm == "dynamics" else 1.10
        records[f"{arm}_native"] = _records(native_morph, native_nll)
        records[f"{arm}_shuffled"] = _records(native_morph - 0.10, native_nll + 0.20)
        slow_gap = 0.0 if null_slow_dynamics and arm == "dynamics" else 0.08
        records[f"{arm}_slow"] = _records(
            native_morph - slow_gap,
            native_nll + 2.0 * slow_gap,
        )
    return records


def _registered_frames():
    return [
        {
            "shot_id": 24001 + index,
            "frame_key": f"frame-{index}",
            "dalpha_burst_time_s": 0.1 * index,
            "actual_horizon_ms": 10.0,
        }
        for index in range(4)
    ]


def _census_records():
    records = {}
    for arm in RUN_DIRECTORIES:
        for variant in RUN_DIRECTORIES[arm]:
            run_key = f"{arm}_{variant}"
            records[run_key] = [
                {
                    "shot_id": frame["shot_id"],
                    "frame_key": frame["frame_key"],
                    "actual_horizon_ms": frame["actual_horizon_ms"],
                    "morphology_fidelity": 0.5,
                    "token_nll": 1.0,
                }
                for frame in _registered_frames()
            ]
    records["dynamics_slow"] = (
        records["dynamics_slow"][:1] + records["dynamics_slow"][2:]
    )
    records["baseline_shuffled"][2]["actual_horizon_ms"] = 13.0
    records["baseline_slow"][3]["actual_horizon_ms"] = 9.5
    return records


def test_shot_clustered_delta_preserves_frame_and_cluster_counts():
    native = _records(0.8, 1.0, morphology_step=0.02)
    control = _records(0.6, 1.3, morphology_step=0.02)

    morph = shot_clustered_delta(
        native,
        control,
        "morphology_fidelity",
        lower_is_better=False,
    )
    nll = shot_clustered_delta(
        native,
        control,
        "token_nll",
        lower_is_better=True,
    )

    assert morph["mean"] == pytest.approx(0.2)
    assert morph["lo"] == pytest.approx(0.2)
    assert morph["favours_native"] is True
    assert morph["n_elm_frames"] == 6
    assert morph["n_shots"] == 3
    assert morph["bootstrap_unit"] == "shot mean"
    assert nll["mean"] == pytest.approx(-0.3)
    assert nll["favours_native"] is True


def test_shot_clustered_delta_refuses_unpaired_frames():
    native = _records(0.8, 1.0)
    control = _records(0.6, 1.3)[:-1]

    with pytest.raises(ValueError, match="exactly paired"):
        shot_clustered_delta(
            native,
            control,
            "morphology_fidelity",
            lower_is_better=False,
        )


def test_build_verdict_requires_both_metrics_and_controls_in_both_arms():
    verdict = build_verdict(_six_run_records())

    assert verdict["fast_dalpha_beats_both_controls"] is True
    for arm in RUN_DIRECTORIES:
        arm_result = verdict["arms"][arm]
        assert arm_result["fast_dalpha_beats_both_controls"] is True
        assert (
            arm_result["comparisons"]["native_minus_shuffled"]["native_beats_control"]
            is True
        )
        assert (
            arm_result["comparisons"]["native_minus_slow"]["native_beats_control"]
            is True
        )


def test_build_verdict_preserves_a_null_control_result():
    verdict = build_verdict(_six_run_records(null_slow_dynamics=True))

    comparison = verdict["arms"]["dynamics"]["comparisons"]["native_minus_slow"]
    assert comparison["morphology"]["clear_of_zero"] is False
    assert comparison["token_nll"]["clear_of_zero"] is False
    assert comparison["native_beats_control"] is False
    assert verdict["fast_dalpha_beats_both_controls"] is False


def test_records_from_evaluation_accepts_nested_frame_records():
    payload = {
        "held_out": {
            "elm_morphology": {
                "per_window": [
                    {
                        "shot_id": 24065,
                        "window_index": 4,
                        "actual_horizon_ms": 9.8,
                        "metrics": {
                            "morphology_fidelity": 0.42,
                            "edge_divertor_nll": 1.23,
                        },
                    }
                ]
            }
        }
    }

    records = records_from_evaluation(payload)

    assert records == [
        {
            "shot_id": 24065,
            "frame_key": "24065:window-4",
            "actual_horizon_ms": 9.8,
            "morphology_fidelity": 0.42,
            "token_nll": 1.23,
        }
    ]


def test_aggregate_only_evaluation_cannot_supply_paired_records():
    payload = {
        "held_out": {
            "masked_nll": {"mean": 10.1, "std": 2.0, "n": 700_000},
            "n_scored_tokens": 700_000,
        }
    }

    assert records_from_evaluation(payload) is None


def test_census_accounts_every_registered_frame_and_checkpoint_disposition():
    records = _census_records()

    census = build_census(_registered_frames(), records)

    assert census["registered_frame_count"] == 4
    assert census["scored_frame_count"] == 1
    assert census["disposition_counts"] == {
        "scored": 1,
        "six_way_pairing_failure": 1,
        "horizon_tolerance_failure": 1,
        "checkpoint_disagreement": 1,
    }
    assert census["frames"][1]["missing_checkpoints"] == ["dynamics_slow"]
    assert census["frames"][1]["checkpoint_disagreement"] is True
    assert census["frames"][2]["registered_horizon_within_tolerance"] is True
    assert (
        census["frames"][2]["checkpoint_horizon_within_tolerance"]["baseline_shuffled"]
        is False
    )
    assert census["frames"][3]["checkpoint_disagreement"] is True
    assert census["frames"][3]["six_way_paired"] is True


def test_paired_census_records_keeps_only_scored_frame_keys():
    records = _census_records()
    census = build_census(_registered_frames(), records)

    paired = paired_census_records(records, census)

    assert set(paired) == set(records)
    assert {len(rows) for rows in paired.values()} == {1}
    assert {rows[0]["frame_key"] for rows in paired.values()} == {"frame-0"}


def test_registered_census_defaults_cover_the_powered_population():
    assert REGISTERED_MAX_CANDIDATES == 1_050
    assert REGISTERED_FRAME_COUNT == 59


def test_figure_and_report_line_cover_the_six_runs(tmp_path):
    verdict = build_verdict(_six_run_records())
    payload = {
        **verdict,
        "elm_frame_count": 6,
        "shot_count": 3,
    }
    path = write_figure(payload, tmp_path / "verdict.png")
    line = verdict_line(payload)

    assert path.exists()
    assert path.stat().st_size > 1_000
    assert line.startswith("YES —")
    assert "6 paired ELM frames from 3 shots" in line
    assert "\n" not in line
    report = census_report_line(payload)
    assert "Widened census: YES" in report
    assert "Earlier result: NO" in report
    assert "8 paired ELM frames" in report
    assert "\n" not in report


def test_verdict_payload_remains_strict_json_serialisable():
    payload = build_verdict(_six_run_records(null_slow_dynamics=True))

    encoded = json.dumps(payload, allow_nan=False)

    assert '"fast_dalpha_beats_both_controls": false' in encoded
