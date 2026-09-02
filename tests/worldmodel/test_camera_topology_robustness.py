"""Tests for camera-topology robustness scoring and evidence boundaries."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from imas_ambix.worldmodel.camera_topology_labeller import (
    REGRESSION_NAMES,
    CameraTopologyLabeller,
    LabellerConfig,
    LabellerTargets,
    LoadedWindows,
)
from imas_ambix.worldmodel.camera_topology_robustness import (
    REGISTERED_SOURCE_FRAMES,
    RobustnessAuthorities,
    error_growth_verdict,
    fit_brightness_centroid_baseline,
    load_labeller_checkpoint,
    read_divertor_frame_indices,
    read_robustness_authorities,
    run_robustness_scoring,
    score_predictions,
)
from imas_ambix.worldmodel.camera_topology_targets import TOPOLOGY_CLASS_NAMES


def _targets() -> LabellerTargets:
    values = np.array(
        [
            [0.90, 0.00, 0.75, -0.55, 0.35, -0.70, 1.45, -0.70],
            [0.95, 0.05, 0.80, 0.55, 0.40, -0.68, 1.40, -0.68],
            [1.00, 0.02, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        ],
        dtype=np.float32,
    )
    return LabellerTargets(
        values_m=values,
        finite_mask=np.isfinite(values),
        topology_class=np.array([1, 2, 0], dtype=np.int64),
    )


def _metric(
    o_point: float,
    x_point: float,
    strike_point: float,
    accuracy: float,
) -> dict[str, object]:
    return {
        "shot_clustered_position_error_cm": {
            "o_point": o_point,
            "x_point": x_point,
            "strike_point": strike_point,
        },
        "class_accuracy": accuracy,
    }


def test_prediction_metrics_include_strikes_and_zero_support_classes():
    targets = _targets()
    prediction = np.nan_to_num(targets.values_m, nan=0.0)
    prediction[:, 0] += 0.01
    prediction[:2, 2] += 0.02
    prediction[:2, 4] += 0.03
    prediction[:2, 6] += 0.03

    result = score_predictions(
        prediction,
        np.array([1, 0, 0], dtype=np.int64),
        targets,
        shot_id=21989,
    )

    assert np.isclose(result["position_error_cm"]["o_point"], 1.0)
    assert np.isclose(result["position_error_cm"]["x_point"], 2.0)
    assert np.isclose(result["position_error_cm"]["strike_point"], 3.0)
    assert result["position_support"] == {
        "o_point": 3,
        "x_point": 2,
        "strike_point": 4,
    }
    assert np.isclose(result["class_accuracy"], 2.0 / 3.0)
    assert result["per_class"]["disconnected-double-null"] == {
        "support": 0,
        "accuracy": None,
    }


def test_error_growth_verdict_covers_position_and_class_error():
    clean = _metric(1.0, 2.0, 3.0, 0.80)
    passing = _metric(1.5, 2.5, 4.0, 0.70)
    failing = _metric(1.0, 3.1, 3.0, 0.80)

    pass_result = error_growth_verdict(clean, passing)
    fail_result = error_growth_verdict(clean, failing)

    assert pass_result["passed"] is True
    assert np.isclose(pass_result["ratios"]["class_error"], 1.5)
    assert fail_result["passed"] is False
    assert fail_result["ratios"]["x_point"] > 1.5


def test_brightness_baseline_scores_coordinates_and_classes():
    targets = _targets()
    frames = np.zeros((3, 3, 12, 12), dtype=np.float32)
    frames[0, :, 2:4, 2:4] = 0.5
    frames[1, :, 7:9, 7:9] = 0.8
    frames[2, :, 5:7, 5:7] = 0.3
    windows = LoadedWindows(
        frames=frames,
        targets=targets,
        shot_ids=np.array([1, 2, 3]),
        frame_times=np.array([0.1, 0.2, 0.3]),
    )

    baseline = fit_brightness_centroid_baseline(windows)
    coordinates, classes = baseline.predict(frames)

    assert coordinates.shape == (3, len(REGRESSION_NAMES))
    assert classes.shape == (3,)
    assert np.isfinite(coordinates).all()


def test_checkpoint_loader_recovers_companion_training_shots(tmp_path: Path):
    model = CameraTopologyLabeller(
        LabellerConfig(window_frames=3, image_size=16, width=4, hidden=12)
    )
    checkpoint = tmp_path / "labeller.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "window_frames": 3,
                "image_size": 16,
                "width": 4,
                "hidden": 12,
                "n_classes": len(TOPOLOGY_CLASS_NAMES),
            },
            "target_mean_m": np.zeros(len(REGRESSION_NAMES), dtype=np.float32),
            "target_std_m": np.ones(len(REGRESSION_NAMES), dtype=np.float32),
            "regression_names": REGRESSION_NAMES,
            "class_names": TOPOLOGY_CLASS_NAMES,
        },
        checkpoint,
    )
    checkpoint.with_suffix(".json").write_text(
        '{"train_shots": [21983, 21985]}\n', encoding="utf-8"
    )

    loaded = load_labeller_checkpoint(checkpoint)

    assert loaded.training_shots == (21983, 21985)
    assert loaded.model.config.image_size == 16


def test_report_parsers_resolve_every_subset(tmp_path: Path):
    cohort = tmp_path / "cohort.md"
    cohort.write_text(
        "### Labeller train — counts\n```text\n15162:10\n```\n"
        "### Labeller validation — counts\n```text\n15179:8\n```\n"
        "### Clean same-campaign test — counts\n```text\n15180:7 15183:6\n```\n"
        "### Held-out campaign test — counts\n```text\n28739:5\n```\n"
        "### Gas-puff — counts\n```text\n15180:4\n```\n",
        encoding="utf-8",
    )
    divertor = tmp_path / "divertor.md"
    divertor.write_text(
        "## Per-shot native-frame membership\n"
        "### Campaign\n"
        "15180 (2): 5 9\n"
        "15183 (0): none\n"
        "## Coverage and consistency checks\n",
        encoding="utf-8",
    )

    authorities = read_robustness_authorities(cohort, divertor)

    assert authorities.source_frames == {
        "clean": 13,
        "brightness": 13,
        "gas_puff": 4,
        "divertor_bright": 2,
        "held_out_campaign": 5,
    }
    assert authorities.divertor_indices[15180].tolist() == [5, 9]


def test_divertor_parser_rejects_declared_count_mismatch(tmp_path: Path):
    report = tmp_path / "screen.md"
    report.write_text(
        "## Per-shot native-frame membership\n"
        "### Campaign\n"
        "15162 (2): 5\n"
        "## Coverage and consistency checks\n",
        encoding="utf-8",
    )

    try:
        read_divertor_frame_indices(report)
    except ValueError as error:
        assert "declares 2 frames" in str(error)
    else:
        raise AssertionError("declared count mismatch was accepted")


def test_smoke_receipt_keeps_campaign_separate_and_refuses_gate_claim(
    tmp_path: Path, monkeypatch
):
    authorities = RobustnessAuthorities(
        clean_counts={15162: 240_281},
        campaign_counts={28739: 107_880},
        gas_puff_counts={15162: 124_138},
        divertor_indices={15162: np.arange(2_042, dtype=np.int64)},
    )
    model = CameraTopologyLabeller(
        LabellerConfig(window_frames=3, image_size=16, width=4, hidden=12)
    )
    checkpoint = type(
        "Checkpoint",
        (),
        {
            "model": model,
            "statistics": None,
            "training_shots": (1,),
            "metadata": {},
        },
    )()
    clean_metric = _metric(1.0, 2.0, 3.0, 0.96)
    clean_metric.update(
        {
            "position_error_cm": {
                "o_point": 1.0,
                "x_point": 2.0,
                "strike_point": 3.0,
            },
            "per_class": {},
        }
    )

    def fake_score(name, counts, *_args, gains=(1.0,), **_kwargs):
        source_frames = sum(counts.values())
        return {
            "name": name,
            "authority_source_frames": source_frames,
            "authority_shots": len(counts),
            "resolved_selected_source_frames": 2,
            "scored_source_frames": 2,
            "evaluation_instances": 2 * len(gains),
            "scored_shots": 1,
            "frame_identity_digest": "sha256:test",
            "identical_frames_for_labeller_and_baseline": True,
            "labeller": clean_metric,
            "brightness_centroid_baseline": clean_metric,
            "conditions": {},
        }

    monkeypatch.setattr(
        "imas_ambix.worldmodel.camera_topology_robustness.read_robustness_authorities",
        lambda *_args: authorities,
    )
    monkeypatch.setattr(
        "imas_ambix.worldmodel.camera_topology_robustness.load_labeller_checkpoint",
        lambda *_args, **_kwargs: checkpoint,
    )
    monkeypatch.setattr(
        "imas_ambix.worldmodel.camera_topology_robustness.read_fullshot_spans",
        lambda *_args: {},
    )
    monkeypatch.setattr(
        "imas_ambix.worldmodel.camera_topology_robustness._fit_baseline_from_checkpoint_split",
        lambda *_args, **_kwargs: (object(), {"shots": 1, "windows": 2}),
    )
    monkeypatch.setattr(
        "imas_ambix.worldmodel.camera_topology_robustness._score_membership",
        fake_score,
    )
    args = Namespace(
        cohort_report=Path("cohort.md"),
        divertor_report=Path("divertor.md"),
        checkpoint=Path("smoke.pt"),
        device="cpu",
        fullshot_manifest=Path("manifest.json"),
        level1_root=tmp_path,
        level2_root=tmp_path,
        baseline_max_shots=1,
        baseline_windows_per_shot=2,
        max_shots_per_subset=1,
        windows_per_shot=2,
        evidence_scale="smoke-scale",
        output=None,
    )

    report = run_robustness_scoring(args)

    assert set(report["emission_confounders"]) == {
        "brightness",
        "gas_puff",
        "divertor_bright",
    }
    assert report["held_out_campaign"]["name"] == "held_out_campaign"
    assert "camera-pose" in report["held_out_campaign"]["interpretation"]
    assert report["source_frame_authority"] == REGISTERED_SOURCE_FRAMES
    assert report["brightness_evaluation_instances"] == 961_124
    assert report["qualification"] == {
        "complete_source_census_scored": False,
        "eligible_as_gate_evidence": False,
        "numerical_thresholds_passed": True,
        "verdict": "smoke-scale-not-a-gate-result",
    }
