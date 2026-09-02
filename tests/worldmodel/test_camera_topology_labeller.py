"""Tests for the independent rbb camera topology labeller."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from imas_ambix.worldmodel.camera_topology_labeller import (
    CameraTopologyLabeller,
    LabellerConfig,
    LabellerTargets,
    LoadedWindows,
    assemble_labeller_targets,
    evaluate_predictions,
    fit_full_corpus_labeller,
    fit_labeller,
    read_cohort_frame_counts,
    read_cohort_split,
    read_fullshot_spans,
    run_training,
    target_statistics,
    validate_full_corpus_split,
)
from imas_ambix.worldmodel.camera_topology_targets import (
    TOPOLOGY_CLASS_NAMES,
    CameraTopologyTargets,
)
from imas_ambix.worldmodel.equilibrium_labels import EquilibriumGeometry


def _synthetic_windows(n_samples: int = 18, *, seed: int = 3) -> LoadedWindows:
    rng = np.random.default_rng(seed)
    frames = rng.uniform(0.0, 0.03, size=(n_samples, 3, 24, 24)).astype(np.float32)
    classes = np.arange(n_samples, dtype=np.int64) % 4
    values = np.zeros((n_samples, 8), dtype=np.float32)
    for index in range(n_samples):
        row = 3 + (index * 5) % 17
        column = 3 + (index * 7) % 17
        frames[index, :, row - 1 : row + 2, column - 1 : column + 2] += 0.9
        values[index] = np.array(
            [
                0.55 + column / 50.0,
                -0.30 + row / 50.0,
                0.50 + column / 45.0,
                -0.35 + row / 45.0,
                0.35,
                -0.55,
                1.35,
                -0.55,
            ],
            dtype=np.float32,
        )
    mask = np.ones_like(values, dtype=bool)
    mask[classes == 0, 2:] = False
    values[~mask] = np.nan
    return LoadedWindows(
        frames=frames,
        targets=LabellerTargets(values, mask, classes),
        shot_ids=np.arange(n_samples, dtype=np.int64) + 20_000,
        frame_times=np.arange(n_samples, dtype=np.float64) / 100.0,
    )


def test_assemble_targets_preserves_metres_and_absence_masks():
    times = np.array([0.1, 0.2])
    geometry = EquilibriumGeometry(
        shot_id=7,
        frame_times=times,
        target=np.array(
            [[0.9, 0.1] + [np.nan] * 12, [1.0, -0.1] + [np.nan] * 12],
            dtype=np.float32,
        ),
        finite_mask=np.array(
            [[True, True] + [False] * 12, [True, True] + [False] * 12]
        ),
    )
    topology = CameraTopologyTargets(
        shot_id=7,
        frame_times=times,
        primary_xpoint=np.array([[0.8, -0.5], [np.nan, np.nan]], dtype=np.float32),
        primary_xpoint_mask=np.array([True, False]),
        strike_points=np.array(
            [[[0.4, -0.7], [1.4, -0.7]], [[np.nan, np.nan], [np.nan, np.nan]]],
            dtype=np.float32,
        ),
        strike_point_mask=np.array([[True, True], [False, False]]),
        topology_class=np.array([1, 0], dtype=np.int8),
        boundary_psi=np.array([0.2, 0.2]),
        boundary_flux_mask=np.array([True, True]),
    )

    result = assemble_labeller_targets(geometry, topology)

    np.testing.assert_allclose(
        result.values_m[0], [0.9, 0.1, 0.8, -0.5, 0.4, -0.7, 1.4, -0.7]
    )
    assert result.finite_mask[0].all()
    assert result.finite_mask[1].tolist() == [True, True] + [False] * 6
    assert np.isnan(result.values_m[1, 2:]).all()


def test_cohort_reader_preserves_whole_shot_firewall(tmp_path: Path):
    report = tmp_path / "cohort.md"
    report.write_text(
        "### Labeller train — counts\n```text\nM6 — 15276:12 21983:9\n```\n"
        "### Labeller validation — counts\n```text\nM7 — 21989:8\n```\n"
        "### Clean same-campaign test — counts\n```text\nM7 — 21986:7\n```\n"
        "### Held-out campaign test — counts\n```text\nM9 — 28739:6\n```\n",
        encoding="utf-8",
    )

    assert read_cohort_split(report) == {
        "train": [15276, 21983],
        "validation": [21989],
        "clean_test": [21986],
        "campaign_test": [28739],
    }


def test_model_forward_and_masked_fit_decrease_loss():
    data = _synthetic_windows()
    model = CameraTopologyLabeller(
        LabellerConfig(window_frames=3, image_size=24, width=4, hidden=24)
    )

    coordinates, logits = model(torch.from_numpy(data.frames))
    _, losses = fit_labeller(
        model,
        data.frames,
        data.targets,
        steps=18,
        learning_rate=1.0e-2,
    )

    assert coordinates.shape == (18, 8)
    assert logits.shape == (18, len(TOPOLOGY_CLASS_NAMES))
    assert np.isfinite(losses).all()
    assert losses[-1] < losses[0]


def test_evaluation_exposes_zero_support_class():
    data = _synthetic_windows(12)
    prediction = np.nan_to_num(data.targets.values_m, nan=0.0)
    report = evaluate_predictions(
        prediction, data.targets.topology_class.copy(), data.targets
    )

    assert report["axis_error_cm"] == 0.0
    assert report["class_accuracy"] == 1.0
    unsupported = report["per_class"]["disconnected-double-null"]
    assert unsupported == {"support": 0, "accuracy": None}


def test_training_entrypoint_reports_model_and_centroid_on_same_frames(monkeypatch):
    train = _synthetic_windows(16, seed=4)
    heldout = _synthetic_windows(8, seed=5)
    monkeypatch.setattr(
        "imas_ambix.worldmodel.camera_topology_labeller.read_cohort_split",
        lambda _path: {
            "train": [21983, 21985],
            "validation": [21989],
            "clean_test": [21986],
            "campaign_test": [28739],
        },
    )
    monkeypatch.setattr(
        "imas_ambix.worldmodel.camera_topology_labeller.load_camera_windows",
        lambda shots, **_kwargs: train if shots == [21983, 21985] else heldout,
    )
    args = Namespace(
        cohort_report=Path("unused.md"),
        level1_root=Path("unused"),
        train_shots="21983,21985",
        heldout_shots="21989",
        windows_per_shot=8,
        window_frames=3,
        image_size=24,
        width=4,
        hidden=24,
        steps=12,
        learning_rate=1.0e-2,
        seed=0,
        device="cpu",
        output=None,
    )

    report = run_training(args)

    assert report["training_loss"]["decreased"] is True
    assert report["heldout_windows"] == 8
    assert "axis_error_cm" in report["model"]
    assert "axis_error_cm" in report["brightness_centroid_baseline"]
    assert report["model"]["per_class"]["disconnected-double-null"]["support"] == 0


def test_corpus_authorities_preserve_frame_counts_and_plasma_spans(
    tmp_path: Path, monkeypatch
):
    report = tmp_path / "cohort.md"
    report.write_text(
        "### Labeller train — counts\n```text\nM6 — 15276:12 21983:9\n```\n"
        "### Labeller validation — counts\n```text\nM7 — 21989:8\n```\n"
        "### Clean same-campaign test — counts\n```text\nM7 — 21986:7\n```\n"
        "### Held-out campaign test — counts\n```text\nM9 — 28739:6\n```\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "fullshot.json"
    manifest.write_text(
        '{"windows": ['
        '{"shot_id": 15276, "start_frame": 2, "end_frame": 14},'
        '{"shot_id": 21983, "start_frame": 3, "end_frame": 12}'
        "]}\n",
        encoding="utf-8",
    )
    frame_counts = read_cohort_frame_counts(report)
    monkeypatch.setattr(
        "imas_ambix.worldmodel.camera_topology_labeller.FULL_CORPUS_COUNTS",
        {"train": (2, 21), "validation": (1, 8)},
    )

    validate_full_corpus_split(frame_counts)

    assert frame_counts["train"] == {15276: 12, 21983: 9}
    assert read_fullshot_spans(manifest) == {15276: (2, 14), 21983: (3, 12)}


def test_bounded_corpus_fit_writes_first_improving_heldout_checkpoint(tmp_path: Path):
    train = _synthetic_windows(24, seed=7)
    heldout = _synthetic_windows(24, seed=7)

    def window_loader(shots, **_kwargs):
        return train if shots == [1] else heldout

    model = CameraTopologyLabeller(
        LabellerConfig(window_frames=3, image_size=24, width=4, hidden=24)
    )
    output = tmp_path / "training.json"
    report = fit_full_corpus_labeller(
        model,
        [1],
        [2],
        target_statistics(train.targets),
        window_frames=3,
        image_size=24,
        windows_per_shot=0,
        level1_root=tmp_path,
        frame_spans={1: (0, 24), 2: (0, 24)},
        batch_size=24,
        epochs=8,
        learning_rate=1.0e-2,
        seed=0,
        device="cpu",
        output=output,
        declared_frame_counts={"train": 24, "validation": 24},
        window_loader=window_loader,
    )

    checkpoint = report["first_improving_checkpoint"]
    assert checkpoint is not None
    assert Path(checkpoint).is_file()
    assert report["history"][0]["validation_loss"] < report["initial_validation_loss"]
    assert report["declared_train_frames"] == 24


def test_full_corpus_launcher_requests_reserved_h200_and_complete_split():
    launcher = Path("scripts/slurm/camera_topology_labeller_train.sbatch").read_text(
        encoding="utf-8"
    )

    assert "#SBATCH --partition=betelgeuse" in launcher
    assert "#SBATCH --reservation=gpu_0003_grpA" in launcher
    assert "#SBATCH --account=grpa" in launcher
    assert "#SBATCH --gres=gpu:1" in launcher
    assert "--full-corpus" in launcher
    assert "--windows-per-shot 0" in launcher
    assert "--device cuda" in launcher
