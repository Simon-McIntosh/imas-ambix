from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from imas_ambix.worldmodel.camera_picture_quality import (
    REPORT_JSON,
    measure_camera_frames,
    picture_quality_score,
    write_json_report,
)


def test_picture_metrics_use_shared_crop_and_thresholds() -> None:
    pattern = np.indices((4, 4)).sum(axis=0) % 2
    frames = np.stack(
        [
            np.zeros((4, 4), dtype=np.uint8),
            pattern.astype(np.uint8) * 100,
            pattern.astype(np.uint8) * 200,
            pattern.astype(np.uint8) * 255,
        ]
    )

    metrics = measure_camera_frames(
        frames,
        camera_group="rbb",
        bit_depth=8,
        frame_times=np.array([0.0, 0.1, 0.2, 0.3]),
    )

    assert metrics.frame_count == 4
    assert (metrics.height, metrics.width, metrics.channel_count) == (4, 4, 1)
    assert metrics.saturated_fraction == 0.125
    assert metrics.blank_like_fraction == 0.25
    assert metrics.robust_intensity_range == 150.0
    assert metrics.frame_motion_median == 50.0
    assert metrics.frame_motion_p90 == 50.0


def test_native_bit_depth_is_normalised_before_screening() -> None:
    pattern = np.indices((4, 4)).sum(axis=0) % 2
    frames = np.stack([pattern * 1023, pattern * 512]).astype(np.uint16)

    metrics = measure_camera_frames(frames, camera_group="rba", bit_depth=10)

    assert metrics.saturated_fraction == 0.25
    assert metrics.blank_like_fraction == 0.0
    assert metrics.robust_intensity_range > 190.0
    assert metrics.frame_motion_p90 > 63.0


def test_quality_score_penalises_blank_and_saturated_content() -> None:
    good = {
        "saturated_fraction": 0.02,
        "blank_like_fraction": 0.01,
        "robust_intensity_range": 190.0,
        "frame_motion_p90": 15.0,
    }
    clipped = {**good, "saturated_fraction": 0.40}
    blank = {**good, "blank_like_fraction": 0.70}

    assert picture_quality_score(good) > picture_quality_score(clipped)
    assert picture_quality_score(good) > picture_quality_score(blank)


def test_json_report_preserves_rank_components_and_readiness(tmp_path: Path) -> None:
    camera = {
        "camera_group": "rbb",
        "frame_count": 4,
        "height": 8,
        "width": 8,
        "channel_count": 1,
        "bit_depth": 8,
        "time_start_s": 0.0,
        "time_end_s": 0.3,
        "saturated_fraction": 0.01,
        "blank_like_fraction": 0.02,
        "robust_intensity_range": 190.0,
        "frame_motion_median": 8.0,
        "frame_motion_p90": 15.0,
        "source_quality": "Not Checked",
        "source_uuid": "synthetic",
    }
    report = {
        "source_revision": "deadbeef",
        "conclusion": "0 of 1 carriers are ready.",
        "shots": [
            {
                "shot": 21989,
                "picture_quality_rank": 1,
                "picture_quality_score": picture_quality_score(camera),
                "reference_camera": "rbb",
                "thomson": {"group": "atm", "channel_count": 36},
                "corpus_session": {"status": "absent"},
                "label_cartoon_ready_today": False,
                "cameras": [camera],
            }
        ],
    }

    json_path = write_json_report(report, tmp_path / REPORT_JSON)

    assert json_path.name == REPORT_JSON
    recorded = json.loads(json_path.read_text())
    assert recorded["shots"][0]["shot"] == 21989
    assert recorded["shots"][0]["picture_quality_rank"] == 1
    assert recorded["shots"][0]["thomson"] == {
        "group": "atm",
        "channel_count": 36,
    }
    assert recorded["shots"][0]["corpus_session"]["status"] == "absent"
    assert recorded["shots"][0]["label_cartoon_ready_today"] is False
