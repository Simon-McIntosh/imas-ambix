from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[2]
MODULE_PATH = ROOT / "scripts" / "challenge_ip_channel_audit.py"
ARTIFACT_PATH = (
    ROOT / "imas_ambix" / "challenge" / "artifacts" / "ip_channel_audit.json"
)
SPEC = importlib.util.spec_from_file_location("challenge_ip_channel_audit", MODULE_PATH)
assert SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def test_frame_causes_partition_low_current_values() -> None:
    label_time = np.arange(-1.0, 10.0)
    native_time = np.arange(0.0, 9.0)
    native_current = np.array([0.0, 0.0, 100.0, 100.0, 0.0, 100.0, 100.0, 0.0, 0.0])
    interpolated, audited, causes = audit.classify_low_current_frames(
        label_time, native_time, native_current
    )

    assert interpolated.shape == audited.shape == causes.shape
    assert causes[audited].tolist() == [
        "outside_native_support",
        "aligned_leading_edge",
        "aligned_leading_edge",
        "interior_low_current",
        "aligned_trailing_edge",
        "aligned_trailing_edge",
        "outside_native_support",
    ]
    assert np.count_nonzero(audited) == np.count_nonzero(causes != "")


def test_shot_level_causes_distinguish_encoding_and_alignment() -> None:
    labels = np.array([0.0, 1.0, 2.0])
    native_time = np.arange(5.0)

    _, audited, causes = audit.classify_low_current_frames(
        labels, native_time, np.zeros(5)
    )
    assert audited.all()
    assert set(causes) == {"whole_channel_below_threshold"}

    _, audited, causes = audit.classify_low_current_frames(
        labels, native_time + 10.0, np.array([0.0, 100.0, 100.0, 0.0, 0.0])
    )
    assert audited.all()
    assert set(causes) == {"outside_native_support"}

    sparse_labels = np.array([0.0, 2.0, 4.0])
    _, audited, causes = audit.classify_low_current_frames(
        sparse_labels, native_time, np.array([0.0, 100.0, 0.0, 100.0, 0.0])
    )
    assert audited.all()
    assert set(causes) == {"label_grid_misses_native_activity"}


def test_committed_artifact_partitions_the_full_audited_population() -> None:
    receipt = json.loads(ARTIFACT_PATH.read_text())
    population = receipt["population_contract"]
    assert population["shot_count"] == 7_041
    assert population["labelled_frame_count"] == 1_559_340
    assert population["audited_frame_count"] == 658_787
    assert sum(item["frame_count"] for item in receipt["classes"]) == 658_787
    assert {item["name"] for item in receipt["classes"]} == set(
        receipt["exclusion_policy"]["default_excluded_classes"]
    )
    for item in receipt["classes"]:
        assert item["time_alignment"]
        assert item["sentinel_or_missing_encoding"]
        assert item["other_actuator_covariation"]
        assert item["downstream_verdict"]["exclude_from_default_consumers"] is True
