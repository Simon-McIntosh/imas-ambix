"""Smoke tests for the FAIR-MAST data probe + manifest pipeline.

These tests exercise the pure logic of the data module without touching
the network or the s5cmd binary. Network/s5cmd-driven tests run
exclusively from a network-enabled host and are out of scope here.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from imas_ambix.data import manifest as manifest_mod
from imas_ambix.data import paths as paths_mod
from imas_ambix.data import probe as probe_mod


def _fake_index() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "shot_id": [11695, 30420, 51056],
            "campaign": ["M5", "M9", "M9"],
        }
    )


# --- paths ------------------------------------------------------------


def test_paths_constants_match_plan():
    assert str(paths_mod.MIRROR_ROOT) == "/work/projects/imas_gpu/mast"
    assert str(paths_mod.LEVEL1_DIR) == "/work/projects/imas_gpu/mast/level1/shots"
    assert str(paths_mod.LEVEL2_DIR) == "/work/projects/imas_gpu/mast/level2/shots"
    assert paths_mod.S3_ENDPOINT == "https://s3.echo.stfc.ac.uk"
    assert paths_mod.S3_BUCKET == "mast"
    assert paths_mod.SHOT_INDEX_URL == "https://mastapp.site/parquet/level2/shots"


def test_s3_paths_are_tier_aware():
    assert paths_mod.s3_shot_path(30420) == "s3://mast/level2/shots/30420.zarr"
    assert (
        paths_mod.s3_shot_path(30420, tier="level1")
        == "s3://mast/level1/shots/30420.zarr"
    )
    assert (
        paths_mod.s3_group_path(30420, "rba", tier="level1")
        == "s3://mast/level1/shots/30420.zarr/rba"
    )


def test_local_shot_path_picks_right_root():
    assert str(paths_mod.local_shot_path(30420, tier="level2")).endswith(
        "level2/shots/30420.zarr"
    )
    assert str(paths_mod.local_shot_path(30420, tier="level1")).endswith(
        "level1/shots/30420.zarr"
    )


def test_camera_sources_constants():
    assert "rba" in paths_mod.CAMERA_SOURCES
    assert "rbb" in paths_mod.CAMERA_SOURCES
    assert "rir" in paths_mod.CAMERA_SOURCES
    # Control vector kept separately
    assert "anb" in paths_mod.CONTROL_SOURCES
    assert "anb" not in paths_mod.CAMERA_SOURCES


# --- manifest module --------------------------------------------------


def test_shot_ids_from_index():
    df = _fake_index()
    assert manifest_mod.shot_ids_from_index(df) == (11695, 30420, 51056)


def test_filter_by_groups_any_mode():
    inv = {
        100: ("magnetics", "equilibrium"),
        200: ("magnetics", "equilibrium", "camera_visible"),
        300: ("magnetics", "rba"),
        400: (),
    }
    assert manifest_mod.filter_by_groups(inv, ("camera_visible", "rba"), "any") == [
        200,
        300,
    ]


def test_filter_by_groups_all_mode():
    inv = {
        100: ("magnetics", "equilibrium"),
        200: ("magnetics", "equilibrium", "rba"),
        300: ("rba", "rbb"),
    }
    assert manifest_mod.filter_by_groups(inv, ("rba", "rbb"), "all") == [300]


def test_filter_by_groups_rejects_bad_mode():
    with pytest.raises(ValueError, match="mode must be"):
        manifest_mod.filter_by_groups({}, (), mode="nope")


def test_group_coverage_counts_per_group():
    inv = {
        1: ("magnetics", "equilibrium"),
        2: ("magnetics", "rba"),
        3: ("rba", "rbb"),
    }
    cov = manifest_mod.group_coverage(inv)
    assert cov["magnetics"] == 2
    assert cov["rba"] == 2
    assert cov["rbb"] == 1
    # Ordering is by descending count
    counts = list(cov.values())
    assert counts == sorted(counts, reverse=True)


def test_build_manifest_groups_and_targets():
    m = manifest_mod.build_manifest(
        tier="level1",
        shot_ids=[100, 200],
        groups=("rba", "rbb"),
        total_in_index=11573,
        filter_description="camera shots",
    )
    assert m.tier == "level1"
    assert m.shot_ids == (100, 200)
    assert m.groups == ("rba", "rbb")
    assert m.total_in_index == 11573
    targets = m.targets()
    # Cross product: 2 shots × 2 groups = 4 targets
    assert len(targets) == 4
    assert {(t.shot_id, t.group) for t in targets} == {
        (100, "rba"),
        (100, "rbb"),
        (200, "rba"),
        (200, "rbb"),
    }


def test_build_manifest_no_groups_means_whole_shot():
    m = manifest_mod.build_manifest(tier="level2", shot_ids=[100])
    targets = m.targets()
    assert len(targets) == 1
    assert targets[0].group is None
    assert targets[0].tier == "level2"


def test_emit_shot_ids():
    m = manifest_mod.build_manifest(tier="level2", shot_ids=[11695, 30420])
    assert manifest_mod.emit_shot_ids(m) == "11695\n30420\n"


def test_emit_targets_as_s5cmd_whole_shot():
    m = manifest_mod.build_manifest(tier="level2", shot_ids=[30420])
    script = manifest_mod.emit_targets_as_s5cmd(m)
    assert "cp s3://mast/level2/shots/30420.zarr/* ./level2/shots/30420.zarr/" in script


def test_emit_targets_as_s5cmd_per_group():
    m = manifest_mod.build_manifest(
        tier="level1", shot_ids=[30420], groups=("rba", "rbb")
    )
    script = manifest_mod.emit_targets_as_s5cmd(m)
    assert (
        "cp s3://mast/level1/shots/30420.zarr/rba/* "
        "./level1/shots/30420.zarr/rba/" in script
    )
    assert (
        "cp s3://mast/level1/shots/30420.zarr/rbb/* "
        "./level1/shots/30420.zarr/rbb/" in script
    )


def test_manifest_to_json_roundtrip():
    m = manifest_mod.build_manifest(
        tier="level1",
        shot_ids=[100, 200],
        groups=("rba",),
        total_in_index=11573,
    )
    payload = json.loads(m.to_json())
    assert payload["tier"] == "level1"
    assert payload["shot_ids"] == [100, 200]
    assert payload["groups"] == ["rba"]
    assert payload["total_in_index"] == 11573


# --- probe ------------------------------------------------------------


def test_sample_shots_deterministic_under_seed():
    df = pd.DataFrame({"shot_id": list(range(100))})
    a = probe_mod.sample_shots(df, sample_size=10, seed=42)
    b = probe_mod.sample_shots(df, sample_size=10, seed=42)
    c = probe_mod.sample_shots(df, sample_size=10, seed=99)
    assert a == b
    assert a != c
    assert len(a) == 10


def test_sample_shots_clamps_to_index_size():
    df = pd.DataFrame({"shot_id": [1, 2, 3]})
    assert len(probe_mod.sample_shots(df, sample_size=100, seed=0)) == 3


def test_probe_report_acceptance_pass_level2():
    samples = [
        probe_mod.ShotSample(
            shot_id=30420 + i,
            groups=("magnetics", "equilibrium"),
            bytes_copied=12_000_000,
            elapsed_s=1.0,
        )
        for i in range(5)
    ]
    report = probe_mod.ProbeReport(
        tier="level2",
        n_shots_in_index=11_573,
        n_shots_in_tier=11_573,
        sample_size=5,
        samples=samples,
        sustained_throughput_mbps=250.0,
        median_shot_size_mb=12.0,
        p95_shot_size_mb=15.0,
        extrapolated_total_size_tb=0.14,
        camera_coverage_fraction=0.0,
    )
    acc = report.acceptance_summary()
    assert acc["throughput"] == "pass"
    assert acc["total_size"] == "pass"
    assert acc["camera_coverage"] == "n/a"  # level-2 doesn't have cameras


def test_probe_report_acceptance_camera_fail_at_level1():
    report = probe_mod.ProbeReport(
        tier="level1",
        n_shots_in_index=11_573,
        n_shots_in_tier=17_111,
        sample_size=0,
        sustained_throughput_mbps=300.0,
        median_shot_size_mb=100.0,
        p95_shot_size_mb=200.0,
        extrapolated_total_size_tb=1.5,
        camera_coverage_fraction=0.1,  # below 0.3 gate
    )
    acc = report.acceptance_summary()
    assert acc["camera_coverage"] == "fail"


def test_probe_report_acceptance_camera_pass_at_level1():
    report = probe_mod.ProbeReport(
        tier="level1",
        n_shots_in_index=11_573,
        n_shots_in_tier=17_111,
        sample_size=0,
        sustained_throughput_mbps=300.0,
        median_shot_size_mb=100.0,
        p95_shot_size_mb=200.0,
        extrapolated_total_size_tb=1.5,
        camera_coverage_fraction=0.6,
    )
    acc = report.acceptance_summary()
    assert acc["camera_coverage"] == "pass"


def test_probe_report_acceptance_throughput_degraded():
    report = probe_mod.ProbeReport(
        tier="level2",
        n_shots_in_index=11_573,
        n_shots_in_tier=11_573,
        sample_size=0,
        sustained_throughput_mbps=100.0,
        median_shot_size_mb=12.0,
        p95_shot_size_mb=15.0,
        extrapolated_total_size_tb=0.14,
    )
    acc = report.acceptance_summary()
    assert acc["throughput"] == "degraded"


def test_probe_report_to_json_is_parseable():
    report = probe_mod.ProbeReport(
        tier="level2",
        n_shots_in_index=11_573,
        n_shots_in_tier=11_573,
        sample_size=0,
        sustained_throughput_mbps=300.0,
        median_shot_size_mb=12.0,
        p95_shot_size_mb=15.0,
        extrapolated_total_size_tb=0.14,
        group_coverage={"magnetics": 5, "equilibrium": 5},
        camera_coverage_fraction=0.0,
        notes=["happy path"],
    )
    payload = json.loads(report.to_json())
    assert payload["tier"] == "level2"
    assert payload["acceptance"]["throughput"] == "pass"
    assert payload["group_coverage"]["magnetics"] == 5
    assert payload["notes"] == ["happy path"]


def test_shot_sample_has_camera_detects_visible():
    s = probe_mod.ShotSample(
        shot_id=30420,
        groups=("magnetics", "rba"),
        bytes_copied=100,
        elapsed_s=1.0,
    )
    assert s.has_camera is True


def test_shot_sample_has_camera_negative():
    s = probe_mod.ShotSample(
        shot_id=30420,
        groups=("magnetics", "equilibrium"),
        bytes_copied=100,
        elapsed_s=1.0,
    )
    assert s.has_camera is False


def test_run_probe_errors_without_s5cmd(monkeypatch):
    """run_probe must refuse to start if s5cmd is missing on PATH."""
    monkeypatch.setattr(probe_mod, "s5cmd_available", lambda: False)
    with pytest.raises(manifest_mod.S5cmdMissingError, match="s5cmd is not on PATH"):
        probe_mod.run_probe(sample_size=1)
