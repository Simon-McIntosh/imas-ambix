"""Smoke tests for the FAIR-MAST data probe + manifest pipeline.

These tests exercise the pure logic of the data module without touching
the network or the s5cmd binary. They confirm:

- Module imports work without pulling in heavy deps at module load.
- The probe report serializes to JSON.
- Manifest construction filters by camera columns correctly.
- The acceptance summary picks the right gate state.

Networked / s5cmd-driven tests live behind a marker and run only on a
sirius compute node — out of scope for the v0 smoke test.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from imas_ambix.data import manifest as manifest_mod
from imas_ambix.data import paths as paths_mod
from imas_ambix.data import probe as probe_mod


def _fake_index() -> pd.DataFrame:
    """Three-shot fake index with two camera-bearing rows."""
    return pd.DataFrame(
        {
            "shot_id": [11695, 30420, 51056],
            "campaign": ["M5", "M9", "M9"],
            "camera_visible_present": [False, True, True],
            "camera_ir_present": [False, False, True],
            "magnetics_present": [True, True, True],
        }
    )


def test_paths_constants_match_plan():
    """Storage layout matches plans/data-acquisition.md §4.1."""
    assert str(paths_mod.MIRROR_ROOT) == "/work/projects/imas_gpu/mast"
    assert str(paths_mod.LEVEL2_DIR) == "/work/projects/imas_gpu/mast/level2/shots"
    assert paths_mod.S3_ENDPOINT == "https://s3.echo.stfc.ac.uk"
    assert paths_mod.S3_BUCKET == "mast"
    assert paths_mod.SHOT_INDEX_URL == "https://mastapp.site/parquet/level2/shots"
    assert paths_mod.s3_shot_path(30420) == "s3://mast/level2/shots/30420.zarr"


def test_detect_camera_columns_finds_visible_and_ir():
    df = _fake_index()
    cols = manifest_mod.detect_camera_columns(df)
    assert set(cols) == {"camera_visible_present", "camera_ir_present"}


def test_filter_camera_bearing_drops_magnetics_only_shots():
    df = _fake_index()
    subset = manifest_mod.filter_camera_bearing(df)
    assert list(subset["shot_id"]) == [30420, 51056]


def test_build_manifest_camera_only_filters_and_records_provenance():
    df = _fake_index()
    m = manifest_mod.build_manifest(df, camera_only=True)
    assert m.shot_ids == (30420, 51056)
    assert m.total_in_index == 3
    assert "camera_visible_present" in m.camera_flag_columns
    assert m.source_url == paths_mod.SHOT_INDEX_URL
    assert m.filter_description.startswith("camera-bearing")
    payload = json.loads(m.to_json())
    assert payload["shot_ids"] == [30420, 51056]
    assert payload["total_in_index"] == 3


def test_build_manifest_all_shots_includes_magnetics_only():
    df = _fake_index()
    m = manifest_mod.build_manifest(df, camera_only=False)
    assert m.shot_ids == (11695, 30420, 51056)
    assert m.filter_description == "all level-2 shots"


def test_emit_shot_ids_is_newline_delimited():
    df = _fake_index()
    m = manifest_mod.build_manifest(df, camera_only=True)
    text = manifest_mod.emit_shot_ids(m)
    assert text == "30420\n51056\n"


def test_probe_report_acceptance_pass():
    samples = [
        probe_mod.ShotSample(
            shot_id=30420 + i,
            has_camera=True,
            bytes_copied=500_000_000,  # 500 MB
            elapsed_s=2.0,
            error=None,
        )
        for i in range(5)
    ]
    report = probe_mod.ProbeReport(
        n_shots_in_index=11_573,
        n_camera_shots=3_000,
        sample_size=len(samples),
        samples=samples,
        sustained_throughput_mbps=250.0,
        median_shot_size_mb=500.0,
        p95_shot_size_mb=550.0,
        extrapolated_total_size_tb=5.0,
        started_at="2026-05-19T18:00:00Z",
        finished_at="2026-05-19T18:30:00Z",
    )
    acc = report.acceptance_summary()
    assert acc == {
        "throughput": "pass",
        "total_size": "pass",
        "camera_shots": "pass",
        "per_shot_p95": "pass",
    }


def test_probe_report_acceptance_degraded_throughput():
    report = probe_mod.ProbeReport(
        n_shots_in_index=11_573,
        n_camera_shots=3_000,
        sample_size=0,
        sustained_throughput_mbps=100.0,
        median_shot_size_mb=500.0,
        p95_shot_size_mb=550.0,
        extrapolated_total_size_tb=5.0,
    )
    acc = report.acceptance_summary()
    assert acc["throughput"] == "degraded"
    assert acc["total_size"] == "pass"


def test_probe_report_acceptance_fail_camera_shortfall():
    report = probe_mod.ProbeReport(
        n_shots_in_index=11_573,
        n_camera_shots=500,  # below the 1000 gate
        sample_size=0,
        sustained_throughput_mbps=300.0,
        median_shot_size_mb=500.0,
        p95_shot_size_mb=550.0,
        extrapolated_total_size_tb=5.0,
    )
    acc = report.acceptance_summary()
    assert acc["camera_shots"] == "fail"


def test_probe_report_acceptance_fail_oversize():
    report = probe_mod.ProbeReport(
        n_shots_in_index=11_573,
        n_camera_shots=3_000,
        sample_size=0,
        sustained_throughput_mbps=300.0,
        median_shot_size_mb=500.0,
        p95_shot_size_mb=550.0,
        extrapolated_total_size_tb=20.0,  # above the 12 TB gate
    )
    acc = report.acceptance_summary()
    assert acc["total_size"] == "fail"


def test_probe_report_to_json_is_parseable():
    report = probe_mod.ProbeReport(
        n_shots_in_index=11_573,
        n_camera_shots=3_000,
        sample_size=0,
        sustained_throughput_mbps=300.0,
        median_shot_size_mb=500.0,
        p95_shot_size_mb=550.0,
        extrapolated_total_size_tb=5.0,
        notes=["happy path"],
    )
    payload = json.loads(report.to_json())
    assert payload["sustained_throughput_mbps"] == 300.0
    assert payload["acceptance"]["throughput"] == "pass"
    assert payload["notes"] == ["happy path"]


def test_sample_shots_deterministic_under_seed():
    df = pd.DataFrame(
        {
            "shot_id": list(range(100)),
            "camera_visible_present": [True] * 100,
        }
    )
    a = probe_mod.sample_shots(df, sample_size=10, camera_only=True, seed=42)
    b = probe_mod.sample_shots(df, sample_size=10, camera_only=True, seed=42)
    c = probe_mod.sample_shots(df, sample_size=10, camera_only=True, seed=99)
    assert a == b
    assert a != c
    assert len(a) == 10
    assert all(0 <= s < 100 for s in a)


def test_run_probe_errors_without_s5cmd(monkeypatch):
    """run_probe must refuse to start if s5cmd is missing on PATH."""
    monkeypatch.setattr(probe_mod, "s5cmd_available", lambda: False)
    with pytest.raises(RuntimeError, match="s5cmd is not on PATH"):
        probe_mod.run_probe(sample_size=1)
