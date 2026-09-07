from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr
import zarr

from imas_ambix.worldmodel.carrier_camera_census import (
    MIN_CONVERGED_SLICES,
    build_ranking_payload,
    camera_richness_score,
    write_ranking,
)


def _write_session(root: Path, shot: int, converged: int) -> None:
    slices = [
        {
            "row": index,
            "time": index * 0.005,
            "written": True,
            "converged": index < converged,
        }
        for index in range(max(converged, MIN_CONVERGED_SLICES))
    ]
    manifest = {"shot": shot, "status": "complete", "slices": slices}
    (root / f"{shot}.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    dataset = xr.Dataset(coords={"time": np.arange(len(slices), dtype=np.float64)})
    dataset.to_netcdf(root / f"{shot}.nc", group="steering", engine="h5netcdf")


def _write_level1(
    root: Path,
    shot: int,
    *,
    camera: str,
    height: int,
    width: int,
    frames: int,
    span: float,
    thomson: str | None = "atm",
) -> None:
    store = zarr.open_group(str(root / f"{shot}.zarr"), mode="w", zarr_format=2)
    if thomson is not None:
        store.create_group(thomson)
    group = store.create_group(camera)
    group.create_array(
        "data",
        shape=(frames, height, width),
        dtype=np.uint8,
        chunks=(1, height, width),
    )
    group.create_array("time", data=np.linspace(0.0, span, frames))


def _add_camera(
    root: Path,
    shot: int,
    *,
    camera: str,
    height: int,
    width: int,
    frames: int,
    cadence_us: float,
) -> None:
    store = zarr.open_group(str(root / f"{shot}.zarr"), mode="a", zarr_format=2)
    group = store.create_group(camera)
    group.create_array(
        "data",
        shape=(frames, height, width),
        dtype=np.uint8,
        chunks=(1, height, width),
    )
    group.create_array(
        "time",
        data=np.arange(frames, dtype=np.float64) * cadence_us / 1_000_000.0,
    )


def _write_candidate(
    session_root: Path,
    level1_root: Path,
    shot: int,
    *,
    converged: int,
    camera: str,
    height: int,
    width: int,
    frames: int,
    span: float,
    thomson: str | None = "atm",
) -> None:
    _write_session(session_root, shot, converged)
    _write_level1(
        level1_root,
        shot,
        camera=camera,
        height=height,
        width=width,
        frames=frames,
        span=span,
        thomson=thomson,
    )


def test_richness_score_requires_strength_in_every_factor() -> None:
    balanced, _ = camera_richness_score(
        converged_slice_count=100,
        frame_area=160_000,
        temporal_span_s=0.5,
        frame_count=1_000,
    )
    narrow_high_cadence, _ = camera_richness_score(
        converged_slice_count=100,
        frame_area=12_000,
        temporal_span_s=0.5,
        frame_count=30_000,
    )
    assert balanced > narrow_high_cadence


def test_census_ranks_every_session_and_writes_eligible_top_five(
    tmp_path: Path,
) -> None:
    session_root = tmp_path / "sessions"
    level1_root = tmp_path / "level1"
    session_root.mkdir()
    level1_root.mkdir()

    specifications = [
        (20001, 110, "rbc", 120, 140, 180, 0.70, "atm"),
        (20002, 100, "rbb", 110, 130, 160, 0.65, "ayc"),
        (20003, 90, "rba", 100, 120, 140, 0.60, "aye"),
        (20004, 80, "rbc", 90, 110, 120, 0.55, "atm"),
        (20005, 70, "rbb", 80, 100, 100, 0.50, "ayc"),
        (20006, 60, "rba", 70, 90, 80, 0.45, "aye"),
        (20007, 120, "rbc", 160, 180, 220, 0.80, None),
        (20008, 39, "rbb", 150, 170, 200, 0.75, "atm"),
    ]
    for specification in specifications:
        (
            shot,
            converged,
            camera,
            height,
            width,
            frames,
            span,
            thomson,
        ) = specification
        _write_candidate(
            session_root,
            level1_root,
            shot,
            converged=converged,
            camera=camera,
            height=height,
            width=width,
            frames=frames,
            span=span,
            thomson=thomson,
        )
    _add_camera(
        level1_root,
        20001,
        camera="rbb",
        height=512,
        width=512,
        frames=3,
        cadence_us=667.0,
    )
    _add_camera(
        level1_root,
        20003,
        camera="rbb",
        height=128,
        width=172,
        frames=3,
        cadence_us=20.0,
    )

    payload = build_ranking_payload(
        session_root=session_root,
        level1_root=level1_root,
    )
    assert payload["session_count"] == 8
    assert payload["eligible_count"] == 6
    assert payload["excluded_count"] == 2
    assert payload["winner"]["shot"] == 20001
    assert [row["shot"] for row in payload["top_five"]] == [
        20001,
        20002,
        20003,
        20004,
        20005,
    ]
    assert [row["rank"] for row in payload["ranking"]] == list(range(1, 9))
    assert {row["shot"] for row in payload["ranking"]} == {
        item[0] for item in specifications
    }
    for row in payload["top_five"]:
        assert row["thomson_groups"]
        assert row["converged_slice_count"] >= MIN_CONVERGED_SLICES
        assert row["camera_group"] in {"rba", "rbb", "rbc"}
        assert row["frame_height"] > 0
        assert row["frame_width"] > 0
        assert row["frame_count"] > 0
        assert row["temporal_span_s"] > 0.0
        assert row["score"] > 0.0

    rows = {row["shot"]: row for row in payload["ranking"]}
    assert [camera["camera_group"] for camera in rows[20001]["available_cameras"]] == [
        "rbb",
        "rbc",
    ]
    assert rows[20001]["camera_group"] == "rbc"
    carrier_camera = rows[20001]["available_cameras"][0]
    assert carrier_camera["frame_height"] == 512
    assert carrier_camera["frame_width"] == 512
    assert carrier_camera["frame_count"] == 3
    assert carrier_camera["cadence_us"] == 667.0
    measurement = payload["measurement"]
    assert measurement["session_total"] == 8
    assert measurement["available_camera_group_total"] == 10
    assert measurement["available_camera_group_counts"] == {
        "rba": 2,
        "rbb": 5,
        "rbc": 3,
    }
    assert measurement["manifest_status_counts"] == {"complete": 8}
    assert measurement["measured_at_utc"] == payload["generated_at_utc"]
    assert payload["demo_camera_family"] == {
        "family": "demo-rbb",
        "definition": {
            "camera_group": "rbb",
            "frame_height": 128,
            "frame_width": 172,
            "cadence_target_us": 20.0,
            "cadence_tolerance_us": 5.0,
            "cadence_minimum_us": 15.0,
            "cadence_maximum_us": 25.0,
            "cadence_bounds_inclusive": True,
            "dimension_tolerance_pixels": 0,
        },
        "shot_count": 1,
        "shots": [20003],
        "complete_shot_count": 1,
        "complete_shots": [20003],
        "present_in_measured_sessions": True,
        "finding": (
            f"1 shot found among 8 sessions at {measurement['measured_at_utc']}; "
            "1 had complete manifests"
        ),
        "session_total": 8,
        "measured_at_utc": measurement["measured_at_utc"],
    }
    assert payload["square_camera_family"]["shot_count"] == 1
    assert payload["square_camera_family"]["shots"] == [20001]
    geometry_rows = payload["rbb_geometry_distribution"]["rows"]
    demo_geometry = next(
        row
        for row in geometry_rows
        if (row["frame_height"], row["frame_width"]) == (128, 172)
    )
    assert demo_geometry["cadence_band_lower_us"] == 15.0
    assert demo_geometry["cadence_band_upper_us"] == 25.0
    assert demo_geometry["shots"] == [20003]
    assert demo_geometry["complete_shot_count"] == 1
    assert demo_geometry["complete_shots"] == [20003]
    assert "trainability is not inferred" in payload["trainability_scope"]

    excluded = {row["shot"]: row for row in payload["ranking"] if not row["eligible"]}
    assert excluded[20007]["exclusion_reasons"] == ["missing_thomson_group"]
    assert excluded[20008]["exclusion_reasons"] == ["fewer_than_40_converged_slices"]

    output = tmp_path / "ranking.json"
    write_ranking(payload, output)
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["winner"] == payload["winner"]
    assert written["ranking_sha256"] == payload["ranking_sha256"]
