from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import zarr

from imas_ambix.worldmodel.camera_topology_targets import MAST_WALL_SOURCE_SHOT
from imas_ambix.worldmodel.limiter_polygon_validation import (
    compare_limiter_polygons,
    load_machine_description_wall,
)


def _circle(radius: float, count: int = 720) -> tuple[np.ndarray, np.ndarray]:
    angle = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    return radius * np.cos(angle), radius * np.sin(angle)


def test_known_radial_offset_is_recovered_in_both_directions() -> None:
    inner_r, inner_z = _circle(0.9)
    outer_r, outer_z = _circle(1.0)

    report = compare_limiter_polygons(
        inner_r,
        inner_z,
        outer_r,
        outer_z,
        dense_sample_count=1440,
    )

    inward = report["constant_vertices_to_machine_wall"]["summary"]
    outward = report["dense_machine_wall_to_constant_polygon"]["summary"]
    assert inward["median_signed_mm"] == pytest.approx(-100.0, abs=0.01)
    assert outward["median_signed_mm"] == pytest.approx(100.0, abs=0.01)
    assert inward["inside_count"] == 720
    assert outward["outside_count"] == 1440


def test_circle_against_inscribed_coarse_polygon_reports_inward_bias() -> None:
    machine_r, machine_z = _circle(1.0)
    constant_r, constant_z = _circle(1.0, count=6)

    report = compare_limiter_polygons(
        constant_r,
        constant_z,
        machine_r,
        machine_z,
        dense_sample_count=3600,
    )

    constant_to_machine = report["constant_vertices_to_machine_wall"]["summary"]
    machine_to_constant = report["dense_machine_wall_to_constant_polygon"]["summary"]
    expected_sagitta_mm = 1000.0 * (1.0 - np.cos(np.pi / 6.0))
    assert constant_to_machine["maximum_absolute_mm"] < 0.02
    assert machine_to_constant["median_signed_mm"] > 0.0
    assert machine_to_constant["maximum_absolute_mm"] == pytest.approx(
        expected_sagitta_mm, abs=0.02
    )
    assert machine_to_constant["outside_count"] > 3500


def test_wall_loader_records_era_constant_fill_provenance(tmp_path: Path) -> None:
    requested_shot = 22086
    zarr.open_group(str(tmp_path / f"{requested_shot}.zarr"), mode="w")
    source_store = zarr.open_group(
        str(tmp_path / f"{MAST_WALL_SOURCE_SHOT}.zarr"), mode="w"
    )
    wall = source_store.create_group("wall")
    wall_r = np.asarray([0.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    wall_z = np.asarray([0.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float32)
    wall.create_array("limiter_r", data=wall_r)
    wall.create_array("limiter_z", data=wall_z)
    wall.create_array(
        "limiter_geometry_channel",
        data=np.asarray([f"element_{index}" for index in range(1, 6)]),
    )

    result = load_machine_description_wall(requested_shot, level2_root=tmp_path)

    assert result.requested_shot_id == requested_shot
    assert result.source_shot_id == MAST_WALL_SOURCE_SHOT
    assert result.source_branch == "era_constant_fill"
    assert result.source_store == tmp_path / f"{MAST_WALL_SOURCE_SHOT}.zarr"
    assert result.description_2d_entry_count == 1
    assert result.limiter_unit_count == 5
    assert (
        result.digest == hashlib.sha256(wall_r.tobytes() + wall_z.tobytes()).hexdigest()
    )
