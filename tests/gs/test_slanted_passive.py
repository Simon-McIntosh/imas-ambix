"""Public polygon-column contract for MAST's eight slanted passive sections."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from imas_ambix.gs.operator import polygon_section_column


@dataclass(frozen=True)
class SlantedSection:
    name: str
    circuit: int
    centroid_r: float
    centroid_z: float
    width: float
    height: float
    angle1: float
    angle2: float
    vertices: np.ndarray


_SLANTED_CATALOG = (
    ("botcol", 0.2354, -2.0250, 0.0, 295.324),
    ("topcol", 0.2356, 2.0250, 0.0, 64.676),
    ("p2larm", 0.3532, -1.6308, 45.0, 0.0),
    ("p2larm_out", 0.6827, -1.6506, 0.0, 320.0),
    ("p2ldivpl", 0.6198, -1.6337, 320.0, 0.0),
    ("p2uarm", 0.3532, 1.6308, 315.0, 0.0),
    ("p2uarm_out", 0.6827, 1.6506, 0.0, 40.0),
    ("p2udivpl", 0.6198, 1.6337, 40.0, 0.0),
)

SLANTED_NAMES = {
    "botcol",
    "topcol",
    "p2larm",
    "p2larm_out",
    "p2ldivpl",
    "p2uarm",
    "p2uarm_out",
    "p2udivpl",
}


def _section_vertices(
    r: float,
    z: float,
    width: float,
    height: float,
    angle1: float,
    angle2: float,
) -> np.ndarray:
    radial_shear = 1.0 / np.tan(np.deg2rad(angle2)) if angle2 > 0 else 0.0
    vertical_shear = np.tan(np.deg2rad(angle1)) if angle1 > 0 else 0.0
    return np.array(
        [
            (
                r - width / 2 - height / 2 * radial_shear,
                z - height / 2 - width / 2 * vertical_shear,
            ),
            (
                r + width / 2 - height / 2 * radial_shear,
                z - height / 2 + width / 2 * vertical_shear,
            ),
            (
                r + width / 2 + height / 2 * radial_shear,
                z + height / 2 + width / 2 * vertical_shear,
            ),
            (
                r - width / 2 + height / 2 * radial_shear,
                z + height / 2 - width / 2 * vertical_shear,
            ),
        ],
        dtype=np.float64,
    )


@pytest.fixture(scope="module")
def sections() -> tuple[SlantedSection, ...]:
    width, height = 0.05, 0.10
    return tuple(
        SlantedSection(
            name=name,
            circuit=100 + index,
            centroid_r=r,
            centroid_z=z,
            width=width,
            height=height,
            angle1=angle1,
            angle2=angle2,
            vertices=_section_vertices(r, z, width, height, angle1, angle2),
        )
        for index, (name, r, z, angle1, angle2) in enumerate(_SLANTED_CATALOG)
    )


def _poly_area(vertices: np.ndarray) -> float:
    rolled = np.roll(vertices, -1, axis=0)
    return 0.5 * abs(
        np.sum(vertices[:, 0] * rolled[:, 1] - rolled[:, 0] * vertices[:, 1])
    )


def _public_column(vertices: np.ndarray) -> np.ndarray:
    return polygon_section_column(
        vertices,
        1.0,
        sensor_r=np.array([0.9, 1.2, 1.5]),
        sensor_z=np.array([-0.7, 0.0, 0.8]),
        sensor_ang=np.array([0.0, -90.0, 0.0]),
        is_flux=np.array([True, False, True]),
    )


def test_public_fixture_carries_all_eight_slanted_sections(sections):
    assert {section.name for section in sections} == SLANTED_NAMES
    assert len(sections) == 8


def test_each_section_targets_one_distinct_circuit(sections):
    circuits = [section.circuit for section in sections]
    assert len(circuits) == len(set(circuits)) == 8


def test_sections_preserve_each_conductors_area(sections):
    for section in sections:
        assert _poly_area(section.vertices) == pytest.approx(
            section.width * section.height, abs=1e-12
        )


def test_sections_keep_their_conductor_centroids(sections):
    for section in sections:
        np.testing.assert_allclose(
            np.mean(section.vertices, axis=0),
            np.array([section.centroid_r, section.centroid_z]),
            atol=1e-12,
        )


def test_bottom_column_shape_keeps_the_catalogued_shear(sections):
    section = next(item for item in sections if item.name == "botcol")
    lower_midpoint = float(np.mean(section.vertices[:2, 0]))
    upper_midpoint = float(np.mean(section.vertices[2:, 0]))
    expected_shift = section.height / np.tan(np.deg2rad(section.angle2))
    assert upper_midpoint - lower_midpoint == pytest.approx(expected_shift, abs=1e-12)


def test_every_slanted_fixture_builds_a_finite_public_column(sections):
    columns = [_public_column(section.vertices) for section in sections]
    assert all(column.shape == (3,) for column in columns)
    assert all(np.isfinite(column).all() for column in columns)


def test_slanted_shape_changes_the_public_column_from_its_box(sections):
    section = next(item for item in sections if item.name == "botcol")
    box = _section_vertices(
        section.centroid_r,
        section.centroid_z,
        section.width,
        section.height,
        0.0,
        0.0,
    )
    assert not np.allclose(
        _public_column(section.vertices),
        _public_column(box),
        rtol=1e-12,
        atol=1e-18,
    )
