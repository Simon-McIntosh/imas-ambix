"""Ingesting MAST's eight slanted passive sections as polygon overrides.

The vessel end-column crowns and the P2 arm / divertor-plate structures are
parallelograms in the MAST Data Catalog ``pf_passive`` group but were carried
as axis-aligned bounding boxes.  These tests pin the shape-angle convention and
the box→parallelogram ingestion (area preserved, field reshaped) without
touching the level-2 data store.
"""

from __future__ import annotations

import numpy as np

from imas_ambix.gs.geometry import (
    _MAST_SLANTED_PASSIVES,
    PFFilament,
    mast_slanted_polygon_sections,
    parallelogram_vertices,
    shaped_section_vertices,
)


def _poly_area(v: np.ndarray) -> float:
    rolled = np.roll(v, -1, axis=0)
    return 0.5 * abs(np.sum(v[:, 0] * rolled[:, 1] - rolled[:, 0] * v[:, 1]))


def _canon(v: np.ndarray) -> np.ndarray:
    return v[np.lexsort((v[:, 1], v[:, 0]))]


def test_shape_angle2_matches_parallelogram_helper():
    """The catalog side-edge shear (angle2) == parallelogram_vertices(90−a2)."""
    r, z, w, h, a2 = 0.2354, -2.0250, 0.0471, 0.30, 295.324  # botcol crown
    cat = shaped_section_vertices(r, z, w, h, 0.0, a2)
    helper = parallelogram_vertices(r, z, w, h, 90.0 - a2)
    np.testing.assert_allclose(_canon(cat), _canon(helper), atol=1e-12)


def test_shaped_vertices_preserve_area_both_shears():
    """Either shear keeps the true cross-section area (→ ring R unchanged)."""
    for a1, a2 in ((45.0, 0.0), (0.0, 320.0), (0.0, 0.0)):
        v = shaped_section_vertices(0.35, -1.63, 0.1265, 0.041, a1, a2)
        assert abs(_poly_area(v) - 0.1265 * 0.041) < 1e-12


def test_shaped_vertices_axis_aligned_at_zero_angle():
    v = shaped_section_vertices(0.9, 0.1, 0.12, 0.18, 0.0, 0.0)
    exp = np.array([(0.84, 0.01), (0.96, 0.01), (0.96, 0.19), (0.84, 0.19)])
    np.testing.assert_allclose(_canon(v), _canon(exp), atol=1e-12)


def test_slanted_sections_built_for_all_eight():
    """One single-filament passive circuit per catalog centroid ⇒ 8 sections,
    each area-preserving and keyed to its circuit."""
    filaments = [
        PFFilament(r=ref_r, z=ref_z, turns=1.0, width=0.05, height=0.10,
                   circuit=100 + i, xmult=1.0)
        for i, (_name, ref_r, ref_z, _a1, _a2) in enumerate(_MAST_SLANTED_PASSIVES)
    ]
    sections = mast_slanted_polygon_sections(filaments)
    assert len(sections) == len(_MAST_SLANTED_PASSIVES) == 8
    circuits = {ps.circuit for ps in sections}
    assert circuits == {100 + i for i in range(8)}
    for ps in sections:
        assert abs(_poly_area(ps.vertices) - 0.05 * 0.10) < 1e-12
        assert ps.xmult == 1.0
        assert ps.name


def test_non_mast_filaments_yield_no_sections():
    """A filament set far from every catalog centroid gets no override."""
    filaments = [
        PFFilament(r=5.0, z=5.0, turns=1.0, width=0.1, height=0.1,
                   circuit=1, xmult=1.0)
    ]
    assert mast_slanted_polygon_sections(filaments) == []


def test_multi_filament_circuit_is_skipped():
    """A slanted passive is single-filament in efm; a multi-filament match is
    skipped rather than guessed."""
    name, ref_r, ref_z, _a1, _a2 = _MAST_SLANTED_PASSIVES[0]
    filaments = [
        PFFilament(r=ref_r, z=ref_z, turns=1.0, width=0.05, height=0.10,
                   circuit=7, xmult=0.5),
        PFFilament(r=ref_r + 0.001, z=ref_z, turns=1.0, width=0.05, height=0.10,
                   circuit=7, xmult=0.5),
    ]
    assert mast_slanted_polygon_sections(filaments) == []
