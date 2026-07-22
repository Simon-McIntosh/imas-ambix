"""Wiring the analytic polygon-section kernel into the forward operator.

A :class:`PolygonSection` replaces the axis-aligned bounding-box column of a
chosen fcoil circuit with the exact Urankar-Part-V shaped field.  The override
is opt-in: an operator built from a table with no polygon sections is
byte-identical to before, and where all sensors sit in the finite-area near
band a box polygon reproduces the rectangular-kernel column it replaces.
"""

from __future__ import annotations

import numpy as np

import imas_ambix.gs.geometry as gsg
from imas_ambix.gs.cylinder import cylinder_greens
from imas_ambix.gs.geometry import PolygonSection, parallelogram_vertices
from imas_ambix.gs.operator import (
    _project_bprobe,
    build_operator,
    polygon_section_column,
)

# ----------------------------------------------------------- vertex builder --


def test_parallelogram_vertices_rectangle_at_zero_angle():
    v = parallelogram_vertices(0.9, 0.1, 0.12, 0.18, 0.0)
    exp = np.array([(0.84, 0.01), (0.96, 0.01), (0.96, 0.19), (0.84, 0.19)])
    np.testing.assert_allclose(v, exp, atol=1e-12)


def test_parallelogram_vertices_area_preserved_and_sheared():
    v = parallelogram_vertices(1.0, 0.0, 0.10, 0.20, 45.0)
    rolled = np.roll(v, -1, axis=0)
    area = 0.5 * abs(np.sum(v[:, 0] * rolled[:, 1] - rolled[:, 0] * v[:, 1]))
    assert abs(area - 0.10 * 0.20) < 1e-12
    dr = 0.5 * 0.20 * np.tan(np.deg2rad(45.0))
    # the left edge tilts by 2·dr in R across the full height
    assert abs((v[3, 0] - v[0, 0]) - 2 * dr) < 1e-12


# ----------------------------------------------------------- column builder --

_SR = np.array([1.30, 1.30, 0.70])
_SZ = np.array([0.30, 0.50, 0.40])
_SANG = np.array([90.0, 0.0, 0.0])
_ISF = np.array([False, False, True])


def test_polygon_column_reduces_to_box_kernel():
    """A box PolygonSection column == the exact finite-area rectangle column."""
    a, z0, da, dz = 1.0, 0.30, 0.12, 0.18
    v = parallelogram_vertices(a, z0, da, dz, 0.0)
    col = polygon_section_column(v, 1.0, _SR, _SZ, _SANG, _ISF)
    psi, br, bz = cylinder_greens(_SR, _SZ, a, z0, da, dz)
    ref = np.where(_ISF, psi, _project_bprobe(bz, br, _SANG))
    np.testing.assert_allclose(col, ref, rtol=1e-9, atol=1e-20)


def test_polygon_column_shaped_differs_from_box():
    box = parallelogram_vertices(1.0, 0.30, 0.12, 0.18, 0.0)
    slant = parallelogram_vertices(1.0, 0.30, 0.12, 0.18, 45.0)
    c_box = polygon_section_column(box, 1.0, _SR, _SZ, _SANG, _ISF)
    c_slant = polygon_section_column(slant, 1.0, _SR, _SZ, _SANG, _ISF)
    rel = np.max(np.abs(c_slant - c_box)) / np.max(np.abs(c_box))
    assert rel > 1e-3, f"shaping should move the column, got {rel:.2e}"


# ------------------------------------------------ build_operator integration --


def _passive_table(polygon_sections=None) -> gsg.GeometryTable:
    """Minimal synthetic table: one INFERRED passive box circuit + 3 sensors.

    No amc channels ⇒ every circuit is inferred passive; sensors sit in the
    finite-area near band of the box (< 3·max(da,dz) from its centroid).
    """
    sig = gsg.SetupSignature(
        n_bprobe=2,
        n_fluxloop=1,
        n_pf_filament=1,
        n_limiter=4,
        digest="deadbeef0000abcd",
    )
    b_probes = [
        gsg.BProbe(index=0, r=1.30, z=0.30, angle_deg=90.0, length=0.001),
        gsg.BProbe(index=1, r=1.30, z=0.50, angle_deg=0.0, length=0.001),
    ]
    flux_loops = [gsg.FluxLoop(index=0, r=0.70, z=0.40)]
    pf_filaments = [
        gsg.PFFilament(
            r=1.0, z=0.30, turns=1.0, width=0.12, height=0.18, circuit=2, xmult=1.0
        )
    ]
    sensor_map = [
        gsg.SensorMapping("b1", "b_probe", 0, 1.30, 0.30, 90.0, 0.001, ""),
        gsg.SensorMapping("b2", "b_probe", 1, 1.30, 0.50, 0.0, 0.001, ""),
        gsg.SensorMapping("f1", "flux_loop", 0, 0.70, 0.40, None, 0.001, ""),
    ]
    return gsg.GeometryTable(
        signature=sig,
        shots=[1],
        b_probes=b_probes,
        flux_loops=flux_loops,
        pf_filaments=pf_filaments,
        limiter_r=[0.3, 1.6, 1.6, 0.3],
        limiter_z=[-1.0, -1.0, 1.0, 1.0],
        sensor_map=sensor_map,
        passive_structures=[],
        amc_current_channels=[],
        unmatched_amb=[],
        polygon_sections=polygon_sections or [],
    )


def test_build_operator_empty_polygon_is_baseline():
    """No polygon sections ⇒ the passive block is byte-identical to before."""
    base = build_operator(_passive_table())
    same = build_operator(_passive_table(polygon_sections=[]))
    assert base.g_passive.shape == (3, 1)
    np.testing.assert_array_equal(base.g_passive, same.g_passive)


def test_build_operator_box_polygon_matches_baseline():
    """A box PolygonSection reproduces the bounding-box column (near-band)."""
    base = build_operator(_passive_table())
    box = PolygonSection(
        circuit=2,
        vertices=parallelogram_vertices(1.0, 0.30, 0.12, 0.18, 0.0),
        xmult=1.0,
    )
    shaped = build_operator(_passive_table(polygon_sections=[box]))
    np.testing.assert_allclose(shaped.g_passive, base.g_passive, rtol=1e-9, atol=1e-20)


def test_build_operator_slanted_polygon_reshapes_column():
    """A slanted PolygonSection reshapes the passive column (≠ box, = kernel)."""
    base = build_operator(_passive_table())
    slant = PolygonSection(
        circuit=2,
        vertices=parallelogram_vertices(1.0, 0.30, 0.12, 0.18, 40.0),
        xmult=1.0,
    )
    shaped = build_operator(_passive_table(polygon_sections=[slant]))
    rel = np.max(np.abs(shaped.g_passive - base.g_passive)) / np.max(
        np.abs(base.g_passive)
    )
    assert rel > 1e-3, f"slanted section should reshape the column, got {rel:.2e}"
    # and it equals the direct polygon-section column at the operator's sensors
    expect = polygon_section_column(slant.vertices, 1.0, _SR, _SZ, _SANG, _ISF)
    np.testing.assert_allclose(shaped.g_passive[:, 0], expect, rtol=1e-12, atol=1e-20)
