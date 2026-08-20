"""Wiring the analytic polygon-section kernel into the forward operator.

A :class:`PolygonSection` replaces the axis-aligned bounding-box column of a
chosen fcoil circuit with the exact Urankar-Part-V shaped field.  The override
is opt-in and has no effect when a table declares no polygon sections.  Where
all sensors sit in the finite-area near band, a box polygon reproduces the
rectangular-kernel column it replaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from imas_ambix.gs.cylinder import cylinder_greens
from imas_ambix.gs.operator import (
    _project_bprobe,
    build_operator,
    polygon_section_column,
)


class _Filament(SimpleNamespace):
    pass


class _SensorMapping(SimpleNamespace):
    def __init__(self, amb_channel, kind, efm_index, r, z, angle_deg, residual_m, flag):
        super().__init__(
            amb_channel=amb_channel,
            kind=kind,
            efm_index=efm_index,
            r=r,
            z=z,
            angle_deg=angle_deg,
            residual_m=residual_m,
            flag=flag,
        )


class _Signature(SimpleNamespace):
    @property
    def key(self):
        counts = (
            f"mp{self.n_bprobe}-fl{self.n_fluxloop}-fc{self.n_pf_filament}"
            f"-lim{self.n_limiter}"
        )
        return f"{counts}-{self.digest}"


class _GeometryFixture(SimpleNamespace):
    def __init__(self, **values):
        values.setdefault("circuit_drives", [])
        values.setdefault("active_circuits", [])
        values.setdefault("provenance_flags", [])
        values.setdefault("r0", 0.85)
        values.setdefault("minor_radius", 0.65)
        values.setdefault("passive_structures", [])
        super().__init__(**values)


def parallelogram_vertices(r, z, width, height, angle_deg):
    shear = 0.5 * height * np.tan(np.deg2rad(angle_deg))
    return np.array(
        [
            (r - width / 2 - shear, z - height / 2),
            (r + width / 2 - shear, z - height / 2),
            (r + width / 2 + shear, z + height / 2),
            (r - width / 2 + shear, z + height / 2),
        ]
    )


@dataclass(frozen=True)
class PolygonSection:
    circuit: int
    vertices: np.ndarray
    xmult: float
    name: str = ""


_fixtures = SimpleNamespace(
    BProbe=SimpleNamespace,
    FluxLoop=SimpleNamespace,
    GeometryTable=_GeometryFixture,
    PFFilament=_Filament,
    SensorMapping=_SensorMapping,
    SetupSignature=_Signature,
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
_SANG = np.array([-90.0, 0.0, 0.0])
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


def _passive_table(polygon_sections=None) -> _fixtures.GeometryTable:
    """Minimal synthetic table: one INFERRED passive box circuit + 3 sensors.

    No amc channels ⇒ every circuit is inferred passive; sensors sit in the
    finite-area near band of the box (< 3·max(da,dz) from its centroid).
    """
    sig = _fixtures.SetupSignature(
        n_bprobe=2,
        n_fluxloop=1,
        n_pf_filament=1,
        n_limiter=4,
        digest="deadbeef0000abcd",
    )
    b_probes = [
        _fixtures.BProbe(index=0, r=1.30, z=0.30, angle_deg=-90.0, length=0.001),
        _fixtures.BProbe(index=1, r=1.30, z=0.50, angle_deg=0.0, length=0.001),
    ]
    flux_loops = [_fixtures.FluxLoop(index=0, r=0.70, z=0.40)]
    pf_filaments = [
        _fixtures.PFFilament(
            r=1.0, z=0.30, turns=1.0, width=0.12, height=0.18, circuit=2, xmult=1.0
        )
    ]
    sensor_map = [
        _fixtures.SensorMapping("b1", "b_probe", 0, 1.30, 0.30, -90.0, 0.001, ""),
        _fixtures.SensorMapping("b2", "b_probe", 1, 1.30, 0.50, 0.0, 0.001, ""),
        _fixtures.SensorMapping("f1", "flux_loop", 0, 0.70, 0.40, None, 0.001, ""),
    ]
    return _fixtures.GeometryTable(
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
    """No polygon sections leave the passive block unchanged."""
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
