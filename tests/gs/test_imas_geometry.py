"""Tests for the IMAS static-IDS machine-geometry reader.

Two layers, matching ``tests/gs/test_geometry.py``'s pattern:

* **Pure-logic** tests build a tiny SYNTHETIC ``pf_active``/``wall``/
  ``magnetics`` triple with imas-python itself (written to a tmp dir, then
  read back) -- no real machine-description database needed, so these run
  anywhere imas-python is installed.  Exercises every reader branch:
  rectangle + annulus + one unsupported element shape, an unset
  ``turns_with_sign``, a 2-unit limiter needing endpoint-chaining, a
  single-point flux loop AND a multi-point "partial" flux loop, and a
  b-probe with an unset ``length``.
* **Integration** tests read the real ITER machine description from
  ``~/public/imasdb/iter_md`` and run the vacuum-field sanity gate;
  skipped when that database is absent.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from imas_ambix.gs import geometry as gsg
from imas_ambix.gs import imas_geometry as img

imas = pytest.importorskip("imas")

# --- fixture: a tiny synthetic pf_active / wall / magnetics triple -----


def _write_synthetic_md(tmp_path: Path) -> tuple[str, str, str]:
    """Write a tiny pf_active/wall/magnetics triple; return their directories."""
    pf_path = str(tmp_path / "pf_active")
    wall_path = str(tmp_path / "wall")
    mag_path = str(tmp_path / "magnetics")

    entry = imas.DBEntry(f"imas:hdf5?path={pf_path}", "x")
    pf = imas.IDSFactory().pf_active()
    pf.ids_properties.homogeneous_time = 2
    pf.coil.resize(3)

    c0 = pf.coil[0]
    c0.name = "COIL_RECT"
    c0.element.resize(2)
    c0.element[0].geometry.geometry_type = 2
    c0.element[0].geometry.rectangle.r = 1.5
    c0.element[0].geometry.rectangle.z = 0.5
    c0.element[0].geometry.rectangle.width = 0.1
    c0.element[0].geometry.rectangle.height = 0.2
    c0.element[0].turns_with_sign = 184.0
    c0.element[1].geometry.geometry_type = 2
    c0.element[1].geometry.rectangle.r = 1.5
    c0.element[1].geometry.rectangle.z = 0.7
    c0.element[1].geometry.rectangle.width = 0.1
    c0.element[1].geometry.rectangle.height = 0.2
    # element[1].turns_with_sign intentionally left UNSET

    c1 = pf.coil[1]
    c1.name = "COIL_ANNULUS"
    c1.element.resize(1)
    c1.element[0].geometry.geometry_type = 5
    c1.element[0].geometry.annulus.r = 2.0
    c1.element[0].geometry.annulus.z = -0.3
    c1.element[0].geometry.annulus.radius_inner = 0.01
    c1.element[0].geometry.annulus.radius_outer = 0.02
    c1.element[0].turns_with_sign = -8.0

    c2 = pf.coil[2]
    c2.name = "COIL_OUTLINE_UNSUPPORTED"
    c2.element.resize(1)
    c2.element[0].geometry.geometry_type = 1
    c2.element[0].geometry.outline.r = [1.0, 1.1, 1.1, 1.0]
    c2.element[0].geometry.outline.z = [-0.1, -0.1, 0.1, 0.1]
    c2.element[0].turns_with_sign = 5.0
    entry.put(pf)
    entry.close()

    entry = imas.DBEntry(f"imas:hdf5?path={wall_path}", "x")
    wall = imas.IDSFactory().wall()
    wall.ids_properties.homogeneous_time = 1
    wall.description_2d.resize(1)
    d2d = wall.description_2d[0]
    d2d.limiter.unit.resize(2)
    u0 = d2d.limiter.unit[0]
    u0.name = "unit0"
    u0.outline.r = [1.0, 2.0, 2.0, 1.0]
    u0.outline.z = [-1.0, -1.0, 1.0, 1.0]
    u1 = d2d.limiter.unit[1]
    u1.name = "unit1"
    u1.outline.r = [1.02, 0.5, 0.4]  # nearest endpoint (1.02,1.03) meets u0's tail
    u1.outline.z = [1.03, 1.5, 1.0]
    entry.put(wall)
    entry.close()

    entry = imas.DBEntry(f"imas:hdf5?path={mag_path}", "x")
    mag = imas.IDSFactory().magnetics()
    mag.ids_properties.homogeneous_time = 2
    mag.b_field_pol_probe.resize(2)
    p0 = mag.b_field_pol_probe[0]
    p0.position.r, p0.position.z = 2.2, 0.3
    p0.poloidal_angle = float(np.pi / 2.0)
    p0.length = 0.05
    p1 = mag.b_field_pol_probe[1]
    p1.position.r, p1.position.z = 2.2, -0.3
    p1.poloidal_angle = 0.0
    # p1.length intentionally left UNSET

    mag.flux_loop.resize(2)
    f0 = mag.flux_loop[0]
    f0.name = "single-point loop"
    f0.position.resize(1)
    f0.position[0].r, f0.position[0].z = 1.8, 1.2
    f1 = mag.flux_loop[1]
    f1.name = "partial loop"
    f1.position.resize(3)
    for i, (r, z) in enumerate([(1.9, 1.1), (2.0, 1.2), (2.1, 1.3)]):
        f1.position[i].r, f1.position[i].z = r, z
    entry.put(mag)
    entry.close()

    return pf_path, wall_path, mag_path


def _synthetic_reader(tmp_path: Path, machine: str = "synth") -> img.ImasGeometryReader:
    pf_path, wall_path, mag_path = _write_synthetic_md(tmp_path)
    return img.ImasGeometryReader(
        machine=machine,
        pf_active_path=pf_path,
        wall_path=wall_path,
        magnetics_path=mag_path,
    )


# --- pure-logic: the reader against the synthetic fixture --------------


def test_reader_produces_geometry_table(tmp_path):
    table = _synthetic_reader(tmp_path).read()
    assert isinstance(table, gsg.GeometryTable)
    assert table.machine == "synth"
    assert table.signature.machine == "synth"
    assert table.signature.key.startswith("synth-")


def test_reader_conforms_to_machine_geometry_reader_protocol(tmp_path):
    reader = _synthetic_reader(tmp_path)
    assert isinstance(reader, gsg.MachineGeometryReader)


def test_pf_active_rectangle_and_annulus_read_turns_with_sign(tmp_path):
    table = _synthetic_reader(tmp_path).read()
    # COIL_RECT's two stacked elements (turns 184 + unset->0) FILL one rectangle
    # (Z 0.4..0.6 + 0.6..0.8) so the canonical table collapses them to one
    # thick cylinder carrying the signed-turns sum (184).  COIL_ANNULUS (a ring,
    # not a filled rectangle) is left as one filament; the unsupported outline
    # coil is dropped.  Signed turns survive the collapse.
    assert len(table.pf_filaments) == 2
    by_rz = {(round(f.r, 3), round(f.z, 3)): f for f in table.pf_filaments}
    rect = by_rz[(1.5, 0.6)]  # bounding-box centre of the two stacked elements
    assert rect.turns == pytest.approx(184.0)  # 184 + 0, signed sum preserved
    assert rect.width == pytest.approx(0.1)
    assert rect.height == pytest.approx(0.4)  # bbox spans both elements
    annulus = by_rz[(2.0, -0.3)]
    assert annulus.turns == pytest.approx(-8.0)  # sign preserved
    assert annulus.width == pytest.approx(0.04)  # 2 * radius_outer
    assert annulus.height == pytest.approx(0.04)


def test_unsupported_element_shape_is_flagged_not_fabricated(tmp_path):
    table = _synthetic_reader(tmp_path).read()
    assert any("unsupported element" in f for f in table.provenance_flags)
    assert any("turns_with_sign unset" in f for f in table.provenance_flags)
    # no filament was fabricated at the outline coil's centroid
    assert not any(f.r == pytest.approx(1.05) for f in table.pf_filaments)


def test_limiter_units_chained_into_one_closed_contour(tmp_path):
    table = _synthetic_reader(tmp_path).read()
    assert len(table.limiter_r) == 4 + 3  # both units' points, in order
    # unit0 starts the chain unchanged; unit1 attaches at its FIRST point
    # (nearest to unit0's tail), so it is not reversed.
    assert table.limiter_r[0] == pytest.approx(1.0)
    assert table.limiter_z[0] == pytest.approx(-1.0)
    assert table.limiter_r[-1] == pytest.approx(0.4)
    assert table.limiter_z[-1] == pytest.approx(1.0)
    assert any("2 units" in f for f in table.provenance_flags)


def test_flux_loop_centroid_and_partial_loop_flag(tmp_path):
    table = _synthetic_reader(tmp_path).read()
    assert len(table.flux_loops) == 2
    single = table.flux_loops[0]
    assert single.r == pytest.approx(1.8)
    assert single.z == pytest.approx(1.2)
    partial = table.flux_loops[1]
    assert partial.r == pytest.approx((1.9 + 2.0 + 2.1) / 3.0)
    assert partial.z == pytest.approx((1.1 + 1.2 + 1.3) / 3.0)
    assert any(
        "partial loop represented by its centroid" in f for f in table.provenance_flags
    )


def test_bpol_probe_angle_degrees_and_empty_length_defaults_to_zero(tmp_path):
    table = _synthetic_reader(tmp_path).read()
    assert len(table.b_probes) == 2
    p0, p1 = table.b_probes
    assert p0.angle_deg == pytest.approx(90.0)
    assert p0.length == pytest.approx(0.05)
    assert p1.angle_deg == pytest.approx(0.0)
    assert p1.length == 0.0  # unset -> 0, never fabricated


def test_r0_and_minor_radius_derived_from_limiter_not_a_constant(tmp_path):
    table = _synthetic_reader(tmp_path).read()
    lr = table.limiter_r
    assert table.r0 == pytest.approx((max(lr) + min(lr)) / 2.0)
    assert table.minor_radius == pytest.approx((max(lr) - min(lr)) / 2.0)
    assert table.r0 != gsg.MAST_R0
    assert table.minor_radius != gsg.MAST_A


def test_sensor_map_covers_every_sensor_cleanly(tmp_path):
    table = _synthetic_reader(tmp_path).read()
    assert len(table.sensor_map) == len(table.b_probes) + len(table.flux_loops)
    assert all(not m.flag for m in table.sensor_map)  # identity map, nothing ambiguous
    assert table.unmatched_amb == []
    assert table.amc_current_channels == []  # no signal-channel concept for IMAS MD


def test_two_machines_never_collide_on_the_same_signature_key(tmp_path):
    """Two readers built from the SAME geometry but different ``machine`` tags
    must produce different signature keys (never resolve through the same
    cache entry)."""
    pf_path, wall_path, mag_path = _write_synthetic_md(tmp_path)
    t_a = img.ImasGeometryReader(
        machine="alpha",
        pf_active_path=pf_path,
        wall_path=wall_path,
        magnetics_path=mag_path,
    ).read()
    t_b = img.ImasGeometryReader(
        machine="beta",
        pf_active_path=pf_path,
        wall_path=wall_path,
        magnetics_path=mag_path,
    ).read()
    assert t_a.signature.key != t_b.signature.key
    assert t_a.signature.digest == t_b.signature.digest  # same geometry, same digest
    assert t_a.signature.key.startswith("alpha-")
    assert t_b.signature.key.startswith("beta-")


# --- signature byte-stability: MAST reader must be an untouched adapter --


def test_mast_reader_key_matches_direct_setup_signature_no_prefix():
    """The 'mast' default must NOT change the untagged key format."""
    geom = {
        "magpr_r": np.array([0.18]),
        "magpr_z": np.array([1.0]),
        "magpr_ang": np.array([90.0]),
        "silop_r": np.array([0.5]),
        "silop_z": np.array([0.6]),
        "fcoil_r": np.array([0.3]),
        "fcoil_z": np.array([0.4]),
        "fcoil_turns": np.array([1.0]),
        "limiterr": np.array([1.0, 1.5]),
        "limiterz": np.array([-0.5, 0.5]),
    }
    sig = gsg.setup_signature(geom)
    assert sig.machine == "mast"
    assert not sig.key.startswith("mast-")
    assert sig.key.startswith("mp1-fl1-fc1-lim2-")


# --- integration: the real ITER machine description (skipped when absent) --

_ITER_MD_BASE = Path(os.path.expanduser("~/public/imasdb/iter_md/3"))
_ITER_PF_ACTIVE = str(_ITER_MD_BASE / "111001" / "4")
_ITER_WALL = str(_ITER_MD_BASE / "116000" / "2")
_ITER_MAGNETICS = str(_ITER_MD_BASE / "150100" / "3")
_HAVE_ITER_MD = all(
    (Path(p) / "master.h5").exists()
    for p in (_ITER_PF_ACTIVE, _ITER_WALL, _ITER_MAGNETICS)
)
_skip_no_iter_md = pytest.mark.skipif(
    not _HAVE_ITER_MD, reason="ITER machine-description entries not available"
)


@pytest.fixture(scope="module")
def iter_table():
    reader = img.ImasGeometryReader(
        machine="iter",
        pf_active_path=_ITER_PF_ACTIVE,
        wall_path=_ITER_WALL,
        magnetics_path=_ITER_MAGNETICS,
        entry_id=111001,
    )
    return reader.read()


@_skip_no_iter_md
def test_iter_machine_description_ingests_a_geometry_table(iter_table):
    cov = iter_table.coverage()
    assert cov["n_bprobe"] > 0
    assert cov["n_fluxloop"] > 0
    assert cov["n_pf_filament"] > 0
    assert cov["n_limiter"] > 3
    assert iter_table.r0 > 0.0
    assert iter_table.minor_radius > 0.0
    # geometry-derived, not the MAST device constants
    assert iter_table.r0 != gsg.MAST_R0
    assert iter_table.minor_radius != gsg.MAST_A


@_skip_no_iter_md
def test_iter_vacuum_field_gate(iter_table, tmp_path):
    """The vacuum-field sanity gate (§5 Workstream C acceptance criteria).

    (a) one real PF filament's finite-area field vs the exact analytic
        point-filament formula, at a point far from the coil;
    (b) the coil-driven psi field is finite everywhere on an
        EquilibriumGrid built from the table (limiter mask + Green's grid
        are geometry-only, independent of operator.classify_circuits);
    (c) sensors respond to a real coil's field -- computed with the SAME
        finite-area kernel PatchBasis uses, applied DIRECTLY to the
        table's filaments/sensors (bypassing
        operator.classify_circuits, whose KNOWN-PF match is hardcoded to
        MAST coil centroids/amc-channel names -- confirmed below to leave
        PatchBasis.m_coil empty for this table, an out-of-scope operator.py
        gap flagged in the module docstring and the workstream report).

    Also confirms `PatchBasis.from_table` ASSEMBLES for a second machine
    with zero solver-code changes (the literal acceptance bar), while
    recording that its m_coil pathway is a known-coil-free by-product of
    the operator.py gap above.
    """
    from imas_ambix.gs.cylinder import hybrid_greens
    from imas_ambix.gs.operator import (
        build_operator,
        classify_circuits,
        greens_bz_br,
        greens_psi,
    )
    from imas_ambix.latent.gs_solve import EquilibriumGrid
    from imas_ambix.latent.patch_basis import PatchBasis

    table = iter_table
    result: dict[str, object] = {
        "schema": "imas-geometry-iter-gate-v0",
        "machine": table.machine,
    }

    # (a) finite-area kernel vs exact point-filament formula, far field.
    fil = max(table.pf_filaments, key=lambda f: max(abs(f.width), abs(f.height)))
    extent = max(abs(fil.width), abs(fil.height))
    far_r = np.array([fil.r + 15.0 * extent])
    far_z = np.array([fil.z + 15.0 * extent])
    psi_fa, br_fa, bz_fa = hybrid_greens(
        far_r, far_z, fil.r, fil.z, abs(fil.width), abs(fil.height)
    )
    psi_pt = greens_psi(far_r, far_z, fil.r, fil.z)
    bz_pt, br_pt = greens_bz_br(far_r, far_z, fil.r, fil.z)
    psi_rel = float(abs(psi_fa[0] - psi_pt[0]) / max(abs(psi_pt[0]), 1e-300))
    br_rel = float(abs(br_fa[0] - br_pt[0]) / max(abs(br_pt[0]), 1e-300))
    bz_rel = float(abs(bz_fa[0] - bz_pt[0]) / max(abs(bz_pt[0]), 1e-300))
    result["a_analytic_loop_check"] = {
        "filament_r_z_w_h": [fil.r, fil.z, fil.width, fil.height],
        "far_test_point_r_z": [float(far_r[0]), float(far_z[0])],
        "psi_rel_error": psi_rel,
        "br_rel_error": br_rel,
        "bz_rel_error": bz_rel,
        "threshold": 1.0e-3,
        "pass": max(psi_rel, br_rel, bz_rel) < 1.0e-3,
    }
    assert psi_rel < 1.0e-3, f"psi far-field rel error {psi_rel:.2e} >= 1e-3"
    assert br_rel < 1.0e-3, f"Br far-field rel error {br_rel:.2e} >= 1e-3"
    assert bz_rel < 1.0e-3, f"Bz far-field rel error {bz_rel:.2e} >= 1e-3"

    # (b) coil-driven psi is finite everywhere on the geometry-derived grid.
    grid = EquilibriumGrid.from_table(table, nr=33, nz=49)
    psi_map = np.zeros(grid.flat_r.size, dtype=np.float64)
    for f in table.pf_filaments:
        psi_f, _br, _bz = hybrid_greens(
            grid.flat_r,
            grid.flat_z,
            f.r,
            f.z,
            max(abs(f.width), 0.01),
            max(abs(f.height), 0.01),
        )
        psi_map += f.turns * psi_f
    n_finite = int(np.isfinite(psi_map).sum())
    result["b_psi_contour_check"] = {
        "grid_nr_nz": [grid.nr, grid.nz],
        "n_grid_points": int(psi_map.size),
        "n_finite": n_finite,
        "psi_min": float(np.nanmin(psi_map)),
        "psi_max": float(np.nanmax(psi_map)),
        "pass": n_finite == psi_map.size,
    }
    assert n_finite == psi_map.size, "coil-driven psi has non-finite grid points"

    # (c) sensors respond -- direct kernel, bypassing classify_circuits.
    all_r = np.array([m.r for m in table.sensor_map])
    all_z = np.array([m.z for m in table.sensor_map])
    is_flux = np.array([m.kind == "flux_loop" for m in table.sensor_map])
    ang = np.array(
        [0.0 if m.angle_deg is None else m.angle_deg for m in table.sensor_map]
    )
    sensor_resp = np.zeros(all_r.size, dtype=np.float64)
    fil_group = [f for f in table.pf_filaments if f.circuit == fil.circuit]
    for f in fil_group:
        psi_s, br_s, bz_s = hybrid_greens(
            all_r, all_z, f.r, f.z, max(abs(f.width), 0.01), max(abs(f.height), 0.01)
        )
        proj = br_s * np.cos(np.deg2rad(ang)) + bz_s * np.sin(np.deg2rad(ang))
        sensor_resp += f.turns * np.where(is_flux, psi_s, proj)
    dist = np.hypot(all_r - fil.r, all_z - fil.z)
    near = dist < 3.0
    far = dist > 10.0
    n_finite_sensors = int(np.isfinite(sensor_resp).sum())
    near_mag = float(np.abs(sensor_resp[near]).mean()) if near.any() else 0.0
    far_mag = float(np.abs(sensor_resp[far]).mean()) if far.any() else 0.0
    result["c_sensor_response_check"] = {
        "note": (
            "direct hybrid_greens kernel applied to the coil's own filaments + "
            "the table's sensor list -- NOT through PatchBasis.m_coil, which "
            "operator.classify_circuits leaves empty for this (non-MAST) table "
            "(see m_coil_shape below)."
        ),
        "coil_circuit": fil.circuit,
        "n_sensor": int(all_r.size),
        "n_finite": n_finite_sensors,
        "n_near_lt_3m": int(near.sum()),
        "n_far_gt_10m": int(far.sum()),
        "mean_abs_response_near": near_mag,
        "mean_abs_response_far": far_mag,
        "pass": n_finite_sensors == all_r.size and near_mag > far_mag > 0.0,
    }
    assert n_finite_sensors == all_r.size
    assert near_mag > far_mag > 0.0, "near sensors do not respond more than far sensors"

    # PatchBasis assembles with ZERO solver-code changes (literal acceptance bar).
    basis = PatchBasis.from_table(table, nr=33, nz=49, cache_dir=tmp_path)
    classes = classify_circuits(table.pf_filaments, table.amc_current_channels)
    n_known = sum(1 for c in classes if c.role == "known_pf")
    fwd = build_operator(table)
    result["patch_basis_assembly"] = {
        "assembled": True,
        "g_pg_shape": list(basis.g_pg.shape),
        "m_sens_shape": list(basis.m_sens.shape),
        "m_coil_shape": list(basis.m_coil.shape),
        "psi_coil_grid_shape": list(basis.psi_coil_grid.shape),
        "n_known_pf_circuits": n_known,
        "n_total_circuits": len(classes),
        "operator_gap": (
            "operator.classify_circuits hardcodes MAST coil centroids "
            "(_PF_COIL_CENTROID) and amc-channel names (_PF_COIL_AMC); this "
            "table's circuits are all outside MAST's centroid tolerance and "
            "have no amc channels, so every circuit classifies "
            "'inferred_passive' and PatchBasis.m_coil / psi_coil_grid are "
            "(*, 0) -- a real, load-bearing, out-of-scope (operator.py) gap "
            "for the KNOWN-PF-coil pathway on any non-MAST machine."
        ),
    }
    # consistency: the same sensor count/grid feed both the operator and PatchBasis
    assert fwd.g_plasma.shape[0] == basis.m_sens.shape[0] == len(table.sensor_map)
    assert basis.g_pg.shape[0] == basis.psi_coil_grid.shape[0]
    assert n_known == 0, (
        "classify_circuits gap has been fixed elsewhere -- update this pin"
    )

    out_path = (
        Path(__file__).resolve().parents[2]
        / "imas_ambix"
        / "gs"
        / "artifacts"
        / "imas_geometry_iter.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import json

    payload = {
        **result,
        "geometry_summary": table.coverage(),
        "r0": table.r0,
        "minor_radius": table.minor_radius,
        "signature_key": table.signature.key,
        "n_provenance_flags": len(table.provenance_flags),
    }
    out_path.write_text(json.dumps(payload, indent=2))
