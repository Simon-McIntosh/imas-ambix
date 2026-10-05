"""Tests for the SELENE-deck to per-phase IDS netCDF converter.

The decks are synthetic files written in the three facility grammars
(``EQSLE.DATA``, ``geo.in``, ``coil_vv_*.dat``) with a known element layout so
every parsed value, provenance line range and netCDF read-back can be asserted.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import imas
import numpy as np
import pytest

from imas_ambix.data import selene_deck as sd

REAL_EQSLE = Path(
    "/home/ITER/mcintos/.config/reckon/crew/reports/imas-ambix/"
    "jtmm-geometry-source/copies/jt-60sa/work/efit_jt60sa/EQSLE.DATA"
)

EQSLE_TEXT = """\
 &DSK DEVICE='JT-60SA',IWRITE=65, /$
 &EQU IRESET=2,$
 / $
 2               PF COIL
 2 2
  1.00000  0.50000   0.10000   0.10000   0.20000  PF1
  1.00000  0.50000   0.20000   0.10000   0.20000
  2.00000  1.50000   0.30000   0.05000   0.06000  PF2
  2.00000  1.50000   0.40000   0.05000   0.06000
 3    NV, Vturn,Vr,Vz,Va,Vb,Vrho
  1.00000   4.90000   0.30000   0.04000   0.25000  7.76e-07
  1.00000   4.80000   0.60000   0.04000   0.25000  7.76e-07
  1.00000   4.70000   0.90000   0.04000   0.25000  7.20e-07
 2
  1.400  0.0  3.500 -60.0  60.0  2   VV OUTER SKIN
  1.400  0.0  3.500  60.0 300.0  2
  3.316  1.897  3.612  1.516   0.0  1  -Outer First Wall
  1.705 -1.757  1.705  1.827   0.0  1  Inner First Wall
  1.550  0.0  3.750   0.0   53.52  2  TFC INSIDE
  2.430  1.190  2.270  53.52 90.0  2
  1.550  0.0  4.010   0.0   53.52  2  TFC OUTSIDE
  2.430  1.190  2.530  53.52 90.0  2
"""

GEO_IN_TEXT = """\
JT-60SA
 2               Limiter
 1.705 -1.875
 1.705  1.875
 2               Magnetic Probe
 4.7355   -0.0283     0.28
 1.6356   -0.0002   180.21
 2               Flux Loop
 1.8798    2.9331
 2.3023   -3.1209
"""

COIL_VV_TEXT = """\
 2
 1
 2 2
  1.00000  0.50000   0.10000   0.10000   0.20000  PF1
  1.00000  0.50000   0.20000   0.10000   0.20000
  2.00000  1.50000   0.30000   0.05000   0.06000  PF2
  2.00000  1.50000   0.40000   0.05000   0.06000
 2   NV,     Vturn,Vr,Vz,Va,Vb,Vrho
  1.00000   4.95480   0.12482   0.03600   0.24969  7.20e-007
  1.00000   4.93720   0.37383   0.03600   0.25000  7.20e-007
"""


@pytest.fixture()
def decks(tmp_path: Path):
    eqsle = tmp_path / "EQSLE.DATA"
    eqsle.write_text(EQSLE_TEXT, encoding="latin-1")
    geo = tmp_path / "geo.in"
    geo.write_text(GEO_IN_TEXT, encoding="latin-1")
    coil_vv = tmp_path / "coil_vv_OP2.dat"
    coil_vv.write_text(COIL_VV_TEXT, encoding="latin-1")
    return eqsle, geo, coil_vv


def test_eqsle_parses_pf_coils(decks):
    eqsle, _, _ = decks
    deck = sd.parse_eqsle_deck(eqsle)
    assert [c.name for c in deck.pf_coils] == ["CS1", "CS2"]
    first = deck.pf_coils[0].elements[0]
    assert first.turns == pytest.approx(1.0)
    assert first.r == pytest.approx(0.5)
    assert first.z == pytest.approx(0.1)
    assert first.dr == pytest.approx(0.1)
    assert first.dz == pytest.approx(0.2)
    # Deck extent columns are full extents, so width/height equal dR/dZ.
    assert first.width == pytest.approx(0.1)
    assert first.height == pytest.approx(0.2)
    assert len(deck.pf_coils[0].elements) == 2
    # Positive turns only; sign is a circuit property, not a turn sign.
    assert all(e.turns > 0 for c in deck.pf_coils for e in c.elements)


def test_eqsle_parses_vessel_tfc_and_wall(decks):
    eqsle, _, _ = decks
    deck = sd.parse_eqsle_deck(eqsle)
    assert len(deck.vessel) == 3
    fil = deck.vessel[0]
    assert fil.r == pytest.approx(4.9)
    assert fil.z == pytest.approx(0.3)
    assert fil.dr == pytest.approx(0.04)
    assert fil.dz == pytest.approx(0.25)
    assert fil.resistivity == pytest.approx(7.76e-7)
    assert len(deck.tfc_inside) == 2
    assert len(deck.tfc_outside) == 2
    kinds = {seg.kind for seg in deck.limiter_and_first_wall}
    assert kinds == {"line", "arc"}
    assert any("First Wall" in s.comment for s in deck.limiter_and_first_wall)
    line = next(s for s in deck.limiter_and_first_wall if s.kind == "line")
    (r1, z1), (r2, z2) = line.line_points()
    assert (r1, z1) == pytest.approx((3.316, 1.897))
    assert (r2, z2) == pytest.approx((3.612, 1.516))


def test_provenance_carries_path_sha256_and_line_range(decks):
    eqsle, geo, _ = decks
    deck = sd.parse_eqsle_deck(eqsle)
    digest = hashlib.sha256(eqsle.read_bytes()).hexdigest()
    n_lines = len(eqsle.read_text(encoding="latin-1").splitlines())

    elements = [e for c in deck.pf_coils for e in c.elements]
    elements += list(deck.vessel)
    elements += list(deck.tfc_inside) + list(deck.tfc_outside)
    elements += list(deck.limiter_and_first_wall)
    for element in elements:
        prov = element.provenance
        assert prov.source == str(eqsle)
        assert prov.sha256 == digest
        assert 1 <= prov.line_start <= prov.line_end <= n_lines

    assert deck.pf_coils[0].elements[0].provenance.line_range() == [6, 6]
    assert deck.vessel[0].provenance.line_range() == [11, 11]
    assert deck.tfc_inside[0].provenance.line_range() == [19, 19]

    geo_parsed = sd.parse_geo_in(geo)
    assert geo_parsed.probes[0].provenance.sha256 == hashlib.sha256(
        geo.read_bytes()
    ).hexdigest()


def test_geo_in_parses_probes_and_flux_loops(decks):
    _, geo, _ = decks
    parsed = sd.parse_geo_in(geo)
    assert len(parsed.probes) == 2
    assert len(parsed.flux_loops) == 2
    probe = parsed.probes[0]
    assert probe.r == pytest.approx(4.7355)
    assert probe.z == pytest.approx(-0.0283)
    assert probe.angle == pytest.approx(0.28)
    loop = parsed.flux_loops[0]
    assert loop.r == pytest.approx(1.8798)
    assert loop.z == pytest.approx(2.9331)
    assert loop.angle is None


def test_coil_vv_vessel_block_parses(decks):
    _, _, coil_vv = decks
    parsed = sd.parse_coil_vv_deck(coil_vv)
    assert len(parsed.vessel) == 2
    fil = parsed.vessel[0]
    assert fil.r == pytest.approx(4.9548)
    assert fil.z == pytest.approx(0.12482)
    assert fil.dr == pytest.approx(0.036)
    assert fil.dz == pytest.approx(0.24969)
    assert fil.resistivity == pytest.approx(7.20e-7)


def test_writer_reads_back_at_dd_4_1_1(decks, tmp_path: Path):
    eqsle, geo, coil_vv = decks
    deck = sd.parse_eqsle_deck(eqsle)
    geo_parsed = sd.parse_geo_in(geo)
    coil_parsed = sd.parse_coil_vv_deck(coil_vv)
    receipt = sd.write_phase_description(
        phase="OP1",
        out_dir=tmp_path,
        eqsle=deck,
        geo=geo_parsed,
        coil_vv=coil_parsed,
        converter_commit="deadbeef",
    )
    phase_dir = tmp_path / "OP1"

    def read(name):
        with imas.DBEntry(phase_dir / f"{name}.nc", "r", dd_version="4.1.1") as e:
            return e.get(name)

    pf = read("pf_active")
    assert str(pf.coil[0].name) == "CS1"
    assert float(pf.coil[0].element[0].turns_with_sign) > 0
    assert float(pf.coil[0].element[0].geometry.rectangle.r) == pytest.approx(0.5)
    assert float(pf.coil[0].element[0].geometry.rectangle.width) == pytest.approx(0.1)
    assert float(pf.coil[0].element[0].geometry.rectangle.height) == pytest.approx(0.2)
    assert np.asarray(pf.circuit[0].connections).size >= 1

    passive = read("pf_passive")
    assert float(passive.loop[0].element[0].geometry.rectangle.r) == pytest.approx(
        4.9548
    )
    # Va/Vb are full extents, so width equals the deck column directly.
    assert float(passive.loop[0].element[0].geometry.rectangle.width) == pytest.approx(
        0.036
    )
    assert float(
        passive.loop[0].element[0].geometry.rectangle.height
    ) == pytest.approx(0.24969)
    assert float(passive.loop[0].resistivity) == pytest.approx(7.20e-7)

    mag = read("magnetics")
    assert float(mag.b_field_pol_probe[0].position.r) == pytest.approx(4.7355)
    assert float(mag.b_field_pol_probe[0].poloidal_angle) == pytest.approx(
        math.radians(0.28)
    )
    assert float(mag.flux_loop[0].position[0].r) == pytest.approx(1.8798)

    wall = read("wall")
    outline_r = np.asarray(wall.description_2d[0].limiter.unit[0].outline.r)
    assert outline_r.size >= 2
    assert outline_r[0] == pytest.approx(3.316)

    tf = read("tf")
    # The deck carries no TF turn or coil count, so both are left unset and
    # recorded as validation gaps rather than written as a guessed zero.
    gap_paths = {gap["path"] for gap in receipt["validation_gaps"]}
    assert "tf/coils_n" in gap_paths
    assert "tf/coil[:]/turns" in gap_paths
    start_r = np.asarray(tf.coil[0].conductor[0].elements.start_points.r)
    assert start_r.size >= 1

    assert receipt["outputs"]["pf_active"] == str(phase_dir / "pf_active.nc")
    assert (phase_dir / "receipt.json").exists()


def test_receipt_names_sources_and_converter_commit(decks, tmp_path: Path):
    eqsle, geo, coil_vv = decks
    receipt = sd.write_phase_description(
        phase="OP2",
        out_dir=tmp_path,
        eqsle=sd.parse_eqsle_deck(eqsle),
        geo=sd.parse_geo_in(geo),
        coil_vv=sd.parse_coil_vv_deck(coil_vv),
    )
    on_disk = json.loads((tmp_path / "OP2" / "receipt.json").read_text())
    assert on_disk["converter"]["git_commit"]
    roles = {s["role"] for s in on_disk["sources"]}
    assert roles == {"eqsle_deck", "geo_in", "coil_vv_deck"}
    for source in on_disk["sources"]:
        assert Path(source["path"]).exists()
        assert len(source["sha256"]) == 64
        assert source["line_ranges"]
    eqsle_entry = next(s for s in on_disk["sources"] if s["role"] == "eqsle_deck")
    assert eqsle_entry["blocks"]["PF COIL"] == [[6, 6], [7, 7], [8, 8], [9, 9]]
    assert set(receipt["outputs"]) == {
        "pf_active",
        "pf_passive",
        "magnetics",
        "wall",
        "tf",
    }


def test_b_field_phi_vacuum_r_relation():
    value = sd.derive_b_field_phi_vacuum_r(15000.0, 100, 2.96)
    expected = (4.0e-7 * math.pi) * 100 * 15000.0 / (2 * math.pi * 2.96)
    assert value == pytest.approx(expected)
    assert value > 0


def test_pf_passive_preserves_two_resistivities(decks, tmp_path: Path):
    eqsle, geo, _ = decks
    receipt = sd.write_phase_description(
        phase="OP1",
        out_dir=tmp_path,
        eqsle=sd.parse_eqsle_deck(eqsle),
        geo=sd.parse_geo_in(geo),
    )
    with imas.DBEntry(
        tmp_path / "OP1" / "pf_passive.nc", "r", dd_version="4.1.1"
    ) as entry:
        passive = entry.get("pf_passive")
    # The two deck resistivities become two loops, each carrying its own value.
    assert len(passive.loop) == 2
    assert [len(loop.element) for loop in passive.loop] == [2, 1]
    assert [float(loop.resistivity) for loop in passive.loop] == pytest.approx(
        [7.76e-7, 7.20e-7]
    )
    assert receipt["phase"] == "OP1"


def test_extent_columns_are_full_extents_of_a_real_element():
    if not REAL_EQSLE.exists():
        pytest.skip(f"real deck copy absent: {REAL_EQSLE}")
    deck = sd.parse_eqsle_deck(REAL_EQSLE)
    cs1 = next(coil for coil in deck.pf_coils if coil.name == "CS1")
    assert len(cs1.elements) == 40
    first = cs1.elements[0]
    assert first.dr == pytest.approx(0.08175)
    assert first.dz == pytest.approx(0.15740)
    assert first.width == pytest.approx(first.dr)
    assert first.height == pytest.approx(first.dz)
    # Tiling with whole extents widens the centre span by one extent, which
    # reproduces the facility CS1 rectangle (0.327 m x 1.574 m).
    rs = [el.r for el in cs1.elements]
    zs = [el.z for el in cs1.elements]
    assert (max(rs) - min(rs)) + first.dr == pytest.approx(0.327, abs=1e-3)
    assert (max(zs) - min(zs)) + first.dz == pytest.approx(1.574, abs=1e-3)

    factory = imas.IDSFactory("4.1.1")
    passive = sd.build_pf_passive(factory, deck.vessel)
    assert [len(loop.element) for loop in passive.loop] == [63, 57]
    assert [float(loop.resistivity) for loop in passive.loop] == pytest.approx(
        [7.76e-7, 7.20e-7]
    )
