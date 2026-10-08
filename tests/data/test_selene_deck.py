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
from imas.ids_metadata import IDSDataType

from imas_alambic.machine_map import bundles_carrying
from imas_ambix.data import selene_deck as sd
from imas_ambix.data.paths import JT60SA_DESCRIPTION_DIR, JT60SA_MAP_DIR
from tests.jt60sa_bundle import requires_jt60sa

# The facility decks the converter reads live in the project description store,
# not in any reckon reports directory.
REAL_DECK_SOURCE = JT60SA_DESCRIPTION_DIR / "source"
REAL_EQSLE = REAL_DECK_SOURCE / "EQSLE.DATA"
REAL_GEO = REAL_DECK_SOURCE / "geo.in"
REAL_COIL_VV = REAL_DECK_SOURCE / "coil_vv_OP2.dat"
REAL_COIL_VV_OP1 = REAL_DECK_SOURCE / "coil_vv_OP1.dat"


@requires_jt60sa
def test_ambix_entry_point_finds_jt60sa_without_a_map_path(monkeypatch):
    """With no ``IMAS_ALAMBIC_MAP_PATH``, ambix's entry point carries jt-60sa."""

    monkeypatch.delenv("IMAS_ALAMBIC_MAP_PATH", raising=False)

    (bundle,) = bundles_carrying("jt-60sa")

    assert bundle.root == JT60SA_MAP_DIR

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
  1.400  0.0  3.000 -60.0  60.0  2   VV OUTER SKIN
  1.400  0.0  3.000  60.0 300.0  2
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
 4  1  nlim,line mimiter option
    1.7050    1.8750
    1.7750    2.0138
    1.7750    2.5000
    1.7050    1.8750
 1
12
     1.6250    0.0000    1.6250    2.3830              1         %%Inner VV=4
     1.6250    2.3830    4.0000    2.3830              1
     4.0000    2.3830    4.0000    0.0000              1
     4.0000    0.0000    1.6250    0.0000              1
     1.4310    0.0000    1.4310    2.3830              1         %%Outer VV=4
     1.4310    2.3830    4.2000    2.3830              1
     4.2000    2.3830    4.2000    0.0000              1
     4.2000    0.0000    1.4310    0.0000              1
     9.0000    9.0000    9.1000    9.0000              1         %%D-probe#1
     9.1000    9.0000    9.1000    9.1000              1
     9.1000    9.1000    9.0000    9.1000              1
     9.0000    9.1000    9.0000    9.0000              1
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
    # The passive block splits into the vessel run and the cryostat run.
    assert len(deck.vessel) == 2
    assert len(deck.cryostat) == 1
    fil = deck.vessel[0]
    assert fil.r == pytest.approx(4.9)
    assert fil.z == pytest.approx(0.3)
    assert fil.dr == pytest.approx(0.04)
    assert fil.dz == pytest.approx(0.25)
    assert fil.resistivity == pytest.approx(7.76e-7)
    assert deck.cryostat[0].resistivity == pytest.approx(7.20e-7)
    assert len(deck.tfc_inside) == 2
    assert len(deck.tfc_outside) == 2
    # The contour table's labels split the first wall from the vessel skin.
    assert {seg.kind for seg in deck.first_wall} == {"line"}
    assert {seg.kind for seg in deck.vessel_skin_outer} == {"arc"}
    assert deck.vessel_skin_inner == []
    assert any("First Wall" in s.comment for s in deck.first_wall)
    line = deck.first_wall[0]
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
    elements += list(deck.cryostat)
    elements += list(deck.tfc_inside) + list(deck.tfc_outside)
    elements += list(deck.first_wall)
    elements += list(deck.vessel_skin_inner) + list(deck.vessel_skin_outer)
    for element in elements:
        prov = element.provenance
        assert prov.source == str(eqsle)
        assert prov.sha256 == digest
        assert 1 <= prov.line_start <= prov.line_end <= n_lines

    assert deck.pf_coils[0].elements[0].provenance.line_range() == [6, 6]
    assert deck.vessel[0].provenance.line_range() == [11, 11]
    assert deck.tfc_inside[0].provenance.line_range() == [19, 19]

    geo_parsed = sd.parse_geo_in(geo)
    assert (
        geo_parsed.probes[0].provenance.sha256
        == hashlib.sha256(geo.read_bytes()).hexdigest()
    )


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
    # The limiter block's vertices become the line segments of a closed
    # contour; the typed contour table after it is not part of the block.
    assert len(parsed.limiter_and_first_wall) == 3
    assert all(seg.kind == "line" for seg in parsed.limiter_and_first_wall)
    (r1, z1), (r2, z2) = parsed.limiter_and_first_wall[0].line_points()
    assert (r1, z1) == pytest.approx((1.7050, 1.8750))
    assert (r2, z2) == pytest.approx((1.7750, 2.0138))
    assert parsed.limiter_and_first_wall[0].provenance.line_range() == [12, 13]
    assert (
        parsed.vessel[0].provenance.sha256
        == hashlib.sha256(coil_vv.read_bytes()).hexdigest()
    )

    # The contour block's Inner/Outer VV runs are the two vessel skins; its
    # D-probe run is a diagnostic probe and is excluded from both.
    inner_r, inner_z = sd._wall_outline(parsed.vessel_skin_inner)
    outer_r, outer_z = sd._wall_outline(parsed.vessel_skin_outer)
    assert len(parsed.vessel_skin_inner) == 4
    assert len(parsed.vessel_skin_outer) == 4
    assert (inner_r[0], inner_z[0]) == (inner_r[-1], inner_z[-1])
    assert (outer_r[0], outer_z[0]) == (outer_r[-1], outer_z[-1])
    assert max(outer_r) > max(inner_r)
    assert not np.isclose(inner_r, 9.0).any()
    assert not np.isclose(inner_z, 9.0).any()
    assert not np.isclose(outer_r, 9.0).any()
    assert not np.isclose(outer_z, 9.0).any()


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
    assert float(passive.loop[0].element[0].geometry.rectangle.height) == pytest.approx(
        0.24969
    )
    assert float(passive.loop[0].resistivity) == pytest.approx(7.20e-7)

    mag = read("magnetics")
    assert float(mag.b_field_pol_probe[0].position.r) == pytest.approx(4.7355)
    # The deck's 0.28 deg is the outward wall-normal omega; the stored axis is
    # the tangential sensing axis, (90 - omega) mod 360.
    assert float(mag.b_field_pol_probe[0].poloidal_angle) == pytest.approx(
        math.radians(89.72)
    )
    assert float(mag.flux_loop[0].position[0].r) == pytest.approx(1.8798)

    wall = read("wall")
    # A coil_vv deck carries a limiter and both vessel skins, so the wall is
    # that deck's limiter plus one annular vessel unit.
    assert int(wall.description_2d[0].type.index) == 2
    outline_r = np.asarray(wall.description_2d[0].limiter.unit[0].outline.r)
    # The deck's limiter block traces a closed contour of four outline points.
    assert outline_r.size == 4
    assert outline_r[0] == pytest.approx(1.7050)
    assert outline_r[-1] == pytest.approx(1.7050)
    annular = wall.description_2d[0].vessel.unit[0].annular
    inner_r = np.asarray(annular.outline_inner.r)
    outer_r = np.asarray(annular.outline_outer.r)
    assert inner_r.size == 5
    assert outer_r.size == 5
    assert float(outer_r.max()) > float(inner_r.max())

    tf = read("tf")
    # The deck carries no TF turn or coil count, so both are left unset and
    # recorded as validation gaps rather than written as a guessed zero.
    gap_paths = {gap["path"] for gap in receipt["validation_gaps"]}
    assert "tf/coils_n" in gap_paths
    assert "tf/coil[:]/turns" in gap_paths
    conductor = tf.coil[0].conductor[0].elements
    start_r = np.asarray(conductor.start_points.r)
    start_z = np.asarray(conductor.start_points.z)
    end_r = np.asarray(conductor.end_points.r)
    end_z = np.asarray(conductor.end_points.z)
    # The synthetic TFC INSIDE section is two arcs; each is one element.
    assert start_r.size == 2
    length_sq = (start_r - end_r) ** 2 + (start_z - end_z) ** 2
    assert np.min(length_sq) > 0
    assert list(np.asarray(conductor.types)) == [2, 2]

    assert receipt["outputs"]["pf_active"] == str(phase_dir / "pf_active.nc")
    assert (phase_dir / "receipt.json").exists()


def _axis_difference_deg(first_deg: float, second_deg: float) -> float:
    """Smallest angle between two axis directions, folded onto 0..90 degrees."""
    return abs((first_deg - second_deg + 90.0) % 180.0 - 90.0)


def _skin_polyline(segments) -> tuple[np.ndarray, np.ndarray]:
    """The sampled (r, z) vertices of a vessel-skin segment run."""
    skin_r, skin_z = sd._wall_outline(segments)
    return np.asarray(skin_r), np.asarray(skin_z)


def _nearest_skin_tangent_deg(
    skin_r: np.ndarray, skin_z: np.ndarray, r: float, z: float
) -> float:
    """Sensing-axis angle of the skin chord nearest (r, z), clockwise from +R."""
    best: tuple[float, float] | None = None
    for i in range(skin_r.size - 1):
        r1, z1 = float(skin_r[i]), float(skin_z[i])
        dr, dz = float(skin_r[i + 1]) - r1, float(skin_z[i + 1]) - z1
        length_sq = dr * dr + dz * dz
        if length_sq == 0.0:
            continue
        t = min(1.0, max(0.0, ((r - r1) * dr + (z - z1) * dz) / length_sq))
        cr, cz = r1 + t * dr, z1 + t * dz
        dist_sq = (r - cr) ** 2 + (z - cz) ** 2
        if best is None or dist_sq < best[0]:
            # The DD leaf is a clockwise-from-+R angle: the negation of the
            # counter-clockwise atan2 of the chord direction.
            best = (dist_sq, (-math.degrees(math.atan2(dz, dr))) % 360.0)
    assert best is not None
    return best[1]


def test_stored_probe_axes_lie_along_the_vessel_inner_skin():
    """Every stored probe axis is the vessel inner-skin tangent.

    The facility probe positions sit on the description's vessel inner skin, so
    the converter must store each probe's sensing axis along that skin's tangent
    rather than the deck's outward wall-normal angle.  Both phases are checked,
    each against the deck its store draws the vessel skin from.
    """
    if not (
        REAL_DECK_SOURCE.is_dir()
        and REAL_GEO.exists()
        and REAL_COIL_VV.exists()
        and REAL_COIL_VV_OP1.exists()
    ):
        pytest.skip(f"project deck store absent: {REAL_DECK_SOURCE}")
    geo = sd.parse_geo_in(REAL_GEO)
    factory = imas.IDSFactory(sd.DD_VERSION)
    magnetics = sd.build_magnetics(factory, geo)
    stored_deg = [
        math.degrees(float(probe.poloidal_angle))
        for probe in magnetics.b_field_pol_probe
    ]
    # Each phase's store draws its vessel skin from its own coil_vv deck.
    skins = {
        "OP1": sd.parse_coil_vv_deck(REAL_COIL_VV_OP1).vessel_skin_inner,
        "OP2": sd.parse_coil_vv_deck(REAL_COIL_VV).vessel_skin_inner,
    }
    for phase, skin in skins.items():
        skin_r, skin_z = _skin_polyline(skin)
        assert skin_r.size >= 2, phase
        for index, probe in enumerate(geo.probes):
            tangent = _nearest_skin_tangent_deg(skin_r, skin_z, probe.r, probe.z)
            difference = _axis_difference_deg(stored_deg[index], tangent)
            assert difference <= 5.0, (phase, index + 1, stored_deg[index], tangent)

    # MP1 sits at the outboard midplane, where the tangential axis points down.
    assert _axis_difference_deg(stored_deg[0], 90.0) <= 1.0


def test_differential_flux_loops_follow_the_physical_loops():
    """build_magnetics appends one differential entry per non-reference loop.

    Each of the 27 geo.in flux loops keeps a type-1 entry carrying its
    position, and after the 27 come 26 type-6 (differential) entries, one for
    every loop L other than the reference loop 7.  Each type-6 entry's
    ``indices_differential`` names the reference then L, which the DD
    documents as ``loop(second index) - loop(first index)`` over the 1-based
    ``flux_loop`` array of structures, and carries no position or area of its
    own.
    """
    if not (REAL_DECK_SOURCE.is_dir() and REAL_GEO.exists()):
        pytest.skip(f"project deck store absent: {REAL_DECK_SOURCE}")
    geo = sd.parse_geo_in(REAL_GEO)
    assert len(geo.flux_loops) == 27
    magnetics = sd.build_magnetics(imas.IDSFactory(sd.DD_VERSION), geo)

    loops = list(magnetics.flux_loop)
    assert len(loops) == 53
    physical, differential = loops[:27], loops[27:]
    assert [int(f.type.index) for f in physical] == [1] * 27
    assert [int(f.type.index) for f in differential] == [6] * 26
    for i, f in enumerate(physical):
        assert float(f.position[0].r) == pytest.approx(geo.flux_loops[i].r)
        assert float(f.position[0].z) == pytest.approx(geo.flux_loops[i].z)
    expected = [[7, L] for L in range(1, 28) if L != 7]
    stored = [
        [int(v) for v in np.asarray(f.indices_differential)] for f in differential
    ]
    assert stored == expected
    for f in differential:
        assert len(f.position) == 0
        assert not f.area.has_value


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


def test_pf_passive_names_the_vessel_and_cryostat_loops(decks, tmp_path: Path):
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
    # The vessel run and the cryostat run become two named loops, each carrying
    # its own resistivity, and their filaments are named after their loop.
    assert [str(loop.name) for loop in passive.loop] == ["VV", "CRYOSTAT"]
    assert [len(loop.element) for loop in passive.loop] == [2, 1]
    assert [float(loop.resistivity) for loop in passive.loop] == pytest.approx(
        [7.76e-7, 7.20e-7]
    )
    assert [str(el.name) for el in passive.loop[0].element] == ["VV_1", "VV_2"]
    assert [str(el.name) for el in passive.loop[1].element] == ["CRYOSTAT_1"]
    assert receipt["phase"] == "OP1"


def _fil(r: float, z: float, resistivity: float) -> sd.VesselFilament:
    return sd.VesselFilament(
        turns=1.0,
        r=r,
        z=z,
        dr=0.1,
        dz=0.1,
        resistivity=resistivity,
        provenance=sd.Provenance(
            source="synthetic", sha256="0" * 64, line_start=1, line_end=1
        ),
    )


def _line(r1: float, z1: float, r2: float, z2: float) -> sd.Segment:
    return sd.Segment(
        kind="line",
        params=(r1, z1, r2, z2),
        comment="",
        provenance=sd.Provenance(
            source="synthetic", sha256="0" * 64, line_start=1, line_end=1
        ),
    )


# A square vessel outer skin in the poloidal plane spanning R 1..3, Z -1..1.
_SQUARE_SKIN = [
    _line(1.0, -1.0, 3.0, -1.0),
    _line(3.0, -1.0, 3.0, 1.0),
    _line(3.0, 1.0, 1.0, 1.0),
    _line(1.0, 1.0, 1.0, -1.0),
]


def test_cryostat_inside_the_vessel_outer_skin_is_refused():
    factory = imas.IDSFactory("4.1.1")
    vessel = [_fil(1.0, -0.5, 7.76e-7), _fil(3.0, 0.5, 7.76e-7)]

    # A filament inside the vessel's outer-skin polyline is vessel material,
    # not a cryostat, so labelling the group CRYOSTAT must be refused.
    inside = [_fil(2.0, 0.0, 7.20e-7)]
    with pytest.raises(sd.CryostatEnvelopeError):
        sd.build_pf_passive(factory, vessel, inside, vessel_outer_skin=_SQUARE_SKIN)

    # A filament inside the outer skin but outside the vessel filaments'
    # rectangular R-Z envelope is still refused: the containment test bounds
    # the vessel shape, not a box around its centres.
    corner = [_fil(2.5, -0.75, 7.20e-7)]
    with pytest.raises(sd.CryostatEnvelopeError):
        sd.build_pf_passive(factory, vessel, corner, vessel_outer_skin=_SQUARE_SKIN)

    # A group outside the outer-skin polyline is accepted and becomes a loop.
    outside = [_fil(5.0, 0.0, 7.20e-7)]
    passive = sd.build_pf_passive(
        factory, vessel, outside, vessel_outer_skin=_SQUARE_SKIN
    )
    assert [str(loop.name) for loop in passive.loop] == ["VV", "CRYOSTAT"]


def test_cryostat_loop_carries_57_filaments_in_both_phases(tmp_path: Path):
    """The cryostat is the same 57-filament loop in OP1 and OP2.

    OP2's vessel comes from its own coil_vv deck (98 filaments) while its
    cryostat is read from the shared EQSLE deck, so the CRYOSTAT loop must be
    identical across the phase boundary even though the VV loop is not.
    """
    if not (REAL_EQSLE.exists() and REAL_GEO.exists() and REAL_COIL_VV.exists()):
        pytest.skip("real deck copies absent")
    eqsle = sd.parse_eqsle_deck(REAL_EQSLE)
    geo = sd.parse_geo_in(REAL_GEO)
    coil_vv = sd.parse_coil_vv_deck(REAL_COIL_VV)

    def cryostat(phase, **kwargs):
        sd.write_phase_description(
            phase=phase, out_dir=tmp_path, eqsle=eqsle, geo=geo, **kwargs
        )
        with imas.DBEntry(
            tmp_path / phase / "pf_passive.nc", "r", dd_version="4.1.1"
        ) as entry:
            passive = entry.get("pf_passive")
        loops = {str(loop.name): loop for loop in passive.loop}
        return loops["CRYOSTAT"], loops["VV"]

    op1_cryostat, op1_vessel = cryostat("OP1")
    op2_cryostat, op2_vessel = cryostat("OP2", coil_vv=coil_vv)

    assert str(op1_cryostat.name) == str(op2_cryostat.name) == "CRYOSTAT"
    assert len(op1_cryostat.element) == len(op2_cryostat.element) == 57
    assert float(op1_cryostat.resistivity) == pytest.approx(7.20e-7)
    assert float(op2_cryostat.resistivity) == pytest.approx(7.20e-7)
    # The two cryostat loops carry the same filament geometry.
    for a, b in zip(op1_cryostat.element, op2_cryostat.element, strict=True):
        assert float(a.geometry.rectangle.r) == pytest.approx(
            float(b.geometry.rectangle.r)
        )
        assert float(a.geometry.rectangle.z) == pytest.approx(
            float(b.geometry.rectangle.z)
        )
    # The vessel differs across the boundary: 63 EQSLE filaments, 98 coil_vv.
    assert len(op1_vessel.element) == 63
    assert len(op2_vessel.element) == 98


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
    passive = sd.build_pf_passive(
        factory,
        deck.vessel,
        deck.cryostat,
        vessel_outer_skin=deck.vessel_skin_outer,
    )
    assert [str(loop.name) for loop in passive.loop] == ["VV", "CRYOSTAT"]
    assert [len(loop.element) for loop in passive.loop] == [63, 57]
    assert [float(loop.resistivity) for loop in passive.loop] == pytest.approx(
        [7.76e-7, 7.20e-7]
    )


def test_real_tfc_sections_and_wall_arcs_round_trip(tmp_path: Path):
    if not (REAL_EQSLE.exists() and REAL_GEO.exists()):
        pytest.skip("real deck copies absent")
    deck = sd.parse_eqsle_deck(REAL_EQSLE)
    geo = sd.parse_geo_in(REAL_GEO)
    assert len(deck.tfc_inside) == 7
    assert len(deck.tfc_outside) == 7

    receipt = sd.write_phase_description(
        phase="RT", out_dir=tmp_path, eqsle=deck, geo=geo
    )
    assert {gap["path"] for gap in receipt["validation_gaps"]} >= {
        "tf/coils_n",
        "tf/coil[:]/turns",
    }

    with imas.DBEntry(tmp_path / "RT" / "wall.nc", "r", dd_version="4.1.1") as e:
        wall = e.get("wall")
    with imas.DBEntry(tmp_path / "RT" / "tf.nc", "r", dd_version="4.1.1") as e:
        tf = e.get("tf")

    # Every TFC section round-trips as one element per deck segment, and no
    # element is written with equal start and end points.
    for c in range(2):
        els = tf.coil[0].conductor[c].elements
        start_r = np.asarray(els.start_points.r)
        end_r = np.asarray(els.end_points.r)
        start_z = np.asarray(els.start_points.z)
        end_z = np.asarray(els.end_points.z)
        assert start_r.size == 7
        assert np.min((start_r - end_r) ** 2 + (start_z - end_z) ** 2) > 0
    types = list(np.asarray(tf.coil[0].conductor[0].elements.types))
    assert types == [2, 2, 2, 1, 2, 2, 2]

    # The limiter outline is the first wall alone: one closed contour whose
    # first point repeats as its last.  Every first-wall arc is represented,
    # with a point within 1e-6 m of its circle.
    limiter_r = np.asarray(wall.description_2d[0].limiter.unit[0].outline.r)
    limiter_z = np.asarray(wall.description_2d[0].limiter.unit[0].outline.z)
    assert int(wall.description_2d[0].type.index) == 2
    assert limiter_r[0] == pytest.approx(limiter_r[-1])
    assert limiter_z[0] == pytest.approx(limiter_z[-1])
    first_wall_arcs = [seg for seg in deck.first_wall if seg.kind == "arc"]
    assert len(first_wall_arcs) == 2
    for arc in first_wall_arcs:
        rc, zc, radius = arc.params[:3]
        distance = np.abs(np.hypot(limiter_r - rc, limiter_z - zc) - radius)
        assert float(np.min(distance)) < 1e-6

    # The vessel unit's two skins become its annular inner and outer outlines,
    # each carrying its own arcs.
    annular = wall.description_2d[0].vessel.unit[0].annular
    for skin, key in (
        (deck.vessel_skin_inner, "outline_inner"),
        (deck.vessel_skin_outer, "outline_outer"),
    ):
        outline = getattr(annular, key)
        skin_r = np.asarray(outline.r)
        skin_z = np.asarray(outline.z)
        # Each skin is six deck segments: five arcs and the one straight run
        # that closes it.
        assert len(skin) == 6
        assert len([s for s in skin if s.kind == "arc"]) == 5
        for arc in (s for s in skin if s.kind == "arc"):
            rc, zc, radius = arc.params[:3]
            distance = np.abs(np.hypot(skin_r - rc, skin_z - zc) - radius)
            assert float(np.min(distance)) < 1e-6
    # The outer skin sits outside the inner skin at the outboard midplane.
    assert float(np.max(annular.outline_outer.r)) > float(
        np.max(annular.outline_inner.r)
    )


def _d_probe_vertices(deck_path: Path) -> set[tuple[float, float]]:
    """Every ``R Z`` vertex of the deck's D-probe rows, from the raw file."""
    lines = deck_path.read_text(encoding="latin-1").splitlines()
    vertices: set[tuple[float, float]] = set()
    for line in lines:
        if "D-probe" not in line:
            continue
        parts = line.split()
        nums = []
        for tok in parts:
            try:
                nums.append(float(tok))
            except ValueError:
                break
        if len(nums) >= 4:
            vertices.add((nums[0], nums[1]))
            vertices.add((nums[2], nums[3]))
    return vertices


def test_both_phases_take_their_wall_from_their_own_coil_vv_deck(tmp_path: Path):
    """OP1 and OP2 both draw limiter and vessel from their coil_vv tables."""
    if not (
        REAL_EQSLE.exists()
        and REAL_GEO.exists()
        and REAL_COIL_VV.exists()
        and REAL_COIL_VV_OP1.exists()
    ):
        pytest.skip("real deck copies absent")
    eqsle = sd.parse_eqsle_deck(REAL_EQSLE)
    geo = sd.parse_geo_in(REAL_GEO)
    decks = {
        "OP1": sd.parse_coil_vv_deck(REAL_COIL_VV_OP1),
        "OP2": sd.parse_coil_vv_deck(REAL_COIL_VV),
    }

    # Each deck's nlim polygon is 51 points closing on the first, and its
    # inner/outer skins are 50-segment polylines with the D-probe rows excluded.
    for phase, deck in decks.items():
        assert len(deck.limiter_and_first_wall) == 50, phase
        assert all(seg.kind == "line" for seg in deck.limiter_and_first_wall)
        assert len(deck.vessel_skin_inner) == 50, phase
        assert len(deck.vessel_skin_outer) == 50, phase
        for skin in (deck.vessel_skin_inner, deck.vessel_skin_outer):
            assert not any("D-PROBE" in s.comment.upper() for s in skin)
        assert deck.sha256 == hashlib.sha256(Path(deck.path).read_bytes()).hexdigest()

    def outlines(phase: str, coil_vv: sd.CoilVesselDeck) -> dict:
        out = tmp_path / phase
        sd.write_phase_description(
            phase=phase,
            out_dir=tmp_path,
            eqsle=eqsle,
            geo=geo,
            coil_vv=coil_vv,
        )
        with imas.DBEntry(out / "wall.nc", "r", dd_version="4.1.1") as e:
            wall = e.get("wall")
        d = wall.description_2d[0]
        annular = d.vessel.unit[0].annular

        def pair(node):
            return np.asarray(node.r), np.asarray(node.z)

        lim_r, lim_z = pair(d.limiter.unit[0].outline)
        in_r, in_z = pair(annular.outline_inner)
        out_r, out_z = pair(annular.outline_outer)
        return {
            "type": int(d.type.index),
            "lim_r": lim_r,
            "lim_z": lim_z,
            "in_r": in_r,
            "in_z": in_z,
            "out_r": out_r,
            "out_z": out_z,
        }

    op1 = outlines("OP1", decks["OP1"])
    op2 = outlines("OP2", decks["OP2"])

    for phase, o in (("OP1", op1), ("OP2", op2)):
        assert o["type"] == 2, phase
        # A limiter of 51 points that is a closed contour.
        assert o["lim_r"].size == 51, phase
        assert o["lim_r"][0] == pytest.approx(o["lim_r"][-1]), phase
        assert o["lim_z"][0] == pytest.approx(o["lim_z"][-1]), phase
        # Inner and outer outlines of 51 points each.
        assert o["in_r"].size == 51, phase
        assert o["out_r"].size == 51, phase
        # No D-probe point appears in either outline.
        probes = _d_probe_vertices(Path(decks[phase].path))
        assert probes, phase
        skin_points = set(
            zip(o["in_r"].tolist(), o["in_z"].tolist(), strict=True)
        ) | set(zip(o["out_r"].tolist(), o["out_z"].tolist(), strict=True))
        assert skin_points.isdisjoint(probes), phase

    # The first wall changed for OP2, so the two limiters differ.
    assert op1["lim_r"].size == op2["lim_r"].size == 51
    assert not np.array_equal(op1["lim_r"], op2["lim_r"]) or not np.array_equal(
        op1["lim_z"], op2["lim_z"]
    )


# The five IDSs the converter writes.  Each carries one or more first-level
# struct arrays whose DD element has a ``name`` leaf, so every populated entry
# must be given a unique non-empty name.
STRUCT_ARRAY_IDS = ("pf_active", "pf_passive", "magnetics", "wall", "tf")


def _element_has_name(meta) -> bool:
    """True when a struct-array element type carries a DD ``name`` leaf."""
    try:
        meta["name"]
    except KeyError:
        return False
    return True


@pytest.fixture()
def phase_ids(decks, tmp_path: Path):
    """Every written IDS for one converted phase, read back at DD 4.1.1."""
    eqsle, geo, coil_vv = decks
    sd.write_phase_description(
        phase="OP1",
        out_dir=tmp_path,
        eqsle=sd.parse_eqsle_deck(eqsle),
        geo=sd.parse_geo_in(geo),
        coil_vv=sd.parse_coil_vv_deck(coil_vv),
    )
    out = {}
    for name in STRUCT_ARRAY_IDS:
        with imas.DBEntry(
            tmp_path / "OP1" / f"{name}.nc", "r", dd_version="4.1.1"
        ) as entry:
            out[name] = entry.get(name)
    return out


@pytest.mark.parametrize("ids_name", STRUCT_ARRAY_IDS)
def test_every_written_struct_array_entry_is_named(phase_ids, ids_name):
    """No populated first-level struct array the converter writes is unnamed.

    The sweep reads the DD metadata rather than a fixed list, so a struct array
    added to the writer later is covered without being enumerated here.  An
    array whose element type carries no ``name`` leaf (the wall's
    ``description_2d``) has no name to assign and is skipped.
    """
    ids = phase_ids[ids_name]
    for meta in ids.metadata:
        if meta.data_type is not IDSDataType.STRUCT_ARRAY:
            continue
        if not _element_has_name(meta):
            continue
        array = getattr(ids, meta.name)
        names = [str(element.name) for element in array]
        for i, name in enumerate(names):
            assert name, f"{ids_name}/{meta.name}[{i}] carries an empty name"
        assert len(set(names)) == len(names), (
            f"duplicate names in {ids_name}/{meta.name}: {names}"
        )


def test_expected_names_are_assigned_in_deck_order(decks, phase_ids):
    # The EQSLE passive block holds two resistivity groups, so its loops are
    # named for what each group is: the vessel and the cryostat.
    eqsle, _, _ = decks
    deck = sd.parse_eqsle_deck(eqsle)
    passive = sd.build_pf_passive(
        imas.IDSFactory("4.1.1"),
        deck.vessel,
        deck.cryostat,
        vessel_outer_skin=deck.vessel_skin_outer,
    )
    assert [str(loop.name) for loop in passive.loop] == ["VV", "CRYOSTAT"]
    assert [str(p.name) for p in phase_ids["magnetics"].b_field_pol_probe] == [
        "MP1",
        "MP2",
    ]
    assert [str(f.name) for f in phase_ids["magnetics"].flux_loop] == ["FL1", "FL2"]
    assert str(phase_ids["tf"].coil[0].name) == "TF1"
    # Each pf_active circuit is named after the coil it drives, and the coil
    # names come from the deck's own labels.
    assert [str(c.name) for c in phase_ids["pf_active"].coil] == ["CS1", "CS2"]
    assert [str(c.name) for c in phase_ids["pf_active"].circuit] == ["CS1", "CS2"]


def test_every_pf_active_coil_element_is_named_by_coil_and_position(phase_ids):
    """Every conductor element names its coil and 2-deck-order position.

    The write scope is the coil element, one struct-array level below the
    struct array the every-entry-is-named sweep reaches, so a coil whose
    elements are unnamed (or whose names repeat within the coil) is caught here.
    """
    pf = phase_ids["pf_active"]
    for coil in pf.coil:
        coil_name = str(coil.name)
        names = [str(element.name) for element in coil.element]
        assert all(names), f"{coil_name} carries an unnamed element: {names}"
        expected = [f"{coil_name}_{k}" for k in range(1, len(names) + 1)]
        assert names == expected
        assert len(set(names)) == len(names), (
            f"{coil_name} repeats an element name: {names}"
        )
    # Names are unique within a coil and the coil prefix keeps them unique
    # across the phase.
    all_names = [str(element.name) for coil in pf.coil for element in coil.element]
    assert len(set(all_names)) == len(all_names)


# --------------------------------------------------------------------------
# coil_vv contour-block hardening: the parse refuses decks it would mis-slice
# --------------------------------------------------------------------------
_FACILITY_SOURCE = JT60SA_DESCRIPTION_DIR / "source"
_FACILITY_EQSLE = _FACILITY_SOURCE / "EQSLE.DATA"
_FACILITY_COIL_VV_OP1 = _FACILITY_SOURCE / "coil_vv_OP1.dat"
_FACILITY_COIL_VV_OP2 = _FACILITY_SOURCE / "coil_vv_OP2.dat"

# The coil_vv deck up to and including the line before the contour count, taken
# from COIL_VV_TEXT so the refusal fixtures share the parser-valid preamble.
_COIL_VV_PREFIX = "\n".join(COIL_VV_TEXT.splitlines()[:16]) + "\n"


def _chord(r1, z1, r2, z2, label=None) -> str:
    """One contour chord row ``R1 Z1 R2 Z2 index`` with an optional run label."""
    row = f"{r1:9.4f} {z1:9.4f} {r2:9.4f} {z2:9.4f}              1"
    if label is not None:
        row += f"         %%{label}"
    return row


def _coil_vv_deck_text(rows: list[str]) -> str:
    """A coil_vv deck whose contour block is ``rows`` behind its own count."""
    return _COIL_VV_PREFIX + f"{len(rows)}\n" + "\n".join(rows) + "\n"


def _write_coil_vv(tmp_path: Path, rows: list[str]) -> Path:
    path = tmp_path / "coil_vv_OP2.dat"
    path.write_text(_coil_vv_deck_text(rows), encoding="latin-1")
    return path


def _closed_skin_rows(count: int, label: str) -> list[str]:
    """``count`` chord rows tracing a closed ``count``-gon, first row labelled."""
    pts = [
        (
            2.0 + 0.5 * math.cos(2 * math.pi * k / count),
            0.5 * math.sin(2 * math.pi * k / count),
        )
        for k in range(count)
    ]
    rows = []
    for k in range(count):
        r1, z1 = pts[k]
        r2, z2 = pts[(k + 1) % count]
        rows.append(_chord(r1, z1, r2, z2, label if k == 0 else None))
    return rows


def test_contour_header_skips_a_count_without_a_labelled_row():
    """A bare count whose next numeric row carries no run label is not the block."""
    prefix = _COIL_VV_PREFIX.splitlines()
    start = len(prefix) - 1  # the first line the header search scans
    lines = prefix + [
        "7",
        _chord(1.0, 2.0, 3.0, 4.0),  # five numbers, no run label
        "2",
        _chord(1.0, 0.0, 2.0, 0.0, "Inner VV=2"),
        _chord(2.0, 0.0, 1.0, 0.0),
    ]
    header, count = sd._find_contour_header(lines, start)
    assert count == 2
    assert lines[header] == "2"


def test_contour_header_absent_is_refused():
    prefix = _COIL_VV_PREFIX.splitlines()
    start = len(prefix) - 1
    lines = prefix + [
        "3",
        _chord(1.0, 0.0, 2.0, 0.0),
        _chord(2.0, 0.0, 1.0, 0.0),
        _chord(1.0, 0.0, 1.0, 0.0),
    ]
    with pytest.raises(ValueError, match="no coil_vv contour block header found"):
        sd._find_contour_header(lines, start)


def test_contour_row_before_the_first_label_is_refused():
    lines = _coil_vv_deck_text(
        [
            _chord(1.0, 0.0, 2.0, 0.0),
            _chord(2.0, 0.0, 1.0, 0.0, "Inner VV=2"),
            _chord(3.0, 0.0, 4.0, 0.0, "Outer VV=1"),
            _chord(4.0, 0.0, 3.0, 0.0),
        ]
    ).splitlines()
    header = len(_COIL_VV_PREFIX.splitlines())
    with pytest.raises(ValueError, match="counted row precedes the first run label"):
        sd._parse_contour_block(lines, header, "synthetic", "0" * 64)


def test_unknown_contour_run_label_is_refused(tmp_path: Path):
    rows = [
        _chord(1.0, 0.0, 2.0, 0.0, "Inner VV=3"),
        _chord(2.0, 0.0, 1.0, 0.0),
        _chord(2.0, 0.0, 1.0, 0.0, "SomethingElse"),
        _chord(3.0, 0.0, 4.0, 0.0, "Outer VV=1"),
        _chord(4.0, 0.0, 3.0, 0.0),
    ]
    with pytest.raises(ValueError, match="unrecognised run label"):
        sd.parse_coil_vv_deck(_write_coil_vv(tmp_path, rows))


def test_contour_run_shorter_than_its_label_is_refused(tmp_path: Path):
    rows = _closed_skin_rows(49, "Inner VV=50") + _closed_skin_rows(4, "Outer VV=4")
    with pytest.raises(ValueError, match="INNER VV run declares 50 rows, found 49"):
        sd.parse_coil_vv_deck(_write_coil_vv(tmp_path, rows))


def test_contour_block_missing_a_skin_is_refused(tmp_path: Path):
    rows = [
        _chord(1.0, 0.0, 2.0, 0.0, "Inner VV=2"),
        _chord(2.0, 0.0, 1.0, 0.0),
    ]
    with pytest.raises(ValueError, match="no Outer VV skin"):
        sd.parse_coil_vv_deck(_write_coil_vv(tmp_path, rows))


def test_contour_chords_that_do_not_join_are_refused(tmp_path: Path):
    rows = [
        _chord(1.0, 0.0, 2.0, 0.0, "Inner VV=2"),
        _chord(3.0, 0.0, 4.0, 0.0),  # start does not meet the first chord's end
        _chord(5.0, 0.0, 6.0, 0.0, "Outer VV=2"),
        _chord(6.0, 0.0, 5.0, 0.0),
    ]
    with pytest.raises(ValueError, match="does not join the previous chord's end"):
        sd.parse_coil_vv_deck(_write_coil_vv(tmp_path, rows))


def test_contour_skin_that_does_not_close_is_refused(tmp_path: Path):
    rows = [
        _chord(1.0, 0.0, 2.0, 0.0, "Inner VV=2"),
        _chord(2.0, 0.0, 3.0, 0.0),  # last chord ends away from the first start
        _chord(5.0, 0.0, 6.0, 0.0, "Outer VV=2"),
        _chord(6.0, 0.0, 5.0, 0.0),
    ]
    with pytest.raises(ValueError, match="does not close"):
        sd.parse_coil_vv_deck(_write_coil_vv(tmp_path, rows))


def _point_segment_distance(px: float, pz: float, seg: sd.Segment) -> float:
    """Distance from a point to a contour segment, an arc measured as an arc."""
    if seg.kind == "line":
        (r1, z1), (r2, z2) = seg.line_points()
        vr, vz = r2 - r1, z2 - z1
        length_sq = vr * vr + vz * vz
        if length_sq == 0.0:
            return math.hypot(px - r1, pz - z1)
        t = min(1.0, max(0.0, ((px - r1) * vr + (pz - z1) * vz) / length_sq))
        return math.hypot(px - (r1 + t * vr), pz - (z1 + t * vz))
    rc, zc, radius, a_start, a_end = seg.params[:5]
    lo, hi = min(a_start, a_end), max(a_start, a_end)
    theta = math.degrees(math.atan2(pz - zc, px - rc))
    while theta < lo:
        theta += 360.0
    while theta > hi + 360.0:
        theta -= 360.0
    if lo <= theta <= hi:
        return abs(math.hypot(px - rc, pz - zc) - radius)
    ends = [
        (
            rc + radius * math.cos(math.radians(a_start)),
            zc + radius * math.sin(math.radians(a_start)),
        ),
        (
            rc + radius * math.cos(math.radians(a_end)),
            zc + radius * math.sin(math.radians(a_end)),
        ),
    ]
    return min(math.hypot(px - er, pz - ez) for er, ez in ends)


def test_coil_vv_skin_vertices_lie_on_the_eqsle_skins():
    """Each coil_vv skin vertex sits on its EQSLE arc/line skin to 1e-3 m.

    The two skins describe the same vessel in two representations, so every
    vertex of a coil_vv polyline must lie on the matching EQSLE.DATA skin
    (inner to ``vessel_skin_inner``, outer to ``vessel_skin_outer``).  The
    distance is measured to the nearest segment, with EQSLE arcs measured as
    arcs rather than as their chord endpoints.
    """
    needed = [_FACILITY_EQSLE, _FACILITY_COIL_VV_OP1, _FACILITY_COIL_VV_OP2]
    if not all(p.exists() for p in needed):
        pytest.skip("facility source decks absent")
    eqsle = sd.parse_eqsle_deck(_FACILITY_EQSLE)
    for deck_path in (_FACILITY_COIL_VV_OP1, _FACILITY_COIL_VV_OP2):
        deck = sd.parse_coil_vv_deck(deck_path)
        for skin, eq_skin, name in (
            (deck.vessel_skin_inner, eqsle.vessel_skin_inner, "inner"),
            (deck.vessel_skin_outer, eqsle.vessel_skin_outer, "outer"),
        ):
            for chord in skin:
                (r1, z1), (r2, z2) = chord.line_points()
                for px, pz in ((r1, z1), (r2, z2)):
                    distance = min(
                        _point_segment_distance(px, pz, seg) for seg in eq_skin
                    )
                    assert distance <= 1e-3, (
                        f"{deck_path.name} {name} vertex ({px}, {pz}) lies "
                        f"{distance:.3e} m from the EQSLE skin"
                    )
