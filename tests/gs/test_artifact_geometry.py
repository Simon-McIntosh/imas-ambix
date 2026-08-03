"""Tests for the artifact-backed machine-geometry reader.

Two layers, matching ``tests/gs/test_imas_geometry.py``:

* **Pure logic** builds a tiny synthetic ``pf_active`` / ``pf_passive`` /
  ``wall`` / ``magnetics`` set with imas-python and drives the reader's parsing
  functions directly, so every branch runs anywhere imas-python is installed.
  These pin the behaviour that must not drift: an unsourced winding becomes NaN
  and a named failure, never a number; an outline keeps its true corners beside
  the box that stands in for it; a probe with no orientation is dropped rather
  than pointed at an assumed axis.
* **Integration** resolves a real published artifact and cross-checks its
  geometry against the ``efm`` reader for the same machine.  It is skipped
  unless the cache is named in the environment, because the artifact is
  content-addressed in a local cache rather than committed.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from imas_ambix.gs import artifact_geometry as ag
from imas_ambix.gs.geometry import GeometryTable, SetupSignature, build_table_for_shot

imas = pytest.importorskip("imas")


# --- synthetic IDS fixtures --------------------------------------------------


def _square(r: float, z: float, half: float) -> tuple[list[float], list[float]]:
    return (
        [r - half, r + half, r + half, r - half],
        [z - half, z - half, z + half, z + half],
    )


@pytest.fixture
def synthetic_pf_active():
    """Two coils: one with an unsourced winding, one with a filled one."""
    pf = imas.IDSFactory().pf_active()
    pf.ids_properties.homogeneous_time = 2
    pf.coil.resize(2)

    unsourced = pf.coil[0]
    unsourced.name = "unsourced_winding"
    unsourced.element.resize(1)
    outline = unsourced.element[0].geometry.outline
    r, z = _square(1.5, 0.4, 0.05)
    outline.r, outline.z = r, z
    # turns_with_sign deliberately left at the IMAS EMPTY sentinel

    sourced = pf.coil[1]
    sourced.name = "sourced_winding"
    sourced.element.resize(2)
    for index, centre in enumerate((0.8, 1.0)):
        loop_outline = sourced.element[index].geometry.outline
        # a repeated closing vertex is legal DD and must not survive as a corner
        r, z = _square(2.0, centre, 0.1)
        loop_outline.r, loop_outline.z = [*r, r[0]], [*z, z[0]]
        sourced.element[index].turns_with_sign = -12.0
    return pf


@pytest.fixture
def synthetic_pf_passive():
    """One loop of two elements, plus one loop whose element has no shape."""
    pp = imas.IDSFactory().pf_passive()
    pp.ids_properties.homogeneous_time = 2
    pp.loop.resize(2)
    pp.loop[0].name = "vessel"
    pp.loop[0].element.resize(2)
    for index, centre in enumerate((-0.5, 0.5)):
        outline = pp.loop[0].element[index].geometry.outline
        r, z = _square(1.8, centre, 0.02)
        outline.r, outline.z = r, z
        pp.loop[0].element[index].turns_with_sign = 1.0
    pp.loop[1].name = "shapeless"
    pp.loop[1].element.resize(1)
    return pp


def _magnetics(angles: list[float | None], loop_positions: list[list[tuple]]):
    magnetics = imas.IDSFactory().magnetics()
    magnetics.ids_properties.homogeneous_time = 2
    magnetics.b_field_pol_probe.resize(len(angles))
    for index, angle in enumerate(angles):
        probe = magnetics.b_field_pol_probe[index]
        probe.name = f"probe_{index}"
        probe.position.r = 1.0 + 0.1 * index
        probe.position.z = 0.2 * index
        probe.length = 0.05
        if angle is not None:
            probe.poloidal_angle = angle
    magnetics.flux_loop.resize(len(loop_positions))
    for index, points in enumerate(loop_positions):
        loop = magnetics.flux_loop[index]
        loop.name = f"loop_{index}"
        loop.position.resize(len(points))
        for k, (r, z) in enumerate(points):
            loop.position[k].r = r
            loop.position[k].z = z
    return magnetics


# --- pf_active ---------------------------------------------------------------


def test_unsourced_winding_becomes_nan_not_a_number(synthetic_pf_active):
    filaments, sections, flags = ag.read_artifact_pf_active(synthetic_pf_active)

    unsourced = [f for f in filaments if f.circuit == 0]
    assert len(unsourced) == 1
    assert np.isnan(unsourced[0].turns)
    assert not any(f.turns == 1.0 for f in unsourced), "a default turn count leaked in"
    assert any("turns_with_sign is unresolved" in flag for flag in flags)
    assert len(sections) == len(filaments)


def test_a_filled_winding_keeps_its_value_and_sign(synthetic_pf_active):
    filaments, _sections, _flags = ag.read_artifact_pf_active(synthetic_pf_active)

    sourced = [f for f in filaments if f.circuit == 1]
    assert len(sourced) == 2
    assert all(f.turns == -12.0 for f in sourced)


def test_a_coil_is_one_circuit(synthetic_pf_active):
    filaments, _sections, _flags = ag.read_artifact_pf_active(synthetic_pf_active)

    assert sorted({f.circuit for f in filaments}) == [0, 1]


def test_the_outline_survives_beside_the_box_that_stands_in_for_it(
    synthetic_pf_active,
):
    filaments, sections, _flags = ag.read_artifact_pf_active(synthetic_pf_active)

    first = filaments[0]
    assert first.r == pytest.approx(1.5)
    assert first.z == pytest.approx(0.4)
    assert first.width == pytest.approx(0.1)
    assert first.height == pytest.approx(0.1)
    assert sections[0].circuit == first.circuit
    assert sections[0].vertices.shape == (4, 2)


def test_a_repeated_closing_vertex_is_not_kept_as_a_corner(synthetic_pf_active):
    _filaments, sections, _flags = ag.read_artifact_pf_active(synthetic_pf_active)

    closed = [s for s in sections if s.circuit == 1]
    assert all(s.vertices.shape == (4, 2) for s in closed)


# --- pf_passive --------------------------------------------------------------


def test_each_passive_element_is_its_own_circuit(synthetic_pf_passive):
    filaments, sections, structures, flags = ag.read_artifact_pf_passive(
        synthetic_pf_passive, first_circuit=7
    )

    assert [f.circuit for f in filaments] == [7, 8]
    assert [s.name for s in structures] == ["vessel_0", "vessel_1"]
    assert len(sections) == 2
    assert any("one per ELEMENT" in flag for flag in flags)


def test_a_passive_element_without_a_shape_is_dropped_and_named(
    synthetic_pf_passive,
):
    filaments, _sections, _structures, flags = ag.read_artifact_pf_passive(
        synthetic_pf_passive, first_circuit=0
    )

    assert len(filaments) == 2
    assert any("shapeless" in flag and "dropped" in flag for flag in flags)


# --- magnetics ---------------------------------------------------------------


def test_a_probe_without_an_orientation_is_dropped_not_assumed():
    magnetics = _magnetics([0.0, None, np.pi / 2], [[(1.0, 0.0)]])

    b_probes, _flux_loops, flags = ag.read_artifact_magnetics(magnetics)

    assert [p.index for p in b_probes] == [0, 2]
    assert any("no poloidal_angle" in flag for flag in flags)


def test_one_shared_orientation_across_every_probe_is_flagged():
    magnetics = _magnetics([np.pi / 2] * 4, [[(1.0, 0.0)]])

    _b_probes, _flux_loops, flags = ag.read_artifact_magnetics(magnetics)

    assert any("cannot separate" in flag for flag in flags)


def test_a_mixed_orientation_probe_set_is_not_flagged():
    magnetics = _magnetics([0.0, np.pi / 2, 0.0, np.pi / 2], [[(1.0, 0.0)]])

    _b_probes, _flux_loops, flags = ag.read_artifact_magnetics(magnetics)

    assert not any("cannot separate" in flag for flag in flags)


def test_a_toroidally_spread_flux_loop_is_flagged_but_a_co_located_one_is_not():
    magnetics = _magnetics(
        [0.0, np.pi / 2],
        [[(1.0, 0.5), (1.0, 0.5)], [(1.0, 0.5), (1.4, 0.9)]],
    )

    _b_probes, flux_loops, flags = ag.read_artifact_magnetics(magnetics)

    assert len(flux_loops) == 2
    spread_flags = [flag for flag in flags if "position points spanning" in flag]
    assert len(spread_flags) == 1
    assert "loop_1" in spread_flags[0]
    assert flux_loops[1].r == pytest.approx(1.2)


# --- the unresolved-turn guard ----------------------------------------------


def _table(turns: list[float]) -> GeometryTable:
    from imas_ambix.gs.geometry import PFFilament

    return GeometryTable(
        signature=SetupSignature(
            n_bprobe=0, n_fluxloop=0, n_pf_filament=len(turns), n_limiter=0, digest="x"
        ),
        shots=[1],
        b_probes=[],
        flux_loops=[],
        pf_filaments=[
            PFFilament(
                r=1.0,
                z=0.0,
                turns=t,
                width=0.1,
                height=0.1,
                circuit=i,
                xmult=1.0,
            )
            for i, t in enumerate(turns)
        ],
        limiter_r=[],
        limiter_z=[],
        sensor_map=[],
        passive_structures=[],
        amc_current_channels=[],
        unmatched_amb=[],
    )


def test_the_guard_names_the_circuits_that_block_an_operator():
    table = _table([10.0, ag.UNRESOLVED_TURNS, 5.0, ag.UNRESOLVED_TURNS])

    assert ag.unresolved_turn_circuits(table) == (1, 3)
    with pytest.raises(ag.UnresolvedTurnsError, match=r"\[1, 3\]"):
        ag.require_resolved_turns(table)


def test_the_guard_passes_a_fully_sourced_table():
    ag.require_resolved_turns(_table([10.0, 5.0]))


# --- the campaign-side inputs the artifact cannot supply ---------------------


def test_the_sensor_arrays_are_presented_in_the_shape_the_mapper_reads():
    from imas_ambix.gs.geometry import BProbe, FluxLoop

    arrays = ag.sensor_position_arrays(
        [BProbe(index=0, r=1.0, z=0.5, angle_deg=90.0, length=0.02)],
        [FluxLoop(index=0, r=1.4, z=-0.3)],
    )

    assert set(arrays) == {"magpr_r", "magpr_z", "magpr_ang", "silop_r", "silop_z"}
    assert arrays["magpr_ang"].tolist() == [90.0]
    assert arrays["silop_r"].tolist() == [1.4]


# --- integration against a published artifact --------------------------------

_CACHE = os.environ.get("AMBIX_MACHINE_ARTIFACT_CACHE", "")
_DIGEST = os.environ.get("AMBIX_MACHINE_ARTIFACT_DIGEST", "")
_PHYSICAL_DIGEST = "76cf833561e602a7"
_REGISTRY_DIGEST = "73ecabaa030a476d80cc24c1fe35d038876a12454ebd7b0c7055aac1d3cf3ab2"
_SEMANTIC_IDENTITY = (
    "sha256:680076be575bf625a8f546ba90c11563a5e9e22b81aa3dd6388f6f64be1d276e"
)
_SHOT = 21983

_skip_no_artifact = pytest.mark.skipif(
    not (_CACHE and _DIGEST),
    reason="no machine-description artifact named in the environment "
    "(AMBIX_MACHINE_ARTIFACT_CACHE + AMBIX_MACHINE_ARTIFACT_DIGEST)",
)


@pytest.fixture(scope="module")
def artifact_table():
    return ag.MachineArtifactGeometryReader(
        cache_directory=_CACHE,
        digest=_DIGEST,
        shot=_SHOT,
        expected_physical_digest=_PHYSICAL_DIGEST,
        expected_registry_digest=_REGISTRY_DIGEST,
    ).read()


@pytest.fixture(scope="module")
def efm_table():
    return build_table_for_shot(_SHOT)


@_skip_no_artifact
def test_the_table_carries_the_identity_it_was_built_from(artifact_table):
    flags = "\n".join(artifact_table.provenance_flags)

    assert _SEMANTIC_IDENTITY in flags
    assert _PHYSICAL_DIGEST in flags
    assert _REGISTRY_DIGEST in flags
    assert "dictionary pin 4.1.1" in flags
    assert "registry evidence for the selected shot: observed" in flags
    assert "forward-model blocker: pf_active/coil/element/turns_with_sign" in flags


@_skip_no_artifact
def test_the_published_winding_is_unresolved_and_blocks_an_operator(artifact_table):
    circuits = ag.unresolved_turn_circuits(artifact_table)

    assert len(circuits) == 13, "every active coil's winding is unsourced"
    assert all(
        np.isnan(f.turns) for f in artifact_table.pf_filaments if f.circuit in circuits
    )
    with pytest.raises(ag.UnresolvedTurnsError):
        ag.require_resolved_turns(artifact_table)


@_skip_no_artifact
def test_the_limiter_contour_matches_the_efm_reader(artifact_table, efm_table):
    """The plasma-facing boundary is the same contour to a few microns."""
    efm_r = np.asarray(efm_table.limiter_r)
    efm_z = np.asarray(efm_table.limiter_z)
    art_r = np.asarray(artifact_table.limiter_r)
    art_z = np.asarray(artifact_table.limiter_z)
    assert art_r.size == efm_r.size

    distance = np.hypot(
        efm_r[:, None] - art_r[None, :], efm_z[:, None] - art_z[None, :]
    )
    assert distance.min(axis=1).max() < 1e-5

    def enclosed(r, z):
        return 0.5 * abs(np.dot(r, np.roll(z, -1)) - np.dot(z, np.roll(r, -1)))

    assert enclosed(art_r, art_z) == pytest.approx(enclosed(efm_r, efm_z), rel=1e-5)


@_skip_no_artifact
def test_the_probe_positions_match_the_efm_reader(artifact_table, efm_table):
    """Position is a bijection to well under a millimetre."""
    from scipy.optimize import linear_sum_assignment

    efm_r = np.array([p.r for p in efm_table.b_probes])
    efm_z = np.array([p.z for p in efm_table.b_probes])
    art_r = np.array([p.r for p in artifact_table.b_probes])
    art_z = np.array([p.z for p in artifact_table.b_probes])
    assert art_r.size == efm_r.size

    distance = np.hypot(
        efm_r[:, None] - art_r[None, :], efm_z[:, None] - art_z[None, :]
    )
    rows, columns = linear_sum_assignment(distance)
    assert distance[rows, columns].max() < 1e-6


@_skip_no_artifact
def test_the_probe_orientations_do_not_yet_match_the_efm_reader(
    artifact_table, efm_table
):
    """The artifact revision records one sensitive axis for every probe.

    ``efm`` carries a mixed set -- outboard radial probes at 0 degrees beside
    vertical probes at 90 -- so the two sources disagree on which field
    component those rows measure.  That is a much larger error than any
    positional tolerance would catch, and the reader flags it from the data
    alone.  This test is the standing record of the disagreement: it fails, and
    must be rewritten to assert agreement, when a revision resolves the
    orientations.
    """
    efm_angles = np.array([p.angle_deg for p in efm_table.b_probes])
    art_angles = np.array([p.angle_deg for p in artifact_table.b_probes])

    assert np.unique(np.round(np.mod(art_angles, 180.0), 3)).size == 1
    assert np.unique(np.round(np.mod(efm_angles, 180.0), 3)).size > 1
    assert any("cannot separate" in flag for flag in artifact_table.provenance_flags)


@_skip_no_artifact
def test_each_coil_outline_reproduces_the_efm_winding_envelope(
    artifact_table, efm_table
):
    """The two sources describe the same conductors at different granularity.

    ``efm`` tiles a coil with filaments; the artifact publishes one outline.
    Take the filaments that fall inside an outline and the envelope they span
    must be the outline's own extent -- which is the statement that no
    conductor moved, made without depending on either side's discretization.
    """
    from matplotlib.path import Path as PolygonPath

    filament_r = np.array([f.r for f in efm_table.pf_filaments])
    filament_z = np.array([f.z for f in efm_table.pf_filaments])
    filament_w = np.array([abs(f.width) for f in efm_table.pf_filaments])
    filament_h = np.array([abs(f.height) for f in efm_table.pf_filaments])
    points = np.column_stack([filament_r, filament_z])

    active = [f for f in artifact_table.pf_filaments if np.isnan(f.turns)]
    sections = {s.circuit: s for s in artifact_table.polygon_sections}
    assert len(active) == 13

    for coil in active:
        inside = PolygonPath(sections[coil.circuit].vertices).contains_points(points)
        assert inside.any(), f"no efm filament inside circuit {coil.circuit}"
        r, z = filament_r[inside], filament_z[inside]
        w, h = filament_w[inside], filament_h[inside]
        width = (r + w / 2).max() - (r - w / 2).min()
        height = (z + h / 2).max() - (z - h / 2).min()
        assert width == pytest.approx(coil.width, abs=1e-4)
        assert height == pytest.approx(coil.height, abs=1e-4)


@_skip_no_artifact
def test_the_flux_loop_coverage_difference_is_bounded_and_known(
    artifact_table, efm_table
):
    """Most loops coincide; a named minority has no artifact counterpart.

    The two sources do not carry the same loop set, so this pins how far apart
    they are rather than pretending they agree: over half the ``efm`` loops sit
    on an artifact loop to within a millimetre, and the loops with no
    counterpart are counted so a growing gap is a test failure.
    """
    efm_r = np.array([f.r for f in efm_table.flux_loops])
    efm_z = np.array([f.z for f in efm_table.flux_loops])
    art_r = np.array([f.r for f in artifact_table.flux_loops])
    art_z = np.array([f.z for f in artifact_table.flux_loops])

    nearest = np.hypot(
        efm_r[:, None] - art_r[None, :], efm_z[:, None] - art_z[None, :]
    ).min(axis=1)
    assert (nearest < 1e-3).sum() >= 26
    assert (nearest < 1e-2).sum() >= 39
    assert (nearest > 0.05).sum() <= 7


@_skip_no_artifact
def test_a_campaign_channel_set_is_what_makes_the_table_drivable():
    """The artifact describes conductors; it does not describe an acquisition system.

    Without the campaign's measured coil-current channels no circuit is
    classified as driven and the vacuum coil block is empty, so the reader takes
    them from the caller.  This pins that the pass-through is what turns a
    geometry table into one a forward operator can build a coil block from.
    """
    from imas_ambix.gs import operator as op
    from imas_ambix.gs.geometry import read_amc_current_channels

    channels = tuple(read_amc_current_channels(_SHOT))
    assert channels

    bare = ag.MachineArtifactGeometryReader(
        cache_directory=_CACHE, digest=_DIGEST, shot=_SHOT
    ).read()
    driven = ag.MachineArtifactGeometryReader(
        cache_directory=_CACHE,
        digest=_DIGEST,
        shot=_SHOT,
        amc_current_channels=channels,
    ).read()

    assert op.build_operator(bare).pf_merged_circuits == []
    assert len(op.build_operator(driven).pf_merged_circuits) == 13


@_skip_no_artifact
def test_the_carried_channel_set_does_not_map_onto_this_revision():
    """Carrying the campaign's amb channels onto artifact geometry loses most of them.

    ``map_amb_sensors`` picks its B-probe candidates by exact equality against
    the orientation a channel's name implies.  This revision stores every
    poloidal angle in radians, so the degree value lands 2e-4 away from a whole
    degree and matches nothing; the radial channels then have no candidate at
    all, because no probe here is radial.  The mapping is therefore not usable
    as published, which is a blocker to record rather than a tolerance to widen.
    """
    from imas_ambix.gs.geometry import canonical_amb_channels

    channels = tuple(canonical_amb_channels([_SHOT]))
    table = ag.MachineArtifactGeometryReader(
        cache_directory=_CACHE, digest=_DIGEST, shot=_SHOT, amb_channels=channels
    ).read()

    assert len(table.unmatched_amb) > len(table.sensor_map)
