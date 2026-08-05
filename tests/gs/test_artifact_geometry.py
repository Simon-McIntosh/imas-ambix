"""Tests for the artifact-backed machine-geometry reader.

Two layers:

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
    filaments, sections, _drives, flags = ag.read_artifact_pf_active(
        synthetic_pf_active
    )

    unsourced = [f for f in filaments if f.circuit == 0]
    assert len(unsourced) == 1
    assert np.isnan(unsourced[0].turns)
    assert not any(f.turns == 1.0 for f in unsourced), "a default turn count leaked in"
    assert any("turns_with_sign is unresolved" in flag for flag in flags)
    assert len(sections) == len(filaments)


def test_a_filled_winding_keeps_its_value_and_sign(synthetic_pf_active):
    filaments, _sections, _drives, _flags = ag.read_artifact_pf_active(
        synthetic_pf_active
    )

    sourced = [f for f in filaments if f.circuit == 1]
    assert len(sourced) == 2
    assert all(f.turns == -12.0 for f in sourced)


def test_a_coil_is_one_circuit(synthetic_pf_active):
    filaments, _sections, _drives, _flags = ag.read_artifact_pf_active(
        synthetic_pf_active
    )

    assert sorted({f.circuit for f in filaments}) == [0, 1]


def test_the_outline_survives_beside_the_box_that_stands_in_for_it(
    synthetic_pf_active,
):
    filaments, sections, _drives, _flags = ag.read_artifact_pf_active(
        synthetic_pf_active
    )

    first = filaments[0]
    assert first.r == pytest.approx(1.5)
    assert first.z == pytest.approx(0.4)
    assert first.width == pytest.approx(0.1)
    assert first.height == pytest.approx(0.1)
    assert sections[0].circuit == first.circuit
    assert sections[0].vertices.shape == (4, 2)


def test_a_repeated_closing_vertex_is_not_kept_as_a_corner(synthetic_pf_active):
    _filaments, sections, _drives, _flags = ag.read_artifact_pf_active(
        synthetic_pf_active
    )

    closed = [s for s in sections if s.circuit == 1]
    assert all(s.vertices.shape == (4, 2) for s in closed)


# --- pf_passive --------------------------------------------------------------


def test_each_passive_element_is_its_own_circuit(synthetic_pf_passive):
    filaments, sections, structures, _drives, flags = ag.read_artifact_pf_passive(
        synthetic_pf_passive, first_circuit=7
    )

    assert [f.circuit for f in filaments] == [7, 8]
    assert [s.name for s in structures] == ["vessel_0", "vessel_1"]
    assert len(sections) == 2
    assert any("one per ELEMENT" in flag for flag in flags)


def test_a_passive_element_without_a_shape_is_dropped_and_named(
    synthetic_pf_passive,
):
    filaments, _sections, _structures, _drives, flags = ag.read_artifact_pf_passive(
        synthetic_pf_passive, first_circuit=0
    )

    assert len(filaments) == 2
    assert any("shapeless" in flag and "dropped" in flag for flag in flags)


# --- the electrical drive map ------------------------------------------------


def _drive(
    conductor: str,
    elements: tuple[int, ...],
    weight: float,
    *,
    channel: str = "",
    container: str = "pf_active",
    distribution: str = "single",
    evidence: str = "measured",
) -> ag.ResolvedDrive:
    return ag.ResolvedDrive(
        channel=channel or f"{conductor}_current",
        container=container,
        conductor=conductor,
        elements=elements,
        weight=weight,
        distribution=distribution,
        evidence=evidence,
    )


def test_a_stated_weight_supersedes_the_conductors_turn_count(synthetic_pf_active):
    """A drive weight and a turn count answer the same question; applying both
    scales the winding by its turns twice."""
    drive = _drive("sourced_winding", (0, 1), 5.0, distribution="section_area")
    filaments, _sections, drives, _flags = ag.read_artifact_pf_active(
        synthetic_pf_active, [drive]
    )

    driven = [f for f in filaments if f.circuit == 1]
    assert sum(f.xmult for f in driven) == pytest.approx(5.0)
    # the turn count is still recorded, and is NOT what scaled the column
    assert all(f.turns == -12.0 for f in driven)
    assert drives[0].ampere_turns_per_ampere == pytest.approx(5.0)


def test_a_conductor_no_drive_names_keeps_unit_weight(synthetic_pf_active):
    filaments, _sections, drives, _flags = ag.read_artifact_pf_active(
        synthetic_pf_active,
        [_drive("sourced_winding", (0, 1), 5.0, distribution="section_area")],
    )

    undriven = [f for f in filaments if f.circuit == 0]
    assert [f.xmult for f in undriven] == [1.0]
    assert [d.conductor for d in drives] == ["sourced_winding"]


def test_a_drive_reaching_one_element_leaves_the_others_unpowered(
    synthetic_pf_active,
):
    """A conductor's element the drive does not name carries none of its current."""
    filaments, _sections, _drives, _flags = ag.read_artifact_pf_active(
        synthetic_pf_active, [_drive("sourced_winding", (0,), 3.0)]
    )

    driven = [f for f in filaments if f.circuit == 1]
    assert [f.xmult for f in driven] == [3.0, 0.0]


def test_a_group_the_drive_names_is_one_circuit_split_by_section_area(
    synthetic_pf_passive,
):
    """A drive is the passive topology the artifact otherwise leaves unsourced:
    these plates carry ONE measured current, shared out by cross-section."""
    pp = synthetic_pf_passive
    # widen the second element so the two sections differ 4:1 in area
    outline = pp.loop[0].element[1].geometry.outline
    outline.r = [1.76, 1.84, 1.84, 1.76]
    outline.z = [0.46, 0.46, 0.54, 0.54]

    drive = _drive(
        "vessel",
        (0, 1),
        1.0,
        container="pf_passive",
        distribution="section_area",
        evidence="generated",
    )
    filaments, sections, structures, drives, flags = ag.read_artifact_pf_passive(
        pp, first_circuit=7, drives=[drive]
    )

    assert {f.circuit for f in filaments} == {7}
    assert sum(f.xmult for f in filaments) == pytest.approx(1.0)
    areas = [0.04 * 0.04, 0.08 * 0.08]
    expected = [a / sum(areas) for a in areas]
    assert [f.xmult for f in filaments] == pytest.approx(expected)
    assert [d.circuit for d in drives] == [7]
    assert [s.name for s in structures] == ["vessel_0", "vessel_1"]
    # a section replaces ONE circuit's box, so a grouped circuit gets none
    assert sections == []
    assert any("split by section area" in flag for flag in flags)


def test_an_element_outside_every_drive_keeps_its_own_circuit(synthetic_pf_passive):
    drive = _drive("vessel", (0,), 1.0, container="pf_passive", evidence="generated")
    filaments, sections, _structures, drives, _flags = ag.read_artifact_pf_passive(
        synthetic_pf_passive, first_circuit=7, drives=[drive]
    )

    assert [f.circuit for f in filaments] == [7, 8]
    assert [f.xmult for f in filaments] == [1.0, 1.0]
    assert [d.circuit for d in drives] == [7]
    # only the induced element keeps a polygon section
    assert [s.circuit for s in sections] == [8]


class _FakeDrive:
    def __init__(self, channel, container, conductor, elements, weight, evidence):
        self.channel = channel
        self.container = container
        self.conductor = conductor
        self.elements = elements
        self.ampere_turns_per_ampere = weight
        self.distribution = "single" if len(elements) == 1 else "section_area"
        self.evidence = evidence


class _FakeDriveMap:
    def __init__(self, drives):
        self.drives = tuple(drives)

    def select(self, channels):
        wanted = set(channels)
        return _FakeDriveMap([d for d in self.drives if d.channel in wanted])


class _FakeManifest:
    def __init__(self, drives):
        self.channel_drive = tuple(drives)
        self.drive_map = _FakeDriveMap(drives)


def test_a_conductor_measured_twice_is_one_column_read_once():
    """A coil's own current and the feed current behind it reach the same
    elements, so they are one column -- and the direct measurement is what it
    is read through, never the fitted conversion standing in for one."""
    manifest = _FakeManifest(
        [
            _FakeDrive(
                "p4u_coil_current", "pf_active", "p4_upper", (0,), 1.0, "measured"
            ),
            _FakeDrive(
                "p4u_feed_current", "pf_active", "p4_upper", (0,), 22.15, "fitted"
            ),
        ]
    )

    resolved, flags = ag.resolve_drives(manifest)

    assert len(resolved) == 1
    assert resolved[0].channel == "p4u_coil_current"
    assert resolved[0].weight == pytest.approx(1.0)
    assert flags == []


def test_a_column_this_campaign_does_not_record_is_left_induced():
    """The description saying a conductor is supplied and the acquisition set
    saying nothing measured it are both true; no weight stands in."""
    manifest = _FakeManifest(
        [
            _FakeDrive(
                "p4u_coil_current", "pf_active", "p4_upper", (0,), 1.0, "measured"
            ),
            _FakeDrive("p7_coil_current", "pf_active", "p7", (0,), 1.0, "measured"),
        ]
    )

    resolved, flags = ag.resolve_drives(manifest, ["p4u_coil_current"])

    assert [d.conductor for d in resolved] == ["p4_upper"]
    assert any("left induced" in flag and "pf_active/p7" in flag for flag in flags)


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
_PHYSICAL_DIGEST = "ca06c8f64481114f"
_REGISTRY_DIGEST = "7083e8029c879310d4b811ecc58f5eefdd40b2bfe01b4a1714b177b03a307366"
_SEMANTIC_IDENTITY = (
    "sha256:18c75c19493714108fc71f88a55cc775836218e489e073f43942fd007d937bdc"
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


@pytest.fixture(scope="module")
def carried_artifact_table():
    """The artifact table with the campaign's own sensor channels carried onto it."""
    from imas_ambix.gs.geometry import canonical_amb_channels

    return ag.MachineArtifactGeometryReader(
        cache_directory=_CACHE,
        digest=_DIGEST,
        shot=_SHOT,
        amb_channels=tuple(canonical_amb_channels([_SHOT])),
    ).read()


@_skip_no_artifact
def test_the_table_carries_the_identity_it_was_built_from(artifact_table):
    flags = "\n".join(artifact_table.provenance_flags)

    assert _SEMANTIC_IDENTITY in flags
    assert _PHYSICAL_DIGEST in flags
    assert _REGISTRY_DIGEST in flags
    assert "dictionary pin 4.1.1" in flags
    assert "registry evidence for the selected shot: observed" in flags
    assert "forward-model blocker: pf_active/coil(p6_upper)/element/" in flags
    assert "forward-model blocker: pf_active/coil(p6_lower)/element/" in flags


@_skip_no_artifact
def test_only_the_two_windings_the_fit_could_not_reach_are_unresolved(artifact_table):
    """The turn counts are sourced except where the vacuum fit had no leverage.

    An earlier revision left every active winding unsourced; this one carries a
    fitted count for all but the P6 pair, whose supplies the campaign holds at
    zero, so no measured current ever excites them and no fit can separate their
    turns.  Those two remain NaN rather than taking a plausible number, and the
    guard still names them, because a consumer that scales with turns must not
    proceed on a guess for any coil.
    """
    circuits = ag.unresolved_turn_circuits(artifact_table)
    sections = {s.circuit: s.name for s in artifact_table.polygon_sections}

    assert sorted(sections[c] for c in circuits) == ["p6_lower", "p6_upper"]
    assert all(
        np.isnan(f.turns) for f in artifact_table.pf_filaments if f.circuit in circuits
    )
    resolved = [
        f
        for f in artifact_table.pf_filaments
        if f.circuit in artifact_table.active_circuits and f.circuit not in circuits
    ]
    assert len(resolved) == 11
    assert all(np.isfinite(f.turns) and f.turns != 0.0 for f in resolved)
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
def test_the_probe_orientations_match_the_efm_reader(artifact_table, efm_table):
    """Both sources split the same probes between the same two sensitive axes.

    A poloidal probe enters the operator as ``B_R cos(theta) + B_Z sin(theta)``,
    so an orientation decides which field component a row measures, not merely
    how it is scaled -- an error of the whole 90 degrees, far above anything a
    positional tolerance would catch.  An earlier revision authored every probe
    on one axis; this one places the outboard radial family along R, and the
    two sources now agree probe for probe on the assignment and on how many
    probes each axis carries.  The residual disagreement is the fraction of a
    degree a source holding radians rounds to.
    """
    efm_angles = np.array([p.angle_deg for p in efm_table.b_probes])
    art_angles = np.array([p.angle_deg for p in artifact_table.b_probes])
    assert art_angles.size == efm_angles.size

    efm_axes, efm_counts = np.unique(np.round(efm_angles, 0), return_counts=True)
    art_axes, art_counts = np.unique(np.round(art_angles, 0), return_counts=True)
    assert efm_axes.tolist() == [0.0, 90.0]
    assert art_axes.tolist() == efm_axes.tolist()
    assert art_counts.tolist() == efm_counts.tolist() == [19, 59]

    # co-located radial/vertical pairs make any position-only pairing ambiguous,
    # so compare the (position, axis) triples as sets: that is the statement
    # that each probe carries the same axis in both sources, without depending
    # on either side's ordering
    def triples(probes):
        return sorted(
            (round(p.r, 6), round(p.z, 6), round(p.angle_deg, 0)) for p in probes
        )

    assert triples(artifact_table.b_probes) == triples(efm_table.b_probes)
    assert not any(
        "cannot separate" in flag for flag in artifact_table.provenance_flags
    )


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

    active = [
        f
        for f in artifact_table.pf_filaments
        if f.circuit in set(artifact_table.active_circuits)
    ]
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
    assert len(op.build_operator(driven).pf_merged_circuits) == 21


@_skip_no_artifact
def test_the_carried_channel_set_resolves_onto_the_same_sensors(
    carried_artifact_table, efm_table
):
    """The campaign's channels land on artifact geometry as they do on ``efm``.

    This is the statement that lets one campaign's acquisition be read against
    a published machine description: the same 96 channels map and the same four
    do not, each mapped channel keeps its kind, and the sensor it resolves to is
    the same physical sensor -- probes to within tens of nanometres and a
    fraction of a degree, flux loops to within the few millimetres by which the
    two sources place a loop that both carry.
    """
    table = carried_artifact_table
    art = {m.amb_channel: m for m in table.sensor_map}
    efm = {m.amb_channel: m for m in efm_table.sensor_map}
    assert set(art) == set(efm)
    assert sorted(table.unmatched_amb) == sorted(efm_table.unmatched_amb)
    assert all(art[c].kind == efm[c].kind for c in art)

    def offset(kind):
        return max(
            np.hypot(art[c].r - efm[c].r, art[c].z - efm[c].z)
            for c in art
            if art[c].kind == kind
        )

    assert offset("b_probe") < 1e-6
    assert offset("flux_loop") < 1e-2
    assert (
        max(
            abs(art[c].angle_deg - efm[c].angle_deg)
            for c in art
            if art[c].kind == "b_probe"
        )
        < 1e-3
    )


@_skip_no_artifact
def test_the_extra_artifact_flux_loops_never_reach_the_forward_model(
    carried_artifact_table, efm_table
):
    """A source may describe more loops than a campaign acquired, and that is fine.

    The artifact carries 80 flux loops against ``efm``'s 46, which asks whether
    a parity comparison has to restrict the loop set by hand.  It does not: the
    forward operator's rows come from the sensor map, which is keyed by the
    campaign's own channels, so a loop no channel names is never given a row.
    Both sources therefore present the same 19 flux-loop rows, and the loops
    only one source describes are simply unused rather than a coverage gap.
    """
    art_mapped = [m for m in carried_artifact_table.sensor_map if m.kind == "flux_loop"]
    efm_mapped = [m for m in efm_table.sensor_map if m.kind == "flux_loop"]

    assert len(carried_artifact_table.flux_loops) > len(efm_table.flux_loops)
    assert [m.amb_channel for m in art_mapped] == [m.amb_channel for m in efm_mapped]
    assert len({m.efm_index for m in art_mapped}) == len(
        {m.efm_index for m in efm_mapped}
    )


@_skip_no_artifact
def test_the_supplied_conductors_are_the_ones_the_source_files_as_active(
    artifact_table,
):
    """Position alone would drive the structure around a coil with its current.

    The artifact resolves its structure far more finely than the campaign
    arrays do, and much of it sits closer to a winding than the radius a
    positional classifier matches within -- so classifying on geometry alone
    promotes far more circuits to driven coils than the machine supplies, each
    fed a winding's measured current.  The source states which conductors it
    supplies and through which channel, and that statement is what the
    classification takes.
    """
    from imas_ambix.gs import operator as op
    from imas_ambix.gs.geometry import read_amc_current_channels

    channels = tuple(read_amc_current_channels(_SHOT))
    table = ag.MachineArtifactGeometryReader(
        cache_directory=_CACHE,
        digest=_DIGEST,
        shot=_SHOT,
        amc_current_channels=channels,
    ).read()

    assert len(table.active_circuits) == 13  # the pf_active windings
    stated = op.classify_circuits(
        table.pf_filaments,
        table.amc_current_channels,
        table.active_circuits,
        table.circuit_drives,
    )
    positional = op.classify_circuits(table.pf_filaments, table.amc_current_channels)

    known = [c for c in stated if c.role != "inferred_passive"]
    assert [c.circuit for c in known] == sorted(d.circuit for d in table.circuit_drives)
    assert len({c.amc_channel for c in known}) == 21
    assert all(c.source_stated_weight for c in known)
    # every declared column is driven by its own channel, never a neighbour's
    assert {c.amc_channel for c in known} == {d.channel for d in table.circuit_drives}
    # position alone over-promotes, which is the whole reason the source is read
    assert sum(1 for c in positional if c.role != "inferred_passive") > len(known)


@pytest.fixture(scope="module")
def driven_operators():
    """The artifact operator and the campaign operator, on shared sensor rows."""
    from imas_ambix.gs import operator as op
    from imas_ambix.gs.geometry import canonical_amb_channels

    efm = build_table_for_shot(_SHOT)
    artifact = ag.MachineArtifactGeometryReader(
        cache_directory=_CACHE,
        digest=_DIGEST,
        shot=_SHOT,
        amb_channels=tuple(canonical_amb_channels([_SHOT])),
        amc_current_channels=tuple(efm.amc_current_channels),
    ).read()
    op_efm, op_art = op.build_operator(efm), op.build_operator(artifact)
    rows = {channel: i for i, channel in enumerate(op_efm.sensor_channels)}
    shared_art = [
        i for i, channel in enumerate(op_art.sensor_channels) if channel in rows
    ]
    shared_efm = [
        rows[channel] for channel in op_art.sensor_channels if channel in rows
    ]
    return op_art, op_efm, shared_art, shared_efm


def _column_ratio(op_art, op_efm, shared_art, shared_efm, channel: str) -> float:
    """Least-squares scale between the two operators' columns for one channel."""
    j = op_art.pf_amc_channels.index(channel)
    k = op_efm.pf_amc_channels.index(channel)
    a = op_art.g_pf[shared_art, j]
    e = op_efm.g_pf[shared_efm, k]
    return float(a @ e / (e @ e))


@_skip_no_artifact
def test_the_two_descriptions_drive_the_same_columns(driven_operators):
    """Both sources supply the same 21 conductors through the same channels."""
    op_art, op_efm, _shared_art, _shared_efm = driven_operators

    assert len(op_art.pf_amc_channels) == 21
    assert sorted(op_art.pf_amc_channels) == sorted(op_efm.pf_amc_channels)


@_skip_no_artifact
def test_every_coil_column_reproduces_the_campaigns(driven_operators):
    """The twelve PF windings agree to better than a percent.

    They are the columns both sources describe the same way -- a measured
    channel already in ampere turns driving one winding -- so what is left is
    the difference between the two discretisations of the same conductor.
    """
    op_art, op_efm, shared_art, shared_efm = driven_operators

    coils = [
        channel
        for channel in op_art.pf_amc_channels
        if channel.endswith("_current")
        and "_case_" not in channel
        and channel != "sol_current"
    ]
    assert len(coils) == 12
    for channel in coils:
        ratio = _column_ratio(op_art, op_efm, shared_art, shared_efm, channel)
        assert ratio == pytest.approx(1.0, abs=0.01), channel


@_skip_no_artifact
def test_every_case_column_reproduces_the_campaigns(driven_operators):
    """The eight case groups agree once the drive supplies their topology.

    The artifact files every case as passive structure; the drive map is what
    says which plates share a measured current, and splitting that current by
    section area is the same non-uniform share the campaign arrays carry.  The
    residual few percent is discretisation -- the two sources cut the same
    enclosures into different numbers of plates.
    """
    op_art, op_efm, shared_art, shared_efm = driven_operators

    cases = [c for c in op_art.pf_amc_channels if c.endswith("_case_current")]
    assert len(cases) == 8
    for channel in cases:
        ratio = _column_ratio(op_art, op_efm, shared_art, shared_efm, channel)
        assert ratio == pytest.approx(1.0, abs=0.05), channel


@_skip_no_artifact
def test_the_solenoid_weight_the_source_states_replaces_the_fitted_correction(
    driven_operators,
):
    """The two sources agree on the solenoid to 3%, not the 5% a raw turn
    count suggests.

    The campaign arrays carry 328 ampere turns per ampere of ``sol_current``
    and the operator multiplies that column by
    :data:`~imas_ambix.gs.operator.SOLENOID_RESPONSE_SCALE`, a vacuum-measured
    correction its own documentation records as degenerate with a turn count.
    The campaign's EFFECTIVE weight is therefore ``328 x 1.0825 = 355.06``,
    which the artifact's fitted ``344.657`` sits within 3% of -- and inside the
    fitted interval, where the raw 328 is not.  Applying the correction to a
    stated weight as well would put the two 5% apart in the other direction,
    which is the same 8% counted twice.
    """
    from imas_ambix.gs.operator import SOLENOID_RESPONSE_SCALE

    op_art, op_efm, shared_art, shared_efm = driven_operators
    ratio = _column_ratio(op_art, op_efm, shared_art, shared_efm, "sol_current")

    assert ratio == pytest.approx(344.657 / (328.0 * SOLENOID_RESPONSE_SCALE), abs=0.01)
    assert ratio == pytest.approx(0.971, abs=0.01)
    # the correction is withheld, so the disagreement is 3% and not 5%
    assert abs(ratio - 1.0) < abs(ratio * SOLENOID_RESPONSE_SCALE - 1.0)
