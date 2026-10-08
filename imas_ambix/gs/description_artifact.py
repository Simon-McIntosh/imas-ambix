"""Author a JT-60SA machine-description artifact for each operating phase.

Nova's machine-artifact layer packs, verifies and caches a directory of IDS
containers as any named machine, but it takes the machine's identity -- the
physical and registry digests and the shot ranges -- as input, and both
producers that exist are hard-wired to one machine.  This module is the JT-60SA
identity producer over that generic layer.

It authors one artifact per operating phase.  OP1 and OP2 are two physical
identities, not one: their description stores carry the same five IDS names but
different ``pf_passive`` and ``wall`` content, so each phase's physical digest
is derived from its own geometry content and the two artifacts carry disjoint
shot ranges but one shared registry digest.

The geometry, registry and shot-range identity is derived, and the operator-
ready channel drives are authored from each phase's drive topology in
``machine_map.json``: one nova drive per ``pf_active`` coil, its channel taken
from the phase's acquisition declaration and its weight the coil's total
ampere-turns per ampere over the topology's connections.  The geometry
field-evidence ledger is not authored here, so each phase artifact is created
``complete=False``, its
``unresolved_gaps`` naming that absence, and it becomes ``complete`` only when
the ledger is supplied -- which this producer does not do.

For each phase the producer reads the phase's five IDSs from its description
store through imas, writes them into a fresh IMAS HDF5 entry
(``imas:hdf5?path=``, the directory form
:class:`~imas_ambix.gs.artifact_geometry.MachineArtifactGeometryReader` opens),
and calls nova's generic layer to assemble and materialize the manifest.  Only
the five IDSs are staged, so the store's converter ``receipt.json`` is never an
artifact member, and the source netCDF files are re-written in the IMAS HDF5
form rather than copied.

The physical digest is taken over the geometry content, never the file bytes:
two authorings of the same store publish different manifest digests, because
the containers carry library write metadata, but they describe one machine and
must carry one physical digest.
"""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import imas
import numpy as np

from imas_ambix.data.geometry_adapter import current_channel_from_conductors
from imas_ambix.data.paths import GHCR_OWNER, JT60SA_MAP_DIR, package_for_machine
from imas_ambix.gs.artifact_geometry import (
    read_artifact_limiter,
    read_artifact_magnetics,
    read_artifact_pf_active,
    read_artifact_pf_passive,
    representation_digest,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

#: The machine identity these artifacts are authored as.
MACHINE = "jt-60sa"

#: The operating phases, each a separate physical identity.
PHASES: tuple[str, ...] = ("OP1", "OP2")

#: The IDSs each phase store carries, and the members each artifact authors.
IDS_NAMES: tuple[str, ...] = ("magnetics", "pf_active", "pf_passive", "tf", "wall")

#: The data dictionary every phase store declares and every artifact pins.
DD_VERSION = "4.1.1"

#: The schema of the JT-60SA geometry registry this module builds.
REGISTRY_SCHEMA = "imas-ambix-jt-60sa-geometry-registry"

#: The in-checkout phase description stores.  The maps tree reaches a worktree
#: through a symlink to the main checkout, so these derive from the repository's
#: own map address rather than a hard-wired absolute path.
DEFAULT_DESCRIPTION_ROOT = JT60SA_MAP_DIR / "machine_description"

#: The packaged catalogue carrying each phase's first and last shot.
DEFAULT_MACHINE_MAP = JT60SA_MAP_DIR / "machine_map.json"

#: Each phase's shot-range evidence, keyed by phase.  OP1 is ``observed`` because
#: its description was checked against acquired OP1 pulses.  OP2 is
#: ``inherited`` because EDDB holds no OP2 pulse, so its identity comes from the
#: declared description alone.
PHASE_SHOT_EVIDENCE: dict[str, str] = {
    "OP1": "observed",
    "OP2": "inherited",
}

#: The absolute spread two elements of one coil may show on the per-element
#: ampere-turns and still be one winding.  The catalogue declares each
#: connection's weight rounded to two decimals, so two elements of one uniform
#: winding differ on their last declared digit by at most 0.01; the band sits
#: just above that so a pair rounded to the same winding (3.83 against 3.84)
#: is accepted while any larger spread is a genuine disagreement refused by
#: name.  The band is absolute rather than relative to the coil's scale, so a
#: small-magnitude coil is judged by the catalogue's own resolution rather than
#: by a fraction of its weight.
_WEIGHT_AGREEMENT = 0.011

#: The evidence the producer does not yet author: the geometry field ledger.
#: The absence is named on every phase artifact, so each stays incomplete.
NO_FIELD_EVIDENCE_GAP = "no field evidence ledger is authored"
DRIVE_TOPOLOGY_GAP = (
    "the channel drive map is not authored from the {phase} drive topology "
    "({drive_topology}) in machine_map.json"
)


@dataclass(frozen=True)
class Jt60saPhaseRange:
    """One operating phase's closed shot range and the identity it selects.

    The range is the phase's catalogue dates; the evidence state records how the
    phase's identity was checked, ``observed`` where its description was tested
    against acquired pulses and ``inherited`` where it comes from the declared
    description alone.
    """

    phase: str
    first_shot: int
    last_shot: int
    evidence: str
    physical_digest: str

    def contains(self, shot: int) -> bool:
        """Return whether ``shot`` lies in this closed range."""
        return self.first_shot <= shot <= self.last_shot


@dataclass(frozen=True)
class Jt60saConfiguration:
    """One JT-60SA operating phase's physical configuration.

    ``authoring_gaps`` is the phase artifact's ``unresolved_gaps``, so an
    identity resolved from this configuration reports the same incompleteness
    the artifact does, and a consumer can tell not-operator-ready from ready.
    """

    physical_digest: str
    geometry: Mapping[str, Any]
    authoring_gaps: tuple[str, ...]


@dataclass(frozen=True)
class Jt60saSelection:
    """The one phase a shot selects, with the evidence behind the selection."""

    shot: int
    configuration: Jt60saConfiguration
    evidence: str


@dataclass(frozen=True)
class Jt60saGeometryRegistry:
    """A JT-60SA geometry registry built over both operating phases.

    One registry spans the two phases, so both phase artifacts carry one shared
    registry digest while each carries its own physical digest.  The registry is
    a frozen expectation of the geometry a phase stands for; it is derived from
    the phase geometry content, never from the container bytes.  It exposes the
    five members the identity resolvers read from Nova's MAST registry:
    ``registry_digest``, ``dd_version``, ``configurations`` keyed by physical
    digest, :meth:`select` and :meth:`resolve_representation`.
    """

    schema: str
    machine: str
    dd_version: str
    configurations: Mapping[str, Jt60saConfiguration]
    representation_aliases: Mapping[str, str]
    ranges: tuple[Jt60saPhaseRange, ...]
    registry_digest: str

    def physical_digest(self, phase: str) -> str:
        """Return the physical digest the registry holds for ``phase``."""
        for row in self.ranges:
            if row.phase == phase:
                return row.physical_digest
        raise KeyError(f"the registry holds no phase {phase!r}")

    def select(self, shot: int) -> Jt60saSelection:
        """Select the one phase whose range holds ``shot``.

        Raises :class:`KeyError` when the shot lies outside both ranges; the
        phases' ranges are disjoint, so a shot resolves to at most one phase.
        """
        matches = [row for row in self.ranges if row.contains(int(shot))]
        if len(matches) != 1:
            raise KeyError(
                f"shot {shot} lies outside the JT-60SA phase ranges "
                f"({self.registry_digest[:12]})"
            )
        row = matches[0]
        return Jt60saSelection(
            shot=int(shot),
            configuration=self.configurations[row.physical_digest],
            evidence=row.evidence,
        )

    def resolve_representation(self, digest: str) -> Jt60saConfiguration:
        """Return the configuration a phase's representation alias names.

        Raises :class:`KeyError` for any digest no phase aliases, so a setup
        signature from another machine or another revision is refused rather
        than answered with the wrong phase's geometry.
        """
        try:
            physical = self.representation_aliases[digest]
        except KeyError as error:
            raise KeyError(
                f"unknown JT-60SA setup representation {digest!r} "
                f"(registry {self.registry_digest[:12]})"
            ) from error
        return self.configurations[physical]


def _points(points: Any) -> list[list[float]]:
    """Return a point structure's ``(r, phi, z)`` rows as plain floats."""
    r = np.asarray(points.r, dtype=float)
    phi = np.asarray(points.phi, dtype=float)
    z = np.asarray(points.z, dtype=float)
    return [
        [float(ri), float(pi), float(zi)] for ri, pi, zi in zip(r, phi, z, strict=True)
    ]


def _tf_geometry_payload(tf: Any) -> list[dict[str, Any]]:
    """Return the toroidal-field coil conductor paths as canonical geometry."""
    coils: list[dict[str, Any]] = []
    for coil in tf.coil:
        conductors: list[dict[str, Any]] = []
        for conductor in coil.conductor:
            elements = conductor.elements
            conductors.append(
                {
                    "centres": _points(elements.centres),
                    "end_points": _points(elements.end_points),
                    "start_points": _points(elements.start_points),
                }
            )
        coils.append({"name": str(coil.name), "conductors": conductors})
    return coils


@dataclass(frozen=True)
class _PhaseGeometry:
    """One phase's canonical geometry payload and the parts it was taken from.

    ``payload`` is the JSON-compatible snapshot hashed into the physical and
    registry digests; the remaining fields are the reader's own extracted
    objects, which the representation digest is computed over.  Both come from
    one extraction, so the payload and the alias cannot be taken from different
    reads of the store.
    """

    payload: Mapping[str, Any]
    b_probes: tuple[Any, ...]
    flux_loops: tuple[Any, ...]
    filaments: tuple[Any, ...]
    limiter_r: tuple[float, ...]
    limiter_z: tuple[float, ...]

    def representation_digest(self) -> str:
        """Return the setup representation digest for this phase's geometry."""
        return representation_digest(
            self.b_probes,
            self.flux_loops,
            self.filaments,
            self.limiter_r,
            self.limiter_z,
        )


def _read_phase_geometry(ids: Mapping[str, Any]) -> _PhaseGeometry:
    """Extract one phase's geometry payload and reader parts from its IDSs.

    The snapshot is taken through the same extractors the geometry reader uses,
    so it is the geometry a consumer of the artifact sees rather than the
    discretization the store happens to record.  Turns are carried as the
    description records them; a moved conductor changes the payload.  The
    filament list joins the active and passive filaments in the order the reader
    emits them, so the representation digest taken over it matches the digest a
    reader stamps from the authored artifact.
    """
    active, _active_sections, _active_drives, _ = read_artifact_pf_active(
        ids["pf_active"]
    )
    passive, _passive_sections, _st, _passive_drives, _ = read_artifact_pf_passive(
        ids["pf_passive"], 0
    )
    limiter_r, limiter_z, _ = read_artifact_limiter(ids["wall"])
    probes, loops, _ = read_artifact_magnetics(ids["magnetics"])
    filaments = tuple(active) + tuple(passive)
    payload = {
        "magnetics": {
            "b_probes": [[p.r, p.z, p.angle_deg, p.length] for p in probes],
            "flux_loops": [[f.r, f.z] for f in loops],
        },
        "pf_active": [
            [f.r, f.z, f.width, f.height, f.turns, f.circuit] for f in active
        ],
        "pf_passive": [
            [f.r, f.z, f.width, f.height, f.turns, f.circuit] for f in passive
        ],
        "tf": _tf_geometry_payload(ids["tf"]),
        "wall": {"limiter_r": list(limiter_r), "limiter_z": list(limiter_z)},
    }
    return _PhaseGeometry(
        payload=payload,
        b_probes=tuple(probes),
        flux_loops=tuple(loops),
        filaments=filaments,
        limiter_r=tuple(limiter_r),
        limiter_z=tuple(limiter_z),
    )


def _canonical_digest(payload: Any, *, length: int = 64) -> str:
    """Return a deterministic SHA-256 digest of JSON-compatible content."""
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()[:length]


def _physical_digest(geometry: Mapping[str, Any]) -> str:
    """Return the physical identity of one phase from its geometry content."""
    return _canonical_digest(geometry, length=16)


def build_jt60sa_registry(
    geometries: Mapping[str, Mapping[str, Any]],
    shot_ranges: Mapping[str, tuple[int, int]],
    representations: Mapping[str, str],
    authoring_gaps: Mapping[str, Sequence[str]],
) -> Jt60saGeometryRegistry:
    """Build one registry over both phases' geometry, ranges and aliases.

    ``representations`` is each phase's setup representation alias -- the
    :func:`~imas_ambix.gs.artifact_geometry.representation_digest` a reader
    stamps from that phase's artifact -- and ``authoring_gaps`` is each phase
    artifact's unresolved gaps.  Both are hashed into the registry digest, so a
    registry records the aliases and gaps it was built from rather than
    accepting either later.
    """
    digests = {phase: _physical_digest(geometries[phase]) for phase in PHASES}
    configurations = {
        digests[phase]: Jt60saConfiguration(
            physical_digest=digests[phase],
            geometry=geometries[phase],
            authoring_gaps=tuple(authoring_gaps[phase]),
        )
        for phase in PHASES
    }
    ranges = tuple(
        Jt60saPhaseRange(
            phase=phase,
            first_shot=int(shot_ranges[phase][0]),
            last_shot=int(shot_ranges[phase][1]),
            evidence=PHASE_SHOT_EVIDENCE[phase],
            physical_digest=digests[phase],
        )
        for phase in PHASES
    )
    representation_aliases = {
        representations[phase]: digests[phase] for phase in PHASES
    }
    if len(set(representation_aliases)) != len(PHASES):
        raise ValueError(
            "the two phases carry the same setup representation alias "
            f"{sorted(representations.values())}; each phase must alias "
            "distinctly"
        )
    for alias, digest in representation_aliases.items():
        if digest not in configurations:
            raise ValueError(
                f"representation alias {alias!r} names unknown digest {digest!r}"
            )
    payload = {
        "configurations": {
            digest: {
                "authoring_gaps": list(configuration.authoring_gaps),
                "geometry": configuration.geometry,
                "physical_digest": configuration.physical_digest,
            }
            for digest, configuration in configurations.items()
        },
        "dd_version": DD_VERSION,
        "machine": MACHINE,
        "ranges": [
            {
                "evidence": row.evidence,
                "first_shot": row.first_shot,
                "last_shot": row.last_shot,
                "phase": row.phase,
                "physical_digest": row.physical_digest,
            }
            for row in ranges
        ],
        "representation_aliases": representation_aliases,
        "schema": REGISTRY_SCHEMA,
    }
    return Jt60saGeometryRegistry(
        schema=REGISTRY_SCHEMA,
        machine=MACHINE,
        dd_version=DD_VERSION,
        configurations=configurations,
        representation_aliases=representation_aliases,
        ranges=ranges,
        registry_digest=_canonical_digest(payload),
    )


def _read_phase_ids(store: Path) -> dict[str, Any]:
    """Read a phase store's five IDSs from their ``{ids}.nc`` containers."""
    ids: dict[str, Any] = {}
    for name in IDS_NAMES:
        with imas.DBEntry(
            str(store / f"{name}.nc"), "r", dd_version=DD_VERSION
        ) as entry:
            ids[name] = entry.get(name, autoconvert=False)
    return ids


@dataclass(frozen=True)
class _PhaseMapRow:
    """One phase's catalogue row: its shot range and its drive-map identities.

    It carries the drive topology's connections and the acquisition declaration
    that names their current channels, so the drive authoring reads one loaded
    structure rather than re-loading the catalogue per concern.
    """

    phase: str
    first_shot: int
    last_shot: int
    drive_topology: str
    connections: tuple[Any, ...]
    acquisition: Any


def _machine_map_payload(
    machine_map_path: Mapping[str, Any] | Path | str,
) -> Mapping[str, Any]:
    """Return the catalogue, reading it from disk only when given a path."""
    if isinstance(machine_map_path, (Path, str)):
        return json.loads(Path(machine_map_path).read_text())
    return machine_map_path


def _phase_map_rows(
    machine_map_path: Mapping[str, Any] | Path | str,
) -> dict[str, _PhaseMapRow]:
    """Read the catalogue once and serve each phase's shot range and drive map.

    One load replaces the separate loads the shot ranges and drive topologies
    each made.  Each phase row carries the drive topology it names and the
    acquisition declaration that topology's ``current_channel_declaration``
    names, so the channel a drive carries and the measured current it receives
    are read from one place.
    """
    from imas_alambic.machine_map import (  # noqa: PLC0415
        AcquisitionDeclaration,
        DriveTopology,
    )

    payload = _machine_map_payload(machine_map_path)
    topologies = {
        str(row["name"]): DriveTopology.from_dict(
            row, f"drive_topologies[{row['name']}]"
        )
        for row in payload["drive_topologies"]
    }
    declarations = {
        str(row["name"]): AcquisitionDeclaration.from_dict(
            row, f"acquisition_declarations[{row['name']}]"
        )
        for row in payload["acquisition_declarations"]
    }
    rows: dict[str, _PhaseMapRow] = {}
    for row in payload["maps"]:
        phase = str(row["name"])
        if phase not in PHASES:
            continue
        topology = topologies[str(row["drive_topology"])]
        rows[phase] = _PhaseMapRow(
            phase=phase,
            first_shot=int(row["first_shot"]),
            last_shot=int(row["last_shot"]),
            drive_topology=topology.name,
            connections=topology.connections,
            acquisition=declarations[topology.current_channel_declaration],
        )
    missing = [phase for phase in PHASES if phase not in rows]
    if missing:
        raise ValueError(f"machine map names no row for {missing}")
    return rows


def _phase_completeness(
    phase: str,
    drive_topology: str,
    *,
    channel_drive: Sequence[Any] = (),
) -> tuple[bool, tuple[str, ...]]:
    """Derive one phase's completeness and the gaps it has not yet closed.

    The producer authors no geometry field-evidence ledger, so that absence is
    always named and the phase stays incomplete.  The channel drive map is
    authored from the phase's drive topology, so its gap is named only when no
    drive is supplied.  The gap text is trimmed and canonically ordered, so
    nova's manifest validator accepts it unchanged.
    """
    gaps: list[str] = [NO_FIELD_EVIDENCE_GAP]
    if not channel_drive:
        gaps.append(
            DRIVE_TOPOLOGY_GAP.format(phase=phase, drive_topology=drive_topology)
        )
    return (not gaps, tuple(sorted(gaps)))


def _element_index(geometry_element_identifier: str) -> int:
    """Return the coil-relative element index a topology geometry name holds.

    The topology names a coil's elements ``<stem>_<ordinal>`` with a one-based
    ordinal, so the store element at index ``i`` is ``<stem>_<i+1>``.
    """
    segment = geometry_element_identifier.rsplit("/", 1)[-1]
    match = re.fullmatch(r".+_(\d+)", segment)
    if match is None:
        raise ValueError(
            "drive topology geometry name "
            f"{geometry_element_identifier!r} carries no element ordinal"
        )
    return int(match.group(1)) - 1


def _connections_by_coil(connections: Sequence[Any]) -> dict[str, tuple[Any, ...]]:
    """Group a topology's connections by the coil their geometry names stem to."""
    grouped: dict[str, list[Any]] = {}
    for connection in connections:
        stem = re.sub(
            r"_\d+$", "", connection.geometry_element_identifier.rsplit("/", 1)[-1]
        )
        grouped.setdefault(stem, []).append(connection)
    return {stem: tuple(rows) for stem, rows in grouped.items()}


def _coil_drive_weight(coil: str, connections: Sequence[Any]) -> float:
    """Return one coil's total ampere-turns per ampere.

    Each connection declares ``turns * current_weight * direction``.  The coil's
    drive weight is their sum -- the ampere turns one ampere of the channel
    drives through the whole coil, matching how
    :func:`imas_ambix.data.geometry_adapter._materialise_circuit_drives` sums a
    circuit.  A reader divides that total across the elements the drive names in
    proportion to section area, so a per-element value here would cut the coil's
    current by its element count.

    A coil's elements are one winding when their per-element weights agree to
    within the two-decimal rounding the catalogue carries; a coil whose elements
    disagree by more is refused by name, because its connections do not describe
    a symmetric winding and their sum would be unverifiable.
    """
    weights = [
        float(connection.turns)
        * float(connection.current_weight)
        * float(connection.direction)
        for connection in connections
    ]
    low, high = min(weights), max(weights)
    if high - low > _WEIGHT_AGREEMENT:
        raise ValueError(
            f"pf_active coil {coil!r} elements disagree on the ampere-turns per "
            f"ampere: {sorted(set(weights))}"
        )
    return sum(weights)


def _phase_channel_drives(
    row: _PhaseMapRow, pf_active: Any
) -> tuple[list[Any], list[Any]]:
    """Author one nova ChannelDrive and its evidence record per pf_active coil.

    The channel is the current channel the coil's geometry element names stem to
    under the rule ``current_channel_from_conductors`` applies, accepted only
    when the phase's acquisition declaration lists it.  The elements are the
    coil's own element indices, and the weight is the coil's total ampere-turns
    per ampere, which a reader divides across those elements by section area.
    Each drive's path points at an evidence record citing the drive topology, so
    the manifest carries the provenance the weight needs.
    """
    from nova.imas.machine_drive import (  # noqa: PLC0415
        SECTION_AREA,
        SINGLE_ELEMENT,
        ChannelDrive,
    )
    from nova.imas.machine_evidence import (  # noqa: PLC0415
        EvidenceRecord,
        FieldEvidence,
        SourceReference,
    )

    grouped = _connections_by_coil(row.connections)
    drives: list[Any] = []
    records: list[Any] = []
    for coil in pf_active.coil:
        name = str(coil.name)
        connections = grouped.get(name)
        if connections is None:
            raise ValueError(
                f"drive topology {row.drive_topology!r} names no circuit for "
                f"pf_active coil {name!r}"
            )
        channel = current_channel_from_conductors(
            tuple(item.geometry_element_identifier for item in connections),
            row.acquisition,
        )
        if channel is None:
            raise ValueError(
                f"pf_active coil {name!r} stems to no current channel the "
                f"acquisition declaration {row.acquisition.name!r} lists"
            )
        elements = tuple(
            sorted(
                _element_index(item.geometry_element_identifier)
                for item in connections
            )
        )
        if elements != tuple(range(len(coil.element))):
            raise ValueError(
                f"drive topology {row.drive_topology!r} reaches elements "
                f"{elements} of pf_active coil {name!r}, not its "
                f"{len(coil.element)} elements"
            )
        circuits = {str(item.circuit_identifier) for item in connections}
        if len(circuits) != 1:
            raise ValueError(
                f"pf_active coil {name!r} spans circuits {sorted(circuits)}, "
                "not one"
            )
        weight = _coil_drive_weight(name, connections)
        path = f"pf_active/coil({name})/current({channel})"
        drives.append(
            ChannelDrive(
                channel=channel,
                container="pf_active",
                conductor=name,
                elements=elements,
                circuit=circuits.pop(),
                ampere_turns_per_ampere=weight,
                distribution=SINGLE_ELEMENT if len(elements) == 1 else SECTION_AREA,
                evidence=FieldEvidence.PUBLISHED,
                path=path,
            )
        )
        records.append(
            EvidenceRecord(
                path=path,
                evidence=FieldEvidence.PUBLISHED,
                first_shot=row.first_shot,
                last_shot=row.last_shot,
                statement=(
                    f"one ampere of {channel} drives {weight:.6g} ampere turns "
                    f"through coil {name}, the {row.drive_topology} drive "
                    "topology's total over its connections"
                ),
                source=SourceReference(
                    title="JT-60SA machine map drive topology",
                    url=(
                        f"https://ghcr.io/{GHCR_OWNER}/"
                        f"{package_for_machine(MACHINE)}"
                    ),
                    locator=(
                        f"drive_topologies[{row.drive_topology}].connections "
                        "in machine_map.json"
                    ),
                    machine=MACHINE,
                    text_verified=True,
                ),
            )
        )
    return drives, records


def _write_phase_ids(ids: Mapping[str, Any], directory: Path) -> None:
    """Write a phase's five IDSs into an IMAS HDF5 entry in ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    with imas.DBEntry(
        f"imas:hdf5?path={directory}", "x", dd_version=DD_VERSION
    ) as entry:
        for name in IDS_NAMES:
            entry.put(ids[name])


def _registry_inputs(
    ids_by_phase: Mapping[str, Mapping[str, Any]],
    rows: Mapping[str, _PhaseMapRow],
) -> tuple[
    dict[str, _PhaseGeometry],
    dict[str, tuple[list[Any], list[Any]]],
    dict[str, tuple[str, ...]],
    dict[str, str],
    dict[str, tuple[int, int]],
]:
    """Derive, from the phase stores, everything the registry is built over.

    Returns the per-phase geometry (with the reader parts its representation
    digest is taken from), the authored channel drives and their evidence
    records, the authoring gaps, the representation aliases and the shot ranges.
    The gaps follow from the drives, so the drive authoring happens here once
    and both the registry's payload and the artifacts read the same result.
    """
    geometries = {
        phase: _read_phase_geometry(ids_by_phase[phase]) for phase in PHASES
    }
    drives_by_phase = {
        phase: _phase_channel_drives(rows[phase], ids_by_phase[phase]["pf_active"])
        for phase in PHASES
    }
    gaps_by_phase = {
        phase: _phase_completeness(
            phase, rows[phase].drive_topology, channel_drive=drives_by_phase[phase][0]
        )[1]
        for phase in PHASES
    }
    aliases = {phase: geometries[phase].representation_digest() for phase in PHASES}
    shot_ranges = {
        phase: (rows[phase].first_shot, rows[phase].last_shot) for phase in PHASES
    }
    return geometries, drives_by_phase, gaps_by_phase, aliases, shot_ranges


def build_jt60sa_registry_from_stores(
    *,
    description_root: str | Path = DEFAULT_DESCRIPTION_ROOT,
    machine_map_path: str | Path = DEFAULT_MACHINE_MAP,
) -> Jt60saGeometryRegistry:
    """Return the JT-60SA geometry registry built from the phase stores.

    This is the entry a machine-keyed identity registry calls for JT-60SA: it
    reads the two phase stores and the catalogue, derives the physical digests,
    the representation aliases and the authoring gaps exactly as artifact
    authoring does, and returns the registry without materializing an artifact.
    Nothing is cached and no artifact cache is required, so a resolver can read
    an identity the moment the stores are present.
    """
    root = Path(description_root)
    rows = _phase_map_rows(machine_map_path)
    ids_by_phase = {phase: _read_phase_ids(root / phase) for phase in PHASES}
    geometries, _drives, gaps, aliases, shot_ranges = _registry_inputs(
        ids_by_phase, rows
    )
    return build_jt60sa_registry(
        {phase: geometries[phase].payload for phase in PHASES},
        shot_ranges,
        aliases,
        gaps,
    )


def author_jt60sa_machine_artifacts(
    cache_directory: str | Path,
    *,
    description_root: str | Path = DEFAULT_DESCRIPTION_ROOT,
    machine_map_path: str | Path = DEFAULT_MACHINE_MAP,
) -> dict[str, Any]:
    """Author, verify and cache one JT-60SA artifact for each operating phase.

    Returns a mapping of phase name to the resolved
    :class:`~nova.imas.machine_artifact.VerifiedMachineArtifact`, each verified
    against its own expected physical and registry digest.
    """
    from nova.imas.machine_artifact import (  # noqa: PLC0415
        ArtifactShotRange,
        create_machine_artifact_manifest,
        materialize_machine_artifact,
        resolve_machine_artifact,
    )

    root = Path(description_root)
    rows = _phase_map_rows(machine_map_path)
    ids_by_phase = {phase: _read_phase_ids(root / phase) for phase in PHASES}
    geometries, drives_by_phase, gaps_by_phase, aliases, shot_ranges = (
        _registry_inputs(ids_by_phase, rows)
    )
    registry = build_jt60sa_registry(
        {phase: geometries[phase].payload for phase in PHASES},
        shot_ranges,
        aliases,
        gaps_by_phase,
    )

    artifacts: dict[str, Any] = {}
    for phase in PHASES:
        physical_digest = registry.physical_digest(phase)
        drives, drive_evidence = drives_by_phase[phase]
        gaps = gaps_by_phase[phase]
        complete = not gaps
        phase_ranges = (
            ArtifactShotRange(
                first_shot=shot_ranges[phase][0],
                last_shot=shot_ranges[phase][1],
                physical_digest=physical_digest,
                evidence=PHASE_SHOT_EVIDENCE[phase],
            ),
        )
        with tempfile.TemporaryDirectory() as work:
            source = Path(work) / "machine_description"
            _write_phase_ids(ids_by_phase[phase], source)
            manifest = create_machine_artifact_manifest(
                source,
                machine=MACHINE,
                dd_version=DD_VERSION,
                registry_digest=registry.registry_digest,
                physical_digest=physical_digest,
                shot_ranges=phase_ranges,
                complete=complete,
                unresolved_gaps=gaps,
                field_evidence=drive_evidence,
                channel_drive=drives,
            )
            materialize_machine_artifact(source, cache_directory, manifest)
        artifacts[phase] = resolve_machine_artifact(
            cache_directory,
            manifest.digest,
            expected_physical_digest=physical_digest,
            expected_registry_digest=registry.registry_digest,
            allow_incomplete=not manifest.complete,
        )
    return artifacts


__all__ = [
    "Jt60saConfiguration",
    "Jt60saGeometryRegistry",
    "Jt60saPhaseRange",
    "Jt60saSelection",
    "author_jt60sa_machine_artifacts",
    "build_jt60sa_registry",
    "build_jt60sa_registry_from_stores",
]
