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

#: The relative spread two elements of one coil may show on the per-element
#: ampere-turns and still be one winding.  The catalogue declares turns to two
#: decimals, so a uniform winding's last declared digit may differ between its
#: elements; a larger spread is a genuine disagreement and is refused by name.
_WEIGHT_AGREEMENT = 1e-2

#: The evidence the producer does not yet author: the geometry field ledger.
#: The absence is named on every phase artifact, so each stays incomplete.
NO_FIELD_EVIDENCE_GAP = "no field evidence ledger is authored"
DRIVE_TOPOLOGY_GAP = (
    "the channel drive map is not authored from the {phase} drive topology "
    "({drive_topology}) in machine_map.json"
)


@dataclass(frozen=True)
class Jt60saGeometryRegistry:
    """A JT-60SA geometry registry built over both operating phases.

    One registry spans the two phases, so both phase artifacts carry one shared
    registry digest while each carries its own physical digest.  The registry is
    a frozen expectation of the geometry a phase stands for; it is derived from
    the phase geometry content, never from the container bytes.
    """

    schema: str
    machine: str
    dd_version: str
    configurations: Mapping[str, Mapping[str, Any]]
    ranges: tuple[Mapping[str, Any], ...]
    registry_digest: str

    def physical_digest(self, phase: str) -> str:
        """Return the physical digest the registry holds for ``phase``."""
        return str(self.configurations[phase]["physical_digest"])


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


def _phase_geometry_payload(ids: Mapping[str, Any]) -> dict[str, Any]:
    """Return the geometry content of one phase's IDSs as canonical data.

    The snapshot is taken through the same extractors the geometry reader uses,
    so it is the geometry a consumer of the artifact sees rather than the
    discretization the store happens to record.  Turns are carried as the
    description records them; a moved conductor changes the payload.
    """
    active, _active_sections, _active_drives, _ = read_artifact_pf_active(
        ids["pf_active"]
    )
    passive, _passive_sections, _st, _passive_drives, _ = read_artifact_pf_passive(
        ids["pf_passive"], 0
    )
    limiter_r, limiter_z, _ = read_artifact_limiter(ids["wall"])
    probes, loops, _ = read_artifact_magnetics(ids["magnetics"])
    return {
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
) -> Jt60saGeometryRegistry:
    """Build one registry over both phases' geometry content and shot ranges."""
    configurations = {
        phase: {
            "physical_digest": _physical_digest(geometries[phase]),
            "geometry": geometries[phase],
        }
        for phase in PHASES
    }
    ranges = tuple(
        {
            "evidence": PHASE_SHOT_EVIDENCE[phase],
            "first_shot": int(shot_ranges[phase][0]),
            "last_shot": int(shot_ranges[phase][1]),
            "phase": phase,
            "physical_digest": configurations[phase]["physical_digest"],
        }
        for phase in PHASES
    )
    payload = {
        "configurations": configurations,
        "dd_version": DD_VERSION,
        "machine": MACHINE,
        "ranges": list(ranges),
        "schema": REGISTRY_SCHEMA,
    }
    return Jt60saGeometryRegistry(
        schema=REGISTRY_SCHEMA,
        machine=MACHINE,
        dd_version=DD_VERSION,
        configurations=configurations,
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
    if high - low > _WEIGHT_AGREEMENT * abs(high):
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
    shot_ranges = {
        phase: (rows[phase].first_shot, rows[phase].last_shot) for phase in PHASES
    }

    ids_by_phase = {phase: _read_phase_ids(root / phase) for phase in PHASES}
    geometries = {
        phase: _phase_geometry_payload(ids_by_phase[phase]) for phase in PHASES
    }
    registry = build_jt60sa_registry(geometries, shot_ranges)

    artifacts: dict[str, Any] = {}
    for phase in PHASES:
        physical_digest = registry.physical_digest(phase)
        drives, drive_evidence = _phase_channel_drives(
            rows[phase], ids_by_phase[phase]["pf_active"]
        )
        complete, gaps = _phase_completeness(
            phase, rows[phase].drive_topology, channel_drive=drives
        )
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


__all__ = ["author_jt60sa_machine_artifacts"]
