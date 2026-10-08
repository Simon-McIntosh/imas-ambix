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
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import imas
import numpy as np

from imas_ambix.gs.artifact_geometry import (
    read_artifact_limiter,
    read_artifact_magnetics,
    read_artifact_pf_active,
    read_artifact_pf_passive,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

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

#: The in-checkout phase description stores, addressed absolutely because the
#: maps tree reaches a worktree through a symlink to the main checkout.
DEFAULT_DESCRIPTION_ROOT = Path(
    "/home/ITER/mcintos/Code/imas-ambix/maps/jt-60sa/machine_description"
)

#: The packaged catalogue carrying each phase's first and last shot.
DEFAULT_MACHINE_MAP = Path(
    "/home/ITER/mcintos/Code/imas-ambix/maps/jt-60sa/machine_map.json"
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
            "evidence": "observed",
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


def _phase_shot_ranges(
    machine_map_path: Mapping[str, Any] | Path | str,
) -> dict[str, tuple[int, int]]:
    """Return each phase's closed shot range from the packaged catalogue."""
    if isinstance(machine_map_path, (Path, str)):
        payload = json.loads(Path(machine_map_path).read_text())
    else:
        payload = machine_map_path
    ranges = {
        str(row["name"]): (int(row["first_shot"]), int(row["last_shot"]))
        for row in payload["maps"]
    }
    missing = [phase for phase in PHASES if phase not in ranges]
    if missing:
        raise ValueError(f"machine map names no shot range for {missing}")
    return ranges


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
    shot_ranges = _phase_shot_ranges(machine_map_path)

    ids_by_phase = {phase: _read_phase_ids(root / phase) for phase in PHASES}
    geometries = {
        phase: _phase_geometry_payload(ids_by_phase[phase]) for phase in PHASES
    }
    registry = build_jt60sa_registry(geometries, shot_ranges)

    artifacts: dict[str, Any] = {}
    for phase in PHASES:
        physical_digest = registry.physical_digest(phase)
        phase_ranges = (
            ArtifactShotRange(
                first_shot=shot_ranges[phase][0],
                last_shot=shot_ranges[phase][1],
                physical_digest=physical_digest,
                evidence="observed",
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
                complete=True,
                unresolved_gaps=(),
            )
            materialize_machine_artifact(source, cache_directory, manifest)
        artifacts[phase] = resolve_machine_artifact(
            cache_directory,
            manifest.digest,
            expected_physical_digest=physical_digest,
            expected_registry_digest=registry.registry_digest,
        )
    return artifacts


__all__ = ["author_jt60sa_machine_artifacts"]
