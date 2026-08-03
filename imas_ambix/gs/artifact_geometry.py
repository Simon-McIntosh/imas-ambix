"""Build a :class:`GeometryTable` from a content-addressed machine-description artifact.

Third implementation of
:class:`~imas_ambix.gs.geometry.MachineGeometryReader`.  The first
(:class:`~imas_ambix.gs.geometry.MastZarrGeometryReader`) reads MAST geometry out
of the per-shot FAIR-MAST ``efm`` arrays; the second
(:class:`~imas_ambix.gs.imas_geometry.ImasGeometryReader`) reads loose IMAS
static IDSs by path.  This one reads a *published artifact*: a verified,
content-addressed directory of DD 4.1.1 IDSs whose manifest carries the physical
identity of the machine configuration, the registry it was authored from, and a
per-field evidence ledger saying which of its numbers are sourced.

Why that difference matters.  The ``efm`` arrays are one campaign's
discretization of the machine and carry no statement about where any number came
from, so a table built from them is trusted implicitly.  The artifact states its
own provenance: every table this reader returns records the artifact's semantic
identity, its physical and registry digests, the dictionary version it is pinned
to, and the registry's evidence state for the shot it was selected for.  A
consumer can therefore tell what it is holding instead of assuming.

Unresolved active turns (the load-bearing gap)
----------------------------------------------
In the artifact revision this reader was built against,
``pf_active/coil/element/turns_with_sign`` is unsourced -- the manifest reports
it as a forward-model blocker, and the stored values are the IMAS EMPTY_FLOAT
sentinel.  Turns scale every PF column of the Green's operator, so a fabricated
count would silently rescale the whole vacuum field.

This reader therefore carries turns as **NaN**, never a guess and never a 1.0
default.  NaN is chosen deliberately over a plausible number: it propagates, so
an operator built from these filaments produces NaN rather than a quietly wrong
field, and it cannot be mistaken for a measurement.  A consumer that must not
proceed without turns calls :func:`require_resolved_turns` and gets a named
failure listing the blocking circuits before any matrix is assembled.
Everything geometric -- outlines, sections, sensor positions and orientations,
the limiter contour -- is faithful and complete.

Conductor sections
------------------
``pf_active`` and ``pf_passive`` elements here are outline polygons, not the
axis-aligned ``fcoil`` rectangles the MAST reader produces.  Each element
becomes a :class:`~imas_ambix.gs.geometry.PFFilament` at its outline's bounding
box (the representation every consumer understands) *plus* a
:class:`~imas_ambix.gs.geometry.PolygonSection` carrying the true vertices, so a
consumer wired for polygon sections evaluates the real cross-section and one
that is not still gets a co-located conductor of the right extent.

Electrical grouping
-------------------
Each ``pf_active`` coil is one circuit: its elements are the same winding and
carry the same current.  Each ``pf_passive`` **element** is its own circuit,
because grouping a loop's elements would impose a series constraint the artifact
does not source (its manifest records passive electrical topology as unresolved)
and an axisymmetric passive element carries an independently induced current.
Both choices are recorded in the table's provenance flags.

Access-layer rules (binding)
----------------------------
Every IDS is read through imas-python at the artifact's own pinned dictionary
version -- read from the manifest, never assumed and never the installed default
-- and ``DBEntry`` opens in its constructor.  No h5py, no version guessing: the
manifest is the authority on the pin, which is the whole point of reading a
verified artifact rather than a loose directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from imas_ambix.gs.geometry import (
    BProbe,
    FluxLoop,
    GeometryTable,
    PassiveStructure,
    PFFilament,
    PolygonSection,
    SensorMapping,
    SetupSignature,
    map_amb_sensors,
    round_geometry_hash,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

#: IMAS EMPTY_FLOAT is -9e40; anything this large (or non-finite) is unfilled.
_EMPTY_FLOAT_FLOOR = 1.0e38

#: The value an unresolved turn count takes in :attr:`PFFilament.turns`.
#: NaN propagates through every downstream product, so an operator assembled
#: over an unresolved winding is visibly wrong instead of quietly wrong.
UNRESOLVED_TURNS = float("nan")


class UnresolvedTurnsError(ValueError):
    """Raised when a consumer needing winding turns is handed a table without them."""


def _is_empty(x: float) -> bool:
    """Return whether ``x`` is the IMAS EMPTY_FLOAT sentinel or otherwise unfilled."""
    return (not np.isfinite(x)) or abs(x) > _EMPTY_FLOAT_FLOOR


def _outline_vertices(geometry: Any) -> np.ndarray | None:
    """Return an element's ``(n, 2)`` outline, or ``None`` if it carries none."""
    outline = geometry.outline
    r = np.asarray(outline.r, dtype=np.float64)
    z = np.asarray(outline.z, dtype=np.float64)
    if (
        r.size < 3
        or r.size != z.size
        or not (np.isfinite(r).all() and np.isfinite(z).all())
    ):
        return None
    # a repeated closing vertex is a valid DD outline but PolygonSection wants
    # the corner list without it
    if abs(r[0] - r[-1]) < 1e-12 and abs(z[0] - z[-1]) < 1e-12:
        r, z = r[:-1], z[:-1]
    if r.size < 3:
        return None
    return np.column_stack([r, z])


def _bounding_box(vertices: np.ndarray) -> tuple[float, float, float, float]:
    """Return ``(r_centre, z_centre, width, height)`` of an outline's extent."""
    r, z = vertices[:, 0], vertices[:, 1]
    r_min, r_max = float(r.min()), float(r.max())
    z_min, z_max = float(z.min()), float(z.max())
    return (
        0.5 * (r_min + r_max),
        0.5 * (z_min + z_max),
        r_max - r_min,
        z_max - z_min,
    )


def _rectangle_vertices(geometry: Any) -> np.ndarray | None:
    """Return the ``(4, 2)`` corners of a filled ``rectangle`` element, else None."""
    r = float(geometry.rectangle.r)
    if _is_empty(r):
        return None
    z = float(geometry.rectangle.z)
    half_width = 0.5 * abs(float(geometry.rectangle.width))
    half_height = 0.5 * abs(float(geometry.rectangle.height))
    return np.array(
        [
            (r - half_width, z - half_height),
            (r + half_width, z - half_height),
            (r + half_width, z + half_height),
            (r - half_width, z + half_height),
        ]
    )


def _element_vertices(geometry: Any) -> np.ndarray | None:
    """Return an element's cross-section corners from whichever shape is filled."""
    vertices = _outline_vertices(geometry)
    if vertices is not None:
        return vertices
    return _rectangle_vertices(geometry)


# --- pf_active --------------------------------------------------------------


def read_artifact_pf_active(
    pf_active: Any,
) -> tuple[list[PFFilament], list[PolygonSection], list[str]]:
    """One filament + one polygon section per ``(coil, element)``; one circuit per coil.

    Turns are carried as :data:`UNRESOLVED_TURNS` wherever the artifact leaves
    ``turns_with_sign`` unfilled, with the coil named in a provenance flag.  A
    filled value is used as-is, sign included.
    """
    filaments: list[PFFilament] = []
    sections: list[PolygonSection] = []
    flags: list[str] = []
    unresolved: list[str] = []
    for circuit in range(len(pf_active.coil)):
        coil = pf_active.coil[circuit]
        name = str(coil.name)
        for index in range(len(coil.element)):
            element = coil.element[index]
            turns = float(element.turns_with_sign)
            if _is_empty(turns):
                turns = UNRESOLVED_TURNS
                unresolved.append(f"{name}[{index}]")
            vertices = _element_vertices(element.geometry)
            if vertices is None:
                flags.append(
                    f"pf_active coil {name!r} element {index}: no usable outline or "
                    f"rectangle (geometry_type="
                    f"{int(element.geometry.geometry_type)}) -- dropped"
                )
                continue
            r, z, width, height = _bounding_box(vertices)
            filaments.append(
                PFFilament(
                    r=r,
                    z=z,
                    turns=turns,
                    width=width,
                    height=height,
                    circuit=circuit,
                    xmult=1.0,
                )
            )
            sections.append(
                PolygonSection(circuit=circuit, vertices=vertices, xmult=1.0, name=name)
            )
    if unresolved:
        flags.append(
            "pf_active/coil/element/turns_with_sign is unresolved in this artifact "
            f"revision for {len(unresolved)} element(s) ({', '.join(unresolved)}); "
            f"turns carried as NaN -- see {__name__}.require_resolved_turns"
        )
    flags.append(
        "pf_active circuits are one per coil (a coil's elements share its current)"
    )
    return filaments, sections, flags


# --- pf_passive -------------------------------------------------------------


def read_artifact_pf_passive(
    pf_passive: Any, first_circuit: int
) -> tuple[list[PFFilament], list[PolygonSection], list[PassiveStructure], list[str]]:
    """One filament, section and structure entry per ``pf_passive`` element.

    Each element is its own circuit: the artifact does not source passive
    electrical topology, and an axisymmetric passive element carries an
    independently induced current, so grouping a loop's elements would impose a
    series constraint nothing backs.  Passive turns are read as filled (they
    describe a conductor's own multiplicity, not an unsourced winding).
    """
    filaments: list[PFFilament] = []
    sections: list[PolygonSection] = []
    structures: list[PassiveStructure] = []
    flags: list[str] = []
    circuit = first_circuit
    for loop_index in range(len(pf_passive.loop)):
        loop = pf_passive.loop[loop_index]
        name = str(loop.name)
        for index in range(len(loop.element)):
            element = loop.element[index]
            vertices = _element_vertices(element.geometry)
            if vertices is None:
                flags.append(
                    f"pf_passive loop {name!r} element {index}: no usable outline "
                    "or rectangle -- dropped"
                )
                continue
            turns = float(element.turns_with_sign)
            if _is_empty(turns):
                turns = 1.0
                flags.append(
                    f"pf_passive loop {name!r} element {index}: turns_with_sign "
                    "unfilled -> 1 (a passive element is a single conductor)"
                )
            r, z, width, height = _bounding_box(vertices)
            element_name = name if len(loop.element) == 1 else f"{name}_{index}"
            filaments.append(
                PFFilament(
                    r=r,
                    z=z,
                    turns=turns,
                    width=width,
                    height=height,
                    circuit=circuit,
                    xmult=1.0,
                )
            )
            sections.append(
                PolygonSection(
                    circuit=circuit, vertices=vertices, xmult=1.0, name=element_name
                )
            )
            structures.append(
                PassiveStructure(name=element_name, r=r, z=z, obsolete=False)
            )
            circuit += 1
    flags.append(
        "pf_passive circuits are one per ELEMENT: the artifact does not source "
        "passive electrical topology, so no series grouping is imposed"
    )
    return filaments, sections, structures, flags


# --- magnetics --------------------------------------------------------------


def _orientation_diversity_flags(b_probes: Sequence[BProbe]) -> list[str]:
    """Flag a probe set that carries only one sensitive-axis orientation.

    A poloidal probe enters the operator as ``B_R cos(theta) + B_Z sin(theta)``,
    so ``theta`` decides which field component the row measures, not merely how
    it is scaled.  A tokamak's poloidal set normally mixes orientations -- an
    outboard array typically carries radial and vertical probes at the same
    positions -- and a set in which every probe shares one angle therefore
    cannot distinguish the two components at all.  That is a property of the
    numbers, not of any channel name, so it is checked here rather than assumed
    resolved: whether a single-orientation set is correct is a question for the
    source, and this flag is what makes a consumer ask it.
    """
    if len(b_probes) < 2:
        return []
    angles = np.array([p.angle_deg for p in b_probes], dtype=np.float64)
    distinct = np.unique(np.round(np.mod(angles, 180.0), 3))
    if distinct.size > 1:
        return []
    return [
        f"all {len(b_probes)} poloidal probes carry one sensitive-axis "
        f"orientation ({distinct[0]:.1f} deg): this probe set cannot separate "
        "B_R from B_Z, so any row whose true axis is the other component is "
        "measuring the wrong field"
    ]


def read_artifact_magnetics(
    magnetics: Any,
) -> tuple[list[BProbe], list[FluxLoop], list[str]]:
    """Poloidal probes 1:1; flux loops collapsed to their position centroid.

    A probe with no ``poloidal_angle`` is dropped rather than given an assumed
    orientation: the projection ``B_R cos(theta) + B_Z sin(theta)`` has no
    meaning without one, and any default would silently pick an axis.  This is
    also what separates the probes a poloidal forward model can use from the
    channels whose bank position the artifact records as unsourced -- the data
    makes that split, no name list does.

    A loop whose position points are not co-located is a partial (toroidally
    limited) loop that a single axisymmetric point cannot represent; those are
    flagged rather than silently averaged away.
    """
    flags: list[str] = []
    b_probes: list[BProbe] = []
    without_angle: list[str] = []
    for index in range(len(magnetics.b_field_pol_probe)):
        probe = magnetics.b_field_pol_probe[index]
        length = float(probe.length)
        if _is_empty(length):
            length = 0.0
        angle = float(probe.poloidal_angle)
        if _is_empty(angle):
            without_angle.append(str(probe.name))
            continue
        b_probes.append(
            BProbe(
                index=index,
                r=float(probe.position.r),
                z=float(probe.position.z),
                angle_deg=float(np.rad2deg(angle)),
                length=length,
            )
        )

    if without_angle:
        families = sorted({name.rsplit("_", 1)[0] for name in without_angle})
        flags.append(
            f"{len(without_angle)} of {len(magnetics.b_field_pol_probe)} poloidal "
            f"probes carry no poloidal_angle and are dropped (families "
            f"{', '.join(families)}); an orientation is never assumed"
        )
    flags += _orientation_diversity_flags(b_probes)

    flux_loops: list[FluxLoop] = []
    for index in range(len(magnetics.flux_loop)):
        loop = magnetics.flux_loop[index]
        n_position = len(loop.position)
        if n_position == 0:
            flags.append(
                f"magnetics.flux_loop[{index}] {str(loop.name)!r}: no position "
                "points -- dropped"
            )
            continue
        r = np.array([float(loop.position[k].r) for k in range(n_position)])
        z = np.array([float(loop.position[k].z) for k in range(n_position)])
        spread = float(max(r.max() - r.min(), z.max() - z.min()))
        if spread > 1e-9:
            flags.append(
                f"magnetics.flux_loop[{index}] {str(loop.name)!r}: {n_position} "
                f"position points spanning {spread:.4f} m represented by their "
                "centroid (non-axisymmetric approximation)"
            )
        flux_loops.append(FluxLoop(index=index, r=float(r.mean()), z=float(z.mean())))
    return b_probes, flux_loops, flags


def sensor_position_arrays(
    b_probes: Sequence[BProbe], flux_loops: Sequence[FluxLoop]
) -> dict[str, np.ndarray]:
    """Present the artifact's sensors in the array shape :func:`map_amb_sensors` reads.

    That mapper resolves an ``amb`` channel to a geometry index by nearest
    neighbour under a name-derived orientation restriction.  Handing it the
    artifact's positions instead of ``efm``'s is what lets one campaign's
    channel set be carried onto artifact geometry without duplicating the
    mapping rules, and the per-channel ``residual_m`` it returns is then the
    distance between the two geometry sources for that sensor.
    """
    return {
        "magpr_r": np.array([p.r for p in b_probes], dtype=np.float64),
        "magpr_z": np.array([p.z for p in b_probes], dtype=np.float64),
        "magpr_ang": np.array([p.angle_deg for p in b_probes], dtype=np.float64),
        "silop_r": np.array([f.r for f in flux_loops], dtype=np.float64),
        "silop_z": np.array([f.z for f in flux_loops], dtype=np.float64),
    }


# --- wall -------------------------------------------------------------------


def read_artifact_limiter(wall: Any) -> tuple[list[float], list[float], list[str]]:
    """Return the first ``description_2d`` limiter unit's contour."""
    flags: list[str] = []
    if not len(wall.description_2d):
        return [], [], ["wall.description_2d is empty -- no limiter contour"]
    limiter = wall.description_2d[0].limiter
    units = []
    for index in range(len(limiter.unit)):
        unit = limiter.unit[index]
        r = np.asarray(unit.outline.r, dtype=np.float64)
        z = np.asarray(unit.outline.z, dtype=np.float64)
        if r.size and z.size:
            units.append((r, z))
    if not units:
        return [], [], ["wall limiter carries no populated unit"]
    if len(units) > 1:
        flags.append(
            f"wall limiter carries {len(units)} units; only the first is used as "
            "the plasma-facing contour"
        )
    r, z = units[0]
    return r.tolist(), z.tolist(), flags


# --- unresolved-turn guard --------------------------------------------------


def unresolved_turn_circuits(table: GeometryTable) -> tuple[int, ...]:
    """Return the circuits whose winding turns the source could not resolve."""
    return tuple(
        sorted({f.circuit for f in table.pf_filaments if not np.isfinite(f.turns)})
    )


def require_resolved_turns(table: GeometryTable) -> None:
    """Raise unless every filament carries a real turn count.

    The check a consumer runs before assembling anything that scales with turns.
    Failing here names the blocking circuits; failing later means reading NaN out
    of a Green's matrix and working backwards to find out why.
    """
    circuits = unresolved_turn_circuits(table)
    if circuits:
        raise UnresolvedTurnsError(
            f"{len(circuits)} circuit(s) carry unresolved winding turns "
            f"{list(circuits)} in table {table.signature.key}; the source records "
            "pf_active/coil/element/turns_with_sign as unsourced, so no operator "
            "that scales with turns may be built from it"
        )


# --- the reader -------------------------------------------------------------


@dataclass(frozen=True)
class MachineArtifactGeometryReader:
    """Reads one machine configuration from a verified machine-description artifact.

    Implements :class:`~imas_ambix.gs.geometry.MachineGeometryReader`.

    ``cache_directory`` + ``digest`` address the artifact; resolution verifies
    the manifest against the cache address and every file against the manifest
    before a single IDS is opened.  ``expected_physical_digest`` and
    ``expected_registry_digest``, when given, are enforced during that
    resolution, so a reader can pin the exact machine configuration it was
    written against rather than trusting whatever the cache happens to hold.

    ``shot`` selects the registry evidence state recorded in the table's
    provenance and is what the table reports in :attr:`GeometryTable.shots`.

    ``amb_channels`` (a campaign's ``(channel, description)`` pairs) carries an
    existing sensor channel set onto this geometry through the shared
    :func:`~imas_ambix.gs.geometry.map_amb_sensors` rules.  Left empty, the
    table's sensor map is keyed by the artifact's own sensor names instead.

    ``amc_current_channels`` names the campaign's measured coil-current
    channels.  The artifact describes conductors, not an acquisition system, so
    it cannot supply them -- and without them
    :func:`~imas_ambix.gs.operator.classify_circuits` finds no driven circuit
    and the vacuum coil block comes out empty.  A caller building a forward
    operator therefore has to pass the campaign's list; a caller doing geometry
    work does not.
    """

    cache_directory: str | Path
    digest: str
    shot: int
    machine: str = "mast"
    amb_channels: tuple[tuple[str, str], ...] = ()
    amc_current_channels: tuple[str, ...] = ()
    expected_physical_digest: str = ""
    expected_registry_digest: str = ""
    allow_incomplete: bool = True

    def resolve(self) -> Any:
        """Return the verified artifact this reader addresses."""
        from nova.imas.mast_artifact import resolve_machine_artifact  # noqa: PLC0415

        return resolve_machine_artifact(
            self.cache_directory,
            self.digest,
            expected_physical_digest=self.expected_physical_digest or None,
            expected_registry_digest=self.expected_registry_digest or None,
            allow_incomplete=self.allow_incomplete,
        )

    def provenance(self, artifact: Any) -> dict[str, Any]:
        """Return the identity and evidence block recorded alongside the geometry."""
        manifest = artifact.manifest
        ledger = manifest.evidence
        block: dict[str, Any] = {
            "semantic_identity": manifest.semantic_identity(),
            "manifest_digest": artifact.digest,
            "physical_digest": manifest.physical_digest,
            "registry_digest": manifest.registry_digest,
            "dd_version": manifest.dd_version,
            "complete": bool(manifest.complete),
            "evidence_states": ledger.state_counts(),
            "forward_model_blockers": list(manifest.forward_model_blockers()),
            "unresolved_gaps": list(manifest.unresolved_gaps),
        }
        block["shot_evidence"] = self._shot_evidence()
        return block

    def _shot_evidence(self) -> str:
        """Return the registry evidence state for :attr:`shot`, or why it is absent."""
        from imas_ambix.gs.machine_identity import (  # noqa: PLC0415
            MachineIdentityError,
            identity_for_shot,
        )

        try:
            return identity_for_shot(int(self.shot)).evidence
        except MachineIdentityError as error:
            return f"unavailable ({error})"

    def read(self) -> GeometryTable:
        import imas  # noqa: PLC0415

        artifact = self.resolve()
        provenance = self.provenance(artifact)
        entry = imas.DBEntry(
            f"imas:hdf5?path={artifact.directory}",
            "r",
            dd_version=provenance["dd_version"],
        )
        try:
            pf_active = entry.get("pf_active")
            pf_passive = entry.get("pf_passive")
            wall = entry.get("wall")
            magnetics = entry.get("magnetics")
        finally:
            entry.close()

        active, active_sections, active_flags = read_artifact_pf_active(pf_active)
        n_active_circuit = 1 + max((f.circuit for f in active), default=-1)
        passive, passive_sections, structures, passive_flags = read_artifact_pf_passive(
            pf_passive, n_active_circuit
        )
        limiter_r, limiter_z, wall_flags = read_artifact_limiter(wall)
        b_probes, flux_loops, magnetics_flags = read_artifact_magnetics(magnetics)

        filaments = active + passive
        sections = active_sections + passive_sections
        sensor_map, unmatched = self._sensor_map(b_probes, flux_loops)

        signature = SetupSignature(
            n_bprobe=len(b_probes),
            n_fluxloop=len(flux_loops),
            n_pf_filament=len(filaments),
            n_limiter=len(limiter_r),
            digest=round_geometry_hash(
                [
                    np.array([p.r for p in b_probes]),
                    np.array([p.z for p in b_probes]),
                    np.array([p.angle_deg for p in b_probes]),
                    np.array([f.r for f in flux_loops]),
                    np.array([f.z for f in flux_loops]),
                    np.array([f.r for f in filaments]),
                    np.array([f.z for f in filaments]),
                    np.array([f.width for f in filaments]),
                    np.array([f.height for f in filaments]),
                    np.array(limiter_r),
                    np.array(limiter_z),
                ]
            ),
            machine=self.machine,
        )
        return GeometryTable(
            signature=signature,
            shots=[int(self.shot)],
            b_probes=b_probes,
            flux_loops=flux_loops,
            pf_filaments=filaments,
            limiter_r=limiter_r,
            limiter_z=limiter_z,
            sensor_map=sensor_map,
            passive_structures=structures,
            amc_current_channels=list(self.amc_current_channels),
            unmatched_amb=unmatched,
            provenance_flags=(
                _provenance_lines(provenance)
                + active_flags
                + passive_flags
                + wall_flags
                + magnetics_flags
            ),
            polygon_sections=sections,
        )

    def _sensor_map(
        self, b_probes: Sequence[BProbe], flux_loops: Sequence[FluxLoop]
    ) -> tuple[list[SensorMapping], list[str]]:
        """Key the sensor map by a carried channel set, or by artifact-native names."""
        if self.amb_channels:
            return map_amb_sensors(
                sensor_position_arrays(b_probes, flux_loops),
                [(channel, description) for channel, description in self.amb_channels],
            )
        native = [
            SensorMapping(
                amb_channel=f"bpol_{i:03d}",
                kind="b_probe",
                efm_index=i,
                r=probe.r,
                z=probe.z,
                angle_deg=probe.angle_deg,
                residual_m=0.0,
                flag="",
            )
            for i, probe in enumerate(b_probes)
        ] + [
            SensorMapping(
                amb_channel=f"floop_{i:03d}",
                kind="flux_loop",
                efm_index=i,
                r=loop.r,
                z=loop.z,
                angle_deg=None,
                residual_m=0.0,
                flag="",
            )
            for i, loop in enumerate(flux_loops)
        ]
        return native, []


def _provenance_lines(provenance: dict[str, Any]) -> list[str]:
    """Flatten the identity block into the table's flat provenance-flag list."""
    lines = [
        f"machine artifact semantic identity {provenance['semantic_identity']}",
        f"machine artifact physical digest {provenance['physical_digest']}",
        f"machine artifact registry digest {provenance['registry_digest']}",
        f"machine artifact dictionary pin {provenance['dd_version']}",
        f"registry evidence for the selected shot: {provenance['shot_evidence']}",
        "artifact evidence states "
        + ", ".join(
            f"{state}={count}"
            for state, count in sorted(provenance["evidence_states"].items())
        ),
    ]
    if not provenance["complete"]:
        lines.append(
            "artifact is INCOMPLETE: " + "; ".join(provenance["unresolved_gaps"])
        )
    lines += [
        f"forward-model blocker: {path}"
        for path in provenance["forward_model_blockers"]
    ]
    return lines


__all__ = [
    "UNRESOLVED_TURNS",
    "MachineArtifactGeometryReader",
    "UnresolvedTurnsError",
    "read_artifact_limiter",
    "read_artifact_magnetics",
    "read_artifact_pf_active",
    "read_artifact_pf_passive",
    "require_resolved_turns",
    "sensor_position_arrays",
    "unresolved_turn_circuits",
]
