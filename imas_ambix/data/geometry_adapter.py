"""Adapt DD-shaped machine descriptions to the legacy geometry table seam.

The adapter consumes only declarations already carried by a machine catalog:
Data Dictionary paths identify geometry roles, binding identities assemble
repeated structures, and the selected drive topology supplies circuit
connectivity.  Source machine names never select behaviour.

Some catalog declarations are intentionally incomplete for the legacy table.
Those omissions remain visible through ``GeometryTable.provenance_flags``;
the adapter does not substitute device constants or inspect a legacy reader.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

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
    collapse_rectangular_circuits,
    round_geometry_hash,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from imas_ambix.data.machine_map import (
        AcquisitionDeclaration,
        DescriptionSupplement,
        DriveTopology,
        MachineMapCatalog,
    )
    from imas_ambix.data.transform_engine import EmittedArray, MachineDescription


class GeometryAdapterError(ValueError):
    """Raised when an emitted description cannot form a geometry table."""


@dataclass(frozen=True)
class _Family:
    """One repeated DD structure assembled from related emitted bindings."""

    stem: str
    order: int
    arrays: Mapping[str, EmittedArray]


@dataclass(frozen=True)
class _Conductor:
    """One source-declared conductor element before circuit association."""

    source_group: str
    identifier: str
    name: str
    r: float
    z: float
    width: float
    height: float


_ROLE_SUFFIXES: Mapping[str, tuple[str, ...]] = {
    "r": ("-r",),
    "z": ("-z",),
    "length": ("-length",),
    "width": ("-width",),
    "height": ("-height",),
    "name": ("-geometry-channel", "-coordinate-element"),
}


def _role_and_stem(binding_name: str) -> tuple[str, str] | None:
    for role, suffixes in _ROLE_SUFFIXES.items():
        for suffix in suffixes:
            if binding_name.endswith(suffix):
                return role, binding_name[: -len(suffix)]
    return None


def _families(
    description: MachineDescription,
    *,
    path_prefix: str,
) -> tuple[_Family, ...]:
    grouped: dict[str, dict[str, EmittedArray]] = {}
    order: dict[str, int] = {}
    for index, emitted in enumerate(description.arrays):
        if not emitted.dd_path.startswith(path_prefix):
            continue
        identity = _role_and_stem(emitted.binding_name)
        if identity is None:
            continue
        role, stem = identity
        grouped.setdefault(stem, {})[role] = emitted
        order.setdefault(stem, index)
    for emitted in description.arrays:
        identity = _role_and_stem(emitted.binding_name)
        if identity is None:
            continue
        role, stem = identity
        if role == "name" and stem in grouped:
            grouped[stem][role] = emitted
    return tuple(
        _Family(stem, order[stem], grouped[stem])
        for stem in sorted(grouped, key=order.__getitem__)
    )


def _same_shape(family: _Family, roles: Iterable[str]) -> bool:
    shapes = {np.asarray(family.arrays[role].values).shape for role in roles}
    return len(shapes) == 1


def _names(family: _Family, size: int) -> tuple[str, ...]:
    emitted = family.arrays.get("name")
    if emitted is None:
        return tuple(f"element-{index}" for index in range(size))
    values = np.asarray(emitted.values).reshape(-1)
    if values.size != size:
        raise GeometryAdapterError(
            f"binding family {family.stem!r} has {values.size} names for {size} "
            "geometry elements"
        )
    return tuple(str(value) for value in values)


def _b_probes(
    description: MachineDescription,
) -> tuple[list[BProbe], list[SensorMapping]]:
    probes: list[BProbe] = []
    mappings: list[SensorMapping] = []
    for family in _families(
        description,
        path_prefix="magnetics/b_field_pol_probe/",
    ):
        roles = ("r", "z", "length")
        if not set(roles).issubset(family.arrays) or not _same_shape(family, roles):
            continue
        r = np.asarray(family.arrays["r"].values).reshape(-1)
        z = np.asarray(family.arrays["z"].values).reshape(-1)
        length = np.asarray(family.arrays["length"].values).reshape(-1)
        names = _names(family, r.size)
        for item in range(r.size):
            index = len(probes)
            probes.append(
                BProbe(
                    index=index,
                    r=float(r[item]),
                    z=float(z[item]),
                    angle_deg=float("nan"),
                    length=float(length[item]),
                )
            )
            mappings.append(
                SensorMapping(
                    amb_channel=names[item],
                    kind="b_probe",
                    efm_index=index,
                    r=float(r[item]),
                    z=float(z[item]),
                    angle_deg=None,
                    residual_m=0.0,
                    flag="poloidal angle is absent from the emitted description",
                )
            )
    return probes, mappings


def _flux_loops(
    description: MachineDescription,
    supplement: DescriptionSupplement,
) -> tuple[list[FluxLoop], list[SensorMapping]]:
    loops: list[FluxLoop] = []
    mappings: list[SensorMapping] = []
    for family in _families(description, path_prefix="magnetics/flux_loop/"):
        roles = ("r", "z")
        if not set(roles).issubset(family.arrays) or not _same_shape(family, roles):
            continue
        r = np.asarray(family.arrays["r"].values)
        z = np.asarray(family.arrays["z"].values)
        if r.ndim != 1:
            continue
        names = _names(family, r.size)
        for item in range(r.size):
            index = len(loops)
            loops.append(FluxLoop(index=index, r=float(r[item]), z=float(z[item])))
            mappings.append(
                SensorMapping(
                    amb_channel=names[item],
                    kind="flux_loop",
                    efm_index=index,
                    r=float(r[item]),
                    z=float(z[item]),
                    angle_deg=None,
                    residual_m=0.0,
                    flag="",
                )
            )
    loops.extend(
        FluxLoop(index=len(loops) + index, r=item.r, z=item.z)
        for index, item in enumerate(supplement.point_flux_loops)
    )
    return loops, mappings


def _assembly_identifiers(
    catalog: MachineMapCatalog,
    description: MachineDescription,
    family: _Family,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    member_bindings = {
        item.binding_name for role, item in family.arrays.items() if role != "name"
    }
    matches = tuple(
        assembly
        for assembly in catalog.structure_assemblies
        if member_bindings.issubset(assembly.member_bindings)
        and assembly.element_identifiers
    )
    if len(matches) != 1:
        raise GeometryAdapterError(
            f"conductor family {family.stem!r} resolves to {len(matches)} "
            "identifier-bearing structure assemblies"
        )
    assembly = matches[0]
    name_rows = tuple(
        item
        for item in description.arrays
        if item.binding_name == assembly.name_binding
    )
    if len(name_rows) != 1:
        raise GeometryAdapterError(
            f"assembly {assembly.name!r} name binding resolves to "
            f"{len(name_rows)} emitted arrays"
        )
    names = tuple(str(item) for item in np.asarray(name_rows[0].values).reshape(-1))
    identifiers = assembly.element_identifiers
    if len(identifiers) != len(names):
        raise GeometryAdapterError(
            f"assembly {assembly.name!r} declares {len(identifiers)} identifiers "
            f"for {len(names)} emitted elements"
        )
    for identifier, name in zip(identifiers, names, strict=True):
        if identifier.rsplit("/", 1)[-1] != name:
            raise GeometryAdapterError(
                f"assembly identifier {identifier!r} does not name emitted "
                f"element {name!r}"
            )
    return identifiers, names


def _conductors(
    description: MachineDescription,
    catalog: MachineMapCatalog,
) -> list[_Conductor]:
    conductors: list[_Conductor] = []
    prefixes = (
        "pf_active/coil/element/geometry/",
        "pf_passive/loop/element/geometry/",
    )
    for prefix in prefixes:
        for family in _families(description, path_prefix=prefix):
            roles = ("r", "z", "width", "height")
            if not set(roles).issubset(family.arrays) or not _same_shape(family, roles):
                continue
            r = np.asarray(family.arrays["r"].values).reshape(-1)
            z = np.asarray(family.arrays["z"].values).reshape(-1)
            width = np.asarray(family.arrays["width"].values).reshape(-1)
            height = np.asarray(family.arrays["height"].values).reshape(-1)
            identifiers, names = _assembly_identifiers(catalog, description, family)
            if len(names) != r.size:
                raise GeometryAdapterError(
                    f"conductor family {family.stem!r} has {r.size} geometry "
                    f"elements but {len(names)} declared names"
                )
            source_group = family.arrays["r"].source_group
            conductors.extend(
                _Conductor(
                    source_group=source_group,
                    identifier=identifiers[index],
                    name=names[index],
                    r=float(r[index]),
                    z=float(z[index]),
                    width=float(width[index]),
                    height=float(height[index]),
                )
                for index in range(r.size)
            )
    identifiers = tuple(item.identifier for item in conductors)
    if len(identifiers) != len(set(identifiers)):
        raise GeometryAdapterError("emitted conductor identifiers are not unique")
    return conductors


def _selected_topology(
    catalog: MachineMapCatalog,
    description: MachineDescription,
) -> DriveTopology:
    selected_name = description.machine_map.drive_topology
    matches = tuple(
        item for item in catalog.drive_topologies if item.name == selected_name
    )
    if len(matches) != 1:
        raise GeometryAdapterError(
            f"range-selected drive topology {selected_name!r} resolves to "
            f"{len(matches)} declarations"
        )
    return matches[0]


def _selected_supplement(
    catalog: MachineMapCatalog,
    description: MachineDescription,
) -> DescriptionSupplement:
    selected_name = description.machine_map.description_supplement
    matches = tuple(
        item for item in catalog.description_supplements if item.name == selected_name
    )
    if len(matches) != 1:
        raise GeometryAdapterError(
            f"range-selected description supplement {selected_name!r} resolves to "
            f"{len(matches)} declarations"
        )
    return matches[0]


def _selected_acquisition(
    catalog: MachineMapCatalog,
    supplement: DescriptionSupplement,
) -> AcquisitionDeclaration:
    matches = tuple(
        item
        for item in catalog.acquisition_declarations
        if item.name == supplement.acquisition_declaration
    )
    if len(matches) != 1:
        raise GeometryAdapterError(
            f"supplement acquisition {supplement.acquisition_declaration!r} "
            f"resolves to {len(matches)} declarations"
        )
    return matches[0]


def _pf_filaments(
    catalog: MachineMapCatalog,
    description: MachineDescription,
    conductors: list[_Conductor],
) -> tuple[list[PFFilament], list[int], list[str]]:
    topology = _selected_topology(catalog, description)
    circuit_order = tuple(
        dict.fromkeys(item.circuit_identifier for item in topology.connections)
    )
    circuit_index = {
        identifier: index + 1 for index, identifier in enumerate(circuit_order)
    }
    by_identifier = {item.identifier: item for item in conductors}
    missing = tuple(
        connection.geometry_element_identifier
        for connection in topology.connections
        if connection.geometry_element_identifier not in by_identifier
    )
    if missing:
        raise GeometryAdapterError(
            f"selected topology has {len(missing)} rows without emitted conductor "
            f"geometry; first missing identifier is {missing[0]!r}"
        )
    raw_filaments: list[PFFilament] = []
    raw_groups: list[str] = []
    for connection in topology.connections:
        conductor = by_identifier[connection.geometry_element_identifier]
        raw_filaments.append(
            PFFilament(
                r=conductor.r,
                z=conductor.z,
                turns=float(connection.turns),
                width=conductor.width,
                height=conductor.height,
                circuit=circuit_index[connection.circuit_identifier],
                xmult=float(connection.direction),
            )
        )
        raw_groups.append(conductor.source_group)
    active_circuits = sorted(
        {
            filament.circuit
            for source_group, filament in zip(raw_groups, raw_filaments, strict=True)
            if source_group == "pf_active"
        }
    )
    return collapse_rectangular_circuits(raw_filaments), active_circuits, []


def _sensor_map_for_acquisition(
    mappings: Iterable[SensorMapping],
    acquisition: AcquisitionDeclaration,
) -> list[SensorMapping]:
    by_channel = {item.amb_channel.casefold(): item for item in mappings}
    unmatched = set(acquisition.unmatched_sensor_addresses)
    selected: list[SensorMapping] = []
    for address in acquisition.sensor_addresses:
        if address in unmatched:
            continue
        mapping = by_channel.get(address.casefold())
        if mapping is None:
            selected.append(
                SensorMapping(
                    amb_channel=address,
                    kind="unresolved",
                    efm_index=-1,
                    r=float("nan"),
                    z=float("nan"),
                    angle_deg=None,
                    residual_m=float("nan"),
                    flag=(
                        "the acquisition declaration supplies the address but no "
                        "association to emitted sensor geometry"
                    ),
                )
            )
        else:
            selected.append(replace(mapping, amb_channel=address))
    return selected


def _polygon_sections(
    supplement: DescriptionSupplement,
    topology: DriveTopology,
) -> list[PolygonSection]:
    circuit_order = tuple(
        dict.fromkeys(item.circuit_identifier for item in topology.connections)
    )
    circuit_index = {
        identifier: index + 1 for index, identifier in enumerate(circuit_order)
    }
    sections: list[PolygonSection] = []
    for declaration in supplement.polygon_sections:
        if declaration.circuit_identifier not in circuit_index:
            raise GeometryAdapterError(
                f"polygon {declaration.name!r} names unknown circuit "
                f"{declaration.circuit_identifier!r}"
            )
        sections.append(
            PolygonSection(
                circuit=circuit_index[declaration.circuit_identifier],
                vertices=np.column_stack((declaration.vertex_r, declaration.vertex_z)),
                xmult=declaration.current_scale,
                name=declaration.name,
            )
        )
    return sections


def _passive_structures(conductors: list[_Conductor]) -> list[PassiveStructure]:
    return [
        PassiveStructure(
            name=conductor.name,
            r=conductor.r,
            z=conductor.z,
            obsolete=False,
        )
        for conductor in conductors
        if conductor.source_group == "pf_passive"
    ]


def _limiter(description: MachineDescription) -> tuple[list[float], list[float]]:
    by_path = description.arrays_by_dd_path
    r_rows = by_path.get("wall/description_2d/limiter/unit/outline/r", ())
    z_rows = by_path.get("wall/description_2d/limiter/unit/outline/z", ())
    if len(r_rows) != 1 or len(z_rows) != 1:
        raise GeometryAdapterError(
            "description must emit exactly one limiter R array and one limiter Z array"
        )
    r = np.asarray(r_rows[0].values).reshape(-1)
    z = np.asarray(z_rows[0].values).reshape(-1)
    size = min(r.size, z.size)
    return r[:size].tolist(), z[:size].tolist()


def _signature(
    description: MachineDescription,
    b_probes: list[BProbe],
    flux_loops: list[FluxLoop],
    filaments: list[PFFilament],
    limiter_r: list[float],
    limiter_z: list[float],
) -> SetupSignature:
    arrays = (
        np.asarray([item.r for item in b_probes]),
        np.asarray([item.z for item in b_probes]),
        np.asarray([item.angle_deg for item in b_probes]),
        np.asarray([item.r for item in flux_loops]),
        np.asarray([item.z for item in flux_loops]),
        np.asarray([item.r for item in filaments]),
        np.asarray([item.z for item in filaments]),
        np.asarray([item.turns * item.xmult for item in filaments]),
        np.asarray(limiter_r),
        np.asarray(limiter_z),
    )
    return SetupSignature(
        n_bprobe=len(b_probes),
        n_fluxloop=len(flux_loops),
        n_pf_filament=len(filaments),
        n_limiter=len(limiter_r),
        digest=round_geometry_hash(arrays),
        machine=description.machine_map.machine,
    )


def geometry_table_from_description(
    description: MachineDescription,
    catalog: MachineMapCatalog,
) -> GeometryTable:
    """Convert one emitted machine description into a legacy geometry table.

    Unavailable legacy-only information is represented by empty collections,
    ``NaN`` device radii, and explicit provenance flags.  The adapter never
    reads source stores or legacy geometry; callers must obtain the
    ``MachineDescription`` through a transform engine first.
    """
    if description.status != "emitted":
        raise GeometryAdapterError(
            f"description status is {description.status!r}, not 'emitted'"
        )
    if description.machine_map not in catalog.maps:
        raise GeometryAdapterError("description map does not belong to the catalog")

    b_probes, b_mappings = _b_probes(description)
    supplement = _selected_supplement(catalog, description)
    acquisition = _selected_acquisition(catalog, supplement)
    topology = _selected_topology(catalog, description)
    flux_loops, flux_mappings = _flux_loops(description, supplement)
    conductors = _conductors(description, catalog)
    filaments, active_circuits, topology_notices = _pf_filaments(
        catalog, description, conductors
    )
    limiter_r, limiter_z = _limiter(description)
    sensor_map = _sensor_map_for_acquisition((*b_mappings, *flux_mappings), acquisition)
    notices = [
        "b_probes.angle_deg: catalog qualification records that the source "
        "does not expose poloidal probe orientation",
        *topology_notices,
        "sensor_map: emitted geometry does not carry the legacy nearest-neighbour "
        "residual or probe-orientation values",
        "minor_radius: the catalog qualification records that the Data Dictionary "
        "has no fixed machine-description minor-radius leaf",
    ]
    signature = _signature(
        description,
        b_probes,
        flux_loops,
        filaments,
        limiter_r,
        limiter_z,
    )
    return GeometryTable(
        signature=signature,
        shots=[description.shot],
        b_probes=b_probes,
        flux_loops=flux_loops,
        pf_filaments=filaments,
        limiter_r=limiter_r,
        limiter_z=limiter_z,
        sensor_map=sensor_map,
        passive_structures=_passive_structures(conductors),
        amc_current_channels=list(acquisition.current_channels),
        unmatched_amb=list(acquisition.unmatched_sensor_addresses),
        r0=supplement.reference_radius,
        minor_radius=float("nan"),
        provenance_flags=notices,
        active_circuits=active_circuits,
        circuit_drives=[],
        polygon_sections=_polygon_sections(supplement, topology),
    )


__all__ = ["GeometryAdapterError", "geometry_table_from_description"]
