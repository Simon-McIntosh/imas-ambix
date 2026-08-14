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

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from imas_ambix.gs.geometry import (
    BProbe,
    FluxLoop,
    GeometryTable,
    PassiveStructure,
    PFFilament,
    SensorMapping,
    SetupSignature,
    round_geometry_hash,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from imas_ambix.data.machine_map import DriveTopology, MachineMapCatalog
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
    return loops, mappings


def _conductors(description: MachineDescription) -> list[_Conductor]:
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
            names = _names(family, r.size)
            source_group = family.arrays["r"].source_group
            conductors.extend(
                _Conductor(
                    source_group=source_group,
                    name=names[index],
                    r=float(r[index]),
                    z=float(z[index]),
                    width=float(width[index]),
                    height=float(height[index]),
                )
                for index in range(r.size)
            )
    return conductors


def _topology_for_element_count(
    catalog: MachineMapCatalog,
    description: MachineDescription,
    element_count: int,
) -> tuple[DriveTopology, str | None]:
    selected_name = description.machine_map.drive_topology
    selected = next(
        (item for item in catalog.drive_topologies if item.name == selected_name),
        None,
    )
    if selected is not None and len(selected.connections) == element_count:
        return selected, None
    matches = tuple(
        item
        for item in catalog.drive_topologies
        if len(item.connections) == element_count
    )
    if len(matches) != 1:
        raise GeometryAdapterError(
            f"description has {element_count} conductor elements but resolves to "
            f"{len(matches)} cardinality-compatible drive topologies"
        )
    selected_count = len(selected.connections) if selected is not None else 0
    return matches[0], (
        "pf_filaments: range-selected drive topology has "
        f"{selected_count} elements while the emitted geometry has {element_count}; "
        "used the unique retained topology with matching discretisation"
    )


def _pf_filaments(
    catalog: MachineMapCatalog,
    description: MachineDescription,
    conductors: list[_Conductor],
) -> tuple[list[PFFilament], list[int], list[str]]:
    topology, topology_notice = _topology_for_element_count(
        catalog, description, len(conductors)
    )
    circuit_order = tuple(
        dict.fromkeys(item.circuit_identifier for item in topology.connections)
    )
    circuit_index = {
        identifier: index + 1 for index, identifier in enumerate(circuit_order)
    }
    filaments = [
        PFFilament(
            r=conductor.r,
            z=conductor.z,
            turns=float(connection.turns),
            width=conductor.width,
            height=conductor.height,
            circuit=circuit_index[connection.circuit_identifier],
            xmult=float(connection.direction),
        )
        for conductor, connection in zip(conductors, topology.connections, strict=True)
    ]
    active_circuits = sorted(
        {
            filament.circuit
            for conductor, filament in zip(conductors, filaments, strict=True)
            if conductor.source_group == "pf_active"
        }
    )
    notices = [
        "pf_filaments: emitted element identifiers and topology element "
        "identifiers occupy different namespaces; associated in declaration order"
    ]
    if topology_notice is not None:
        notices.insert(0, topology_notice)
    return filaments, active_circuits, notices


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
    flux_loops, flux_mappings = _flux_loops(description)
    conductors = _conductors(description)
    filaments, active_circuits, topology_notices = _pf_filaments(
        catalog, description, conductors
    )
    limiter_r, limiter_z = _limiter(description)
    notices = [
        "b_probes.angle_deg: catalog qualification records that the source "
        "does not expose poloidal probe orientation",
        *topology_notices,
        "sensor_map: DD sensor names are used directly because acquisition "
        "address descriptions are not declared",
        "amc_current_channels: catalog qualification records that acquisition "
        "current-channel addressing is absent",
        "circuit_drives: current-channel addressing is absent, so topology "
        "supplies no measured-channel association",
        "polygon_sections: conductor geometry cannot be joined to topology "
        "by a shared declared element identifier",
        "r0: no device reference-radius binding is emitted",
        "minor_radius: no device minor-radius binding is emitted",
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
        sensor_map=[*b_mappings, *flux_mappings],
        passive_structures=_passive_structures(conductors),
        amc_current_channels=[],
        unmatched_amb=[],
        r0=float("nan"),
        minor_radius=float("nan"),
        provenance_flags=notices,
        active_circuits=active_circuits,
        circuit_drives=[],
        polygon_sections=[],
    )


__all__ = ["GeometryAdapterError", "geometry_table_from_description"]
