"""Build the private geometry kernel from declared machine descriptions.

This module is the sole machine-description acquisition route. Description
content is emitted by the reviewed machine map and a store-format transform
engine, then adapted to the private compatibility kernel behind
``MachineGeometryService``. Consumers therefore do not need to know which
source arrays or store layout supplied the description.

MAST level-2 stores do not carry the directed angle of a poloidal field probe.
The acquisition declaration does carry stable addresses whose prefixes state
the sensitive axis.  This boundary supplies that declared axis on the sensor
mapping while leaving every emitted coordinate and conductor value untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from imas_ambix.data.geometry_adapter import geometry_table_from_description
from imas_ambix.data.machine_map import (
    load_packaged_machine_map,
    resolve_description_store_root,
)
from imas_ambix.data.transform_engine import transform_machine_description

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

class DescriptionReadError(RuntimeError):
    """Raised when a declared description cannot produce a geometry table."""


@dataclass(frozen=True)
class AcquisitionChannels:
    """Declared acquisition addresses carried beside a machine description."""

    sensors: tuple[tuple[str, str], ...]
    currents: tuple[str, ...]


def _probe_angle_from_address(address: str) -> float | None:
    """Return the directed poloidal angle declared by a probe address prefix."""
    name = address.lower()
    if name.startswith(("ccbv", "obv")):
        return -90.0
    if name.startswith("obr"):
        return 0.0
    return None


def _supply_declared_probe_angles(table: Any) -> Any:
    """Fill sensor-map angles from acquisition identities, without source reads."""
    mappings = []
    missing = []
    for mapping in table.sensor_map:
        if mapping.kind != "b_probe":
            mappings.append(mapping)
            continue
        angle = _probe_angle_from_address(mapping.amb_channel)
        if angle is None:
            missing.append(mapping.amb_channel)
            mappings.append(mapping)
            continue
        mappings.append(replace(mapping, angle_deg=angle, flag=""))
    if missing:
        raise DescriptionReadError(
            "declared MAST probe addresses do not state a sensitive axis: "
            + ", ".join(missing)
        )
    return replace(
        table,
        sensor_map=mappings,
        provenance_flags=[
            *table.provenance_flags,
            "sensor_map.angle_deg: directed probe axes supplied by the reviewed "
            "MAST acquisition-address convention",
        ],
    )


_PROBE_ANGLE_DD_PATH = "magnetics/b_field_pol_probe/poloidal_angle"


def _supply_emitted_probe_angles(description: Any, table: Any) -> Any:
    """Fill sensor-map angles from the emitted DD probe-angle array.

    The ``description`` probe-angle source carries the directed axis in the
    emitted ``magnetics/b_field_pol_probe/poloidal_angle`` array rather than in
    an acquisition address.  The array's ``target_unit`` decides the unit the
    table reports: the DD leaf is radians while the geometry kernel's
    ``angle_deg`` is degrees.  A description that emits no such array leaves
    every probe's absent-angle flag standing.
    """

    angle_arrays = tuple(
        array
        for array in description.arrays
        if array.dd_path == _PROBE_ANGLE_DD_PATH
    )
    if not angle_arrays:
        return table
    if len(angle_arrays) > 1:
        raise DescriptionReadError(
            "the emitted description carries more than one "
            f"{_PROBE_ANGLE_DD_PATH} array"
        )
    emitted = angle_arrays[0]
    values = np.asarray(emitted.values, dtype=np.float64).reshape(-1)
    probe_indices = [
        index
        for index, mapping in enumerate(table.sensor_map)
        if mapping.kind == "b_probe"
    ]
    if values.size != len(probe_indices):
        raise DescriptionReadError(
            f"the emitted {_PROBE_ANGLE_DD_PATH} array holds {values.size} "
            f"angles for {len(probe_indices)} mapped probes"
        )
    angles = np.rad2deg(values) if emitted.target_unit == "rad" else values
    mappings = list(table.sensor_map)
    for index, angle in zip(probe_indices, angles, strict=True):
        mappings[index] = replace(mappings[index], angle_deg=float(angle), flag="")
    return replace(
        table,
        sensor_map=mappings,
        provenance_flags=[
            *table.provenance_flags,
            "sensor_map.angle_deg: directed probe axes supplied by the emitted "
            f"{_PROBE_ANGLE_DD_PATH}",
        ],
    )


def _resolve_store_addressing(
    catalog: Any,
    machine: str,
    store_format: str | None,
    store_root: Path | str | None,
) -> tuple[str, Path | str]:
    """Resolve the store format and root a description read addresses.

    An explicit ``store_format``/``store_root`` override wins; otherwise the
    loaded catalog's own declaration supplies them.  A catalog that declares no
    description store has neither, so a read against it is refused naming the
    machine rather than silently resolving to another machine's root.
    """
    if store_format is None:
        if catalog.description_store_format is None:
            raise DescriptionReadError(
                f"machine {machine!r} declares no description store; supply an "
                "explicit store_format and store_root to read one"
            )
        store_format = catalog.description_store_format
    if store_root is None:
        if catalog.description_store_root is None:
            raise DescriptionReadError(
                f"machine {machine!r} declares no description store; supply an "
                "explicit store_format and store_root to read one"
            )
        store_root = resolve_description_store_root(catalog.description_store_root)
    return store_format, store_root


def read_geometry_table(
    shot: int,
    *,
    machine: str = "mast",
    store_format: str | None = None,
    store_root: Path | str | None = None,
) -> Any:
    """Emit and adapt the declared machine description covering ``shot``.

    The store format, root and probe-angle source come from the loaded
    catalog.  ``store_format`` and ``store_root`` remain optional overrides so
    a caller that points at a specific store keeps working; when omitted the
    catalog's own declaration is used.
    """
    shot_id = int(shot)
    catalog = load_packaged_machine_map(machine)
    resolved_format, resolved_root = _resolve_store_addressing(
        catalog, machine, store_format, store_root
    )
    description = transform_machine_description(
        catalog,
        shot_id,
        resolved_format,
        resolved_root,
    )
    if description.status != "emitted":
        raise DescriptionReadError(
            f"shot {shot_id} machine description is {description.status}: "
            f"{description.detail}"
        )
    table = geometry_table_from_description(description, catalog)
    if catalog.probe_angle_source == "acquisition-address":
        table = _supply_declared_probe_angles(table)
    elif catalog.probe_angle_source == "description":
        table = _supply_emitted_probe_angles(description, table)
    return table


def read_acquisition_channels(
    shots: Iterable[int],
    *,
    machine: str = "mast",
    store_format: str | None = None,
    store_root: Path | str | None = None,
) -> AcquisitionChannels:
    """Return the stable union of declared sensor and current addresses."""
    sensors: dict[str, str] = {}
    currents: dict[str, None] = {}
    for shot in shots:
        table = read_geometry_table(
            int(shot),
            machine=machine,
            store_format=store_format,
            store_root=store_root,
        )
        for mapping in table.sensor_map:
            sensors.setdefault(
                mapping.amb_channel,
                f"r={mapping.r:.17g}, z={mapping.z:.17g}",
            )
        for channel in table.amc_current_channels:
            currents.setdefault(channel, None)
    return AcquisitionChannels(
        sensors=tuple(sensors.items()),
        currents=tuple(currents),
    )


__all__ = [
    "AcquisitionChannels",
    "DescriptionReadError",
    "read_acquisition_channels",
    "read_geometry_table",
]
