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
from imas_alambic.machine_map import (
    load_packaged_machine_map,
    resolve_description_store_root,
)
from imas_alambic.transform_engine import transform_machine_description

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
_PROBE_NAME_DD_PATH = "magnetics/b_field_pol_probe/name"

_ANGLE_UNIT_CONVERTERS: dict[str, Any] = {
    "rad": np.rad2deg,
    "deg": lambda values: values,
}


def _supply_emitted_probe_angles(description: Any, table: Any) -> Any:
    """Fill sensor-map angles from the emitted DD probe-angle array.

    The ``description`` probe-angle source carries the directed axis in the
    emitted ``magnetics/b_field_pol_probe/poloidal_angle`` array rather than in
    an acquisition address.  The array's ``target_unit`` decides the unit the
    table reports: the DD leaf is radians while the geometry kernel's
    ``angle_deg`` is degrees, so ``rad`` is converted and ``deg`` is carried
    through; any other unit spelling is refused rather than read as degrees.

    Each angle is joined to a mapped probe by probe identity: the emitted
    ``magnetics/b_field_pol_probe/name`` array names each angle's probe, and
    that name is the mapped probe's ``amb_channel``.  An angle whose name
    matches no mapped probe, and a mapped probe with no emitted angle, are both
    refused; array order is never used to place an angle.  A description that
    emits no probe-angle array leaves every probe's absent-angle flag standing.
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
    converter = _ANGLE_UNIT_CONVERTERS.get(emitted.target_unit)
    if converter is None:
        raise DescriptionReadError(
            f"the emitted {_PROBE_ANGLE_DD_PATH} array declares angle unit "
            f"{emitted.target_unit!r}; only 'rad' and 'deg' are accepted"
        )
    values = np.asarray(emitted.values, dtype=np.float64).reshape(-1)
    angles = np.asarray(converter(values), dtype=np.float64).reshape(-1)

    name_arrays = tuple(
        array
        for array in description.arrays
        if array.dd_path == _PROBE_NAME_DD_PATH
    )
    if len(name_arrays) != 1:
        raise DescriptionReadError(
            "the emitted description must carry exactly one "
            f"{_PROBE_NAME_DD_PATH} array to join {_PROBE_ANGLE_DD_PATH} "
            f"angles to mapped probes; found {len(name_arrays)}"
        )
    names = tuple(
        str(value) for value in np.asarray(name_arrays[0].values).reshape(-1)
    )
    if len(names) != angles.size:
        raise DescriptionReadError(
            f"the emitted {_PROBE_NAME_DD_PATH} array holds {len(names)} names "
            f"for {angles.size} {_PROBE_ANGLE_DD_PATH} angles"
        )
    angle_for_probe: dict[str, float] = {}
    for name, angle in zip(names, angles, strict=True):
        if name in angle_for_probe:
            raise DescriptionReadError(
                f"the emitted {_PROBE_NAME_DD_PATH} array names probe {name!r} "
                "more than once"
            )
        angle_for_probe[name] = float(angle)

    mappings = list(table.sensor_map)
    matched: set[str] = set()
    for index, mapping in enumerate(mappings):
        if mapping.kind != "b_probe":
            continue
        angle = angle_for_probe.get(mapping.amb_channel)
        if angle is None:
            raise DescriptionReadError(
                f"the emitted {_PROBE_ANGLE_DD_PATH} array has no angle for "
                f"mapped probe {mapping.amb_channel!r}"
            )
        matched.add(mapping.amb_channel)
        mappings[index] = replace(mapping, angle_deg=angle, flag="")
    unmatched = [name for name in names if name not in matched]
    if unmatched:
        raise DescriptionReadError(
            f"the emitted {_PROBE_ANGLE_DD_PATH} array holds angles for probes "
            "whose identities are not mapped: " + ", ".join(unmatched)
        )
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
