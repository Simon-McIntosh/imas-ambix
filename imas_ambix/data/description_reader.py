"""Read declared machine descriptions through the geometry-table contract.

This module is the sole machine-description acquisition route. Description
content is emitted by the reviewed machine map and a store-format transform
engine, then adapted to :class:`imas_ambix.gs.geometry.GeometryTable`.
Consumers therefore do not need to know which source arrays or store layout
supplied the description.

MAST level-2 stores do not carry the directed angle of a poloidal field probe.
The acquisition declaration does carry stable addresses whose prefixes state
the sensitive axis.  This boundary supplies that declared axis on the sensor
mapping while leaving every emitted coordinate and conductor value untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from imas_ambix.data.geometry_adapter import geometry_table_from_description
from imas_ambix.data.machine_map import load_packaged_machine_map
from imas_ambix.data.paths import LEVEL2_DIR
from imas_ambix.data.transform_engine import transform_machine_description

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from imas_ambix.gs.geometry import GeometryTable


class DescriptionReadError(RuntimeError):
    """Raised when a declared description cannot produce a geometry table."""


@dataclass(frozen=True)
class AcquisitionChannels:
    """Declared acquisition addresses carried beside a machine description."""

    sensors: tuple[tuple[str, str], ...]
    currents: tuple[str, ...]


def _mast_probe_angle(address: str) -> float | None:
    """Return the directed poloidal angle declared by a MAST probe address."""
    name = address.lower()
    if name.startswith(("ccbv", "obv")):
        return -90.0
    if name.startswith("obr"):
        return 0.0
    return None


def _supply_declared_probe_angles(table: GeometryTable) -> GeometryTable:
    """Fill sensor-map angles from acquisition identities, without source reads."""
    mappings = []
    missing = []
    for mapping in table.sensor_map:
        if mapping.kind != "b_probe":
            mappings.append(mapping)
            continue
        angle = _mast_probe_angle(mapping.amb_channel)
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


def read_geometry_table(
    shot: int,
    *,
    machine: str = "mast",
    store_format: str = "zarr",
    store_root: Path | str = LEVEL2_DIR,
) -> GeometryTable:
    """Emit and adapt the declared machine description covering ``shot``."""
    shot_id = int(shot)
    catalog = load_packaged_machine_map(machine)
    description = transform_machine_description(
        catalog,
        shot_id,
        store_format,
        store_root,
    )
    if description.status != "emitted":
        raise DescriptionReadError(
            f"shot {shot_id} machine description is {description.status}: "
            f"{description.detail}"
        )
    table = geometry_table_from_description(description, catalog)
    if machine == "mast":
        table = _supply_declared_probe_angles(table)
    return table


def read_acquisition_channels(
    shots: Iterable[int],
    *,
    machine: str = "mast",
    store_format: str = "zarr",
    store_root: Path | str = LEVEL2_DIR,
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
