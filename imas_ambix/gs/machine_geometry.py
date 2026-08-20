"""Shot-addressed projections of resolved machine geometry.

The public boundary is deliberately narrower than the compatibility table used
to construct it.  Representation identity remains byte-for-byte compatible
with existing compute caches, while callers receive only the operator, sensor,
or identity view they need.
"""

from __future__ import annotations

import dataclasses as _dataclasses
import types as _types
import typing as _typing

import numpy as _np

from imas_ambix.gs import geometry as _geometry
from imas_ambix.gs import geometry_export as _geometry_export
from imas_ambix.gs import machine_selection as _machine_selection
from imas_ambix.gs import operator as _operator


@_dataclasses.dataclass(frozen=True)
class GeometryIdentity:
    """Selection provenance beside the unchanged representation identity."""

    representation_key: str
    representation_digest: str
    derivation_id: str
    physical_digest: str
    registry_digest: str


@_dataclasses.dataclass(frozen=True, eq=False)
class OperatorGeometry:
    """Immutable geometry and coil-column matrix needed by forward operators."""

    identity: GeometryIdentity
    probes: tuple[_typing.Any, ...]
    loops: tuple[_typing.Any, ...]
    conductors: tuple[_typing.Any, ...]
    passives: tuple[_typing.Any, ...]
    limiter_r: tuple[float, ...]
    limiter_z: tuple[float, ...]
    polygon_sections: tuple[_typing.Any, ...]
    drive_map: tuple[_typing.Any, ...]
    unresolved_turns: _typing.Mapping[str, None]
    coil_channels: tuple[str, ...]
    coil_column_matrix: _np.ndarray


@_dataclasses.dataclass(frozen=True, eq=False)
class SensorGeometry:
    """Dense positional features aligned to an explicitly requested channel list."""

    identity: GeometryIdentity
    channels: tuple[str, ...]
    feature_names: tuple[str, ...]
    sensor_kinds: tuple[str, ...]
    feature_matrix: _np.ndarray


def _readonly_array(value: _typing.Any) -> _np.ndarray:
    array = _np.array(value, copy=True)
    array.setflags(write=False)
    return array


def _readonly_sections(
    sections: _typing.Iterable[_typing.Any],
) -> tuple[_typing.Any, ...]:
    return tuple(
        _dataclasses.replace(section, vertices=_readonly_array(section.vertices))
        for section in sections
    )


def _unresolved_turns(table: _typing.Any) -> _typing.Mapping[str, None]:
    unresolved_circuits = {
        int(filament.circuit)
        for filament in table.pf_filaments
        if not _np.isfinite(filament.turns)
    }
    drives = {int(drive.circuit): drive for drive in table.circuit_drives}
    named: dict[str, None] = {}
    for circuit in sorted(unresolved_circuits):
        drive = drives.get(circuit)
        if drive is None:
            name = f"circuit_{circuit}"
        else:
            name = str(drive.conductor or drive.channel)
        named[name] = None
    return _types.MappingProxyType(named)


class MachineGeometryService:
    """Resolve one shot and expose narrow, cached geometry projections."""

    def __init__(
        self,
        *,
        channel_shots: _typing.Iterable[int] = (),
        amc_channel_shot: int | None = None,
    ) -> None:
        self._selector = _machine_selection.ArtifactMachineSelector(
            channel_shots=tuple(int(shot) for shot in channel_shots),
            amc_channel_shot=amc_channel_shot,
        )
        self._selections: dict[int, _typing.Any] = {}
        self._identities: dict[int, GeometryIdentity] = {}
        self._operators: dict[int, _typing.Any] = {}
        self._operator_projections: dict[int, OperatorGeometry] = {}

    def _selection(self, shot: int) -> _typing.Any:
        addressed_shot = int(shot)
        if addressed_shot not in self._selections:
            self._selections[addressed_shot] = self._selector.select(addressed_shot)
        return self._selections[addressed_shot]

    def _compatibility_kernel(self, shot: int) -> _typing.Any:
        """Return the private table while consumers move onto projections."""
        return self._selection(shot).table

    def _compatibility_operator(self, shot: int) -> _typing.Any:
        addressed_shot = int(shot)
        if addressed_shot not in self._operators:
            self._operators[addressed_shot] = _operator.build_operator(
                self._compatibility_kernel(addressed_shot)
            )
        return self._operators[addressed_shot]

    def identity(self, shot: int) -> GeometryIdentity:
        """Return representation and selection identities for ``shot``."""
        addressed_shot = int(shot)
        if addressed_shot not in self._identities:
            selected = self._selection(addressed_shot)
            signature = selected.table.signature
            self._identities[addressed_shot] = GeometryIdentity(
                representation_key=signature.key,
                representation_digest=signature.digest,
                derivation_id=_geometry.GEOMETRY_TABLE_VERSION,
                physical_digest=selected.identity.physical_digest,
                registry_digest=selected.identity.registry_digest,
            )
        return self._identities[addressed_shot]

    def operator(self, shot: int) -> OperatorGeometry:
        """Return the operator-facing projection for ``shot``."""
        addressed_shot = int(shot)
        if addressed_shot not in self._operator_projections:
            table = self._compatibility_kernel(addressed_shot)
            existing = self._compatibility_operator(addressed_shot)
            self._operator_projections[addressed_shot] = OperatorGeometry(
                identity=self.identity(addressed_shot),
                probes=tuple(table.b_probes),
                loops=tuple(table.flux_loops),
                conductors=tuple(table.pf_filaments),
                passives=tuple(table.passive_structures),
                limiter_r=tuple(float(value) for value in table.limiter_r),
                limiter_z=tuple(float(value) for value in table.limiter_z),
                polygon_sections=_readonly_sections(table.polygon_sections),
                drive_map=tuple(table.circuit_drives),
                unresolved_turns=_unresolved_turns(table),
                coil_channels=tuple(existing.pf_amc_channels),
                coil_column_matrix=_readonly_array(existing.g_pf),
            )
        return self._operator_projections[addressed_shot]

    def sensors(self, shot: int, channels: _typing.Iterable[str]) -> SensorGeometry:
        """Return sensor features aligned to ``channels`` for ``shot``."""
        addressed_shot = int(shot)
        requested = tuple(str(channel) for channel in channels)
        fields = _geometry_export.build_geometry_fields_from_table(
            self._compatibility_kernel(addressed_shot),
            extra_channel_names=requested,
        )
        matrix, kinds = fields.feature_matrix(requested)
        return SensorGeometry(
            identity=self.identity(addressed_shot),
            channels=requested,
            feature_names=tuple(_geometry_export.GEOMETRY_FEATURE_NAMES),
            sensor_kinds=tuple(kinds),
            feature_matrix=_readonly_array(matrix),
        )


__all__ = [
    "GeometryIdentity",
    "MachineGeometryService",
    "OperatorGeometry",
    "SensorGeometry",
]

globals().pop("annotations", None)
