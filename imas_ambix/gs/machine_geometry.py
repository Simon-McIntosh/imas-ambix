"""Shot-addressed projections of resolved machine geometry.

The public boundary is deliberately narrower than the compatibility table used
to construct it.  Representation identity remains byte-for-byte compatible
with existing compute caches, while callers receive only the operator, sensor,
or identity view they need.
"""

from __future__ import annotations

import dataclasses as _dataclasses
import re as _re
import types as _types
import typing as _typing

import numpy as _np

from imas_ambix.gs import geometry as _geometry
from imas_ambix.gs import machine_selection as _machine_selection

_SENSOR_FEATURE_NAMES: tuple[str, ...] = (
    "r",
    "z",
    "phi",
    "angle_deg",
    "normal_r",
    "normal_z",
    "chord_r1",
    "chord_z1",
    "chord_r2",
    "chord_z2",
)

_SENSOR_NAME_SEPARATOR_RE = _re.compile(r"[\s_\-/.]+")
_INTERFEROMETER_PREFIXES = ("interfer", "nbar", "density", "ne_", "ne")
_SXR_PREFIXES = ("sxr", "softxray", "xsx")
_COIL_PREFIXES = ("p1", "p2", "p3", "p4", "p5", "p6", "pf", "sol", "tf", "ip")

_KIND_BPOL_PROBE = "bpol_probe"
_KIND_FLUX_LOOP = "flux_loop"
_KIND_INTERFEROMETER_CHORD = "interferometer_chord"
_KIND_SXR_CHORD = "sxr_chord"
_KIND_COIL = "coil"
_KIND_SCALAR = "scalar"


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
    sensor_map: tuple[_typing.Any, ...]
    unmatched_channels: tuple[str, ...]
    active_circuits: tuple[int, ...]
    available_current_channels: tuple[str, ...]
    r0: float
    minor_radius: float
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


def _normalise_sensor_name(name: str) -> str:
    return _SENSOR_NAME_SEPARATOR_RE.sub("", str(name).lower())


def _unmapped_sensor_kind(channel_name: str) -> str:
    normalised = _normalise_sensor_name(channel_name)
    if normalised.startswith(_SXR_PREFIXES):
        return _KIND_SXR_CHORD
    if normalised.startswith(_INTERFEROMETER_PREFIXES):
        return _KIND_INTERFEROMETER_CHORD
    if normalised.startswith(_COIL_PREFIXES):
        return _KIND_COIL
    return _KIND_SCALAR


def _empty_sensor_row(*, phi: float = _np.nan) -> _np.ndarray:
    row = _np.full(len(_SENSOR_FEATURE_NAMES), _np.nan, dtype=_np.float64)
    row[2] = phi
    return row


def _mapped_sensor_row(mapping: _typing.Any) -> tuple[_np.ndarray, str]:
    if mapping.kind == "b_probe":
        row = _empty_sensor_row(phi=0.0)
        row[0] = float(mapping.r)
        row[1] = float(mapping.z)
        angle = (
            float(mapping.angle_deg) if mapping.angle_deg is not None else float("nan")
        )
        row[3] = angle
        if _np.isfinite(angle):
            radians = _np.deg2rad(angle)
            row[4] = float(_np.cos(radians))
            row[5] = float(_np.sin(radians))
        return row, _KIND_BPOL_PROBE

    if mapping.kind == "flux_loop":
        row = _empty_sensor_row(phi=0.0)
        row[0] = float(mapping.r)
        row[1] = float(mapping.z)
        return row, _KIND_FLUX_LOOP

    kind = _unmapped_sensor_kind(mapping.amb_channel)
    return _empty_sensor_row(phi=0.0 if kind != _KIND_SCALAR else _np.nan), kind


def _project_sensor_features(
    kernel: _typing.Any,
    channels: tuple[str, ...],
) -> tuple[_np.ndarray, tuple[str, ...]]:
    """Project the private kernel onto aligned sensor features and kinds."""
    rows_by_name: dict[str, tuple[_np.ndarray, str]] = {}
    for mapping in kernel.sensor_map:
        rows_by_name[_normalise_sensor_name(mapping.amb_channel)] = _mapped_sensor_row(
            mapping
        )

    for channel in kernel.amc_current_channels:
        key = _normalise_sensor_name(channel)
        rows_by_name.setdefault(key, (_empty_sensor_row(phi=0.0), _KIND_COIL))

    matrix = _np.full(
        (len(channels), len(_SENSOR_FEATURE_NAMES)),
        _np.nan,
        dtype=_np.float32,
    )
    kinds: list[str] = []
    for index, channel in enumerate(channels):
        row_and_kind = rows_by_name.get(_normalise_sensor_name(channel))
        if row_and_kind is None:
            kind = _unmapped_sensor_kind(channel)
            row = _empty_sensor_row(phi=0.0 if kind != _KIND_SCALAR else _np.nan)
        else:
            row, kind = row_and_kind
        matrix[index] = _np.asarray(row, dtype=_np.float32)
        kinds.append(kind)
    return matrix, tuple(kinds)


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


def _project_operator_geometry(
    kernel: _typing.Any,
    *,
    identity: GeometryIdentity | None = None,
    resolve_identity: bool = False,
) -> OperatorGeometry:
    """Project a private compatibility kernel onto the operator boundary."""
    if isinstance(kernel, OperatorGeometry):
        return kernel

    if identity is None:
        signature = kernel.signature
        physical_digest = ""
        registry_digest = ""
        if resolve_identity:
            from imas_ambix.gs.machine_identity import (  # noqa: PLC0415
                MachineIdentityError,
                identity_for_table,
            )

            try:
                resolved = identity_for_table(kernel)
            except MachineIdentityError, ImportError, OSError, TypeError, ValueError:
                pass
            else:
                physical_digest = resolved.physical_digest
                registry_digest = resolved.registry_digest
        identity = GeometryIdentity(
            representation_key=signature.key,
            representation_digest=signature.digest,
            derivation_id=_geometry.GEOMETRY_TABLE_VERSION,
            physical_digest=physical_digest,
            registry_digest=registry_digest,
        )

    return OperatorGeometry(
        identity=identity,
        probes=tuple(kernel.b_probes),
        loops=tuple(kernel.flux_loops),
        conductors=tuple(kernel.pf_filaments),
        passives=tuple(kernel.passive_structures),
        limiter_r=tuple(float(value) for value in kernel.limiter_r),
        limiter_z=tuple(float(value) for value in kernel.limiter_z),
        polygon_sections=_readonly_sections(kernel.polygon_sections),
        drive_map=tuple(kernel.circuit_drives),
        sensor_map=tuple(kernel.sensor_map),
        unmatched_channels=tuple(str(value) for value in kernel.unmatched_amb),
        active_circuits=tuple(int(value) for value in kernel.active_circuits),
        available_current_channels=tuple(
            str(value) for value in kernel.amc_current_channels
        ),
        r0=float(kernel.r0),
        minor_radius=float(kernel.minor_radius),
        unresolved_turns=_unresolved_turns(kernel),
        coil_channels=(),
        coil_column_matrix=_readonly_array(_np.zeros((len(kernel.sensor_map), 0))),
    )


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
            from imas_ambix.gs import operator as _operator  # noqa: PLC0415

            projected = _project_operator_geometry(
                self._compatibility_kernel(addressed_shot),
                identity=self.identity(addressed_shot),
            )
            self._operators[addressed_shot] = _operator.build_operator(projected)
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
            projected = _project_operator_geometry(
                table,
                identity=self.identity(addressed_shot),
            )
            self._operator_projections[addressed_shot] = _dataclasses.replace(
                projected,
                coil_channels=tuple(existing.pf_amc_channels),
                coil_column_matrix=_readonly_array(existing.g_pf),
            )
        return self._operator_projections[addressed_shot]

    def sensors(self, shot: int, channels: _typing.Iterable[str]) -> SensorGeometry:
        """Return sensor features aligned to ``channels`` for ``shot``."""
        addressed_shot = int(shot)
        requested = tuple(str(channel) for channel in channels)
        matrix, kinds = _project_sensor_features(
            self._compatibility_kernel(addressed_shot), requested
        )
        return SensorGeometry(
            identity=self.identity(addressed_shot),
            channels=requested,
            feature_names=_SENSOR_FEATURE_NAMES,
            sensor_kinds=kinds,
            feature_matrix=_readonly_array(matrix),
        )


__all__ = [
    "GeometryIdentity",
    "MachineGeometryService",
    "OperatorGeometry",
    "SensorGeometry",
]

globals().pop("annotations", None)
