"""Ambix metadata adapters for Nova conductor and circuit representations."""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
from nova.biot.coupling import (
    CaseChannel,
    CircuitClass,
    CircuitCoupling,
    CircuitTable,
    CoilChannel,
    CouplingColumn,
    CouplingPlan,
    classify_circuits,
    couple_circuits,
)
from nova.circuit import (
    ConductorSet,
    CoreCells,
    CoupledCircuit,
    PassiveCircuit,
    PassiveCircuitSystem,
    PassiveEigenbasis,
    PatchTiling,
    PlasmaCircuit,
    PolygonSection,
    ScreeningBasis,
    SensorSet,
    build_passive_circuit_system,
    build_passive_eigenbasis,
    build_plasma_circuit,
    build_plasma_circuit_from_state,
    reduce_passive_system,
    screening_eigenbasis,
    screening_trajectory,
)

Record = Mapping[str, Any] | object


def _optional_value(record: Record, *names: str, default: Any = None) -> Any:
    """Read the first available mapping key or object attribute."""
    for name in names:
        if isinstance(record, Mapping):
            if name in record:
                return record[name]
        elif hasattr(record, name):
            return getattr(record, name)
    return default


def _required_value(record: Record, *names: str) -> Any:
    """Read one required field, naming all accepted spellings on failure."""
    value = _optional_value(record, *names)
    if value is None:
        joined = ", ".join(names)
        raise KeyError(f"record does not provide any of: {joined}")
    return value


def _coil_channels(record: Record) -> tuple[str, ...]:
    """Return a stable preference order from Ambix or generic channel fields."""
    explicit = _optional_value(record, "channels")
    if explicit is not None:
        return tuple(str(channel) for channel in explicit if channel)
    preferred = _optional_value(record, "preferred_current_channel")
    if callable(preferred):
        preferred = preferred()
    candidates = (
        preferred,
        _optional_value(record, "l1_coil_channel"),
        _optional_value(record, "l1_feed_channel"),
    )
    return tuple(dict.fromkeys(str(channel) for channel in candidates if channel))


def circuit_table_from_metadata(
    active_circuits: Sequence[Record],
    case_circuits: Sequence[Record],
    coil_centroids: Mapping[str, Sequence[float]],
    *,
    match_radius: float = 0.08,
) -> CircuitTable:
    """Translate Ambix circuit records into Nova's machine-description table."""
    coils = []
    for record in active_circuits:
        label = str(_required_value(record, "coil_label", "label"))
        centroid = coil_centroids[label]
        channels = _coil_channels(record)
        if not channels:
            raise ValueError(f"active circuit {label!r} has no current channel")
        coils.append(
            CoilChannel(
                label=label,
                centroid=(float(centroid[0]), float(centroid[1])),
                channels=channels,
            )
        )
    cases = tuple(
        CaseChannel(
            circuit=int(_required_value(record, "circuit_id", "circuit")),
            coil_label=str(
                _required_value(
                    record,
                    "geometry_confusable_with",
                    "coil_label",
                )
            ),
            channel=_optional_value(record, "l1_case_channel", "channel"),
            constrained_zero=bool(
                _optional_value(record, "constrained_zero", default=False)
            ),
        )
        for record in case_circuits
    )
    return CircuitTable(
        coils=tuple(coils),
        cases=cases,
        match_radius=float(match_radius),
    )


def emit_circuit_coupling(
    filaments: Sequence[Record],
    channels: Sequence[str],
    table: CircuitTable,
    *,
    measured_channels: Iterable[str] = (),
    polygon_sections: Sequence[Record] = (),
) -> CircuitCoupling:
    """Classify Ambix filament records and emit Nova's circuit-tier payload."""
    r = np.asarray([_required_value(row, "r") for row in filaments], dtype=np.float64)
    z = np.asarray([_required_value(row, "z") for row in filaments], dtype=np.float64)
    dr = np.asarray(
        [_required_value(row, "width", "dr") for row in filaments],
        dtype=np.float64,
    )
    dz = np.asarray(
        [_required_value(row, "height", "dz") for row in filaments],
        dtype=np.float64,
    )
    current_share = np.asarray(
        [_required_value(row, "xmult", "current_share") for row in filaments],
        dtype=np.float64,
    )
    circuit = np.asarray(
        [_required_value(row, "circuit", "circuit_id") for row in filaments],
        dtype=np.int64,
    )
    sections = tuple(
        PolygonSection(
            circuit=int(_required_value(section, "circuit")),
            vertices=np.asarray(
                _required_value(section, "vertices"),
                dtype=np.float64,
            ),
            current_share=float(_optional_value(section, "current_share", default=1.0)),
        )
        for section in polygon_sections
    )
    classes = classify_circuits(circuit, r, z, current_share, channels, table)
    return couple_circuits(classes).emit(
        r=r,
        z=z,
        dr=dr,
        dz=dz,
        current_share=current_share,
        circuit=circuit,
        measured_channels=measured_channels,
        polygon_sections=sections,
    )


__all__ = [
    "CaseChannel",
    "CircuitClass",
    "CircuitCoupling",
    "CircuitTable",
    "CoilChannel",
    "ConductorSet",
    "CoreCells",
    "CoupledCircuit",
    "CouplingColumn",
    "CouplingPlan",
    "PassiveCircuit",
    "PassiveCircuitSystem",
    "PassiveEigenbasis",
    "PatchTiling",
    "PlasmaCircuit",
    "PolygonSection",
    "ScreeningBasis",
    "SensorSet",
    "build_passive_circuit_system",
    "build_passive_eigenbasis",
    "build_plasma_circuit",
    "build_plasma_circuit_from_state",
    "circuit_table_from_metadata",
    "classify_circuits",
    "couple_circuits",
    "emit_circuit_coupling",
    "reduce_passive_system",
    "screening_eigenbasis",
    "screening_trajectory",
]
