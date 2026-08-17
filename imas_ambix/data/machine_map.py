"""Strict loader for declarative, shot-range-scoped machine maps.

The packaged YAML document is the LinkML authoring contract.  Runtime loading
keeps the dependency surface small by enforcing the same closed-world slots in
Python: a map selects one reusable binding set for one inclusive shot range,
and every binding declares its source location, DD path, units, and sign rule.
No conditional expression or executable hook is accepted.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

PACKAGED_MACHINE_MAP_ROOT = Path(__file__).with_name("machine_maps")
LINKML_SCHEMA_PATH = PACKAGED_MACHINE_MAP_ROOT / "schema.yaml"

_SIGN_CONVENTIONS = {
    "identity",
    "negate",
    "not-applicable",
    "unknown-unvalidated",
}
_SOURCE_ROLES = {"value", "identifier", "dimension-coordinate"}
_SOURCE_STATUSES = {"corpus-observed", "legacy-only", "range-absent"}
_VALIDATION_STATES = {"corpus-validated", "source-only"}
_IDENTITY_CASE_RULES = {"case-fold"}
_IDENTITY_NUMERIC_TOKEN_RULES = {"integer-value"}
_COCOS_IDENTIFIERS = {*range(1, 9), *range(11, 19)}
_UNDECLARED_COCOS = 0


class MachineMapError(ValueError):
    """Raised when a machine-map document violates its LinkML contract."""


def _object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MachineMapError(f"{label} must be an object")
    return value


def _exact_keys(
    value: Mapping[str, Any], required: set[str], optional: set[str], label: str
) -> None:
    missing = required.difference(value)
    extra = set(value).difference(required | optional)
    if missing or extra:
        raise MachineMapError(
            f"{label} keys differ: missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise MachineMapError(f"{label} must be non-empty trimmed text")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise MachineMapError(f"{label} must be an integer >= {minimum}")
    return value


def _cocos_identifier(value: Any, label: str) -> int:
    identifier = _integer(value, label, minimum=1)
    if identifier not in _COCOS_IDENTIFIERS:
        raise MachineMapError(
            f"{label} must be a recognised COCOS identifier, got {identifier}"
        )
    return identifier


def _positive_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise MachineMapError(f"{label} must be a number > 0")
    return float(value)


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MachineMapError(f"{label} must be a number")
    return float(value)


def _text_tuple(
    value: Any, label: str, *, allow_empty: bool = False
) -> tuple[str, ...]:
    if not isinstance(value, list) or (not value and not allow_empty):
        qualifier = "a list" if allow_empty else "a non-empty list"
        raise MachineMapError(f"{label} must be {qualifier}")
    items = tuple(_text(item, f"{label}[{index}]") for index, item in enumerate(value))
    if len(items) != len(set(items)):
        raise MachineMapError(f"{label} must be unique")
    return items


def _number_tuple(value: Any, label: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise MachineMapError(f"{label} must be a non-empty list")
    return tuple(_number(item, f"{label}[{index}]") for index, item in enumerate(value))


@dataclass(frozen=True)
class ChannelBinding:
    """One immutable store-array to Data Dictionary path declaration."""

    name: str
    source_group: str
    source_array: str
    source_rank: int
    source_role: str
    source_location: str
    dd_path: str
    source_unit: str
    target_unit: str
    sign_convention: str
    evidence: str
    source_cocos_override: int | None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], label: str) -> ChannelBinding:
        required = {
            "name",
            "source_group",
            "source_array",
            "source_rank",
            "source_role",
            "source_location",
            "dd_path",
            "source_unit",
            "target_unit",
            "sign_convention",
            "evidence",
        }
        _exact_keys(payload, required, {"source_cocos_override"}, label)
        values = {
            key: _text(payload[key], f"{label}.{key}")
            for key in required.difference({"source_rank"})
        }
        if values["sign_convention"] not in _SIGN_CONVENTIONS:
            raise MachineMapError(
                f"{label}.sign_convention must be one of {sorted(_SIGN_CONVENTIONS)}"
            )
        if values["source_role"] not in _SOURCE_ROLES:
            raise MachineMapError(
                f"{label}.source_role must be one of {sorted(_SOURCE_ROLES)}"
            )
        if "://" not in values["source_location"]:
            raise MachineMapError(f"{label}.source_location must be an absolute URI")
        if "/" not in values["dd_path"]:
            raise MachineMapError(f"{label}.dd_path must include IDS and target path")
        return cls(
            source_rank=_integer(payload["source_rank"], f"{label}.source_rank"),
            source_cocos_override=(
                _cocos_identifier(
                    payload["source_cocos_override"],
                    f"{label}.source_cocos_override",
                )
                if "source_cocos_override" in payload
                else None
            ),
            **values,
        )


@dataclass(frozen=True)
class MachineMap:
    """One named machine-map selection valid for an inclusive shot range."""

    name: str
    machine: str
    first_shot: int
    last_shot: int
    transition: str | None
    binding_set: str
    drive_topology: str | None
    description_supplement: str | None
    validation_state: str
    source_representation_signature: str | None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], label: str) -> MachineMap:
        required = {
            "name",
            "machine",
            "first_shot",
            "last_shot",
            "transition",
            "binding_set",
            "drive_topology",
            "description_supplement",
            "validation_state",
        }
        _exact_keys(payload, required, {"source_representation_signature"}, label)
        transition = payload["transition"]
        if transition is not None:
            transition = _text(transition, f"{label}.transition")
        drive_topology = payload["drive_topology"]
        if drive_topology is not None:
            drive_topology = _text(drive_topology, f"{label}.drive_topology")
        description_supplement = payload["description_supplement"]
        if description_supplement is not None:
            description_supplement = _text(
                description_supplement, f"{label}.description_supplement"
            )
        machine_map = cls(
            name=_text(payload["name"], f"{label}.name"),
            machine=_text(payload["machine"], f"{label}.machine"),
            first_shot=_integer(payload["first_shot"], f"{label}.first_shot"),
            last_shot=_integer(payload["last_shot"], f"{label}.last_shot"),
            transition=transition,
            binding_set=_text(payload["binding_set"], f"{label}.binding_set"),
            drive_topology=drive_topology,
            description_supplement=description_supplement,
            validation_state=_text(
                payload["validation_state"], f"{label}.validation_state"
            ),
            source_representation_signature=(
                _text(
                    payload["source_representation_signature"],
                    f"{label}.source_representation_signature",
                )
                if "source_representation_signature" in payload
                else None
            ),
        )
        if machine_map.first_shot > machine_map.last_shot:
            raise MachineMapError(f"{label} has an inverted shot range")
        if machine_map.validation_state not in _VALIDATION_STATES:
            raise MachineMapError(
                f"{label}.validation_state must be one of {sorted(_VALIDATION_STATES)}"
            )
        return machine_map


@dataclass(frozen=True)
class ValidationGap:
    """One binding whose declaration could not be checked against pulse data."""

    binding: str
    reason: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], label: str) -> ValidationGap:
        _exact_keys(payload, {"binding", "reason"}, set(), label)
        return cls(
            binding=_text(payload["binding"], f"{label}.binding"),
            reason=_text(payload["reason"], f"{label}.reason"),
        )


@dataclass(frozen=True)
class SourceQualification:
    """One required field that cannot become an authoritative DD binding."""

    name: str
    source_group: str
    source_array: str
    source_location: str
    source_shape: tuple[int, ...]
    source_status: str
    source_unit: str
    reason: str
    evidence: str
    range_first_shot: int | None
    range_last_shot: int | None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], label: str) -> SourceQualification:
        required = {
            "name",
            "source_group",
            "source_array",
            "source_location",
            "source_shape",
            "source_status",
            "source_unit",
            "reason",
            "evidence",
        }
        range_keys = {"range_first_shot", "range_last_shot"}
        _exact_keys(payload, required, range_keys, label)
        shape = payload["source_shape"]
        if not isinstance(shape, list) or not shape:
            raise MachineMapError(f"{label}.source_shape must be a non-empty list")
        source_shape = tuple(
            _integer(size, f"{label}.source_shape[{index}]", minimum=1)
            for index, size in enumerate(shape)
        )
        values = {
            key: _text(payload[key], f"{label}.{key}")
            for key in required.difference({"source_shape"})
        }
        if "://" not in values["source_location"]:
            raise MachineMapError(f"{label}.source_location must be an absolute URI")
        if values["source_status"] not in _SOURCE_STATUSES:
            raise MachineMapError(
                f"{label}.source_status must be one of {sorted(_SOURCE_STATUSES)}"
            )
        present_range_keys = range_keys.intersection(payload)
        if present_range_keys and present_range_keys != range_keys:
            raise MachineMapError(
                f"{label} must declare both range_first_shot and range_last_shot"
            )
        if values["source_status"] == "range-absent" and not present_range_keys:
            raise MachineMapError(
                f"{label} range-absent qualification requires an explicit shot range"
            )
        if values["source_status"] != "range-absent" and present_range_keys:
            raise MachineMapError(
                f"{label} shot ranges are reserved for range-absent qualifications"
            )
        range_first_shot = (
            _integer(payload["range_first_shot"], f"{label}.range_first_shot")
            if present_range_keys
            else None
        )
        range_last_shot = (
            _integer(payload["range_last_shot"], f"{label}.range_last_shot")
            if present_range_keys
            else None
        )
        if (
            range_first_shot is not None
            and range_last_shot is not None
            and range_last_shot < range_first_shot
        ):
            raise MachineMapError(
                f"{label}.range_last_shot precedes range_first_shot"
            )
        return cls(
            source_shape=source_shape,
            range_first_shot=range_first_shot,
            range_last_shot=range_last_shot,
            **values,
        )


@dataclass(frozen=True)
class SensorIdentityRule:
    """Declarative canonicalisation shared by acquisition sensor identities."""

    name: str
    case_rule: str
    numeric_token_rule: str
    evidence: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], label: str) -> SensorIdentityRule:
        required = {"name", "case_rule", "numeric_token_rule", "evidence"}
        _exact_keys(payload, required, set(), label)
        case_rule = _text(payload["case_rule"], f"{label}.case_rule")
        numeric_token_rule = _text(
            payload["numeric_token_rule"], f"{label}.numeric_token_rule"
        )
        if case_rule not in _IDENTITY_CASE_RULES:
            raise MachineMapError(
                f"{label}.case_rule must be one of {sorted(_IDENTITY_CASE_RULES)}"
            )
        if numeric_token_rule not in _IDENTITY_NUMERIC_TOKEN_RULES:
            raise MachineMapError(
                f"{label}.numeric_token_rule must be one of "
                f"{sorted(_IDENTITY_NUMERIC_TOKEN_RULES)}"
            )
        return cls(
            name=_text(payload["name"], f"{label}.name"),
            case_rule=case_rule,
            numeric_token_rule=numeric_token_rule,
            evidence=_text(payload["evidence"], f"{label}.evidence"),
        )

    def normalise(self, identity: str) -> str:
        """Apply the declared case and numeric-token rules to one identity."""
        normalised = _text(identity, "sensor identity")
        if self.case_rule == "case-fold":
            normalised = normalised.casefold()
        if self.numeric_token_rule == "integer-value":
            normalised = re.sub(
                r"\d+", lambda match: str(int(match.group())), normalised
            )
        return normalised


@dataclass(frozen=True)
class IdentityQualification:
    """One upstream spelling repaired by a declared sensor-identity rule."""

    name: str
    source_location: str
    malformed_identity: str
    canonical_identity: str
    sensor_identity_rule: str
    reason: str
    evidence: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], label: str) -> IdentityQualification:
        required = {
            "name",
            "source_location",
            "malformed_identity",
            "canonical_identity",
            "sensor_identity_rule",
            "reason",
            "evidence",
        }
        _exact_keys(payload, required, set(), label)
        source_location = _text(payload["source_location"], f"{label}.source_location")
        if "://" not in source_location:
            raise MachineMapError(f"{label}.source_location must be an absolute URI")
        return cls(
            name=_text(payload["name"], f"{label}.name"),
            source_location=source_location,
            malformed_identity=_text(
                payload["malformed_identity"], f"{label}.malformed_identity"
            ),
            canonical_identity=_text(
                payload["canonical_identity"], f"{label}.canonical_identity"
            ),
            sensor_identity_rule=_text(
                payload["sensor_identity_rule"], f"{label}.sensor_identity_rule"
            ),
            reason=_text(payload["reason"], f"{label}.reason"),
            evidence=_text(payload["evidence"], f"{label}.evidence"),
        )


@dataclass(frozen=True)
class CircuitCurrentJoin:
    """One circuit's explicit measured-current and conductor identity."""

    circuit_identifier: str
    current_channel: str
    conductor_identifier: str
    evidence: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], label: str) -> CircuitCurrentJoin:
        required = {
            "circuit_identifier",
            "current_channel",
            "conductor_identifier",
            "evidence",
        }
        _exact_keys(payload, required, set(), label)
        return cls(**{key: _text(payload[key], f"{label}.{key}") for key in required})


@dataclass(frozen=True)
class CircuitConnection:
    """One sparse supply-to-element connection in a named circuit."""

    circuit_identifier: str
    supply_identifier: str
    element_identifier: str
    geometry_element_identifier: str
    turns: float
    direction: int

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], label: str) -> CircuitConnection:
        required = {
            "circuit_identifier",
            "supply_identifier",
            "element_identifier",
            "geometry_element_identifier",
            "turns",
            "direction",
        }
        _exact_keys(payload, required, set(), label)
        direction = payload["direction"]
        if isinstance(direction, bool) or direction not in {-1, 1}:
            raise MachineMapError(f"{label}.direction must be -1 or 1")
        return cls(
            circuit_identifier=_text(
                payload["circuit_identifier"], f"{label}.circuit_identifier"
            ),
            supply_identifier=_text(
                payload["supply_identifier"], f"{label}.supply_identifier"
            ),
            element_identifier=_text(
                payload["element_identifier"], f"{label}.element_identifier"
            ),
            geometry_element_identifier=_text(
                payload["geometry_element_identifier"],
                f"{label}.geometry_element_identifier",
            ),
            turns=_positive_number(payload["turns"], f"{label}.turns"),
            direction=direction,
        )


@dataclass(frozen=True)
class DriveTopology:
    """Declarative sparse connectivity from supplies to conductor elements."""

    name: str
    source_location: str
    circuit_identity_source: str
    current_scale_source: str
    current_channel_source: str
    current_channel_declaration: str
    circuit_identifier_path: str
    supply_identifier_path: str
    element_identifier_path: str
    turns_path: str
    connections_path: str
    connections: tuple[CircuitConnection, ...]
    passive_loop_names: tuple[str, ...]
    evidence: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], label: str) -> DriveTopology:
        required = {
            "name",
            "source_location",
            "circuit_identity_source",
            "current_scale_source",
            "current_channel_source",
            "current_channel_declaration",
            "circuit_identifier_path",
            "supply_identifier_path",
            "element_identifier_path",
            "turns_path",
            "connections_path",
            "connections",
            "passive_loop_names",
            "evidence",
        }
        _exact_keys(payload, required, set(), label)
        passive_loop_names = payload["passive_loop_names"]
        if not isinstance(passive_loop_names, list) or not passive_loop_names:
            raise MachineMapError(
                f"{label}.passive_loop_names must be a non-empty list"
            )
        loop_names = tuple(
            _text(name, f"{label}.passive_loop_names[{index}]")
            for index, name in enumerate(passive_loop_names)
        )
        if len(loop_names) != len(set(loop_names)):
            raise MachineMapError(f"{label}.passive_loop_names must be unique")
        connections_raw = payload["connections"]
        if not isinstance(connections_raw, list) or not connections_raw:
            raise MachineMapError(f"{label}.connections must be a non-empty list")
        connections = tuple(
            CircuitConnection.from_dict(
                _object(entry, f"{label}.connections[{index}]"),
                f"{label}.connections[{index}]",
            )
            for index, entry in enumerate(connections_raw)
        )
        connection_keys = {
            (item.circuit_identifier, item.element_identifier) for item in connections
        }
        if len(connection_keys) != len(connections):
            raise MachineMapError(
                f"{label}.connections must identify unique circuit-element pairs"
            )
        values = {
            key: _text(payload[key], f"{label}.{key}")
            for key in required.difference({"passive_loop_names", "connections"})
        }
        if "://" not in values["source_location"]:
            raise MachineMapError(f"{label}.source_location must be an absolute URI")
        return cls(
            passive_loop_names=loop_names,
            connections=connections,
            **values,
        )


@dataclass(frozen=True)
class StructureAssembly:
    """Bindings assembled into one repeated, typed DD structure."""

    name: str
    structure_path: str
    type_path: str
    type_index: int
    name_binding: str
    member_bindings: tuple[str, ...]
    element_identifiers: tuple[str, ...]
    evidence: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], label: str) -> StructureAssembly:
        required = {
            "name",
            "structure_path",
            "type_path",
            "type_index",
            "name_binding",
            "member_bindings",
            "evidence",
        }
        _exact_keys(payload, required, {"element_identifiers"}, label)
        member_bindings = payload["member_bindings"]
        if not isinstance(member_bindings, list) or not member_bindings:
            raise MachineMapError(f"{label}.member_bindings must be a non-empty list")
        members = tuple(
            _text(name, f"{label}.member_bindings[{index}]")
            for index, name in enumerate(member_bindings)
        )
        if len(members) != len(set(members)):
            raise MachineMapError(f"{label}.member_bindings must be unique")
        return cls(
            name=_text(payload["name"], f"{label}.name"),
            structure_path=_text(payload["structure_path"], f"{label}.structure_path"),
            type_path=_text(payload["type_path"], f"{label}.type_path"),
            type_index=_integer(payload["type_index"], f"{label}.type_index"),
            name_binding=_text(payload["name_binding"], f"{label}.name_binding"),
            member_bindings=members,
            element_identifiers=_text_tuple(
                payload.get("element_identifiers", []),
                f"{label}.element_identifiers",
                allow_empty=True,
            ),
            evidence=_text(payload["evidence"], f"{label}.evidence"),
        )


@dataclass(frozen=True)
class AcquisitionDeclaration:
    """Range-compatible acquisition identities retained as catalog data."""

    name: str
    source_location: str
    sensor_identity_key: str
    sensor_identity_rule: str
    current_channels: tuple[str, ...]
    sensor_addresses: tuple[str, ...]
    unmatched_sensor_addresses: tuple[str, ...]
    evidence: str

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any], label: str
    ) -> AcquisitionDeclaration:
        required = {
            "name",
            "source_location",
            "sensor_identity_key",
            "sensor_identity_rule",
            "current_channels",
            "sensor_addresses",
            "unmatched_sensor_addresses",
            "evidence",
        }
        _exact_keys(payload, required, set(), label)
        source_location = _text(payload["source_location"], f"{label}.source_location")
        if "://" not in source_location:
            raise MachineMapError(f"{label}.source_location must be an absolute URI")
        sensor_addresses = _text_tuple(
            payload["sensor_addresses"], f"{label}.sensor_addresses"
        )
        unmatched = _text_tuple(
            payload["unmatched_sensor_addresses"],
            f"{label}.unmatched_sensor_addresses",
            allow_empty=True,
        )
        if not set(unmatched).issubset(sensor_addresses):
            raise MachineMapError(
                f"{label}.unmatched_sensor_addresses must be sensor addresses"
            )
        return cls(
            name=_text(payload["name"], f"{label}.name"),
            source_location=source_location,
            sensor_identity_key=_text(
                payload["sensor_identity_key"], f"{label}.sensor_identity_key"
            ),
            sensor_identity_rule=_text(
                payload["sensor_identity_rule"], f"{label}.sensor_identity_rule"
            ),
            current_channels=_text_tuple(
                payload["current_channels"], f"{label}.current_channels"
            ),
            sensor_addresses=sensor_addresses,
            unmatched_sensor_addresses=unmatched,
            evidence=_text(payload["evidence"], f"{label}.evidence"),
        )


@dataclass(frozen=True)
class PointFluxLoopDeclaration:
    """One static point-loop position absent from emitted pulse arrays."""

    name: str
    acquisition_address: str | None
    r: float
    z: float
    r_path: str
    z_path: str
    type_path: str
    type_index: int
    evidence: str

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any], label: str
    ) -> PointFluxLoopDeclaration:
        required = {
            "name",
            "r",
            "z",
            "r_path",
            "z_path",
            "type_path",
            "type_index",
            "evidence",
        }
        _exact_keys(payload, required, {"acquisition_address"}, label)
        acquisition_address = payload.get("acquisition_address")
        if acquisition_address is not None:
            acquisition_address = _text(
                acquisition_address, f"{label}.acquisition_address"
            )
        return cls(
            name=_text(payload["name"], f"{label}.name"),
            acquisition_address=acquisition_address,
            r=_number(payload["r"], f"{label}.r"),
            z=_number(payload["z"], f"{label}.z"),
            r_path=_text(payload["r_path"], f"{label}.r_path"),
            z_path=_text(payload["z_path"], f"{label}.z_path"),
            type_path=_text(payload["type_path"], f"{label}.type_path"),
            type_index=_integer(payload["type_index"], f"{label}.type_index"),
            evidence=_text(payload["evidence"], f"{label}.evidence"),
        )


@dataclass(frozen=True)
class PolygonSectionDeclaration:
    """A shaped conductor section joined to circuit and geometry identities."""

    name: str
    circuit_identifier: str
    geometry_element_identifier: str
    vertex_r: tuple[float, ...]
    vertex_z: tuple[float, ...]
    current_scale: float
    evidence: str

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any], label: str
    ) -> PolygonSectionDeclaration:
        required = {
            "name",
            "circuit_identifier",
            "geometry_element_identifier",
            "vertex_r",
            "vertex_z",
            "current_scale",
            "evidence",
        }
        _exact_keys(payload, required, set(), label)
        vertex_r = _number_tuple(payload["vertex_r"], f"{label}.vertex_r")
        vertex_z = _number_tuple(payload["vertex_z"], f"{label}.vertex_z")
        if len(vertex_r) != len(vertex_z) or len(vertex_r) < 3:
            raise MachineMapError(
                f"{label}.vertex_r and vertex_z must describe the same polygon"
            )
        return cls(
            name=_text(payload["name"], f"{label}.name"),
            circuit_identifier=_text(
                payload["circuit_identifier"], f"{label}.circuit_identifier"
            ),
            geometry_element_identifier=_text(
                payload["geometry_element_identifier"],
                f"{label}.geometry_element_identifier",
            ),
            vertex_r=vertex_r,
            vertex_z=vertex_z,
            current_scale=_number(payload["current_scale"], f"{label}.current_scale"),
            evidence=_text(payload["evidence"], f"{label}.evidence"),
        )


@dataclass(frozen=True)
class DescriptionSupplement:
    """Static description values selected with one range-scoped machine map."""

    name: str
    source_location: str
    acquisition_declaration: str
    point_flux_loops: tuple[PointFluxLoopDeclaration, ...]
    reference_radius: float
    reference_radius_path: str
    reference_radius_unit: str
    minor_radius_qualification: str
    polygon_sections: tuple[PolygonSectionDeclaration, ...]
    evidence: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], label: str) -> DescriptionSupplement:
        required = {
            "name",
            "source_location",
            "acquisition_declaration",
            "point_flux_loops",
            "reference_radius",
            "reference_radius_path",
            "reference_radius_unit",
            "minor_radius_qualification",
            "polygon_sections",
            "evidence",
        }
        _exact_keys(payload, required, set(), label)
        source_location = _text(payload["source_location"], f"{label}.source_location")
        if "://" not in source_location:
            raise MachineMapError(f"{label}.source_location must be an absolute URI")
        loops_raw = payload["point_flux_loops"]
        if not isinstance(loops_raw, list):
            raise MachineMapError(f"{label}.point_flux_loops must be a list")
        polygons_raw = payload["polygon_sections"]
        if not isinstance(polygons_raw, list):
            raise MachineMapError(f"{label}.polygon_sections must be a list")
        return cls(
            name=_text(payload["name"], f"{label}.name"),
            source_location=source_location,
            acquisition_declaration=_text(
                payload["acquisition_declaration"],
                f"{label}.acquisition_declaration",
            ),
            point_flux_loops=tuple(
                PointFluxLoopDeclaration.from_dict(
                    _object(item, f"{label}.point_flux_loops[{index}]"),
                    f"{label}.point_flux_loops[{index}]",
                )
                for index, item in enumerate(loops_raw)
            ),
            reference_radius=_positive_number(
                payload["reference_radius"], f"{label}.reference_radius"
            ),
            reference_radius_path=_text(
                payload["reference_radius_path"], f"{label}.reference_radius_path"
            ),
            reference_radius_unit=_text(
                payload["reference_radius_unit"], f"{label}.reference_radius_unit"
            ),
            minor_radius_qualification=_text(
                payload["minor_radius_qualification"],
                f"{label}.minor_radius_qualification",
            ),
            polygon_sections=tuple(
                PolygonSectionDeclaration.from_dict(
                    _object(item, f"{label}.polygon_sections[{index}]"),
                    f"{label}.polygon_sections[{index}]",
                )
                for index, item in enumerate(polygons_raw)
            ),
            evidence=_text(payload["evidence"], f"{label}.evidence"),
        )


@dataclass(frozen=True)
class MachineMapCatalog:
    """A schema-bound collection of maps, bindings, and explicit qualifications."""

    schema_version: str
    dd_version: str
    source: str
    source_revision: str
    source_cocos: int | None
    binding_sets: Mapping[str, tuple[ChannelBinding, ...]]
    maps: tuple[MachineMap, ...]
    validation_gaps: tuple[ValidationGap, ...]
    source_qualifications: tuple[SourceQualification, ...]
    sensor_identity_rules: tuple[SensorIdentityRule, ...]
    identity_qualifications: tuple[IdentityQualification, ...]
    drive_topologies: tuple[DriveTopology, ...]
    structure_assemblies: tuple[StructureAssembly, ...]
    acquisition_declarations: tuple[AcquisitionDeclaration, ...]
    description_supplements: tuple[DescriptionSupplement, ...]
    circuit_current_joins: tuple[CircuitCurrentJoin, ...] = ()

    def cocos_for_binding(self, binding: ChannelBinding | None = None) -> int | None:
        """Resolve a binding override before the machine-level declaration."""
        if binding is not None and binding.source_cocos_override is not None:
            return binding.source_cocos_override
        return self.source_cocos

    def bindings_for(self, machine_map: MachineMap) -> tuple[ChannelBinding, ...]:
        """Resolve the binding set selected by ``machine_map``."""
        return self.binding_sets[machine_map.binding_set]

    def normalise_sensor_identity(
        self, acquisition: AcquisitionDeclaration, identity: str
    ) -> str:
        """Resolve one acquisition identity through its declared reusable rule."""
        rule = next(
            (
                item
                for item in self.sensor_identity_rules
                if item.name == acquisition.sensor_identity_rule
            ),
            None,
        )
        if rule is None:
            raise MachineMapError(
                f"acquisition {acquisition.name!r} references unknown sensor identity "
                f"rule {acquisition.sensor_identity_rule!r}"
            )
        return rule.normalise(identity)

    @property
    def bound_channel_count(self) -> int:
        """Count unique declarations across reusable binding sets."""
        return sum(len(bindings) for bindings in self.binding_sets.values())

    @property
    def bound_channel_counts(self) -> Mapping[str, int]:
        """Count unique declarations by immutable source group."""
        counts: dict[str, int] = {}
        for bindings in self.binding_sets.values():
            for binding in bindings:
                counts[binding.source_group] = counts.get(binding.source_group, 0) + 1
        return MappingProxyType(counts)

    @property
    def qualified_channel_count(self) -> int:
        """Count required fields deliberately excluded from DD bindings."""
        return len(self.source_qualifications)

    @property
    def qualified_channel_counts(self) -> Mapping[str, int]:
        """Count explicit field qualifications by immutable source group."""
        counts: dict[str, int] = {}
        for qualification in self.source_qualifications:
            counts[qualification.source_group] = (
                counts.get(qualification.source_group, 0) + 1
            )
        return MappingProxyType(counts)


def load_linkml_schema(path: Path | str = LINKML_SCHEMA_PATH) -> Mapping[str, Any]:
    """Load and structurally verify the packaged LinkML schema document."""
    payload = yaml.safe_load(Path(path).read_text())
    schema = _object(payload, "LinkML schema")
    required_classes = {
        "AcquisitionDeclaration",
        "BindingSet",
        "ChannelBinding",
        "CircuitCurrentJoin",
        "CircuitConnection",
        "DescriptionSupplement",
        "DriveTopology",
        "IdentityQualification",
        "MachineMap",
        "PointFluxLoopDeclaration",
        "PolygonSectionDeclaration",
        "SourceQualification",
        "SensorIdentityRule",
        "StructureAssembly",
        "ValidationGap",
        "MachineMapCatalog",
    }
    classes = _object(schema.get("classes"), "LinkML schema.classes")
    if not required_classes.issubset(classes):
        raise MachineMapError(
            "LinkML schema lacks classes: "
            f"{sorted(required_classes.difference(classes))}"
        )
    if schema.get("default_range") != "string":
        raise MachineMapError("LinkML schema.default_range must be 'string'")
    return schema


def load_machine_map(path: Path | str) -> MachineMapCatalog:
    """Load a JSON machine-map catalog and validate every declared slot."""
    load_linkml_schema()
    try:
        raw = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise MachineMapError(f"cannot read machine map {path!s}: {error}") from error
    payload = _object(raw, "machine-map catalog")
    required = {
        "schema_version",
        "dd_version",
        "source",
        "source_revision",
        "source_cocos",
        "binding_sets",
        "maps",
        "validation_gaps",
        "source_qualifications",
        "sensor_identity_rules",
        "identity_qualifications",
        "circuit_current_joins",
        "drive_topologies",
        "structure_assemblies",
        "acquisition_declarations",
        "description_supplements",
    }
    _exact_keys(payload, required, set(), "machine-map catalog")

    raw_source_cocos = payload["source_cocos"]
    if raw_source_cocos == _UNDECLARED_COCOS:
        source_cocos = None
    else:
        source_cocos = _cocos_identifier(raw_source_cocos, "source_cocos")

    binding_sets_raw = payload["binding_sets"]
    if not isinstance(binding_sets_raw, list) or not binding_sets_raw:
        raise MachineMapError("binding_sets must be a non-empty list")
    binding_sets: dict[str, tuple[ChannelBinding, ...]] = {}
    binding_names: set[str] = set()
    for set_index, raw_set in enumerate(binding_sets_raw):
        set_context = f"binding_sets[{set_index}]"
        binding_set = _object(raw_set, set_context)
        _exact_keys(binding_set, {"name", "bindings"}, set(), set_context)
        name = _text(binding_set["name"], f"{set_context}.name")
        if name in binding_sets:
            raise MachineMapError(f"duplicate binding set {name!r}")
        entries = binding_set["bindings"]
        if not isinstance(entries, list) or not entries:
            raise MachineMapError(f"{set_context}.bindings must be a non-empty list")
        bindings = tuple(
            ChannelBinding.from_dict(
                _object(entry, f"{set_context}.bindings[{index}]"),
                f"{set_context}.bindings[{index}]",
            )
            for index, entry in enumerate(entries)
        )
        for binding in bindings:
            if binding.name in binding_names:
                raise MachineMapError(f"duplicate binding name {binding.name!r}")
            binding_names.add(binding.name)
        binding_sets[name] = bindings

    maps_raw = payload["maps"]
    if not isinstance(maps_raw, list) or not maps_raw:
        raise MachineMapError("maps must be a non-empty list")
    maps = tuple(
        MachineMap.from_dict(_object(entry, f"maps[{index}]"), f"maps[{index}]")
        for index, entry in enumerate(maps_raw)
    )
    map_names = [item.name for item in maps]
    if len(map_names) != len(set(map_names)):
        raise MachineMapError("map names must be unique")
    for item in maps:
        if item.binding_set not in binding_sets:
            raise MachineMapError(
                f"map {item.name!r} references unknown binding set {item.binding_set!r}"
            )

    gaps_raw = payload["validation_gaps"]
    if not isinstance(gaps_raw, list):
        raise MachineMapError("validation_gaps must be a list")
    gaps = tuple(
        ValidationGap.from_dict(
            _object(entry, f"validation_gaps[{index}]"),
            f"validation_gaps[{index}]",
        )
        for index, entry in enumerate(gaps_raw)
    )
    gap_names = [gap.binding for gap in gaps]
    if len(gap_names) != len(set(gap_names)):
        raise MachineMapError("validation gap bindings must be unique")
    if not set(gap_names).issubset(binding_names):
        raise MachineMapError("validation gaps must name declared bindings")

    qualifications_raw = payload["source_qualifications"]
    if not isinstance(qualifications_raw, list):
        raise MachineMapError("source_qualifications must be a list")
    qualifications = tuple(
        SourceQualification.from_dict(
            _object(entry, f"source_qualifications[{index}]"),
            f"source_qualifications[{index}]",
        )
        for index, entry in enumerate(qualifications_raw)
    )
    qualification_names = [item.name for item in qualifications]
    if len(qualification_names) != len(set(qualification_names)):
        raise MachineMapError("source qualification names must be unique")
    if set(qualification_names).intersection(binding_names):
        raise MachineMapError("source qualifications must not be declared bindings")
    bound_sources = {
        (binding.source_group, binding.source_array)
        for bindings in binding_sets.values()
        for binding in bindings
    }
    qualified_sources = {
        (item.source_group, item.source_array) for item in qualifications
    }
    if len(qualified_sources) != len(qualifications):
        raise MachineMapError("source qualifications must identify unique arrays")
    if qualified_sources.intersection(bound_sources):
        raise MachineMapError("qualified source arrays must not also be bound")

    identity_rules_raw = payload["sensor_identity_rules"]
    if not isinstance(identity_rules_raw, list):
        raise MachineMapError("sensor_identity_rules must be a list")
    identity_rules = tuple(
        SensorIdentityRule.from_dict(
            _object(entry, f"sensor_identity_rules[{index}]"),
            f"sensor_identity_rules[{index}]",
        )
        for index, entry in enumerate(identity_rules_raw)
    )
    identity_rule_names = [item.name for item in identity_rules]
    if len(identity_rule_names) != len(set(identity_rule_names)):
        raise MachineMapError("sensor identity rule names must be unique")

    identity_qualifications_raw = payload["identity_qualifications"]
    if not isinstance(identity_qualifications_raw, list):
        raise MachineMapError("identity_qualifications must be a list")
    identity_qualifications = tuple(
        IdentityQualification.from_dict(
            _object(entry, f"identity_qualifications[{index}]"),
            f"identity_qualifications[{index}]",
        )
        for index, entry in enumerate(identity_qualifications_raw)
    )
    identity_qualification_names = [item.name for item in identity_qualifications]
    if len(identity_qualification_names) != len(set(identity_qualification_names)):
        raise MachineMapError("identity qualification names must be unique")
    identity_rules_by_name = {item.name: item for item in identity_rules}
    for qualification in identity_qualifications:
        if qualification.sensor_identity_rule not in identity_rules_by_name:
            raise MachineMapError(
                f"identity qualification {qualification.name!r} references unknown "
                "sensor identity rule"
            )
        rule = identity_rules_by_name[qualification.sensor_identity_rule]
        if qualification.malformed_identity == qualification.canonical_identity:
            raise MachineMapError(
                f"identity qualification {qualification.name!r} must change spelling"
            )
        if rule.normalise(qualification.malformed_identity) != rule.normalise(
            qualification.canonical_identity
        ):
            raise MachineMapError(
                f"identity qualification {qualification.name!r} is not resolved by "
                f"rule {rule.name!r}"
            )

    topologies_raw = payload["drive_topologies"]
    if not isinstance(topologies_raw, list):
        raise MachineMapError("drive_topologies must be a list")
    topologies = tuple(
        DriveTopology.from_dict(
            _object(entry, f"drive_topologies[{index}]"),
            f"drive_topologies[{index}]",
        )
        for index, entry in enumerate(topologies_raw)
    )
    topology_names = [item.name for item in topologies]
    if len(topology_names) != len(set(topology_names)):
        raise MachineMapError("drive topology names must be unique")
    for item in maps:
        if (
            item.drive_topology is not None
            and item.drive_topology not in topology_names
        ):
            raise MachineMapError(
                f"map {item.name!r} references unknown drive topology "
                f"{item.drive_topology!r}"
            )

    assemblies_raw = payload["structure_assemblies"]
    if not isinstance(assemblies_raw, list):
        raise MachineMapError("structure_assemblies must be a list")
    assemblies = tuple(
        StructureAssembly.from_dict(
            _object(entry, f"structure_assemblies[{index}]"),
            f"structure_assemblies[{index}]",
        )
        for index, entry in enumerate(assemblies_raw)
    )
    assembly_names = [item.name for item in assemblies]
    if len(assembly_names) != len(set(assembly_names)):
        raise MachineMapError("structure assembly names must be unique")
    geometry_element_identifiers = {
        identifier
        for assembly in assemblies
        for identifier in assembly.element_identifiers
    }
    if sum(len(item.element_identifiers) for item in assemblies) != len(
        geometry_element_identifiers
    ):
        raise MachineMapError("structure element identifiers must be globally unique")
    for assembly in assemblies:
        references = {assembly.name_binding, *assembly.member_bindings}
        if not references.issubset(binding_names):
            raise MachineMapError(
                f"structure assembly {assembly.name!r} references unknown bindings"
            )
    for topology in topologies:
        unresolved = {
            item.geometry_element_identifier for item in topology.connections
        }.difference(geometry_element_identifiers)
        if unresolved:
            raise MachineMapError(
                f"drive topology {topology.name!r} has unresolved geometry element "
                f"identifiers: {sorted(unresolved)[:3]}"
            )

    acquisitions_raw = payload["acquisition_declarations"]
    if not isinstance(acquisitions_raw, list):
        raise MachineMapError("acquisition_declarations must be a list")
    acquisitions = tuple(
        AcquisitionDeclaration.from_dict(
            _object(entry, f"acquisition_declarations[{index}]"),
            f"acquisition_declarations[{index}]",
        )
        for index, entry in enumerate(acquisitions_raw)
    )
    acquisition_names = [item.name for item in acquisitions]
    if len(acquisition_names) != len(set(acquisition_names)):
        raise MachineMapError("acquisition declaration names must be unique")
    for acquisition in acquisitions:
        if acquisition.sensor_identity_rule not in identity_rules_by_name:
            raise MachineMapError(
                f"acquisition declaration {acquisition.name!r} references unknown "
                "sensor identity rule"
            )
        rule = identity_rules_by_name[acquisition.sensor_identity_rule]
        normalised_addresses = tuple(
            rule.normalise(address) for address in acquisition.sensor_addresses
        )
        if len(normalised_addresses) != len(set(normalised_addresses)):
            raise MachineMapError(
                f"acquisition declaration {acquisition.name!r} has colliding "
                "normalised sensor identities"
            )
    declared_sensor_addresses = {
        address
        for acquisition in acquisitions
        for address in acquisition.sensor_addresses
    }
    for qualification in identity_qualifications:
        if qualification.canonical_identity not in declared_sensor_addresses:
            raise MachineMapError(
                f"identity qualification {qualification.name!r} canonical identity "
                "is not declared by an acquisition"
            )
    for topology in topologies:
        if topology.current_channel_declaration not in acquisition_names:
            raise MachineMapError(
                f"drive topology {topology.name!r} references unknown acquisition "
                "declaration"
            )

    joins_raw = payload["circuit_current_joins"]
    if not isinstance(joins_raw, list):
        raise MachineMapError("circuit_current_joins must be a list")
    circuit_current_joins = tuple(
        CircuitCurrentJoin.from_dict(
            _object(entry, f"circuit_current_joins[{index}]"),
            f"circuit_current_joins[{index}]",
        )
        for index, entry in enumerate(joins_raw)
    )
    join_circuits = [item.circuit_identifier for item in circuit_current_joins]
    join_channels = [item.current_channel for item in circuit_current_joins]
    join_conductors = [item.conductor_identifier for item in circuit_current_joins]
    for join_label, identifiers in (
        ("circuit identifiers", join_circuits),
        ("current channels", join_channels),
        ("conductor identifiers", join_conductors),
    ):
        if len(identifiers) != len(set(identifiers)):
            raise MachineMapError(f"circuit current join {join_label} must be unique")
    acquisitions_by_name = {item.name: item for item in acquisitions}
    for topology in topologies:
        declared_circuits = {item.circuit_identifier for item in topology.connections}
        unresolved_circuits = set(join_circuits).difference(declared_circuits)
        if unresolved_circuits:
            raise MachineMapError(
                f"drive topology {topology.name!r} lacks joined circuits: "
                f"{sorted(unresolved_circuits)}"
            )
        acquisition = acquisitions_by_name[topology.current_channel_declaration]
        unresolved_channels = set(join_channels).difference(
            acquisition.current_channels
        )
        if unresolved_channels:
            raise MachineMapError(
                f"acquisition declaration {acquisition.name!r} lacks joined "
                f"current channels: {sorted(unresolved_channels)}"
            )
    if circuit_current_joins and not topologies:
        raise MachineMapError("circuit current joins require a drive topology")

    supplements_raw = payload["description_supplements"]
    if not isinstance(supplements_raw, list):
        raise MachineMapError("description_supplements must be a list")
    supplements = tuple(
        DescriptionSupplement.from_dict(
            _object(entry, f"description_supplements[{index}]"),
            f"description_supplements[{index}]",
        )
        for index, entry in enumerate(supplements_raw)
    )
    supplement_names = [item.name for item in supplements]
    if len(supplement_names) != len(set(supplement_names)):
        raise MachineMapError("description supplement names must be unique")
    for supplement in supplements:
        if supplement.acquisition_declaration not in acquisition_names:
            raise MachineMapError(
                f"description supplement {supplement.name!r} references unknown "
                "acquisition declaration"
            )
        if supplement.minor_radius_qualification not in qualification_names:
            raise MachineMapError(
                f"description supplement {supplement.name!r} references unknown "
                "minor-radius qualification"
            )
        unresolved_polygons = {
            item.geometry_element_identifier for item in supplement.polygon_sections
        }.difference(geometry_element_identifiers)
        if unresolved_polygons:
            raise MachineMapError(
                f"description supplement {supplement.name!r} has unresolved polygon "
                f"elements: {sorted(unresolved_polygons)}"
            )
    for item in maps:
        if (
            item.description_supplement is not None
            and item.description_supplement not in supplement_names
        ):
            raise MachineMapError(
                f"map {item.name!r} references unknown description supplement "
                f"{item.description_supplement!r}"
            )

    catalog = MachineMapCatalog(
        schema_version=_text(payload["schema_version"], "schema_version"),
        dd_version=_text(payload["dd_version"], "dd_version"),
        source=_text(payload["source"], "source"),
        source_revision=_text(payload["source_revision"], "source_revision"),
        source_cocos=source_cocos,
        binding_sets=MappingProxyType(binding_sets),
        maps=maps,
        validation_gaps=gaps,
        source_qualifications=qualifications,
        sensor_identity_rules=identity_rules,
        identity_qualifications=identity_qualifications,
        drive_topologies=topologies,
        structure_assemblies=assemblies,
        acquisition_declarations=acquisitions,
        description_supplements=supplements,
        circuit_current_joins=circuit_current_joins,
    )
    source_only = {
        binding.name
        for item in maps
        if item.validation_state == "source-only"
        for binding in catalog.bindings_for(item)
    }
    if source_only != set(gap_names):
        raise MachineMapError(
            "source-only bindings and validation_gaps must agree exactly"
        )
    return catalog


def load_packaged_machine_map(machine: str) -> MachineMapCatalog:
    """Load the reviewed catalog for ``machine`` from the package."""
    component = _text(machine, "machine")
    if not component.replace("-", "").isalnum():
        raise MachineMapError("machine must contain only letters, digits, or hyphens")
    return load_machine_map(PACKAGED_MACHINE_MAP_ROOT / f"{component}.json")


def map_for_shot(catalog: MachineMapCatalog, shot: int) -> MachineMap:
    """Return the single range-scoped map covering ``shot``."""
    matches = [
        item for item in catalog.maps if item.first_shot <= shot <= item.last_shot
    ]
    if len(matches) != 1:
        raise LookupError(f"shot {shot} resolves to {len(matches)} machine maps")
    return matches[0]


def assert_transition_alignment(
    catalog: MachineMapCatalog, transitions: Sequence[Any]
) -> None:
    """Require maps to cover each transition with contiguous contained ranges."""
    transitions_by_name = {item.name: item for item in transitions}
    if len(transitions_by_name) != len(transitions):
        raise MachineMapError("geometry transition names must be unique")

    errors: list[str] = []
    for machine_map in catalog.maps:
        transition = transitions_by_name.get(machine_map.transition)
        if transition is None:
            errors.append(
                f"map {machine_map.name!r} names unknown transition "
                f"{machine_map.transition!r}"
            )
            continue
        if not (
            transition.first_shot <= machine_map.first_shot
            and machine_map.last_shot <= transition.last_shot
        ):
            errors.append(
                f"map {machine_map.name!r} range "
                f"{machine_map.first_shot}-{machine_map.last_shot} lies outside "
                f"transition {transition.name!r} range "
                f"{transition.first_shot}-{transition.last_shot}"
            )

    for transition in transitions:
        ranges = sorted(
            (
                machine_map.first_shot,
                machine_map.last_shot,
                machine_map.name,
            )
            for machine_map in catalog.maps
            if machine_map.transition == transition.name
        )
        expected_first = transition.first_shot
        for first_shot, last_shot, name in ranges:
            if first_shot != expected_first:
                errors.append(
                    f"transition {transition.name!r} expected map coverage at "
                    f"shot {expected_first}, got {name!r} at shot {first_shot}"
                )
            expected_first = last_shot + 1
        if expected_first != transition.last_shot + 1:
            errors.append(
                f"transition {transition.name!r} map coverage ends at "
                f"{expected_first - 1}, expected {transition.last_shot}"
            )

    if errors:
        raise MachineMapError(
            "map ranges differ from geometry transitions: " + "; ".join(errors)
        )
