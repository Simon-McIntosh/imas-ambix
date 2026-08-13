"""Strict loader for declarative, shot-range-scoped machine maps.

The packaged YAML document is the LinkML authoring contract.  Runtime loading
keeps the dependency surface small by enforcing the same closed-world slots in
Python: a map selects one reusable binding set for one inclusive shot range,
and every binding declares its source location, DD path, units, and sign rule.
No conditional expression or executable hook is accepted.
"""

from __future__ import annotations

import json
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
_SOURCE_STATUSES = {"corpus-observed", "legacy-only"}
_VALIDATION_STATES = {"corpus-validated", "source-only"}


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
        _exact_keys(payload, required, set(), label)
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
    validation_state: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], label: str) -> MachineMap:
        required = {
            "name",
            "machine",
            "first_shot",
            "last_shot",
            "transition",
            "binding_set",
            "validation_state",
        }
        _exact_keys(payload, required, set(), label)
        transition = payload["transition"]
        if transition is not None:
            transition = _text(transition, f"{label}.transition")
        machine_map = cls(
            name=_text(payload["name"], f"{label}.name"),
            machine=_text(payload["machine"], f"{label}.machine"),
            first_shot=_integer(payload["first_shot"], f"{label}.first_shot"),
            last_shot=_integer(payload["last_shot"], f"{label}.last_shot"),
            transition=transition,
            binding_set=_text(payload["binding_set"], f"{label}.binding_set"),
            validation_state=_text(
                payload["validation_state"], f"{label}.validation_state"
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
    """One required field that cannot become an executable DD binding."""

    name: str
    source_group: str
    source_array: str
    source_location: str
    source_shape: tuple[int, ...]
    source_status: str
    source_unit: str
    reason: str
    evidence: str

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
        _exact_keys(payload, required, set(), label)
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
        return cls(source_shape=source_shape, **values)


@dataclass(frozen=True)
class DriveTopology:
    """Declarative relationship between conductors and their current channels."""

    name: str
    source_location: str
    circuit_identity_source: str
    current_scale_source: str
    current_channel_source: str
    circuit_identity_qualification: str
    current_scale_qualification: str
    current_channel_qualification: str
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
            "circuit_identity_qualification",
            "current_scale_qualification",
            "current_channel_qualification",
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
        values = {
            key: _text(payload[key], f"{label}.{key}")
            for key in required.difference({"passive_loop_names"})
        }
        if "://" not in values["source_location"]:
            raise MachineMapError(f"{label}.source_location must be an absolute URI")
        return cls(passive_loop_names=loop_names, **values)


@dataclass(frozen=True)
class StructureAssembly:
    """Bindings assembled into one repeated, typed DD structure."""

    name: str
    structure_path: str
    type_path: str
    type_index: int
    name_binding: str
    member_bindings: tuple[str, ...]
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
        _exact_keys(payload, required, set(), label)
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
            evidence=_text(payload["evidence"], f"{label}.evidence"),
        )


@dataclass(frozen=True)
class MachineMapCatalog:
    """A schema-bound collection of maps, bindings, and explicit qualifications."""

    schema_version: str
    dd_version: str
    source: str
    source_revision: str
    binding_sets: Mapping[str, tuple[ChannelBinding, ...]]
    maps: tuple[MachineMap, ...]
    validation_gaps: tuple[ValidationGap, ...]
    source_qualifications: tuple[SourceQualification, ...]
    drive_topologies: tuple[DriveTopology, ...]
    structure_assemblies: tuple[StructureAssembly, ...]

    def bindings_for(self, machine_map: MachineMap) -> tuple[ChannelBinding, ...]:
        """Resolve the binding set selected by ``machine_map``."""
        return self.binding_sets[machine_map.binding_set]

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
        "BindingSet",
        "ChannelBinding",
        "DriveTopology",
        "MachineMap",
        "SourceQualification",
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
        "binding_sets",
        "maps",
        "validation_gaps",
        "source_qualifications",
        "drive_topologies",
        "structure_assemblies",
    }
    _exact_keys(payload, required, set(), "machine-map catalog")

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
    for topology in topologies:
        referenced = {
            topology.circuit_identity_qualification,
            topology.current_scale_qualification,
            topology.current_channel_qualification,
        }
        if not referenced.issubset(qualification_names):
            raise MachineMapError(
                f"drive topology {topology.name!r} references unknown qualifications"
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
    for assembly in assemblies:
        references = {assembly.name_binding, *assembly.member_bindings}
        if not references.issubset(binding_names):
            raise MachineMapError(
                f"structure assembly {assembly.name!r} references unknown bindings"
            )

    catalog = MachineMapCatalog(
        schema_version=_text(payload["schema_version"], "schema_version"),
        dd_version=_text(payload["dd_version"], "dd_version"),
        source=_text(payload["source"], "source"),
        source_revision=_text(payload["source_revision"], "source_revision"),
        binding_sets=MappingProxyType(binding_sets),
        maps=maps,
        validation_gaps=gaps,
        source_qualifications=qualifications,
        drive_topologies=topologies,
        structure_assemblies=assemblies,
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
    """Require every transition range to have one identically bounded map."""
    map_ranges = {
        (item.first_shot, item.last_shot, item.transition) for item in catalog.maps
    }
    transition_ranges = {
        (item.first_shot, item.last_shot, item.name) for item in transitions
    }
    if map_ranges != transition_ranges:
        raise MachineMapError(
            f"map ranges differ from geometry transitions: "
            f"maps={sorted(map_ranges)}, transitions={sorted(transition_ranges)}"
        )
