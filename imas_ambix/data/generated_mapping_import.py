"""Import generated facility mappings against ordered machine-description members."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from imas_alambic.signal_map import MAP_SCHEMA_VERSION, SignalMap, SignalRule

if TYPE_CHECKING:
    from imas_alambic.machine_map import MachineMapCatalog, SensorIdentityRule


class GeneratedMappingImportError(ValueError):
    """The handoff or the identity context cannot be interpreted safely."""


# The Data Dictionary's cocos_label_transformation vocabulary. A label whose
# value changes under a COCOS transformation leaves the target's sign
# undetermined while the machine's source COCOS is undeclared; such a row is
# held out of the draft maps as pending. A neutral label carries no
# transformation, so the row imports as the hand-off stands.
COCOS_NEUTRAL_LABELS = frozenset({"none", "one_like"})
COCOS_DEPENDENT_LABELS = frozenset(
    {
        "b0_like",
        "dodpsi_like",
        "ip_like",
        "pol_angle_like",
        "psi_like",
        "q_like",
        "tor_angle_like",
    }
)


def _cocos_dependent_label(label: str) -> bool:
    return label in COCOS_DEPENDENT_LABELS or label.startswith("grid_type")


def _pending_reason(label: str, source: str | None) -> str:
    origin = source if source is not None else "a source it does not name"
    return (
        f"COCOS-dependent label {label!r} from {origin} is withheld pending "
        f"an undeclared source COCOS"
    )


@dataclass(frozen=True)
class UnresolvedMapping:
    """A handoff row withheld from a draft map, with its specific reason."""

    ids_name: str
    source_id: str
    source_array: str | None
    target_path: str
    reason: str


@dataclass(frozen=True)
class ResolvedTimeBinding:
    """A handoff time row confirmed by a draft value rule for its channel."""

    ids_name: str
    source_id: str
    source_array: str
    target_path: str
    value_rule: SignalRule


@dataclass(frozen=True)
class PendingCocosRule:
    """A COCOS-dependent row held out of the draft maps until COCOS is declared."""

    ids_name: str
    source_id: str
    source_array: str
    target_path: str
    cocos_label: str
    cocos_label_source: str | None
    open_source_cocos: int | None
    reason: str
    rule: SignalRule


@dataclass(frozen=True)
class GeneratedMappingImport:
    """Draft maps, confirmed time vectors, pending-COCOS rules, and withheld rows."""

    maps: tuple[SignalMap, ...]
    time_bindings: tuple[ResolvedTimeBinding, ...]
    unresolved: tuple[UnresolvedMapping, ...]
    pending: tuple[PendingCocosRule, ...] = ()


def _document(source: Path | str | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(source, Mapping):
        return source
    try:
        value = json.loads(Path(source).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GeneratedMappingImportError(
            f"cannot read mapping handoff: {error}"
        ) from error
    if not isinstance(value, Mapping):
        raise GeneratedMappingImportError("mapping handoff must be an object")
    return value


def _identity_rule(
    catalogue: MachineMapCatalog, name: str | None
) -> SensorIdentityRule:
    rules = catalogue.sensor_identity_rules
    if name is None:
        if len(rules) != 1:
            raise GeneratedMappingImportError(
                "select an identity rule when the catalogue has zero or multiple rules"
            )
        return rules[0]
    matches = [rule for rule in rules if rule.name == name]
    if len(matches) != 1:
        raise GeneratedMappingImportError(f"unknown sensor identity rule {name!r}")
    return matches[0]


def _target_index(
    row: Mapping[str, Any],
    description_members: Mapping[str, Sequence[str]],
    rule: SensorIdentityRule,
) -> tuple[int | None, str | None]:
    path = row.get("target_path")
    member = row.get("member_identifier")
    source_array = row.get("source_array")
    if not all(
        isinstance(value, str) and value for value in (path, member, source_array)
    ):
        return None, "target path, member identifier, and source array are required"
    matching_paths = [key for key in description_members if path.startswith(f"{key}/")]
    if not matching_paths:
        return None, f"no machine-description member array covers {path}"
    structure = max(matching_paths, key=len)
    source_member_identity = rule.normalise(member)
    source_identity = rule.normalise(source_array)
    source_tokens = {
        rule.normalise(token) for token in re.findall(r"\d+", source_array)
    }
    member_tokens = {rule.normalise(token) for token in re.findall(r"\d+", member)}
    present_in_source = (
        source_member_identity in source_tokens
        if source_member_identity.isdigit()
        else source_member_identity in source_identity
        and member_tokens <= source_tokens
    )
    if not present_in_source:
        return None, (
            f"member identifier {member!r} is absent from source array {source_array!r}"
        )
    identity = rule.normalise(rule.member_aliases.get(member, member))
    names = description_members[structure]
    matches = [
        index for index, name in enumerate(names) if rule.normalise(name) == identity
    ]
    if not matches:
        return None, (
            f"member identifier {member!r} has no identity match in "
            f"machine-description {structure}"
        )
    if len(matches) != 1:
        return None, (
            f"member identifier {member!r} matches {len(matches)} elements in "
            f"machine-description {structure}"
        )
    return matches[0], None


def _partner_key(
    row: Mapping[str, Any],
) -> tuple[str | None, str | None, str | None, str]:
    return (
        row.get("source_group"),
        row.get("source_array"),
        row.get("member_identifier"),
        row["target_path"].rsplit("/", maxsplit=1)[0],
    )


def import_generated_mappings(
    source: Path | str | Mapping[str, Any],
    catalogue: MachineMapCatalog,
    description_members: Mapping[str, Sequence[str]],
    *,
    identity_rule_name: str | None = None,
    target_cocos: int = 17,
) -> GeneratedMappingImport:
    """Build draft maps using ordered identity values from the machine description."""

    document = _document(source)
    if (
        document.get("format") != "imas-codex-mapping-handoff"
        or document.get("format_version") != 1
    ):
        raise GeneratedMappingImportError("unsupported mapping handoff format")
    ids_rows = document.get("ids")
    if not isinstance(ids_rows, list):
        raise GeneratedMappingImportError("mapping handoff ids must be an array")
    rule = _identity_rule(catalogue, identity_rule_name)
    maps: list[SignalMap] = []
    time_bindings: list[ResolvedTimeBinding] = []
    unresolved: list[UnresolvedMapping] = []
    pending: list[PendingCocosRule] = []
    for ids in ids_rows:
        if not isinstance(ids, Mapping):
            raise GeneratedMappingImportError("each ids entry must be an object")
        ids_name = ids["ids_name"]
        mapping_id = ids["mapping_id"]
        signals: list[SignalRule] = []
        value_rules: dict[
            tuple[str | None, str | None, str | None, str], SignalRule
        ] = {}
        value_reasons: dict[tuple[str | None, str | None, str | None, str], str] = {}
        time_rows: list[Mapping[str, Any]] = []
        used_targets: set[tuple[str, int]] = set()
        for row in ids["signals"]:
            field = row["target_path"].rsplit("/", maxsplit=1)[-1]
            source_property = row.get("source_property")
            if field == "time" and source_property == "time":
                time_rows.append(row)
                continue
            if field != "data":
                if field in {
                    "data_error_upper",
                    "data_error_lower",
                    "data_error_index",
                }:
                    reason = (
                        "imas-codex derived error bounds from the value; "
                        "no error signal exists"
                    )
                elif field == "time":
                    if source_property is None:
                        reason = (
                            "handoff has no source_property; time-vector binding "
                            "cannot be distinguished from a value binding"
                        )
                    else:
                        reason = (
                            f"source_property={source_property!r} cannot target time"
                        )
                else:
                    reason = "target is not a data field"
                unresolved.append(
                    UnresolvedMapping(
                        ids_name=ids_name,
                        source_id=row["source_id"],
                        source_array=row.get("source_array"),
                        target_path=row["target_path"],
                        reason=reason,
                    )
                )
                continue
            index, reason = _target_index(row, description_members, rule)
            if reason is None and source_property not in (None, "value"):
                reason = f"source_property={source_property} cannot target data"
            if reason is None and row["source_units"] != row["target_units"]:
                reason = "source and target units differ without a conversion factor"
            expression = row["transform_expression"]
            if reason is None and expression not in (None, "one_like", "value"):
                reason = f"unsupported transform expression {expression!r}"
            cocos = row.get("cocos_label")
            cocos_label_source = row.get("cocos_label_source")
            if reason is None and cocos is not None:
                label = str(cocos)
                if label not in COCOS_NEUTRAL_LABELS and not _cocos_dependent_label(
                    label
                ):
                    reason = f"unrecognised COCOS label {cocos!r}"
            target = (row["target_path"], index)
            if reason is None and target in used_targets:
                reason = f"target {target!r} is already assigned"
            if reason is not None:
                value_reasons[_partner_key(row)] = reason
                unresolved.append(
                    UnresolvedMapping(
                        ids_name=ids_name,
                        source_id=row["source_id"],
                        source_array=row.get("source_array"),
                        target_path=row["target_path"],
                        reason=reason,
                    )
                )
                continue
            signal = SignalRule(
                semantic_id=row["signal_id"],
                source_group=row["source_group"],
                source_array=row["source_array"],
                source_unit=row["source_units"],
                target_path=row["target_path"],
                target_unit=row["target_units"],
                target_index=index,
                transformation=(
                    "one_like" if expression in (None, "value") else expression
                ),
                source_cocos=None,
                unit_factor=1.0,
                channel_factor=1.0,
                standard_name=None,
                evidence="; ".join(
                    part
                    for part in (
                        f"imas-codex mapping_id={mapping_id}",
                        f"status={ids['status']}",
                        f"cocos_label={cocos}" if cocos is not None else "",
                        (
                            f"cocos_label_source={cocos_label_source}"
                            if cocos is not None and cocos_label_source is not None
                            else ""
                        ),
                        row["evidence"].strip(),
                    )
                    if part
                ),
                validation_state="draft",
            )
            signal.validate()
            key = _partner_key(row)
            open_source_cocos = getattr(catalogue, "source_cocos", None)
            if (
                cocos is not None
                and _cocos_dependent_label(str(cocos))
                and open_source_cocos in (None, 0)
            ):
                pending_reason = _pending_reason(str(cocos), cocos_label_source)
                pending.append(
                    PendingCocosRule(
                        ids_name=ids_name,
                        source_id=row["source_id"],
                        source_array=row["source_array"],
                        target_path=row["target_path"],
                        cocos_label=str(cocos),
                        cocos_label_source=cocos_label_source,
                        open_source_cocos=open_source_cocos,
                        reason=pending_reason,
                        rule=signal,
                    )
                )
                value_rules[key] = signal
                continue
            signals.append(signal)
            value_rules[key] = signal
            used_targets.add(target)
        for row in time_rows:
            key = _partner_key(row)
            partner = value_rules.get(key)
            if partner is not None:
                time_bindings.append(
                    ResolvedTimeBinding(
                        ids_name=ids_name,
                        source_id=row["source_id"],
                        source_array=row["source_array"],
                        target_path=row["target_path"],
                        value_rule=partner,
                    )
                )
            else:
                detail = value_reasons.get(key)
                reason = (
                    f"value partner unresolved: {detail}"
                    if detail is not None
                    else (
                        "value partner absent for the same channel, member, "
                        "and target structure"
                    )
                )
                unresolved.append(
                    UnresolvedMapping(
                        ids_name=ids_name,
                        source_id=row["source_id"],
                        source_array=row["source_array"],
                        target_path=row["target_path"],
                        reason=reason,
                    )
                )
        for row in ids["unexpanded"]:
            unresolved.append(
                UnresolvedMapping(
                    ids_name=ids_name,
                    source_id=row["source_id"],
                    source_array=None,
                    target_path=row["target_path"],
                    reason=f"unexpanded: {row['reason']}",
                )
            )
        if signals:
            maps.append(
                SignalMap.create(
                    schema_version=MAP_SCHEMA_VERSION,
                    set_version="0.1.0",
                    machine=document["facility"],
                    system=ids_name,
                    source_dataset=document["facility"],
                    target_dd_version=document["dd_version"],
                    target_cocos=target_cocos,
                    discovery_producer="imas-codex",
                    discovery_receipt=mapping_id,
                    signals=signals,
                )
            )
    return GeneratedMappingImport(
        tuple(maps), tuple(time_bindings), tuple(unresolved), tuple(pending)
    )
