"""Versioned, lightweight maps from immutable source arrays to canonical signals.

Discovery systems may produce a large evidence graph.  The Ambix runtime does
not query that graph.  It loads a reviewed distillation containing only source
bindings, target identities, separable conversion factors, validity intervals,
and evidence receipts.  The document is byte-canonical and content-addressed;
its human release version describes compatibility while its digest identifies
the exact bytes.

Standard Names are optional during the catalogue bootstrap.  ``semantic_id``
and the DDv4 target path are mandatory, so a signal is stable and useful before
the provisional name is accepted.  Adding an accepted Standard Name later does
not change the source binding or numerical transform.

Compilation resolves shot-dependent calibration once and reduces every served
row to one affine operation.  Data access then performs no JSON parsing,
convention lookup, interval search, or string-based calibration lookup in the
chunk loop.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from imas_ambix.cocos import (
    canonical_factor,
    require_canonical_contract,
)

MAP_SCHEMA_VERSION = "1.0.0"
"""Schema understood by this reader."""

PACKAGED_MAP_ROOT = Path(__file__).with_name("maps")
"""Reviewed, Git-versioned maps shipped with Ambix."""

_SEMANTIC_VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
_MAP_COMPONENT = re.compile(r"^[a-z][a-z0-9_-]*$")


class SignalMapError(ValueError):
    """Raised when a distilled map is incomplete, ambiguous, or inconsistent."""


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise SignalMapError(f"{label} must be non-empty trimmed text")
    return value


def _finite(value: Any, label: str, *, nonzero: bool = False) -> float:
    if isinstance(value, bool):
        raise SignalMapError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise SignalMapError(f"{label} must be a finite number") from error
    if not math.isfinite(number):
        raise SignalMapError(f"{label} must be a finite number")
    if nonzero and number == 0.0:
        raise SignalMapError(f"{label} cannot be zero because it erases the signal")
    return number


def _exact_keys(row: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(row)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise SignalMapError(f"{label} keys differ: missing={missing}, extra={extra}")


@dataclass(frozen=True, order=True)
class SignalRule:
    """One immutable source binding and its static source-to-DD conversion."""

    semantic_id: str
    source_group: str
    source_array: str
    source_unit: str
    target_path: str
    target_unit: str
    target_index: int | None
    transformation: str
    source_cocos: int | None
    unit_factor: float
    channel_factor: float
    standard_name: str | None
    evidence: str

    @property
    def source_key(self) -> tuple[str, str]:
        return self.source_group, self.source_array

    @property
    def target_key(self) -> tuple[str, int | None]:
        return self.target_path, self.target_index

    @property
    def convention_factor(self) -> float:
        if self.source_cocos is None:
            return 1.0
        return canonical_factor(
            self.transformation,
            source_cocos=int(self.source_cocos),
        )

    @property
    def static_scale(self) -> float:
        return (
            float(self.unit_factor)
            * float(self.channel_factor)
            * self.convention_factor
        )

    def validate(self) -> None:
        for value, label in (
            (self.semantic_id, "semantic id"),
            (self.source_group, "source group"),
            (self.source_array, "source array"),
            (self.source_unit, "source unit"),
            (self.target_path, "target path"),
            (self.target_unit, "target unit"),
            (self.transformation, "transformation"),
            (self.evidence, "signal evidence"),
        ):
            _text(value, label)
        if self.standard_name is not None:
            _text(self.standard_name, "standard name")
        if self.target_index is not None:
            if isinstance(self.target_index, bool) or not isinstance(
                self.target_index, int
            ):
                raise SignalMapError("target index must be an integer or null")
            if self.target_index < 0:
                raise SignalMapError("target index must be non-negative")
        _finite(self.unit_factor, "unit factor", nonzero=True)
        _finite(self.channel_factor, "channel factor", nonzero=True)
        if self.source_cocos is None:
            if self.transformation != "one_like":
                raise SignalMapError(
                    f"{self.semantic_id!r} needs source_cocos for transformation "
                    f"{self.transformation!r}"
                )
        elif isinstance(self.source_cocos, bool) or not isinstance(
            self.source_cocos, int
        ):
            raise SignalMapError("source COCOS must be an integer or null")
        try:
            _finite(self.convention_factor, "convention factor", nonzero=True)
        except ValueError as error:
            raise SignalMapError(str(error)) from error

    def as_dict(self) -> dict[str, Any]:
        return {
            "channel_factor": float(self.channel_factor),
            "evidence": self.evidence,
            "semantic_id": self.semantic_id,
            "source_array": self.source_array,
            "source_cocos": self.source_cocos,
            "source_group": self.source_group,
            "source_unit": self.source_unit,
            "standard_name": self.standard_name,
            "target_index": self.target_index,
            "target_path": self.target_path,
            "target_unit": self.target_unit,
            "transformation": self.transformation,
            "unit_factor": float(self.unit_factor),
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> SignalRule:
        expected = {
            "channel_factor",
            "evidence",
            "semantic_id",
            "source_array",
            "source_cocos",
            "source_group",
            "source_unit",
            "standard_name",
            "target_index",
            "target_path",
            "target_unit",
            "transformation",
            "unit_factor",
        }
        _exact_keys(row, expected, "signal rule")
        rule = cls(
            semantic_id=row["semantic_id"],
            source_group=row["source_group"],
            source_array=row["source_array"],
            source_unit=row["source_unit"],
            target_path=row["target_path"],
            target_unit=row["target_unit"],
            target_index=row["target_index"],
            transformation=row["transformation"],
            source_cocos=row["source_cocos"],
            unit_factor=row["unit_factor"],
            channel_factor=row["channel_factor"],
            standard_name=row["standard_name"],
            evidence=row["evidence"],
        )
        rule.validate()
        return rule


@dataclass(frozen=True, order=True)
class CalibrationRule:
    """One already-distilled affine calibration over a shot interval.

    ``canonical = statically_converted * scale + offset``.  Discovery-specific
    gain, range, polarity, and baseline records are combined into this explicit
    target-unit operation before publication.  Intervals for one signal may
    touch but may never overlap.
    """

    semantic_id: str
    first_shot: int | None
    last_shot: int | None
    scale: float
    offset: float
    evidence: str

    def covers(self, shot: int) -> bool:
        return (self.first_shot is None or self.first_shot <= shot) and (
            self.last_shot is None or shot <= self.last_shot
        )

    def validate(self) -> None:
        _text(self.semantic_id, "calibration semantic id")
        _text(self.evidence, "calibration evidence")
        for value, label in (
            (self.first_shot, "first shot"),
            (self.last_shot, "last shot"),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise SignalMapError(f"{label} must be a non-negative integer or null")
        if (
            self.first_shot is not None
            and self.last_shot is not None
            and self.first_shot > self.last_shot
        ):
            raise SignalMapError("calibration first shot exceeds its last shot")
        _finite(self.scale, "calibration scale", nonzero=True)
        _finite(self.offset, "calibration offset")

    def as_dict(self) -> dict[str, Any]:
        return {
            "evidence": self.evidence,
            "first_shot": self.first_shot,
            "last_shot": self.last_shot,
            "offset": float(self.offset),
            "scale": float(self.scale),
            "semantic_id": self.semantic_id,
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> CalibrationRule:
        expected = {
            "evidence",
            "first_shot",
            "last_shot",
            "offset",
            "scale",
            "semantic_id",
        }
        _exact_keys(row, expected, "calibration rule")
        rule = cls(**{key: row[key] for key in expected})
        rule.validate()
        return rule


@dataclass(frozen=True, order=True)
class BlockedSignal:
    """A source array deliberately absent from the served map."""

    source_group: str
    source_array: str
    reason: str
    unmet: str

    @property
    def source_key(self) -> tuple[str, str]:
        return self.source_group, self.source_array

    def validate(self) -> None:
        for value, label in (
            (self.source_group, "blocked source group"),
            (self.source_array, "blocked source array"),
            (self.reason, "blocked reason"),
            (self.unmet, "blocked unmet condition"),
        ):
            _text(value, label)

    def as_dict(self) -> dict[str, str]:
        return {
            "reason": self.reason,
            "source_array": self.source_array,
            "source_group": self.source_group,
            "unmet": self.unmet,
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> BlockedSignal:
        expected = {"reason", "source_array", "source_group", "unmet"}
        _exact_keys(row, expected, "blocked signal")
        blocked = cls(**{key: row[key] for key in expected})
        blocked.validate()
        return blocked


@dataclass(frozen=True)
class CompiledSignal:
    """One source row reduced to the affine operation used by the hot path."""

    rule: SignalRule
    scale: float
    offset: float

    def apply(self, values: Any) -> np.ndarray:
        result = np.multiply(np.asarray(values), self.scale)
        if self.offset != 0.0:
            np.add(result, self.offset, out=result)
        return result


class CompiledSignalMap:
    """Shot-resolved map with constant-time lookup and vectorized batch transforms."""

    def __init__(
        self,
        *,
        shot: int,
        digest: str,
        signals: Sequence[CompiledSignal],
    ) -> None:
        self.shot = int(shot)
        self.digest = digest
        self.signals = tuple(signals)
        self._by_id = MappingProxyType(
            {signal.rule.semantic_id: signal for signal in self.signals}
        )

    def __len__(self) -> int:
        return len(self.signals)

    def __iter__(self):
        return iter(self.signals)

    def __getitem__(self, semantic_id: str) -> CompiledSignal:
        try:
            return self._by_id[semantic_id]
        except KeyError as error:
            raise KeyError(f"signal {semantic_id!r} is not served") from error

    def apply(self, semantic_id: str, values: Any) -> np.ndarray:
        return self[semantic_id].apply(values)

    def apply_columns(
        self,
        values: Any,
        semantic_ids: Sequence[str],
        *,
        axis: int = -1,
    ) -> np.ndarray:
        """Apply a fused affine conversion to a packed channel axis."""

        array = np.asarray(values)
        normalized_axis = np.lib.array_utils.normalize_axis_index(axis, array.ndim)
        if array.shape[normalized_axis] != len(semantic_ids):
            raise SignalMapError(
                f"channel axis has {array.shape[normalized_axis]} columns for "
                f"{len(semantic_ids)} semantic ids"
            )
        selected = [self[semantic_id] for semantic_id in semantic_ids]
        shape = [1] * array.ndim
        shape[normalized_axis] = len(selected)
        scales = np.asarray([signal.scale for signal in selected]).reshape(shape)
        offsets = np.asarray([signal.offset for signal in selected]).reshape(shape)
        result = np.multiply(array, scales)
        if np.any(offsets != 0.0):
            np.add(result, offsets, out=result)
        return result


@dataclass(frozen=True)
class SignalMap:
    """Reviewed source map owned by Ambix and independent of the raw store."""

    schema_version: str
    set_version: str
    machine: str
    system: str
    source_dataset: str
    target_dd_version: str
    target_cocos: int
    discovery_producer: str
    discovery_receipt: str
    signals: tuple[SignalRule, ...]
    calibrations: tuple[CalibrationRule, ...] = ()
    blocked: tuple[BlockedSignal, ...] = ()

    @classmethod
    def create(
        cls,
        *,
        schema_version: str,
        set_version: str,
        machine: str,
        system: str,
        source_dataset: str,
        target_dd_version: str,
        target_cocos: int,
        discovery_producer: str,
        discovery_receipt: str,
        signals: Iterable[SignalRule],
        calibrations: Iterable[CalibrationRule] = (),
        blocked: Iterable[BlockedSignal] = (),
    ) -> SignalMap:
        result = cls(
            schema_version=schema_version,
            set_version=set_version,
            machine=machine,
            system=system,
            source_dataset=source_dataset,
            target_dd_version=target_dd_version,
            target_cocos=target_cocos,
            discovery_producer=discovery_producer,
            discovery_receipt=discovery_receipt,
            signals=tuple(sorted(signals, key=lambda row: row.semantic_id)),
            calibrations=tuple(
                sorted(
                    calibrations,
                    key=lambda row: (
                        row.semantic_id,
                        -1 if row.first_shot is None else row.first_shot,
                        math.inf if row.last_shot is None else row.last_shot,
                    ),
                )
            ),
            blocked=tuple(sorted(blocked, key=lambda row: row.source_key)),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.schema_version != MAP_SCHEMA_VERSION:
            raise SignalMapError(
                f"map schema {self.schema_version!r} is not supported; expected "
                f"{MAP_SCHEMA_VERSION!r}"
            )
        if not _SEMANTIC_VERSION.fullmatch(str(self.set_version)):
            raise SignalMapError("map set version must be semantic major.minor.patch")
        for value, label in (
            (self.machine, "machine"),
            (self.system, "system"),
            (self.source_dataset, "source dataset"),
            (self.discovery_producer, "discovery producer"),
            (self.discovery_receipt, "discovery receipt"),
        ):
            _text(value, label)
        try:
            require_canonical_contract(self.target_dd_version, self.target_cocos)
        except ValueError as error:
            raise SignalMapError(str(error)) from error
        if not self.signals:
            raise SignalMapError("a signal map must serve at least one signal")
        for signal in self.signals:
            signal.validate()
        for calibration in self.calibrations:
            calibration.validate()
        for blocked in self.blocked:
            blocked.validate()

        semantic_ids = [signal.semantic_id for signal in self.signals]
        if len(set(semantic_ids)) != len(semantic_ids):
            raise SignalMapError("semantic ids must be unique")
        targets = [signal.target_key for signal in self.signals]
        if len(set(targets)) != len(targets):
            raise SignalMapError("one canonical target cannot be served twice")
        served_sources = {signal.source_key for signal in self.signals}
        blocked_sources = {signal.source_key for signal in self.blocked}
        overlap = served_sources & blocked_sources
        if overlap:
            raise SignalMapError(
                f"source arrays are both served and blocked: {sorted(overlap)}"
            )

        known = set(semantic_ids)
        by_signal: dict[str, list[CalibrationRule]] = {}
        for calibration in self.calibrations:
            if calibration.semantic_id not in known:
                raise SignalMapError(
                    f"calibration targets unknown signal {calibration.semantic_id!r}"
                )
            by_signal.setdefault(calibration.semantic_id, []).append(calibration)
        for semantic_id, rows in by_signal.items():
            for first, second in zip(rows, rows[1:], strict=False):
                first_end = math.inf if first.last_shot is None else first.last_shot
                second_start = -1 if second.first_shot is None else second.first_shot
                if second_start <= first_end:
                    raise SignalMapError(
                        f"calibration intervals overlap for {semantic_id!r}"
                    )

    def as_dict(self) -> dict[str, Any]:
        return {
            "blocked": [row.as_dict() for row in self.blocked],
            "calibrations": [row.as_dict() for row in self.calibrations],
            "discovery": {
                "producer": self.discovery_producer,
                "receipt": self.discovery_receipt,
            },
            "machine": self.machine,
            "schema_version": self.schema_version,
            "set_version": self.set_version,
            "signals": [row.as_dict() for row in self.signals],
            "source_dataset": self.source_dataset,
            "system": self.system,
            "target": {
                "cocos": int(self.target_cocos),
                "dd_version": self.target_dd_version,
            },
        }

    def canonical_bytes(self) -> bytes:
        self.validate()
        return (
            json.dumps(
                self.as_dict(),
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        )

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def compile(self, shot: int) -> CompiledSignalMap:
        if isinstance(shot, bool) or not isinstance(shot, int) or shot < 0:
            raise SignalMapError("shot must be a non-negative integer")
        calibration_by_id: dict[str, CalibrationRule] = {}
        for calibration in self.calibrations:
            if calibration.covers(shot):
                calibration_by_id[calibration.semantic_id] = calibration
        compiled = []
        for rule in self.signals:
            calibration = calibration_by_id.get(rule.semantic_id)
            scale = rule.static_scale
            offset = 0.0
            if calibration is not None:
                scale *= calibration.scale
                offset = calibration.offset
            compiled.append(CompiledSignal(rule=rule, scale=scale, offset=offset))
        return CompiledSignalMap(shot=shot, digest=self.digest, signals=compiled)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SignalMap:
        expected = {
            "blocked",
            "calibrations",
            "discovery",
            "machine",
            "schema_version",
            "set_version",
            "signals",
            "source_dataset",
            "system",
            "target",
        }
        _exact_keys(payload, expected, "signal map")
        discovery = payload["discovery"]
        target = payload["target"]
        if not isinstance(discovery, Mapping):
            raise SignalMapError("discovery must be an object")
        if not isinstance(target, Mapping):
            raise SignalMapError("target must be an object")
        _exact_keys(discovery, {"producer", "receipt"}, "discovery")
        _exact_keys(target, {"cocos", "dd_version"}, "target")
        for key in ("signals", "calibrations", "blocked"):
            if isinstance(payload[key], (str, bytes)) or not isinstance(
                payload[key], Sequence
            ):
                raise SignalMapError(f"{key} must be an array")
        return cls.create(
            schema_version=payload["schema_version"],
            set_version=payload["set_version"],
            machine=payload["machine"],
            system=payload["system"],
            source_dataset=payload["source_dataset"],
            target_dd_version=target["dd_version"],
            target_cocos=target["cocos"],
            discovery_producer=discovery["producer"],
            discovery_receipt=discovery["receipt"],
            signals=(SignalRule.from_dict(row) for row in payload["signals"]),
            calibrations=(
                CalibrationRule.from_dict(row) for row in payload["calibrations"]
            ),
            blocked=(BlockedSignal.from_dict(row) for row in payload["blocked"]),
        )


def load_signal_map(path: Path | str) -> SignalMap:
    """Load and validate a distilled map without touching its source dataset."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SignalMapError(f"cannot read signal map {path!s}: {error}") from error
    if not isinstance(payload, Mapping):
        raise SignalMapError("signal map root must be an object")
    return SignalMap.from_dict(payload)


def load_packaged_signal_map(machine: str, system: str) -> SignalMap:
    """Load one reviewed map by machine and system, without reading source data."""

    for value, label in ((machine, "machine"), (system, "system")):
        if not isinstance(value, str) or _MAP_COMPONENT.fullmatch(value) is None:
            raise SignalMapError(
                f"packaged map {label} must match {_MAP_COMPONENT.pattern!r}"
            )
    path = PACKAGED_MAP_ROOT / machine / f"{system}.json"
    source_map = load_signal_map(path)
    if source_map.machine != machine or source_map.system != system:
        raise SignalMapError(
            f"map {path} identifies {source_map.machine}/{source_map.system}, "
            f"not {machine}/{system}"
        )
    return source_map


__all__ = [
    "MAP_SCHEMA_VERSION",
    "PACKAGED_MAP_ROOT",
    "BlockedSignal",
    "CalibrationRule",
    "CompiledSignal",
    "CompiledSignalMap",
    "SignalMap",
    "SignalMapError",
    "SignalRule",
    "load_packaged_signal_map",
    "load_signal_map",
]
