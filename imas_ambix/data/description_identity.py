"""Observable identity for machine descriptions whose geometry can move.

A single global digest is the wrong assertion for a machine whose geometry
changes with shot: legitimate range transitions would look like corruption.
Identity is instead observable locally as deterministic bytes for one emitted
description, consistent content within each declared version, and named field
changes when a version boundary is crossed.

The transform also emits pulse quantities from description-bearing IDSs.
Those ``data`` and ``time`` leaves describe an experiment, not the machine, so
they are deliberately excluded from the stable description representation.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from imas_ambix.data.geometry_transitions import geometry_fields_changed

if TYPE_CHECKING:
    from imas_ambix.data.transform_engine import EmittedArray, MachineDescription


def _is_description_array(array: EmittedArray) -> bool:
    components = array.dd_path.split("/")
    return components[-1] not in {"data", "time"}


def _json_value(value: object) -> object:
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def _array_payload(array: np.ndarray) -> Mapping[str, object]:
    values = np.asarray(array)
    if values.dtype.hasobject:
        encoded_values: object = _json_value(values.tolist())
        encoding = "json"
    else:
        canonical_dtype = (
            values.dtype
            if values.dtype.byteorder == "|"
            else values.dtype.newbyteorder("<")
        )
        canonical = np.ascontiguousarray(values.astype(canonical_dtype, copy=False))
        encoded_values = canonical.tobytes().hex()
        encoding = "little-endian-hex"
    return {
        "dtype": values.dtype.str,
        "shape": list(values.shape),
        "encoding": encoding,
        "values": encoded_values,
    }


def machine_description_bytes(description: MachineDescription) -> bytes:
    """Return canonical bytes for the stable content emitted for one version.

    Shot number and range endpoints select the description but are not part of
    its physical content. Array ordering is canonicalized by binding identity,
    and pulse ``data`` and ``time`` leaves are excluded.
    """
    machine_map = description.machine_map
    arrays = []
    for emitted in sorted(description.arrays, key=lambda item: item.binding_name):
        if not _is_description_array(emitted):
            continue
        arrays.append(
            {
                "binding_name": emitted.binding_name,
                "source_group": emitted.source_group,
                "source_array": emitted.source_array,
                "dd_path": emitted.dd_path,
                "source_unit": emitted.source_unit,
                "target_unit": emitted.target_unit,
                "array": _array_payload(emitted.values),
            }
        )
    payload = {
        "machine": machine_map.machine,
        "description_version": machine_map.transition,
        "dd_version": description.dd_version,
        "status": description.status,
        "arrays": arrays,
    }
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=True,
    ).encode("ascii")


def geometry_description_for_transition(
    payload: Mapping[str, Any], transition_name: str | None
) -> Mapping[str, Any]:
    """Resolve the geometry-table row named by a map transition."""
    if transition_name is None:
        raise LookupError("machine map has no declared geometry transition")
    campaigns = payload.get("campaigns", payload)
    if not isinstance(campaigns, Mapping):
        raise ValueError("geometry payload has no campaign mapping")
    signature_suffix = transition_name.rsplit("-", maxsplit=1)[-1]
    matches = [
        row
        for row in campaigns.values()
        if isinstance(row, Mapping)
        and str(row.get("signature_key", "")).endswith(signature_suffix)
    ]
    if len(matches) != 1:
        raise LookupError(
            f"transition {transition_name!r} resolves to {len(matches)} geometry rows"
        )
    return matches[0]


def description_field_changes(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> tuple[str, ...]:
    """Return the inspectable machine-description fields changed at a boundary."""
    return geometry_fields_changed(before, after)
