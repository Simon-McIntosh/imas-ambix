"""Parity receipts for the DD-description to geometry-table adapter."""

from __future__ import annotations

import dataclasses
import inspect
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import imas_ambix.data.geometry_adapter as adapter_module
from imas_ambix.data.geometry_adapter import geometry_table_from_description
from imas_ambix.data.machine_map import load_packaged_machine_map
from imas_ambix.data.transform_engine import transform_machine_description
from imas_ambix.gs.geometry import GeometryTable, build_table_for_shot

LEVEL2_ROOT = Path("/work/projects/imas_gpu/mast/level2/shots")
LEVEL1_ROOT = Path("/work/projects/imas_gpu/mast/level1/shots")
RANGE_FIRST_SHOTS = (11_766, 12_417)


@dataclass(frozen=True)
class _ExpectedDivergence:
    status: str
    cause: str
    reason: str


_NONMATCHING_FIELDS = {
    "signature": _ExpectedDivergence(
        "differing",
        "representation-divergence",
        "the signature fingerprints the DD sensor and conductor discretisation, "
        "whose declared cardinalities differ from the legacy source",
    ),
    "b_probes": _ExpectedDivergence(
        "differing",
        "catalog-unavailable",
        "the catalog qualifies probe orientation as unavailable and its level-2 "
        "probe Z coordinates differ from the legacy efm coordinates",
    ),
    "flux_loops": _ExpectedDivergence(
        "differing",
        "catalog-unavailable",
        "the catalog emits 44 point loops while the legacy efm table exposes 46; "
        "saddle trajectories are not substitute point loops",
    ),
    "pf_filaments": _ExpectedDivergence(
        "differing",
        "catalog-unavailable",
        "the DD description exposes 938 conductor elements without a shared join "
        "identifier to the selected 1004-row topology, while the legacy table "
        "collapses its rows into field-equivalent rectangles",
    ),
    "sensor_map": _ExpectedDivergence(
        "differing",
        "catalog-unavailable",
        "acquisition-address descriptions are not declared, so DD sensor names "
        "cannot reproduce the legacy nearest-neighbour channel mapping",
    ),
    "passive_structures": _ExpectedDivergence(
        "differing",
        "representation-divergence",
        "DD passive conductor elements and legacy induced-current source entries "
        "are different declared structures",
    ),
    "amc_current_channels": _ExpectedDivergence(
        "missing",
        "catalog-unavailable",
        "the catalog explicitly qualifies acquisition current-channel addressing "
        "as unavailable",
    ),
    "unmatched_amb": _ExpectedDivergence(
        "missing",
        "catalog-unavailable",
        "unmatched acquisition channels cannot be enumerated without acquisition "
        "address descriptions",
    ),
    "r0": _ExpectedDivergence(
        "missing",
        "catalog-unavailable",
        "no device reference-radius binding is emitted",
    ),
    "minor_radius": _ExpectedDivergence(
        "missing",
        "catalog-unavailable",
        "no device minor-radius binding is emitted",
    ),
    "provenance_flags": _ExpectedDivergence(
        "differing",
        "representation-divergence",
        "the adapter records unresolved declarations while the legacy reader "
        "reports no provenance qualifications",
    ),
    "active_circuits": _ExpectedDivergence(
        "differing",
        "representation-divergence",
        "the DD description distinguishes actively supplied conductors while the "
        "legacy table leaves that classification implicit",
    ),
    "polygon_sections": _ExpectedDivergence(
        "missing",
        "catalog-unavailable",
        "oblique geometry is declared but has no shared element identifier for "
        "joining a section to a legacy circuit",
    ),
}


def _assert_exact(actual: Any, expected: Any) -> None:
    if isinstance(actual, np.ndarray) or isinstance(expected, np.ndarray):
        assert np.array_equal(np.asarray(actual), np.asarray(expected))
        return
    if dataclasses.is_dataclass(actual) and dataclasses.is_dataclass(expected):
        assert type(actual) is type(expected)
        for field in dataclasses.fields(actual):
            _assert_exact(getattr(actual, field.name), getattr(expected, field.name))
        return
    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_exact(actual_item, expected_item)
        return
    assert actual == expected


def _equal(actual: Any, expected: Any) -> bool:
    try:
        _assert_exact(actual, expected)
    except AssertionError, TypeError, ValueError:
        return False
    return True


def _is_missing(actual: Any, expected: Any) -> bool:
    if isinstance(actual, float) and math.isnan(actual):
        return not (isinstance(expected, float) and math.isnan(expected))
    if isinstance(actual, (list, tuple)) and not actual:
        return bool(expected)
    return False


def _field_status(actual: Any, expected: Any) -> str:
    if _equal(actual, expected):
        return "matching"
    if _is_missing(actual, expected):
        return "missing"
    return "differing"


def test_adapter_source_contains_no_machine_selection() -> None:
    source = inspect.getsource(adapter_module).casefold()
    forbidden_machine_names = ("mast", "diii")

    assert all(name not in source for name in forbidden_machine_names)


@pytest.mark.skipif(
    not all(
        (LEVEL2_ROOT / f"{shot}.zarr").is_dir()
        and (LEVEL1_ROOT / f"{shot}.zarr").is_dir()
        for shot in RANGE_FIRST_SHOTS
    ),
    reason="local level-1 and level-2 geometry stores are not mounted",
)
def test_adapter_accounts_for_every_legacy_geometry_field() -> None:
    catalog = load_packaged_machine_map("mast")
    table_fields = tuple(field.name for field in dataclasses.fields(GeometryTable))

    assert (
        tuple(machine_map.first_shot for machine_map in catalog.maps)
        == RANGE_FIRST_SHOTS
    )
    assert set(_NONMATCHING_FIELDS).issubset(table_fields)
    for shot in RANGE_FIRST_SHOTS:
        description = transform_machine_description(catalog, shot, "zarr", LEVEL2_ROOT)
        adapted = geometry_table_from_description(description, catalog)
        legacy = build_table_for_shot(shot)
        counts = {"matching": 0, "differing": 0, "missing": 0}
        classified_causes: list[str] = []

        for field_name in table_fields:
            actual = getattr(adapted, field_name)
            expected = getattr(legacy, field_name)
            status = _field_status(actual, expected)
            counts[status] += 1
            divergence = _NONMATCHING_FIELDS.get(field_name)
            if status == "matching":
                assert divergence is None, (
                    f"{field_name} now matches and its stale divergence receipt "
                    "must be removed"
                )
                _assert_exact(actual, expected)
                continue

            assert divergence is not None, (
                f"unclassified adapter divergence for {field_name}: {status}"
            )
            assert status == divergence.status
            classified_causes.append(divergence.cause)
            print(
                "GEOMETRY_ADAPTER_FIELD "
                f"shot={shot} field={field_name} status={status} "
                f"cause={divergence.cause} reason={divergence.reason}"
            )

        assert "adapter-incorrect" not in classified_causes
        assert sum(counts.values()) == len(table_fields)
        signature_equal = adapted.signature.key == legacy.signature.key
        print(
            "GEOMETRY_ADAPTER_PARITY "
            f"shot={shot} matching={counts['matching']} "
            f"differing={counts['differing']} missing={counts['missing']}"
        )
        print(
            "GEOMETRY_ADAPTER_SIGNATURE "
            f"shot={shot} adapter={adapted.signature.key} "
            f"legacy={legacy.signature.key} equal={signature_equal}"
        )
