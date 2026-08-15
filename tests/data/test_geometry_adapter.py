"""Parity receipts for the DD-description to geometry-table adapter."""

from __future__ import annotations

import dataclasses
import inspect
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import imas_ambix.data.geometry_adapter as adapter_module
from imas_ambix.data.geometry_adapter import (
    CURRENT_SOURCE_ABSENT,
    CURRENT_SOURCE_NEEDS_DECLARATION,
    RECOVERABLE_CURRENT_SOURCE,
    current_source_resolutions,
    geometry_table_from_description,
)
from imas_ambix.data.machine_map import load_packaged_machine_map
from imas_ambix.data.operator_parity import compare_operator_parity
from imas_ambix.data.transform_engine import transform_machine_description
from imas_ambix.gs.geometry import GeometryTable, build_table_for_shot
from imas_ambix.gs.operator import build_operator, classify_circuits

LEVEL2_ROOT = Path("/work/projects/imas_gpu/mast/level2/shots")
LEVEL1_ROOT = Path("/work/projects/imas_gpu/mast/level1/shots")
RANGE_FIRST_SHOTS = (11_766, 12_417)
BASELINE_PARITY_COUNTS = {"matching": 8, "differing": 8, "missing": 1}
MATERIALISED_PARITY_COUNTS = {"matching": 7, "differing": 9, "missing": 1}
BASELINE_OPERATOR_COLUMN_COUNT = 231
LEGACY_OPERATOR_COLUMN_COUNT = 251
BASELINE_DIFFERING_CELL_COUNTS = {11_766: 20_238, 12_417: 24_254}
COLUMN_ADDING = "column-adding"
COLUMN_MODIFYING = "column-modifying"
COLUMN_UNCHANGED = "neither"


@dataclass(frozen=True)
class _ExpectedDivergence:
    status: str
    cause: str
    reason: str


_NONMATCHING_FIELDS = {
    "signature": _ExpectedDivergence(
        "differing",
        "representation-divergence",
        "the signature fingerprints the DD sensor coordinates and positive-turn "
        "connectivity representation, which differ from the legacy source",
    ),
    "b_probes": _ExpectedDivergence(
        "differing",
        "source-divergence",
        "the catalog qualifies probe orientation as unavailable and its level-2 "
        "probe Z coordinates differ from the legacy efm coordinates",
    ),
    "flux_loops": _ExpectedDivergence(
        "differing",
        "source-divergence",
        "the supplement closes the count at 46, but the level-2 point-loop "
        "coordinates only partially coincide with the legacy efm coordinates",
    ),
    "pf_filaments": _ExpectedDivergence(
        "differing",
        "representation-divergence",
        "all 1004 topology rows join and collapse to 299 rectangles, but the DD "
        "topology declares positive combined turn magnitude plus direction while "
        "the legacy table retains turns and current weight as separate values",
    ),
    "sensor_map": _ExpectedDivergence(
        "differing",
        "catalog-unavailable",
        "the acquisition declaration supplies address identities but not every "
        "address-to-geometry association, probe angle, or matching residual",
    ),
    "passive_structures": _ExpectedDivergence(
        "differing",
        "representation-divergence",
        "DD passive conductor elements and legacy induced-current source entries "
        "are different declared structures",
    ),
    "minor_radius": _ExpectedDivergence(
        "missing",
        "catalog-unavailable",
        "the catalog qualification records that the Data Dictionary has no fixed "
        "machine-description minor-radius leaf",
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
    "circuit_drives": _ExpectedDivergence(
        "differing",
        "representation-divergence",
        "the catalog declares circuit-to-current joins while the legacy reader "
        "leaves those drive identities implicit",
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


def _operator_column_identity(circuit_class: Any) -> tuple[str, str | int]:
    if circuit_class.role in {"known_pf", "known_case"}:
        return "driven", circuit_class.amc_channel
    return "passive", circuit_class.circuit


def _correction_effects(
    resolutions: tuple[Any, ...],
    circuit_indices: dict[str, int],
    baseline_classes: list[Any],
    corrected_classes: list[Any],
    baseline_filaments: list[Any],
    corrected_filaments: list[Any],
) -> dict[str, str]:
    baseline_by_circuit = {item.circuit: item for item in baseline_classes}
    corrected_by_circuit = {item.circuit: item for item in corrected_classes}
    baseline_columns = {_operator_column_identity(item) for item in baseline_classes}
    corrected_columns = {_operator_column_identity(item) for item in corrected_classes}
    baseline_geometry = {
        circuit: tuple(item for item in baseline_filaments if item.circuit == circuit)
        for circuit in circuit_indices.values()
    }
    corrected_geometry = {
        circuit: tuple(item for item in corrected_filaments if item.circuit == circuit)
        for circuit in circuit_indices.values()
    }
    effects: dict[str, str] = {}
    for resolution in resolutions:
        circuit = circuit_indices[resolution.circuit_identifier]
        before = _operator_column_identity(baseline_by_circuit[circuit])
        after = _operator_column_identity(corrected_by_circuit[circuit])
        geometry_changed = not _equal(
            baseline_geometry[circuit], corrected_geometry[circuit]
        )
        if after not in baseline_columns and before in corrected_columns:
            effect = COLUMN_ADDING
        elif before != after or geometry_changed:
            effect = COLUMN_MODIFYING
        else:
            effect = COLUMN_UNCHANGED
        effects[resolution.circuit_identifier] = effect
    return effects


def _baseline_projection_table(
    adapted: GeometryTable,
    description: Any,
    catalog: Any,
) -> GeometryTable:
    supplement = adapter_module._selected_supplement(catalog, description)
    acquisition = adapter_module._selected_acquisition(catalog, supplement)
    conductors = adapter_module._conductors(description, catalog)
    filaments, active_circuits, _ = adapter_module._pf_filaments(
        catalog,
        description,
        conductors,
        acquisition,
        resolve_direct_geometry=False,
    )
    return dataclasses.replace(
        adapted,
        pf_filaments=filaments,
        active_circuits=active_circuits,
        circuit_drives=[],
    )


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
        reordered = geometry_table_from_description(
            dataclasses.replace(
                description,
                arrays=tuple(reversed(description.arrays)),
            ),
            catalog,
        )
        legacy = build_table_for_shot(shot)
        resolutions = current_source_resolutions(description, catalog)
        counts = {"matching": 0, "differing": 0, "missing": 0}
        classified_causes: list[str] = []
        topology = next(
            item
            for item in catalog.drive_topologies
            if item.name == description.machine_map.drive_topology
        )
        conductor_identifiers = {
            identifier
            for assembly in catalog.structure_assemblies
            for identifier in assembly.element_identifiers
        }
        unjoined = {
            item.geometry_element_identifier
            for item in topology.connections
            if item.geometry_element_identifier not in conductor_identifiers
        }

        assert len(topology.connections) == 1_004
        assert len(unjoined) == 0
        circuit_order = tuple(
            dict.fromkeys(item.circuit_identifier for item in topology.connections)
        )
        circuit_indices = {
            identifier: index + 1 for index, identifier in enumerate(circuit_order)
        }
        emitted_circuits = {item.circuit for item in adapted.pf_filaments}
        for resolution in resolutions:
            circuit = circuit_indices[resolution.circuit_identifier]
            assert circuit in emitted_circuits
            assert circuit not in adapted.active_circuits
            assert resolution.conductor_identifiers
        assert len(adapted.flux_loops) == 46
        _assert_exact(adapted.pf_filaments, reordered.pf_filaments)
        _assert_exact(adapted.circuit_drives, reordered.circuit_drives)
        drives_by_circuit = {item.circuit: item for item in adapted.circuit_drives}
        assert len(drives_by_circuit) == len(adapted.circuit_drives) == 21
        joined_channels = {}
        for join in catalog.circuit_current_joins:
            circuit = circuit_indices[join.circuit_identifier]
            drive = drives_by_circuit[circuit]
            assert drive.channel == join.current_channel
            assert drive.conductor == join.conductor_identifier
            assert drive.evidence == join.evidence
            joined_channels[join.circuit_identifier] = drive.channel
        assert len(joined_channels) == 8
        assert all(
            drive.ampere_turns_per_ampere > 0 for drive in adapted.circuit_drives
        )
        print(
            "GEOMETRY_ADAPTER_JOIN "
            f"shot={shot} topology_rows={len(topology.connections)} "
            f"conductor_identifiers={len(conductor_identifiers)} "
            f"unjoined={len(unjoined)} "
            f"collapsed_filaments={len(adapted.pf_filaments)} "
            "declaration_order_invariant=true"
        )
        print(
            "GEOMETRY_ADAPTER_FLUX_LOOPS "
            f"shot={shot} emitted={len(adapted.flux_loops)} "
            f"legacy={len(legacy.flux_loops)}"
        )

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
        assert counts == MATERIALISED_PARITY_COUNTS
        signature_equal = adapted.signature.key == legacy.signature.key
        deltas = {
            status: counts[status] - BASELINE_PARITY_COUNTS[status] for status in counts
        }
        print(
            "GEOMETRY_ADAPTER_PARITY "
            f"shot={shot} matching={counts['matching']} "
            f"differing={counts['differing']} missing={counts['missing']} "
            f"previous_matching={BASELINE_PARITY_COUNTS['matching']} "
            f"previous_differing={BASELINE_PARITY_COUNTS['differing']} "
            f"previous_missing={BASELINE_PARITY_COUNTS['missing']} "
            f"delta_matching={deltas['matching']:+d} "
            f"delta_differing={deltas['differing']:+d} "
            f"delta_missing={deltas['missing']:+d} adapter_incorrect=0"
        )
        print(
            "GEOMETRY_ADAPTER_SIGNATURE "
            f"shot={shot} adapter={adapted.signature.key} "
            f"legacy={legacy.signature.key} equal={signature_equal}"
        )
        operator_receipt = compare_operator_parity(shot, adapted, legacy)
        baseline = _baseline_projection_table(adapted, description, catalog)
        baseline_operator = build_operator(baseline)
        baseline_classes = classify_circuits(
            baseline.pf_filaments,
            baseline.amc_current_channels,
            baseline.active_circuits,
            baseline.circuit_drives,
        )
        corrected_classes = classify_circuits(
            adapted.pf_filaments,
            adapted.amc_current_channels,
            adapted.active_circuits,
            adapted.circuit_drives,
        )
        correction_effects = _correction_effects(
            resolutions,
            circuit_indices,
            baseline_classes,
            corrected_classes,
            baseline.pf_filaments,
            adapted.pf_filaments,
        )
        effect_counts = Counter(correction_effects.values())
        column_adding = tuple(
            resolution
            for resolution in resolutions
            if correction_effects[resolution.circuit_identifier] == COLUMN_ADDING
        )
        resolution_counts = Counter(item.classification for item in column_adding)
        baseline_blocks = (
            baseline_operator.g_pf.shape,
            baseline_operator.g_plasma.shape,
            baseline_operator.g_passive.shape,
        )
        baseline_columns = sum(shape[1] for shape in baseline_blocks)
        adapted_columns = operator_receipt.greens.adapted_shape[1]
        legacy_columns = operator_receipt.greens.legacy_shape[1]
        adapted_block_columns = tuple(
            shape[1] for shape in operator_receipt.greens.adapted_block_shapes
        )
        legacy_block_columns = tuple(
            shape[1] for shape in operator_receipt.greens.legacy_block_shapes
        )
        current_differing_cells = operator_receipt.greens.differing_cell_count
        previous_differing_cells = BASELINE_DIFFERING_CELL_COUNTS[shot]

        assert len(resolutions) == 28
        assert sum(effect_counts.values()) == 28
        assert effect_counts[COLUMN_ADDING] == 20
        assert effect_counts[COLUMN_MODIFYING] == 8
        assert effect_counts[COLUMN_UNCHANGED] == 0
        assert len({item.circuit_identifier for item in column_adding}) == 20
        assert resolution_counts[RECOVERABLE_CURRENT_SOURCE] == 12
        assert resolution_counts[CURRENT_SOURCE_NEEDS_DECLARATION] == 8
        assert resolution_counts[CURRENT_SOURCE_ABSENT] == 0
        assert sum(resolution_counts.values()) == 20
        for resolution in resolutions:
            print(
                "GEOMETRY_ADAPTER_COLUMN_EFFECT "
                f"shot={shot} circuit={resolution.circuit_identifier} "
                f"effect={correction_effects[resolution.circuit_identifier]} "
                f"conductors={','.join(resolution.conductor_identifiers)}"
            )
        for resolution in column_adding:
            print(
                "GEOMETRY_ADAPTER_CURRENT_SOURCE "
                f"shot={shot} circuit={resolution.circuit_identifier} "
                f"classification={resolution.classification} "
                f"channel={resolution.current_channel or 'none'} "
                f"conductors={','.join(resolution.conductor_identifiers)} "
                f"evidence_topology={resolution.evidence_topology} "
                f"reason={resolution.reason}"
            )
        print(
            "GEOMETRY_ADAPTER_CURRENT_SOURCE_COUNTS "
            f"shot={shot} recoverable="
            f"{resolution_counts[RECOVERABLE_CURRENT_SOURCE]} "
            f"requires_declaration="
            f"{resolution_counts[CURRENT_SOURCE_NEEDS_DECLARATION]} "
            f"absent={resolution_counts[CURRENT_SOURCE_ABSENT]} total="
            f"{sum(resolution_counts.values())}"
        )
        print(
            "GEOMETRY_ADAPTER_COLUMN_EFFECT_COUNTS "
            f"shot={shot} adding={effect_counts[COLUMN_ADDING]} "
            f"modifying={effect_counts[COLUMN_MODIFYING]} "
            f"neither={effect_counts[COLUMN_UNCHANGED]} "
            f"total={sum(effect_counts.values())}"
        )
        assert baseline_columns == BASELINE_OPERATOR_COLUMN_COUNT
        assert legacy_columns == LEGACY_OPERATOR_COLUMN_COUNT
        assert adapted_columns == legacy_columns
        assert adapted_columns - baseline_columns == effect_counts[COLUMN_ADDING]
        assert adapted_block_columns == legacy_block_columns == (21, 84, 146)
        assert sum(adapted_block_columns) == 251
        assert operator_receipt.greens.nonfinite_mismatch_count == 0
        assert operator_receipt.unattributed_count == 0
        assert operator_receipt.unattributed_metrics == ()
        print(
            "GEOMETRY_ADAPTER_OPERATOR_COLUMNS "
            f"shot={shot} previous={baseline_columns} "
            f"adapted={adapted_columns} legacy={legacy_columns} "
            f"added={adapted_columns - baseline_columns} "
            f"baseline_blocks={baseline_blocks} "
            f"adapted_blocks={operator_receipt.greens.adapted_block_shapes} "
            f"legacy_blocks={operator_receipt.greens.legacy_block_shapes} "
            f"differing_cells={current_differing_cells} "
            f"previous_differing_cells={previous_differing_cells} "
            f"differing_cells_delta="
            f"{current_differing_cells - previous_differing_cells:+d} "
            f"nonfinite_mismatches="
            f"{operator_receipt.greens.nonfinite_mismatch_count} "
            f"unattributed_count={operator_receipt.unattributed_count}"
        )
