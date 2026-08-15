"""Consumer-level receipts for adapter-versus-legacy operator parity."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import imas_ambix.data.operator_parity as parity_module
from imas_ambix.data.geometry_adapter import geometry_table_from_description
from imas_ambix.data.machine_map import load_packaged_machine_map
from imas_ambix.data.operator_parity import (
    DEFAULT_OPERATOR_PARITY_LOG,
    KNOWN_DIFFERING_DESCRIPTION_FIELDS,
    compare_operator_parity,
    write_operator_parity_log,
)
from imas_ambix.data.transform_engine import transform_machine_description
from imas_ambix.gs.geometry import SensorMapping, build_table_for_shot

LEVEL2_ROOT = Path("/work/projects/imas_gpu/mast/level2/shots")


def test_parity_rejects_unresolved_sensor_before_operator_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapping = SensorMapping(
        amb_channel="missing-position",
        kind="unresolved",
        efm_index=-1,
        r=float("nan"),
        z=float("nan"),
        angle_deg=None,
        residual_m=float("nan"),
        flag="no emitted sensor geometry",
    )
    invalid_table = SimpleNamespace(sensor_map=[mapping])
    build_calls: list[object] = []
    monkeypatch.setattr(
        parity_module.gs_operator,
        "build_operator",
        build_calls.append,
    )

    with pytest.raises(ValueError, match="missing-position"):
        compare_operator_parity(1, invalid_table, invalid_table)

    assert build_calls == []


@pytest.mark.skipif(
    not LEVEL2_ROOT.is_dir(),
    reason="local level-2 geometry stores are not mounted",
)
def test_range_first_operators_have_fully_attributed_parity_receipts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = load_packaged_machine_map("mast")
    shots = tuple(machine_map.first_shot for machine_map in catalog.maps)
    original_build_operator = parity_module.gs_operator.build_operator
    built_tables: list[object] = []

    def counted_build_operator(table: object) -> object:
        built_tables.append(table)
        return original_build_operator(table)

    monkeypatch.setattr(
        parity_module.gs_operator,
        "build_operator",
        counted_build_operator,
    )

    assert len(shots) == 2
    previous_row_counts = {11_766: 81, 12_417: 97}
    expected_adapter_only_exclusions = {
        11_766: (),
        12_417: ("fl_cc10", "fl_p6u_1"),
    }
    receipts = []
    for shot in shots:
        assert (LEVEL2_ROOT / f"{shot}.zarr").is_dir()
        description = transform_machine_description(
            catalog,
            shot,
            "zarr",
            LEVEL2_ROOT,
        )
        adapted_table = geometry_table_from_description(description, catalog)
        legacy_table = build_table_for_shot(shot)
        builds_before = len(built_tables)

        receipt = compare_operator_parity(shot, adapted_table, legacy_table)

        assert len(built_tables) - builds_before == 2
        assert built_tables[-2] is adapted_table
        assert built_tables[-1] is legacy_table
        assert receipt.shot == shot
        assert receipt.unattributed_count == 0
        assert receipt.unattributed_metrics == ()
        assert receipt.grid.equal
        assert receipt.grid.differing_cell_count == 0
        assert receipt.limiter_mask.equal
        assert receipt.limiter_mask.differing_cell_count == 0
        assert not np.isnan(receipt.greens.max_absolute_difference)
        assert not np.isnan(receipt.greens.max_relative_difference)
        assert receipt.greens.adapted_nonfinite_count == 0
        assert receipt.greens.legacy_nonfinite_count == 0
        assert receipt.greens.nonfinite_mismatch_count == 0

        expected_exclusions = expected_adapter_only_exclusions[shot]
        assert receipt.exclusions.coordinate_missing_channels == expected_exclusions
        assert receipt.channel_order.legacy_count == previous_row_counts[shot]
        assert receipt.channel_order.adapted_count == (
            previous_row_counts[shot] - len(expected_exclusions)
        )
        assert all(mapping.kind != "unresolved" for mapping in adapted_table.sensor_map)
        assert all(
            np.isfinite((mapping.r, mapping.z)).all()
            for mapping in adapted_table.sensor_map
        )
        if expected_exclusions:
            exclusion_notice = next(
                notice
                for notice in adapted_table.provenance_flags
                if notice.startswith(
                    "sensor_map: excluded sensor addresses with no finite emitted "
                    "coordinates:"
                )
            )
            assert all(channel in exclusion_notice for channel in expected_exclusions)

        channel_order = receipt.channel_order
        if channel_order.equal:
            assert channel_order.first_differing_index is None
            assert channel_order.adapted_value is None
            assert channel_order.legacy_value is None
        else:
            assert channel_order.first_differing_index is not None

        differing_metrics = {
            metric
            for metric, equal in (
                ("channel_order", receipt.channel_order.equal),
                ("greens", receipt.greens.equal),
                ("grid", receipt.grid.equal),
                ("limiter_mask", receipt.limiter_mask.equal),
                ("cache_key", receipt.cache_key.equal),
            )
            if not equal
        }
        attributed_metrics = {item.metric for item in receipt.attributions}
        assert attributed_metrics == differing_metrics
        for attribution in receipt.attributions:
            assert attribution.fields
            assert set(attribution.fields).issubset(KNOWN_DIFFERING_DESCRIPTION_FIELDS)
            assert attribution.reason
        receipts.append(receipt)

    assert len(built_tables) == 2 * len(shots)
    log_path = write_operator_parity_log(receipts)
    log_lines = log_path.read_text().splitlines()

    assert log_path == DEFAULT_OPERATOR_PARITY_LOG
    assert log_lines
    assert len(log_lines) == 7 * len(shots)
    assert all(any(f"shot={shot}" in line for shot in shots) for line in log_lines)
    for shot in shots:
        shot_lines = [line for line in log_lines if f"shot={shot}" in line]
        assert len(shot_lines) == 7
        assert any("OPERATOR_PARITY_CHANNELS" in line for line in shot_lines)
        assert any("first_differing_index=" in line for line in shot_lines)
        assert any("OPERATOR_PARITY_GREENS" in line for line in shot_lines)
        assert any("max_absolute_difference=" in line for line in shot_lines)
        assert any("max_relative_difference=" in line for line in shot_lines)
        assert any("OPERATOR_PARITY_GRID" in line for line in shot_lines)
        assert any("OPERATOR_PARITY_LIMITER_MASK" in line for line in shot_lines)
        assert any("differing_cells=" in line for line in shot_lines)
        exclusion_line = next(
            line for line in shot_lines if "OPERATOR_PARITY_EXCLUSIONS" in line
        )
        assert (
            "coordinate_missing_reason=no_finite_emitted_coordinates" in exclusion_line
        )
        assert (
            f"coordinate_missing_count={len(expected_adapter_only_exclusions[shot])}"
        ) in exclusion_line
        assert all(
            channel in exclusion_line
            for channel in expected_adapter_only_exclusions[shot]
        )
        assert any("OPERATOR_PARITY_CACHE" in line for line in shot_lines)
        assert any("unattributed_count=0" in line for line in shot_lines)

    print(log_path.read_text(), end="")
