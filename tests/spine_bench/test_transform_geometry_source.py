"""Source selection and geometry receipts for the transform-backed stamp."""

from __future__ import annotations

import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from imas_ambix.data.paths import LEVEL2_DIR
from imas_ambix.gs.geometry import SensorMapping
from imas_ambix.spine_bench.runner import CampaignGeometrySource
from imas_ambix.spine_bench.shots import FROZEN_SHOTSET
from imas_ambix.spine_bench.transform_geometry_source import (
    TRANSFORM_SOURCE_LABEL,
    TransformGeometrySource,
    _require_residual_coverage,
    coordinate_divergence,
    resolve_geometry_source,
)


def test_campaign_selection_preserves_the_existing_source_bytes() -> None:
    """Selecting the added arm must not wrap or alter the established source."""
    shot = int(FROZEN_SHOTSET[0].shot_id)
    direct = CampaignGeometrySource().table_for(shot)
    selected = resolve_geometry_source("campaign").table_for(shot)

    assert pickle.dumps(selected, protocol=5) == pickle.dumps(direct, protocol=5)


def test_source_selection_is_explicit() -> None:
    assert isinstance(resolve_geometry_source("campaign"), CampaignGeometrySource)
    assert resolve_geometry_source("transform").label == TRANSFORM_SOURCE_LABEL
    with pytest.raises(ValueError, match="unknown geometry source"):
        resolve_geometry_source("")


def test_an_empty_residual_stamp_fails_visibly() -> None:
    shots = [
        SimpleNamespace(substrate=arm, topology_read="hard", n_slices_attempted=0)
        for arm in ("greens-matvec", "grid-delstar")
    ]
    stamp = SimpleNamespace(
        geometry_source=TRANSFORM_SOURCE_LABEL,
        aggregate={"greens-matvec": {}, "grid-delstar": {}},
        shots=shots,
    )

    with pytest.raises(RuntimeError, match="attempted slices=.*0"):
        _require_residual_coverage(stamp)


def test_coordinate_divergence_counts_positions_and_maximum_separation() -> None:
    transformed = SimpleNamespace(
        sensor_map=[
            SensorMapping("probe-a", "b_probe", 0, 1.0, 0.0, None, 0.0, ""),
            SensorMapping("probe-b", "b_probe", 1, 2.0, 0.0, None, 0.0, ""),
            SensorMapping("probe-c", "b_probe", 2, 3.0, 0.4, None, 0.0, ""),
            SensorMapping("loop-a", "flux_loop", 0, 1.0, 0.0, None, 0.0, ""),
            SensorMapping("loop-b", "flux_loop", 1, 2.0, 0.0, None, 0.0, ""),
        ]
    )
    campaign = SimpleNamespace(
        sensor_map=[
            SensorMapping("probe-a", "b_probe", 0, 1.0, 0.0, 0.0, 0.0, ""),
            SensorMapping("probe-b", "b_probe", 1, 2.0, 0.3, 0.0, 0.0, ""),
            SensorMapping("probe-c", "b_probe", 2, 3.0, 0.0, 0.0, 0.0, ""),
            SensorMapping("probe-d", "b_probe", 3, 4.0, 0.0, 0.0, 0.0, ""),
            SensorMapping("loop-a", "flux_loop", 0, 1.0, 0.0, None, 0.0, ""),
            SensorMapping("loop-b", "flux_loop", 1, 2.3, 0.4, None, 0.0, ""),
        ]
    )

    receipt = coordinate_divergence(transformed, campaign)

    assert receipt.b_probes.transformed_count == 3
    assert receipt.b_probes.campaign_count == 4
    assert receipt.b_probes.compared_count == 3
    assert receipt.b_probes.transformed_only_count == 0
    assert receipt.b_probes.campaign_only_count == 1
    assert receipt.b_probes.differing_count == 2
    assert receipt.b_probes.max_separation_m == pytest.approx(0.4)
    assert receipt.flux_loops.compared_count == 2
    assert receipt.flux_loops.differing_count == 1
    assert receipt.flux_loops.max_separation_m == pytest.approx(0.5)


@pytest.mark.skipif(
    not (LEVEL2_DIR / f"{FROZEN_SHOTSET[0].shot_id}.zarr").is_dir(),
    reason="frozen-shot level-2 store is not mounted",
)
def test_transform_source_keeps_level2_positions_and_supplies_finite_angles() -> None:
    from imas_ambix.data.geometry_adapter import geometry_table_from_description
    from imas_ambix.data.machine_map import load_packaged_machine_map
    from imas_ambix.data.transform_engine import transform_machine_description

    shot = int(FROZEN_SHOTSET[0].shot_id)
    catalog = load_packaged_machine_map("mast")
    direct = geometry_table_from_description(
        transform_machine_description(catalog, shot, "zarr", Path(LEVEL2_DIR)),
        catalog,
    )
    source = TransformGeometrySource(
        evidence_shot=shot, store_root=Path(LEVEL2_DIR), catalog=catalog
    )

    table = source.table_for(shot)
    campaign_angles = {
        mapping.amb_channel: mapping.angle_deg
        for mapping in CampaignGeometrySource().table_for(shot).sensor_map
        if mapping.kind == "b_probe"
    }

    assert [(item.r, item.z) for item in table.b_probes] == [
        (item.r, item.z) for item in direct.b_probes
    ]
    assert [(item.r, item.z) for item in table.flux_loops] == [
        (item.r, item.z) for item in direct.flux_loops
    ]
    assert all(
        mapping.kind != "b_probe" or np.isfinite(mapping.angle_deg)
        for mapping in table.sensor_map
    )
    assert all(
        mapping.kind != "b_probe"
        or mapping.angle_deg == campaign_angles[mapping.amb_channel]
        for mapping in table.sensor_map
    )
    provenance = source.provenance()
    assert provenance["probe_orientation_source"] == CampaignGeometrySource.label
    assert provenance["coordinate_divergence"]["reference_shot"] == shot
