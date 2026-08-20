"""Source selection and geometry receipts for the transform-backed stamp."""

from __future__ import annotations

import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from imas_ambix.data.paths import LEVEL2_DIR
from imas_ambix.spine_bench.runner import CampaignGeometrySource
from imas_ambix.spine_bench.shots import FROZEN_SHOTSET
from imas_ambix.spine_bench.transform_geometry_source import (
    IDENTITY_BOUND_SOURCE_LABEL,
    TRANSFORM_SOURCE_LABEL,
    IdentityBoundCampaignGeometrySource,
    TransformGeometrySource,
    _require_residual_coverage,
    coordinate_divergence,
    resolve_geometry_source,
)


def _sensor_mapping(
    channel: str,
    kind: str,
    index: int,
    r: float,
    z: float,
    angle_deg: float | None,
) -> SimpleNamespace:
    return SimpleNamespace(
        amb_channel=channel,
        kind=kind,
        efm_index=index,
        r=r,
        z=z,
        angle_deg=angle_deg,
        flag="",
    )


def test_campaign_selection_preserves_the_existing_source_bytes() -> None:
    """Selecting the added arm must not wrap or alter the established source."""
    shot = int(FROZEN_SHOTSET[0].shot_id)
    direct = CampaignGeometrySource().table_for(shot)
    selected = resolve_geometry_source("campaign").table_for(shot)

    assert pickle.dumps(selected, protocol=5) == pickle.dumps(direct, protocol=5)


def test_source_selection_is_explicit() -> None:
    assert isinstance(resolve_geometry_source("campaign"), CampaignGeometrySource)
    assert (
        resolve_geometry_source("identity-bound-campaign").label
        == IDENTITY_BOUND_SOURCE_LABEL
    )
    assert resolve_geometry_source("transform").label == TRANSFORM_SOURCE_LABEL
    with pytest.raises(ValueError, match="unknown geometry source"):
        resolve_geometry_source("")


@pytest.mark.skipif(
    not (LEVEL2_DIR.parent.parent / "level1" / "shots" / "21978.zarr").is_dir(),
    reason="frozen-shot level-1 store is not mounted",
)
def test_identity_bound_campaign_replaces_the_aliased_loop_join() -> None:
    shot = int(FROZEN_SHOTSET[0].shot_id)
    source = IdentityBoundCampaignGeometrySource(evidence_shot=shot)
    aliased = CampaignGeometrySource().table_for(shot)
    corrected = source.table_for(shot)
    before = {
        item.amb_channel: item
        for item in aliased.sensor_map
        if item.kind == "flux_loop"
    }
    after = {
        item.amb_channel: item
        for item in corrected.sensor_map
        if item.kind == "flux_loop"
    }

    assert len(before) == len(after) == 19
    assert len({item.efm_index for item in after.values()}) == 19
    assert (
        sum(before[channel].efm_index != after[channel].efm_index for channel in after)
        == 19
    )
    assert corrected.signature.key == "mp78-fl46-fc938-lim37-532938247d31ec5c"
    assert after["fl_cc09"].efm_index == 8
    assert after["fl_p6l_1"].efm_index == 44
    assert after["fl_p6u_1"].efm_index == 26
    assert (after["fl_p6u_1"].r, after["fl_p6u_1"].z) == pytest.approx(
        (1.402500033378601, 0.8889999985694885)
    )
    assert {
        channel: (item.r, item.z, item.flag) for channel, item in after.items()
    } == {channel: (item.r, item.z, item.flag) for channel, item in before.items()}
    provenance = source.provenance()
    assert provenance["identity_channel_count"] == 19
    assert provenance["identity_geometry_rows_rebound"] == 19
    assert provenance["identity_minimum_winning_correlation"] > 0.99999


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
            _sensor_mapping("probe-a", "b_probe", 0, 1.0, 0.0, None),
            _sensor_mapping("probe-b", "b_probe", 1, 2.0, 0.0, None),
            _sensor_mapping("probe-c", "b_probe", 2, 3.0, 0.4, None),
            _sensor_mapping("loop-a", "flux_loop", 0, 1.0, 0.0, None),
            _sensor_mapping("loop-b", "flux_loop", 1, 2.0, 0.0, None),
        ]
    )
    campaign = SimpleNamespace(
        sensor_map=[
            _sensor_mapping("probe-a", "b_probe", 0, 1.0, 0.0, 0.0),
            _sensor_mapping("probe-b", "b_probe", 1, 2.0, 0.3, 0.0),
            _sensor_mapping("probe-c", "b_probe", 2, 3.0, 0.0, 0.0),
            _sensor_mapping("probe-d", "b_probe", 3, 4.0, 0.0, 0.0),
            _sensor_mapping("loop-a", "flux_loop", 0, 1.0, 0.0, None),
            _sensor_mapping("loop-b", "flux_loop", 1, 2.3, 0.4, None),
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
def test_transform_source_binds_loop_identity_and_supplies_finite_angles() -> None:
    from imas_ambix.data.description_reader import read_geometry_table

    shot = int(FROZEN_SHOTSET[0].shot_id)
    direct = read_geometry_table(shot, store_root=Path(LEVEL2_DIR))
    source = TransformGeometrySource(evidence_shot=shot, store_root=Path(LEVEL2_DIR))

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
    loop_mappings = {
        mapping.amb_channel: mapping
        for mapping in table.sensor_map
        if mapping.kind == "flux_loop"
    }
    assert len(loop_mappings) == 19
    assert len({mapping.efm_index for mapping in loop_mappings.values()}) == 19
    assert (loop_mappings["fl_p3l_1"].r, loop_mappings["fl_p3l_1"].z) == (
        pytest.approx(1.163),
        pytest.approx(-1.08259),
    )
    assert "nominal-table" in loop_mappings["fl_p3l_1"].flag
    assert (loop_mappings["fl_p4l_1"].r, loop_mappings["fl_p4l_1"].z) == (
        pytest.approx(1.5984),
        pytest.approx(-1.04443),
    )
    assert "reconstruction" in loop_mappings["fl_p4l_1"].flag
    assert all("undecided" not in mapping.flag for mapping in loop_mappings.values())
    assert any(
        "no undecided positions present in this acquisition" in notice
        for notice in table.provenance_flags
    )
    assert loop_mappings["fl_p6u_1"].efm_index == 26
    assert (loop_mappings["fl_p6u_1"].r, loop_mappings["fl_p6u_1"].z) == (
        pytest.approx(1.402500033378601),
        pytest.approx(0.8889999985694885),
    )
    provenance = source.provenance()
    assert provenance["probe_orientation_source"] == "declared acquisition address"
    assert provenance["identity_binding"] == "unique highest waveform correlation"
    assert provenance["identity_channel_count"] == 19
    assert provenance["identity_geometry_rows_rebound"] == 19
    assert provenance["coordinate_divergence"]["reference_shot"] == shot
