"""Run the frozen physics-spine stamp from an explicitly selected geometry source.

``campaign`` selects the runner's established per-campaign efm source unchanged.
``transform`` emits the level-2 machine description through the package's
transform engine and converts it with the public geometry adapter.  Level-2 does
not expose the directed poloidal angle of a field probe, so that one quantity is
carried from the campaign table by stable probe index.  Positions, conductors,
drive declarations, limiter geometry, and sensor addressing remain those of the
transformed description, and the enrichment is recorded in source provenance.

Run one source explicitly as::

    python -m imas_ambix.spine_bench.transform_geometry_source \
        --geometry-source transform --out-dir /path/to/results
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import socket
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from imas_ambix.data.description_identity import machine_description_bytes
from imas_ambix.data.geometry_adapter import geometry_table_from_description
from imas_ambix.data.machine_map import (
    PACKAGED_MACHINE_MAP_ROOT,
    load_packaged_machine_map,
)
from imas_ambix.data.paths import LEVEL2_DIR
from imas_ambix.data.transform_engine import transform_machine_description
from imas_ambix.spine_bench.runner import (
    CampaignGeometrySource,
    GeometrySource,
    run_stamp,
    write_yaml,
)
from imas_ambix.spine_bench.shots import FROZEN_SHOTSET

logger = logging.getLogger(__name__)

CAMPAIGN_SOURCE_NAME = "campaign"
TRANSFORM_SOURCE_NAME = "transform"
TRANSFORM_SOURCE_LABEL = "level2-transform"


@dataclass(frozen=True)
class PositionDivergence:
    """Coordinate separation after aligning one sensor family by address."""

    transformed_count: int
    campaign_count: int
    compared_count: int
    transformed_only_count: int
    campaign_only_count: int
    differing_count: int
    max_separation_m: float


@dataclass(frozen=True)
class CoordinateDivergence:
    """Level-2 versus efm position differences for probes and flux loops."""

    b_probes: PositionDivergence
    flux_loops: PositionDivergence


def _position_divergence(
    transformed: Any, campaign: Any, kind: str
) -> PositionDivergence:
    transformed_positions = {
        item.amb_channel: (float(item.r), float(item.z))
        for item in transformed.sensor_map
        if item.kind == kind
    }
    campaign_positions = {
        item.amb_channel: (float(item.r), float(item.z))
        for item in campaign.sensor_map
        if item.kind == kind
    }
    common = sorted(transformed_positions.keys() & campaign_positions.keys())
    separations = np.asarray(
        [
            np.hypot(
                transformed_positions[channel][0] - campaign_positions[channel][0],
                transformed_positions[channel][1] - campaign_positions[channel][1],
            )
            for channel in common
        ],
        dtype=np.float64,
    )
    return PositionDivergence(
        transformed_count=len(transformed_positions),
        campaign_count=len(campaign_positions),
        compared_count=int(separations.size),
        transformed_only_count=len(
            transformed_positions.keys() - campaign_positions.keys()
        ),
        campaign_only_count=len(
            campaign_positions.keys() - transformed_positions.keys()
        ),
        differing_count=int(np.count_nonzero(separations > 0.0)),
        max_separation_m=float(np.max(separations, initial=0.0)),
    )


def coordinate_divergence(transformed: Any, campaign: Any) -> CoordinateDivergence:
    """Compare level-2 and efm sensor positions by acquisition address."""
    return CoordinateDivergence(
        b_probes=_position_divergence(transformed, campaign, "b_probe"),
        flux_loops=_position_divergence(transformed, campaign, "flux_loop"),
    )


def _supply_probe_orientations(transformed: Any, campaign: Any) -> Any:
    """Fill the one geometry quantity level-2 does not declare for this benchmark."""
    campaign_angles = {
        mapping.amb_channel: mapping.angle_deg
        for mapping in campaign.sensor_map
        if mapping.kind == "b_probe"
    }

    mappings = []
    for mapping in transformed.sensor_map:
        if mapping.kind != "b_probe":
            mappings.append(mapping)
            continue
        angle = campaign_angles.get(mapping.amb_channel)
        if angle is None or not np.isfinite(angle):
            raise ValueError(
                f"probe mapping {mapping.amb_channel!r} has no finite campaign "
                "orientation"
            )
        mappings.append(replace(mapping, angle_deg=float(angle)))

    return replace(
        transformed,
        sensor_map=mappings,
        provenance_flags=[
            *transformed.provenance_flags,
            "sensor_map.angle_deg: joined by acquisition address from the efm "
            "campaign source because level-2 does not expose directed probe "
            "orientation",
        ],
    )


@dataclass
class TransformGeometrySource:
    """A shot-resolved level-2 transform and geometry-adapter benchmark source."""

    evidence_shot: int
    store_root: Path = LEVEL2_DIR
    catalog: Any = field(default_factory=lambda: load_packaged_machine_map("mast"))
    campaign_source: GeometrySource = field(default_factory=CampaignGeometrySource)
    label: str = TRANSFORM_SOURCE_LABEL
    _tables: dict[int, Any] = field(default_factory=dict, init=False, repr=False)
    _description_digests: dict[int, str] = field(
        default_factory=dict, init=False, repr=False
    )
    _coordinate_divergence: dict[int, CoordinateDivergence] = field(
        default_factory=dict, init=False, repr=False
    )

    def table_for(self, shot: int) -> Any:
        """Emit, adapt, qualify, and cache one shot's transform-backed table."""
        shot_id = int(shot)
        if shot_id in self._tables:
            return self._tables[shot_id]
        description = transform_machine_description(
            self.catalog, shot_id, "zarr", self.store_root
        )
        if description.status != "emitted":
            raise RuntimeError(
                f"shot {shot_id} machine description is {description.status}: "
                f"{description.detail}"
            )
        transformed = geometry_table_from_description(description, self.catalog)
        campaign = self.campaign_source.table_for(shot_id)
        if campaign is None:
            raise RuntimeError(f"campaign geometry is unavailable for shot {shot_id}")
        self._coordinate_divergence[shot_id] = coordinate_divergence(
            transformed, campaign
        )
        self._description_digests[shot_id] = hashlib.sha256(
            machine_description_bytes(description)
        ).hexdigest()
        table = _supply_probe_orientations(transformed, campaign)
        self._tables[shot_id] = table
        return table

    def divergence_for(self, shot: int) -> CoordinateDivergence:
        """Return the recorded level-2 versus efm coordinate comparison."""
        self.table_for(int(shot))
        return self._coordinate_divergence[int(shot)]

    @property
    def revision(self) -> str:
        """Digest the catalog and all description contents observed by the run."""
        if not self._description_digests:
            self.table_for(self.evidence_shot)
        digest = hashlib.sha256()
        digest.update((PACKAGED_MACHINE_MAP_ROOT / "mast.json").read_bytes())
        for value in sorted(set(self._description_digests.values())):
            digest.update(value.encode("ascii"))
        return f"sha256:{digest.hexdigest()}"

    def provenance(self) -> dict[str, Any]:
        """State the transformed source, efm orientation input, and coordinate gap."""
        reference = self.divergence_for(self.evidence_shot)
        return {
            "source": self.label,
            "reader": (
                "imas_ambix.data.transform_engine.transform_machine_description"
            ),
            "adapter": (
                "imas_ambix.data.geometry_adapter.geometry_table_from_description"
            ),
            "store_format": "zarr",
            "store_root": str(self.store_root),
            "evidence_shot": int(self.evidence_shot),
            "shots_emitted": sorted(self._description_digests),
            "description_digests": dict(sorted(self._description_digests.items())),
            "probe_orientation_source": CampaignGeometrySource.label,
            "probe_orientation_reason": (
                "level-2 does not expose directed poloidal probe orientation"
            ),
            "coordinate_divergence": {
                "reference_shot": int(self.evidence_shot),
                "b_probes": reference.b_probes.__dict__,
                "flux_loops": reference.flux_loops.__dict__,
            },
        }


def resolve_geometry_source(selection: str) -> GeometrySource:
    """Resolve one explicit benchmark source name; there is no implicit default."""
    if selection == CAMPAIGN_SOURCE_NAME:
        return CampaignGeometrySource()
    if selection == TRANSFORM_SOURCE_NAME:
        return TransformGeometrySource(evidence_shot=int(FROZEN_SHOTSET[0].shot_id))
    raise ValueError(f"unknown geometry source {selection!r}")


def _log_coordinate_divergence(source: TransformGeometrySource) -> None:
    divergence = source.divergence_for(source.evidence_shot)
    logger.info(
        "COORDINATE_DIVERGENCE reference_shot=%d "
        "probe_level2_count=%d probe_efm_count=%d probe_positions_compared=%d "
        "probe_level2_only=%d probe_efm_only=%d probe_positions_differing=%d "
        "probe_max_separation_m=%.12g loop_level2_count=%d loop_efm_count=%d "
        "loop_positions_compared=%d loop_level2_only=%d loop_efm_only=%d "
        "loop_positions_differing=%d loop_max_separation_m=%.12g",
        source.evidence_shot,
        divergence.b_probes.transformed_count,
        divergence.b_probes.campaign_count,
        divergence.b_probes.compared_count,
        divergence.b_probes.transformed_only_count,
        divergence.b_probes.campaign_only_count,
        divergence.b_probes.differing_count,
        divergence.b_probes.max_separation_m,
        divergence.flux_loops.transformed_count,
        divergence.flux_loops.campaign_count,
        divergence.flux_loops.compared_count,
        divergence.flux_loops.transformed_only_count,
        divergence.flux_loops.campaign_only_count,
        divergence.flux_loops.differing_count,
        divergence.flux_loops.max_separation_m,
    )


def _require_residual_coverage(stamp: Any) -> None:
    """Refuse a nominally successful stamp that never reached either solve arm."""
    metric = "magnetics_residual_whitened_rms"
    missing = [
        arm
        for arm in ("greens-matvec", "grid-delstar")
        if metric not in stamp.aggregate.get(arm, {})
    ]
    if not missing:
        return
    attempted = {
        arm: sum(
            item.n_slices_attempted
            for item in stamp.shots
            if item.substrate == arm and item.topology_read == "hard"
        )
        for arm in missing
    }
    raise RuntimeError(
        f"source {stamp.geometry_source!r} produced no {metric} for "
        f"{', '.join(missing)}; attempted slices={attempted}"
    )


def main(argv: list[str] | None = None) -> int:
    """Run the frozen CPU stamp under one explicitly named geometry source."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--geometry-source",
        required=True,
        choices=(CAMPAIGN_SOURCE_NAME, TRANSFORM_SOURCE_NAME),
    )
    parser.add_argument("--max-slices", type=int, default=6)
    parser.add_argument("--sigma", type=float, default=0.02)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    source = resolve_geometry_source(args.geometry_source)
    logger.info(
        "BENCHMARK_SOURCE selection=%s label=%s host=%s",
        args.geometry_source,
        source.label,
        socket.gethostname(),
    )
    if isinstance(source, TransformGeometrySource):
        _log_coordinate_divergence(source)
    stamp = run_stamp(
        created_utc=datetime.now(UTC).isoformat(),
        max_slices=args.max_slices,
        sigma=args.sigma,
        geometry_source=source,
    )
    _require_residual_coverage(stamp)
    path = write_yaml(stamp, args.out_dir)
    cohort = {item.shot_id for item in stamp.shots}
    logger.info(
        "BENCHMARK_RUN node_class=%s host=%s cohort_shots=%d source=%s wall_s=%.2f",
        stamp.machine.slurm_partition or "non-slurm",
        stamp.machine.hostname,
        len(cohort),
        stamp.geometry_source,
        stamp.complete_run_wall_s,
    )
    logger.info("AGGREGATE %s", json.dumps(stamp.aggregate, sort_keys=True))
    logger.info("STAMP %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
