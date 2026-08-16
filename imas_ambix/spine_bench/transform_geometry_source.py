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
from imas_ambix.gs.geometry import _open_group
from imas_ambix.spine_bench.runner import (
    CampaignGeometrySource,
    GeometrySource,
    run_stamp,
    write_yaml,
)
from imas_ambix.spine_bench.shots import FROZEN_SHOTSET

logger = logging.getLogger(__name__)

CAMPAIGN_SOURCE_NAME = "campaign"
IDENTITY_BOUND_SOURCE_NAME = "identity-bound-campaign"
TRANSFORM_SOURCE_NAME = "transform"
IDENTITY_BOUND_SOURCE_LABEL = "efm-campaign-signal-identity"
TRANSFORM_SOURCE_LABEL = "level2-transform"

_IDENTITY_CORRELATION_FLOOR = 0.9999


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


@dataclass(frozen=True)
class SignalIdentityMatch:
    """One acquisition channel's independently observed EFM column identity."""

    channel: str
    efm_index: int
    correlation: float
    runner_up_correlation: float


def _signal_identity_matches(
    evidence_shot: int, channels: tuple[str, ...]
) -> tuple[SignalIdentityMatch, ...]:
    """Bind flux-loop addresses to EFM columns through their measured waveforms.

    The acquisition description coordinate is deliberately absent from this
    calculation.  Each raw ``amb`` waveform is interpolated onto the EFM time
    base and matched to the unique highest-correlation ``silop_x`` column.  The
    correlation is used only to establish stable sensor identity; geometry still
    comes from the static ``silop_r`` and ``silop_z`` arrays.
    """
    amb = _open_group(int(evidence_shot), "amb")
    efm = _open_group(int(evidence_shot), "efm")
    amb_time = np.asarray(amb["time"][:], dtype=np.float64)
    efm_time = np.asarray(efm["time"][:], dtype=np.float64)
    efm_signals = np.asarray(efm["silop_x"][:], dtype=np.float64)
    matches: list[SignalIdentityMatch] = []
    for channel in channels:
        signal = np.asarray(amb[channel][:], dtype=np.float64)
        if signal.shape != amb_time.shape:
            raise RuntimeError(
                f"channel {channel!r} has shape {signal.shape}, not its declared "
                f"time-base shape {amb_time.shape}"
            )
        sampled = np.interp(
            efm_time,
            amb_time,
            signal,
            left=np.nan,
            right=np.nan,
        )
        correlations = np.full(efm_signals.shape[1], -np.inf, dtype=np.float64)
        for index, column in enumerate(efm_signals.T):
            finite = np.isfinite(sampled) & np.isfinite(column)
            if (
                np.count_nonzero(finite) >= 3
                and np.std(sampled[finite]) > 0.0
                and np.std(column[finite]) > 0.0
            ):
                correlations[index] = np.corrcoef(sampled[finite], column[finite])[0, 1]
        order = np.argsort(correlations)[::-1]
        winner = int(order[0])
        runner_up = int(order[1])
        if correlations[winner] < _IDENTITY_CORRELATION_FLOOR:
            raise RuntimeError(
                f"channel {channel!r} best EFM identity correlation "
                f"{correlations[winner]:.9f} is below "
                f"{_IDENTITY_CORRELATION_FLOOR:.4f}"
            )
        if correlations[winner] <= correlations[runner_up]:
            raise RuntimeError(f"channel {channel!r} has no unique EFM identity")
        matches.append(
            SignalIdentityMatch(
                channel=channel,
                efm_index=winner,
                correlation=float(correlations[winner]),
                runner_up_correlation=float(correlations[runner_up]),
            )
        )
    indices = [item.efm_index for item in matches]
    if len(indices) != len(set(indices)):
        raise RuntimeError("flux-loop signal identities are not one-to-one")
    return tuple(matches)


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


@dataclass
class IdentityBoundCampaignGeometrySource:
    """The frozen campaign table with flux-loop identity bound by signal.

    The historical campaign source maps an acquisition loop to the nearest EFM
    coordinate parsed from its raw description.  Several descriptions reuse a
    placeholder coordinate, so that join is not an identity relation.  This
    source keeps the campaign table and all of its static geometry but replaces
    only those loop mappings with a one-to-one binding measured on the frozen
    cohort's evidence shot.
    """

    evidence_shot: int
    campaign_source: GeometrySource = field(default_factory=CampaignGeometrySource)
    label: str = IDENTITY_BOUND_SOURCE_LABEL
    _table: Any | None = field(default=None, init=False, repr=False)
    _matches: tuple[SignalIdentityMatch, ...] = field(
        default=(), init=False, repr=False
    )
    _aliased_count: int = field(default=0, init=False, repr=False)

    def _build(self) -> Any:
        table = self.campaign_source.table_for(self.evidence_shot)
        if table is None:
            raise RuntimeError(
                f"campaign geometry is unavailable for shot {self.evidence_shot}"
            )
        channels = tuple(
            item.amb_channel for item in table.sensor_map if item.kind == "flux_loop"
        )
        self._matches = _signal_identity_matches(self.evidence_shot, channels)
        by_channel = {item.channel: item for item in self._matches}
        loops = {item.index: item for item in table.flux_loops}
        corrected = []
        aliased_count = 0
        for mapping in table.sensor_map:
            if mapping.kind != "flux_loop":
                corrected.append(mapping)
                continue
            match = by_channel[mapping.amb_channel]
            loop = loops.get(match.efm_index)
            if loop is None:
                raise RuntimeError(
                    f"channel {mapping.amb_channel!r} matched absent EFM geometry "
                    f"column {match.efm_index}"
                )
            aliased_count += int(mapping.efm_index != match.efm_index)
            corrected.append(
                replace(
                    mapping,
                    efm_index=match.efm_index,
                    r=loop.r,
                    z=loop.z,
                    residual_m=0.0,
                    flag="",
                )
            )
        self._aliased_count = aliased_count
        return replace(
            table,
            sensor_map=corrected,
            provenance_flags=[
                *table.provenance_flags,
                "flux-loop sensor identity: acquisition waveforms joined one-to-one "
                "to EFM columns on the evidence shot; description coordinates are "
                "not identity keys",
            ],
        )

    def table_for(self, shot: int) -> Any:
        """Return the corrected table after checking campaign compatibility."""
        if self._table is None:
            self._table = self._build()
        candidate = self.campaign_source.table_for(int(shot))
        if candidate is None:
            raise RuntimeError(f"campaign geometry is unavailable for shot {shot}")
        if candidate.signature.key != self._table.signature.key:
            raise RuntimeError(
                f"shot {shot} campaign {candidate.signature.key!r} differs from "
                f"identity evidence campaign {self._table.signature.key!r}"
            )
        return self._table

    @property
    def revision(self) -> str:
        """Digest the exact address-to-column binding carried by this source."""
        if not self._matches:
            self.table_for(self.evidence_shot)
        payload = json.dumps(
            [(item.channel, item.efm_index) for item in self._matches],
            separators=(",", ":"),
        ).encode("utf-8")
        return f"sha256:{hashlib.sha256(payload).hexdigest()}"

    def provenance(self) -> dict[str, Any]:
        """Record the independent identity evidence and displaced alias count."""
        if not self._matches:
            self.table_for(self.evidence_shot)
        return {
            "source": self.label,
            "reader": "imas_ambix.gs.geometry.read_efm_geometry",
            "identity_binding": "unique highest waveform correlation",
            "identity_evidence_shot": int(self.evidence_shot),
            "identity_channel_count": len(self._matches),
            "identity_aliased_nearest_coordinate_rows_replaced": self._aliased_count,
            "identity_minimum_winning_correlation": min(
                item.correlation for item in self._matches
            ),
            "identity_minimum_winner_margin": min(
                item.correlation - item.runner_up_correlation for item in self._matches
            ),
            "identity_binding_sha256": self.revision.removeprefix("sha256:"),
        }


def resolve_geometry_source(selection: str) -> GeometrySource:
    """Resolve one explicit benchmark source name; there is no implicit default."""
    if selection == CAMPAIGN_SOURCE_NAME:
        return CampaignGeometrySource()
    if selection == IDENTITY_BOUND_SOURCE_NAME:
        return IdentityBoundCampaignGeometrySource(
            evidence_shot=int(FROZEN_SHOTSET[0].shot_id)
        )
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
        choices=(
            CAMPAIGN_SOURCE_NAME,
            IDENTITY_BOUND_SOURCE_NAME,
            TRANSFORM_SOURCE_NAME,
        ),
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
