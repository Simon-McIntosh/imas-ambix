"""Measure consumer-level parity between two machine geometry tables.

The comparison deliberately observes the public forward operator rather than
inferring impact from table-field counts.  It builds each operator once,
compares its positional Green's column matrix, sensor order, masked plasma
grid, limiter mask, and cache identity, then relates every observed difference
to the description fields known to differ between the two inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

import imas_ambix.gs.operator as gs_operator
from imas_ambix.data.geometry_adapter import SENSOR_COORDINATE_EXCLUSION_PREFIX

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from imas_ambix.gs.geometry import GeometryTable
    from imas_ambix.gs.operator import ForwardOperator


DEFAULT_OPERATOR_PARITY_LOG = Path(
    "/work/projects/imas_gpu/store-bench/operator-parity.log"
)

KNOWN_DIFFERING_DESCRIPTION_FIELDS = (
    "signature",
    "b_probes",
    "flux_loops",
    "pf_filaments",
    "sensor_map",
    "passive_structures",
    "provenance_flags",
    "active_circuits",
)


@dataclass(frozen=True)
class SequenceComparison:
    """Exact ordered-sequence comparison with its first disagreement."""

    equal: bool
    adapted_count: int
    legacy_count: int
    first_differing_index: int | None
    adapted_value: str | None
    legacy_value: str | None


@dataclass(frozen=True)
class MatrixComparison:
    """Positionally aligned comparison of the complete Green's column matrix."""

    equal: bool
    adapted_shape: tuple[int, int]
    legacy_shape: tuple[int, int]
    adapted_block_shapes: tuple[tuple[int, int], ...]
    legacy_block_shapes: tuple[tuple[int, int], ...]
    adapted_nonfinite_count: int
    legacy_nonfinite_count: int
    differing_cell_count: int
    nonfinite_mismatch_count: int
    max_absolute_difference: float
    max_relative_difference: float


@dataclass(frozen=True)
class GridComparison:
    """Exact comparison of plasma-grid cells retained by the limiter mask."""

    equal: bool
    adapted_count: int
    legacy_count: int
    differing_cell_count: int


@dataclass(frozen=True)
class MaskComparison:
    """Exact comparison of the full default-grid limiter masks."""

    equal: bool
    adapted_count: int
    legacy_count: int
    differing_cell_count: int


@dataclass(frozen=True)
class CacheKeyComparison:
    """Exact comparison of the operator cache identities."""

    equal: bool
    adapted_key: str
    legacy_key: str


@dataclass(frozen=True)
class ExclusionComparison:
    """Operator-excluded channels and the exclusions unique to the adapter."""

    adapted_channels: tuple[str, ...]
    legacy_channels: tuple[str, ...]
    coordinate_missing_channels: tuple[str, ...]


@dataclass(frozen=True)
class DifferenceAttribution:
    """Description fields that causally feed one differing operator metric."""

    metric: str
    fields: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class OperatorParityReceipt:
    """All consumer-level parity measurements for one shot."""

    shot: int
    channel_order: SequenceComparison
    greens: MatrixComparison
    grid: GridComparison
    limiter_mask: MaskComparison
    cache_key: CacheKeyComparison
    exclusions: ExclusionComparison
    attributions: tuple[DifferenceAttribution, ...]
    unattributed_metrics: tuple[str, ...]

    @property
    def unattributed_count(self) -> int:
        """Number of observed differences without a known field cause."""
        return len(self.unattributed_metrics)


_ATTRIBUTION_FIELDS = {
    "channel_order": ("sensor_map",),
    "greens": (
        "b_probes",
        "flux_loops",
        "pf_filaments",
        "sensor_map",
        "active_circuits",
    ),
    "cache_key": ("signature",),
}

_ATTRIBUTION_REASONS = {
    "channel_order": (
        "sensor_map declares the consumer row identities and their order"
    ),
    "greens": (
        "sensor geometry sets the rows while PF geometry and active-circuit "
        "classification set the source columns"
    ),
    "cache_key": "signature is copied directly to the operator cache identity",
}


def _compare_sequence(
    adapted: Sequence[str], legacy: Sequence[str]
) -> SequenceComparison:
    adapted_values = tuple(str(item) for item in adapted)
    legacy_values = tuple(str(item) for item in legacy)
    if adapted_values == legacy_values:
        return SequenceComparison(
            equal=True,
            adapted_count=len(adapted_values),
            legacy_count=len(legacy_values),
            first_differing_index=None,
            adapted_value=None,
            legacy_value=None,
        )

    common = min(len(adapted_values), len(legacy_values))
    first = next(
        (
            index
            for index in range(common)
            if adapted_values[index] != legacy_values[index]
        ),
        common,
    )
    return SequenceComparison(
        equal=False,
        adapted_count=len(adapted_values),
        legacy_count=len(legacy_values),
        first_differing_index=first,
        adapted_value=(adapted_values[first] if first < len(adapted_values) else None),
        legacy_value=legacy_values[first] if first < len(legacy_values) else None,
    )


def _greens_matrix(
    operator: ForwardOperator,
) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
    blocks = tuple(
        np.asarray(block, dtype=np.float64)
        for block in (operator.g_pf, operator.g_plasma, operator.g_passive)
    )
    shapes = tuple(tuple(int(size) for size in block.shape) for block in blocks)
    return np.column_stack(blocks), shapes


def _compare_matrix(
    adapted: np.ndarray,
    legacy: np.ndarray,
    *,
    adapted_block_shapes: tuple[tuple[int, int], ...],
    legacy_block_shapes: tuple[tuple[int, int], ...],
) -> MatrixComparison:
    adapted = np.asarray(adapted, dtype=np.float64)
    legacy = np.asarray(legacy, dtype=np.float64)
    rows = max(adapted.shape[0], legacy.shape[0])
    columns = max(adapted.shape[1], legacy.shape[1])
    adapted_aligned = np.zeros((rows, columns), dtype=np.float64)
    legacy_aligned = np.zeros((rows, columns), dtype=np.float64)
    adapted_aligned[: adapted.shape[0], : adapted.shape[1]] = adapted
    legacy_aligned[: legacy.shape[0], : legacy.shape[1]] = legacy

    finite_adapted = np.isfinite(adapted_aligned)
    finite_legacy = np.isfinite(legacy_aligned)
    equal_nonfinite = (
        (np.isnan(adapted_aligned) & np.isnan(legacy_aligned))
        | (np.isposinf(adapted_aligned) & np.isposinf(legacy_aligned))
        | (np.isneginf(adapted_aligned) & np.isneginf(legacy_aligned))
    )
    nonfinite_mismatch = ~(finite_adapted & finite_legacy) & ~equal_nonfinite
    finite_pairs = finite_adapted & finite_legacy
    absolute = np.zeros_like(adapted_aligned)
    absolute[finite_pairs] = np.abs(
        adapted_aligned[finite_pairs] - legacy_aligned[finite_pairs]
    )
    differing = (finite_pairs & (absolute != 0.0)) | nonfinite_mismatch

    if np.any(nonfinite_mismatch):
        max_absolute = float("inf")
        max_relative = float("inf")
    else:
        max_absolute = float(np.max(absolute, initial=0.0))
        denominator = np.maximum(np.abs(legacy_aligned), np.finfo(np.float64).tiny)
        relative = np.zeros_like(absolute)
        relative[finite_pairs] = absolute[finite_pairs] / denominator[finite_pairs]
        max_relative = float(np.max(relative, initial=0.0))

    exact = adapted.shape == legacy.shape and np.array_equal(
        adapted, legacy, equal_nan=True
    )
    return MatrixComparison(
        equal=exact,
        adapted_shape=tuple(int(size) for size in adapted.shape),
        legacy_shape=tuple(int(size) for size in legacy.shape),
        adapted_block_shapes=adapted_block_shapes,
        legacy_block_shapes=legacy_block_shapes,
        adapted_nonfinite_count=int(np.count_nonzero(~finite_adapted)),
        legacy_nonfinite_count=int(np.count_nonzero(~finite_legacy)),
        differing_cell_count=int(np.count_nonzero(differing)),
        nonfinite_mismatch_count=int(np.count_nonzero(nonfinite_mismatch)),
        max_absolute_difference=max_absolute,
        max_relative_difference=max_relative,
    )


def _compare_grid(adapted: np.ndarray, legacy: np.ndarray) -> GridComparison:
    adapted = np.asarray(adapted, dtype=np.float64).reshape(-1, 2)
    legacy = np.asarray(legacy, dtype=np.float64).reshape(-1, 2)
    common = min(adapted.shape[0], legacy.shape[0])
    equal_rows = np.all(
        (adapted[:common] == legacy[:common])
        | (np.isnan(adapted[:common]) & np.isnan(legacy[:common])),
        axis=1,
    )
    differing = int(np.count_nonzero(~equal_rows)) + abs(
        adapted.shape[0] - legacy.shape[0]
    )
    return GridComparison(
        equal=adapted.shape == legacy.shape and differing == 0,
        adapted_count=int(adapted.shape[0]),
        legacy_count=int(legacy.shape[0]),
        differing_cell_count=differing,
    )


def _limiter_mask(table: GeometryTable, *, nr: int = 9, nz: int = 13) -> np.ndarray:
    limiter_r = np.asarray(table.limiter_r, dtype=np.float64)
    limiter_z = np.asarray(table.limiter_z, dtype=np.float64)
    if limiter_r.size < 3:
        return np.ones(nr * nz, dtype=bool)
    grid_r = np.linspace(float(limiter_r.min()), float(limiter_r.max()), nr)
    grid_z = np.linspace(float(limiter_z.min()), float(limiter_z.max()), nz)
    mesh_r, mesh_z = np.meshgrid(grid_r, grid_z)
    return gs_operator._inside_polygon(
        mesh_r.ravel(), mesh_z.ravel(), limiter_r, limiter_z
    )


def _compare_mask(adapted: np.ndarray, legacy: np.ndarray) -> MaskComparison:
    adapted = np.asarray(adapted, dtype=bool).reshape(-1)
    legacy = np.asarray(legacy, dtype=bool).reshape(-1)
    common = min(adapted.size, legacy.size)
    differing = int(np.count_nonzero(adapted[:common] != legacy[:common])) + abs(
        adapted.size - legacy.size
    )
    return MaskComparison(
        equal=adapted.shape == legacy.shape and differing == 0,
        adapted_count=int(adapted.size),
        legacy_count=int(legacy.size),
        differing_cell_count=differing,
    )


def _difference_attributions(
    differences: Iterable[str],
) -> tuple[tuple[DifferenceAttribution, ...], tuple[str, ...]]:
    known = set(KNOWN_DIFFERING_DESCRIPTION_FIELDS)
    attributed: list[DifferenceAttribution] = []
    unattributed: list[str] = []
    for metric in differences:
        fields = _ATTRIBUTION_FIELDS.get(metric, ())
        if not fields or not set(fields).issubset(known):
            unattributed.append(metric)
            continue
        attributed.append(
            DifferenceAttribution(
                metric=metric,
                fields=fields,
                reason=_ATTRIBUTION_REASONS[metric],
            )
        )
    return tuple(attributed), tuple(unattributed)


def _require_operator_ready(table: GeometryTable, source: str) -> None:
    invalid = tuple(
        mapping.amb_channel
        for mapping in table.sensor_map
        if mapping.kind == "unresolved" or not np.isfinite((mapping.r, mapping.z)).all()
    )
    if invalid:
        raise ValueError(
            f"{source} table has sensor mappings without finite coordinates: "
            f"{', '.join(invalid)}"
        )


def _coordinate_missing_channels(table: GeometryTable) -> tuple[str, ...]:
    notices = tuple(
        notice.removeprefix(SENSOR_COORDINATE_EXCLUSION_PREFIX)
        for notice in table.provenance_flags
        if notice.startswith(SENSOR_COORDINATE_EXCLUSION_PREFIX)
    )
    if len(notices) > 1:
        raise ValueError("table records more than one sensor-coordinate exclusion")
    if not notices:
        return ()
    return tuple(channel.strip() for channel in notices[0].split(","))


def compare_operator_parity(
    shot: int,
    adapted_table: GeometryTable,
    legacy_table: GeometryTable,
) -> OperatorParityReceipt:
    """Build both operators once and return their consumer-level comparison."""
    _require_operator_ready(adapted_table, "adapted")
    _require_operator_ready(legacy_table, "legacy")
    adapted_operator = gs_operator.build_operator(adapted_table)
    legacy_operator = gs_operator.build_operator(legacy_table)

    channel_order = _compare_sequence(
        adapted_operator.sensor_channels, legacy_operator.sensor_channels
    )
    adapted_greens, adapted_blocks = _greens_matrix(adapted_operator)
    legacy_greens, legacy_blocks = _greens_matrix(legacy_operator)
    greens = _compare_matrix(
        adapted_greens,
        legacy_greens,
        adapted_block_shapes=adapted_blocks,
        legacy_block_shapes=legacy_blocks,
    )
    grid = _compare_grid(adapted_operator.plasma_rz, legacy_operator.plasma_rz)
    limiter_mask = _compare_mask(
        _limiter_mask(adapted_table), _limiter_mask(legacy_table)
    )
    cache_key = CacheKeyComparison(
        equal=adapted_operator.signature_key == legacy_operator.signature_key,
        adapted_key=adapted_operator.signature_key,
        legacy_key=legacy_operator.signature_key,
    )
    adapted_excluded = tuple(adapted_operator.excluded_channels)
    legacy_excluded = tuple(legacy_operator.excluded_channels)
    exclusions = ExclusionComparison(
        adapted_channels=adapted_excluded,
        legacy_channels=legacy_excluded,
        coordinate_missing_channels=_coordinate_missing_channels(adapted_table),
    )
    differences = tuple(
        metric
        for metric, equal in (
            ("channel_order", channel_order.equal),
            ("greens", greens.equal),
            ("grid", grid.equal),
            ("limiter_mask", limiter_mask.equal),
            ("cache_key", cache_key.equal),
        )
        if not equal
    )
    attributions, unattributed = _difference_attributions(differences)
    return OperatorParityReceipt(
        shot=int(shot),
        channel_order=channel_order,
        greens=greens,
        grid=grid,
        limiter_mask=limiter_mask,
        cache_key=cache_key,
        exclusions=exclusions,
        attributions=attributions,
        unattributed_metrics=unattributed,
    )


def _value(value: object | None) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return format(value, ".17g")
    return str(value)


def _shape(shape: tuple[int, ...]) -> str:
    return "x".join(str(size) for size in shape)


def format_operator_parity(receipt: OperatorParityReceipt) -> str:
    """Format a receipt with the shot repeated on every numerical record."""
    shot = receipt.shot
    channel = receipt.channel_order
    greens = receipt.greens
    grid = receipt.grid
    mask = receipt.limiter_mask
    cache = receipt.cache_key
    exclusions = receipt.exclusions
    adapted_blocks = ",".join(_shape(item) for item in greens.adapted_block_shapes)
    legacy_blocks = ",".join(_shape(item) for item in greens.legacy_block_shapes)
    attribution = ";".join(
        f"{item.metric}:{','.join(item.fields)}" for item in receipt.attributions
    )
    lines = [
        (
            f"OPERATOR_PARITY_CHANNELS shot={shot} equal={_value(channel.equal)} "
            f"adapted_count={channel.adapted_count} "
            f"legacy_count={channel.legacy_count} "
            f"first_differing_index={_value(channel.first_differing_index)} "
            f"adapted_value={_value(channel.adapted_value)} "
            f"legacy_value={_value(channel.legacy_value)}"
        ),
        (
            f"OPERATOR_PARITY_GREENS shot={shot} equal={_value(greens.equal)} "
            f"adapted_shape={_shape(greens.adapted_shape)} "
            f"legacy_shape={_shape(greens.legacy_shape)} "
            f"adapted_blocks={adapted_blocks} "
            f"legacy_blocks={legacy_blocks} "
            f"adapted_nonfinite={greens.adapted_nonfinite_count} "
            f"legacy_nonfinite={greens.legacy_nonfinite_count} "
            f"differing_cells={greens.differing_cell_count} "
            f"nonfinite_mismatches={greens.nonfinite_mismatch_count} "
            f"max_absolute_difference={_value(greens.max_absolute_difference)} "
            f"max_relative_difference={_value(greens.max_relative_difference)}"
        ),
        (
            f"OPERATOR_PARITY_GRID shot={shot} equal={_value(grid.equal)} "
            f"adapted_cells={grid.adapted_count} legacy_cells={grid.legacy_count} "
            f"differing_cells={grid.differing_cell_count}"
        ),
        (
            f"OPERATOR_PARITY_LIMITER_MASK shot={shot} equal={_value(mask.equal)} "
            f"adapted_cells={mask.adapted_count} legacy_cells={mask.legacy_count} "
            f"differing_cells={mask.differing_cell_count}"
        ),
        (
            f"OPERATOR_PARITY_EXCLUSIONS shot={shot} "
            f"operator_adapted_count={len(exclusions.adapted_channels)} "
            f"operator_legacy_count={len(exclusions.legacy_channels)} "
            f"coordinate_missing_count={len(exclusions.coordinate_missing_channels)} "
            "coordinate_missing_reason=no_finite_emitted_coordinates "
            "coordinate_missing_channels="
            f"{','.join(exclusions.coordinate_missing_channels) or 'none'}"
        ),
        (
            f"OPERATOR_PARITY_CACHE shot={shot} equal={_value(cache.equal)} "
            f"adapted_key={cache.adapted_key} legacy_key={cache.legacy_key}"
        ),
        (
            f"OPERATOR_PARITY_ATTRIBUTION shot={shot} "
            f"known_differing_fields={len(KNOWN_DIFFERING_DESCRIPTION_FIELDS)} "
            f"attributions={attribution or 'none'} "
            f"unattributed_count={receipt.unattributed_count} "
            f"unattributed_metrics={','.join(receipt.unattributed_metrics) or 'none'}"
        ),
    ]
    return "\n".join(lines)


def write_operator_parity_log(
    receipts: Iterable[OperatorParityReceipt],
    path: Path = DEFAULT_OPERATOR_PARITY_LOG,
) -> Path:
    """Write complete parity receipts to one named, durable text log."""
    rendered = tuple(format_operator_parity(receipt) for receipt in receipts)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rendered) + "\n")
    return path


__all__ = [
    "DEFAULT_OPERATOR_PARITY_LOG",
    "KNOWN_DIFFERING_DESCRIPTION_FIELDS",
    "CacheKeyComparison",
    "DifferenceAttribution",
    "ExclusionComparison",
    "GridComparison",
    "MaskComparison",
    "MatrixComparison",
    "OperatorParityReceipt",
    "SequenceComparison",
    "compare_operator_parity",
    "format_operator_parity",
    "write_operator_parity_log",
]
