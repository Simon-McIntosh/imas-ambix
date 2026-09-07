"""Render decoder rollout degradation from two committed render receipts."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from statistics import fmean
from typing import Any

DEFAULT_OUTPUT = Path(
    "docs/figures/physics-carried-playable-plasma/seed-window/rollout-degradation.png"
)


@dataclass(frozen=True)
class RolloutDegradation:
    """Receipt-derived values required to explain one autoregressive rollout."""

    full_window_receipt: Path
    seeded_receipt: Path
    transition_indices: tuple[int, ...]
    target_times_s: tuple[float, ...]
    decoded_error_u8: tuple[float, ...]
    persistence_error_u8: tuple[float, ...]
    phase_boundary_time_s: float
    phase_boundary_transition: float
    early_transition_count: int
    late_transition_count: int
    early_ratio: float
    late_ratio: float
    aggregate_ratio: float
    seeded_ratio: float
    seeded_transition_count: int
    history_transition_count: int


def _read_receipt(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path}: receipt root must be a JSON object")
    return payload


def _required_mapping(
    payload: Mapping[str, Any], key: str, source: Path
) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{source}: missing required '{key}' block")
    return value


def _finite_number(value: Any, field: str, source: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{source}: '{field}' must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{source}: '{field}' must be a finite number")
    return number


def _positive_integer(value: Any, field: str, source: Path) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{source}: '{field}' must be a positive integer")
    return value


def _finite_sequence(
    block: Mapping[str, Any], key: str, source: Path
) -> tuple[float, ...]:
    value = block.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{source}: '{key}' must be a non-empty list")
    return tuple(
        _finite_number(item, f"{key}[{index}]", source)
        for index, item in enumerate(value)
    )


def _error_ratio(decoded: Sequence[float], persistence: Sequence[float]) -> float:
    persistence_mean = fmean(persistence)
    if persistence_mean <= 0.0:
        raise ValueError("persistence error mean must be positive")
    return fmean(decoded) / persistence_mean


def load_rollout_degradation(
    full_window_receipt: Path, seeded_receipt: Path
) -> RolloutDegradation:
    """Load and validate the curve and its distinct real-seeded reference."""
    full_path = Path(full_window_receipt)
    seeded_path = Path(seeded_receipt)
    full_payload = _read_receipt(full_path)
    seeded_payload = _read_receipt(seeded_path)

    per_frame = _required_mapping(full_payload, "per_frame_error", full_path)
    decoded = _finite_sequence(per_frame, "decoded_frame_mae_u8_scored", full_path)
    persistence = _finite_sequence(
        per_frame, "persistence_frame_mae_u8_scored", full_path
    )
    target_times = _finite_sequence(
        per_frame, "scored_transition_target_times_s", full_path
    )
    if len({len(decoded), len(persistence), len(target_times)}) != 1:
        raise ValueError(
            f"{full_path}: scored decoded, persistence, and target-time lengths "
            "must match"
        )
    if any(right <= left for left, right in pairwise(target_times)):
        raise ValueError(f"{full_path}: scored target times must be increasing")

    phase = _required_mapping(full_payload, "phase_error", full_path)
    boundary_time = _finite_number(
        phase.get("boundary_time_s"), "phase_error.boundary_time_s", full_path
    )
    early_positions = tuple(
        index for index, time_s in enumerate(target_times) if time_s < boundary_time
    )
    late_positions = tuple(
        index for index, time_s in enumerate(target_times) if time_s >= boundary_time
    )
    if not early_positions or not late_positions:
        raise ValueError(
            f"{full_path}: phase boundary must split the scored transitions"
        )
    first_late_transition = late_positions[0] + 1
    phase_boundary_transition = first_late_transition - 0.5

    seed_provenance = _required_mapping(full_payload, "seed_provenance", full_path)
    history_indices = seed_provenance.get("session_slice_indices")
    if not isinstance(history_indices, list) or not history_indices:
        raise ValueError(
            f"{full_path}: 'seed_provenance.session_slice_indices' must be a "
            "non-empty list"
        )
    history_count = len(history_indices)
    if history_count >= len(decoded):
        raise ValueError(
            f"{full_path}: real-frame history must end within the scored rollout"
        )

    seeded_error = _required_mapping(seeded_payload, "pixel_error", seeded_path)
    seeded_ratio = _finite_number(
        seeded_error.get("decoded_to_persistence_ratio"),
        "pixel_error.decoded_to_persistence_ratio",
        seeded_path,
    )
    seeded_count = _positive_integer(
        seeded_error.get("scored_frame_count"),
        "pixel_error.scored_frame_count",
        seeded_path,
    )

    early_decoded = tuple(decoded[index] for index in early_positions)
    early_persistence = tuple(persistence[index] for index in early_positions)
    late_decoded = tuple(decoded[index] for index in late_positions)
    late_persistence = tuple(persistence[index] for index in late_positions)
    return RolloutDegradation(
        full_window_receipt=full_path,
        seeded_receipt=seeded_path,
        transition_indices=tuple(range(1, len(decoded) + 1)),
        target_times_s=target_times,
        decoded_error_u8=decoded,
        persistence_error_u8=persistence,
        phase_boundary_time_s=boundary_time,
        phase_boundary_transition=phase_boundary_transition,
        early_transition_count=len(early_positions),
        late_transition_count=len(late_positions),
        early_ratio=_error_ratio(early_decoded, early_persistence),
        late_ratio=_error_ratio(late_decoded, late_persistence),
        aggregate_ratio=_error_ratio(decoded, persistence),
        seeded_ratio=seeded_ratio,
        seeded_transition_count=seeded_count,
        history_transition_count=history_count,
    )


def build_figure(data: RolloutDegradation):
    """Build a self-explaining rollout curve with receipt provenance."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: PLC0415
    from matplotlib.figure import Figure  # noqa: PLC0415

    figure = Figure(figsize=(12.0, 7.2), dpi=160)
    FigureCanvasAgg(figure)
    axes = figure.add_subplot(1, 1, 1)
    transition_end = len(data.transition_indices) + 0.5

    early_phase = axes.axvspan(
        0.5,
        data.phase_boundary_transition,
        color="#dceaf7",
        alpha=0.72,
        zorder=0,
    )
    early_phase.set_gid("early-phase")
    late_phase = axes.axvspan(
        data.phase_boundary_transition,
        transition_end,
        color="#f5e3dc",
        alpha=0.62,
        zorder=0,
    )
    late_phase.set_gid("late-phase")

    axes.plot(
        data.transition_indices,
        data.decoded_error_u8,
        color="#c23b22",
        linewidth=2.2,
        marker="o",
        markersize=4.2,
        label="Decoded frame error",
        zorder=3,
    )
    axes.plot(
        data.transition_indices,
        data.persistence_error_u8,
        color="#2166ac",
        linewidth=1.8,
        marker="o",
        markersize=3.5,
        label="Persistence error (repeat previous real frame)",
        zorder=3,
    )
    axes.axvline(
        data.phase_boundary_transition,
        color="#555555",
        linewidth=1.1,
        linestyle=":",
        zorder=2,
    )
    history_exhaustion = data.history_transition_count + 0.5
    axes.axvline(
        history_exhaustion,
        color="#111111",
        linewidth=1.4,
        linestyle="--",
        zorder=2,
    )

    maximum_error = max((*data.decoded_error_u8, *data.persistence_error_u8))
    axes.set_ylim(0.0, maximum_error * 1.27)
    axes.set_xlim(0.5, transition_end)
    phase_label_y = maximum_error * 1.17
    early_midpoint = (0.5 + data.phase_boundary_transition) / 2.0
    late_midpoint = (data.phase_boundary_transition + transition_end) / 2.0
    axes.text(
        early_midpoint,
        phase_label_y,
        f"Early: t < {data.phase_boundary_time_s:.3f} s\n"
        f"{data.early_ratio:.3f}× persistence · "
        f"{data.early_transition_count} transitions",
        ha="center",
        va="top",
        fontsize=9,
        color="#234a70",
    )
    axes.text(
        late_midpoint,
        phase_label_y,
        f"Late: t ≥ {data.phase_boundary_time_s:.3f} s\n"
        f"{data.late_ratio:.3f}× persistence · "
        f"{data.late_transition_count} transitions",
        ha="center",
        va="top",
        fontsize=9,
        color="#7b3828",
    )

    first_fully_autoregressive = data.history_transition_count + 1
    axes.annotate(
        "real-frame history exhausted\n"
        f"after transition {data.history_transition_count}",
        xy=(
            first_fully_autoregressive,
            data.decoded_error_u8[first_fully_autoregressive - 1],
        ),
        xytext=(history_exhaustion + 2.0, maximum_error * 0.87),
        arrowprops={"arrowstyle": "->", "color": "#111111", "linewidth": 1.0},
        fontsize=9,
        ha="left",
        va="center",
    )
    figure.text(
        0.085,
        0.885,
        "Real-seeded short horizon (distinct render)\n"
        f"{data.seeded_ratio:.3f}× persistence over "
        f"{data.seeded_transition_count} transitions",
        ha="left",
        va="center",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "white",
            "edgecolor": "#777777",
            "alpha": 0.92,
        },
    )
    figure.text(
        0.985,
        0.885,
        "Free-running full window\n"
        f"{data.aggregate_ratio:.3f}× persistence over "
        f"{len(data.transition_indices)} transitions",
        ha="right",
        va="center",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "white",
            "edgecolor": "#777777",
            "alpha": 0.92,
        },
    )

    axes.set_xlabel("Autoregressive transition index")
    axes.set_ylabel("Mean absolute frame error (uint8 intensity levels)")
    axes.grid(axis="y", alpha=0.22)
    axes.legend(loc="upper center", ncol=2, frameon=False, fontsize=9)
    figure.text(
        0.5,
        0.975,
        "Decoder quality collapses once real-frame history leaves the rollout",
        ha="center",
        va="top",
        fontsize=14,
        fontweight="bold",
    )
    figure.text(
        0.01,
        0.015,
        "Source receipts (different renders):\n"
        f"free-running curve: {data.full_window_receipt}\n"
        f"real-seeded reference: {data.seeded_receipt}",
        ha="left",
        va="bottom",
        fontsize=7.5,
        color="#444444",
    )
    figure.subplots_adjust(left=0.085, right=0.985, top=0.82, bottom=0.20)
    return figure


def write_figure(
    full_window_receipt: Path, seeded_receipt: Path, output: Path
) -> RolloutDegradation:
    """Read both receipts and write their combined degradation figure."""
    data = load_rollout_degradation(full_window_receipt, seeded_receipt)
    figure = build_figure(data)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    return data


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("full_window_receipt", type=Path)
    parser.add_argument("seeded_receipt", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Render the rollout degradation figure from two receipt paths."""
    args = _parser().parse_args(argv)
    data = write_figure(
        args.full_window_receipt,
        args.seeded_receipt,
        args.output,
    )
    print(
        f"wrote {args.output}: seeded {data.seeded_ratio:.3f}x over "
        f"{data.seeded_transition_count}, free-running "
        f"{data.aggregate_ratio:.3f}x over {len(data.transition_indices)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
