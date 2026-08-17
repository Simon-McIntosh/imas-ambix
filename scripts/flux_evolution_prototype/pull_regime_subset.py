#!/usr/bin/env python3
"""Select a regime-spanning DIII-D subset from downloaded shot parquet files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

SELECTION_COLUMNS = (
    "efit_times",
    "magnetics_plasma_current",
    "magnetics_plasma_current_times",
)


def _shot_features(path: Path) -> dict[str, float | str | int | bool]:
    row = pq.read_table(path, columns=list(SELECTION_COLUMNS)).to_pylist()[0]
    times = np.asarray(row["efit_times"], dtype=np.float64)
    current_time = np.asarray(row["magnetics_plasma_current_times"], dtype=np.float64)
    current = np.asarray(row["magnetics_plasma_current"], dtype=np.float64)
    current_on_frames = np.abs(np.interp(times, current_time, current))
    peak_current = float(np.nanquantile(current_on_frames, 0.95))
    scaled = current_on_frames / max(peak_current, 1.0)
    duration_ms = float(np.ptp(times)) if len(times) > 1 else 0.0
    timestep_ms = duration_ms / max(len(times) - 1, 1)
    flat_fraction = float(np.mean(scaled >= 0.8))
    ramp_fraction = float(np.mean((scaled >= 0.15) & (scaled < 0.8)))
    flat_ms = flat_fraction * duration_ms
    ramp_ms = ramp_fraction * duration_ms
    return {
        "shot": path.stem,
        "path": str(path),
        "frame_count": int(len(times)),
        "duration_ms": duration_ms,
        "timestep_ms": timestep_ms,
        "peak_current_ka": peak_current,
        "flat_fraction": flat_fraction,
        "flat_ms": flat_ms,
        "ramp_fraction": ramp_fraction,
        "ramp_ms": ramp_ms,
        "long_flat_top": bool(flat_ms >= 500.0),
        "substantial_ramp": bool(ramp_ms >= 200.0 and ramp_fraction >= 0.15),
    }


def _ranked_unique(
    rows: list[dict[str, object]], key: str, count: int, selected: list[dict]
) -> None:
    used = {row["shot"] for row in selected}
    for row in sorted(rows, key=lambda item: float(item[key]), reverse=True):
        if row["shot"] not in used:
            selected.append(row)
            used.add(row["shot"])
        if len(selected) >= count:
            return


def select_subset(rows: list[dict[str, object]], count: int) -> list[dict]:
    """Prefer strong flat-top and ramp receipts, then fill by feature diversity."""
    selected: list[dict] = []
    anchor_count = min(5, max(3, count // 4))
    _ranked_unique(rows, "flat_ms", anchor_count, selected)
    _ranked_unique(rows, "ramp_ms", 2 * anchor_count, selected)

    features = np.array(
        [
            [
                float(row["duration_ms"]),
                float(row["peak_current_ka"]),
                float(row["flat_fraction"]),
                float(row["ramp_fraction"]),
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    scale = np.std(features, axis=0)
    scale[scale == 0.0] = 1.0
    normalised = (features - np.mean(features, axis=0)) / scale
    row_by_shot = {row["shot"]: index for index, row in enumerate(rows)}
    used = {row["shot"] for row in selected}
    while len(selected) < min(count, len(rows)):
        candidates = [
            index for index, row in enumerate(rows) if row["shot"] not in used
        ]
        chosen_indices = [row_by_shot[row["shot"]] for row in selected]
        if chosen_indices:
            distances = np.min(
                np.linalg.norm(
                    normalised[candidates, None, :] - normalised[chosen_indices, :],
                    axis=2,
                ),
                axis=1,
            )
            chosen = candidates[int(np.argmax(distances))]
        else:
            chosen = candidates[0]
        selected.append(rows[chosen])
        used.add(rows[chosen]["shot"])
    return selected[:count]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("selection_json", type=Path)
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()

    paths = sorted(args.source_dir.glob("d3d_shot_*.parquet"))
    rows = [_shot_features(path) for path in paths]
    selected = select_subset(rows, args.count)
    payload = {
        "candidate_count": len(rows),
        "selected_count": len(selected),
        "long_flat_top_count": sum(bool(row["long_flat_top"]) for row in selected),
        "substantial_ramp_count": sum(
            bool(row["substantial_ramp"]) for row in selected
        ),
        "selection": selected,
    }
    args.selection_json.parent.mkdir(parents=True, exist_ok=True)
    args.selection_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps({key: value for key, value in payload.items() if key != "selection"})
    )


if __name__ == "__main__":
    main()
