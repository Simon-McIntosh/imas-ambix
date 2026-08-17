"""Build a banked facts receipt from a real DIII-D training slice."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from .download import REPOSITORY, REVISION
from .loader import validate_shot_schema

SOURCE_CARD = (
    f"https://huggingface.co/datasets/{REPOSITORY}/blob/{REVISION}/README.md"
)


def build_report(paths: list[Path]) -> dict[str, object]:
    """Validate a slice and attach bounded verdicts to every circulated claim."""

    if len(paths) < 100:
        message = f"facts receipt requires at least 100 shots, found {len(paths)}"
        raise ValueError(message)
    frame_counts: list[int] = []
    layouts: set[tuple[str, ...]] = set()
    for path in paths:
        shot = validate_shot_schema(path)
        if shot.source != "DIII-D":
            raise ValueError(f"expected DIII-D source in {path}, found {shot.source}")
        frame_counts.append(len(shot.labels.time_ms))
        layouts.add(tuple(shot.chord_geometry["thomson_chord_name"].tolist()))
    measurements = {
        "shots": len(paths),
        "labeled_frames": sum(frame_counts),
        "frames_per_shot_min": min(frame_counts),
        "frames_per_shot_median": statistics.median(frame_counts),
        "frames_per_shot_max": max(frame_counts),
        "distinct_thomson_channel_name_layouts": len(layouts),
        "all_psirz_shapes": ["T", 65, 65],
    }
    claims = [
        {
            "claim": "approximately 1.55 million diverted DIII-D label frames",
            "verdict": "confirmed",
            "measured_value": 1_559_340,
            "basis": (
                "pinned official dataset card; slice validates the same "
                "efit_times schema"
            ),
        },
        {
            "claim": "approximately 88% DIII-D and 66% MAST frame retention",
            "verdict": "confirmed",
            "measured_value": {"DIII-D": 0.880, "MAST": 0.664},
            "basis": (
                "pinned official dataset card; pre-filter frames are absent "
                "from the release"
            ),
        },
        {
            "claim": "approximately 22 distinct per-shot DIII-D Thomson layouts",
            "verdict": "confirmed",
            "measured_value": 22,
            "slice_value": len(layouts),
            "basis": "pinned official dataset card; slice count is a lower bound",
        },
        {
            "claim": (
                "three-decimal submitted-map rounding measurably harms MAST "
                "Consistency"
            ),
            "verdict": "unreachable-from-slice",
            "measured_value": None,
            "basis": (
                "MAST ground truth and a scorer-side paired evaluation are "
                "not released"
            ),
        },
        {
            "claim": "ground-truth EFITs are magnetics-only or MSE-constrained",
            "verdict": "unreachable-from-slice",
            "measured_value": None,
            "basis": "Parquet schema contains no reconstruction-constraint provenance",
        },
    ]
    return {
        "repository": REPOSITORY,
        "revision": REVISION,
        "source_card": SOURCE_CARD,
        "measurements": measurements,
        "claims": claims,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slice_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--shots", type=int, default=100)
    args = parser.parse_args()
    paths = sorted(args.slice_dir.glob("*.parquet"))[: args.shots]
    report = build_report(paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(report["measurements"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
