"""CLI: stamp the physics-spine performance + quality baseline to a versioned YAML.

    OMP_NUM_THREADS=1 uv run python -m scripts.spine_benchmark

Produces imas_ambix/spine_bench/results/physics-spine-<shotset>-<commit>-<host>.yaml —
a schema-versioned, git-commit + machine keyed evolution metric (perf + physics quality)
on the FROZEN shot set. See imas_ambix/spine_bench/README.md.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("spine_benchmark")

DEFAULT_OUT = Path("imas_ambix/spine_bench/results")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-slices", type=int, default=6)
    ap.add_argument("--sigma", type=float, default=0.02)
    ap.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT))
    ap.add_argument(
        "--shots",
        type=str,
        default="",
        help="explicit comma list (overrides the frozen set — for testing)",
    )
    return ap


def main() -> int:
    args = build_parser().parse_args()
    from imas_ambix.spine_bench.runner import run_stamp, write_yaml
    from imas_ambix.spine_bench.shots import BenchShot

    shots = None
    if args.shots:
        shots = [
            BenchShot(shot_id=int(s), role="ad-hoc")
            for s in args.shots.split(",")
            if s.strip()
        ]
        logger.info(
            "ad-hoc shot set (NOT the frozen metric): %s", [s.shot_id for s in shots]
        )

    created = datetime.now(UTC).isoformat()
    logger.info("stamping physics-spine baseline @ %s", created)
    stamp = run_stamp(
        created_utc=created, max_slices=args.max_slices, sigma=args.sigma, shots=shots
    )
    path = write_yaml(stamp, Path(args.out_dir))

    logger.info(
        "\n=== aggregate (median across %d frozen shots) ===",
        len({s.shot_id for s in stamp.shots}),
    )
    logger.info(json.dumps(stamp.aggregate, indent=2))
    logger.info("\nwrote %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
