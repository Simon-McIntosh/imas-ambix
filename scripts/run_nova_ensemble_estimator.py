"""Run or merge the Nova-propagated ensemble estimator artifact."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--members", type=int, default=16)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int)
    parser.add_argument("--merge-root", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.merge_root is not None:
        from imas_ambix.statespace.nova_ensemble_estimator import merge_shards

        merge_shards(args.merge_root, args.output_dir)
        return 0

    if (args.shard_index is None) != (args.shard_count is None):
        raise SystemExit("--shard-index and --shard-count must be supplied together")
    if args.shard_count is not None and not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("--shard-index must lie inside --shard-count")

    os.environ["JAX_PLATFORMS"] = "cpu" if args.backend == "cpu" else "cuda"
    if args.backend == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from imas_ambix.statespace.nova_ensemble_estimator import (
        EstimatorConfig,
        NovaEnsembleEstimator,
        write_result,
    )

    shard_index = args.shard_index or 0
    config = EstimatorConfig(
        backend=args.backend,
        members=args.members,
        devices=args.devices,
        seed=args.seed,
        member_offset=shard_index * args.members,
    )
    result = NovaEnsembleEstimator(config).run()
    name = "result" if args.shard_count is None else f"shard-{shard_index:03d}"
    write_result(
        result,
        args.output_dir,
        name=name,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
