"""Encode one shard of the rbb corpus on a single GPU.

Called as one task of a SLURM array job. Each task is allocated --gres=gpu:1
by SLURM itself, so CUDA_VISIBLE_DEVICES is set correctly by the scheduler
(avoids the multiprocessing-vs-CUDA-init issue we hit when sharding inside
a single 4-GPU Python process).

Usage:

    sbatch --array=0-3 ... scripts/encode_one_shard.sbatch
    # the sbatch wrapper sets SHARD_ID=$SLURM_ARRAY_TASK_ID and N_SHARDS=4
    # then calls: python scripts/encode_one_shard.py \
    #                 --shard $SHARD_ID --n-shards $N_SHARDS --camera rbb ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True, help="this shard index (0-based)")
    parser.add_argument("--n-shards", type=int, required=True, help="total number of shards")
    parser.add_argument("--camera", default="rbb")
    parser.add_argument("--output-dir", default="/work/projects/imas_gpu/mast/tokens")
    parser.add_argument("--report", required=True)
    parser.add_argument("--vocab-version", default="v1")
    parser.add_argument("--max-shots", type=int, default=None)
    parser.add_argument("--no-skip-existing", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.path.insert(0, str(REPO_ROOT))

    from imas_ambix.data.encoding import bulk_encode_frames
    from imas_ambix.tokenizer.frames import OpenMagvit2Tokenizer

    # Build the shotlist + shard it deterministically by stride
    manifest = json.loads(
        Path("/work/projects/imas_gpu/mast/manifests/level1-cameras.json").read_text()
    )
    all_shots = sorted(manifest["shot_ids"])
    l1_root = Path("/work/projects/imas_gpu/mast/level1/shots")
    shots = [s for s in all_shots if (l1_root / f"{s}.zarr" / args.camera).is_dir()]
    if args.max_shots:
        shots = shots[: args.max_shots]
    shard = shots[args.shard :: args.n_shards]
    print(
        f"[shard{args.shard}/{args.n_shards}] {len(shard)} of {len(shots)} shots, "
        f"camera={args.camera}",
        flush=True,
    )

    t0 = time.monotonic()
    tok = OpenMagvit2Tokenizer(device="cuda", batch_size=8)
    t_ready = time.monotonic() - t0
    print(f"[shard{args.shard}] daemon ready in {t_ready:.1f}s", flush=True)

    reports = bulk_encode_frames(
        shard,
        args.camera,
        lambda: tok,
        max_workers=1,
        skip_existing=not args.no_skip_existing,
        vocab_version=args.vocab_version,
    )
    elapsed = time.monotonic() - t0
    n_ok = sum(1 for r in reports if r.error is None)
    n_fail = len(reports) - n_ok
    print(
        f"[shard{args.shard}] done: {n_ok} ok, {n_fail} fail, {elapsed:.1f}s "
        f"({elapsed / max(len(shard), 1):.2f}s/shot)",
        flush=True,
    )

    Path(args.report).write_text(json.dumps({
        "shard": args.shard,
        "n_shards": args.n_shards,
        "n_shots": len(shard),
        "n_ok": n_ok,
        "n_fail": n_fail,
        "daemon_startup_s": round(t_ready, 1),
        "total_s": round(elapsed, 1),
        "errors": [
            {"shot_id": r.shot_id, "error": r.error}
            for r in reports if r.error is not None
        ][:50],
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
