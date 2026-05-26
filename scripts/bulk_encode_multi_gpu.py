"""Multi-GPU driver for Open-MAGVIT2 frame bulk-encode.

Saturates the 4× H200 Group A reservation by spawning one driver process
per GPU, each holding a persistent ``OpenMagvit2Tokenizer`` daemon. The
12,000-shot rbb corpus shards evenly across GPUs; each process pays the
checkpoint-load cost ONCE then streams shots through its daemon.

Usage (inside SLURM with --gres=gpu:4):

    python scripts/bulk_encode_multi_gpu.py \
        --camera rbb \
        --output-dir /work/projects/imas_gpu/mast/tokens \
        --report /tmp/bulk-rbb-${SLURM_JOB_ID}.json \
        --gpus 4 \
        [--shot-ids 15085,15086,...]  # default: all L1 cameras manifest
        [--max-shots N]                # cap for testing

Per-GPU monitoring is started before encoding and logged to
``${OUTPUT_DIR}/gpu-util-${SLURM_JOB_ID}.csv`` — sampled every 2 s via
nvidia-smi.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _worker(
    gpu_id: int,
    shot_ids: list[int],
    camera: str,
    output_dir: str,
    vocab_version: str,
    skip_existing: bool,
    result_queue: "mp.Queue",
) -> None:
    """Per-GPU worker: pin to one GPU, spawn one daemon, encode the shard.

    DO NOT set CUDA_VISIBLE_DEVICES here — inside a SLURM --gres=gpu:N
    allocation, overriding CVD in a child process breaks the CUDA driver
    init (cuda_avail=False even though device_count()=1). Instead leave
    CVD alone (all GPUs visible to the cgroup) and pass an explicit
    cuda:<i> device string so each daemon binds to a different physical
    device.
    """
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    sys.path.insert(0, str(REPO_ROOT))
    from imas_ambix.data.encoding import bulk_encode_frames
    from imas_ambix.tokenizer.frames import OpenMagvit2Tokenizer

    t0 = time.monotonic()
    tok = OpenMagvit2Tokenizer(device=f"cuda:{gpu_id}", batch_size=8)
    t_ready = time.monotonic() - t0
    print(f"[gpu{gpu_id}] daemon ready in {t_ready:.1f}s, encoding {len(shot_ids)} shots", flush=True)

    reports = bulk_encode_frames(
        shot_ids,
        camera,
        lambda: tok,
        max_workers=1,
        skip_existing=skip_existing,
        vocab_version=vocab_version,
    )
    elapsed = time.monotonic() - t0
    n_ok = sum(1 for r in reports if r.error is None)
    n_fail = len(reports) - n_ok
    print(
        f"[gpu{gpu_id}] done: {n_ok} ok, {n_fail} fail, {elapsed:.1f}s "
        f"({elapsed / max(len(shot_ids), 1):.2f}s/shot)",
        flush=True,
    )
    result_queue.put({
        "gpu_id": gpu_id,
        "n_shots": len(shot_ids),
        "n_ok": n_ok,
        "n_fail": n_fail,
        "daemon_startup_s": t_ready,
        "total_s": elapsed,
        "errors": [
            {"shot_id": r.shot_id, "error": r.error}
            for r in reports if r.error is not None
        ][:20],
    })


def _start_gpu_sampler(csv_path: Path, interval_s: float = 2.0) -> subprocess.Popen:
    """Start a background nvidia-smi sampler that logs all GPUs every interval_s.

    CSV columns: timestamp, gpu_idx, util_gpu_pct, util_mem_pct, mem_used_mib, power_w
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # nvidia-smi --query-gpu writes one row per sample per GPU
    cmd = [
        "nvidia-smi",
        "--query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,power.draw",
        "--format=csv,noheader,nounits",
        f"--loop-ms={int(interval_s * 1000)}",
    ]
    f = open(csv_path, "w")
    f.write("timestamp,gpu_idx,util_gpu_pct,util_mem_pct,mem_used_mib,power_w\n")
    f.flush()
    return subprocess.Popen(cmd, stdout=f, stderr=subprocess.DEVNULL)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", default="rbb")
    parser.add_argument("--output-dir", default="/work/projects/imas_gpu/mast/tokens")
    parser.add_argument("--report", required=True, help="Final report JSON output path")
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument(
        "--shot-ids",
        default=None,
        help="Comma-separated shot ids; default = the L1 cameras manifest",
    )
    parser.add_argument("--max-shots", type=int, default=None)
    parser.add_argument("--vocab-version", default="v1")
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument(
        "--gpu-util-csv",
        default=None,
        help="Path for nvidia-smi sampling CSV (default: ${OUTPUT_DIR}/gpu-util-${SLURM_JOB_ID}.csv)",
    )
    parser.add_argument("--gpu-util-interval-s", type=float, default=2.0)
    args = parser.parse_args()

    # Resolve shot list
    if args.shot_ids:
        shots = [int(s) for s in args.shot_ids.split(",")]
    else:
        manifest = json.loads(
            Path("/work/projects/imas_gpu/mast/manifests/level1-cameras.json").read_text()
        )
        shots = sorted(manifest["shot_ids"])
        # Filter to shots that actually have the camera dir on disk
        l1_root = Path("/work/projects/imas_gpu/mast/level1/shots")
        shots = [s for s in shots if (l1_root / f"{s}.zarr" / args.camera).is_dir()]
        print(f"[main] {len(shots)} shots in manifest with {args.camera!r} on disk", flush=True)
    if args.max_shots:
        shots = shots[: args.max_shots]

    # Shard across GPUs
    n_gpus = args.gpus
    shards = [shots[i::n_gpus] for i in range(n_gpus)]
    print(f"[main] sharding {len(shots)} shots across {n_gpus} GPUs: {[len(s) for s in shards]}", flush=True)

    # Start GPU sampler
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    gpu_csv = Path(
        args.gpu_util_csv
        or f"{args.output_dir}/gpu-util-{job_id}.csv"
    )
    sampler = _start_gpu_sampler(gpu_csv, args.gpu_util_interval_s)
    print(f"[main] gpu sampler started → {gpu_csv}", flush=True)

    # Spawn workers
    mp.set_start_method("spawn", force=True)
    result_queue: mp.Queue = mp.Queue()
    procs = []
    t_start = time.monotonic()
    for gpu_id in range(n_gpus):
        if not shards[gpu_id]:
            continue
        p = mp.Process(
            target=_worker,
            args=(
                gpu_id,
                shards[gpu_id],
                args.camera,
                args.output_dir,
                args.vocab_version,
                not args.no_skip_existing,
                result_queue,
            ),
            name=f"encode-gpu{gpu_id}",
        )
        p.start()
        procs.append(p)

    # Collect results
    results = []
    for _ in procs:
        results.append(result_queue.get())
    for p in procs:
        p.join()
    elapsed = time.monotonic() - t_start

    # Stop sampler
    sampler.terminate()
    try:
        sampler.wait(timeout=5)
    except subprocess.TimeoutExpired:
        sampler.kill()

    # Summarise GPU util from the CSV
    try:
        rows = gpu_csv.read_text().strip().splitlines()[1:]  # skip header
        per_gpu_util: dict[int, list[float]] = {}
        for r in rows:
            parts = [p.strip() for p in r.split(",")]
            if len(parts) < 4:
                continue
            try:
                gpu_idx = int(parts[1])
                util = float(parts[2])
            except ValueError:
                continue
            per_gpu_util.setdefault(gpu_idx, []).append(util)
        util_summary = {
            f"gpu{idx}_util_mean_pct": round(sum(vs) / len(vs), 1)
            for idx, vs in per_gpu_util.items() if vs
        } | {
            f"gpu{idx}_util_p95_pct": round(sorted(vs)[int(len(vs) * 0.95)], 1)
            for idx, vs in per_gpu_util.items() if vs
        }
    except Exception as exc:  # noqa: BLE001
        util_summary = {"error": str(exc)}

    report = {
        "job_id": job_id,
        "camera": args.camera,
        "n_shots_total": len(shots),
        "n_gpus": n_gpus,
        "elapsed_s": round(elapsed, 1),
        "throughput_shots_per_s": round(len(shots) / max(elapsed, 1), 3),
        "per_gpu": results,
        "gpu_util_csv": str(gpu_csv),
        "gpu_util_summary": util_summary,
    }
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
