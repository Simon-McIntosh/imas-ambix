"""T1: STOP-FILE clean-exit validation (SAFE — no drain risk).

Validates that the AMBIX_STOP_FILE mechanism causes all DDP ranks to exit
cleanly without triggering SLURM's UnkillableStepTimeout → drain sequence.

HOW TO RUN:
    sbatch scripts/tests/t1_stopfile.sbatch

HOW TO TRIGGER:
    After the job starts and prints "Ready. Touch stop-file to test:", run:
        touch /work/projects/imas_gpu/stops/<JOBID>.stop

PASS CRITERIA:
    All ranks print "STOP-FILE detected → clean exit" and exit code 0 within
    ~5 s of the file being created. The SLURM job completes with exit 0.

FAIL CRITERIA:
    Ranks hang past 60 s after stop-file creation, or job exits non-zero,
    or slurmd logs "Kill task failed" and drains the node.
"""
import os
import sys
import time
import pathlib

import torch
import torch.distributed as dist

STOP_FILE = pathlib.Path(os.environ.get("AMBIX_STOP_FILE", "/tmp/t1_stop"))
POLL_S = 0.25  # check stop-file every N seconds between collectives


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")

    buf = torch.ones(2048, 2048, device=device)  # ~16 MB tensor for all_reduce

    if rank == 0:
        print(f"[T1] world={world}  stop-file={STOP_FILE}", flush=True)
        print(f"[T1] Ready. Touch stop-file to test:", flush=True)
        print(f"[T1]   touch {STOP_FILE}", flush=True)

    step = 0
    t_start = time.monotonic()

    while True:
        # ── NCCL collective (SIGTERM cannot interrupt here) ──────────────────
        t_collective = time.monotonic()
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        dt_collective = time.monotonic() - t_collective

        # ── STOP-FILE check — safe boundary between collectives ──────────────
        if STOP_FILE.exists():
            elapsed = time.monotonic() - t_start
            print(
                f"[T1] rank={rank} STOP-FILE detected at step={step} "
                f"elapsed={elapsed:.1f}s last_collective={dt_collective*1000:.1f}ms "
                f"→ clean exit",
                flush=True,
            )
            break

        step += 1
        if rank == 0 and step % 40 == 0:
            elapsed = time.monotonic() - t_start
            print(f"[T1] step={step} elapsed={elapsed:.1f}s", flush=True)

        time.sleep(POLL_S)

    # ── Clean DDP teardown ───────────────────────────────────────────────────
    dist.barrier()
    dist.destroy_process_group()
    torch.cuda.empty_cache()

    if rank == 0:
        elapsed = time.monotonic() - t_start
        print(f"[T1] ✓ PASS — all ranks exited cleanly in {elapsed:.1f}s", flush=True)

    sys.exit(0)


if __name__ == "__main__":
    main()
