"""
MWE-04: Simple NCCL All-Reduce + Scancel
==========================================
PURPOSE
  Tests whether issuing `scancel` during a simple NCCL all_reduce on
  H200 NVLink puts the processes into D-state (causing node drain) or
  R-state (CPU spin-wait, SIGKILL succeeds, no drain).

  Prior empirical results (job 1209723 on this cluster) showed that a
  simple all_reduce puts ranks into R-state (CPU spin-wait), NOT D-state.
  SIGKILL succeeds; no drain occurs. This MWE re-confirms that finding
  and is expected to NOT drain the node.

RISK LEVEL
  ⚠️  LOW DRAIN RISK (expected R-state; empirically confirmed SAFE on this
  cluster). If your cluster uses a different NCCL collective implementation
  or older NVLink, D-state is possible. Monitor carefully.

WHAT IT DOES
  1. Initialises a 4-rank DDP job (NCCL backend) across 4 GPUs.
  2. Enters an infinite loop of dist.all_reduce on a large buffer.
  3. Prints "READY" when stable — this is the signal to scancel.
  4. A timeout-based watchdog (default: 120 s) self-terminates the job
     if scancel is not issued, making this test safe for automated runs.
  5. After scancel, the node state should remain idle/allocated.

HOW TO TEST
  1. Submit:
       sbatch mwe-04-nccl-scancel.sbatch
       # note the job ID from the output

  2. Wait for "READY" in the log:
       tail -f <log-file>

  3. Issue scancel (this is the drain trigger):
       scancel <jobid>

  4. Check node state after ~70 s (UnkillableStepTimeout + margin):
       sinfo -N -n 98dci4-gpu-0003 --noheader -o "%T"
       # Expected: "idle" or "allocated" (R-state, no drain)
       # Unexpected: "drain" (D-state, NVLink entered kernel sleep)

  5. Optionally monitor process state from the compute node before
     scancel to confirm R-state:
       ssh <compute-node>
       cat /proc/<pid>/status | grep State
       # Expect: "R (running)" or "S (sleeping)"

EXPECTED RESULT
  Node remains idle after scancel. No drain event. Confirms that simple
  NCCL all_reduce does not create D-state processes on this hardware.

ENVIRONMENT VARIABLES
  BUFFER_MB          Buffer size in MB per all-reduce (default: 512)
  WATCHDOG_SECONDS   Self-terminate after N seconds (default: 120)
  DRAIN_TEST_LOG_DIR Log directory (default: /tmp)
"""

import os
import sys
import signal
import threading
import time

import torch
import torch.distributed as dist

BUFFER_MB = int(os.environ.get("BUFFER_MB", "512"))
WATCHDOG_SEC = float(os.environ.get("WATCHDOG_SECONDS", "120"))
LOG_DIR = os.environ.get("DRAIN_TEST_LOG_DIR", "/tmp")
JOB_ID = os.environ.get("SLURM_JOB_ID", "local")

_stop_event = threading.Event()


def _watchdog(rank: int) -> None:
    """Self-terminate after WATCHDOG_SECONDS if scancel was not issued."""
    if not _stop_event.wait(timeout=WATCHDOG_SEC):
        if rank == 0:
            print(f"[MWE-04] WATCHDOG: {WATCHDOG_SEC:.0f}s elapsed; self-terminating.", flush=True)
        os.kill(os.getpid(), signal.SIGTERM)


def setup() -> tuple[int, int]:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    return rank, local_rank


def main() -> None:
    rank, local_rank = setup()
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()

    n_floats = (BUFFER_MB * 1024 * 1024) // 4
    buf = torch.randn(n_floats, device=device)

    if rank == 0:
        print(f"[MWE-04] ranks={world_size}, buffer={BUFFER_MB} MB", flush=True)
        print(f"[MWE-04] job={JOB_ID}  pid={os.getpid()}", flush=True)
        print(f"[MWE-04] Watchdog: {WATCHDOG_SEC:.0f} s", flush=True)
        wt = threading.Thread(target=_watchdog, args=(rank,), daemon=True)
        wt.start()

    # Warm up
    for _ in range(3):
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize(device)

    if rank == 0:
        print(f"[MWE-04] READY — issue 'scancel {JOB_ID}' now", flush=True)
        print(f"[MWE-04] Expected result: node stays idle (R-state, SIGKILL works)", flush=True)
        print(f"[MWE-04] Drain result:    node drains (D-state, NVLink kernel sleep)", flush=True)

    n = 0
    t_last_report = time.monotonic()
    while not _stop_event.is_set():
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        n += 1
        if rank == 0 and time.monotonic() - t_last_report > 10:
            print(f"[MWE-04] Still running: {n} collectives", flush=True)
            t_last_report = time.monotonic()

    if rank == 0:
        print(f"[MWE-04] Exiting cleanly after {n} collectives.", flush=True)

    dist.barrier()
    dist.destroy_process_group()
    torch.cuda.empty_cache()
    if rank == 0:
        print("[MWE-04] DONE", flush=True)


if __name__ == "__main__":
    main()
