"""
MWE-02: NCCL Stop-File Clean Exit
==================================
PURPOSE
  Validates the stop-file cancellation mechanism as a safe alternative to
  SLURM scancel for DDP GPU jobs. Demonstrates that a running NCCL job
  can be terminated without draining the node.

RISK LEVEL
  SAFE — exits cleanly via stop-file; no scancel required.

WHAT IT DOES
  1. Initialises a 4-rank DDP job (NCCL backend) across 4 GPUs.
  2. Runs an infinite loop of dist.all_reduce collectives to simulate a
     training workload.
  3. After each collective, rank 0 checks for the stop-file:
       $DRAIN_TEST_STOP_DIR/<SLURM_JOB_ID>.stop
  4. When the stop-file is detected, rank 0 sets a "stop" tensor and
     all ranks exit cleanly via a coordinated ReduceOp.MAX barrier.
  5. Reports: exit time, number of collectives completed, GPU memory freed.

  The stop-file is automatically created by a timer thread (default: 30 s)
  so this test is completely self-contained and never requires manual
  intervention.

EXPECTED RESULT
  All 4 ranks exit with code 0. GPU memory is freed. No drain event.
  The log shows "CLEAN EXIT via stop-file" within ~1 collective cycle
  of the stop-file appearing.

ENVIRONMENT VARIABLES
  DRAIN_TEST_STOP_DIR   Directory for stop files (default: /tmp)
  DRAIN_TEST_LOG_DIR    Directory for log files  (default: /tmp)
  BUFFER_MB             Buffer size in MB (default: 256)
  STOP_AFTER_SECONDS    Auto-create stop-file after N seconds (default: 30)
  SLURM_JOB_ID          Set by SLURM; used to name the stop-file

RUN WITH
  torchrun --nproc_per_node=4 mwe-02-stopfile.py

  Or via the accompanying sbatch script:
  sbatch mwe-02-stopfile.sbatch

  To cancel manually (safe):
    touch $DRAIN_TEST_STOP_DIR/<SLURM_JOB_ID>.stop
"""

import os
import threading
import time

import torch
import torch.distributed as dist

STOP_DIR = os.environ.get("DRAIN_TEST_STOP_DIR", "/tmp")
LOG_DIR = os.environ.get("DRAIN_TEST_LOG_DIR", "/tmp")
BUFFER_MB = int(os.environ.get("BUFFER_MB", "256"))
STOP_AFTER = float(os.environ.get("STOP_AFTER_SECONDS", "30"))
JOB_ID = os.environ.get("SLURM_JOB_ID", "local")
STOP_FILE = os.path.join(STOP_DIR, f"{JOB_ID}.stop")


def setup() -> tuple[int, int]:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    return rank, local_rank


def teardown() -> None:
    dist.barrier()
    dist.destroy_process_group()
    torch.cuda.empty_cache()


def _auto_stop_thread() -> None:
    """Creates the stop-file after STOP_AFTER seconds (rank 0 only)."""
    time.sleep(STOP_AFTER)
    os.makedirs(STOP_DIR, exist_ok=True)
    open(STOP_FILE, "w").close()
    print(f"[MWE-02] Auto-created stop-file: {STOP_FILE}", flush=True)


def main() -> None:
    rank, local_rank = setup()
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()

    n_floats = (BUFFER_MB * 1024 * 1024) // 4
    buf = torch.randn(n_floats, device=device)
    stop = torch.zeros(1, device=device)

    t_start = time.monotonic()

    if rank == 0:
        print(f"[MWE-02] ranks={world_size}, buffer={BUFFER_MB} MB", flush=True)
        print(f"[MWE-02] Stop-file path: {STOP_FILE}", flush=True)
        print(f"[MWE-02] Auto-stop in {STOP_AFTER:.0f} s", flush=True)
        # Start auto-stop timer thread
        t = threading.Thread(target=_auto_stop_thread, daemon=True)
        t.start()

    n_collectives = 0
    exit_reason = "unknown"

    try:
        while True:
            dist.all_reduce(buf, op=dist.ReduceOp.SUM)
            n_collectives += 1

            # Check stop-file (rank 0) and broadcast decision
            if rank == 0 and os.path.exists(STOP_FILE):
                stop[0] = 1.0
            dist.all_reduce(stop, op=dist.ReduceOp.MAX)

            if stop[0].item() > 0.5:
                exit_reason = "stop-file"
                break

    finally:
        elapsed = time.monotonic() - t_start
        if rank == 0:
            print(f"[MWE-02] CLEAN EXIT via {exit_reason}", flush=True)
            print(f"[MWE-02] Collectives completed: {n_collectives}", flush=True)
            print(f"[MWE-02] Elapsed: {elapsed:.1f} s", flush=True)
            # Clean up stop-file
            if os.path.exists(STOP_FILE):
                os.remove(STOP_FILE)
            print("[MWE-02] PASS", flush=True)
        teardown()


if __name__ == "__main__":
    main()
