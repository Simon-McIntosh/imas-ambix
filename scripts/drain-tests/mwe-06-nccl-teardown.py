"""
MWE-06: NCCL Communicator Teardown D-State Reproducer
======================================================
PURPOSE
  Reproduces the rank-asymmetric collective drain mechanism.
  The drain was NOT caused by a collective deadlocking during the run — the
  original script printed "PASS" and completed all 20 rounds. The drain
  was caused by the NCCL communicator teardown (dist.destroy_process_group)
  entering D-state.

MECHANISM (empirically confirmed from job 1209813 error log):
  1. Run N rounds of count-mismatch all_reduce:
       Step A (all 4): all_reduce(buf_large)  — completes normally
       Step B (rank 0 only): all_reduce(buf_large)  — seq mismatch with others
       Step C (all 4): all_reduce(buf_small)  — coord collective, count mismatch
     Each completed collective leaves the NCCL ring state slightly corrupted.
  2. Individual collectives return (NCCL 2.21 truncates to minimum count).
  3. Collective timing degrades: ~156s for init + ~1000ms/round thereafter
     (NCCL recovering from ring state corruption each round).
  4. After all rounds complete, script prints "PASS" and calls teardown().
  5. dist.destroy_process_group() invokes ncclCommAbort() on all PG communicators.
  6. ncclCommAbort() cannot resolve the desynchronized NVLink ring state
     → hangs in D-state (TASK_UNINTERRUPTIBLE).
  7. PyTorch waits 600,000 ms (10 min) then logs:
     "[rankN]: Future for ProcessGroup abort timed out after 600000 ms"
  8. The ncclCommAbort background thread remains in D-state.
  9. SLURM job time limit fires → SIGTERM → process cannot exit (D-state) →
     SIGKILL → process still unkillable → "Kill task failed" → node drain.

EVIDENCE (from job 1209813 stderr, 2026-06-01):
  [rank2]: Future for ProcessGroup abort timed out after 600000 ms
  [rank1]: Future for ProcessGroup abort timed out after 600000 ms
  [rank3]: Future for ProcessGroup abort timed out after 600000 ms
  [rank0]: Future for ProcessGroup abort timed out after 600000 ms
  slurmstepd: error: *** JOB 1209813 STEPD TERMINATED ... DUE TO JOB NOT ENDING WITH SIGNALS ***
  slurmstepd: error: Container ... has 5 processes, giving up after 63 sec

RISK LEVEL
  CRITICAL — DESIGNED TO DRAIN THE NODE.
  Reserve 30 minutes for admin recovery after this test.
  Recovery after drain:
    nvidia-smi -i 0,1,2,3 --gpu-reset
    scontrol update nodename=98dci4-gpu-0003 state=resume reason=""

ENVIRONMENT VARIABLES
  NROUNDS        Number of count-mismatch rounds (default: 10)
  BUFFER_MB      Size of large buffer in MB (default: 256)

RUN WITH
  sbatch scripts/drain-tests/mwe-06-nccl-teardown.sbatch
"""

import os
import sys
import time

import torch
import torch.distributed as dist

NROUNDS = int(os.environ.get("NROUNDS", "10"))
BUFFER_MB = int(os.environ.get("BUFFER_MB", "256"))
LOG_DIR = os.environ.get("DRAIN_TEST_LOG_DIR", "/tmp")
JOB_ID = os.environ.get("SLURM_JOB_ID", "local")


def main() -> None:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    n_floats = (BUFFER_MB * 1024 * 1024) // 4
    buf_large = torch.randn(n_floats, dtype=torch.float32, device=device)
    buf_small = torch.zeros(1, dtype=torch.float32, device=device)

    if rank == 0:
        print(
            f"[MWE-06] job={JOB_ID}  ranks={world_size}  rounds={NROUNDS}  buffer={BUFFER_MB}MB",
            flush=True,
        )
        print(
            "[MWE-06] Reproducing the rank-asymmetric collective drain mechanism",
            flush=True,
        )
        print(
            "[MWE-06] Each round runs count-mismatch all_reduce to corrupt NCCL ring state",
            flush=True,
        )
        print(
            "[MWE-06] After rounds complete, ncclCommAbort() will enter D-state in teardown",
            flush=True,
        )
        print("[MWE-06] Recovery after drain:", flush=True)
        print("[MWE-06]   nvidia-smi -i 0,1,2,3 --gpu-reset", flush=True)
        print(
            '[MWE-06]   scontrol update nodename=98dci4-gpu-0003 state=resume reason=""',
            flush=True,
        )

    for rnd in range(NROUNDS):
        t_start = time.perf_counter()

        # Step A: all 4 ranks — healthy collective (seq N for all)
        dist.all_reduce(buf_large, op=dist.ReduceOp.SUM)

        # Step B: rank 0 ONLY calls an extra collective (rank 0 now at seq N+1)
        # Ranks 1-3 skip this — they stay at seq N
        if rank == 0:
            dist.all_reduce(buf_large, op=dist.ReduceOp.SUM)

        # Step C: all 4 call coordination collective
        # Rank 0 at seq N+2, counts buf_small (1 element)
        # Ranks 1-3 at seq N+1, counts buf_small (1 element)
        # NCCL matches by sequence position with buf_large for rank 0 — COUNT MISMATCH
        dist.all_reduce(buf_small, op=dist.ReduceOp.MAX)

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        if rank == 0:
            print(
                f"[MWE-06] Round {rnd + 1}/{NROUNDS}  {elapsed_ms:.1f} ms", flush=True
            )

    if rank == 0:
        print(
            f"[MWE-06] *** All {NROUNDS} rounds completed (NCCL ring state corrupted) ***",
            flush=True,
        )
        print("[MWE-06] Entering teardown — dist.destroy_process_group()", flush=True)
        print(
            "[MWE-06] EXPECTED: ncclCommAbort() hangs 600,000 ms in D-state", flush=True
        )
        print(
            "[MWE-06] SLURM time limit will fire -> SIGTERM -> unkillable -> DRAIN",
            flush=True,
        )
        sys.stdout.flush()

    # Teardown: this is where the drain happens.
    # dist.barrier() may or may not hang; dist.destroy_process_group() definitely hangs.
    dist.barrier()
    dist.destroy_process_group()

    # This line is never reached — the process drains before getting here.
    if rank == 0:
        print(
            "[MWE-06] UNEXPECTED: teardown completed without hang (no drain)",
            flush=True,
        )


if __name__ == "__main__":
    main()
