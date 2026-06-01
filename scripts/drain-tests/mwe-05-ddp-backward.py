"""
MWE-05: NCCL Collective Type Mismatch — All-Rank D-State Drain Reproducer (v5)
================================================================================
PURPOSE
  Reproduces the drain mechanism from Event #4 (job 1209813, 2026-06-01)
  by creating a NCCL collective type mismatch that blocks ALL 4 ranks
  simultaneously in incompatible NCCL operations. None can signal peer
  disconnection to the others, so all 4 stay stuck indefinitely → all in
  D-state → SIGTERM undeliverable → drain.

REVISION HISTORY
  v1 Finding
    dim=4096, depth=8, batch=32. ~130 steps/sec → 7ms/step. Backward+NCCL
    window ~1ms. Scancel arrived between steps → clean cancel, no drain.

  v2 Finding
    dim=8192, depth=16, batch=256, bfloat16. loss.backward() returns in 2ms
    — PyTorch DDP dispatches NCCL all-reduce asynchronously. CPU returns
    immediately; actual GPU+NCCL work runs in background on CUDA stream.
    Scancel arrived while CPU was in Python R/S state → clean cancel, no drain.

  v3 Finding
    Added torch.cuda.synchronize(device) after backward (backward=112ms with
    sync). Despite the 112ms CPU-blocking wait, the process was killed cleanly.
    Conclusion: cudaDeviceSynchronize() uses interruptible wait (S-state futex /
    CPU polling), NOT the D-state NVLink ioctl.

  v4 Finding
    Rank 0 alone in dist.all_reduce(), ranks 1-3 in time.sleep() (S-state).
    When SIGTERM fired: ranks 1-3 exited cleanly → NCCL on rank 0 detected
    peer disconnect → rank 0's all_reduce returned with error → no drain.
    Conclusion: a single D-state rank is insufficient if the other ranks can
    close NCCL channels by exiting cleanly.

  v5 Fix (this version)
    Collective TYPE MISMATCH: rank 0 calls dist.all_reduce() while ranks 1-3
    call dist.all_gather_into_tensor(). These map to different NCCL operations
    (ncclAllReduce vs ncclAllGather). NCCL matches collectives by call sequence
    number — a type mismatch causes all 4 ranks to hang indefinitely, each
    waiting for a matching operation that never arrives. ALL 4 are in D-state
    simultaneously; none can exit to signal the others. SIGTERM is
    undeliverable to all 4 → SIGKILL fails → UnkillableStepTimeout → drain.

RISK LEVEL
  ⚠️⚠️  DESIGNED TO DRAIN THE NODE.
  Recovery after drain:
    nvidia-smi -i 0,1,2,3 --gpu-reset
    scontrol update nodename=98dci4-gpu-0003 state=resume reason=""

MECHANISM
  1. All 4 ranks init DDP + NCCL, then dist.barrier() to sync.
  2. Rank 0:      dist.all_reduce(t)                → ncclAllReduce
     Ranks 1-3:  dist.all_gather_into_tensor(...)   → ncclAllGather
  3. NCCL operation type mismatch → all 4 hang in NCCL kernel (D-state via
     NVLink driver ioctl).
  4. Auto-scancel fires → SIGTERM → all 4 D-state → SIGKILL → still D-state
     → UnkillableStepTimeout (60s) → "Kill task failed" → node drains.

ENVIRONMENT VARIABLES
  TENSOR_NUMEL        Elements per rank (default: 64*1024*1024 = 256MB fp32)
  AUTO_SCANCEL_DELAY  Seconds after READY before auto-cancel (default: 2.0)
  DRAIN_TEST_LOG_DIR  Log directory (default: /tmp)
"""

import os
import subprocess
import threading
import time

import torch
import torch.distributed as dist

TENSOR_NUMEL = int(os.environ.get("TENSOR_NUMEL", str(64 * 1024 * 1024)))
AUTO_SCANCEL_DELAY = float(os.environ.get("AUTO_SCANCEL_DELAY", "2.0"))
LOG_DIR = os.environ.get("DRAIN_TEST_LOG_DIR", "/tmp")
JOB_ID = os.environ.get("SLURM_JOB_ID", "local")


def main() -> None:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"[MWE-05] *** v5: NCCL type mismatch — all-rank simultaneous D-state ***", flush=True)
        print(f"[MWE-05] job={JOB_ID}  ranks={world_size}  DESIGNED TO DRAIN", flush=True)
        print(f"[MWE-05] Mechanism: rank 0 → AllReduce, ranks 1-3 → AllGather", flush=True)
        print(f"[MWE-05]   type mismatch → all 4 D-state simultaneously → drain", flush=True)
        print(f"[MWE-05] Tensor: {TENSOR_NUMEL/1e6:.0f}M floats "
              f"({TENSOR_NUMEL*4/1e9:.2f} GB fp32 per rank)", flush=True)

    # Sync all ranks — NCCL init complete, all ranks coordinated.
    dist.barrier()

    if rank == 0:
        print(f"[MWE-05] All ranks synced. Entering mismatched collectives...", flush=True)
        print(f"[MWE-05] READY — rank 0 → AllReduce; ranks 1-3 → AllGather (mismatch)", flush=True)
        print(f"[MWE-05] Auto-scancel in {AUTO_SCANCEL_DELAY}s", flush=True)
        print(f"[MWE-05] Recovery after drain:", flush=True)
        print(f"[MWE-05]   nvidia-smi -i 0,1,2,3 --gpu-reset", flush=True)
        print(f'[MWE-05]   scontrol update nodename=98dci4-gpu-0003 state=resume reason=""', flush=True)

        def _scancel():
            time.sleep(AUTO_SCANCEL_DELAY)
            print(f"[MWE-05] AUTO-SCANCEL: issuing scancel {JOB_ID}", flush=True)
            subprocess.run(["scancel", JOB_ID], timeout=10)

        threading.Thread(target=_scancel, daemon=True).start()

        # Rank 0: AllReduce — expects ncclAllReduce from all peers.
        # Peers are calling AllGather → type mismatch → rank 0 hangs in D-state.
        t = torch.ones(TENSOR_NUMEL, dtype=torch.float32, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)

        print(f"[MWE-05] UNEXPECTED: all_reduce returned (NCCL detected mismatch?) — no drain", flush=True)

    else:
        # Ranks 1-3: AllGather (ncclAllGather) — incompatible with rank 0's AllReduce.
        # All 3 are stuck waiting for NCCL to match their AllGather, which never happens.
        output = torch.zeros(world_size * TENSOR_NUMEL, dtype=torch.float32, device=device)
        local_t = torch.ones(TENSOR_NUMEL, dtype=torch.float32, device=device)
        dist.all_gather_into_tensor(output, local_t)

        print(f"[MWE-05 rank {rank}] UNEXPECTED: all_gather returned — no drain", flush=True)


if __name__ == "__main__":
    main()
