"""
MWE-05: DDP/NCCL Mutual Communicator Deadlock — Drain Reproducer (v6)
=======================================================================
PURPOSE
  Reproduces the exact drain mechanism of Event #4 (job 1209813, 2026-06-01):
  a mutual deadlock across two NCCL communicators that keeps ALL 4 ranks
  permanently blocked, prevents SIGTERM delivery, and triggers
  UnkillableStepTimeout -> node drain.

  MECHANISM (why ALL 4 ranks become D-state):
    DDP creates a private NCCL communicator (comm_DDP) for gradient
    averaging, separate from the default process-group communicator
    (comm_PG).

    After a clean DDP backward step (all 4 ranks synced on comm_DDP):
      - Rank 0 calls dist.all_reduce(t) on comm_PG. This waits for all 4
        ranks to join the collective. Ranks 1-3 never make this call, so
        rank 0 blocks indefinitely.
      - Ranks 1-3 proceed to the next training step and call loss.backward().
        DDP fires the gradient all-reduce hook on comm_DDP, which waits for
        rank 0. Rank 0 is stuck in comm_PG and never joins comm_DDP.

    TWO-WAY DEADLOCK:
      comm_PG: rank 0 waiting for ranks 1-3  (ranks 1-3 are in comm_DDP)
      comm_DDP: ranks 1-3 waiting for rank 0  (rank 0 is in comm_PG)

    All 4 processes are ALIVE. No peer disconnect event fires. NCCL waits via
    NVLink P2P DMA ioctl for data that never arrives -> TASK_UNINTERRUPTIBLE
    (D-state) on all 4 ranks. SIGTERM and SIGKILL both fail. Drain follows.

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

  v5 No-run
    Collective type mismatch (AllReduce vs AllGather) designed; never ran.
    Port 29503 was still held by looping job 1209890 (torchrun retry storm
    from v4). EADDRINUSE on startup. Job failed before NCCL initialised.

  v6 Fix (this version)
    Exact replication of Event #4 mutual communicator deadlock:
      - Rank 0 calls dist.all_reduce(t) on comm_PG after its DDP backward.
        comm_PG waits for all 4 ranks to join -> rank 0 blocks indefinitely.
      - Ranks 1-3 proceed to step N+1 -> loss.backward() fires the DDP
        gradient all-reduce hook on comm_DDP -> waits for rank 0 (stuck).
      TWO-WAY DEADLOCK: comm_PG blocks on ranks 1-3; comm_DDP blocks on
      rank 0. Both sides alive, no disconnect event, NCCL waits via NVLink
      P2P DMA ioctl -> D-state all 4 ranks -> drain.

RISK LEVEL
  DESIGNED TO DRAIN THE NODE.
  Recovery after drain:
    nvidia-smi -i 0,1,2,3 --gpu-reset
    scontrol update nodename=98dci4-gpu-0003 state=resume reason=""

ENVIRONMENT VARIABLES
  MODEL_DIM           Linear layer width (default: 4096)
  BATCH_SIZE          Per-rank batch size (default: 64)
  AUTO_SCANCEL_DELAY  Seconds after READY before auto-cancel (default: 2.0)
"""

import os
import subprocess
import threading
import time

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

MODEL_DIM = int(os.environ.get("MODEL_DIM", "4096"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))
AUTO_SCANCEL_DELAY = float(os.environ.get("AUTO_SCANCEL_DELAY", "2.0"))
JOB_ID = os.environ.get("SLURM_JOB_ID", "local")


def build_model(dim: int) -> nn.Module:
    layers: list[nn.Module] = []
    for _ in range(8):
        layers.extend([nn.Linear(dim, dim), nn.ReLU()])
    return nn.Sequential(*layers)


def main() -> None:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dtype = torch.bfloat16

    if rank == 0:
        print(f"[MWE-05] *** v6: DDP/NCCL mutual communicator deadlock — Event #4 exact replica ***", flush=True)
        print(f"[MWE-05] job={JOB_ID}  ranks={world_size}  DESIGNED TO DRAIN", flush=True)
        print(f"[MWE-05] Mechanism:", flush=True)
        print(f"[MWE-05]   rank 0: extra dist.all_reduce (comm_PG) after backward", flush=True)
        print(f"[MWE-05]   ranks 1-3: step-N+1 backward -> DDP gradient all_reduce (comm_DDP)", flush=True)
        print(f"[MWE-05]   mutual deadlock -> all 4 D-state -> SIGTERM fails -> drain", flush=True)

    model = build_model(MODEL_DIM).to(device=device, dtype=dtype)
    ddp_model = DDP(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-4)

    n_params = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(f"[MWE-05] Parameters: {n_params:,}  dim: {MODEL_DIM}  batch: {BATCH_SIZE}", flush=True)

    # Warm up: NCCL init + DDP communicator established (comm_DDP created here).
    for _ in range(2):
        x = torch.randn(BATCH_SIZE, MODEL_DIM, device=device, dtype=dtype)
        ddp_model(x).sum().backward()
        optimizer.step()
        optimizer.zero_grad()

    if rank == 0:
        print(f"[MWE-05] Warm-up done. DDP+NCCL communicators established.", flush=True)

    # Step N: all 4 ranks do a clean DDP backward together (comm_DDP all-reduce).
    x = torch.randn(BATCH_SIZE, MODEL_DIM, device=device, dtype=dtype)
    ddp_model(x).sum().backward()
    optimizer.step()
    optimizer.zero_grad()

    # --- INJECT EVENT #4 BUG ---
    # Rank 0: extra all_reduce on comm_PG (waits for all 4 ranks to join).
    # Ranks 1-3: proceed to step N+1 backward (DDP fires gradient all_reduce on
    #            comm_DDP, waits for rank 0 that is stuck in comm_PG).
    # Result: TWO-WAY DEADLOCK — all 4 ranks blocked in NCCL D-state.

    if rank == 0:
        t = torch.ones(64 * 1024 * 1024, dtype=torch.float32, device=device)  # 256 MB
        print(f"[MWE-05] READY — rank 0 entering extra all_reduce (comm_PG)", flush=True)
        print(f"[MWE-05] Deadlock: ranks 1-3 step-N+1 backward (comm_DDP) will block", flush=True)
        print(f"[MWE-05] Auto-scancel in {AUTO_SCANCEL_DELAY}s", flush=True)
        print(f"[MWE-05] Recovery after drain:", flush=True)
        print(f"[MWE-05]   nvidia-smi -i 0,1,2,3 --gpu-reset", flush=True)
        print(f'[MWE-05]   scontrol update nodename=98dci4-gpu-0003 state=resume reason=""', flush=True)

        def _scancel() -> None:
            time.sleep(AUTO_SCANCEL_DELAY)
            print(f"[MWE-05] AUTO-SCANCEL: scancel {JOB_ID}", flush=True)
            subprocess.run(["scancel", JOB_ID], timeout=10)

        threading.Thread(target=_scancel, daemon=True).start()

        # HANGS: ranks 1-3 never join this call (they are in comm_DDP).
        dist.all_reduce(t)
        print(f"[MWE-05] UNEXPECTED: all_reduce returned — check deadlock timing", flush=True)

    else:
        # Ranks 1-3: start step N+1. DDP backward fires gradient all_reduce on
        # comm_DDP, which waits for rank 0 (stuck in comm_PG). HANGS.
        x = torch.randn(BATCH_SIZE, MODEL_DIM, device=device, dtype=dtype)
        ddp_model(x).sum().backward()
        optimizer.step()
        print(f"[MWE-05 rank {rank}] UNEXPECTED: backward returned — check deadlock timing", flush=True)


if __name__ == "__main__":
    main()
