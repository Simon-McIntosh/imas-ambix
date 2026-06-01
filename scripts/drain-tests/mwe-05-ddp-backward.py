"""
MWE-05: NCCL D-State Drain Reproducer (v8)
==========================================
PURPOSE
  Empirically determines which NCCL operations enter D-state (TASK_UNINTERRUPTIBLE)
  on this H200 NVLink cluster, reproducing the drain mechanism of Event #4
  (job 1209813, 2026-06-01).

REVISION HISTORY
  v1 Finding: 7ms/step backward window too short → scancel between steps → clean cancel.
  v2 Finding: DDP backward async (2ms CPU return) → CPU never blocks → clean cancel.
  v3 Finding: cudaDeviceSynchronize() after backward → S-state (interruptible futex).
              112ms wait, clean kill. cudaStreamSync is NOT the D-state path.
  v4 Finding: rank 0 alone in all_reduce, ranks 1-3 sleeping → NCCL detects peer
              disconnect → rank 0 returns, no drain. Single-rank blocking insufficient.
  v5 No-run:  AllReduce/AllGather mismatch designed; EADDRINUSE from prior looping job.
  v6 No-drain (job 1209900): DDP uses comm_PG (not a separate comm_DDP). Rank 0 extra
              all_reduce + ranks 1-3 DDP backward → same NCCL sequence → accidental
              match. All ranks returned "UNEXPECTED". Lesson: must use new_group.
  v7 No-drain (jobs 1209902, 1209905):
     cross_group mode: new_group(all_ranks) × 2. Rank 0 on comm_a, ranks 1-3 on
       comm_b. Returned immediately — NCCL heartbeat/watchdog aborted within 2s.
     ring_deadlock mode: all ranks recv(src=(N+1)%4) before send(dst=(N-1)%4).
       SIGTERM interrupted the blocking recv cleanly (S-state).
  v8 (this version):
     seq_mismatch mode: genuine NCCL sequence mismatch on the SAME communicator.
       All 4 ranks stuck simultaneously: rank 0 calling seq N, ranks 1-3 at seq N+1.
       Reproduces the EXACT original MWE-01 bug pattern that caused Event #4.
       NCCL heartbeat disabled via env vars so watchdog does not abort early.
     gpu_sleep mode: all 4 ranks launch torch.cuda.sleep(infinite) then exit the
       Python main thread. Tests whether cuCtxDestroy with pending GPU kernel is
       D-state. Exercises the GPU context destruction cleanup path.

RISK LEVEL
  seq_mismatch and gpu_sleep are DESIGNED TO DRAIN THE NODE.
  Recovery after drain:
    nvidia-smi -i 0,1,2,3 --gpu-reset
    scontrol update nodename=98dci4-gpu-0003 state=resume reason=""

ENVIRONMENT VARIABLES
  DRAIN_MODE         "seq_mismatch" (default) | "cross_group" | "ring_deadlock" |
                     "gpu_sleep"
  TENSOR_NUMEL       Elements per rank (default: 64*1024*1024 = 256 MB fp32)
  AUTO_SCANCEL_DELAY Seconds after READY before auto-cancel (default: 2.0)
"""

import os
import subprocess
import threading
import time

import torch
import torch.distributed as dist

DRAIN_MODE = os.environ.get("DRAIN_MODE", "seq_mismatch")
TENSOR_NUMEL = int(os.environ.get("TENSOR_NUMEL", str(64 * 1024 * 1024)))
AUTO_SCANCEL_DELAY = float(os.environ.get("AUTO_SCANCEL_DELAY", "2.0"))
JOB_ID = os.environ.get("SLURM_JOB_ID", "local")


def auto_scancel(delay: float) -> None:
    def _run() -> None:
        time.sleep(delay)
        print(f"[MWE-05] AUTO-SCANCEL: scancel {JOB_ID}", flush=True)
        subprocess.run(["scancel", JOB_ID], timeout=10)

    threading.Thread(target=_run, daemon=True).start()


def run_seq_mismatch(rank: int, world_size: int, device: torch.device) -> None:
    """Faithful reproduction of the original Event #4 (MWE-01 v1 bug).

    The original bug was: the per-round loop in MWE-01 v1 ran two all_reduce
    calls for rank 0 (timing + blind-window measurement) but only one for
    ranks 1-3, then a coordination all_reduce for all 4. This created:

      All 4 call all_reduce(buf_large)   → NCCL seq 1, completes.
      Rank 0: all_reduce(buf_large)      → enters NCCL seq 2, count=N, blocks
      Ranks 1-3: all_reduce(buf_small)   → enters NCCL seq 2, count=1, blocks

    All 4 are simultaneously stuck at NCCL sequence 2 on the SAME communicator
    with MISMATCHED element counts (N vs 1). The NCCL data-transfer protocol
    deadlocks because neither side can satisfy the other's size expectation.
    If this wait is in D-state (TASK_UNINTERRUPTIBLE), SIGTERM cannot wake it
    → SIGKILL also fails → UnkillableStepTimeout → node drain.

    Key difference from prior (failed) design: ranks 1-3 enter IMMEDIATELY
    without waiting — they are stuck at seq 2 BEFORE rank 0 joins. The
    auto_scancel fires as a background thread; rank 0 then enters the
    all_reduce and both sides are deadlocked simultaneously when SIGTERM arrives.
    """
    # buf_large and buf_small must have DIFFERENT element counts — that is the
    # count mismatch that corrupts the NCCL collective handshake.
    buf_large = torch.randn(TENSOR_NUMEL, dtype=torch.float32, device=device)
    buf_small = torch.zeros(1, dtype=torch.float32, device=device)

    # Warm-up: all 4 participate, NCCL seq advances to 1 for all ranks
    dist.all_reduce(buf_large)

    if rank == 0:
        # Schedule scancel in background BEFORE entering the deadlock
        print("[MWE-05] READY seq_mismatch (count mismatch) — entering deadlock", flush=True)
        print("[MWE-05]   rank 0: all_reduce(buf_large) count=N  at NCCL seq 2", flush=True)
        print("[MWE-05]   ranks 1-3: all_reduce(buf_small) count=1 at NCCL seq 2", flush=True)
        print("[MWE-05]   NCCL count mismatch → mutual D-state deadlock", flush=True)
        print(f"[MWE-05] Auto-scancel in {AUTO_SCANCEL_DELAY}s", flush=True)
        auto_scancel(AUTO_SCANCEL_DELAY)  # Background thread — returns immediately
        # Rank 0 now enters seq 2 with large buffer (count=N)
        # Ranks 1-3 are already blocked at seq 2 with small buffer (count=1)
        dist.all_reduce(buf_large)
    else:
        # Ranks 1-3 skip rank-0's block and immediately enter seq 2 with count=1
        # They will be stuck here BEFORE rank 0 arrives
        dist.all_reduce(buf_small)

    print(f"[MWE-05 rank {rank}] UNEXPECTED: seq_mismatch returned (no drain?)", flush=True)


def run_gpu_sleep(rank: int, world_size: int, device: torch.device) -> None:
    """Test whether cuCtxDestroy is D-state when GPU has a pending kernel.

    Mechanism:
      - All 4 ranks launch torch.cuda.sleep(~infinite) — an async CUDA kernel
        that spins for ~30 years (10^18 GPU clock cycles at ~10 GHz).
      - CPU returns immediately from the non-blocking dispatch.
      - All 4 ranks print READY and the auto-scancel fires.
      - SIGTERM arrives → Python tries to exit → cuCtxDestroy called.
      - If cuCtxDestroy blocks on the spinning GPU kernel: D-state → drain.
      - If cuCtxDestroy aborts the kernel: clean exit.

    This isolates the GPU context destruction cleanup path from NCCL/NVLink.
    """
    dist.barrier()

    # Non-blocking GPU kernel launch: CPU returns immediately, GPU spins
    torch.cuda.sleep(int(1e18))  # ~30 years at 10 GHz GPU clock

    if rank == 0:
        print("[MWE-05] READY gpu_sleep — all 4 GPUs have infinite sleep kernels running", flush=True)
        print("[MWE-05]   torch.cuda.sleep(1e18) dispatched async on all 4 GPUs", flush=True)
        print("[MWE-05]   CPU returned immediately — GPU kernels still spinning", flush=True)
        print("[MWE-05]   Testing: is cuCtxDestroy D-state with pending GPU kernel?", flush=True)
        print(f"[MWE-05] Auto-scancel in {AUTO_SCANCEL_DELAY}s", flush=True)
        auto_scancel(AUTO_SCANCEL_DELAY)

    # Sleep long enough for auto-scancel to fire; do NOT sync
    # (we deliberately do NOT call torch.cuda.synchronize here)
    time.sleep(3600)  # Wait for scancel
    print(f"[MWE-05 rank {rank}] UNEXPECTED: time.sleep returned (job not cancelled?)", flush=True)


def run_cross_group(rank: int, world_size: int, device: torch.device) -> None:
    """Rank 0 on comm_a, ranks 1-3 on comm_b. Both alive, neither ever
    participates in the other's collective. True cross-communicator deadlock."""
    all_ranks = list(range(world_size))

    comm_a = dist.new_group(ranks=all_ranks)
    comm_b = dist.new_group(ranks=all_ranks)

    dist.barrier()  # Ensure both groups are fully initialised

    t = torch.ones(TENSOR_NUMEL, dtype=torch.float32, device=device)

    if rank == 0:
        print(f"[MWE-05] READY cross_group — rank 0 all_reduce on comm_a", flush=True)
        print(f"[MWE-05]   ranks 1-3 are on comm_b — no cross-group match", flush=True)
        print(f"[MWE-05] Auto-scancel in {AUTO_SCANCEL_DELAY}s", flush=True)
        auto_scancel(AUTO_SCANCEL_DELAY)
        dist.all_reduce(t, group=comm_a)
        print(f"[MWE-05] UNEXPECTED (rank 0): comm_a all_reduce returned", flush=True)
    else:
        dist.all_reduce(t, group=comm_b)
        print(f"[MWE-05 rank {rank}] UNEXPECTED: comm_b all_reduce returned", flush=True)


def run_ring_deadlock(rank: int, world_size: int, device: torch.device) -> None:
    """All ranks simultaneously wait to recv from next rank before sending.
    Nobody ever sends first -> circular deadlock on P2P NVLink path."""
    dist.barrier()

    t = torch.zeros(TENSOR_NUMEL, dtype=torch.float32, device=device)
    src = (rank + 1) % world_size
    dst = (rank - 1) % world_size

    if rank == 0:
        print(f"[MWE-05] READY ring_deadlock — all 4 ranks entering recv", flush=True)
        print(f"[MWE-05]   rank N: recv(src=(N+1)%4) before send(dst=(N-1)%4)", flush=True)
        print(f"[MWE-05]   nobody sends first -> ring deadlock via P2P NVLink", flush=True)
        print(f"[MWE-05] Auto-scancel in {AUTO_SCANCEL_DELAY}s", flush=True)
        auto_scancel(AUTO_SCANCEL_DELAY)

    dist.recv(t, src=src)   # All 4 blocked here simultaneously
    dist.send(t, dst=dst)   # Never reached
    print(f"[MWE-05 rank {rank}] UNEXPECTED: recv returned", flush=True)


def main() -> None:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"[MWE-05] *** v8: mode={DRAIN_MODE} tensor={TENSOR_NUMEL//1024//1024}M floats ***", flush=True)
        print(f"[MWE-05] job={JOB_ID}  ranks={world_size}", flush=True)
        print(f"[MWE-05] Recovery after drain:", flush=True)
        print(f"[MWE-05]   nvidia-smi -i 0,1,2,3 --gpu-reset", flush=True)
        print(f'[MWE-05]   scontrol update nodename=98dci4-gpu-0003 state=resume reason=""', flush=True)

    if DRAIN_MODE == "seq_mismatch":
        run_seq_mismatch(rank, world_size, device)
    elif DRAIN_MODE == "gpu_sleep":
        run_gpu_sleep(rank, world_size, device)
    elif DRAIN_MODE == "ring_deadlock":
        run_ring_deadlock(rank, world_size, device)
    else:
        run_cross_group(rank, world_size, device)


if __name__ == "__main__":
    main()
