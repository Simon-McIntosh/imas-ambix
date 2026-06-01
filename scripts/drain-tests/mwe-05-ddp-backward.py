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

  v6 No-drain (job 1209900)
    Theory: DDP uses separate comm_DDP, so rank 0 extra all_reduce (comm_PG)
    and ranks 1-3 DDP backward (comm_DDP) would deadlock. WRONG: in PyTorch
    2.x, DDP gradient all_reduce uses the DEFAULT process group (comm_PG).
    Rank 0 extra all_reduce and ranks 1-3 DDP backward hit the SAME NCCL
    sequence number on the SAME communicator -> accidentally completed as a
    valid all_reduce. All ranks printed "UNEXPECTED: returned". No drain.
    Lesson: DDP and dist.all_reduce share comm_PG. Must use EXPLICIT new_group
    to create truly separate communicators.

  v7 Fix (this version)
    Approach A — cross-group deadlock:
      All 4 ranks create comm_a and comm_b via new_group.
      Rank 0: dist.all_reduce on comm_a (waits for ALL ranks on comm_a).
      Ranks 1-3: dist.all_reduce on comm_b (waits for ALL ranks on comm_b).
      No rank ever calls on the other's communicator -> true deadlock.
    Approach B — ring send/recv deadlock:
      Rank N: dist.recv(src=(N+1)%W) before dist.send(dst=(N-1)%W).
      All 4 ranks blocked in recv. Nobody sends. Ring deadlock via P2P NVLink.

RISK LEVEL
  DESIGNED TO DRAIN THE NODE (if NCCL cross-group or P2P recv is D-state).
  If all operations prove S-state, job exits cleanly and reports no drain.
  Recovery after drain:
    nvidia-smi -i 0,1,2,3 --gpu-reset
    scontrol update nodename=98dci4-gpu-0003 state=resume reason=""

ENVIRONMENT VARIABLES
  DRAIN_MODE         "cross_group" (default) or "ring_deadlock"
  TENSOR_NUMEL       Elements per rank (default: 64*1024*1024 = 256MB fp32)
  AUTO_SCANCEL_DELAY Seconds after READY before auto-cancel (default: 2.0)
"""

import os
import subprocess
import threading
import time

import torch
import torch.distributed as dist

DRAIN_MODE = os.environ.get("DRAIN_MODE", "cross_group")
TENSOR_NUMEL = int(os.environ.get("TENSOR_NUMEL", str(64 * 1024 * 1024)))
AUTO_SCANCEL_DELAY = float(os.environ.get("AUTO_SCANCEL_DELAY", "2.0"))
JOB_ID = os.environ.get("SLURM_JOB_ID", "local")


def auto_scancel(delay: float) -> None:
    def _run() -> None:
        time.sleep(delay)
        print(f"[MWE-05] AUTO-SCANCEL: scancel {JOB_ID}", flush=True)
        subprocess.run(["scancel", JOB_ID], timeout=10)

    threading.Thread(target=_run, daemon=True).start()


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
        print(f"[MWE-05] *** v7: mode={DRAIN_MODE} tensor={TENSOR_NUMEL//1024//1024}M floats ***", flush=True)
        print(f"[MWE-05] job={JOB_ID}  ranks={world_size}", flush=True)
        print(f"[MWE-05] Recovery after drain:", flush=True)
        print(f"[MWE-05]   nvidia-smi -i 0,1,2,3 --gpu-reset", flush=True)
        print(f'[MWE-05]   scontrol update nodename=98dci4-gpu-0003 state=resume reason=""', flush=True)

    if DRAIN_MODE == "ring_deadlock":
        run_ring_deadlock(rank, world_size, device)
    else:
        run_cross_group(rank, world_size, device)


if __name__ == "__main__":
    main()
