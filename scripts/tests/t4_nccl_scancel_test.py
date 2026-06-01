"""T4: Minimal NCCL collective + scancel drain reproducer.

╔══════════════════════════════════════════════════════════════════════════╗
║  WARNING: THIS JOB WILL DRAIN 98dci4-gpu-0003 (betelgeuse)              ║
║  Run only with an SDCC admin ready to issue:                             ║
║    nvidia-smi --gpu-reset -i 0,1,2,3  (or reboot the node)              ║
║    scontrol update nodename=98dci4-gpu-0003 state=resume reason=""       ║
║  Expected recovery time: 20–30 minutes (GPU reset required)             ║
╚══════════════════════════════════════════════════════════════════════════╝

WHAT THIS PROVES:
    Reproduces the 2026-05-28 13:52 drain (Event 3 / Drain #4).
    All DDP ranks enter kernel D-state during an NCCL all_reduce collective.
    scancel → SIGTERM → ranks cannot process signal mid-C++ → D-state ×N
    → UnkillableStepTimeout (60s) → "Kill task failed" → node DRAIN.

    This is distinct from T3: it proves the CUDA/NCCL-specific mechanism
    and leaves orphaned CUDA contexts that require nvidia-smi --gpu-reset.

MECHANISM TIMELINE (expected):
    T+0s  : Job starts, 2 DDP ranks initialise NCCL communicators on GPU 0/1
    T+10s : Both ranks are inside dist.all_reduce() on a large tensor
    T+10s : (Operator) scancel <JOBID>   ← SLURM sends SIGTERM to both ranks
    T+10s : Python signal handler is QUEUED — cannot fire inside NCCL C++
    T+70s : UnkillableStepTimeout expires (60s after SIGTERM)
             slurmd: "Kill task failed for job <JOBID>"
             Node state → DRAIN + orphaned CUDA contexts on GPU 0/1

RECOVERY (after drain):
    1. nvidia-smi --gpu-reset -i 0,1  (clear orphaned CUDA contexts)
    2. scontrol update nodename=98dci4-gpu-0003 state=resume reason=""
"""
import os
import signal
import sys
import time

import torch
import torch.distributed as dist

# Large tensor → long all_reduce → maximises window for SIGTERM during collective
NUMEL = int(os.environ.get("T4_NUMEL", str(64 * 1024 * 1024)))  # 256 MB float32


def _sigterm_handler(signum: int, _frame) -> None:
    # This handler WILL be queued by the OS but will only run after the
    # current NCCL C++ call returns. If we're deep in an NCCL collective,
    # this function body may not execute for 10s–60s.
    t = time.monotonic()
    rank = dist.get_rank() if dist.is_initialized() else -1
    print(
        f"[T4] rank={rank} SIGTERM handler fired at t={t:.3f} "
        f"(may be very delayed relative to signal delivery)",
        flush=True,
    )
    # NOTE: we do NOT call sys.exit() here — we just log and let the loop
    # eventually detect the signal. In practice with NCCL blocking, this
    # entire body may never execute before the process is force-killed.


def main() -> None:
    signal.signal(signal.SIGTERM, _sigterm_handler)  # installed but ineffective mid-NCCL

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")

    buf = torch.ones(NUMEL, device=device, dtype=torch.float32)

    if rank == 0:
        print(f"[T4] world={world}  tensor={NUMEL*4/1e6:.0f}MB", flush=True)
        print(f"[T4] Running infinite NCCL all_reduce loop. No stop-file.", flush=True)
        print(f"[T4] To reproduce drain: scancel <JOBID>", flush=True)
        print(f"[T4] Expected: D-state → 60s → Kill task failed → DRAIN", flush=True)

    step = 0
    while True:
        # This call blocks both ranks in uninterruptible NCCL C++.
        # With NUMEL=64M floats (256 MB) the operation takes ~100–500 ms.
        # Any SIGTERM during this window is undeliverable until it returns.
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        step += 1
        if rank == 0 and step % 20 == 0:
            print(f"[T4] step={step}", flush=True)
        # No stop-file check — intentional; demonstrates the vulnerability


if __name__ == "__main__":
    main()
