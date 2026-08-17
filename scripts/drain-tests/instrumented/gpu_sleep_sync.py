"""
gpu_sleep_sync.py — minimal, deterministic, single-GPU drain reproducer.
========================================================================
Isolates the ROOT mechanism with NO NCCL and NO distributed code:

  1. Launch a GPU kernel that runs for ~20 years (torch.cuda._sleep), async.
  2. Call torch.cuda.synchronize() -> the CPU thread blocks inside the
     NVIDIA driver ioctl waiting for that kernel to finish. It never does.

The synchronize wait is UNINTERRUPTIBLE (D-state) in the NVIDIA driver
(wchan = os_acquire_semaphore / os_acquire_rwlock_* / a cuda ioctl). Because
the kernel never completes, the wait is PERMANENT — the process survives
SIGKILL. When the SLURM --time limit fires, slurmstepd cannot reap the step
within UnkillableStepTimeout (60 s) -> "Kill task failed" -> node DRAIN.

This is the same kernel-level wait observed transiently during NCCL teardown
(job 1210225); here it is made permanent and reproducible without NCCL, which
proves the drain is a CUDA-driver property, not an NCCL-specific one. NCCL is
merely the most common way a production job ends up with a stuck GPU kernel.

Single process, single GPU -> smallest possible blast radius for recovery.

Recovery (admin): nvidia-smi -i <id> --gpu-reset (or reboot if it hangs);
                   scontrol update nodename=98dci4-gpu-0003 state=resume reason=""
"""

import os
import time

import torch


def main() -> None:
    pid = os.getpid()
    print(f"[sleepsync pid={pid}] init CUDA on cuda:0", flush=True)
    torch.cuda.set_device(0)
    x = torch.randn(1024, 1024, device="cuda")
    x = (x @ x).relu()
    torch.cuda.synchronize()
    print(
        f"[sleepsync pid={pid}] launching ~20yr GPU kernel (async), then synchronize",
        flush=True,
    )
    print(
        f"[sleepsync pid={pid}] >>> EXPECT PERMANENT D-STATE in nvidia driver; SIGKILL cannot reap <<<",
        flush=True,
    )
    torch.cuda._sleep(int(1e18))  # async dispatch; CPU returns immediately
    torch.cuda.synchronize()  # blocks forever in the driver ioctl -> D-state
    print(
        f"[sleepsync pid={pid}] synchronize RETURNED (did NOT wedge — unexpected)",
        flush=True,
    )
    time.sleep(3600)


if __name__ == "__main__":
    main()
