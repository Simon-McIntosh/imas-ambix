"""
gpu_block_sync.py — deterministic single-GPU SUSTAINED-drain reproducer.
========================================================================
Forces a PERMANENT uninterruptible (D) wait inside the NVIDIA driver:

  1. torch.cuda._sleep(~20 years) — a GPU kernel that never completes and does
     NOT poll any abort flag (unlike an NCCL kernel, which ncclCommAbort can
     cancel — that is why a healthy-GPU NCCL hang stays transient/killable).
  2. A *blocking* CUDA event sync (cudaEventBlockingSync): the CPU thread sleeps
     on a driver semaphore (wchan os_acquire_semaphore — the exact function in
     open GPU kernel-module failure mode) waiting for that kernel. It never
     returns -> permanent D-state.

We also set the device sync policy to blocking via ctypes as belt-and-braces, so
even torch.cuda.synchronize() blocks rather than spins.

A D-state task ignores SIGTERM and SIGKILL. When the SLURM --time limit fires,
slurmstepd cannot reap the step within UnkillableStepTimeout (60 s) ->
"Kill task failed" -> NODE DRAIN. This is the controlled, NCCL-free validation
of the mechanism behind production drains 1208980 / 1209813.

Recovery (admin): nvidia-smi -i <id> --gpu-reset (or reboot if it hangs);
                  scontrol update nodename=98dci4-gpu-0003 state=resume reason=""
"""

import ctypes
import os
import time

import torch


def _set_blocking_sync() -> None:
    """cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync=0x04) before context init."""
    try:
        cudart = ctypes.CDLL("libcudart.so")
        rc = cudart.cudaSetDeviceFlags(ctypes.c_uint(0x04))
        print(f"[blocksync] cudaSetDeviceFlags(BlockingSync) rc={rc}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(
            f"[blocksync] cudaSetDeviceFlags failed ({e}); relying on blocking Event",
            flush=True,
        )


def main() -> None:
    pid = os.getpid()
    _set_blocking_sync()
    torch.cuda.set_device(0)
    x = torch.randn(1024, 1024, device="cuda")
    x = (x @ x).relu()
    torch.cuda.synchronize()
    print(f"[blocksync pid={pid}] launching ~20yr GPU kernel (async)", flush=True)
    print(
        f"[blocksync pid={pid}] >>> blocking-event sync -> PERMANENT D in nvidia driver; SIGKILL cannot reap <<<",
        flush=True,
    )
    torch.cuda._sleep(int(1e18))  # async; never completes
    ev = torch.cuda.Event(blocking=True)  # cudaEventBlockingSync
    ev.record()
    ev.synchronize()  # sleeps on driver semaphore forever -> D-state
    print(
        f"[blocksync pid={pid}] synchronize RETURNED (did NOT wedge — unexpected)",
        flush=True,
    )
    time.sleep(3600)


if __name__ == "__main__":
    main()
