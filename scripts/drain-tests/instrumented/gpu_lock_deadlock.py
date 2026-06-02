"""
gpu_lock_deadlock.py — attempt a PERMANENT NVIDIA-driver rwlock deadlock.
=========================================================================
Targets the exact wchan observed transiently during NCCL teardown
(os_acquire_rwlock_write / os_acquire_semaphore) and drives it permanent.

ROLE=holder:
  set_device(0); launch torch.cuda._sleep(~20yr) (infinite, abort-immune kernel);
  call cudaDeviceReset() — context teardown must WAIT for the kernel while holding
  the device write-lock -> blocks forever holding os_acquire_rwlock_write.
ROLE=reader (xN, same GPU):
  tight loop of cudaMalloc/free (needs the device read-lock) -> blocks behind the
  held write-lock -> permanent D.

All participants end in uninterruptible D-state on the driver lock; none can be
reaped -> SLURM UnkillableStepTimeout -> drain. NCCL-free, single GPU.
"""
import ctypes
import os
import time

import torch

ROLE = os.environ.get("ROLE", "holder")
GPU = int(os.environ.get("GPU", "0"))


def log(m: str) -> None:
    print(f"[{ROLE} pid={os.getpid()} gpu={GPU} {time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> None:
    cudart = ctypes.CDLL("libcudart.so")
    torch.cuda.set_device(GPU)
    x = torch.randn(512, 512, device=f"cuda:{GPU}")
    x = (x @ x).relu()
    torch.cuda.synchronize()

    if ROLE == "holder":
        log("launching infinite kernel, then cudaDeviceReset() — EXPECT D-state holding write-lock")
        torch.cuda._sleep(int(1e18))
        rc = cudart.cudaDeviceReset()   # waits for the infinite kernel; should block holding the lock
        log(f"cudaDeviceReset returned rc={rc} (did NOT wedge)")
        time.sleep(3600)
    else:
        log("hammering cudaMalloc/free on the same device — EXPECT D-state behind the write-lock")
        ptr = ctypes.c_void_p()
        n = 0
        while True:
            rc = cudart.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(64 * 1024 * 1024))
            if rc == 0:
                cudart.cudaFree(ptr)
            n += 1
            if n % 1000 == 0:
                log(f"{n} malloc/free cycles (still progressing rc={rc})")


if __name__ == "__main__":
    main()
