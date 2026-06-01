"""
MWE-05: NCCL Collective Hang — Confirmed D-State Drain Reproducer (v4)
=======================================================================
PURPOSE
  Reproduces the confirmed drain mechanism from the accidental MWE-01 run
  (Event #4, job 1209813, 2026-06-01): rank 0 calls a collective that no
  other rank joins, causing rank 0 to block indefinitely inside the NCCL
  kernel. The blocking wait is via NVLink driver ioctl → D-state
  (TASK_UNINTERRUPTIBLE). SIGTERM and SIGKILL cannot reach a D-state
  process. SLURM's UnkillableStepTimeout (60 s) fires → "Kill task failed"
  → node drain.

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
    sync). Despite the 112ms CPU-blocking wait, the process was still killed
    cleanly. Conclusion: cudaDeviceSynchronize() uses an interruptible wait
    mechanism (S-state futex / CPU polling), NOT the D-state NVLink ioctl.

  v4 Fix (this version)
    The ONLY confirmed D-state mechanism in this cluster is the NCCL kernel
    blocking on a MISSING PEER: when rank 0 enters a collective and other
    ranks do NOT, rank 0 waits forever in ncclAllReduce via NVLink P2P ioctl
    → D-state. This is exactly what caused the accidental Event #4 drain
    (collective mismatch bug in MWE-01). This version reproduces it
    deliberately: rank 0 calls dist.all_reduce(); ranks 1-3 sleep.
    Auto-scancel fires 2s after READY → SIGTERM to rank 0 in D-state →
    torchrun SIGKILLed → rank 0 orphaned D-state process → drain.

RISK LEVEL
  ⚠️⚠️  DESIGNED TO DRAIN THE NODE (this is the confirmed mechanism).
  Recovery after drain:
    nvidia-smi -i 0,1,2,3 --gpu-reset
    scontrol update nodename=98dci4-gpu-0003 state=resume reason=""

MECHANISM
  dist.all_reduce() (rank 0 only) → NCCL kernel waits for peers via
  NVLink P2P ioctl → D-state (TASK_UNINTERRUPTIBLE).

  torchrun receives SIGTERM → forwards to rank 0 → rank 0 in D-state
  → torchrun blocks in waitpid() → SLURM sends SIGKILL to torchrun →
  torchrun dies → rank 0 becomes orphan D-state process →
  UnkillableStepTimeout (60 s) → "Kill task failed" → node drains.

ENVIRONMENT VARIABLES
  TENSOR_NUMEL        Elements in the all_reduce tensor (default: 256*1024*1024)
  AUTO_SCANCEL_DELAY  Seconds after READY before auto-cancel (default: 2.0)
  DRAIN_TEST_LOG_DIR  Log directory (default: /tmp)
"""

import os
import subprocess
import threading
import time

import torch
import torch.distributed as dist

TENSOR_NUMEL = int(os.environ.get("TENSOR_NUMEL", str(256 * 1024 * 1024)))
AUTO_SCANCEL_DELAY = float(os.environ.get("AUTO_SCANCEL_DELAY", "2.0"))
LOG_DIR = os.environ.get("DRAIN_TEST_LOG_DIR", "/tmp")
JOB_ID = os.environ.get("SLURM_JOB_ID", "local")


def main() -> None:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)

    if rank == 0:
        print(f"[MWE-05] *** v4: NCCL collective hang — confirmed D-state mechanism ***", flush=True)
        print(f"[MWE-05] job={JOB_ID}  ranks={dist.get_world_size()}  DESIGNED TO DRAIN", flush=True)
        print(f"[MWE-05] Mechanism: rank 0 alone in all_reduce → NCCL waits for", flush=True)
        print(f"[MWE-05]   missing peers via NVLink ioctl → D-state → drain", flush=True)

    # Synchronise all ranks at startup (ensures NCCL init is complete)
    dist.barrier()

    if rank == 0:
        t = torch.ones(TENSOR_NUMEL, dtype=torch.float32, device=f"cuda:{local_rank}")
        print(f"[MWE-05] All ranks initialised. Tensor: {TENSOR_NUMEL/1e6:.0f}M floats "
              f"({TENSOR_NUMEL*4/1e9:.1f} GB)", flush=True)
        print(f"[MWE-05] READY — rank 0 entering lone all_reduce (peers absent → D-state)", flush=True)
        print(f"[MWE-05] Auto-scancel in {AUTO_SCANCEL_DELAY}s", flush=True)
        print(f"[MWE-05] Recovery after drain:", flush=True)
        print(f"[MWE-05]   nvidia-smi -i 0,1,2,3 --gpu-reset", flush=True)
        print(f"[MWE-05]   scontrol update nodename=98dci4-gpu-0003 state=resume reason=\"\"", flush=True)

        def _scancel():
            time.sleep(AUTO_SCANCEL_DELAY)
            print(f"[MWE-05] AUTO-SCANCEL: issuing scancel {JOB_ID}", flush=True)
            subprocess.run(["scancel", JOB_ID], timeout=10)

        threading.Thread(target=_scancel, daemon=True).start()

        # Rank 0 calls all_reduce; ranks 1-3 do NOT.
        # Rank 0 blocks indefinitely in NCCL NVLink P2P ioctl → D-state.
        # SIGTERM cannot be delivered. drain follows.
        dist.all_reduce(t)

        print(f"[MWE-05] UNEXPECTED: all_reduce returned (peers joined?) — no drain", flush=True)

    else:
        # Ranks 1-3: sleep indefinitely. They are NOT in the collective.
        # This ensures rank 0 waits forever and is in D-state when SIGTERM arrives.
        time.sleep(3600)


if __name__ == "__main__":
    main()
