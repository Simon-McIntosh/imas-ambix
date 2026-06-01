"""
MWE-05: DDP loss.backward() + Scancel — Drain Reproducer (v3)
==============================================================
PURPOSE
  Reproduces the confirmed drain mechanism from Event 3 (2026-05-28,
  job 1208975): issuing SLURM scancel on a DDP job mid-``loss.backward()``
  puts ranks into D-state (cudaStreamSynchronize inside NCCL gradient
  all-reduce), making them immune to SIGKILL, triggering
  UnkillableStepTimeout, and draining the node.

v1 FINDING
  First run (dim=4096, depth=8, batch=32) ran at ~130 steps/sec → 7ms/step.
  Scancel arrived between steps, not mid-backward → clean cancel, no drain.

v2 FINDING
  Model scaled to dim=8192, depth=16, batch=256, bfloat16. backward()
  returned in 2 ms — PyTorch DDP dispatches to the CUDA stream asynchronously.
  CPU returns immediately; actual GPU work runs in background. Scancel
  arrived between async dispatches (CPU in Python R/S state) → clean cancel,
  no drain. The D-state window was never exposed to SIGTERM.

v3 FIX
  Added torch.cuda.synchronize(device) after loss.backward(). This forces
  the CPU to block inside cudaDeviceSynchronize() until the entire GPU stream
  drains — including the NCCL gradient all-reduce over NVLink. This blocking
  call enters D-state (TASK_UNINTERRUPTIBLE) via the NVLink kernel driver
  ioctl. Step time is now ~400 ms of solid D-state per step. Auto-scancel
  fires 0.5 s after READY → mid-step-3 synchronize() → SIGTERM in D-state
  → drain.

RISK LEVEL
  ⚠️⚠️  DESIGNED TO DRAIN THE NODE.
  Recovery after drain:
    nvidia-smi -i 0,1,2,3 --gpu-reset
    scontrol update nodename=98dci4-gpu-0003 state=resume reason=""

MECHANISM
  loss.backward() with DDP triggers NCCL gradient all-reduce hooks. These
  hooks call cudaStreamSynchronize() on the NCCL P2P stream, which enters
  D-state (TASK_UNINTERRUPTIBLE) in the NVLink kernel driver. SIGTERM and
  SIGKILL cannot reach D-state processes. SLURM UnkillableStepTimeout (60s)
  fires → "Kill task failed" → node drain.

ENVIRONMENT VARIABLES
  MODEL_DEPTH        Hidden layers (default: 16)
  HIDDEN_DIM         Feature size (default: 8192)
  BATCH_SIZE         Per-rank batch size (default: 256)
  AUTO_SCANCEL_DELAY Seconds after READY before auto-cancel (default: 0.5)
  DRAIN_TEST_LOG_DIR Log directory (default: /tmp)
"""

import os
import subprocess
import sys
import threading
import time

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

MODEL_DEPTH = int(os.environ.get("MODEL_DEPTH", "16"))
HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", "8192"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "256"))
AUTO_SCANCEL_DELAY = float(os.environ.get("AUTO_SCANCEL_DELAY", "0.5"))
LOG_DIR = os.environ.get("DRAIN_TEST_LOG_DIR", "/tmp")
JOB_ID = os.environ.get("SLURM_JOB_ID", "local")


def build_model(depth: int, dim: int) -> nn.Module:
    """Deep MLP sized to produce ~500ms backward on H200 (target D-state window)."""
    layers: list[nn.Module] = [nn.Linear(dim, dim)]
    for _ in range(depth - 1):
        layers.extend([nn.ReLU(), nn.Linear(dim, dim)])
    return nn.Sequential(*layers)


def setup() -> tuple[int, int]:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    return rank, local_rank


def main() -> None:
    rank, local_rank = setup()
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()
    dtype = torch.bfloat16

    if rank == 0:
        print(f"[MWE-05] ranks={world_size}, depth={MODEL_DEPTH}, dim={HIDDEN_DIM}, "
              f"batch={BATCH_SIZE}, dtype=bfloat16", flush=True)
        print(f"[MWE-05] job={JOB_ID}  *** DESIGNED TO DRAIN THE NODE ***", flush=True)
        print(f"[MWE-05] Auto-scancel fires {AUTO_SCANCEL_DELAY}s after READY", flush=True)
        print(f"[MWE-05] Mechanism: loss.backward() → NCCL all-reduce → "
              f"cudaStreamSynchronize → D-state → UnkillableStepTimeout → drain", flush=True)

    model = build_model(MODEL_DEPTH, HIDDEN_DIM).to(device=device, dtype=dtype)
    ddp_model = DDP(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    n_params = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(f"[MWE-05] Parameters: {n_params:,} ({n_params*2/1e9:.2f} GB bf16 weights)", flush=True)

    # Warm-up: trigger NCCL init before READY so the training loop D-state
    # window is purely the synchronous cudaStreamSynchronize, not NCCL init.
    x = torch.randn(BATCH_SIZE, HIDDEN_DIM, device=device, dtype=dtype)
    y = torch.randn(BATCH_SIZE, HIDDEN_DIM, device=device, dtype=dtype)
    warmup_loss = criterion(ddp_model(x), y)
    warmup_loss.backward()
    torch.cuda.synchronize(device)
    optimizer.zero_grad()
    if rank == 0:
        print(f"[MWE-05] Warm-up complete (NCCL initialised). Starting timed loop...", flush=True)

    step = 0
    t_start = time.monotonic()

    # Training loop: no stop-file, no SIGTERM handler, no watchdog.
    # The auto-scancel daemon (rank 0 only) will issue scancel mid-backward.
    while True:
        x = torch.randn(BATCH_SIZE, HIDDEN_DIM, device=device, dtype=dtype)
        y = torch.randn(BATCH_SIZE, HIDDEN_DIM, device=device, dtype=dtype)

        optimizer.zero_grad()
        out = ddp_model(x)
        loss = criterion(out, y)

        t_bwd = time.monotonic()
        # DDP hooks fire NCCL all-reduce; dispatched asynchronously to CUDA stream.
        loss.backward()

        # v3: Force CPU to block until GPU stream (including NCCL all-reduce)
        # completes. cudaDeviceSynchronize() enters D-state via NVLink kernel
        # driver ioctl. This creates a ~400ms D-state window per step.
        torch.cuda.synchronize(device)
        bwd_ms = (time.monotonic() - t_bwd) * 1000

        optimizer.step()
        step += 1

        if rank == 0:
            elapsed = time.monotonic() - t_start
            if step == 1:
                print(f"[MWE-05] Step 1: backward={bwd_ms:.0f} ms, elapsed={elapsed:.2f}s", flush=True)
                print(f"[MWE-05] READY — auto-scancel fires in {AUTO_SCANCEL_DELAY}s", flush=True)
                print(f"[MWE-05] Recovery after drain:", flush=True)
                print(f"[MWE-05]   nvidia-smi -i 0,1,2,3 --gpu-reset", flush=True)
                print(f"[MWE-05]   scontrol update nodename=98dci4-gpu-0003 state=resume reason=\"\"", flush=True)

                # Daemon thread issues scancel after AUTO_SCANCEL_DELAY seconds,
                # which will be mid-backward of step 2.
                def _scancel():
                    time.sleep(AUTO_SCANCEL_DELAY)
                    print(f"[MWE-05] AUTO-SCANCEL: issuing scancel {JOB_ID} at "
                          f"{time.monotonic()-t_start:.2f}s", flush=True)
                    subprocess.run(["scancel", JOB_ID], timeout=10)

                threading.Thread(target=_scancel, daemon=True).start()

            elif step % 5 == 0:
                print(f"[MWE-05] step={step}, bwd={bwd_ms:.0f}ms, "
                      f"elapsed={elapsed:.1f}s, loss={loss.item():.4f}", flush=True)


if __name__ == "__main__":
    main()
