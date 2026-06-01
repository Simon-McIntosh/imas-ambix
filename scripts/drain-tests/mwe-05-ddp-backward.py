"""
MWE-05: DDP loss.backward() + Scancel — Drain Reproducer
==========================================================
PURPOSE
  Reproduces the confirmed drain mechanism from Event 3 (2026-05-28,
  job 1208975): issuing SLURM scancel on a DDP job mid-``loss.backward()``
  puts ranks into D-state (cudaStreamSynchronize inside NCCL gradient
  all-reduce), making them immune to SIGKILL, triggering
  UnkillableStepTimeout, and draining the node.

RISK LEVEL
  ⚠️⚠️  LIKELY DRAIN — GPU reset + admin scontrol resume expected.
  Do NOT run without admin contact ready and explicit user authorisation.

  Required recovery after drain:
    nvidia-smi -i 0,1,2,3 --gpu-reset
    scontrol update nodename=98dci4-gpu-0003 state=resume reason=""

MECHANISM
  loss.backward() with DDP triggers NCCL gradient all-reduce hooks. These
  hooks call cudaStreamSynchronize() on the NCCL P2P stream, which enters
  D-state (TASK_UNINTERRUPTIBLE) in the kernel NVLink driver. SIGTERM
  cannot be queued; SIGKILL cannot kill the D-state process. SLURM's
  UnkillableStepTimeout (~60 s) fires → "Kill task failed" → node drain.

  This is distinct from a simple all_reduce (MWE-04), which uses CPU
  spin-wait (R-state) and can be killed cleanly.

WHAT IT DOES
  1. Initialises 4-rank DDP (NCCL backend) across 4 GPUs.
  2. Creates a real model (ResNet-50 equivalent depth) so that backward()
     creates substantial gradient tensors needing all-reduce.
  3. Runs a training loop: forward → loss → backward → optimizer.step().
  4. Prints "READY" after the first successful backward pass.
  5. Runs indefinitely (no watchdog, no stop-file, no SIGUSR1 handler)
     so that scancel mid-backward can create D-state.
  6. The accompanying sbatch has NO --signal or three-layer defence
     (intentionally omitted to expose the drain mechanism).

HOW TO TEST
  1. Submit:
       sbatch mwe-05-ddp-backward.sbatch
       # note the job ID

  2. Wait for "READY" in the log (first backward pass complete, ~30 s):
       tail -f <log-file>

  3. Immediately issue scancel:
       scancel <jobid>

  4. Wait 70 s and check node state:
       sinfo -N -n 98dci4-gpu-0003 --noheader -o "%T"
       # "drain" → mechanism confirmed (expected result)
       # "idle"  → D-state did not form (try again or check NCCL version)

  5. If drained, notify admin for recovery:
       nvidia-smi -i 0,1,2,3 --gpu-reset
       scontrol update nodename=98dci4-gpu-0003 state=resume reason=""

ENVIRONMENT VARIABLES
  MODEL_DEPTH        Number of hidden layers (default: 8; more = slower backward)
  HIDDEN_DIM         Feature size (default: 4096)
  BATCH_SIZE         Per-rank batch size (default: 32)
  DRAIN_TEST_LOG_DIR Log directory (default: /tmp)
"""

import os
import sys
import time

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

MODEL_DEPTH = int(os.environ.get("MODEL_DEPTH", "8"))
HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", "4096"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))
LOG_DIR = os.environ.get("DRAIN_TEST_LOG_DIR", "/tmp")
JOB_ID = os.environ.get("SLURM_JOB_ID", "local")


def build_model(depth: int, dim: int) -> nn.Module:
    """Simple deep MLP with enough parameters to make backward() slow."""
    layers: list[nn.Module] = []
    layers.append(nn.Linear(dim, dim))
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

    if rank == 0:
        print(f"[MWE-05] ranks={world_size}, depth={MODEL_DEPTH}, dim={HIDDEN_DIM}, batch={BATCH_SIZE}", flush=True)
        print(f"[MWE-05] job={JOB_ID}  WARNING: this test is designed to DRAIN THE NODE", flush=True)
        print(f"[MWE-05] Mechanism: DDP loss.backward() → NCCL grad all-reduce → cudaStreamSynchronize → D-state", flush=True)

    # Build and wrap model
    model = build_model(MODEL_DEPTH, HIDDEN_DIM).to(device)
    ddp_model = DDP(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    n_params = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(f"[MWE-05] Model parameters: {n_params:,} ({n_params*4/1e6:.1f} MB)", flush=True)

    # Warm-up forward pass
    x = torch.randn(BATCH_SIZE, HIDDEN_DIM, device=device)
    y = torch.randn(BATCH_SIZE, HIDDEN_DIM, device=device)
    with torch.no_grad():
        _ = ddp_model(x)
    torch.cuda.synchronize(device)

    if rank == 0:
        print(f"[MWE-05] Warm-up done. Starting training loop...", flush=True)

    step = 0
    t_start = time.monotonic()

    # Training loop — no stop-file, no SIGTERM handler, no watchdog.
    # This is intentional: we want scancel to find the process mid-backward.
    while True:
        x = torch.randn(BATCH_SIZE, HIDDEN_DIM, device=device)
        y = torch.randn(BATCH_SIZE, HIDDEN_DIM, device=device)

        optimizer.zero_grad()
        out = ddp_model(x)
        loss = criterion(out, y)

        # This is where D-state occurs under scancel.
        # DDP backward hooks fire NCCL gradient all-reduce at this point.
        # cudaStreamSynchronize() inside the NCCL kernel enters D-state.
        loss.backward()

        optimizer.step()
        step += 1

        if rank == 0:
            if step == 1:
                elapsed = time.monotonic() - t_start
                print(f"[MWE-05] First backward complete in {elapsed:.2f} s", flush=True)
                print(f"[MWE-05] READY — issue 'scancel {JOB_ID}' NOW for drain test", flush=True)
                print(f"[MWE-05] Expected: node drains within 70 s of scancel", flush=True)
                print(f"[MWE-05] Recovery: nvidia-smi -i 0,1,2,3 --gpu-reset && scontrol update nodename=98dci4-gpu-0003 state=resume reason=\"\"", flush=True)
            elif step % 10 == 0:
                elapsed = time.monotonic() - t_start
                print(f"[MWE-05] step={step}, elapsed={elapsed:.1f}s, loss={loss.item():.4f}", flush=True)


if __name__ == "__main__":
    main()
