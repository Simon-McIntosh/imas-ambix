# ============================================================
# Single-rank control: Single-rank (nproc=1), CUDA context held, no NCCL
# ============================================================
# Base:  track_a_mwe01_v1_frozen.py (verbatim b3a71b6)
# Change: No DDP / NCCL init. Single process, holds a CUDA context and a
#         large GPU allocation, then hangs in an infinite Python loop.
#         Run with: torchrun --nproc_per_node=1 bisect_04_single_rank_only.py
#
# What this isolates (negative control):
#   The RCA (memory: "single cause = NVIDIA-driver D-state surviving SIGKILL")
#   establishes that a drain requires a D-state process — a process wedged in
#   an uninterruptible kernel sleep, typically caused by NCCL or GPFS.
#
#   A Python infinite-loop process (CPU-only) is trivially SIGKILL-able and
#   never enters D-state. Adding a CUDA context (but no blocking NCCL call)
#   asks: does merely *holding* a CUDA allocation make a process D-state-capable,
#   or does the drain require an active blocking NCCL call?
#
#   Expected: NO DRAIN. A process looping in Python with a live GPU allocation
#   but no blocking NCCL collective is SIGKILL-able. SLURM terminates it cleanly.
#   This is the "does any unkillable process drain?" negative control.
#
#   If bisect_04 DOES drain: re-examine the RCA — GPU allocation alone may
#   be sufficient to prevent SIGKILL under certain driver conditions.
#
# ⚠️  sbatch must use --nproc_per_node=1 (not 4) to match single-rank intent.
#     The accompanying bisect_04.sbatch is configured accordingly.
# ============================================================

import os
import time

import torch

BUFFER_MB = int(os.environ.get("BUFFER_MB", "256"))
HANG_SECONDS = int(
    os.environ.get("HANG_SECONDS", "600")
)  # 10 min — long enough for SLURM to scancel


def main() -> None:
    print(
        "[bisect_04] single_rank_only — no NCCL, CUDA context held, hang in Python loop",
        flush=True,
    )

    # Initialize a CUDA context (fair comparison: real job holds GPU allocation)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Allocate a large GPU buffer (same size as v1 default) — holds GPU memory
    n_floats = (BUFFER_MB * 1024 * 1024) // 4
    buf = torch.randn(n_floats, device=device)
    torch.cuda.synchronize(device)

    print(
        f"[bisect_04] GPU context initialized, {BUFFER_MB} MB allocated on {device}",
        flush=True,
    )
    print(
        f"[bisect_04] Entering {HANG_SECONDS}s hang loop (expects SIGKILL from SLURM time-limit)",
        flush=True,
    )
    print(
        "[bisect_04] Expected: SLURM kills cleanly — no drain (this is a negative control)",
        flush=True,
    )

    # Hang in Python CPU loop — no blocking NCCL call, no D-state trigger
    t_start = time.monotonic()
    i = 0
    while time.monotonic() - t_start < HANG_SECONDS:
        # Periodically touch the buffer to confirm GPU allocation remains active
        if i % 10000 == 0:
            _ = buf[0].item()  # host-device transfer — fast, not a blocking collective
        i += 1
        time.sleep(0.0001)  # yield CPU — don't spin-burn

    # If we reach here: the loop expired naturally (shouldn't happen before --time fires)
    print(
        f"[bisect_04] Hang loop expired after {HANG_SECONDS}s — exiting cleanly (unexpected in drain test)",
        flush=True,
    )

    # Release GPU explicitly before exit
    del buf
    torch.cuda.empty_cache()
    print("[bisect_04] GPU allocation released. Exiting.", flush=True)


if __name__ == "__main__":
    main()
