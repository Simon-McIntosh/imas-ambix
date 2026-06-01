"""
MWE-03: Network Black-Hole D-State Hang — v3 (DDP + CUDA + urllib)
===================================================================
PURPOSE
  Reproduces the EXACT mechanism of Event #3 (2026-05-27b, drain rca
  doc rca-node-drain-2026-05-27b.html): model-weight download via urllib
  attempted during DDP initialisation on a betelgeuse GPU node (no
  outbound network, firewall DROPs packets). The download put torchrun
  workers into D-state; SLURM could not kill the orphaned D-state
  processes → UnkillableStepTimeout → node drained.

v1/v2 FINDING (single-process, no CUDA)
  Raw socket.connect() to bare IP → S-state (interruptible). No drain.
  urllib.request.urlopen() in single process → S-state (TCP/DNS is
  interruptible when there is no kernel driver involvement). No drain.

  Key difference from the original event: the original used torchrun
  DDP workers. SLURM sends SIGTERM to torchrun, which forwards to workers.
  If workers are in D-state, they cannot receive SIGTERM. torchrun then
  blocks in waitpid() and is SIGKILLed, leaving D-state workers as
  orphans that survive SIGKILL → drain.

v3 CHANGES (closing all three gaps)
  1. DDP: 4-rank torchrun (matches original multi-process topology)
  2. CUDA: each rank calls torch.cuda.set_device() and allocates a tensor
     (live CUDA context — increases chance of ioctl D-state on cancel)
  3. GPFS write target: urllib.request.urlretrieve(URL, gpfs_path)
     Writing to GPFS via urlretrieve involves GPFS client kernel code
     that can enter D-state, compounding the firewall-DROP hang.
  4. AUTO-SCANCEL: rank 0 daemon thread issues scancel after READY

RISK LEVEL
  ⚠️⚠️  LIKELY DRAIN — torchrun orphan D-state workers on betelgeuse.
  Recovery (no CUDA reset needed if CUDA context was not deep):
    nvidia-smi -i 0,1,2,3 --gpu-reset   (may be needed if ioctl D-state)
    scontrol update nodename=98dci4-gpu-0003 state=resume reason=""

DRAIN MECHANISM (two possible paths)
  Path A (GPFS): urlretrieve kernel write to GPFS → GPFS client D-state
    → torchrun SIGTERM → SIGKILL torchrun (parent) → orphaned workers
    → workers survive SIGKILL in D-state → UnkillableStepTimeout → drain
  Path B (network): urllib TCP SYN to dropped network → DNS/connect D-state
    → same torchrun kill chain as Path A

ENVIRONMENT VARIABLES
  DOWNLOAD_URL        URL to fetch (default: vgg_lpips URL from Event #3)
  GPFS_DEST           File path for urlretrieve output (must be on GPFS)
  WAIT_BEFORE         Seconds before download attempt (default: 5)
  AUTO_SCANCEL_DELAY  Seconds after READY before auto-cancel (default: 2)
  DRAIN_TEST_LOG_DIR  Log directory (default: /tmp)
"""

import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request

import torch
import torch.distributed as dist

DOWNLOAD_URL = os.environ.get(
    "DOWNLOAD_URL",
    "https://heibox.uni-heidelberg.de/f/607503859d364012b37e/?dl=1",
)
GPFS_DEST = os.environ.get(
    "GPFS_DEST",
    "/work/projects/imas_gpu/mwe03-download-test.bin",
)
WAIT_BEFORE = float(os.environ.get("WAIT_BEFORE", "5"))
AUTO_SCANCEL_DELAY = float(os.environ.get("AUTO_SCANCEL_DELAY", "2"))
LOG_DIR = os.environ.get("DRAIN_TEST_LOG_DIR", "/tmp")
JOB_ID = os.environ.get("SLURM_JOB_ID", "local")


def main() -> None:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    # CUDA context live (gap 2): mirrors original VQModel GPU init before download
    torch.cuda.set_device(local_rank)
    _dummy = torch.ones(1, device=f"cuda:{local_rank}")
    torch.cuda.synchronize()

    hostname = socket.gethostname()
    pid = os.getpid()

    if rank == 0:
        print(f"[MWE-03] *** v3: DDP + CUDA + urllib to GPFS ***", flush=True)
        print(f"[MWE-03] hostname={hostname}  pid={pid}  job={JOB_ID}", flush=True)
        print(f"[MWE-03] URL:  {DOWNLOAD_URL}", flush=True)
        print(f"[MWE-03] DEST: {GPFS_DEST}", flush=True)
        print(f"[MWE-03] All {dist.get_world_size()} DDP ranks alive. "
              f"CUDA contexts live. Barrier...", flush=True)

    dist.barrier()

    if rank == 0:
        print(f"[MWE-03] READY — all ranks past barrier, auto-scancel in {AUTO_SCANCEL_DELAY}s", flush=True)
        print(f"[MWE-03] All ranks now attempting urlretrieve to GPFS (no timeout).", flush=True)
        print(f"[MWE-03] Expected: hang → D-state → torchrun orphan → node drains", flush=True)
        print(f"[MWE-03] Recovery if drained:", flush=True)
        print(f"[MWE-03]   nvidia-smi -i 0,1,2,3 --gpu-reset", flush=True)
        print(f"[MWE-03]   scontrol update nodename=98dci4-gpu-0003 state=resume reason=\"\"", flush=True)

        def _scancel():
            time.sleep(AUTO_SCANCEL_DELAY)
            print(f"[MWE-03] AUTO-SCANCEL: issuing scancel {JOB_ID}", flush=True)
            subprocess.run(["scancel", JOB_ID], timeout=10)

        threading.Thread(target=_scancel, daemon=True).start()

    # All ranks attempt the download — no SIGTERM handler, no timeout.
    # On betelgeuse (DROP firewall): getaddrinfo() or TCP SYN blocks in D-state.
    # The GPFS write path also involves kernel code that may enter D-state.
    dest_rank = f"{GPFS_DEST}.rank{rank}"
    try:
        urllib.request.urlretrieve(DOWNLOAD_URL, dest_rank)
        print(f"[MWE-03] rank={rank} UNEXPECTED: urlretrieve succeeded — "
              f"node has outbound network access!", flush=True)
    except OSError as exc:
        # Fast failure (REJECT): process stayed killable, no D-state
        print(f"[MWE-03] rank={rank} urlretrieve raised {type(exc).__name__}: {exc}", flush=True)
    except Exception as exc:
        print(f"[MWE-03] rank={rank} unexpected exception: {exc}", flush=True)


if __name__ == "__main__":
    main()
