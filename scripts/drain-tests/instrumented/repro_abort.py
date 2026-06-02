"""
repro_abort.py — faithful reproducer of the production GPU-node drain.
========================================================================
Reproduces the SINGLE mechanism shared by both captured drains on
98dci4-gpu-0003:

  job 1209813 (2026-06-01): NCCL sequence divergence over 20 real-NVLink
      all_reduce rounds, then dist.destroy_process_group().
  job 1208980 (2026-05-28, Event 3 follow-on): DDP param-shape verify
      mismatch (rank 0 "inconsistent 0 params"), then teardown.

Both logged the identical signature:
  "[E ProcessGroupNCCL.cpp:1154] Future for ProcessGroup abort timed out
   after 600000 ms"   ->  STEPD TERMINATED ... JOB NOT ENDING WITH SIGNALS
   ->  Container ... has N processes, giving up after 63 sec  ->  DRAIN.

The hypothesis under test: dist.destroy_process_group() on a *broken*
communicator calls ncclCommAbort(), which enters an UNINTERRUPTIBLE (D-state)
wait in the NVIDIA driver. The process then survives SIGKILL, so SLURM cannot
reap the step within UnkillableStepTimeout (60 s) -> node drain.

The kill is delivered by the SLURM --time limit (faithful to 1209813, which
died "DUE TO TIME LIMIT"); no scancel is issued by this script.

A separate observer job (observe_state.sh) records every rank's /proc state
and wchan through the whole window, so we OBSERVE the D-state directly rather
than inferring it from log strings (which is all the prior analysis ever did).

MODE (env REPRO_MODE):
  seq_divergence (default) -> replicate 1209813
  ddp_mismatch             -> replicate 1208980
"""
import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist

MODE = os.environ.get("REPRO_MODE", "seq_divergence")
ROUNDS = int(os.environ.get("ROUNDS", "20"))
NUMEL = int(os.environ.get("NUMEL", str(64 * 1024 * 1024)))  # 256 MB fp32


def log(m: str) -> None:
    print(f"[repro {time.strftime('%H:%M:%S')} pid={os.getpid()}] {m}", flush=True)


def main() -> None:
    _pg_to = os.environ.get("AMBIX_PG_TIMEOUT")
    if _pg_to:
        dist.init_process_group(backend="nccl", timeout=timedelta(seconds=int(_pg_to)))
    else:
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    lr = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(lr)
    dev = torch.device(f"cuda:{lr}")

    if rank == 0:
        log(f"*** MODE={MODE} world={world} ROUNDS={ROUNDS} NUMEL={NUMEL} ***")
        log("*** DESIGNED TO WEDGE ncclCommAbort AND DRAIN THE NODE ***")
        log("*** kill arrives via SLURM --time limit; observer captures /proc state ***")

    # Hot, real compute kernels (cuBLAS) + establish the NCCL ring, so GPU
    # contexts are warm exactly like a real trained job (1208975 ran to step 400).
    a = torch.randn(4096, 4096, device=dev)
    for _ in range(30):
        a = (a @ a).relu() * 1e-3
    torch.cuda.synchronize()
    buf = torch.randn(NUMEL, dtype=torch.float32, device=dev)
    dist.all_reduce(buf)
    torch.cuda.synchronize()
    dist.barrier()
    log(f"rank {rank} warmup complete (hot CUDA context + NCCL ring established)")

    if MODE == "abort_watchdog":
        # PRODUCTION-FAITHFUL (matches 1208980/1209813 timeline): DEFAULT torch
        # watchdog ON + short PG timeout (AMBIX_PG_TIMEOUT). Ranks 1..N-1 leave a
        # collective genuinely stuck in flight; rank 0 stays alive and never joins.
        # At the PG timeout the watchdog fires and drives ncclCommAbort; the SLURM
        # --time SIGKILL is timed to land WHILE that abort hangs. If the abort
        # wedges in the driver (D-state) it survives SIGKILL -> drain.
        if rank == 0:
            log("rank 0 ALIVE, NOT joining (peer kernels stay stuck; watchdog will fire on peers)")
            time.sleep(3600)
        else:
            log(f"rank {rank} posting all_reduce + synchronize -> stuck in flight; watchdog fires at PG timeout")
            dist.all_reduce(buf)
            torch.cuda.synchronize()
            log(f"rank {rank} synchronize returned (watchdog/abort did not wedge)")
        time.sleep(3600)
        return

    if MODE == "abort_stuck":
        # NCCL #829 FAITHFUL repro of the production signature
        # ("Future for ProcessGroup abort timed out after 600000 ms"):
        # ranks 1..N-1 launch an ASYNC all_reduce (GPU kernel goes in flight and
        # stalls forever waiting for rank 0, who stays alive but never joins so the
        # ring is not torn down), then immediately call destroy_process_group().
        # That invokes ncclCommAbort, which internally calls cudaStreamSynchronize
        # on the stream holding the stuck kernel -> uninterruptible (D) wait that
        # never returns. With the torch/NCCL watchdog disabled (see sbatch) nothing
        # SIGABRTs the process, so the SLURM --time SIGKILL lands inside the
        # abort-hang window -> unkillable -> drain.
        if rank == 0:
            log("rank 0 ALIVE, NOT joining collective (keeps ring up so peer kernels stay stuck)")
            time.sleep(3600)
        else:
            log(f"rank {rank} launching ASYNC all_reduce (NO synchronize) -> kernel stuck in flight")
            dist.all_reduce(buf)  # async: returns immediately, GPU kernel stalls
            log(f"rank {rank} >>> destroy_process_group() = ncclCommAbort on in-flight stuck kernel (EXPECT D-STATE HANG) <<<")
            t0 = time.time()
            dist.destroy_process_group()
            log(f"rank {rank} destroy RETURNED after {time.time() - t0:.1f}s (did NOT wedge)")
        time.sleep(3600)
        return

    if MODE == "stuck_collective":
        # NCCL #829 path: a genuinely STUCK NCCL GPU kernel.
        # Ranks 1..N-1 post a real all_reduce (GPU kernel launched, needs rank 0's
        # contribution) then cudaStreamSynchronize. Rank 0 stays ALIVE (so NCCL
        # sees no peer disconnect and does not auto-abort) but NEVER joins the
        # collective. The synchronize on ranks 1..N-1 blocks in the NVIDIA ioctl
        # that can never return -> uninterruptible (D) state -> survives SIGKILL.
        # The NCCL/torch watchdog is disabled via env (see sbatch) so nothing
        # aborts the stuck kernel before the SLURM --time kill arrives.
        if rank == 0:
            log("rank 0 ALIVE but NOT joining the collective (keeps ring connected)")
            time.sleep(3600)
        else:
            log(f"rank {rank} >>> posting all_reduce that will STALL (rank 0 absent) + synchronize <<<")
            dist.all_reduce(buf)
            torch.cuda.synchronize()  # blocks in cudaStreamSynchronize ioctl -> EXPECT D-state
            log(f"rank {rank} synchronize RETURNED (did NOT wedge)")
        time.sleep(3600)
        return

    if MODE == "ddp_mismatch":
        import torch.nn as nn
        from torch.nn.parallel import DistributedDataParallel as DDP

        # Mismatched parameter shapes across ranks -> DDP's
        # _verify_param_shape_across_processes() allgather mismatches and raises,
        # leaving the communicator broken mid-collective (as in 1208980).
        if rank == 0:
            model = nn.Linear(8, 8).to(dev)
        else:
            model = nn.Sequential(nn.Conv2d(3, 16, 3), nn.Conv2d(16, 16, 3)).to(dev)
        log(f"rank {rank} building DDP with MISMATCHED model (expect verify failure)")
        try:
            DDP(model, device_ids=[lr])
            log(f"rank {rank} UNEXPECTED: DDP built without error")
        except Exception as e:  # noqa: BLE001
            log(f"rank {rank} DDP raised: {type(e).__name__}: {str(e)[:140]}")
    else:  # seq_divergence (1209813)
        for r in range(ROUNDS):
            if rank == 0:
                dist.all_reduce(buf)  # extra op on rank 0 -> accumulating seq divergence
            dist.all_reduce(buf)
            torch.cuda.synchronize()
            if rank == 0 and r % 5 == 0:
                log(f"round {r}/{ROUNDS} done (rank-0 seq divergence accumulating)")
        log(f"rank {rank} completed {ROUNDS} divergent rounds; ring desynchronised")

    log(f"rank {rank} >>> calling dist.destroy_process_group() — EXPECT ncclCommAbort HANG <<<")
    t0 = time.time()
    dist.destroy_process_group()
    log(f"rank {rank} destroy_process_group RETURNED after {time.time() - t0:.1f}s (did NOT wedge)")

    # Stay alive so (a) the observer keeps sampling and (b) the SLURM --time
    # SIGTERM lands on a live process, reproducing the production kill timing.
    time.sleep(3600)
    log(f"rank {rank} sleep returned")


if __name__ == "__main__":
    main()
