# SLURM wrappers — ambix exclusive-pause jobs

## Shared `#SBATCH` flag set

Every script in this directory uses the same base flags:

| Flag | Value | Why |
|------|-------|-----|
| `--partition` | `betelgeuse` | GPU node 98dci4-gpu-0003 |
| `--reservation` | `gpu_0003_grpA` | Group A 4×H200 allocation |
| `--account` | `grpa` | Required for betelgeuse access |
| `--gres` | `gpu:4` | All four Group A GPUs |
| `--cpus-per-task` | `30` | Group A NUMA share |
| `--mem` | `640G` | Group A memory limit |
| `--output` | `/tmp/ambix-%x-%j.out` | SLURM log — scratch, not GPFS |

**`--qos` is intentionally absent.** The QoS `gpu_0003_grpa` is
auto-applied by SLURM for this account/reservation combination. Passing
it explicitly causes: `error: QOS not permitted for this job submission`.

## Env vars each script reads

Override at submission time with `sbatch --export=ALL,VAR=val …`.

| Script | Var | Default |
|--------|-----|---------|
| `train_exclusive.sbatch` | `CONFIG_PATH` | `imas_ambix/train/configs/v0-125m.yaml` |
| `train_exclusive.sbatch` | `OUTPUT_DIR` | `/work/projects/imas_gpu/mast-checkpoints` |
| `bulk_encode_frames.sbatch` | `CAMERA` | `rbb` |
| `bulk_encode_frames.sbatch` | `QUALITY_INDEX` | `/tmp/audit-full.json` |
| `bulk_encode_frames.sbatch` | `OUTPUT_DIR` | `/work/projects/imas_gpu/mast/tokens` |
| `bulk_encode_signals.sbatch` | `QUALITY_INDEX` | `/tmp/audit-full.json` |
| `bulk_encode_signals.sbatch` | `SIGNAL_GROUPS` | `magnetics summary` |
| `bulk_encode_signals.sbatch` | `OUTPUT_DIR` | `/work/projects/imas_gpu/mast/tokens` |
| `bench_rbb.sbatch` | `BENCH_CONFIG` | `imas_ambix/bench/configs/v0-rbb-25shot.yaml` |
| `bench_rbb.sbatch` | `RESULTS_DIR` | `imas_ambix/bench/results` |

## Dry-run (no GPU allocation)

```bash
sbatch --test-only scripts/slurm/train_exclusive.sbatch
sbatch --test-only scripts/slurm/bulk_encode_frames.sbatch
sbatch --test-only scripts/slurm/bulk_encode_signals.sbatch
sbatch --test-only scripts/slurm/bench_rbb.sbatch
```

## Preempt V4-Flash before submitting

`train_exclusive.sbatch` calls `scripts/slurm/preempt_v4flash.sh` automatically
in its prologue. For encode and bench jobs you can run it manually first:

```bash
bash scripts/slurm/preempt_v4flash.sh
sbatch --export=ALL,CAMERA=rbb,QUALITY_INDEX=/tmp/audit-full.json \
    scripts/slurm/bulk_encode_frames.sbatch
```
