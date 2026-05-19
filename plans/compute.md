# Plan: Compute — SLURM Patterns for World-Model Training

Status: **Draft** — reservation request not yet filed.

This plan covers the SLURM patterns for training the Fusion World Model.
The agent-serving deployment under `imas-ambix/imas_ambix/agent/` already
documents the **serving** patterns in `docs/cluster-usage.md`; the
training patterns are distinct enough to warrant their own document.

The headline question this plan answers: **how do we get 4 × H200
exclusively for training while the LLM-serving job is still useful?**
The short answer: request a separate reservation and run training in
scheduled-around-serving windows in the meantime.

---

## 1. Hardware budget recap

The Group A reservation on `98dci4-gpu-0003` (`betelgeuse` SLURM
partition) currently grants:

| Resource | Allocated to Group A |
|---|---|
| GPUs | 4 × NVIDIA H200 NVL (140.4 GB HBM3e each, 561 GB total) |
| CPU cores | 30 (NUMA: 15-29, 47-61) |
| RAM | 650 GB |
| QoS | `gpu_0003_grpa` |
| Reservation name | `gpu_0003_grpA` |
| Account | `grpa` |
| NVLink | 18 links / GPU @ 26.562 GB/s (478 GB/s bidirectional per GPU) |
| Local scratch | 5.9 TB NVMe at `/scratch_local` (ephemeral, cleared on reboot) |
| Shared storage | 1.5 PB GPFS at `/work/projects` (576 TB free at time of writing) |

The other half of the node is allocated to Group B; we cannot reach it
from Group A's reservation.

---

## 2. Why training cannot co-locate with DeepSeek V4-Flash serving

DeepSeek V4-Flash, the current production serve, uses ~41 GB / GPU
(weights 164 GB / 4 = 41 GB) and reserves additional headroom for KV
cache. That leaves ~90 GB free per GPU. From the world-model FSDP budget
in `world-model-v0.md` §4.2:

| Model | Per-GPU peak (bf16, FSDP, activation ckpt, 16 K ctx, micro-batch 4) |
|---|---|
| 125 M | ~30 GB |
| **500 M** | **~85 GB** |
| 1 B | ~140 GB (exceeds total, requires CPU offload or 8 GPU) |

500 M training **just barely fits** alongside V4-Flash on paper, with
zero margin for unexpected OOMs (e.g. PyTorch CUDA-graph artefacts,
optimizer state spikes). In practice this is too tight to operate
reliably. We therefore plan for a **dedicated training reservation**.

The 125 M curriculum step **does** fit alongside V4-Flash (30 GB +
41 GB = 71 GB / GPU << 141 GB), so the training loop can be brought up
and debugged in shared mode. The 500 M step requires the dedicated
slice.

---

## 3. Dedicated training-reservation request

### 3.1 Reservation name (proposed)

`gpu_0003_grpA_train` — same node, same group, separate reservation slot
that can be drained without affecting the production serve.

### 3.2 Request body

To file with the SDCC operations team. The body below quotes the
existing `gpu_0003_grpA` pattern from `docs/cluster-usage.md` §2 so the
ops team can copy-paste the boilerplate.

```text
Subject: New SLURM reservation request — gpu_0003_grpA_train

Node:        98dci4-gpu-0003 (existing Group A allocation)
Reservation: gpu_0003_grpA_train (new)
Group:       gpu_0003_grpA (same as existing)
QoS:         gpu_0003_grpa_train (new, mirrors existing)
Account:     grpa (same as existing)
Resources:   4 × H200, 30 cores, 650 GB RAM (same as existing)
Time policy: per-job up to 72 h, no concurrency cap

Justification:
The Fusion World Model training campaign (imas-ambix plan
plans/world-model-v0.md, ~500 M parameters, FSDP across 4 × H200, peak
~85 GB / GPU) does not co-locate with the running DeepSeek V4-Flash
serve which already holds ~41 GB / GPU plus KV-cache headroom. We need
the ability to drain the serve briefly, run training, and bring the
serve back without affecting the existing Group A users.

Operating model:
- The serve and the train reservations share the same 4 GPUs.
- Only one of them is active at any time. Training jobs cannot be
  submitted while the serve job is running, and vice versa.
- Scheduling is co-ordinated by the imas-ambix maintainer; submitted
  via a small CLI command (ambix train start / stop) that already
  manages SLURM submissions for the agent CLI.
- The change is administrative: we are not asking for additional
  hardware, only for the ability to submit training jobs to a separate
  reservation slot that can be held while the serve is paused.

Owner: <maintainer name + ITER account>
Stakeholder: Science Division (S. Pinches)
```

### 3.3 Interim — scheduled-around-serving

Until the dedicated reservation lands, training runs in windows when the
serve is paused. The protocol:

```bash
# 1. Pause the serve (CLI exists; this just `scancel`s the serve job
#    after recording its config so we can restart it later)
ambix agent stop deepseek-v4-flash

# 2. Submit the train job — same reservation, same QoS as the serve
ambix train submit --config v0-500m --duration 24h

# 3. Wait for the train job to finish or fail
squeue -u $USER --start

# 4. Restart the serve
ambix agent serve deepseek-v4-flash
```

Reasonable windows: weekends, overnight (00:00 – 06:00 local), and any
explicitly-coordinated maintenance window. The maintainer announces in
the SDCC GPU / Ambix Teams channel before stopping the serve.

---

## 4. SLURM submission patterns

### 4.1 Training submission (writes the script the CLI submits)

```bash
#!/bin/bash
#SBATCH --job-name=ambix-train-v0-500m
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA_train   # once granted; until then, gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=30
#SBATCH --mem=640G
#SBATCH --time=72:00:00
#SBATCH --output=ambix-train-%j.log

set -euo pipefail

# Per AGENTS.md SDCC notes: TMPDIR fixup
export TMPDIR=/scratch_local/$SLURM_JOB_ID && mkdir -p "$TMPDIR"

# Per AGENTS.md SDCC notes: BLAS threads
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Source the project env (creates by ambix install)
source /work/projects/imas_gpu/ambix-train-venv/bin/activate

# Launch FSDP via accelerate — 4 GPUs, ZeRO-3
accelerate launch \
    --num_processes 4 \
    --num_machines 1 \
    --use_fsdp \
    --fsdp_sharding_strategy FULL_SHARD \
    --fsdp_state_dict_type SHARDED_STATE_DICT \
    --fsdp_offload_params false \
    --mixed_precision bf16 \
    -m imas_ambix.train.loop \
    --config-name v0-500m
```

Note that we deliberately **do not** pass `--qos=...` explicitly — per
`docs/cluster-usage.md`, the QoS auto-applies from the reservation; an
explicit `--qos` causes a SLURM error in this environment.

### 4.2 Smoke-test submission (debug)

```bash
#!/bin/bash
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:1                       # single-GPU smoke
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00

# Smoke target: one training step, finite loss, no NaN
python -m imas_ambix.train.loop --config-name v0-125m --debug
```

Smoke tests run during the serving window because they fit (125 M model
@ 30 GB << remaining 90 GB).

---

## 5. FSDP sharding strategy

- Strategy: **FULL_SHARD** (ZeRO-3). Shards weights, gradients, optimizer
  state across the 4 ranks. Maximum memory savings, minimal coding
  overhead via `accelerate`'s FSDP integration.
- **Sharded state-dict checkpoints**: each rank writes its own shard;
  reassembled at load. This avoids OOMing the CPU during checkpoint
  consolidation.
- **No CPU offload** for v0 — model fits in HBM with activation
  checkpointing.
- **Activation checkpointing** every other transformer block (Llama
  default in `accelerate`).
- **FlashAttention-2** via the HF Llama implementation; default with the
  declared `torch ≥ 2.6` dep.

For v1 (multi-billion params), the choice between FSDP and Megatron-LM
tensor parallelism becomes interesting. Out of scope here.

---

## 6. Checkpoint and artefact paths

```
/work/projects/imas_gpu/mast-checkpoints/
└── {run-id}/                              # run-id = config-hash + manifest-hash + timestamp
    ├── manifest.json                      # data manifest + tokenizer vocab version
    ├── config.yaml                        # the resolved Hydra config
    ├── step-2000/
    │   ├── pytorch_model_fsdp.bin         # sharded safetensors
    │   ├── optimizer.pt
    │   └── trainer-state.json
    ├── step-4000/
    │   └── ...
    └── final/                             # symlink to the best checkpoint
```

Checkpoint cadence: every 2 K steps. Retention policy: keep all
checkpoints for a run while the run is "active" (training); after a run
finishes or is abandoned, keep only `step-0`, `step-{halfway}`, and
`final/`. The cleanup is a manual operator job (no auto-prune script in
v0) so we cannot accidentally lose state.

---

## 7. Future scaling

The 4 × H200 reservation is a single-node budget. v1+ growth options:

| Direction | Trigger | Cost |
|---|---|---|
| **Larger model on the same node** (1 B / 2 B) | v0 demo successful, more partner data online | Needs CPU offload or 8-GPU reservation; check if Group A can be expanded to all 8 H200 on the node (this is the same node, just the other half) |
| **Multi-node** | 2 B+ models or larger context windows | Requires multi-node interconnect QoS that we do not currently have — file with SDCC; cost real |
| **FP8 training** | If H200 FP8 path becomes stable for FSDP in HF `transformers` | Free other than engineering time |

Note: the GPU-procurement doc anticipated 8-GPU access for the
serving-class agents (see §3.2 of `gpu-cluster-scoping.md`). Expanding
the training reservation onto the same 8 GPUs is the natural Phase 4
expansion path.

---

## 8. Risks (training-compute-specific)

| Risk | Mitigation |
|---|---|
| Dedicated reservation request denied / delayed | Stay on scheduled-around-serving until granted; v0 demo can still complete in interim windows. |
| FSDP + sharded state-dict not stable on HF `transformers` 4.48 + accelerate 1.0 | Single-shard fallback documented (write full state on rank 0); 500 M model fits in 650 GB CPU RAM. |
| H200 NVL NCCL flakiness (seen in MiniMax serve, see `AGENTS.md`) | Run `accelerate launch` with `NCCL_DEBUG=INFO`; fall back to `gloo` for CPU-only debug if NCCL throws. |
| Disk full at `/work/projects/imas_gpu/mast-checkpoints/` | Each checkpoint is ~5 GB at 500 M params; full run = ~150 GB. 576 TB free is fine; monitor with the existing `df` cron. |

---

## 9. Related plans

- `STRATEGY.md` — roadmap that this compute plan supports.
- `world-model-v0.md` — the model whose budget drives this plan.
- `data-acquisition.md` — the data the training reads from GPFS.
- `../docs/cluster-usage.md` — the canonical serving / SLURM docs that
  this plan references.
