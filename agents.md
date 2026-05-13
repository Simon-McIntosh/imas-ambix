## 1. GPU Server Hardware

Node: 98dci4-gpu-0003 on the `betelgeuse` SLURM partition.

| Resource | Spec |
|----------|------|
| GPUs | 8× NVIDIA H200 NVL (143,771 MiB / 140.4 GB each) |
| NVLink | 18 links/GPU @ 26.562 GB/s (478 GB/s bidirectional per GPU) |
| CPU | 2× Intel Xeon 6530P (32 cores/socket, 64 total, HT off) |
| RAM | 1.5 TB (1,536 GB) |
| Local scratch | 5.9 TB NVMe at `/scratch_local` (ephemeral — cleared on reboot) |
| Shared storage | 1.5 PB GPFS at `/work/projects` (582 TB free) |
| Driver | NVIDIA 595.58.03, Compute Capability 9.0 |
| CUDA | 13.2.1 system-wide at `/usr/local/cuda/` |
| OS | RHEL 10.1, Linux 6.12, Python 3.12.12 |

## 2. Access via SLURM

The node is split between groups via reservations + QoS:

**Group A (gpu_0003_grpA):**
- 4 GPUs, 30 cores (NUMA: cores 15-29, 47-61), 650 GB RAM
- Account: `grpa`
- Reservation: `gpu_0003_grpA`
- QoS: `gpu_0003_grpa` (auto-applied — do NOT pass `--qos` explicitly, it errors)

**SLURM submission pattern:**
```bash
sbatch --partition=betelgeuse \
       --reservation=gpu_0003_grpA \
       --account=grpa \
       --gres=gpu:4 \
       --cpus-per-task=30 \
       --mem=640G \
       your_script.sh
```

**Network access:**
- **GPU nodes** (betelgeuse): NO outbound network. Model downloads and package installs must happen elsewhere.
- **Standard compute nodes** (sirius, rigel, etc.): Full outbound network AND access to `/work/projects/imas_gpu/`.
- **Login nodes**: Full network but NO access to `/work/projects/imas_gpu/` (requires `sdcc-imas_gpu` group).
- **Strategy**: Download models and install packages from standard compute nodes into shared GPFS, then serve from GPU nodes.

**SLURM workarounds:**
- `export TMPDIR=/scratch_local/$SLURM_JOB_ID && mkdir -p "$TMPDIR"` — default TMPDIR is broken on this node
- Use `srun` with the same flags for interactive jobs

## 3. Storage Paths

| Path | Purpose | Type | Size |
|------|---------|------|------|
| `/work/projects/imas_gpu/` | Shared project directory for GPU workloads | GPFS (persistent) | Shared 1.5 PB |
| `/work/projects/imas_gpu/agents/<slug>/model/` | Model weights per deployment | GPFS | — |
| `/work/projects/imas_gpu/agents/<slug>/.cache/` | HF download cache per deployment | GPFS | — |
| `~/.local/share/ambix/agent-venv/` | Shared Python 3.12 venv with inference stack | NFS (home) | ~20 GB |
| `/scratch_local/` | Fast staging, ephemeral per-node | Local NVMe | 5.9 TB |

**Inference venv contents:** torch 2.11+cu130, sglang 0.5.11, kt-kernel 0.6.1, flashinfer 0.6.8

## 4. Composable Agent CLI

The `ambix agent` CLI manages LLM deployments via TOML model profiles:

```bash
ambix agent list                          # List available profiles
ambix agent info kimi-k2-6               # Show profile details + memory budget
ambix agent download kimi-k2-6           # Submit SLURM download job (sirius partition)
ambix agent serve kimi-k2-6              # Submit SLURM serve job (betelgeuse partition)
ambix agent serve kimi-k2-6 --dry-run    # Print script without submitting
ambix agent status                        # Show running ambix SLURM jobs
```

Adding a new model: create a TOML file in `imas_ambix/agent/profiles/<slug>.toml`.

## 5. Kimi-K2.6 Deployment

**Engine:** KTransformers + SGLang (CPU+GPU hybrid MoE inference)
**Why not vLLM:** vLLM with TP=4 fills all 4×H200 VRAM (~134 GB/GPU), leaving no room for GPU sharing. KTransformers keeps only hot experts on GPU (~32 GB/GPU), leaving ~90 GB/GPU free.

**Model path:** `/work/projects/imas_gpu/agents/kimi-k2-6/model`
**Download cache:** `/work/projects/imas_gpu/agents/kimi-k2-6/.cache`

MLA compression makes KV cache tiny — full 256K context uses only 34.3 GB (9% of the 377 GB GPU KV budget), leaving ~90 GB/GPU free for sharing.

**Memory budget (per GPU):**
- Model: ~32 GB (22 GB non-expert TP=4 + 10 GB hot experts)
- KV cache: ~34 GB (262K tokens — full model context)
- Safety: ~10 GB
- **Free for other work: ~90 GB**

**CPU memory budget (650 GB QoS limit):**
- Cold experts: ~467 GB (354 × 1.32 GB)
- Overhead: ~65 GB (OS, KV spill, buffers)
- **Headroom: ~118 GB** — monitor with `sacct --format=MaxRSS`

**Client access:**

Users only get SSH access to compute nodes once a SLURM job is launched. The serve job is the entry point. Clients connect via SSH tunnel from the login node:
```bash
# Find the compute node
squeue -j <jobid> -o %N

# Set up tunnel
ssh -N -L 8000:<compute-node>:8000 <login-node>

# Verify
curl http://localhost:8000/v1/models
```

**Tuning knobs:**
- `--kt-num-gpu-experts N`: Increase N for speed (more VRAM), decrease for sharing (less VRAM)
- `--max-total-tokens N`: Increase for longer context, decrease to save memory
- `--mem-fraction-static`: GPU memory fraction for static allocation (0.90 is conservative)

**Chat modes:**
- Thinking (default): temperature=1.0, top_p=0.95
- Instant: pass `extra_body={'chat_template_kwargs': {"thinking": False}}`
- Preserve thinking: `extra_body={"chat_template_kwargs": {"thinking": True, "preserve_thinking": True}}`

## 6. Confluence Reference

GPU server documentation: https://confluence.iter.org/spaces/SDCC/pages/935667046/GPU+Server+-+98dci4-gpu-0003
(Requires ITER network authentication)
