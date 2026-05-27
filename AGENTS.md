# Agent Guidelines — IMAS Ambix

Repo-specific guardrails for this workstation.  The user-global rules in
[`~/.agents/AGENTS.md`](~/.agents/AGENTS.md) apply in full — git safety,
parallel-agent safety, model selection, compute infrastructure, IMAS data
access, shell hygiene, test execution.  Only the **repo-specific**
overrides and additions live here.

## Git Workflow

**Primary branch: `main`** (per [`~/.agents/AGENTS.md`](~/.agents/AGENTS.md)
"Branch Hygiene").  Trunk-based — all agent commits land directly on `main`.
Feature branches are only created by the user when preparing a PR.

Remotes:

| Remote | Repository | Purpose |
|--------|------------|---------|
| `origin` | [`Simon-McIntosh/imas-ambix`](https://github.com/Simon-McIntosh/imas-ambix) | Fork — daily development; **GitHub Pages source** for the public plans dashboard at <https://simon-mcintosh.github.io/imas-ambix/> |
| `upstream` | [`iterorganization/imas-ambix`](https://github.com/iterorganization/imas-ambix) | Canonical |

Routine flow:

```bash
git checkout main
git pull --no-rebase origin main
# ... edit ...
git add <specific-files>
git commit -m "type(scope): ..."
git push origin main
```

When the user asks for an upstream PR, only then create a feature branch:

```bash
git checkout -b feature/my-change
git push origin feature/my-change
gh pr create --repo iterorganization/imas-ambix --base main
```

## Plans & docs (reckon — HTML-first)

All plans **and** non-plan structured docs (RCAs, incident reports,
SDCC/ops tickets, reviews, explainers, dashboards) live under
[`docs/`](docs/) as **HTML** — never markdown (markdown only for READMEs /
brief prose). State lives **in the HTML island** (`<meta name="plan-*">`
scalars + `data-reckon` sections); there are **no per-plan sidecar JSON
files** (migrated 2026-05-27 — `docs/state/<project>/index.json` keeps
only project config: sprints, milestones, timeline).

Use the reckon skills — do not hand-edit `docs/*.html`:

| Intent | Skill |
|---|---|
| New plan **or** non-plan doc (RCA, ticket, explainer → `reckon-type=doc`) | `reckon-create` |
| Edit / lock decision / followup / sprint / archive | `reckon-edit` |
| Implement plan work + record outcomes + collapse-on-landing | `reckon-ship` |
| Read-only status / health audit | `reckon-status` |
| Set up / refresh reckon infra | `reckon-sync` |

State mutations go through the reckon MCP tools (or `POST /plan/...`); if
the MCP server is down, still author the HTML via `reckon-create` and
apply state changes once it reconnects — **never fall back to markdown.**
Closure: a shipped section collapses to a 2-4 line landed-summary on the
evergreen with full detail archived under `docs/archive/` (reckon-ship §5b);
plans retire via `reckon-edit`. Full architecture:
[`~/Code/reckon/AGENTS.md`](~/Code/reckon/AGENTS.md).

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
- `export TMPDIR=/scratch_local/$SLURM_JOB_ID && mkdir -p "$TMPDIR"` — default TMPDIR is broken on this node (on non-betelgeuse partitions like `sun` use `TMPDIR=/tmp`; `/scratch_local` only exists on betelgeuse)
- Use `srun` with the same flags for interactive jobs

## 2a. GPU Job Safety — node-drain prevention (MANDATORY)

`98dci4-gpu-0003` is the **only** H200 node, shared with Group B. It has been
**drained twice** by hung ambix GPU jobs (2026-05-26, 2026-05-27 — see
`docs/rca-node-drain-2026-05-27.md`). A drain takes the whole reservation
offline until an admin resumes it. **Do not crash this node.**

**Failure mechanism.** A GPU process stuck in uninterruptible kernel sleep
(`D` state — wedged on a CUDA or GPFS call) cannot be reaped by SIGTERM or
SIGKILL. When `scancel` can't kill the step within SLURM's
`UnkillableStepTimeout` (~60 s), slurmd marks **"Kill task failed"** and
auto-DRAINs the node. Stale GPU memory from the un-reaped process then
requires `nvidia-smi --gpu-reset` or a reboot to clear.

**Binding rules for any long-running GPU job:**

1. **Install a SIGTERM/SIGINT handler** that cleanly shuts down workers,
   flushes writers, releases the model, and exits in < 5 s — well under
   `UnkillableStepTimeout`. A clean self-exit does NOT drain the node; an
   unkillable kill DOES.
2. **Add a per-shot / per-batch watchdog timeout.** Abort a single stuck unit
   (e.g. > N× the median time) rather than letting it hang the whole job.
3. **No deadlock-prone IPC.** Prefer the **in-process streaming encoder**
   (`imas_ambix/data/stream_encode.py`: torch `DataLoader` + cross-shot
   continuous batching, no subprocess daemon, no prefetch producer/consumer
   threads) over the legacy file-IPC daemon. Both removed surfaces were drain
   causes: a prefetch producer dying on a bad shot blocked the consumer
   forever; a subprocess daemon mid-CUDA was unkillable.
4. **The legacy frame-daemon / prefetch path is deprecated.** `stream_encode.py`
   (in-process, hardened: graceful SIGTERM + per-batch watchdog, commit
   `4f820da`) is the sole frame encoder going forward. The legacy path
   (`OpenMagvit2Tokenizer` subprocess daemon + the `bulk_encode_frames`
   prefetch + `encode_one_shard.py`) **deadlocked and drained the node** — do
   NOT run it. It is slated for deletion once `stream_encode` passes GPU
   validation; we do not harden or carry it.
5. **Never `scancel` a CUDA-wedged job and assume clean teardown.** Detect the
   hang early — a token-rate / heartbeat watchdog that exits cleanly — instead
   of killing a wedged process and triggering the drain.

**When a drain happens (RCA procedure):**
1. `sacct -a -N 98dci4-gpu-0003 --starttime=<window>` — list ALL users' jobs to
   determine cause and rule Group B in/out (check for non-`grpa` accounts).
2. Match the node `Reason=...[root@<ts>]` timestamp against your `scancel` /
   job-end times (`sacct` End + `.batch` ExitCode) to attribute it.
3. Write `docs/rca-node-drain-<date>.md` and give ordered admin instructions:
   **check stuck procs (`nvidia-smi`, `ps -eo pid,stat,cmd`) → `nvidia-smi
   --gpu-reset` or reboot → `scontrol update nodename=98dci4-gpu-0003
   state=resume reason=""`** (resume only after the GPUs are confirmed clean,
   else the next job inherits a dirty GPU).

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

## 5. Available Model Profiles

| Profile | Model | Engine | Size | Context | License |
|---------|-------|--------|------|---------|---------|
| `kimi-k2-6` | Kimi-K2.6 (1T MoE) | KTransformers+SGLang | 555 GB | 262K | Modified MIT |
| `deepseek-v4-flash` | DeepSeek V4-Flash (284B MoE) | SGLang | 164 GB | 1M | MIT |
| `minimax-m2-7` | MiniMax M2.7 (~220B MoE) | SGLang | 220 GB | 200K | Custom |

**Kimi-K2.6** — CPU-offloaded via KTransformers. 5 tok/s, best code quality (SWE 65.8%).
**DeepSeek V4-Flash** — Full GPU, FP4+FP8. 500–800 tok/s est., 1M context, MIT license.
**MiniMax M2.7** — Full GPU, FP8 native. 400–600 tok/s est., best agentic (GDPval-AA 1495).

### Kimi-K2.6 Deployment

**Engine:** KTransformers + SGLang (CPU+GPU hybrid MoE inference)
**Why not vLLM:** vLLM with TP=4 fills all 4×H200 VRAM (~134 GB/GPU), leaving no room for GPU sharing. KTransformers keeps only hot experts on GPU (~32 GB/GPU), leaving ~90 GB/GPU free.

**Model path:** `/work/projects/imas_gpu/agents/kimi-k2-6/model`
**Download cache:** `/work/projects/imas_gpu/agents/kimi-k2-6/.cache`

MLA compression makes KV cache tiny — full 256K context uses only 34.3 GB (9% of the 377 GB GPU KV budget), leaving ~90 GB/GPU free for sharing.

**Memory budget (per GPU, gpu_experts=280, mem_fraction=0.90):**
- Model + hot experts: ~100 GB
- KV pool: ~14 GB (49152 tokens — capped low so prefill_64k bench
  returns HTTP 400 instead of crashing the server with OOM)
- Scratch (FlashInfer workspace, prefill intermediates): ~25 GB
- **Measured live: 25.89 GB free at startup**

Previous config (gpu_experts=350, mem_fraction=0.96, max_total_tokens=
131072) crashed during prefill_16k with `torch.OutOfMemoryError`
(only 3.28 GB free) — left here as a warning.

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

### DeepSeek V4-Flash Deployment

**Engine:** SGLang native (full GPU serving, no CPU offloading)
**Architecture:** 284B total, 13B activated, 256+1 experts, 6 selected/token
**Attention:** CSA+HCA hybrid — 1 KV head, head_dim 512, compression [128, 4]
**Weights:** FP4 (experts) + FP8 (others) mixed precision

**Model path:** `/work/projects/imas_gpu/agents/deepseek-v4-flash/model`

**Memory budget (4×H200):**
- Weights: ~164 GB total (~41 GB/card)
- KV cache: ultra-efficient (~42 KB/token) — millions of tokens fit
- **Free per GPU: ~100 GB** for KV and other workloads

**Deploy:**
```bash
ambix agent download deepseek-v4-flash   # ~164 GB, run from sirius
ambix agent serve deepseek-v4-flash      # submit to betelgeuse
```

### MiniMax M2.7 Deployment

**Engine:** SGLang native (full GPU serving, no CPU offloading)
**Architecture:** ~220B total, ~10B activated, 256 experts, 8 selected/token
**Attention:** GQA with 48 Q heads, 8 KV heads, head_dim 128
**Weights:** FP8 native (float8_e4m3fn)

**Model path:** `/work/projects/imas_gpu/agents/minimax-m2-7/model`

**Memory budget (4×H200):**
- Weights: ~220 GB total (~55 GB/card)
- KV cache: ~830K tokens capacity
- **Free per GPU: ~85 GB** for KV and other workloads

**Officially tested on 4-GPU configs** (96G×4; our 140G×4 is larger).

**Engine quirks discovered 2026-05-15:**
- `torch.ops.sgl_kernel.fp8_blockwise_scaled_mm` returns
  `RuntimeError: Error Internal` on this H200 NVL + sgl-kernel 0.5.x
  build (both in CUDA-graph capture and eager mode). Workaround in
  the profile: `disable_cuda_graph=true`,
  `disable_piecewise_cuda_graph=true`, `moe_runner_backend="triton"`,
  `fp8_gemm_runner_backend="triton"`.
- The performance cost is real: measured 17 tok/s decode (vs ~110 for
  flash on the same node) because every dense FP8 matmul takes the
  triton fallback path. When sgl-kernel ships a working CUTLASS path
  this profile should switch back.

**Deploy:**
```bash
ambix agent download minimax-m2-7   # ~220 GB, run from sirius
ambix agent serve minimax-m2-7      # submit to betelgeuse
```

**Recommended inference params:** temperature=1.0, top_p=0.95, top_k=40

## 6. Models we evaluated and rejected

### DeepSeek-V4-Pro — does NOT fit on the Group A 4×H200 reservation

**Repo:** `deepseek-ai/DeepSeek-V4-Pro`
**Spec:** 1.6T total / 49B activated MoE, FP4 experts + FP8 attention/dense,
~882 GB on disk (64 safetensor shards). CSA+HCA hybrid attention, 1M context.

**Why we did not add a profile:**
- Weights alone are 882 GB → 4×H200 = 561 GB VRAM is **321 GB short**, so a
  pure-GPU deployment with TP=4 is impossible.
- The SGLang DeepSeek-V4 cookbook explicitly requires **8 GPUs on H200**
  for the FP8 checkpoint and **16 GPUs across 2 nodes** for the original
  FP4 checkpoint with Marlin — i.e. a full `98dci4-gpu-0003` node, which
  Group A's reservation only owns half of.
- A KTransformers CPU-offload deployment would in principle have room
  (650 GB CPU + ~440 GB GPU after non-expert weights), but the CSA+HCA
  hybrid attention is not currently supported by `kt-kernel` and would
  require substantial engineering before it could be tried.

**When this could be revisited:** if Group A is granted the full 8-GPU
allocation on the node (1,120 GB aggregate VRAM, enough for the FP8
SGLang checkpoint with `--tp 8`). Until then, V4-Flash is the practical
DeepSeek choice — it delivers the same 1M-token context, the same MIT
licence, and ~110 tok/s decode on the same 4 GPUs.

## 7. Confluence Reference

GPU server documentation: https://confluence.iter.org/spaces/SDCC/pages/935667046/GPU+Server+-+98dci4-gpu-0003
(Requires ITER network authentication)
