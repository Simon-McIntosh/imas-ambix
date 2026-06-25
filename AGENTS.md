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
brief prose). State lives **in the plan's own HTML** — `<meta name="plan-*">`
scalars in the head + `data-reckon` section elements in the body; there are
**no per-plan sidecar JSON files**. Agents edit the plan file directly
(reckon MCP tools / `reckon-edit` for version-safe writes, or by hand with a
bypass note) — the HTML *is* the store, not a separate state object (migrated 2026-05-27 — `docs/state/<project>/index.json` keeps
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
- QoS: **do NOT pass `--qos`** — jobs then default to QOS `normal` (priority 1, **no GPU cap**). The
  `gpu_0003_grpa` QOS carries the 4-GPU cap (`gres/gpu=4` GrpTRES + `DenyOnLimit`); passing it
  explicitly is what limits you to 4 cards (and `QOSGrpGRES`-denies a 6/8-GPU request at submit).
  Under `normal` the only enforced per-group limit is the reservation's 30 CPU cores.

**We are authorised to use the full GPU server (all 8 H200 cards), and the reservation is ours to
utilise — use the free cards wherever possible (idle GPUs are capacity to use, not a cost to conserve).**
A single Group A submit under `normal` QOS bursts onto any free cards node-wide (the reservation reserves
only cores); `CUDA_VISIBLE_DEVICES` is remapped to `0..N`, so the job never knows which physical silicon
it is on. **Check `squeue` for the live state and request what is free:**
- **DSv4 down (its default state unless someone has started a serve) → all 8 cards are free → request them**
  (`--gres=gpu:8`). This is the common case; size big runs for the full 8.
- **DSv4 up → it holds 2 cards → a 6-card job coexists with it** (6 + 2 = 8/8) — confirmed 2026-06-19:
  `--gres=gpu:6` started immediately on physical cards `0,1,2,3,6,7` (skipping the cards DSv4 held); an
  8-GPU job is then accepted and queues until they free. Keep DSv4 up only when it is actually serving; do
  not leave it occupying cards an active training campaign needs.

Mechanism + the cooperative-yield watcher: `docs/gpu-preemptible-scheduling.html`.

**SLURM submission pattern (training run — request all free cards, normal QOS):**
```bash
sbatch --partition=betelgeuse \
       --reservation=gpu_0003_grpA \
       --account=grpa \
       --gres=gpu:8 \        # all 8 when DSv4 is down (check squeue); use gpu:6 to coexist if DSv4 is up
       --cpus-per-task=12 \   # PROBE with srun --test-only first; more cores free when DSv4 is down
       --mem=640G \
       your_script.sh        # do NOT pass --qos → defaults to normal → no 4-GPU cap
```

**CPU sizing — the "30 cores" is nominal, not schedulable when a serve co-runs (confirmed 2026-06-21).**
The node carries TWO overlapping 30-core reservations (grpA + grpB) and the DeepSeek serve runs on general
cores, so the cores actually schedulable for a NEW grpA job while DeepSeek is up are **~12, not 30**. A
`--cpus-per-task=30` request then pends forever on `Reason=Resources` even though all 6 GPUs are free — it is
CPU, not GPU, that is short (incident: re-train job pended + cancelled, 2026-06-21). **Probe before sizing:**
`srun --test-only --reservation=gpu_0003_grpA --account=grpa --gres=gpu:6 --cpus-per-task=N --mem=… true`.
Size `--cpus-per-task` (and the DataLoader `num_workers`) to leave room for the co-running serve — ~12 cores
is ample for a token-DataLoader DDP run (tokens are tiny). The GPU burst itself is unaffected (node-wide
under normal QOS). If you genuinely need the full core count, drop `--reservation` and run on the node's
general free pool (~49 cores idle) instead. **When DSv4 is down the contention vanishes — the general core
pool is free too, so an 8-GPU run can size cores up; still PROBE with `srun --test-only` before committing.**

**Design every training run for fast takedown (binding etiquette).** Preemption is OFF cluster-wide,
so SLURM cannot evict us — bursting onto cards 6–7 without a fast give-back makes us a bad neighbour to
Group B. Every run MUST be **resume-safe**: frequent checkpoints (`latest.pt`), a clean `SIGTERM` flush
(< 5 s), and checkpoint-resume on restart, so a `scancel` (or a cooperative `scontrol requeue`) costs
~nothing and frees the cards in seconds.

**Network access:**
- **GPU nodes** (betelgeuse): GPFS `/work/projects/imas_gpu/` **IS mounted and accessible** here — read/write
  the corpus, tokens, and checkpoints directly from the GPU job. The *only* thing missing is **outbound
  internet**, so model downloads and package installs must happen elsewhere; data I/O against `/work` does NOT.
- **Standard compute nodes** (sirius, rigel, etc.): Full outbound network AND access to `/work/projects/imas_gpu/`.
- **Login nodes**: Full network but NO access to `/work/projects/imas_gpu/` (requires `sdcc-imas_gpu` group).
- **Strategy**: Download models and install packages from standard compute nodes into shared GPFS; everything
  thereafter (encode, train, eval) reads/writes `/work` directly from the GPU nodes — no offload needed for I/O.

**SLURM workarounds:**
- `export TMPDIR=/scratch_local/$SLURM_JOB_ID && mkdir -p "$TMPDIR"` — default TMPDIR is broken on this node (on non-betelgeuse partitions like `sun` use `TMPDIR=/tmp`; `/scratch_local` only exists on betelgeuse)
- Use `srun` with the same flags for interactive jobs

## 2a. GPU node stability — not an agent concern (settled 2026-06-10)

`98dci4-gpu-0003` occasionally drains because of an **environmental,
node-level condition** — a kernel-driver / CXI-Slingshot-fabric / GPFS D-state
that occasionally fails to resolve within SLURM's `UnkillableStepTimeout`. It is
**outside user-space control and is not an agent concern.** Settled root cause +
the full test record: `docs/rca-node-drain-final-2026-06-03.html`; the actual
fix is admin-side (`docs/proposal-drain-auto-recovery.html`).

- **`scancel`, `SIGKILL`, and `#SBATCH --time` expiry are exonerated.** All were
  fired deliberately at live CUDA/NCCL workloads, repeatedly, and never drained
  the node. Use `scancel` and ordinary `--time` limits freely — they are the
  supported ways to stop a job.
- **Do NOT build drain-defence scaffolding.** The former STOP-FILE contract,
  never-`scancel` rule, three-layer time-limit rule, and "hardened sbatch
  header" defended against a mechanism that does not exist on this stack
  (16 deliberate, zero-drain reproduction attempts). They were removed
  2026-06-10; do not reintroduce them — in code, sbatch headers, or plans.
- **A drain is an SDCC/admin recovery** (GPU reset + `scontrol … state=resume`),
  not something to engineer around or treat as a code defect. If one recurs, log
  it to SDCC against the admin proposal.

Normal good practice that stands on its own merits — a clean `SIGTERM` for
lossless cancellation, in-process model-load-once for throughput, fail-fast on
missing offline assets, symmetric collectives — is covered in §2b. Keep it
because it is good code, **not** because of drains.

## 2b. Performant GPU code (in-process default)

**Principle:** GPU code in this repo runs in a **single long-lived process** that
loads the model once and processes many shots in a loop. Any pattern that
pays model-load cost per item — subprocess-per-shot, daemon-with-IPC, or
prefetch producer/consumer threads — is **PROHIBITED** for production code.
(Smoke tests and one-shot CLI tools are exempt.)

**Why — three measured regressions from this repo:**

1. **Corpus encode:** the legacy frame daemon deadlocked (prefetch
   producer/consumer hang, 2026-05-27) and delivered poor throughput.
   Replacing it with in-process `stream_encode.py`
   (torch `DataLoader`, bounded queues) yielded 308 fps/GPU peak and encoded
   4.02B tokens in ~7 h on a single GPU.
2. **Bench — subprocess-per-shot:** 100-shot rbb bench took **65 min** (job
   1208872) because `OpenMagvit2Tokenizer` spawned ~200 subprocesses, each
   reloading the VQModel. In-process `stream_worker.py` (job 1208918) ran the
   same 100 shots in **8 min 46 s** — ~55× speedup on the GPU phase, ~7.4×
   overall including CPU metrics.
3. **Fragility:** interim subprocess-per-shot bench (job 1208896) hit
   "Bus error (core dumped)" mid-run after a filesystem hiccup. The in-process
   run immediately after processed all 100 shots without incident.

**Canonical reference modules — copy these patterns:**

- `imas_ambix/data/stream_encode.py` — corpus encoder. Holds VQModel across
  all shots. torch `DataLoader` for shot loading. SIGTERM handler flushes async
  Zarr writers and tears down DataLoader workers cleanly on shutdown.
- `imas_ambix/bench/stream_worker.py` — bench encoder + decoder. Same
  hardening: SIGTERM handler sets a `STOP` flag, per-shot watchdog
  auto-tunes its timeout from the running median, `try/finally` releases
  model + calls `torch.cuda.empty_cache()`.

**Required of every new GPU-bound code path:**

- Model loaded **once** outside the per-item loop.
- `SIGTERM`/`SIGINT` handler sets a `STOP` flag and exits cleanly in < 5 s.
- Per-item watchdog that sets `STOP` on timeout rather than blocking forever.
- `try/finally` releasing the model + `torch.cuda.empty_cache()`.
- `torch.backends.cudnn.benchmark = False` + `torch.backends.cudnn.deterministic = True`
  for reproducibility (see root-cause comment in `stream_encode.py`).
- `torch.set_float32_matmul_precision("high")` + bf16 on CUDA (H200 tensor cores).

**Forbidden patterns:**

- `subprocess.run` / `subprocess.Popen` per shot to a worker that reloads the
  model. Model load is the dominant cost; this is never performant.
- A persistent worker daemon driven by named pipes / FIFOs / file-IPC. These
  have real deadlock failure modes (the 2026-05-27 prefetch hang) and offer
  no advantage over in-process now that `stream_encode.py` exists.
- Prefetch producer/consumer threads with unbounded queues. A failure in either
  thread wedges the other. Use the torch `DataLoader` worker pool with bounded
  queues instead.

**Tooling:**

- `imas-ambix tokenize bench --in-process` is the default. The `--no-in-process`
  flag is a debug escape hatch only.
- `scripts/slurm/bench_rbb.sbatch` always runs in-process.

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

The `imas-ambix agent` CLI manages LLM deployments via TOML model profiles:

```bash
imas-ambix agent list                          # List available profiles (marks any serving)
imas-ambix agent info kimi-k2-6               # Show profile details + memory budget
imas-ambix agent download kimi-k2-6           # Submit SLURM download job (sirius partition)
imas-ambix agent serve kimi-k2-6              # Submit SLURM serve job (betelgeuse partition)
imas-ambix agent serve kimi-k2-6 --dry-run    # Print script without submitting
imas-ambix agent status                        # Jobs + connection block (URL, key, readiness)
imas-ambix agent tunnel glm-5-2               # SSH-forward the serve port to localhost
imas-ambix agent key --rotate                 # Rotate the shared API key + restart serve
imas-ambix agent clive --deploy               # Generate+deploy the clive launcher (see §4a)
```

Adding a new model: create a TOML file in `imas_ambix/agent/profiles/<slug>.toml`.

## 4a. Driving an interactive agent against the local model (`clive`)

`clive` ("CLI + live") points **Claude Code** (Anthropic Messages API) or the
**OpenAI Codex CLI** (OpenAI API) at the served model. The vLLM server exposes
**both** `/v1/messages` and `/v1/chat/completions` on the same port, so one
server, key, and model back either harness — with full reasoning, tool calling,
and prompt caching (vLLM automatic prefix caching; >0.17.1 handles Claude
Code's per-request hash, so caching is not defeated). No gateway/LiteLLM/router
is needed — the Anthropic endpoint is native to vLLM.

```bash
clive "explain this repo"          # Claude Code (default), auto-tunnels from a login node
clive --codex "write a test"       # Codex CLI, same server/key/model
clive --model glm-5-2-fp8 ...       # override the served-model name
imas-ambix agent clive --deploy    # (re)generate the launcher to GPFS after a config change
imas-ambix agent clive --path      # print the ~/.bashrc PATH line
```

- **Generated, not hand-written.** `imas-ambix agent clive --deploy` renders the
  script from `SiteConfig` (URL, port, key-file, default model) to
  `/work/projects/imas_gpu/agents/clive` (mode 755, group `sdcc-imas_gpu`), so
  it never drifts from what the server serves. The generator is
  `imas_ambix/agent/clive.py`; do not edit the deployed copy in place.
- **Shared by group, not by copying keys.** The launcher reads the shared
  mode-640 key file at runtime; the key never appears on a command line.
  Group-mates add `/work/projects/imas_gpu/agents` to their PATH and run `clive`.
- **Auto-tunnels.** From a login node, `clive` opens an SSH forward
  (`localhost:PORT → GPU-node:PORT`) and points the harness at localhost; on the
  GPU node it connects directly.
- **Operator vs consumer.** `imas-ambix` is the *operator* CLI (serve/manage,
  per-user repo venv); `clive` is the *consumer* launcher (shared on GPFS).

## 5. Available Model Profiles

| Profile | Model | Engine | Size | Context | License |
|---------|-------|--------|------|---------|---------|
| `kimi-k2-6` | Kimi-K2.6 (1T MoE) | KTransformers+SGLang | 555 GB | 262K | Modified MIT |
| `deepseek-v4-flash` | DeepSeek V4-Flash (284B MoE) | SGLang | 164 GB | 1M | MIT |
| `minimax-m2-7` | MiniMax M2.7 (~220B MoE) | SGLang | 220 GB | 200K | Custom |
| `glm-5-2` | GLM-5.2 (~744B MoE) | vLLM | 744 GB | 256K¹ | MIT |

¹ GLM-5.2 context is capped at **256K** on this hardware, not its native 1M —
see the deployment note below.

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

**Engine:** vLLM native (full GPU serving, no CPU offloading)
**Architecture:** 284B total, 13B activated, 256+1 experts, 6 selected/token
**Attention:** CSA+HCA hybrid — 1 KV head, head_dim 512, compression [128, 4]
**Weights:** FP4 (experts) + FP8 (others) mixed precision

**Model path:** `/work/projects/imas_gpu/agents/deepseek-v4-flash/model`

**Profiles:**
- `deepseek-v4-flash` — 4×H200, TP=4, target 500–800 tok/s
- `deepseek-v4-flash-2x` — 2×H200, TP=2, ~100 tok/s single / ~114 tok/s 8-way concurrent

**Memory budget (4×H200):**
- Weights: ~164 GB total (~41 GB/card)
- KV cache: ultra-efficient (~42 KB/token) — millions of tokens fit
- **Free per GPU: ~100 GB** for KV and other workloads
- KV pool at 2×H200: 1,372,204 tokens (measured 2026-06-04)

**Thinking mode API (vLLM, all client-side per request):**
- Non-think: no `chat_template_kwargs` (default)
- Think: `chat_template_kwargs={"thinking": True}`  → `message.reasoning` populated
- Think Max: `chat_template_kwargs={"thinking": True, "reasoning_effort": "max"}` → REASONING_EFFORT_MAX prefix injected
- Think High: `chat_template_kwargs={"thinking": True, "reasoning_effort": "high"}`
- Think + tools: works — `message.reasoning` AND `message.tool_calls` both present

**Deploy:**
```bash
imas-ambix agent serve deepseek-v4-flash      # 4× GPUs — 400+ tok/s target
imas-ambix agent serve deepseek-v4-flash-2x   # 2× GPUs — share node with other work
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
imas-ambix agent download minimax-m2-7   # ~220 GB, run from sirius
imas-ambix agent serve minimax-m2-7      # submit to betelgeuse
```

**Recommended inference params:** temperature=1.0, top_p=0.95, top_k=40

### GLM-5.2 Deployment

**Engine:** vLLM native, full-GPU, **TP=8 on all 8×H200** (no CPU offload).
**Model path:** `/work/projects/imas_gpu/agents/glm-5-2/model`
**Why all 8 cards:** the ~744 GB FP8 checkpoint needs the full 8×140 GB VRAM;
this run fills the node (no coexisting GPU job). vLLM exposes both the OpenAI
API and the **Anthropic Messages API** (`/v1/messages`) natively → drive it with
`clive` (§4a).

**Hard-won deployment facts (measured 2026-06-25):**
- **Context capped at 256K, not 1M.** The FP8 weights + MTP draft model leave a
  fixed ~24 GiB aggregate for KV at `mem_fraction 0.95`. Measured ceilings: 1M
  KV-OOMs (needs 60.8 GiB; est. max 233K at 0.90), 512K KV-OOMs (needs 30.4 GiB;
  est. max 419K at 0.95). `max_total_tokens=262144` needs ~15 GiB → fits with
  1.72× concurrency headroom. The native 1M genuinely needs 8×B200 (180 GB).
- **Eager mode (`disable_cuda_graph=true`).** CUDA-graph capture stalled at ~88%
  of the piecewise graph set on three consecutive launches (each ~60 min, never
  reaching startup; the third left an unkillable step that auto-rebooted the
  node). The TP=8 + MTP piecewise capture is the hang surface on this H200 NVL +
  vLLM 0.23.0 build. Eager skips capture → ready right after weight-load + KV
  init. Decode throughput is lower without graphs; re-enable + tune (disable MTP
  capture, or cap `cuda_graph_max_bs`) once a stable capture config is found.
- **MTP speculative decoding** (`--speculative-config.method mtp`,
  `num_speculative_tokens 5`) is GLM-5.2's headline throughput feature and is
  enabled in the profile.
- **Reasoning effort wired** end-to-end (Claude Code `effort` → vLLM
  `reasoning_effort` → GLM template `enable_thinking`), but GLM-5.2's chat
  template effectively offers **two levels** — `high`, or `max` for everything
  else.
- **vLLM env required transformers 5.x + flashinfer 0.6.12 etc.** The serve venv
  was upgraded 0.20.2 → **0.23.0** for GLM-5.2 (`GlmMoeDsaForCausalLM`, glm45/
  glm47 parsers, MTP). The setup wheel-resolver was fixed for the
  manylinux_2_28 tag drop.

**Deploy:**
```bash
imas-ambix agent download glm-5-2   # ~744 GB, run from sirius
imas-ambix agent serve glm-5-2      # all 8 cards; ~30 min weight load in eager mode
clive "..."                          # drive Claude Code against it (§4a)
```

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
