# Agent Guidelines — IMAS Ambix · GPU Serving & Model Deployment

Scoped to the `imas_ambix/agent/` sub-tree: the GPU server, SLURM access, the
`imas-ambix agent` CLI, model profiles, and serving. This file is read
automatically by agentic tools when working anywhere under `imas_ambix/agent/`.
Repo-wide rules (git workflow, plans) live in the repo-root `AGENTS.md`; the
user-global rules in `~/.agents/AGENTS.md` apply in full.

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
| `/work/projects/imas_gpu/agents/{vllm,sglang}/.venv/` | Shared serving environment per engine | GPFS | — |
| `/scratch_local/` | Fast staging, ephemeral per-node | Local NVMe | 5.9 TB |

Engine versions are established by the serving-node setup verification log.
Do not infer readiness from files observed only on the network-enabled install
node.

## 4. Composable Agent CLI

The `imas-ambix agent` CLI manages LLM deployments via TOML model profiles:

```bash
imas-ambix agent list                          # List available profiles (marks any serving)
imas-ambix agent info kimi-k2-6               # Show profile details + memory budget
imas-ambix agent setup vllm                   # Install, then verify on the serving node
imas-ambix agent download kimi-k2-6           # Submit SLURM download job (sirius partition)
imas-ambix agent serve kimi-k2-6              # Submit SLURM serve job (betelgeuse partition)
imas-ambix agent serve kimi-k2-6 --dry-run    # Print script without submitting
imas-ambix agent status                        # Jobs + connection block (URL, key, readiness)
imas-ambix agent key --rotate                 # Rotate the shared API key + restart serve
imas-ambix agent clive --deploy               # Generate+deploy the clive launcher (see §4a)
```

Adding a new model: create a TOML file in `imas_ambix/agent/profiles/<slug>.toml`.

**Setup readiness contract:** `agent setup` submits a network-enabled install
job followed by a dependent runtime verification job on `betelgeuse`. The
verification runs after the producer allocation exits and checks the
consumer-visible interpreter plus engine package metadata. The environment is
ready only when the runtime verification job reaches `COMPLETED`; an install
job reaching `COMPLETED` by itself is not readiness. If verification fails,
do not submit a serve job—the GPFS namespace visible to the serving node does
not contain an executable environment.

## 4a. Driving an interactive agent against the local model (`clive`)

`clive` ("CLI + live") points **Claude Code** (Anthropic Messages API) or the
**OpenAI Codex CLI** (OpenAI API) at the served model. The vLLM server exposes
**both** `/v1/messages` and `/v1/chat/completions` on the same port, so one
server, key, and model back either harness — with full reasoning, tool calling,
and prompt caching (vLLM automatic prefix caching; >0.17.1 handles Claude
Code's per-request hash, so caching is not defeated). No gateway/LiteLLM/router
is needed — the Anthropic endpoint is native to vLLM.

```bash
clive "explain this repo"          # local H200 model; no OR key = direct
clive "refactor this"              # with OR key = tiered routing (local+OR)
clive --codex "write a test"       # local model via Codex CLI, same server
orclaude "..."                     # all tiers -> OpenRouter (bashrc fn; key-holders)
imas-ambix agent clive --deploy    # (re)generate + sync launcher + routing config
imas-ambix agent clive --path      # print the ~/.bashrc PATH line
```

**One launcher (`clive`), adaptive routing:**

- **Without an OpenRouter key** (`~/.config/openrouter/key`): clive connects
  directly to the local H200 model (all tiers → one model). Auto-detects the
  running model from `/v1/models`. What every SDCC cluster user runs.
- **With an OpenRouter key**: clive ensures the **per-user `imas-ambix-llm`
  systemd service** is running (`systemctl --user start` — idempotent, **0 s
  when already up**; ~11 s cold start once per login) and routes Claude Code
  tiers through its LiteLLM proxy:
  - Default / haiku → the local H200 model (named after the served model,
    e.g. `deepseek-v4-flash`)
  - Sonnet → `or-sonnet-4.6`, Opus → `or-opus-4.8` (OpenRouter)
  - Also in the picker: `or-gpt-5.5`, `or-glm-5.2`
  The proxy is **per-user** (runs as the invoking user with *their own* OR key,
  bound to 127.0.0.1 only — NOT shared, no shared-key/budget exposure). No OR
  key → no proxy, local-only.
- **`orclaude` — all tiers → OpenRouter** (the `~/.bashrc` function).

**Why systemd, not a per-run daemon:** litellm's import costs 10-20 s; starting
it per `clive` invocation was the bottleneck. The `imas-ambix-llm.service`
(installed by `--deploy`) stays alive across runs, so only the first launch
after login pays the cost. `EnvironmentFile` MUST be `-`-optional (the
ExecStartPre helper creates it) or systemd fails with `Result=resources`.

**Model picker control (Claude Code 2.1+) — env vars, not discovery:** the
picker is populated AND labelled purely with **startup env vars**, no settings
file and no `/v1/models` gateway discovery. Two facts kill the discovery route:
Claude Code **ignores any `/v1/models` entry whose `id` doesn't begin with
`claude`/`anthropic`** (our `or-*`/local ids don't, so they're silently
dropped), and `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC` (which clive sets)
disables discovery anyway. Instead clive uses the **five labellable slots Claude
Code honours behind an `ANTHROPIC_BASE_URL` gateway** — the `opus` / `sonnet` /
`haiku` / `fable` aliases plus one `ANTHROPIC_CUSTOM_MODEL_OPTION` — each taking
a `_NAME` + `_DESCRIPTION` (model-config doc: the `_NAME`/`_DESCRIPTION` vars
"also take effect when `ANTHROPIC_BASE_URL` points to an LLM gateway"). We have
exactly five models, so the mapping is one-to-one:

| Picker slot | Model | Notes |
|---|---|---|
| `haiku` (+ session default via `ANTHROPIC_MODEL=haiku`) | local H200 | also backs background tasks; free |
| `opus` | `or-opus-4.8` | OpenRouter Claude, frontier |
| `sonnet` | `or-sonnet-4.6` | OpenRouter Claude, balanced |
| custom option | `or-gpt-5.5` | plain entry, no Claude-specific shaping |
| `fable` | `or-glm-5.2` | ⚠ Claude Code treats the fable slot as Fable 5 → always-on adaptive thinking + safety auto-fallback to the opus slot; benign behind `drop_params`, and it's the only remaining labelled slot |

This shows real names + descriptions (not "Custom * model"), needs no
`--settings` and no deployed settings file, and touches **only clive's own
process env** — the user's plain `claude` is completely unaffected. Set
`ANTHROPIC_MODEL=haiku` (the alias), NOT the raw served id, or Claude Code adds
a duplicate "Custom model" row alongside the haiku slot.

**OpenRouter caching — use `anthropic/<model>` + `api_base`, NOT `openrouter/`:**
verified 2026-06-26 (or-opus, 10,815-token prompt): `cache_creation` then
`cache_read=10815` on the repeat → cost $0.068→$0.0055. The Anthropic-native
passthrough preserves `cache_control`; `openrouter/anthropic/<model>` routes via
the OpenAI chat_completions wire format and **silently drops caching**. Claude
`or-*` models use `anthropic/` + `api_base`; OpenAI-family (`or-gpt-*`) use
`openai/`.

**Routing config = LiteLLM YAML, generated like the launcher.** Routing lives
in `litellm_config.yaml` (deployed to `/work`), generated from
`imas_ambix/agent/litellm_config.py` with the local URL/model auto-filled from
SiteConfig. It is **secret-free** — the OpenRouter key is `os.environ/OPENROUTER_API_KEY`,
injected by the systemd unit's env helper from the per-user
`~/.config/openrouter/key`. To change routing, edit the generator's YAML and
re-deploy.

**Sync discipline — repo is the source of truth (binding):** `imas-ambix agent
clive --deploy` generates and writes all of: `clive`, `litellm_config.yaml`,
`imas-ambix-llm-env.sh` (→ `/work/.../agents/`, group
`sdcc-imas_gpu`), and the per-user `imas-ambix-llm.service` (→
`~/.config/systemd/user/`, then `daemon-reload`). Generators:
`imas_ambix/agent/{clive,litellm_config,litellm_service}.py`. **NEVER hand-edit
the deployed copies** — they are disposable artifacts. To change any: edit the
generator in the repo, commit, then re-run `--deploy` to re-sync. This is the
only thing that keeps the deployed copies from drifting from the repo.

- **Direct route, no tunnel.** Login and standard compute nodes route directly
  to `<gpu-node>:PORT` (verified 2026-06-25: login → `98dci4-gpu-0003:18800` =
  200). SSH `-L` port-forwarding to the compute node is **administratively
  prohibited** (`channel … open failed: administratively prohibited`), so the
  launchers use the direct URL and do not tunnel.
- **Operator vs consumer.** `imas-ambix` is the *operator* CLI (serve/manage,
  per-user repo venv); `clive` is the *consumer* launcher (shared on GPFS,
  secret-free).

## 5. Available Model Profiles

| Profile | Model | Engine | Size | Context | License |
|---------|-------|--------|------|---------|---------|
| `kimi-k2-6` | Kimi-K2.6 (1T MoE) | KTransformers+SGLang | 555 GB | 262K | Modified MIT |
| `deepseek-v4-flash` | DeepSeek V4-Flash (284B MoE) | SGLang | 164 GB | 1M | MIT |
| `minimax-m2-7` | MiniMax M2.7 (~220B MoE) | SGLang | 220 GB | 200K | Custom |
| `glm-5-2` | GLM-5.2 (~744B MoE) | vLLM | 744 GB | 224K¹ | MIT |

¹ GLM-5.2 context is capped at **224K** on this hardware, not its native 1M —
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

Login and standard compute nodes route **directly** to the GPU node's serve
port — no SSH tunnel (verified 2026-06-25: login → `98dci4-gpu-0003:18800` =
200; SSH `-L` forwarding to the compute node is administratively prohibited).
```bash
# Find the compute node
squeue -j <jobid> -o %N

# Connect directly (the key is enforced; see `imas-ambix agent key`)
curl -H "Authorization: Bearer $KEY" http://<compute-node>:18800/v1/models
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

**Hard-won deployment facts (measured 2026-06-25; working config = 224K @ 0.86,
job 1222821 — verified with a real decode + a live `clive`/Claude Code request).**
- **Two HBM constraints fight; context capped at 224K, not 1M.** The FP8 weights
  + MTP draft model + the FP8 sparse-MLA attention kernel (which allocates
  **~8 GiB/card of scratch lazily on the FIRST real decode**) leave little room
  for KV. Measured: at `mem_fraction 0.95` the KV pool was ~24 GiB (256K fits)
  but the first real decode **OOM'd** (8 GiB needed, 2.4 free → EngineDeadError,
  job 1222807); at `0.86` decode scratch is ample but KV is only 13.63 GiB, so
  256K KV-init **fails** (needs 15.2; job 1222818). Resolution: keep
  `mem_fraction 0.86` (decode-safe) and size context to fit the KV that leaves —
  **`max_total_tokens=229376` (224K) needs ~13.3 GiB < 13.63**. **Startup +
  KV-init passing does NOT prove inference fits** — the kernel scratch is lazy,
  so always validate with a real generation. The native 1M needs 8×B200.
- **CUDA graphs are ON** (graph-accelerated). The earlier "capture stalls" on
  jobs 1222717/1222745/1222792 were **memory pressure** at `mem_fraction`
  0.90/0.95, not an inherent TP=8+MTP bug — at 0.86 capture completes in ~70 s
  ("Graph capturing finished in 70 secs, took 2.84 GiB", job 1222821) and the
  server serves real traffic. (`disable_cuda_graph` in the profile is a no-op
  here: vLLM ignores it — it is an SGLang-only flag.) One root cause —
  mem_fraction too high — caused both the capture stalls and the decode OOM.
- **MTP speculative decoding** (`--speculative-config.method mtp`,
  `num_speculative_tokens 5`) is GLM-5.2's headline throughput feature and is
  enabled in the profile.
- **Throughput (measured 2026-06-26, job 1222821, 224K/0.86):** single request
  **~33–37 tok/s** decode; **aggregate** throughput scales with concurrency —
  ~210 tok/s @ 8 concurrent, ~457 @ 16, ~680 @ 32, **~974 tok/s @ 48**. Healthy
  concurrency is **~16–32 simultaneous requests** (per-request rate holds ~25–35
  tok/s through n=16, then trades latency for aggregate). The scheduler cap is
  `max_num_seqs=1024` (not the bottleneck); the real ceiling is the ~13.6 GiB KV
  pool — at 224K/request the KV-limited concurrency is ~1× (one full-context
  request), but typical agent requests use far less context, so dozens run
  concurrently. **Low concurrency (n=1–4) is MTP-overhead-dominated** and slower
  per-aggregate than n=8+; the model shines under batch load.
- **Reasoning effort wired** end-to-end (Claude Code `effort` → vLLM
  `reasoning_effort` → GLM template `enable_thinking`), but GLM-5.2's chat
  template effectively offers **two levels** — `high`, or `max` for everything
  else.
- **vLLM env required transformers 5.x + flashinfer 0.6.12 etc.** The serve venv
  was upgraded 0.20.2 → **0.23.0** for GLM-5.2 (`GlmMoeDsaForCausalLM`, glm45/
  glm47 parsers, MTP). The setup wheel-resolver was fixed for the
  manylinux_2_28 tag drop.
- **FP8 is the only practical precision on H200 — FP4 is Blackwell-only.**
  Native FP4 tensor cores do not exist on Hopper (H200, CC 9.0); they were
  introduced with Blackwell (B200/GB200/RTX PRO 6000). A `GLM-5.2-NVFP4`
  checkpoint exists (~459 GB vs our 744 GB FP8 — would free ~285 GB for KV), but
  it **requires Blackwell**: on H200 vLLM can only load it via the Marlin
  software fallback, which is *unaccelerated* (no FP4 silicon → likely slower
  than FP8, not faster) and *correctness-buggy* ("NVFP4 Marlin assumes scales
  ≥0 but encountered negative scales"). So **224K-FP8 is the best/only sound
  setup on this hardware** — the tightness is the 744 GB footprint + MTP, not a
  quantization choice. The lever that would actually free headroom on H200 is
  **dropping MTP** (reclaims the draft model's weights + scratch for KV/context),
  trading its throughput benefit. FP4 only becomes viable if Group A gets
  Blackwell cards. (NVFP4 details: <https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/>;
  H200 = Hopper/FP8, B200 = Blackwell/FP4.)

**Deploy:**
```bash
imas-ambix agent download glm-5-2   # ~744 GB, run from sirius
imas-ambix agent serve glm-5-2      # all 8 cards; ~30 min weight load (GPFS-cold)
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
