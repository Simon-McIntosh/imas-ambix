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

**CPU sizing — cores, not cards, bind concurrent work.** The Group A reservation
contains 30 cores. In a live 2026-09-01 snapshot, the DeepSeek serve held 8 and
a co-running CPU-only job held 16. Those two jobs alone left at most 6 reserved
cores, so a new 12-core serve could not be admitted even while enough cards
were free. The available core count is therefore live workload state, not a
number to infer from the requested card count.
Size `--cpus-per-task` (and the DataLoader `num_workers`) to leave room for
co-running jobs. The GPU burst itself is unaffected under normal QOS, but free
cards do not imply that the reservation can admit the accompanying core request.

**Memory binds too, and `--mem=0` is an exclusive-node request in disguise.**
Measured 2026-09-06: a one-card, eight-core job sat in `Resources` while three
cards and sixteen group-A cores were free, because it requested
`mem=1500G` — the node's entire `RealMemory` of 1,536,000 MB. `--mem=0` reads
as "no limit" and means "reserve everything". Such a job **cannot start while
anyone else holds memory**, including group B, so on a shared node it
effectively never runs. Worse, it **head-of-line blocks**: a sensible 128G job
behind it showed `Priority`, not `Resources`, so nothing in the queue state
pointed at the real cause — `Resources` looks identical to genuine contention.

Size `--mem` to actual need, and check before blaming contention:

```bash
scontrol show job <id> | grep -oE 'ReqTRES=[^ ]*'                # mem near 1500G?
scontrol show node 98dci4-gpu-0003 | grep -E 'RealMemory|AllocMem'
```

A request above `RealMemory − AllocMem` cannot start now; one above
`RealMemory` less whatever group B routinely holds will effectively never start.

**Do NOT size from `srun --test-only` — it is inert on these reservations and
answers every request identically.** Measured 2026-09-05: against
`--reservation=gpu_0003_grpA` it reported a start time of exactly *now plus
seven days* for `--gres=gpu:4 --cpus-per-task=12`, for `gpu:2/12`, for `gpu:4/8`,
for `gpu:4/4` — and for a **one-core, no-GPU, 1 GB** request. A probe that
returns the same answer for a trivial request and a four-card one is measuring
nothing; the likely cause is the reservation's `IGNORE_JOBS` flag. Read
literally it says a serve cannot start for a week. It is wrong: the four-card
DSpark serve submitted minutes later started **immediately**. Dropping the
reservation to probe around it fails with `Access/permission denied`, so there
is no unreserved control either.

**Size from the live core ledger instead**, which is readable and correct:

```bash
scontrol show res gpu_0003_grpA | grep -E 'CoreCnt|CoreIDs'   # 30 cores, IDs 15-29,47-61
squeue -w 98dci4-gpu-0003 -o "%.10i %.9u %.18j %.4C %.22v"    # cores per job, per reservation
```

Sum the `%C` column for jobs whose reservation is **grpA** and subtract from 30 —
group B's jobs run on their own 30 and do not draw on ours, so counting
node-wide `CPUAlloc` overstates what binds you. Then submit and watch: a job
pending on `Resources` is the real signal, and cancelling it is cheap.

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
| `~/.local/share/ambix/engine-envs/{vllm,sglang}/.venv/` | Per-user serving environment per engine | Home GPFS | — |
| `/scratch_local/` | Fast staging, ephemeral per-node | Local NVMe | 5.9 TB |

Engine versions are established by the serving-node setup verification log.
Do not infer readiness from files observed only on the network-enabled install
node.

The environment root is configured by `AMBIX_AGENT_ENGINE_ENV_ROOT`; its
default follows `XDG_DATA_HOME` or `~/.local/share`. Setup requires at least
32 GiB free by default, configurable with
`AMBIX_AGENT_ENGINE_ENV_MIN_FREE_GB`.

## 3a. Serving-interpreter compatibility (binding)

The generated serve script runs repository modules with the selected engine's
`SiteConfig.python_path(profile.engine.type)`, not the project interpreter. The
pin is the `requires-python` entry in
`imas_ambix/agent/envs/<engine>/pyproject.toml`; both serving environments
currently require `>=3.12,<3.13`. Read the selected environment's project file
as the authority. The repository root instead requires Python 3.14 and Ruff
targets `py314`, so whole-tree lint does not prove serving-node compatibility.

Every serve executes `imas_ambix.agent.registry` on the compute node. A vLLM
serve also loads `imas_ambix.agent.vllm_catalog` and
`imas_ambix.agent.vllm_think_marker` as middleware, and a generated router starts
`imas_ambix.cli` under the vLLM interpreter. Any imported repository module must
therefore parse at the serving environment's lower bound: newer syntax is a hard
failure at import time, before the endpoint can start.

The incompatibility has recurred three times as an unparenthesized exception
tuple. Python 3.14 accepts the first form; the serving interpreter requires the
second:

```python
# Incorrect for the serving interpreter
except OSError, TypeError, ValueError:

# Correct across both interpreters
except (OSError, TypeError, ValueError):
```

Check the package grammar before committing:

```bash
uv run pytest tests/test_agent_serve_registry.py \
  -k 'agent_package_modules_parse_with_serving_python or serving_python_guard_rejects_unparenthesized_exception_tuple' -q
```

## 3b. Live endpoints — what to point a client at

Two services, two different shapes. Both are **keyless** and both are reached
**directly** from login and standard compute nodes (measured: compute node to
each port, open). There is no SSH tunnel — port-forwarding to a compute node is
administratively prohibited.

| Service | Origin | Shape | Owner |
|---|---|---|---|
| LLM serving (this repo) | `http://98dci4-gpu-0003:18800` | OpenAI **and** Anthropic native | `imas-ambix agent serve` |
| Text embeddings | `http://98dci4-gpu-0002:18765` | custom, **not** OpenAI-shaped | imas-codex |

The serve port is a `SiteConfig` default (`AMBIX_AGENT_PORT`), so a second
concurrent model is served on a **different port on the same host**. One vLLM
process serves one model, so the origin above is one model's catalog, not the
site's. `imas-ambix agent status` lists every live route with its port; `clive
--list` lists what a consumer can select.

**Service level:** the LLM endpoint is a group-sized, best-effort service for
the storage group. It runs on spare cards with no reserved floor, so a serve may
be cancelled when a training campaign needs the node. Admission limits and the
bounded queue are sized for a handful of concurrent consumers, not a published
cluster-wide wait. Widening the audience later requires a larger card budget and
a reserved floor; it does not require a different endpoint or routing design.

**LLM serving — paths that exist on the engine:**

```bash
curl http://98dci4-gpu-0003:18800/v1/models                  # catalog + ambix metadata
curl http://98dci4-gpu-0003:18800/v1/messages       -d @body  # Anthropic native, SSE-capable
curl http://98dci4-gpu-0003:18800/v1/messages/count_tokens -d @body
curl http://98dci4-gpu-0003:18800/v1/chat/completions -d @body # OpenAI
```

The Anthropic surface is complete — streaming emits real `thinking` content
blocks and tool definitions are accepted — which is why a consumer points
`ANTHROPIC_BASE_URL` straight at the origin and needs no translating proxy.

**The model id is the profile's `served_name`, not its slug:**

| Profile slug | Model id a client sends |
|---|---|
| `deepseek-v4-flash`, `deepseek-v4-flash-2x` | `deepseek-v4-flash` |
| `glm-5-2`, `glm-5-2` INT4 variants | `glm-5.2` |
| `glm-5-3`, `glm-5-3` FP8 variant | `glm-5.3` |
| `minimax-m2-7` | `minimax-m2.7` |
| `minimax-m3` | `minimax-m3` |

One release keeps one client-visible name across every card count, so a
consumer never has to know which node size answered.

**Embeddings — a different API, so do not send it OpenAI requests:**

```bash
curl http://98dci4-gpu-0002:18765/health                       # 200 when ready
curl http://98dci4-gpu-0002:18765/info                         # model, device, dims, stats
curl http://98dci4-gpu-0002:18765/embed -H 'Content-Type: application/json' \
     -d '{"texts":["plasma current"]}'                          # -> {"embeddings":[[...]]}
```

Paths are `/embed`, `/health`, `/info`, `/workers` — there is **no `/v1/`
prefix** and no `/v1/embeddings`. Qwen3-Embedding-0.6B on a P100, native
dimension 1024 reduced to 256 on output, L2-normalised.

**Auth:** a serve is open unless launched with `--auth` (§4). Where a key IS
enforced it is `AMBIX_AGENT_API_KEY` in the shared `agents/.env`, and the engine
accepts it **only** as `Authorization: Bearer` — never `x-api-key`.

## 3c. Concurrency on the local lane — the router bounds it, not the model

**The ceiling that agent fleets hit is `AdmissionLimits` in
`imas_ambix/agent/router.py`, and for most of 2026 it was six.** The shipped
defaults are `max_in_flight = 2` and `max_queued = 4`; **2 + 4 = 6** concurrent
requests, after which the router answers **HTTP 429** with
`consumer queue full; retry after N seconds`. Raised to **16 in flight and 48
queued** on 2026-09-06 after that arithmetic was traced to a fleet-wide worker
die-off. Set them at launch, never in code:

```bash
imas-ambix agent router --submit --port 18802 --max-in-flight 16 --max-queued 48
```

**The allowance is PER SOURCE IP — shared by every worker on one host, and
multiplied by the number of hosts.** Admission is keyed on
`_consumer_id(scope)`, which returns `scope["client"][0]`. Every clive worker
dispatched from one login node is *one consumer* against one allowance, whatever
project dispatched it — **but a second host gets its own full allowance.**
Measured 2026-09-06: two client IPs (1,284 requests from one, 26 from another),
and engine concurrency reached **17** against a configured 16, which is 16 from
the busy host plus 1 from the other. **The lane-wide in-flight ceiling is
therefore `max_in_flight × distinct client hosts`, not `max_in_flight`.** Check
before sizing:

```bash
grep -oE 'INFO: +[0-9.]+:' ambix-router-<job>.log | grep -oE '[0-9.]+' | sort -u
```

Three further consequences that cost a night to learn:

- **No project can see the binding quantity from its own ledger.** One session
  measured its own maximum at four simultaneous runs and concluded the six-figure
  could not apply, while another session's runs were consuming the same
  allowance. Count clive runs *host-wide* or the number is meaningless.
- **Sessions are not requests.** One turn issuing several parallel tool calls
  bursts a limit of two on its own, so refusals were observed with as few as two
  sessions live.
- **A reckon-side concurrency ceiling and this one bound the same resource in
  DIFFERENT UNITS, so setting both to the same number does not align them.**
  Reckon's per-backend ceiling counts **live runs**; this one counts
  **simultaneous requests**, and the ratio is variable — four agentic sessions
  were measured producing two concurrent requests, because a worker spends most
  of its wall clock between turns. A reckon ceiling of sixteen *runs* might
  therefore produce only eight simultaneous requests and silently throttle the
  lane to half its allowance while appearing to match. If both are set, the
  tighter wins invisibly and nobody can tell which. **Reckon's stays unset by
  default** — a number an operator cannot convert is a number they should not be
  invited to set — and where it is set it is a coarse bound on how much work one
  fleet may hold *open* against the lane, not a model of this queue.

**The conversion between the two units, measured 2026-09-06 — and it is not a
constant.** Joined samples across three fleets: 7-8 live clive runs produced 4-5
simultaneous requests (0.5-0.7), while 15 live runs coincided with 15 running
(near 1.0). An agentic worker spends most of its wall clock between turns, so
the ratio is low when the fleet is small and **rises toward 1.0 as it grows**,
because overlapping turns become likelier.

**Convert a ceiling at 1.0, never at the low-end figure.** The observed band is
0.5 to 1.0, and the conversion **shrinks exactly when it is being relied upon** —
the moment a fleet is large enough for the ceiling to matter is the moment a run
is worth close to a whole request. Sixteen in flight is therefore about sixteen
runs at saturation, not the 23-32 the low end implies. The band is measured; the
mechanism is inferred.

**Any single spot reading of concurrency is a draw from a distribution, not a
level.** Eight reads five seconds apart spanned **10 to 14** with the fleet
steady. Pair a series against a series; a synchronised pair of single values is
no better than two adjacent ones.

**A quiet endpoint is not evidence of headroom.** A deliberate pressure test on
2026-09-06 told three fleets to run hot and the engine never exceeded **five**
concurrent requests, with zero capacity waits, zero deferrals and zero
preemptions. It would be wrong to conclude that sixteen is comfortable: the
correct conclusion is that **not enough independent work existed to find out**.
The roster was not the constraint either — 174 members with 3 holding runs, so
171 free. What bound every fleet was **the width of its dependency graph**: the
number of ready nodes that do not contend for the same files. That is a property
of the plan, not of capacity, and no lifted limit moves it.

**Those two conclusions look identical in a graph of engine metrics and are
entirely different facts.** Before reporting headroom, establish that the offered
load was actually there — a lane-wide live-run count beside the engine samples,
at a shared wall-clock stamp.

**Sixteen is a working ceiling on a shared best-effort endpoint, not a target.**
The endpoint runs on spare cards with no reserved floor and competes with any
training campaign on the same node.

### The KV pool is a function of the profile, not a property of the cards

**Do not copy a pool figure between engines, and never quote one without its job
id.** Measured on two four-card serves of the same checkpoint on the same day:

| Job | `max_model_len` | mem fraction | KV memory | **Pool** | KV per token |
|---|---|---|---|---|---|
| 1262921 | 524,288 | 0.90 | 67.18 GiB | 1,313,935 | 54.9 KB |
| 1262952 | 1,048,576 | 0.92 | 70.01 GiB | **2,557,835** | 28.7 KB |

The pool nearly **doubled** while the memory behind it grew 4.2%, so per-token
KV cost halved. `kv_cache_dtype` was `fp8` on both, so the obvious explanation —
a precision change — is **ruled out**; the only other delta is `max_model_len`.

**The strongest clue is the ratio the engines report themselves.**
`kv_cache_max_concurrency` was **2.506** and **2.439** — near-constant across a
doubling of the window. That figure is pool over `max_model_len`, so a constant
means the reported pool tracks the declared window almost exactly. A physical
capacity that merely happened to change would not hold that ratio. This model's
hybrid compressed attention accounting per-token KV against the declared window
predicts precisely this; a draft-model reservation predicts the opposite sign.
**The mechanism is still not established and must not be asserted.**

**The decisive experiment costs one restart** and has not been run: bring an
engine up on the current checkpoint and memory fraction with `max_model_len`
back at 524,288. A pool near 1.3M isolates the window as the cause with every
other variable fixed; a pool near 2.56M means something else changed between the
two serves and the window is a coincidence.

**Name the assumption before dividing the pool by a session size.** Concurrency
figures derived that way are valid only because vLLM's paged attention allocates
KV per token *actually used* rather than per declared window. That holds here,
but it is the load-bearing premise under every sizing number in this section and
it was left unstated through two rounds of argument.

**The operational rule stands regardless: re-measure after any profile change,
because a limit sized to yesterday's pool silently stops being right.**

Read the live figure rather than a document:

```bash
curl -s http://98dci4-gpu-0003:18800/metrics | grep '^vllm:cache_config_info' \
  | tr ',' '\n' | grep -E 'kv_cache_size_tokens|kv_cache_max_concurrency'
```

At the ~83k tokens an agentic session was measured to hold, 2,557,835 divides
into roughly thirty concurrent sessions — so at six the hardware was never
close, and `max_num_seqs` is 1024.

### Diagnosing a dead local-lane worker

**`stderr.log` carries no status for any outcome.** Every clive run's stderr is
byte-identical boilerplate — banner, connectors warning, and a
`[claude-code:unrecognized_model]` line — whether the run succeeded or died.
**That last line is benign and appears on healthy runs; three sessions
misattributed deaths to it before measuring.** The status lives in the stream:

```bash
grep -c '"error_status":429' <run-dir>/stream.jsonl <run-dir>/resume-*.jsonl
```

**A 429 is recoverable until it is not.** The client honours `Retry-After`
exactly, then escalates to about 40 s, and gives up after **ten retries on a
single call** — roughly 200 s of patience — after which it emits a synthetic
assistant message and an error result, and in print mode the turn ending is the
process ending. Runs survive short bursts: nodes carrying seven and eighteen
retry records completed and committed real work. Reckon passes no retry
configuration, so that ten is not tunable from the dispatch side; the lever is
this router's limits.

**Two 429-and-`turn.completed` counts classify the death mechanically, before
you read anything else.** Measured across four dead runs, the separation was
exact:

| `429` count | `turn.completed` | Diagnosis | Does the router limit fix it? |
|---|---|---|---|
| > 0 | 0 | admission exhaustion, died mid-turn | **yes** |
| 0 | 1 | process ended with its turn, deliverable written, manifest missing | **no** |

```bash
for f in <run-dir>/stream.jsonl <run-dir>/resume-*.jsonl; do
  printf '%-28s 429=%s turn.completed=%s\n' "$(basename $f)" \
    "$(grep -c '"error_status":429' $f)" "$(grep -c 'turn.completed' $f)"
done
```

**Key on the structured field, never on the rejection sentence.** `api_retry`
and `"error_status":429` are the same records — counted 16/16, 8/8, 5/5 and 0/0
across six streams — so either works. But the literal text
`Request rejected (429)` / `consumer queue full` returned **zero across 29 real
429 records** on one session's runs while appearing as assistant turn text on
another's. A grep for the sentence will report no rejections on a run that had
twenty-nine.

**Only compare `turn.completed` across codex-format streams** — a claude-backend
run does not emit that record at all, so its `0/0` is a format artefact rather
than a diagnosis.

**The 429s are lane pressure, not a property of the node.** One node's original
stream carried zero; the 429s appeared only in its later resumes, when the
fleet was busiest.

**At least two distinct failures wear the label "process gone without a complete
manifest", and a fix aimed at one will appear to half-work.** One is 429
exhaustion mid-turn. The other is a worker whose stream contains
`turn.completed` with its deliverable fully written and only the manifest
missing — the process ending with its turn, one turn short, which has nothing to
do with admission control. That second shape has at least two measured causes of
its own: a manifest written to the node's declared report path instead of the
run directory, and a promotion that reads the stream before its writer has
finished. **If manifest-missing rates fall after a concurrency change, suspect
that the admission deaths were merely masking the other cause rather than that
it was fixed.**

## 4. Composable Agent CLI

The `imas-ambix agent` CLI manages LLM deployments via TOML model profiles:

```bash
imas-ambix agent list                          # List available profiles (marks any serving)
imas-ambix agent info kimi-k2-6               # Show profile details + memory budget
imas-ambix agent setup vllm                   # Install, then verify on the serving node
imas-ambix agent download kimi-k2-6           # Submit SLURM download job (sirius partition)
imas-ambix agent serve kimi-k2-6              # Submit SLURM serve job (betelgeuse partition)
imas-ambix agent serve kimi-k2-6 --dry-run    # Print script without submitting
imas-ambix agent serve glm-5-3 --gpus 4 --cpus 12 --port 18801   # coexist with another serve
imas-ambix agent serve glm-5-3 --auth         # Require a key; open endpoint is the DEFAULT
imas-ambix agent status                        # Jobs + connection block (URL, key, readiness)
imas-ambix agent key --rotate                 # Operator-only authenticated-backend maintenance
imas-ambix agent clive --deploy               # Generate+deploy the clive launcher (see §4a)
```

Adding a new model: create a TOML file in `imas_ambix/agent/profiles/<slug>.toml`.

**A serve is an OPEN endpoint by default; `--auth` opts in.** The cluster is the
authentication boundary, and the shared key is readable only inside the
`sdcc-imas_gpu` group, so a keyed endpoint locks the standalone launcher out of
the endpoint it exists to reach. Naming `--api-key` arms enforcement on its own;
asking for `--auth` with no resolvable key is a launch error, not a downgrade to
an open port. There is no `--no-auth` — it fails loudly.

**Cores do not follow cards, and `--gpus` no longer scales them.** A serve runs
one API server, one engine core and one worker per rank, each mostly blocked on
the device, so a full-GPU profile declares 12 cores at four cards and 16 at
eight — not the 30-core reservation ceiling. The exception is mechanism, not
preference: a KTransformers profile computes cold experts on the host, so
`glm-5-1`, `kimi-k2-6` and `mimo-v2-5-pro` legitimately keep 30. Before raising
any of them, read the live core ledger as §2 describes — **not**
`srun --test-only`, which is inert here — and use `--cpus` for a one-off. Taking
the ceiling is what leaves a co-running job pending on `Resources` while every
GPU it wants sits idle.

**Operator authority (binding):** setup, download, serve, key rotation, global
endpoint configuration, and deployment are operator work. Run those commands
only in a node explicitly assigned that authority. Documentation, review,
read-only verification, operator-unassigned work, and default shared-consumer
operation must not start, stop, restart, or cancel scheduler jobs or services.
`clive` has no scheduler or deployment authority. Its sole service exception is
explicit OpenRouter opt-in: after successful global model selection, it may
start an already-installed per-user proxy, but it does not install, stop,
restart, or cancel that service.

**Setup readiness contract:** `agent setup` submits a network-enabled install
job followed by a dependent runtime verification job on `betelgeuse`. The
verification runs after the producer allocation exits and checks the
consumer-visible interpreter, an exact per-run identity marker, and engine
package metadata. The environment is ready only when the runtime verification
job reaches `COMPLETED`; an install job reaching `COMPLETED` by itself is
not readiness. If verification fails, do not submit a serve job. Relocation is
rollback-free: setup writes a new engine-isolated home path and leaves the
project-backed and shared legacy environments untouched.

## 4a. Driving an interactive agent against the global catalog (`clive`)

`clive` ("CLI + live") is the standalone, shared consumer for **Claude Code**
(Anthropic Messages API) and the **OpenAI Codex CLI** (OpenAI API). An operator
sets `AMBIX_AGENT_GLOBAL_URL` when generating the launcher; the normalized
origin is embedded in the deployed script and is identical for every user.
Consumers do not need the repository, its virtual environment, operator CLI,
credentials, or scheduler access.

```bash
clive --list                                      # list the global catalog
clive "explain this repo"                         # select interactively if needed
clive --selector deepseek-v4-flash@4xh200 "..."  # exact release and topology
clive --model glm-5.3 "..."                       # exact native release id
clive --codex --model glm-5.3 "write a test"      # same origin through Codex
```

**Anonymous global discovery contract:**

- Each discovery run makes exactly one anonymous
  `GET $AMBIX_AGENT_GLOBAL_URL/v1/models` request to the embedded origin. It
  sends no `Authorization` header and never reads a user name, home directory,
  key file, repository, `AMBIX_AGENT_*` consumer override, profile, batch
  script, or SLURM state. It never runs `squeue`, `scontrol`, `sbatch`, or the
  operator CLI. Redirect responses are rejected rather than followed, so no
  second endpoint can become catalog authority.
- A successful, non-empty response is the complete availability authority.
  Clive uses each native model-card `id` as the release identity; it never
  substitutes a profile slug, local alias, filename-derived name, or URL from
  the response.
- The server-owned `ambix` metadata supplies checkpoint precision and runtime
  topology. `accelerator_family` is displayed with an explicit
  `accelerator_count` of 2, 4, 6, or 8, yielding `2×H200`, `4×H200`,
  `6×H200`, or `8×H200`. Clive validates those fields and does not infer
  them from scheduler state or client-side profiles.
- Selection changes only the native model id sent back to the same global
  origin. A future catalog item with `id: glm-5.3` is therefore immediately
  selectable by `--model glm-5.3`, without a Clive source or deployment change.
- The server-reported `max_model_len`, when present, becomes
  `CLAUDE_CODE_MAX_CONTEXT_TOKENS`; absence leaves the context unknown rather
  than guessed.
- Unreachable, non-successful, empty, malformed, duplicate, or incompletely
  annotated catalogs fail closed before either harness starts. Endpoint-down
  evidence is a service failure; Clive never falls back to scheduler discovery,
  a personal provider, a key file, or another endpoint.

After selection, Claude Code receives the global origin through
`ANTHROPIC_BASE_URL`; Codex receives the same origin with `/v1` appended. The
fixed client API-key values satisfy harness configuration only and are not
credentials. The global vLLM endpoint is keyless and both harnesses send
inference to the selected native release at that same origin.

**Explicit external-provider mode is separate from discovery.** A readable
personal key never changes the default route. `--openrouter` or
`CLIVE_OPENROUTER=1` is an explicit opt-in after the anonymous global catalog
has selected a native release; catalog failure still stops before the per-user
proxy is considered. Installing its artifacts is operator-authorized deployment
work. Once installed, explicit opt-in may start that per-user proxy; the proxy
is not part of default shared-consumer discovery.

**Sync discipline — repo is the source of truth (binding):** `imas-ambix agent
clive --deploy` generates the default shared artifact at
`/work/projects/imas_gpu/agents/clive` from `imas_ambix/agent/clive.py`.
`--destination PATH` writes the same generated launcher elsewhere.
`AMBIX_AGENT_GLOBAL_URL` is operator input at generation time, not a consumer
override. **NEVER hand-edit a deployed copy** — edit the generator, commit it,
and redeploy. Read-only verification may compare
`imas-ambix agent clive --print` byte-for-byte with a deployed copy; deployment
itself requires explicit operator authority.

- **Direct route, no tunnel.** Login and standard compute nodes route directly
  to `<gpu-node>:PORT` (verified 2026-06-25: login → `98dci4-gpu-0003:18800` =
  200). SSH `-L` port-forwarding to the compute node is **administratively
  prohibited** (`channel … open failed: administratively prohibited`), so the
  launchers use the direct URL and do not tunnel.
- **Operator vs consumer.** `imas-ambix agent` owns profiles, scheduler
  inspection, serving, key management for authenticated backends, and launcher
  deployment from the repository environment. `clive` owns only anonymous
  catalog discovery, selection, and harness launch from the shared GPFS script.
  Neither user identity nor operator state crosses that boundary.

## 5. Available Model Profiles

| Profile | Model | Engine | Checkpoint | Cards¹ | Context | License |
|---------|-------|--------|------------|--------|---------|---------|
| `kimi-k2-6` | Kimi-K2.6 (1T MoE) | KTransformers+SGLang | 555 GB | 4 | 262K | Modified MIT |
| `deepseek-v4-flash` | DeepSeek V4-Flash (284B MoE) | vLLM | 164 GB FP4+FP8 | 4 | 1M | MIT |
| `minimax-m2-7` | MiniMax M2.7 (~220B MoE) | SGLang | 220 GB FP8 | 4 | 200K | Custom |
| `glm-5-2` | GLM-5.2 (~744B MoE) | vLLM | 744 GB FP8 | 8 | 112K² | MIT |
| `glm-5-2-int4` | GLM-5.2 (~744B MoE) | vLLM | ~447 GiB INT4 AWQ | 4 | 128K | MIT |
| `glm-5-3` | GLM-5.3 (~744B MoE) | vLLM | 321 GiB INT4 / 704 GiB FP8 | 4 / 8 | 262K / 200K | unconfirmed³ |

¹ The card count is the profile's default sizing, not part of its identity —
`imas-ambix agent serve <slug> --gpus N` rescales tensor-parallel width, cores,
and memory at launch. See the naming convention below.
² Far below the native 1M, and **provisional**: the eight-card path has not been
launched against the installed engine, because doing so needs all eight cards and
is sequenced after the four-card serve. The 112K figure is the 224K that was
measured with an FP8 KV pool, halved for the bf16 entry size the installed engine
forces. Re-measure the startup KV line before quoting any ceiling.
³ **GLM-5.3 is downloaded and servable.** The four-card deployment uses the
downloaded 321 GiB community INT4 requant; the eight-card deployment replaces
it with the downloaded 704 GiB vendor FP8 release. Both resolve from the
`glm-5-3` profile and inherit client-visible name `glm-5.3`; their explicit
checkpoint precision is respectively `int4` and `fp8`. The FP8 release fills
the node, so INT4 is the deployment that can coexist with another serve. The
upstream license is still unconfirmed — treat output accordingly.

**Kimi-K2.6** — CPU-offloaded via KTransformers. 5 tok/s, best code quality (SWE 65.8%).
**DeepSeek V4-Flash** — Full GPU, FP4+FP8. 500–800 tok/s est., 1M context, MIT license.
**MiniMax M2.7** — Full GPU, FP8 native. 400–600 tok/s est., best agentic (GDPval-AA 1495).
**GLM-5.2** — Full GPU, vLLM, MTP speculative decoding. Two checkpoints for two
node sizes: FP8 on eight cards, INT4 AWQ on four.

### Variant naming convention (binding)

- **One release, one client-visible name.** Every variant of a release is served
  under the same `served_name` (`glm-5.2` — never `glm-5.2-fp8` or
  `glm-5.2-int4`), so a consumer never has to know which node size answered its
  request. `model.name` is likewise the plain release name (`GLM-5.2`) on every
  variant, exactly as the two-card DeepSeek variant reports `DeepSeek-V4-Flash`.
- **The card count never appears in a profile slug, a display name, or a SLURM
  job name.** It is a launch-time flag — `imas-ambix agent serve <slug> --gpus N`
  — which rescales tensor-parallel width, cores, and memory off the base profile.
  Two DeepSeek cards are served from the plain `deepseek-v4-flash` slug that way,
  and that is the pattern to follow; a card-count-suffixed profile is only a thin
  topology override of its base, not a distinct model.
- **Where two top-level profiles are genuinely needed, the slug names the
  checkpoint precision** (`glm-5-2-int4`), because the checkpoint is the real
  differentiator. A single profile may instead declare a card-specific
  checkpoint override: `glm-5-3` uses INT4 at four cards and replaces it with
  vendor FP8 at eight. Both patterns name checkpoint differences rather than
  topology; DeepSeek needs neither because both sizes load the same checkpoint.

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
- `deepseek-v4-flash-2x` — 2×H200, TP=2, ~100 tok/s single / ~114 tok/s 8-way
  concurrent. A topology override of the base profile that also raises the
  context cap to 1M and halves the sequence cap. The card count in its slug is
  not a pattern to copy — a plain two-card serve is `--gpus 2` on the base slug.

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

### GLM-5.2 on eight cards (FP8)

**Profile:** `glm-5-2` — vLLM native, full-GPU, **TP=8 across all 8×H200**, no
CPU offload. The ~744 GB FP8 checkpoint needs the whole 8×140 GB of VRAM, so this
serve fills the node and no other GPU job coexists with it. vLLM exposes the
OpenAI API and the **Anthropic Messages API** (`/v1/messages`) on the same port,
so `clive` (§4a) drives it directly.
**Model path:** `/work/projects/imas_gpu/agents/glm-5-2/model`

Deployment facts — eight-card FP8 only. The memory behaviour below was measured
2026-06-25/26 at `mem_fraction 0.86` and verified with a real decode and a live
Claude Code request, but on an earlier engine with an FP8 KV pool; the KV note
below governs what still holds:

- **Two HBM constraints fight, and the ceiling lands far below the native
  1M.** The FP8 weights, the MTP draft model, and the sparse-MLA decode
  kernel — which allocates **~8 GiB/card of scratch lazily, on the first real
  decode** — leave little room for KV. At `mem_fraction 0.95` the KV pool reaches
  ~24 GiB (256K fits) but the first decode finds only ~2.4 GiB of the 8 GiB it
  needs and the engine dies; at 0.86 the decode scratch is ample and the KV pool
  is 13.63 GiB, so a 256K pool (needs 15.2 GiB) cannot initialise while 224K
  (`max_total_tokens=229376`, ~13.3 GiB) does. **Startup and KV-init passing are
  not proof that inference fits** — the scratch is lazy, so always validate with
  a real generation. The native 1M context needs Blackwell cards.
- **CUDA graphs are on and carry the decode.** At 0.86 capture completes in ~70 s
  for 2.84 GiB. Capture stalls at `mem_fraction` 0.90/0.95 are the same memory
  pressure that kills the first decode, not a TP=8-plus-MTP defect: one cause,
  two symptoms. `disable_cuda_graph` is an SGLang-only flag and vLLM ignores it.
- **MTP speculative decoding** (`--speculative-config.method mtp`,
  `num_speculative_tokens 5`) is this model's headline throughput feature and is
  enabled in the profile.
- **Throughput:** single request **~33–37 tok/s** decode; aggregate scales with
  concurrency — ~210 tok/s @ 8 concurrent, ~457 @ 16, ~680 @ 32, **~974 @ 48**.
  Healthy concurrency is **~16–32 simultaneous requests** (per-request rate holds
  ~25–35 tok/s through n=16, then trades latency for aggregate). The scheduler
  cap (`max_num_seqs=1024`) is not the bottleneck; the ~13.6 GiB KV pool is —
  only one full-context request fits in it, but typical agent requests
  use far less context, so dozens run concurrently. **Low concurrency (n=1–4) is
  MTP-overhead-dominated** and worse per-aggregate than n=8+; the model shines
  under batch load.
- **Reasoning effort is wired end to end** (Claude Code `effort` → vLLM
  `reasoning_effort` → GLM template `enable_thinking`), but the chat template
  offers effectively two levels — `high`, or `max` for everything else.
- **Engine floor:** vLLM ≥ 0.23.0 for `GlmMoeDsaForCausalLM`, the glm45/glm47
  parsers, and MTP, alongside transformers 5.x and flashinfer. The vLLM engine
  environment is shared across profiles, so the higher four-card INT4 floor below
  governs whichever version is actually installed.
- **This profile now declares `bfloat16` KV, and its context ceiling is
  unvalidated.** The KV-dtype constraint recorded for the four-card path below is
  a property of the same sparse attention, and `"fp8"` resolves to the very
  `fp8_e4m3` those backends reject, so an FP8 KV entry cannot start on the
  installed engine at all. Because a bf16 entry costs twice the bytes per token,
  the profile is set to 112K — half the previously measured 224K — and both
  numbers are engine-dependent. Treat the ceiling as provisional until a real
  decode confirms it; the eight-card launch needs the whole node, so it is
  sequenced after the four-card serve is confirmed.
- **FP8 is the only practical precision on H200 — FP4 is Blackwell-only.**
  Native FP4 tensor cores do not exist on Hopper (H200, CC 9.0); they arrived
  with Blackwell (B200/GB200/RTX PRO 6000). A `GLM-5.2-NVFP4` checkpoint exists
  (~459 GB vs 744 GB FP8 — would free ~285 GB for KV), but it **requires
  Blackwell**: on H200 vLLM can only load it through the Marlin software
  fallback, which is *unaccelerated* (no FP4 silicon, so likely slower than FP8,
  not faster) and *correctness-buggy* (NVFP4 Marlin assumes scales ≥0 and
  encounters negative ones). The tightness here is the 744 GB footprint plus MTP,
  not a quantization choice. The lever that frees headroom on H200 is **dropping
  MTP** (reclaims the draft model's weights and scratch for KV/context), trading
  its throughput benefit; FP4 becomes viable only on Blackwell cards. (NVFP4
  details: <https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/>;
  H200 = Hopper/FP8, B200 = Blackwell/FP4.)

**Deploy:**
```bash
imas-ambix agent download glm-5-2   # ~744 GB, run from sirius
imas-ambix agent serve glm-5-2      # eight cards; ~30 min weight load (GPFS-cold)
clive "..."                          # drive Claude Code against it (§4a)
```

### GLM-5.2 on four cards (INT4 AWQ)

**Profile:** `glm-5-2-int4` — vLLM native, TP=4, INT4 AWQ
(compressed-tensors/Marlin), 128K context (`max_total_tokens=131072`) at
`mem_fraction_static 0.90`.
**Weights:** `cyankiwi/GLM-5.2-AWQ-INT4` — 83 shards, ~447 GiB, at
`/work/projects/imas_gpu/agents/glm-5-2-int4/model`.
**MTP draft:** `CosmicRaisins/GLM-5.2-MTP-INT4` (~5.5 GiB, 3122 tensors) in the
`mtp-draft/` subdirectory of that model directory. AWQ requantisation damages the
integrated MTP head, so the draft is loaded as a separate model.
**Coexistence:** this serve holds **4 cards on port 18801** while the two-card
DeepSeek serve holds 18800 — 6 of the 8 cards in use, both endpoints live.

Deployment facts — four-card INT4 only (measured 2026-08-26):

- **The KV cache must be `bfloat16`.** GLM-5.2's sparse (DSA) attention has no
  FP8 KV kernel in vLLM: every candidate attention backend rejects `fp8_e4m3`,
  each reporting either that sparse attention is unsupported or that the KV dtype
  is. The FP8-KV throughput gain widely quoted for this model is an SGLang result
  and does not transfer to vLLM. It costs KV capacity — bf16 holds half the
  tokens per byte — and is worth re-testing when a later vLLM ships the kernel.
- **The accepted spelling is `bfloat16`.** `bf16` is rejected at argument parse.
- **vLLM 0.27.1 requires torch 2.13.0.** The wheel is compiled against that ABI;
  against torch 2.11.0 the bundled `deep_gemm` extension fails to import on an
  undefined `c10` symbol, and because the sparse-attention indexer depends on
  that extension the server refuses to start with "Sparse Attention Indexer CUDA
  op requires DeepGEMM support". `deep_gemm` additionally needs the CUDA 13
  runtime (`libcudart.so.13`, `libnvrtc.so.13`), which the environment carries
  under `site-packages/nvidia/cu13/lib` and which the generated serve script puts
  on `LD_LIBRARY_PATH` through its `nvidia/*/lib` glob. torchaudio and
  torchvision are not needed for serving and are not pinned.
- **MTP speculative decoding needs vLLM ≥ 0.27.0.** The community INT4 draft
  stores experts individually
  (`model.layers.78.mlp.experts.N.{gate,up,down}_proj.weight_packed`); a loader
  that expects a single fused `routed_experts` tensor raises a missing-key error
  on it. The 0.27.x loader rewrites the layer prefix to `mtp_block.` and fuses
  the per-expert weights itself, so this layout is the supported one.
- **The scheduler sequence cap is 64** — above the benchmark's maximum
  concurrency of 32, and low enough not to inflate CUDA-graph capture against a
  bf16 KV pool.

**Deploy:**
```bash
imas-ambix agent download glm-5-2-int4                  # ~447 GiB + draft, from sirius
imas-ambix agent serve glm-5-2-int4 --port 18801        # four cards, beside DeepSeek
```

### Measuring serving throughput

Serving throughput is recorded with `imas-ambix agent bench`; saved runs land
under `~/.local/share/ambix/bench/`. A run is comparable only when it carries its
provenance — the serving configuration it ran against and the engine version —
and a draft-token acceptance rate, because a silently degraded speculator costs
throughput without raising an error. Comparison tooling over the saved runs is
still being built; read its flags from its own `--help` rather than from here.

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
