# Connecting to the ITER SDCC LLM Endpoints

This guide is for **users of the ITER SDCC GPU cluster** who want to run
agent harnesses (Hermes, OpenHands, CrewAI, AutoGen, etc.) against the
LLM endpoints hosted on the shared betelgeuse GPU node.

If you are managing or extending the deployments themselves, see
[../AGENTS.md](../AGENTS.md) instead.

## 1. What is served

A single OpenAI-compatible HTTP server runs on the GPU node and
exposes one model at a time:

| Field          | Value                                       |
|----------------|---------------------------------------------|
| Host           | `98dci4-gpu-0003`                           |
| Port           | `18800`                                     |
| Base URL       | `http://98dci4-gpu-0003:18800/v1`           |
| Protocol       | OpenAI Chat Completions (`/v1/chat/completions`) |
| Auth           | `Authorization: Bearer <key>` (required)    |
| Health probe   | `GET /v1/models` (also requires the key)    |

The currently-served model can change. Always start from
`GET /v1/models` to find the `id` to pass in your request body — do not
hard-code a model slug.

## 2. Prerequisites

You need:

1. An ITER SSO account with access to the SDCC HPC cluster.
2. Membership in the `sdcc-imas_gpu` group (request via your local
   service desk if you do not already have it; without it the shared
   `/work/projects/imas_gpu/` tree is invisible).
3. SSH access to an SDCC **login** node (`io-ls-hpc.iter.org` or
   similar — check the SDCC onboarding page).
4. **An API key for the endpoint** — see §3.

You do **not** need GPU access or SLURM allocation to use the endpoint;
those are only needed if you intend to *operate* the server.

## 3. Getting an API key

> ⚠️ **The API key is not committed to this repository, and must not be
> shared in chat, tickets, or screenshots.**

Request the current key out-of-band from one of the deployment
maintainers (see `git log AGENTS.md` for active maintainers, or ask in
the ITER SDCC Mattermost / Teams channel for the GPU server). They
will pass it to you via a secure channel.

When the key is rotated (which can happen at any time), your old key
will start returning `401 Unauthorized` and you will need to request
the new one — there is no automatic refresh.

## 4. Storing the key on your machine

Use a `~/.hermes/.env` file (or `~/.config/<tool>/.env`, depending on
the agent harness). Permissions **must** be `0600` (owner read/write
only); the file is unreadable to other users on shared systems.

```bash
mkdir -p ~/.hermes
touch ~/.hermes/.env
chmod 600 ~/.hermes/.env   # MANDATORY — without this, anyone on the
                           # login node who can read your homedir can
                           # steal the key
```

Edit `~/.hermes/.env` and add:

```env
# Endpoint
OPENAI_BASE_URL=http://98dci4-gpu-0003:18800/v1

# API key — obtained out-of-band; never commit this file
OPENAI_API_KEY=<paste-the-key-you-were-given>

# Optional: lock provider to the local endpoint so harnesses don't
# silently route to public OpenAI / Anthropic when an SDK auto-detects
# a different env var.
HERMES_INFERENCE_PROVIDER=openai-compat
HERMES_DISABLE_LAZY_INSTALLS=1
```

Verify the permissions are correct:

```bash
$ stat -c '%a %n' ~/.hermes/.env
600 /home/ITER/<you>/.hermes/.env
```

Anything other than `600` is a finding — fix it before continuing:

```bash
chmod 600 ~/.hermes/.env
```

### Why 0600?

The SDCC home directory is mounted on every login and compute node.
Any process running as a different user on the same node — including
shared service accounts — can `cat` your files if they are world- or
group-readable. The API key acts as a bearer token for the model
server, and a leaked key allows anyone on the ITER network to consume
GPU time on your behalf.

If you ever suspect a key has leaked (committed to git, pasted into a
chat, printed in a build log), tell a maintainer immediately so they
can rotate.

## 5. Reaching the endpoint

### 5a. From a login node — direct (recommended)

The GPU node is on the internal network and reachable from any SDCC
login or compute node. From a login shell:

```bash
source ~/.hermes/.env
curl -s -H "Authorization: Bearer $OPENAI_API_KEY" \
  "$OPENAI_BASE_URL/models" | jq .
```

Expected response: a JSON `data` array with one entry per served
model. The current `id` (e.g. `deepseek-v4-flash`) is what you pass as
the `model` field in chat requests.

### 5b. From your laptop — SSH tunnel

If your harness runs on your own machine rather than on an SDCC node,
forward the port through your SSH session:

```bash
ssh -N -L 18800:98dci4-gpu-0003:18800 <your-user>@<sdcc-login-host>
```

Leave that running, then point your client at
`http://localhost:18800/v1` instead.

### 5c. A first chat call

```bash
source ~/.hermes/.env
curl -s -H "Authorization: Bearer $OPENAI_API_KEY" \
       -H "Content-Type: application/json" \
       -d '{
             "model": "deepseek-v4-flash",
             "messages": [
               {"role": "user", "content": "Reply with the single word OK."}
             ],
             "max_tokens": 4
           }' \
       "$OPENAI_BASE_URL/chat/completions" | jq .
```

If you see `401 Unauthorized`, the key is wrong, missing, or rotated.
If you see a network error, check that you are on the ITER network /
VPN and that the SSH tunnel (if any) is still up.

## 6. Wiring it into an agent harness

Most harnesses pick up `OPENAI_BASE_URL` + `OPENAI_API_KEY`
automatically from the environment. Sourcing `~/.hermes/.env` in your
shell profile is the simplest setup:

```bash
# Append to ~/.bashrc (or ~/.zshrc)
[ -f "$HOME/.hermes/.env" ] && set -a && . "$HOME/.hermes/.env" && set +a
```

Per-harness notes:

- **Hermes Agent (`hermes` / `hermes-cli`)** — reads
  `~/.hermes/config.yaml`. Point `model.base_url` and
  `delegation.base_url` at `http://98dci4-gpu-0003:18800/v1`. The CLI
  re-reads `~/.hermes/.env` on each launch.
- **OpenHands (`oh`, `oh-yolo`)** — same env vars; set
  `LLM_MODEL=openai/<served-model-id>`.
- **CrewAI / AutoGen / MAF** — pass an explicit
  `OpenAIChatCompletion` client constructed from
  `os.environ["OPENAI_BASE_URL"]` and `os.environ["OPENAI_API_KEY"]`.
- **Python SDK** —

  ```python
  from openai import OpenAI
  client = OpenAI()   # picks up OPENAI_BASE_URL + OPENAI_API_KEY
  ```

## 7. Available models

Check `/v1/models` for the live answer. The current deployment
catalogue (see [../AGENTS.md](../AGENTS.md) §5 for the full table) is:

| Slug                | Approx. size | Engine        | Context | Notes |
|---------------------|--------------|---------------|---------|-------|
| `deepseek-v4-flash` | 164 GB       | vLLM          | 1M (262K served) | Default — fast decode, FP4+FP8 mixed |
| `kimi-k2-6`         | 555 GB       | KTransformers | 256K    | 1T MoE, CPU+GPU offload, slower decode |
| `minimax-m2-7`      | 220 GB       | SGLang        | 200K    | 220B MoE, agentic-tuned, FP8 |

Only **one** of these is served at any given moment — switching models
is an operator action (see AGENTS.md §4 — `ambix agent serve <slug>`).
The current value is whatever `/v1/models` returns.

## 8. Throughput expectations

Single-stream decode on 4×H200 (Group A reservation), measured via
`ambix agent bench` (throughput / prefill / tools / reasoning) on
2026-05-15:

| Model               | Decode TPS | TTFT 16K | Tools pass | Notes |
|---------------------|-----------:|---------:|-----------:|-------|
| `deepseek-v4-flash` |        111 |   586 ms |        4/4 | vLLM, FP4 experts + FP8 dense, KV cache fp8 |
| `minimax-m2-7`      |         17 |   406 ms |        4/4 | SGLang, FP8 native, CUDA graphs disabled, fp8 GEMM via triton (see profile) |
| `kimi-k2-6`         |          4 |  53.2 s  |        2/4 | KTransformers CPU+GPU offload, INT4 experts; prefill is CPU-bound |

Reading those numbers:

- `deepseek-v4-flash` is **~6× faster** than `minimax-m2-7` and
  **~30× faster** than `kimi-k2-6` on this hardware. Pick it unless
  you have a specific reason not to.
- `minimax-m2-7` decode is steady at 17 TPS across output lengths
  128 → 2048; prefill TTFT scales sub-linearly (~400 ms across 1K-16K
  prompts thanks to FlashInfer chunked prefill).
- `kimi-k2-6` runs at ~4 TPS but every additional token of *context*
  costs more than every additional token of *output*: TTFT was
  ~7.7 s for 1K context, ~17 s for 4K, ~53 s for 16K, ~210 s for 64K.
  Cold-expert fetches from CPU dominate. Best used for short prompts
  where reasoning quality matters more than latency.
- Tool calling: flash + minimax handle parallel tool calls and
  structured arg dictionaries reliably; kimi only passed the
  single-tool and no-tool cases (2/4) — the model emitted prose
  instead of tool calls on the parallel and structured tests.

For multi-worker / concurrent throughput, run
`ambix agent bench --category concurrency <slug>` against the live
endpoint; with vLLM continuous batching the aggregate TPS scales
roughly linearly to 8–16 workers before tailing off.

Raw JSON reports are saved under
`~/.local/share/ambix/bench/<slug>_<timestamp>.json`.

## 9. Things that will save you a ticket

- The endpoint **requires** auth — even `GET /v1/models` returns 401
  without the bearer token. Don't be fooled by a server response of
  "Unauthorized" into thinking the server is down.
- The model `id` you pass in the chat payload must match what
  `/v1/models` returns *right now*. After a model swap, your hard-coded
  slug will start returning 404 on completions even though `/v1/models`
  works.
- The GPU node has **no outbound network** — if you are SSH'd in and
  trying to `pip install` something to test, it will hang. Install on a
  login or `sirius` node first.
- `~/.hermes/.env` mode 600 is mandatory. CI hooks may eventually start
  rejecting commits that contain anything matching the key shape;
  treat the key like a password.
- The default 262K-token context is large but not free; passing the
  whole `IMAS Data Dictionary` as a single prompt will burn through
  GPU memory before it burns through your patience. Summarise first.
- The endpoint serves *one* model. If a colleague is benchmarking, you
  may see a different model from what you started with — check
  `/v1/models` if results look off.

## 10. Reporting problems

- **Slow / hung requests, 5xx errors, or model swap mistakes** —
  message a maintainer; check `squeue -p betelgeuse` first to see if
  the serve job is even running.
- **401s after a known-good key** — the key has been rotated; request
  the new one.
- **Suspected key leak** — request rotation **immediately** (do not
  wait), then audit `git log -p` and any chat history for exposure.

The deployment code lives at
<https://github.com/iterorganization/imas-ambix>; bugs in the harness
itself or model profiles can go through GitHub issues.
