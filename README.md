# IMAS Ambix ⚗️

[![Plans](https://img.shields.io/badge/plans-dashboard-2563eb?logo=github)](https://simon-mcintosh.github.io/imas-ambix/)

**Fusion World Model — distilling experimental data into physics-informed generative models**

> 📋 **Plans dashboard.** Project plans, strategy, v0 runway and decisions —
> plus non-plan docs (RCAs, tickets, explainers) — are tracked as **HTML**
> under [`docs/`](docs/), published via GitHub Pages at
> <https://simon-mcintosh.github.io/imas-ambix/>.  State lives in each page's
> HTML island (`<meta name="plan-*">` + `data-reckon` sections); there are no
> per-plan sidecar JSON files.  Authoring goes through the **reckon** skill set
> (`reckon-create` / `reckon-edit` / `reckon-ship` / `reckon-status`) — see
> [`AGENTS.md`](AGENTS.md) "Plans & docs" and the architecture in
> [`~/Code/reckon/AGENTS.md`](https://github.com/Simon-McIntosh/reckon).

Ambix is a machine learning framework for training generative world models on
tokamak experimental data. It distils data access patterns and IMAS mappings
discovered by [imas-codex](https://github.com/iterorganization/imas-codex) into
a unified training corpus, then trains transformer-based models capable of
predicting plasma evolution from control inputs.

## Vision

A Fusion World Model trained on experimental data from ITER's partner tokamaks
(JET, TCV, JT-60SA, ASDEX Upgrade, DIII-D) that can:

- **Pre-play planned pulses** — generate predicted plasma evolution given control
  waveforms, before execution on the real machine
- **Validate physics assumptions** — compare predicted plasma response against
  expectations in silico
- **Assess disruption susceptibility** — identify trajectories passing through
  regions associated with disruption precursors
- **Explore alternative scenarios** — modify control waveforms and compare
  predicted outcomes without consuming machine time

The underlying approach follows Microsoft's WHAM (World and Human Action Model)
architecture: given the current plasma state (diagnostic measurements at time *t*)
and control actions (coil currents, gas injection, heating power), predict the
plasma state at time *t+1*.

The end-to-end plan from raw FAIR-MAST data through tokenization, training, and
the wide-angle-camera forward-prediction demo lives under
[`docs/`](docs/README.md) — start with
[`docs/STRATEGY.md`](docs/STRATEGY.md) for the vision and roadmap.

## Architecture

```
imas-codex (discovery)          imas-ambix (distillation & training)
┌─────────────────────┐         ┌─────────────────────────────────────┐
│ Federated Fusion     │         │                                     │
│ Knowledge Graph      │────────▶│  Distilled Data Patterns            │
│ (Neo4j artifact)     │         │  ├─ IMAS Mappings (source→target)   │
└─────────────────────┘         │  ├─ Signal Metadata                 │
                                │  └─ Coordinate Transforms           │
                                │                                     │
                                │  Training Pipeline                  │
                                │  ├─ Data Loaders (HDF5/MDSplus)     │
                                │  ├─ Tokenization (state→tokens)     │
                                │  ├─ Model (transformer)             │
                                │  └─ Evaluation (physics metrics)    │
                                └─────────────────────────────────────┘
                                          │
                                          ▼
                                ┌─────────────────────┐
                                │  GPU Server          │
                                │  98dci4-gpu-0003     │
                                │  4× NVIDIA H200      │
                                │  564 GB VRAM         │
                                └─────────────────────┘
```

## Relationship to imas-codex

| Project | Role | Output |
|---------|------|--------|
| **imas-codex** | Discovery & mapping | Federated Fusion Knowledge Graph |
| **imas-ambix** | Distillation & training | Fusion World Model weights |

Codex discovers *what data exists* and *how it maps to IMAS*. Ambix consumes
those mappings to build unified training datasets and train generative models.

## Infrastructure

Training targets the ITER Science Division GPU server:
- **4× NVIDIA H200** (141 GB HBM3e each, 564 GB total)
- **NVLink mesh** (900 GB/s per GPU)
- **FP8 Tensor Cores** (15,832 TFLOPS aggregate)

Estimated training corpus: ~26 billion state transitions from partner tokamaks.

## Quick Start

```bash
# Install with uv (development mode)
uv sync --dev

# Run CLI
ambix status

# Run tests
uv run pytest
```

## Agent Serving

The `ambix agent` CLI manages LLM deployments on the ITER SDCC GPU cluster.
See [AGENTS.md](AGENTS.md) for hardware specs and deployment details,
and [docs/cluster-usage.md](docs/cluster-usage.md) for end-user
connection instructions (env setup, API key handling, harness wiring).

```bash
# List available model profiles
ambix agent list

# Show profile details
ambix agent info kimi-k2-6

# Download model weights (submits SLURM job to sirius partition)
ambix agent download kimi-k2-6

# Serve model (submits SLURM job to betelgeuse GPU partition)
ambix agent serve kimi-k2-6

# Serve with API key authentication
ambix agent serve deepseek-v4-flash --api-key "your-secret-key"

# Preview generated script without submitting
ambix agent serve kimi-k2-6 --dry-run

# Check running jobs
ambix agent status
```

### API Key Authentication

The `--api-key` flag protects `/v1/*` endpoints on the model server.  Health
and metrics endpoints remain open for monitoring.

```bash
# Generate a key
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Serve with key
ambix agent serve deepseek-v4-flash --api-key "$(cat ~/.ambix-api-key)"

# Or via environment variable
export AMBIX_AGENT_API_KEY="your-secret-key"
ambix agent serve deepseek-v4-flash

# Client-side: pass key in Authorization header
# (endpoint URL is exchanged over Teams — see docs/getting-started.md)
curl -H "Authorization: Bearer $OPENAI_API_KEY" "$OPENAI_BASE_URL/models"
```

### Hermes Agent Harness

The [Hermes Agent](https://github.com/NousResearch/hermes-agent) harness
provides an autonomous agent runtime that connects to ambix-served models.
It supports depth-2 agent swarms with up to 64 concurrent leaf agents.

**Installation and docs**: `/work/projects/imas_gpu/tools/hermes-agent/`

Quick start:
```bash
# Add to ~/.bashrc (one time)
alias hermes='/work/projects/imas_gpu/tools/hermes-agent/bin/hermes'

# Set up per-user config (one time)
mkdir -p ~/.hermes
cp /work/projects/imas_gpu/tools/hermes-agent/templates/config.yaml.template ~/.hermes/config.yaml
cp /work/projects/imas_gpu/tools/hermes-agent/templates/.env.template ~/.hermes/.env
chmod 600 ~/.hermes/config.yaml ~/.hermes/.env
# Edit ~/.hermes/.env — set TERMINAL_SSH_USER to your username

# Run
hermes
```

**Current deployment:**
- **DeepSeek V4-Flash** — 284B MoE / 13B activated, FP4 experts + FP8
  dense, vLLM on 4×H200, 262K served context.
  - Decode throughput: ~110 tok/s
  - OpenAI-compatible Chat Completions endpoint
  - Endpoint URL + API key exchanged over Microsoft Teams with a
    cluster maintainer — see [docs/getting-started.md](docs/getting-started.md).

## Development

```bash
# Create virtual environment with preferred Python
uv venv --python 3.14

# Install in development mode
uv sync --dev

# Lint
uv run ruff check src/ tests/

# Type check
uv run mypy src/
```

## Licence

LGPL-3.0-or-later — see [LICENSE](LICENSE).

Copyright © 2026 ITER Organization.
