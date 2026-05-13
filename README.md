# IMAS Ambix ⚗️

**Fusion World Model — distilling experimental data into physics-informed generative models**

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
See [agents.md](agents.md) for hardware specs and deployment details.

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
curl -H "Authorization: Bearer your-secret-key" http://98dci4-gpu-0003:18800/v1/models
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

**Current deployments:**
- **Kimi-K2.6** — 1T-param MoE (32B activated), KTransformers+SGLang engine,
  4×H200 GPUs, ~18K context (auto-fitted to VRAM), OpenAI-compatible API at
  `http://98dci4-gpu-0003:18800/v1`
  - Decode throughput: ~4–5 tok/s
  - Reasoning model with chain-of-thought (use `max_tokens≥1024` for complex prompts)
  - Access from ITER login nodes or compute nodes via SLURM

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
