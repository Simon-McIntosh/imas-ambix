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

## Python environment (binding, repo-wide)

Policy lives in [`~/.agents/AGENTS.md`](~/.agents/AGENTS.md) "Development
Environment": one virtualenv per repository, at its root; agents keep it
current; every dependency change is durable and `pyproject.toml`-backed. The
repo-specific parts:

**Main checkout** — the project environment is
`/home/ITER/mcintos/Code/imas-ambix/.venv`. Run plain `uv run <cmd>`, which
syncs that environment first; sync it directly with `uv sync` when it is stale,
incomplete, or absent. Bringing it up to date is agent work, not a blocker to
hand back — report only if the sync itself fails, with its output. Change
dependencies with `uv add` / `uv remove` (or by editing `pyproject.toml`) and
commit `pyproject.toml` together with `uv.lock`; never `pip install` into the
environment.

**Detached worktree** — point back at the main checkout's environment; a
worktree must never build its own. One environment here is ~70k filesystem
entries and ~1.8 GiB on GPFS, so a fleet each building one trips the storage
alert. An inherited `VIRTUAL_ENV` does not achieve the reuse (uv warns the path
does not match the project and creates `.venv` anyway):

```bash
UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-ambix/.venv PYTHONPATH="$PWD" \
  uv run --no-sync pytest <targets>
```

`--no-sync` belongs to the worktree case only, because that environment is
shared with the main checkout and any concurrent workers. It is not how to run
in the main checkout.

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

## GPU server, model serving & the agent CLI

All guidance for the **GPU server (8×H200), SLURM access, the `imas-ambix
agent` CLI, model profiles, serving (GLM-5.2, DeepSeek-V4-Flash, …), and the
`clive` launcher** lives next to that code, in
[`imas_ambix/agent/AGENTS.md`](imas_ambix/agent/AGENTS.md). Agentic tools read
it automatically when working under `imas_ambix/agent/`. Look there for:
hardware specs, the SLURM submission pattern and CPU/GPU sizing, storage paths,
`imas-ambix agent {list,info,download,serve,status,key,clive}`, per-model
deployment notes and memory budgets, and the `clive` interactive-harness setup.

## Physics-spine benchmark (perf + quality, incl. H200/GPU policy)

The equilibrium-engine benchmark — how to run the frozen CPU metric on SLURM,
the GPU-capable inventory, and the **binding H200 capability-demonstration
policy** (reservation, CUDA-jaxlib install, `scripts/fsa_gpu_capability.py`) —
lives next to that code, in
[`imas_ambix/spine_bench/AGENTS.md`](imas_ambix/spine_bench/AGENTS.md). Read it
before running or extending any benchmark.
