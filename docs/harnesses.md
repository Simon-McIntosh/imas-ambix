# Agent harnesses available on the ITER SDCC GPU cluster

Ten LLM agent harnesses are installed under `/work/projects/imas_gpu/tools/`,
all wired to the locally-served **DeepSeek V4-Flash** endpoint by default.
Membership in `sdcc-imas_gpu` is required; the alias prefix
`sg sdcc-imas_gpu -c "..."` handles that automatically.

This document is the **reference**. For day-one setup, see
[`getting-started.md`](./getting-started.md).

> The endpoint URL and the API key are not in this document.
> Request both via Microsoft Teams from a cluster maintainer (channel
> *SDCC GPU / Ambix*). The wrapper scripts read the key from the
> group-shared `/work/projects/imas_gpu/agents/.env` (mode 600) at
> launch — once you are in the group, you do not need to handle
> the key directly for any of these harnesses.

---

## Quick comparison

Trial: identical "create `palindrome.py` + `test_palindrome.py` with five
pytest cases, run pytest" prompt; clean tmpdir each run; same model
(`deepseek-v4-flash`). Wall time is end-to-end including tool loop.

| Harness     | Wall  | Tests | Agent teams | One-line headless |
|-------------|------:|------:|:-----------:|-------------------|
| **OpenCode**   | 7.5 s | 5/5  | ✗ | `opencode run "..."` |
| **Continue (cn)** | 10 s  | 5/5  | ✗ | `cn -p --auto "..."` |
| **Aider**      | 13 s  | 5/5  | ✗ | `aider --message "..."` (needs `--file` for new files) |
| **Goose**      | 15 s  | 5/5  | ✗ | `goose run --no-session --quiet -t "..."` |
| **Kimi-CLI**   | 18 s  | 5/5  | ✗ | `kimi --print --yolo --prompt "..."` |
| **Hermes**     | 21 s  | 5/5  | ✓ swarms | `hermes-cli -z "..." --yolo` (default = SSH sandbox) |
| **OpenHands**  | 36 s  | 5/5  | ✗ | `oh --headless --always-approve --exit-without-confirmation -t "..."` |
| **AutoGen**    | 7 s   | (team task) | ✓ RoundRobin / Selector / Swarm | python script using `autogen` venv |
| **CrewAI**     | (workflow) | n/a | ✓ role-based crews | python project scaffolding via `crew` |
| **MAF**        | (workflow) | n/a | ✓ Microsoft Agent Framework | `maf path/to/script.py` or `maf-ui` |

All 10 produced semantically equivalent code and all interactive
harnesses passed 5/5 on first run after their respective configuration
nudges (see per-harness sections).

---

## Per-harness reference

For every harness:
- `install`: where it lives on `/work/projects/imas_gpu/`
- `alias`: bashrc entry (already present)
- `endpoint`: how it learns the URL + key
- `headless`: scriptable invocation pattern
- `teams`: whether the tool itself coordinates multiple agents
- `best for`: workloads it excels at on `deepseek-v4-flash`
- `gotchas`: things that bit us on this cluster

### 1. OpenHands

- **install:** `/work/projects/imas_gpu/tools/openhands/` (uv venv, pip pkg `openhands-ai`).
- **alias:** `openhands`, `oh`, `oh-yolo`, `oh-headless`.
- **endpoint:** wrapper exports `LLM_BASE_URL`, `LLM_MODEL=openai/deepseek-v4-flash`, and `LLM_API_KEY` (read from shared `.env` at launch).
- **headless:** `oh-headless -t "prompt"` (auto-approves all actions, exits cleanly with a "CONVERSATION SUMMARY" panel and a resume hint).
- **teams:** single agent; no multi-agent orchestration.
- **best for:** *one-shot agentic tasks where you want a self-contained loop and a tidy end-of-run summary*. The cleanest UX of the lot. Supports MCP servers via `oh mcp`. Has an ACP mode if you want to drive it from Zed / Toad CLI.
- **gotchas:** Rich terminal detection fires "interactive UI may not render correctly" — set `TTY_INTERACTIVE=1` if it annoys you. Cold start is slow (~30 s end-to-end) because Rich initialisation + first model warm-up is not optimised for one-shot use.

### 2. Hermes (Nous Research)

- **install:** `/work/projects/imas_gpu/tools/hermes-agent/` (uv venv, pip pkg `hermes-agent`).
- **alias:** `hermes` (TUI), `hermes-cli` (scriptable).
- **endpoint:** `~/.hermes/config.yaml` carries `model.base_url` and `delegation.base_url`; the per-user `~/.hermes/.env` holds `OPENAI_API_KEY`. Wrapper sets `HERMES_INFERENCE_PROVIDER=custom` and `HERMES_DISABLE_LAZY_INSTALLS=1`.
- **headless:** `hermes-cli -z "prompt" --yolo` (auto-approves tools).
- **teams:** **yes** — depth-2 agent swarms with up to 64 concurrent leaf agents, configured under `delegation:` in `config.yaml`.
- **best for:** *long-running autonomous loops, agent delegation/swarms, and sandboxed exploration*. Its `terminal.backend: ssh` (default) makes every shell action happen on a remote sandbox host — your local files stay untouched. Built-in webhook gateway, dictation, kanban, MCP, ACP — the "everything app" of agent harnesses.
- **gotchas:** Default SSH sandbox writes files to the sandbox's filesystem, not yours. For local edits, either set `terminal.backend: local` in `~/.hermes/config.yaml`, or do your work inside the sandboxed workspace. The TUI (`hermes`) is the rich experience; `hermes-cli` is the headless equivalent.

### 3. Aider

- **install:** `/work/projects/imas_gpu/tools/aider/` (uv venv, pip pkg `aider-chat`).
- **alias:** `aider`.
- **endpoint:** wrapper exports `OPENAI_API_BASE` and `OPENAI_API_KEY`; passes `--model openai/deepseek-v4-flash` to aider.
- **headless:** `aider --message "prompt" --no-auto-commits --yes-always`.
- **teams:** single-agent.
- **best for:** *incremental edits of existing files with git-aware diff application*. Cheapest tokens of any harness here (it sends only relevant repo-map slices). Pair-programming model: you point it at files, it edits them with diffs, commits per change.
- **gotchas:** **For new files, declare them up front** with `--file palindrome.py --file test_palindrome.py`, otherwise Aider's edit-format parser bails when the model emits code blocks without filename headers. Also: by default it tries to create a `.git` repo; pass `--no-git` to skip.

### 4. Kimi-CLI

- **install:** `/work/projects/imas_gpu/tools/kimi-cli/` (uv venv, pip pkg `kimi-cli`).
- **alias:** `kimi`.
- **endpoint:** wrapper bootstraps `~/.kimi/config.toml` with a `[providers.ambix]` block of `type = "openai_legacy"` and the live key on first run. Mode 0600.
- **headless:** `kimi --print --yolo --prompt "prompt"`.
- **teams:** single-agent, but supports MCP tool servers and ACP for IDE integration.
- **best for:** *coding sessions where you want MCP / Agent Client Protocol (ACP) on by default*. Pairs naturally with Zed and Toad CLI through ACP. Has a built-in shell mode (Ctrl-X) for raw shell commands.
- **gotchas:** stdout is verbose (Python-repr style — `TextPart(...)`, `ToolCallPart(...)`) but is parseable.

### 5. OpenCode

- **install:** `/work/projects/imas_gpu/tools/opencode/` (npm — wraps a Rust binary).
- **alias:** `opencode`.
- **endpoint:** wrapper bootstraps `~/.config/opencode/opencode.json` declaring the `ambix` provider with `npm: "@ai-sdk/openai-compatible"` and the Ambix base URL. Mode 0600. The wrapper also defaults `opencode run …` to `-m ambix/deepseek-v4-flash` if you don't specify a model.
- **headless:** `opencode run "prompt"` (the wrapper picks the default model automatically).
- **teams:** single-agent; routes to 75+ providers via the AI SDK.
- **best for:** *quick coding tasks, and as a "swap LLM provider in one config line" tool*. Fastest of the headless harnesses here.
- **gotchas:** The very first invocation does an sqlite DB migration that can stall for >10 minutes — run a throw-away `opencode run "hi"` once in advance.

### 6. Continue CLI (`cn`)

- **install:** `/work/projects/imas_gpu/tools/continue-cli/` (npm pkg `@continuedev/cli`).
- **alias:** `cn`.
- **endpoint:** wrapper bootstraps `~/.continue/config.yaml` with `provider: openai`, `apiBase` = Ambix URL, and the live key inlined (Continue's YAML does not expand shell env vars at runtime). Mode 0600.
- **headless:** `cn -p --auto "prompt"` (`-p` is print mode, `--auto` allows all tools).
- **teams:** single-agent; shares config with the Continue VS Code / JetBrains extension.
- **best for:** *users who already run the Continue plugin in VS Code or JetBrains and want the same config to drive a terminal*. The unified config is the killer feature; the CLI itself is a competent but not standout coder.
- **gotchas:** YAML's `${{ env.OPENAI_API_KEY }}` syntax is only resolved by Continue Hub server-side, not by the local CLI. Hence we inline the key (file is 600).

### 7. Goose (Block → AAIF Linux Foundation)

- **install:** `/work/projects/imas_gpu/tools/goose/` (Rust binary, no Python/Node deps).
- **alias:** `goose`.
- **endpoint:** wrapper bootstraps `~/.config/goose/config.yaml` with `GOOSE_PROVIDER: openai`, `GOOSE_MODEL: deepseek-v4-flash`, and **`GOOSE_CONTEXT_LIMIT: 200000`** (without that, goose's default for unknown models is so small it errors immediately). Exports `OPENAI_BASE_URL` + `OPENAI_API_KEY`.
- **headless:** `goose run --no-session --quiet -t "prompt"` (`--no-session` skips the sqlite session record; `--quiet` suppresses chrome).
- **teams:** single-agent, but has a rich "extension" ecosystem (built-in `developer` extension covers shell + files; many MCP servers plug in directly).
- **best for:** *operator-style tasks where the agent needs to run a sequence of shell + file commands*. Fast cold start, clean output. Strong "engineer that just runs commands" persona.
- **gotchas:** **Must** set `GOOSE_CONTEXT_LIMIT` for unknown OpenAI-compatible models, or every run will fail with `Context limit reached. Compacting…`. The wrapper does this; if you write your own config, copy that field.

### 8. CrewAI

- **install:** `/work/projects/imas_gpu/tools/crewai/` (uv venv, pip pkg `crewai`).
- **alias:** `crewai`, `crew`.
- **endpoint:** wrapper exports `OPENAI_API_BASE`, `OPENAI_MODEL_NAME=deepseek-v4-flash`, and `OPENAI_API_KEY` (read from `.env`). Sets `CREWAI_TELEMETRY_OPT_OUT=1`.
- **headless:** the CLI is for project scaffolding — write Python that constructs `Crew(agents=[…], tasks=[…])`.
- **teams:** **yes** — its central abstraction. Role-typed agents (researcher / writer / reviewer) with delegation, hierarchical processes, and async tasks.
- **best for:** *role-driven multi-agent workflows where each agent has a distinct expertise* (e.g. a research crew, a doc-generation pipeline, an analysis pipeline).
- **gotchas:** CrewAI is a framework, not a CLI agent. Expect to write ~50 lines of Python per workflow. Long agent chains can blow the context budget; cap `max_iterations` aggressively.

### 9. AutoGen (AG2)

- **install:** `/work/projects/imas_gpu/tools/autogen/` (uv venv, pip pkg `autogen-agentchat` + `autogen-ext`).
- **alias:** `autogen` (runs Python in the autogen venv).
- **endpoint:** wrapper exports `OAI_CONFIG_LIST` JSON with `{model, api_key, base_url}` (key read from `.env`). Examples in `/work/projects/imas_gpu/tools/autogen/examples/` use `OpenAIChatCompletionClient(model=…, base_url=…, api_key=…)`.
- **headless:** `autogen my_script.py` (the wrapper just `exec`s the venv's python).
- **teams:** **yes** — `RoundRobinGroupChat`, `SelectorGroupChat`, `Swarm`, `Society`. We verified a two-agent planner+coder team on this endpoint produces working code in 6.8 s.
- **best for:** *flexible multi-agent patterns, especially when you want to mix retrieval, code execution, and review*. The most mature multi-agent framework here.
- **gotchas:** The bundled examples have `api_key="REQUEST_FROM_OPERATOR"` placeholders — replace with `os.environ["OPENAI_API_KEY"]` (the wrapper exports it) or read from the shared `.env`.

### 10. MAF (Microsoft Agent Framework)

- **install:** `/work/projects/imas_gpu/tools/maf/` (uv venv, pip pkg `microsoft-agent-framework`).
- **alias:** `maf` (Python runner), `maf-ui` (launches `devui` — a local browser UI for inspecting agent flows).
- **endpoint:** wrapper exports `MAF_MODEL`, `MAF_BASE_URL`, `MAF_API_KEY`, plus the standard `OPENAI_API_KEY` / `OPENAI_BASE_URL`.
- **headless:** `maf path/to/agent_script.py`.
- **teams:** **yes** — multi-agent workflows with a visual graph in `devui`.
- **best for:** *building agent workflows you want to inspect visually* (the `devui` graph is genuinely useful for debugging multi-step flows). Pairs well with Azure ecosystems but works fine pointed at any OpenAI-compatible endpoint.
- **gotchas:** `devui` opens on `http://localhost:8080` — if you are on a remote node, tunnel with `ssh -L 8080:localhost:8080 …`.

---

## Choosing between agent teams: which framework when?

| Need | Pick |
|---|---|
| Two or three agents in a back-and-forth (planner ↔ coder, writer ↔ reviewer) | **AutoGen** — `RoundRobinGroupChat` is the path of least resistance |
| Many specialists with distinct roles and delegation | **CrewAI** |
| Visual flow debugging | **MAF** with `maf-ui` |
| Concurrent autonomous loops (swarm), not strict hand-offs | **Hermes** delegation (up to 64 leaf agents) |
| You actually only need one agent | Any of the seven single-agent CLIs above — start with **OpenHands** or **OpenCode** |

---

## What `deepseek-v4-flash` is and isn't good at

Knowing the model's strengths helps you pick a harness that won't fight it:

**Strong**:
- Tool calling (all 4 single-tool / parallel / structured / no-call tests pass).
- Throughput — 110 tok/s decode, ~106 tok/s prefill on 4×H200.
- Long context — 262K served (1M architectural).
- Code generation, especially Python.
- MIT licensed weights — no contractual surprises.

**Weak**:
- Deep multi-turn reasoning that benefits from explicit chain-of-thought
  (no thinking-mode head trained). For long reasoning trajectories, the
  cluster also serves Kimi-K2.6 at ~4 tok/s — slower but better at deep
  thought; see `AGENTS.md`.
- Tasks that depend on very strong RLHF instruction following — Flash
  occasionally over-interprets brief prompts (we saw "ping" turn into a
  full HTTP server). Be explicit.

---

## Where the configs and keys live

| File | Role | Mode | Owner |
|---|---|---|---|
| `/work/projects/imas_gpu/agents/.env` | Shared `AMBIX_AGENT_API_KEY=…` — the live key the wrappers read | 0600 | The operator who rotated it last |
| `~/.hermes/.env` | Per-user copy of the same key for Hermes | 0600 | You |
| `~/.kimi/config.toml` | Kimi-CLI provider config (created by the `kimi` wrapper) | 0600 | You |
| `~/.continue/config.yaml` | Continue CLI / IDE config (created by the `cn` wrapper) | 0600 | You |
| `~/.config/goose/config.yaml` | Goose config (created by the `goose` wrapper) | 0600 | You |
| `~/.config/opencode/opencode.json` | OpenCode provider config (created by the `opencode` wrapper) | 0600 | You |

If any of these become 0640 / 0644, fix immediately with `chmod 600`.

---

## Harnesses we evaluated and skipped

| Tool | Why not |
|---|---|
| **Codex CLI** (OpenAI) | Since Jan 2026 requires the OpenAI **Responses API**; our vLLM/SGLang only serve Chat Completions. Will not work against `deepseek-v4-flash`. |
| **Cline**, **Roo Code**, **Cursor Agent** | VS Code / IDE extensions, not terminal CLIs. Worth knowing for IDE users; pair them with Continue's shared config. |
| **SWE-Agent** | Research tool for SWE-bench; not designed for daily engineering. |
| **GitHub Copilot CLI** | Uses GitHub's hosted model, not our endpoint. Already aliased as `copilot` if you want it. |
