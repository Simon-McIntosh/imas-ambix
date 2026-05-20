# Getting started on the SDCC GPU LLM cluster

For members of the `sdcc-imas_gpu` group who want to drive the
locally-served **DeepSeek V4-Flash** model from the command line.

Estimated time to first agent run: **~5 minutes** if you are already
in the group.

---

## 0. Are you in the group?

```bash
id -nG | tr ' ' '\n' | grep -c sdcc-imas_gpu
```

If this prints `1`, you are in. If `0`, request membership from your
local service desk and come back when it is granted — none of the
shared tooling under `/work/projects/imas_gpu/` is readable without it.

---

## 1. Get the endpoint URL and an API key (out-of-band)

> **The endpoint URL and the API key are not in this repository.**
> They must be requested out-of-band, over Microsoft Teams, from the
> Ambix maintainer (channel **SDCC GPU / Ambix** or direct message
> the deployment owner — see `git log AGENTS.md` for active operators).

You will receive two values:

1. The **base URL** of the inference endpoint (an `http://<host>:<port>/v1`
   on a SLURM GPU node — reachable from any SDCC login node).
2. The **API key** — a long random token. Treat it like a password.

Do **not** paste either into chat, tickets, screenshots, or commits.

---

## 2. One-time configuration

Most harnesses read the key from a shared `/work/projects/imas_gpu/agents/.env`
file that the maintainer keeps current. If you are using **Hermes** or
running anything in your own scripts, you also want a per-user copy at
`~/.hermes/.env`:

```bash
mkdir -p ~/.hermes
touch ~/.hermes/.env
chmod 600 ~/.hermes/.env   # MANDATORY — without this, anyone with
                           # access to your home directory on any
                           # login or compute node can steal the key
```

Open `~/.hermes/.env` in your editor and add:

```env
# Endpoint URL — paste the value the maintainer sent you over Teams
OPENAI_BASE_URL=<the URL from Teams>

# API key — paste the value the maintainer sent you over Teams.
# This file MUST stay at mode 600.
OPENAI_API_KEY=<the key from Teams>

# Lock harnesses to this provider so SDK auto-detection cannot route
# to public OpenAI / Anthropic accidentally.
HERMES_INFERENCE_PROVIDER=custom
HERMES_DISABLE_LAZY_INSTALLS=1
```

Verify the file is private:

```bash
stat -c '%a %n' ~/.hermes/.env
# expected output: 600 /home/ITER/<you>/.hermes/.env
```

Anything other than `600` is a finding — fix it now:

```bash
chmod 600 ~/.hermes/.env
```

### Why 0600?

The SDCC home directory is mounted on every login and compute node.
Any process running as a different user on the same node can `cat`
your files if they are world- or group-readable. A leaked key allows
anyone on the ITER network to consume GPU time on your behalf — and
to read whatever else your code happens to be sending to the model.

### What if the key stops working?

The maintainer rotates the key on a schedule and after any suspected
exposure. When that happens, every harness on your machine starts
returning `401 Unauthorized`. Request the new key over Teams and
update `~/.hermes/.env` (still 0600). No other config changes needed.

---

## 3. Auto-load your env

Append to `~/.bashrc` (one line, idempotent):

```bash
[ -f "$HOME/.hermes/.env" ] && set -a && . "$HOME/.hermes/.env" && set +a
```

Then either `source ~/.bashrc` or open a new shell. Now any tool that
reads `OPENAI_BASE_URL` + `OPENAI_API_KEY` will pick them up.

---

## 4. First run

The cluster has ten agent harnesses installed under
`/work/projects/imas_gpu/tools/`, each fronted by an alias that
handles group switching and config bootstrapping. The recommended
harness for first contact is **OpenHands** in headless mode:

```bash
mkdir -p ~/work/firstrun && cd ~/work/firstrun
oh-headless -t "Create palindrome.py with is_palindrome(s) using regex \
and test_palindrome.py with five pytest cases, then run pytest -v."
```

After ~30 s you should see a "CONVERSATION SUMMARY" panel and two
new `.py` files in your working directory. Run them:

```bash
python -m pytest -v
```

If you get `5 passed`, you are operational.

---

## 5. Recommendation for day-to-day work

There is no single "best" harness — each is good at different things.
Pick based on the shape of the task in front of you:

| Daily task                                    | Use                | Why |
|-----------------------------------------------|--------------------|-----|
| "Build me X from scratch" (one-shot)          | `oh-headless`      | Cleanest agent loop, end-of-run summary |
| Tight loop of edits to an existing file       | `aider`            | Diff-based, cheap tokens, git-aware |
| "I want to talk through a problem first"      | `oh` (TUI)         | Interactive, supports REPL-style refinement |
| Quick exploration in an isolated sandbox      | `hermes` (TUI)     | SSH sandbox isolates your real fs from agent shell commands |
| Two-agent workflow (planner + coder, etc.)    | AutoGen script     | `RoundRobinGroupChat` is 20 lines |
| Multi-role crew with delegation               | CrewAI             | Designed for role-based teams |
| Visual debugging of an agent workflow         | `maf-ui`           | Browser-based flow inspector |
| Same config also driving VS Code / JetBrains  | `cn` (Continue)    | One `~/.continue/config.yaml` covers all three |
| Latest tooling, multi-provider routing        | `opencode`         | Easy provider swap; fastest in our trial |

For 80% of one-off engineering tasks, **`oh-headless`** is the
right default.

For incremental editing of code you already have, **`aider`** is
the right default.

For multi-agent work, **AutoGen** is the right default unless you
have a specific reason to pick another framework.

---

## 6. Aliases at a glance

After your `.bashrc` reload, these aliases are live (all routed through
`sg sdcc-imas_gpu -c "…"`):

| Alias            | Tool                | Usage |
|------------------|---------------------|-------|
| `hermes`         | Hermes Agent (TUI)  | Rich interactive UI |
| `hermes-cli`     | Hermes Agent        | Headless: `hermes-cli -z "task" --yolo` |
| `openhands`, `oh`| OpenHands           | Interactive |
| `oh-yolo`        | OpenHands           | Interactive, auto-approve |
| `oh-headless`    | OpenHands           | Function: `oh-headless -t "task"` |
| `aider`          | Aider               | `aider --message "task"` (use `--file foo.py` for new files) |
| `kimi`           | Kimi-CLI            | `kimi --print --yolo --prompt "task"` |
| `goose`          | Goose               | `goose run --no-session --quiet -t "task"` |
| `opencode`       | OpenCode            | `opencode run "task"` |
| `cn`             | Continue CLI        | `cn -p --auto "task"` |
| `crewai`, `crew` | CrewAI              | Project scaffolding — write Python that builds a Crew |
| `autogen`        | AutoGen             | `autogen path/to/script.py` |
| `maf`            | MAF                 | `maf path/to/script.py` |
| `maf-ui`         | MAF devui           | Browser UI on `http://localhost:8080` |

Full per-harness reference and tradeoffs: see [`harnesses.md`](./harnesses.md).

---

## 7. A note on cost and etiquette

- The endpoint is one shared node. **One concurrent agentic task per
  user** is a reasonable rule; sustained `concurrency >= 16` will be
  noticed.
- Don't paste credentials, PII, or sensitive RDD material into prompts.
  The model is local but logs and session files persist.
- If something looks broken, check the operator channel before
  filing — there might be a scheduled rotation or restart in flight.

---

## 8. Where to look when things break

- `~/.hermes/.env` mode 600, key matches what the maintainer last
  sent. `chmod 600 ~/.hermes/.env` if not.
- Re-source `~/.hermes/.env` in your current shell:
  `set -a && . ~/.hermes/.env && set +a`.
- 401 from everything → key was rotated. Get the new one over Teams.
- 4xx from a specific harness → likely a per-harness config file got
  stale. Delete `~/.config/<harness>/…` or `~/.kimi/config.toml` etc.
  and let the wrapper recreate it on next launch.
- Per-tool config / install / known gotcha is documented in
  [`harnesses.md`](./harnesses.md) §1–10.

---

## 9. Ambix plan site (HTML coordination hub)

The `plans/` directory contains the full Fusion World Model plan set.
An interactive HTML site generated from those plans lives under
`docs/plans-html/`:

```bash
xdg-open docs/plans-html/index.html
```

The site provides:

- **Timeline** — milestone phases with status badges (click to expand)
- **Status table** — all work items with filter buttons (Done / In flight / Next / Blocked)
- **Blocking tasks** — what is currently blocked and what it is waiting for
- **Open-decision widgets** — 9 interactive forms that generate copy-pasteable
  follow-on prompts to paste back into the AI coordinator

To regenerate after editing plan docs:

```bash
python docs/plans-html/_generate.py
```

See [`docs/plans-html/README.md`](./plans-html/README.md) for full operator instructions.
