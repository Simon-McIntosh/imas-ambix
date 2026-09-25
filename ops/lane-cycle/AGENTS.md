# Agent Guidelines — Lane cycling

**Scope.** This directory owns the procedure for cycling the local serving
lane — pausing admission, relaunching the engine with new settings (or
restarting the router), and resuming. It applies to any work under
`ops/lane-cycle/`, and to any request to change the lane's serve settings,
bench the quiet engine, or deploy router code onto the running lane.

**Why a pause and not a drain.** A run-level drain waits for every in-flight
worker node to finish and charges about an hour of held dispatch to other
projects. A pause stops admission at the lane's front door instead: the engine
finishes the requests it already has, later arrivals wait in the router's FIFO,
and `resume` admits them in arrival order. Nothing is lost while the pause
holds, so a pause is the right instrument wherever a drain used to be the only
route.

**Run every command in this runbook from the main checkout**
(`/home/ITER/mcintos/Code/imas-ambix`), never from a worktree. A serve
submitted from a detached worktree resolves the SGLang kernel patch and
`scripts/slurm/drain_sidecar.sh` by absolute path from the tree it was
generated in, so it holds that worktree open for the serve's whole life;
reclaiming the worktree — ordinary housekeeping — then breaks the lane. Check
the generated script rather than trusting it: **no path in the submitted
script may contain `.reckon-worktrees`**.

A cycle is an operational action on shared services outside the repository, so
the coordinator runs it as a reported inline exception rather than as a worker
node.

## The seven steps

Each step carries its own command. Read what each command prints rather than
assuming it worked.

### 1. Announce the pause to the live coordinators

`crew(view="directory")` lists the live reckon coordinators; add the
interactive sessions running on this machine. Message every session whose
workers dispatch to the lane before touching the gate:

- that a pause is coming, and why;
- the bound the pause will be held inside (see **Bounds** below), so a
  coordinator can decide whether to hold dispatch or let a turn ride out the
  pause.

The pause is also published, so a consumer that never got the message can see
it: `lane.json`'s `router_generation_gate` block carries `paused` and `reason`
while the pause holds, and admission is FIFO-ordered on resume.

### 2. Pause

```bash
uv run imas-ambix agent pause --reason "<why, and the bound>"
```

The reason is required: a pause has to read as a stated decision rather than
as a lane that happens to look quiet. The command writes `"paused": true` and
the reason into `router-gate.json` — the gate file the running router reads
live, so no restart is needed — and prints the lane's counts. The pause
outranks the width, including `width 0`.

### 3. Wait for the engine to drain

The drain is the requests already in flight, normally 1–3 minutes. Wait until
the engine's running count reads **0**, and only from a reading that
succeeded — a failed scrape is not zero. The published lane document is the
reading to use; `classify_reading` names whether it is current:

```bash
uv run python -c "
import json, pathlib
from imas_ambix.agent.lane import classify_reading
doc = json.loads(pathlib.Path.home().joinpath('public/imas-ambix/lane.json').read_text())
print(classify_reading(doc), 'running=' + str(doc.get('running')), 'waiting=' + str(doc.get('waiting')))
"
```

Proceed only on a line that reads `measured` with `running=0`. `stale` and
`unavailable` are the two ways this check fails, and neither is a zero: a
lane with nothing running and a lane nobody can read look identical in the
counters alone. Do not read the router's own `in_flight` as the drain signal —
that counts requests the router still holds, and it stays non-zero for the
whole pause by design.

### 4. Relaunch with the new settings

Change the settings first, then relaunch. Settings that are CLI options can be
passed; a serve setting with no CLI flag is a profile edit
(`imas_ambix/agent/profiles/<slug>.toml`), committed like any other change.

Check the generated script before submitting it:

```bash
uv run imas-ambix agent restart deepseek-v4-1-flash --dry-run | grep -c '\.reckon-worktrees'   # must print 0
```

Then relaunch (this is shutdown plus serve, so the endpoint document
republishes):

```bash
uv run imas-ambix agent restart deepseek-v4-1-flash
```

Add `--no-speculative` for a serve without speculative decoding. A depth cell
that needs a draft-block-size other than the profile's own key changes the
profile (the relaunch override for the sweep arrives with the sweep tooling).

To stop a serve and not start a new one, use `uv run imas-ambix agent shutdown
deepseek-v4-1-flash --yes`. **Never `scancel` a serve**: `shutdown` cancels
the profile's own jobs *and republishes the endpoint document*, and a bare
`scancel` skips the republish, leaving the published document claiming a lane
that is gone. Never `scancel` a serve with peer runs live against it.

### 5. Wait for readiness, then validate

Measure readiness **from the job's `RUNNING` transition, not from submit** —
time spent pending on `Resources` says nothing about whether the engine will
start. The bound is **15 minutes from `RUNNING`**; a launch carrying the fused
draft module took 485 s.

`agent status` prints a compact job table and a connection block for each
RUNNING serve, including a live `/v1/models` readiness probe:

```bash
uv run imas-ambix agent status
```

Readiness is not the gate for DSPARK: the fused-MoE workspace allocates
lazily at generation, so validate with a real multi-hundred-token completion
and read the completion, not the HTTP status:

```bash
curl -sS -m 300 http://98dci4-gpu-0003:18810/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4.1-flash","max_tokens":400,
       "messages":[{"role":"user","content":"Write twenty numbered lines about tokamak equilibrium."}]}' \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('usage'), d.get('choices',[{}])[0].get('finish_reason'))"
```

Require a `usage.completion_tokens` in the hundreds and a
`finish_reason` of `stop` or `length`. An empty or absent body means the
engine is not ready however healthy the process looks.

The engine port is bound so that login and standard compute nodes reach it
directly; once the engine binds loopback, run this curl and any bench on the
engine's own node — a bench against the quiet engine is launched there.

### 6. Resume

```bash
uv run imas-ambix agent resume
```

The router admits the held requests in arrival order. The command clears
`paused` and the reason from the gate file and prints the counts.

### 7. Record the cycle

Record four figures against the cycle:

| Field | Where it comes from |
|---|---|
| Pause start | the moment the flag took effect — the `pause` write, confirmed by the gate block's `paused` turning true in `lane.json` |
| Pause end | the `resume` command's own output |
| Peak held count | the highest `router_generation_gate.waiting` read during the pause; the count of 529 refusals for the window comes from the router's own receipts |
| Final failures | whether any held request exhausted its client retries during the pause — a held request that received the 529 past the gate's wait bound is the evidence, not the absence of an error message |

Write the cycle's record to the plan evidence record that owns the cycle, so
the figures sit beside the settings they were measured under. A cycle that ran
as an inline exception also leaves a one-line note in the run's own directory.

## Bounds

**A pause never runs open-ended.** A held client rides out a pause of about
**56 minutes** — 11 attempts of 305 s: the gate holds each request for
`wait_seconds` (300 s), then answers the existing 529 `overloaded_error` with
`Retry-After: 5`, which the client honours exactly and retries ten times. Past
that the turn fails.

The components of a bounded pause are a 1–3 minute drain, a readiness bound of
15 minutes from `RUNNING`, and validation plus resume. Keep the whole pause
inside **40 minutes**, which leaves about 16 minutes of margin on the ~56
minute ride-out. If the readiness bound is missed, roll back rather than
pushing the pause past its ceiling.

**Rollback to the previous settings.** Put the previous settings back — the
previous `speculative_dspark_block_size`, or `--no-speculative` where the
cycle added speculation, or the previous gate width — and relaunch, then
resume:

```bash
uv run imas-ambix agent restart deepseek-v4-1-flash        # with the previous settings restored
uv run imas-ambix agent resume
```

`resume` is never skipped: a lane left paused holds every arrival in a FIFO
until each client's retries are exhausted, which is the outcome the pause
exists to prevent. Resume is the step that closes the cycle, whether the new
settings validated or the previous ones were restored.

## A router restart is not an engine relaunch

Both are deployments, and their costs differ in a way that decides when each
may run:

| | Engine relaunch | Router restart |
|---|---|---|
| What it drops | Nothing. In-flight requests finish; later arrivals wait in the router's FIFO | Every request the router is holding — the FIFO lives in the router's memory, so it dies with the process |
| What the client sees if a request is caught | A wait, then normal service on resume | A dropped connection, not the 529-with-`Retry-After` path that ends in a clean retry |
| When it may run | Any time admission can be paused; the engine reaches the quiet state of step 3 first | Inside a pause, **after** step 3 — the engine must already be quiet, because a held request at the swap instant is lost |
| Does the pause survive it? | Unaffected — an engine relaunch does not touch the gate file | Yes: `paused` is persisted in the gate file, and the new router reads it at start |
| Time budget | The 40-minute pause ceiling above | The router's own startup, which must finish inside the client's retry budget — measure it before relying on it, since a caught request does not get the retry schedule |

A router restart that deploys code therefore sequences as: announce, pause,
drain to a genuine zero, restart the router, confirm the pause survived and
the new router serves, then resume. The two never run concurrently: one of
them must leave the lane quiet for the other.