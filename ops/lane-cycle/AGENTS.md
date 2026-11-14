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
holds, so a pause is the right instrument when a drain would hold other dispatches
while every in-flight worker node finishes.

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

Run the directory command to list the live reckon coordinators:

```bash
/home/ITER/mcintos/Code/reckon/.venv/bin/reckon crew directory --project imas-ambix
```

Add the
interactive sessions running on this machine. Message every session whose
workers dispatch to the lane before touching the gate:

- that a pause is coming, and why;
- the bound the pause will be held inside (see **Bounds** below), so a
  coordinator can decide whether to hold dispatch or let a turn ride out the
  pause.

The pause is also published, so a consumer that never got the message can see
it: `lane.json`'s `router_generation_gate` block carries `paused` and `reason`
while the pause holds, and admission is FIFO-ordered on resume.

### 2–6. Run the paused engine reload

The command saves the existing batch script for rollback, pauses admission,
waits for a fresh measured zero, and proceeds only while the paused FIFO holds
new arrivals. It replaces the serve, waits for scheduler `RUNNING`, waits for
`/v1/models` readiness, and validates with a real completion. A failed
readiness or completion restores the saved batch script. The command resumes
in all paths it paused and appends its measurements to `lane-cycles.jsonl`.
The complete pause is bounded at 40 minutes; a drain that does not reach a
measured zero within 10 minutes resumes without cancelling the serve.

Run it from the main checkout, after announcing the pause as in step 1:

```bash
uv run imas-ambix agent restart deepseek-v4-1-flash --reason "<why and bound>"
```

Pass launch overrides such as `--no-speculative`, `--dspark-block-size`,
`--gpus`, `--cpus`, `--time`, `--port`, and authentication options directly to
`restart`. `--dry-run` prints the replacement script and selected job without
pausing the lane or changing scheduler state.

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
Set `RECKON_RUN_ID` to that inline exception's run id and set the four value
variables from the readings above, then append the machine-readable record to
`/home/ITER/mcintos/.config/reckon/crew/runs/$RECKON_RUN_ID/lane-cycle.jsonl`:

```bash
CYCLE_RECORD="/home/ITER/mcintos/.config/reckon/crew/runs/${RECKON_RUN_ID:?set RECKON_RUN_ID to this inline cycle's run id}/lane-cycle.jsonl"
printf '{"pause_start":"%s","pause_end":"%s","peak_held":%s,"final_failures":%s}\n' \
  "${PAUSE_START:?set PAUSE_START from pause confirmation}" \
  "${PAUSE_END:?set PAUSE_END from resume output}" \
  "${PEAK_HELD:?set PEAK_HELD from pause reads}" \
  "${FINAL_FAILURES:?set FINAL_FAILURES from retry receipts}" >> "$CYCLE_RECORD"
```

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
| What it drops | Nothing. In-flight requests finish; later arrivals wait in the router's FIFO | Nothing the router is holding. On SIGTERM or SIGINT the router stops admitting and answers every request waiting at the gate with `529` and `Retry-After` before uvicorn's own shutdown begins; a relay already streaming is given the graceful timeout to finish and is recorded as cut if it outlasts it |
| What the client sees if a request is caught | A wait, then normal service on resume | A `529` with a five-second `Retry-After` — the same retryable refusal the wait bound produces — or, for a relay already answering, its completed response |
| When it may run | Any time admission can be paused; the engine reaches the quiet state of step 3 first | Inside a pause, after step 3. The release covers what the router itself holds; the drain still runs first because a request that has not yet reached the router has no gate to be answered from |
| Does the pause survive it? | Unaffected — an engine relaunch does not touch the gate file | Yes: `paused` is persisted in the gate file, and the new router reads it at start |
| Time budget | The 40-minute pause ceiling above | The router's own startup, measured **1.3 s warm and 4.3 s cold** on 2026-10-27 on `98dci4-clu-2068` (8 cores), from process start to the first answered `/v1/models` against a fake upstream on a free port — two samples, one cold at 4.34 s and one warm at 1.30 s |

The graceful timeout is `timeout_graceful_shutdown`, set by the launch and
defaulting to 30 s: it bounds how long a relay already streaming may take
before the process cancels it. The 529 answers are not bounded by it — they are
written at the signal, which is why the release is installed on the signal
rather than in the lifespan handler: uvicorn reaches the lifespan only after
that wait, and with the timeout unset it would wait forever.

A router restart that deploys code therefore sequences as: announce, pause,
drain to a genuine zero, restart the router, confirm the pause survived and
the new router serves, then resume. The two never run concurrently: one of
them must leave the lane quiet for the other. If a request is caught anyway,
it is answered rather than lost, and the `reason` on its receipt row
(`router shutdown`) counts it.

**The router belongs on `rigel`, and it moves there once consumers follow its
published address (lead decision 2026-11-14).** A router on a CPU node shares
no server's fate, which matters once a second GPU server serves. The router's
job publishes its address: it rewrites `routing_origins` in
`~/public/imas-ambix/endpoints.json` when it starts answering and again when it
exits. A consumer that reads that entry follows the router to any node; in
Python, `imas_ambix.agent.registry.published_router_origin(path)` returns it.
`clive` and this repository's own callers already read it.

Until every consumer does, the router stays pinned. Consumers that still hold
the fixed address (`AMBIX_AGENT_GLOBAL_URL`, by default
`http://98dci4-gpu-0003:18802`) are served only by a router on that host and
port, and the site's default entry is a `rigel` job with no node pinned, so
`agent router --submit` refuses the default. Name the pinned placement in the
submitting shell:

```bash
export AMBIX_AGENT_ROUTER='{"partition": "betelgeuse", "reservation": "gpu_0003_grpA", "node": "98dci4-gpu-0003"}'
imas-ambix agent router --port 18802 --submit --dry-run   # read the header first
imas-ambix agent router --port 18802 --submit
```

Submit the new router and see it answer `/v1/models` on that address before
the old job is cancelled by id; a second router on the same node cannot bind
the port while the first holds it, so on one node the order is cancel, then
submit, inside the pause. Measured 2026-11-13: a restart submitted with the
default entry started on `rigel`, and the consumers' address went unanswered
for about two and a half minutes until a pinned job replaced it.

**Moving to `rigel`.** First confirm each consumer outside this repository
resolves the published address: reckon's local backend and imas-codex's
generation endpoint are the two known to hold the fixed one. Then, inside a
pause, cancel the pinned job by id and submit the default entry with
`imas-ambix agent router --port 18802 --submit --any-node`. Read
`routing_origins` afterwards: it must name the `rigel` node the job started on
before admission resumes.
