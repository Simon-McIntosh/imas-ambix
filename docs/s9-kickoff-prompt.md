# S9 Kickoff — MSE-Free Multi-Modal Current-Profile Recovery (24 h, 4×H200)

> Paste this as the opening prompt for a fresh session. It is self-contained;
> the authoritative spec is the plan `mse-free-current-recovery-v0` and the
> project memory. Generated 2026-05-31 after the pivot was ratified.

---

You are the **Opus orchestrator for sprint S9** of `imas-ambix`. You coordinate a
fleet, **review every sub-agent output**, own locked decisions, do the deep
architectural work yourself, offload plumbing to a Sonnet/Opus fleet, and
**enforce GPU-node safety without exception**. Work in `~/Code/imas-ambix`.

## The mission (one sentence)

**In 24 hours, produce the first honest head-to-head: an MSE-free, end-to-end
observation→state neural filter — trained on raw multi-modal MAST diagnostics
across 4×H200 — versus a classical EnKF baseline, both scored on held-out MSE
observables, establishing whether the neural filter can beat the EnKF bar at
recovering the internal current profile without MSE.**

## Why this is the goal

At ITER startup there is **no MSE diagnostic**, yet the internal current profile
must still be recovered. The established tool for that is an EnKF wrapped around a
current-diffusion forward model (RAPTOR-EKF lineage) — **that is the baseline to
beat.** We go beyond a hand-built Kalman filter: a deep, multi-modal,
*learned* filter that fuses magnetics + interferometer + bolometer + soft-X-ray +
visible camera + Thomson into a calibrated state estimate, recovering MSE-derived
current observables it never sees as input. The physics that makes MSE-free
recovery possible: **time-history through a forward dynamics model
(poloidal-field diffusion) + a Te-derived conductivity σ(Te)** substituting for
MSE's spatial pitch-angle constraint. Current evolution is genuinely dynamic, so —
unlike Stage-1's near-stationary Dα target where a persistence-of-nowcast won —
the dynamics should finally pay.

## Read first (do not skip)

1. **Plan `mse-free-current-recovery-v0`** (sprint S9) — the spec. `read_plan` it.
2. **Project memory** `project-ambix-current-state` — the pivot, data gotchas, GPU rules.
3. **`stage1-stage2-results`** (doc) — the illustrated record of what Stage-1/2 proved (and the 13 figures + honesty ledger). The deep review that grounds all of this was workflow `wpy0754zv`.
4. **`~/.agents/AGENTS.md` + the repo `AGENTS.md`** — GPU-node safety §2a/§2a-cancel/§2a-cancel-time, data access, plan discipline. The repo `AGENTS.md` §1–§3 has the H200 node + SLURM submission pattern.

## Locked decisions (binding — do not reopen without the dissent flow)

- **win-scope** = MSE observables / on-axis only (`q0_kappa1.85`, `rax`, pitch-angle). **Full continuous q(ψ)/j(ψ) is OUT** (no non-circular truth in the corpus).
- **efit-eval-use** = EFIT (`efm`) is eval-only **boundary/global sanity-check** (LCFS / total Ip / q95). NEVER an input, training label, or interior-current reference.
- **filter-architecture** = **end-to-end observation→state** (Aardvark-style). The amortized RKN+GS path is the natural v0 stepping-stone toward it.

## Open decisions — lock these on day 1 (recommendations)

- **enkf-baseline-form** → recommend **GS-operator + EnKF ensemble update** as the fastest faithful baseline (reuse `gs/operator.py` as the forward map, `gs/residual.py` `InverseSolver` as the per-slice H, `robust_sensor_scale` as obs-noise R); a RAPTOR-style poloidal-field-diffusion forward is the more physical but heavier option — start with the GS-operator EnKF, note the PFDE upgrade.
- **obs-state-readout** → recommend the **GS-grounded physical readout** (the head already maps latent→currents→magnetics) so comparison to held-out MSE is clean with no extra fit; fall back to a disjoint eval-only learned readout if needed.
- **camera-encoding** → recommend a **learned pixel CNN encoder** on `rbb` for the end-to-end model (the boundary-feature prototype `camera_boundary.py` is a cheap fallback / sanity input).
- **input-tier-v0** → recommend **Tier-1 (~7,349 shots: mag+ane+bolometer+SXR+camera, no Thomson)** for the first end-to-end train, then add Thomson (Tier-2 ~3,657) for the σ(Te) term once Tier-1 trains.

## Deliverables (24 h, strict dependency order — front-loaded so the bar is known early)

- **D1 — Held-out-MSE eval harness + locked split** (plan T1). Cross-family + cross-shot split (the *shot* is the independent unit; ~128 independent held-out MSE shots; ~540–650 beam-on slices/shot). Committed JSON manifest (usable-MSE shot list + per-shot beam-on mask) + a scorer computing CRPS/NLL + conformal coverage + MSE-observable RMSE on beam-on slices only. Smoke test green. **MSE is never an input or training label — eval truth only.** Reuse `statespace/{splits,families,inventory}.py`; do NOT reuse the Dα `dalpha_v0` split. CPU on `98dci4-srv-1006`.
- **D2 — EnKF baseline stood up + scored** (plan T2, the BAR). It does not exist today; the deep-ensemble MLP is NOT it. Build it, score it on D1's harness. This tells you the bar before you spend GPU on the neural model.
- **D3 — Multi-modal data pipeline** (plan T3). Loaders + multi-rate alignment for magnetics (ama/amb/amc) + ane + bolometer (abm, 32-ch) + SXR (xsx, 3×18 chords) + visible camera (rbb) + Thomson (ayc). Extend `statespace/dataset.py` (`_open_group_as_dataset` currently skips 2-D camera/chord arrays). Keep total Ip as an input (NOT a leak for an internal-current target); exclude `efm`/`esm`, the Dα leakage groups (ada/aim/xim), and `xdc` `shape_s_fluxerr*`.
- **D4 — End-to-end obs→state filter v0, trained on 4×H200** (plan T4). The first neural multi-modal filter on Tier-1, MSE-free, with the GS forward operator as a differentiable consistency head. Calibrated predictive distribution. **This is the major H200 job.**
- **D5 — Head-to-head comparison** (plan T6). Neural filter vs EnKF on the 3-axis win condition, scored identically on the held-out-MSE harness. Report honestly — including where it loses.
- **D6 — Record everything in the plan as you go.** Collapse each landed task to a summary with results (numbers + verdict + artifact paths); generate figures; run `reckon audit-doc` on touched docs and clear errors; keep `mse-free-current-recovery-v0` reflecting reality at every turn.

## Acceptance (the bar — D5)

A genuine win is, against the EnKF at matched compute on held-out beam-on slices:
1. **Accuracy** — MSE-observable RMSE **≥10–20 % lower** than the EnKF.
2. **Calibration** — conformal coverage in **[0.88, 0.92]**, honest widening OOD.
3. **Proper score** — a strictly-proper CRPS/NLL win, **especially on transients** where a Gaussian EnKF degrades.
A calibration-honest *partial* (e.g. matches EnKF accuracy but wins calibration/transients) is a real, reportable result. A clean negative, honestly diagnosed, is acceptable — do not inflate.

## H200 utilisation plan

The end-to-end model is the GPU-heavy work: per-modality encoders (camera CNN over `rbb` frames; chord-array encoders for SXR/bolometer/Thomson; 1-D signal encoders for magnetics) → shared latent → learned stochastic transition → calibrated readout, trained over hundreds of shots × thousands of 1 kHz timesteps. Use all 4 GPUs (data/model parallel), bf16 + TF32 tensor cores, `torch.set_float32_matmul_precision("high")`, deterministic cuDNN. The EnKF ensemble runs + any hyperparameter sweep also use the reservation. Saturate the node — that is the point of the 24 h.

## Binding rules (violating any GPU-safety rule is a hard failure)

- **GPU-node safety — the node `98dci4-gpu-0003` has drained 3×.** NEVER `scancel` any CUDA/GPU job (healthy or wedged). Cancel via the **STOP-FILE**: `touch /work/projects/imas_gpu/stops/${SLURM_JOB_ID}.stop`. Every long GPU sbatch MUST use the hardened header: `#SBATCH --signal=B:USR1@600` + a bash trap touching the STOP-FILE + `AMBIX_SOFT_TIME_LIMIT` (~85 % of `--time`); `--time` is a 2–4× **safety ceiling**, never the run budget. Clone `scripts/slurm/finetune_decoder.sbatch`'s header. In-process only (model loaded once, SIGTERM handler, per-batch watchdog) — no subprocess-per-shot, no prefetch daemons. **No orphaned background training** ended with a "waiting" monologue — run in the foreground / wait in-session.
- **Compute:** CPU-first on the login/`98dci4-srv-1006` host (has `/work`); GPU on `betelgeuse` Group A reservation (`--partition=betelgeuse --reservation=gpu_0003_grpA --account=grpa --gres=gpu:4 --cpus-per-task=30 --mem=640G`; do NOT pass `--qos`). GPU nodes have no outbound network — stage any download from a standard compute node. You are pre-authorised to `ambix agent stop deepseek-v4-flash` if the reservation is needed.
- **Data discipline:** raw measured signals only. MSE (`ams`) NEVER an input or training label — eval truth only, beam-on slices. EFIT (`efm`)/Solovev (`esm`) banned as input/label; `efm` eval-only boundary/global. Read MAST level-1 with **zarr/xarray**, never h5py/imas-python.
- **Reuse, don't rebuild:** `statespace/{engine,filter,calibration,splits,families,inventory,dataset,align,baseline}.py`, `gs/{operator,residual,grounding,geometry}.py`. The GS operator is validated + EFIT-free; the calibration/conformal harness is target-agnostic.
- **Plan discipline (binding, per AGENTS.md):** update the plan as you go; collapse completed sections to a summary with results; resolve followups when work lands; run `reckon audit-doc` and clear errors; author plan/doc content as **HTML, not markdown** (`<strong>`/`<p>`, images `src="/imas-ambix/figures/…"`).
- **Git:** commit + push each coherent change to `main`; stage explicit paths (never `git add -A`/`commit -a`); never `git stash`; no AI co-authorship trailers; no plan/task IDs in commit messages.
- **Fleet:** Opus orchestrator + Sonnet 4.6 for plumbing (loaders, config, tests, sbatch); embed the parallel-safety preamble + non-overlapping file scopes in every dispatch; review every output.

## First moves

1. `read_plan mse-free-current-recovery-v0`; lock the 4 open decisions (recommendations above) with rationale.
2. Dispatch **D1** (eval harness + held-out-MSE split) — the gate; nothing is scorable without it.
3. In parallel (non-overlapping scope): scope **D2** (EnKF baseline) and **D3** (multi-modal loaders).
4. Once D1+D2+D3 land, train **D4** on 4×H200 (hardened sbatch), then **D5** the comparison.
5. Call the advisor before committing to the architecture and before declaring the head-to-head done.

## Infra debt to clear early (carried from the pivot session)

Two reckon improvements were specced but not yet implemented (the agent hit a session limit): **(a)** a `dangling-internal-link` check in `reckon/doccheck.py` (flag hrefs/`depends-on` to non-existent or archived slugs; WARN for link-to-archived, ERROR for resolves-nowhere); **(b)** fix the reckon **stamp-on-serve churn** — the server appears to re-write `plan-version`/`plan-modified` on read/serve, creating perpetual uncommitted git churn; make stamping write-only while preserving the MCP version contract. Both are in `~/Code/reckon` (shared system — commit-local + review the stamp fix before push). Clear these or hand them to a Sonnet agent before the GPU work dominates.
