# Prepared (HELD) GPU/CPU encodes — corpus expansion

Three encodes are PREPARED and **HELD**: do NOT launch until the lead signals a
free card (the unified corpus build + current runs hold 8/8 GPUs). Each is
validated to wire up; the launch commands below are ready to fire as one
coordinated batch once a slot opens.

All paths are project-absolute. All GPU encodes follow AGENTS.md §2/§2b:
betelgeuse, `--gres=gpu:N`, **no `--qos`** (→ normal, no 4-GPU cap, coexists with
the DeepSeek serve), in-process, SIGTERM-clean.

---

## 1. rbc — untokenised visible camera (Open-MAGVIT2 frames)

`rbc` has **0** token stores; it is encoded exactly like rbb — same VQ model,
same `stream_encode.py` path — only the `CAMERA` and shotlist differ. Reuse the
production frame encoder verbatim.

**Step 1 (CPU/sun): build the rbc shotlist** (shots whose L1 carries rbc but have
no rbc tokens):
```bash
srun --partition=sun --cpus-per-task=8 --mem=16G --time=01:00:00 \
  bash -lc 'export TMPDIR=/tmp; cd /home/ITER/mcintos/Code/imas-ambix; \
    uv run python scripts/slurm/build_encode_shotlists.py --phase rbc \
      --out /work/projects/imas_gpu/agents/excitation-corpus/shotlist_rbc.json'
```

**Step 2 (GPU/betelgeuse, HELD): encode** via the production sbatch with
`CAMERA=rbc` and the shotlist as `EXPLICIT_SHOTS`. The sbatch already shards
across the array; point `STREAM_ROOT` at the live frames root so the tokens land
beside the other cameras:
```bash
# read the shotlist into a comma list, then (per shard) hand it to the encoder.
# N_SHARDS over the free cards; do NOT pass --qos.
SHOTS=$(uv run python -c 'import json;print(",".join(map(str,json.load(open("/work/projects/imas_gpu/agents/excitation-corpus/shotlist_rbc.json"))["shot_ids"])))')
sbatch --array=0-5 \
  --export=ALL,CAMERA=rbc,N_SHARDS=6,EXPLICIT_SHOTS="$SHOTS",STREAM_ROOT=/work/projects/imas_gpu/mast-tokens/v1/frames \
  scripts/slurm/stream_encode_rbb.sbatch
```
Reported count to confirm against: lead said ~2982 rbc shots (the shotlist
builder prints the exact number).

---

## 2. M5/M6 rbb backfill — early campaigns below the tokenised floor

~3378 shots with id < 15085 carry rbb in L1 but were never tokenised (confirmed
on disk). Same rbb encoder, just the early-campaign shotlist.

**Step 1 (CPU/sun): build the backfill shotlist**:
```bash
srun --partition=sun --cpus-per-task=8 --mem=16G --time=01:00:00 \
  bash -lc 'export TMPDIR=/tmp; cd /home/ITER/mcintos/Code/imas-ambix; \
    uv run python scripts/slurm/build_encode_shotlists.py --phase backfill \
      --out /work/projects/imas_gpu/agents/excitation-corpus/shotlist_rbb_backfill.json'
```

**Step 2 (GPU/betelgeuse, HELD): encode** (rbb, the backfill shotlist):
```bash
SHOTS=$(uv run python -c 'import json;print(",".join(map(str,json.load(open("/work/projects/imas_gpu/agents/excitation-corpus/shotlist_rbb_backfill.json"))["shot_ids"])))')
sbatch --array=0-5 \
  --export=ALL,CAMERA=rbb,N_SHARDS=6,EXPLICIT_SHOTS="$SHOTS",STREAM_ROOT=/work/projects/imas_gpu/mast-tokens/v1/frames \
  scripts/slurm/stream_encode_rbb.sbatch
```
This extends the rbb corpus down to the M5/M6 campaigns, so the unified corpus
re-run afterwards will pick them up (one window per (shot, camera)).

---

## 3. ait — divertor heat-flux SIGNAL stream (NOT camera frames)

`ait` is the divertor IR heat-flux ANALYSIS: per-shot time-resolved strike-point
traces (`etot_{isp,osp}`, `lampow*`, `ptot`, `peakpower_pos`, `temperature`,
`satpixels`) + (time, 186) `qprofile`/`tprofile` heat-flux/temperature profiles
on `rcoord_{isp,osp}` — confirmed ~4.6k time samples per shot. The raw IR camera
itself is ~13 real frames → **skipped** (no imaging). ait is a DIAGNOSTIC
measurement (admissible conditioning / W3 probe), not a reconstruction → no
leakage-ban concern.

These traces are time-resolved at a moderate cadence (NOT the MHz raw rate the HF
phase-tokeniser `signal_hf_encode` targets), so ait fits the **L2 measured-signal
pattern** the world-model v2 signal loader already consumes (like `summary_l2` /
`pf_active_l2`), not the HF tokeniser. The prepared script STAGES the ait traces
into a per-shot signal Zarr keyed by the ait `time` axis — the input the
downstream uniform-quantiser tokeniser / conditioning loader reads.

**Stage (CPU/sun — no GPU; can run anytime, but HELD with the batch for a single
coordinated launch):**
```bash
# sharded across sun (IO-bound read+reshape).
sbatch --partition=sun --cpus-per-task=8 --mem=24G --time=02:00:00 --array=0-5 \
  --output=/work/projects/imas_gpu/agents/excitation-corpus/ait_shard_%a.log \
  --wrap='export TMPDIR=/tmp; cd /home/ITER/mcintos/Code/imas-ambix; \
    uv run python scripts/slurm/encode_ait_signal.py \
      --shard $SLURM_ARRAY_TASK_ID --n-shards 6 \
      --report /work/projects/imas_gpu/agents/excitation-corpus/ait_report_$SLURM_ARRAY_TASK_ID.json'
```
Staged streams land at `/work/projects/imas_gpu/mast-tokens/v1/signals-ait/ait/<shot>/ait.zarr`.
Reported count to confirm: lead said ~2846 ait shots.

**Tokeniser hookup (the gated follow-on, after staging):** add an `ait` entry to
the L2 measured-signal modality list the v2 dataset reads
(`default_signal_modalities` in `spacetime_dataset_v2.py` — a peer-owned file, so
coordinate the one-line addition), pointing at the staged `signals-ait` group,
and tokenise with the uniform quantiser like the other L2 measured groups. The
staging here is the part that needs the data read; the modality registration is a
config one-liner the model agent applies when ait conditioning is wanted.

---

## Launch order when the slot opens
1. CPU shotlist builds (rbc, backfill) + ait staging — can overlap, all on `sun`.
2. GPU rbc + rbb-backfill encodes — betelgeuse, sharded over the freed cards, no
   `--qos`, coexist with DeepSeek.
3. Re-run the unified corpus build afterwards so the new rbc + backfill tokens
   enter `curated_windows_unified.json`.
