# Plan: Wide-Angle Viewing System Forward-Prediction Demo

Status: **Draft** — depends on `world-model-v0.md` model converging.

The demo is the singular v0 deliverable that an outside reader can
understand at a glance: given a held-out MAST shot's control schedule
and the first ~150 ms of plasma evolution, generate the next ~1 second of
the wide-angle visible camera and compare side-by-side with the ground
truth. This document specifies the inputs, outputs, evaluation metrics,
shot split, and the notebook layout.

---

## 1. Demo target

**Task:** Forward-predict the **`camera_visible.camera_center.image_raw`**
frames of a held-out MAST shot, conditioned on:

1. **Initial state** — the first ~150 ms of the same shot's full
   diagnostic + control vector (Open-MAGVIT2 + Chronos + PatchTST tokens
   for that prefix).
2. **Ongoing control schedule** — the ground-truth control vector for
   the remainder of the shot (the model is *not* asked to invent the
   schedule).

**Output:** Predicted next-frame tokens for ~1 second of physical time,
decoded back to 256² RGB frames at 100 Hz model cadence; written
alongside the ground truth in a single Zarr archive plus a static MP4
overlay.

We pick `camera_center` (RBB) rather than `camera_lower` or
`camera_color` because RBB is the wide-angle midplane view that is most
generally informative about the plasma — the same view a human operator
would skim in post-pulse analysis.

---

## 2. Why a forward-prediction demo (not classification, not retrieval)

A forward-prediction demo is the cleanest test of "did the world model
learn dynamics, or did it learn an autoencoder". A reconstruction demo
proves only the tokenizer round-trip; a classification demo only proves
that a representation exists. Forward prediction requires the model to
extrapolate from prefix → suffix using the control schedule, which is
the only capability that actually unlocks the downstream pre-play /
disruption-exploration use cases.

---

## 3. Shot split

From the camera-bearing MAST shot subset (verified during the probe in
`data-acquisition.md` §3.4 — target ≥ 3 K shots):

| Split | Fraction | Selection rule |
|---|---|---|
| Train | 80 % | all camera-bearing shots not in val/test/demo |
| Val | 10 % | random sample with stratified campaign distribution (M5 – M9 equally represented) |
| Test | 10 % | held out for v1 quantitative evaluation; **not used in v0** |
| **Demo** | 3 specific shot ids | pinned-by-name in `plans/demo.md` (this file) |

Demo shots (subject to confirmation post-probe — the IDs below are
candidates with known good camera coverage from the FAIR-MAST tooling
examples and need replacement with whatever the probe surfaces):

- `30420` — appears in `bulk_download.html` quickstart; representative
  M9 shot, all level-2 groups present including `camera_visible`.
- `30421` — adjacent, used in the FAIR-MAST quickstart for the level-1
  examples. Provides continuity with the existing tutorial.
- One M6-era shot from the camera-bearing set — to be picked from the
  probe manifest based on `pulse_schedule` diversity.

The demo shots are *frozen for v0*. Changing them invalidates the demo
comparisons; if a demo shot turns out to lack camera coverage we
document the replacement in this plan with a dated update note.

---

## 4. Evaluation metrics

### 4.1 Reconstruction-quality metrics

| Metric | Computed on | Target |
|---|---|---|
| rFID (reconstruction FID) | predicted vs ground-truth frames | ≤ 8 |
| PSNR | per-frame, averaged across rollout | ≥ 14 dB |
| LPIPS | per-frame, averaged | ≤ 0.45 |

### 4.2 Physics-derived metrics

Compute these from both ground-truth and predicted frames using a fixed
post-processing pipeline (`imas_ambix/eval/metrics.py`):

- **Frame centroid trajectory**: brightness-weighted centroid `(x, y)`
  per frame. Compare predicted vs ground-truth as MSE of the trajectory
  over the rollout horizon.
- **Chord-integrated emission**: integrate the brightness along the
  midplane chord defined by `camera_center` geometry. Compare predicted
  vs ground-truth time-series as NRMSE.
- **Outer-boundary edge detection**: simple intensity-threshold edge.
  Compare predicted vs ground-truth edge displacement as
  median-absolute-deviation.

The point is not perfect physics. The point is that the predicted frames
contain *enough structure* that physically meaningful quantities can be
extracted, and those quantities track the truth at least in sign and
order of magnitude.

### 4.3 Qualitative

- A 4-panel matplotlib figure per demo shot: top = ground-truth frames
  at t=0.0, 0.2, 0.5, 1.0 s; bottom = predicted frames at the same
  times. Saved as `demo-{shot_id}.png`.
- A side-by-side MP4 (ground truth left, prediction right) at the model
  cadence. Saved as `demo-{shot_id}.mp4`.

We do not require the model to predict pixel-accurate plasma boundary in
v0. We require the model to predict a sequence that, looked at by a
plasma physicist, is recognisably plasma evolving in roughly the right
direction.

---

## 5. CLI surface

```bash
# Pull the demo shot's ground-truth tokens (already part of the
# mast-tokens corpus) and run the rollout
ambix demo wham-mast --shot 30420 \
    --checkpoint /work/projects/imas_gpu/mast-checkpoints/v0-500m/step-30000 \
    --prefix-ms 150 \
    --rollout-ms 1000 \
    --output /work/projects/imas_gpu/mast-demos/v0/30420/
```

Generates:

```
/work/projects/imas_gpu/mast-demos/v0/30420/
├── ground-truth.zarr          # frame tensor (decoded)
├── prediction.zarr            # frame tensor from rollout
├── tokens-ground-truth.zarr   # raw token sequence
├── tokens-prediction.zarr     # raw token sequence
├── metrics.json               # rFID, PSNR, LPIPS, centroid MSE, chord NRMSE
├── demo-30420.png             # 4-panel ground-truth vs prediction
└── demo-30420.mp4             # side-by-side video
```

The corresponding `ambix demo` lazy group is defined in
`imas_ambix/cli.py` alongside the existing `imas-ambix agent` group.

---

## 6. Demo notebook outline

`docs/demo.ipynb` — generated by `ambix demo notebook --shot 30420` so
the notebook is reproducible and version-controlled by configuration,
not by hand-editing.

Cells:

1. Markdown header: shot id, model checkpoint, date, manifest hashes.
2. Load ground-truth and predicted Zarr archives.
3. Show 4-panel figure inline (the same one written to PNG).
4. Embed MP4 via IPython's `Video` widget.
5. Print metrics dict as a markdown table.
6. Markdown discussion: what's qualitatively right, what's qualitatively
   wrong, residual artefacts. **This is the only hand-edited cell** —
   it captures the human commentary that the metrics cannot.

The notebook is intended for emailable distribution. It contains no
secrets and no private data.

---

## 7. Acceptance for v0 sign-off

All of the following must hold simultaneously for one demo shot before
v0 is declared complete:

1. rFID ≤ 8 on the rollout (computed against the corresponding
   ground-truth frames).
2. Centroid trajectory MSE within 2× of the ground-truth centroid
   variance across the rollout window.
3. The MP4 is "recognisable as plasma" by an unprompted physicist
   colleague — qualitative sign-off, captured in the notebook's
   commentary cell with name and date.
4. The full demo pipeline runs end-to-end from `ambix demo wham-mast` in
   < 5 min on the 4 × H200 reservation.

Failing 1 or 2 is a science problem (loss curve insufficient, more
training needed). Failing 3 is the strongest argument for revisiting the
tokenizer choice. Failing 4 is an engineering problem with the rollout
code path.

---

## 8. Demo-day sequencing

| Step | Owner | Wall time |
|---|---|---|
| Pull latest checkpoint, freeze it to a tagged release | maintainer | 5 min |
| Run `ambix demo wham-mast` for all three demo shots in turn | maintainer | 15 min |
| Render the notebook | maintainer | 5 min |
| Email the notebook + one MP4 to the stakeholder list | maintainer | 5 min |

This pipeline is intentionally short — the v0 demo is a smoke test, not
a polished release.

---

## 9. Related plans

- `world-model-v0.md` — defines the model that produces the rollout.
- `tokenizers.md` — defines the encode/decode used in this demo.
- `data-acquisition.md` — the shot split and the demo-shot list depend
  on the probe-time manifest, which is what tells us which shots really
  do carry camera frames.
