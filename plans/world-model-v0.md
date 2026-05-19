# Plan: World-Model v0 — WHAM-style Decoder-Only AR Transformer

Status: **Draft** — depends on `tokenizers.md` round-trip passing.

This plan specifies the v0 Fusion World Model: a Microsoft-WHAM-flavoured
decoder-only autoregressive transformer trained on interleaved
multi-modal token streams from FAIR-MAST. The aim is *workflow proof*, not
parameter records. The success criterion is the demo described in
`demo.md` — produce a recognisable wide-angle-camera rollout for a
held-out shot given the diagnostic+control context.

The reference architecture is Kanervisto *et al.*, "World and Human Action
Models towards gameplay ideation", *Nature* **638** (2025), DOI
[`10.1038/s41586-025-08600-3`](https://doi.org/10.1038/s41586-025-08600-3).
We follow their recipe with three deliberate substitutions:

1. Frame tokenizer = **Open-MAGVIT2** (theirs is ViT-VQGAN).
2. Action stream = **diagnostic + control vector** (theirs is keyboard /
   controller buttons).
3. Backbone = **Llama-class decoder-only** built on HuggingFace
   `transformers` instead of WHAM's internal framework, because the
   training code is not open-released.

WHAMM (the real-time variant) replaces the dense decoder with a MaskGIT
parallel sampler. v0 stays with WHAM's dense AR sampling; WHAMM-style
parallel decoding is a v1 question once we have a stable baseline.

---

## 1. Model architecture

| Parameter | v0 value | v1 ambition |
|---|---|---|
| Backbone | Decoder-only Llama-class transformer | same |
| Param count | **125 M → 500 M** (curriculum) | 1 – 2 B |
| Hidden size | 1024 | 2048 |
| Layers | 24 | 32 |
| Attention heads | 16 (head_dim 64) | 32 (head_dim 64) |
| Vocab size | ~280 K (see `tokenizers.md` §4) | same |
| Positional encoding | RoPE (Llama default) | same |
| Norm | RMSNorm (Llama default) | same |
| Activation | SwiGLU (Llama default) | same |
| Context window | **16 K tokens** | 64 K |
| FFN ratio | 4× (Llama default) | same |
| Sliding-window attention | None (full attention) | Optional 4 K sliding window |
| Tying input/output embeddings | yes | yes |

The 125 M → 500 M curriculum exists because the cheapest way to validate
the data-loader + tokenizer + training-loop integration is to train a
tiny model first. 125 M trains end-to-end on 1 GPU in ~6 h on a 10-shot
subset — small enough to debug interactively, large enough to produce
a sane loss curve.

---

## 2. Token-stream layout

The model sees one long interleaved sequence per training sample.
Schema, per **timestep**:

```
<step_start> <frame_tokens> <signal_tokens> <action_tokens> <step_end>
```

with token counts as follows (single-camera, 4 kHz signal grid):

| Block | Token count |
|---|---|
| `<step_start>` | 1 |
| `<frame_tokens>` (Open-MAGVIT2 visible camera, 256² → 32² spatial after 8× compression, 4× temporal grouping → effectively 256 tokens per 4-frame chunk) | **256 per 4-frame chunk → 64 tokens / frame amortised** |
| `<signal_tokens>` (Chronos quantized: 30 diagnostic channels × 1 token / channel + PatchTST magnetics ~20 patch tokens per chunk) | ~50 |
| `<action_tokens>` (control vector: PF coil currents × 6 + gas valves × 4 + heating × 2 + pulse-schedule waypoint id, all quantized) | ~12 |
| `<step_end>` | 1 |

Per timestep ≈ **130 tokens**. At 4 kHz the model would see 4,000 timesteps
per second of physical pulse time — clearly too much. We downsample the
*model time grid* to 100 Hz (every 40th 4 kHz sample), giving:

- **Tokens per second of pulse time** = 130 × 100 = 13,000.
- **Context window 16 K tokens** = **~1.25 seconds of physical pulse time**.

That is just enough for a multi-second MAST pulse to fit in a sliding
window. The model can predict the future of a pulse by re-tiling sliding
windows during rollout (KV-cache carries state across tiles), the way
WHAMM does.

If 1.25 s windows turn out to be too short for the camera-frame demo,
the v0 mitigation is dropping the magnetics PatchTST patch-token rate
(halve to ~10 tokens per chunk), pushing the window to ~1.7 s. We do not
extend context window beyond 16 K in v0 because positional encoding
behaviour with RoPE past 16 K needs proper testing we are not budgeted
for.

### Why per-step framing (not strict frame-then-signal-then-action)

We tried both orderings on paper. The per-step `<frame, signal, action>`
ordering keeps each timestep's blocks contiguous in the sequence, which:

- Makes the causal mask correct without special-case masking.
- Lets the loss weighting (next-frame loss > next-signal loss > next-
  action loss) attach cleanly to each block.
- Matches WHAM's `(image, action, image, action, ...)` interleaving even
  though our content is richer.

---

## 3. Training data shape

Training samples are 16 K-token windows extracted from the per-shot
token streams under `/work/projects/imas_gpu/mast-tokens/`. Window
stride during training: 4 K tokens (so the model sees every part of every
shot at least 4× per epoch).

Estimated dataset sizes:

- **Camera-bearing MAST shots**: ~3,000 shots (verified via probe
  manifest).
- Average pulse length ~0.4 s of recorded physical time → 0.4 s × 13 K
  tokens = ~5 K tokens / shot.
- That gives ~15 M raw training tokens — far too small for a 500 M model.

Mitigations:

1. **Multi-resolution training**: also include 10 Hz-downsampled token
   windows (covers ~12.5 s per 16 K window), so longer-horizon dynamics
   show up in the loss.
2. **Multi-camera duplication**: each shot has up to 3 visible cameras.
   Treat each camera as an independent sample; codebook is shared.
3. **Cross-shot continuation training**: pack 4 short shots into one 16 K
   window with `<eos>` separator tokens. This is the WHAM trick for
   short-clip data.
4. **Phase 3 expansion**: once TCV / JT-60SA land, the corpus grows by
   1 – 2 orders of magnitude.

The v0 success criterion is *qualitative*. The model is not expected to
extrapolate plasma physics from 15 M tokens; it is expected to demonstrate
the pipeline.

---

## 4. Training recipe

### 4.1 Optimization

- Optimizer: **AdamW** with `betas=(0.9, 0.95)`, `weight_decay=0.1`,
  `eps=1e-8` (Llama-2 standard).
- Schedule: linear warm-up 2 % of steps → cosine decay to 10 % of peak
  LR.
- Peak LR: **3e-4** for 125 M; **1.5e-4** for 500 M.
- Loss: token-level cross-entropy. **Block-weighted**: w_frame = 1.0,
  w_signal = 0.3, w_action = 0.1, w_control = 0.0 (control tokens are
  ground-truth observations, not predictions — masked out of loss).
- Batch: micro-batch 4 sequences × 16 K tokens per GPU × 4 GPUs = 256 K
  tokens / step.
- Total steps: 30 K for 125 M; 60 K for 500 M.

### 4.2 Parallelism on 4 × H200

- **FSDP** via HuggingFace `accelerate` ≥ 1.0. ZeRO-3 sharding for
  weights, gradients, optimizer state.
- Activation checkpointing on (every other transformer block).
- bf16 mixed precision (FP8 considered for v1).
- Effective memory budget per GPU under FSDP + activation checkpointing
  for a 500 M model at bf16, 16 K context, micro-batch 4: ~85 GB —
  comfortable on 141 GB H200, but **does not co-locate with DeepSeek
  V4-Flash serving** (which already occupies ~41 GB / GPU). This is the
  driving reason for the dedicated reservation request in `compute.md`.

### 4.3 Configs

```
imas_ambix/train/configs/
├── v0-125m.yaml
└── v0-500m.yaml
```

Hydra-managed. The 125 M config inherits from the 500 M config and
overrides `hidden_size`, `num_hidden_layers`, `num_attention_heads`,
`max_steps`.

### 4.4 Checkpoints + logging

- W&B project: `imas-ambix-world-model`. Run names derived from the Hydra
  config hash plus the data manifest hash.
- Checkpoints: `safetensors` under
  `/work/projects/imas_gpu/mast-checkpoints/{run-id}/step-{step}/`, every
  2 K steps. Resume-friendly.
- Eval rollouts on the held-out demo shot every 5 K steps; latest
  rollout PNG/JSON in the W&B artefact.

---

## 5. Evaluation

We follow `demo.md` for the headline demo. The training-time evaluation
hooks (run every 5 K steps) are subset:

| Metric | What it measures | Acceptance for v0 demo |
|---|---|---|
| Validation loss | overall token CE | strictly decreasing across the cosine schedule |
| Held-out frame rFID | reconstruction quality at the next frame | ≤ 8 (qualitative recognisability) |
| Centroid MSE | physics-derived: emission centroid trajectory MSE vs ground truth | ≤ 2× the centroid trajectory variance of the held-out shot — i.e. trajectory is "in the same ballpark" |
| Chord-integrated emission MSE | physics-derived | order-of-magnitude agreement |

We **do not** require state-of-the-art frame-FID. The aim is a
sufficiently good baseline that the pipeline is unblocked, not a
publishable model.

---

## 6. Sequence-packing strategy

Each training-data file under `/work/projects/imas_gpu/mast-tokens/`
holds the per-shot token stream as a 1-D `int32` Zarr array. The data
loader:

1. Samples a shot id uniformly from the training set.
2. Samples a window start uniformly from `[0, len - 16K)`.
3. Yields a `(input_ids, labels, attn_mask, loss_mask)` tuple where
   `loss_mask` zero-weights the action and control tokens.

For continuation training (§3 item 3), the loader concatenates up to 4
short shots end-to-end with an `<eos>` separator and yields a single
window; the loss mask zeroes out any cross-shot positions to prevent the
model learning fake transitions between unrelated shots.

The loader is implemented in `imas_ambix/data/loaders.py` (created
later); the persistent token files are written by
`imas_ambix/tokenizer/...`. There is no per-step `xr.open_zarr` round
trip — we mmap the 1-D `int32` array and slice it directly.

---

## 7. Generation / rollout

At inference, the model receives:

- An initial context window of length `N_ctx_init` (typically 2 K
  tokens, ~0.15 s of physical time).
- A control schedule that extends beyond the context window (the
  diagnostic / control vector the operator wants to "play forward").

It then alternates:

1. **Predict next-frame block**: autoregress the 64-token frame block,
   feeding back into the prefix.
2. **Splice in known signal + action tokens for the next timestep**: do
   not let the model hallucinate the control input — the control is the
   "action" the human / scheduler is supplying.
3. Continue until `N_target` steps reached or `<eos>` predicted.

Output of rollout: the predicted frame-token sequence is decoded back
through the Open-MAGVIT2 decoder into RGB frames; the predicted signal
tokens are inverse-quantized via Chronos / PatchTST scaling info; both
are written as a Zarr next to the model run for downstream visualization.

The rollout code lives in `imas_ambix/eval/rollout.py`. It uses HF
`transformers` `model.generate()` with logits processors that:

- Force the signal and action token positions to the **ground-truth /
  scheduler-provided values** instead of sampling them.
- Use top-k sampling (k=64) with temperature 0.8 on frame tokens to
  reduce drift relative to greedy.

---

## 8. What v0 deliberately leaves out

- **No classifier-free guidance.** WHAM uses it; we will add it in v1
  once we have a clean baseline.
- **No latent-action learning.** Genie's "actions inferred from video"
  approach is appealing but adds engineering scope we do not have.
- **No physics losses.** No first-principles solver in the loop. Loss
  is pure token CE.
- **No multi-camera consistency loss.** Each camera is trained
  independently with the shared codebook. Multi-camera coherence is a
  v1 feature.
- **No equilibrium 2-D tokenization.** Equilibrium enters as a
  continuous tensor via cross-attention to the model's first 4 layers.
  Tokenizing 2-D fields is a v1 task.

---

## 9. References

- WHAM Nature paper: <https://doi.org/10.1038/s41586-025-08600-3>
- WHAMM technical article (real-time variant):
  <https://www.microsoft.com/en-us/research/articles/whamm-real-time-world-modelling-of-interactive-environments/>
- Open-MAGVIT2: <https://github.com/TencentARC/Open-MAGVIT2>
- HF `transformers` Llama implementation: `transformers.models.llama`
- HF `accelerate` FSDP guide:
  <https://huggingface.co/docs/accelerate/usage_guides/fsdp>
- Chronos: <https://github.com/amazon-science/chronos-forecasting>
- PatchTST paper (ICLR 2023): <https://arxiv.org/abs/2211.14730>
