# Plan: Multi-Modal Tokenizers

Status: **Draft** — depends on `data-acquisition.md` mirror completing.

The Fusion World Model treats every diagnostic stream as a sequence of
integer tokens drawn from a shared vocabulary. This plan fixes the
per-modality tokenizer choice, the codebook layout, the token-id
namespacing across modalities, the training data, and the persistence
format so that downstream model code (`imas_ambix/model/wham.py`) and
training code (`imas_ambix/train/loop.py`) consume a stable interface.

The choice is biased toward off-the-shelf, Apache-2.0-licensed
components. We do not invent a new tokenizer. If a chosen tokenizer
underperforms, the fallback is documented inline; the public interface
does not change when we swap implementations.

---

## 1. Modality breakdown and tokenizer choices

| Modality | Source IDS / column | Native shape | Tokenizer | Reason |
|---|---|---|---|---|
| Visible camera | `camera_visible.{camera_lower,camera_center,camera_color}.image_raw` | `(H, W, T)` per camera, native cadence | Open-MAGVIT2 (TencentARC) | 2^18 LFQ codebook, 8× spatial / 4× temporal compression, Apache-2.0, rFID 0.39 @ 8× ImageNet (state-of-the-art on open discrete tokenizers). |
| IR camera | `camera_ir.{camera_lower,etc.}.image_raw` | `(H, W, T)`, single channel | Open-MAGVIT2 (same checkpoint, fine-tuned on IR) | Sharing the visible tokenizer is the v0 default; IR-specific fine-tune deferred to v1. |
| Magnetics (high rate) | `magnetics.flux_loop[*].flux.data`, `magnetics.b_field_pol_probe[*].field.data`, OMAHA channels | 1D, MHz cadence, up to 7.26 M samples / shot | PatchTST (patch length 64, stride 32) | Channel-independent patches; fastest to retrofit; output dimension feeds a learned linear projection into the token id space. |
| Low-frequency signals | `equilibrium.time_slice[*].global_quantities.*`, `pf_active.coil[*].current.data`, `summary.*`, `gas_injection.*`, `thomson_scattering.*` (profile-time series) | 1D or low-D, 4 kHz interpolated grid | Chronos T5-small (Amazon) | Scale-and-quantize → discrete token IDs already, Apache-2.0, HF `transformers`-native. |
| Scalar state / control actions | `pulse_schedule.*`, time-step index, shot id, campaign tag | scalar per step | Learned embedding table | Trivial; treat each scalar field as a tiny categorical with a learned embedding row. |

We do **not** tokenize equilibrium 2-D profiles (`profiles_2d[*].j_phi`,
`psi`, …) in v0. They go into a separate "auxiliary" conditioning path
read directly as continuous tensors by the model's cross-attention. v1 may
revisit this once Open-MAGVIT2 has been fine-tuned for grid-shape data.

---

## 2. Why Open-MAGVIT2 (not Cosmos-Tokenizer, not VQ-GAN)

The three credible discrete frame tokenizers for v0:

| Option | License | Compression | Codebook | Eval (rFID lower is better) |
|---|---|---|---|---|
| Open-MAGVIT2 (TencentARC) | **Apache-2.0** | 8×8×4 | 2^18 LFQ | **0.39** ImageNet 256×, 0.50 on 128×-class |
| Cosmos-Tokenizer-DV (NVIDIA) | Apache-2.0 code + NVIDIA OML weights | 8×8×8 | 64 K | 0.50-class |
| ViT-VQGAN (WHAM original) | MIT/Apache mixes | 16× spatial | ~16 K | ~1.5 |

Open-MAGVIT2 wins on every axis that matters for v0: pure-Apache stack,
strongest reconstruction on the open benchmarks, and the same temporal
compression (4×) that WHAM/WHAMM are converging on. The Cosmos OML weights
are commercial-friendly but bind us to NVIDIA's terms; the v0 plan stays
pure-Apache so we can publish weights and inputs together without a
license question. ViT-VQGAN is the WHAM reference but its rFID is too
loose to keep plasma camera artefacts intact at 8× compression.

Fallback ladder if Open-MAGVIT2 reconstruction on plasma imagery
disappoints (rFID > 10 on the held-out MAST sample):

1. Fine-tune the Open-MAGVIT2 *decoder only* on a few thousand plasma
   frames — keep the encoder frozen so the codebook stays stable.
2. If that fails, switch to **Cosmos-Tokenizer-CV (continuous, 16-channel
   latent)**. This drops the discrete codebook but lets us pair with a
   continuous-token AR head (e.g. NextStep-1 style) without throwing away
   the rest of the model code.
3. If that fails, escalate — the v0 demo target moves to magnetics-only
   forecasting and the camera output becomes a v1 concern.

Reference repos:

- <https://github.com/TencentARC/Open-MAGVIT2> — primary
- <https://github.com/NVIDIA/Cosmos-Tokenizer> — fallback continuous variant
- <https://arxiv.org/abs/2409.04410> — Open-MAGVIT2 paper

---

## 3. Why Chronos + PatchTST for signals

The signal landscape is heterogeneous: 4 kHz interpolated low-frequency
signals (most diagnostics), MHz raw magnetics (OMAHA), and 5 ms / 65×65
equilibrium 2-D grids. No single signal tokenizer covers all three well.
The v0 choice:

| Tokenizer | Best fit modality | Why |
|---|---|---|
| **Chronos T5-small** | 4 kHz interpolated signals | Scale + uniform quantize → token IDs, already T5-shaped, drop-in HF `transformers`. Apache-2.0. |
| **PatchTST** | MHz magnetics, OMAHA, raw probe signals | Channel-independent patch tokens; patch length 64 means ~110 K tokens per shot at MHz cadence is reduced to a tractable ~1.7 K patch tokens. |
| **(none — direct tensor)** | Equilibrium 2-D, scalar metadata | Skip discrete tokenization for v0. |

Chronos paper / repo: <https://github.com/amazon-science/chronos-forecasting>.
PatchTST: ICLR 2023 paper; HF wrapper available.

We deliberately reject Moirai-MoE for v0 even though its "any-variate"
attention is the cleanest answer to multi-rate signals. The reason is
operational: Moirai is BSD-3 licensed but its open implementation is
smaller and less HF-integrated than Chronos. Adopting it adds engineering
overhead we cannot afford in v0. v1 should revisit.

---

## 4. Token id namespacing

A single global vocabulary serves all modalities. Each tokenizer is
assigned a contiguous id range; the model never sees the modality string,
only token ids. The mapping is captured in
`imas_ambix/tokenizer/registry.py`:

```text
[       0,        4) — control tokens: <pad>, <bos>, <eos>, <sep>
[       4,    4+N1) — Open-MAGVIT2 frame tokens (visible)         N1 ≈ 2^18
[ 4+N1,   4+N1+N2) — Open-MAGVIT2 frame tokens (IR — shared CB?)  N2 = N1 in v0
[ 4+N1+N2,   ...  ) — Chronos signal tokens, per-channel offset by channel-id
[          ...    ) — PatchTST patch ids (discretised via FSQ projection)
[          ...    ) — Scalar / metadata embeddings (no shared vocab — own table)
```

Open-MAGVIT2's LFQ codebook is shared between visible and IR by default in
v0 to keep the vocabulary compact. If reconstruction shows the
visible/IR distribution gap is large enough to need separate codebooks,
we extend the registry to allocate a second range for IR; the registry
file is the single source of truth.

The registry is a pure Python module — no runtime DB. Versioning is by
the registry module's `VOCAB_VERSION` constant. Any change increments the
version and requires re-tokenisation of cached tokens.

---

## 5. Persistence layout

Tokenized data lives under
`/work/projects/imas_gpu/mast-tokens/`, parallel to but separate from the
raw mirror at `/work/projects/imas_gpu/mast/`. Layout:

```
/work/projects/imas_gpu/mast-tokens/
├── vocab/
│   └── v1/
│       ├── registry.json           # exported from tokenizer/registry.py
│       ├── open-magvit2-decoder.safetensors
│       └── chronos-t5-small.safetensors
├── frames/
│   └── v1/
│       └── {shot}/{camera}.zarr        # int32 token ids, shape (T, h, w) where h=H/8, w=W/8, with t-axis compressed 4×
├── signals/
│   └── v1/
│       └── {shot}/{group}.zarr         # int32 token ids per signal channel
└── manifests/
    └── v1-{ts}.json                    # per-shot tokenization metadata, vocab hash, source mirror checkpoint
```

Why Zarr again: the encode pass is incremental — token streams are large
(per-shot frame tokens for one camera can run to several MB), the model
loads them with the same `xr.open_zarr` machinery as the raw data, and
the chunking lines up with the model's sequence-window stride.

`v1/` is the vocab generation. Re-encoding under `v2/` lets us keep
old token caches around for direct A/B comparisons. The per-vocab
`open-magvit2-decoder.safetensors` is the only weight file we need to
reconstruct frames from tokens — the encoder is not needed at training
time.

---

## 6. Training / fine-tuning the tokenizers

### 6.1 Frame tokenizer

- **v0 default:** **no training** — use the pretrained Open-MAGVIT2
  ImageNet checkpoint as-is, downsample MAST visible frames to its native
  256× input.
- **v0 with fine-tune:** decoder-only fine-tune on ~5 K MAST frames if the
  default rFID is > 5. Frozen encoder + frozen codebook, ~1 GPU-hour on
  the 4×H200 reservation; this fine-tune runs *before* the world-model
  training starts.
- **v1:** consider end-to-end re-train. Out of scope for this plan.

### 6.2 Signal tokenizer

- **Chronos:** ships pre-trained on a broad time-series corpus
  (electricity, traffic, etc.). v0 uses the **pretrained T5-small
  checkpoint as-is**. The "quantize then index" step is calibrated
  per-channel from the MAST training split (compute per-channel mean +
  std on the training shots; freeze for inference).
- **PatchTST:** v0 uses **random initialisation**. PatchTST is a small
  model (~1 M params) and trains during the world-model training itself
  through gradient backprop — the patch-projection layer is part of the
  WHAM transformer's input pipeline. This avoids a separate pre-training
  pass.

### 6.3 Retraining triggers

| Trigger | Action |
|---|---|
| New MAST campaign data lands (M10+) | Recompute Chronos per-channel calibration; do not retrain Open-MAGVIT2 unless rFID degrades. |
| New facility added (TCV, JT-60SA, …) | Recompute per-channel calibration; re-evaluate Open-MAGVIT2 decoder fine-tune for facility-specific cameras. |
| Vocab-impacting bug fix | Bump `VOCAB_VERSION`, re-encode in a new `mast-tokens/v{N+1}/` directory; keep `v{N}/` until model trained on `v{N+1}/` is validated. |

---

## 7. Round-trip evaluation

A standalone tool (`ambix tokenize round-trip --shot {id}`) encodes a
single shot through the full pipeline, decodes back, and reports
per-modality reconstruction metrics:

- Frame tokenizer: rFID and PSNR on each camera, per-shot histogram.
- Chronos signals: NRMSE per channel, autocorrelation deviation.
- PatchTST: NRMSE on the magnetics raw probe traces.

The acceptance gate for graduating the tokenizer stage to the
world-model training is `rFID ≤ 5 on visible camera frames` and
`NRMSE ≤ 0.05 on each of the headline diagnostic channels` (pulse
schedule control variables, PF coil currents, plasma current, line-
integrated density).

---

## 8. Open questions to revisit before Phase 2 kicks off

- **IR codebook sharing**: confirm after the round-trip that the visible
  codebook reconstructs IR frames acceptably. If not, allocate a
  separate range in the registry.
- **PatchTST patch length**: 64 is chosen by analogy with the LLM
  context-window economics. If the world-model rollout shows aliasing
  from the patching, halve to 32.
- **Equilibrium 2-D path**: in v0 the equilibrium grid feeds the model as
  a continuous tensor via cross-attention. v1 should consider a 2-D
  Open-MAGVIT2 fine-tune on the grid representation so equilibrium
  enters the token stream the same way frames do.

---

## 9. v0 scaffold landed — 2026-05-19

The `imas_ambix.tokenizer` package now exists with the protocol surfaces
and placeholder implementations described above:

| Module | Role |
|---|---|
| `base.py` | `Tokenizer` / `FrameTokenizer` / `SignalTokenizer` protocols, `EncodedFrames` / `EncodedSignals` dataclasses |
| `registry.py` | `TokenRegistry` global id allocator, `CONTROL_TOKENS` reserved range, `VOCAB_VERSION` |
| `alignment.py` | `TimeGrid`, `shot_time_window`, `resample_to_grid` |
| `frames.py` | `PlaceholderFrameTokenizer` (working) + `OpenMagvit2Tokenizer` (stub) |
| `signals.py` | `UniformQuantizer` (working) + `ChronosSignalTokenizer` (stub) |
| `multimodal.py` | `ShotTokenizer` — interleaves frame+signal tokens with `<bos>/<sep>/<eos>` |
| `cli.py` | `ambix tokenize {registry, inspect, frames, signals}` |

The two `*_v1` placeholder tokenizers (`frames_placeholder_v1`,
`signals_uniform_v1`) emit valid global token ids inside the
registry-allocated range, and round-trip cleanly:

```text
$ ambix tokenize frames --shot 15085 --camera rbb --temporal-compression 4 --spatial-compression 8
loaded rbb for shot 15085: shape=(149, 536, 560), dtype=uint16
encoded shape: (37, 67, 70)  vocab range used: [4, 259]
decode shape:  (148, 536, 560)  input vs decoded MAE: 631.09

$ ambix tokenize signals --shot 11766 --group summary --n-bins 64
input vars: 4
tokenized channels: 4
token shape: (1652, 4)
global id range: [14, 60]
vocab_size (per ch.): 64
```

These work without the Open-MAGVIT2 or Chronos weights — they exist so
the rest of the pipeline (model loader, training loop, evaluation) can
be exercised before the real tokenizers are plumbed in.

### 9.1 Open-MAGVIT2 — live (2026-05-20)

The real `OpenMagvit2Tokenizer` is now wired up. Staging layout:

```
/work/projects/imas_gpu/mast-tokens/v1/open-magvit2/
├── src/                       # github.com/TencentARC/Open-MAGVIT2 @ c1544ef
├── weights/
│   └── imagenet_256_L.ckpt    # 921 MB, 2^18 LFQ codebook, Apache-2.0
├── .venv/                     # uv venv, Python 3.11.5
│                              # torch 2.1.1+cpu, lightning 2.2.0,
│                              # numpy<2, setuptools<81, transformers 4.37.2
└── worker.py                  # ambix bridge — encode/decode via .npy tempfiles
```

Discovered details that the docs didn't quote:

| Setting | Reality |
|---|---|
| Tokenizer 256 checkpoint name | `imagenet_256_L.ckpt` (the README's `_B/_L/_XL` suffix refers to **AR head** sizes — there is no `imagenet_256_B.ckpt`) |
| Spatial compression | **16×** (256×256 → 16×16 tokens) — not 8× as the comparison table implies; 8× is only via the 128 model run at 256 input |
| Temporal compression | **1×** for the image tokenizer (we will revisit for v1 if we add the 5×16×16 video tokenizer) |
| `get_codebook_entry` API | `(indices: (B, h*w), bhwc: (B, h, w, embed_dim=18), order: "pre")` — undocumented but discovered in `lookup_free_quantize.py:192` |
| Setuptools | Needs `<81` because Lightning 2.2.0 still calls `pkg_resources.declare_namespace` |
| Numpy | Needs `<2` for torch 2.1.1's pre-numpy-2 ABI |

The `OpenMagvit2Tokenizer` is process-isolated:

- ambix `frames.py` runs in our main venv (torch ≥ 2.6).
- Each `encode()` / `decode()` call serialises numpy arrays through `.npy`
  temp files, invokes `.venv/bin/python worker.py {encode,decode}` in the
  isolated venv, then loads the result back.
- Per-call overhead on the login node CPU: ~5 s python startup + model load,
  plus ~30 s per frame for encode and similar for decode. The login node is
  fine for smoke tests; the production token pass for ~3,000 rbb-bearing
  shots × ~150 frames = ~450,000 frames needs the GPU node (where weights
  warm to GPU once and each batch is sub-second).

### 9.2 Chronos + PatchTST landed — 2026-05-20

Both signal tokenizers are now wired up in `imas_ambix/tokenizer/signals.py`
and covered by the test suite (`tests/test_tokenizer.py` — 31 tests, all
green).

#### Chronos T5-small

| Setting | Value |
|---------|-------|
| HF model id | `amazon/chronos-t5-small` |
| Package | `chronos-forecasting>=1.3` (Apache-2.0) |
| Install | `uv pip install chronos-forecasting` or `uv pip install "imas-ambix[train]"` |
| Vocab range allocated | `[0, 4096)` local → shifted into global registry |
| Tokenizer class used | `chronos.MeanScaleUniformBins` (no T5 weights needed for encode/decode) |
| Lazy-import pattern | `_build_chronos_tokenizer()` is called on first `encode` / `decode`; `ChronosUnavailableError` (RuntimeError subclass) raised if package absent |
| Per-channel calibration | `fit(datasets)` accumulates mean + std; normalised values fed to Chronos's internal mean-scale quantizer |
| Round-trip result | Pearson r > 0.98 on 64-step sine/cosine synthetic; quantisation is lossy (≥ 0.9 is the acceptance gate) |

The T5 *transformer* weights are **not** loaded by this tokenizer. Only
the `MeanScaleUniformBins` arithmetic (scale + uniform-bin assignment) is
used, constructed from the published config constants (`n_tokens=4096`,
`n_special_tokens=2`, `low_limit=-1.0`, `high_limit=1.0`). The config
matches `amazon/chronos-t5-small` exactly — no download required.

#### PatchTST (identity passthrough)

| Setting | Value |
|---------|-------|
| Registry name | `signals_patchtst_v1` |
| `vocab_size` | `1` (single "identity" id per patch) |
| `patch_size` | `64` samples |
| Token ids | All zeros (shifted into registry range) — the raw floats live in `metadata["patches"]` |
| Round-trip | Exact — `np.allclose` verified in `test_patchtst_roundtrip_exact` |

The patch-projection matrix trains end-to-end inside the WHAM transformer
(see `plans/world-model-v0.md` §2). The tokenizer's only job is to slice
each channel into `(n_patches, 64)` float arrays and preserve them for
the model's input pipeline.

### 9.3 Implementation notes for the real tokenizers

- The decoders need to run on **CPU during data prep** (the betelgeuse
  GPU node has no network access for weight downloads). Once the
  encoder weights are staged on GPFS, the actual encode pass for the
  full corpus moves to the GPU node.
- Encoded tokens persist under `mast-tokens/v1/` per §5. The PR that
  swaps placeholders for the real tokenizers also writes
  `mast-tokens/v1/registry.json` capturing the global vocabulary.
- The round-trip evaluation in §7 is the gate. The placeholder hits
  ~1% MAE on uint16 frames; Open-MAGVIT2 should hit rFID ≤ 5 once the
  encoder is plumbed in.
