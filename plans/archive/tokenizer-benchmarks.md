# Plan: Tokenizer Benchmark Framework

Status: **In flight** — implementation in `imas_ambix/bench/`, tracked by
`sonnet-bench-framework` concurrent agent.

This plan defines the quantitative comparison framework for tokenizer
evaluation. Tokenizer choice is the highest-leverage architectural decision
in the entire pipeline: the world model trains on token IDs, and a tokenizer
that produces poor reconstructions means the model never sees high-fidelity
plasma imagery regardless of how well it trains. Closing the loop with
numbers — not intuition — is therefore a prerequisite to committing to any
tokenizer configuration beyond the v0 defaults.

The benchmark framework runs *outside* of the pytest suite (no GPU in CI)
but is a first-class software artefact: version-controlled, reproducible,
and hooked into a persistent results archive.

---

## 1. Why this exists

### 1.1 Quantitative tokenizer comparison

The current Open-MAGVIT2 smoke test reports MAE 324 on a single rbb shot
(per `tokenizers.md` §9.1) and the Chronos round-trip gives Pearson r > 0.98
on a synthetic sine/cosine. These figures answer "does the code run" — they
do not answer the design questions that matter for v0:

- Is rFID 324 MAE acceptable, or does it hide structural plasma-boundary
  artefacts that the world model cannot recover?
- Does Chronos reconstructruct the *headline* diagnostic channels (plasma
  current, PF coil currents, gas injection) with sufficient fidelity, or
  only the smooth synthetic channels?
- How do throughput numbers translate to GPU-hours for the ~450,000-frame
  rbb corpus?
- When we eventually fine-tune Open-MAGVIT2 on plasma imagery (§6.1), by
  how much does rFID improve?

The benchmark framework answers all of these with structured, archived,
comparable results.

### 1.2 Gating tokenizer promotion

The acceptance gates in `tokenizers.md` §7 (`rFID ≤ 5`, `NRMSE ≤ 0.05`) are
stated as thresholds for graduating the tokenizer stage to world-model
training. Without a rigorous measurement framework those gates are
aspirational text. The benchmark framework makes them executable: a
`BenchResult` either passes or fails each gate, and the pass/fail decision
is recorded in the archive.

### 1.3 Comparative baselines for future tokenizer upgrades

When the plasma-domain decoder fine-tune runs (`tokenizers.md` §12.1, this
doc §4.1), the improvement in rFID relative to the ImageNet checkpoint needs
to be measured against a stable baseline. The benchmark framework provides
that baseline: every run is archived with a config hash, so the same shot
subset at the same resolution against the same reference produces comparable
numbers regardless of which checkpoint is being tested.

---

## 2. Framework architecture

The framework lives under `imas_ambix/bench/`. The five modules are:

```
imas_ambix/bench/
├── __init__.py          # public API: BenchConfig, BenchResult
├── config.py            # BenchConfig dataclass + YAML loader
├── frame.py             # benchmark_frame_tokenizer
├── signal.py            # benchmark_signal_tokenizer
├── runner.py            # SLURM batch runner + results archiver
└── cli.py               # ambix tokenize bench (sub-command of tokenize group)
```

### 2.1 BenchConfig

```python
@dataclass(frozen=True)
class BenchConfig:
    # Shot selection
    shot_ids: list[int]                  # pinned shot list for reproducibility
    camera: str = "rbb"                  # which camera group for frames

    # Tokenizer under test (any registered name from TokenRegistry)
    frame_tokenizer_name: str = "frames_open_magvit2_v1"
    signal_tokenizer_name: str = "signals_chronos_v1"

    # Frame benchmark settings
    frame_spatial_compression: int = 16  # Open-MAGVIT2 actual
    frame_temporal_compression: int = 1  # image tokenizer
    frame_target_height: int = 256       # resize before encode
    frame_target_width: int = 256

    # Signal benchmark settings
    signal_groups: list[str] = field(
        default_factory=lambda: ["magnetics", "summary", "pf_active"]
    )
    signal_n_bins: int = 4096            # Chronos default

    # Infrastructure
    device: str = "cuda"                 # or "cpu" for smoke runs
    batch_size: int = 32                 # frames per encode call
    n_warmup_batches: int = 3            # discarded for throughput measurement

    # Acceptance gates (cross-references tokenizers.md §7)
    rfid_max: float = 5.0
    psnr_min_db: float = 20.0
    lpips_max: float = 0.35
    signal_pearson_r_min: float = 0.90
    throughput_frames_per_s_min: float = 100.0
```

Configs are stored as YAML in `imas_ambix/bench/configs/`:

```
imas_ambix/bench/configs/
├── v0-rbb-100shot.yaml       # reference: 100-shot rbb baseline
├── v0-rbb-10shot-smoke.yaml  # fast smoke run (10 shots, CPU)
└── v0-signals-100shot.yaml   # signal tokenizer baseline
```

### 2.2 BenchResult

```python
@dataclass
class BenchResult:
    config_hash: str             # SHA-256 of the serialised BenchConfig
    tokenizer_name: str
    tokenizer_version: str       # from TokenRegistry
    timestamp: str               # ISO-8601 UTC
    shot_ids: list[int]

    # Frame metrics (None if frame benchmark not run)
    rfid: float | None
    psnr_db: float | None
    lpips: float | None
    mae_per_pixel: float | None
    codebook_utilisation: float | None   # fraction of codebook entries used
    compression_ratio: float | None      # bytes_in / bytes_out

    # Signal metrics (per-channel dicts keyed by "{group}/{variable}")
    mae_per_channel: dict[str, float] | None
    nrmse_per_channel: dict[str, float] | None
    pearson_r_per_channel: dict[str, float] | None
    signal_codebook_utilisation: float | None

    # Throughput metrics
    encode_fps: float | None             # frames per second (encode)
    decode_fps: float | None             # frames per second (decode)
    total_wall_s: float

    # Acceptance gate outcomes
    gates_passed: dict[str, bool]        # keyed by gate name
    all_gates_passed: bool
    failure_reasons: list[str]
```

Results are JSON-serialisable and archived at:
```
/work/projects/imas_gpu/mast-tokens/v1/benchmarks/
└── {tokenizer_name}-{config_hash[:8]}-{timestamp}.json
```

### 2.3 benchmark_frame_tokenizer

Signature:
```python
def benchmark_frame_tokenizer(
    tokenizer: FrameTokenizer,
    shot_ids: list[int],
    config: BenchConfig,
    level1_root: str = "/work/projects/imas_gpu/mast/level1/shots",
) -> BenchResult:
    ...
```

Steps:
1. For each shot: open the rbb Zarr, resize frames to 256×256.
2. Encode in `config.batch_size` batches; record encode time.
3. Decode; record decode time.
4. Compute rFID, PSNR, LPIPS, MAE on the decoded vs original frames.
5. Compute codebook utilisation (number of unique token IDs used / codebook
   size) and compression ratio (raw uint16 bytes / encoded int32 bytes).
6. Aggregate across shots; apply acceptance gates.

rFID is the primary gate. It requires an Inception v3 feature extractor
(via `torchvision`). rFID is computed shot-by-shot against the
ground-truth frames from the same shot (not against ImageNet statistics).
This is the *reconstruction* FID — lower is better, and it measures how
much the round-trip degrades the distribution of frame features.

### 2.4 benchmark_signal_tokenizer

Signature:
```python
def benchmark_signal_tokenizer(
    tokenizer: SignalTokenizer,
    shot_ids: list[int],
    config: BenchConfig,
    level2_root: str = "/work/projects/imas_gpu/mast/level2/shots",
) -> BenchResult:
    ...
```

Steps:
1. For each shot and each group in `config.signal_groups`: open the Zarr,
   resample to the 100 Hz model grid.
2. Encode; decode.
3. Compute MAE, NRMSE, Pearson r per channel.
4. Codebook utilisation: fraction of bin IDs used out of `n_bins`.
5. Apply acceptance gates: `pearson_r >= signal_pearson_r_min` for all
   headline channels.

"Headline channels" are: `summary/ip` (plasma current), `summary/b0`
(toroidal field), `pf_active/*/current` (PF coil currents),
`pulse_schedule/tf/b_field_tor` (target toroidal field), and
`summary/global_quantities/li_3` (internal inductance). These are the
control-adjacent signals most tightly coupled to the plasma macro-state.

### 2.5 render_comparison_table

```python
def render_comparison_table(results: list[BenchResult]) -> str:
    ...
```

Renders a markdown table comparing multiple `BenchResult` objects. Columns:
tokenizer name, rFID, PSNR, LPIPS, Pearson-r (mean across headline channels),
encode fps, gates. Used by the `ambix tokenize bench compare` sub-command.

---

## 3. Metric definitions

### 3.1 Frame metrics

**rFID (reconstruction FID).** Fréchet distance between the Inception v3
feature distributions of the original frames and the decoded frames, both
from the same shot set. This is *not* the generative FID (which compares to
a reference dataset like ImageNet); it measures how much the tokenizer
round-trip shifts the feature distribution. Lower is better. Reference:
`imas_ambix/eval/metrics.py:rFID`.

**PSNR.** Peak signal-to-noise ratio between original and decoded frames,
computed per-frame and averaged. In dB; higher is better. Formula:
`10 * log10(MAX² / MSE)` where `MAX = 255` for uint8 frames.

**LPIPS.** Learned perceptual image patch similarity (VGG backbone). In [0,1];
lower is better. Captures perceptual quality beyond pixel-level MSE. Reference:
`imas_ambix/eval/metrics.py:LPIPS`.

**MAE per pixel.** Mean absolute error between original and decoded frames,
averaged over all pixels and frames. In the same units as the input (uint16
for raw rbb data; uint8 for the 256×256 resized input to Open-MAGVIT2).

**Codebook utilisation.** The fraction of the codebook's entries that appear
at least once in the encoded output for the benchmark shot set. An
Open-MAGVIT2 codebook with 2^18 = 262,144 entries will never be fully
utilised on a 100-shot rbb sample; a utilisation below 1 % suggests the
encoder is collapsing tokens into a small effective vocabulary, which hurts
generation diversity.

**Compression ratio.** Raw bytes in (uint16 frames, no compression) divided
by encoded bytes out (int32 token IDs, no compression). Open-MAGVIT2 at
16×16 spatial compression on a 256×256 frame: `(256×256×2 bytes) /
(16×16×4 bytes) = 32`. This is the theoretical value; measure empirically
to confirm.

### 3.2 Signal metrics

**MAE per channel.** Mean absolute error between original and decoded signal,
per variable, in the variable's native units after inverse-normalisation.
The Chronos `MeanScaleUniformBins` normalisation must be inverted before
computing MAE.

**NRMSE per channel.** Normalised RMSE = `RMSE / std(original)`. Scale-free
metric. The acceptance gate in `tokenizers.md` §7 is `NRMSE ≤ 0.05` for
headline channels.

**Pearson r.** Linear correlation between original and decoded channel.
Gate: ≥ 0.90 for all headline channels. This is a loose gate; a pure
autoencoder achieves r > 0.999 on smooth time series. A shot tokenizer that
fails this gate is producing qualitatively wrong reconstructions.

**Signal codebook utilisation.** For Chronos: fraction of the 4,096 bins
used across all shots + channels. For PatchTST: always 1 (single identity
token per patch).

### 3.3 Throughput metrics

**encode_fps.** Frames per second for the encode call (excluding model load
time, which is amortised). Measured by timing the encode loop over
`n_warmup_batches + N` batches and reporting only the last `N` batches.

**decode_fps.** Same, for the decode call.

**total_wall_s.** Total elapsed time for the full benchmark including data
loading, encode, decode, and metric computation (but not model load).

---

## 4. Reference comparisons for v0

### 4.1 Frame tokenizer comparisons

All comparisons run on the **same 100-shot rbb subset** to ensure
comparability. The 100 shots are pinned in `v0-rbb-100shot.yaml` (sampled
from the training set only — never from val/test/demo).

| Comparison | Tokenizer A | Tokenizer B | Purpose |
|---|---|---|---|
| Baseline vs placeholder | `frames_open_magvit2_v1` (ImageNet ckpt) | `frames_placeholder_v1` | Confirm that Open-MAGVIT2 beats the placeholder on rFID |
| Plasma fine-tune delta | `frames_open_magvit2_v1` (ImageNet) | `frames_open_magvit2_plasma_v1` (plasma fine-tuned) | Measure rFID improvement from decoder fine-tune (§5 of this doc; see also `tokenizers.md` §12.1) |
| IR baseline | `frames_open_magvit2_v1` on rir | same | Confirm shared visible/IR codebook is acceptable; gate for IR codebook decision (`tokenizers.md` §12.5) |

The Cosmos-Tokenizer-DV alternative is explicitly **not** benchmarked in v0
because its NVIDIA OML weights are not pure-Apache. If the rFID after
fine-tune is still > 5, Cosmos becomes the escalation path — add a
comparison run at that point.

### 4.2 Signal tokenizer comparisons

All comparisons run on the **same 100-shot level-2 subset**.

| Comparison | Tokenizer A | Tokenizer B | Purpose |
|---|---|---|---|
| Chronos vs uniform | `signals_chronos_v1` | `signals_uniform_v1` | Confirm Chronos beats the placeholder on Pearson r |
| PatchTST identity check | `signals_patchtst_v1` | N/A | Confirm exact round-trip (Pearson r = 1.0, MAE = 0) |
| Chronos vs PatchTST-real | `signals_chronos_v1` | `signals_patchtst_real_v1` (HF model, see `tokenizers.md` §12.2) | When PatchTST-real lands: compare reconstruction + codebook utilisation |

### 4.3 Throughput targets

| Tokenizer | Target (GPU) | Target (CPU) | Rationale |
|---|---|---|---|
| Open-MAGVIT2 encode (256×256) | ≥ 100 fps | ≥ 2 fps | 100 fps on GPU makes the 450k-frame encode pass finish in ~75 min (see `tokenizers.md` §9.1) |
| Chronos encode (100 Hz, 1 s window) | ≥ 500 steps/s | ≥ 200 steps/s | Signal encoding is not the bottleneck; 500 steps/s gives 1,000× margin on GPU |
| PatchTST patch-project (64-sample patches) | ≥ 50,000 patches/s | ≥ 20,000 patches/s | Magnetics has ~1.7K patches per shot; throughput is not a concern |

---

## 5. Acceptance gates per tokenizer

Gates cross-reference `tokenizers.md` §7. A benchmark pass requires:

### 5.1 Frame tokenizer (Open-MAGVIT2)

| Gate | Threshold | On fail |
|---|---|---|
| rFID on 100-shot rbb | ≤ 5.0 | Trigger decoder fine-tune (§4.1 this doc; `tokenizers.md` §12.1) |
| PSNR | ≥ 20 dB | Informational; not a hard gate |
| LPIPS | ≤ 0.35 | Informational |
| Codebook utilisation | ≥ 0.5 % | If < 0.5 %: codebook collapse suspected — investigate encoder temperature |
| encode_fps on GPU | ≥ 100 fps | If < 100 fps: profile the worker.py bridge; the bottleneck is likely the temp-file IPC — consider memory-mapped approach |

**Passing rFID ≤ 5** is the gate that unlocks tokenizer graduation to world-model
training (`tokenizers.md` §7, `world-model-v0.md` §5).

### 5.2 Signal tokenizer (Chronos)

| Gate | Threshold | On fail |
|---|---|---|
| Pearson r (headline channels) | ≥ 0.90 for all | Investigate per-channel calibration; check for out-of-range scaling |
| NRMSE (headline channels) | ≤ 0.05 for all | If > 0.10: check bin count — may need `n_bins=8192` rather than 4096 |
| Codebook utilisation | ≥ 5 % | If < 5 %: signals cluster in a narrow quantile range; consider per-campaign recalibration |

### 5.3 PatchTST (identity passthrough)

| Gate | Threshold | Notes |
|---|---|---|
| Pearson r | = 1.000 (exact) | Identity: no quantisation, no lossy encoding |
| MAE | = 0.000 (exact) | Same |

A failure here indicates a bug in the patch-slice / reassembly logic.

---

## 6. CI integration

### 6.1 Why not in pytest

The full benchmark requires a GPU node, the Open-MAGVIT2 weights (921 MB),
and a sample of the actual MAST corpus. All three are unavailable in CI.
Running a CPU-only smoke test (10 shots, placeholder tokenizers) in pytest is
acceptable for regression detection — this is implemented as `test_bench.py`
with the real tokenizers guarded by `pytest.mark.skipif(no_gpu)`.

The quantitative benchmarks with real weights run only on SLURM.

### 6.2 SLURM batch job design

The benchmark job is submitted via `ambix tokenize bench run`:

```bash
ambix tokenize bench run \
    --config imas_ambix/bench/configs/v0-rbb-100shot.yaml \
    --output /work/projects/imas_gpu/mast-tokens/v1/benchmarks/ \
    --slurm                          # submits to gpu_0003_grpA, exclusive
```

The `--slurm` flag wraps the benchmark in an `sbatch` script that:
1. Stops the DeepSeek V4-Flash serving job (per `compute.md` §2 protocol).
2. Runs the benchmark.
3. Restarts the serving job.

The benchmark result JSON is written to the benchmarks archive; a summary
table is printed to the SLURM log and also appended to the job's W&B run
(project: `imas-ambix-benchmarks`).

### 6.3 Result archive and comparison

Results accumulate in `/work/projects/imas_gpu/mast-tokens/v1/benchmarks/`.
Compare any two results:

```bash
ambix tokenize bench compare \
    /work/projects/imas_gpu/mast-tokens/v1/benchmarks/frames_open_magvit2_v1-*.json
```

This renders a markdown comparison table (via `render_comparison_table`) to
stdout. Useful for communicating rFID deltas to stakeholders without
manually extracting numbers from JSON.

---

## 7. Open questions to revisit

1. **rFID reference dataset.** The current rFID implementation computes
   reconstruction FID against the same-shot ground truth. An alternative
   is to compute against all 100 benchmark shots (pooled). The pooled
   version is more stable (larger sample) but less sensitive to per-shot
   degradation. Decide after the first 100-shot run — if per-shot rFID
   variance is low, switch to pooled.

2. **Throughput measurement validity.** The Open-MAGVIT2 worker uses temp-file
   IPC (see `tokenizers.md` §9.1 process isolation note). The measured
   encode_fps on the GPU node includes subprocess startup time per batch.
   For the benchmark to measure true GPU throughput, the worker must be kept
   alive across batches (persistent mode). Implement before reporting any
   throughput numbers as "GPU throughput".

3. **Signal benchmark temporal scope.** The 100 Hz model grid over a 1 s
   window gives 100 timesteps per benchmark window. Is that enough to measure
   Pearson r reliably for slowly-varying channels (e.g. PF coil current ramps)?
   Consider using full-pulse windows (up to 2 s) for signal benchmarks.

4. **MAE unit reporting.** MAE is computed after inverse-normalisation, so it
   carries physical units (e.g. kA for plasma current). The benchmark result
   must store the unit string alongside the MAE value so the comparison table
   is interpretable. Add a `units` dict to `BenchResult`.

5. **Benchmark cadence.** Should benchmarks run automatically after every
   tokenizer weight update, or only on demand? An automatic post-update run
   would close the loop cleanly but costs ~30 min of GPU time per run. Design
   as on-demand for v0; automate if the workflow stabilises.

6. **IR camera benchmark.** The rir source has only 25 shots in FAIR-MAST
   (per `data-acquisition.md` §10.4). A 100-shot rFID comparison is not
   possible. Define a separate `v0-rir-25shot.yaml` config and lower the
   expected rFID threshold (fewer shots = noisier FID estimate; use a
   relaxed gate of rFID ≤ 10 for the IR codebook decision).

7. **Multi-modal coherence metric.** The benchmark currently tests each
   modality independently. A joint coherence metric — do signal tokens at
   timestep t correlate with frame tokens at t? — would strengthen confidence
   in the interleaved stream. This is defined in `tokenizers.md` §12.6 as
   a future metric. Cross-link here once implemented.
