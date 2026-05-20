# Plan: Data Quality Framework

Status: **In flight** — implementation in `imas_ambix/quality/`, tracked by
`sonnet-data-quality` concurrent agent.

This plan defines the pre-training data validation layer for the FAIR-MAST
training corpus. Before any shot's token stream enters the training set it
must pass a quality audit. Without this layer we risk training on corrupted
Zarr files, NaN-poisoned signal channels, or shots with degenerate physics
(sub-threshold plasma current, near-zero pulse duration). A world model
trained on such shots learns from noise, and the resulting loss curves are
indistinguishable from a legitimately bad model.

The quality framework is therefore not optional scaffolding — it is a
**load-bearing gate** in the `mast-tokens → training-set` pipeline.

---

## 1. Why this exists

### 1.1 Pre-training data validation

The tokenized corpus under `/work/projects/imas_gpu/mast-tokens/v1/` is a
derived product from the FAIR-MAST level-2 and level-1 mirrors. Zarr files
can be partially written, truncated by network interruption, or missing
expected groups — the bulk download job at `plans/data-acquisition.md` §4
uses resumable `s5cmd cp` which skips existing objects, so a shot that was
half-copied on a previous run will not be re-fetched. Every shot therefore
needs to be validated before it is allowed into the training index.

Additionally, FAIR-MAST covers MAST campaigns M5 – M9 spanning ~15 years.
Earlier campaigns have sparser diagnostics, lower plasma current, and shorter
pulses. Including every shot with equal weight risks overrepresenting low-
quality shots in a training batch and wastes the 16 K-token context window
on sub-threshold physics.

### 1.2 Surfacing failure modes early

The first full corpus audit will surface systematic issues we cannot see from
a 10-shot sample:

- Missing `equilibrium` group in ~1.7 % of shots (per the full inventory in
  `data-acquisition.md` §11.1).
- `charge_exchange` present in only 37.9 % of shots — the model must not
  treat absence as a zero measurement.
- Unknown but possible: NaN-poisoned time slices in `magnetics` from sensor
  failures, or discontinuous time axes from stitched acquisitions.

The quality framework converts these into per-shot flags that the data loader
reads to filter training samples. Problems are visible and actionable rather
than silently absorbed into gradient noise.

### 1.3 Building the train / val / test split decision

The quality report includes a corpus-level view: plasma-current histogram,
pulse-duration distribution, campaign distribution, and group-presence joint
distribution. This view is the input to the stratified split in
`plans/demo.md` §3 — equal campaign representation in val/test, demo shots
frozen from the camera-bearing subset, etc. The split decision cannot be made
correctly without knowing the quality distribution of the full corpus.

---

## 2. Framework architecture

The framework lives under `imas_ambix/quality/`. The four modules are:

```
imas_ambix/quality/
├── __init__.py          # public API: ShotQualityReport, CorpusAudit
├── shot.py              # per-shot checks → ShotQualityReport
├── corpus.py            # aggregate over all shots → CorpusAudit
└── cli.py               # ambix data audit (sub-command of the data group)
```

### 2.1 ShotQualityReport

A frozen dataclass capturing the quality outcome for a single shot:

```python
@dataclass(frozen=True)
class ShotQualityReport:
    shot_id: int
    zarr_open_ok: bool                   # level-2 Zarr opens without error
    groups_present: list[str]            # which level-2 groups were found
    nan_fraction: dict[str, float]       # per-group NaN fraction
    time_monotonic: dict[str, bool]      # per-group time axis check
    time_gaps: dict[str, int]            # number of >2-sample gaps per group
    dynamic_range_ok: dict[str, bool]    # per-variable range plausibility
    homogeneous_time_ok: bool            # ids_properties.homogeneous_time == 1
    dd_version: str                      # ids_properties.version_put.data_dictionary
    plasma_current_max_ka: float | None  # from summary.global_quantities.ip
    pulse_duration_ms: float | None      # inferred from summary time axis
    training_grade: bool                 # aggregate pass/fail (§4)
    failure_reasons: list[str]           # if not training_grade, why
```

All fields are populated in a single pass over the shot's Zarr files. The
report is JSON-serialisable (all values are primitives or lists/dicts of
primitives).

### 2.2 CorpusAudit

Aggregates across all shot reports:

```python
@dataclass
class CorpusAudit:
    timestamp: str                       # ISO-8601 UTC
    manifest_hash: str                   # hash of the level-2 manifest
    n_shots_total: int
    n_shots_training_grade: int
    n_shots_failed: int
    failure_mode_counts: dict[str, int]  # keyed by failure_reasons values
    nan_fraction_p50: dict[str, float]   # median per group
    nan_fraction_p95: dict[str, float]   # p95 per group
    plasma_current_histogram: list[float]# 20-bin counts, 0–10 MA
    pulse_duration_histogram: list[float]# 20-bin counts, 0–2000 ms
    campaign_distribution: dict[str, int]# counts per campaign tag
    group_presence_matrix: dict[str, float]  # group → fraction of shots
    shot_reports: list[ShotQualityReport]  # full per-shot data
```

Serialised to JSON and stored at
`/work/projects/imas_gpu/mast-tokens/v1/quality/audit-{ts}.json`. The
training data loader reads the **latest** audit from this directory on
startup.

### 2.3 CLI surface

```bash
# Full corpus audit (writes to mast-tokens/v1/quality/)
ambix data audit --manifest level2-all.json [--workers 16]

# Single-shot quick check (prints the ShotQualityReport as JSON)
ambix data audit --shot 30420

# Summary table from the latest audit
ambix data audit --summary

# Inspect per-group NaN fractions across the corpus
ambix data audit --nan-fractions
```

The `audit` sub-command is implemented in `imas_ambix/quality/cli.py` and
registered under the existing `ambix data` CLI group defined in
`imas_ambix/data/cli.py`.

---

## 3. Quality dimensions

### 3.1 Zarr-open / no-corruption

Every shot's level-2 Zarr directory is opened with `xarray.open_zarr` before
any other check. A failure here is an immediate `training_grade = False`.

Common failure modes:
- Truncated `.zarray` metadata (half-written file from interrupted download).
- Missing `.zmetadata` consolidated metadata (Zarr v2 default, required by
  xarray's consolidated mode).
- Byte-level corruption in a chunk file (rare but possible after an OOM kill).

Detection: open with `xr.open_zarr(path, consolidated=True)` and catch
`Exception`. On failure, log the error text into `failure_reasons`.

Level-1 camera Zarr files (the rbb/rba sources under
`/work/projects/imas_gpu/mast/level1/shots/{shot}/`) receive the same open
check. A shot whose level-2 opens but whose rbb Zarr is corrupt is still
`training_grade = True` for the signal-only training set; the frame
tokenizer marks its output as unavailable.

### 3.2 Per-variable NaN fraction

For each open group and each data variable, compute:

```python
nan_frac = float(np.isnan(ds[var].values).mean())
```

Groups checked for the training-grade gate (§4): `magnetics`, `summary`,
`pulse_schedule`, `equilibrium`. NaN fraction threshold: **< 5 %**.

Why 5 %? A NaN fraction of 5 % means that in a 1-second pulse at 100 Hz
(100 timesteps), 5 timesteps have missing values. The training window is
16 K tokens; a 5 % NaN rate will produce ~800 NaN-corrupted token positions.
At the per-channel mean-scale normalisation used by Chronos, NaN values in
the input are forward-filled by the loader — so they do not cause a gradient
explosion, but they do corrupt the signal. 5 % is the threshold below which
this corruption is tolerable as label noise; above it, the shot is excluded.

### 3.3 Dynamic-range plausibility

For a pre-defined set of "headline" variables, check that the observed
min/max falls inside a physics-reasonable interval:

| Variable | IDS path | Expected range |
|---|---|---|
| Plasma current | `summary.global_quantities.ip.value` | –20 MA to +20 MA |
| Line-integrated density | `summary.line_average.n_e.value` | 0 to 10²¹ m⁻² |
| Toroidal field | `summary.global_quantities.b0.value` | –10 T to +10 T |
| PF coil current | `pf_active.coil[*].current.data` | –200 kA to +200 kA |
| Electron temperature (TS) | `thomson_scattering.channel[*].t_e.data` | 0 to 30 keV |

Out-of-range values are not always wrong (e.g. a MAST shot with a disruption
spike could transiently exceed the PF limit), but values that are *orders of
magnitude* out of range — say, a plasma current of 10¹² A — indicate a unit
error or a corrupted scaling factor in the IMAS ingestion. Flag and exclude.

Implementation: soft thresholds (2 × range gives a warning, 10 × range gives
a hard fail). The thresholds are stored as constants in
`imas_ambix/quality/shot.py`.

### 3.4 Time-axis monotonic / no gaps

For each group, verify:

1. The time axis is strictly monotonically increasing.
2. No gap larger than 2× the median time step (`dt_median`).

Non-monotonic time axes indicate a stitching artefact or data-ingest bug.
Large time gaps (> 100 ms at the 100 Hz model grid) indicate that the
Chronos resampler will extrapolate across a region with no data — this can
introduce spurious transients in the token stream.

Any gap > 100 ms is recorded in `time_gaps` with the gap start time. Shots
with > 3 such gaps in any of the mandatory groups are `training_grade = False`.

### 3.5 IMAS attribute consistency (replaces IDS-format checks)

**Superseded by §10.1 findings.**  FAIR-MAST is xarray-on-Zarr, NOT IDS
format.  The original checks for `homogeneous_time` and `dd_version` have
been removed because these IDS attributes do not exist in the bucket.

The new checks are:

- `check_imas_label_matches_group`: verifies that the group-level
  `ds.attrs["imas"]` string matches the group's directory name.  Info-level
  only — a mismatch is a metadata note, not a training-grade failure.
- `check_imas_pointer(ds, var)`: reports whether a variable carries an
  `imas` path-pointer attribute.  Info-level; absence is not a failure.

Neither check affects `usable_for_training`.

### 3.6 Cross-shot corpus metrics

Computed at corpus level by `CorpusAudit`:

**Campaign distribution.** The val/test split in `plans/demo.md` §3 requires
stratified campaign sampling. The audit counts shots per campaign tag and
flags campaigns with < 50 shots (too small to split meaningfully).

**Plasma-current histogram.** 20 bins from 0 to 10 MA. Expected shape:
skewed, with a mode around 300 – 600 kA for MAST. Shots below 100 kA are
excluded by the acceptance gate (§4). The histogram is plotted in the audit
summary.

**Pulse-duration distribution.** 20 bins from 0 to 2,000 ms. MAST pulses
range from a few hundred milliseconds to ~1 second for long-pulse campaigns.
Very short pulses (< 100 ms) are excluded by the acceptance gate. The
distribution drives the context-window economics: the 16 K-token window at
100 Hz covers ~1.25 s, so most MAST pulses fit in 2 windows.

**Group-presence joint distribution.** A sparse matrix: for each pair
(group_A, group_B), what fraction of shots have both? This surfaces which
diagnostic pairs are always co-present (e.g. `magnetics` + `pf_active`) and
which are weakly correlated (e.g. `charge_exchange` + `equilibrium`). The
training loop cannot assume a fixed diagnostic set; this matrix informs the
design of the `ShotTokenizer`'s group-absence handling.

---

## 4. Acceptance for "shot is training-grade"

**Updated 2026-05-20 — see §10 for FAIR-MAST format reality.**

A shot is `usable_for_training = True` (new flag name) if and only if
**all** of the following conditions hold:

| Condition | Detail |
|---|---|
| All groups open | `all_groups_open == True` — every discovered Zarr group opens without error |
| `magnetics` present | `has_magnetics == True` — signal-level training requires magnetics |
| `pulse_schedule` present | group opens — pulse structure required for training window selection |
| No corruption errors | `no_corrupt_nans == True` — no all-NaN variables, no `\|value\| > 1e25`, no inf |

Dropped conditions (see §10 for rationale):

| Dropped condition | Reason |
|---|---|
| NaN fraction < 5 % | Per-variable all-NaN catches real corruption; partial NaN is acceptable |
| Plasma current > 100 kA | Not yet wired — parquet metadata not loaded by default audit path |
| Pulse duration > 100 ms | Not yet wired — as above |
| Time axis monotonic | All FAIR-MAST time axes are monotonic (verified on 50-shot sample) |
| Time gap ≤ 3 | No time gaps found — uniform Δt throughout corpus |
| Dynamic range in bounds (old) | Threshold 1e15 was below physical plasma densities; replaced by 1e25 hard cap |

Shots that fail are not deleted from the mirror — they remain in
`/work/projects/imas_gpu/mast/` but are excluded from the training manifest.
Their `failure_reasons` are recorded in the audit JSON. If a systematic
failure mode appears (e.g. all M5 shots have a NaN-poisoned
`pulse_schedule`), that is actionable data for the ingestion pipeline.

---

## 5. Operational protocol

### 5.1 First-time corpus audit

Run after the level-2 and level-1-camera downloads complete (tracked in
`data-acquisition.md` §11.4). Steps:

```bash
# 1. Allocate a sirius node (quality audit reads GPFS; not run on login node)
srun --partition=sun_debug --cpus-per-task=8 --mem=32G --time=01:00:00 --pty bash -l

# 2. Run the full audit (~16 workers; I/O-bound, not CPU-bound)
cd /home/ITER/mcintos/Code/imas-ambix
uv run ambix data audit \
    --manifest /work/projects/imas_gpu/mast/manifests/level2-all.json \
    --output /work/projects/imas_gpu/mast-tokens/v1/quality/ \
    --workers 16

# 3. Print summary
uv run ambix data audit --summary
```

Expected wall time: ~2 h for 11,573 shots at 16 workers (each shot check
involves an `xr.open_zarr` + a scan of 3 – 5 variables × 2 – 4 groups).

The audit writes to:
```
/work/projects/imas_gpu/mast-tokens/v1/quality/
└── audit-{YYYYMMDD-HHMMSS}.json    # full CorpusAudit JSON
```

### 5.2 Re-audit after new data ingestion

Any time new shots are added to the mirror (e.g. a new MAST campaign becomes
available, or the FAIR-MAST ingestion runs for camera sources that are
currently level-2 absent), re-run the audit with the updated manifest. The
audit compares the new manifest hash against the latest audit; if unchanged,
it is a no-op.

### 5.3 Training data loader integration

`imas_ambix/data/loaders.py:ShotTokenDataset` reads the latest audit on
construction:

```python
dataset = ShotTokenDataset(
    token_root="/work/projects/imas_gpu/mast-tokens/v1/",
    quality_dir="/work/projects/imas_gpu/mast-tokens/v1/quality/",
    # loads the most-recent audit-*.json, filters to training_grade=True
)
```

Shot ids with `training_grade = False` are silently excluded. The number of
excluded shots is logged at `INFO` level on construction so anomalies are
visible in the training log.

### 5.4 Audit artefact retention

Old audit JSON files are retained indefinitely — they provide a history of
data quality over time. The latest audit is determined by lexicographic
sort on the timestamp suffix. No automatic pruning.

---

## 6. Open questions to revisit

1. **Camera Zarr open check.** The level-1 rbb/rba files are raw arrays, not
   IMAS Zarr groups. The open check needs a separate code path (`xr.open_zarr`
   on the level-1 prefix vs. the level-2 IMAS layout). Confirm before the
   first audit run.

2. **DD version multiplicity.** If shots from different campaigns were
   ingested with different DD versions (e.g. M5 with `3.38.0`, M9 with
   `3.40.0`), the loader needs to open each shot with its stored DD version —
   not a single corpus-wide version. The audit surfaces the per-shot
   `dd_version`; the loader needs to consume this. Cross-reference to the
   AGENTS.md rule: "Always open data with the DD version it was written in."

3. **NaN threshold for magnetics.** The 5 % threshold is heuristic. If the
   magnetics `flux_loop` channels have systematic fill-value NaNs for probes
   that were not operational in some campaigns, 5 % may be too tight. Consider
   per-variable thresholds after the first corpus audit surfaces the actual
   distribution.

4. **plasma_current_max source.** The plan reads from `summary` — confirm
   that `summary.global_quantities.ip.value` is populated for all 100 % of
   shots where `summary` is present. If the field is often NaN, fall back to
   the `pf_active` derived current or skip the threshold check entirely.

5. **Partial-shot quality.** A shot may have a good `magnetics` group but a
   corrupt `equilibrium`. Should the shot be training-grade for signal-only
   windows? The current design uses a single `training_grade` boolean. A more
   granular `training_grade_signals / training_grade_frames` split would allow
   the loader to use such shots for signal training while excluding them from
   the mixed frame+signal windows. This adds complexity; defer to post-audit
   analysis.

6. **Quality metric evolution.** As the benchmark framework
   (`plans/tokenizer-benchmarks.md`) matures, tokenizer-level reconstruction
   quality may feed back into the quality filter — e.g. excluding shots where
   the frame tokenizer achieves an anomalously high rFID (suggesting the
   camera content is degenerate or the frames are mostly noise). This would
   require running the tokenizer before the quality check completes, which
   creates a circular dependency. Design decision deferred.

---

## 10. FAIR-MAST format reality — 2026-05-20

### 10.1 Storage format

FAIR-MAST level-2 is **xarray-on-Zarr v3**, NOT IDS format.  The data model
is:

- One Zarr v3 store per shot (e.g. `11766.zarr/`).
- Sub-directories are Zarr groups named after IMAS IDS names (`magnetics/`,
  `equilibrium/`, `pf_active/`, …).
- Each group opens as an `xr.Dataset`.  Data variables are physics signals;
  coordinates include time axes.
- The group-level `ds.attrs["imas"]` is a short **label string** equal to the
  group name (e.g. `"magnetics"`).  It is NOT an IDS container.
- Per-variable `ds["var"].attrs["imas"]` is an **IDS-path pointer string**
  (e.g. `"magnetics.b_field_pol_probe[:].field.data"`).  It is a reference,
  not a container.
- There are **no** `ids_properties`, `version_put`, `data_dictionary`, or
  `homogeneous_time` attributes anywhere in the bucket.

This means the checks `check_dd_version` and `check_homogeneous_time_flag`
that were present in the original implementation were chasing IDS attributes
that do not exist.  Both have been removed.

### 10.2 Time coordinate layout

Time coordinates are per-group, not at root.  Each group exposes one or more
time coordinates:

| Group | Primary time coord | Rate | Notes |
|---|---|---|---|
| `equilibrium` | `time` | 200 Hz | N≈83, dt=5 ms |
| `magnetics` | `time` | 5 kHz | N≈2065, dt=200 µs |
| `magnetics` | `time_saddle` | 20 kHz | dt=20 µs (saddle loops) |
| `pf_active`, `summary`, `pulse_schedule`, `gas_injection` | `time` | 4 kHz | dt=250 µs |
| `spectrometer_visible` | `time` | 50 kHz | dt=20 µs |
| `pf_passive`, `soft_x_rays`, `wall` | — | — | Static geometry, no time dim |

All `time` coords are monotonic with uniform Δt.  The audit's `check_time_axis`
has been rewritten to enumerate all coords containing `"time"` in their name,
report one `CheckResult` per coord, and return an info-level "static group"
result for groups with no time coord.

### 10.3 Physical dynamic ranges

FAIR-MAST plasma physics quantities span large numeric ranges that are
physically correct, NOT corruption:

| Quantity | Expected magnitude |
|---|---|
| Line-integrated electron density | ~10¹⁹ m⁻² |
| Neutral-beam / gas injection (particle rate) | ~10²² s⁻¹ |
| Density gradient (spectrometer) | ~10²¹ m⁻⁴ |
| Plasma current | 0–750 kA |
| PF coil current | ±50–200 kA |
| Reconstruction quality rating (`da_rating`) | 0–10 (integer-equivalent) |

The original audit used an `abs_max ≥ 1e15` threshold that flagged all of
the above as warnings.  The new `check_dynamic_range` uses a hard corruption
threshold of `1e25` — well above all physical quantities — to catch only
genuine bit-pattern errors.

### 10.4 Corpus-level findings from the 2026-05-20 audit (25-shot sample)

Running the recalibrated audit on a random 25-shot L2 sample revealed:

- **All groups open** (100 %): every group in every shot opens successfully —
  no Zarr-level corruption.
- **charge_exchange corruption** (systematic): `charge_exchange/t_i` and
  `charge_exchange/v_i` carry bit-pattern errors in ~50 % of audited shots,
  with values ranging from 10²⁶ to 10³⁸.  Physical ion temperature is
  ≤30 keV (~3×10⁴ eV) and ion velocity is ≤10⁷ m/s.  These values are
  12–28 orders of magnitude beyond physical range and represent genuine
  float-encoding defects in the FAIR-MAST ingestion pipeline for CX
  diagnostics.  **This is the primary cause of shots failing the audit.**
- **gas_injection/valve_target_voltage all-NaN**: present in ~20 % of shots.
  Likely channels not connected in some campaigns.
- **All false-positive IDS-format warnings eliminated**: the recalibrated
  audit no longer fires `dd_version`, `homogeneous_time_flag`, or
  `abs_max ≥ 1e15` warnings against physically correct data.
- **Usable-for-training rate**: ~44 % of the random sample.  The primary
  exclusion reason is `charge_exchange` corruption — shots without this
  diagnostic or with clean CX data pass at nearly 100 %.

### 10.5 Audit check changes made in this revision

The following table summarises the changes to §3 and §4:

| Check | Before | After |
|---|---|---|
| `check_dd_version` | Warned on every group (missing attribute) | **Removed** — attribute does not exist in FAIR-MAST |
| `check_homogeneous_time_flag` | Warned on every group (missing IDS flag) | **Removed** — no IDS `ids_properties` anywhere |
| `check_time_axis` | Checked only `"time"` coord, failed with warn if absent | Rewrites to enumerate all `"time*"` coords, info-level for static groups |
| `check_dynamic_range` | Warned on abs_max ≥ 1e15 (flagged physical densities) | Errors only on abs_max > 1e25 (hard corruption) or inf; warns on constant non-zero time-series |
| `check_no_all_nan` | Unchanged — correct | Unchanged |
| `check_open` | Unchanged — correct | Unchanged |
| `check_imas_pointer` | New | Info-level per-variable IDS-path pointer presence |
| `check_imas_label_matches_group` | New | Info-level group-label consistency check |

Quality flags changed:

| Flag | Before | After |
|---|---|---|
| `sane_dynamic_range` | Derived from `check_dynamic_range` warns | **Removed** (recalibrated check no longer warns on physical data) |
| `monotonic_time` | Per-group boolean | **Removed** (rolled into per-coord `time_axis:*` check results) |
| `all_groups_open` | Not present | **Added** — true iff every discovered group opened successfully |
| `no_corrupt_nans` | Not present | **Added** — true iff no all-NaN variable and no hard-corruption error fired |
| `usable_for_training` | No error-severity checks anywhere | `all_groups_open AND has_magnetics AND has_pulse_schedule AND no_error_checks` |
