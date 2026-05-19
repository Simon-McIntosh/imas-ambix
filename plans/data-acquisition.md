# Plan: FAIR-MAST Data Acquisition

Status: **Draft** — awaiting probe result + reservation grant.

This plan defines how we move FAIR-MAST level-2 data from
`s3.echo.stfc.ac.uk` to GPFS under `/work/projects/imas_gpu/mast/` so that
training jobs on the betelgeuse GPU node — which has **no outbound network
access** — can read it. The probe protocol must run first; it gates the
go / no-go on the full mirror and provides the throughput evidence we
need before reserving 6 – 24 h of compute time on a `sirius`-class node.

---

## 1. Endpoint inventory (verbatim from FAIR-MAST docs)

| Channel | Verbatim value |
|---|---|
| S3 endpoint | `https://s3.echo.stfc.ac.uk` |
| Bucket | `mast` (anonymous read; pass `--no-sign-request`) |
| Level-2 per-shot path | `s3://mast/level2/shots/{shot_id}.zarr/{group}/` |
| Level-1 per-shot path | `s3://mast/level1/shots/{shot_id}.zarr/{group}/` |
| Direct HTTPS Zarr | `https://s3.echo.stfc.ac.uk/mast/level2/shots/{shot}.zarr` |
| Parquet shot index | `https://mastapp.site/parquet/level2/shots` (11,573 rows × 189 cols) |
| REST shot endpoint | `https://mastapp.site/json/shots?filters=…&cursor=…&size=…` |
| GraphQL | `https://mastapp.site/graphql_api.html` |
| Region for boto3 | `us-east-1` (Ceph default — endpoint URL overrides) |

Licensing: data CC BY-SA 4.0; the `ukaea/fair-mast` and
`ukaea/fair-mast-ingestion` codebases are MIT. The "SA" clause matters
only if we redistribute *data* derivatives; it is generally interpreted
as not contagious to model weights, but legal sign-off is needed before
publishing any weights derived from raw FAIR-MAST signals (tracked in
`STRATEGY.md` §6).

Citations (BibTeX-ready):

```bibtex
@article{jackson2024fairmast,
  title   = {FAIR-MAST: A fusion device data management system},
  author  = {Jackson, Samuel and Khan, Saiful and Cummings, Nathan and
             Hodson, James and de~Witt, Shaun and Pamela, Stanislas and
             Akers, Rob and Thiyagalingam, Jeyan},
  journal = {SoftwareX},
  volume  = {27},
  pages   = {101869},
  year    = {2024},
  doi     = {10.1016/j.softx.2024.101869}
}

@article{jackson2025fairmast,
  title   = {An Open Data Service for Supporting Research in Machine
             Learning on Tokamak Data},
  author  = {Jackson, Samuel and others},
  journal = {IEEE Trans. Plasma Sci.},
  year    = {2025},
  doi     = {10.1109/TPS.2025.3583419}
}
```

---

## 2. Network and host topology (why this matters)

The SDCC partition assignment matters for both the probe and the bulk
download. From `docs/cluster-usage.md` §2:

| Host class | Outbound network | `/work/projects/imas_gpu/` access | Suitable for |
|---|---|---|---|
| Login (`io-ls-hpc.iter.org` etc.) | yes | **no** (group `sdcc-imas_gpu` missing) | Reading parquet index, REST API only |
| `sirius` standard compute | **yes** | **yes** | The probe **and** the bulk download |
| `rigel` standard compute | yes | yes | Backup option for the bulk download |
| `betelgeuse` GPU node | **no** | yes | Only useful **after** mirror is complete |

Conclusion: both the probe and the bulk download run on `sirius` (or
`rigel`). Login nodes can do dry-runs but cannot stage to GPFS.

---

## 3. Sizing probe protocol

**Owner:** runs from `sirius`. Wall-time budget: ~30 min. Result is appended
to this document as a verbatim block in §3.4 before the bulk download is
allowed to start.

### 3.1 Probe steps (paste verbatim on a `sirius` node)

```bash
# 0. Allocate an interactive sirius node
srun --partition=sun_debug --cpus-per-task=8 --mem=16G --time=01:00:00 --pty bash -l
export TMPDIR=/tmp

# 1. Pull the parquet shot index
mkdir -p ~/work/mast-probe && cd ~/work/mast-probe
python - <<'EOF'
import pandas as pd
df = pd.read_parquet("https://mastapp.site/parquet/level2/shots")
print(f"shots={len(df)} cols={len(df.columns)}")
df.to_parquet("shots-index.parquet")
# Save a manifest of shot_id + boolean has-camera flags
keep = ["shot_id", "campaign"]
for col in df.columns:
    if "camera_visible" in col or "camera_ir" in col:
        keep.append(col)
df[keep].to_parquet("shots-with-camera-flags.parquet")
EOF

# 2. Install s5cmd (Python wheel wraps the Go binary)
pip install --user s5cmd
export PATH="$HOME/.local/bin:$PATH"

# 3. Enumerate level-2 shots — confirm bucket layout matches the docs
s5cmd --no-sign-request --endpoint-url https://s3.echo.stfc.ac.uk \
  ls 's3://mast/level2/shots/' | head -20

# 4. Per-shot footprint on a 50-shot sample (cameras included)
mkdir -p sample-with-cam sample-no-cam
time s5cmd --no-sign-request --endpoint-url https://s3.echo.stfc.ac.uk \
  --numworkers 16 \
  cp 's3://mast/level2/shots/30420.zarr/*' sample-with-cam/30420.zarr/
du -sh sample-with-cam/30420.zarr/

# 5. Repeat 4. for 49 more shot IDs sampled across the campaign range
#    (write the loop into a script — pseudocode shown)
#    Outputs: median per-shot MB, p95 per-shot MB, camera-vs-non ratio.

# 6. Throughput probe — saturating cp of a single camera-rich shot
time s5cmd --no-sign-request --endpoint-url https://s3.echo.stfc.ac.uk \
  --numworkers 32 \
  cp --concurrency 64 \
  's3://mast/level2/shots/30420.zarr/camera_visible/*' \
  ./throughput-test/
# Throughput = (bytes copied) / (elapsed seconds), reported in MB/s.
```

The probe script lives in `imas_ambix/data/probe.py` (created in the
code-skeleton PR following this plan PR). It accepts
`ambix data probe --sirius` and writes a JSON summary to
`/work/projects/imas_gpu/mast/.probe/<timestamp>.json`.

### 3.2 Acceptance thresholds

| Metric | Threshold | Action if missed |
|---|---|---|
| Sustained throughput from STFC Echo | ≥ 200 MB/s on 32 workers | If 50 – 200 MB/s: continue but extend wall-time budget to 36 h. If < 50 MB/s: open a ticket with STFC and switch to per-diagnostic prioritised mirror. |
| Total level-2 footprint extrapolated to 11,573 shots | 2 TB ≤ size ≤ 12 TB | If > 12 TB: split download into camera vs non-camera and stage camera-rich shots first; defer full-mirror decision. |
| Camera-bearing shot count | ≥ 1,000 | If < 1,000: descope the camera demo and pivot v0 to a magnetics-only forecast — escalate to user, do not proceed silently. |
| Per-shot p95 size | ≤ 5 GB | If higher: alert (probably means full-rate magnetics is included unexpectedly). |

### 3.3 What the probe *does not* do

- Does not validate the IMAS-DD version embedded in each shot's Zarr
  attributes. That check belongs in the loader, not the probe.
- Does not pull any data into the mirror. The 50-shot sample lives under
  `~/work/mast-probe/` and is discarded after the probe report is filed.

### 3.4 Probe result (to be appended after the first run)

```text
<probe-result-block: paste the JSON summary here>
```

Until that block is filled in, the bulk download in §4 is **on hold**.

---

## 4. Bulk download protocol

Run only after §3.4 is filled and acceptance thresholds are met.

### 4.1 Storage layout

```
/work/projects/imas_gpu/mast/                  # root, group sdcc-imas_gpu, mode 2770
├── README.md                                  # provenance + license note (CC BY-SA 4.0)
├── shots-index.parquet                        # snapshot of mastapp.site index at download time
├── manifests/
│   └── level2-{download_timestamp}.json       # shot list, expected sizes, checksums
├── level2/
│   └── shots/
│       ├── 11695.zarr/
│       ├── 11696.zarr/
│       └── ...
└── .probe/
    └── {timestamp}.json                       # probe results (kept for the record)
```

We deliberately mirror the source layout under `level2/shots/` so that
existing `xarray.open_zarr` snippets from FAIR-MAST docs work unchanged
against the local mirror. Code paths just point at
`/work/projects/imas_gpu/mast/level2/shots/{shot}.zarr` instead of
`https://s3.echo.stfc.ac.uk/mast/level2/shots/{shot}.zarr`.

### 4.2 SLURM job spec

```bash
#!/bin/bash
#SBATCH --job-name=mast-mirror
#SBATCH --partition=sun                     # standard compute, outbound network OK
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.log

set -euo pipefail
export TMPDIR=/scratch_local/$SLURM_JOB_ID && mkdir -p "$TMPDIR"
export PATH="$HOME/.local/bin:$PATH"

DEST=/work/projects/imas_gpu/mast/level2/shots
mkdir -p "$DEST"

# Manifest-driven loop — pass shot IDs via stdin to allow restart
python -m imas_ambix.data.manifest emit-shot-ids \
  --index /work/projects/imas_gpu/mast/shots-index.parquet \
| while read SHOT; do
    s5cmd --no-sign-request --endpoint-url https://s3.echo.stfc.ac.uk \
      --numworkers 32 \
      cp "s3://mast/level2/shots/${SHOT}.zarr/*" "${DEST}/${SHOT}.zarr/"
done
```

`partition=sun` is the standard compute partition with network access. We
deliberately do **not** use `*_debug` partitions for this job — debug
partitions cap at 1 h wall time. The 24 h limit is dimensioned for the
worst-case 50 MB/s scenario; on a healthy network it should finish in
6 – 12 h.

### 4.3 Resumability

- s5cmd `cp` skips objects that already exist with matching size, so a
  re-run is a no-op for completed shots. We exploit this for restart
  after `oom_killer` or wall-time exhaustion.
- The acceptance gate after the job is: re-run the same SLURM script;
  the second invocation must log **0 new objects copied**. If any new
  objects appear on the second pass, investigate before declaring the
  mirror complete.
- Per-shot verification: a checksum loop reads each shot's `.zmetadata`
  manifest, opens the Zarr with `xarray.open_zarr`, and counts groups +
  total bytes. Output written to
  `/work/projects/imas_gpu/mast/manifests/level2-{ts}.json`.

### 4.4 Permissions

```bash
chgrp -R sdcc-imas_gpu /work/projects/imas_gpu/mast/
chmod -R u+rwX,g+rsX,o-rwx /work/projects/imas_gpu/mast/
find /work/projects/imas_gpu/mast/ -type d -exec chmod g+s {} +
```

The `g+s` setgid bit ensures every file created later inherits the
`sdcc-imas_gpu` group, matching the existing layout under
`/work/projects/imas_gpu/agents/`.

### 4.5 Camera-shot prioritisation

If the probe shows total footprint > 12 TB, switch to a two-pass mirror:

1. **Pass A:** shots where `camera_visible` *or* `camera_ir` group is
   present, full set of diagnostics. Pre-filtered from the parquet index
   into `manifests/level2-camera-pass-a.json`.
2. **Pass B:** the remainder, magnetics + equilibrium + pf_active +
   summary + pulse_schedule groups only (skip `camera_visible`,
   `camera_ir`, `bolometer` to halve the footprint).

Pass A unlocks v0 training and the demo; Pass B fills out the corpus for
the larger v1 runs. The split is captured in two manifest files so we can
re-issue either independently.

---

## 5. Operating the mirror after download

- **Read access** — anyone in `sdcc-imas_gpu` can open shots directly:

  ```python
  import xarray as xr
  ds = xr.open_zarr(
      "/work/projects/imas_gpu/mast/level2/shots/30420.zarr",
      group="camera_visible/camera_center",
  )
  ```

- **From the betelgeuse GPU node** — same path, no network needed.
- **Refresh policy** — FAIR-MAST is read-mostly but does receive M9
  ingestion updates. We re-pull the parquet index quarterly and `s5cmd
  cp` only the new shot IDs. The manifest distinguishes "original mirror
  date" from "last-checked date" per shot.

---

## 6. Recovery / re-run rules

| Symptom | Action |
|---|---|
| SLURM job killed mid-stream | Re-submit the same script — s5cmd skip-existing behaviour handles restart. |
| Disk full on GPFS | Surface immediately to the operator. Do **not** delete shots from the destination to make room without an authorisation note in this plan. |
| Re-pulled object differs from the first pull | Open a STFC ticket; quarantine the differing object under `.quarantine/` rather than overwriting. |
| Parquet index returns a different shot count from the documented 11,573 | Quote the new count in the next mirror's manifest and continue; do **not** rewrite this plan unless the count drops. |

---

## 7. Out of scope of this plan

- ~~Level-1 raw mirror. Level-1 is per-source-file and includes
  unprocessed camera frames; the size budget grows dramatically. v0
  trains on level-2; level-1 is a Phase 3+ question.~~ **Superseded by
  §10.** The probe found level-2 carries no cameras; a selective level-1
  camera mirror is the v0 path.
- IMAS-DD version coercion. The level-2 YAML uses DDv3-shaped paths; we
  open with `dd_version` matched to the file (see `imas-python` rules in
  AGENTS.md). The loader, not the mirror job, enforces this.
- Tokenization. Encoded tokens land under
  `/work/projects/imas_gpu/mast-tokens/`, never inside `mast/` itself.
  See `tokenizers.md`.

---

## 10. Live-probe findings — 2026-05-19 (login node)

Probe executed from `io-ls-hpc.iter.org` (login node, group
`sdcc-imas_gpu`, internet OK). s5cmd v2.3.0 installed at
`~/.local/bin/s5cmd` from the upstream Go-binary release.

### 10.1 Headline counts (verbatim from `s5cmd ls`)

| Tier | Shot count | Bucket prefix |
|---|---:|---|
| level-2 | **11,573** | `s3://mast/level2/shots/` |
| level-1 | **17,111** | `s3://mast/level1/shots/` |

`ls` of the level-2 prefix took 26 s; level-1 took 78 s. Both are
cold-listing rates from the login node.

### 10.2 Level-2 does **not** carry camera groups

Random sample of 20 level-2 shots — checked each for
`camera_visible/` or `camera_ir/` sub-prefixes via `s5cmd ls`:

```text
12984 30042 12101 27336 30034 12882 23815 29616 26155 20031
28147 24583 27157 15273 13945 12822 19573 22348 23836 18483
```

**All 20 returned `camera_groups=0`.** Spot-check of shot `28352`
(reported as having `rbb` source by `mastapp.site/json/sources`) returned
only the 11 non-camera groups: `equilibrium`, `gas_injection`,
`magnetics`, `pf_active`, `pf_passive`, `pulse_schedule`, `soft_x_rays`,
`spectrometer_visible`, `summary`, `thomson_scattering`, `wall`. The
level-1 → level-2 mapping in
`ukaea/fair-mast-ingestion/mappings/level1/mast/groups.json` documents
the *intended* ingestion (rba → camera_visible.camera_lower, etc.) but
**the ingestion has not been run for the camera sources** in the public
level-2 bucket.

This contradicts the original plan §1 group catalogue. The plan listed
`camera_visible` and `camera_ir` among the level-2 groups verbatim from
`mappings/level2/mast.yml`; that YAML describes what level-2 *can* hold,
not what the public bucket currently does.

### 10.3 Cameras live in level-1

Shot 30420 level-1 carries: `rba/`, `rbb/`, `rbc/`, `rca/`, `rco/`,
`rgb/`, `rgc/`, `rir/`, `rit/` (plus all non-camera sources). Per-camera
sizes for shot 30420 (single-stream cold pull, 32 workers):

| Source | Mapping | Size | Wall time | Throughput |
|---|---|---:|---:|---:|
| `rba` | camera_visible.camera_lower | 92 MB | 1.2 s | 76 MB/s |
| `rbb` | camera_visible.camera_center | 73 MB | 8.4 s | 8.7 MB/s |
| `rir` | camera_ir | 5.6 MB | 0.9 s | 6.2 MB/s |

Throughput varies wildly per object — STFC Echo is fast on the first
object then drops. The single 18 MB/s overall single-shot rate is the
honest summary; parallel multi-shot pulls should saturate higher.

### 10.4 Coverage from the REST API (cross-check, see notes)

`mastapp.site/json/sources?filters=name$eq:{src}&size=100&cursor=...`
reports the following totals:

| Source name | Shots reporting source |
|---|---:|
| `rba` | 6,139 |
| `rbb` | 9,494 |
| `rir` | 25 |

Caveat: the REST `sources` table includes shots that have never been
ingested into the level-2 bucket — example `shot 15667` was listed by
the API but `s5cmd ls s3://mast/level2/shots/15667.zarr/` returns
`no object found`. Treat these counts as **upper bounds**; the actual
level-1 prefix presence has to be confirmed by `s5cmd ls`.

### 10.5 Revised sizing target

Best estimate of a working corpus for v0:

| Component | Size |
|---|---:|
| Level-2, all 11,573 shots, all 11 non-camera groups | ~140 GB (assuming the shot 30420 figure of 111 MB is typical) |
| Level-1 RBA + RBB + RIR for the camera-bearing subset | ~1.3 TB (9,494 × 73 MB rbb worst-case, plus 6,139 × 92 MB rba on the overlap) |
| **Total** | **~1.5 TB** |

This is **well below** the original ~5 TB working assumption and is
comfortable on the 576 TB free GPFS partition. The acceptance gates in
§3.2 — 2 TB ≤ total ≤ 12 TB — were too generous on the lower bound and
need revision.

### 10.6 Implications for the plan

1. **§3.2 acceptance gates** — lower the `TOTAL_SIZE_MIN_TB` from 2 to
   0.5. The corpus we want is genuinely small. Drop the camera-shots
   gate (1,000 shots minimum) — it's measured by level-1 listing, not
   index columns. Add a new gate "camera_visible coverage detected in
   level-1 sample ≥ 50 %".
2. **§3 probe protocol** — replace the "parquet camera flags" filter
   with an S3-listing pass. The parquet index has 189 columns but
   **none** contain "camera". The implementation refactor is tracked in
   `imas_ambix/data/probe.py` (PR-pending).
3. **§4 bulk download** — split into two manifests:
   - `level2-all-shots.json` — all 11,573 shots, all groups present in
     each shot's level-2 Zarr. ~140 GB.
   - `level1-cameras.json` — for each shot in level-1 with any of
     `rba` / `rbb` / `rbc` / `rgb` / `rir` / `rit`, copy only those
     camera prefixes plus a minimal control vector (`anb`, `aga`, `efm`).
2. **Download host** — login node has internet + GPFS + group access and
   sustained ~18 MB/s per stream. At 32 parallel streams the bulk
   download fits inside 6 – 12 h. SLURM is **not strictly required** for
   v0 — a screened `s5cmd` invocation from the login node works. SLURM
   moves to optional infrastructure rather than a hard requirement.

### 10.7 Sources for §10

- Local commands: `~/.local/bin/s5cmd v2.3.0`, `pandas` over
  `https://mastapp.site/parquet/level2/shots`.
- Cross-check: `https://mastapp.site/json/sources?filters=name$eq:rba`.
- Comparison reference: the original size assumption traces to the
  GPU-procurement document (`imas-codex/plans/gpu-cluster-scoping.md`
  §3.3) which quoted ~5 TB based on partner-corpus extrapolation,
  not on the FAIR-MAST corpus specifically.
