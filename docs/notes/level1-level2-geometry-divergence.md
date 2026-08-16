# Level-1 versus level-2 magnetic geometry

## Finding

Level 2 does **not** copy the R and Z values in the level-1 `amb` channel
descriptions. The FAIR-MAST level-2 reader recognises the R/Z profiles as
`geometry`, bypasses the shot's signal loader, and asks
`Level2UDAGeometryLoader` for rows from two static geometry files:

- poloidal probes come from `pickup_coils.nc`, whose stored provenance names
  `create_netcdf_pickup.py` and configuration PDFs as its sources;
- point flux loops come from `fluxloops.nc`, whose stored provenance names
  `create_netcdf_fluxloops.py` and `detectors.dat_M4` as its source.

The loader uses each static row's `name` as `*_geometry_channel` and its `r`
or `z` field as the profile value. The shot-specific level-1 `amb` descriptions
are not read anywhere on this path. In mechanism terms, level 2 substitutes a
nominal geometry table; it neither copies level 1 nor recomputes positions from
the shot's measurements.

This is visible both in the
[upstream mapping at commit `ab435c7`](https://github.com/ukaea/fair-mast-ingestion/blob/ab435c799d892956fb042d55391f7d1be0c950e6/mappings/level2/mast.yml#L3591-L3886)
and in the
[geometry branch of `DatasetReader`](https://github.com/ukaea/fair-mast-ingestion/blob/ab435c799d892956fb042d55391f7d1be0c950e6/src/level2/reader.py#L48-L61).
`read_geometry` delegates to
[`Level2UDAGeometryLoader`](https://github.com/ukaea/fair-mast-ingestion/blob/ab435c799d892956fb042d55391f7d1be0c950e6/src/core/load.py#L520-L582),
which fetches the configured geometry tree and constructs the named xarray.
The upstream ingestion repository also records that this geometry is the same
for every shot even though it is currently ingested once per shot.

The local stores prove that statement for the two representatives. Hashing the
names, R and Z arrays for all 78 probes and 44 loops gives the identical SHA-256
digest
`856b8e0d5542ecfe2af480a9cd78a0c6d1160248449e6ab2b8a17f7ea10d74ba`
at both shots 11766 and 12417.

## Decomposition of all 186 disagreements

The comparison is by case-folded acquisition identity, not array index or
nearest position. There are 83 shared identities at shot 11766 and 103 at shot
12417. None has exactly equal level-1 and level-2 R/Z.

The distribution has a natural gap: every close loop is below 0.020 m, while
every displaced loop is above 0.115 m. That separates the causes without an
arbitrary boundary through the observations.

| Cause | Shot 11766 | Shot 12417 | Total | Separation range | Mechanism |
|---|---:|---:|---:|---:|---|
| Nominal probe table replaces `amb` description coordinates | 69 | 76 | **145** | 0.005700003–0.014142178 m | The acquisition descriptions use their own rounded positions; level 2 independently reads the PDF-derived `pickup_coils.nc` table. |
| Same-site loop values from two tables | 5 | 7 | **12** | 0.000410016–0.019956451 m | A named `amb` loop and the `detectors.dat_M4` row describe the same local placement at different precision or survey position. |
| Point-loop identity reassigned or displaced | 9 | 20 | **29** | 0.115433913–2.148100509 m | Level-1 descriptions contain repeated placeholders or positions belonging to another loop group, while level 2 publishes the independently named detector-table row. Fourteen of these 29 displacements are at least 0.5 m. |
| **Total** | **83** | **103** | **186** | 0.000410016–2.148100509 m | Two independent source tables are being compared as though one had been copied from the other. |

The largest cause by count is the nominal probe-table substitution: 145 of 186
rows. It is also why “no position agrees” initially looks more severe than the
probe geometry warrants. Every probe moves, but all probe movements are only
5.7–14.1 mm.

The metre-scale tail is instead a point-loop identity problem. For example,
level 1 assigns `fl_cc01`, `fl_cc02`, `fl_cc03`, `fl_cc04`, `fl_cc05`,
`fl_cc07`, `fl_cc09` and `fl_cc10` the same description coordinate
`(0.180, 1.215) m` at shot 12417. The nominal table spreads those identities
along the centre column. Identity matching therefore moves `fl_cc09` to
`(0.1785, -0.9331) m`, the largest observed displacement at **2.148100509 m**.
This is not a coordinate convention, rounding error, or shot transition; it is
the consequence of replacing a duplicated lookup placeholder with a named
detector-table row.

## Comparison with the reconstruction geometry used by the spine

`build_table_for_shot` reads `efm.magpr_r/z` and `efm.silop_r/z`. Its probe join
uses the level-1 `amb` description only as a lookup key and then stores the EFM
coordinate in `SensorMapping`. Thus the EFM columns below are exactly the
geometry the current spine uses after its join.

| Shot | Sensor | Level-1 `(R, Z)` m | Level-2 `(R, Z)` m | Spine EFM `(R, Z)` m | Distance L1→EFM | Distance L2→EFM | Closer store |
|---:|---|---|---|---|---:|---:|---|
| 11766 | `ccbv10` | (0.186000, 0.762000) | (0.180300, 0.758870) | (0.180300, 0.758870) | 0.006502837 m | 0 | level 2, exact |
| 11766 | `obr01` | (1.440000, 1.321000) | (1.442000, 1.335000) | (1.442000, 1.330700) | 0.009904085 m | 0.004299998 m | level 2 |
| 11766 | `fl_p3u_1` | (1.163000, 1.083000) | (1.163000, 1.082590) | (1.163000, 1.082590) | 0.000410016 m | 0 | level 2, exact |
| 12417 | `ccbv01` | (0.186000, 1.449000) | (0.180300, 1.448750) | (0.180300, 1.448750) | 0.005705482 m | 0 | level 2, exact |
| 12417 | `obr01` | (1.440000, 1.321000) | (1.442000, 1.335000) | (1.442000, 1.334540) | 0.013686927 m | 0.000460029 m | level 2 |
| 12417 | `obv01` | (1.440000, 1.321000) | (1.442000, 1.335000) | (1.442000, 1.334540) | 0.013686927 m | 0.000460029 m | level 2 |
| 12417 | `fl_cc01` | (0.180000, 1.215000) | (0.178500, 1.234900) | (0.178500, 1.234900) | 0.019956451 m | 0 | level 2, exact |
| 12417 | `fl_cc09` | (0.180000, 1.215000) | (0.178500, −0.933100) | (0.178500, 1.234900) | 0.019956451 m | 2.167999983 m | level 1, but the EFM join is non-unique |

The aggregate probe result is stronger than the examples: level 2 is closer to
the spine's EFM geometry for 56 of 69 probes at shot 11766 and for all 76 of 76
probes at shot 12417. It is exact for 20 and 39 probes respectively. Level 1
is exact for none. Across both shots, level 2 is closer for **132 of 145** probe
identities and exact for **59**.

The loop aggregate must not be read the same way. The spine's legacy loop join
is nearest-neighbour from the duplicated level-1 description coordinate. At
shot 12417 it maps every centre-column identity to EFM `silop[0]`; the
`fl_cc09` row shows the resulting false appearance that level 1 agrees. The
mapping records this as non-unique, so it is evidence about the join mechanism,
not evidence that seven distinct loops occupy one point. Settled identities
such as `fl_p3u_1` and `fl_cc01` instead show exact level-2/EFM agreement.

## Authority recommendation

**Use the level-2 nominal geometry as the machine-description authority, with
explicit corrections or qualifications for its known loop-table defects.** Use
the level-1 `amb` descriptions for acquisition identity and provenance, not as
the physical position authority.

The evidence is independent and consistent:

1. The production mapping declares the level-2 R/Z arrays to be geometry and
   sources them from configuration PDFs or an EFIT detector file intended to
   describe the machine, rather than from a shot signal.
2. The resulting 122-position payload is byte-identical across the two
   physical-range representatives.
3. Level-1 loop descriptions demonstrably contain duplicated placeholders and
   a wrong-side value; treating them as coordinates collapses distinct sensors
   and corrupts the EFM identity join.
4. For probes, where the orientation-constrained join is unambiguous, level 2
   is closer to the reconstruction table for 132 of 145 identities and level 1
   is exact for zero.

This recommendation does not assert that every level-2 row is correct. The
nominal loop table contains a copied P4-lower block, a spelling defect for the
tenth centre-column loop, and no P6-upper entry. Those are catalog defects to
correct or qualify on top of the nominal authority; they are not reasons to
promote the known-placeholder acquisition descriptions.

## The two shot-12417 omissions

### `fl_cc10`

This is explained: level 2 does carry the tenth centre-column geometry row, but
names it **`fl_cc010`**. The static 44-row geometry list ends `fl_cc01` through
`fl_cc09`, `fl_cc010`; identity comparison against level-1 `fl_cc10` therefore
reports an omission. The mechanism is an extra-zero spelling defect in the
nominal geometry source, not absent R/Z.

### `fl_p6u_1`

This is also localised. Level 1 publishes the `fl_p6u_1` acquisition channel at
shot 12417, but the 44-row `detectors.dat_M4` geometry product contains two
P6-lower rows and **zero P6-upper rows**. The level-2 signal mapping also does
not request `AMB_FL/P6U/1`. Consequently the channel is measurable in level 1
but neither described nor ingested into level 2. The historical reason the
static detector source omitted the P6-upper row is not recorded in the local
store or current ingestion mapping; that provenance remains unexplained.

## Missing `silop_dphi` at shot 11766

The absence occurs upstream in the level-1 EFM group. `efm0117.66` simply has
no `silop_dphi` array, while `efm0124.17` publishes 46 finite values, all equal
to 2π within 2e-7 rad. The level-2 point-loop R/Z path cannot have removed the
field because it never reads it: it obtains R/Z independently from
`fluxloops.nc`.

No producer commit or source metadata in the mounted level-1 store explains why
the early EFM dataset omitted this field. The defensible mechanism-level record
is therefore: **an upstream EFM publication/schema omission in the early
representative, motivation unexplained**, not a level-1-to-level-2 conversion
loss and not evidence that the early loops had zero toroidal extent.

## Evidence and reproducibility

- `/work/projects/imas_gpu/store-bench/mmte-level1-divergence-findings-20260816.log`
  — concise counts, hashes, EFM agreement totals, omissions and side-by-side
  receipts; command exit 0.
- `/work/projects/imas_gpu/store-bench/mmte-level1-divergence-residuals-20260816.log`
  — every shared identity with level-1, level-2 and joined EFM coordinates;
  command exit 0.
- `/work/projects/imas_gpu/store-bench/mmte-level1-divergence-metadata-20260816.log`
  — level-2 array provenance naming the geometry builders and their PDF/EFIT
  sources; command exit 0.

The measurements use source commit
`d7370fd57445c2efb30af2c0d46f780d52ecf6f1`. No production code was changed.
