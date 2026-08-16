# Level-1 versus level-2 sensor disentanglement

## Result

The 2.148 m disagreement is not a naming, array-order, units, reflection,
scale, or constant-offset effect. For the original 186 identities shared by
case-insensitive name, reordering level-2 into level-1 identity order leaves
**186/186 R/Z pairs different**. The least-squares best of the requested
single transforms is a **-0.0380419 m Z offset**, but it still leaves a
**2.11006 m maximum residual** and makes every probe-family residual exceed
1 cm. The discrepancy is therefore in the per-sensor positions published by
the two tiers, with the largest and most structured disagreement in flux
loops.

The level-1 `amb` descriptions are rounded nominal coordinates, while the
eight-sensor EFM cross-check below puts level-2 at, or within 3 mm of, EFM for
the sampled identities. In particular, `fl_cc09` is
`(R,Z)=(0.180,1.215) m` in the level-1 description and
`(0.178500,-0.933100) m` in both level-2 and signal-identified EFM. This is a
position-source difference, not an order or coordinate-system conversion.

## Inputs and comparison rule

The two range-first snapshots were used because they cover both declared
machine-description regimes:

| Shot | Level-1 source | Level-2 source |
|---:|---|---|
| 11766 | `level1/shots/11766.zarr/amb`, consolidated Zarr V2 | `level2/shots/11766.zarr/magnetics`, Zarr V3 |
| 12417 | `level1/shots/12417.zarr/amb`, consolidated Zarr V2 | `level2/shots/12417.zarr/magnetics`, Zarr V3 |

Level-1 R/Z came from the `r=..., z=...` text in each acquired sensor
array's `description`. Level-2 identity and R/Z came from the paired
`*_geometry_channel`, `*_r`, and `*_z` arrays for `ccbv`, `obr`, `obv`, and
point flux loops. The counts below are sensor occurrences across the two
shots, not a deduplicated machine-wide inventory; level-2 contains the static
122-sensor geometry on each shot, while level-1 contains only arrays acquired
on that shot.

## Naming

The exact-string test is byte-for-byte after decoding the Zarr string. The
normalised test applies, in order: Unicode NFKC, surrounding-whitespace trim,
Unicode case-folding, replacement of whitespace, `/`, and `-` runs by `_`,
and removal of leading zeroes from numeric suffixes inside a token. The
normalised keys were verified unique in each store, so normalisation does not
merge two sensors.

| Shot | Level-1 names | Level-2 names | Exact | Normalisation only | No level-2 counterpart | Level-2 only |
|---:|---:|---:|---:|---:|---:|---:|
| 11766 | 83 | 122 | 69 | 14 | 0 | 39 |
| 12417 | 105 | 122 | 76 | 28 | 1 | 18 |
| **Occurrence total** | **188** | **244** | **145** | **42** | **1** | **57** |

The 42 normalisation-only matches have two concrete patterns:

- 41 point-loop occurrences are lowercase `fl_*` at level-1 and uppercase
  `FL_*` at level-2.
- Level-1 `fl_cc10` is level-2 `FL_CC010`: level-2 adds a third digit of zero
  padding to this one centre-column loop.

After normalisation, 187/188 level-1 occurrences have a level-2 counterpart.
The sole remainder is `fl_p6u_1` on shot 12417. Conversely, the 57 level-2-only
occurrences are mostly static geometry for sensors not acquired in that
level-1 shot; they are not alternative spellings. The previously reported
186-pair cohort used case-folding only, so it excludes both `fl_cc10` (the
extra-padding naming case) and `fl_p6u_1` (no counterpart) on shot 12417.

## Ordering

For level-1, array index means ordinal position among the filtered sensor keys
returned by the consolidated `amb` group. For level-2, it means ordinal
position after concatenating the four paired geometry arrays in their stored
family order: `ccbv`, `obr`, `obv`, then `flux_loop`. To avoid counting a
missing sensor as an order shift, indices were recomputed after restricting
both sequences to the matched identity set.

| Shot | Match rule | Matched | Index differs | Index fixed | One identity permutation? |
|---:|---|---:|---:|---:|---|
| 11766 | case-fold | 83 | 44 | 39 | yes, bijective |
| 12417 | case-fold | 103 | 64 | 39 | yes, bijective |
| **Total used for position fits** | **case-fold** | **186** | **108** | **78** | **yes, per shot** |
| 11766 | full normalisation | 83 | 44 | 39 | yes, bijective |
| 12417 | full normalisation | 104 | 65 | 39 | yes, bijective |

The permutation is a family-block move: level-1 places acquired flux-loop
keys between `ccbv` and the outboard probes, whereas level-2 places flux loops
after both outboard-probe families. Thus ordering genuinely differs for
108/186 identities. It does not explain the geometry disagreement: applying
the single identity permutation to each shot produces identity-aligned arrays
and still leaves **186/186 non-zero separations**.

## Coordinate-transformation hypotheses

Each transform maps level-1 `(R,Z)` to level-2 and is fitted over the original
186 case-fold identities. The residual is Euclidean separation in metres.
Offsets and scale minimise the summed squared residual; the table reports both
RMS and maximum residual so a transform cannot hide one device-scale error in
many small probe shifts.

| Single hypothesis | Fitted value | RMS residual (m) | Maximum residual (m) | Residual below 0.01 m |
|---|---:|---:|---:|---:|
| Identity | - | 0.300763 | 2.148101 | 129/186 |
| Constant R offset | `dR=+0.0281914 m` | 0.299438 | 2.148305 | 0/186 |
| Constant Z offset | `dZ=-0.0380419 m` | **0.298347** | **2.110059** | 0/186 |
| Uniform scale about `(0,0)` | `s=0.978803802` | 0.299240 | 2.122348 | 22/186 |
| Z reflection about `z=0` | `Z'=-Z` | 1.783526 | 3.127605 | 4/186 |

A two-axis translation was also checked as a diagnostic, although it is not
one of the requested single-axis hypotheses: `dR=+0.0281914 m` and
`dZ=-0.0380419 m` gives RMS 0.297012 m and maximum 2.110267 m. It does not
change the verdict.

**No single transform reduces the 2.148101 m maximum below 0.01 m.** The best
allowed transform by RMS is the constant Z offset, and its maximum is still
211 times the threshold.

### Units test

Multiplying both level-1 coordinates by 10, 100, or 1000 gives RMS residuals
of 12.8662, 141.188, and 1424.44 m, respectively. The reciprocal directions
were checked as well: factors 0.1, 0.01, and 0.001 give RMS residuals 1.28826,
1.41339, and 1.42594 m. None improves the identity RMS of 0.300763 m, and none
leaves any of 186 pairs below 1 cm. The factor 0.1 happens to reduce the single
largest residual to 1.76968 m while making the population fit more than four
times worse; it is not a units solution.

## Residual structure

The first three columns show why the untransformed data already rejects a
global correction: every `ccbv` disagreement is sub-centimetre, while flux
loops contain the metre-scale tail. The last columns are after the
least-squares best requested transform, the -0.0380419 m Z offset.

| Family | Pairs | Identity RMS (m) | Identity maximum (m) | Identity below 0.01 m | Best-transform RMS (m) | Best-transform maximum (m) |
|---|---:|---:|---:|---:|---:|---:|
| `ccbv` | 78 | 0.005719 | 0.006503 | 78 | 0.038423 | 0.038961 |
| `obr` | 34 | 0.009733 | 0.014142 | 21 | 0.045032 | 0.052080 |
| `obv` | 33 | 0.009645 | 0.014142 | 20 | 0.044963 | 0.052080 |
| `flux_loop` | 41 | 0.640434 | 2.148101 | 10 | 0.630625 | 2.110059 |

The fitted Z offset slightly reduces the aggregate flux-loop residual while
turning all 129 initially sub-centimetre pairs into larger residuals. This is
the opposite of a global coordinate-frame correction. The difference is
family- and sensor-specific.

## Level-1 versus level-2 versus EFM

The table uses shot 12417 and spans every family. EFM B-probe identity follows
the published 78-row family blocks (`ccbv[0:40]`, `obv[40:59]`,
`obr[59:78]`), whose angle medians are 90 degrees, 90 degrees, and 0 degrees.
Flux-loop identity is coordinate-independent: the acquired waveform was
interpolated to EFM time and correlated against every `efm/silop_x` column.
`fl_cc09` uniquely selects EFM index 8 at correlation 0.999998914;
`fl_p5l_1` selects index 40 at correlation 0.999999999999987.

| Sensor | Family | Level-1 `(R,Z)` m | Level-2 `(R,Z)` m | EFM index and `(R,Z)` m | L1-L2 (m) | L1-EFM (m) | L2-EFM (m) |
|---|---|---|---|---|---:|---:|---:|
| `ccbv01` | `ccbv` | `(0.186000, 1.449000)` | `(0.180300, 1.448750)` | 0: `(0.180300, 1.448750)` | 0.005705 | 0.005705 | 0.000000 |
| `ccbv20` | `ccbv` | `(0.186000, 0.000000)` | `(0.180300, 0.000000)` | 19: `(0.180300, 0.000000)` | 0.005700 | 0.005700 | 0.000000 |
| `obv01` | `obv` | `(1.440000, 1.321000)` | `(1.442000, 1.335000)` | 40: `(1.442000, 1.334540)` | 0.014142 | 0.013687 | 0.000460 |
| `obv14` | `obv` | `(1.850000, -0.300000)` | `(1.844900, -0.293020)` | 53: `(1.844900, -0.293000)` | 0.008645 | 0.008661 | 0.000020 |
| `obr01` | `obr` | `(1.440000, 1.321000)` | `(1.442000, 1.335000)` | 59: `(1.442000, 1.334540)` | 0.014142 | 0.013687 | 0.000460 |
| `obr14` | `obr` | `(1.850000, -0.300000)` | `(1.844900, -0.293020)` | 72: `(1.844900, -0.293000)` | 0.008645 | 0.008661 | 0.000020 |
| `fl_cc09` | `flux_loop` | `(0.180000, 1.215000)` | `(0.178500, -0.933100)` | 8: `(0.178500, -0.933100)` | **2.148101** | **2.148101** | **0.000000** |
| `fl_p5l_1` | `flux_loop` | `(1.163000, -1.089000)` | `(1.749300, -0.442240)` | 40: `(1.746300, -0.442240)` | 0.872953 | 0.870941 | 0.003000 |

The sampled level-2 coordinates are identical to EFM for both `ccbv` sensors
and `fl_cc09`, and are 0.02-3.00 mm from EFM for the other five. The level-1
description discrepancies are 5.7 mm to 2.148 m. This does not prove every
level-2 row is correct, but it independently localises the observed
level-1/level-2 disagreement to the nominal coordinates embedded in level-1
`amb` descriptions rather than to level-2 array order or a global transform.

## Accounting over the original 186 pairs

| Hypothesis | Observed metadata difference | Fraction of the 186 position disagreements explained |
|---|---:|---:|
| Naming | 145 raw-exact identities; 41 differ only by `FL_*` case | **0/186 (0%)** |
| Ordering | 108 identities move matched-set index; one bijection maps each shot | **0/186 (0%)** after applying it |
| Global coordinate transformation | Best requested transform leaves 2.110059 m maximum | **0/186 (0%)** under the 0.01 m criterion |
| Per-sensor position values | All 186 identity-aligned R/Z pairs are non-zero; 129 are below 0.01 m and 57 are at least 0.01 m | **186/186 (100%)** |
| Unexplained within the 186 | None: every pair is identity-aligned and its residual is directly measured | **0/186 (0%)** |

There are two occurrences outside that original cohort: `fl_cc10` is a naming
case (`FL_CC010`) and becomes a 187th matched pair under full normalisation;
`fl_p6u_1` has no level-2 counterpart on shot 12417 and remains explicitly
unmatched. The latter is not absorbed into the 186-pair position verdict.

No production code or machine-map declaration was changed by this analysis.
