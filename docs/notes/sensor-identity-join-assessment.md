# Sensor identity join assessment

## Conclusion

The legacy nearest-coordinate join does not preserve flux-loop identity. Across
three audited shots, **37 of 50 mapped flux-loop rows** select a different EFM
column from the measured-signal identity: **9/12** on shot 11766 and **14/19**
on each of shots 21978 and 21983. The waveform matches are one-to-one. Their
minimum winning Pearson correlation is 0.999954 on the early-range shot and
0.999993 on the two later-range shots.

The legacy reader notices most, but not all, of the damage. It records a
`non-unique` flag on **35/37 mismatch instances (94.6%)**. The remaining two
instances are `fl_p6l_1` on shots 21978 and 21983: the coordinate residual is
only 0.000565690956 m, the flag is empty, and the row is nevertheless joined to
`silop[34]` instead of its signal-identified `silop[44]`. More importantly for
the frozen spine, the flags are informational: the benchmark does not turn them
into a mask.

Consequently, the frozen greens-matvec residual
**0.7401030841614555 is affected by this aliasing**. It was computed with the
aliased sensor positions in both the upstream disc-centroid read and the final
cell-to-sensor forward check. This audit does not claim a corrected residual or
a numerical delta: obtaining either requires changing the identity binding and
re-running the frozen CPU benchmark. The existing number remains an exact
receipt for the legacy geometry semantics, not a clean reference for judging a
correctly identified machine description.

## Evidence boundary and method

The corpus inventory is the union of the two acquisition declarations in
`imas_ambix/data/machine_maps/mast.json`, whose source revision is
`local-level2-11573-shots-11766-30471-machine-description`. Those declarations
cover the two live-plan geometry ranges, 11766–12416 and 12417–30471. They
contain 188 address occurrences and 106 distinct acquisition addresses. Every
one of the 83 and 105 declared addresses, respectively, has a parseable
description coordinate on its declaration's range-first evidence shot (11766
and 12417). Thus the collision count below is the catalog-declared corpus
inventory, with coordinates verified against those two source shots.

An attempted shot-by-shot metadata scan was stopped by the command execution
ceiling before it could flush a result; it is not used as evidence. The bounded
declaration census is recorded in
`mmte-legacy-alias-impact-declaration-census.log` in the worker scratchpad.

For identity, each acquired `amb` loop waveform was interpolated onto the EFM
time base and correlated against every column of `efm/silop_x`; the unique
highest-correlation column is the signal identity. This is intentionally
independent of both the acquisition description and either geometry source's
coordinates. Shot 12417 was not counted because `fl_cc10` has 7,500 samples
while that shot's common `amb/time` has 11,289, and no channel-specific time
base is declared. Substituting an assumed truncation would weaken the receipt,
so the later range is represented by shots 21978 and 21983 instead. The raw
per-row receipts are in `mmte-legacy-alias-impact-shot-comparison.log` and
`mmte-legacy-alias-impact-late-shot-comparison.log`.

## Corpus coordinate collisions

There are **56 distinct acquisition addresses at 23 duplicated coordinates**:

- **20 flux-loop addresses at 5 coordinates**. These are unsafe identity keys
  because every member of a group has the same sensor kind and no orientation
  discriminator.
- **36 B-probe addresses at 18 coordinates**, all 18 being intentional
  `obrNN`/`obvNN` radial/vertical pairs. These are not subject to this flux-loop
  failure: the legacy B-probe join first restricts candidates to the
  name-expected orientation (`geometry.py:868-946`).

Every duplicated flux-loop address is listed here:

| Description coordinate (m) | Acquisition addresses sharing it |
|---|---|
| `r=0.180, z=1.215` | `fl_cc01`, `fl_cc02`, `fl_cc03`, `fl_cc04`, `fl_cc05`, `fl_cc07`, `fl_cc09`, `fl_cc10` |
| `r=1.035, z=-1.089` | `fl_p3l_4`, `fl_p4l_4`, `fl_p5l_4` |
| `r=1.035, z=1.083` | `fl_p3u_4`, `fl_p5u_4` |
| `r=1.163, z=-1.089` | `fl_p3l_1`, `fl_p4l_1`, `fl_p5l_1`, `fl_p6u_1` |
| `r=1.163, z=1.083` | `fl_p3u_1`, `fl_p4u_4`, `fl_p5u_1` |

The proposed six-name set at `r=0.180, z=1.215` is therefore **correct for the
addresses present on shot 21978**, namely `fl_cc01`, `fl_cc03`, `fl_cc04`,
`fl_cc05`, `fl_cc07`, and `fl_cc09`, but it is **not the complete corpus set**.
The corpus declaration union adds `fl_cc02` and `fl_cc10`, for eight identities
at that coordinate. The early declaration contains the four-member subset
`fl_cc01`, `fl_cc02`, `fl_cc03`, `fl_cc05`.

Every duplicated B-probe address is listed here:

| Description coordinate (m) | Co-located radial/vertical addresses |
|---|---|
| `r=1.440, z=-1.325` | `obr19`, `obv19` |
| `r=1.440, z=-1.250` | `obr18`, `obv18` |
| `r=1.440, z=1.250` | `obr02`, `obv02` |
| `r=1.440, z=1.321` | `obr01`, `obv01` |
| `r=1.590, z=-0.800` | `obr17`, `obv17` |
| `r=1.590, z=-0.725` | `obr16`, `obv16` |
| `r=1.590, z=-0.650` | `obr15`, `obv15` |
| `r=1.590, z=0.650` | `obr05`, `obv05` |
| `r=1.590, z=0.725` | `obr04`, `obv04` |
| `r=1.590, z=0.800` | `obr03`, `obv03` |
| `r=1.850, z=-0.300` | `obr14`, `obv14` |
| `r=1.850, z=-0.225` | `obr13`, `obv13` |
| `r=1.850, z=-0.150` | `obr12`, `obv12` |
| `r=1.850, z=-0.075` | `obr11`, `obv11` |
| `r=1.850, z=0.075` | `obr09`, `obv09` |
| `r=1.850, z=0.150` | `obr08`, `obv08` |
| `r=1.850, z=0.225` | `obr07`, `obv07` |
| `r=1.850, z=0.300` | `obr06`, `obv06` |

## Nearest-coordinate index errors

| Shot | Declared range | Mapped loop rows | Legacy index differs from signal identity | Flagged | Silent | Signal-match receipt |
|---:|---|---:|---:|---:|---:|---|
| 11766 | 11766–12416 | 12 | **9** | 9 | 0 | minimum winner 0.999954; unique winning indices |
| 21978 | 12417–30471 | 19 | **14** | 13 | 1 | minimum winner 0.999995; unique winning indices |
| 21983 | 12417–30471 | 19 | **14** | 13 | 1 | minimum winner 0.999993; unique winning indices |

The following grouped receipt names every mismatching row and reports the
legacy `residual_m` and `flag`. An arrow gives `address: legacy efm_index ->
signal efm_index`.

| Shot(s) | Mismatching rows | Legacy `residual_m` (m) | Legacy `flag` |
|---|---|---:|---|
| 11766 | `fl_cc02: 0->1`, `fl_cc03: 0->2`, `fl_cc05: 0->4` | 0.019956450559 | `non-unique: silop[0]` claimed by `fl_cc01, fl_cc02, fl_cc03, fl_cc05` |
| 11766 | `fl_p4l_1: 32->36`, `fl_p5l_1: 32->40` | 0.000399960518 | `non-unique: silop[32]` claimed by `fl_p4l_1, fl_p5l_1` |
| 11766 | `fl_p4l_4: 35->39`, `fl_p5l_4: 35->43` | 0.000565680842 | `non-unique: silop[35]` claimed by `fl_p4l_4, fl_p5l_4` |
| 11766 | `fl_p4u_4: 14->21`, `fl_p5u_1: 14->22` | 0.000410016060 | `non-unique: silop[14]` claimed by `fl_p3u_1, fl_p4u_4, fl_p5u_1` |
| 21978, 21983 | `fl_cc03: 0->2`, `fl_cc04: 0->3`, `fl_cc05: 0->4`, `fl_cc07: 0->6`, `fl_cc09: 0->8` | 0.019956450559 | `non-unique: silop[0]` claimed by `fl_cc01, fl_cc03, fl_cc04, fl_cc05, fl_cc07, fl_cc09` |
| 21978, 21983 | `fl_p4l_1: 32->36`, `fl_p5l_1: 32->40`, `fl_p6u_1: 32->26` | 0.000399960518 | `non-unique: silop[32]` claimed by `fl_p3l_1, fl_p4l_1, fl_p5l_1, fl_p6u_1` |
| 21978, 21983 | `fl_p4l_4: 35->39`, `fl_p5l_4: 35->43` | 0.000565680842 | `non-unique: silop[35]` claimed by `fl_p3l_4, fl_p4l_4, fl_p5l_4` |
| 21978, 21983 | `fl_p4u_4: 14->21`, `fl_p5u_1: 14->22` | 0.000410016060 | `non-unique: silop[14]` claimed by `fl_p3u_1, fl_p4u_4, fl_p5u_1` |
| 21978, 21983 | `fl_p5u_4: 17->25` | 0.000410020716 | `non-unique: silop[17]` claimed by `fl_p3u_4, fl_p5u_4` |
| 21978, 21983 | `fl_p6l_1: 34->44` | 0.000565690956 | empty — **accepted silently** |

The mechanism is visible in `imas_ambix/gs/geometry.py:948-968`: the loop
description is compared to every finite `silop` coordinate and the nearest
index is stored whenever its distance is below 50 mm. Only a second pass over
multiple claims adds the non-unique flag (`geometry.py:971-986`). A physically
wrong but unique nearest point, as for `fl_p6l_1`, cannot be detected by that
logic.

## Frozen-spine impact

The default campaign source used by the frozen benchmark delegates to
`scripts/heldout_mse_gate_eval.py::_campaign_table`. In the current checkout,
the six frozen shots 21978, 21983, 21985, 21986, 21989, and 22086 all reuse the
table built from shot 21978. It has **19 flux-loop rows: 18 flagged and one
unflagged (`fl_p6l_1`)**. The two frozen shots audited by waveform both have
14/19 wrong EFM identities, including the silent row.

The spine consumers divide into these two categories:

### Consumers of sensor coordinates

- `imas_ambix/gs/operator.py::_sensor_rows` copies every mapped `(r,z,angle)`
  into the forward operator, including flagged rows (`operator.py:743-779,
  875`). Those coordinates build the known-coil vacuum and other sensor-space
  Green columns; `flagged_channels` is only retained on the result
  (`operator.py:1019-1022`).
- `imas_ambix/latent/gs_solve.py::EquilibriumGrid.sensor_greens` evaluates the
  cell-to-sensor Green function at every `SensorMapping.r/z`
  (`gs_solve.py:454-490`).
- `imas_ambix/latent/patch_basis.py::_coil_sensor_matrix` and
  `PatchBasis.from_table` use the coordinates for coil-to-sensor and
  cell-to-sensor matrices (`patch_basis.py:97-125, 225-232`).
- `imas_ambix/latent/boundary_disc.py::sensor_signature_arrays` copies the
  coordinates into the filament signature used by `fit_current_centroid`
  (`boundary_disc.py:96-160`); its passive sensor couplings use the same arrays.
  Thus aliasing can move the disc seed before either benchmark substrate runs.
- `imas_ambix/spine_bench/runner.py::_magnetics_residual` calls
  `sensor_greens` for the final forward prediction (`runner.py:161-189`). The
  reported 0.7401030841614555 is the aggregate of this coordinate-dependent
  check.

### Consumers of ordering only

- The alignment portion of `scripts/spine_label_factory.py::factory_shot_payloads`
  takes the channel names returned in `sensor_map` order and reindexes measured,
  vacuum, scale, and mask vectors by name (`spine_label_factory.py:100-128`).
  This preserves row ordering but does not independently use `r` or `z`.
- The runner's per-slice accumulators and substrate pairing consume the already
  aligned payload order; they do not reinterpret sensor coordinates.

The distinction does not protect the metric. `factory_shot_payloads` constructs
the coordinate-dependent operator, grid, and patch basis before doing its
order-only alignment. Its `present` mask tests only whether a channel name
exists; it does not consult `ForwardOperator.flagged_channels`. Therefore the
legacy reader **flagged most aliasing but the frozen spine accepted those rows
as ordinary inputs and score targets**.

## Bounded repair and re-measurement

1. Bind each acquisition address to a stable flux-loop structure identity in
   the machine map; do not infer identity from `(r,z)`. Cover all five duplicate
   loop-coordinate groups and the unique-looking `fl_p6l_1` failure.
2. Materialise that identity in the transform-backed `sensor_map`, preserving
   channel order separately from coordinates. Add cross-range tests with
   waveform-correlation receipts and an explicit assertion that no two present
   loop channels claim one EFM identity.
3. Run the focused geometry-adapter and spine-source tests, then run the frozen
   CPU benchmark on its required SLURM lane with the corrected identity source.
4. Report the corrected greens-matvec residual and its signed/relative change
   from 0.7401030841614555. Keep the old value labelled as legacy-join
   provenance; it must not be treated as an unaffected physics baseline.

No production code was changed by this assessment.
