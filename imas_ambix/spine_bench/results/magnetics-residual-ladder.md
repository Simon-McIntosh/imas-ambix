# The whitened magnetics misfit, stamp by stamp

Every banked stamp that measures `magnetics_residual_whitened_rms` on the frozen
six-shot set, in the order it was taken. The point of the sequence is that the
metric is a forward-model check: the frozen spine never fits these channels, so
the number moves only when the machine description behind the Green's functions
moves. A calibration campaign is therefore readable as a ladder — each landing
re-stamps, and the residual must not rise.

Two descriptions are benched. **`efm-campaign`** is the campaign's own static efm
arrays, the historical geometry, whose calibration was already spent against this
machine's data. **`machine-artifact`** is the published machine description, read
through `machine_artifact_arm.py`; its `revision` is the semantic identity of the
republication that was read, which is the only field separating two
republications of one machine.

| created | commit | description | revision | greens-matvec | grid-Δ* | what changed | entry point |
|---|---|---|---|---|---|---|---|
| 08-03 23:15 | `08ae0dee74` | efm-campaign | — | 0.7401030841612 | 0.7395612732950 | the metric itself lands | committed |
| 08-04 00:40 | `27aaed2bb9` | machine-artifact | unrecorded | 1.2098421278914 | 1.2100082308559 | first bench of a described machine | out-of-tree |
| 08-04 02:43 | `e76b0dc65c` | machine-artifact | unrecorded | 0.7639208113227 | 0.7637534273248 | drive the machine through the channels its source names | out-of-tree |
| 08-04 17:46 | `180455e846` | machine-artifact | unrecorded | 0.7632104919483 | 0.7631398908964 | probe axes and case-plate orientation corrected | out-of-tree |
| 08-04 23:35 | `60b10eae69` | efm-campaign | — | 0.7401030841612 | 0.7395612732950 | nova pin moved; incumbent unperturbed | committed |
| 08-04 23:36 | `60b10eae69` | efm-campaign | — | 0.7401030841612 | 0.7395612732950 | same, second host | committed |
| 08-05 00:28 | `a2adbc5028` | machine-artifact | `85993ba9491b` | 0.7632104919483 | 0.7631398908964 | corrected-registry revision, from committed code | committed |
| 08-05 00:28 | `a2adbc5028` | machine-artifact | `c20bc7e157fa` | 0.7632104919483 | 0.7631398908964 | ten turn counts restated on archive integers | committed |
| 08-05 00:28 | `a2adbc5028` | machine-artifact | `3aba565a2e40` | 0.7632104919483 | 0.7631398908964 | channel scales promoted; uniform density kept | committed |
| 08-05 13:29 | `a380769d9b` | efm-campaign | — | 0.7401030841612 | 0.7395612732950 | acquisition range setting divided out of the read | committed |
| 08-05 13:29 | `a380769d9b` | machine-artifact | `3aba565a2e40` | 0.7632104919483 | 0.7631398908964 | same read change, other description | committed |
| 08-05 16:57 | `09e0555384` | machine-artifact | `18c75c194937` | 0.7632104919483 | 0.7631398908964 | republished: P2 pack interconnection published | committed |

## What the ladder shows

**The residual never rose.** The artifact arm runs
1.2098 → 0.7639 → 0.7632 → 0.7632 (seven times), non-increasing at every rung. The
incumbent is bit-identical across four stamps on three different hosts, which is
what makes it usable as a bar rather than a moving target.

**Only two rungs moved it, and neither was a calibration.** The first was reading
the drive map correctly; the second was the corrected probe axes and case-plate
orientation, worth −0.000710 greens / −0.000614 grid. That second figure is an
**upper bound** on the geometry corrections, not a measurement of them: the pair
it spans also crosses a different conductor count, a moved registry digest and
eleven engine commits, and the artifact revision on its near side is no longer on
disk, so it cannot be decomposed further from here.

**Every calibration rung is flat, and each is flat for a stated reason** rather
than by coincidence:

- *Turn counts* (`85993ba9` → `c20bc7e1`) are inert. A poloidal channel measures
  its coil's own current with the turns already multiplied out, so the drive
  carries one ampere-turn per ampere and the vacuum term is built from that
  weight, never from the filament count. Positions, drive weights and multipliers
  are byte-identical across the pair.
- *Channel scales* (`c20bc7e1` → `3aba565a`) are not read by this forward model.
  Sensors reach the solve as position and angle only —
  `artifact_geometry.sensor_position_arrays` returns `r`, `z`, `angle_deg` and
  nothing else — so a promoted gain is description evidence the gate cannot see.
- *Winding lattices* never entered the description: the layout was derivable and
  the data refused it, so there was nothing to re-stamp.
- *The acquisition range normalisation* is flat to float equality, measured, not
  assumed — see below.
- *The passive calibration's publication* (`3aba565a` → `18c75c19`) is flat to
  float equality too. The resistance fit was refused, so nominal values stood; what
  the republication carries is the P2 pack interconnection moving to published,
  plus the accumulated sensor records — one more published field, one more
  generated, one fewer unresolved. Worth measuring rather than assuming, and the
  flatness is what says the closure below was scored against the *whole* current
  description rather than an older snapshot of it.

**The read change is not a discontinuity here.** Nineteen probe channels were
recorded at more than one acquisition range setting, so both arms were re-stamped
together to find out what dividing the setting out costs. Both came back
bit-identical: on all six frozen shots every channel with a measured setting sits
in a block recorded at the reference rung, so the read divides by exactly one —
372 bracketed reads, 6 refused, 204 unmeasured, zero divisions. The frozen set was
curated as one campaign configuration and that curation, made for other reasons,
also placed it inside one acquisition configuration. Stamps under schema 1.5
therefore remain directly comparable with 1.6 ones.

The two probes carrying most of the cross-source excess are why this could not
have closed the gap either: `obv06` and `obv14` each hold a single block at the
reference setting across the whole archive, so no range correction reaches them.

## The closure verdict: not met

The criterion was the artifact description's absolute misfit at or below the
incumbent's on both hard-read arms, scored under the untouched same-source budget.

| arm | incumbent | artifact | artifact − incumbent | verdict |
|---|---|---|---|---|
| greens-matvec | 0.7401031 | 0.7632105 | **+0.0231074** | artifact worse |
| grid-Δ* | 0.7395613 | 0.7631399 | **+0.0235786** | artifact worse |

`compare_paths(campaign → artifact, same-source)` **FAILS** on both arms — 0.76321
against a bound of 0.747504, and 0.76314 against 0.746957. The same pair **PASSES
15/15 under the source-cutover budget**, unchanged from every previous stamp, so
the cutover remains sound on the budget it was registered against: the qualified
cross-source status **stands and is not retired**. No tolerance was modified in
either direction.

The gap is stable to seven decimal places across nine artifact-arm stamps and four
artifact revisions, so it is not noise and it is not a solver setting. It is a
property of the description, and the decomposition below says which channels hold
it.

The two arms score the **same 72 channels**, so this is a like-for-like
comparison. The artifact's operator carries 96 sensor rows against the campaign's
97, but the three channels that differ (`fl_cc02`, `fl_cc10`, `ccbv10`) are not
measured on these shots — the verdict is not a channel census.

## Where the misfit is, and how far above the noise

`channel-gap-*.json` beside this file, both arms, from the same solves the closing
stamps ran. Each record reproduces its own stamp's aggregate to float equality,
which is what makes it a decomposition of the gate's number rather than of
something adjacent to it. Shares are of the mean square, so they add to one.

| channel | artifact share | residual | its measured floor | ratio to floor |
|---|---|---|---|---|
| `obr17` | 22.6% | 15.9 mT | 77 µT | **206×** |
| `obv06` | 11.4% | 12.8 mT | 86 µT | **149×** |
| `obv14` | 8.2% | 10.4 mT | 48 µT | **219×** |
| `obr04` | 6.9% | 12.8 mT | 51 µT | **253×** |
| `obr10` | 4.2% | 5.3 mT | 59 µT | **91×** |
| `ccbv25` | 3.3% | 76.3 mT | 417 µT | **183×** |

**The noise envelope is not the constraint anywhere.** Six channels carry 56.6% of
the misfit and every one of them sits between 91 and 253 times its own measured
floor. The pooled floor recomputed from the same per-channel measurements is
387 µT, which reproduces the envelope this ladder was given as its stretch target
— and the description stands two orders of magnitude above it on its worst
channels. Whatever is left is model error with a great deal of room in it, not an
instrument limit.

**The single largest channel is shared, not a cutover cost.** `obr17` carries the
most misfit on *both* arms (22.6% artifact, 25.0% incumbent) at ~206× its floor.
Neither description explains that probe. It is also the channel a promoted scale of
0.5011 was fitted for, which the gate never reads — consistent with the flat rung
above, and a standing lead rather than a closed item.

**The cutover's own +0.023 is an outboard-vertical-probe loss.** Comparing shares
channel by channel, the artifact is worse on the `obv` family — `obv06` +1.51% of
the total, `obv14` +1.23%, then `obv13`, `obv03`, `obv04` — and *better* on
`obr17` (−2.44%), `obr04` and `ccbv25`. So the description change trades radial
probes for vertical ones and loses on the trade, on exactly the outboard verticals
beside P4/P5 that the earlier attribution named. With noise, gain, turns, lattice
and now acquisition range all excluded by measurement, what is left for those
probes is pose or a conductor absent from both descriptions.

## The six physical gaps, re-dispositioned

The description states its own open gaps, and the republished manifest lists
exactly six. That list is the authority here, and it has moved in one place: the
turns gap now names only `p6_lower` and `p6_upper`, where it used to cover turn
magnitude, absolute polarity, the connections matrix and the P2 pack link.

| gap (the manifest's own wording) | disposition now | what moved |
|---|---|---|
| turns are not sourced for `p6_lower`, `p6_upper` | **structural**; still the only forward-model blocker | Ten of thirteen counts published on exact archive integers (P4/P5 = 23); the solenoid weight reconciled at 344.657 [332.97, 356.34], overlapping the independent stratum route; the P2 pack interconnection published. P6 is excited as ampere-turns, so no shot can identify its physical count — closed by evidence rather than left open. |
| passive material, resistance and electrical topology are not sourced | **bounded provisional**, inductive half now exact | The 57-circuit passive inductance matrix landed on corrected case geometry (reciprocity 3.0e-3). The resistive half stays nominal as a visible negative: the four-class fit converged in sample (+7.2%) then failed the promotion contract four ways, with the identifying measurement named. |
| poloidal probe channel-to-bank toroidal position assignment is not sourced | **closed search, still dual-valued** | Measured to be unobtainable from this archive: separating the 150° and 330° candidates needs ~50 kA of error-field-coil drive against the 12 kA it holds. Both candidates stay carried; no midpoint. |
| independent toroidal probe orientation is not sourced | **unresolved, non-blocking** (unchanged) | Corroborated rather than moved: the floor is error-field-coil-free, shifting 1.75 pT, so the 36 position-only probes are not what the residual waits on. |
| detailed toroidal-field winding geometry is not sourced | **not required for this lane** (unchanged) | Topology published; conductor elements unresolved and non-blocking under the locked axisymmetric-TF decision. |
| saddle traversal sign is not sourced | **explicit discrete choice** (unchanged) | No new evidence. Geometry stays direction-neutral, both signs carried, digest unmoved. |

Two of the six are now closed as far as this archive can close them — P6's turn
count and the bank assignment each have a *measured* reason why no shot identifies
them, which is a more useful state than "open". And none of the six is what holds
the remaining misfit: that sits on specific probe channels, and the two dominant
ones are neither mis-gained nor mis-scaled.

Timing metrics in the re-baselined pair ran two-up on one node and are contended;
residuals are deterministic and unaffected.
