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

## What the ladder shows

**The residual never rose.** The artifact arm runs
1.2098 → 0.7639 → 0.7632 → 0.7632 (six times), non-increasing at every rung. The
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

The gap is stable to seven decimal places across eight artifact-arm stamps and
three artifact revisions, so it is not noise and it is not a solver setting. It is
a property of the description, and the per-channel decomposition beside this file
says which channels hold it.

## What this ladder did not test

The closing stamp read revision `3aba565a2e40`, the newest machine description on
disk. The passive-calibration work that landed afterwards did not republish the
artifact — its resistance fit was a recorded negative, so nominal values stood,
but its positive rider (the P2 pack interconnection, published from archive
evidence) is **not** in the revision benched here. Whether that rider moves the
residual is untested.

Timing metrics in the closing pair ran two-up on one node and are contended; the
residuals are deterministic and unaffected.
