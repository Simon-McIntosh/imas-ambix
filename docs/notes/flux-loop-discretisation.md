# Flux-loop discretisation assessment

## Result

The point-flux-loop family containing `fl_cc05` and `fl_cc09` is represented
by **one stored `(R, Z)` point per loop in all three sources**. Neither
level 1, level 2, nor EFM stores an ordered point set for these loops. EFM also
records a toroidal extent of `2π` for all 46 finite loops at shot 12417, but
that is a scalar extent attached to one `(R, Z)` position, not a discretised
trajectory. Every point on the implied full toroidal circle has the same
`R` and `Z`, so changing toroidal position cannot produce a 2.148 m separation
in the poloidal plane.

Level 2 does contain a genuinely multi-point **saddle-loop** representation:
the lower, middle, and upper saddle families each have `r`, `z`, and `phi`
arrays of shape `(12, 28)`, hence 28 ordered points per saddle loop. Those are
the distinct `b_field_tor_probe_saddle_*` arrays and do not contain the
`fl_cc*` point-flux-loop family.

The 2.148100509 m `fl_cc09` disagreement is therefore not two vertices from
one stored loop and not a mesh-discretisation difference. It is the distance
between level 1's single repeated description point and the single position
used by the signal-identified EFM column; level 2 is exactly coincident with
EFM on this loop. The same mechanism gives 1.200100938 m for `fl_cc05`.

## Representation in each source

The two range representatives, shots 11766 and 12417, were inspected directly.

| Source | Field consulted | Stored shape | Effective geometry shape | Points per point-flux-loop |
|---|---|---:|---:|---:|
| Level 1, shot 11766 | `amb/<fl_name>.attrs["description"]`, parsed for its one `r=..., z=...` pair | 14 acquired loop signal arrays, each `(7500,)` | `(14, 2)` | **1** |
| Level 1, shot 12417 | same field | 28 signal arrays `(11289,)`; `fl_cc10` is `(7500,)` | `(29, 2)` | **1** |
| Level 2, both shots | `magnetics/flux_loop_geometry_channel`, `flux_loop_r`, `flux_loop_z` | `(44,)`, `(44,)`, `(44,)` | `(44, 2)` | **1** |
| EFM, shot 11766 | `efm/silop_r`, `efm/silop_z` | `(46,)`, `(46,)` | `(46, 2)` finite | **1** |
| EFM, shot 12417 | same fields | `(78,)`, `(78,)`, source-declared as `[1, 1, 46]` | `(46, 2)` finite plus 32 NaN padding rows | **1** |

The level-1 geometry is metadata on each measured channel rather than a
separate position array. Each description has exactly one parsable coordinate
pair. The signal-array shapes in the table are included to keep the distinction
explicit: the thousands of samples are measurements over time, not thousands
of geometry points.

At shot 12417, `efm/silop_dphi` is also stored as `(78,)` with 46 finite values
and 32 NaNs. The 46 finite values all equal `2π` within the precision already
recorded for the source. Shot 11766 has no `silop_dphi` field. In neither case
does EFM publish per-loop `R`, `Z`, or `phi` vertices.

### The separate multi-point saddle family

For both representative shots, level 2 has:

| Family | Fields | Shape | Loops | Points per loop |
|---|---|---:|---:|---:|
| Lower saddle | `b_field_tor_probe_saddle_l_{r,z,phi}` | `(12, 28)` each | 12 | **28** |
| Middle saddle | `b_field_tor_probe_saddle_m_{r,z,phi}` | `(12, 28)` each | 12 | **28** |
| Upper saddle | `b_field_tor_probe_saddle_u_{r,z,phi}` | `(12, 28)` each | 12 | **28** |

This establishes that the store can represent a loop as a point trajectory,
but does so only for the saddle-loop sensor class. The full stored point sets
for the two centre-column loops under investigation remain the following
singletons at shot 12417:

| Loop | Level-1 full point set `(R,Z)` m | Level-2 full point set `(R,Z)` m | EFM full point set `(R,Z)` m |
|---|---|---|---|
| `fl_cc05` | `[(0.180000000, 1.215000000)]` | `[(0.178499997, 0.014900000)]` | `[(0.178499997, 0.014900000)]`, `silop[4]` |
| `fl_cc09` | `[(0.180000000, 1.215000000)]` | `[(0.178499997, -0.933099985)]` | `[(0.178499997, -0.933099985)]`, `silop[8]` |

The corresponding Euclidean separations are:

| Loop | Level 1 to level 2 | Level 1 to EFM | Level 2 to EFM |
|---|---:|---:|---:|
| `fl_cc05` | 1.200100938 m | 1.200100938 m | **0 m** |
| `fl_cc09` | **2.148100509 m** | **2.148100509 m** | **0 m** |

The level-1 point `(0.180, 1.215)` is reused for multiple centre-column
addresses. Its nearest EFM point is `silop[0]`, signal-identified as `fl_cc01`,
at a still non-zero distance of 0.019956451 m. In contrast, the measured
`fl_cc09` waveform uniquely selects `silop[8]` with correlation
0.999998914393. Thus the 2.148 m value is not a distance from the level-1
representative to another vertex of `fl_cc09`; no such vertex set exists. It
is a same-hardware, different-single-position disagreement, and the repeated
level-1 point lies near the first centre-column loop rather than the ninth.

## EFM carries measurements and positions

EFM carries both sides needed by a reconstruction:

| Shot | Position arrays | Experimental input measurements | Computed fit | Per-loop fit statistic |
|---:|---|---|---|---|
| 11766 | `silop_r`, `silop_z`: `(46,)`, `(46,)` | `silop_x`: `(50, 46)` | `silop_c`: `(50, 46)` | `silop_chisq`: `(50, 46)` |
| 12417 | `silop_r`, `silop_z`: raw `(78,)`, `(78,)`, 46 finite each | `silop_x`: `(61, 46)` | `silop_c`: `(61, 46)` | `silop_chisq`: `(61, 46)` |

The `silop_x` metadata describes it as the input experimental fitted magnetic
flux-probe signal, in `Wb / rad`; `silop_c` is the corresponding computed
signal. The second dimension is 46 in both, and exactly 46 finite position
pairs are present. EFM therefore constrains on 46 loop measurement columns
with one associated position pair per column. The raw length 78 at shot 12417
is shared-dimension padding to the 78 magnetic-probe slots, not 78 flux-loop
points and not multiple points per loop.

## Position comparison against the reconstruction

EFM does not publish loop names, so an index must not be assigned from
coordinate proximity when testing geometry. Identities were established
without positions by interpolating every acquired level-1 `amb` loop waveform
to EFM time and correlating it against all 46 `silop_x` columns. The accepted
matches all exceed correlation 0.9999, have unique best EFM indices, and cover
14 loops at shot 11766 plus 28 at shot 12417. This is every reliably shared
level-1/EFM loop occurrence.

One occurrence is deliberately not absorbed into the counts: shot-12417
`fl_cc10` has a truncated `(7500,)` signal against an `(11289,)` group time
base and its attempted best EFM correlation is only -0.0183. It therefore does
not establish an EFM identity. Level 1's `fl_p6u_1` does establish an EFM
identity but has no level-2 counterpart, leaving 41 three-way shared loop
occurrences.

“Agreeing” below means exact zero Euclidean separation after the independent
signal identity join.

| Shot | Comparison | Reliably shared | Agreeing | Differing | Maximum separation |
|---:|---|---:|---:|---:|---|
| 11766 | EFM vs level 1 | 14 | 0 | 14 | 1.200100938 m, `fl_cc05` |
| 11766 | EFM vs level 2 | 14 | 9 | 5 | 0.437069046 m, `fl_p4l_1` |
| 12417 | EFM vs level 1 | 28 | 0 | 28 | **2.148100509 m, `fl_cc09`** |
| 12417 | EFM vs level 2 | 27 | 13 | 14 | 0.437069046 m, `fl_p4l_1` |
| **Both** | **EFM vs level 1** | **42** | **0** | **42** | **2.148100509 m** |
| **Both** | **EFM vs level 2** | **41** | **22** | **19** | **0.437069046 m** |

The reconstruction uses its own `efm/silop_r` and `efm/silop_z` setup arrays,
aligned one-for-one with `silop_x`; it does not consume either store's point
metadata directly. Those EFM setup positions agree exactly with level 2 for
22 of the 41 three-way occurrences and with level 1 for 0 of 42 pairwise
occurrences. They also agree exactly with level 2 for both centre-column cases
that exposed the metre-scale tail. This establishes that the reconstruction's
own centre-column geometry is the level-2 geometry, while retaining the
important qualification that EFM and level 2 still have 19 genuine residuals,
up to 0.437069046 m, elsewhere in the loop inventory.

## Classification

There is **no discretisation or alternate-representative-vertex component** in
the point-flux-loop residual:

- EFM versus level 1: 42 reliable shared occurrences comprise 0 exact,
  **0 discretisation/alternate-vertex**, and **42 genuine stored-position
  disagreements**.
- EFM versus level 2: 41 reliable shared occurrences comprise **22 exact**,
  **0 discretisation/alternate-vertex**, and **19 genuine stored-position
  disagreements**.
- For `fl_cc05` and `fl_cc09` specifically, level 2 and EFM are exact and the
  level-1 singleton is displaced by 1.200100938 m and 2.148100509 m,
  respectively.

The population is therefore a mixture of exact EFM/level-2 agreements and
genuine source-position disagreements, not a mixture of loop discretisations.
The metre-scale centre-column discrepancy is a genuine disagreement between
singleton representative coordinates for the same signal-proven hardware;
it cannot be explained as two points on the same stored continuous loop.

## Evidence and reproducibility

- `/work/projects/imas_gpu/store-bench/mmte-loop-discretisation-validated-20260816.log`
  — asserted representation shapes, saddle-family separation, independently
  signal-matched aggregate counts, singleton point sets and pairwise distances;
  `RESULT PASS`, exit 0.
- `/work/projects/imas_gpu/store-bench/mmte-loop-discretisation-inventory-20260816.log`
  — source metadata and all position, measured-input, computed-output and fit
  array shapes; exit 0.
- `/work/projects/imas_gpu/store-bench/mmte-loop-discretisation-efm-shapes-20260816.log`
  — full EFM position vectors showing the 46 finite plus 32 NaN-padded layout
  at shot 12417; exit 0.

The measurements use source commit
`9a95276cd9b9db6af5eb1e11b2cb06a781c6bc7f`. No production code was changed.
