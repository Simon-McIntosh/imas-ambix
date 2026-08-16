# Origin of symmetry-axis polygon evaluations

## Finding

No production geometry source, operator builder, solve grid, or escaped-centroid
path currently emits a polygon-kernel target at exactly `R=0`. The only exact
axis targets in the repository are synthetic inputs constructed directly by
`tests/test_polygon_singularity.py` to exercise the analytic limit.

The premise that the escaped centroid exposed an axis evaluation is therefore
refuted. The escaped centroid is not inserted into an evaluation set. It is used
only to compute its radial clearance from the limiter, after which `disc_read`
returns `None` because that clearance is non-positive.

All measurements below used source commit
`f8258f84ff8eb4f2f682249573cc949f9a2e21c0` and shot 21978, the first frozen-spine
shot and the shot whose transform-backed table produced the clearest centroid
escape.

## Quantitative census

“Distinct targets” counts coordinates in the table or grid once. “Kernel target
entries” counts the coordinates again for every polygon section passed to
`polygon_greens`; this is the actual number of target entries seen by the
kernel during the instrumented operator build.

| Evaluation path | Distinct targets | Polygon sections / calls | Kernel target entries | Exact `R=0` entries | Minimum target `R` |
|---|---:|---:|---:|---:|---:|
| Normal campaign operator, shot 21978 | 96 sensors | 8 | 768 | **0** | 0.17849999666213989 m |
| Transform-table operator associated with the escaped centroid | 95 sensors | 4 | 380 | **0** | 0.17849999666213989 m |
| Normal 65×97 solve grid | 6,305 grid points | not passed to polygon by `build_operator` | 0 | **0** | 0.19524440169334412 m |
| Transform-table 65×97 solve grid | 6,305 grid points | not passed to polygon by `build_operator` | 0 | **0** | 0.19524440169334412 m |
| Default limiter-masked operator plasma basis, either table | 84 points | uses the filament/cylinder path, not polygon | 0 | **0** | 0.19524440169334412 m |
| Escaped-centroid portion of `disc_read` | no evaluation set constructed | 0 | 0 | **0** | not applicable |
| Direct axis regression input | 3 synthetic points per axis call | 1 | 3 | **3** | 0 m |

The focused regression suite makes two exact-axis calls of three points each,
so it deliberately supplies six `R=0` kernel target entries across the suite.
The first call pins finiteness and the physical values; the second compares the
analytic limit with a near-axis evaluation.

The full transform-backed benchmark independently recorded 36 payloads over six
shots. All **36/36** returned at the same non-positive-clearance branch, with
zero other exit mechanisms. Thus the one-slice trace is not a special case of
the escaped cohort.

## End-to-end call paths

### Normal operator build

The ordinary operator route is:

1. `CampaignGeometrySource.table_for(21978)` obtains the campaign
   `GeometryTable`.
2. `build_operator(table)` calls `_sensor_rows(table)` and copies each
   `SensorMapping.r` and `.z` into the operator target arrays.
3. For a passive circuit represented by a `PolygonSection`, `build_operator`
   calls `polygon_section_column`.
4. `polygon_section_column` passes those sensor arrays unchanged to
   `polygon_greens`.

The component that emits polygon evaluation points on this route is therefore
the table's `sensor_map`, not a plasma grid. The instrumented build observed 96
finite sensor targets, no zero-radius coordinate, and eight calls over the same
target set. The smallest sensor radius was 0.17849999666213989 m.

### Grid-backed optional passive coupling

The only production route that passes a solve grid to the polygon kernel is the
optional passive-current stage:

1. `EquilibriumGrid.from_table` constructs
   `rg = linspace(max(limiter_r.min(), 0.06), limiter_r.max(), nr)`.
2. `EquilibriumGrid.__init__` forms `grid.flat_r` and `grid.flat_z` from that
   one-dimensional grid.
3. If and only if `DiscReadConfig.passive_k > 0`, `disc_read` calls
   `passive_coupling_matrices`.
4. For polygon-backed passive circuits, `passive_coupling_matrices` passes
   `grid.flat_r` and `grid.flat_z` to `polygon_greens`.

For shot 21978, `limiter_r.min()` is 0.19524440169334412 m, above the 0.06 m
numerical floor. The inner grid edge is therefore the inner limiter, not the
symmetry axis. Both measured 65×97 grids contain 0 of 6,305 points at `R=0`.
The frozen spine also uses the default `passive_k=0`, so this optional polygon
route is not entered during its boundary read.

### Transform table and escaped centroid

The escaped-centroid route is:

1. `TransformGeometrySource.table_for(21978)` runs
   `transform_machine_description`, `geometry_table_from_description`, and the
   declared probe-orientation join.
2. The spine factory builds its `EquilibriumGrid` and `PatchBasis` from that
   table. Any polygon calls made during the operator builds use the 95 positive-
   radius sensor coordinates: 0 of 380 repeated kernel target entries are at
   `R=0`.
3. `run_stamp` calls `disc_read` for a payload.
4. `fit_current_centroid` evaluates candidate filaments with `hybrid_greens`,
   the finite-area cylinder path, against the fixed sensor coordinates. It does
   not call the polygon kernel and does not make the candidate centroid a field
   target.
5. On the first slice, the order-one moment seed is already outside the vessel
   at `(R, Z) = (7.26058167199104, -1.15713028830696) m`. The unconstrained
   filament fit then returns
   `(363812710.066013, -10792672.174583) m`.
6. `disc_read` computes the limiter crossings at that height and then
   `d_minor = min(r0 - r_hfs, r_lfs - r0)`. The measured value is
   `-363812708.166013 m`, so the `d_minor <= 0` guard returns `None` immediately.

The functions after that guard would select grid cells around a valid centroid
and, only with an opt-in passive stage, could call the polygon/grid coupling.
They never run in the escaped case. The diverging centroid causes an early exit;
it causes neither a zero-radius default nor an axis target.

### The actual `R=0` emitter

`test_symmetry_axis_uses_finite_physical_limit` constructs
`target_r = np.zeros(3)` and calls `polygon_greens` directly. The continuity
test constructs the same three-point axis set directly before comparing it with
`R=1e-4 m`. There is no intervening table, grid, operator, adapter, or centroid
calculation. These regression functions are the complete end-to-end call path
for exact-axis evaluation in the current tree.

The proxy benchmark does not introduce another source: its target constructor
drops every point below the declared 0.2 m machine bore.

## Candidate-mechanism assessment

### Grid inner edge placed at zero

**Not the mechanism.** The solve-grid constructor explicitly takes the larger
of the inner limiter radius and 0.06 m. In the measured normal and transformed
tables, the inner edge is 0.19524440169334412 m. The coarser operator plasma
basis also starts at the inner limiter and has 0 of 84 points at zero radius.

### Missing-coordinate default

**Not the mechanism.** The instrumented normal and transformed operator tables
contain no zero-radius sensor mapping. Their minimum mapped sensor radius is
0.17849999666213989 m. Missing mappings are excluded rather than defaulted to
the origin; no code on the traced paths substitutes `(0, 0)` for an absent
coordinate.

### Consequence of the diverging centroid

**Not the mechanism.** The centroid fit uses the cylinder kernel at the table's
fixed sensor targets. The fitted centroid is checked against the limiter and is
never appended to the grid or sensor evaluation arrays. Its divergence triggers
the early `d_minor <= 0` return.

### Direct synthetic axis input

**This is the mechanism.** The regression test deliberately constructs exact
axis points so the public polygon kernel's analytic limit is pinned even though
the current physical consumers do not request it.

## Is the analytic limit load-bearing?

The analytic `R=0` branch in `imas_ambix/gs/polygon.py` is **not load-bearing in
normal operator or frozen-spine operation today**. Removing it would not change
either measured production build because both have zero axis targets, and the
escaped centroid does not reach the kernel.

It remains a valid defensive and mathematical guard for the public kernel. A
field target on the symmetry axis has a finite physical limit: poloidal flux and
radial field vanish by symmetry, while axial field remains finite. Computing
that limit analytically is preferable to clamping or returning a fabricated
value. The branch is load-bearing for the direct regression contract and for
any future caller that deliberately evaluates the axis, but it is not evidence
that the present machine geometry requires an axis evaluation.

## Evidence

- `/work/projects/imas_gpu/store-bench/mmte-r-zero-origin-operator-census-rerun-20260816.log`
  — instrumented normal and transform-table operator builds, including every
  polygon target count and both grid minima; command exit 0.
- `/work/projects/imas_gpu/store-bench/mmte-benchmark-source-disc-branches-cohort-20260815.log`
  — all 36 transform-backed payloads take the non-positive-clearance branch;
  zero other exits.
- `/work/projects/imas_gpu/store-bench/mmte-benchmark-source-centroid-stages-20260815.log`
  — first-slice moment seed and escaped filament-fit coordinates.
- `/work/projects/imas_gpu/store-bench/mmte-r-zero-origin-focused-20260816.log`
  — three axis-limit regression tests passed in 0.16 s.

No production code was changed for this assessment.
