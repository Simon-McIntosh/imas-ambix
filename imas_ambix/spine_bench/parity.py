"""Numeric parity gate for the frozen shot set.

A parity run compares two equilibrium paths on identical inputs and asks whether
they are the same physics.  That question needs numbers, so this module registers
one tolerance per scored metric and the structural admissibility rules a stamp
must satisfy before its numbers mean anything at all.

Reference values
----------------
Every reference EXCEPT the sensor-space misfit is the aggregate median of the
committed stamp
``results/physics-spine-v0-mast-heldout-6-0447fb2e0d-98dci4-clu-3141.yaml``
(schema ``spine-bench/1.3``, 18 rows, ``git_dirty: false``, all six frozen shots
under one campaign signature).  That stamp is also the SLOWEST of the four
committed stamps on both gated arms, so using it as the timing reference cannot
be gamed by comparing against a lucky fast run.

The sensor-space misfit did not exist at schema 1.3, so its reference is the
earliest stamp that measures it -- :data:`BEFORE_PATH_STAMP` at schema 1.4.
Every deterministic metric in that stamp is bit-identical to the 1.3 stamp it
succeeds, which is what makes the two references mutually consistent: the run
that first measured the misfit reproduced everything already gated, digit for
digit, so the new number describes the same engine the older references do.

Which arms are gated
--------------------
Only the two hard-topology-read arms, ``grid-delstar`` and ``greens-matvec``.
The ``greens-matvec+connectivity`` arm is measured but NOT gated: it converged on
0.833 and 0.917 of its slices in the two stamps that ran it, so it is already
losing slices and cannot carry a no-loss gate.  Its reproduction spread is also
wide (``profile_smoothread_rms`` moved 117% between those two stamps), which is
the smooth read's own open question, not a parity signal.

Margin policy
-------------
The four committed stamps span three schema versions and four engine commits on
one host, which separates what the code determines from what the node does.
Three metric classes fall out, each with its own margin mechanism:

**Deterministic reproduction** (axis, LCFS, profile, FSA roughness) showed
*exactly zero* spread across all four stamps -- identical to every printed digit.
There is therefore no noise envelope to clear, and the tolerance is a change
budget instead: a measured value may reach ``REPRODUCTION_CHANGE_BUDGET`` times
its reference.  The multiplier is calibrated against the grid rather than chosen
for roundness.  The benchmark grid is 65x97 over the MAST limiter extent
(R 0.195-1.900 m, Z +-1.825 m), so one radial cell is 2.66 cm.  The axis ceiling
is 4 x 0.0234 = 0.094 cm, about a thirtieth of that cell, so anything that passes
is still deeply sub-cell.  Meanwhile the smallest geometry defect that could
plausibly survive review -- one poloidal-field filament displaced by a single
grid cell -- moves the reconstructed axis by order 1 cm, two orders of magnitude
above the ceiling.  The budget is thus loose enough to absorb an equivalent
geometry assembled in a different order (whose perturbation enters at the
solver's own convergence tolerance) and far too tight for a real error to pass.

**Quantized solve health** (converged and confined fractions) is gated at exact
equality with 1.0, with no margin, because a margin here cannot be expressed.
The aggregate is a median over shots of a per-shot fraction with at most six
scored slices, so it moves in steps of at least 1/6 = 0.167: any margin below one
step is unreachable, and any margin at or above one step silently accepts a lost
slice.  Losing a slice is exactly the failure this gate exists to catch.

**Timing** (throughput) spread 16.1% on ``greens-matvec`` and 19.3% on
``grid-delstar`` across the four stamps while the physics metrics stayed
bit-identical, which makes that spread shared-node scatter rather than code.
``THROUGHPUT_REGRESSION_BUDGET`` allows a 25% shortfall against the reference:
above the 19.3% observed so the gate does not fire on node noise, and far below
the cost of a real algorithmic regression (rebuilding the Green's matrices per
slice instead of per campaign, or adding a solve, each at least doubles the
wall).  ``solve_wall_ms_per_slice`` is the exact reciprocal of throughput, so it
is deliberately not registered -- one number, one gate.  ``latency_ms_p99`` is
also ungated: it spread up to 30%, is a tail statistic over at most five timed
slices per shot, and carries no parity information.

**Sensor-space misfit** (``magnetics_residual_whitened_rms``) is the fourth
class and needs its own mechanism, because it is the only registered metric that
is an ABSOLUTE misfit rather than a difference between two solves.  A change
budget expressed as a multiple of the reference would be meaningless here: four
times a misfit that already sits near one whitened sigma admits an equilibrium
with no relationship to the magnetics at all.  The margin is therefore a
fractional one, derived the way the timing margin is -- from the measured spread
of the quantity under a change that is not the one being gated.

That spread is available directly, because a stamp measures the residual twice.
``grid-delstar`` and ``greens-matvec`` solve the same equilibrium from the same
measurements on the same geometry through two different substrates -- a gridded
elliptic inversion and an analytic Green's matvec -- and their residuals differ
by 7.3e-4 in relative terms (0.739561 against 0.740103).  A two-slice ad-hoc run
put the same figure at 1.3e-3.  That is what this number does when the
reconstruction path changes and the machine does not, which is the perturbation
class a geometry-source cutover belongs to, so it is the floor a margin must
clear.

The ceiling was measured rather than assumed, because a fractional budget has no
grid cell to calibrate against the way the reproduction budget does.  Displacing
every sensor radially and recomputing the residual on the SAME converged
equilibrium moves the median by 0.20% at 1 mm, 0.88% at 5 mm, 1.65% at 10 mm and
3.86% at one radial grid cell (26.6 mm).  Those figures are lower bounds: the
probe holds the known-coil vacuum term fixed, so a real change of geometry source
moves the number by more than the same displacement does here.

``MAGNETICS_RESIDUAL_REGRESSION_BUDGET`` therefore allows 1%.  That is about
eight times the observed cross-path spread and below the 1.65% a 10 mm sensor
displacement costs, so the gate fires on any geometry error from roughly a
quarter of a grid cell upward while leaving substrate-level differences alone.  A
5% budget would have admitted a whole-cell displacement, which is the error this
metric exists to catch.  The gate is one-sided, matching the metric's direction:
a run that fits the magnetics BETTER than the before-path is not a parity
failure.

Both hard-read arms carry it, like solve health and throughput: the misfit is an
absolute property of each arm's own equilibrium, not a cross-arm comparison, so
each arm's number stands on its own.

Comparing two descriptions of one machine
-----------------------------------------
The 1% budget above is a SAME-SOURCE number: its floor and its ceiling were both
measured with one description of the machine held fixed while the reconstruction
substrate or the engine moved.  Scoring a change of geometry SOURCE against it
asks a different question, because two independent descriptions of one machine
disagree about the machine itself, and that disagreement is measurable before any
equilibrium is solved.

For the pair this module carries -- the campaign tables of
:data:`BEFORE_PATH_STAMP` against the machine-artifact reader of
:data:`AFTER_PATH_STAMP` -- every driven Green's column was tabulated on both
sides over the 95 channels both map.  The columns agree to 1.2% or better on the
twelve poloidal-field windings, to 2.9% on the solenoid (an amplitude
difference: the sources state different turn weights), and to between 1.8% and
13.0% on the eight case circuits, whose section discretisation differs.

That disagreement reaches the misfit without a solve, because the vacuum term is
linear in the driven currents.  Exchanging all 21 columns at the measured
currents perturbs the whitened prediction by a median 0.0777 per slice against a
before-path residual of 0.7401, and reducing the perturbed residual the way the
runner does moves the metric by 2.85%.  That propagation is arithmetic on the
two sources' tables and the before-path residual -- it reads nothing from the
after-path solve -- and it reproduces the instrumented exchange-all-columns
measurement to every printed digit, which is what makes it a prediction rather
than a fit.

Three things the propagation cannot see set the margin, and each is measured:

**Alignment.**  The propagation adds the perturbation at whatever alignment the
before-path solution happens to give it.  The same perturbation moves the metric
by 0.55% if it is orthogonal to the residual and by 12.6% if it is parallel (the
per-slice triangle bound), so the propagated value carries no margin of its own:
alignment alone spans more than an order of magnitude, and re-solving is free to
rotate it.

**Solve feedback.**  How far re-solving rotates it was measured directly.
Restoring one source's winding filament lattices into the other's machine and
RE-SOLVING moved the metric to the opposite side of the prediction the same swap
made with the solve held fixed: -0.0184 predicted against +0.0065 measured, a
discrepancy of 3.37% of the reference.  The case-discretisation swap put the
same discrepancy at 0.06%.  ``SOLVE_FEEDBACK_ALLOWANCE`` takes the larger,
because it is the one a vacuum-term change of this size actually produced.

**Stated-weight calibration.**  The incumbent path carries a solenoid response
scale fitted against the very magnetics this metric scores, while a path that
takes its weights from the source is judged on fidelity alone.  Sweeping the
solenoid weight across the interval the two sources jointly admit moves the
misfit by 0.93% to 1.71% of the reference, which is the size of the advantage
the fitted path holds over the source-stated one.

:data:`SOURCE_CUTOVER_RESIDUAL_BUDGET` is the sum of those three and nothing
else.  It does not answer the question the 1% budget answers, and deliberately
gives up that budget's discrimination against a localised geometry defect: this
pair's ADMITTED case-section disagreement already costs more than a whole-cell
sensor displacement does, so no cross-source budget wide enough to accept the
pair can also reject a displacement that small.  What carries that duty across a
cutover is the column census -- both paths driving the same columns, channel for
channel -- and not this metric.  What the budget does still catch is a source
read that loses or mis-weights a column: the first artifact read of this pair
entered the solenoid at 1/328 of its stated weight and scored 1.210, 63% above
the reference and eight times the budget.

The choice of budget is a property of the comparison, not of the stamp, so it is
named at the call: :class:`ComparisonKind` selects which margin the misfit is
scored under, and every other tolerance is identical across them.

Migration comparator after correcting sensor identity
------------------------------------------------------
The migration comparator is the clean frozen-set stamp
``physics-spine-v0-mast-heldout-6-3d3ed8c56d-98dci4-clu-3141-efm-campaign-``
``signal-identity-1d41708300ef.yaml``.  It replaces 14 of 19 flux-loop rows
whose nearest-description-coordinate join selected the wrong EFM column.  The
replacement is one-to-one and independent of description coordinates: each raw
acquisition waveform is bound to its unique highest-correlation ``silop_x``
column on shot 21978 (minimum winning correlation 0.9999948002).

The corrected residuals are 0.6890842357802178 on ``greens-matvec`` and
0.6889769655392166 on ``grid-delstar``.  Against the aliased greens reference
0.7401030841614555 those are signed relative movements of -6.8934786887% and
-6.9079726482%, respectively.  Both solves still clear all 15 registered
tolerances.  The stamp does carry a coverage qualification: shots 21985 and
21986 score five of six attempted slices on each arm.  Because an aggregate
median of six per-shot fractions can hide those losses, path comparison also
pins exact per-shot attempted/scored coverage; a migrated seam cannot silently
lose another slice or change the population being compared.

The per-seam margin is re-derived from this sound pair rather than inherited
from the aliased pair.  Their relative cross-substrate residual spread is
0.0155670723%.  The same eight-times safety multiple used to place the prior
same-source margin above its measured cross-path spread gives
``MIGRATION_SEAM_RESIDUAL_BUDGET = 0.1245365782%``.  Thus the budget retains the
gate's established margin mechanism while every numerical input now comes from
the corrected comparator.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from imas_ambix.spine_bench.shots import FROZEN_SHOTSET, SHOTSET_VERSION

#: The reference stamp every tolerance below is derived from.
REFERENCE_STAMP = "physics-spine-v0-mast-heldout-6-0447fb2e0d-98dci4-clu-3141.yaml"

#: The measured before-path: the frozen gate run on the engine as it stands at
#: the cutover, clean-tree and two-arm.  The absolute table above is the floor a
#: run must clear on its own; this stamp is what an after-path run is compared
#: AGAINST, because parity asks whether the new path reproduces the path being
#: replaced, not one that predates it.  The two differ: 94 engine commits
#: separate them, over which the reproduction metrics moved (axis 0.0234 to
#: 0.0253 cm, profile RMS 0.0162 to 0.0190) while staying far inside the change
#: budget, which is the budget absorbing real solver evolution as designed.
#: The stamp named here is the schema-1.4 re-run of that same engine state: it
#: reproduces every deterministic metric of the ca82e50d2f stamp bit for bit and
#: adds the sensor-space misfit, which is why it can serve as the before-path for
#: a comparison that includes the new metric.
BEFORE_PATH_STAMP = "physics-spine-v0-mast-heldout-6-08ae0dee74-98dci4-clu-3141.yaml"

#: The measured after-path: the same frozen gate run with the machine read from
#: its own description instead of the campaign tables.  It is named here because
#: the cross-source budget below was propagated from THIS pair's tabulated
#: Green's columns, so the pair is what makes that derivation reproducible.
AFTER_PATH_STAMP = "physics-spine-v0-mast-heldout-6-e76b0dc65c-98dci4-clu-3141.yaml"

#: Clean frozen-set comparator with the flux-loop identity join corrected.
MIGRATION_REFERENCE_STAMP = (
    "physics-spine-v0-mast-heldout-6-3d3ed8c56d-98dci4-clu-3141-"
    "efm-campaign-signal-identity-1d41708300ef.yaml"
)

#: Arms carrying the gate: the hard-topology-read solves on both substrates.
GATED_ARMS = ("grid-delstar", "greens-matvec")


class ComparisonKind(enum.StrEnum):
    """What differs between the two stamps, which fixes the misfit's budget.

    Every other tolerance is identical in both: a reproduction metric, a solve
    fraction and a throughput are properties of one arm's own solve and do not
    know where the geometry came from.  Only the sensor-space misfit does,
    because it is the one metric a change of machine description moves without
    any error being present.
    """

    #: One description of the machine; the substrate, engine or schema moved.
    SAME_SOURCE = "same-source"

    #: Two independent descriptions of one machine.
    SOURCE_CUTOVER = "source-cutover"

    #: Successive consumer seams against the corrected-identity comparator.
    MIGRATION_SEAM = "migration-seam"


#: A deterministic reproduction metric may reach this multiple of its reference.
REPRODUCTION_CHANGE_BUDGET = 4.0

#: Throughput may fall this fraction below its reference before failing.
THROUGHPUT_REGRESSION_BUDGET = 0.25

#: The whitened magnetics misfit may rise this fraction above its reference.
MAGNETICS_RESIDUAL_REGRESSION_BUDGET = 0.01

#: Residual from the aliased nearest-description-coordinate join.  Retained as
#: provenance for the corrected comparator's signed movement, not as its gate.
ALIASED_NEAREST_COORDINATE_RESIDUAL = 0.7401030841614555

#: Corrected-identity residuals from :data:`MIGRATION_REFERENCE_STAMP`.
MIGRATION_REFERENCE_RESIDUALS = {
    "greens-matvec": 0.6890842357802178,
    "grid-delstar": 0.6889769655392166,
}

#: Cross-substrate spread of the corrected residual pair, relative to the
#: primary greens-matvec reference.
MIGRATION_CROSS_SUBSTRATE_RESIDUAL_SPREAD = (
    abs(
        MIGRATION_REFERENCE_RESIDUALS["greens-matvec"]
        - MIGRATION_REFERENCE_RESIDUALS["grid-delstar"]
    )
    / MIGRATION_REFERENCE_RESIDUALS["greens-matvec"]
)

#: Preserve the measured safety multiple used by the same-source gate while
#: deriving the numerical margin entirely from the corrected comparator.
MIGRATION_SEAM_MARGIN_MULTIPLIER = 8.0
MIGRATION_SEAM_RESIDUAL_BUDGET = (
    MIGRATION_SEAM_MARGIN_MULTIPLIER * MIGRATION_CROSS_SUBSTRATE_RESIDUAL_SPREAD
)

#: Exchanging every driven Green's column between the two descriptions, at the
#: measured currents and with the plasma solve held at the before-path solution,
#: moves the misfit this fraction.  Arithmetic on the two sources' tabulated
#: columns and the before-path residual; no after-path solve is read.
CROSS_SOURCE_COLUMN_PROPAGATION = 0.0285

#: What the hold-fixed propagation misses by ignoring the solve re-adapting to
#: the changed vacuum term.  Measured on the winding-lattice swap, that
#: re-adaptation landed this fraction of the reference away from the hold-fixed
#: prediction of the same swap, and on the far side of it.
SOLVE_FEEDBACK_ALLOWANCE = 0.0337

#: The advantage a path holds when its solenoid response scale was fitted
#: against the scored magnetics.  Sweeping that weight across the interval the
#: two descriptions jointly admit moves the misfit by up to this fraction.
STATED_WEIGHT_CALIBRATION_ALLOWANCE = 0.0171

#: The whitened magnetics misfit may rise this fraction above its reference when
#: the two paths read different descriptions of one machine.  It is the sum of
#: the propagated column disagreement and the two mechanisms that propagation
#: cannot see -- nothing is added for roundness.
SOURCE_CUTOVER_RESIDUAL_BUDGET = (
    CROSS_SOURCE_COLUMN_PROPAGATION
    + SOLVE_FEEDBACK_ALLOWANCE
    + STATED_WEIGHT_CALIBRATION_ALLOWANCE
)

#: The metric scored under the sensor-space-misfit margin mechanism.
MAGNETICS_RESIDUAL_METRIC = "magnetics_residual_whitened_rms"


@dataclass(frozen=True)
class ParityTolerance:
    """One metric's admissible band on one arm.

    ``lower_better`` mirrors the direction in the metric registry and decides
    which side of ``bound`` passes: a lower-better metric must not exceed it, a
    higher-better metric must not fall below it.
    """

    metric: str
    arm: str
    reference: float
    bound: float
    lower_better: bool
    basis: str

    def passes(self, value: float) -> bool:
        """Return whether a measured aggregate value clears this tolerance."""
        if value != value:  # NaN never passes: a missing measurement is a failure
            return False
        return value <= self.bound if self.lower_better else value >= self.bound


def _reproduction(metric: str, arm: str, reference: float) -> ParityTolerance:
    return ParityTolerance(
        metric=metric,
        arm=arm,
        reference=reference,
        bound=reference * REPRODUCTION_CHANGE_BUDGET,
        lower_better=True,
        basis=(
            "deterministic across the four committed stamps (zero spread); "
            f"change budget {REPRODUCTION_CHANGE_BUDGET:g}x reference"
        ),
    )


def _solve_health(metric: str, arm: str) -> ParityTolerance:
    return ParityTolerance(
        metric=metric,
        arm=arm,
        reference=1.0,
        bound=1.0,
        lower_better=False,
        basis=(
            "quantized in steps of >=1/6 by the six-slice cap, so the only "
            "expressible gate is exact equality with 1.0"
        ),
    )


def _throughput(arm: str, reference: float) -> ParityTolerance:
    return ParityTolerance(
        metric="throughput_slices_per_core_s",
        arm=arm,
        reference=reference,
        bound=reference * (1.0 - THROUGHPUT_REGRESSION_BUDGET),
        lower_better=False,
        basis=(
            "reference is the slowest committed stamp on this arm; budget "
            f"{THROUGHPUT_REGRESSION_BUDGET:.0%} clears the observed "
            "shared-node timing scatter (16.1% / 19.3%)"
        ),
    )


def _magnetics_residual(arm: str, reference: float) -> ParityTolerance:
    return ParityTolerance(
        metric=MAGNETICS_RESIDUAL_METRIC,
        arm=arm,
        reference=reference,
        bound=reference * (1.0 + MAGNETICS_RESIDUAL_REGRESSION_BUDGET),
        lower_better=True,
        basis=(
            "absolute misfit, so the margin is fractional rather than a change "
            f"budget; {MAGNETICS_RESIDUAL_REGRESSION_BUDGET:.0%} sits above the "
            "7.3e-4 cross-substrate spread and below the 1.65% a 10 mm sensor "
            "displacement costs"
        ),
    )


def _cross_source_magnetics_residual(arm: str, reference: float) -> ParityTolerance:
    return ParityTolerance(
        metric=MAGNETICS_RESIDUAL_METRIC,
        arm=arm,
        reference=reference,
        bound=reference * (1.0 + SOURCE_CUTOVER_RESIDUAL_BUDGET),
        lower_better=True,
        basis=(
            "two descriptions of one machine, so the margin is the measured "
            "column disagreement propagated through the metric "
            f"({CROSS_SOURCE_COLUMN_PROPAGATION:.2%}) plus what that "
            f"propagation cannot see: solve feedback "
            f"({SOLVE_FEEDBACK_ALLOWANCE:.2%}) and the stated-weight "
            f"calibration asymmetry ({STATED_WEIGHT_CALIBRATION_ALLOWANCE:.2%})"
        ),
    )


def _migration_seam_magnetics_residual(arm: str, reference: float) -> ParityTolerance:
    return ParityTolerance(
        metric=MAGNETICS_RESIDUAL_METRIC,
        arm=arm,
        reference=reference,
        bound=reference * (1.0 + MIGRATION_SEAM_RESIDUAL_BUDGET),
        lower_better=True,
        basis=(
            "successive consumer seam against the corrected-identity frozen "
            "comparator; margin is "
            f"{MIGRATION_SEAM_MARGIN_MULTIPLIER:g}x its measured "
            f"cross-substrate residual spread "
            f"({MIGRATION_CROSS_SUBSTRATE_RESIDUAL_SPREAD:.6%})"
        ),
    )


#: Which misfit mechanism each comparison kind scores under.  Every other
#: tolerance is built the same way for both, so this mapping is the whole of the
#: difference between them.
_RESIDUAL_TOLERANCE = {
    ComparisonKind.SAME_SOURCE: _magnetics_residual,
    ComparisonKind.SOURCE_CUTOVER: _cross_source_magnetics_residual,
    ComparisonKind.MIGRATION_SEAM: _migration_seam_magnetics_residual,
}


#: The registered gate. Reproduction metrics live on ``greens-matvec`` because
#: that arm is the one scored against the grid baseline; solve health and
#: throughput are gated on both hard-read arms.
PARITY_TOLERANCES: tuple[ParityTolerance, ...] = (
    _reproduction("axis_reproduce_cm", "greens-matvec", 0.023398397214736938),
    _reproduction("lcfs_reproduce_cm", "greens-matvec", 0.00827735914166669),
    _reproduction("profile_reproduce_rms", "greens-matvec", 0.016230933274198972),
    _reproduction("fsa_d_roughness_nrho32", "greens-matvec", 0.34638211670150604),
    _reproduction("fsa_d_roughness_nrho96", "greens-matvec", 0.17119111122512323),
    _reproduction("fsa_d_roughness_nrho32", "grid-delstar", 0.3431132830615231),
    _reproduction("fsa_d_roughness_nrho96", "grid-delstar", 0.17419467402462369),
    _solve_health("converged_fraction", "greens-matvec"),
    _solve_health("confined_fraction", "greens-matvec"),
    _solve_health("converged_fraction", "grid-delstar"),
    _solve_health("confined_fraction", "grid-delstar"),
    _throughput("greens-matvec", 0.1903098833334077),
    _throughput("grid-delstar", 0.3206149571463025),
    _magnetics_residual("greens-matvec", 0.7401030841611733),
    _magnetics_residual("grid-delstar", 0.7395612732950347),
)


@dataclass(frozen=True)
class ParityFailure:
    """One tolerance that a stamp did not clear."""

    metric: str
    arm: str
    measured: float
    bound: float
    reference: float

    def describe(self) -> str:
        """Return a one-line human-readable statement of the miss."""
        return (
            f"{self.arm}/{self.metric}: measured {self.measured:.6g} "
            f"outside bound {self.bound:.6g} (reference {self.reference:.6g})"
        )


@dataclass(frozen=True)
class ParityReport:
    """The verdict on one stamp: structural admissibility plus metric outcomes."""

    admissibility: tuple[str, ...]
    failures: tuple[ParityFailure, ...]
    checked: int

    @property
    def ok(self) -> bool:
        """Return whether the stamp is admissible and clears every tolerance."""
        return not self.admissibility and not self.failures

    def describe(self) -> str:
        """Return a multi-line summary suitable for a log or a commit note."""
        if self.ok:
            return f"parity PASS: {self.checked} tolerances cleared"
        lines = [f"parity FAIL ({self.checked} tolerances checked)"]
        lines += [f"  inadmissible: {reason}" for reason in self.admissibility]
        lines += [f"  {failure.describe()}" for failure in self.failures]
        return "\n".join(lines)


def check_admissibility(stamp) -> tuple[str, ...]:
    """Return the reasons a stamp may not be scored as a frozen-set parity run.

    Partial scoring is a failure, not a qualified pass, so a stamp that is
    missing a shot, an arm, or the frozen label is rejected before any metric is
    compared.  A single campaign signature is required too: the frozen set was
    curated as one machine configuration, and a second signature would mean the
    run silently mixed geometries.
    """
    reasons: list[str] = []
    if stamp.shotset_version != SHOTSET_VERSION:
        reasons.append(
            f"shotset_version {stamp.shotset_version!r} is not the frozen "
            f"{SHOTSET_VERSION!r}"
        )
    expected_shots = {shot.shot_id for shot in FROZEN_SHOTSET}
    present_shots = {row.shot_id for row in stamp.shots}
    if missing := sorted(expected_shots - present_shots):
        reasons.append(f"missing frozen shots {missing}")
    if extra := sorted(present_shots - expected_shots):
        reasons.append(f"shots outside the frozen set {extra}")

    expected_roles = {shot.shot_id: shot.role for shot in FROZEN_SHOTSET}
    for row in stamp.shots:
        role = expected_roles.get(row.shot_id)
        if role is not None and row.role != role:
            reasons.append(
                f"shot {row.shot_id} role {row.role!r} is not the frozen role {role!r}"
            )
    signatures = {row.campaign_signature for row in stamp.shots}
    if len(signatures) > 1:
        reasons.append(f"multiple campaign signatures {sorted(signatures)}")

    for arm in GATED_ARMS:
        arm_shots = {
            row.shot_id
            for row in stamp.shots
            if row.substrate == arm and row.topology_read == "hard"
        }
        if absent := sorted(expected_shots - arm_shots):
            reasons.append(f"arm {arm} is missing shots {absent}")
    return tuple(reasons)


def _score(stamp, tolerances) -> list[ParityFailure]:
    failures = []
    for tolerance in tolerances:
        measured = stamp.aggregate.get(tolerance.arm, {}).get(
            tolerance.metric, float("nan")
        )
        if not tolerance.passes(measured):
            failures.append(
                ParityFailure(
                    metric=tolerance.metric,
                    arm=tolerance.arm,
                    measured=measured,
                    bound=tolerance.bound,
                    reference=tolerance.reference,
                )
            )
    return failures


def evaluate(stamp) -> ParityReport:
    """Score a benchmark stamp against the registered tolerances."""
    return ParityReport(
        admissibility=check_admissibility(stamp),
        failures=tuple(_score(stamp, PARITY_TOLERANCES)),
        checked=len(PARITY_TOLERANCES),
    )


def tolerances_from(
    stamp, kind: ComparisonKind = ComparisonKind.SAME_SOURCE
) -> tuple[ParityTolerance, ...]:
    """Re-derive the registered table against a measured stamp.

    Same metrics, same arms, same margin policy -- only the reference values move
    to what ``stamp`` measured.  This is how an after-path run is gated against
    the before-path it must reproduce rather than against a historical anchor.

    ``kind`` states what differs between the two runs being compared, which
    selects the misfit's budget: a substrate or engine change is scored against
    the same-source spread, a change of machine description against the column
    disagreement the two descriptions themselves carry.
    """
    residual_tolerance = _RESIDUAL_TOLERANCE[kind]
    derived: list[ParityTolerance] = []
    for tolerance in PARITY_TOLERANCES:
        reference = stamp.aggregate.get(tolerance.arm, {}).get(tolerance.metric)
        if reference is None:
            raise KeyError(
                f"stamp does not measure {tolerance.arm}/{tolerance.metric}, so it "
                "cannot serve as a parity reference"
            )
        if tolerance.metric == "throughput_slices_per_core_s":
            derived.append(_throughput(tolerance.arm, reference))
        elif tolerance.metric == MAGNETICS_RESIDUAL_METRIC:
            derived.append(residual_tolerance(tolerance.arm, reference))
        elif tolerance.reference == 1.0 and not tolerance.lower_better:
            derived.append(_solve_health(tolerance.metric, tolerance.arm))
        else:
            derived.append(_reproduction(tolerance.metric, tolerance.arm, reference))
    return tuple(derived)


def compare_paths(
    before, after, kind: ComparisonKind = ComparisonKind.SAME_SOURCE
) -> ParityReport:
    """Score an after-path stamp against the before-path it must reproduce.

    Both stamps must be admissible frozen-set runs, because a comparison between
    two differently-scoped runs is not a parity measurement.  The after-path is
    then held to the margin policy ``kind`` names, with the before-path's numbers
    as the reference, so a cutover that leaves the equilibrium unchanged passes
    and one that moves it by a multiple of the change budget does not.

    The default is the stricter of the two.  A caller comparing across geometry
    sources must say so, because the wider budget is only derivable once the two
    descriptions' Green's columns have been tabulated against each other.
    """
    admissibility = [f"before-path: {reason}" for reason in check_admissibility(before)]
    admissibility += [f"after-path: {reason}" for reason in check_admissibility(after)]
    before_coverage = {
        (row.shot_id, row.substrate, row.topology_read): (
            row.n_slices_attempted,
            row.n_slices_scored,
        )
        for row in before.shots
        if row.substrate in GATED_ARMS and row.topology_read == "hard"
    }
    after_coverage = {
        (row.shot_id, row.substrate, row.topology_read): (
            row.n_slices_attempted,
            row.n_slices_scored,
        )
        for row in after.shots
        if row.substrate in GATED_ARMS and row.topology_read == "hard"
    }
    for key in sorted(before_coverage.keys() | after_coverage.keys()):
        if before_coverage.get(key) != after_coverage.get(key):
            shot, arm, topology_read = key
            admissibility.append(
                f"coverage shot {shot} arm {arm} read {topology_read}: "
                f"before {before_coverage.get(key)}, after {after_coverage.get(key)}"
            )
    tolerances = tolerances_from(before, kind)
    return ParityReport(
        admissibility=tuple(admissibility),
        failures=tuple(_score(after, tolerances)),
        checked=len(tolerances),
    )
