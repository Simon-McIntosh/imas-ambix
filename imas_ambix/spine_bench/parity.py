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
"""

from __future__ import annotations

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

#: Arms carrying the gate: the hard-topology-read solves on both substrates.
GATED_ARMS = ("grid-delstar", "greens-matvec")

#: A deterministic reproduction metric may reach this multiple of its reference.
REPRODUCTION_CHANGE_BUDGET = 4.0

#: Throughput may fall this fraction below its reference before failing.
THROUGHPUT_REGRESSION_BUDGET = 0.25

#: The whitened magnetics misfit may rise this fraction above its reference.
MAGNETICS_RESIDUAL_REGRESSION_BUDGET = 0.01

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


def tolerances_from(stamp) -> tuple[ParityTolerance, ...]:
    """Re-derive the registered table against a measured stamp.

    Same metrics, same arms, same margin policy -- only the reference values move
    to what ``stamp`` measured.  This is how an after-path run is gated against
    the before-path it must reproduce rather than against a historical anchor.
    """
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
            derived.append(_magnetics_residual(tolerance.arm, reference))
        elif tolerance.reference == 1.0 and not tolerance.lower_better:
            derived.append(_solve_health(tolerance.metric, tolerance.arm))
        else:
            derived.append(_reproduction(tolerance.metric, tolerance.arm, reference))
    return tuple(derived)


def compare_paths(before, after) -> ParityReport:
    """Score an after-path stamp against the before-path it must reproduce.

    Both stamps must be admissible frozen-set runs, because a comparison between
    two differently-scoped runs is not a parity measurement.  The after-path is
    then held to the same margin policy with the before-path's numbers as the
    reference, so a cutover that leaves the equilibrium unchanged passes and one
    that moves it by a multiple of the change budget does not.
    """
    admissibility = [f"before-path: {reason}" for reason in check_admissibility(before)]
    admissibility += [f"after-path: {reason}" for reason in check_admissibility(after)]
    tolerances = tolerances_from(before)
    return ParityReport(
        admissibility=tuple(admissibility),
        failures=tuple(_score(after, tolerances)),
        checked=len(tolerances),
    )
