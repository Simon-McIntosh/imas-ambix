"""The frozen-set parity gate: tolerances, and what makes a stamp inadmissible."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from imas_ambix.latent.gs_solve import SUBSTRATE_GREENS, SUBSTRATE_GRID
from imas_ambix.spine_bench import parity
from imas_ambix.spine_bench.schema import SpineBenchmarkStamp
from imas_ambix.spine_bench.shots import (
    AD_HOC_SHOTSET_VERSION,
    FROZEN_SHOTSET,
    SHOTSET_VERSION,
)

RESULTS = Path(__file__).parents[2] / "imas_ambix" / "spine_bench" / "results"


def _load(name: str) -> SpineBenchmarkStamp:
    payload = yaml.safe_load((RESULTS / name).read_text())
    return SpineBenchmarkStamp.model_validate(payload)


@pytest.fixture
def reference() -> SpineBenchmarkStamp:
    """A stamp whose every metric IS the registered reference for that metric.

    The absolute table spans two committed stamps, because the sensor-space
    misfit was registered at a later schema version than the other thirteen
    tolerances and the older stamp therefore does not measure it.  This fixture
    puts the two together: the historical reference, carrying the misfit values
    the current-schema stamp measured.  Every test below that perturbs one metric
    by a multiple of its budget depends on that identity holding -- otherwise the
    arithmetic is against one number and the gate against another.
    """
    stamp = _load(parity.REFERENCE_STAMP)
    misfit = _load(parity.BEFORE_PATH_STAMP).aggregate
    for arm in parity.GATED_ARMS:
        stamp.aggregate[arm][parity.MAGNETICS_RESIDUAL_METRIC] = misfit[arm][
            parity.MAGNETICS_RESIDUAL_METRIC
        ]
    return stamp


def _without(stamp: SpineBenchmarkStamp, **overrides) -> SpineBenchmarkStamp:
    """Return a deep copy of ``stamp`` with top-level fields replaced."""
    clone = copy.deepcopy(stamp)
    for name, value in overrides.items():
        setattr(clone, name, value)
    return clone


# --- the reference must satisfy its own gate ------------------------------


def test_reference_stamp_clears_every_registered_tolerance(reference):
    """A gate its own reference cannot pass would be miscalibrated."""
    report = parity.evaluate(reference)
    assert report.ok, report.describe()
    assert report.checked == len(parity.PARITY_TOLERANCES)


def test_gated_arms_are_the_substrate_constants_the_solver_defines():
    """The arm names are solver identities, not strings that may drift apart."""
    assert parity.GATED_ARMS == (SUBSTRATE_GRID, SUBSTRATE_GREENS)


def test_every_tolerance_names_a_registered_metric():
    """A tolerance on an unregistered metric would score a number with no meaning."""
    from imas_ambix.spine_bench.schema import METRICS

    for tolerance in parity.PARITY_TOLERANCES:
        assert tolerance.metric in METRICS
        assert tolerance.arm in parity.GATED_ARMS


def test_tolerance_direction_matches_the_metric_registry():
    """A gate applied on the wrong side of the bound would pass every regression."""
    from imas_ambix.spine_bench.schema import METRICS

    for tolerance in parity.PARITY_TOLERANCES:
        expected = METRICS[tolerance.metric].direction == "lower_better"
        assert tolerance.lower_better is expected, tolerance.metric


def test_the_connectivity_arm_is_not_gated(reference):
    """It loses slices, so it cannot carry a no-loss gate; measured, not scored."""
    arms = {tolerance.arm for tolerance in parity.PARITY_TOLERANCES}
    assert f"{SUBSTRATE_GREENS}+connectivity" not in arms
    # and the reference records it below a no-loss fraction, which is the reason
    connectivity = reference.aggregate[f"{SUBSTRATE_GREENS}+connectivity"]
    assert connectivity["converged_fraction"] < 1.0


# --- metric outcomes -----------------------------------------------------


def test_a_reproduction_metric_beyond_its_change_budget_fails(reference):
    """Sub-cell today; a multiple of the budget is a geometry error, not noise."""
    stamp = copy.deepcopy(reference)
    reference_axis = stamp.aggregate[SUBSTRATE_GREENS]["axis_reproduce_cm"]
    stamp.aggregate[SUBSTRATE_GREENS]["axis_reproduce_cm"] = reference_axis * (
        parity.REPRODUCTION_CHANGE_BUDGET + 1.0
    )
    report = parity.evaluate(stamp)
    assert not report.ok
    assert [f.metric for f in report.failures] == ["axis_reproduce_cm"]


def test_a_reproduction_metric_inside_its_change_budget_passes(reference):
    """The budget absorbs an equivalent geometry assembled in a different order."""
    stamp = copy.deepcopy(reference)
    for arm, metric in (
        (SUBSTRATE_GREENS, "axis_reproduce_cm"),
        (SUBSTRATE_GREENS, "lcfs_reproduce_cm"),
        (SUBSTRATE_GREENS, "profile_reproduce_rms"),
    ):
        stamp.aggregate[arm][metric] *= parity.REPRODUCTION_CHANGE_BUDGET * 0.9
    assert parity.evaluate(stamp).ok


def test_one_lost_slice_fails_the_solve_health_gate(reference):
    """Five of six converged is the failure the gate exists to catch."""
    stamp = copy.deepcopy(reference)
    stamp.aggregate[SUBSTRATE_GREENS]["converged_fraction"] = 5.0 / 6.0
    report = parity.evaluate(stamp)
    assert not report.ok
    assert [f.metric for f in report.failures] == ["converged_fraction"]


def test_throughput_within_the_regression_budget_passes(reference):
    """Shared-node timing scatter must not fire the gate."""
    stamp = copy.deepcopy(reference)
    for arm in parity.GATED_ARMS:
        stamp.aggregate[arm]["throughput_slices_per_core_s"] *= (
            1.0 - parity.THROUGHPUT_REGRESSION_BUDGET * 0.75
        )
    assert parity.evaluate(stamp).ok


def test_throughput_beyond_the_regression_budget_fails(reference):
    """A halved throughput is an algorithmic regression, not node noise."""
    stamp = copy.deepcopy(reference)
    stamp.aggregate[SUBSTRATE_GRID]["throughput_slices_per_core_s"] *= 0.5
    report = parity.evaluate(stamp)
    assert not report.ok
    assert [f.metric for f in report.failures] == ["throughput_slices_per_core_s"]


def test_a_missing_metric_fails_rather_than_being_skipped(reference):
    """An unmeasured gate item is a failure; silence must never read as a pass."""
    stamp = copy.deepcopy(reference)
    del stamp.aggregate[SUBSTRATE_GREENS]["axis_reproduce_cm"]
    report = parity.evaluate(stamp)
    assert not report.ok
    assert [f.metric for f in report.failures] == ["axis_reproduce_cm"]


def test_a_missing_arm_fails_every_tolerance_registered_on_it(reference):
    """Dropping an arm cannot reduce the number of things checked."""
    stamp = copy.deepcopy(reference)
    del stamp.aggregate[SUBSTRATE_GRID]
    report = parity.evaluate(stamp)
    expected = sum(1 for t in parity.PARITY_TOLERANCES if t.arm == SUBSTRATE_GRID)
    assert len(report.failures) == expected


# --- structural admissibility: partial scoring is a failure --------------


def test_an_ad_hoc_labelled_stamp_is_inadmissible(reference):
    """The gate scores the frozen metric, never an override."""
    stamp = _without(reference, shotset_version=AD_HOC_SHOTSET_VERSION)
    reasons = parity.check_admissibility(stamp)
    assert any("frozen" in reason for reason in reasons)
    assert not parity.evaluate(stamp).ok


def test_a_stamp_missing_a_frozen_shot_is_inadmissible(reference):
    """Five of six shots is not a six-shot run."""
    dropped = FROZEN_SHOTSET[-1].shot_id
    stamp = _without(
        reference, shots=[row for row in reference.shots if row.shot_id != dropped]
    )
    reasons = parity.check_admissibility(stamp)
    assert any(str(dropped) in reason for reason in reasons)


def test_a_stamp_carrying_a_shot_outside_the_frozen_set_is_inadmissible(reference):
    """A substituted shot changes what the metric means."""
    stamp = copy.deepcopy(reference)
    stamp.shots[0].shot_id = 99999
    reasons = parity.check_admissibility(stamp)
    assert any("99999" in reason for reason in reasons)


def test_an_altered_role_is_inadmissible(reference):
    """Roles document why each shot is in the set; a changed role is a changed set."""
    stamp = copy.deepcopy(reference)
    stamp.shots[0].role = "ad-hoc"
    reasons = parity.check_admissibility(stamp)
    assert any("role" in reason for reason in reasons)


def test_a_mixed_campaign_signature_is_inadmissible(reference):
    """Two signatures mean the run silently spanned two geometries."""
    stamp = copy.deepcopy(reference)
    stamp.shots[0].campaign_signature = "mp78-fl46-fc1004-lim37-9425ae4a8bf3bc15"
    reasons = parity.check_admissibility(stamp)
    assert any("campaign signature" in reason for reason in reasons)


def test_an_arm_missing_a_shot_is_inadmissible(reference):
    """Both gated arms must cover every frozen shot for a comparison to exist."""
    stamp = copy.deepcopy(reference)
    stamp.shots = [
        row
        for row in stamp.shots
        if not (row.substrate == SUBSTRATE_GRID and row.shot_id == 21985)
    ]
    reasons = parity.check_admissibility(stamp)
    assert any(SUBSTRATE_GRID in reason and "21985" in reason for reason in reasons)


def test_the_reference_stamp_is_structurally_admissible(reference):
    """Guards the admissibility rules against being unsatisfiable in practice."""
    assert parity.check_admissibility(reference) == ()
    assert reference.shotset_version == SHOTSET_VERSION


# --- the before-path and the old-vs-new comparison ------------------------


@pytest.fixture
def before_path() -> SpineBenchmarkStamp:
    """The measured before-path an after-path run must reproduce."""
    return _load(parity.BEFORE_PATH_STAMP)


def test_the_before_path_stamp_is_a_genuine_clean_two_arm_frozen_run(before_path):
    """A stamp with a dirty tree or a partial arm cannot anchor a comparison."""
    assert parity.check_admissibility(before_path) == ()
    assert before_path.env.git_dirty is False
    assert {row.substrate for row in before_path.shots} == set(parity.GATED_ARMS)
    assert {row.topology_read for row in before_path.shots} == {"hard"}
    assert all(
        row.n_slices_scored == row.n_slices_attempted for row in before_path.shots
    )


def test_the_before_path_clears_the_registered_absolute_table(before_path):
    """The engine at the cutover satisfies the gate it will be measured under."""
    assert parity.evaluate(before_path).ok


def test_a_path_compared_against_itself_is_perfect_parity(before_path):
    """The comparison's fixed point: identical stamps cannot fail."""
    assert parity.compare_paths(before_path, before_path).ok


def test_the_comparison_rebases_the_reference_onto_the_before_path(before_path):
    """Tolerances must follow the measured before-path, not the historical anchor."""
    derived = {
        (t.arm, t.metric): t.reference for t in parity.tolerances_from(before_path)
    }
    measured = before_path.aggregate["greens-matvec"]["axis_reproduce_cm"]
    assert derived[("greens-matvec", "axis_reproduce_cm")] == measured
    assert measured != pytest.approx(
        next(
            t.reference
            for t in parity.PARITY_TOLERANCES
            if t.arm == "greens-matvec" and t.metric == "axis_reproduce_cm"
        )
    )


def test_an_after_path_that_moves_the_equilibrium_fails_the_comparison(before_path):
    """The check the cutover exists to survive."""
    after = copy.deepcopy(before_path)
    after.aggregate["greens-matvec"]["axis_reproduce_cm"] *= (
        parity.REPRODUCTION_CHANGE_BUDGET + 1.0
    )
    report = parity.compare_paths(before_path, after)
    assert not report.ok
    assert [f.metric for f in report.failures] == ["axis_reproduce_cm"]


def test_an_after_path_that_loses_a_slice_fails_the_comparison(before_path):
    after = copy.deepcopy(before_path)
    after.aggregate["greens-matvec"]["converged_fraction"] = 5.0 / 6.0
    assert not parity.compare_paths(before_path, after).ok


def test_an_inadmissible_after_path_fails_even_with_matching_numbers(before_path):
    """Identical metrics over the wrong shot set is not parity."""
    after = _without(before_path, shotset_version=AD_HOC_SHOTSET_VERSION)
    report = parity.compare_paths(before_path, after)
    assert not report.ok
    assert any("after-path" in reason for reason in report.admissibility)


def test_a_stamp_missing_a_gated_metric_cannot_serve_as_a_reference(before_path):
    """A reference with a hole would silently drop a gate item."""
    incomplete = copy.deepcopy(before_path)
    del incomplete.aggregate["greens-matvec"]["axis_reproduce_cm"]
    with pytest.raises(KeyError, match="cannot serve as a parity reference"):
        parity.tolerances_from(incomplete)
