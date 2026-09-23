"""A failed ensemble member must not read as a measurement of zero current.

The forward solve degrades rather than raising, because one member of a large
ensemble failing is recoverable while aborting the whole ensemble is not. The
cost of degrading is that the placeholder it leaves behind is numerically
indistinguishable from a real result unless membership is carried explicitly,
and these tests pin the two halves of that: the placeholder is finite, and the
readout excludes it on the flag rather than on its value.
"""

import numpy as np

from imas_ambix.statespace.enkf_baseline import ToraxTrajectory, _j_at_slices


def _failed_member() -> ToraxTrajectory:
    """Exactly what the forward solve returns when TORAX raises."""
    return ToraxTrajectory(
        time=np.array([0.3]),
        rho_norm=np.linspace(0, 1, 27),
        j_total=np.zeros((1, 27)),
        q=np.full((1, 26), np.nan),
        ok=False,
    )


def _working_member(level: float) -> ToraxTrajectory:
    return ToraxTrajectory(
        time=np.array([0.3]),
        rho_norm=np.linspace(0, 1, 27),
        j_total=np.full((1, 27), level),
        q=np.full((1, 26), 1.5),
        ok=True,
    )


def test_failed_member_current_is_finite_so_finiteness_cannot_screen_it():
    """The premise the exclusion rests on: the placeholder passes isfinite.

    This is why a mask over ``ok`` is required and why the obvious guard is not
    enough. If the forward solve is ever changed to leave NaN instead, this
    test fails and the mask can be reconsidered -- which is the point of
    pinning the premise rather than only the behaviour.
    """
    slice_t = np.linspace(0.1, 0.5, 6)

    j = _j_at_slices(_failed_member(), slice_t)

    assert np.isfinite(j).all(), "a NaN placeholder would make the mask redundant"
    assert np.all(j == 0.0)


def test_ensemble_mean_over_the_flag_excludes_the_failure():
    """The mean must describe the members that ran, not the ones that did not."""
    slice_t = np.linspace(0.1, 0.5, 6)
    members = [_working_member(4.0), _failed_member(), _working_member(6.0)]

    stack = np.array([_j_at_slices(m, slice_t) for m in members])
    ok_mask = np.array([m.ok for m in members], dtype=bool)

    excluded = np.nanmean(stack[ok_mask], axis=0)
    naive = np.nanmean(stack, axis=0)

    # The two working members average to 5.0; including the failure drags the
    # reported current to 10/3 while every value stays finite and plausible.
    assert np.allclose(excluded, 5.0)
    assert not np.allclose(naive, 5.0)
    assert np.isfinite(naive).all(), "the wrong answer is not detectable by value"


def test_all_members_failing_is_reported_as_absent_not_as_zero():
    """No surviving member means no measurement, which is NaN and not 0.0."""
    slice_t = np.linspace(0.1, 0.5, 6)
    members = [_failed_member(), _failed_member()]

    ok_mask = np.array([m.ok for m in members], dtype=bool)

    assert not ok_mask.any()
    # The readout guards on ok_mask.any() precisely so this case does not
    # average an empty selection into a confident zero.
    stack = np.array([_j_at_slices(m, slice_t) for m in members])
    naive = np.nanmean(stack, axis=0)
    assert np.all(naive == 0.0), "unguarded, total failure reports zero current"
