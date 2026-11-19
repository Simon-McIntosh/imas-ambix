"""The one median the agent package computes for itself."""

from __future__ import annotations

import pytest

from imas_ambix.agent.median import median, median_or_none


def test_an_even_count_averages_the_two_middles() -> None:
    """The rule a hand-written copy gets wrong first.

    At an even count the median is the mean of the two middle values: picking
    either one is a different estimator, and one line of arithmetic separates
    the two answers. Every reader in this package -- the lane's gate waits, a
    benchmark's rates, a price table's rates -- is promised this rule, so it is
    pinned here once rather than at each caller.
    """
    assert median([0.0, 1.0, 2.0, 9.0]) == pytest.approx(1.5)
    # Order of the series is not its order of magnitude: the estimator sorts,
    # so an unsorted series is not a different answer.
    assert median([9.0, 1.0, 2.0, 0.0]) == pytest.approx(1.5)


def test_an_odd_count_is_the_middle_value() -> None:
    assert median([7.0, 1.0, 2.0]) == pytest.approx(2.0)


def test_an_empty_series_is_none_only_when_the_caller_asks_for_it() -> None:
    """Two contracts, one estimator: a measured figure and a missing table.

    ``median`` raises on an empty series, which is the standard library's
    behaviour and is correct for a caller that has already established it has
    a value; ``median_or_none`` returns ``None`` for a reader whose series is
    legitimately not measured yet, where a zero would be a measurement that
    was never taken.
    """
    assert median_or_none([]) is None
    with pytest.raises(ValueError):
        median([])


def test_the_two_contracts_are_one_estimator() -> None:
    """Both entry points answer from the same arithmetic.

    The none-contract exists for a reader whose series may be unmeasured; it
    is not a second estimator. Feeding both the same series has to give the
    same figure, because a reader comparing a lane median against a benchmark
    median is promised exactly that.
    """
    series = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0]
    assert median_or_none(series) == pytest.approx(median(series))


def test_the_estimator_averages_the_middles_of_this_series() -> None:
    """A six-value series: the two middles are 3.0 and 4.0."""
    assert median([3.0, 1.0, 4.0, 1.0, 5.0, 9.0]) == pytest.approx(3.5)
