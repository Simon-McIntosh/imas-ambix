"""The medians the agent package computes, stated once.

Several readers summarise a published series with a median: the lane's gate
waits, a benchmark's rates, a price table's rates. Each has to agree on what a
median is, and the rule that a copy gets wrong first is the even count, where
the estimator averages the two middle values rather than picking one of them.
So the estimator is taken from the standard library, which already states that
rule, and the two contracts callers here need are named once each:

- :func:`median` is the standard library's -- it raises on an empty series,
  which is right for a caller that has already established it has a value.
- :func:`median_or_none` is the same estimator for a caller reading not-yet-
  measured figures, where an empty series means "no value" and a zero would be
  a measurement that was never taken.
"""

from __future__ import annotations

import statistics
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = ["median", "median_or_none"]


def median(values: Iterable[float]) -> float:
    """The median of *values*, which must not be empty."""
    return float(statistics.median(values))


def median_or_none(values: Iterable[float]) -> float | None:
    """The median of *values*, or ``None`` when there is nothing to average."""
    values = list(values)
    return float(statistics.median(values)) if values else None
