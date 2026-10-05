"""The lane's availability percentage, the share of the gate width still open."""

from __future__ import annotations

import pytest

from imas_ambix.agent.lane import availability_percent


def test_availability_is_the_open_share_of_the_effective_width() -> None:
    assert availability_percent(13, 16) == 81.25


def test_full_gate_reports_zero_availability() -> None:
    assert availability_percent(0, 16) == 0.0


def test_headroom_past_the_width_clamps_to_full_availability() -> None:
    assert availability_percent(20, 16) == 100.0


def test_paused_gate_reports_zero_availability() -> None:
    assert availability_percent(4, 8, paused=True) == 0.0


def test_absent_headroom_omits_the_key() -> None:
    assert availability_percent(None, 16) is None


def test_non_positive_effective_width_omits_the_key() -> None:
    assert availability_percent(4, 0) is None


def test_negative_headroom_clamps_to_zero_rather_than_dividing_negative() -> None:
    assert availability_percent(-3, 16) == 0.0


@pytest.mark.parametrize("bad", ["13", object()])
def test_non_numeric_headroom_omits_the_key(bad: object) -> None:
    assert availability_percent(bad, 16) is None


@pytest.mark.parametrize("non_finite", [float("inf"), float("-inf"), float("nan")])
def test_non_finite_headroom_omits_the_key(non_finite: float) -> None:
    """An infinite or NaN headroom yields no percentage, not a NaN one."""
    assert availability_percent(non_finite, 16) is None


@pytest.mark.parametrize("non_finite", [float("inf"), float("-inf"), float("nan")])
def test_non_finite_effective_width_omits_the_key(non_finite: float) -> None:
    """An infinite or NaN width yields no percentage, not a NaN one."""
    assert availability_percent(4, non_finite) is None


def test_configured_and_effective_width_differ_so_the_denominator_is_pinned() -> None:
    """Half the configured width open is 50%, not 25% of the configured width.

    The configured width is 16 and the width in force is 8 with headroom 4, so
    the gate is half open. Dividing by the configured width instead of the
    effective width would read 25.0 here.
    """
    assert availability_percent(4, 8) == 50.0
