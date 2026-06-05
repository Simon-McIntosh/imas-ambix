"""Unit tests for internal diagnostics evidence probes."""

from __future__ import annotations

import numpy as np

from imas_ambix.statespace.internal_diagnostics_probe import (
    act_key_assessment,
    atm_has_usable_near_axis,
)


def test_atm_has_usable_near_axis_1d_radius():
    radius = np.array([0.55, 0.76, 0.84, 0.96, 1.10])
    pe = np.array(
        [
            [np.nan, 10.0, 20.0, np.nan, np.nan],
            [np.nan, np.nan, 30.0, 40.0, np.nan],
        ]
    )
    assert atm_has_usable_near_axis(pe, radius, r0=0.85, radius_tol=0.08)


def test_atm_has_usable_near_axis_2d_radius():
    radius = np.array(
        [
            [0.60, 0.70, 0.83, 1.00],
            [0.60, 0.70, 0.84, 1.00],
        ]
    )
    pe = np.array(
        [
            [np.nan, np.nan, 5.0, np.nan],
            [np.nan, np.nan, np.nan, np.nan],
        ]
    )
    assert atm_has_usable_near_axis(pe, radius, r0=0.85, radius_tol=0.05)


def test_act_key_assessment_reports_ti_without_zeff():
    assessment = act_key_assessment(
        {
            "c_pla_temperature",
            "c_pla_temperature_error",
            "c_pla_velocity",
            "c_pla_velocity_error",
            "c_pla_cx_counts",
            "time",
            "majorradius",
        }
    )
    assert assessment["ti_available"] is True
    assert assessment["velocity_available"] is True
    assert assessment["cx_counts_available"] is True
    assert assessment["zeff_available"] is False
    assert assessment["zeff_key_candidates"] == []
