"""Regression tests for the measured FAIR-MAST coordinate convention."""

from __future__ import annotations

from math import tau

import pytest
from nova.io.cocos import transform_factor

from imas_ambix.data.cocos_convention import (
    COCOS_CANDIDATES,
    MAST_LEVEL2_ROOT,
    MAST_LEVEL2_SIGN_TABLE,
    MAST_SOURCE_COCOS,
    MAST_TO_COCOS_17_FACTORS,
    ShotSignObservation,
    format_sign_report,
    read_level2_sign_table,
    score_conventions,
    surviving_conventions,
)


def test_stored_sign_table_covers_two_shots_at_each_current_polarity():
    signs = [row.plasma_current_sign for row in MAST_LEVEL2_SIGN_TABLE]

    assert len(MAST_LEVEL2_SIGN_TABLE) == 4
    assert signs.count(-1) == 2
    assert signs.count(+1) == 2
    assert sum(row.retained_slices for row in MAST_LEVEL2_SIGN_TABLE) == 251


def test_stored_sign_table_reproduces_unique_convention_determination():
    scores = score_conventions(MAST_LEVEL2_SIGN_TABLE)

    assert len(COCOS_CANDIDATES) == 16
    assert len(scores) == len(COCOS_CANDIDATES)
    assert surviving_conventions(MAST_LEVEL2_SIGN_TABLE) == (4,)
    assert MAST_SOURCE_COCOS == 4
    assert next(score for score in scores if score.identifier == 4).violations == ()
    assert all(
        score.violations for score in scores if score.identifier != MAST_SOURCE_COCOS
    )


def test_each_polarity_independently_selects_the_same_convention():
    positive = tuple(
        row for row in MAST_LEVEL2_SIGN_TABLE if row.plasma_current_sign > 0
    )
    negative = tuple(
        row for row in MAST_LEVEL2_SIGN_TABLE if row.plasma_current_sign < 0
    )

    assert surviving_conventions(positive) == (4,)
    assert surviving_conventions(negative) == (4,)


def test_inconsistent_sign_table_is_reported_as_no_single_convention():
    row = MAST_LEVEL2_SIGN_TABLE[-1]
    inconsistent = MAST_LEVEL2_SIGN_TABLE[:-1] + (
        ShotSignObservation(
            shot=row.shot,
            plasma_current_a=row.plasma_current_a,
            toroidal_field_t=row.toroidal_field_t,
            poloidal_flux_edge_minus_axis_wb_per_rad=(
                -row.poloidal_flux_edge_minus_axis_wb_per_rad
            ),
            poloidal_angle_signed_area_m2=row.poloidal_angle_signed_area_m2,
            safety_factor=row.safety_factor,
            flux_exponent=row.flux_exponent,
            retained_slices=row.retained_slices,
        ),
    )

    assert surviving_conventions(inconsistent) == ()
    assert "0 conventions survive" in format_sign_report(inconsistent)


def test_cocos_four_to_seventeen_factors_are_committed_as_data():
    assert dict(MAST_TO_COCOS_17_FACTORS) == pytest.approx(
        {
            "psi_like": -tau,
            "ip_like": -1.0,
            "b0_like": -1.0,
            "q_like": 1.0,
            "dodpsi_like": -1.0 / tau,
            "tor_angle_like": -1.0,
            "pol_angle_like": -1.0,
            "one_like": 1.0,
        }
    )

    for transformation in (
        "psi_like",
        "ip_like",
        "b0_like",
        "q_like",
        "dodpsi_like",
        "one_like",
    ):
        assert MAST_TO_COCOS_17_FACTORS[transformation] == pytest.approx(
            transform_factor(
                transformation,
                source=MAST_SOURCE_COCOS,
                target=17,
            )
        )


def test_report_prints_every_shot_candidate_and_explicit_verdict(capsys):
    print(format_sign_report())
    report = capsys.readouterr().out

    for row in MAST_LEVEL2_SIGN_TABLE:
        assert str(row.shot) in report
    for candidate in COCOS_CANDIDATES:
        assert f"{candidate:5d}" in report
    assert "exactly 1 convention survives: COCOS-4" in report


@pytest.mark.skipif(
    not all(
        (MAST_LEVEL2_ROOT / f"{row.shot}.zarr").is_dir()
        for row in MAST_LEVEL2_SIGN_TABLE
    ),
    reason="FAIR-MAST level-2 convention cohort is not mounted",
)
def test_live_level_two_cohort_reproduces_committed_signs():
    live = read_level2_sign_table()

    assert live == MAST_LEVEL2_SIGN_TABLE
    assert surviving_conventions(live) == (MAST_SOURCE_COCOS,)
