"""Regression tests for the measured FAIR-MAST coordinate convention."""

from __future__ import annotations

from dataclasses import replace
from math import tau

import pytest
from nova.io.cocos import transform_factor

from imas_ambix.data.cocos_convention import (
    COCOS_3_4_MEASUREMENT_DISTINGUISHABLE,
    COCOS_CANDIDATES,
    COEFFICIENT_ASSESSMENTS,
    IP_LIKE_CANDIDATE_FACTORS,
    IP_LIKE_TARGETS,
    MAST_LEVEL2_ROOT,
    MAST_LEVEL2_SIGN_TABLE,
    MAST_SOURCE_COCOS,
    MAST_TO_COCOS_17_FACTORS,
    RELATIVE_SIGN_PRODUCTS,
    SOURCE_COCOS_RECOMMENDATION,
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
    assert sum(row.raw_flux_loop_channels for row in MAST_LEVEL2_SIGN_TABLE) == 56
    assert (
        sum(row.raw_flux_loop_opposite_sign_channels for row in MAST_LEVEL2_SIGN_TABLE)
        == 56
    )
    assert all(row.raw_flux_loop_response_sign == -1 for row in MAST_LEVEL2_SIGN_TABLE)


def test_stored_sign_table_retains_both_handedness_candidates():
    scores = score_conventions(MAST_LEVEL2_SIGN_TABLE)

    assert len(COCOS_CANDIDATES) == 16
    assert len(scores) == len(COCOS_CANDIDATES)
    assert surviving_conventions(MAST_LEVEL2_SIGN_TABLE) == (3, 4)
    assert MAST_SOURCE_COCOS == 3
    assert all(
        next(score for score in scores if score.identifier == candidate).violations
        == ()
        for candidate in (3, 4)
    )
    assert all(score.violations for score in scores if score.identifier not in (3, 4))


def test_each_polarity_independently_retains_the_same_candidate_pair():
    positive = tuple(
        row for row in MAST_LEVEL2_SIGN_TABLE if row.plasma_current_sign > 0
    )
    negative = tuple(
        row for row in MAST_LEVEL2_SIGN_TABLE if row.plasma_current_sign < 0
    )

    assert surviving_conventions(positive) == (3, 4)
    assert surviving_conventions(negative) == (3, 4)


def test_each_coefficient_has_a_binary_evidence_classification_and_exact_sources():
    assessments = {item.coefficient: item for item in COEFFICIENT_ASSESSMENTS}

    assert set(assessments) == {
        "sigma_Bp",
        "e_Bp",
        "sigma_R_phi_Z",
        "sigma_rho_theta_phi",
    }
    assert {name: item.classification for name, item in assessments.items()} == {
        "sigma_Bp": "measurable-from-data",
        "e_Bp": "requires-an-external-declaration",
        "sigma_R_phi_Z": "requires-an-external-declaration",
        "sigma_rho_theta_phi": "measurable-from-data",
    }
    assert {name: item.value for name, item in assessments.items()} == {
        "sigma_Bp": -1,
        "e_Bp": 0,
        "sigma_R_phi_Z": None,
        "sigma_rho_theta_phi": -1,
    }
    assert {
        name: tuple((source.path, source.kind) for source in item.sources)
        for name, item in assessments.items()
    } == {
        "sigma_Bp": (
            ("magnetics/time", "measurement"),
            ("magnetics/ip", "measurement"),
            ("magnetics/flux_loop_flux", "measurement"),
            ("equilibrium/time", "reconstruction-output"),
            ("equilibrium/psi", "reconstruction-output"),
            ("equilibrium/major_radius", "reconstruction-output"),
            ("equilibrium/z", "reconstruction-output"),
            ("equilibrium/magnetic_axis_r", "reconstruction-output"),
            ("equilibrium/magnetic_axis_z", "reconstruction-output"),
            ("equilibrium/lcfs_r", "reconstruction-output"),
            ("equilibrium/lcfs_z", "reconstruction-output"),
        ),
        "e_Bp": (
            (
                "equilibrium/psi:units",
                "reconstruction-metadata-declaration",
            ),
        ),
        "sigma_R_phi_Z": (),
        "sigma_rho_theta_phi": (
            ("magnetics/time", "measurement"),
            ("magnetics/ip", "measurement"),
            ("equilibrium/time", "reconstruction-output"),
            ("equilibrium/bvac_rmag", "reconstruction-output"),
            ("equilibrium/q95", "reconstruction-output"),
        ),
    }


def test_measured_sigma_bp_is_untouched_by_the_external_handedness_declaration():
    sigma_bp = next(
        item for item in COEFFICIENT_ASSESSMENTS if item.coefficient == "sigma_Bp"
    )
    candidate_scores = {
        score.identifier: score for score in score_conventions(MAST_LEVEL2_SIGN_TABLE)
    }

    assert sigma_bp.classification == "measurable-from-data"
    assert sigma_bp.value == -1
    assert {candidate_scores[candidate].sigma_bp for candidate in (3, 4)} == {-1}
    assert MAST_SOURCE_COCOS == 3


def test_relative_sign_products_are_not_promoted_to_individual_handedness():
    products = {item.expression: item for item in RELATIVE_SIGN_PRODUCTS}

    assert {name: item.value for name, item in products.items()} == {
        "sigma_Bp*sigma_rho_theta_phi": 1,
        "sigma_R_phi_Z*sigma_rho_theta_phi": 1,
    }
    assert "excluded" in products["sigma_R_phi_Z*sigma_rho_theta_phi"].scope
    assert surviving_conventions() == (3, 4)


def test_candidate_pair_requires_external_handedness_declaration():
    assert COCOS_3_4_MEASUREMENT_DISTINGUISHABLE is False
    assert SOURCE_COCOS_RECOMMENDATION == "external-declaration"
    assert len(IP_LIKE_TARGETS) == 3
    assert dict(IP_LIKE_CANDIDATE_FACTORS) == {3: 1.0, 4: -1.0}
    for candidate, factor in IP_LIKE_CANDIDATE_FACTORS.items():
        assert factor == transform_factor("ip_like", source=candidate, target=17)


def test_inconsistent_sign_table_is_reported_as_no_single_convention():
    row = MAST_LEVEL2_SIGN_TABLE[-1]
    inconsistent = MAST_LEVEL2_SIGN_TABLE[:-1] + (
        replace(
            row,
            raw_flux_loop_response_wb_per_a=(-row.raw_flux_loop_response_wb_per_a),
            toroidal_field_t=row.toroidal_field_t,
            poloidal_flux_edge_minus_axis_wb_per_rad=(
                -row.poloidal_flux_edge_minus_axis_wb_per_rad
            ),
        ),
    )

    assert surviving_conventions(inconsistent) == ()
    assert "0 conventions survive" in format_sign_report(inconsistent)


def test_declared_source_to_seventeen_factors_are_committed_as_data():
    assert dict(MAST_TO_COCOS_17_FACTORS) == pytest.approx(
        {
            "psi_like": tau,
            "ip_like": 1.0,
            "b0_like": 1.0,
            "q_like": -1.0,
            "dodpsi_like": 1.0 / tau,
            "tor_angle_like": 1.0,
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


def test_report_prints_sources_candidate_pair_and_external_recommendation(capsys):
    print(format_sign_report())
    report = capsys.readouterr().out

    for row in MAST_LEVEL2_SIGN_TABLE:
        assert str(row.shot) in report
    for candidate in COCOS_CANDIDATES:
        assert f"{candidate:5d}" in report
    assert "2 conventions survive: (3, 4)" in report
    assert "no level-2 measurement distinguishes them" in report
    assert "explicit external declaration" in report
    assert "COCOS 3 is an owner assumption" in report
    assert "pending a facility statement of positive-phi direction" in report
    assert "not a measurement" in report
    assert "factor +1 to all 3 targets" in report
    assert "factor -1 to all 3 targets" in report
    assert "moves factor -1 to +1 for each affected target" in report
    assert "magnetics/ip, pf_active/coil/current, pf_active/solenoid/current" in report
    for assessment in COEFFICIENT_ASSESSMENTS:
        assert assessment.coefficient in report
        for source in assessment.sources:
            assert f"{source.path} [{source.kind}]" in report


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
    assert surviving_conventions(live) == (3, 4)
