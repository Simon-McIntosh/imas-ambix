"""Regression tests for the measured FAIR-MAST coordinate convention."""

from __future__ import annotations

from dataclasses import replace
from math import tau
from types import MappingProxyType

import numpy as np
import pytest
import zarr
from nova.io.cocos import transform_factor

from imas_ambix.challenge.loader import EfitLabels
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
    read_catalogue_observation,
    read_level2_sign_table,
    score_conventions,
    surviving_conventions,
)
from imas_ambix.data.machine_map import (
    ChannelBinding,
    MachineMap,
    MachineMapCatalog,
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


_SYNTHETIC_SHOT = 12345
_SYNTHETIC_TIME = np.arange(8.0)
_SYNTHETIC_CURRENT = np.array([0.0, 0.0, 8.0e5, 8.0e5, 8.0e5, 8.0e5, 0.0, 0.0])
_DECOY_CURRENT = np.array([0.0, 0.0, -3.0e5, -3.0e5, -3.0e5, -3.0e5, 0.0, 0.0])


def _binding(name, source_array, dd_path):
    return ChannelBinding(
        name=name,
        source_group="PSRC",
        source_array=source_array,
        source_rank=2,
        source_role="value",
        source_location="ssh://jt-60sa/EDDB/PSRC",
        dd_path=dd_path,
        source_unit="A",
        target_unit="A",
        sign_convention="identity",
        evidence="synthetic store for the catalogue observation reader test",
        source_cocos_override=None,
    )


def _synthetic_catalogue(plasma_current_array="Ip"):
    binding_set = (
        _binding("synthetic-ip", plasma_current_array, "magnetics/ip"),
        _binding("synthetic-flux", "FL", "magnetics/flux_loop_flux"),
        _binding("synthetic-bt", "BT", "magnetics/b_field_tor_probe/field"),
    )
    machine_map = MachineMap(
        name="synthetic",
        machine="jt-60sa",
        first_shot=0,
        last_shot=200_000,
        transition=None,
        binding_set="synthetic",
        drive_topology=None,
        description_supplement=None,
        validation_state="source-only",
        source_representation_signature=None,
    )
    return MachineMapCatalog(
        schema_version="1.0.0",
        dd_version="4.1.1",
        source="synthetic",
        source_revision="synthetic",
        source_cocos=None,
        description_store_format="zarr",
        description_store_root="JT60SA_ROOT",
        description_store_layout="per-shot",
        probe_angle_source="description",
        binding_sets=MappingProxyType({"synthetic": binding_set}),
        maps=(machine_map,),
        validation_gaps=(),
        source_qualifications=(),
        sensor_identity_rules=(),
        identity_qualifications=(),
        flux_loop_position_declarations=(),
        drive_topologies=(),
        structure_assemblies=(),
        acquisition_declarations=(),
        description_supplements=(),
    )


def _write_synthetic_store(root):
    group = zarr.open_group(root / f"{_SYNTHETIC_SHOT}.zarr", mode="w")
    category = group.require_group("PSRC")
    category.create_array("Ip", data=_SYNTHETIC_CURRENT)
    category.create_array("Ip_time", data=_SYNTHETIC_TIME)
    category.create_array("IpDecoy", data=_DECOY_CURRENT)
    category.create_array("IpDecoy_time", data=_SYNTHETIC_TIME)
    flux = np.vstack((2.0e-3 * _SYNTHETIC_CURRENT, 3.0e-3 * _SYNTHETIC_CURRENT))
    category.create_array("FL", data=flux)
    category.create_array("FL_time", data=_SYNTHETIC_TIME)
    category.create_array("BT", data=np.full(_SYNTHETIC_TIME.size, 2.5))
    category.create_array("BT_time", data=np.linspace(0.0, 7.0, _SYNTHETIC_TIME.size))


def _synthetic_equilibrium():
    radius = np.linspace(2.0, 4.0, 9)
    height = np.linspace(-1.5, 1.5, 9)
    frames = 5
    psirz = np.empty((frames, height.size, radius.size))
    for frame in range(frames):
        psirz[frame] = (
            (radius[np.newaxis, :] - 3.0) ** 2
            + height[:, np.newaxis] ** 2
            + frame
        )
    angle = np.linspace(0.0, tau, 16, endpoint=False)
    lcfs_r = np.tile(3.0 + 0.6 * np.cos(angle), (frames, 1))
    lcfs_z = np.tile(0.6 * np.sin(angle), (frames, 1))
    return EfitLabels(
        time_ms=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        psirz=psirz,
        grid_r_m=radius,
        grid_z_m=height,
        lcfs_r_m=lcfs_r,
        lcfs_z_m=lcfs_z,
        scalars={
            "efit_q95": np.full(frames, 2.0),
            "efit_r_axis": np.full(frames, 3.0),
            "efit_z_axis": np.zeros(frames),
        },
        cocos=17,
    )


def test_catalogue_reader_measures_the_bound_raw_arrays(tmp_path, monkeypatch):
    _write_synthetic_store(tmp_path)
    monkeypatch.setattr(
        MachineMapCatalog,
        "description_store_root_path",
        lambda self, _root=tmp_path: _root,
    )

    observation = read_catalogue_observation(
        _SYNTHETIC_SHOT, _synthetic_catalogue(), _synthetic_equilibrium()
    )

    assert observation.shot == _SYNTHETIC_SHOT
    assert observation.plasma_current_a == pytest.approx(8.0e5)
    assert observation.plasma_current_sign == 1
    assert observation.raw_flux_loop_channels == 2
    assert observation.raw_flux_loop_response_sign == 1
    assert observation.toroidal_field_t == pytest.approx(2.5)
    assert observation.toroidal_field_sign == 1
    assert observation.safety_factor_sign == 1
    assert observation.retained_slices == 4


def test_catalogue_reader_follows_the_plasma_current_binding(tmp_path, monkeypatch):
    _write_synthetic_store(tmp_path)
    monkeypatch.setattr(
        MachineMapCatalog,
        "description_store_root_path",
        lambda self, _root=tmp_path: _root,
    )

    observation = read_catalogue_observation(
        _SYNTHETIC_SHOT,
        _synthetic_catalogue(plasma_current_array="IpDecoy"),
        _synthetic_equilibrium(),
    )

    assert observation.plasma_current_a == pytest.approx(-3.0e5)
    assert observation.plasma_current_sign == -1
