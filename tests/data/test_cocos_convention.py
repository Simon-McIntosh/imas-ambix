"""Regression tests for the measured FAIR-MAST coordinate convention."""

from __future__ import annotations

from dataclasses import replace
from math import tau

import numpy as np
import pytest
import zarr
from nova_cocos import transform_factor

from imas_alambic.signal_map import (
    MAP_SCHEMA_VERSION,
    SignalMap,
    SignalRule,
)
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
    read_level2_sign_table,
    read_signal_map_observation,
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
            "one_like": 1.0,
        }
    )

    # Every committed entry agrees with nova's derived factor, and no entry is
    # excluded from the cross-check: a class nova does not own has no factor to
    # agree with, so the loop refuses it rather than skipping it.
    for transformation, factor in MAST_TO_COCOS_17_FACTORS.items():
        assert factor == pytest.approx(
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


def _signal_rule(
    semantic_id,
    source_group,
    source_array,
    target_path,
    *,
    target_index=None,
    channel_factor=1.0,
):
    return SignalRule(
        semantic_id=semantic_id,
        source_group=source_group,
        source_array=source_array,
        source_unit="Wb",
        target_path=target_path,
        target_unit="Wb",
        target_index=target_index,
        transformation="one_like",
        source_cocos=None,
        unit_factor=1.0,
        channel_factor=channel_factor,
        standard_name=None,
        evidence="synthetic signal map for the observation reader test",
        validation_state="source-only",
    )


def _signal_map(system, signals):
    return SignalMap.create(
        schema_version=MAP_SCHEMA_VERSION,
        set_version="0.1.0",
        machine="jt-60sa",
        system=system,
        source_dataset="synthetic",
        target_dd_version="4.1.1",
        target_cocos=17,
        discovery_producer="synthetic",
        discovery_receipt="synthetic",
        signals=signals,
    )


def _synthetic_magnetics_map(*, plasma_current_factor=-1.0):
    """A magnetics map whose plasma-current rule carries channel_factor -1.

    The three flux-loop rules are declared with semantic ids out of target
    order, so the reader's target-index ordering — not the declaration order —
    is what gathers the channels.
    """

    return _signal_map(
        "magnetics",
        (
            _signal_rule(
                "synthetic-ip",
                "PSRC",
                "Ip",
                "magnetics/ip/data",
                channel_factor=plasma_current_factor,
            ),
            _signal_rule(
                "synthetic-flux-c",
                "MDAC",
                "magFlxLp3",
                "magnetics/flux_loop/flux/data",
                target_index=2,
            ),
            _signal_rule(
                "synthetic-flux-a",
                "MDAC",
                "magFlxLp1",
                "magnetics/flux_loop/flux/data",
                target_index=0,
            ),
            _signal_rule(
                "synthetic-flux-b",
                "MDAC",
                "magFlxLp2",
                "magnetics/flux_loop/flux/data",
                target_index=1,
            ),
        ),
    )


def _synthetic_tf_map():
    return _signal_map(
        "tf",
        (
            _signal_rule(
                "synthetic-tf-current",
                "MMSYS",
                "cur1TFLKAT",
                "tf/coil/current/data",
            ),
        ),
    )


def _synthetic_maps(*, plasma_current_factor=-1.0):
    return {
        "magnetics": _synthetic_magnetics_map(
            plasma_current_factor=plasma_current_factor
        ),
        "tf": _synthetic_tf_map(),
    }


def _write_synthetic_store(root):
    group = zarr.open_group(root / f"{_SYNTHETIC_SHOT}.zarr", mode="w")
    plasma_current = group.require_group("PSRC")
    plasma_current.create_array("Ip", data=_SYNTHETIC_CURRENT)
    plasma_current.create_array("Ip_time", data=_SYNTHETIC_TIME)
    flux_loops = group.require_group("MDAC")
    for name, factor in (
        ("magFlxLp1", 2.0e-3),
        ("magFlxLp2", 3.0e-3),
        ("magFlxLp3", 4.0e-3),
    ):
        flux_loops.create_array(name, data=factor * _SYNTHETIC_CURRENT)
        flux_loops.create_array(f"{name}_time", data=_SYNTHETIC_TIME)
    tf = group.require_group("MMSYS")
    tf.create_array("cur1TFLKAT", data=np.full(_SYNTHETIC_TIME.size, 3.0))
    tf.create_array("cur1TFLKAT_time", data=np.linspace(0.0, 7.0, _SYNTHETIC_TIME.size))


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
            "magnetics_bcoil": np.full(frames, 2.5),
        },
        cocos=17,
    )


def _stub_signal_maps(monkeypatch, maps):
    monkeypatch.setattr(
        "imas_ambix.data.cocos_convention.load_packaged_signal_map",
        lambda machine, system: maps[system],
    )


def _single_loop_maps():
    return {
        "magnetics": _signal_map(
            "magnetics",
            (
                _signal_rule("synthetic-ip", "PSRC", "Ip", "magnetics/ip/data"),
                _signal_rule(
                    "synthetic-flux-a",
                    "MDAC",
                    "magFlxLp1",
                    "magnetics/flux_loop/flux/data",
                    target_index=0,
                ),
            ),
        ),
        "tf": _synthetic_tf_map(),
    }


def _write_loop_store(root, loop_time, loop_values):
    root.mkdir(parents=True, exist_ok=True)
    group = zarr.open_group(root / f"{_SYNTHETIC_SHOT}.zarr", mode="w")
    plasma_current = group.require_group("PSRC")
    plasma_current.create_array("Ip", data=_SYNTHETIC_CURRENT)
    plasma_current.create_array("Ip_time", data=_SYNTHETIC_TIME)
    flux_loops = group.require_group("MDAC")
    flux_loops.create_array("magFlxLp1", data=np.asarray(loop_values, dtype="<f8"))
    flux_loops.create_array("magFlxLp1_time", data=np.asarray(loop_time, dtype="<f8"))
    tf = group.require_group("MMSYS")
    tf.create_array("cur1TFLKAT", data=np.full(loop_time.size, 3.0))
    tf.create_array("cur1TFLKAT_time", data=np.linspace(0.0, 7.0, loop_time.size))


def test_flux_loop_on_a_shifted_time_base_is_resampled_onto_the_current(
    tmp_path, monkeypatch
):
    """A loop and the current are separate channels on their own time vectors.

    The loop here is sampled on a grid shifted from the current's but covering
    it; the reader must interpolate it, so its observation equals the one from
    a store whose loop is already sampled at the current's own times.
    """

    _stub_signal_maps(monkeypatch, _single_loop_maps())
    loop_time = np.linspace(-0.5, 7.0, _SYNTHETIC_TIME.size)
    loop_values = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0])
    shifted_root = tmp_path / "shifted"
    aligned_root = tmp_path / "aligned"
    _write_loop_store(shifted_root, loop_time, loop_values)
    _write_loop_store(
        aligned_root,
        _SYNTHETIC_TIME,
        np.interp(_SYNTHETIC_TIME, loop_time, loop_values),
    )

    shifted = read_signal_map_observation(
        _SYNTHETIC_SHOT, "jt-60sa", _synthetic_equilibrium(), root=shifted_root
    )
    aligned = read_signal_map_observation(
        _SYNTHETIC_SHOT, "jt-60sa", _synthetic_equilibrium(), root=aligned_root
    )

    assert shifted.raw_flux_loop_response_wb_per_a == pytest.approx(
        aligned.raw_flux_loop_response_wb_per_a
    )


def test_flux_loop_not_covering_the_current_time_base_is_refused(
    tmp_path, monkeypatch
):
    _stub_signal_maps(monkeypatch, _single_loop_maps())
    loop_time = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.5])
    _write_loop_store(tmp_path, loop_time, np.ones(loop_time.size))

    with pytest.raises(ValueError, match="does not cover"):
        read_signal_map_observation(
            _SYNTHETIC_SHOT, "jt-60sa", _synthetic_equilibrium(), root=tmp_path
        )


def test_signal_map_reader_measures_the_raw_cached_channels(tmp_path, monkeypatch):
    _write_synthetic_store(tmp_path)
    _stub_signal_maps(monkeypatch, _synthetic_maps())

    observation = read_signal_map_observation(
        _SYNTHETIC_SHOT, "jt-60sa", _synthetic_equilibrium(), root=tmp_path
    )

    assert observation.shot == _SYNTHETIC_SHOT
    assert observation.plasma_current_a == pytest.approx(8.0e5)
    assert observation.plasma_current_sign == 1
    assert observation.raw_flux_loop_channels == 3
    assert observation.raw_flux_loop_opposite_sign_channels == 0
    assert observation.raw_flux_loop_response_sign == 1
    assert observation.toroidal_field_t == pytest.approx(2.5)
    assert observation.toroidal_field_sign == 1
    assert observation.safety_factor_sign == 1
    assert observation.retained_slices == 4
    assert observation.tf_coil_current_sign == 1


def test_signal_map_reader_ignores_a_rules_channel_factor(tmp_path, monkeypatch):
    _write_synthetic_store(tmp_path)
    _stub_signal_maps(monkeypatch, _synthetic_maps(plasma_current_factor=-1.0))

    observation = read_signal_map_observation(
        _SYNTHETIC_SHOT, "jt-60sa", _synthetic_equilibrium(), root=tmp_path
    )

    # The plasma-current rule declares channel_factor -1, yet the reader must
    # surface the raw cached value: the factor selects the assumed sign this
    # reader exists to measure, so applying it here would erase the measurement.
    assert observation.plasma_current_a == pytest.approx(8.0e5)
    assert observation.plasma_current_sign == 1


def test_signal_map_reader_stacks_every_flux_loop_rule_in_target_order(
    tmp_path, monkeypatch
):
    _write_synthetic_store(tmp_path)
    _stub_signal_maps(monkeypatch, _synthetic_maps())

    observation = read_signal_map_observation(
        _SYNTHETIC_SHOT, "jt-60sa", _synthetic_equilibrium(), root=tmp_path
    )

    assert observation.raw_flux_loop_channels == 3
    assert observation.raw_flux_loop_response_sign == 1
    assert observation.raw_flux_loop_opposite_sign_channels == 0
