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
from imas_ambix.challenge.loader import EfitLabels, load_geqdsk
from imas_ambix.data.cocos_convention import (
    _RAW_FLUX_LOOP_TARGETS,
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
    _absolute_flux_loop_rules,
    _differential_flux_loop_targets,
    format_sign_report,
    read_level2_sign_table,
    read_signal_map_observation,
    score_convention,
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
    validation_state="corpus-validated",
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
        validation_state=validation_state,
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


def _synthetic_equilibrium_map():
    """A gridless equilibrium map: the signed q and the field scalar only.

    Both rules stay ``source-only`` because the reader reads their raw values
    and owns the sign itself, exactly as it does for the magnetics half.
    """

    return _signal_map(
        "equilibrium",
        (
            _signal_rule(
                "synthetic-q95",
                "FAME",
                "Q95",
                "equilibrium/time_slice/global_quantities/q_95",
                validation_state="source-only",
            ),
            _signal_rule(
                "synthetic-toroidal-field",
                "FAME",
                "BTV",
                "equilibrium/vacuum_toroidal_field/b0",
                validation_state="source-only",
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


def _write_synthetic_store(root, *, equilibrium_field_sign=1.0):
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
    fame = group.require_group("FAME")
    fame.create_array("Q95", data=np.full(_SYNTHETIC_TIME.size, 8.0))
    fame.create_array("Q95_time", data=_SYNTHETIC_TIME)
    fame.create_array(
        "BTV", data=np.full(_SYNTHETIC_TIME.size, 6.0 * equilibrium_field_sign)
    )
    fame.create_array("BTV_time", data=_SYNTHETIC_TIME)


def _synthetic_equilibrium():
    radius = np.linspace(2.0, 4.0, 9)
    height = np.linspace(-1.5, 1.5, 9)
    frames = 5
    psirz = np.empty((frames, height.size, radius.size))
    for frame in range(frames):
        psirz[frame] = (
            (radius[np.newaxis, :] - 3.0) ** 2 + height[:, np.newaxis] ** 2 + frame
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


def _write_synthetic_geqdsk(path, *, psi_sign=1.0):
    """Write one small G-EQDSK under a temporary path.

    The flux is quadratic in the coordinates with its minimum on the axis, so
    the edge-minus-axis flux carries the same sign as ``psi_sign`` and the
    written file reproduces the shape and sign a real G-EQDSK carries.
    """

    from eqdsk import EQDSKInterface

    size = 9
    radius = np.linspace(2.0, 4.0, size)
    height = np.linspace(-1.5, 1.5, size)
    flux = psi_sign * np.transpose(
        (radius[np.newaxis, :] - 3.0) ** 2 + height[:, np.newaxis] ** 2
    )
    angle = np.linspace(0.0, tau, 16, endpoint=False)
    instance = EQDSKInterface(
        bcentre=2.5,
        cplasma=8.0e5,
        dxc=np.zeros(0),
        dzc=np.zeros(0),
        ffprime=np.zeros(size),
        fpol=np.full(size, 6.0),
        Ic=np.zeros(0),
        name="synthetic",
        nbdry=angle.size,
        ncoil=0,
        nlim=0,
        nx=size,
        nz=size,
        pprime=np.zeros(size),
        pressure=np.zeros(size),
        psi=flux,
        psibdry=float(flux[0, 0]),
        psimag=float(flux[size // 2, size // 2]),
        xbdry=3.0 + 0.6 * np.cos(angle),
        xc=np.zeros(0),
        xcentre=3.0,
        xdim=2.0,
        xgrid1=2.0,
        xlim=np.zeros(0),
        xmag=3.0,
        zbdry=0.6 * np.sin(angle),
        zc=np.zeros(0),
        zdim=3.0,
        zlim=np.zeros(0),
        zmag=0.0,
        zmid=0.0,
        qpsi=np.linspace(2.0, 5.0, size),
    )
    instance.write(path, file_format="geqdsk")
    return path


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


def test_flux_loop_not_covering_the_current_time_base_is_refused(tmp_path, monkeypatch):
    _stub_signal_maps(monkeypatch, _single_loop_maps())
    loop_time = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.0, 7.4])
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


def test_row_without_a_flux_half_scores_with_the_flux_relation_unscored():
    """A source with no psi grid has no edge-minus-axis flux to score.

    The relation is left out of the violation set for that row rather than
    failed, and the report names it beside its reason so an unscored relation
    is never mistaken for a satisfied one.
    """

    row = replace(
        MAST_LEVEL2_SIGN_TABLE[0],
        poloidal_flux_edge_minus_axis_wb_per_rad=None,
        poloidal_angle_signed_area_m2=None,
    )

    assert row.poloidal_flux_sign is None
    assert row.poloidal_angle_direction is None
    for candidate in COCOS_CANDIDATES:
        violations = score_convention(candidate, (row,)).violations
        assert f"{row.shot}:reconstructed_poloidal_flux" not in violations

    report = format_sign_report((row,))
    assert "UNSCORED RELATIONS" in report
    assert f"{row.shot}: reconstructed_poloidal_flux not scored — " in report


def test_geqdsk_flux_row_scores_the_poloidal_flux_relation(tmp_path, monkeypatch):
    """A G-EQDSK supplies the cohort's one row carrying a flux half.

    The record carries psi on its grid, so the reconstructed-poloidal-flux
    relation is scored rather than left unscored, and the row keeps the
    file's own psi sign.  Negating the file's psi flips
    ``poloidal_flux_sign`` and the relation stops matching the raw
    flux-loop response, so a candidate with ``sigma_bp = -1`` is the one
    that reports the violation here.
    """

    _stub_signal_maps(monkeypatch, _synthetic_maps())
    _write_synthetic_store(tmp_path)
    record = load_geqdsk(
        _write_synthetic_geqdsk(tmp_path / "e101011.geqdsk"), time_ms=3.0
    )

    observation = read_signal_map_observation(
        _SYNTHETIC_SHOT, "jt-60sa", record, root=tmp_path
    )

    assert observation.poloidal_flux_edge_minus_axis_wb_per_rad is not None
    assert observation.poloidal_flux_sign == 1
    assert observation.plasma_current_sign == 1
    assert observation.raw_flux_loop_response_sign == 1
    assert observation.flux_exponent == 0
    violated = score_convention(4, (observation,))
    assert f"{_SYNTHETIC_SHOT}:reconstructed_poloidal_flux" in violated.violations


def test_signal_map_reader_supplies_the_equilibrium_half_from_the_map(
    tmp_path, monkeypatch
):
    """With no equilibrium record the reader takes q and F from the map.

    The q scalar and the field scalar come from the machine's equilibrium
    signal map, so a store carrying no psi grid still yields the q relation's
    two signs while every flux-dependent relation stays unscored.
    """

    _write_synthetic_store(tmp_path)
    maps = _synthetic_maps()
    maps["equilibrium"] = _synthetic_equilibrium_map()
    _stub_signal_maps(monkeypatch, maps)

    observation = read_signal_map_observation(_SYNTHETIC_SHOT, "jt-60sa", root=tmp_path)

    assert observation.safety_factor == pytest.approx(8.0)
    assert observation.safety_factor_sign == 1
    assert observation.toroidal_field_t == pytest.approx(6.0)
    assert observation.toroidal_field_sign == 1
    assert observation.poloidal_flux_sign is None
    assert observation.poloidal_flux_edge_minus_axis_wb_per_rad is None


def test_negating_the_equilibrium_field_makes_the_q_relation_report_a_violation(
    tmp_path, monkeypatch
):
    """The sign the reader reports for F is the store's own, not an assumption.

    COCOS 17 carries ``sigma_rho_theta_phi = +1``, so the relation
    ``sign(q) = sign(Ip) * sign(B0) * sigma_rho_theta_phi`` holds while the
    store's F is positive and fails once the same store records it negative.
    If the reader assumed a sign instead of reading it, the negative store
    would score identically to the positive one.
    """

    maps = _synthetic_maps()
    maps["equilibrium"] = _synthetic_equilibrium_map()
    _stub_signal_maps(monkeypatch, maps)
    positive_root = tmp_path / "positive"
    negative_root = tmp_path / "negative"
    _write_synthetic_store(positive_root, equilibrium_field_sign=1.0)
    _write_synthetic_store(negative_root, equilibrium_field_sign=-1.0)

    positive = read_signal_map_observation(
        _SYNTHETIC_SHOT, "jt-60sa", root=positive_root
    )
    negative = read_signal_map_observation(
        _SYNTHETIC_SHOT, "jt-60sa", root=negative_root
    )

    assert positive.toroidal_field_sign == 1
    assert negative.toroidal_field_sign == -1
    assert not any(
        violation.endswith(":reconstructed_safety_factor")
        for violation in score_convention(17, (positive,)).violations
    )
    assert any(
        violation.endswith(":reconstructed_safety_factor")
        for violation in score_convention(17, (negative,)).violations
    )


def test_signal_map_reader_reads_only_the_proven_absolute_flux_loop(
    tmp_path, monkeypatch
):
    """A ``source-only`` loop carries an assumed sign and must not be scored.

    The reference loop is proven; the differential loop beside it is not, so
    only the reference enters the raw response and the channel count is one.
    """

    _write_synthetic_store(tmp_path)
    maps = _synthetic_maps()
    maps["magnetics"] = _signal_map(
        "magnetics",
        (
            _signal_rule("synthetic-ip", "PSRC", "Ip", "magnetics/ip/data"),
            _signal_rule(
                "synthetic-flux-reference",
                "MDAC",
                "magFlxLp1",
                "magnetics/flux_loop/flux/data",
                target_index=6,
                channel_factor=-1.0,
            ),
            _signal_rule(
                "synthetic-flux-differential",
                "MDAC",
                "magFlxLp2",
                "magnetics/flux_loop/flux/data",
                target_index=3,
                validation_state="source-only",
            ),
        ),
    )
    _stub_signal_maps(monkeypatch, maps)

    observation = read_signal_map_observation(
        _SYNTHETIC_SHOT, "jt-60sa", _synthetic_equilibrium(), root=tmp_path
    )

    assert observation.raw_flux_loop_channels == 1


def test_signal_map_reader_refuses_a_map_with_no_proven_flux_loop(
    tmp_path, monkeypatch
):
    _write_synthetic_store(tmp_path)
    maps = _synthetic_maps()
    maps["magnetics"] = _signal_map(
        "magnetics",
        (
            _signal_rule("synthetic-ip", "PSRC", "Ip", "magnetics/ip/data"),
            _signal_rule(
                "synthetic-flux-differential",
                "MDAC",
                "magFlxLp2",
                "magnetics/flux_loop/flux/data",
                target_index=3,
                validation_state="source-only",
            ),
        ),
    )
    _stub_signal_maps(monkeypatch, maps)

    with pytest.raises(ValueError, match="no proven rule"):
        read_signal_map_observation(
            _SYNTHETIC_SHOT, "jt-60sa", _synthetic_equilibrium(), root=tmp_path
        )


def test_absolute_loop_selector_drops_a_proven_differential_entry():
    """A proven loop whose target is a differential pair is not an absolute flux.

    The selector keeps a rule only when its convention is proven *and* its
    target entry carries no ``indices_differential``.  A type-6 entry stores the
    difference between two loops, so its channel cannot carry a sign product
    however well the pair's convention is known.
    """

    rules = (
        _signal_rule(
            "synthetic-flux-reference",
            "MDAC",
            "magFlxLp1",
            "magnetics/flux_loop/flux/data",
            target_index=6,
        ),
        _signal_rule(
            "synthetic-flux-differential",
            "MDAC",
            "magFlxLp2",
            "magnetics/flux_loop/flux/data",
            target_index=37,
        ),
    )

    kept = _absolute_flux_loop_rules(
        rules,
        _RAW_FLUX_LOOP_TARGETS,
        "the flux-loop flux",
        differential_targets=frozenset({37}),
    )

    assert [rule.semantic_id for rule in kept] == ["synthetic-flux-reference"]


def test_jt60sa_differential_loop_targets_are_the_type6_entries():
    """The resolver reads the pair entries, it does not assume the store order.

    The 53-loop store holds 27 absolute entries then 26 differential pairs, so
    the differential target indices run from 27 to 52 inclusive.
    """

    assert _differential_flux_loop_targets("jt-60sa") == frozenset(range(27, 53))


def _catalogue_with_maps(monkeypatch, machine_maps):
    """Stub the packaged catalogue to carry the given machine maps."""

    from imas_alambic.machine_map import load_packaged_machine_map

    catalog = load_packaged_machine_map("jt-60sa")
    monkeypatch.setattr(
        "imas_ambix.data.cocos_convention.load_packaged_machine_map",
        lambda machine: replace(catalog, maps=tuple(machine_maps)),
    )
    return catalog


def test_differential_targets_resolve_when_the_first_map_is_not_magnetics(monkeypatch):
    """The store is found by the declared magnetics system, not the first map.

    The first declared map is renamed so its directory carries no magnetics
    store; a resolver keyed on ``catalog.maps[0]`` would then find no such file
    and admit every proven rule, but the magnetics IDS the catalogue declares is
    still carried by the maps that remain, so the differential entries resolve.
    """

    from imas_alambic.machine_map import load_packaged_machine_map

    catalog = load_packaged_machine_map("jt-60sa")
    shadowed = replace(catalog.maps[0], name="no-magnetics-here")
    _catalogue_with_maps(monkeypatch, (shadowed, *catalog.maps[1:]))

    assert _differential_flux_loop_targets("jt-60sa") == frozenset(range(27, 53))


def test_differential_targets_refuse_a_declared_store_with_no_magnetics_file(
    monkeypatch,
):
    """A declared description store whose magnetics file is absent is refused.

    A catalogue that declares a description store but whose declared maps carry
    no ``magnetics.nc`` cannot resolve its differential entries; resolving that
    to the empty set would admit every proven flux-loop rule as an absolute
    loop, so the store is refused instead.
    """

    from imas_alambic.machine_map import load_packaged_machine_map

    catalog = load_packaged_machine_map("jt-60sa")
    absent = tuple(replace(item, name=f"absent-{item.name}") for item in catalog.maps)
    _catalogue_with_maps(monkeypatch, absent)

    with pytest.raises(ValueError, match="magnetics.nc"):
        _differential_flux_loop_targets("jt-60sa")


def test_signal_map_reader_drops_a_proven_differential_loop(tmp_path, monkeypatch):
    """A ``corpus-validated`` differential rule is still not an absolute flux.

    Loop 7 is proven and targets its absolute type-1 entry; a second proven
    rule targets a type-6 entry, which stores a difference to the reference.
    The reader keeps the absolute loop alone, so the channel count is one.
    """

    _write_synthetic_store(tmp_path)
    maps = _synthetic_maps()
    maps["magnetics"] = _signal_map(
        "magnetics",
        (
            _signal_rule("synthetic-ip", "PSRC", "Ip", "magnetics/ip/data"),
            _signal_rule(
                "synthetic-flux-reference",
                "MDAC",
                "magFlxLp1",
                "magnetics/flux_loop/flux/data",
                target_index=6,
                channel_factor=-1.0,
            ),
            _signal_rule(
                "synthetic-flux-differential",
                "MDAC",
                "magFlxLp2",
                "magnetics/flux_loop/flux/data",
                target_index=37,
                channel_factor=-1.0,
            ),
        ),
    )
    _stub_signal_maps(monkeypatch, maps)

    observation = read_signal_map_observation(
        _SYNTHETIC_SHOT, "jt-60sa", _synthetic_equilibrium(), root=tmp_path
    )

    assert observation.raw_flux_loop_channels == 1


def test_single_absolute_loop_row_is_not_a_consensus_violation():
    """An agreeing one-loop row fixes the sign, it is not a split consensus.

    A store that carries a single absolute flux loop has one response and no
    disagreement to report, so the consensus test stays silent for it rather
    than flagging the every-channel-negative form a multi-loop store needs.
    """

    row = replace(
        MAST_LEVEL2_SIGN_TABLE[0],
        raw_flux_loop_channels=1,
        raw_flux_loop_opposite_sign_channels=0,
    )

    for candidate in COCOS_CANDIDATES:
        assert not any(
            violation.endswith(":raw_flux_loop_channel_consensus")
            for violation in score_convention(candidate, (row,)).violations
        )


def test_split_flux_loop_consensus_is_a_violation():
    """A store whose loop responses are split violates every candidate."""

    row = replace(
        MAST_LEVEL2_SIGN_TABLE[0],
        raw_flux_loop_channels=2,
        raw_flux_loop_opposite_sign_channels=1,
    )

    assert any(
        violation.endswith(":raw_flux_loop_channel_consensus")
        for violation in score_convention(3, (row,)).violations
    )


def test_flux_loop_gap_of_one_sampling_interval_is_accepted(tmp_path, monkeypatch):
    """A loop opening one sample after the current still covers it.

    The loop's own sampling interval is its resolution, so a record that opens
    one interval after the plasma current's start carries a measured value
    there; the reader holds the nearest value across that gap rather than
    refusing the channel.
    """

    _stub_signal_maps(monkeypatch, _single_loop_maps())
    loop_time = np.arange(1.0, 9.0, 1.0)
    _write_loop_store(tmp_path, loop_time, np.power(2.0, np.arange(loop_time.size)))

    observation = read_signal_map_observation(
        _SYNTHETIC_SHOT, "jt-60sa", _synthetic_equilibrium(), root=tmp_path
    )

    assert observation.raw_flux_loop_channels == 1


def test_flux_loop_gap_wider_than_one_sampling_interval_is_refused(
    tmp_path, monkeypatch
):
    """A gap wider than one interval is beyond the channel's resolution."""

    _stub_signal_maps(monkeypatch, _single_loop_maps())
    loop_time = np.arange(2.0, 10.0, 1.0)
    _write_loop_store(tmp_path, loop_time, np.power(2.0, np.arange(loop_time.size)))

    with pytest.raises(ValueError, match="does not cover"):
        read_signal_map_observation(
            _SYNTHETIC_SHOT, "jt-60sa", _synthetic_equilibrium(), root=tmp_path
        )


def test_row_without_psi_leaves_the_flux_exponent_unscored():
    """No psi grid means no declared flux exponent to score either.

    A row whose equilibrium carries no psi grid records ``None`` for the flux
    exponent rather than MAST's declared zero, so the declared-flux-exponent
    check is left unscored and the report names the relation and the source
    that would fix it.
    """

    row = replace(
        MAST_LEVEL2_SIGN_TABLE[0],
        poloidal_flux_edge_minus_axis_wb_per_rad=None,
        poloidal_angle_signed_area_m2=None,
        flux_exponent=None,
    )

    assert row.flux_exponent is None
    for candidate in COCOS_CANDIDATES:
        assert not any(
            violation.endswith(":declared_flux_exponent")
            for violation in score_convention(candidate, (row,)).violations
        )

    report = format_sign_report((row,))
    assert "declared_flux_exponent not scored" in report
    assert "E101011's G-EQDSK (section 7)" in report


def test_foreign_cohort_report_states_only_its_own_measurements():
    """A non-MAST cohort reports the coefficients its own rows measure.

    The raw absolute-loop response fixes sigma_Bp and the q relation fixes
    sigma_rho_theta_phi; e_Bp and sigma_R_phi_Z stay undetermined, so the
    verdict lists every candidate the scored relations leave — here
    (1, 2, 11, 12) — and MAST's fixed blocks are suppressed.
    """

    cohort = tuple(
        replace(
            MAST_LEVEL2_SIGN_TABLE[0],
            shot=shot,
            plasma_current_a=8.0e5,
            raw_flux_loop_response_wb_per_a=1.0e-6,
            raw_flux_loop_channels=1,
            raw_flux_loop_opposite_sign_channels=0,
            toroidal_field_t=2.5,
            poloidal_flux_edge_minus_axis_wb_per_rad=None,
            poloidal_angle_signed_area_m2=None,
            safety_factor=8.0,
            flux_exponent=None,
        )
        for shot in (100595, 100579)
    )

    assert surviving_conventions(cohort) == (1, 2, 11, 12)

    report = format_sign_report(cohort)
    assert "4 conventions survive: (1, 2, 11, 12)" in report
    assert "COHORT COEFFICIENT CLASSIFICATION" in report
    assert "sigma_Bp: measurable-from-data; value=+1" in report
    assert "sigma_rho_theta_phi: measurable-from-data; value=+1" in report
    assert "sigma_R_phi_Z: requires-an-external-declaration; value=unknown" in report
    assert "e_Bp: requires-an-external-declaration; value=unknown" in report
    assert "value=+0" not in report
    assert "E101011's G-EQDSK (section 7)" in report
    assert "DETERMINABLE RELATIVE-SIGN PRODUCTS" not in report
    assert "COCOS 3 versus COCOS 4" not in report
    assert "RECOMMENDATION" not in report
    assert "IP-LIKE CONSEQUENCE" not in report
    assert "IP-LIKE TARGETS" not in report


def test_mast_cohort_report_still_prints_its_fixed_blocks():
    """MAST's report text is unchanged: its five blocks still print."""

    report = format_sign_report()

    assert "COEFFICIENT CLASSIFICATION" in report
    assert "COHORT COEFFICIENT CLASSIFICATION" not in report
    assert "DETERMINABLE RELATIVE-SIGN PRODUCTS" in report
    assert "COCOS 3 versus COCOS 4" in report
    assert "RECOMMENDATION" in report
    assert "IP-LIKE CONSEQUENCE" in report
    assert "IP-LIKE TARGETS: magnetics/ip, pf_active/coil/current, " in report
    assert "e_Bp: requires-an-external-declaration; value=+0" in report
    assert "2 conventions survive: (3, 4)" in report
