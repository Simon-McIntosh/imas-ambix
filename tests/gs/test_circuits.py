"""Tests for the MAST PF power-supply / circuit description.

Pure-logic tests only (no mirror, no network): the circuit topology is a
hardcoded machine description (like ``operator._PF_COIL_CENTROID``), so its
correctness is checked against (a) internal invariants, (b) consistency with
the sibling :mod:`imas_ambix.gs.operator` module's independent coil→amc-channel
map, and (c) ``amc`` channel listings **recorded** from three real shots
spanning both in-use ``fcoil`` discretisations (938- and 1004-filament
campaigns) — recorded by the declared acquisition map
on 2026-07-04, not read live, so this file never depends on mirror access.
"""

from __future__ import annotations

from imas_ambix.gs import circuits as pfc
from imas_ambix.gs import operator as op

# --- recorded amc channel listings (see module docstring for provenance) ---

# shot 18502 — fc938 signature (mp78-fl46-fc938-lim37-1cb6f2ee742c4ee4)
_AMC_18502 = frozenset(
    {
        "efps_current",
        "error_field_02",
        "error_field_05",
        "p2il_coil_current",
        "p2il_feed_current",
        "p2iu_coil_current",
        "p2iu_feed_current",
        "p2l_case_current",
        "p2l_current",
        "p2ol_coil_current",
        "p2ol_feed_current",
        "p2ou_coil_current",
        "p2ou_feed_current",
        "p2u_case_current",
        "p2u_current",
        "p3l_case_current",
        "p3l_coil_current",
        "p3l_current",
        "p3l_feed_current",
        "p3u_case_current",
        "p3u_coil_current",
        "p3u_current",
        "p3u_feed_current",
        "p4l_case_current",
        "p4l_coil_current",
        "p4l_current",
        "p4l_feed_current",
        "p4u_case_current",
        "p4u_coil_current",
        "p4u_current",
        "p4u_feed_current",
        "p5l_case_current",
        "p5l_coil_current",
        "p5l_current",
        "p5l_feed_current",
        "p5u_case_current",
        "p5u_coil_current",
        "p5u_current",
        "p5u_feed_current",
        "p6l_current",
        "p6u_current",
        "plasma_current",
        "sol_current",
        "tf_current",
    }
)

# shot 11764 — fc1004 signature (mp78-fl46-fc1004-lim37-9425ae4a8bf3bc15)
_AMC_11764 = frozenset(
    {
        "efps_current",
        "error_field_a",
        "error_field_b",
        "p2il_coil_current",
        "p2il_feed_current",
        "p2iu_coil_current",
        "p2iu_feed_current",
        "p2l_case_current",
        "p2l_current",
        "p2ol_coil_current",
        "p2ol_feed_current",
        "p2ou_coil_current",
        "p2ou_feed_current",
        "p2u_case_current",
        "p2u_current",
        "p3l_case_current",
        "p3l_coil_current",
        "p3l_current",
        "p3l_feed_current",
        "p3u_case_current",
        "p3u_coil_current",
        "p3u_current",
        "p3u_feed_current",
        "p4l_case_current",
        "p4l_coil_current",
        "p4l_current",
        "p4l_feed_current",
        "p4u_case_current",
        "p4u_coil_current",
        "p4u_current",
        "p4u_feed_current",
        "p5l_case_current",
        "p5l_coil_current",
        "p5l_current",
        "p5l_feed_current",
        "p5u_case_current",
        "p5u_coil_current",
        "p5u_current",
        "p5u_feed_current",
        "p6l_current",
        "p6u_current",
        "plasma_current",
        "sol_current",
        "tf_current",
    }
)

_RECORDED_SIGNATURES = {
    "mp78-fl46-fc938-lim37-1cb6f2ee742c4ee4 (shot 18502)": _AMC_18502,
    "mp78-fl46-fc1004-lim37-9425ae4a8bf3bc15 (shot 11764)": _AMC_11764,
}


# --- structural invariants ----------------------------------------------


def test_active_and_case_counts():
    assert len(pfc.active_circuits()) == pfc.MAST_ACTIVE_CIRCUIT_COUNT == 13
    assert len(pfc.case_circuits()) == pfc.MAST_CASE_CIRCUIT_COUNT == 10


def test_circuit_ids_are_1_to_23_contiguous():
    ids = sorted(c.circuit_id for c in pfc.active_circuits()) + sorted(
        c.circuit_id for c in pfc.case_circuits()
    )
    assert sorted(ids) == list(range(1, 24))


def test_series_solenoid_is_the_only_series_circuit():
    """Circuit 1 (ohmic) is the sole series circuit: solenoid1+solenoid2, one supply."""
    ohmic = pfc.active_circuit_for_coil("sol")
    assert ohmic.series is True
    assert ohmic.pf_coil_names == ("solenoid1", "solenoid2")
    assert ohmic.supply_signal_name == "amc_sol current"
    others = [c for c in pfc.active_circuits() if c.coil_label != "sol"]
    assert all(not c.series for c in others)
    assert all(len(c.pf_coil_names) == 1 for c in others)


def test_each_active_circuit_has_a_dedicated_supply():
    """13 active circuits -> 13 distinct supply ids (no shared supply besides ohmic)."""
    supply_ids = [c.supply_id for c in pfc.active_circuits()]
    assert len(set(supply_ids)) == len(supply_ids) == 13


def test_p6_case_circuits_are_the_only_constrained_zero_ones():
    constrained = [c for c in pfc.case_circuits() if c.constrained_zero]
    assert {c.coils_encased for c in constrained} == {"P6U", "P6L"}
    assert all(c.l1_case_channel is None for c in constrained)
    assert all(c.supply_scaling_a == 0.0 for c in constrained)
    non_constrained = [c for c in pfc.case_circuits() if not c.constrained_zero]
    assert len(non_constrained) == 8
    assert all(c.l1_case_channel is not None for c in non_constrained)


def test_turns_property_matches_measured_coil_to_feed_ratio():
    """Measured on 4 held-out shots (docs/mast-coil-circuits.html §3): coil_current
    / feed_current is an exact integer equal to supply_scaling_a/1000."""
    measured_turns = {
        "p2iu": 12.0,
        "p2ou": 8.0,
        "p2il": 12.0,
        "p2ol": 8.0,
        "p3u": 8.0,
        "p3l": 8.0,
        "p4u": 23.0,
        "p4l": 23.0,
        "p5u": 23.0,
        "p5l": 23.0,
    }
    for label, turns in measured_turns.items():
        circuit = pfc.active_circuit_for_coil(label)
        assert circuit.turns == turns


# --- cross-consistency with the independent operator.py machine map ------


def test_preferred_channel_matches_operator_pf_coil_amc_map():
    """circuits.py and operator.py encode the coil->channel map independently
    (different modules, different provenance path); they must agree."""
    for label, expected_channel in op._PF_COIL_AMC.items():
        assert pfc.preferred_current_channel(label) == expected_channel


def test_active_circuit_for_coil_unknown_label_raises():
    import pytest

    with pytest.raises(KeyError):
        pfc.active_circuit_for_coil("not_a_real_coil")


def test_case_circuit_for_active_coil_reverse_lookup():
    case = pfc.case_circuit_for_active_coil("p3u")
    assert case is not None
    assert case.name == "P3U case current"
    assert pfc.case_circuit_for_active_coil("sol") is None  # ohmic has no case circuit


# --- channel presence, checked against RECORDED amc listings -------------


def test_every_active_circuit_channel_present_in_every_recorded_signature():
    for sig_label, amc in _RECORDED_SIGNATURES.items():
        for ac in pfc.active_circuits():
            assert ac.l1_coil_channel in amc, (
                f"{ac.l1_coil_channel!r} missing from {sig_label}"
            )
            if ac.l1_feed_channel is not None:
                assert ac.l1_feed_channel in amc, (
                    f"{ac.l1_feed_channel!r} missing from {sig_label}"
                )


def test_every_measured_case_channel_present_in_every_recorded_signature():
    for sig_label, amc in _RECORDED_SIGNATURES.items():
        for cc in pfc.case_circuits():
            if cc.constrained_zero:
                continue
            assert cc.l1_case_channel in amc, (
                f"{cc.l1_case_channel!r} missing from {sig_label}"
            )


def test_p6_case_channels_confirmed_absent_from_recorded_signatures():
    """Cross-validates the pfSystems.xml claim ('unknown for MAST, constrained
    to 0') against real data: no p6u/p6l_case_current channel ever appears."""
    for amc in _RECORDED_SIGNATURES.values():
        assert "p6u_case_current" not in amc
        assert "p6l_case_current" not in amc


def test_verify_amc_channels_reports_clean_on_recorded_signatures():
    for amc in _RECORDED_SIGNATURES.values():
        report = pfc.verify_amc_channels(amc)
        assert report["missing_active"] == []
        assert report["missing_case"] == []
        assert report["unexpectedly_present_zero_channels"] == []


def test_verify_amc_channels_flags_a_missing_channel():
    amc = set(next(iter(_RECORDED_SIGNATURES.values()))) - {"p4u_coil_current"}
    report = pfc.verify_amc_channels(amc)
    assert "p4u_coil_current" in report["missing_active"]


# --- geometry-classifier caveat (documents, does not fix, operator.py) ---


def test_geometry_confusable_with_covers_every_active_coil_except_sol_and_p2_outer():
    """8 of the 10 case circuits will be geometrically folded into an active
    coil's G_pf column (measured against classify_circuits on real data — see
    module docstring).  P2OU/P2OL have no case-circuit counterpart because the
    combined P2U/P2L case sits closer to the INNER (p2iu/p2il) coil than the
    outer one, so only p2iu/p2il get a confusable case (not p2ou/p2ol)."""
    labels_with_case = {cc.geometry_confusable_with for cc in pfc.case_circuits()}
    active_labels = {ac.coil_label for ac in pfc.active_circuits()}
    assert labels_with_case == active_labels - {"sol", "p2ou", "p2ol"}


def test_to_dict_round_trips_json_serialisable():
    import json

    payload = pfc.to_dict()
    text = json.dumps(payload)  # must not raise
    reloaded = json.loads(text)
    assert reloaded["n_active_circuits"] == 13
    assert reloaded["n_case_circuits"] == 10


def test_write_artifact(tmp_path):
    out = pfc.write_artifact(tmp_path / "mast_pf_circuits.json")
    assert out.exists()
    import json

    payload = json.loads(out.read_text())
    assert payload["schema"] == "mast-pf-circuits-v0"
