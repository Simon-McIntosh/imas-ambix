"""Generated handoff rows become draft rules only with description identities."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from imas_alambic.machine_map import SensorIdentityRule
from imas_alambic.signal_map import SignalMap, SignalRule
from imas_ambix.data.generated_mapping_import import import_generated_mappings

FIXTURE = Path(__file__).parents[1] / "fixtures" / "mapping_handoff_example.json"
DESCRIPTION_MEMBERS = {
    "magnetics/b_field_pol_probe": ("011", "010"),
    "pf_active/coil": ("02", "01"),
}


def _catalogue():
    return SimpleNamespace(
        sensor_identity_rules=(
            SensorIdentityRule(
                name="sensor-identity",
                case_rule="case-fold",
                numeric_token_rule="integer-value",
                evidence="description identities use different zero padding",
            ),
        )
    )


def test_fixture_indices_follow_description_identity_not_handoff_order():
    imported = import_generated_mappings(FIXTURE, _catalogue(), DESCRIPTION_MEMBERS)

    assert {mapping.system for mapping in imported.maps} == {"magnetics", "pf_active"}
    by_source = {
        signal.source_array: signal
        for mapping in imported.maps
        for signal in mapping.signals
    }
    assert {source: signal.target_index for source, signal in by_source.items()} == {
        "magPbTC10": 1,
        "magPbTC11": 0,
        "curCS1LKAT": 1,
        "curCS2LKAT": 0,
    }
    assert all(signal.validation_state == "draft" for signal in by_source.values())
    assert by_source["magPbTC10"].evidence == (
        "imas-codex mapping_id=jt-60sa:magnetics; "
        "status=generated; EDAS MDAC pickup probe channel"
    )
    assert imported.unresolved[0].reason == (
        "unexpanded: No FacilitySignal member is linked to this source"
    )


def test_facility_member_alias_resolves_to_description_index():
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    row = document["ids"][1]["signals"][0]
    row["member_identifier"] = "UFP"
    row["source_array"] = "curUFPLKAT"
    members = dict(DESCRIPTION_MEMBERS)
    members["pf_active/coil"] = ("02", "FPPC_UP")
    catalogue = _catalogue()
    rule = catalogue.sensor_identity_rules[0]
    catalogue.sensor_identity_rules = (
        SensorIdentityRule(
            name=rule.name,
            case_rule=rule.case_rule,
            numeric_token_rule=rule.numeric_token_rule,
            evidence=rule.evidence,
            member_aliases={"UFP": "FPPC_UP"},
        ),
    )

    imported = import_generated_mappings(document, catalogue, members)

    assert any(
        signal.source_array == "curUFPLKAT" and signal.target_index == 1
        for mapping in imported.maps
        for signal in mapping.signals
    )
    assert all(item.source_array != "curUFPLKAT" for item in imported.unresolved)


def test_unresolvable_member_is_reported_without_an_index():
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    missing = document["ids"][0]["signals"][0]
    missing["member_identifier"] = "99"
    missing["source_array"] = "magPbTC99"

    imported = import_generated_mappings(document, _catalogue(), DESCRIPTION_MEMBERS)

    assert all(
        signal.source_array != "magPbTC99"
        for mapping in imported.maps
        for signal in mapping.signals
    )
    assert imported.unresolved[0].source_array == "magPbTC99"
    assert imported.unresolved[0].reason == (
        "member identifier '99' has no identity match in "
        "machine-description magnetics/b_field_pol_probe"
    )


def test_member_identifier_must_match_the_source_array():
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    document["ids"][0]["signals"][0]["source_array"] = "magPbTC12"

    imported = import_generated_mappings(document, _catalogue(), DESCRIPTION_MEMBERS)

    assert imported.unresolved[0].reason == (
        "member identifier '10' is absent from source array 'magPbTC12'"
    )
    assert all(
        signal.source_array != "magPbTC12"
        for mapping in imported.maps
        for signal in mapping.signals
    )


def test_alphanumeric_member_rejects_a_longer_numeric_token():
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    signal = document["ids"][1]["signals"][0]
    signal["member_identifier"] = "CS1"
    signal["source_array"] = "curCS11LKAT"
    members = dict(DESCRIPTION_MEMBERS)
    members["pf_active/coil"] = ("CS1", "CS11")

    imported = import_generated_mappings(document, _catalogue(), members)

    assert any(
        item.source_array == "curCS11LKAT"
        and item.reason
        == "member identifier 'CS1' is absent from source array 'curCS11LKAT'"
        for item in imported.unresolved
    )


def test_draft_rule_and_map_round_trip_through_serialisation():
    imported = import_generated_mappings(FIXTURE, _catalogue(), DESCRIPTION_MEMBERS)
    rule = imported.maps[0].signals[0]

    assert SignalRule.from_dict(rule.as_dict()) == rule
    assert (
        SignalMap.from_dict(json.loads(imported.maps[0].canonical_bytes()))
        == (imported.maps[0])
    )


def test_value_expression_imports_as_identity():
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    document["ids"][0]["signals"][0]["transform_expression"] = "value"
    document["ids"][0]["signals"][0]["evidence"] = ""

    imported = import_generated_mappings(document, _catalogue(), DESCRIPTION_MEMBERS)

    assert imported.maps[0].signals[0].transformation == "one_like"
    assert imported.maps[0].signals[0].evidence == (
        "imas-codex mapping_id=jt-60sa:magnetics; status=generated"
    )
    assert all(
        row.target_path != "magnetics/b_field_pol_probe/field/data"
        for row in imported.unresolved
    )


def test_derived_error_and_unqualified_time_rows_are_unresolved():
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    value = document["ids"][0]["signals"][0]
    derived = []
    for field in ("data_error_upper", "data_error_lower", "data_error_index", "time"):
        row = dict(
            value, target_path=value["target_path"].replace("/data", f"/{field}")
        )
        row["transform_expression"] = "value"
        row["source_property"] = None
        row["signal_id"] = f"{value['signal_id']}:{field}"
        derived.append(row)
    document["ids"][0]["signals"].extend(derived)

    imported = import_generated_mappings(document, _catalogue(), DESCRIPTION_MEMBERS)

    error_reason = (
        "imas-codex derived error bounds from the value; no error signal exists"
    )
    time_reason = (
        "handoff has no source_property; time-vector binding cannot be "
        "distinguished from a value binding"
    )
    assert {
        row.target_path.rsplit("/", 1)[-1]: row.reason
        for row in imported.unresolved
        if row.source_array == value["source_array"]
    } == {
        "data_error_upper": error_reason,
        "data_error_lower": error_reason,
        "data_error_index": error_reason,
        "time": time_reason,
    }
    assert all(
        signal.target_path.endswith("/data")
        for mapping in imported.maps
        for signal in mapping.signals
    )


def test_time_row_resolves_to_its_value_rule_even_when_first():
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    value = document["ids"][0]["signals"][0]
    assert value["source_property"] == "value"
    time = dict(value, source_property="time")
    time["target_path"] = value["target_path"].replace("/data", "/time")
    time["signal_id"] += ":time"
    time["source_units"] = time["target_units"] = "s"
    document["ids"][0]["signals"].insert(0, time)

    imported = import_generated_mappings(document, _catalogue(), DESCRIPTION_MEMBERS)
    signals = imported.maps[0].signals
    partner = next(
        rule for rule in signals if rule.source_array == value["source_array"]
    )
    assert not any(rule.target_path.endswith("/time") for rule in signals)
    assert len(imported.time_bindings) == 1
    assert imported.time_bindings[0].target_path == time["target_path"]
    assert imported.time_bindings[0].value_rule is partner
    assert partner.semantic_id == value["signal_id"]
    assert all(row.target_path != time["target_path"] for row in imported.unresolved)


def test_time_row_without_a_resolved_value_partner_reports_why():
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    value = document["ids"][0]["signals"][0]
    time = dict(value, source_property="time")
    time["target_path"] = value["target_path"].replace("/data", "/time")
    time["source_units"] = time["target_units"] = "s"
    document["ids"][0]["signals"].append(time)

    value["source_array"] = "magPbTC99"
    imported = import_generated_mappings(document, _catalogue(), DESCRIPTION_MEMBERS)
    assert imported.time_bindings == ()
    assert (
        next(
            row.reason
            for row in imported.unresolved
            if row.target_path == time["target_path"]
        )
        == "value partner absent for the same channel, member, and target structure"
    )

    value["source_array"] = time["source_array"]
    value["source_units"] = "not-tesla"
    imported = import_generated_mappings(document, _catalogue(), DESCRIPTION_MEMBERS)
    assert imported.time_bindings == ()
    assert (
        next(
            row.reason
            for row in imported.unresolved
            if row.target_path == time["target_path"]
        )
        == "value partner unresolved: source and target units differ "
        "without a conversion factor"
    )


@pytest.mark.parametrize(
    ("field", "different"),
    [
        ("source_group", "OTHER"),
        ("source_array", "magPbTC11"),
        ("member_identifier", "11"),
        ("target_path", "magnetics/b_field_pol_probe/other/time"),
    ],
)
def test_time_partner_requires_same_channel_member_and_structure(field, different):
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    value = document["ids"][0]["signals"][0]
    time = dict(value, source_property="time")
    time["target_path"] = value["target_path"].replace("/data", "/time")
    time[field] = different
    document["ids"][0]["signals"].append(time)

    imported = import_generated_mappings(document, _catalogue(), DESCRIPTION_MEMBERS)
    assert imported.time_bindings == ()
    assert any(
        row.target_path == time["target_path"]
        and row.reason.startswith("value partner absent")
        for row in imported.unresolved
    )


def test_time_property_cannot_be_imported_as_data():
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    document["ids"][0]["signals"][0]["source_property"] = "time"
    imported = import_generated_mappings(document, _catalogue(), DESCRIPTION_MEMBERS)
    assert all(
        rule.source_array != "magPbTC10"
        for mapping in imported.maps
        for rule in mapping.signals
    )
    assert any("source_property=time" in row.reason for row in imported.unresolved)


def _labelled_document(label, source):
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    row = document["ids"][0]["signals"][0]
    row["cocos_label"] = label
    row["cocos_label_source"] = source
    return document


def _imported_rule(imported, source_array):
    return next(
        rule
        for mapping in imported.maps
        for rule in mapping.signals
        if rule.source_array == source_array
    )


def test_one_like_label_imports_with_its_label_and_source_in_evidence():
    imported = import_generated_mappings(
        _labelled_document("one_like", "xml"), _catalogue(), DESCRIPTION_MEMBERS
    )

    rule = _imported_rule(imported, "magPbTC10")
    assert "cocos_label=one_like" in rule.evidence
    assert "cocos_label_source=xml" in rule.evidence
    assert imported.pending == ()


def test_none_label_imports_and_is_not_pending():
    imported = import_generated_mappings(
        _labelled_document("none", "none"), _catalogue(), DESCRIPTION_MEMBERS
    )

    assert "cocos_label=none" in _imported_rule(imported, "magPbTC10").evidence
    assert imported.pending == ()


def test_cocos_dependent_label_becomes_a_pending_rule_absent_from_the_maps():
    imported = import_generated_mappings(
        _labelled_document("ip_like", "inferred_forward"),
        _catalogue(),
        DESCRIPTION_MEMBERS,
    )

    assert len(imported.pending) == 1
    pending = imported.pending[0]
    assert pending.cocos_label == "ip_like"
    assert pending.cocos_label_source == "inferred_forward"
    assert pending.open_source_cocos is None
    assert "ip_like" in pending.reason
    assert "inferred_forward" in pending.reason
    assert "undeclared source COCOS" in pending.reason
    assert pending.rule.source_array == "magPbTC10"
    assert pending.rule.validation_state == "draft"
    assert all(
        rule.source_array != "magPbTC10"
        for mapping in imported.maps
        for rule in mapping.signals
    )


def test_unknown_cocos_label_is_refused_by_name():
    imported = import_generated_mappings(
        _labelled_document("not_a_label", "xml"), _catalogue(), DESCRIPTION_MEMBERS
    )

    assert any(
        row.source_array == "magPbTC10"
        and row.reason == "unrecognised COCOS label 'not_a_label'"
        for row in imported.unresolved
    )
    assert imported.pending == ()


def test_null_cocos_label_keeps_the_unlabelled_behaviour():
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert document["ids"][0]["signals"][0]["cocos_label"] is None

    imported = import_generated_mappings(document, _catalogue(), DESCRIPTION_MEMBERS)

    assert "cocos_label" not in _imported_rule(imported, "magPbTC10").evidence
    assert imported.pending == ()


def test_declared_source_cocos_imports_a_dependent_label_into_the_maps():
    catalogue = _catalogue()
    catalogue.source_cocos = 17

    imported = import_generated_mappings(
        _labelled_document("ip_like", "xml"), catalogue, DESCRIPTION_MEMBERS
    )

    assert imported.pending == ()
    assert "cocos_label=ip_like" in _imported_rule(imported, "magPbTC10").evidence


def test_zero_source_cocos_is_undeclared_and_holds_the_row_pending():
    catalogue = _catalogue()
    catalogue.source_cocos = 0

    imported = import_generated_mappings(
        _labelled_document("psi_like", "inferred_forward"),
        catalogue,
        DESCRIPTION_MEMBERS,
    )

    assert len(imported.pending) == 1
    assert imported.pending[0].open_source_cocos == 0
    assert all(
        rule.source_array != "magPbTC10"
        for mapping in imported.maps
        for rule in mapping.signals
    )


def test_time_row_binds_to_its_pending_cocos_value_placeholder():
    document = _labelled_document("ip_like", "xml")
    value = document["ids"][0]["signals"][0]
    time = dict(value, source_property="time")
    time["target_path"] = value["target_path"].replace("/data", "/time")
    time["signal_id"] += ":time"
    time["source_units"] = time["target_units"] = "s"
    time["cocos_label"] = "none"
    time["cocos_label_source"] = "none"
    document["ids"][0]["signals"].insert(0, time)

    imported = import_generated_mappings(document, _catalogue(), DESCRIPTION_MEMBERS)

    assert len(imported.pending) == 1
    assert len(imported.time_bindings) == 1
    assert imported.time_bindings[0].value_rule is imported.pending[0].rule
    assert all(row.target_path != time["target_path"] for row in imported.unresolved)
