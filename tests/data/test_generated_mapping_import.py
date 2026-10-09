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
