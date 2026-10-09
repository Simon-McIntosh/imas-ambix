"""Generated handoff rows become draft rules only with description identities."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

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
        "imas-codex mapping_id=jt-60sa:magnetics:4.1.1; "
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
