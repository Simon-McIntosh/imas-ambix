"""The hand-off score partitions rows and compares complete signal identities."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

from imas_alambic.machine_map import SensorIdentityRule
from imas_ambix.data import handoff_score
from imas_ambix.data.generated_mapping_import import import_generated_mappings

FIXTURE = Path(__file__).parents[1] / "fixtures" / "mapping_handoff_example.json"
MEMBERS = {
    "magnetics/b_field_pol_probe": ("011", "010"),
    "pf_active/coil": ("02", "01"),
}


def _catalogue(source_cocos=1):
    return SimpleNamespace(
        source_cocos=source_cocos,
        sensor_identity_rules=(
            SensorIdentityRule(
                name="sensor-identity",
                case_rule="case-fold",
                numeric_token_rule="integer-value",
                evidence="ordered description identities",
            ),
        ),
    )


def _case():
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    signals = document["ids"][0]["signals"]
    absent = dict(signals[0], member_identifier="99", source_array="magPbTC99")
    refused = dict(signals[0], source_property="time")
    signals.extend((absent, refused))
    imported = import_generated_mappings(document, _catalogue(), MEMBERS)
    reference = imported.maps[0]
    first, second = reference.signals
    reference = dataclasses.replace(
        reference,
        signals=(first, dataclasses.replace(second, channel_factor=-1.0)),
    )
    return document, reference


def test_counts_rows_and_requires_source_and_total_scale_for_agreement():
    document, reference = _case()
    result = handoff_score.score_handoff(document, _catalogue(), MEMBERS, [reference])

    magnetics = result["by_ids"]["magnetics"]
    assert (magnetics["exported"], magnetics["imported"], magnetics["agreeing"]) == (
        5,
        2,
        1,
    )
    assert (magnetics["unplaced"], magnetics["refused"]) == (1, 2)
    assert (
        magnetics["refused_reasons"][
            "unexpanded: No FacilitySignal member is linked to this source"
        ]
        == 1
    )
    assert result["by_ids"]["pf_active"]["agreeing"] == "no reference"
    assert (result["total"]["exported"], result["total"]["imported"]) == (7, 4)
    assert result["total"]["agreeing"] == 1
    assert result["total"]["sign_unscored"] == 0


def test_one_ids_splits_into_separate_structure_rows():
    document, reference = _case()
    result = handoff_score.score_handoff(document, _catalogue(), MEMBERS, [reference])

    probe = result["by_structure"]["magnetics/b_field_pol_probe"]
    loops = result["by_structure"]["magnetics/flux_loop"]
    assert (probe["exported"], probe["imported"], probe["agreeing"]) == (4, 2, 1)
    assert (probe["unplaced"], probe["refused"]) == (1, 1)
    assert (loops["exported"], loops["imported"]) == (1, 0)
    assert loops["agreeing"] == "no reference"
    assert (loops["unplaced"], loops["refused"]) == (0, 1)


def test_one_like_stays_sign_scored_without_source_cocos():
    document, reference = _case()
    assert (
        handoff_score._target_cocos_label(
            "4.1.1", "magnetics/b_field_pol_probe/field/data"
        )
        == "one_like"
    )

    result = handoff_score.score_handoff(
        document, _catalogue(None), MEMBERS, [reference]
    )

    assert result["by_ids"]["magnetics"]["agreeing"] == 1
    assert result["by_ids"]["magnetics"]["sign_unscored"] == 0


def _graph_labelled_case(monkeypatch):
    document, reference = _case()
    imported = import_generated_mappings(document, _catalogue(), MEMBERS)
    document["ids"][0]["signals"][0]["cocos_label"] = "psi_like"
    monkeypatch.setattr(
        handoff_score, "import_generated_mappings", lambda *args: imported
    )
    monkeypatch.setattr(handoff_score, "_target_cocos_label", lambda *args: None)
    return document, reference


def test_handoff_label_overrides_absent_dd_label(monkeypatch):
    document, reference = _graph_labelled_case(monkeypatch)

    result = handoff_score.score_handoff(
        document, _catalogue(None), MEMBERS, [reference]
    )

    assert result["by_ids"]["magnetics"]["agreeing"] == 1
    assert result["by_ids"]["magnetics"]["sign_unscored"] == 1


def test_unscored_sign_agrees_on_scale_magnitude(monkeypatch):
    document, reference = _graph_labelled_case(monkeypatch)
    first, second = reference.signals
    reference = dataclasses.replace(
        reference,
        signals=(dataclasses.replace(first, channel_factor=-1.0), second),
    )

    result = handoff_score.score_handoff(
        document, _catalogue(None), MEMBERS, [reference]
    )

    assert result["by_ids"]["magnetics"]["agreeing"] == 1
    assert result["by_ids"]["magnetics"]["sign_unscored"] == 1


def test_matching_target_with_a_different_source_array_does_not_agree():
    document, reference = _case()
    first, second = reference.signals
    reference = dataclasses.replace(
        reference,
        signals=(
            first,
            dataclasses.replace(second, channel_factor=1.0, source_array="other"),
        ),
    )

    result = handoff_score.score_handoff(document, _catalogue(), MEMBERS, [reference])

    assert result["by_ids"]["magnetics"]["agreeing"] == 1


def test_pending_cocos_rule_counts_as_imported_with_the_sign_unscored():
    document, _ = _case()
    document["ids"][0]["signals"][0]["cocos_label"] = "ip_like"
    document["ids"][0]["signals"][0]["cocos_label_source"] = "xml"
    imported = import_generated_mappings(document, _catalogue(None), MEMBERS)
    assert len(imported.pending) == 1
    reference = dataclasses.replace(
        imported.maps[0],
        signals=(
            dataclasses.replace(imported.pending[0].rule, channel_factor=-1.0),
        ),
    )

    result = handoff_score.score_handoff(
        document, _catalogue(None), MEMBERS, [reference]
    )

    magnetics = result["by_ids"]["magnetics"]
    assert magnetics["agreeing"] == 1
    assert magnetics["sign_unscored"] == 1
    assert magnetics["imported"] == 2
