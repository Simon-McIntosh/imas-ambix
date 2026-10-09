"""The hand-off score partitions rows and compares complete signal identities."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

from imas_alambic.machine_map import SensorIdentityRule
from imas_ambix.data import handoff_score
from imas_ambix.data.generated_mapping_import import (
    _rule_id,
    import_generated_mappings,
)

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


def test_score_completes_with_a_row_the_dd_does_not_define_refused():
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    value = document["ids"][0]["signals"][0]
    undefined = dict(
        value, target_path="magnetics/b_field_pol_probe/conductor/current/data"
    )
    undefined["signal_id"] += ":conductor"
    document["ids"][0]["signals"].append(undefined)
    imported = import_generated_mappings(document, _catalogue(), MEMBERS)

    result = handoff_score.score_handoff(
        document, _catalogue(), MEMBERS, [imported.maps[0]]
    )

    magnetics = result["by_ids"]["magnetics"]
    assert magnetics["refused_reasons"] == {
        "unexpanded: No FacilitySignal member is linked to this source": 1,
        "target path 'magnetics/b_field_pol_probe/conductor/current/data' is not "
        "defined in Data Dictionary 4.1.1": 1,
    }
    assert magnetics["agreeing"] == 2
    assert magnetics["conflicting"] == 0


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


def test_a_rule_reads_its_own_row_when_one_signal_binds_two_structures(monkeypatch):
    """Two rules from one hand-off signal each read their own row's cocos_label."""
    signal_id = "jt-60sa:general/mdac_coilpair"
    rows = [
        {
            "signal_id": signal_id,
            "target_path": "pf_active/coil/current/data",
            "source_group": "MDAC",
            "source_array": "coilA",
            "cocos_label": "ip_like",
        },
        {
            "signal_id": signal_id,
            "target_path": "pf_active/circuit/current/data",
            "source_group": "MDAC",
            "source_array": "coilA",
            "cocos_label": "one_like",
        },
    ]
    monkeypatch.setattr(
        handoff_score, "_target_cocos_label", lambda *args: "dd_fallback"
    )

    def rule(target_path):
        return SimpleNamespace(
            semantic_id=_rule_id(signal_id, target_path, 0),
            target_path=target_path,
            source_group="MDAC",
            source_array="coilA",
        )

    assert (
        handoff_score._handoff_label(rows, rule("pf_active/coil/current/data"), "4.1.1")
        == "ip_like"
    )
    assert (
        handoff_score._handoff_label(
            rows, rule("pf_active/circuit/current/data"), "4.1.1"
        )
        == "one_like"
    )


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


def test_conflicting_counts_a_target_whose_source_array_differs():
    """A row on a hand-built target bound by another array is a conflict."""
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

    magnetics = result["by_ids"]["magnetics"]
    assert magnetics["agreeing"] == 1
    assert magnetics["conflicting"] == 1
    assert magnetics["conflicts"] == [
        {
            "target_path": "magnetics/b_field_pol_probe/field/data",
            "target_index": second.target_index,
            "imported_source_array": "magPbTC11",
            "hand_built_source_arrays": ["other"],
        }
    ]
    assert result["total"]["conflicting"] == 1
    assert (
        result["by_structure"]["magnetics/b_field_pol_probe"]["conflicting"] == 1
    )


def test_a_target_without_a_hand_built_rule_is_not_conflicting():
    """An imported row with no hand-built rule on its target is not a conflict."""
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    imported = import_generated_mappings(document, _catalogue(), MEMBERS)
    probe = imported.maps[0]
    reference = dataclasses.replace(probe, signals=(probe.signals[0],))

    result = handoff_score.score_handoff(document, _catalogue(), MEMBERS, [reference])

    magnetics = result["by_ids"]["magnetics"]
    assert magnetics["agreeing"] == 1
    assert magnetics["conflicting"] == 0
    assert magnetics["conflicts"] == []


def test_an_agreeing_row_is_not_conflicting():
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    imported = import_generated_mappings(document, _catalogue(), MEMBERS)

    result = handoff_score.score_handoff(
        document, _catalogue(), MEMBERS, [imported.maps[0]]
    )

    magnetics = result["by_ids"]["magnetics"]
    assert magnetics["agreeing"] == 2
    assert magnetics["conflicting"] == 0
    assert magnetics["conflicts"] == []


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
