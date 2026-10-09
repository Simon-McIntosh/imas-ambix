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
    assert result["by_structure"]["magnetics/b_field_pol_probe"]["conflicting"] == 1


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
        signals=(dataclasses.replace(imported.pending[0].rule, channel_factor=-1.0),),
    )

    result = handoff_score.score_handoff(
        document, _catalogue(None), MEMBERS, [reference]
    )

    magnetics = result["by_ids"]["magnetics"]
    assert magnetics["agreeing"] == 1
    assert magnetics["sign_unscored"] == 1
    assert magnetics["imported"] == 2


def _launcher_document() -> dict:
    """A hand-off for an IDS the machine description carries no store for."""
    return {
        "format": "imas-codex-mapping-handoff",
        "format_version": 1,
        "facility": "jt-60sa",
        "dd_version": "4.1.1",
        "exported_at": "2026-10-09T11:21:53Z",
        "ids": [
            {
                "ids_name": "ec_launchers",
                "mapping_id": "jt-60sa:ec_launchers",
                "status": "generated",
                "signals": [
                    {
                        "signal_id": "jt-60sa:general/ech_freu8",
                        "source_id": "jt-60sa:ec_launchers:beam",
                        "data_source": "edas",
                        "source_group": "ECH",
                        "source_array": "freU8",
                        "member_identifier": "8",
                        "source_property": "value",
                        "target_path": "ec_launchers/beam/frequency/data",
                        "transform_expression": None,
                        "source_units": "Hz",
                        "target_units": "Hz",
                        "cocos_label": None,
                        "confidence": 0.8,
                        "evidence": "ECH beam frequency",
                    },
                    {
                        "signal_id": "jt-60sa:general/ech_powu7omode",
                        "source_id": "jt-60sa:ec_launchers:beam",
                        "data_source": "edas",
                        "source_group": "ECH",
                        "source_array": "powU7Omode",
                        "member_identifier": None,
                        "source_property": "value",
                        "target_path": "ec_launchers/beam/power_launched/data",
                        "transform_expression": None,
                        "source_units": "W",
                        "target_units": "W",
                        "cocos_label": None,
                        "confidence": 0.8,
                        "evidence": "ECH beam power",
                    },
                ],
                "unexpanded": [],
            }
        ],
    }


def test_sign_unscored_counts_a_non_agreeing_pending_rule():
    """A pending rule the hand-built map does not match still counts unscored.

    The old sign_unscored counted only agreeing rules, so a COCOS-pending row
    that landed on a different index left no trace in the score at all.
    """
    document, _ = _case()
    document["ids"][0]["signals"][0]["cocos_label"] = "ip_like"
    document["ids"][0]["signals"][0]["cocos_label_source"] = "xml"
    imported = import_generated_mappings(document, _catalogue(None), MEMBERS)
    assert len(imported.pending) == 1
    pending = imported.pending[0].rule
    elsewhere = dataclasses.replace(pending, target_index=pending.target_index + 5)
    reference = dataclasses.replace(imported.maps[0], signals=(elsewhere,))

    result = handoff_score.score_handoff(
        document, _catalogue(None), MEMBERS, [reference]
    )

    magnetics = result["by_ids"]["magnetics"]
    assert magnetics["agreeing"] == 0
    assert magnetics["sign_unscored"] == 1


def test_a_channel_served_elsewhere_is_listed_under_cross_structure():
    """A channel the hand-built map serves on another structure is reported.

    The imported rule targets a poloidal probe; the hand-built map serves the
    same source channel on a phi probe, so neither the target match nor the
    same-target conflict test reaches it. Both targets are named.
    """
    document, _ = _case()
    imported = import_generated_mappings(document, _catalogue(), MEMBERS)
    probe = imported.maps[0]
    first, second = probe.signals
    moved = dataclasses.replace(
        second,
        target_path="magnetics/b_field_phi_probe/field/data",
        target_index=3,
    )
    reference = dataclasses.replace(probe, signals=(first, moved))

    result = handoff_score.score_handoff(document, _catalogue(), MEMBERS, [reference])

    magnetics = result["by_ids"]["magnetics"]
    assert magnetics["agreeing"] == 1
    assert magnetics["cross_structure"] == [
        {
            "source_group": "MDAC",
            "source_array": "magPbTC11",
            "imported_target_path": "magnetics/b_field_pol_probe/field/data",
            "imported_target_index": 0,
            "hand_built_targets": [
                {
                    "target_path": "magnetics/b_field_phi_probe/field/data",
                    "target_index": 3,
                }
            ],
        }
    ]
    assert result["total"]["cross_structure"] == magnetics["cross_structure"]


def test_a_channel_served_on_another_index_is_listed_under_cross_structure():
    """A channel bound at index 0 by the import but index 27 by the hand-built map.

    Both rules share the source channel, structure and target path, so only the
    element index separates them; the score names both targets rather than
    reading the pair as neither agreeing nor conflicting.
    """
    document, _ = _case()
    imported = import_generated_mappings(document, _catalogue(), MEMBERS)
    probe = imported.maps[0]
    first, second = probe.signals
    moved = dataclasses.replace(second, target_index=second.target_index + 27)
    reference = dataclasses.replace(probe, signals=(first, moved))

    result = handoff_score.score_handoff(document, _catalogue(), MEMBERS, [reference])

    magnetics = result["by_ids"]["magnetics"]
    assert magnetics["agreeing"] == 1
    assert magnetics["conflicting"] == 0
    assert magnetics["cross_structure"] == [
        {
            "source_group": "MDAC",
            "source_array": "magPbTC11",
            "imported_target_path": "magnetics/b_field_pol_probe/field/data",
            "imported_target_index": 0,
            "hand_built_targets": [
                {
                    "target_path": "magnetics/b_field_pol_probe/field/data",
                    "target_index": 27,
                }
            ],
        }
    ]


def test_an_ids_without_a_description_store_is_reported_unplaced_by_name():
    """Every row of an IDS the description holds no store for is unplaced.

    The cause names the IDS, so a study reads the missing store apart from a
    row that carries no member pattern of its own.
    """
    result = handoff_score.score_handoff(_launcher_document(), _catalogue(), {}, [])

    launchers = result["by_ids"]["ec_launchers"]
    assert launchers["agreeing"] == "no reference"
    assert launchers["imported"] == 0
    assert launchers["unplaced"] == 2
    assert launchers["refused"] == 0
    assert launchers["unplaced_reasons"] == {
        "no machine-description store for ec_launchers": 1,
        "target path, member identifier, and source array are required": 1,
    }
    assert result["total"]["unplaced_reasons"] == launchers["unplaced_reasons"]
