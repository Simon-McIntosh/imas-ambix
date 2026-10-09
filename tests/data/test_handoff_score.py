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


def test_labelled_target_under_undeclared_source_has_unscored_sign(monkeypatch):
    document, reference = _case()
    monkeypatch.setattr(
        handoff_score,
        "_target_cocos_label",
        lambda version, path: "psi_like" if path.startswith("magnetics/") else None,
    )

    result = handoff_score.score_handoff(
        document, _catalogue(None), MEMBERS, [reference]
    )

    assert result["by_ids"]["magnetics"]["sign"] == "unscored"
    assert result["by_ids"]["magnetics"]["unscored_sign"] == 2
    assert result["by_ids"]["magnetics"]["agreeing"] == 0


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
