"""Geometry transitions select exactly one machine setup for every shot."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from imas_ambix.data.geometry_transitions import (
    CONDUCTOR_OUTLINE_TOLERANCE_M,
    DISCRETISATION_ONLY,
    build_geometry_transitions,
    classify_geometry_boundaries,
    transition_for_shot,
)
from imas_ambix.data.manifest import build_manifest
from imas_ambix.data.paths import MANIFEST_DIR

_GEOMETRY_TABLE_PATH = MANIFEST_DIR / "gs_geometry_tables.json"
_CORPUS_MANIFEST_PATH = MANIFEST_DIR / "level2-all.json"
_HAVE_LOCAL_CORPUS = _GEOMETRY_TABLE_PATH.is_file() and _CORPUS_MANIFEST_PATH.is_file()


def _table(signature: str, shots: list[int], *, offset: float, filaments: int):
    return {
        "signature_key": signature,
        "shots": shots,
        "b_probes": [{"r": 1.0, "z": offset, "angle_deg": 0.0, "length": 0.1}],
        "flux_loops": [{"r": 1.2, "z": 0.0}],
        "pf_filaments": [
            {
                "r": 0.5 + offset,
                "z": 0.1,
                "turns": 1.0,
                "width": 0.02,
                "height": 0.03,
                "circuit": position + 1,
                "xmult": 1.0,
            }
            for position in range(filaments)
        ],
        "limiter_r": [0.2, 1.4],
        "limiter_z": [-1.0, 1.0],
        "r0": 0.85,
        "minor_radius": 0.65,
    }


def _mesh_table(
    signature: str, shots: list[int], elements: list[dict[str, float | int]]
):
    table = _table(signature, shots, offset=0.0, filaments=0)
    table["pf_filaments"] = elements
    return table


def _element(
    *,
    r: float,
    z: float,
    width: float,
    height: float,
    circuit: int = 1,
    weight: float = 1.0,
) -> dict[str, float | int]:
    return {
        "r": r,
        "z": z,
        "turns": 1.0,
        "width": width,
        "height": height,
        "circuit": circuit,
        "xmult": weight,
    }


@pytest.fixture
def transition_inputs():
    shots = list(range(100, 161))
    index = pd.DataFrame(
        {
            "shot_id": shots,
            "campaign": ["alpha" if shot < 130 else "beta" for shot in shots],
        }
    )
    payload = {
        "campaigns": {
            "geometry-a": _table("geometry-a", [95, 110], offset=0.0, filaments=2),
            "geometry-b": _table("geometry-b", [120, 140], offset=0.01, filaments=2),
            "geometry-c": _table(
                "geometry-c", [125, 135, 150], offset=0.02, filaments=1
            ),
        }
    }
    return shots, index, payload


def test_transition_ranges_partition_corpus_and_lookup_is_unique(transition_inputs):
    shots, index, payload = transition_inputs
    transitions = build_geometry_transitions(shots, index, payload)

    assert [(item.first_shot, item.last_shot) for item in transitions] == [
        (100, 119),
        (120, 124),
        (125, 139),
        (140, 149),
        (150, 160),
    ]
    assert transitions[0].first_shot == min(shots)
    assert transitions[-1].last_shot == max(shots)
    assert all(
        left.last_shot + 1 == right.first_shot
        for left, right in zip(transitions, transitions[1:], strict=False)
    )
    for shot in range(min(shots), max(shots) + 1):
        matches = [item for item in transitions if item.contains(shot)]
        assert matches == [transition_for_shot(transitions, shot)]


def test_transitions_name_changed_geometry_and_boundary_campaign(transition_inputs):
    shots, index, payload = transition_inputs
    transitions = build_geometry_transitions(shots, index, payload)

    assert [item.campaign for item in transitions] == [
        "alpha",
        "alpha",
        "alpha",
        "beta",
        "beta",
    ]
    assert transitions[1].name == "mast-geometry-120-b"
    assert transitions[1].changed_fields == (
        "b_probes.z",
        "pf_filaments.r",
    )
    assert "pf_filaments.count" in transitions[2].changed_fields


def test_manifest_reports_boundaries_strictly_inside_campaigns(transition_inputs):
    shots, index, payload = transition_inputs
    manifest = build_manifest(
        tier="level2",
        shot_ids=shots,
        campaign_index=index,
        geometry_payload=payload,
    )
    encoded = json.loads(manifest.to_json())

    assert encoded["geometry_transition_count"] == 5
    assert encoded["geometry_transitions_inside_campaign"] == 4
    assert encoded["geometry_transitions"][0]["first_shot"] == 100
    assert encoded["geometry_transitions"][-1]["last_shot"] == 160


def test_manifest_requires_campaign_and_geometry_inputs_together(transition_inputs):
    shots, index, _ = transition_inputs
    with pytest.raises(ValueError, match="must be provided together"):
        build_manifest(tier="level2", shot_ids=shots, campaign_index=index)


def test_resolution_only_change_does_not_open_a_geometry_range():
    shots = list(range(100, 141))
    index = pd.DataFrame({"shot_id": shots, "campaign": ["alpha"] * len(shots)})
    whole = [_element(r=1.0, z=0.0, width=0.4, height=0.6)]
    subdivided = [
        _element(
            r=r,
            z=z,
            width=0.2,
            height=0.3,
            weight=0.25,
        )
        for z in (-0.15, 0.15)
        for r in (0.9, 1.1)
    ]
    payload = {
        "campaigns": {
            "geometry-coarse": _mesh_table("geometry-coarse", [95], whole),
            "geometry-fine": _mesh_table("geometry-fine", [120], subdivided),
        }
    }

    assessments = classify_geometry_boundaries(shots, payload)
    transitions = build_geometry_transitions(shots, index, payload)

    assert len(assessments) == 1
    assert assessments[0].classification == DISCRETISATION_ONLY
    assert assessments[0].before_element_count == 1
    assert assessments[0].after_element_count == 4
    assert assessments[0].conductor_outline_residual_m == pytest.approx(0.0)
    assert assessments[0].conductor_outline_residual_m <= (
        CONDUCTOR_OUTLINE_TOLERANCE_M
    )
    assert [(item.first_shot, item.last_shot) for item in transitions] == [(100, 140)]
    assert transition_for_shot(transitions, 120) is transitions[0]


def test_changed_conductor_outline_opens_a_geometry_range():
    shots = list(range(100, 141))
    index = pd.DataFrame({"shot_id": shots, "campaign": ["alpha"] * len(shots)})
    payload = {
        "campaigns": {
            "geometry-left": _mesh_table(
                "geometry-left",
                [95],
                [_element(r=1.0, z=0.0, width=0.4, height=0.6)],
            ),
            "geometry-right": _mesh_table(
                "geometry-right",
                [120],
                [_element(r=1.02, z=0.0, width=0.4, height=0.6)],
            ),
        }
    }

    assessments = classify_geometry_boundaries(shots, payload)
    transitions = build_geometry_transitions(shots, index, payload)

    assert assessments[0].opens_range
    assert assessments[0].conductor_outline_residual_m == pytest.approx(0.02)
    assert [(item.first_shot, item.last_shot) for item in transitions] == [
        (100, 119),
        (120, 140),
    ]


@pytest.mark.skipif(not _HAVE_LOCAL_CORPUS, reason="local MAST corpus unavailable")
def test_local_filament_resolution_boundaries_are_discretisation():
    corpus = json.loads(_CORPUS_MANIFEST_PATH.read_text())["shot_ids"]
    payload = json.loads(_GEOMETRY_TABLE_PATH.read_text())
    index = pd.DataFrame(
        {"shot_id": corpus, "campaign": ["single-campaign"] * len(corpus)}
    )

    assessments = classify_geometry_boundaries(corpus, payload)
    transitions = build_geometry_transitions(corpus, index, payload)
    dropped = [assessment for assessment in assessments if not assessment.opens_range]

    print(f"geometry ranges: {len(transitions)} (previously {len(assessments) + 1})")
    for assessment in dropped:
        print(
            f"dropped {assessment.name}: {assessment.before_element_count}->"
            f"{assessment.after_element_count} elements; outline residual "
            f"{assessment.conductor_outline_residual_m:.9g} m; {assessment.reason}"
        )

    assert len(corpus) == 11_573
    assert len(assessments) == 10
    assert len(transitions) == 2
    assert [assessment.first_shot for assessment in dropped] == [
        12533,
        12561,
        12878,
        12905,
        12964,
        12990,
        13021,
        13049,
        13379,
    ]
    assert {
        (assessment.before_element_count, assessment.after_element_count)
        for assessment in dropped
    } == {(1004, 938), (938, 1004)}
    assert all(
        assessment.classification == DISCRETISATION_ONLY for assessment in dropped
    )
    assert max(assessment.conductor_outline_residual_m for assessment in dropped) == 0
    assert transitions[0].first_shot == min(corpus)
    assert transitions[-1].last_shot == max(corpus)
    assert all(
        left.last_shot + 1 == right.first_shot
        for left, right in zip(transitions, transitions[1:], strict=False)
    )
    assert all(
        transition_for_shot(transitions, shot).contains(shot)
        for shot in range(min(corpus), max(corpus) + 1)
    )
