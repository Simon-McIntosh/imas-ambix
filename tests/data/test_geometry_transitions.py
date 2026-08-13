"""Geometry transitions select exactly one machine setup for every shot."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from imas_ambix.data.geometry_transitions import (
    build_geometry_transitions,
    transition_for_shot,
)
from imas_ambix.data.manifest import build_manifest


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
