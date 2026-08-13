"""Machine identity assertions over the local multi-campaign MAST corpus."""

from __future__ import annotations

import json
from itertools import islice

import pytest

from imas_ambix.data.description_identity import (
    description_field_changes,
    geometry_description_for_transition,
    machine_description_bytes,
)
from imas_ambix.data.geometry_transitions import (
    GeometryTransition,
    build_geometry_transitions,
    load_geometry_table_payload,
)
from imas_ambix.data.machine_map import load_packaged_machine_map
from imas_ambix.data.manifest import load_index
from imas_ambix.data.paths import LEVEL2_DIR, MANIFEST_DIR
from imas_ambix.data.transform_engine import transform_machine_description

CORPUS_MANIFEST = MANIFEST_DIR / "level2-all.json"
GEOMETRY_TABLE = MANIFEST_DIR / "gs_geometry_tables.json"
DETERMINISM_SHOTS = (11_766, 12_417, 12_533, 12_906, 30_471)


def _corpus_shots() -> tuple[int, ...]:
    payload = json.loads(CORPUS_MANIFEST.read_text())
    return tuple(int(shot) for shot in payload["shot_ids"])


def _declared_transitions(corpus: tuple[int, ...]) -> tuple[GeometryTransition, ...]:
    return build_geometry_transitions(
        corpus,
        load_index(),
        load_geometry_table_payload(GEOMETRY_TABLE),
    )


def _partition_samples(
    corpus: tuple[int, ...], transitions: tuple[GeometryTransition, ...]
) -> tuple[tuple[int, ...], ...]:
    samples = tuple(
        tuple(islice((shot for shot in corpus if transition.contains(shot)), 2))
        for transition in transitions
    )
    assert all(len(range_samples) == 2 for range_samples in samples)
    return samples


def _emit(shot: int):
    result = transform_machine_description(
        load_packaged_machine_map("mast"), shot, "zarr", LEVEL2_DIR
    )
    assert result.status == "emitted"
    return result


@pytest.mark.skipif(
    not CORPUS_MANIFEST.is_file() or not GEOMETRY_TABLE.is_file(),
    reason="local FAIR-MAST level-2 corpus manifests are not mounted",
)
def test_description_emission_is_deterministic_for_five_real_shots():
    equal_count = 0
    byte_counts: list[int] = []
    for shot in DETERMINISM_SHOTS:
        assert (LEVEL2_DIR / f"{shot}.zarr").is_dir()
        first = machine_description_bytes(_emit(shot))
        second = machine_description_bytes(_emit(shot))
        assert first == second
        equal_count += 1
        byte_counts.append(len(first))

    assert equal_count == len(DETERMINISM_SHOTS) == 5
    print(
        "DETERMINISM "
        f"sampled_shots={equal_count} byte_equal_pairs={equal_count} "
        f"description_bytes_min={min(byte_counts)} "
        f"description_bytes_max={max(byte_counts)}"
    )


@pytest.mark.skipif(
    not CORPUS_MANIFEST.is_file() or not GEOMETRY_TABLE.is_file(),
    reason="local FAIR-MAST level-2 corpus manifests are not mounted",
)
def test_description_versions_partition_the_real_corpus_consistently():
    catalog = load_packaged_machine_map("mast")
    corpus = _corpus_shots()
    maps = catalog.maps
    transitions = _declared_transitions(corpus)
    partition_samples = _partition_samples(corpus, transitions)

    assert [(item.first_shot, item.last_shot, item.transition) for item in maps] == [
        (item.first_shot, item.last_shot, item.name) for item in transitions
    ]
    assert len(maps) == len(transitions) == len(partition_samples)
    assert maps[0].first_shot == min(corpus)
    assert maps[-1].last_shot == max(corpus)
    assert all(
        left.last_shot + 1 == right.first_shot
        for left, right in zip(maps, maps[1:], strict=False)
    )
    assert all(
        sum(item.first_shot <= shot <= item.last_shot for item in maps) == 1
        for shot in corpus
    )

    sampled_shots = 0
    for machine_map, samples in zip(maps, partition_samples, strict=True):
        first = _emit(samples[0])
        assert first.machine_map == machine_map
        first_bytes = machine_description_bytes(first)
        del first
        second = _emit(samples[1])
        assert second.machine_map == machine_map
        second_bytes = machine_description_bytes(second)
        del second
        assert machine_map.transition is not None
        assert first_bytes == second_bytes
        sampled_shots += len(samples)

    assert sampled_shots == sum(len(samples) for samples in partition_samples)
    print(
        "PARTITION_CONSISTENCY "
        f"corpus_shots={len(corpus)} ranges={len(maps)} "
        f"interior_boundaries={len(maps) - 1} sampled_shots={sampled_shots}"
    )


@pytest.mark.skipif(
    not CORPUS_MANIFEST.is_file() or not GEOMETRY_TABLE.is_file(),
    reason="local FAIR-MAST level-2 corpus manifests are not mounted",
)
def test_every_declared_boundary_has_a_visible_description_change():
    catalog = load_packaged_machine_map("mast")
    corpus = _corpus_shots()
    transitions = _declared_transitions(corpus)
    geometry_payload = json.loads(GEOMETRY_TABLE.read_text())
    changed_counts: list[int] = []
    map_boundaries = tuple(zip(catalog.maps, catalog.maps[1:], strict=False))
    transition_boundaries = tuple(zip(transitions, transitions[1:], strict=False))

    assert [
        (item.first_shot, item.last_shot, item.transition) for item in catalog.maps
    ] == [(item.first_shot, item.last_shot, item.name) for item in transitions]
    assert len(map_boundaries) == len(transition_boundaries)

    # Shot 12417 is the only boundary whose element count stays 1004 -> 1004
    # while its outline moves 0.0217 m, so it survives the discretisation filter
    # that removes the 1004 -> 938 alternations.
    for before_map, after_map in map_boundaries:
        before_shot = max(shot for shot in corpus if shot <= before_map.last_shot)
        after_shot = min(shot for shot in corpus if shot >= after_map.first_shot)
        emitted_before_map = _emit(before_shot).machine_map
        emitted_after_map = _emit(after_shot).machine_map
        assert emitted_before_map == before_map
        assert emitted_after_map == after_map

        before_fields = geometry_description_for_transition(
            geometry_payload, emitted_before_map.transition
        )
        after_fields = geometry_description_for_transition(
            geometry_payload, emitted_after_map.transition
        )
        changes = description_field_changes(before_fields, after_fields)
        assert changes
        changed_counts.append(len(changes))
        print(
            "VISIBLE_CHANGE "
            f"boundary={after_map.first_shot} before_shot={before_shot} "
            f"after_shot={after_shot} changed_field_count={len(changes)} "
            f"fields={','.join(changes)}"
        )

    assert len(changed_counts) == len(transition_boundaries)
    assert all(changed_counts)
    print(
        "VISIBLE_CHANGE_SUMMARY "
        f"boundaries={len(changed_counts)} changed_field_counts={changed_counts}"
    )
