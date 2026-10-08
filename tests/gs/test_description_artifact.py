"""Tests for the JT-60SA machine-description artifact producer.

The producer authors, verifies and caches one artifact per operating phase from
the private JT-60SA description stores.  These tests read those stores read-only
through imas and check the identity the producer derives rather than accepts:
two authorings of the same store carry one physical digest (the content, not the
container bytes), the two phases carry different physical digests but one shared
registry digest over disjoint shot ranges, a moved conductor moves one phase's
digest, and each authored directory resolves and reopens through imas.
"""

from __future__ import annotations

import shutil

import pytest

from imas_ambix.gs import description_artifact as da

imas = pytest.importorskip("imas")

PHASES = da.PHASES


@pytest.fixture(scope="module")
def baseline(tmp_path_factory) -> dict:
    """Author both phase artifacts once from the packaged stores."""
    cache = tmp_path_factory.mktemp("jt60sa-artifact-cache")
    return da.author_jt60sa_machine_artifacts(cache)


def test_authoring_twice_gives_same_semantic_identity_and_physical_digest(
    baseline, tmp_path
) -> None:
    """Two authorings of one store describe one machine with one identity."""
    second = da.author_jt60sa_machine_artifacts(tmp_path / "second-cache")
    for phase in PHASES:
        first_manifest = baseline[phase].manifest
        second_manifest = second[phase].manifest
        assert first_manifest.physical_digest == second_manifest.physical_digest
        assert first_manifest.semantic_identity() == second_manifest.semantic_identity()


def test_phases_differ_in_physical_but_share_registry_on_disjoint_ranges(
    baseline,
) -> None:
    """OP1 and OP2 are two physical identities of one registry."""
    op1 = baseline["OP1"].manifest
    op2 = baseline["OP2"].manifest
    assert op1.physical_digest != op2.physical_digest
    assert op1.registry_digest == op2.registry_digest

    op1_shots = {
        shot for row in op1.shot_ranges for shot in (row.first_shot, row.last_shot)
    }
    op2_shots = {
        shot for row in op2.shot_ranges for shot in (row.first_shot, row.last_shot)
    }
    op1_span = (min(op1_shots), max(op1_shots))
    op2_span = (min(op2_shots), max(op2_shots))
    assert op1_span[1] < op2_span[0], (op1_span, op2_span)


def test_moving_one_pf_coil_changes_only_that_phase_physical_digest(
    baseline, tmp_path
) -> None:
    """The physical digest follows the geometry content, not the store layout."""
    moved_root = tmp_path / "stores"
    for phase in PHASES:
        shutil.copytree(da.DEFAULT_DESCRIPTION_ROOT / phase, moved_root / phase)

    store = moved_root / "OP1" / "pf_active.nc"
    with imas.DBEntry(str(store), "r", dd_version=da.DD_VERSION) as entry:
        pf_active = entry.get("pf_active", autoconvert=False)
    element = pf_active.coil[0].element[0]
    element.geometry.rectangle.r = float(element.geometry.rectangle.r) + 0.05
    with imas.DBEntry(str(store), "w", dd_version=da.DD_VERSION) as entry:
        entry.put(pf_active)

    moved = da.author_jt60sa_machine_artifacts(
        tmp_path / "moved-cache", description_root=moved_root
    )
    assert (
        moved["OP1"].manifest.physical_digest
        != baseline["OP1"].manifest.physical_digest
    )
    assert (
        moved["OP2"].manifest.physical_digest
        == baseline["OP2"].manifest.physical_digest
    )


def test_each_artifact_resolves_with_its_own_expected_digests(baseline) -> None:
    """Resolution enforces each phase's own physical and shared registry digest."""
    from nova.imas.machine_artifact import resolve_machine_artifact

    for phase in PHASES:
        manifest = baseline[phase].manifest
        verified = resolve_machine_artifact(
            baseline[phase].directory.parent.parent,
            baseline[phase].digest,
            expected_physical_digest=manifest.physical_digest,
            expected_registry_digest=manifest.registry_digest,
            allow_incomplete=not manifest.complete,
        )
        assert verified.manifest.physical_digest == manifest.physical_digest
        assert verified.manifest.registry_digest == manifest.registry_digest


def test_artifact_directory_reopens_through_imas_hdf5(baseline) -> None:
    """Every artifact is a valid IMAS HDF5 entry at its manifest's pin."""
    for phase in PHASES:
        directory = baseline[phase].directory
        dd_version = baseline[phase].manifest.dd_version
        with imas.DBEntry(
            f"imas:hdf5?path={directory}", "r", dd_version=dd_version
        ) as entry:
            for ids_name in ("pf_active", "pf_passive", "wall", "magnetics"):
                ids = entry.get(ids_name, autoconvert=False)
                assert ids is not None, (phase, ids_name)


def test_artifact_members_are_the_five_ids_and_master_only(baseline) -> None:
    """Each artifact packs master.h5 plus the five IDS files, nothing else."""
    expected = {"master.h5"} | {f"{name}.h5" for name in da.IDS_NAMES}
    for phase in PHASES:
        names = {artifact_file.name for artifact_file in baseline[phase].manifest.files}
        assert names == expected, (phase, names)
        assert "receipt.json" not in names
        assert not any(name.endswith(".nc") for name in names)


def test_each_artifact_is_incomplete_and_names_both_gaps(baseline) -> None:
    """No field evidence and no channel drive are authored, so both are named."""
    for phase in PHASES:
        manifest = baseline[phase].manifest
        assert manifest.complete is False, phase
        gaps = manifest.unresolved_gaps
        assert da.NO_FIELD_EVIDENCE_GAP in gaps, (phase, gaps)
        assert any("channel drive map" in gap for gap in gaps), (phase, gaps)


def test_op1_shot_evidence_is_observed_and_op2_inherited(baseline) -> None:
    """OP1's range was checked against pulses; OP2's identity is declared only."""
    op1 = {row.evidence for row in baseline["OP1"].manifest.shot_ranges}
    op2 = {row.evidence for row in baseline["OP2"].manifest.shot_ranges}
    assert op1 == {"observed"}, op1
    assert op2 == {"inherited"}, op2
