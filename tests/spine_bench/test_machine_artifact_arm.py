"""The artifact arm: the geometry-source seam, and the table it reads.

The benchmark's sensor-space misfit is a forward-model check on the machine
geometry, so which description supplied that geometry decides what the number
means.  These tests cover the seam that makes the choice explicit -- a run states
its source, a stamp records it, and the machine is chosen by physical identity
rather than by the address a description was found at -- and that the arm's table
is the committed reader's own output rather than a parallel construction of it.
The description is resolved from the pinned identity, so none of this needs an
artifact to be named from outside.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from imas_ambix.gs import artifact_resolution as resolution
from imas_ambix.spine_bench import machine_artifact_arm as arm
from imas_ambix.spine_bench.runner import CampaignGeometrySource, write_yaml
from imas_ambix.spine_bench.schema import EnvInfo, MachineInfo, SpineBenchmarkStamp
from imas_ambix.spine_bench.shots import FROZEN_SHOTSET, SHOTSET_VERSION

RESULTS = Path(__file__).parents[2] / "imas_ambix" / "spine_bench" / "results"


# --- the seam ----------------------------------------------------------------


def test_a_stamp_that_does_not_say_otherwise_read_the_campaign_arrays():
    """The default names the historical source, so old stamps stay honest.

    Every stamp written before the field existed measured the campaign arrays, so
    the default has to be that source and not an empty string -- a reader must not
    have to treat 'unstated' as a third possibility.
    """
    banked = yaml.safe_load(
        (
            RESULTS / "physics-spine-v0-mast-heldout-6-08ae0dee74-98dci4-clu-3141.yaml"
        ).read_text()
    )
    assert "geometry_source" not in banked

    stamp = SpineBenchmarkStamp.model_validate(banked)
    assert stamp.geometry_source == CampaignGeometrySource.label == "efm-campaign"
    assert stamp.geometry_revision == ""


def _stamp(**overrides) -> SpineBenchmarkStamp:
    fields = dict(
        shotset_version=SHOTSET_VERSION,
        created_utc="2026-08-05T00:00:00+00:00",
        machine=MachineInfo(hostname="host-1.example", platform="linux"),
        env=EnvInfo(
            python_version="3.14.0", git_commit="0123456789abcdef", git_dirty=False
        ),
    )
    fields.update(overrides)
    return SpineBenchmarkStamp(**fields)


def test_the_campaign_arm_keeps_the_historical_filename(tmp_path):
    """The names the parity module cites as its references must not move."""
    path = write_yaml(_stamp(), tmp_path)

    assert path.name == f"physics-spine-{SHOTSET_VERSION}-0123456789-host-1.yaml"


def test_two_geometry_sources_at_one_commit_do_not_overwrite_each_other(tmp_path):
    campaign = write_yaml(_stamp(), tmp_path)
    artifact = write_yaml(
        _stamp(
            geometry_source=arm.ARTIFACT_SOURCE_LABEL, geometry_revision="sha256:abc"
        ),
        tmp_path,
    )

    assert campaign != artifact
    assert arm.ARTIFACT_SOURCE_LABEL in artifact.name


def test_two_revisions_of_one_description_do_not_overwrite_each_other(tmp_path):
    """Two republications can state the same conductors, so the signature cannot
    separate their stamps and the revision has to."""
    first = write_yaml(
        _stamp(
            geometry_source=arm.ARTIFACT_SOURCE_LABEL,
            geometry_revision="sha256:c20bc7e157fa117318883826373f41ea03e8539011b16c",
        ),
        tmp_path,
    )
    second = write_yaml(
        _stamp(
            geometry_source=arm.ARTIFACT_SOURCE_LABEL,
            geometry_revision="sha256:3aba565a2e407a621c38ec1abc0d458752bf5be85a2",
        ),
        tmp_path,
    )

    assert first != second
    assert "c20bc7e157fa" in first.name
    assert "3aba565a2e40" in second.name


def test_an_unnamed_artifact_still_names_a_machine(monkeypatch):
    """A source built from nothing measures the description this package pins.

    The objection defaulting has to answer is that a run might quietly measure a
    different machine than the caller meant.  What answers it is that the default
    is not a filesystem path but a pinned identity, verified after resolution: a
    source that was told nothing is still pinned to one machine, and says which.
    """
    monkeypatch.delenv(arm.CACHE_ENV, raising=False)
    monkeypatch.delenv(arm.DIGEST_ENV, raising=False)

    source = arm.resolve_geometry_source()

    assert source.cache_directory is None
    assert source.digest is None
    assert source.expected_physical_digest == arm.PINNED_PHYSICAL_DIGEST
    assert source.expected_registry_digest == arm.PINNED_REGISTRY_DIGEST


def test_the_source_reads_the_artifact_named_in_the_environment(monkeypatch):
    """Naming one explicitly overrides the pin, which is how a second revision of
    the same machine is benched against it."""
    monkeypatch.setenv(arm.CACHE_ENV, "/somewhere/cache")
    monkeypatch.setenv(arm.DIGEST_ENV, "sha256:feed")

    source = arm.resolve_geometry_source(
        os.environ[arm.CACHE_ENV], os.environ[arm.DIGEST_ENV]
    )

    assert source.cache_directory == "/somewhere/cache"
    assert source.digest == "sha256:feed"
    assert source.label == "machine-artifact"
    assert source.expected_physical_digest == arm.PINNED_PHYSICAL_DIGEST
    assert source.expected_registry_digest == arm.PINNED_REGISTRY_DIGEST


def test_the_channel_shots_span_the_whole_frozen_set():
    """A one-shot channel scan would make the sensor set an artifact of that
    shot's own acquisition gaps rather than of the geometry."""
    source = arm.resolve_geometry_source()

    assert source.channel_shots == tuple(int(s.shot_id) for s in FROZEN_SHOTSET)
    assert source.evidence_shot == FROZEN_SHOTSET[0].shot_id


# --- integration against the pinned published artifact -----------------------


@pytest.fixture(scope="module")
def source() -> arm.MachineArtifactGeometrySource:
    built = arm.resolve_geometry_source()
    built.build()
    return built


def test_the_arm_reads_the_table_the_committed_reader_produces(source):
    """The arm must not be a second construction of the geometry.

    Built directly from the reader with the arm's own arguments, the table has to
    come out identical -- otherwise the benchmarked machine is whatever this
    module assembles rather than what the reader publishes, and the artifact-backed
    geometry tests would be guarding something the benchmark does not use.
    """
    from imas_ambix.data.description_reader import read_acquisition_channels
    from imas_ambix.gs.artifact_geometry import MachineArtifactGeometryReader

    described = resolution.resolve_machine_description()
    shots = [int(s.shot_id) for s in FROZEN_SHOTSET]
    sensor_acquisition = read_acquisition_channels(shots)
    current_acquisition = read_acquisition_channels((shots[0],))
    direct = MachineArtifactGeometryReader(
        cache_directory=described.cache_directory,
        digest=described.digest,
        shot=shots[0],
        amb_channels=sensor_acquisition.sensors,
        amc_current_channels=current_acquisition.currents,
        expected_physical_digest=arm.PINNED_PHYSICAL_DIGEST,
        expected_registry_digest=arm.PINNED_REGISTRY_DIGEST,
    ).read()
    table = source.build()

    assert table.signature.key == direct.signature.key
    assert [f.r for f in table.pf_filaments] == [f.r for f in direct.pf_filaments]
    assert [f.z for f in table.pf_filaments] == [f.z for f in direct.pf_filaments]
    assert [f.xmult for f in table.pf_filaments] == [
        f.xmult for f in direct.pf_filaments
    ]
    assert list(table.limiter_r) == list(direct.limiter_r)
    assert len(table.sensor_map) == len(direct.sensor_map)
    assert {d.channel for d in table.circuit_drives} == {
        d.channel for d in direct.circuit_drives
    }


def test_the_machine_is_pinned_so_a_swapped_cache_cannot_be_benched(source):
    provenance = source.provenance()

    assert provenance["physical_digest"] == arm.PINNED_PHYSICAL_DIGEST
    assert provenance["registry_digest"] == arm.PINNED_REGISTRY_DIGEST
    assert provenance["source"] == arm.ARTIFACT_SOURCE_LABEL
    assert provenance["artifact_digest"] == (
        resolution.resolve_machine_description().digest
    )


def test_the_machine_was_chosen_by_identity_rather_than_by_address(source):
    """What the stamp has to record for the selection to be auditable.

    The address a description was found at says nothing about which machine it
    describes -- that is the whole reason selection moved onto the physical
    digest.  So the stamp carries the shot that selected, the digest it resolved
    to, and the registry's evidence for it, and a reader can check the machine
    without knowing anything about the cache.
    """
    provenance = source.provenance()

    assert provenance["selected_by"] == "physical"
    assert provenance["selected_shot"] == FROZEN_SHOTSET[0].shot_id
    assert provenance["registry_evidence"] == "observed"
    assert provenance["matches_pinned_description"] is True
    assert provenance["semantic_identity"] == resolution.PINNED_SEMANTIC_IDENTITY


def test_the_compute_key_stays_the_setup_signature_the_geometry_determines(source):
    """The dual read, asserted where a cutover would break it.

    Selection moved to the physical digest; every cached matrix downstream is a
    function of the discretization and must stay keyed by the signature, because
    two representations of one machine carry different filament counts and a
    digest-keyed cache would hand a caller the wrong filament set.
    """
    provenance = source.provenance()
    table = source.build()

    assert provenance["table_signature"] == table.signature.key
    assert provenance["table_signature"] != provenance["physical_digest"]


def test_the_revision_is_the_identity_that_moves_on_republication(source):
    """The semantic identity, not the physical digest: a republication of the same
    conductors leaves the physical digest and the signature untouched."""
    provenance = source.provenance()

    assert source.revision == provenance["semantic_identity"]
    assert source.revision.startswith("sha256:")
    assert source.revision != provenance["physical_digest"]


def test_the_signature_alone_cannot_say_which_revision_was_measured(source):
    """Why the stamp records the revision and keys its filename on it.

    The setup signature hashes conductor and sensor POSITIONS and the limiter, not
    turn counts.  A republication that restates a turn count therefore predicts
    different fields -- it moves the vacuum term and so the sensor-space misfit --
    under an unchanged signature.  Pinned as a property of the signature rather
    than by naming two revisions, so it holds for whichever one is on hand: a
    signature that agreed with the revision would make this test's subject moot.
    """
    table = source.build()
    provenance = source.provenance()

    assert provenance["table_signature"] == table.signature.key
    assert source.revision.split(":")[-1] not in table.signature.key
    turns = [f.turns for f in table.pf_filaments if f.turns == f.turns]
    assert turns and not any(str(round(t, 4)) in table.signature.key for t in turns)


def test_the_description_stands_for_the_machine_on_every_frozen_shot(source):
    """One table for every shot -- a machine description is a statement about the
    device, so unlike the campaign arrays it has no per-campaign geometry to
    resolve.  The evidence shot it reports is fixed rather than being whichever
    shot happened to be solved first."""
    tables = {id(source.table_for(int(s.shot_id))) for s in FROZEN_SHOTSET}

    assert len(tables) == 1
    assert source.provenance()["evidence_shot"] == FROZEN_SHOTSET[0].shot_id


def test_the_drive_map_is_read_from_the_description_not_a_position_match(source):
    """Every driven column comes from a stated weight, which is what lets the
    campaign's channel names address the artifact's conductors without the
    centroid-radius match the efm path is forced into."""
    table = source.build()

    assert len(table.circuit_drives) == 21
    assert all(drive.evidence for drive in table.circuit_drives)
    assert all(
        drive.ampere_turns_per_ampere == drive.ampere_turns_per_ampere
        for drive in table.circuit_drives
    )
