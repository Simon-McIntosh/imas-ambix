"""Resolving the pinned machine description without being told where it is.

These tests set no environment, and that is the property under test rather than a
convenience.  Artifact-backed coverage guarded on hand-exported variables runs
only where somebody exported them, so it reports nothing about every other
machine -- a skip is not a pass.  Anything below that needs a description gets it
from the resolution path itself.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from imas_ambix.gs import artifact_resolution as resolution


@pytest.fixture(scope="module")
def resolved():
    """The description the package pins, resolved however this machine can."""
    return resolution.resolve_machine_description()


# --- the default path --------------------------------------------------------


def test_the_description_resolves_with_nothing_set(monkeypatch):
    """The whole point: no cache directory, no digest, no environment."""
    monkeypatch.delenv(resolution.CACHE_ENV, raising=False)
    monkeypatch.delenv(resolution.DIGEST_ENV, raising=False)

    resolved = resolution.resolve_machine_description()

    assert resolved.semantic_identity == resolution.PINNED_SEMANTIC_IDENTITY
    assert resolved.matches_pin
    assert resolved.route in {resolution.ROUTE_CACHED, resolution.ROUTE_AUTHORED}


def test_the_resolved_description_states_the_pinned_machine(resolved):
    """Identity is checked during resolution, not asserted about afterwards."""
    assert resolved.physical_digest == resolution.PINNED_PHYSICAL_DIGEST
    assert resolved.registry_digest == resolution.PINNED_REGISTRY_DIGEST


def test_the_second_resolution_reads_the_cache_the_first_one_left(resolved):
    """Authoring happens at most once per machine; the object is then found."""
    again = resolution.resolve_machine_description()

    assert again.route == resolution.ROUTE_CACHED
    assert again.digest == resolved.digest


def test_a_revision_is_found_by_what_it_says_not_by_its_file_hashes(resolved):
    found = resolution.find_revision(
        resolved.cache_directory, resolution.PINNED_SEMANTIC_IDENTITY
    )

    assert found == resolved.digest


def test_an_absent_revision_is_reported_as_absent_rather_than_guessed(resolved):
    absent = resolution.find_revision(resolved.cache_directory, "sha256:" + "0" * 64)

    assert absent is None


def test_a_cache_that_does_not_exist_is_not_an_error_to_search(tmp_path):
    assert resolution.find_revision(tmp_path / "nothing-here") is None


# --- why the pin is a semantic identity --------------------------------------


def test_re_authoring_reproduces_the_identity_the_package_pins(tmp_path, resolved):
    """The property the pin choice rests on, and the reason it is not a digest.

    A manifest digest covers the stored dictionary containers, whose bytes carry
    library write metadata that changes between writes -- so authoring the same
    description again need not reproduce it, and a machine with no cache could
    never satisfy a digest pin.  The semantic identity excludes the file table and
    is reproduced exactly, which is what makes the default resolution able to fall
    back on authoring instead of failing.
    """
    from nova.imas.mast_geometry import publish_refined_artifact

    republished = publish_refined_artifact(resolution.staging_cache_root())

    assert republished.manifest.semantic_identity() == resolved.semantic_identity
    assert republished.manifest.physical_digest == resolved.physical_digest

    stored = json.loads(
        (
            resolution.publish_object(republished, tmp_path) / "manifest.json"
        ).read_bytes()
    )
    assert stored["physical_digest"] == resolution.PINNED_PHYSICAL_DIGEST


# --- publishing where the atomic no-clobber rename is unavailable ------------


def test_an_object_publishes_into_a_cache_that_refuses_no_clobber_rename(
    tmp_path, resolved
):
    """The durable cache here is GPFS, where ``renameat2`` fails with EINVAL.

    ``publish_object`` therefore copies, fsyncs and renames rather than using the
    flag, and the result must still verify through nova's own reader -- an object
    is never trusted because this module wrote it.
    """
    published = resolution.publish_object(resolved.artifact, tmp_path)

    assert published.is_dir()
    mirrored = resolution.resolve_machine_description(tmp_path, resolved.digest)
    assert mirrored.semantic_identity == resolved.semantic_identity
    assert mirrored.route == resolution.ROUTE_ENVIRONMENT


def test_publishing_an_object_that_is_already_stored_leaves_it_untouched(
    tmp_path, resolved
):
    """The address is the hash of the manifest, so a second write can only harm."""
    first = resolution.publish_object(resolved.artifact, tmp_path)
    before = sorted((path.name, path.stat().st_mtime_ns) for path in first.iterdir())

    second = resolution.publish_object(resolved.artifact, tmp_path)

    assert second == first
    assert (
        sorted((path.name, path.stat().st_mtime_ns) for path in second.iterdir())
        == before
    )


def test_publishing_leaves_no_partial_directory_behind(tmp_path, resolved):
    """A torn copy under the object root would be inventoried as a stored object."""
    resolution.publish_object(resolved.artifact, tmp_path)

    entries = sorted(path.name for path in (tmp_path / "sha256").iterdir())
    assert entries == [Path(resolved.artifact.directory).name]


# --- the override ------------------------------------------------------------


def test_the_environment_names_a_description_instead_of_the_pinned_one(
    monkeypatch, resolved
):
    monkeypatch.setenv(resolution.CACHE_ENV, str(resolved.cache_directory))
    monkeypatch.setenv(resolution.DIGEST_ENV, resolved.digest)

    named = resolution.resolve_machine_description()

    assert named.route == resolution.ROUTE_ENVIRONMENT
    assert named.digest == resolved.digest


def test_a_named_description_records_which_revision_it_actually_read(
    resolved, tmp_path
):
    """A named artifact is held to the pinned MACHINE, not the pinned DESCRIPTION.

    Benching a second revision of one machine is what the override exists for, so
    it must not be refused for failing the pin.  What keeps that from being
    ambiguous is that the identity read and the identity pinned are both recorded,
    so a consumer can always tell whether it measured the default.
    """
    resolution.publish_object(resolved.artifact, tmp_path)

    provenance = resolution.resolve_machine_description(
        tmp_path, resolved.digest
    ).provenance()

    assert provenance["semantic_identity"] == resolved.semantic_identity
    assert provenance["pinned_semantic_identity"] == (
        resolution.PINNED_SEMANTIC_IDENTITY
    )
    assert provenance["matches_pinned_description"] is True
    assert provenance["resolution_route"] == resolution.ROUTE_ENVIRONMENT


def test_half_an_override_is_refused_rather_than_half_applied(monkeypatch):
    """A cache with no digest cannot be completed by falling back to the pin."""
    monkeypatch.setenv(resolution.CACHE_ENV, "/somewhere/cache")
    monkeypatch.delenv(resolution.DIGEST_ENV, raising=False)

    with pytest.raises(resolution.ArtifactResolutionError, match="needs both"):
        resolution.resolve_machine_description()


def test_a_named_artifact_that_is_not_this_machine_is_refused(tmp_path):
    with pytest.raises(resolution.ArtifactResolutionError, match="does not verify"):
        resolution.resolve_machine_description(tmp_path, "sha256:" + "0" * 64)


# --- where each cache root lives ---------------------------------------------


def test_the_staging_root_is_node_local_so_the_atomic_publish_is_available():
    """Authoring must land where ``renameat2(RENAME_NOREPLACE)`` works."""
    root = resolution.staging_cache_root()

    assert not root.is_relative_to(Path.home())


def test_the_environment_moves_the_durable_root(monkeypatch, tmp_path):
    monkeypatch.setenv(resolution.CACHE_ENV, str(tmp_path))

    assert resolution.durable_cache_root() == tmp_path


def test_the_durable_root_falls_back_to_the_user_cache(monkeypatch, tmp_path):
    monkeypatch.delenv(resolution.CACHE_ENV, raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    assert (
        resolution.durable_cache_root() == tmp_path / "imas-ambix" / "machine-artifact"
    )


def test_the_staging_root_follows_the_scratch_a_compute_node_hands_out(
    monkeypatch, tmp_path
):
    """A SLURM node's own scratch is reached through ``TMPDIR``, not ``/tmp``."""
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr("tempfile.tempdir", None)

    assert resolution.staging_cache_root().is_relative_to(tmp_path)
