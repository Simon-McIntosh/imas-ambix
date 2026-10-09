"""Find the published machine description a run reads, without being told where it is.

:class:`~imas_ambix.gs.artifact_geometry.MachineArtifactGeometryReader` addresses
an artifact by a cache directory and a manifest digest.  Both are facts about one
machine's filesystem, which no caller can know without being handed them from
outside -- so a reader addressed that way is reachable only where somebody has
already set the two up, and code guarded on them is code that does not run
anywhere else.  This module supplies both, so a caller names the MACHINE and not
a path.

Why the pin is a semantic identity and not a digest
---------------------------------------------------
The obvious pin is the manifest digest, and it does not work.  The artifact's
files are dictionary containers whose bytes carry library write metadata, so
authoring the same description twice publishes two different manifest digests
over byte-identical semantics.  Measured on this description: two authoring runs
produced ``sha256:b41c076e...`` and ``sha256:889cbc2a...``, with every file the
same size and every file hash different.  A digest pin is therefore not a
statement about a machine at all -- it names one materialization on one
filesystem, and cannot be re-satisfied anywhere else.

``MachineArtifactManifest.semantic_identity`` is the address that survives
re-authoring: it covers the dictionary pin, the physical and registry identity,
the shot extent and every field's provenance, and excludes the file table.  Both
authoring runs above returned the same semantic identity for it.  That is what
this module pins, and what it verifies after resolving, so a machine with no
cache at all reproduces the pinned description rather than failing.

The pin is a table keyed by machine
-----------------------------------
A machine states one pinned identity per physical configuration it has.  MAST
pins one row; JT-60SA pins one per operating phase, because OP1 and OP2 are two
physical identities of one device.  :data:`PINNED_DESCRIPTIONS` holds those rows,
each carrying the semantic identity that survives re-authoring, the physical and
registry digests the row's artifact must state, and the route that authors it.
Nothing here derives a row from the registry it guards: the registry is the live
object and the row is the frozen expectation recorded when a benchmark was
stamped, so a registry change fails the stamp loudly instead of silently
re-labelling it.

Where the artifact is looked for
--------------------------------
In order: the environment override, the durable cache root, the node-local
staging root, and finally re-authoring it.  Re-authoring costs about three
seconds, which is why the fallback is a resolution step rather than an error --
there is no state a caller must set up before the geometry is available.

Publishing where no-clobber rename exists
-----------------------------------------
:func:`nova.imas.machine_artifact.materialize_machine_artifact` publishes a
freshly-authored bundle with ``renameat2(RENAME_NOREPLACE)``, which makes the
publish atomic and refuses to overwrite a colliding object.  That call is not
available everywhere: on the GPFS filesystems holding ``/home`` and ``/work`` it
fails with ``EINVAL``, and nova correctly refuses to publish rather than risk a
non-atomic write.  Only node-local storage (``/tmp``, ``/run/user``) supports it.

So authoring always happens under :func:`staging_cache_root`, which is node-local
by construction, and the durable copy is made by :func:`publish_object`, which
does not need the flag: it copies into a sibling temporary directory, fsyncs the
files and the directory, and renames that into place.  A plain directory rename
cannot clobber a populated destination -- it fails with ``ENOTEMPTY`` -- so a
racing writer is detected rather than overwritten, and because the destination is
content-addressed, losing that race means the other writer stored the same bytes.
The copy is verified through nova's own reader afterwards, so a durable object is
never trusted because it was written here.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class PinRow:
    """One pinned physical identity of one machine.

    ``semantic_identity`` is the address that survives re-authoring, and is what
    resolution verifies after producing a description.  ``physical_digest`` and
    ``registry_digest`` are the machine and registry revisions the row's artifact
    must state, enforced during resolution so a cache holding a different machine
    fails rather than being silently read.  ``authoring_route`` takes a cache
    directory and returns the freshly authored, materialized artifact, which is
    how the row is reproduced where no cache holds it.
    """

    semantic_identity: str
    physical_digest: str
    registry_digest: str
    authoring_route: Callable[[Path], Any]


def author_mast_machine_artifact(cache_directory: Path) -> Any:
    """Author and materialize MAST's machine artifact through Nova's producer."""
    from nova.imas.mast_geometry import publish_refined_artifact  # noqa: PLC0415

    return publish_refined_artifact(cache_directory)


def author_jt60sa_machine_artifact(phase: str) -> Callable[[Path], Any]:
    """Return the route that authors one JT-60SA phase artifact.

    The producer authors both phases from the shared stores on each call, so the
    route returns the phase this row names rather than a second producer.
    """

    def _author(cache_directory: Path) -> Any:
        from imas_ambix.gs.description_artifact import (  # noqa: PLC0415
            author_jt60sa_machine_artifacts,
        )

        return author_jt60sa_machine_artifacts(cache_directory)[phase]

    return _author


#: Every machine's pinned physical identities, keyed by machine.  Moving a row is
#: how the package adopts a republished description; nothing else in the
#: resolution path names a revision.  MAST holds one row; JT-60SA holds one per
#: operating phase, ordered OP1 then OP2.
PINNED_DESCRIPTIONS: dict[str, tuple[PinRow, ...]] = {
    #: MAST's identity advances when the description's provenance advances: the
    #: semantic identity covers every field's evidence path, so re-authoring the
    #: wall as an indexed limiter-unit collection moves it while the physical and
    #: registry identities, which name the machine, stay put.
    "mast": (
        PinRow(
            semantic_identity=(
                "sha256:f20d64dbde63756150fecd5a41d34a0d7a055900877370aae40a594ca7d039d5"
            ),
            physical_digest="b55c5bb005a2cb67",
            registry_digest=(
                "2a26cc0a3a22e7fb8f42a53ee4c45e639290f0c5587e5f56405772b007f31bfd"
            ),
            authoring_route=author_mast_machine_artifact,
        ),
    ),
    "jt-60sa": (
        PinRow(
            semantic_identity=(
                "sha256:1ea4dac766f47ba01bb6f9e35d5ca74f47d816772006379076f466a51e58f372"
            ),
            physical_digest="f4ba9a35aab22485",
            registry_digest=(
                "058b377fda345c171706e94feb78a1e1a7694b3cf70a2b15cb0676adf16db742"
            ),
            authoring_route=author_jt60sa_machine_artifact("OP1"),
        ),
        PinRow(
            semantic_identity=(
                "sha256:b3c079a077b818f1ecd51ee608653338b8d53b1a565f432d4250e8dd25685e49"
            ),
            physical_digest="08807c77e7b35487",
            registry_digest=(
                "058b377fda345c171706e94feb78a1e1a7694b3cf70a2b15cb0676adf16db742"
            ),
            authoring_route=author_jt60sa_machine_artifact("OP2"),
        ),
    ),
}


def pinned_rows(machine: str) -> tuple[PinRow, ...]:
    """Return a machine's pinned rows, refusing a machine with none by name."""
    try:
        return PINNED_DESCRIPTIONS[machine]
    except KeyError:
        raise ArtifactResolutionError(
            f"no machine description is pinned for machine {machine!r}"
        ) from None


def pinned_row(machine: str, physical_digest: str | None = None) -> PinRow:
    """Return the pinned row a machine holds for ``physical_digest``.

    The digest may be omitted only when the machine's entry holds one row; a
    machine that pins several identities refuses the omission by name rather than
    defaulting to one of them, and a digest the machine's entry does not hold is
    refused the same way.
    """
    rows = pinned_rows(machine)
    if physical_digest is None:
        if len(rows) != 1:
            raise ArtifactResolutionError(
                f"machine {machine!r} pins {len(rows)} physical identities; name "
                "the physical digest of the shot's identity to choose one"
            )
        return rows[0]
    for row in rows:
        if row.physical_digest == physical_digest:
            return row
    raise ArtifactResolutionError(
        f"machine {machine!r} pins no physical identity {physical_digest!r}"
    )

#: Naming an artifact explicitly overrides the default resolution.  The pair is
#: kept because it is the contract every existing caller and run script uses; it
#: is now an override rather than a precondition.
CACHE_ENV = "AMBIX_MACHINE_ARTIFACT_CACHE"
DIGEST_ENV = "AMBIX_MACHINE_ARTIFACT_DIGEST"

#: How each route arrived at the artifact, recorded in provenance so a stamp
#: says whether it read a named cache, a found one, or one it authored.
ROUTE_ENVIRONMENT = "environment"
ROUTE_CACHED = "cached"
ROUTE_AUTHORED = "authored"

_OBJECT_SUBDIRECTORY = "sha256"
_MANIFEST_FILENAME = "manifest.json"
_PUBLISH_PREFIX = ".incoming-"


class ArtifactResolutionError(RuntimeError):
    """Raised when no artifact matching the pinned description can be produced."""


@dataclass(frozen=True)
class ResolvedArtifact:
    """One verified artifact, the address it was found at, and how it was found."""

    cache_directory: Path
    digest: str
    semantic_identity: str
    physical_digest: str
    registry_digest: str
    route: str
    artifact: Any
    pinned_semantic_identity: str

    @property
    def matches_pin(self) -> bool:
        """Return whether this is the description the row pins for the machine."""
        return self.semantic_identity == self.pinned_semantic_identity

    def provenance(self) -> dict[str, Any]:
        """Return the block a consumer records to say what it resolved and how."""
        return {
            "cache_directory": str(self.cache_directory),
            "artifact_digest": self.digest,
            "semantic_identity": self.semantic_identity,
            "physical_digest": self.physical_digest,
            "registry_digest": self.registry_digest,
            "resolution_route": self.route,
            "pinned_semantic_identity": self.pinned_semantic_identity,
            "matches_pinned_description": self.matches_pin,
        }


def durable_cache_root() -> Path:
    """Return the cache that survives a reboot and is shared between nodes.

    A home-directory cache on this platform is GPFS, which is exactly the
    filesystem that cannot take nova's atomic publish -- so this root is a place
    to READ from and to mirror into through :func:`publish_object`, never a place
    to author into directly.
    """
    override = os.environ.get(CACHE_ENV, "").strip()
    if override:
        return Path(override)
    base = os.environ.get("XDG_CACHE_HOME", "").strip() or (Path.home() / ".cache")
    return Path(base) / "imas-ambix" / "machine-artifact"


def staging_cache_root() -> Path:
    """Return the node-local cache an artifact may be authored into.

    ``tempfile.gettempdir`` honours ``TMPDIR``, which is how a compute node's own
    scratch is reached; both it and ``/tmp`` support the no-clobber rename nova's
    publish requires, which the durable root may not.
    """
    return Path(tempfile.gettempdir()) / "imas-ambix-machine-artifact"


def _object_root(cache_directory: Path | str) -> Path:
    return Path(cache_directory) / _OBJECT_SUBDIRECTORY


def _semantic_identity_of(manifest_path: Path) -> str | None:
    """Return a stored object's semantic identity, or ``None`` if unreadable.

    Deliberately does not verify the object's files: this runs over every
    candidate in a cache to find the one worth verifying, and full verification
    re-hashes every dictionary container.  The winner is then resolved through
    nova's own reader, which does verify.
    """
    from nova.imas.machine_artifact import MachineArtifactManifest  # noqa: PLC0415

    try:
        manifest = MachineArtifactManifest.from_bytes(manifest_path.read_bytes())
    except (OSError, ValueError):
        return None
    try:
        return manifest.semantic_identity()
    except ValueError:
        return None


def find_revision(
    cache_directory: Path | str,
    semantic_identity: str,
) -> str | None:
    """Return the manifest digest of a stored revision, or ``None``.

    Searches by semantic identity because the manifest digest is a property of
    one materialization: the same description authored on two machines is stored
    under two addresses, and only this identity connects them.
    """
    object_root = _object_root(cache_directory)
    if not object_root.is_dir():
        return None
    for entry in sorted(object_root.iterdir()):
        if not entry.is_dir() or entry.is_symlink():
            continue
        manifest_path = entry / _MANIFEST_FILENAME
        if not manifest_path.is_file():
            continue
        if _semantic_identity_of(manifest_path) == semantic_identity:
            return f"sha256:{entry.name}"
    return None


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _copy_durably(source: Path, destination: Path) -> None:
    """Copy one file and force it to disk before the directory rename names it."""
    shutil.copyfile(source, destination)
    descriptor = os.open(destination, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_object(artifact: Any, cache_directory: Path | str) -> Path:
    """Mirror a verified artifact into a cache whose filesystem may lack the flag.

    Returns the destination directory.  An object already present is left exactly
    as it is: the address is the hash of the manifest and every file is hashed
    against that manifest on read, so an existing object with this name already
    holds these bytes and rewriting it could only turn a good object into a torn
    one.
    """
    object_root = _object_root(cache_directory)
    destination = object_root / Path(artifact.directory).name
    if destination.is_dir():
        return destination
    object_root.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=_PUBLISH_PREFIX, dir=object_root),
    )
    try:
        for entry in sorted(Path(artifact.directory).iterdir()):
            if entry.is_file() and not entry.is_symlink():
                _copy_durably(entry, staging / entry.name)
        _fsync_directory(staging)
        try:
            os.rename(staging, destination)
        except OSError:
            # a concurrent writer published the same content-addressed object
            # first; its bytes hash to this name, so there is nothing to redo
            if not destination.is_dir():
                raise
            return destination
        _fsync_directory(object_root)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return destination


def _resolve_stored(
    row: PinRow,
    cache_directory: Path | str,
    digest: str,
    route: str,
    *,
    require_pin: bool,
) -> ResolvedArtifact:
    from nova.imas.machine_artifact import (  # noqa: PLC0415
        MachineArtifactError,
        resolve_machine_artifact,
    )

    try:
        artifact = resolve_machine_artifact(
            cache_directory,
            digest,
            expected_physical_digest=row.physical_digest,
            expected_registry_digest=row.registry_digest,
            allow_incomplete=True,
        )
    except MachineArtifactError as error:
        raise ArtifactResolutionError(
            f"artifact {digest} under {cache_directory} does not verify as the "
            f"pinned machine (physical {row.physical_digest}, registry "
            f"{row.registry_digest[:12]}): {error}"
        ) from error
    semantic_identity = artifact.manifest.semantic_identity()
    if require_pin and semantic_identity != row.semantic_identity:
        raise ArtifactResolutionError(
            f"artifact {digest} under {cache_directory} states description "
            f"{semantic_identity}, not the pinned {row.semantic_identity}"
        )
    return ResolvedArtifact(
        cache_directory=Path(cache_directory),
        digest=digest,
        semantic_identity=semantic_identity,
        physical_digest=artifact.manifest.physical_digest,
        registry_digest=artifact.manifest.registry_digest,
        route=route,
        artifact=artifact,
        pinned_semantic_identity=row.semantic_identity,
    )


def author_pinned_description(
    machine: str,
    physical_digest: str | None = None,
    durable_root: Path | str | None = None,
) -> ResolvedArtifact:
    """Author the pinned description locally and store it for the next caller.

    Authoring runs into :func:`staging_cache_root` because that is where nova's
    atomic publish works.  The result is then mirrored into the durable root so
    the next process on any node finds it without re-authoring; a durable root
    that cannot be written to is not an error, because the node-local object the
    caller was just handed is already verified and complete.
    """
    row = pinned_row(machine, physical_digest)
    from nova.imas.machine_artifact import MachineArtifactError  # noqa: PLC0415

    staging = staging_cache_root()
    try:
        published = row.authoring_route(staging)
    except (MachineArtifactError, OSError) as error:
        raise ArtifactResolutionError(
            f"cannot author the pinned description for machine {machine!r} "
            f"into {staging}: {error}"
        ) from error

    semantic_identity = published.manifest.semantic_identity()
    if semantic_identity != row.semantic_identity:
        raise ArtifactResolutionError(
            f"the installed producer authors machine {machine!r} description "
            f"{semantic_identity}, not the pinned {row.semantic_identity}; the pin "
            "and the producer revision have diverged and one of them must move"
        )

    durable = Path(durable_root) if durable_root is not None else durable_cache_root()
    try:
        publish_object(published, durable)
    except OSError:
        return _resolve_stored(
            row,
            staging,
            published.digest,
            ROUTE_AUTHORED,
            require_pin=True,
        )
    return _resolve_stored(
        row,
        durable,
        published.digest,
        ROUTE_AUTHORED,
        require_pin=True,
    )


def resolve_machine_description(
    machine: str,
    physical_digest: str | None = None,
    *,
    cache_directory: Path | str | None = None,
    digest: str | None = None,
) -> ResolvedArtifact:
    """Return the pinned machine description, producing it if no cache holds it.

    An explicitly named artifact -- by argument or by :data:`CACHE_ENV` and
    :data:`DIGEST_ENV` -- is read as named and is NOT held to the pinned
    description, only to the pinned machine.  That asymmetry is the point of the
    override: comparing two revisions of one machine is exactly what it exists
    for, and a revision that failed the pin could never be benched.  The identity
    it did read is recorded either way, so a run always states which description
    it measured.
    """
    row = pinned_row(machine, physical_digest)
    named_cache = str(
        cache_directory
        if cache_directory is not None
        else os.environ.get(CACHE_ENV, "")
    ).strip()
    named_digest = str(
        digest if digest is not None else os.environ.get(DIGEST_ENV, "")
    ).strip()
    if named_cache and named_digest:
        return _resolve_stored(
            row,
            named_cache,
            named_digest,
            ROUTE_ENVIRONMENT,
            require_pin=False,
        )
    if named_cache or named_digest:
        raise ArtifactResolutionError(
            "a named artifact needs both a cache directory and a digest "
            f"({CACHE_ENV} + {DIGEST_ENV}); leave both unset to resolve the "
            "pinned description"
        )

    for root in (durable_cache_root(), staging_cache_root()):
        found = find_revision(root, row.semantic_identity)
        if found is not None:
            return _resolve_stored(row, root, found, ROUTE_CACHED, require_pin=True)
    return author_pinned_description(machine, physical_digest)


__all__ = [
    "CACHE_ENV",
    "DIGEST_ENV",
    "PINNED_DESCRIPTIONS",
    "PinRow",
    "pinned_row",
    "pinned_rows",
    "ROUTE_AUTHORED",
    "ROUTE_CACHED",
    "ROUTE_ENVIRONMENT",
    "ArtifactResolutionError",
    "ResolvedArtifact",
    "author_pinned_description",
    "durable_cache_root",
    "find_revision",
    "publish_object",
    "resolve_machine_description",
    "staging_cache_root",
]
