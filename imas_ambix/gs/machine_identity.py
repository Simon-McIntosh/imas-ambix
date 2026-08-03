"""Select a machine configuration by physical identity rather than discretization.

:class:`~imas_ambix.gs.geometry.SetupSignature` fingerprints a campaign by four
discretization counts plus a hash of the rounded sensor and filament positions.
That is an excellent *representation* key -- it is byte-stable, it is sensitive to
sub-centimetre drift, and every cache and checkpoint on disk resolves through it
-- but it is the wrong thing to call an identity.  Mesh density, filament
multiplicity, wall-element count and source ordering all change it while the
machine stands still, so a re-subdivided reconstruction looks like a new device.

The physical identity is the machine and its diagnostic pose.  Nova publishes it:
each physical configuration is content-addressed by a physical digest over the
component geometry, and the registry carries a
``representation_aliases`` table mapping historical setup digests onto the
configuration they actually describe.  For MAST all three recorded
representation digests -- including the frozen benchmark's
``1cb6f2ee742c4ee4`` -- alias onto the single configuration
``76cf833561e602a7``, which is the identity rule working as intended: three
discretizations, one machine.

Selecting on physical identity therefore MERGES campaign groups; it never renames
one.  Anything keyed by representation stays exactly where it is, which is what
makes the migration free.

What physical identity is NOT
-----------------------------
It is not a compute-cache address.  Every cached object in the equilibrium path
-- the grid, the Delta-star factorisation, the Green's and interaction matrices,
the flattened per-channel geometry features -- is a function of the
discretization, so two representations of one machine require two different
cached objects.  The patch-scoping cache shows this concretely: it holds matrices
under all three aliased digests, and two of them carry a different filament count
(``fc938`` against ``fc1004``) at the same resolution and differ in size by two
orders of magnitude.  All three name one machine.  Keying that cache on the
physical digest would therefore collapse three distinct matrices onto one address
and hand a caller the wrong filament set, which is why
:meth:`EquilibriumGrid.from_table` keys its build cache on the setup signature
and says so.

The distinction is the whole point.  Identity governs SELECTION and PROVENANCE:
which machine and diagnostic pose a shot resolves to, what evidence backs it,
whether its electrical semantics are sourced.  Discretization governs
COMPUTATION.  Conflating them in either direction is a defect -- treating a
re-subdivision as a new machine, or treating one machine's two subdivisions as
interchangeable numerics.

Migration story per cache family
--------------------------------
Nothing is rewritten and nothing is orphaned, because the representation key
remains the address of everything already written:

``imas_ambix/latent/artifacts/patch_scoping/g_pg_{key}_{nr}x{nz}.npz``
    Keyed by :attr:`SetupSignature.key` and left that way.  The matrix it holds is
    a function of the grid and the limiter outline -- discretization -- so the
    representation key is the *correct* address for it, not a legacy one.  These
    files are read unchanged.

``imas_ambix/gs/artifacts/*.json`` (``signature_key``)
    Keeps its ``signature_key`` field as written.  Consumers that want physical
    identity resolve it forward through :func:`identity_for_representation`, so
    old artifacts gain physical identity on read without being rewritten.

``imas_ambix/spine_bench/results/*.yaml`` (``campaign_signature``)
    Historical stamps keep their representation signature; the field is
    provenance, not a lookup key, and rewriting committed stamps would break the
    comparability the benchmark exists to provide.  A reader resolves a stamp's
    signature to a physical digest through this module when it needs to compare
    across an identity change.

Trained checkpoints
    Resolve through the representation key recorded at training time.  Because
    every MAST representation aliases onto one configuration, a checkpoint's
    campaign grouping is unchanged under the physical key -- the merge is a no-op
    for a single-configuration corpus.  A future corpus spanning two real
    configurations must re-derive its grounding, which the physical digest makes
    detectable instead of silent.

The representation key stays byte-stable throughout: this module reads it and
never computes or alters it.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

#: Group by :attr:`SetupSignature.key` -- the historical behaviour and the
#: default, so every existing caller and every file on disk is unaffected.
IDENTITY_REPRESENTATION = "representation"

#: Group by the Nova registry physical digest -- the machine-and-pose identity.
IDENTITY_PHYSICAL = "physical"

IDENTITY_MODES = (IDENTITY_REPRESENTATION, IDENTITY_PHYSICAL)


class MachineIdentityError(LookupError):
    """Raised when a machine identity cannot be resolved from the registry."""


@dataclass(frozen=True)
class MachineIdentity:
    """One machine configuration, carrying both identities and its evidence.

    ``physical_digest`` is identity.  ``representation_key`` is provenance and
    the on-disk cache address, retained so a resolved identity can always be
    traced back to the discretization that produced it.
    """

    physical_digest: str
    representation_key: str
    evidence: str
    authoring_gaps: tuple[str, ...]
    registry_digest: str
    dd_version: str

    @property
    def is_operator_ready(self) -> bool:
        """Return whether the configuration has no unsourced authoring gaps.

        False for the current provisional MAST revision, which carries six
        recorded gaps.  A forward operator built from it is admissible for
        geometry work and not yet quantitatively complete, so a caller that
        needs sourced electrical semantics must check this rather than assume it.
        """
        return not self.authoring_gaps

    def key(self, mode: str = IDENTITY_REPRESENTATION) -> str:
        """Return the grouping key for ``mode``."""
        if mode == IDENTITY_REPRESENTATION:
            return self.representation_key
        if mode == IDENTITY_PHYSICAL:
            return self.physical_digest
        raise ValueError(f"unknown identity mode {mode!r}; expected {IDENTITY_MODES}")


@lru_cache(maxsize=1)
def default_registry() -> Any:
    """Return the packaged Nova MAST geometry registry, loaded once.

    Imported lazily and cached because loading validates the whole payload
    (re-hashing every configuration and the registry digest), which is wasted
    work on the many code paths that never ask about identity.
    """
    from nova.catalog.mast_geometry import MachineGeometryRegistry  # noqa: PLC0415

    return MachineGeometryRegistry.default()


def _identity(
    representation_key: str,
    physical_digest: str,
    evidence: str,
    authoring_gaps: tuple[str, ...],
    registry: Any,
) -> MachineIdentity:
    return MachineIdentity(
        physical_digest=physical_digest,
        representation_key=representation_key,
        evidence=evidence,
        authoring_gaps=authoring_gaps,
        registry_digest=registry.registry_digest,
        dd_version=registry.dd_version,
    )


def identity_for_representation(
    signature, *, registry: Any | None = None
) -> MachineIdentity:
    """Resolve a :class:`SetupSignature` (or its digest) to a physical identity.

    Accepts the signature object or a bare digest string.  The digest -- not the
    full key -- is what the registry aliases, so the discretization counts in the
    key are correctly ignored here.

    Evidence is reported as ``"aliased"``: the alias table states which machine
    the representation describes but says nothing about the source quality of any
    particular shot, which only :func:`identity_for_shot` can answer.
    """
    registry = registry or default_registry()
    digest = getattr(signature, "digest", signature)
    representation_key = getattr(signature, "key", digest)
    try:
        configuration = registry.resolve_representation(digest)
    except KeyError as error:
        raise MachineIdentityError(
            f"setup representation {digest!r} is not in the Nova registry alias "
            f"table (registry {registry.registry_digest[:12]}); a new campaign "
            "must be aliased in Nova before it can select a physical identity"
        ) from error
    return _identity(
        representation_key,
        configuration.physical_digest,
        "aliased",
        tuple(configuration.authoring_gaps),
        registry,
    )


def identity_for_shot(
    shot_id: int, representation_key: str = "", *, registry: Any | None = None
) -> MachineIdentity:
    """Resolve a shot to its physical identity and per-shot evidence state.

    This is the stronger route: it reports the registry's evidence state for the
    shot (``observed``, ``inherited``, or ``missing`` where source coverage is
    incomplete), which the alias table cannot supply.
    """
    registry = registry or default_registry()
    try:
        selection = registry.select(int(shot_id))
    except KeyError as error:
        raise MachineIdentityError(
            f"shot {shot_id} lies outside the Nova registry shot ranges "
            f"(registry {registry.registry_digest[:12]})"
        ) from error
    return _identity(
        representation_key,
        selection.configuration.physical_digest,
        str(getattr(selection.evidence, "value", selection.evidence)),
        tuple(selection.configuration.authoring_gaps),
        registry,
    )


def identity_for_table(table, *, registry: Any | None = None) -> MachineIdentity:
    """Resolve a :class:`GeometryTable` to a physical identity.

    Prefers the per-shot route when the table records the shots it was built
    from, because that route carries the evidence state; falls back to the alias
    table otherwise.  A shot the registry does not cover falls back rather than
    failing, so a table remains usable when only its representation is known.
    """
    registry = registry or default_registry()
    signature = table.signature
    for shot in getattr(table, "shots", ()) or ():
        try:
            identity = identity_for_shot(int(shot), signature.key, registry=registry)
        except MachineIdentityError, TypeError, ValueError:
            continue
        return identity
    return identity_for_representation(signature, registry=registry)


def same_machine(left, right, *, registry: Any | None = None) -> bool:
    """Return whether two geometry tables describe one physical configuration.

    The question ``signature.key`` equality cannot answer.  Two tables that
    differ only by subdivision compare equal here and unequal by signature, which
    is the identity rule stated as a predicate.
    """
    registry = registry or default_registry()
    return (
        identity_for_table(left, registry=registry).physical_digest
        == identity_for_table(right, registry=registry).physical_digest
    )


def describe_identity(table, *, registry: Any | None = None) -> dict[str, Any]:
    """Return a provenance block recording both identities and the evidence.

    The shape consumers embed in an artifact or a stamp so a later reader can
    tell which machine produced it, which discretization it was computed on, and
    whether the electrical semantics behind it were sourced.
    """
    identity = identity_for_table(table, registry=registry)
    return {
        "physical_digest": identity.physical_digest,
        "representation_key": identity.representation_key,
        "evidence": identity.evidence,
        "registry_digest": identity.registry_digest,
        "dd_version": identity.dd_version,
        "operator_ready": identity.is_operator_ready,
        "authoring_gaps": list(identity.authoring_gaps),
    }
