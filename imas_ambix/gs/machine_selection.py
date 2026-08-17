"""Select a shot's machine geometry by which machine it is, not by how it was meshed.

The existing selection route is
:func:`~imas_ambix.gs.geometry.build_table_for_shot`: a shot names its own static
``efm`` arrays, those arrays are reduced to a
:class:`~imas_ambix.gs.geometry.SetupSignature`, and that signature is what every
downstream consumer groups by.  The signature is a discretization fingerprint, so
that route answers "which mesh did this shot arrive on" and treats the answer as
if it were "which machine is this".  Re-subdividing the same device produces a
different signature and therefore, to every consumer, a different machine.

This route asks the other question.  A shot resolves through the Nova registry to
a PHYSICAL identity -- the machine and its diagnostic pose -- and that identity
selects the published description to read.  Two shots that differ only in how
their campaign discretized the device select the same description here, which is
the identity rule doing its job.

The dual read
-------------
Physical identity governs SELECTION and PROVENANCE; it is not a cache address.
The table this returns still carries the ``SetupSignature`` its geometry
determines, unchanged and uncomputed by this module, because every cached object
downstream -- the grid, the Delta-star factorisation, the Green's and interaction
matrices -- is a function of the discretization and must stay keyed by it.  Two
representations of one machine legitimately require two cached matrices, so
keying those caches on the physical digest would hand a caller the wrong filament
set.  Selection moves to the physical digest; computation stays on the signature.

What is verified before a table is returned
-------------------------------------------
The registry's physical digest for the shot and the description's own physical
digest must agree.  They are two independent statements -- one from the shot-range
table, one content-addressed over the description's component geometry -- and a
disagreement means the shot has been mapped onto a machine the description does
not describe.  That is refused rather than reconciled, because the whole value of
selecting on identity is lost if the selection can silently miss.

Acquisition addressing
----------------------
The reviewed map declares stable sensor and current addresses beside the machine
geometry.  Those addresses say which measured column reaches each described
conductor or sensor; positions, orientations, outlines and drive weights remain
properties of the description itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from imas_ambix.gs.artifact_resolution import (
    ResolvedArtifact,
    resolve_machine_description,
)
from imas_ambix.gs.machine_identity import IDENTITY_PHYSICAL, identity_for_shot

if TYPE_CHECKING:
    from collections.abc import Sequence

    from imas_ambix.gs.geometry import GeometryTable
    from imas_ambix.gs.machine_identity import MachineIdentity


class MachineSelectionError(LookupError):
    """Raised when a shot's registry identity and the description disagree."""


@dataclass(frozen=True)
class SelectedMachine:
    """One shot's physical identity, the description it selected, and its table."""

    shot: int
    identity: MachineIdentity
    artifact: ResolvedArtifact
    table: GeometryTable
    #: What the description itself states about its own provenance -- the
    #: reader's block, carried through unchanged so a consumer records the
    #: evidence ledger and the forward-model blockers alongside the identity.
    description: dict[str, Any] = field(default_factory=dict)

    @property
    def selection_key(self) -> str:
        """The key selection groups by: the physical digest, not the signature."""
        return self.identity.key(IDENTITY_PHYSICAL)

    @property
    def computation_key(self) -> str:
        """The key every compute cache stays addressed by: the setup signature."""
        return self.table.signature.key

    def provenance(self) -> dict[str, Any]:
        """Return both identities, the evidence behind them, and how they resolved."""
        return {
            "selected_shot": int(self.shot),
            "selected_by": IDENTITY_PHYSICAL,
            "physical_digest": self.identity.physical_digest,
            "registry_evidence": self.identity.evidence,
            "operator_ready": self.identity.is_operator_ready,
            "authoring_gaps": list(self.identity.authoring_gaps),
            "table_signature": self.computation_key,
            **self.artifact.provenance(),
            **self.description,
        }


@dataclass
class ArtifactMachineSelector:
    """Selects machine geometry by physical identity and reads it from a description.

    Built once and reused: a description is a statement about a device, not about
    a campaign, so every shot resolving to one physical identity gets the same
    geometry.  Reads are cached per evidence shot rather than shared outright,
    because the registry EVIDENCE state is the one part of a selection that IS
    per-shot -- a table must record the evidence for the shot that asked, not
    inherit whichever shot arrived first.

    ``channel_shots`` names the shots whose acquisition channel sets are unioned
    to address the description's sensors.  Scanning more than one keeps the
    channel set a property of the machine rather than of one shot's gaps.
    """

    channel_shots: tuple[int, ...] = ()
    amc_channel_shot: int | None = None
    #: Name a description explicitly instead of resolving the pinned one.  Both
    #: must be given together; both left unset is the default resolution.
    cache_directory: str | None = None
    digest: str | None = None
    _artifact: ResolvedArtifact | None = field(default=None, init=False, repr=False)
    _reads: dict[int, tuple[GeometryTable, dict[str, Any]]] = field(
        default_factory=dict, init=False, repr=False
    )

    def artifact(self) -> ResolvedArtifact:
        """Resolve the pinned description once."""
        if self._artifact is None:
            self._artifact = resolve_machine_description(
                self.cache_directory, self.digest
            )
        return self._artifact

    def select(self, shot: int) -> SelectedMachine:
        """Return what the shot's physical identity selects, and its table."""
        artifact = self.artifact()
        identity = identity_for_shot(int(shot))
        if identity.physical_digest != artifact.physical_digest:
            raise MachineSelectionError(
                f"shot {shot} resolves to physical identity "
                f"{identity.physical_digest} but the resolved description states "
                f"{artifact.physical_digest}; the shot is not this machine"
            )
        table, description = self._read(shot, artifact, identity)
        return SelectedMachine(
            shot=int(shot),
            identity=identity,
            artifact=artifact,
            table=table,
            description=description,
        )

    def _read(
        self, shot: int, artifact: ResolvedArtifact, identity: MachineIdentity
    ) -> tuple[GeometryTable, dict[str, Any]]:
        from imas_ambix.data.description_reader import (  # noqa: PLC0415
            read_acquisition_channels,
        )
        from imas_ambix.gs.artifact_geometry import (  # noqa: PLC0415
            MachineArtifactGeometryReader,
        )

        evidence_shot = int(shot)
        cached = self._reads.get(evidence_shot)
        if cached is not None:
            return cached
        channel_shots: Sequence[int] = self.channel_shots or (evidence_shot,)
        amc_shot = (
            evidence_shot if self.amc_channel_shot is None else self.amc_channel_shot
        )
        sensor_acquisition = read_acquisition_channels(
            int(s) for s in channel_shots
        )
        current_acquisition = read_acquisition_channels((int(amc_shot),))
        reader = MachineArtifactGeometryReader(
            cache_directory=artifact.cache_directory,
            digest=artifact.digest,
            shot=evidence_shot,
            amb_channels=sensor_acquisition.sensors,
            amc_current_channels=current_acquisition.currents,
            # the digest the SHOT resolved to, so the read is pinned by the
            # selection rather than by whatever the cache happened to hold
            expected_physical_digest=identity.physical_digest,
            expected_registry_digest=artifact.registry_digest,
        )
        read = (reader.read(), reader.provenance(artifact.artifact))
        self._reads[evidence_shot] = read
        return read


def select_machine(
    shot: int, *, channel_shots: Sequence[int] = (), amc_channel_shot: int | None = None
) -> SelectedMachine:
    """Select one shot's machine by physical identity and read its geometry table."""
    selector = ArtifactMachineSelector(
        channel_shots=tuple(int(s) for s in channel_shots),
        amc_channel_shot=amc_channel_shot,
    )
    return selector.select(int(shot))


__all__ = [
    "ArtifactMachineSelector",
    "MachineSelectionError",
    "SelectedMachine",
    "select_machine",
]
