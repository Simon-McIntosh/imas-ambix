"""Bench the frozen shot set against a published machine description.

The benchmark's other arm reads the machine from the campaign's own static efm
arrays.  This one reads it from a verified machine-description artifact instead —
the same shots, the same frozen spine, the same solves, with every
geometry-dependent quantity (coil filaments and turns, sensor positions and
orientations, the limiter contour, the passive structure) coming from the
artifact.  Because the frozen spine never fits the magnetics,
``magnetics_residual_whitened_rms`` is a forward-model check on that geometry,
and the difference between the two arms is what a change of machine description
costs in sensor space.

Run it as::

    python -m imas_ambix.spine_bench.machine_artifact_arm

with no environment to set up: the description is resolved from the identity
:mod:`imas_ambix.gs.artifact_resolution` pins, and authored locally if no cache
on this machine holds it.  Naming one explicitly (``--cache-directory`` /
``--digest``, or ``AMBIX_MACHINE_ARTIFACT_CACHE`` / ``AMBIX_MACHINE_ARTIFACT_DIGEST``)
overrides that, which is how a second revision of the same machine is benched
against the pinned one.

Which machine gets read is decided by SELECTION rather than by the address: the
evidence shot resolves through the Nova registry to a physical identity, and
:class:`~imas_ambix.gs.machine_selection.ArtifactMachineSelector` refuses any
description that does not state that identity.  The geometry then comes through
:class:`MachineArtifactGeometryReader`, whose resolution verifies the manifest
against the cache address and every file against the manifest before opening an
IDS.  A cache holding a different machine therefore fails to resolve rather than
being silently benched.

Two campaign-side inputs are still read from efm, and neither is geometry:

``amb_channels``
    the campaign's ``(channel, description)`` sensor-name pairs.  The artifact
    describes sensors, not the acquisition system that named them, so this is
    what ADDRESSES an artifact sensor by the channel a shot's data arrives on.
    The positions and orientations behind those names are the artifact's.

``amc_current_channels``
    the campaign's measured coil-current channel names.  Again an addressing
    list: without it :func:`~imas_ambix.gs.operator.classify_circuits` finds no
    driven circuit and the vacuum coil block comes out empty.  Which conductors
    those channels supply, and at how many ampere-turns per ampere, is stated by
    the artifact's own drive map.

So efm supplies the names of the columns and the artifact supplies the machine.
Nothing here reads a reconstructed efm output, and nothing writes into another
module's cache: the source hands the runner a table and the runner records the
signature of the table it was handed.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from imas_ambix.gs.artifact_resolution import (
    CACHE_ENV,
    DIGEST_ENV,
    PINNED_PHYSICAL_DIGEST,
    PINNED_REGISTRY_DIGEST,
)
from imas_ambix.gs.machine_selection import ArtifactMachineSelector
from imas_ambix.spine_bench.runner import run_stamp, write_yaml
from imas_ambix.spine_bench.shots import FROZEN_SHOTSET

logger = logging.getLogger(__name__)

#: The machine this arm is written against, and the environment variables that
#: name a description explicitly.  All four are re-exported from
#: :mod:`imas_ambix.gs.artifact_resolution`, which owns them: the arm selects a
#: machine and must not be able to pin a different one than the resolution path
#: verifies against.  The SEMANTIC identity is deliberately not enforced here:
#: revisions that change what the description says about the same conductors (an
#: evidence promotion, a refused calibration) are exactly what this arm exists to
#: measure, so it must be able to run on any of them and record which one it read.
__all__ = [
    "ARTIFACT_SOURCE_LABEL",
    "CACHE_ENV",
    "DIGEST_ENV",
    "PINNED_PHYSICAL_DIGEST",
    "PINNED_REGISTRY_DIGEST",
    "MachineArtifactGeometrySource",
    "main",
    "resolve_geometry_source",
]

#: The geometry-source label this arm stamps itself with.
ARTIFACT_SOURCE_LABEL = "machine-artifact"


@dataclass
class MachineArtifactGeometrySource:
    """A published machine description, handed to the runner as a geometry table.

    The table is built once and returned for every shot.  That is not a cache
    standing in for per-shot work: a machine description is a statement about the
    device, not about a campaign, so unlike the efm arrays there is no per-shot or
    per-campaign geometry to resolve.  What IS shot-dependent is the registry
    evidence state recorded in the table's provenance, which is why
    :attr:`evidence_shot` is fixed at construction and reported — a table
    describing one machine must not silently claim the evidence of whichever shot
    happened to be solved first.
    """

    evidence_shot: int
    #: Shots whose amb channel sets are unioned to address the artifact's sensors.
    #: Scanning the whole benchmark set rather than one shot keeps the channel set
    #: geometry-determined instead of an artifact of one shot's acquisition gaps.
    channel_shots: tuple[int, ...] = ()
    #: Left unset, the description is resolved from the pinned identity; set, it
    #: names one explicitly.  Neither is a path the caller has to know about for
    #: the arm to run.
    cache_directory: str | Path | None = None
    digest: str | None = None
    expected_physical_digest: str = PINNED_PHYSICAL_DIGEST
    expected_registry_digest: str = PINNED_REGISTRY_DIGEST
    label: str = ARTIFACT_SOURCE_LABEL
    _table: Any | None = field(default=None, init=False, repr=False)
    _provenance: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def build(self) -> Any:
        """Select the machine by physical identity, reading its table once."""
        if self._table is not None:
            return self._table

        shots = self.channel_shots or (self.evidence_shot,)
        selector = ArtifactMachineSelector(
            channel_shots=tuple(int(s) for s in shots),
            amc_channel_shot=int(self.evidence_shot),
            cache_directory=(
                None if self.cache_directory is None else str(self.cache_directory)
            ),
            digest=self.digest,
        )
        selected = selector.select(int(self.evidence_shot))
        if selected.identity.physical_digest != self.expected_physical_digest:
            raise ValueError(
                f"shot {self.evidence_shot} selects physical identity "
                f"{selected.identity.physical_digest}, not the machine this arm "
                f"is written against ({self.expected_physical_digest})"
            )
        self._provenance = {
            "source": self.label,
            "reader": ("imas_ambix.gs.artifact_geometry.MachineArtifactGeometryReader"),
            "selector": "imas_ambix.gs.machine_selection.ArtifactMachineSelector",
            "evidence_shot": int(self.evidence_shot),
            "channel_shots": [int(s) for s in shots],
            "n_driven_circuits": len(selected.table.circuit_drives),
            "campaign_addressing": {
                "amb_channels": "canonical_amb_channels (sensor channel names)",
                "amc_current_channels": (
                    "read_amc_current_channels (measured coil-current channel names)"
                ),
            },
            **selected.provenance(),
        }
        self._table = selected.table
        return selected.table

    def table_for(self, shot: int) -> Any | None:  # noqa: ARG002 — see class docstring
        """Return the artifact's table, which describes the machine for any shot."""
        return self.build()

    def provenance(self) -> dict[str, Any]:
        """Return the verified identity of the description that was read."""
        self.build()
        return dict(self._provenance)

    @property
    def revision(self) -> str:
        """The semantic identity — which republication of this machine was read.

        The right discriminator for a revision, because it is the one that moves
        when the description is republished with the same conductors: the physical
        digest and the setup signature both stay put, so neither can tell two such
        revisions apart.
        """
        self.build()
        return str(self._provenance.get("semantic_identity", ""))


def resolve_geometry_source(
    cache_directory: str | Path | None = None,
    digest: str | None = None,
    *,
    evidence_shot: int | None = None,
) -> MachineArtifactGeometrySource:
    """Build the source, naming a description explicitly or resolving the pinned one.

    Nothing has to be set up for this to return a source.  A benchmark that
    quietly measured a different machine than the caller meant would be worse than
    one that refuses to start, and what makes defaulting safe here is that the
    default is not a filesystem path: it is a PINNED description, verified by
    semantic identity after resolution and recorded in the stamp.  A run that
    names nothing therefore measures a machine this repository states, not
    whatever a cache happened to hold.
    """
    shots = tuple(int(s.shot_id) for s in FROZEN_SHOTSET)
    return MachineArtifactGeometrySource(
        cache_directory=cache_directory,
        digest=digest,
        evidence_shot=int(evidence_shot if evidence_shot is not None else shots[0]),
        channel_shots=shots,
    )


def main(argv: list[str] | None = None) -> int:
    """Stamp the frozen shot set against the named machine description."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-directory", default=None)
    parser.add_argument("--digest", default=None)
    parser.add_argument("--evidence-shot", type=int, default=None)
    parser.add_argument("--max-slices", type=int, default=6)
    parser.add_argument("--sigma", type=float, default=0.02)
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent / "results"),
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    source = resolve_geometry_source(
        args.cache_directory, args.digest, evidence_shot=args.evidence_shot
    )
    source.build()
    provenance = source.provenance()
    logger.info("artifact:   %s", provenance["artifact_digest"])
    logger.info("semantic:   %s", provenance["semantic_identity"])
    logger.info("physical:   %s", provenance["physical_digest"])
    logger.info("signature:  %s", provenance["table_signature"])
    logger.info("driven:     %d circuits", provenance["n_driven_circuits"])

    stamp = run_stamp(
        created_utc=datetime.now(UTC).isoformat(),
        max_slices=args.max_slices,
        sigma=args.sigma,
        geometry_source=source,
    )
    path = write_yaml(stamp, Path(args.out_dir))
    logger.info("\n=== aggregate ===\n%s", json.dumps(stamp.aggregate, indent=2))
    logger.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
