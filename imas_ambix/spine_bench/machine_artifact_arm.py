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

    AMBIX_MACHINE_ARTIFACT_CACHE=... AMBIX_MACHINE_ARTIFACT_DIGEST=sha256:... \\
        python -m imas_ambix.spine_bench.machine_artifact_arm

The geometry is read through :class:`MachineArtifactGeometryReader`, whose
resolution verifies the manifest against the cache address and every file against
the manifest before opening an IDS, and which is pinned here to the physical and
registry digests this arm was written against.  A cache holding a different
machine therefore fails to resolve rather than being silently benched.

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
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from imas_ambix.spine_bench.runner import run_stamp, write_yaml
from imas_ambix.spine_bench.shots import FROZEN_SHOTSET

logger = logging.getLogger(__name__)

#: The machine this arm is written against.  The physical digest fixes the
#: conductor and sensor geometry, the registry digest the shot-range identity
#: mapping; both are enforced during artifact resolution.  The SEMANTIC identity
#: is deliberately not pinned: revisions that change what the description says
#: about the same conductors (an evidence promotion, a refused calibration) are
#: exactly what this arm exists to measure, so it must be able to run on any of
#: them and record which one it read.
PINNED_PHYSICAL_DIGEST = "ca06c8f64481114f"
PINNED_REGISTRY_DIGEST = (
    "7083e8029c879310d4b811ecc58f5eefdd40b2bfe01b4a1714b177b03a307366"
)

#: Environment variables naming the artifact, matching the contract the
#: artifact-backed geometry tests already skip-guard on.
CACHE_ENV = "AMBIX_MACHINE_ARTIFACT_CACHE"
DIGEST_ENV = "AMBIX_MACHINE_ARTIFACT_DIGEST"

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

    cache_directory: str | Path
    digest: str
    evidence_shot: int
    #: Shots whose amb channel sets are unioned to address the artifact's sensors.
    #: Scanning the whole benchmark set rather than one shot keeps the channel set
    #: geometry-determined instead of an artifact of one shot's acquisition gaps.
    channel_shots: tuple[int, ...] = ()
    expected_physical_digest: str = PINNED_PHYSICAL_DIGEST
    expected_registry_digest: str = PINNED_REGISTRY_DIGEST
    label: str = ARTIFACT_SOURCE_LABEL
    _table: Any | None = field(default=None, init=False, repr=False)
    _provenance: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def build(self) -> Any:
        """Resolve the artifact and read its geometry table (once)."""
        if self._table is not None:
            return self._table

        from imas_ambix.gs.artifact_geometry import (  # noqa: PLC0415
            MachineArtifactGeometryReader,
        )
        from imas_ambix.gs.geometry import (  # noqa: PLC0415
            canonical_amb_channels,
            read_amc_current_channels,
        )

        shots = self.channel_shots or (self.evidence_shot,)
        reader = MachineArtifactGeometryReader(
            cache_directory=self.cache_directory,
            digest=self.digest,
            shot=int(self.evidence_shot),
            amb_channels=tuple(canonical_amb_channels([int(s) for s in shots])),
            amc_current_channels=tuple(
                read_amc_current_channels(int(self.evidence_shot))
            ),
            expected_physical_digest=self.expected_physical_digest,
            expected_registry_digest=self.expected_registry_digest,
        )
        artifact = reader.resolve()
        table = reader.read()
        self._provenance = {
            "source": self.label,
            "reader": ("imas_ambix.gs.artifact_geometry.MachineArtifactGeometryReader"),
            "artifact_digest": str(artifact.digest),
            "evidence_shot": int(self.evidence_shot),
            "channel_shots": [int(s) for s in shots],
            "table_signature": table.signature.key,
            "n_driven_circuits": len(table.circuit_drives),
            "campaign_addressing": {
                "amb_channels": "canonical_amb_channels (sensor channel names)",
                "amc_current_channels": (
                    "read_amc_current_channels (measured coil-current channel names)"
                ),
            },
            **reader.provenance(artifact),
        }
        self._table = table
        return table

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


def source_from_environment(
    cache_directory: str | Path | None = None,
    digest: str | None = None,
    *,
    evidence_shot: int | None = None,
) -> MachineArtifactGeometrySource:
    """Build the source from explicit arguments, falling back to the environment.

    Raises when the artifact is not named, rather than defaulting to a path: a
    benchmark that quietly measured a different machine than the caller meant is
    worse than one that refuses to start.
    """
    cache = str(cache_directory or os.environ.get(CACHE_ENV, "")).strip()
    resolved_digest = str(digest or os.environ.get(DIGEST_ENV, "")).strip()
    if not (cache and resolved_digest):
        raise ValueError(
            "no machine-description artifact named: pass the cache directory and "
            f"digest, or set {CACHE_ENV} and {DIGEST_ENV}"
        )
    shots = tuple(int(s.shot_id) for s in FROZEN_SHOTSET)
    return MachineArtifactGeometrySource(
        cache_directory=cache,
        digest=resolved_digest,
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

    source = source_from_environment(
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
