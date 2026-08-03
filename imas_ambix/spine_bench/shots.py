"""The FROZEN named benchmark shot set.

The evolution metric is only comparable across time if the shot set is PINNED. This
module is that pin: a small, curated, named set of MAST shots that (a) load and solve
reliably through the frozen spine, (b) span ramp and flat-top slices, and (c) run fast
enough for a routine stamp. Changing the set (adding/removing shots, or the roles)
REQUIRES bumping ``SHOTSET_VERSION`` so old and new stamps are never silently compared.

v0 is drawn from the held-out-MSE split shots exercised by the greens-filament-solver §2
go/no-go gate (all confirmed to load + solve). It is intentionally small; a v1 spanning
both fcoil campaign signatures (938 / 1004) and more machines is a documented follow-up.
"""

from __future__ import annotations

from pydantic import BaseModel

#: Bump when the frozen set (shots or roles) changes.
SHOTSET_VERSION = "v0-mast-heldout-6"

#: The label a stamp carries when it did NOT measure the frozen set. Stamps are
#: named by their shot-set version, so an override that reused the frozen label
#: would produce a file indistinguishable from the real metric.
AD_HOC_SHOTSET_VERSION = "ad-hoc"


class BenchShot(BaseModel):
    """One pinned benchmark shot and its role in the set."""

    shot_id: int
    role: str


#: The pinned set. Ordered; roles document why each is included.
FROZEN_SHOTSET: list[BenchShot] = [
    BenchShot(shot_id=21978, role="ramp+flat-top (low-Ip early + 900kA flat-top)"),
    BenchShot(shot_id=21983, role="flat-top representative"),
    BenchShot(shot_id=21985, role="flat-top representative"),
    BenchShot(shot_id=21986, role="flat-top representative"),
    BenchShot(shot_id=21989, role="flat-top representative"),
    BenchShot(shot_id=22086, role="flat-top representative (campaign-edge)"),
]


def resolve_shotset_version(shots: list[BenchShot] | None) -> str:
    """Return the shot-set label that honestly names what will be measured.

    A stamp's filename and its comparability guard both come from this label, so
    it must be derived from the shot set actually solved rather than assumed.
    Anything other than the frozen shots in their frozen order with their frozen
    roles is :data:`AD_HOC_SHOTSET_VERSION`, which keeps an override from ever
    landing in the results directory under the frozen metric's name.
    """
    if shots is None:
        return SHOTSET_VERSION
    frozen = [(shot.shot_id, shot.role) for shot in FROZEN_SHOTSET]
    given = [(int(shot.shot_id), shot.role) for shot in shots]
    return SHOTSET_VERSION if given == frozen else AD_HOC_SHOTSET_VERSION
