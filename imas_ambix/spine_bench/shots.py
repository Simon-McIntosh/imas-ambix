"""The FROZEN named benchmark shot sets, one per machine.

The evolution metric is only comparable across time if the shot set is PINNED. This
module is that pin: small, curated, named sets of shots that (a) load and solve
reliably through the frozen spine, (b) span ramp and flat-top slices, and (c) run fast
enough for a routine stamp. Changing a set (adding/removing shots, or the roles)
REQUIRES bumping that machine's version label so old and new stamps are never silently
compared.

Each set is keyed by the machine it belongs to, because a shot integer is not unique
across machines: a JT-60SA pulse and a MAST shot can carry the same digits, so a set
must be resolved through its declared machine and never from the shot numbers alone.
MAST's ``v0`` is drawn from the held-out-MSE split shots exercised by the
greens-filament-solver go/no-go gate (all confirmed to load + solve). JT-60SA's
``v0`` is the 2023 OP1 shots on which SELENE ran (PSRC populated), the reference
the reconstruction benchmark is scored against.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel

#: Bump when MAST's frozen set (shots or roles) changes.
SHOTSET_VERSION = "v0-mast-heldout-6"

#: Bump when JT-60SA's frozen set (shots or roles) changes.
JT60SA_SHOTSET_VERSION = "v0-jt60sa-op1-16"

#: The label a stamp carries when it did NOT measure the frozen set. Stamps are
#: named by their shot-set version, so an override that reused the frozen label
#: would produce a file indistinguishable from the real metric.
AD_HOC_SHOTSET_VERSION = "ad-hoc"


class BenchShot(BaseModel):
    """One pinned benchmark shot and its machine and role in the set.

    ``machine`` has no default: a shot number is not unique across machines (a
    JT-60SA pulse integer coincides with no MAST value by construction, but a
    coincidence is not a rule), so the catalogue a shot belongs to is DECLARED
    and never inferred from its digits.
    """

    machine: str
    shot_id: int
    role: str


#: MAST's pinned set. Ordered; roles document why each is included.
FROZEN_SHOTSET: list[BenchShot] = [
    BenchShot(
        machine="mast",
        shot_id=21978,
        role="ramp+flat-top (low-Ip early + 900kA flat-top)",
    ),
    BenchShot(machine="mast", shot_id=21983, role="flat-top representative"),
    BenchShot(machine="mast", shot_id=21985, role="flat-top representative"),
    BenchShot(machine="mast", shot_id=21986, role="flat-top representative"),
    BenchShot(machine="mast", shot_id=21989, role="flat-top representative"),
    BenchShot(
        machine="mast",
        shot_id=22086,
        role="flat-top representative (campaign-edge)",
    ),
]

#: JT-60SA's pinned set: the 2023 OP1 shots on which SELENE ran, i.e. whose EDDB
#: PSRC category carried a plasma-current/boundary record (the OP1 shot survey).
#: Ordered as surveyed; every shot shares one role because the inclusion criterion
#: is the same for each -- there is no ramp/flat-top split within this set the way
#: MAST's roles carry.
JT60SA_FROZEN_SHOTSET: list[BenchShot] = [
    BenchShot(machine="jt60sa", shot_id=100599, role="op1 SELENE-ran (PSRC present)"),
    BenchShot(machine="jt60sa", shot_id=100999, role="op1 SELENE-ran (PSRC present)"),
    BenchShot(machine="jt60sa", shot_id=101017, role="op1 SELENE-ran (PSRC present)"),
    BenchShot(machine="jt60sa", shot_id=101025, role="op1 SELENE-ran (PSRC present)"),
    BenchShot(machine="jt60sa", shot_id=101026, role="op1 SELENE-ran (PSRC present)"),
    BenchShot(machine="jt60sa", shot_id=101029, role="op1 SELENE-ran (PSRC present)"),
    BenchShot(machine="jt60sa", shot_id=101031, role="op1 SELENE-ran (PSRC present)"),
    BenchShot(machine="jt60sa", shot_id=101033, role="op1 SELENE-ran (PSRC present)"),
    BenchShot(machine="jt60sa", shot_id=101039, role="op1 SELENE-ran (PSRC present)"),
    BenchShot(machine="jt60sa", shot_id=101044, role="op1 SELENE-ran (PSRC present)"),
    BenchShot(machine="jt60sa", shot_id=101045, role="op1 SELENE-ran (PSRC present)"),
    BenchShot(machine="jt60sa", shot_id=101046, role="op1 SELENE-ran (PSRC present)"),
    BenchShot(machine="jt60sa", shot_id=101153, role="op1 SELENE-ran (PSRC present)"),
    BenchShot(machine="jt60sa", shot_id=101156, role="op1 SELENE-ran (PSRC present)"),
    BenchShot(machine="jt60sa", shot_id=101162, role="op1 SELENE-ran (PSRC present)"),
    BenchShot(machine="jt60sa", shot_id=101163, role="op1 SELENE-ran (PSRC present)"),
]


@dataclass(frozen=True)
class FrozenShotset:
    """One machine's pinned set and the label its stamps carry.

    The two travel together because they are one identity: a stamp is named by the
    label of the set it measured, so resolving either without the other would let a
    label from one machine describe a set from another.
    """

    machine: str
    version: str
    shots: list[BenchShot]


#: The frozen sets, keyed by machine.  This is the one table the label resolver and
#: the parity guard read, so a run's declared machine selects both the shots it is
#: compared against and the label it is stamped with.
FROZEN_SHOTSETS: dict[str, FrozenShotset] = {
    "mast": FrozenShotset(
        machine="mast", version=SHOTSET_VERSION, shots=FROZEN_SHOTSET
    ),
    "jt60sa": FrozenShotset(
        machine="jt60sa", version=JT60SA_SHOTSET_VERSION, shots=JT60SA_FROZEN_SHOTSET
    ),
}


def resolve_shotset_version(shots: list[BenchShot] | None) -> str:
    """Return the shot-set label that honestly names what will be measured.

    A stamp's filename and its comparability guard both come from this label, so
    it must be derived from the shot set actually solved rather than assumed.
    Anything other than a machine's frozen shots in their frozen order with their
    frozen roles is :data:`AD_HOC_SHOTSET_VERSION`, which keeps an override from
    ever landing in the results directory under a frozen metric's name.

    The set is resolved through the machine the shots DECLARE and never inferred
    from their digits: a JT-60SA shot whose number coincides with a MAST id
    resolves against JT-60SA's row, so it can never be stamped under MAST's
    label.  With no shots the run uses the frozen set, which is MAST's by
    default.  A machine with no frozen set resolves ad-hoc rather than borrowing
    another machine's label.
    """
    if shots is None:
        return SHOTSET_VERSION
    if not shots:
        return AD_HOC_SHOTSET_VERSION
    row = FROZEN_SHOTSETS.get(shots[0].machine)
    if row is None:
        return AD_HOC_SHOTSET_VERSION
    frozen = [(int(shot.shot_id), shot.role) for shot in row.shots]
    given = [(int(shot.shot_id), shot.role) for shot in shots]
    return row.version if given == frozen else AD_HOC_SHOTSET_VERSION
