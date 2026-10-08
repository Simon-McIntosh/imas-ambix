"""The selection route is keyed by machine, with JT-60SA pinned per phase.

The selector resolves a shot's registry identity through the registry that
belongs to the shot's declared machine and reads the description pinned for that
identity's physical digest.  For a machine that pins several physical identities
-- JT-60SA's two operating phases -- the shot's own identity chooses the row, and
a digest the machine does not pin is refused rather than defaulted.  These tests
pin that route on both a single-row machine (MAST) and a two-row one (JT-60SA).
"""

from __future__ import annotations

import pytest

from imas_ambix.gs import artifact_resolution as resolution
from imas_ambix.gs import description_artifact as da
from imas_ambix.gs.machine_selection import ArtifactMachineSelector

imas = pytest.importorskip("imas")

#: The last OP1 pulse: inside the OP1 range and outside OP2's.
OP1_SHOT = 101173

#: A MAST shot every committed measurement resolved through.
MAST_SHOT = 21983


@pytest.fixture(scope="module")
def jt60sa_rows() -> dict[str, resolution.PinRow]:
    """JT-60SA's pinned rows keyed by the physical digest each identity holds."""
    return {row.physical_digest: row for row in resolution.pinned_rows("jt-60sa")}


@pytest.fixture(scope="module")
def jt60sa_registry() -> da.Jt60saGeometryRegistry:
    return da.build_jt60sa_registry_from_stores()


def test_an_op1_shot_selects_to_the_pinned_op1_artifact(jt60sa_rows, jt60sa_registry):
    """The shot's own phase identity chooses the row, not the row's position."""
    op1_digest = jt60sa_registry.physical_digest("OP1")
    op1_row = jt60sa_rows[op1_digest]

    selector = ArtifactMachineSelector(
        machine="jt-60sa",
        channel_shots=(OP1_SHOT,),
        amc_channel_shot=OP1_SHOT,
    )
    selected = selector.select(OP1_SHOT)

    assert selected.table.signature.machine == "jt-60sa"
    assert selected.identity.physical_digest == op1_digest
    assert selected.artifact.physical_digest == op1_digest
    assert selected.artifact.semantic_identity == op1_row.semantic_identity
    assert selected.artifact.registry_digest == op1_row.registry_digest


def test_resolving_a_multi_phase_machine_without_a_digest_is_refused(jt60sa_rows):
    """Two pinned identities cannot be chosen between without the shot's digest."""
    assert len(jt60sa_rows) == 2
    with pytest.raises(resolution.ArtifactResolutionError, match="jt-60sa"):
        resolution.resolve_machine_description("jt-60sa")


def test_an_unknown_machine_is_refused_when_the_selector_is_built():
    """A machine with no pin row and no registry has no route to select on."""
    with pytest.raises(resolution.ArtifactResolutionError, match="d3d"):
        ArtifactMachineSelector(machine="d3d")


def test_re_authoring_both_phases_reproduces_the_pinned_rows(tmp_path):
    """The pin holds values the installed producer authors here and now."""
    authored = da.author_jt60sa_machine_artifacts(tmp_path)

    rows = resolution.pinned_rows("jt-60sa")
    for phase, row in zip(da.PHASES, rows, strict=True):
        manifest = authored[phase].manifest
        assert manifest.semantic_identity() == row.semantic_identity
        assert manifest.physical_digest == row.physical_digest
        assert manifest.registry_digest == row.registry_digest


def test_mast_resolution_and_selection_are_unchanged():
    """The single-row machine still resolves and selects as it always did."""
    mast_row = resolution.pinned_rows("mast")[0]

    resolved = resolution.resolve_machine_description("mast")
    assert resolved.physical_digest == mast_row.physical_digest
    assert resolved.semantic_identity == mast_row.semantic_identity

    selector = ArtifactMachineSelector(
        machine="mast",
        channel_shots=(MAST_SHOT,),
        amc_channel_shot=MAST_SHOT,
    )
    selected = selector.select(MAST_SHOT)
    assert selected.table.signature.machine == "mast"
    assert selected.identity.physical_digest == mast_row.physical_digest
