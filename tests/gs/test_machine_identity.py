"""Machine identity is the physical configuration, not the discretization.

These tests pin both halves of the contract: a re-subdivided machine resolves to
the same physical identity, and the setup signature that addresses every cache on
disk stays byte-identical while that resolution happens.
"""

from __future__ import annotations

import numpy as np
import pytest

from imas_ambix.gs import geometry as gsg
from imas_ambix.gs.machine_identity import (
    IDENTITY_MODES,
    IDENTITY_PHYSICAL,
    IDENTITY_REPRESENTATION,
    MachineIdentity,
    MachineIdentityError,
    default_registry,
    describe_identity,
    identity_for_representation,
    identity_for_shot,
    identity_for_table,
    same_machine,
)

#: The frozen benchmark campaign's setup digest and the machine it describes.
FROZEN_REPRESENTATION_DIGEST = "1cb6f2ee742c4ee4"
FROZEN_REPRESENTATION_KEY = "mp78-fl46-fc938-lim37-1cb6f2ee742c4ee4"
MAST_PHYSICAL_DIGEST = "ca06c8f64481114f"

#: The frozen benchmark shots, whose identity must resolve without the alias table.
FROZEN_SHOTS = (21978, 21983, 21985, 21986, 21989, 22086)


class _Signature:
    """Minimal stand-in carrying the two attributes resolution reads."""

    def __init__(self, digest: str, key: str) -> None:
        self.digest = digest
        self.key = key


class _Table:
    """Minimal stand-in for a GeometryTable's identity-bearing surface."""

    def __init__(self, digest: str, key: str, shots=()) -> None:
        self.signature = _Signature(digest, key)
        self.shots = list(shots)


# --- the setup signature must not move ------------------------------------


def test_the_frozen_campaign_key_is_byte_identical():
    """Every cache, artifact and checkpoint on disk resolves through this string."""
    signature = gsg.SetupSignature(
        n_bprobe=78,
        n_fluxloop=46,
        n_pf_filament=938,
        n_limiter=37,
        digest=FROZEN_REPRESENTATION_DIGEST,
    )
    assert signature.key == FROZEN_REPRESENTATION_KEY
    assert signature.machine == "mast"
    assert not signature.key.startswith("mast-")


def test_resolving_identity_does_not_alter_the_signature_key():
    """Identity resolution is a read: the cache address it was given comes back."""
    table = _Table(FROZEN_REPRESENTATION_DIGEST, FROZEN_REPRESENTATION_KEY)
    identity = identity_for_table(table)
    assert identity.representation_key == FROZEN_REPRESENTATION_KEY
    assert table.signature.key == FROZEN_REPRESENTATION_KEY


def test_the_rounded_position_digest_is_unchanged_by_this_module():
    """A digest recomputed after import must equal one computed before it."""
    arrays = [np.array([0.1, 0.2, 0.3]), np.array([1.0, 2.0])]
    assert gsg.round_geometry_hash(arrays) == gsg._round_hash(arrays)


# --- physical identity ----------------------------------------------------


def test_the_frozen_campaign_resolves_to_the_published_configuration():
    identity = identity_for_representation(FROZEN_REPRESENTATION_DIGEST)
    assert identity.physical_digest == MAST_PHYSICAL_DIGEST
    assert identity.dd_version == "4.1.1"
    assert len(identity.registry_digest) == 64


def test_a_signature_object_and_a_bare_digest_resolve_alike():
    """The discretization counts in the key must not affect resolution."""
    signature = _Signature(FROZEN_REPRESENTATION_DIGEST, FROZEN_REPRESENTATION_KEY)
    assert (
        identity_for_representation(signature).physical_digest
        == identity_for_representation(FROZEN_REPRESENTATION_DIGEST).physical_digest
    )


def test_every_recorded_representation_names_one_machine():
    """Three discretizations, one device: the identity rule as a measurement."""
    registry = default_registry()
    resolved = {
        identity_for_representation(digest, registry=registry).physical_digest
        for digest in registry.representation_aliases
    }
    assert resolved == {MAST_PHYSICAL_DIGEST}
    assert len(registry.representation_aliases) >= 3


def test_a_different_filament_count_is_the_same_machine():
    """fc938 and fc1004 differ in subdivision only, so they must compare equal."""
    fc938 = _Table("1cb6f2ee742c4ee4", "mp78-fl46-fc938-lim37-1cb6f2ee742c4ee4")
    fc1004 = _Table("9425ae4a8bf3bc15", "mp78-fl46-fc1004-lim37-9425ae4a8bf3bc15")
    assert fc938.signature.key != fc1004.signature.key
    assert same_machine(fc938, fc1004)


def test_an_unaliased_representation_raises_rather_than_guessing():
    """An unknown campaign must be aliased in nova, never silently assigned."""
    with pytest.raises(MachineIdentityError, match="not in the Nova registry"):
        identity_for_representation("ffffffffffffffff")


# --- the shot route carries evidence the alias table cannot ---------------


@pytest.mark.parametrize("shot", FROZEN_SHOTS)
def test_each_frozen_shot_selects_the_published_configuration(shot):
    identity = identity_for_shot(shot)
    assert identity.physical_digest == MAST_PHYSICAL_DIGEST
    assert identity.evidence == "observed"


def test_the_shot_route_and_the_alias_route_agree():
    """Two independent paths to identity; a disagreement would be a registry fault."""
    by_alias = identity_for_representation(FROZEN_REPRESENTATION_DIGEST)
    for shot in FROZEN_SHOTS:
        assert identity_for_shot(shot).physical_digest == by_alias.physical_digest


def test_a_shot_outside_the_registry_ranges_raises():
    with pytest.raises(MachineIdentityError, match="outside the Nova registry"):
        identity_for_shot(1)


def test_a_table_prefers_the_shot_route_for_its_evidence_state():
    """The shot route reports source coverage; the alias route cannot."""
    table = _Table(
        FROZEN_REPRESENTATION_DIGEST, FROZEN_REPRESENTATION_KEY, shots=[21983]
    )
    assert identity_for_table(table).evidence == "observed"


def test_a_table_with_no_registered_shot_falls_back_to_the_alias_route():
    """A table stays usable when only its representation is known."""
    table = _Table(FROZEN_REPRESENTATION_DIGEST, FROZEN_REPRESENTATION_KEY, shots=[1])
    identity = identity_for_table(table)
    assert identity.physical_digest == MAST_PHYSICAL_DIGEST
    assert identity.evidence == "aliased"


# --- the provisional revision must not claim to be complete ---------------


def test_the_provisional_configuration_is_not_operator_ready():
    """Unsourced electrical semantics must be visible, not assumed away."""
    identity = identity_for_representation(FROZEN_REPRESENTATION_DIGEST)
    assert not identity.is_operator_ready
    assert identity.authoring_gaps
    assert any("turns" in gap for gap in identity.authoring_gaps)


def test_a_configuration_without_gaps_would_be_operator_ready():
    """The predicate reads the gap list rather than hard-coding today's answer."""
    identity = MachineIdentity(
        physical_digest="a" * 16,
        representation_key="k",
        evidence="observed",
        authoring_gaps=(),
        registry_digest="b" * 64,
        dd_version="4.1.1",
    )
    assert identity.is_operator_ready


# --- the mode seam --------------------------------------------------------


def test_the_two_identity_modes_return_the_two_keys():
    identity = identity_for_representation(FROZEN_REPRESENTATION_DIGEST)
    assert identity.key(IDENTITY_REPRESENTATION) == FROZEN_REPRESENTATION_DIGEST
    assert identity.key(IDENTITY_PHYSICAL) == MAST_PHYSICAL_DIGEST
    assert IDENTITY_MODES == (IDENTITY_REPRESENTATION, IDENTITY_PHYSICAL)


def test_an_unknown_identity_mode_raises():
    identity = identity_for_representation(FROZEN_REPRESENTATION_DIGEST)
    with pytest.raises(ValueError, match="unknown identity mode"):
        identity.key("physical-digest")


# --- provenance propagation onto the built artifacts ----------------------


def _operator(**overrides):
    """A minimal ForwardOperator; only its identity fields matter here."""
    from imas_ambix.gs.operator import ForwardOperator

    empty = np.zeros((0, 0), dtype=np.float64)
    fields = {
        "signature_key": FROZEN_REPRESENTATION_KEY,
        "sensor_channels": [],
        "sensor_kind": [],
        "g_pf": empty,
        "g_plasma": empty,
        "g_passive": empty,
        "pf_circuits": [],
        "pf_amc_channels": [],
        "pf_merged_circuits": [],
        "plasma_rz": np.zeros((0, 2)),
        "passive_rz": np.zeros((0, 2)),
        "circuit_classes": [],
        "excluded_channels": [],
        "flagged_channels": [],
    }
    return ForwardOperator(**{**fields, **overrides})


def test_an_operator_without_resolved_identity_reports_the_historical_summary():
    """An operator built with no registry access must be unchanged and usable."""
    shapes = _operator().shapes()
    assert shapes["signature_key"] == FROZEN_REPRESENTATION_KEY
    assert "physical_digest" not in shapes


def test_an_operator_carries_the_physical_digest_beside_the_signature():
    """The two identities coexist: discretization built it, the machine owns it."""
    shapes = _operator(physical_digest=MAST_PHYSICAL_DIGEST).shapes()
    assert shapes["signature_key"] == FROZEN_REPRESENTATION_KEY
    assert shapes["physical_digest"] == MAST_PHYSICAL_DIGEST


def test_exported_geometry_fields_omit_the_digest_until_it_is_resolved():
    """An export written without identity stays byte-comparable with older ones."""
    from imas_ambix.gs.geometry_export import GeometryFields, MachineGeometry

    machine = MachineGeometry(
        limiter_r=[0.2, 1.9],
        limiter_z=[-1.8, 1.8],
        pf_coil_r=[0.5],
        pf_coil_z=[1.0],
        r0=0.85,
        minor_radius=0.6,
    )
    fields = GeometryFields(
        signature_key=FROZEN_REPRESENTATION_KEY,
        shots=[21983],
        machine=machine,
    )
    assert "physical_digest" not in fields.to_dict()
    fields.physical_digest = MAST_PHYSICAL_DIGEST
    assert fields.to_dict()["physical_digest"] == MAST_PHYSICAL_DIGEST


def test_the_provenance_block_records_both_identities_and_the_evidence():
    table = _Table(
        FROZEN_REPRESENTATION_DIGEST, FROZEN_REPRESENTATION_KEY, shots=[21983]
    )
    block = describe_identity(table)
    assert block["physical_digest"] == MAST_PHYSICAL_DIGEST
    assert block["representation_key"] == FROZEN_REPRESENTATION_KEY
    assert block["evidence"] == "observed"
    assert block["operator_ready"] is False
    assert len(block["authoring_gaps"]) == 6
