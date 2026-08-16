"""Canonical coordinate-convention contract for Ambix data and operators.

Ambix serves Data Dictionary v4 quantities in COCOS 17.  Facility archives may
use another convention, but that fact belongs to a reviewed source map and is
resolved at the read boundary.  Raw arrays are never rewritten.

The scalar algebra is reused from :mod:`nova.io.cocos`, whose complete COCOS
digit table is already a pinned Ambix dependency.  Directed poloidal probe
angles need an explicit adapter because an installed sensor orientation is a
property of the source description, not something that can be inferred from
equilibrium magnitudes.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from nova.io.cocos import transform_factor

CANONICAL_COCOS = 17
"""COCOS convention served by Ambix and Data Dictionary v4."""

CANONICAL_DD_MAJOR = 4
"""Data Dictionary major version served by Ambix."""

MAST_SOURCE_COCOS = 3
"""External owner assumption, not a measurement.

This declaration remains pending a facility statement identifying MAST's
positive-phi direction.  The level-2 arrays cannot determine that handedness.
"""

_DD_VERSION = re.compile(r"^(?P<major>[0-9]+)\.(?P<minor>[0-9]+)\.(?P<patch>[0-9]+)$")


class ConventionContractError(ValueError):
    """Raised when data cannot satisfy the canonical DD/COCOS contract."""


def require_canonical_contract(dd_version: str, cocos: int) -> None:
    """Reject data that does not declare the Ambix DDv4/COCOS-17 contract."""

    match = _DD_VERSION.fullmatch(str(dd_version))
    if match is None:
        raise ConventionContractError(
            f"data dictionary version {dd_version!r} is not an exact "
            "major.minor.patch pin"
        )
    major = int(match.group("major"))
    if major != CANONICAL_DD_MAJOR:
        raise ConventionContractError(
            f"data dictionary {dd_version!r} is major {major}; Ambix serves "
            f"DDv{CANONICAL_DD_MAJOR} only"
        )
    if int(cocos) != CANONICAL_COCOS:
        raise ConventionContractError(
            f"COCOS {cocos!r} is not the Ambix canonical COCOS {CANONICAL_COCOS}"
        )


def canonical_factor(transformation: str, *, source_cocos: int) -> float:
    """Return a source-to-COCOS-17 factor from the shared digit algebra."""

    return float(
        transform_factor(
            transformation,
            source=int(source_cocos),
            target=CANONICAL_COCOS,
        )
    )


def mast_angle_to_canonical(angle_degrees: Any) -> np.ndarray:
    """Convert FAIR-MAST probe-axis angles to DDv4 poloidal angles.

    ``magpr_ang`` increases counter-clockwise in the ``(R, Z)`` plane: 0 degrees
    reads ``B_R`` and +90 degrees reads ``+B_Z``.  DDv4's
    ``b_field_pol_probe/poloidal_angle`` increases in the opposite sense and
    projects as ``B_R cos(theta) - B_Z sin(theta)``.  Negating the source angle
    preserves the directed sensitive axis, including its polarity.
    """

    return -np.asarray(angle_degrees, dtype=np.float64)


def project_poloidal_field(
    br: Any,
    bz: Any,
    poloidal_angle_degrees: Any,
) -> np.ndarray:
    """Project ``(B_R, B_Z)`` along a DDv4 directed probe orientation."""

    theta = np.deg2rad(np.asarray(poloidal_angle_degrees, dtype=np.float64))
    return np.asarray(br) * np.cos(theta) - np.asarray(bz) * np.sin(theta)


__all__ = [
    "CANONICAL_COCOS",
    "CANONICAL_DD_MAJOR",
    "MAST_SOURCE_COCOS",
    "ConventionContractError",
    "canonical_factor",
    "mast_angle_to_canonical",
    "project_poloidal_field",
    "require_canonical_contract",
]
