"""Ambix array adapters for Nova's flux-surface transport representation."""

from __future__ import annotations

from dataclasses import MISSING, fields
from typing import TYPE_CHECKING, Any

import numpy as np
from nova.transport import (
    CurrentDiffusion,
    EtaProfile,
    FluxSurfaceGeometry,
    assemble_flux_surface_geometry_jax,
    flux_surface_geometry,
    flux_surface_geometry_jax,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

_SCALAR_GEOMETRY_FIELDS = {
    "axis_psi",
    "boundary_psi",
    "flux_sign",
    "ip_amperes",
    "phi_b",
    "r0",
    "volume",
}


def flux_surface_geometry_from_mapping(
    values: Mapping[str, Any],
) -> FluxSurfaceGeometry:
    """Coerce an Ambix geometry payload into Nova's immutable metric type."""
    converted: dict[str, Any] = {}
    for field in fields(FluxSurfaceGeometry):
        if field.name not in values:
            if field.default is not MISSING or field.default_factory is not MISSING:
                continue
            raise KeyError(f"missing flux-surface geometry field {field.name!r}")
        value = values[field.name]
        converted[field.name] = (
            float(value)
            if field.name in _SCALAR_GEOMETRY_FIELDS
            else np.asarray(value, dtype=np.float64)
        )
    return FluxSurfaceGeometry(**converted)


def current_diffusion_from_mapping(
    geometry: Mapping[str, Any],
    eta: EtaProfile | Mapping[str, float],
    *,
    theta: float = 1.0,
) -> CurrentDiffusion:
    """Build Nova's diffusion solver from serialized Ambix geometry and eta."""
    eta_profile = eta if isinstance(eta, EtaProfile) else EtaProfile(**dict(eta))
    return CurrentDiffusion(
        flux_surface_geometry_from_mapping(geometry),
        eta_profile,
        theta=float(theta),
    )


__all__ = [
    "CurrentDiffusion",
    "EtaProfile",
    "FluxSurfaceGeometry",
    "assemble_flux_surface_geometry_jax",
    "current_diffusion_from_mapping",
    "flux_surface_geometry",
    "flux_surface_geometry_from_mapping",
    "flux_surface_geometry_jax",
]
