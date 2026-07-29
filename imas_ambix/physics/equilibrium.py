"""Ambix-facing construction seams for Nova equilibrium representations.

The numerical types are imported directly from :mod:`nova`.  The functions in
this module only translate Ambix geometry records and column mappings into
those types; all equilibrium kernels and solvers remain owned by Nova.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from nova.equilibrium import (
    ProfileDegrees,
    ProfilePrior,
    ProfileResult,
    ReconstructProfile,
)
from nova.equilibrium.harmonic import (
    HarmonicConfig,
    HarmonicInversion,
    ReconstructHarmonic,
)
from nova.equilibrium.measurement import Magnetics, SliceMeasurement
from nova.equilibrium.moment import (
    CurrentCells,
    MomentConfig,
    MomentInversion,
    MomentOrder,
    MomentReconstruction,
    ReconstructMoment,
)

Record = Mapping[str, Any] | object


def _value(record: Record, name: str) -> Any:
    """Read one field from either a mapping or an Ambix geometry record."""
    if isinstance(record, Mapping):
        return record[name]
    return getattr(record, name)


def magnetics_from_records(
    field_probes: Sequence[Record],
    flux_loops: Sequence[Record],
) -> Magnetics:
    """Convert Ambix diagnostic records to Nova's shared magnetics row order.

    Field-probe records provide ``r``, ``z`` and ``angle_deg``.  Flux-loop
    records provide ``r`` and ``z``; their unused orientation is set to zero.
    The returned rows contain field probes first and flux loops second, matching
    the order of the two supplied sequences.
    """
    probe_r = [float(_value(record, "r")) for record in field_probes]
    probe_z = [float(_value(record, "z")) for record in field_probes]
    probe_angle = [float(_value(record, "angle_deg")) for record in field_probes]
    loop_r = [float(_value(record, "r")) for record in flux_loops]
    loop_z = [float(_value(record, "z")) for record in flux_loops]
    number_probes = len(probe_r)
    number_loops = len(loop_r)
    return Magnetics(
        r=np.asarray(probe_r + loop_r, dtype=np.float64),
        z=np.asarray(probe_z + loop_z, dtype=np.float64),
        angle=np.asarray(probe_angle + [0.0] * number_loops, dtype=np.float64),
        flux_loop=np.asarray(
            [False] * number_probes + [True] * number_loops,
            dtype=bool,
        ),
    )


def profile_reconstructor(
    geometry: Mapping[str, Any],
    *,
    field_probes: Sequence[Record],
    flux_loops: Sequence[Record],
    n_pressure: int,
    n_diamagnetic: int,
    priors: Sequence[ProfilePrior] = (),
    options: Mapping[str, Any] | None = None,
) -> ReconstructProfile:
    """Build Nova's profile solver from Ambix's geometry-column vocabulary.

    Required geometry keys are ``grid_r``, ``grid_z``, ``inside_limiter``,
    ``cell_width``, ``cell_height``, ``source_r``, ``source_z``,
    ``source_width``, ``source_height``, ``source_names``, ``axis_seed``,
    ``wall_r`` and ``wall_z``.  ``options`` is forwarded to Nova's constructor
    for numerical controls such as iteration count and relaxation.
    """
    return ReconstructProfile.from_geometry(
        grid_r=np.asarray(geometry["grid_r"], dtype=np.float64),
        grid_z=np.asarray(geometry["grid_z"], dtype=np.float64),
        inside_limiter=np.asarray(geometry["inside_limiter"], dtype=bool),
        cell_width=np.asarray(geometry["cell_width"], dtype=np.float64),
        cell_height=np.asarray(geometry["cell_height"], dtype=np.float64),
        source_r=np.asarray(geometry["source_r"], dtype=np.float64),
        source_z=np.asarray(geometry["source_z"], dtype=np.float64),
        source_width=np.asarray(geometry["source_width"], dtype=np.float64),
        source_height=np.asarray(geometry["source_height"], dtype=np.float64),
        source_names=tuple(str(name) for name in geometry["source_names"]),
        magnetics=magnetics_from_records(field_probes, flux_loops),
        degrees=ProfileDegrees(n_pressure, n_diamagnetic),
        axis_seed=tuple(float(value) for value in geometry["axis_seed"]),
        wall_r=np.asarray(geometry["wall_r"], dtype=np.float64),
        wall_z=np.asarray(geometry["wall_z"], dtype=np.float64),
        priors=tuple(priors),
        **dict(options or {}),
    )


__all__ = [
    "CurrentCells",
    "HarmonicConfig",
    "HarmonicInversion",
    "Magnetics",
    "MomentConfig",
    "MomentInversion",
    "MomentOrder",
    "MomentReconstruction",
    "ProfileDegrees",
    "ProfilePrior",
    "ProfileResult",
    "ReconstructHarmonic",
    "ReconstructMoment",
    "ReconstructProfile",
    "SliceMeasurement",
    "magnetics_from_records",
    "profile_reconstructor",
]
