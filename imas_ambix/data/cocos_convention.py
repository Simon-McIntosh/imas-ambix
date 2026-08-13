"""Measured FAIR-MAST level-2 coordinate-convention evidence.

Sauter and Medvedev reduce a COCOS convention to four coefficients:
``sigma_bp``, ``e_bp``, ``sigma_r_phi_z`` and
``sigma_rho_theta_phi``.  This module keeps the real-shot observations that
identify those coefficients next to the scoring algebra and the resulting
transform factors.  The raw FAIR-MAST stores remain read-only.

Signs use ``+1`` and ``-1``.  Poloidal direction is ``+1`` for
counter-clockwise and ``-1`` for clockwise when the ``(R, Z)`` cross-section
is viewed from the front.  Poloidal-flux values are edge minus axis, matching
the consistency relation in Sauter and Medvedev Eq. 22.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import tau
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np
from nova.io.cocos import CONVENTION_DIGITS

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

MAST_LEVEL2_ROOT = Path("/work/projects/imas_gpu/mast/level2/shots")

SIGN_SOURCE_PATHS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "plasma_current": ("magnetics/ip",),
        "toroidal_field": ("equilibrium/bvac_rmag",),
        "poloidal_flux": (
            "equilibrium/psi",
            "equilibrium/major_radius",
            "equilibrium/z",
            "equilibrium/magnetic_axis_r",
            "equilibrium/magnetic_axis_z",
            "equilibrium/lcfs_r",
            "equilibrium/lcfs_z",
        ),
        "poloidal_angle_direction": (
            "equilibrium/lcfs_r",
            "equilibrium/lcfs_z",
        ),
        "safety_factor": ("equilibrium/q95",),
        "flux_exponent": ("equilibrium/psi:units",),
    }
)
"""Level-2 arrays and metadata from which each observation is derived."""


@dataclass(frozen=True)
class ShotSignObservation:
    """Robust medians extracted from one real level-2 pulse."""

    shot: int
    plasma_current_a: float
    toroidal_field_t: float
    poloidal_flux_edge_minus_axis_wb_per_rad: float
    poloidal_angle_signed_area_m2: float
    safety_factor: float
    flux_exponent: int
    retained_slices: int

    @property
    def plasma_current_sign(self) -> int:
        """Sign of the measured plasma current."""

        return _finite_sign(self.plasma_current_a, "plasma current")

    @property
    def toroidal_field_sign(self) -> int:
        """Sign of the reconstructed vacuum toroidal field."""

        return _finite_sign(self.toroidal_field_t, "toroidal field")

    @property
    def poloidal_flux_sign(self) -> int:
        """Sign of poloidal flux at the edge relative to the axis."""

        return _finite_sign(
            self.poloidal_flux_edge_minus_axis_wb_per_rad,
            "poloidal flux edge minus axis",
        )

    @property
    def poloidal_angle_direction(self) -> int:
        """Return ``+1`` for counter-clockwise or ``-1`` for clockwise."""

        return _finite_sign(
            self.poloidal_angle_signed_area_m2,
            "ordered LCFS signed area",
        )

    @property
    def safety_factor_sign(self) -> int:
        """Sign of the reconstructed safety factor."""

        return _finite_sign(self.safety_factor, "safety factor")


MAST_LEVEL2_SIGN_TABLE = (
    ShotSignObservation(
        shot=13_277,
        plasma_current_a=732_126.5000001,
        toroidal_field_t=-0.47604525089263916,
        poloidal_flux_edge_minus_axis_wb_per_rad=-0.035180365335017026,
        poloidal_angle_signed_area_m2=-1.613527010949289,
        safety_factor=6.998124599456787,
        flux_exponent=0,
        retained_slices=85,
    ),
    ShotSignObservation(
        shot=13_890,
        plasma_current_a=719_757.4062499921,
        toroidal_field_t=-0.423467755317688,
        poloidal_flux_edge_minus_axis_wb_per_rad=-0.04966495033208261,
        poloidal_angle_signed_area_m2=-1.6475482418671565,
        safety_factor=6.559555530548096,
        flux_exponent=0,
        retained_slices=70,
    ),
    ShotSignObservation(
        shot=13_471,
        plasma_current_a=-675_203.3124999956,
        toroidal_field_t=0.4392752945423126,
        poloidal_flux_edge_minus_axis_wb_per_rad=0.03213906383923698,
        poloidal_angle_signed_area_m2=-1.6472013104793763,
        safety_factor=6.428024053573608,
        flux_exponent=0,
        retained_slices=44,
    ),
    ShotSignObservation(
        shot=13_472,
        plasma_current_a=-716_287.5000000217,
        toroidal_field_t=0.43514707684516907,
        poloidal_flux_edge_minus_axis_wb_per_rad=0.043208963359904554,
        poloidal_angle_signed_area_m2=-1.6141660274907323,
        safety_factor=6.233675718307495,
        flux_exponent=0,
        retained_slices=52,
    ),
)
"""Stored real-shot receipt: two pulses at each plasma-current polarity."""


@dataclass(frozen=True)
class ConventionScore:
    """One candidate convention and every observation it fails to predict."""

    identifier: int
    sigma_bp: int
    e_bp: int
    sigma_r_phi_z: int
    sigma_rho_theta_phi: int
    violations: tuple[str, ...]

    @property
    def survives(self) -> bool:
        """Whether the convention predicts every observation in the cohort."""

        return not self.violations


COCOS_CANDIDATES = tuple(sorted(CONVENTION_DIGITS))
"""All sixteen conventions in Sauter and Medvedev Table I."""


def _finite_sign(value: float, quantity: str) -> int:
    if not np.isfinite(value) or value == 0:
        raise ValueError(f"{quantity} has no finite non-zero sign: {value!r}")
    return 1 if value > 0 else -1


def score_convention(
    identifier: int,
    observations: Sequence[ShotSignObservation] = MAST_LEVEL2_SIGN_TABLE,
) -> ConventionScore:
    """Score one convention against every retained real-shot sign.

    The consistency relations are

    ``sign(psi_edge - psi_axis) = sign(Ip) * sigma_bp``
    ``sign(q) = sign(Ip) * sign(B0) * sigma_rho_theta_phi``

    and Table I gives the front-view poloidal direction as
    ``-sigma_r_phi_z * sigma_rho_theta_phi`` when counter-clockwise is
    positive.  The source flux units determine ``e_bp`` independently.
    """

    try:
        sigma_bp, e_bp, sigma_r_phi_z, sigma_rho_theta_phi = CONVENTION_DIGITS[
            int(identifier)
        ]
    except KeyError as error:
        raise ValueError(f"unknown COCOS convention {identifier!r}") from error

    violations: list[str] = []
    expected_angle_direction = -sigma_r_phi_z * sigma_rho_theta_phi
    for row in observations:
        if row.poloidal_flux_sign != row.plasma_current_sign * sigma_bp:
            violations.append(f"{row.shot}:poloidal_flux")
        if row.safety_factor_sign != (
            row.plasma_current_sign * row.toroidal_field_sign * sigma_rho_theta_phi
        ):
            violations.append(f"{row.shot}:safety_factor")
        if row.poloidal_angle_direction != expected_angle_direction:
            violations.append(f"{row.shot}:poloidal_angle_direction")
        if row.flux_exponent != e_bp:
            violations.append(f"{row.shot}:flux_exponent")

    return ConventionScore(
        identifier=int(identifier),
        sigma_bp=sigma_bp,
        e_bp=e_bp,
        sigma_r_phi_z=sigma_r_phi_z,
        sigma_rho_theta_phi=sigma_rho_theta_phi,
        violations=tuple(violations),
    )


def score_conventions(
    observations: Sequence[ShotSignObservation] = MAST_LEVEL2_SIGN_TABLE,
) -> tuple[ConventionScore, ...]:
    """Score all Sauter and Medvedev Table I conventions."""

    if not observations:
        raise ValueError("at least one sign observation is required")
    return tuple(score_convention(item, observations) for item in COCOS_CANDIDATES)


def surviving_conventions(
    observations: Sequence[ShotSignObservation] = MAST_LEVEL2_SIGN_TABLE,
) -> tuple[int, ...]:
    """Return conventions that predict every sign at every polarity."""

    return tuple(
        score.identifier for score in score_conventions(observations) if score.survives
    )


MAST_SOURCE_COCOS = 4
"""Unique convention surviving the four-shot level-2 sign receipt."""

MAST_TO_COCOS_17_FACTORS: Mapping[str, float] = MappingProxyType(
    {
        "psi_like": -tau,
        "ip_like": -1.0,
        "b0_like": -1.0,
        "q_like": 1.0,
        "dodpsi_like": -1.0 / tau,
        "tor_angle_like": -1.0,
        "pol_angle_like": -1.0,
        "one_like": 1.0,
    }
)
"""Sauter coefficient factors carrying measured COCOS-4 values to COCOS-17."""


def _ordered_polygon_area(r: np.ndarray, z: np.ndarray) -> float:
    valid = np.isfinite(r) & np.isfinite(z) & (r > 0)
    r_valid = r[valid]
    z_valid = z[valid]
    if r_valid.size < 4:
        return float("nan")
    return float(
        0.5 * np.sum(r_valid * np.roll(z_valid, -1) - np.roll(r_valid, -1) * z_valid)
    )


def read_level2_observation(
    shot: int,
    root: Path | str = MAST_LEVEL2_ROOT,
    *,
    minimum_current_a: float = 50_000.0,
) -> ShotSignObservation:
    """Read one observation directly from an immutable FAIR-MAST level-2 store."""

    import zarr  # noqa: PLC0415

    source = Path(root) / f"{int(shot)}.zarr"
    group = zarr.open_group(source, mode="r")
    magnetics = group["magnetics"]
    equilibrium = group["equilibrium"]

    psi_units = str(equilibrium["psi"].attrs.get("units", ""))
    if psi_units.replace(" ", "").lower() not in {"wb/rad", "weber/rad"}:
        raise ValueError(
            f"shot {shot} equilibrium/psi does not declare per-radian flux: "
            f"units={psi_units!r}"
        )

    magnetics_time = np.asarray(magnetics["time"], dtype=np.float64)
    plasma_current = np.asarray(magnetics["ip"], dtype=np.float64)
    current_valid = np.isfinite(magnetics_time) & np.isfinite(plasma_current)
    if np.count_nonzero(current_valid) < 2:
        raise ValueError(f"shot {shot} has no usable magnetics/ip time series")

    equilibrium_time = np.asarray(equilibrium["time"], dtype=np.float64)
    aligned_current = np.interp(
        equilibrium_time,
        magnetics_time[current_valid],
        plasma_current[current_valid],
        left=np.nan,
        right=np.nan,
    )
    toroidal_field = np.asarray(equilibrium["bvac_rmag"], dtype=np.float64)
    safety_factor = np.asarray(equilibrium["q95"], dtype=np.float64)
    retained = (
        np.isfinite(aligned_current)
        & (np.abs(aligned_current) > minimum_current_a)
        & np.isfinite(toroidal_field)
        & np.isfinite(safety_factor)
    )
    retained_indices = np.flatnonzero(retained)
    if retained_indices.size == 0:
        raise ValueError(f"shot {shot} has no plasma-on equilibrium slices")

    radial_grid = np.asarray(equilibrium["major_radius"], dtype=np.float64)
    vertical_grid = np.asarray(equilibrium["z"], dtype=np.float64)
    flux = np.asarray(equilibrium["psi"], dtype=np.float64)
    axis_r = np.asarray(equilibrium["magnetic_axis_r"], dtype=np.float64)
    axis_z = np.asarray(equilibrium["magnetic_axis_z"], dtype=np.float64)
    boundary_r = np.asarray(equilibrium["lcfs_r"], dtype=np.float64)
    boundary_z = np.asarray(equilibrium["lcfs_z"], dtype=np.float64)

    flux_differences: list[float] = []
    signed_areas: list[float] = []
    for index in retained_indices:
        field = flux[:, :, index]
        radial_index = int(np.argmin(np.abs(radial_grid - axis_r[index])))
        vertical_index = int(np.argmin(np.abs(vertical_grid - axis_z[index])))
        axis_flux = field[radial_index, vertical_index]

        r_boundary = boundary_r[:, index]
        z_boundary = boundary_z[:, index]
        boundary_valid = (
            np.isfinite(r_boundary) & np.isfinite(z_boundary) & (r_boundary > 0)
        )
        if not np.isfinite(axis_flux) or np.count_nonzero(boundary_valid) < 4:
            continue
        r_indices = np.abs(
            radial_grid[:, np.newaxis] - r_boundary[boundary_valid]
        ).argmin(axis=0)
        z_indices = np.abs(
            vertical_grid[:, np.newaxis] - z_boundary[boundary_valid]
        ).argmin(axis=0)
        edge_flux = float(np.nanmedian(field[r_indices, z_indices]))
        flux_differences.append(edge_flux - float(axis_flux))
        signed_areas.append(_ordered_polygon_area(r_boundary, z_boundary))

    if not flux_differences or not signed_areas:
        raise ValueError(f"shot {shot} has no usable flux-boundary slices")

    return ShotSignObservation(
        shot=int(shot),
        plasma_current_a=float(np.nanmedian(aligned_current[retained])),
        toroidal_field_t=float(np.nanmedian(toroidal_field[retained])),
        poloidal_flux_edge_minus_axis_wb_per_rad=float(np.nanmedian(flux_differences)),
        poloidal_angle_signed_area_m2=float(np.nanmedian(signed_areas)),
        safety_factor=float(np.nanmedian(safety_factor[retained])),
        flux_exponent=0,
        retained_slices=int(retained_indices.size),
    )


def read_level2_sign_table(
    root: Path | str = MAST_LEVEL2_ROOT,
    shots: Sequence[int] = tuple(row.shot for row in MAST_LEVEL2_SIGN_TABLE),
) -> tuple[ShotSignObservation, ...]:
    """Read the fixed both-polarity cohort from the level-2 mirror."""

    return tuple(read_level2_observation(shot, root) for shot in shots)


def format_sign_report(
    observations: Sequence[ShotSignObservation] = MAST_LEVEL2_SIGN_TABLE,
) -> str:
    """Format the per-shot sign table, every score and the survivor verdict."""

    lines = [
        "shot  Ip  Bphi  psi(edge-axis)  theta  q  eBp  retained",
    ]
    for row in observations:
        direction = "CCW" if row.poloidal_angle_direction > 0 else "CW"
        lines.append(
            f"{row.shot:5d}  {row.plasma_current_sign:+d}  "
            f"{row.toroidal_field_sign:+d}  {row.poloidal_flux_sign:+d}  "
            f"{direction:>5s}  {row.safety_factor_sign:+d}  "
            f"{row.flux_exponent:d}  {row.retained_slices:d}"
        )

    scores = score_conventions(observations)
    lines.append("")
    lines.append("COCOS  sigma_Bp  e_Bp  sigma_RphiZ  sigma_rhothetaphi  result")
    for score in scores:
        result = "SURVIVES" if score.survives else ",".join(score.violations)
        lines.append(
            f"{score.identifier:5d}  {score.sigma_bp:+8d}  {score.e_bp:4d}  "
            f"{score.sigma_r_phi_z:+11d}  "
            f"{score.sigma_rho_theta_phi:+17d}  {result}"
        )

    survivors = tuple(score.identifier for score in scores if score.survives)
    if len(survivors) == 1:
        verdict = f"exactly 1 convention survives: COCOS-{survivors[0]}"
    elif survivors:
        verdict = f"more than 1 convention survives: {survivors}"
    else:
        verdict = "0 conventions survive: observations are not COCOS-expressible"
    lines.extend(("", f"VERDICT: {verdict}"))
    return "\n".join(lines)


def main() -> int:
    """Print the committed, corpus-independent determination receipt."""

    print(format_sign_report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COCOS_CANDIDATES",
    "MAST_LEVEL2_ROOT",
    "MAST_LEVEL2_SIGN_TABLE",
    "MAST_SOURCE_COCOS",
    "MAST_TO_COCOS_17_FACTORS",
    "SIGN_SOURCE_PATHS",
    "ConventionScore",
    "ShotSignObservation",
    "format_sign_report",
    "read_level2_observation",
    "read_level2_sign_table",
    "score_convention",
    "score_conventions",
    "surviving_conventions",
]
