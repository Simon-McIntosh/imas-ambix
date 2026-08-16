"""FAIR-MAST level-2 coordinate-convention evidence.

Sauter and Medvedev reduce a COCOS convention to four coefficients:
``sigma_bp``, ``e_bp``, ``sigma_r_phi_z`` and
``sigma_rho_theta_phi``.  Level-2 values constrain some coefficients and
relative-sign products, but numerical arrays cannot declare the physical
direction of positive toroidal angle.  This module keeps those evidence
classes separate instead of treating reconstruction serialization as a
facility coordinate declaration.  The raw FAIR-MAST stores remain read-only.

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
from typing import TYPE_CHECKING, Literal

import numpy as np
from nova.io.cocos import CONVENTION_DIGITS

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

MAST_LEVEL2_ROOT = Path("/work/projects/imas_gpu/mast/level2/shots")

EvidenceClassification = Literal[
    "measurable-from-data",
    "requires-an-external-declaration",
]
SourceKind = Literal[
    "measurement",
    "reconstruction-output",
    "reconstruction-metadata-declaration",
]


@dataclass(frozen=True)
class EvidenceSource:
    """One exact level-2 path and the provenance of its values."""

    path: str
    kind: SourceKind


@dataclass(frozen=True)
class CoefficientAssessment:
    """What level-2 can establish about one Sauter coefficient."""

    coefficient: str
    classification: EvidenceClassification
    value: int | None
    reasoning: str
    sources: tuple[EvidenceSource, ...]


COEFFICIENT_ASSESSMENTS = (
    CoefficientAssessment(
        coefficient="sigma_Bp",
        classification="measurable-from-data",
        value=-1,
        reasoning=(
            "Baseline-corrected flux-loop response has the opposite sign to "
            "measured plasma current in every usable channel-shot relation; "
            "the reconstructed edge-minus-axis flux sign independently agrees."
        ),
        sources=(
            EvidenceSource("magnetics/time", "measurement"),
            EvidenceSource("magnetics/ip", "measurement"),
            EvidenceSource("magnetics/flux_loop_flux", "measurement"),
            EvidenceSource("equilibrium/time", "reconstruction-output"),
            EvidenceSource("equilibrium/psi", "reconstruction-output"),
            EvidenceSource("equilibrium/major_radius", "reconstruction-output"),
            EvidenceSource("equilibrium/z", "reconstruction-output"),
            EvidenceSource("equilibrium/magnetic_axis_r", "reconstruction-output"),
            EvidenceSource("equilibrium/magnetic_axis_z", "reconstruction-output"),
            EvidenceSource("equilibrium/lcfs_r", "reconstruction-output"),
            EvidenceSource("equilibrium/lcfs_z", "reconstruction-output"),
        ),
    ),
    CoefficientAssessment(
        coefficient="e_Bp",
        classification="requires-an-external-declaration",
        value=0,
        reasoning=(
            "Array magnitudes do not distinguish flux from flux divided by 2pi; "
            "the value zero comes only from the declared Wb/rad units."
        ),
        sources=(
            EvidenceSource(
                "equilibrium/psi:units",
                "reconstruction-metadata-declaration",
            ),
        ),
    ),
    CoefficientAssessment(
        coefficient="sigma_R_phi_Z",
        classification="requires-an-external-declaration",
        value=None,
        reasoning=(
            "No level-2 measurement declares whether positive phi makes "
            "(R, phi, Z) right-handed; ordered contour points are an output "
            "serialization choice, not a physical handedness measurement."
        ),
        sources=(),
    ),
    CoefficientAssessment(
        coefficient="sigma_rho_theta_phi",
        classification="measurable-from-data",
        value=-1,
        reasoning=(
            "The q, plasma-current and vacuum-field relative signs give minus "
            "one for the convention written by EFIT.  Only plasma current is "
            "a raw measurement, so this characterizes reconstruction output."
        ),
        sources=(
            EvidenceSource("magnetics/time", "measurement"),
            EvidenceSource("magnetics/ip", "measurement"),
            EvidenceSource("equilibrium/time", "reconstruction-output"),
            EvidenceSource("equilibrium/bvac_rmag", "reconstruction-output"),
            EvidenceSource("equilibrium/q95", "reconstruction-output"),
        ),
    ),
)
"""Binary classification and exact provenance for all four coefficients."""

SIGN_SOURCE_PATHS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        item.coefficient: tuple(source.path for source in item.sources)
        for item in COEFFICIENT_ASSESSMENTS
    }
)
"""Compatibility view of the exact paths used for each coefficient."""


@dataclass(frozen=True)
class RelativeSignProduct:
    """A coefficient product exposed by stored relative signs."""

    expression: str
    value: int
    scope: str
    sources: tuple[EvidenceSource, ...]


RELATIVE_SIGN_PRODUCTS = (
    RelativeSignProduct(
        expression="sigma_Bp*sigma_rho_theta_phi",
        value=1,
        scope=(
            "EFIT output relation; this adds no discriminator between "
            "COCOS 3 and COCOS 4"
        ),
        sources=(
            EvidenceSource("equilibrium/psi", "reconstruction-output"),
            EvidenceSource("equilibrium/bvac_rmag", "reconstruction-output"),
            EvidenceSource("equilibrium/q95", "reconstruction-output"),
        ),
    ),
    RelativeSignProduct(
        expression="sigma_R_phi_Z*sigma_rho_theta_phi",
        value=1,
        scope=(
            "ordered-LCFS serialization only; excluded from the physical "
            "facility-convention candidate score"
        ),
        sources=(
            EvidenceSource("equilibrium/lcfs_r", "reconstruction-output"),
            EvidenceSource("equilibrium/lcfs_z", "reconstruction-output"),
        ),
    ),
)
"""Products retained separately from individual-coefficient evidence."""


@dataclass(frozen=True)
class ShotSignObservation:
    """Robust medians extracted from one real level-2 pulse."""

    shot: int
    plasma_current_a: float
    raw_flux_loop_response_wb_per_a: float
    raw_flux_loop_channels: int
    raw_flux_loop_opposite_sign_channels: int
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
    def raw_flux_loop_response_sign(self) -> int:
        """Sign of baseline-corrected raw flux-loop response per ampere."""

        return _finite_sign(
            self.raw_flux_loop_response_wb_per_a,
            "raw flux-loop response per plasma-current ampere",
        )

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
        raw_flux_loop_response_wb_per_a=-5.990708310010158e-07,
        raw_flux_loop_channels=14,
        raw_flux_loop_opposite_sign_channels=14,
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
        raw_flux_loop_response_wb_per_a=-3.974035046128424e-07,
        raw_flux_loop_channels=14,
        raw_flux_loop_opposite_sign_channels=14,
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
        raw_flux_loop_response_wb_per_a=-4.4113093628010267e-07,
        raw_flux_loop_channels=14,
        raw_flux_loop_opposite_sign_channels=14,
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
        raw_flux_loop_response_wb_per_a=-4.5849222015861696e-07,
        raw_flux_loop_channels=14,
        raw_flux_loop_opposite_sign_channels=14,
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
    """Score one convention using defensible coefficient constraints.

    The consistency relations are

    ``sign(psi_edge - psi_axis) = sign(Ip) * sigma_bp``
    ``sign(q) = sign(Ip) * sign(B0) * sigma_rho_theta_phi``

    The raw flux-loop response corroborates ``sigma_bp`` without substituting
    EFIT output for an available magnetics measurement.  The q relation
    characterizes the EFIT output convention because level-2 has no raw q or
    toroidal-field reference.  The source flux units declare ``e_bp``.

    ``sigma_r_phi_z`` is deliberately not scored.  The signed area of an
    ordered LCFS array constrains the writer's point ordering, not the physical
    direction of positive toroidal angle.
    """

    try:
        sigma_bp, e_bp, sigma_r_phi_z, sigma_rho_theta_phi = CONVENTION_DIGITS[
            int(identifier)
        ]
    except KeyError as error:
        raise ValueError(f"unknown COCOS convention {identifier!r}") from error

    violations: list[str] = []
    for row in observations:
        if row.raw_flux_loop_response_sign != sigma_bp:
            violations.append(f"{row.shot}:raw_flux_loop_response")
        if row.raw_flux_loop_opposite_sign_channels != row.raw_flux_loop_channels:
            violations.append(f"{row.shot}:raw_flux_loop_channel_consensus")
        if row.poloidal_flux_sign != row.plasma_current_sign * sigma_bp:
            violations.append(f"{row.shot}:reconstructed_poloidal_flux")
        if row.safety_factor_sign != (
            row.plasma_current_sign * row.toroidal_field_sign * sigma_rho_theta_phi
        ):
            violations.append(f"{row.shot}:reconstructed_safety_factor")
        if row.flux_exponent != e_bp:
            violations.append(f"{row.shot}:declared_flux_exponent")

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
    """Return candidates consistent with data and available declarations."""

    return tuple(
        score.identifier for score in score_conventions(observations) if score.survives
    )


MAST_SOURCE_COCOS = 3
"""External owner assumption, not a level-2 measurement.

The declaration remains pending a facility statement identifying MAST's
positive-phi direction.  COCOS 3 and 4 both satisfy the measurable evidence.
"""

SOURCE_COCOS_RECOMMENDATION = "external-declaration"
"""MAST's positive-phi handedness must be declared outside the level-2 arrays."""

COCOS_3_4_MEASUREMENT_DISTINGUISHABLE = False
"""No measurement in the level-2 corpus distinguishes the two candidates."""

IP_LIKE_TARGETS = (
    "magnetics/ip",
    "pf_active/coil/current",
    "pf_active/solenoid/current",
)
"""Bound targets whose factor changes when the external declaration changes."""

IP_LIKE_CANDIDATE_FACTORS: Mapping[int, float] = MappingProxyType(
    {
        3: 1.0,
        4: -1.0,
    }
)
"""Source-to-COCOS-17 factors for the unresolved candidate pair."""

MAST_TO_COCOS_17_FACTORS: Mapping[str, float] = MappingProxyType(
    {
        "psi_like": tau,
        "ip_like": 1.0,
        "b0_like": 1.0,
        "q_like": -1.0,
        "dodpsi_like": 1.0 / tau,
        "tor_angle_like": 1.0,
        "pol_angle_like": -1.0,
        "one_like": 1.0,
    }
)
"""Factors conditional on the external owner declaration being COCOS 3."""


def _ordered_polygon_area(r: np.ndarray, z: np.ndarray) -> float:
    valid = np.isfinite(r) & np.isfinite(z) & (r > 0)
    r_valid = r[valid]
    z_valid = z[valid]
    if r_valid.size < 4:
        return float("nan")
    return float(
        0.5 * np.sum(r_valid * np.roll(z_valid, -1) - np.roll(r_valid, -1) * z_valid)
    )


def _raw_flux_loop_response(
    plasma_current: np.ndarray,
    flux_loops: np.ndarray,
    *,
    minimum_current_a: float,
    baseline_current_a: float,
) -> tuple[float, int, int]:
    """Return the robust raw flux response per ampere and channel consensus."""

    if flux_loops.ndim != 2 or flux_loops.shape[1] != plasma_current.size:
        raise ValueError(
            "magnetics/flux_loop_flux must have one column per magnetics/ip sample"
        )

    responses: list[float] = []
    for signal in flux_loops:
        valid = np.isfinite(plasma_current) & np.isfinite(signal)
        baseline = valid & (np.abs(plasma_current) < baseline_current_a)
        plasma_on = valid & (np.abs(plasma_current) > minimum_current_a)
        if np.count_nonzero(baseline) < 2 or np.count_nonzero(plasma_on) < 2:
            continue
        offset = float(np.nanmedian(signal[baseline]))
        response = float(
            np.nanmedian((signal[plasma_on] - offset) / plasma_current[plasma_on])
        )
        if np.isfinite(response) and response != 0:
            responses.append(response)

    if not responses:
        raise ValueError("shot has no usable magnetics/flux_loop_flux channels")
    opposite = sum(response < 0 for response in responses)
    return float(np.median(responses)), len(responses), opposite


def read_level2_observation(
    shot: int,
    root: Path | str = MAST_LEVEL2_ROOT,
    *,
    minimum_current_a: float = 50_000.0,
    baseline_current_a: float = 10_000.0,
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
    flux_loops = np.asarray(magnetics["flux_loop_flux"], dtype=np.float64)
    current_valid = np.isfinite(magnetics_time) & np.isfinite(plasma_current)
    if np.count_nonzero(current_valid) < 2:
        raise ValueError(f"shot {shot} has no usable magnetics/ip time series")
    raw_flux_response, raw_flux_channels, raw_flux_opposite = _raw_flux_loop_response(
        plasma_current,
        flux_loops,
        minimum_current_a=minimum_current_a,
        baseline_current_a=baseline_current_a,
    )

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
        raw_flux_loop_response_wb_per_a=raw_flux_response,
        raw_flux_loop_channels=raw_flux_channels,
        raw_flux_loop_opposite_sign_channels=raw_flux_opposite,
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
    """Format provenance, coefficient limits and the conditional declaration."""

    lines = [
        "COEFFICIENT CLASSIFICATION",
    ]
    for assessment in COEFFICIENT_ASSESSMENTS:
        value = "unknown" if assessment.value is None else f"{assessment.value:+d}"
        lines.append(
            f"{assessment.coefficient}: {assessment.classification}; value={value}"
        )
        lines.append(f"  reasoning: {assessment.reasoning}")
        if assessment.sources:
            for source in assessment.sources:
                lines.append(f"  source: {source.path} [{source.kind}]")
        else:
            lines.append("  source: none in level-2")

    lines.extend(
        (
            "",
            "RAW AND RECONSTRUCTION SIGN RECEIPT",
            "shot  Ip  raw_flux/Ip  loops  Bphi  psi(edge-axis)  "
            "theta  q  eBp  retained",
        )
    )
    for row in observations:
        direction = "CCW" if row.poloidal_angle_direction > 0 else "CW"
        lines.append(
            f"{row.shot:5d}  {row.plasma_current_sign:+d}  "
            f"{row.raw_flux_loop_response_sign:+d}  "
            f"{row.raw_flux_loop_opposite_sign_channels:d}/"
            f"{row.raw_flux_loop_channels:d}  "
            f"{row.toroidal_field_sign:+d}  {row.poloidal_flux_sign:+d}  "
            f"{direction:>5s}  {row.safety_factor_sign:+d}  "
            f"{row.flux_exponent:d}  {row.retained_slices:d}"
        )

    lines.extend(("", "DETERMINABLE RELATIVE-SIGN PRODUCTS"))
    for product in RELATIVE_SIGN_PRODUCTS:
        lines.append(f"{product.expression}={product.value:+d}; {product.scope}")
        for source in product.sources:
            lines.append(f"  source: {source.path} [{source.kind}]")

    scores = score_conventions(observations)
    lines.append("")
    lines.append("STRICT CANDIDATE SCORE")
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
    if survivors:
        verdict = f"{len(survivors)} conventions survive: {survivors}"
    else:
        verdict = "0 conventions survive: observations are not COCOS-expressible"
    lines.extend(
        (
            "",
            f"VERDICT: {verdict}",
            "COCOS 3 versus COCOS 4: no level-2 measurement distinguishes them; "
            "they differ only in sigma_R_phi_Z.",
            "RECOMMENDATION: treat the MAST source COCOS as an explicit external "
            "declaration; COCOS 3 is an owner assumption pending a facility "
            "statement of positive-phi direction, not a measurement.",
            "IP-LIKE CONSEQUENCE: declaration 3 applies factor +1 to all 3 targets; "
            "declaration 4 applies factor -1 to all 3 targets.",
            "DECLARATION CHANGE: COCOS 4 to COCOS 3 moves factor -1 to +1 for "
            "each affected target.",
            "IP-LIKE TARGETS: " + ", ".join(IP_LIKE_TARGETS),
        )
    )
    return "\n".join(lines)


def main() -> int:
    """Print the committed, corpus-independent determination receipt."""

    print(format_sign_report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COCOS_3_4_MEASUREMENT_DISTINGUISHABLE",
    "COCOS_CANDIDATES",
    "COEFFICIENT_ASSESSMENTS",
    "IP_LIKE_CANDIDATE_FACTORS",
    "IP_LIKE_TARGETS",
    "MAST_LEVEL2_ROOT",
    "MAST_LEVEL2_SIGN_TABLE",
    "MAST_SOURCE_COCOS",
    "MAST_TO_COCOS_17_FACTORS",
    "RELATIVE_SIGN_PRODUCTS",
    "SIGN_SOURCE_PATHS",
    "SOURCE_COCOS_RECOMMENDATION",
    "CoefficientAssessment",
    "ConventionScore",
    "EvidenceSource",
    "RelativeSignProduct",
    "ShotSignObservation",
    "format_sign_report",
    "read_level2_observation",
    "read_level2_sign_table",
    "score_convention",
    "score_conventions",
    "surviving_conventions",
]
