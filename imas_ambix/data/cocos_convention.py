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

from dataclasses import dataclass, replace
from math import tau
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

import numpy as np
from nova_cocos import CONVENTION_DIGITS

from imas_alambic.eddb import normalised_shot
from imas_alambic.signal_map import SignalRule, load_packaged_signal_map
from imas_alambic.virtual_zarr import VirtualZarrView

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from imas_ambix.challenge.loader import EfitLabels

MAST_LEVEL2_ROOT = Path("/work/projects/imas_gpu/mast/level2/shots")

_RAW_PLASMA_CURRENT_TARGETS = ("magnetics/ip",)
_RAW_FLUX_LOOP_TARGETS = ("magnetics/flux_loop_flux", "magnetics/flux_loop/flux")
_RAW_TF_COIL_TARGETS = ("tf/coil/current",)
_RAW_SAFETY_FACTOR_TARGETS = ("equilibrium/time_slice/global_quantities/q_95",)
_RAW_TOROIDAL_FIELD_TARGETS = ("equilibrium/vacuum_toroidal_field/b0",)

#: The two plasma-current floors that separate plasma-on samples from the
#: pre-plasma baseline, shared by every observation reader so a level-2 read
#: and a signal-map read score the same window.
_MINIMUM_CURRENT_A = 50_000.0
_BASELINE_CURRENT_A = 10_000.0

#: The equilibrium half's vacuum toroidal field, bound to the DD b0 leaf when it
#: is read through a signal map rather than assembled into an ``EfitLabels``.
_VACUUM_FIELD_KEY = "magnetics_bcoil"

#: A flux-loop rule enters the raw response only when its sign and value are
#: proven.  A ``source-only`` loop is left out: on JT-60SA every loop but the
#: reference (loop 7) measures a difference to that reference and never fits its
#: own absolute prediction, so only the reference carries an absolute flux a
#: sign product can use.
_PROVEN_VALIDATION_STATE = "corpus-validated"

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
    poloidal_flux_edge_minus_axis_wb_per_rad: float | None
    poloidal_angle_signed_area_m2: float | None
    safety_factor: float
    flux_exponent: int | None
    """The declared poloidal-flux exponent, or ``None`` when the source carries
    no psi grid so the exponent is not measured."""
    retained_slices: int
    tf_coil_current_sign: int | None = None
    """Sign of the raw TF coil current, the polarity cross-check beside the
    observation.  ``None`` when the source is a store that carries no TF map."""

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
    def poloidal_flux_sign(self) -> int | None:
        """Sign of poloidal flux at the edge relative to the axis.

        ``None`` when the equilibrium source carries no psi grid: the edge
        minus axis flux is then not measurable and no sign product can be
        formed from it.
        """

        if self.poloidal_flux_edge_minus_axis_wb_per_rad is None:
            return None
        return _finite_sign(
            self.poloidal_flux_edge_minus_axis_wb_per_rad,
            "poloidal flux edge minus axis",
        )

    @property
    def poloidal_angle_direction(self) -> int | None:
        """Return ``+1`` for counter-clockwise or ``-1`` for clockwise.

        ``None`` when no ordered LCFS was available to bound a signed area.
        """

        if self.poloidal_angle_signed_area_m2 is None:
            return None
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
    EFIT output for an available magnetics measurement.  The raw loop channels
    are scored for unanimity rather than for a fixed sign: a well-formed store
    either has every loop response on one side or is split, and which side is
    read from the response itself, so a store that carries a single absolute
    loop is scored on that loop's own sign instead of against another store's
    sign convention, and a split store violates every candidate.  The q relation
    characterizes the EFIT output convention because level-2 has no raw q or
    toroidal-field reference.  The source flux units declare ``e_bp``.

    ``sigma_r_phi_z`` is deliberately not scored.  The signed area of an
    ordered LCFS array constrains the writer's point ordering, not the physical
    direction of positive toroidal angle.

    A row whose equilibrium source carries no psi grid has no measurable
    edge-minus-axis flux and no declared flux exponent, so both the
    reconstructed-poloidal-flux relation and the declared-flux-exponent check
    are left unscored for that row rather than failed; its ``flux_exponent`` is
    ``None`` rather than a hard-coded zero, and :func:`format_sign_report` names
    each such relation beside its reason.
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
        if 0 < row.raw_flux_loop_opposite_sign_channels < row.raw_flux_loop_channels:
            violations.append(f"{row.shot}:raw_flux_loop_channel_consensus")
        row_flux_sign = row.poloidal_flux_sign
        if row_flux_sign is not None and row_flux_sign != (
            row.plasma_current_sign * sigma_bp
        ):
            violations.append(f"{row.shot}:reconstructed_poloidal_flux")
        if row.safety_factor_sign != (
            row.plasma_current_sign * row.toroidal_field_sign * sigma_rho_theta_phi
        ):
            violations.append(f"{row.shot}:reconstructed_safety_factor")
        if row.flux_exponent is not None and row.flux_exponent != e_bp:
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


def _unscored_relations(
    observations: Sequence[ShotSignObservation],
) -> tuple[tuple[int, str, str], ...]:
    """Name every consistency relation a row cannot be scored against.

    Returns ``(shot, relation, reason)`` for each relation the observations
    leave unscored, so the report states what the cohort could not test rather
    than presenting a shorter list of violations as complete.
    """

    unscored: list[tuple[int, str, str]] = []
    for row in observations:
        if row.poloidal_flux_sign is None:
            unscored.append(
                (
                    row.shot,
                    "reconstructed_poloidal_flux",
                    "the equilibrium source carries no psi grid, so the "
                    "edge-minus-axis flux has no measured value",
                )
            )
        if row.flux_exponent is None:
            unscored.append(
                (
                    row.shot,
                    "declared_flux_exponent",
                    "the equilibrium source carries no psi grid, so the "
                    "declared flux exponent is unmeasured and no row fixes "
                    "e_Bp; E101011's G-EQDSK (section 7) is the source that "
                    "would fix it",
                )
            )
    return tuple(unscored)


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
        "one_like": 1.0,
    }
)
"""Factors conditional on the external owner declaration being COCOS 3.

Each entry is a transformation class nova's ``TRANSFORMATIONS`` owns, so the
table is a data mirror of the derived factors rather than an independent
statement.  A sensitive-axis angle is authored from the binding's measured
``sign_convention`` and carries no COCOS-digit factor, so no angle class
appears here.
"""


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


def _observation_from_series(
    shot: int,
    *,
    plasma_current_time: np.ndarray,
    plasma_current: np.ndarray,
    flux_loops: np.ndarray,
    toroidal_field_time: np.ndarray,
    toroidal_field: np.ndarray,
    equilibrium: EfitLabels,
    minimum_current_a: float,
    baseline_current_a: float,
) -> ShotSignObservation:
    """Build one sign observation from raw series and an equilibrium record.

    This is the read-independent kernel.  Any store supplies the three raw
    series — plasma current, flux-loop flux and toroidal field — and any
    loader supplies one :class:`EfitLabels` equilibrium record, so the
    FAIR-MAST level-2 mirror and a catalogue-bound read share one computation
    of the signs, the raw flux-loop response and the flux-per-radian versus
    total-flux discriminator.  ``equilibrium.psirz`` holds one equilibrium
    frame per leading index, each a ``(height, radius)`` grid over
    ``grid_z_m`` and ``grid_r_m``.
    """

    plasma_current = np.asarray(plasma_current, dtype=np.float64)
    plasma_current_time = np.asarray(plasma_current_time, dtype=np.float64)
    current_valid = np.isfinite(plasma_current_time) & np.isfinite(plasma_current)
    if np.count_nonzero(current_valid) < 2:
        raise ValueError(f"shot {shot} has no usable plasma-current time series")
    raw_flux_response, raw_flux_channels, raw_flux_opposite = _raw_flux_loop_response(
        plasma_current,
        np.asarray(flux_loops, dtype=np.float64),
        minimum_current_a=minimum_current_a,
        baseline_current_a=baseline_current_a,
    )

    equilibrium_time = np.asarray(equilibrium.time_ms, dtype=np.float64)
    aligned_current = np.interp(
        equilibrium_time,
        plasma_current_time[current_valid],
        plasma_current[current_valid],
        left=np.nan,
        right=np.nan,
    )
    aligned_toroidal_field = np.interp(
        equilibrium_time,
        np.asarray(toroidal_field_time, dtype=np.float64),
        np.asarray(toroidal_field, dtype=np.float64),
        left=np.nan,
        right=np.nan,
    )
    safety_factor = np.asarray(equilibrium.scalars["efit_q95"], dtype=np.float64)
    retained = (
        np.isfinite(aligned_current)
        & (np.abs(aligned_current) > minimum_current_a)
        & np.isfinite(aligned_toroidal_field)
        & np.isfinite(safety_factor)
    )
    retained_indices = np.flatnonzero(retained)
    if retained_indices.size == 0:
        raise ValueError(f"shot {shot} has no plasma-on equilibrium slices")

    radial_grid = np.asarray(equilibrium.grid_r_m, dtype=np.float64)
    vertical_grid = np.asarray(equilibrium.grid_z_m, dtype=np.float64)
    flux = np.asarray(equilibrium.psirz, dtype=np.float64)

    flux_difference: float | None = None
    signed_area: float | None = None
    if flux.size:
        axis_r = np.asarray(equilibrium.scalars["efit_r_axis"], dtype=np.float64)
        axis_z = np.asarray(equilibrium.scalars["efit_z_axis"], dtype=np.float64)
        boundary_r = np.asarray(equilibrium.lcfs_r_m, dtype=np.float64)
        boundary_z = np.asarray(equilibrium.lcfs_z_m, dtype=np.float64)

        flux_differences: list[float] = []
        signed_areas: list[float] = []
        for index in retained_indices:
            field = flux[index]
            radial_index = int(np.argmin(np.abs(radial_grid - axis_r[index])))
            vertical_index = int(np.argmin(np.abs(vertical_grid - axis_z[index])))
            axis_flux = field[vertical_index, radial_index]

            r_boundary = boundary_r[index]
            z_boundary = boundary_z[index]
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
            edge_flux = float(np.nanmedian(field[z_indices, r_indices]))
            flux_differences.append(edge_flux - float(axis_flux))
            signed_areas.append(_ordered_polygon_area(r_boundary, z_boundary))

        if not flux_differences or not signed_areas:
            raise ValueError(f"shot {shot} has no usable flux-boundary slices")
        flux_difference = float(np.nanmedian(flux_differences))
        signed_area = float(np.nanmedian(signed_areas))

    return ShotSignObservation(
        shot=int(shot),
        plasma_current_a=float(np.nanmedian(aligned_current[retained])),
        raw_flux_loop_response_wb_per_a=raw_flux_response,
        raw_flux_loop_channels=raw_flux_channels,
        raw_flux_loop_opposite_sign_channels=raw_flux_opposite,
        toroidal_field_t=float(np.nanmedian(aligned_toroidal_field[retained])),
        poloidal_flux_edge_minus_axis_wb_per_rad=flux_difference,
        poloidal_angle_signed_area_m2=signed_area,
        safety_factor=float(np.nanmedian(safety_factor[retained])),
        flux_exponent=0 if flux.size else None,
        retained_slices=int(retained_indices.size),
    )


def _equilibrium_labels_from_level2(equilibrium: object) -> EfitLabels:
    """Assemble one canonical equilibrium record from a FAIR-MAST level-2 group.

    The stored layout orders flux as ``(radial, vertical, time)``; the record
    orders it ``(time, height, radius)``, so the transposition happens here
    once and every reader of the record sees one shape.
    """

    from imas_ambix.challenge.loader import EfitLabels  # noqa: PLC0415

    return EfitLabels(
        time_ms=np.asarray(equilibrium["time"], dtype=np.float64),
        psirz=np.transpose(np.asarray(equilibrium["psi"], dtype=np.float64), (2, 1, 0)),
        grid_r_m=np.asarray(equilibrium["major_radius"], dtype=np.float64),
        grid_z_m=np.asarray(equilibrium["z"], dtype=np.float64),
        lcfs_r_m=np.transpose(np.asarray(equilibrium["lcfs_r"], dtype=np.float64)),
        lcfs_z_m=np.transpose(np.asarray(equilibrium["lcfs_z"], dtype=np.float64)),
        scalars={
            "efit_q95": np.asarray(equilibrium["q95"], dtype=np.float64),
            "efit_r_axis": np.asarray(equilibrium["magnetic_axis_r"], dtype=np.float64),
            "efit_z_axis": np.asarray(equilibrium["magnetic_axis_z"], dtype=np.float64),
        },
        cocos=MAST_SOURCE_COCOS,
    )


def read_level2_observation(
    shot: int,
    root: Path | str = MAST_LEVEL2_ROOT,
    *,
    minimum_current_a: float = _MINIMUM_CURRENT_A,
    baseline_current_a: float = _BASELINE_CURRENT_A,
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

    return _observation_from_series(
        shot,
        plasma_current_time=np.asarray(magnetics["time"], dtype=np.float64),
        plasma_current=np.asarray(magnetics["ip"], dtype=np.float64),
        flux_loops=np.asarray(magnetics["flux_loop_flux"], dtype=np.float64),
        toroidal_field_time=np.asarray(equilibrium["time"], dtype=np.float64),
        toroidal_field=np.asarray(equilibrium["bvac_rmag"], dtype=np.float64),
        equilibrium=_equilibrium_labels_from_level2(equilibrium),
        minimum_current_a=minimum_current_a,
        baseline_current_a=baseline_current_a,
    )


def _rules_targeting(
    rules: Sequence[SignalRule],
    targets: tuple[str, ...],
    quantity: str,
) -> tuple[SignalRule, ...]:
    """Gather every signal-map rule whose DD target serves ``quantity``.

    A quantity may be served by more than one rule — each flux loop its own —
    so every match is kept rather than the first.  The declared target
    spellings are ordered, and within one target the rules come out by their
    structure index, so channels are gathered in target order.
    """

    selected: list[SignalRule] = []
    for target in targets:
        matched = [
            rule
            for rule in rules
            if rule.target_path == target or rule.target_path.startswith(f"{target}/")
        ]
        matched.sort(
            key=lambda rule: (
                rule.target_index is None,
                rule.target_index if rule.target_index is not None else 0,
                rule.semantic_id,
            )
        )
        selected.extend(matched)
    if not selected:
        raise ValueError(
            f"signal map declares no rule targeting {quantity} "
            f"(expected one of {targets})"
        )
    return tuple(selected)


def _one_rule(
    rules: Sequence[SignalRule],
    targets: tuple[str, ...],
    quantity: str,
) -> SignalRule:
    """Select the single rule serving ``quantity``, refusing zero or several."""

    selected = _rules_targeting(rules, targets, quantity)
    if len(selected) != 1:
        raise ValueError(
            f"signal map declares {len(selected)} rules targeting {quantity}; "
            "exactly one is required"
        )
    return selected[0]


def _absolute_flux_loop_rules(
    rules: Sequence[SignalRule],
    targets: tuple[str, ...],
    quantity: str,
) -> tuple[SignalRule, ...]:
    """Keep only the flux-loop rules whose sign and value a sign product may use.

    A differential flux loop stores the difference between two loops
    (``indices_differential``), so its raw channel is not an absolute flux and no
    sign product over it is meaningful: on JT-60SA every loop but the reference
    measures a difference against that reference and never fits its own absolute
    prediction.  A ``source-only`` rule carries an assumed sign the reader exists
    to measure, so it cannot enter either.  What survives is the rules whose
    convention is already proven, which for JT-60SA is the absolute reference
    loop alone.
    """

    selected = tuple(
        rule
        for rule in _rules_targeting(rules, targets, quantity)
        if rule.validation_state == _PROVEN_VALIDATION_STATE
    )
    if not selected:
        raise ValueError(
            f"signal map declares no proven rule targeting {quantity} "
            f"(expected at least one with validation_state "
            f"{_PROVEN_VALIDATION_STATE!r})"
        )
    return selected


def _equilibrium_half_from_signal_map(
    machine: str,
    pulse: Path,
    shot: int,
) -> tuple[EfitLabels, np.ndarray, np.ndarray]:
    """Assemble a gridless equilibrium record from the machine's equilibrium map.

    The safety factor (Q95) and the field-times-radius scalar (BTV) are read raw
    through the machine's ``equilibrium`` signal map, so the reader owns no sign
    of its own and reads the same source the equilibrium half would supply.
    Neither the flux grid nor the boundary is present in this store, so the
    record carries empty arrays and the kernel leaves every relation that needs
    flux unscored.
    """

    from imas_ambix.challenge.loader import EfitLabels  # noqa: PLC0415

    equilibrium_map = load_packaged_signal_map(machine, "equilibrium")
    safety_factor_rule = _one_rule(
        equilibrium_map.signals, _RAW_SAFETY_FACTOR_TARGETS, "the safety factor"
    )
    toroidal_field_rule = _one_rule(
        equilibrium_map.signals,
        _RAW_TOROIDAL_FIELD_TARGETS,
        "the toroidal field",
    )
    view = VirtualZarrView.open(str(pulse), equilibrium_map, shot=shot)
    q_values, q_time = view.raw_series(safety_factor_rule.semantic_id)
    f_values, f_time = view.raw_series(toroidal_field_rule.semantic_id)
    safety_factor = _raw_series(q_values, safety_factor_rule)
    toroidal_field = _raw_series(f_values, toroidal_field_rule)
    empty = np.empty(0, dtype=np.float64)
    return (
        EfitLabels(
            time_ms=np.asarray(q_time, dtype=np.float64),
            psirz=np.empty((0, 0, 0), dtype=np.float64),
            grid_r_m=empty,
            grid_z_m=empty,
            lcfs_r_m=empty,
            lcfs_z_m=empty,
            scalars={"efit_q95": safety_factor},
            cocos=equilibrium_map.target_cocos,
        ),
        np.asarray(f_time, dtype=np.float64),
        toroidal_field,
    )


def _raw_series(values: object, rule: SignalRule) -> np.ndarray:
    """Collapse a rule's raw source values to its one time series."""

    series = np.asarray(values, dtype=np.float64)
    if series.ndim == 1:
        return series
    if series.shape[0] == 1:
        return series[0]
    raise ValueError(
        f"rule {rule.semantic_id!r} reads {series.shape[0]} channels where one "
        "time series is expected"
    )


def _aligned_loop(
    rule: SignalRule,
    series: np.ndarray,
    loop_time: np.ndarray,
    current_time: np.ndarray,
) -> np.ndarray:
    """Resample one flux-loop series onto the plasma-current time base.

    The loop and the current are separate EDDB channels with their own time
    vectors, so aligning them by sample index pairs measurements taken at
    different instants.  The loop is interpolated onto the current's time base
    instead, and a loop whose own time span does not cover the current's is
    refused rather than extrapolated: a flux-loop channel whose record starts
    after the plasma current has no measured value to contribute where the
    current is defined.  The one allowance is a gap no wider than one sampling
    interval of the loop's own record, which is inside the channel's own
    resolution: a record opening one sample after the current does carry a
    measured value there, and ``np.interp`` holds the nearest measured value
    across the gap rather than inventing a trend.
    """

    finite = np.isfinite(loop_time) & np.isfinite(series)
    if np.count_nonzero(finite) < 2:
        raise ValueError(
            f"flux-loop rule {rule.semantic_id!r} has fewer than two timed "
            "samples to interpolate"
        )
    span_time = loop_time[finite]
    span_values = series[finite]
    order = np.argsort(span_time)
    span_time = span_time[order]
    span_values = span_values[order]

    current_finite = np.isfinite(current_time)
    if not np.any(current_finite):
        raise ValueError("plasma-current time base has no finite samples")
    low = float(np.min(span_time))
    high = float(np.max(span_time))
    current_low = float(np.min(current_time[current_finite]))
    current_high = float(np.max(current_time[current_finite]))
    intervals = np.diff(span_time)
    resolution = float(np.median(intervals)) if intervals.size else 0.0
    tolerance = resolution
    if low > current_low + tolerance or high < current_high - tolerance:
        raise ValueError(
            f"flux-loop rule {rule.semantic_id!r} spans [{low}, {high}] s, "
            f"which does not cover the plasma-current time base "
            f"[{current_low}, {current_high}] s"
        )
    return np.interp(current_time, span_time, span_values)


def read_signal_map_observation(
    shot: int,
    machine: str,
    equilibrium: EfitLabels | None = None,
    *,
    root: Path | str,
    minimum_current_a: float = _MINIMUM_CURRENT_A,
    baseline_current_a: float = _BASELINE_CURRENT_A,
) -> ShotSignObservation:
    """Read one observation through a machine's packaged signal maps.

    The raw half — plasma current and every flux-loop channel — is read through
    the view's untransformed accessor,
    :meth:`~imas_alambic.virtual_zarr.VirtualZarrView.raw_series`, which
    resolves each rule to its source array and that channel's own time base
    without applying the compiled transform.  The transform is not used because
    for a ``source-only`` rule it carries the assumed sign this reader exists
    to measure, so the reader takes the raw values and the raw time base and
    owns the alignment itself.  Each flux loop is interpolated onto the
    plasma-current time base, and a loop whose time span does not cover that
    base is refused rather than extrapolated or index-aligned; only the rules
    whose convention is already proven enter at all, because a differential
    loop's raw channel is a difference to the reference rather than an absolute
    flux (see :func:`_absolute_flux_loop_rules`).  The equilibrium half
    supplies the vacuum toroidal field, the same way MAST's reader takes
    ``bvac_rmag`` from the level-2 equilibrium group, so the reader owns no
    field-per-ampere relation.  When no equilibrium record is supplied the
    safety factor and the field are read raw through the machine's
    ``equilibrium`` signal map instead, which yields a gridless record and
    leaves every flux-dependent relation unscored.  The ``tf`` map's coil
    current is read raw the same way and enters only as a polarity cross-check
    reported beside the observation, by sign alone.  The three raw series and
    the :class:`EfitLabels` record are handed to the shared kernel, so a
    signal-map read and a level-2 read share one computation of the signs.
    """

    shot_id = int(shot)
    magnetics_map = load_packaged_signal_map(machine, "magnetics")
    tf_map = load_packaged_signal_map(machine, "tf")
    plasma_current_rule = _one_rule(
        magnetics_map.signals, _RAW_PLASMA_CURRENT_TARGETS, "the plasma current"
    )
    flux_loop_rules = _absolute_flux_loop_rules(
        magnetics_map.signals, _RAW_FLUX_LOOP_TARGETS, "the flux-loop flux"
    )
    tf_coil_rule = _one_rule(
        tf_map.signals, _RAW_TF_COIL_TARGETS, "the TF coil current"
    )

    pulse = Path(root) / f"{normalised_shot(shot_id)}.zarr"
    magnetics_view = VirtualZarrView.open(str(pulse), magnetics_map, shot=shot_id)
    tf_view = VirtualZarrView.open(str(pulse), tf_map, shot=shot_id)

    if equilibrium is None:
        equilibrium, toroidal_field_time, toroidal_field = (
            _equilibrium_half_from_signal_map(machine, pulse, shot_id)
        )
    else:
        toroidal_field_time = np.asarray(equilibrium.time_ms, dtype=np.float64)
        toroidal_field = np.asarray(
            equilibrium.scalars[_VACUUM_FIELD_KEY], dtype=np.float64
        )

    plasma_current_values, plasma_current_time = magnetics_view.raw_series(
        plasma_current_rule.semantic_id
    )
    plasma_current = _raw_series(plasma_current_values, plasma_current_rule)
    plasma_current_time = np.asarray(plasma_current_time, dtype=np.float64)
    aligned_loops: list[np.ndarray] = []
    for rule in flux_loop_rules:
        loop_values, loop_time = magnetics_view.raw_series(rule.semantic_id)
        aligned_loops.append(
            _aligned_loop(
                rule,
                _raw_series(loop_values, rule),
                np.asarray(loop_time, dtype=np.float64),
                plasma_current_time,
            )[np.newaxis, :]
        )
    flux_loops = np.vstack(aligned_loops)
    tf_coil_values, _ = tf_view.raw_series(tf_coil_rule.semantic_id)
    tf_coil_current = np.asarray(tf_coil_values, dtype=np.float64).reshape(-1)
    tf_coil_current_sign = _finite_sign(
        float(np.nanmedian(tf_coil_current)), "TF coil current"
    )

    observation = _observation_from_series(
        shot_id,
        plasma_current_time=plasma_current_time,
        plasma_current=plasma_current,
        flux_loops=flux_loops,
        toroidal_field_time=toroidal_field_time,
        toroidal_field=toroidal_field,
        equilibrium=equilibrium,
        minimum_current_a=minimum_current_a,
        baseline_current_a=baseline_current_a,
    )
    return replace(observation, tf_coil_current_sign=tf_coil_current_sign)


def read_level2_sign_table(
    root: Path | str = MAST_LEVEL2_ROOT,
    shots: Sequence[int] = tuple(row.shot for row in MAST_LEVEL2_SIGN_TABLE),
) -> tuple[ShotSignObservation, ...]:
    """Read the fixed both-polarity cohort from the level-2 mirror."""

    return tuple(read_level2_observation(shot, root) for shot in shots)


def _is_mast_level2_cohort(observations: Sequence[ShotSignObservation]) -> bool:
    """Whether the observations are MAST's own level-2 receipt.

    MAST's coefficient classification, relative-sign products and Ip-like
    consequence are statements about MAST's level-2 layout, so they print only
    when the cohort is that receipt.  A cohort read from another machine's maps
    supports only the coefficients its own rows measure.
    """

    return tuple(observations) == MAST_LEVEL2_SIGN_TABLE


def _common_sign(values: Iterable[int | None]) -> int | None:
    """The single sign every value shares, or ``None`` when they disagree."""

    signs = {sign for sign in values if sign is not None}
    return signs.pop() if len(signs) == 1 else None


def _cohort_coefficient_classification(
    observations: Sequence[ShotSignObservation],
) -> tuple[CoefficientAssessment, ...]:
    """Classify the coefficients this cohort's own rows support.

    ``sigma_Bp`` is read from the raw absolute flux-loop response and
    ``sigma_rho_theta_phi`` from the q relation, so those two are measured from
    the rows.  ``e_Bp`` and ``sigma_R_phi_Z`` are left to an external
    declaration: no row measures the direction of positive toroidal angle, and
    the cohort's equilibrium carries no psi grid, so E101011's G-EQDSK
    (section 7) is the source that would fix the flux exponent.
    """

    sigma_bp = _common_sign(row.raw_flux_loop_response_sign for row in observations)
    sigma_rho = _common_sign(
        row.safety_factor_sign * row.plasma_current_sign * row.toroidal_field_sign
        for row in observations
    )
    return (
        CoefficientAssessment(
            coefficient="sigma_Bp",
            classification="measurable-from-data",
            value=sigma_bp,
            reasoning=(
                "the raw absolute flux-loop response carries one sign over "
                "every row, so this cohort's rows fix it"
            ),
            sources=(),
        ),
        CoefficientAssessment(
            coefficient="sigma_rho_theta_phi",
            classification="measurable-from-data",
            value=sigma_rho,
            reasoning=(
                "the q relation sign(q) = sign(Ip)*sign(B0)*sigma_rho_theta_phi "
                "holds on every row"
            ),
            sources=(),
        ),
        CoefficientAssessment(
            coefficient="e_Bp",
            classification="requires-an-external-declaration",
            value=None,
            reasoning=(
                "the equilibrium carries no psi grid, so the declared flux "
                "exponent is unmeasured; E101011's G-EQDSK (section 7) is the "
                "source that would fix e_Bp"
            ),
            sources=(),
        ),
        CoefficientAssessment(
            coefficient="sigma_R_phi_Z",
            classification="requires-an-external-declaration",
            value=None,
            reasoning="no row measures the direction of positive toroidal angle",
            sources=(),
        ),
    )


def format_sign_report(
    observations: Sequence[ShotSignObservation] = MAST_LEVEL2_SIGN_TABLE,
) -> str:
    """Format the cohort's classification, receipt, score and verdict.

    MAST's fixed blocks — the coefficient classification, the determinable
    relative-sign products and the Ip-like consequence — print only for MAST's
    own level-2 cohort.  Any other cohort reports the coefficients its own rows
    measure, so a JT-60SA cohort leaves ``e_Bp`` and ``sigma_R_phi_Z`` to an
    external declaration and lists every candidate its scored relations leave.
    """

    mast_cohort = _is_mast_level2_cohort(observations)

    lines = [
        "COEFFICIENT CLASSIFICATION"
        if mast_cohort
        else "COHORT COEFFICIENT CLASSIFICATION",
    ]
    assessments = (
        COEFFICIENT_ASSESSMENTS
        if mast_cohort
        else _cohort_coefficient_classification(observations)
    )
    for assessment in assessments:
        value = "unknown" if assessment.value is None else f"{assessment.value:+d}"
        lines.append(
            f"{assessment.coefficient}: {assessment.classification}; value={value}"
        )
        if assessment.sources:
            lines.append(f"  reasoning: {assessment.reasoning}")
            for source in assessment.sources:
                lines.append(f"  source: {source.path} [{source.kind}]")
        elif mast_cohort:
            lines.append(f"  reasoning: {assessment.reasoning}")
            lines.append("  source: none in level-2")
        else:
            lines.append(f"  reasoning: {assessment.reasoning}")

    lines.extend(
        (
            "",
            "RAW AND RECONSTRUCTION SIGN RECEIPT",
            "shot  Ip  raw_flux/Ip  loops  Bphi  psi(edge-axis)  "
            "theta  q  eBp  retained",
        )
    )
    for row in observations:
        angle = row.poloidal_angle_direction
        direction = "n/a" if angle is None else ("CCW" if angle > 0 else "CW")
        flux_sign = row.poloidal_flux_sign
        flux_text = "n/a" if flux_sign is None else f"{flux_sign:+d}"
        exponent = row.flux_exponent
        exponent_text = "n/a" if exponent is None else f"{exponent:d}"
        lines.append(
            f"{row.shot:5d}  {row.plasma_current_sign:+d}  "
            f"{row.raw_flux_loop_response_sign:+d}  "
            f"{row.raw_flux_loop_opposite_sign_channels:d}/"
            f"{row.raw_flux_loop_channels:d}  "
            f"{row.toroidal_field_sign:+d}  {flux_text:>3s}  "
            f"{direction:>5s}  {row.safety_factor_sign:+d}  "
            f"{exponent_text}  {row.retained_slices:d}"
        )

    lines.extend(("", "UNSCORED RELATIONS"))
    unscored = _unscored_relations(observations)
    if unscored:
        for shot, relation, reason in unscored:
            lines.append(f"{shot}: {relation} not scored — {reason}")
    else:
        lines.append("none: every relation was scored on every row")

    if mast_cohort:
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
    lines.extend(("", f"VERDICT: {verdict}"))
    if mast_cohort:
        lines.extend(
            (
                "COCOS 3 versus COCOS 4: no level-2 measurement distinguishes "
                "them; they differ only in sigma_R_phi_Z.",
                "RECOMMENDATION: treat the MAST source COCOS as an explicit "
                "external declaration; COCOS 3 is an owner assumption pending a "
                "facility statement of positive-phi direction, not a measurement.",
                "IP-LIKE CONSEQUENCE: declaration 3 applies factor +1 to all 3 "
                "targets; declaration 4 applies factor -1 to all 3 targets.",
                "DECLARATION CHANGE: COCOS 4 to COCOS 3 moves factor -1 to +1 "
                "for each affected target.",
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
    "read_signal_map_observation",
    "score_convention",
    "score_conventions",
    "surviving_conventions",
]
